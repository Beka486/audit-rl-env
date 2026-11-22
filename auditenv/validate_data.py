"""Independent data validator for exported audit episodes (Python standard library only).

For every episode directory (``public/`` + ``private/``) this module re-derives, from the
published specification alone, that

* the files are well-formed                                   (``format``)
* the documents are internally consistent                     (``documents_consistent``)
* the books *before* error injection balance                  (``clean_books_balanced``)
* the books *before* error injection follow the booking policy
  with zero discrepancies                                     (``clean_books_follow_policy``)
* every declared discrepancy is reproduced from public data   (``declared_discrepancies_present``)
* public data holds no undeclared discrepancy or anomaly      (``truth_complete``)
* every evidence set alone is sufficient to derive its issue  (``evidence_sufficient``)
* the truth file is well-formed                               (``truth_wellformed``)
* variants / interactions are what truth says they are        (``variants_consistent``)
* provenance agrees with the public/clean difference          (``provenance_consistent``)
* nothing private leaks into public files                     (``no_private_leak``)

Independence: this module never imports ``auditenv.generator`` or ``auditenv.checker`` (nor
any other ``auditenv`` module); every rule (period membership, booking rules, FX policy,
rounding, discrepancy definitions) is implemented here from the specification.

Usage::

    python -m auditenv.validate_data <root> [--json] [--max-errors N]
"""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- constants
BANK = "1000-BANK"
AR = "1100-AR"
FXC = "1950-FX-CLEARING"
UNAPPLIED = "2400-UNAPPLIED-CASH"
REVENUE_ACCOUNTS = frozenset({"4000-REVENUE-SERVICES", "4100-REVENUE-SUBSCRIPTIONS", "4200-REVENUE-GOODS"})
ACCOUNTS = frozenset({BANK, AR, FXC, UNAPPLIED}) | REVENUE_ACCOUNTS

LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                  "entity_id", "invoice_id", "payment_id"]
NULLABLE = ("entity_id", "invoice_id", "payment_id")
FX_COLUMNS = ["date", "source_currency", "target_currency", "rate"]
INVOICE_KEYS = frozenset({"doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency",
                          "total_minor", "line_items"})
LINE_KEYS = frozenset({"line_no", "sku", "description", "quantity", "unit_price_minor", "amount_minor",
                       "revenue_account"})
PAYMENT_KEYS = frozenset({"doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"})
REF_KEYS = frozenset({"invoice_id", "amount_minor"})
MANIFEST_KEYS = frozenset({"schema_version", "env_version", "generator_version", "episode_id", "company_name",
                           "base_currency", "currencies", "period_start", "period_end", "timezone", "counts",
                           "rounding", "files", "null_policy"})
COUNT_KEYS = frozenset({"ledger_rows", "invoices", "payments", "fx_rates", "documents"})
FILE_KEYS = {"ledger": "ledger.csv", "invoices": "invoices.jsonl", "payments": "payments.jsonl",
             "fx": "fx.csv", "policy": "policy.json"}

PUBLIC_FILES = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv")
PRIVATE_FILES = ("truth.json", "provenance.json")

ISSUE_TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")
ISSUE_KEYS = frozenset({"issue_id", "type", "affected_ids", "expected_amount_minor", "currency", "tolerance_minor",
                        "evidence_sets", "interaction_id", "variant", "description"})
AFFECTED_COUNT = {"duplicate_invoice": 2, "payment_allocation": 2, "fx_conversion": 2, "period_cutoff": 1}
ALLOCATION_VARIANTS = ("missing", "partial", "misapplied")
FX_VARIANTS = ("document_wall_clock_date", "invoice_issue_date", "next_business_day", "previous_business_day",
               "reverse_pair_rate")
MAX_ISSUES = 12
MAX_EVIDENCE_IDS = 10
MAX_AFFECTED_IDS = 10
LEAK_STRINGS = ("issue_id", "evidence_sets", "provenance", "truth", "interaction_id", "added_ledger_rows",
                "removed_ledger_rows", "added_documents")

CHECKS = ("format", "documents_consistent", "truth_wellformed", "no_private_leak", "clean_books_balanced",
          "clean_books_follow_policy", "declared_discrepancies_present", "truth_complete",
          "evidence_sufficient", "variants_consistent", "provenance_consistent")

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?[+-]\d{2}:\d{2}$")
_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")
_INT_RE = re.compile(r"^(?:0|[1-9]\d*)$")
_RATE_RE = re.compile(r"^(?:0|[1-9]\d*)(?:\.\d+)?$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CCY_RE = re.compile(r"^[A-Z]{3}$")
_MAX_ERRORS_PER_CHECK = 40


# --------------------------------------------------------------------------- small helpers
def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_id(v: Any) -> bool:
    return isinstance(v, str) and v != "" and v == v.strip() and v != "null"


def parse_ts(s: Any) -> datetime:
    """Strict ISO-8601 timestamp with an explicit +HH:MM / -HH:MM offset."""
    if not isinstance(s, str) or not _TS_RE.match(s):
        raise ValueError(f"not an ISO-8601 timestamp with offset: {s!r}")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:  # pragma: no cover - regex guarantees an offset
        raise ValueError(f"timestamp without offset: {s!r}")
    return dt


def parse_offset(s: Any) -> int:
    m = _OFFSET_RE.match(s) if isinstance(s, str) else None
    if not m or int(m.group(3)) >= 60 or int(m.group(2)) > 14:
        raise ValueError(f"bad UTC offset {s!r}")
    minutes = int(m.group(2)) * 60 + int(m.group(3))
    return minutes if m.group(1) == "+" else -minutes


def weekday_rate_date(d: date) -> date:
    """Policy rate date for calendar date ``d``: ``d`` itself on Mon-Fri, else the previous Friday."""
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _prev_business_day(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _next_business_day(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def convert_minor(amount_minor: int, exp_src: int, exp_tgt: int, rate: str) -> int:
    """target_minor = ROUND_HALF_EVEN(amount_minor * 10^-exp_src * rate * 10^exp_tgt); only the result is rounded."""
    with localcontext() as ctx:
        ctx.prec = 100
        value = (Decimal(amount_minor).scaleb(-exp_src) * Decimal(rate)).scaleb(exp_tgt)
        return int(value.to_integral_value(rounding=ROUND_HALF_EVEN))


def tolerance_for(expected_minor: int) -> int:
    """Published amount tolerance: exact amounts (tau = 0) for every generated issue type."""
    del expected_minor
    return 0


def _fmt_counter(c: Counter, limit: int = 4) -> str:
    items = list(c.elements())
    shown = ", ".join(str(x) for x in items[:limit])
    return f"[{shown}{', ...' if len(items) > limit else ''}]"


# --------------------------------------------------------------------------- data containers
class _Books:
    """Ledger + documents with journal indexes (used for both the clean and the public books)."""

    def __init__(self, ledger: List[dict], invoices: List[dict], payments: List[dict]):
        self.ledger = ledger
        self.invoices = invoices
        self.payments = payments
        self.inv = {d["invoice_id"]: d for d in invoices}
        self.pay = {d["payment_id"]: d for d in payments}
        self.journals: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        self.orphans: List[dict] = []
        for r in ledger:
            if r["payment_id"] is not None:
                self.journals[("P", r["payment_id"])].append(r)
            elif r["invoice_id"] is not None:
                self.journals[("I", r["invoice_id"])].append(r)
            else:
                self.orphans.append(r)


class _Episode:
    def __init__(self, path: Path):
        self.path = path
        self.raw: Dict[str, str] = {}
        self.raw_private: Dict[str, str] = {}
        self.manifest: Dict[str, Any] = {}
        self.policy: Dict[str, Any] = {}
        self.truth: Dict[str, Any] = {}
        self.provenance: Dict[str, Any] = {}
        self.ledger: List[dict] = []
        self.invoices: List[dict] = []
        self.payments: List[dict] = []
        self.fx: List[dict] = []
        self.exp: Dict[str, int] = {}
        self.tz: timezone = timezone.utc
        self.offset = timedelta(0)
        self.start: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self.end: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self.rates: Dict[Tuple[date, str, str], str] = {}
        self.fx_dates: Dict[Tuple[str, str], List[date]] = {}
        self.fx_by_id: Dict[str, dict] = {}
        self.pub: Optional[_Books] = None
        self.clean: Optional[_Books] = None
        self.id_index: Optional[Dict[str, Tuple[str, dict]]] = None
        self._pub_recon: Optional[Tuple[List[dict], List[str]]] = None

    # -- derived lookups
    def in_period(self, ts: datetime) -> bool:
        return self.start <= ts <= self.end

    def local_date(self, ts: datetime) -> date:
        return ts.astimezone(self.tz).date()

    def lookup_rate(self, d: date, src: str, tgt: str) -> Optional[Tuple[date, str]]:
        """Most recent published rate for (src -> tgt) on or before ``d``."""
        dates = self.fx_dates.get((src, tgt)) or []
        k = bisect.bisect_right(dates, d) - 1
        if k < 0:
            return None
        return dates[k], self.rates[(dates[k], src, tgt)]

    def policy_rate(self, pay_ts: datetime, src: str, tgt: str) -> Tuple[Optional[Tuple[date, str]], str]:
        d = self.local_date(pay_ts)
        found = self.lookup_rate(d, src, tgt)
        if found is None:
            return None, f"no fx rate {src}->{tgt} published on or before {d}"
        if found[0] != weekday_rate_date(d):
            return None, (f"fx rate {src}->{tgt} for {d}: most recent published date {found[0]} differs from "
                          f"the weekday-rule date {weekday_rate_date(d)} (gap in fx.csv)")
        return found, ""

    def public_recon(self) -> Tuple[List[dict], List[str]]:
        if self._pub_recon is None:
            assert self.pub is not None
            self._pub_recon = _reconcile(self, self.pub)
        return self._pub_recon


# --------------------------------------------------------------------------- loading + format
def _parse_ledger_row(raw: Dict[str, Any], where: str, ep: _Episode, err: List[str], *, from_csv: bool
                      ) -> Optional[dict]:
    """Validate one ledger row (CSV strings or JSON values) and return a typed copy (or None)."""
    ok = True
    row: Dict[str, Any] = {}
    if not _is_id(raw.get("row_id")):
        err.append(f"{where}: bad row_id {raw.get('row_id')!r}")
        ok = False
    row["row_id"] = raw.get("row_id")
    try:
        row["ts"] = parse_ts(raw.get("timestamp"))
    except ValueError as exc:
        err.append(f"{where}: {exc}")
        ok = False
    row["timestamp"] = raw.get("timestamp")
    if raw.get("account") not in ACCOUNTS:
        err.append(f"{where}: unknown account {raw.get('account')!r}")
        ok = False
    row["account"] = raw.get("account")
    for col, key in (("debit_minor", "debit"), ("credit_minor", "credit")):
        v = raw.get(col)
        if from_csv:
            if not isinstance(v, str) or not _INT_RE.match(v):
                err.append(f"{where}: {col} must be a non-negative integer, got {v!r}")
                ok = False
                continue
            row[key] = int(v)
        else:
            if not _is_int(v) or v < 0:
                err.append(f"{where}: {col} must be a non-negative integer, got {v!r}")
                ok = False
                continue
            row[key] = v
    if ok and (row["debit"] > 0) == (row["credit"] > 0):
        err.append(f"{where}: exactly one of debit_minor/credit_minor must be > 0")
        ok = False
    if raw.get("currency") not in ep.exp:
        err.append(f"{where}: currency {raw.get('currency')!r} not declared in manifest")
        ok = False
    row["currency"] = raw.get("currency")
    for col in NULLABLE:
        v = raw.get(col)
        if from_csv:
            if v == "null":
                v = None
            elif not _is_id(v):
                err.append(f"{where}: {col} must be an id or the literal 'null', got {v!r}")
                ok = False
        else:
            if v is not None and not _is_id(v):
                err.append(f"{where}: {col} must be an id or JSON null, got {v!r}")
                ok = False
        row[col] = v
    if not from_csv:
        extra = set(raw) - set(LEDGER_COLUMNS)
        missing = set(LEDGER_COLUMNS) - set(raw)
        if extra or missing:
            err.append(f"{where}: row keys differ from ledger columns "
                       f"(extra {sorted(extra)}, missing {sorted(missing)})")
            ok = False
    return row if ok else None


def _read_json(text: str, name: str, err: List[str]) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        err.append(f"{name}: invalid JSON ({exc})")
        return None


def _read_jsonl(text: str, name: str, err: List[str]) -> Optional[List[dict]]:
    out = []
    for n, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            err.append(f"{name} line {n}: invalid JSON ({exc})")
            return None
        if not isinstance(obj, dict):
            err.append(f"{name} line {n}: not a JSON object")
            return None
        out.append(obj)
    return out


def _load(ep: _Episode, err: List[str]) -> bool:
    """Read all files and parse what is needed by later checks. Returns False if unusable."""
    pub, priv = ep.path / "public", ep.path / "private"
    usable = True
    for folder, names, store in ((pub, PUBLIC_FILES, ep.raw), (priv, PRIVATE_FILES, ep.raw_private)):
        for name in names:
            p = folder / name
            try:
                store[name] = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                err.append(f"cannot read {folder.name}/{name}: {exc}")
                usable = False
    if not usable:
        return False
    loaded = {}
    for name in ("manifest.json", "policy.json"):
        loaded[name] = _read_json(ep.raw[name], name, err)
    for name in PRIVATE_FILES:
        loaded[name] = _read_json(ep.raw_private[name], name, err)
    for name, obj in loaded.items():
        if not isinstance(obj, dict):
            err.append(f"{name}: must be a JSON object")
            usable = False
    if not usable:
        return False
    ep.manifest, ep.policy = loaded["manifest.json"], loaded["policy.json"]
    ep.truth, ep.provenance = loaded["truth.json"], loaded["provenance.json"]
    inv = _read_jsonl(ep.raw["invoices.jsonl"], "invoices.jsonl", err)
    pay = _read_jsonl(ep.raw["payments.jsonl"], "payments.jsonl", err)
    if inv is None or pay is None:
        return False
    ep.invoices, ep.payments = inv, pay

    # manifest essentials (needed to type the rest)
    m = ep.manifest
    cur = m.get("currencies")
    if not isinstance(cur, dict) or not cur:
        err.append("manifest.currencies must be a non-empty object")
        return False
    for code, spec in cur.items():
        if not _CCY_RE.match(str(code)) or not isinstance(spec, dict) or set(spec) != {"exponent"} \
                or not _is_int(spec.get("exponent")) or not 0 <= spec["exponent"] <= 6:
            err.append(f"manifest.currencies[{code!r}] must be {{exponent: int}}")
            usable = False
        else:
            ep.exp[code] = spec["exponent"]
    try:
        off = parse_offset(m.get("timezone"))
        ep.offset = timedelta(minutes=off)
        ep.tz = timezone(ep.offset)
    except ValueError as exc:
        err.append(f"manifest.timezone: {exc}")
        usable = False
    for key in ("period_start", "period_end"):
        try:
            dt = parse_ts(m.get(key))
        except ValueError as exc:
            err.append(f"manifest.{key}: {exc}")
            usable = False
            continue
        setattr(ep, "start" if key == "period_start" else "end", dt)
    if not usable:
        return False

    # ledger.csv
    text = ep.raw["ledger.csv"]
    rd = csv.reader(io.StringIO(text))
    header = next(rd, None)
    if header != LEDGER_COLUMNS:
        err.append(f"ledger.csv: header must be {','.join(LEDGER_COLUMNS)}, got {header}")
        return False
    for n, rec in enumerate(rd, start=2):
        if len(rec) != len(LEDGER_COLUMNS):
            err.append(f"ledger.csv line {n}: expected {len(LEDGER_COLUMNS)} fields, got {len(rec)}")
            continue
        row = _parse_ledger_row(dict(zip(LEDGER_COLUMNS, rec)), f"ledger.csv line {n}", ep, err, from_csv=True)
        if row is not None:
            ep.ledger.append(row)

    # fx.csv
    rd = csv.reader(io.StringIO(ep.raw["fx.csv"]))
    header = next(rd, None)
    if header != FX_COLUMNS:
        err.append(f"fx.csv: header must be {','.join(FX_COLUMNS)}, got {header}")
        return False
    by_pair: Dict[Tuple[str, str], List[date]] = defaultdict(list)
    for n, rec in enumerate(rd, start=2):
        where = f"fx.csv line {n}"
        if len(rec) != len(FX_COLUMNS):
            err.append(f"{where}: expected {len(FX_COLUMNS)} fields, got {len(rec)}")
            continue
        ds, src, tgt, rate = rec
        try:
            if not _DATE_RE.match(ds):
                raise ValueError
            d = date.fromisoformat(ds)
        except ValueError:
            err.append(f"{where}: bad date {ds!r}")
            continue
        if src not in ep.exp or tgt not in ep.exp or src == tgt:
            err.append(f"{where}: bad currency pair {src!r}->{tgt!r}")
            continue
        try:
            if not _RATE_RE.match(rate) or Decimal(rate) <= 0:
                raise InvalidOperation
        except InvalidOperation:
            err.append(f"{where}: rate must be a positive plain decimal string, got {rate!r}")
            continue
        fx_id = f"FX:{ds}:{src}:{tgt}"
        if fx_id in ep.fx_by_id:
            err.append(f"{where}: duplicate fx record {fx_id}")
            continue
        rec_d = {"id": fx_id, "date": d, "source_currency": src, "target_currency": tgt, "rate": rate}
        ep.fx.append(rec_d)
        ep.fx_by_id[fx_id] = rec_d
        ep.rates[(d, src, tgt)] = rate
        by_pair[(src, tgt)].append(d)
    ep.fx_dates = {k: sorted(v) for k, v in by_pair.items()}

    # documents need the keys used for indexing
    for name, docs, keys in (("invoices.jsonl", ep.invoices, ("doc_id", "invoice_id")),
                             ("payments.jsonl", ep.payments, ("doc_id", "payment_id"))):
        for n, d in enumerate(docs, start=1):
            for k in keys:
                if not _is_id(d.get(k)):
                    err.append(f"{name} record {n}: {k} must be a non-empty id, got {d.get(k)!r}")
                    usable = False
    if not usable:
        return False
    ep.pub = _Books(ep.ledger, ep.invoices, ep.payments)
    return True


def _check_format(ep: _Episode, err: List[str]) -> None:
    m = ep.manifest
    missing = MANIFEST_KEYS - set(m)
    if missing:
        err.append(f"manifest.json: missing keys {sorted(missing)}")
    for k in ("schema_version", "env_version", "generator_version", "episode_id", "company_name",
              "null_policy", "timezone", "period_start", "period_end", "base_currency"):
        if k in m and not (isinstance(m[k], str) and m[k]):
            err.append(f"manifest.{k} must be a non-empty string")
    if m.get("base_currency") not in ep.exp:
        err.append(f"manifest.base_currency {m.get('base_currency')!r} not in currencies")
    if ep.start.utcoffset() != ep.offset or ep.end.utcoffset() != ep.offset:
        err.append("manifest period_start/period_end must carry the company timezone offset")
    if not ep.start < ep.end:
        err.append("manifest period_start must be before period_end")
    rounding = m.get("rounding")
    if not isinstance(rounding, dict) or rounding.get("mode") != "ROUND_HALF_EVEN":
        err.append("manifest.rounding.mode must be ROUND_HALF_EVEN")
    files = m.get("files")
    if not isinstance(files, dict) or any(files.get(k) != v for k, v in FILE_KEYS.items()):
        err.append(f"manifest.files must map {FILE_KEYS}")
    counts = m.get("counts")
    actual = {"ledger_rows": _count_csv_rows(ep.raw["ledger.csv"]), "invoices": len(ep.invoices),
              "payments": len(ep.payments), "fx_rates": _count_csv_rows(ep.raw["fx.csv"]),
              "documents": len(ep.invoices) + len(ep.payments)}
    if not isinstance(counts, dict) or set(counts) != COUNT_KEYS:
        err.append(f"manifest.counts must have exactly the keys {sorted(COUNT_KEYS)}")
    else:
        for k, v in actual.items():
            if counts.get(k) != v or not _is_int(counts.get(k)):
                err.append(f"manifest.counts.{k} = {counts.get(k)!r} but files contain {v}")

    # policy.json must agree with the manifest
    pol = ep.policy
    if pol.get("timezone") != m.get("timezone"):
        err.append("policy.timezone differs from manifest.timezone")
    period = pol.get("period")
    if not isinstance(period, dict) or period.get("start") != m.get("period_start") \
            or period.get("end") != m.get("period_end"):
        err.append("policy.period start/end differ from the manifest")
    accounts = pol.get("accounts")
    if not isinstance(accounts, dict) or set(accounts.values()) != set(ACCOUNTS):
        err.append("policy.accounts must list exactly the chart of accounts")

    # invoices
    for n, d in enumerate(ep.invoices, start=1):
        where = f"invoices.jsonl record {n} ({d.get('doc_id')})"
        if set(d) != INVOICE_KEYS:
            err.append(f"{where}: keys must be {sorted(INVOICE_KEYS)}, got {sorted(d)}")
            continue
        if not _is_id(d["entity_id"]):
            err.append(f"{where}: entity_id must be an id")
        for k in ("issued_at", "due_at"):
            try:
                parse_ts(d[k])
            except ValueError as exc:
                err.append(f"{where}: {k}: {exc}")
        if d["currency"] not in ep.exp:
            err.append(f"{where}: currency {d['currency']!r} not in manifest")
        if not _is_int(d["total_minor"]):
            err.append(f"{where}: total_minor must be an integer")
        items = d["line_items"]
        if not isinstance(items, list) or not items:
            err.append(f"{where}: line_items must be a non-empty list")
            continue
        for li in items:
            if not isinstance(li, dict) or set(li) != LINE_KEYS:
                err.append(f"{where}: line item keys must be {sorted(LINE_KEYS)}")
                continue
            for k in ("line_no", "quantity", "unit_price_minor", "amount_minor"):
                if not _is_int(li[k]):
                    err.append(f"{where}: line item {k} must be an integer")
            if not isinstance(li["sku"], str) or not isinstance(li["description"], str):
                err.append(f"{where}: line item sku/description must be strings")
            if li["revenue_account"] not in REVENUE_ACCOUNTS:
                err.append(f"{where}: bad revenue_account {li['revenue_account']!r}")
        nos = [li.get("line_no") for li in items if isinstance(li, dict)]
        if nos != list(range(1, len(items) + 1)):
            err.append(f"{where}: line_no must be 1..{len(items)} in order")

    # payments
    for n, d in enumerate(ep.payments, start=1):
        where = f"payments.jsonl record {n} ({d.get('doc_id')})"
        if set(d) != PAYMENT_KEYS:
            err.append(f"{where}: keys must be {sorted(PAYMENT_KEYS)}, got {sorted(d)}")
            continue
        try:
            parse_ts(d["timestamp"])
        except ValueError as exc:
            err.append(f"{where}: timestamp: {exc}")
        if d["currency"] not in ep.exp:
            err.append(f"{where}: currency {d['currency']!r} not in manifest")
        if not _is_int(d["amount_minor"]):
            err.append(f"{where}: amount_minor must be an integer")
        refs = d["invoice_refs"]
        if refs is None:
            continue
        if not isinstance(refs, list) or not refs:
            err.append(f"{where}: invoice_refs must be null or a non-empty list")
            continue
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != REF_KEYS or not _is_id(ref.get("invoice_id")) \
                    or not _is_int(ref.get("amount_minor")):
                err.append(f"{where}: invoice_refs entries must be {{invoice_id: id, amount_minor: int}}")

    # fx publication calendar: every Mon-Fri in range, never Sat/Sun
    for (src, tgt), dates in ep.fx_dates.items():
        weekend = [d for d in dates if d.weekday() >= 5]
        if weekend:
            err.append(f"fx.csv: {src}->{tgt} published on weekend dates {[str(d) for d in weekend[:3]]}")
        want, d = set(), dates[0]
        while d <= dates[-1]:
            if d.weekday() < 5:
                want.add(d)
            d += timedelta(days=1)
        gaps = sorted(want - set(dates))
        if gaps:
            err.append(f"fx.csv: {src}->{tgt} missing weekdays {[str(x) for x in gaps[:3]]}")

    # id uniqueness (also across namespaces, after the grader's trim/upper-case normalisation)
    namespaces = {
        "row_id": [r["row_id"] for r in ep.ledger],
        "doc_id": [d["doc_id"] for d in ep.invoices + ep.payments],
        "invoice_id": [d["invoice_id"] for d in ep.invoices],
        "payment_id": [d["payment_id"] for d in ep.payments],
        "fx id": [f["id"] for f in ep.fx],
    }
    for name, ids in namespaces.items():
        dup = [k for k, v in Counter(ids).items() if v > 1]
        if dup:
            err.append(f"duplicate {name}s: {dup[:5]}")
    norm = Counter(str(i).strip().upper() for ids in namespaces.values() for i in set(ids))
    clash = [k for k, v in norm.items() if v > 1]
    if clash:
        err.append(f"ids collide across namespaces or by case: {clash[:5]}")


def _count_csv_rows(text: str) -> int:
    rd = csv.reader(io.StringIO(text))
    next(rd, None)
    return sum(1 for _ in rd)


# --------------------------------------------------------------------------- documents
def _check_documents(ep: _Episode, err: List[str]) -> None:
    inv_by_id = {d["invoice_id"]: d for d in ep.invoices}
    for d in ep.invoices:
        where = f"invoice {d['invoice_id']}"
        items = d.get("line_items") or []
        total = 0
        for li in items:
            q, u, a = li.get("quantity"), li.get("unit_price_minor"), li.get("amount_minor")
            if not (_is_int(q) and _is_int(u) and _is_int(a)):
                err.append(f"{where}: non-integer line item")
                continue
            if q <= 0 or u < 0 or a <= 0:
                err.append(f"{where}: line {li.get('line_no')} needs quantity > 0, unit price >= 0, amount > 0")
            if a != q * u:
                err.append(f"{where}: line {li.get('line_no')} amount {a} != {q} * {u}")
            total += a
        if d.get("total_minor") != total:
            err.append(f"{where}: total_minor {d.get('total_minor')} != sum of line amounts {total}")
        if not _is_int(d.get("total_minor")) or d.get("total_minor", 0) <= 0:
            err.append(f"{where}: total_minor must be > 0")
        try:
            if parse_ts(d.get("due_at")) < parse_ts(d.get("issued_at")):
                err.append(f"{where}: due_at before issued_at")
        except ValueError:
            pass  # reported by format
    for p in ep.payments:
        where = f"payment {p['payment_id']}"
        amt, refs, ccy = p.get("amount_minor"), p.get("invoice_refs"), p.get("currency")
        if not _is_int(amt) or amt == 0:
            err.append(f"{where}: amount_minor must be a non-zero integer")
            continue
        if refs is None:
            if amt < 0:
                err.append(f"{where}: refund without an invoice reference")
            continue
        if not isinstance(refs, list):
            continue
        ids = [r.get("invoice_id") for r in refs if isinstance(r, dict)]
        if len(set(ids)) != len(ids):
            err.append(f"{where}: invoice referenced twice in invoice_refs")
        missing = [i for i in ids if i not in inv_by_id]
        if missing:
            err.append(f"{where}: invoice_refs point to unknown invoices {missing}")
            continue
        entities = {inv_by_id[i]["entity_id"] for i in ids}
        if len(entities) != 1:
            err.append(f"{where}: referenced invoices belong to different entities {sorted(map(str, entities))}")
        ref_amts = [r.get("amount_minor") for r in refs if isinstance(r, dict)]
        if not all(_is_int(a) for a in ref_amts):
            continue
        cross = [i for i in ids if inv_by_id[i]["currency"] != ccy]
        if amt > 0:
            if any(a <= 0 for a in ref_amts):
                err.append(f"{where}: positive payment with a non-positive allocation")
            if sum(ref_amts) > amt:
                err.append(f"{where}: allocations {sum(ref_amts)} exceed the payment amount {amt}")
            if cross and len(refs) != 1:
                err.append(f"{where}: cross-currency allocation in a payment with {len(refs)} references")
        else:
            if len(refs) != 1 or ref_amts[0] != amt:
                err.append(f"{where}: refund must carry exactly one allocation equal to its (negative) amount")
            if cross:
                err.append(f"{where}: refund in a currency other than the invoice currency (undefined by policy)")


# --------------------------------------------------------------------------- policy model
def _sig(r: dict) -> Tuple:
    return (r["account"], r["debit"], r["credit"], r["currency"], r["entity_id"], r["invoice_id"])


def _invoice_rows(inv: dict) -> List[Tuple]:
    c, e, i = inv["currency"], inv["entity_id"], inv["invoice_id"]
    rows = [(AR, inv["total_minor"], 0, c, e, i)]
    rows += [(li["revenue_account"], 0, li["amount_minor"], c, e, i) for li in inv["line_items"]]
    return rows


def _dup_sig(inv: dict) -> Tuple:
    items = tuple(sorted((str(li["description"]), li["quantity"], li["unit_price_minor"], li["amount_minor"],
                          li["revenue_account"]) for li in inv["line_items"]))
    return (inv["entity_id"], inv["currency"], inv["total_minor"], items)


def _doc_ts(doc: dict) -> datetime:
    return parse_ts(doc["issued_at"] if "issued_at" in doc else doc["timestamp"])


def _payment_entity(books: _Books, pay: dict, problems: List[str]) -> Optional[str]:
    refs = pay["invoice_refs"]
    if refs is None:
        return None
    ents = set()
    for ref in refs:
        inv = books.inv.get(ref["invoice_id"])
        if inv is None:
            problems.append(f"references unknown invoice {ref['invoice_id']}")
        else:
            ents.add(inv["entity_id"])
    if len(ents) != 1:
        problems.append(f"cannot determine a single entity (refs give {sorted(map(str, ents))})")
        return None
    return ents.pop()


def _payment_rows(ep: _Episode, books: _Books, pay: dict, problems: List[str]) -> Optional[List[Tuple]]:
    """Journal prescribed by the booking policy for one payment (row signatures, no timestamps)."""
    amt, ccy, refs = pay["amount_minor"], pay["currency"], pay["invoice_refs"]
    if refs is None:
        if amt <= 0:
            problems.append("negative payment without invoice reference")
            return None
        return [(BANK, amt, 0, ccy, None, None), (UNAPPLIED, 0, amt, ccy, None, None)]
    n_problems = len(problems)
    ent = _payment_entity(books, pay, problems)
    if len(problems) > n_problems:
        return None
    if amt < 0:
        if len(refs) != 1 or refs[0]["amount_minor"] != amt:
            problems.append("refund must have exactly one allocation equal to its amount")
            return None
        inv = books.inv[refs[0]["invoice_id"]]
        if inv["currency"] != ccy:
            problems.append("cross-currency refund is not defined by the policy")
            return None
        return [(BANK, 0, -amt, ccy, ent, None), (AR, -amt, 0, ccy, ent, inv["invoice_id"])]
    ts = _doc_ts(pay)
    rows = [(BANK, amt, 0, ccy, ent, None)]
    allocated = 0
    for ref in refs:
        inv_id, a = ref["invoice_id"], ref["amount_minor"]
        allocated += a
        tgt = books.inv[inv_id]["currency"]
        if tgt == ccy:
            rows.append((AR, 0, a, ccy, ent, inv_id))
            continue
        if len(refs) != 1:
            problems.append("cross-currency allocation in a multi-reference payment")
            return None
        rate, why = ep.policy_rate(ts, ccy, tgt)
        if rate is None:
            problems.append(why)
            return None
        y = convert_minor(a, ep.exp[ccy], ep.exp[tgt], rate[1])
        if y <= 0:
            problems.append(f"allocation {a} {ccy} converts to {y} {tgt}; no valid ledger row possible")
            return None
        rows += [(FXC, 0, a, ccy, ent, inv_id), (FXC, y, 0, tgt, ent, inv_id), (AR, 0, y, tgt, ent, inv_id)]
    rest = amt - allocated
    if rest < 0:
        problems.append(f"allocations exceed payment amount by {-rest}")
        return None
    if rest > 0:
        rows.append((UNAPPLIED, 0, rest, ccy, ent, None))
    return rows


def _finding(type_: str, affected: Iterable[str], amount: int, currency: str, variant: Optional[str],
             **detail: Any) -> dict:
    return {"type": type_, "affected": tuple(sorted(affected)), "amount": amount, "currency": currency,
            "variant": variant, "detail": detail}


def _fkey(f: dict) -> Tuple:
    return (f["type"], f["affected"], f["amount"], f["currency"])


def _ikey(issue: dict) -> Tuple:
    return (issue.get("type"), tuple(issue.get("affected_ids") or ()), issue.get("expected_amount_minor"),
            issue.get("currency"))


def _cutoff_variant(ep: _Episode, ts: datetime, booked: bool, kind: str) -> str:
    if ts < ep.start:
        side = "start"
    elif ts > ep.end:
        side = "end"
    else:
        side = "start" if ts - ep.start <= ep.end - ts else "end"
    return f"{'early' if booked else 'late'}_{side}_{kind}"


def _compare_journal(actual: Counter, expected: Counter, label: str, anomalies: List[str]) -> bool:
    if actual == expected:
        return True
    anomalies.append(f"{label}: journal differs from the booking policy; missing {_fmt_counter(expected - actual)}"
                     f", unexpected {_fmt_counter(actual - expected)}")
    return False


def _check_times(ep: _Episode, rows: List[dict], doc_ts: datetime, in_period: bool, label: str,
                 anomalies: List[str]) -> None:
    for r in rows:
        ts = r["ts"]
        if in_period:
            good = ts == doc_ts and ts.utcoffset() == ep.offset
            want = "the document instant in the company offset"
        else:  # out-of-period booking: wall-clock time of the document re-stamped with the company offset
            good = ts.utcoffset() == ep.offset and ts.replace(tzinfo=None) == doc_ts.replace(tzinfo=None)
            want = "the document wall-clock time re-stamped with the company offset"
        if not good:
            anomalies.append(f"{label}: row {r['row_id']} timestamp {r['timestamp']} is not {want}")


def _reconcile(ep: _Episode, books: _Books) -> Tuple[List[dict], List[str]]:
    """Own reconciliation of a set of books against the policy: (discrepancy findings, other anomalies)."""
    findings: List[dict] = []
    anomalies: List[str] = []
    for r in books.orphans:
        anomalies.append(f"ledger row {r['row_id']} has neither invoice_id nor payment_id")
    for (kind, ref), rows in books.journals.items():
        if kind == "I" and ref not in books.inv:
            anomalies.append(f"invoice journal {ref} has no invoice document")
        if kind == "P" and ref not in books.pay:
            anomalies.append(f"payment journal {ref} has no payment document")
        bal: Dict[str, int] = defaultdict(int)
        for r in rows:
            bal[r["currency"]] += r["debit"] - r["credit"]
        off = {c: v for c, v in bal.items() if v}
        if off:
            anomalies.append(f"journal {kind}:{ref} does not balance per currency {off}")
    for r in books.ledger:
        if not ep.in_period(r["ts"]):
            anomalies.append(f"row {r['row_id']} timestamp {r['timestamp']} lies outside the period")
        if r["ts"].utcoffset() != ep.offset:
            anomalies.append(f"row {r['row_id']} timestamp {r['timestamp']} is not in the company offset")
        if r["invoice_id"] is not None and r["invoice_id"] not in books.inv:
            anomalies.append(f"row {r['row_id']} references unknown invoice {r['invoice_id']}")

    booked_invoices = []
    for inv in books.invoices:
        rows = books.journals.get(("I", inv["invoice_id"]), [])
        ts = _doc_ts(inv)
        in_p = ep.in_period(ts)
        label = f"invoice {inv['invoice_id']}"
        if rows:
            booked_invoices.append(inv)
            _compare_journal(Counter(_sig(r) for r in rows), Counter(_invoice_rows(inv)), label, anomalies)
            _check_times(ep, rows, ts, in_p, label, anomalies)
        if bool(rows) != in_p:
            findings.append(_finding("period_cutoff", [inv["invoice_id"]], abs(inv["total_minor"]), inv["currency"],
                                     _cutoff_variant(ep, ts, bool(rows), "invoice"), doc=inv["doc_id"]))
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for inv in booked_invoices:
        groups[_dup_sig(inv)].append(inv)
    for group in groups.values():
        for a, b in combinations(group, 2):
            findings.append(_finding("duplicate_invoice", [a["invoice_id"], b["invoice_id"]], a["total_minor"],
                                     a["currency"], None, docs=sorted([a["doc_id"], b["doc_id"]])))

    for pay in books.payments:
        rows = books.journals.get(("P", pay["payment_id"]), [])
        ts = _doc_ts(pay)
        in_p = ep.in_period(ts)
        label = f"payment {pay['payment_id']}"
        if bool(rows) != in_p:
            findings.append(_finding("period_cutoff", [pay["payment_id"]], abs(pay["amount_minor"]), pay["currency"],
                                     _cutoff_variant(ep, ts, bool(rows), "payment"), doc=pay["doc_id"]))
        if not rows:
            continue
        _check_times(ep, rows, ts, in_p, label, anomalies)
        _analyze_payment(ep, books, pay, rows, ts, findings, anomalies)
    return findings, anomalies


def _analyze_payment(ep: _Episode, books: _Books, pay: dict, rows: List[dict], ts: datetime,
                     findings: List[dict], anomalies: List[str]) -> None:
    pid, amt, ccy, refs = pay["payment_id"], pay["amount_minor"], pay["currency"], pay["invoice_refs"]
    label = f"payment {pid}"
    problems: List[str] = []
    expected = _payment_rows(ep, books, pay, problems)
    anomalies.extend(f"{label}: {p}" for p in problems)
    if expected is None:
        return
    actual = Counter(_sig(r) for r in rows)
    if actual == Counter(expected):
        return
    if amt <= 0 or refs is None:  # no discrepancy type covers deviations of refunds / unidentified receipts
        _compare_journal(actual, Counter(expected), label, anomalies)
        return
    ent_policy = expected[0][4]
    ents = {r["entity_id"] for r in rows}
    if ents == {None}:
        ent = None
    elif ents == {ent_policy}:
        ent = ent_policy
    else:
        anomalies.append(f"{label}: rows carry entities {sorted(map(str, ents))}, expected {ent_policy}")
        return
    refmap = {r["invoice_id"]: r["amount_minor"] for r in refs}
    tgt_of = {i: books.inv[i]["currency"] for i in refmap}

    def net(acc: str, cur: str, inv_id: Optional[str], sign: int = 1) -> int:
        return sum(sign * (r["credit"] - r["debit"]) for r in rows
                   if r["account"] == acc and r["currency"] == cur and r["invoice_id"] == inv_id)

    applied, booked = {}, {}
    for inv_id in refmap:
        if tgt_of[inv_id] == ccy:
            applied[inv_id] = net(AR, ccy, inv_id)
        else:
            applied[inv_id] = net(FXC, ccy, inv_id)
            booked[inv_id] = net(AR, tgt_of[inv_id], inv_id)
    misapplied: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r["account"] == AR and r["currency"] == ccy and r["invoice_id"] is not None \
                and r["invoice_id"] not in refmap:
            misapplied[r["invoice_id"]] += r["credit"] - r["debit"]
    unapplied = net(UNAPPLIED, ccy, None)

    model = [(BANK, amt, 0, ccy, ent, None)]
    for inv_id in refmap:
        tgt = tgt_of[inv_id]
        if tgt == ccy:
            if applied[inv_id]:
                model.append((AR, 0, applied[inv_id], ccy, ent, inv_id))
        else:
            if applied[inv_id]:
                model.append((FXC, 0, applied[inv_id], ccy, ent, inv_id))
            if booked[inv_id]:
                model += [(FXC, booked[inv_id], 0, tgt, ent, inv_id), (AR, 0, booked[inv_id], tgt, ent, inv_id)]
    for j, c in misapplied.items():
        if c:
            model.append((AR, 0, c, ccy, ent, j))
    if unapplied:
        model.append((UNAPPLIED, 0, unapplied, ccy, ent, None))
    if Counter(model) != actual:
        anomalies.append(f"{label}: journal deviates from the policy in a way no discrepancy type explains; "
                         f"unexplained {_fmt_counter(actual - Counter(model))}, "
                         f"missing {_fmt_counter(Counter(model) - actual)}")
        return
    if ent is None and (any(applied.values()) or any(booked.values()) or any(misapplied.values())):
        anomalies.append(f"{label}: entity_id is null although cash was applied to invoices")
        return
    for j in misapplied:
        if j not in books.inv:
            anomalies.append(f"{label}: credits unknown invoice {j}")
    for inv_id, a in refmap.items():
        got = applied[inv_id]
        if got > a or got < 0:
            anomalies.append(f"{label}: applied {got} to {inv_id} although the allocation is {a}")
            continue
        if got < a:
            variant = "misapplied" if misapplied else ("missing" if got == 0 else "partial")
            findings.append(_finding("payment_allocation", [pid, inv_id], a - got, ccy, variant, payment=pid,
                                     invoice=inv_id, applied=got, misapplied_to=sorted(misapplied),
                                     entity_null=ent is None, unapplied=unapplied))
        tgt = tgt_of[inv_id]
        if tgt != ccy:
            rate, why = ep.policy_rate(ts, ccy, tgt)
            if rate is None:
                anomalies.append(f"{label}: {why}")
                continue
            correct = convert_minor(got, ep.exp[ccy], ep.exp[tgt], rate[1])
            if booked[inv_id] != correct:
                findings.append(_finding("fx_conversion", [pid, inv_id], abs(booked[inv_id] - correct), tgt, None,
                                         payment=pid, invoice=inv_id, applied=got, booked=booked[inv_id],
                                         correct=correct, rate_date=rate[0], source=ccy, target=tgt))


# --------------------------------------------------------------------------- checks on books
def _check_clean_balanced(ep: _Episode, err: List[str]) -> None:
    prov = ep.provenance
    added_rows = prov.get("added_ledger_rows")
    removed = prov.get("removed_ledger_rows")
    added_docs = prov.get("added_documents")
    if not isinstance(added_rows, list) or not all(isinstance(x, str) for x in added_rows):
        err.append("provenance.added_ledger_rows must be a list of row ids")
        return
    if not isinstance(removed, list) or not all(isinstance(x, dict) for x in removed):
        err.append("provenance.removed_ledger_rows must be a list of ledger rows")
        return
    if not isinstance(added_docs, list) or not all(isinstance(x, str) for x in added_docs):
        err.append("provenance.added_documents must be a list of doc ids")
        return
    pub_ids = {r["row_id"] for r in ep.ledger}
    for rid in added_rows:
        if rid not in pub_ids:
            err.append(f"provenance.added_ledger_rows: {rid} is not in the public ledger")
    if len(set(added_rows)) != len(added_rows):
        err.append("provenance.added_ledger_rows contains duplicates")
    added = set(added_rows)
    kept = [r for r in ep.ledger if r["row_id"] not in added]
    restored = []
    for n, raw in enumerate(removed, start=1):
        row = _parse_ledger_row(raw, f"provenance.removed_ledger_rows[{n}]", ep, err, from_csv=False)
        if row is not None:
            restored.append(row)
    clean_ledger = kept + restored
    dup = [k for k, v in Counter(r["row_id"] for r in clean_ledger).items() if v > 1]
    if dup:
        err.append(f"clean ledger has duplicate row ids {dup[:5]}")
    doc_ids = {d["doc_id"] for d in ep.invoices + ep.payments}
    for d in added_docs:
        if d not in doc_ids:
            err.append(f"provenance.added_documents: {d} is not a public document")
    gone = set(added_docs)
    ep.clean = _Books(clean_ledger, [d for d in ep.invoices if d["doc_id"] not in gone],
                      [d for d in ep.payments if d["doc_id"] not in gone])
    for (kind, ref), rows in sorted(ep.clean.journals.items()):
        bal: Dict[str, int] = defaultdict(int)
        for r in rows:
            bal[r["currency"]] += r["debit"] - r["credit"]
        off = {c: v for c, v in bal.items() if v}
        if off:
            err.append(f"clean journal {kind}:{ref} does not balance per currency {off}")
    for r in ep.clean.orphans:
        err.append(f"clean ledger row {r['row_id']} belongs to no journal")
    tb: Dict[str, int] = defaultdict(int)
    for r in clean_ledger:
        tb[r["currency"]] += r["debit"] - r["credit"]
    off = {c: v for c, v in tb.items() if v}
    if off:
        err.append(f"clean trial balance is not zero per currency: {off}")


def _check_clean_policy(ep: _Episode, err: List[str]) -> None:
    if ep.clean is None:
        err.append("skipped: clean books could not be reconstructed")
        return
    findings, anomalies = _reconcile(ep, ep.clean)
    for f in findings:
        err.append(f"clean books contain a {f['type']} discrepancy {list(f['affected'])} "
                   f"{f['amount']} {f['currency']} ({f['variant']})")
    for a in anomalies:
        err.append(f"clean books: {a}")


def _check_declared_present(ep: _Episode, err: List[str]) -> None:
    findings, _ = ep.public_recon()
    keys = {_fkey(f) for f in findings}
    for issue in _issues(ep):
        if _ikey(issue) in keys:
            continue
        near = [f for f in findings if f["type"] == issue.get("type")
                and f["affected"] == tuple(issue.get("affected_ids") or ())]
        hint = (f"; reconciliation found amount {near[0]['amount']} {near[0]['currency']}" if near
                else "; reconciliation found nothing for these ids")
        err.append(f"{issue.get('issue_id')}: declared {issue.get('type')} {issue.get('affected_ids')} "
                   f"{issue.get('expected_amount_minor')} {issue.get('currency')} is not reproduced{hint}")


def _check_truth_complete(ep: _Episode, err: List[str]) -> None:
    findings, anomalies = ep.public_recon()
    declared = {_ikey(i) for i in _issues(ep)}
    for f in findings:
        if _fkey(f) not in declared:
            err.append(f"undeclared {f['type']} discrepancy {list(f['affected'])} {f['amount']} {f['currency']}"
                       f" ({f['variant']})")
    for a in anomalies:
        err.append(f"public books: {a}")


# --------------------------------------------------------------------------- evidence
def _resolve(ep: _Episode, ident: str) -> Optional[Tuple[str, dict]]:
    if ep.id_index is None:
        idx: Dict[str, Tuple[str, dict]] = {}
        for r in ep.ledger:
            idx[r["row_id"]] = ("row", r)
        for d in ep.invoices:
            idx[d["doc_id"]] = ("invoice", d)
        for d in ep.payments:
            idx[d["doc_id"]] = ("payment", d)
        for f in ep.fx:
            idx[f["id"]] = ("fx", f)
        alias = {}
        for d in ep.invoices:
            alias[d["invoice_id"]] = ("invoice", d)
        for d in ep.payments:
            alias[d["payment_id"]] = ("payment", d)
        for k, v in alias.items():
            idx.setdefault(k, v)
        ep.id_index = idx
    return ep.id_index.get(ident)


def _verify_duplicate(ep: _Episode, invs: List[dict], pays: List[dict], rows: List[dict], fxs: List[dict]):
    if len(invs) != 2 or len(rows) != 2 or pays or fxs:
        return [], "needs exactly 2 invoice documents + 2 ledger rows"
    a, b = invs
    if a["invoice_id"] == b["invoice_id"] or _dup_sig(a) != _dup_sig(b):
        return [], "the two invoices are not identical in entity, currency, total and line items"
    for inv in (a, b):
        own = [r for r in rows if r["payment_id"] is None and r["invoice_id"] == inv["invoice_id"]]
        if len(own) != 1:
            return [], f"needs exactly one booking row of invoice {inv['invoice_id']}"
        r = own[0]
        if _sig(r) not in set(_invoice_rows(inv)) or not ep.in_period(r["ts"]):
            return [], f"row {r['row_id']} is not a booking row of {inv['invoice_id']} in this period"
    return [((tuple(sorted([a["invoice_id"], b["invoice_id"]]))), a["total_minor"], a["currency"])], ""


def _verify_allocation(ep: _Episode, invs: List[dict], pays: List[dict], rows: List[dict], fxs: List[dict]):
    if len(pays) != 1 or len(rows) != 1 or invs or fxs:
        return [], "needs exactly 1 payment document + 1 ledger row"
    p, r = pays[0], rows[0]
    amt, ccy, refs = p["amount_minor"], p["currency"], p["invoice_refs"]
    if amt <= 0 or not refs:
        return [], "payment has no positive allocation"
    if r["payment_id"] != p["payment_id"] or r["currency"] != ccy:
        return [], f"row {r['row_id']} is not a payment-currency row of {p['payment_id']}"
    refmap = {x["invoice_id"]: x["amount_minor"] for x in refs}
    net = r["credit"] - r["debit"]
    cands: List[Tuple[str, int]] = []
    if r["account"] == UNAPPLIED and r["invoice_id"] is None:
        on_account = amt - sum(refmap.values())
        if net == amt:  # the whole payment sits in unapplied cash: no allocation was applied
            cands = list(refmap.items())
        elif len(refmap) == 1:
            (inv_id, a), = refmap.items()
            excess = net - on_account
            if 0 < excess <= a:
                cands = [(inv_id, excess)]
    elif r["account"] in (AR, FXC) and r["invoice_id"] in refmap:
        a = refmap[r["invoice_id"]]
        if 0 <= net < a:
            cands = [(r["invoice_id"], a - net)]
    elif r["account"] == AR and r["invoice_id"] is not None and r["invoice_id"] not in refmap and net > 0:
        if len(refmap) == 1:
            (inv_id, a), = refmap.items()
            cands = [(inv_id, min(a, net))]
        else:
            cands = [(i, a) for i, a in refmap.items() if a == net]
    if not cands:
        return [], f"row {r['row_id']} does not show an unapplied, partially applied or misapplied allocation"
    return [(tuple(sorted([p["payment_id"], i])), amount, ccy) for i, amount in cands], ""


def _verify_fx(ep: _Episode, invs: List[dict], pays: List[dict], rows: List[dict], fxs: List[dict]):
    if len(pays) != 1 or len(fxs) != 1 or invs or not 1 <= len(rows) <= 2:
        return [], "needs exactly 1 payment document + 1 fx record + 1-2 ledger rows"
    p, f = pays[0], fxs[0]
    refs, amt, src = p["invoice_refs"], p["amount_minor"], p["currency"]
    if amt <= 0 or not refs or len(refs) != 1:
        return [], "payment must carry exactly one positive allocation"
    inv_id, a = refs[0]["invoice_id"], refs[0]["amount_minor"]
    tgt = f["target_currency"]
    if f["source_currency"] != src or tgt == src:
        return [], f"fx record {f['id']} is not payment currency -> other currency"
    want = weekday_rate_date(ep.local_date(_doc_ts(p)))
    if f["date"] != want:
        return [], f"fx record {f['id']} is not the policy rate date {want}"
    if any(r["payment_id"] != p["payment_id"] or r["invoice_id"] != inv_id for r in rows):
        return [], "rows must belong to the payment and carry the referenced invoice"
    t_rows = [r for r in rows if r["currency"] == tgt]
    s_rows = [r for r in rows if r["currency"] == src]
    if len(t_rows) != 1 or len(s_rows) + 1 != len(rows):
        return [], "needs exactly one invoice-currency row (and optionally one payment-currency row)"
    t = t_rows[0]
    if t["account"] == AR:
        booked = t["credit"] - t["debit"]
    elif t["account"] == FXC:
        booked = t["debit"] - t["credit"]
    else:
        return [], f"row {t['row_id']} is neither the AR credit nor the FX-clearing debit"
    if s_rows:
        s = s_rows[0]
        if s["account"] != FXC:
            return [], f"row {s['row_id']} is not the 1950-FX-CLEARING row in payment currency"
        applied = s["credit"] - s["debit"]
    else:
        applied = a
    if not 0 < applied <= a:
        return [], "applied amount is not within the allocation"
    if src not in ep.exp or tgt not in ep.exp:
        return [], "currency exponents unknown"
    correct = convert_minor(applied, ep.exp[src], ep.exp[tgt], f["rate"])
    if booked == correct:
        return [], "booked amount equals the policy conversion"
    return [(tuple(sorted([p["payment_id"], inv_id])), abs(booked - correct), tgt)], ""


def _verify_cutoff(ep: _Episode, invs: List[dict], pays: List[dict], rows: List[dict], fxs: List[dict]):
    docs = invs + pays
    if len(docs) != 1 or fxs or len(rows) > 1:
        return [], "needs exactly 1 document (+ at most 1 ledger row)"
    d = docs[0]
    is_inv = bool(invs)
    ts = _doc_ts(d)
    own_id = d["invoice_id"] if is_inv else d["payment_id"]
    amount = abs(d["total_minor"] if is_inv else d["amount_minor"])
    if rows:
        r = rows[0]
        belongs = (r["payment_id"] is None and r["invoice_id"] == own_id) if is_inv else r["payment_id"] == own_id
        if not belongs:
            return [], f"row {r['row_id']} is not part of the journal of {own_id}"
        if ep.in_period(ts):
            return [], f"{own_id} lies inside the period"
        if not ep.in_period(r["ts"]):
            return [], f"row {r['row_id']} is not booked in this period"
    else:
        if not ep.in_period(ts):
            return [], f"{own_id} lies outside the period and is not booked"
        assert ep.pub is not None
        if ep.pub.journals.get(("I" if is_inv else "P", own_id)):
            return [], f"{own_id} is booked in the ledger"
    return [((own_id,), amount, d["currency"])], ""


_VERIFIERS: Dict[str, Callable] = {"duplicate_invoice": _verify_duplicate, "payment_allocation": _verify_allocation,
                                   "fx_conversion": _verify_fx, "period_cutoff": _verify_cutoff}


def verify_evidence_set(ep: _Episode, issue: dict, ids: Sequence[str]) -> Tuple[bool, str]:
    """Re-derive ``issue`` from the records of one evidence set only (plus manifest/policy)."""
    if not isinstance(ids, list) or not ids or len(ids) > MAX_EVIDENCE_IDS:
        return False, f"evidence set must hold 1..{MAX_EVIDENCE_IDS} ids"
    if len(set(ids)) != len(ids):
        return False, "evidence set repeats an id"
    buckets: Dict[str, List[dict]] = {"invoice": [], "payment": [], "row": [], "fx": []}
    for i in ids:
        hit = _resolve(ep, i) if isinstance(i, str) else None
        if hit is None:
            return False, f"id {i!r} does not resolve to a public record"
        buckets[hit[0]].append(hit[1])
    verifier = _VERIFIERS.get(issue.get("type"))  # type: ignore[arg-type]
    if verifier is None:
        return False, f"unknown issue type {issue.get('type')!r}"
    cands, why = verifier(ep, buckets["invoice"], buckets["payment"], buckets["row"], buckets["fx"])
    target = (tuple(issue.get("affected_ids") or ()), issue.get("expected_amount_minor"), issue.get("currency"))
    if target in cands:
        return True, ""
    if cands:
        return False, f"evidence derives {cands} instead of {target}"
    return False, why


def _check_evidence(ep: _Episode, err: List[str]) -> None:
    for issue in _issues(ep):
        sets = issue.get("evidence_sets")
        if not isinstance(sets, list) or not sets:
            err.append(f"{issue.get('issue_id')}: no evidence sets")
            continue
        for k, ids in enumerate(sets):
            ok, why = verify_evidence_set(ep, issue, ids)
            if not ok:
                err.append(f"{issue.get('issue_id')} evidence set {k} {ids}: {why}")


# --------------------------------------------------------------------------- truth / variants / provenance
def _issues(ep: _Episode) -> List[dict]:
    issues = ep.truth.get("issues")
    if not isinstance(issues, list):
        return []
    return [i for i in issues if isinstance(i, dict)]


def _check_truth(ep: _Episode, err: List[str]) -> None:
    t = ep.truth
    if set(t) != {"episode_id", "issues", "schema_version"}:
        err.append(f"truth.json keys must be episode_id, issues, schema_version; got {sorted(t)}")
    if t.get("episode_id") != ep.manifest.get("episode_id"):
        err.append("truth.episode_id differs from manifest.episode_id")
    issues = t.get("issues")
    if not isinstance(issues, list):
        err.append("truth.issues must be a list")
        return
    if len(issues) > MAX_ISSUES:
        err.append(f"truth has {len(issues)} issues (> {MAX_ISSUES})")
    known_ids = {d["invoice_id"] for d in ep.invoices} | {d["payment_id"] for d in ep.payments}
    seen_ids, seen_keys = set(), set()
    groups: Dict[str, List[dict]] = defaultdict(list)
    for n, issue in enumerate(issues):
        where = f"issue[{n}]"
        if not isinstance(issue, dict):
            err.append(f"{where}: not an object")
            continue
        if set(issue) != ISSUE_KEYS:
            err.append(f"{where}: keys must be {sorted(ISSUE_KEYS)}")
            continue
        iid = issue["issue_id"]
        where = f"issue {iid}"
        if not _is_id(iid) or iid in seen_ids:
            err.append(f"{where}: issue_id missing or not unique")
        seen_ids.add(iid)
        typ = issue["type"]
        if typ not in ISSUE_TYPES:
            err.append(f"{where}: unknown type {typ!r}")
            continue
        aff = issue["affected_ids"]
        if not isinstance(aff, list) or not aff or not all(isinstance(x, str) for x in aff):
            err.append(f"{where}: affected_ids must be a non-empty list of ids")
            continue
        if aff != sorted(aff) or len(set(aff)) != len(aff):
            err.append(f"{where}: affected_ids must be sorted and unique")
        if len(aff) > MAX_AFFECTED_IDS or len(aff) != AFFECTED_COUNT[typ]:
            err.append(f"{where}: {typ} needs {AFFECTED_COUNT[typ]} affected ids, got {len(aff)}")
        unknown = [x for x in aff if x not in known_ids]
        if unknown:
            err.append(f"{where}: affected ids {unknown} are not invoice_ids/payment_ids of public documents")
        amt = issue["expected_amount_minor"]
        if not _is_int(amt) or amt <= 0:
            err.append(f"{where}: expected_amount_minor must be a positive integer")
        elif issue["tolerance_minor"] != tolerance_for(amt) or not _is_int(issue["tolerance_minor"]):
            err.append(f"{where}: tolerance_minor {issue['tolerance_minor']!r} != published tolerance 0")
        if issue["currency"] not in ep.exp:
            err.append(f"{where}: currency {issue['currency']!r} not in manifest")
        sets = issue["evidence_sets"]
        if not isinstance(sets, list) or not sets:
            err.append(f"{where}: evidence_sets must be a non-empty list")
        else:
            norm = set()
            for s in sets:
                if not isinstance(s, list) or not s or len(s) > MAX_EVIDENCE_IDS \
                        or not all(isinstance(x, str) for x in s) or len(set(s)) != len(s):
                    err.append(f"{where}: each evidence set must hold 1..{MAX_EVIDENCE_IDS} distinct ids")
                    continue
                fs = frozenset(s)
                if fs in norm:
                    err.append(f"{where}: repeated evidence set {s}")
                norm.add(fs)
        if issue["interaction_id"] is not None:
            if not _is_id(issue["interaction_id"]):
                err.append(f"{where}: interaction_id must be null or an id")
            else:
                groups[issue["interaction_id"]].append(issue)
        var = issue["variant"]
        if typ == "duplicate_invoice" and var is not None:
            err.append(f"{where}: duplicate_invoice variant must be null")
        if typ == "payment_allocation" and var not in ALLOCATION_VARIANTS:
            err.append(f"{where}: payment_allocation variant must be one of {ALLOCATION_VARIANTS}")
        if typ == "fx_conversion" and var not in FX_VARIANTS:
            err.append(f"{where}: fx_conversion variant must be one of {FX_VARIANTS}")
        if typ == "period_cutoff" and not (isinstance(var, str) and re.match(
                r"^(early|late)_(start|end)_(invoice|payment)$", var)):
            err.append(f"{where}: period_cutoff variant must look like early|late_start|end_invoice|payment")
        if not isinstance(issue["description"], str) or not issue["description"]:
            err.append(f"{where}: description must be a non-empty string")
        key = (typ, tuple(aff))
        if key in seen_keys:
            err.append(f"{where}: another issue has the same type and affected ids")
        seen_keys.add(key)
    for gid, members in groups.items():
        kinds = sorted(((m["type"], m["variant"]) for m in members), key=repr)
        if kinds == [("duplicate_invoice", None), ("payment_allocation", "misapplied")]:
            dup = next(m for m in members if m["type"] == "duplicate_invoice")
            alloc = next(m for m in members if m["type"] == "payment_allocation")
            if not set(dup["affected_ids"]) & set(alloc["affected_ids"]):
                err.append(f"interaction {gid}: misapplied payment does not reference one of the duplicates")
        elif len(members) == 2 and {m["type"] for m in members} == {"payment_allocation", "fx_conversion"}:
            alloc = next(m for m in members if m["type"] == "payment_allocation")
            fx = next(m for m in members if m["type"] == "fx_conversion")
            if alloc["variant"] != "partial" or fx["affected_ids"] != alloc["affected_ids"]:
                err.append(f"interaction {gid}: partial allocation and fx conversion must share payment and invoice")
        else:
            err.append(f"interaction {gid}: unsupported combination {kinds}")


def _check_variants(ep: _Episode, err: List[str]) -> None:
    findings, _ = ep.public_recon()
    by_key = {_fkey(f): f for f in findings}
    assert ep.pub is not None
    pub = ep.pub
    matched: Dict[str, dict] = {}
    for issue in _issues(ep):
        f = by_key.get(_ikey(issue))
        if f is None:
            continue  # reported by declared_discrepancies_present
        matched[str(issue.get("issue_id"))] = f
        where = f"issue {issue.get('issue_id')}"
        typ, var = issue.get("type"), issue.get("variant")
        if typ in ("payment_allocation", "period_cutoff") and var != f["variant"]:
            err.append(f"{where}: variant {var!r} but the records show {f['variant']!r}")
        if typ == "payment_allocation" and var == "missing" and not f["detail"].get("entity_null"):
            err.append(f"{where}: 'missing' allocation should leave the payment with entity null")
        if typ == "fx_conversion":
            d = f["detail"]
            pay = pub.pay[d["payment"]]
            inv = pub.inv[d["invoice"]]
            pts, its = _doc_ts(pay), _doc_ts(inv)
            pol = d["rate_date"]
            alt = {
                "previous_business_day": {_prev_business_day(pol), _prev_business_day(ep.local_date(pts))},
                "next_business_day": {_next_business_day(pol), _next_business_day(ep.local_date(pts))},
                "document_wall_clock_date": {pts.date()},
                "invoice_issue_date": {ep.local_date(its), its.date()},
            }.get(var, set())
            explained = False
            if var == "reverse_pair_rate":
                found = ep.lookup_rate(pol, d["target"], d["source"])
                if found and found[0] == pol and convert_minor(d["applied"], ep.exp[d["source"]],
                                                               ep.exp[d["target"]], found[1]) == d["booked"]:
                    explained = True
            for day in alt:
                found = ep.lookup_rate(day, d["source"], d["target"])
                if found and found[0] != pol and convert_minor(d["applied"], ep.exp[d["source"]],
                                                               ep.exp[d["target"]], found[1]) == d["booked"]:
                    explained = True
            if not explained:
                err.append(f"{where}: booked amount {d['booked']} is not the conversion at the {var!r} rate")
    groups: Dict[str, List[dict]] = defaultdict(list)
    for issue in _issues(ep):
        if issue.get("interaction_id") is not None:
            groups[str(issue.get("interaction_id"))].append(issue)
    for gid, members in groups.items():
        fs = {m.get("type"): matched.get(str(m.get("issue_id"))) for m in members}
        if None in fs.values() or len(fs) != len(members):
            continue
        if set(fs) == {"duplicate_invoice", "payment_allocation"}:
            dup, alloc = fs["duplicate_invoice"], fs["payment_allocation"]
            ref_inv = alloc["detail"]["invoice"]
            others = set(dup["affected"]) - {ref_inv}
            if ref_inv not in dup["affected"] or set(alloc["detail"]["misapplied_to"]) != others:
                err.append(f"interaction {gid}: payment is not misapplied to the duplicate of its invoice")
        elif set(fs) == {"payment_allocation", "fx_conversion"}:
            pay = pub.pay[fs["fx_conversion"]["detail"]["payment"]]
            pts = _doc_ts(pay)
            if pts.date() == ep.local_date(pts):
                err.append(f"interaction {gid}: payment is not at a date boundary (wall-clock date = company date)")
            if fs["fx_conversion"]["detail"]["applied"] != fs["payment_allocation"]["detail"]["applied"]:
                err.append(f"interaction {gid}: fx amount not computed on the applied part")


def _check_provenance(ep: _Episode, err: List[str]) -> None:
    prov = ep.provenance
    if prov.get("episode_id", ep.manifest.get("episode_id")) != ep.manifest.get("episode_id"):
        err.append("provenance.episode_id differs from manifest.episode_id")
    changes = prov.get("changes")
    if not isinstance(changes, list):
        err.append("provenance.changes must be a list")
        return
    assert ep.pub is not None
    pub = ep.pub
    doc_ids = {d["doc_id"] for d in ep.invoices + ep.payments}
    change_docs, named_added = set(), set()
    for n, c in enumerate(changes):
        if not isinstance(c, dict) or not isinstance(c.get("op"), str) or not isinstance(c.get("reason"), str) \
                or not isinstance(c.get("journal"), dict) or not isinstance(c["journal"].get("document"), str):
            err.append(f"provenance.changes[{n}] must be {{op, reason, journal: {{document}}}}")
            continue
        doc = c["journal"]["document"]
        if doc not in doc_ids:
            err.append(f"provenance.changes[{n}]: unknown document {doc}")
        change_docs.add(doc)
        if c["op"].startswith("add_document"):
            named_added.update(c.get("documents") or [])
    added_docs = set(prov.get("added_documents") or [])
    if added_docs != named_added:
        err.append(f"added_documents {sorted(added_docs)} != documents added by changes {sorted(named_added)}")

    def doc_of(r: dict) -> Optional[str]:
        if r.get("payment_id") is not None:
            d = pub.pay.get(r["payment_id"])
        else:
            d = pub.inv.get(r.get("invoice_id"))
        return d["doc_id"] if d else None

    added = set(prov.get("added_ledger_rows") or [])
    touched = {doc_of(r) for r in ep.ledger if r["row_id"] in added}
    touched |= {doc_of(r) for r in prov.get("removed_ledger_rows") or [] if isinstance(r, dict)}
    if None in touched:
        err.append("an injected/removed ledger row belongs to no public document")
        touched.discard(None)
    if touched != change_docs:
        err.append(f"journals changed by injection {sorted(map(str, touched))} != journals named in changes "
                   f"{sorted(change_docs)}")
    issue_docs: Dict[str, set] = {}
    alias = {d["invoice_id"]: d["doc_id"] for d in ep.invoices}
    alias.update({d["payment_id"]: d["doc_id"] for d in ep.payments})
    for issue in _issues(ep):
        docs = {alias.get(x) for x in issue.get("affected_ids") or []} - {None}
        issue_docs[str(issue.get("issue_id"))] = docs
        if not docs & change_docs:
            err.append(f"issue {issue.get('issue_id')}: no injected change touches its documents")
    all_issue_docs = set().union(*issue_docs.values()) if issue_docs else set()
    for doc in sorted(change_docs - all_issue_docs):
        err.append(f"injected change on {doc} is not covered by any truth issue")


def _check_leak(ep: _Episode, err: List[str]) -> None:
    pub_dir = ep.path / "public"
    extra = sorted(p.name for p in pub_dir.iterdir() if p.name not in PUBLIC_FILES)
    if extra:
        err.append(f"unexpected files in public/: {extra}")
    ids = [i["issue_id"] for i in _issues(ep) if isinstance(i.get("issue_id"), str) and i["issue_id"]]
    pats = [(iid, re.compile(r"(?<![A-Za-z0-9_])" + re.escape(iid) + r"(?![A-Za-z0-9_])")) for iid in ids]
    for name, text in ep.raw.items():
        for iid, pat in pats:
            if pat.search(text):
                err.append(f"public/{name} contains truth issue id {iid}")
        low = text.lower()
        for s in LEAK_STRINGS:
            if s in low:
                err.append(f"public/{name} contains the private marker {s!r}")


# --------------------------------------------------------------------------- driver
_STEPS: Tuple[Tuple[str, Callable[[_Episode, List[str]], None]], ...] = (
    ("format", _check_format),
    ("documents_consistent", _check_documents),
    ("truth_wellformed", _check_truth),
    ("no_private_leak", _check_leak),
    ("clean_books_balanced", _check_clean_balanced),
    ("clean_books_follow_policy", _check_clean_policy),
    ("declared_discrepancies_present", _check_declared_present),
    ("truth_complete", _check_truth_complete),
    ("evidence_sufficient", _check_evidence),
    ("variants_consistent", _check_variants),
    ("provenance_consistent", _check_provenance),
)


def validate_episode_dir(path: str | Path) -> Dict[str, Any]:
    """Validate one exported episode directory. Never raises for bad data."""
    ep = _Episode(Path(path))
    errs: Dict[str, List[str]] = {c: [] for c in CHECKS}
    try:
        usable = _load(ep, errs["format"])
    except Exception as exc:  # defensive: the loader must never crash the run
        errs["format"].append(f"loader crashed: {type(exc).__name__}: {exc}")
        usable = False
    for name, fn in _STEPS:
        if not usable:
            if name != "format":
                errs[name].append("skipped: episode files could not be loaded")
            continue
        try:
            fn(ep, errs[name])
        except Exception as exc:  # a crash on malformed data counts as a failed check
            errs[name].append(f"internal error while checking: {type(exc).__name__}: {exc}")
    checks = {c: not errs[c] for c in CHECKS}
    errors: List[str] = []
    for c in CHECKS:
        msgs = errs[c]
        errors += [f"{c}: {m}" for m in msgs[:_MAX_ERRORS_PER_CHECK]]
        if len(msgs) > _MAX_ERRORS_PER_CHECK:
            errors.append(f"{c}: ... {len(msgs) - _MAX_ERRORS_PER_CHECK} more")
    return {"episode_id": ep.manifest.get("episode_id") if isinstance(ep.manifest, dict) else None,
            "path": str(path), "ok": all(checks.values()), "checks": checks, "errors": errors}


def find_episode_dirs(root: str | Path) -> List[Path]:
    """Directories under ``root`` (including ``root``) that contain both ``public/`` and ``private/``."""
    out: List[Path] = []

    def walk(d: Path) -> None:
        if (d / "public").is_dir() and (d / "private").is_dir():
            out.append(d)
            return
        for child in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")):
            walk(child)

    root = Path(root)
    if root.is_dir():
        walk(root)
    return out


def validate_tree(root: str | Path) -> Dict[str, Any]:
    results = [validate_episode_dir(d) for d in find_episode_dirs(root)]
    return {"root": str(root), "episodes": len(results), "ok": sum(1 for r in results if r["ok"]),
            "failed": [r["path"] for r in results if not r["ok"]], "results": results}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m auditenv.validate_data",
                                 description="Independently validate exported audit episodes.")
    ap.add_argument("root", help="directory holding episode dirs (each with public/ and private/)")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--max-errors", type=int, default=5, help="errors shown per failing episode")
    args = ap.parse_args(list(argv) if argv is not None else None)
    summary = validate_tree(args.root)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for r in summary["results"]:
            status = "OK  " if r["ok"] else "FAIL"
            failed = [c for c, v in r["checks"].items() if not v]
            print(f"{status} {r['episode_id']}  {r['path']}" + (f"  failed: {', '.join(failed)}" if failed else ""))
            for e in r["errors"][:max(0, args.max_errors)]:
                print(f"       - {e}")
        print(f"{summary['ok']}/{summary['episodes']} episodes valid")
    if summary["episodes"] == 0:
        print(f"no episode directories found under {args.root}", file=sys.stderr)
        return 2
    return 0 if summary["ok"] == summary["episodes"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
