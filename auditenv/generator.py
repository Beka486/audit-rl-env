"""Episode generator.

Order of construction (each stage only reads the previous ones):

1. company, currencies, period, timezone, FX table (synthetic random walk);
2. hidden *history*: customers, invoices, payments (consistent by construction);
3. legitimate distractors (split payments, refunds, recurring subscriptions, ...);
4. hosts for the requested errors, built with the same procedures as legit documents;
5. clean books: the ledger exactly as the booking policy prescribes;
6. error injection: every change to documents/ledger is recorded (provenance) and
   every resulting discrepancy is described in truth with its evidence sets;
7. finalisation: IDs, ordering, public tables, truth.json, provenance.json;
8. self-check: an independent reconciliation of the *public* records must reproduce
   truth exactly, and the clean books must reconcile with zero discrepancies.
"""
from __future__ import annotations

import calendar
import copy
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext, Context, ROUND_HALF_EVEN
from typing import Any, Dict, List, Optional, Set, Tuple

from . import money
from .config import (CURRENCY_POOL, DISTRACTORS, ERROR_TYPES, INTERACTIONS, TIMEZONE_POOL,
                     canonical_json, normalize_config, sample_config)
from .episode import Episode
from .policy_text import ACCOUNTS, build_manifest, build_policy
from .rng import CROCKFORD, Rng, derive_hex
from .timeutil import (fmt, is_business_day, local_date, next_business_day, offset_str,
                       parse_offset, prev_business_day, rate_date_for, render, tz)
from .version import ENV_VERSION, GENERATOR_VERSION, SCHEMA_VERSION

BANK, AR, FXC, UNAPPLIED = ACCOUNTS["bank"], ACCOUNTS["ar"], ACCOUNTS["fx_clearing"], ACCOUNTS["unapplied"]
REV = {"services": ACCOUNTS["revenue_services"], "subscriptions": ACCOUNTS["revenue_subscriptions"],
       "goods": ACCOUNTS["revenue_goods"]}

DOC_OFFSETS = [-480, -300, -180, 0, 60, 180, 240, 300, 330, 360, 480, 540, 600, 720, 780]
MIN_FX_DIFF = 20  # minor units: an FX error must move the booked amount at least this much

# Approximate starting levels (units per 1 abstract numeraire). Every episode perturbs
# them and then runs its own random walk, so no real market data is used.
LEVELS = {"USD": "1", "EUR": "0.92", "GBP": "0.79", "KZT": "505", "JPY": "152", "KWD": "0.307",
          "CNY": "7.2", "AED": "3.67", "TRY": "34", "INR": "84", "UZS": "12700", "CHF": "0.88"}

PRODUCTS = [
    ("Consulting hours", "services", 90, 260), ("Data migration service", "services", 800, 4200),
    ("Cloud storage plan 1TB", "services", 40, 120), ("Hardware maintenance visit", "services", 150, 900),
    ("Freight forwarding", "services", 300, 2500), ("Warehouse handling", "services", 60, 400),
    ("Office supplies pack", "goods", 25, 140), ("Industrial sensor unit", "goods", 120, 950),
    ("Packaging materials pallet", "goods", 80, 600), ("Software license seat", "services", 30, 220),
    ("Training workshop day", "services", 500, 3000), ("Customs brokerage fee", "services", 100, 700),
    ("Spare parts kit", "goods", 45, 520), ("Printing services batch", "services", 70, 480),
    ("Solar panel module", "goods", 180, 1100), ("Lab analysis report", "services", 140, 860),
]
PLANS = [("Basic", 40, 90), ("Pro", 120, 260), ("Enterprise", 400, 900)]
SYLLABLES = ["ka", "ra", "tel", "mon", "vi", "sor", "an", "del", "ta", "ri", "no", "bel", "gar", "zen",
             "lo", "mir", "dan", "sel", "tor", "vel", "qa", "sa", "nur", "ast", "er", "ul", "bo", "ki"]
INDUSTRIES = ["Logistics", "Trading", "Systems", "Foods", "Engineering", "Textiles", "Analytics",
              "Agro", "Energy Services", "Media", "Instruments", "Supply"]
LEGAL = ["LLP", "Ltd", "JSC", "GmbH", "Inc", "S.A.", "Group"]
INVOICE_PREFIX_LETTERS = "ABEGHKMNSTVWXYZ"  # never D/L/P/C/F/R so namespaces cannot collide


class RetryGeneration(Exception):
    """Raised when a random draw cannot be satisfied; the build retries with a new attempt index."""


class GenerationError(RuntimeError):
    pass


@dataclass
class Customer:
    key: str
    currency: str
    offset: int
    alt_currency: str
    plan: Optional[Tuple[str, int]] = None  # (plan name, price minor) for subscriptions
    entity_id: str = ""


@dataclass
class Invoice:
    key: str
    cust: str
    issued: datetime  # aware, UTC
    offset: int
    terms_days: int
    currency: str
    lines: List[Dict[str, Any]]
    tags: Set[str] = field(default_factory=set)
    doc_id: str = ""
    invoice_id: str = ""

    @property
    def total(self) -> int:
        return sum(li["amount_minor"] for li in self.lines)


@dataclass
class Payment:
    key: str
    ts: datetime  # aware, UTC
    offset: int
    currency: str
    amount: int
    refs: Optional[List[Tuple[str, int]]]
    payer: Optional[str]
    tags: Set[str] = field(default_factory=set)
    doc_id: str = ""
    payment_id: str = ""


@dataclass
class Row:
    key: str
    ts: datetime  # aware, company offset
    account: str
    debit: int
    credit: int
    currency: str
    entity: Optional[str]
    invoice: Optional[str]
    payment: Optional[str]
    journal: str
    seq: int
    row_id: str = ""

    def sig(self) -> tuple:
        return (self.ts, self.account, self.debit, self.credit, self.currency, self.entity,
                self.invoice, self.payment)


def generate_episode(seed: int, config: Optional[Dict[str, Any]] = None, secret: str = "") -> Episode:
    """Generate one episode. Same (seed, config, secret, version) -> identical episode."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    cfg = normalize_config(config)
    base_cfg = {k: v for k, v in cfg.items() if k != "transform"}  # a transform never changes the base episode
    key = derive_hex("auditenv-episode", GENERATOR_VERSION, secret, seed, canonical_json(base_cfg))
    last: Exception | None = None
    for attempt in range(8):
        try:
            ep = _Build(key, attempt, base_cfg, seed, secret != "").run()
        except RetryGeneration as exc:
            last = exc
            continue
        if cfg.get("transform"):
            from .transforms import apply_transform
            ep, _ = apply_transform(ep, cfg["transform"])
        return ep
    raise GenerationError(f"could not satisfy config after retries: {last}")


class _Build:
    def __init__(self, key: str, attempt: int, cfg: Dict[str, Any], seed: int, secret_used: bool) -> None:
        self.key = key
        self.attempt = attempt
        self.cfg = cfg
        self.seed = seed
        self.secret_used = secret_used
        self.invoices: Dict[str, Invoice] = {}
        self.payments: Dict[str, Payment] = {}
        self.customers: Dict[str, Customer] = {}
        self.journals: Dict[str, List[Row]] = {}
        self.row_seq = 0
        self.inv_counter = 0
        self.pay_counter = 0
        self.po_used: Set[str] = set()
        self.plans: List[Dict[str, Any]] = []
        self.distractor_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ utils
    def rng(self, name: str) -> Rng:
        return Rng.from_key(self.key, self.attempt, name)

    def in_period(self, instant: datetime) -> bool:
        return self.start_utc <= instant <= self.end_utc

    def loc(self, instant: datetime) -> date:
        return local_date(instant, self.offset)

    def at_local(self, d: date, seconds: int) -> datetime:
        base = datetime(d.year, d.month, d.day, tzinfo=tz(self.offset))
        return (base + timedelta(seconds=seconds)).astimezone(timezone.utc)

    def rate(self, d: date, src: str, tgt: str) -> Tuple[str, str]:
        rd = rate_date_for(d)
        k = (rd.isoformat(), src, tgt)
        if k not in self.fx:
            raise RetryGeneration(f"no fx rate for {k}")
        return rd.isoformat(), self.fx[k]

    def nice(self, minor: int, currency: str, rng: Rng | None = None) -> int:
        """Round to whole major units (at least 1 major unit)."""
        unit = 10 ** money.exponent(currency)
        return max(unit, (minor // unit) * unit)

    # --------------------------------------------------------------- company
    def make_company(self) -> None:
        r = self.rng("company")
        cfg = self.cfg
        pool = list(cfg["currencies"]["pool"])
        base = cfg["currencies"]["base"] or r.choice(pool)
        if base not in pool:
            pool.append(base)
        others = [c for c in pool if c != base]
        nf = min(self.sampled["foreign_count"], len(others))
        self.base = base
        self.foreign = r.sample(others, nf)
        self.currencies = [base] + self.foreign
        self.offset = parse_offset(cfg["timezone"]) if cfg["timezone"] else parse_offset(r.choice(TIMEZONE_POOL))
        if cfg["period"]:
            y, m = int(cfg["period"][:4]), int(cfg["period"][5:])
        else:
            y, m = r.choice([2025, 2026]), r.randint(1, 12)
        self.year, self.month = y, m
        last = calendar.monthrange(y, m)[1]
        self.start_date = date(y, m, 1)
        self.end_date = date(y, m, last)
        self.start_utc = datetime(y, m, 1, 0, 0, 0, tzinfo=tz(self.offset)).astimezone(timezone.utc)
        self.end_utc = datetime(y, m, last, 23, 59, 59, tzinfo=tz(self.offset)).astimezone(timezone.utc)
        name = "".join(r.choice(SYLLABLES) for _ in range(r.randint(2, 3))).capitalize()
        self.company_name = f"{name} {r.choice(INDUSTRIES)} {r.choice(LEGAL)}"
        self.id_style = {
            "invoice": r.randint(0, 3), "invoice_prefix": r.choice(INVOICE_PREFIX_LETTERS) + r.choice(INVOICE_PREFIX_LETTERS),
            "invoice_start": r.randint(100, 4000), "doc": r.randint(0, 1), "payment": r.randint(0, 1),
            "row": r.randint(0, 1), "entity": r.randint(0, 1),
        }

    # -------------------------------------------------------------------- fx
    def make_fx(self) -> None:
        r = self.rng("fx")
        ctx = Context(prec=40, rounding=ROUND_HALF_EVEN)
        first = self.start_date - timedelta(days=50)
        last = self.end_date + timedelta(days=14)
        levels: Dict[str, Decimal] = {}
        with localcontext(ctx):
            for c in self.currencies:
                factor = Decimal(1) + Decimal(r.randint(-800, 800)) / Decimal(10000)
                levels[c] = Decimal(LEVELS[c]) * factor
            self.price_level = dict(levels)
            self.fx: Dict[Tuple[str, str, str], str] = {}
            self.fx_dates: List[str] = []
            d = first
            while d <= last:
                if is_business_day(d):
                    for c in self.currencies:
                        bps = r.randint(-35, 35) + r.randint(-35, 35)
                        levels[c] = levels[c] * (Decimal(1) + Decimal(bps) / Decimal(10000))
                    ds = d.isoformat()
                    self.fx_dates.append(ds)
                    for a in self.currencies:
                        for b in self.currencies:
                            if a != b:
                                self.fx[(ds, a, b)] = money.quantize_rate(levels[b] / levels[a])
                d += timedelta(days=1)

    def price(self, usd_lo: int, usd_hi: int, currency: str, r: Rng) -> int:
        with localcontext(Context(prec=40, rounding=ROUND_HALF_EVEN)):
            usd = Decimal(r.randint(usd_lo * 100, usd_hi * 100)) / Decimal(100)
            usd_level = self.price_level.get("USD", Decimal(LEVELS["USD"]))
            major = usd * self.price_level[currency] / usd_level
            q = Decimal(1).scaleb(major.adjusted() - 2)  # 3 significant digits
            major = major.quantize(q, rounding=ROUND_HALF_EVEN)
            minor = int((major.scaleb(money.exponent(currency))).quantize(Decimal(1)))
            return max(minor, 10 ** money.exponent(currency))

    # ------------------------------------------------------------- customers
    def make_customers(self) -> None:
        r = self.rng("customers")
        n = self.sampled["size"]["customers"]
        n_sub = min(self.sampled["distractors"]["recurring_customers"], n - 1)
        for i in range(n):
            ccy = self.base if r.chance(0.55) else r.choice(self.foreign)
            alt = r.choice([c for c in self.currencies if c != ccy])
            off = self.offset if r.chance(0.55) else r.choice(DOC_OFFSETS)
            self.customers[f"c{i}"] = Customer(f"c{i}", ccy, off, alt)
        for k in r.sample(sorted(self.customers), n_sub):
            c = self.customers[k]
            name, lo, hi = r.choice(PLANS)
            c.plan = (name, self.price(lo, hi, c.currency, r))

    # ----------------------------------------------------------- primitives
    def new_po(self, r: Rng) -> str:
        while True:
            po = r.digits(6)
            if po not in self.po_used and po[0] != "0":
                self.po_used.add(po)
                return po

    def make_lines(self, cust: Customer, r: Rng) -> List[Dict[str, Any]]:
        lines = []
        po = self.new_po(r)
        for i, prod in enumerate(r.sample(PRODUCTS, r.randint(1, 4)), start=1):
            name, cat, lo, hi = prod
            unit = self.price(lo, hi, cust.currency, r)
            qty = r.randint(1, 12)
            lines.append({"line_no": i, "sku": "SKU-" + derive_hex("sku", self.key, name, n=6).upper(),
                          "description": f"{name} - PO {po}", "quantity": qty, "unit_price_minor": unit,
                          "amount_minor": qty * unit, "revenue_account": REV[cat]})
        return lines

    def add_invoice(self, cust_key: str, issued: datetime, r: Rng, lines=None, offset=None, tags=()) -> Invoice:
        cust = self.customers[cust_key]
        key = f"i{self.inv_counter}"
        self.inv_counter += 1
        if offset is None:
            offset = self.offset if r.chance(0.6) else cust.offset
        inv = Invoice(key, cust_key, issued, offset, r.choice([14, 30, 30, 45, 60]), cust.currency,
                      lines if lines is not None else self.make_lines(cust, r), set(tags))
        self.invoices[key] = inv
        return inv

    def add_payment(self, ts: datetime, currency: str, amount: int, refs, payer, r: Rng, offset=None, tags=()) -> Payment:
        key = f"p{self.pay_counter}"
        self.pay_counter += 1
        if offset is None:
            roll = r.random()
            if roll < 0.35:
                offset = 0
            elif roll < 0.8 or payer is None:
                offset = self.offset
            else:
                offset = self.customers[payer].offset
        p = Payment(key, ts, offset, currency, amount, refs, payer, set(tags))
        self.payments[key] = p
        return p

    def random_period_instant(self, r: Rng, lo_day: int = 0, hi_day: int | None = None, business=True) -> datetime:
        days = (self.end_date - self.start_date).days
        hi_day = days if hi_day is None else min(hi_day, days)
        lo_day = max(0, min(lo_day, hi_day))
        for _ in range(50):
            d = self.start_date + timedelta(days=r.randint(lo_day, hi_day))
            if not business or is_business_day(d):
                break
        secs = r.randint(8 * 3600, 19 * 3600 + 3599)
        return self.at_local(d, secs)

    def payments_of(self, inv_key: str) -> List[Payment]:
        return [p for p in self.payments.values() if p.refs and any(k == inv_key for k, _ in p.refs)]

    def stamp_for(self, d: date, shift: int, r: Rng) -> Tuple[datetime, int]:
        """Instant on local date ``d`` whose wall-clock date in the returned offset is d+shift."""
        for _ in range(400):
            if shift > 0:
                secs = r.randint(18 * 3600, 24 * 3600 - 61)
            else:
                secs = r.randint(5 * 60, 6 * 3600)
            inst = self.at_local(d, secs)
            target = d + timedelta(days=shift)
            cands = [o for o in DOC_OFFSETS if local_date(inst, o) == target]
            if cands:
                return inst, r.choice(cands)
        raise RetryGeneration("no offset produces the requested wall-clock date")

    def boundary_stamp(self, side: str, inside: bool, r: Rng) -> Tuple[datetime, int]:
        one = timedelta(days=1)
        if side == "end":
            return self.stamp_for(self.end_date, +1, r) if inside else self.stamp_for(self.end_date + one, -1, r)
        return self.stamp_for(self.start_date, -1, r) if inside else self.stamp_for(self.start_date - one, +1, r)

    # --------------------------------------------------------------- history
    def make_history(self) -> None:
        r = self.rng("history")
        n_inv = self.sampled["size"]["invoices"]
        cust_keys = sorted(self.customers)
        regular = [k for k in cust_keys if self.customers[k].plan is None] or cust_keys
        weights = [r.randint(1, 6) for _ in regular]
        for _ in range(n_inv):
            self.make_regular_invoice(r.weighted(regular, weights), r)
        # prior-period invoices (booked in earlier periods, paid or open now)
        for _ in range(self.sampled["size"]["prior_invoices"]):
            ck = r.choice(regular)
            issued = self.start_utc - timedelta(seconds=r.randint(2 * 86400, 40 * 86400))
            inv = self.add_invoice(ck, issued, r)
            if r.chance(0.8):
                ts = self.random_period_instant(r, 0, 20, business=False)
                self.add_payment(ts, inv.currency, inv.total, [(inv.key, inv.total)], ck, r)
        # recurring subscription invoices: same amount every week, different service period
        for ck in cust_keys:
            c = self.customers[ck]
            if c.plan is None:
                continue
            first = r.randint(0, 6)
            plan, price = c.plan
            day = first
            while day <= (self.end_date - self.start_date).days:
                d0 = self.start_date + timedelta(days=day)
                issued = self.at_local(d0, r.randint(9 * 3600, 11 * 3600))
                period_txt = f"{d0.isoformat()}..{(d0 + timedelta(days=6)).isoformat()}"
                lines = [{"line_no": 1, "sku": "SUB-" + plan.upper(), "description":
                          f"Subscription {plan} - service period {period_txt}", "quantity": 1,
                          "unit_price_minor": price, "amount_minor": price,
                          "revenue_account": REV["subscriptions"]}]
                inv = self.add_invoice(ck, issued, r, lines=lines, offset=self.offset, tags={"subscription"})
                if r.chance(0.8):
                    ts = issued + timedelta(seconds=r.randint(3600, 5 * 86400))
                    if ts <= self.end_utc + timedelta(days=3):
                        self.add_payment(ts, inv.currency, inv.total, [(inv.key, inv.total)], ck, r,
                                         tags={"subscription"})
                day += 7
        self.distractor_log.append({"kind": "recurring_customers", "customers": [
            k for k in cust_keys if self.customers[k].plan is not None],
            "invoices": sorted((k for k, i in self.invoices.items() if "subscription" in i.tags), key=lambda k: int(k[1:])),
            "payments": sorted((k for k, p in self.payments.items() if "subscription" in p.tags), key=lambda k: int(k[1:]))})

    def make_regular_invoice(self, ck: str, r: Rng, lo_day: int = 0, hi_day: int | None = None,
                             pay_prob: float = 0.65) -> Invoice:
        issued = self.random_period_instant(r, lo_day, hi_day)
        inv = self.add_invoice(ck, issued, r)
        if r.chance(pay_prob):
            ts = issued + timedelta(seconds=r.randint(2 * 3600, 26 * 86400))
            if ts <= self.end_utc + timedelta(days=3):
                self.add_payment(ts, inv.currency, inv.total, [(inv.key, inv.total)], ck, r)
        return inv

    # ---------------------------------------------------------- candidates
    def regular_payments(self, cross_ok: bool = False) -> List[Payment]:
        out = []
        for p in self.payments.values():
            if p.tags and not (cross_ok and p.tags == {"cross"}):
                continue
            if not p.refs or len(p.refs) != 1 or p.amount <= 0 or p.refs[0][1] != p.amount:
                continue
            inv = self.invoices[p.refs[0][0]]
            if inv.tags or not self.in_period(p.ts):
                continue
            if p.ts > self.end_utc - timedelta(hours=30) or p.ts < self.start_utc + timedelta(hours=30):
                continue
            if len(self.payments_of(inv.key)) != 1:
                continue
            out.append(p)
        return sorted(out, key=lambda p: p.key)

    def open_regular_invoices(self, lo_day: int, hi_day: int) -> List[Invoice]:
        out = []
        for inv in self.invoices.values():
            if inv.tags or not self.in_period(inv.issued) or self.payments_of(inv.key):
                continue
            dd = (self.loc(inv.issued) - self.start_date).days
            if lo_day <= dd <= hi_day:
                out.append(inv)
        return sorted(out, key=lambda i: i.key)

    def fresh_open_invoice(self, r: Rng, lo_day: int, hi_day: int, currency: str | None = None) -> Invoice:
        cands = [i for i in self.open_regular_invoices(lo_day, hi_day) if currency is None or i.currency == currency]
        if cands and r.chance(0.7):
            return r.choice(cands)
        custs = [k for k in sorted(self.customers) if self.customers[k].plan is None
                 and (currency is None or self.customers[k].currency == currency)]
        if not custs:
            custs = [k for k in sorted(self.customers) if self.customers[k].plan is None]
        return self.make_regular_invoice(r.choice(custs), r, lo_day, hi_day, pay_prob=0.0)

    def make_cross(self, p: Payment, r: Rng) -> None:
        inv = self.invoices[p.refs[0][0]]
        cust = self.customers[inv.cust]
        pay_ccy = cust.alt_currency if r.chance(0.7) else r.choice([c for c in self.currencies if c != inv.currency])
        _, rt = self.rate(self.loc(p.ts), inv.currency, pay_ccy)
        x = money.convert(inv.total, inv.currency, pay_ccy, rt)
        if x <= 10 ** money.exponent(pay_ccy):
            raise RetryGeneration("cross-currency amount too small")
        p.currency, p.amount, p.refs = pay_ccy, x, [(inv.key, x)]
        p.tags.add("cross")

    # ---------------------------------------------------------- distractors
    def make_distractors(self) -> None:
        r = self.rng("distractors")
        d = self.sampled["distractors"]

        def take(n: int) -> List[Payment]:
            cands = self.regular_payments()
            return r.sample(cands, min(n, len(cands)))

        for p in take(d["split_payments"]):
            inv = self.invoices[p.refs[0][0]]
            k = r.randint(2, 3)
            unit = 10 ** money.exponent(p.currency)
            parts = []
            remaining = p.amount
            for i in range(k - 1):
                share = self.nice(remaining * r.randint(25, 60) // 100, p.currency)
                if share >= remaining:
                    share = max(unit, remaining // 2)
                parts.append(share)
                remaining -= share
            parts.append(remaining)
            if any(x <= 0 for x in parts):
                continue
            p.amount, p.refs = parts[0], [(inv.key, parts[0])]
            p.tags.add("split")
            ts = p.ts
            ids = [p.key]
            for share in parts[1:]:
                ts = min(ts + timedelta(seconds=r.randint(86400, 6 * 86400)), self.end_utc - timedelta(hours=r.randint(2, 20)))
                q = self.add_payment(ts, p.currency, share, [(inv.key, share)], p.payer, r, tags={"split"})
                ids.append(q.key)
            inv.tags.add("split")
            self.distractor_log.append({"kind": "split_payments", "invoice": inv.key, "payments": ids})

        for p in take(d["legit_partial_payments"]):
            inv = self.invoices[p.refs[0][0]]
            part = self.nice(p.amount * r.randint(30, 80) // 100, p.currency)
            if not 0 < part < p.amount:
                continue
            p.amount, p.refs = part, [(inv.key, part)]
            p.tags.add("partial")
            inv.tags.add("partial")
            self.distractor_log.append({"kind": "legit_partial_payments", "invoice": inv.key, "payments": [p.key]})

        for _ in range(d["multi_invoice_payments"]):
            by_cust: Dict[Tuple[str, str], List[Payment]] = {}
            for p in self.regular_payments():
                inv = self.invoices[p.refs[0][0]]
                by_cust.setdefault((inv.cust, p.currency), []).append(p)
            groups = [g for _, g in sorted(by_cust.items()) if len(g) >= 2]
            if not groups:
                break
            g = r.choice(groups)
            chosen = r.sample(g, min(len(g), r.randint(2, 3)))
            keep = max(chosen, key=lambda p: (p.ts, p.key))
            refs = [(c.refs[0][0], c.amount) for c in sorted(chosen, key=lambda p: p.key)]
            r.shuffle(refs)
            for c in chosen:
                if c is not keep:
                    del self.payments[c.key]
            keep.refs = refs
            keep.amount = sum(a for _, a in refs)
            keep.tags.add("multi")
            for k, _ in refs:
                self.invoices[k].tags.add("multi")
            self.distractor_log.append({"kind": "multi_invoice_payments", "payments": [keep.key],
                                        "invoices": [k for k, _ in refs]})

        refund_cands = [p for p in self.regular_payments() if p.ts <= self.end_utc - timedelta(days=9)
                        and p.currency == self.invoices[p.refs[0][0]].currency]
        for p in r.sample(refund_cands, min(d["refund_pairs"], len(refund_cands))):
            inv = self.invoices[p.refs[0][0]]
            t2 = p.ts + timedelta(seconds=r.randint(86400, 3 * 86400))
            p2 = self.add_payment(t2, p.currency, p.amount, [(inv.key, p.amount)], p.payer, r, tags={"refund_pair"})
            t3 = t2 + timedelta(seconds=r.randint(86400, 4 * 86400))
            p3 = self.add_payment(t3, p.currency, -p.amount, [(inv.key, -p.amount)], p.payer, r, tags={"refund_pair"})
            p.tags.add("refund_pair")
            inv.tags.add("refund_pair")
            self.distractor_log.append({"kind": "refund_pairs", "invoice": inv.key, "payments": [p.key, p2.key, p3.key]})

        for p in take(d["on_account_payments"]):
            extra = self.nice(p.amount * r.randint(5, 25) // 100, p.currency)
            p.amount += extra
            p.tags.add("on_account")
            self.invoices[p.refs[0][0]].tags.add("on_account")
            self.distractor_log.append({"kind": "on_account_payments", "payments": [p.key]})

        for p in take(d["cross_currency_payments"]):
            self.make_cross(p, r)
            self.invoices[p.refs[0][0]].tags.add("cross")
            self.distractor_log.append({"kind": "cross_currency_payments", "payments": [p.key]})

        for _ in range(d["unidentified_receipts"]):
            ccy = r.choice(self.currencies)
            amount = self.price(50, 3000, ccy, r)
            ts = self.random_period_instant(r, 0, None, business=False)
            p = self.add_payment(ts, ccy, amount, None, None, r, tags={"unidentified"})
            self.distractor_log.append({"kind": "unidentified_receipts", "payments": [p.key]})

        # Boundary distractors scale with the number of cutoff errors, so that flagging the documents
        # closest to a period boundary without checking how they are booked does not pay.
        d["boundary_documents"] += d["boundary_per_cutoff"] * self.sampled["errors"]["period_cutoff"]
        for _ in range(d["boundary_documents"]):
            kind = r.choice(["invoice", "invoice", "payment"])
            side = "end" if kind == "payment" else r.choice(["end", "start"])
            inside = r.chance(0.5)
            doc = self.make_boundary_doc(kind, side, inside, r, tags={"boundary"})
            self.distractor_log.append({"kind": "boundary_documents", "doc": doc.key, "side": side,
                                        "in_period": inside})

    def make_boundary_doc(self, kind: str, side: str, inside: bool, r: Rng, tags=()):
        inst, off = self.boundary_stamp(side, inside, r)
        custs = [k for k in sorted(self.customers) if self.customers[k].plan is None]
        if kind == "invoice":
            return self.add_invoice(r.choice(custs), inst, r, offset=off, tags=set(tags))
        days = (self.end_date - self.start_date).days
        inv = self.fresh_open_invoice(r, 1, days - 3)
        inv.tags.add("boundary_paid")
        return self.add_payment(inst, inv.currency, inv.total, [(inv.key, inv.total)], inv.cust, r,
                                offset=off, tags=set(tags))

    # ----------------------------------------------------------------- hosts
    def make_hosts(self) -> None:
        r = self.rng("hosts")
        s = self.sampled
        days = (self.end_date - self.start_date).days
        for _ in range(s["interactions"]["duplicate_misapplied"]):
            inv = self.fresh_open_invoice(r, 1, days - 10)
            inv.tags.add("host")
            dup_issued = inv.issued + timedelta(seconds=r.randint(3600, 2 * 86400))
            ts = dup_issued + timedelta(seconds=r.randint(86400, 7 * 86400))
            ts = min(ts, self.end_utc - timedelta(hours=r.randint(26, 40)))
            p = self.add_payment(ts, inv.currency, inv.total, [(inv.key, inv.total)], inv.cust, r, tags={"host"})
            self.plans.append({"kind": "duplicate_misapplied", "invoice": inv.key, "payment": p.key,
                               "dup_issued": dup_issued})
        for _ in range(s["interactions"]["fx_partial_boundary"]):
            self.plans.append(self.make_fx_partial_boundary(r))
        for _ in range(s["errors"]["duplicate_invoice"]):
            cands = [i for i in self.invoices.values() if not i.tags and self.in_period(i.issued)
                     and 1 <= (self.loc(i.issued) - self.start_date).days <= days - 5]
            inv = r.choice(sorted(cands, key=lambda i: i.key)) if cands else self.fresh_open_invoice(r, 1, days - 5)
            inv.tags.add("host")
            dup_issued = inv.issued + timedelta(seconds=r.randint(600, 3 * 86400))
            self.plans.append({"kind": "duplicate_invoice", "invoice": inv.key, "dup_issued": dup_issued})
        for _ in range(s["errors"]["payment_allocation"]):
            variant = r.choice(list(self.cfg["variants"]["payment_allocation"]))
            p = self.alloc_host(r, cross=r.chance(0.3))
            applied = 0
            if variant == "partial":
                applied = self.nice(p.amount * r.randint(30, 85) // 100, p.currency)
                if not 0 < applied < p.amount:
                    raise RetryGeneration("partial allocation amount degenerate")
            self.plans.append({"kind": "payment_allocation", "variant": variant, "payment": p.key, "applied": applied})
        for _ in range(s["errors"]["fx_conversion"]):
            self.plans.append(self.make_fx_host(r))
        for _ in range(s["errors"]["period_cutoff"]):
            kind = r.choice(["invoice", "invoice", "payment"])
            side = "end" if kind == "payment" else r.choice(["end", "start"])
            # early: booked although outside; late: inside but not booked
            variant = r.choice(list(self.cfg["variants"]["period_cutoff"]))
            doc = self.make_boundary_doc(kind, side, variant == "late", r, tags={"host"})
            self.plans.append({"kind": "period_cutoff", "variant": variant, "doc_kind": kind, "doc": doc.key, "side": side})

    def alloc_host(self, r: Rng, cross: bool) -> Payment:
        cands = self.regular_payments()
        if cands:
            p = r.choice(cands)
        else:
            days = (self.end_date - self.start_date).days
            inv = self.fresh_open_invoice(r, 1, days - 8)
            ts = inv.issued + timedelta(seconds=r.randint(86400, 5 * 86400))
            p = self.add_payment(ts, inv.currency, inv.total, [(inv.key, inv.total)], inv.cust, r)
        if cross:
            self.make_cross(p, r)
        p.tags.add("host")
        self.invoices[p.refs[0][0]].tags.add("host")
        return p

    def wrong_rate_options(self, p: Payment, inv: Invoice, applied: int) -> List[Tuple[str, str, str, int]]:
        """(variant, wrong_rate_date, wrong_rate, |diff|) for every variant that changes the booking enough."""
        local_d = self.loc(p.ts)
        correct_date, correct = self.rate(local_d, p.currency, inv.currency)
        good = money.convert(applied, p.currency, inv.currency, correct)
        cd = date.fromisoformat(correct_date)
        options = []
        cands = [("next_business_day", next_business_day(cd)), ("previous_business_day", prev_business_day(cd))]
        wall = local_date(p.ts, p.offset)
        if wall != local_d:
            cands.append(("document_wall_clock_date", rate_date_for(wall)))
        cands.append(("invoice_issue_date", rate_date_for(self.loc(inv.issued))))
        # right date, wrong direction: the row target->source is used as if it were source->target
        rev = (correct_date, inv.currency, p.currency)
        if rev in self.fx:
            bad = money.convert(applied, p.currency, inv.currency, self.fx[rev])
            if bad >= 1 and abs(bad - good) >= MIN_FX_DIFF:  # a zero booking is not a valid ledger row
                options.append(("reverse_pair_rate", correct_date, self.fx[rev], abs(bad - good)))
        for variant, wd in cands:
            k = (wd.isoformat(), p.currency, inv.currency)
            if wd.isoformat() == correct_date or k not in self.fx:
                continue
            bad = money.convert(applied, p.currency, inv.currency, self.fx[k])
            if bad >= 1 and abs(bad - good) >= MIN_FX_DIFF:
                options.append((variant, wd.isoformat(), self.fx[k], abs(bad - good)))
        return options

    def make_fx_host(self, r: Rng) -> Dict[str, Any]:
        for _ in range(30):
            p = self.alloc_host(r, cross=True)
            inv = self.invoices[p.refs[0][0]]
            allowed = set(self.cfg["variants"]["fx_conversion"])
            opts = [o for o in self.wrong_rate_options(p, inv, p.amount) if o[0] in allowed]
            if opts:
                variant, wd, wr, _ = r.choice(opts)
                return {"kind": "fx_conversion", "payment": p.key, "variant": variant, "wrong_date": wd, "wrong_rate": wr}
            p.tags.discard("host")
            p.tags.add("cross")  # stays a legit cross-currency payment
            self.invoices[p.refs[0][0]].tags.discard("host")
            self.invoices[p.refs[0][0]].tags.add("cross")
        raise RetryGeneration("no FX host with a material rate difference")

    def make_fx_partial_boundary(self, r: Rng) -> Dict[str, Any]:
        one = timedelta(days=1)
        configs = []
        if is_business_day(self.end_date) and is_business_day(self.end_date + one):
            configs.append(("end", self.end_date, +1))
        if is_business_day(self.start_date) and is_business_day(self.start_date - one):
            configs.append(("start", self.start_date, -1))
        r.shuffle(configs)
        d = self.end_date - one
        while len(configs) < 6 and d > self.end_date - timedelta(days=9):
            if is_business_day(d) and is_business_day(d + one):
                configs.append(("near_end", d, +1))
            d -= one
        custs = [k for k in sorted(self.customers) if self.customers[k].plan is None]
        for side, local_d, shift in configs:
            for _ in range(4):
                ck = r.choice(custs)
                cust = self.customers[ck]
                if side == "start":
                    issued = self.start_utc - timedelta(seconds=r.randint(3 * 86400, 25 * 86400))
                else:
                    hi = (local_d - self.start_date).days - 3
                    if hi < 1:
                        continue
                    issued = self.random_period_instant(r, 1, hi)
                inv = self.add_invoice(ck, issued, r, tags={"host"})
                inst, off = self.stamp_for(local_d, shift, r)
                pay_ccy = cust.alt_currency if r.chance(0.7) else r.choice([c for c in self.currencies if c != inv.currency])
                _, rt = self.rate(local_d, inv.currency, pay_ccy)
                x = money.convert(inv.total * r.randint(40, 75) // 100, inv.currency, pay_ccy, rt)
                applied = self.nice(x * r.randint(45, 85) // 100, pay_ccy)
                if not 0 < applied < x:
                    del self.invoices[inv.key]
                    continue
                p = self.add_payment(inst, pay_ccy, x, [(inv.key, x)], ck, r, offset=off, tags={"host", "cross"})
                opts = [o for o in self.wrong_rate_options(p, inv, applied) if o[0] == "document_wall_clock_date"]
                if opts:
                    _, wd, wr, _ = opts[0]
                    return {"kind": "fx_partial_boundary", "payment": p.key, "invoice": inv.key, "applied": applied,
                            "wrong_date": wd, "wrong_rate": wr, "side": side}
                del self.payments[p.key]
                del self.invoices[inv.key]
        raise RetryGeneration("could not place fx_partial_boundary scenario")

    # ------------------------------------------------------------ booking
    def new_row(self, ts, account, debit, credit, currency, entity, invoice, payment, journal, seq) -> Row:
        self.row_seq += 1
        return Row(f"r{self.row_seq}", ts, account, debit, credit, currency, entity, invoice, payment, journal, seq)

    def book_invoice(self, inv: Invoice, ts: datetime | None = None) -> List[Row]:
        ts = ts or inv.issued.astimezone(tz(self.offset))
        j = "I:" + inv.key
        rows = [self.new_row(ts, AR, inv.total, 0, inv.currency, inv.cust, inv.key, None, j, 0)]
        for i, li in enumerate(inv.lines, start=1):
            rows.append(self.new_row(ts, li["revenue_account"], 0, li["amount_minor"], inv.currency, inv.cust, inv.key, None, j, i))
        return rows

    def book_payment(self, p: Payment, ts: datetime | None = None, mode: Dict[str, Any] | None = None) -> List[Row]:
        mode = mode or {}
        ts = ts or p.ts.astimezone(tz(self.offset))
        j = "P:" + p.key
        rows: List[Row] = []
        seq = iter(range(100))

        def add(account, debit, credit, currency, entity, invoice):
            rows.append(self.new_row(ts, account, debit, credit, currency, entity, invoice, p.key, j, next(seq)))

        if p.amount < 0:
            (ik, a), = p.refs
            add(BANK, 0, -p.amount, p.currency, p.payer, None)
            add(AR, -a, 0, p.currency, self.invoices[ik].cust, ik)
            return rows
        if p.refs is None or mode.get("unidentified"):
            add(BANK, p.amount, 0, p.currency, None, None)
            add(UNAPPLIED, 0, p.amount, p.currency, None, None)
            return rows
        add(BANK, p.amount, 0, p.currency, p.payer, None)
        unapplied = p.amount - sum(a for _, a in p.refs)
        for ik, a in p.refs:
            inv = self.invoices[ik]
            applied = mode.get("applied", {}).get(ik, a)
            unapplied += a - applied
            target = mode.get("redirect", {}).get(ik, ik)
            if applied == 0:
                continue
            if inv.currency == p.currency:
                add(AR, 0, applied, p.currency, inv.cust, target)
            else:
                rt = mode.get("rate") or self.rate(self.loc(p.ts), p.currency, inv.currency)[1]
                y = money.convert(applied, p.currency, inv.currency, rt)
                add(FXC, 0, applied, p.currency, inv.cust, target)
                add(FXC, y, 0, inv.currency, inv.cust, target)
                add(AR, 0, y, inv.currency, inv.cust, target)
        if unapplied:
            add(UNAPPLIED, 0, unapplied, p.currency, p.payer, None)
        return rows

    def book_clean(self) -> None:
        for inv in sorted(self.invoices.values(), key=lambda i: i.key):
            if self.in_period(inv.issued):
                self.journals["I:" + inv.key] = self.book_invoice(inv)
        for p in sorted(self.payments.values(), key=lambda p: p.key):
            if self.in_period(p.ts):
                self.journals["P:" + p.key] = self.book_payment(p)

    def rebook(self, journal: str, new_rows: List[Row]) -> None:
        """Replace a journal; field-identical rows keep their identity (key)."""
        old = list(self.journals.get(journal, []))
        out = []
        for nr in new_rows:
            match = next((o for o in old if o.sig() == nr.sig()), None)
            if match is not None:
                old.remove(match)
                match.seq = nr.seq
                out.append(match)
            else:
                out.append(nr)
        self.journals[journal] = out

    # ------------------------------------------------------------ injection
    def inject(self) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        self.change_notes: List[Dict[str, Any]] = []
        group = 0
        for plan in self.plans:
            kind = plan["kind"]
            if kind == "duplicate_invoice":
                issues.append(self.inj_duplicate(plan, None))
            elif kind == "duplicate_misapplied":
                group += 1
                gid = f"G{group}"
                dup_issue = self.inj_duplicate(plan, gid)
                issues.append(dup_issue)
                issues.append(self.inj_misapply(plan, dup_issue["_dup_key"], gid))
            elif kind == "payment_allocation":
                issues.append(self.inj_allocation(plan))
            elif kind == "fx_conversion":
                issues.append(self.inj_fx(plan))
            elif kind == "period_cutoff":
                issues.append(self.inj_cutoff(plan))
            elif kind == "fx_partial_boundary":
                group += 1
                issues.extend(self.inj_fx_partial(plan, f"G{group}"))
            else:  # pragma: no cover
                raise AssertionError(kind)
        return issues

    def journal_rows(self, journal: str) -> List[Row]:
        return self.journals[journal]

    def inj_duplicate(self, plan, gid) -> Dict[str, Any]:
        orig = self.invoices[plan["invoice"]]
        r = self.rng("dup-" + orig.key)
        dup = Invoice(f"i{self.inv_counter}", orig.cust, plan["dup_issued"], orig.offset, orig.terms_days,
                      orig.currency, copy.deepcopy(orig.lines), {"duplicate"})
        self.inv_counter += 1
        self.invoices[dup.key] = dup
        if not self.in_period(dup.issued):
            raise RetryGeneration("duplicate fell outside the period")
        self.journals["I:" + dup.key] = self.book_invoice(dup)
        self.change_notes.append({"op": "add_document_and_journal", "documents": [dup.key], "journal": "I:" + dup.key,
                                  "copy_of": orig.key, "reason": "invoice entered twice (copy of an existing invoice)"})
        sets = [[("doc", orig.key), ("doc", dup.key), ("row", ra.key), ("row", rb.key)]
                for ra in self.journals["I:" + orig.key] for rb in self.journals["I:" + dup.key]]
        del r
        return {"type": "duplicate_invoice", "affected": [("inv", orig.key), ("inv", dup.key)],
                "amount": orig.total, "currency": orig.currency, "evidence_sets": sets, "interaction_id": gid,
                "description": "The same invoice (same customer, currency, line items and total) is booked twice.",
                "_dup_key": dup.key}

    def inj_misapply(self, plan, dup_key: str, gid: str) -> Dict[str, Any]:
        p = self.payments[plan["payment"]]
        ik = p.refs[0][0]
        j = "P:" + p.key
        self.rebook(j, self.book_payment(p, mode={"redirect": {ik: dup_key}}))
        self.change_notes.append({"op": "modify_journal", "journal": j, "reason":
                                  "payment applied to the duplicate invoice instead of the referenced one"})
        ar = [r for r in self.journals[j] if r.account == AR]
        return {"type": "payment_allocation", "affected": [("pay", p.key), ("inv", ik)], "amount": p.refs[0][1],
                "currency": p.currency, "evidence_sets": [[("doc", p.key), ("row", ar[0].key)]],
                "interaction_id": gid, "variant": "misapplied",
                "description": "The remittance references one invoice but the cash was applied to its duplicate."}

    def inj_allocation(self, plan) -> Dict[str, Any]:
        p = self.payments[plan["payment"]]
        ik, a = p.refs[0]
        j = "P:" + p.key
        if plan["variant"] == "missing":
            self.rebook(j, self.book_payment(p, mode={"unidentified": True}))
            short = a
            rows = [r for r in self.journals[j] if r.account == UNAPPLIED]
            desc = "The remittance references an invoice but the payment was left entirely as unapplied cash."
        else:
            applied = plan["applied"]
            self.rebook(j, self.book_payment(p, mode={"applied": {ik: applied}}))
            short = a - applied
            rows = [r for r in self.journals[j] if r.account == UNAPPLIED or
                    (r.account in (AR, FXC) and r.currency == p.currency and r.invoice == ik)]
            desc = "Only part of the referenced allocation was applied to the invoice; the rest sits in unapplied cash."
        self.change_notes.append({"op": "modify_journal", "journal": j, "reason": f"allocation {plan['variant']}"})
        return {"type": "payment_allocation", "affected": [("pay", p.key), ("inv", ik)], "amount": short,
                "currency": p.currency, "evidence_sets": [[("doc", p.key), ("row", r.key)] for r in rows],
                "interaction_id": None, "variant": plan["variant"], "description": desc}

    def inj_fx(self, plan) -> Dict[str, Any]:
        p = self.payments[plan["payment"]]
        ik, a = p.refs[0]
        inv = self.invoices[ik]
        j = "P:" + p.key
        self.rebook(j, self.book_payment(p, mode={"rate": plan["wrong_rate"]}))
        cdate, crate = self.rate(self.loc(p.ts), p.currency, inv.currency)
        good = money.convert(a, p.currency, inv.currency, crate)
        bad = money.convert(a, p.currency, inv.currency, plan["wrong_rate"])
        if plan["variant"] == "reverse_pair_rate":
            why = f"conversion used the {inv.currency}->{p.currency} rate of {cdate} as if it were {p.currency}->{inv.currency}"
        else:
            why = f"conversion used rate of {plan['wrong_date']} ({plan['variant']}) instead of {cdate}"
        self.change_notes.append({"op": "modify_journal", "journal": j, "reason": why})
        inv_rows = [r for r in self.journals[j] if r.currency == inv.currency and r.account in (AR, FXC)]
        fx_id = f"FX:{cdate}:{p.currency}:{inv.currency}"
        return {"type": "fx_conversion", "affected": [("pay", p.key), ("inv", ik)], "amount": abs(bad - good),
                "currency": inv.currency, "evidence_sets": [[("doc", p.key), ("fx", fx_id), ("row", r.key)] for r in inv_rows],
                "interaction_id": None, "variant": plan["variant"],
                "description": "A cross-currency allocation was converted with a rate other than the policy rate."}

    def inj_cutoff(self, plan) -> Dict[str, Any]:
        is_inv = plan["doc_kind"] == "invoice"
        doc = self.invoices[plan["doc"]] if is_inv else self.payments[plan["doc"]]
        j = ("I:" if is_inv else "P:") + doc.key
        inst = doc.issued if is_inv else doc.ts
        if plan["variant"] == "early":
            wall = inst.astimezone(tz(doc.offset)).replace(tzinfo=tz(self.offset))
            if not self.in_period(wall) or self.in_period(inst):
                raise RetryGeneration("cutoff early variant not reproducible")
            rows = self.book_invoice(doc, ts=wall) if is_inv else self.book_payment(doc, ts=wall)
            self.journals[j] = rows
            sets = [[("doc", doc.key), ("row", r.key)] for r in rows]
            self.change_notes.append({"op": "add_journal", "journal": j, "reason":
                                      "document from another period booked using its wall-clock date read in the company timezone"})
            desc = "A document dated outside the period (company timezone) is booked in this period."
        else:
            if not self.in_period(inst):
                raise RetryGeneration("cutoff late variant not reproducible")
            self.removed_journals[j] = self.journals.pop(j)
            sets = [[("doc", doc.key)]]
            self.change_notes.append({"op": "remove_journal", "journal": j, "reason":
                                      "document of this period booked in the adjacent period (wall-clock date of its own timezone)"})
            desc = "A document dated inside the period (company timezone) is missing from this period's ledger."
        amount = doc.total if is_inv else abs(doc.amount)
        return {"type": "period_cutoff", "affected": [("inv" if is_inv else "pay", doc.key)], "amount": amount,
                "currency": doc.currency, "evidence_sets": sets, "interaction_id": None,
                "variant": f"{plan['variant']}_{plan['side']}_{plan['doc_kind']}", "description": desc}

    def inj_fx_partial(self, plan, gid) -> List[Dict[str, Any]]:
        p = self.payments[plan["payment"]]
        ik, a = p.refs[0]
        inv = self.invoices[ik]
        applied = plan["applied"]
        j = "P:" + p.key
        self.rebook(j, self.book_payment(p, mode={"applied": {ik: applied}, "rate": plan["wrong_rate"]}))
        cdate, crate = self.rate(self.loc(p.ts), p.currency, inv.currency)
        good = money.convert(applied, p.currency, inv.currency, crate)
        bad = money.convert(applied, p.currency, inv.currency, plan["wrong_rate"])
        self.change_notes.append({"op": "modify_journal", "journal": j, "reason":
                                  f"partial application of a cross-currency payment at the period boundary; "
                                  f"rate of {plan['wrong_date']} (wall-clock date) used instead of {cdate}"})
        rows = self.journals[j]
        fxc_pay = [r for r in rows if r.account == FXC and r.currency == p.currency][0]
        unap = [r for r in rows if r.account == UNAPPLIED][0]
        inv_rows = [r for r in rows if r.currency == inv.currency and r.account in (AR, FXC)]
        fx_id = f"FX:{cdate}:{p.currency}:{inv.currency}"
        alloc = {"type": "payment_allocation", "affected": [("pay", p.key), ("inv", ik)], "amount": a - applied,
                 "currency": p.currency, "evidence_sets": [[("doc", p.key), ("row", fxc_pay.key)], [("doc", p.key), ("row", unap.key)]],
                 "interaction_id": gid, "variant": "partial",
                 "description": "Boundary cross-currency payment: only part of the remittance was applied."}
        fx = {"type": "fx_conversion", "affected": [("pay", p.key), ("inv", ik)], "amount": abs(bad - good),
              "currency": inv.currency,
              "evidence_sets": [[("doc", p.key), ("fx", fx_id), ("row", fxc_pay.key), ("row", r.key)] for r in inv_rows],
              "interaction_id": gid, "variant": "document_wall_clock_date",
              "description": "Boundary cross-currency payment: the applied part was converted at the rate of the "
                             "wall-clock date of the bank timestamp instead of the company-timezone date."}
        return [alloc, fx]

    # ------------------------------------------------------------- sizing
    def count_rows(self) -> int:
        n = 0
        for inv in self.invoices.values():
            if self.in_period(inv.issued):
                n += 1 + len(inv.lines)
        for p in self.payments.values():
            if self.in_period(p.ts):
                if p.amount < 0 or p.refs is None:
                    n += 2
                else:
                    n += 1 + sum(1 if self.invoices[k].currency == p.currency else 3 for k, _ in p.refs)
                    n += 1 if p.amount > sum(a for _, a in p.refs) else 0
        return n

    def enforce_size(self) -> None:
        r = self.rng("size")
        size = self.sampled["size"]
        custs = [k for k in sorted(self.customers) if self.customers[k].plan is None]
        guard = 0
        while self.count_rows() < size["min_ledger_rows"] + 30 and guard < 400:
            self.make_regular_invoice(r.choice(custs), r, 0, None)
            guard += 1
        while len(self.invoices) + len(self.payments) > size["max_documents"] - 14:
            cands = sorted([i for i in self.invoices.values() if not i.tags and not self.payments_of(i.key)],
                           key=lambda i: i.key)
            if not cands:
                break
            del self.invoices[r.choice(cands).key]
        if len(self.invoices) + len(self.payments) > size["max_documents"] - 14:
            raise RetryGeneration("documents exceed max_documents")

    # ------------------------------------------------------------ finalise
    def assign_ids(self) -> None:
        r = self.rng("ids")
        used: Set[str] = set()
        st = self.id_style

        def uniq(make):
            for _ in range(1000):
                v = make()
                if v not in used:
                    used.add(v)
                    return v
            raise GenerationError("id space exhausted")

        for k in sorted(self.customers):
            self.customers[k].entity_id = uniq(lambda: ("C-" + r.token(6)) if st["entity"] == 0 else ("CUST" + r.digits(6)))
        n = st["invoice_start"]
        for inv in sorted(self.invoices.values(), key=lambda i: (i.issued, i.key)):
            n += 1
            if st["invoice"] == 0:
                inv.invoice_id = uniq(lambda: f"INV-{self.year}-{n:05d}")
            elif st["invoice"] == 1:
                inv.invoice_id = uniq(lambda: f"{st['invoice_prefix']}{n:06d}")
            elif st["invoice"] == 2:
                inv.invoice_id = uniq(lambda: f"SF/{self.year}/{n:05d}")
            else:
                inv.invoice_id = uniq(lambda: f"B{self.year % 100:02d}-{n:05d}")
        docs: List[Any] = sorted(list(self.invoices.values()) + list(self.payments.values()), key=lambda d: d.key)
        for d in docs:
            d.doc_id = uniq(lambda: ("DOC-" + r.token(8)) if st["doc"] == 0 else ("D" + r.token(9)))
        for p in sorted(self.payments.values(), key=lambda p: p.key):
            p.payment_id = uniq(lambda: ("PAY-" + r.token(9)) if st["payment"] == 0 else ("TX" + r.digits(10)))
        all_rows = [row for rows in self.journals.values() for row in rows]
        all_rows += [row for rows in self.removed_journals.values() for row in rows]
        all_rows += self.removed_rows
        for row in sorted(all_rows, key=lambda x: int(x.key[1:])):
            row.row_id = uniq(lambda: ("L-" + r.token(8)) if st["row"] == 0 else ("GL" + r.digits(9)))

    def row_dict(self, row: Row) -> Dict[str, Any]:
        return {"row_id": row.row_id, "timestamp": fmt(row.ts), "account": row.account,
                "debit_minor": row.debit, "credit_minor": row.credit, "currency": row.currency,
                "entity_id": self.customers[row.entity].entity_id if row.entity else None,
                "invoice_id": self.invoices[row.invoice].invoice_id if row.invoice else None,
                "payment_id": self.payments[row.payment].payment_id if row.payment else None}

    def invoice_dict(self, inv: Invoice) -> Dict[str, Any]:
        return {"doc_id": inv.doc_id, "invoice_id": inv.invoice_id, "entity_id": self.customers[inv.cust].entity_id,
                "issued_at": render(inv.issued, inv.offset),
                "due_at": render(inv.issued + timedelta(days=inv.terms_days), inv.offset),
                "currency": inv.currency, "total_minor": inv.total, "line_items": copy.deepcopy(inv.lines)}

    def payment_dict(self, p: Payment) -> Dict[str, Any]:
        refs = None if p.refs is None else [{"invoice_id": self.invoices[k].invoice_id, "amount_minor": a} for k, a in p.refs]
        return {"doc_id": p.doc_id, "payment_id": p.payment_id, "timestamp": render(p.ts, p.offset),
                "currency": p.currency, "amount_minor": p.amount, "invoice_refs": refs}

    def ledger_dicts(self, journals: Dict[str, List[Row]]) -> List[Dict[str, Any]]:
        def doc_of(j: str):
            return self.invoices[j[2:]] if j.startswith("I:") else self.payments[j[2:]]
        rows = [row for rows in journals.values() for row in rows]
        rows.sort(key=lambda x: (x.ts.astimezone(timezone.utc), doc_of(x.journal).doc_id, x.seq))
        return [self.row_dict(x) for x in rows]

    def resolve_ref(self, ref: Tuple[str, str]) -> str:
        kind, k = ref
        if kind == "doc":
            return (self.invoices.get(k) or self.payments[k]).doc_id
        if kind == "row":
            return self.row_by_key[k].row_id
        if kind == "fx":
            return k
        if kind == "inv":
            return self.invoices[k].invoice_id
        if kind == "pay":
            return self.payments[k].payment_id
        raise AssertionError(kind)

    # ---------------------------------------------------------------- run
    def run(self) -> Episode:
        from .checker import reconcile
        self.sampled = sample_config(self.cfg, self.rng("config"))
        self.removed_journals: Dict[str, List[Row]] = {}
        self.removed_rows: List[Row] = []
        self.make_company()
        self.make_fx()
        self.make_customers()
        self.make_history()
        self.make_distractors()
        self.make_hosts()
        self.enforce_size()
        self.book_clean()
        clean_journals = {j: list(rows) for j, rows in self.journals.items()}
        clean_invoice_keys = set(self.invoices)
        issues_internal = self.inject()
        # rows that disappeared from modified journals
        final_keys = {row.key for rows in self.journals.values() for row in rows}
        for j, rows in clean_journals.items():
            if j in self.removed_journals:
                continue
            for row in rows:
                if row.key not in final_keys:
                    self.removed_rows.append(row)
        self.assign_ids()
        self.row_by_key = {row.key: row for rows in self.journals.values() for row in rows}
        for rows in list(self.removed_journals.values()) + [self.removed_rows]:
            for row in rows:
                self.row_by_key[row.key] = row

        ledger = self.ledger_dicts(self.journals)
        if not self.sampled["size"]["min_ledger_rows"] <= len(ledger) <= self.sampled["size"]["max_ledger_rows"]:
            raise RetryGeneration(f"ledger size {len(ledger)} outside configured bounds")
        invoices = [self.invoice_dict(i) for i in sorted(self.invoices.values(), key=lambda i: (i.issued, i.invoice_id))]
        payments = [self.payment_dict(p) for p in sorted(self.payments.values(), key=lambda p: (p.ts, p.payment_id))]
        fx_rows = [{"date": d, "source_currency": a, "target_currency": b, "rate": v}
                   for (d, a, b), v in sorted(self.fx.items())]
        episode_id = "EP-" + derive_hex("episode-id", self.key, self.attempt, n=12).upper()

        issues = []
        for it in issues_internal:
            issues.append({
                "type": it["type"],
                "affected_ids": sorted(self.resolve_ref(a) for a in it["affected"]),
                "expected_amount_minor": it["amount"],
                "currency": it["currency"],
                "tolerance_minor": money.amount_tolerance(it["amount"]),
                "evidence_sets": sorted({tuple(sorted(self.resolve_ref(e) for e in s)) for s in it["evidence_sets"]}),
                "interaction_id": it["interaction_id"],
                "variant": it.get("variant"),
                "description": it["description"],
            })
        issues.sort(key=lambda x: (ERROR_TYPES.index(x["type"]), x["affected_ids"]))
        for n, it in enumerate(issues, start=1):
            it["issue_id"] = f"ISS-{n:02d}"
            it["evidence_sets"] = [list(s) for s in it["evidence_sets"]]
        issues = [{k: it[k] for k in ("issue_id", "type", "affected_ids", "expected_amount_minor", "currency",
                                      "tolerance_minor", "evidence_sets", "interaction_id", "variant", "description")}
                  for it in issues]
        keyset = [(i["type"], tuple(i["affected_ids"])) for i in issues]
        if len(set(keyset)) != len(keyset):
            raise RetryGeneration("two issues share type and affected ids")

        manifest = build_manifest(
            episode_id=episode_id, company_name=self.company_name, base_currency=self.base,
            currencies=sorted(self.currencies), period_start=fmt(self.start_utc.astimezone(tz(self.offset))),
            period_end=fmt(self.end_utc.astimezone(tz(self.offset))), timezone=offset_str(self.offset),
            counts={"ledger_rows": len(ledger), "invoices": len(invoices), "payments": len(payments),
                    "fx_rates": len(fx_rows), "documents": len(invoices) + len(payments)})
        policy = build_policy(manifest)
        truth = {"schema_version": SCHEMA_VERSION, "episode_id": episode_id, "issues": issues}

        added_docs = sorted(self.invoices[k].doc_id for k in self.invoices if k not in clean_invoice_keys)
        added_rows = sorted(row.row_id for j, rows in self.journals.items() for row in rows
                            if row.key not in {x.key for x in clean_journals.get(j, [])})
        removed = [self.row_dict(row) for rows in self.removed_journals.values() for row in rows]
        removed += [self.row_dict(row) for row in self.removed_rows]
        changes = []
        for note in self.change_notes:
            c = dict(note)
            if "journal" in c:
                j = c["journal"]
                doc = self.invoices.get(j[2:]) if j.startswith("I:") else self.payments.get(j[2:])
                c["journal"] = {"document": doc.doc_id}
            if "documents" in c:
                c["documents"] = [self.invoices[k].doc_id for k in c["documents"]]
            if "copy_of" in c:
                c["copy_of"] = self.invoices[c["copy_of"]].doc_id
            changes.append(c)
        distractors = []
        for dlog in self.distractor_log:
            e = {"kind": dlog["kind"]}
            for fld in ("invoice", "doc"):
                if fld in dlog:
                    k = dlog[fld]
                    e[fld] = (self.invoices.get(k) or self.payments[k]).doc_id
            if "payments" in dlog:
                e["payments"] = [self.payments[k].doc_id for k in dlog["payments"] if k in self.payments]
            if "invoices" in dlog:
                e["invoices"] = [self.invoices[k].doc_id for k in dlog["invoices"] if k in self.invoices]
            if "customers" in dlog:
                e["customers"] = [self.customers[k].entity_id for k in dlog["customers"]]
            for fld in ("side", "in_period"):
                if fld in dlog:
                    e[fld] = dlog[fld]
            distractors.append(e)
        provenance = {
            "schema_version": SCHEMA_VERSION, "episode_id": episode_id,
            "generator_version": GENERATOR_VERSION, "env_version": ENV_VERSION,
            "seed": None if self.secret_used else self.seed, "secret_used": self.secret_used,
            "attempt": self.attempt, "config": self.cfg, "sampled": self.sampled,
            "company": {"name": self.company_name, "base_currency": self.base, "currencies": self.currencies,
                        "timezone": offset_str(self.offset), "period": f"{self.year:04d}-{self.month:02d}"},
            "construction_order": ["company", "fx", "customers", "history", "distractors", "error_hosts",
                                   "clean_books", "error_injection", "ids", "self_check"],
            "distractors": distractors,
            "changes": changes,
            "added_documents": added_docs,
            "added_ledger_rows": added_rows,
            "removed_ledger_rows": sorted(removed, key=lambda x: x["row_id"]),
        }
        ep = Episode(manifest=manifest, policy=policy, ledger=ledger, invoices=invoices, payments=payments,
                     fx=fx_rows, truth=truth, provenance=provenance)
        # ---- self-check: independent reconciliation of public records must equal truth
        found, anomalies = reconcile(ep.public())
        want = sorted((i["type"], tuple(i["affected_ids"]), i["expected_amount_minor"], i["currency"]) for i in issues)
        got = sorted((i["type"], tuple(i["affected_ids"]), i["amount_minor"], i["currency"]) for i in found)
        # a failed self-check never ships: the build is retried with the next attempt index
        # (deterministic, recorded as provenance.attempt); after 8 attempts generation fails loudly
        if anomalies or want != got:
            raise RetryGeneration(f"self-check failed: anomalies={anomalies[:3]} want={want} got={got}")
        clean = ep.clean_public()
        found_c, anomalies_c = reconcile(clean)
        if found_c or anomalies_c:
            raise RetryGeneration(f"clean books do not reconcile: {found_c[:3]} {anomalies_c[:3]}")
        return ep
