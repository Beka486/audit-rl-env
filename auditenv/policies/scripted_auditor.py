"""Scripted auditor (demo policy): applies the published booking policy through the tools only.

Reads every table page by page, fetches line items only for invoices whose (entity, currency,
total) collide, fetches the policy FX rate only for booked cross-currency payments, then
submits findings with evidence. It never reads truth; it shows the task is solvable within
the step budget and that careful evidence analysis beats shortcuts.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, localcontext, Context
from typing import Any, Dict, List, Optional

from .base import TableReader

AR, FXC, UNAPPLIED = "1100-AR", "1950-FX-CLEARING", "2400-UNAPPLIED-CASH"


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _offset(s: str) -> timezone:
    sign = 1 if s[0] == "+" else -1
    return timezone(sign * timedelta(hours=int(s[1:3]), minutes=int(s[4:6])))


def _convert(amount: int, e_src: int, e_tgt: int, rate: str) -> int:
    with localcontext(Context(prec=60)):
        v = Decimal(amount).scaleb(-e_src) * Decimal(rate)
        return int(v.scaleb(e_tgt).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


class ScriptedAuditor(TableReader):
    name = "scripted_auditor"

    def reset(self, seed: int = 0) -> None:
        super().reset(seed)
        self.fetch_queue: Optional[List[Dict[str, Any]]] = None
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.rates: Dict[str, str] = {}

    def on_result(self, action: str, result: Dict[str, Any]) -> None:
        if action == "get_document":
            d = result["document"]
            self.docs[d["doc_id"]] = d
        elif action == "get_record" and result["table"] == "fx":
            self.rates[result["record"]["fx_id"]] = result["record"]["rate"]

    # -------------------------------------------------------------- analysis
    def _prepare(self) -> None:
        m = self.manifest
        self.start, self.end = _parse(m["period_start"]), _parse(m["period_end"])
        self.tz = _offset(m["timezone"])
        self.exp = {c: v["exponent"] for c, v in m["currencies"].items()}
        self.inv = {i["invoice_id"]: i for i in self.data["invoices"]}
        self.pay = {p["payment_id"]: p for p in self.data["payments"]}
        self.inv_rows: Dict[str, List[dict]] = defaultdict(list)
        self.pay_rows: Dict[str, List[dict]] = defaultdict(list)
        for r in self.data["ledger"]:
            if r["payment_id"] is not None:
                self.pay_rows[r["payment_id"]].append(r)
            elif r["invoice_id"] is not None:
                self.inv_rows[r["invoice_id"]].append(r)
        groups: Dict[tuple, List[str]] = defaultdict(list)
        for iid in sorted(self.inv_rows):
            i = self.inv.get(iid)
            if i is not None:
                groups[(i["entity_id"], i["currency"], i["total_minor"])].append(iid)
        self.dup_groups = [g for g in groups.values() if len(g) >= 2]
        queue: List[Dict[str, Any]] = []
        for g in self.dup_groups:
            for iid in g:
                queue.append({"action": "get_document", "args": {"doc_id": self.inv[iid]["doc_id"]}})
        self.cross = []
        for pid, rows in sorted(self.pay_rows.items()):
            p = self.pay.get(pid)
            if p is None or not p["invoice_refs"] or p["amount_minor"] <= 0:
                continue
            for ref in p["invoice_refs"]:
                i = self.inv.get(ref["invoice_id"])
                if i is not None and i["currency"] != p["currency"]:
                    fid = f"FX:{self.rate_date(p['timestamp'])}:{p['currency']}:{i['currency']}"
                    self.cross.append((pid, ref["invoice_id"], fid))
                    if fid not in [q["args"].get("id") for q in queue]:
                        queue.append({"action": "get_record", "args": {"table": "fx", "id": fid}})
        self.fetch_queue = queue

    def inside(self, ts: str) -> bool:
        return self.start <= _parse(ts) <= self.end

    def rate_date(self, ts: str) -> str:
        d = _parse(ts).astimezone(self.tz).date()
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d.isoformat()

    def next_action(self) -> Dict[str, Any]:
        if self.fetch_queue is None:
            self._prepare()
        if self.fetch_queue and self.steps_remaining > 1:
            return self.fetch_queue.pop(0)
        return self.submit(self.findings())

    def findings(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        def add(t, affected, amount, currency, evidence, why):
            out.append({"type": t, "affected_ids": sorted(affected), "claimed_amount_minor": int(amount),
                        "currency": currency, "evidence_ids": evidence[:10], "explanation": why[:1000]})

        # period cutoff
        for iid, i in self.inv.items():
            booked, ins = iid in self.inv_rows, self.inside(i["issued_at"])
            if booked and not ins:
                add("period_cutoff", [iid], abs(i["total_minor"]), i["currency"], [i["doc_id"], self.inv_rows[iid][0]["row_id"]],
                    "Invoice issued outside the period (company timezone) but booked in this ledger.")
            elif ins and not booked:
                add("period_cutoff", [iid], abs(i["total_minor"]), i["currency"], [i["doc_id"]],
                    "Invoice issued inside the period (company timezone) has no ledger rows.")
        for pid, p in self.pay.items():
            booked, ins = pid in self.pay_rows, self.inside(p["timestamp"])
            if booked and not ins:
                add("period_cutoff", [pid], abs(p["amount_minor"]), p["currency"], [p["doc_id"], self.pay_rows[pid][0]["row_id"]],
                    "Payment outside the period (company timezone) booked in this ledger.")
            elif ins and not booked:
                add("period_cutoff", [pid], abs(p["amount_minor"]), p["currency"], [p["doc_id"]],
                    "Payment inside the period (company timezone) has no ledger rows.")
        # duplicates (need line items)
        for g in self.dup_groups:
            by_sig: Dict[tuple, List[str]] = defaultdict(list)
            for iid in g:
                d = self.docs.get(self.inv[iid]["doc_id"])
                if d is None:
                    continue
                sig = tuple(sorted((li["description"], li["quantity"], li["unit_price_minor"], li["amount_minor"],
                                    li["revenue_account"]) for li in d["line_items"]))
                by_sig[sig].append(iid)
            for ids in by_sig.values():
                if len(ids) == 2:
                    a, b = ids
                    ra = next(r for r in self.inv_rows[a] if r["account"] == AR)
                    rb = next(r for r in self.inv_rows[b] if r["account"] == AR)
                    i = self.inv[a]
                    add("duplicate_invoice", [a, b], i["total_minor"], i["currency"],
                        [self.inv[a]["doc_id"], self.inv[b]["doc_id"], ra["row_id"], rb["row_id"]],
                        "Two booked invoices with identical customer, currency, total and line items.")
        # allocation
        for pid, rows in sorted(self.pay_rows.items()):
            p = self.pay.get(pid)
            if p is None or not p["invoice_refs"]:
                continue
            refs = p["invoice_refs"]
            for ref in refs:
                i = self.inv.get(ref["invoice_id"])
                if i is None:
                    continue
                acct = AR if i["currency"] == p["currency"] else FXC
                applied_rows = [r for r in rows if r["account"] == acct and r["invoice_id"] == ref["invoice_id"]
                                and r["currency"] == p["currency"]]
                applied = sum(r["credit_minor"] - r["debit_minor"] for r in applied_rows)
                short = ref["amount_minor"] - applied
                if short > 0:
                    unap = [r for r in rows if r["account"] == UNAPPLIED]
                    stray = [r for r in rows if r["account"] in (AR, FXC) and r["invoice_id"] is not None
                             and r["invoice_id"] not in {x["invoice_id"] for x in refs}]
                    ev_row = (unap or stray or applied_rows)[0]
                    add("payment_allocation", [pid, ref["invoice_id"]], short, p["currency"], [p["doc_id"], ev_row["row_id"]],
                        "Remittance allocation not fully applied to the referenced invoice.")
        # fx conversion
        for pid, iid, fid in self.cross:
            rate = self.rates.get(fid)
            if rate is None:
                continue
            p, i, rows = self.pay[pid], self.inv[iid], self.pay_rows[pid]
            paid_rows = [r for r in rows if r["account"] == FXC and r["invoice_id"] == iid and r["currency"] == p["currency"]]
            applied = sum(r["credit_minor"] - r["debit_minor"] for r in paid_rows)
            ar_rows = [r for r in rows if r["account"] == AR and r["invoice_id"] == iid and r["currency"] == i["currency"]]
            if applied <= 0 or not ar_rows:
                continue
            booked = sum(r["credit_minor"] - r["debit_minor"] for r in ar_rows)
            good = _convert(applied, self.exp[p["currency"]], self.exp[i["currency"]], rate)
            if booked != good:
                add("fx_conversion", [pid, iid], abs(booked - good), i["currency"],
                    [p["doc_id"], fid, ar_rows[0]["row_id"], paid_rows[0]["row_id"]],
                    "Cross-currency allocation not converted at the policy rate.")
        return out
