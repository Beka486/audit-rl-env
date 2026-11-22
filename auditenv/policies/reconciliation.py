"""Simple reconciliation baseline (the kind of shortcut a naive script takes).

* duplicates: booked invoices with the same entity, currency and amount (no line-item check);
* allocation: compares remittance amounts with AR credits without currency conversion;
* fx: not checked;
* cutoff: decides period membership from the date written in the timestamp (ignores timezones).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .base import TableReader

AR = "1100-AR"


class SimpleReconciliation(TableReader):
    name = "reconciliation"

    def next_action(self) -> Dict[str, Any]:
        m = self.manifest
        lo, hi = m["period_start"][:10], m["period_end"][:10]
        inv = {i["invoice_id"]: i for i in self.data["invoices"]}
        pay = {p["payment_id"]: p for p in self.data["payments"]}
        inv_rows: Dict[str, List[dict]] = defaultdict(list)
        pay_rows: Dict[str, List[dict]] = defaultdict(list)
        for r in self.data["ledger"]:
            (pay_rows[r["payment_id"]] if r["payment_id"] else inv_rows[r["invoice_id"]]).append(r)
        out: List[Dict[str, Any]] = []

        def add(t, affected, amount, currency, evidence):
            out.append({"type": t, "affected_ids": sorted(affected), "claimed_amount_minor": max(0, int(amount)),
                        "currency": currency, "evidence_ids": evidence[:10], "explanation": "reconciliation rule"})

        groups: Dict[tuple, List[str]] = defaultdict(list)
        for iid, rows in sorted(inv_rows.items()):
            ar = [r for r in rows if r["account"] == AR]
            if ar and iid in inv:
                groups[(ar[0]["entity_id"], ar[0]["currency"], ar[0]["debit_minor"])].append(iid)
        for key, ids in groups.items():
            for a, b in zip(ids, ids[1:]):
                ra = [r for r in inv_rows[a] if r["account"] == AR][0]
                rb = [r for r in inv_rows[b] if r["account"] == AR][0]
                add("duplicate_invoice", [a, b], key[2], key[1], [inv[a]["doc_id"], inv[b]["doc_id"], ra["row_id"], rb["row_id"]])
        for pid, rows in sorted(pay_rows.items()):
            p = pay.get(pid)
            if not p or not p["invoice_refs"]:
                continue
            for ref in p["invoice_refs"]:
                credit = sum(r["credit_minor"] - r["debit_minor"] for r in rows
                             if r["account"] == AR and r["invoice_id"] == ref["invoice_id"])
                if ref["amount_minor"] - credit > 0:
                    other = [r for r in rows if r["account"] != "1000-BANK"]
                    add("payment_allocation", [pid, ref["invoice_id"]], ref["amount_minor"] - credit, p["currency"],
                        [p["doc_id"]] + [r["row_id"] for r in other[:3]])
        for iid, i in inv.items():
            naive_inside = lo <= i["issued_at"][:10] <= hi
            if naive_inside != (iid in inv_rows):
                ev = [i["doc_id"]] + ([inv_rows[iid][0]["row_id"]] if iid in inv_rows else [])
                add("period_cutoff", [iid], abs(i["total_minor"]), i["currency"], ev)
        for pid, p in pay.items():
            naive_inside = lo <= p["timestamp"][:10] <= hi
            if naive_inside != (pid in pay_rows):
                ev = [p["doc_id"]] + ([pay_rows[pid][0]["row_id"]] if pid in pay_rows else [])
                add("period_cutoff", [pid], abs(p["amount_minor"]), p["currency"], ev)
        return self.submit(out)
