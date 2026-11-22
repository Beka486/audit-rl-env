"""Attack baseline "flag everything": every document under every type, evidence sprayed to the limit."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import TableReader

TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")


class FlagAll(TableReader):
    name = "flag_all"
    tables = ("invoices", "payments")

    def next_action(self) -> Dict[str, Any]:
        invs, pays = self.data["invoices"], self.data["payments"]
        docs = [d["doc_id"] for d in invs + pays]
        cands: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TYPES}

        def mk(t, affected, amount, currency, anchor):
            spray = [anchor] + [d for d in docs if d != anchor][:9]
            return {"type": t, "affected_ids": affected, "claimed_amount_minor": abs(int(amount)), "currency": currency,
                    "evidence_ids": spray, "explanation": "flag everything"}

        for a, b in zip(invs, invs[1:]):
            cands["duplicate_invoice"].append(mk("duplicate_invoice", [a["invoice_id"], b["invoice_id"]],
                                                 a["total_minor"], a["currency"], a["doc_id"]))
        for p in pays:
            refs = p["invoice_refs"] or []
            for ref in refs[:1]:
                cands["payment_allocation"].append(mk("payment_allocation", [p["payment_id"], ref["invoice_id"]],
                                                      ref["amount_minor"], p["currency"], p["doc_id"]))
                cands["fx_conversion"].append(mk("fx_conversion", [p["payment_id"], ref["invoice_id"]],
                                                 ref["amount_minor"] // 100, p["currency"], p["doc_id"]))
        for d in invs:
            cands["period_cutoff"].append(mk("period_cutoff", [d["invoice_id"]], d["total_minor"], d["currency"], d["doc_id"]))
        for p in pays:
            cands["period_cutoff"].append(mk("period_cutoff", [p["payment_id"]], p["amount_minor"], p["currency"], p["doc_id"]))
        out: List[Dict[str, Any]] = []
        i = 0
        while len(out) < 30 and any(cands.values()):
            t = TYPES[i % len(TYPES)]
            if cands[t]:
                out.append(cands[t].pop(0))
            i += 1
        return self.submit(out)
