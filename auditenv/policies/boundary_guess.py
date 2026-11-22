"""Attack policy for the measured failure analysis: blind period-cutoff guessing.

It reads invoices, payments and ledger, then flags the k documents whose timestamps are closest
to a period boundary as period_cutoff (amount |total|, evidence = the document and its own ledger
rows). It never checks how a document is booked or which period its instant belongs to.
Before the fix (boundary distractors did not scale with cutoff errors) it earned a positive
mean reward on cutoff-heavy books; see DESIGN.md section 6 and ``python -m auditenv report``.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

from .base import TableReader


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class BoundaryGuesser(TableReader):
    name = "boundary_guess"
    k = 6

    def next_action(self) -> Dict[str, Any]:
        m = self.manifest
        start, end = _ts(m["period_start"]), _ts(m["period_end"])
        rows = defaultdict(list)
        for r in self.data["ledger"]:
            for key in ("invoice_id", "payment_id"):
                if r.get(key):
                    rows[r[key]].append(r["row_id"])
        docs = [(d["invoice_id"], d["doc_id"], _ts(d["issued_at"]), abs(d["total_minor"]), d["currency"])
                for d in self.data["invoices"]]
        docs += [(d["payment_id"], d["doc_id"], _ts(d["timestamp"]), abs(d["amount_minor"]), d["currency"])
                 for d in self.data["payments"]]
        docs.sort(key=lambda x: (min(abs((x[2] - start).total_seconds()), abs((x[2] - end).total_seconds())), x[0]))
        findings = [{"type": "period_cutoff", "affected_ids": [biz], "claimed_amount_minor": amt, "currency": cur,
                     "evidence_ids": ([doc] + sorted(set(rows.get(biz, []))))[:10], "explanation": "near a boundary"}
                    for biz, doc, _, amt, cur in docs[: self.k]]
        return {"action": "submit", "args": {"findings": {"episode_id": m["episode_id"], "findings": findings}}}
