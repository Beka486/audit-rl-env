"""Random explorer with a fixed seed: random valid tool calls, then a few random findings."""
from __future__ import annotations

from typing import Any, Dict, List

from ..rng import Rng
from .base import Policy

TYPES = ["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"]
TABLES = ["ledger", "invoices", "payments", "fx"]


class RandomExplorer(Policy):
    name = "random"

    def reset(self, seed: int = 0) -> None:
        self.rng = Rng.from_key("random-explorer", seed)
        self.budget = self.rng.randint(5, 40)
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.amounts: List[int] = []
        self.currencies: List[str] = []
        self.cursors: Dict[str, Any] = {t: None for t in TABLES}
        self.episode_id = None
        self.used = 0

    def _absorb(self, rec: Dict[str, Any]) -> None:
        for k in ("row_id", "invoice_id", "payment_id", "fx_id"):
            if rec.get(k):
                self.ids.append(rec[k])
        if rec.get("doc_id"):
            self.docs.append(rec["doc_id"])
        for k in ("debit_minor", "credit_minor", "total_minor", "amount_minor"):
            if isinstance(rec.get(k), int) and rec[k] > 0:
                self.amounts.append(rec[k])
        if rec.get("currency"):
            self.currencies.append(rec["currency"])

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        r = self.rng
        if obs["kind"] == "reset":
            self.episode_id = obs["manifest"]["episode_id"]
            self.currencies = list(obs["manifest"]["currencies"])
        elif obs["kind"] == "result":
            res = obs["result"]
            if obs["action"] == "list_records":
                for rec in res["records"]:
                    self._absorb(rec)
                self.cursors[res["table"]] = res["next_cursor"]
            elif "record" in res:
                self._absorb(res["record"])
            elif "document" in res:
                self._absorb(res["document"])
        self.used += 1
        if self.used > self.budget or obs.get("steps_remaining", 100) <= 1:
            return self._submit()
        roll = r.below(4)
        if roll == 0 or not self.ids:
            t = r.choice(TABLES)
            return {"action": "list_records", "args": {"table": t, "cursor": self.cursors[t], "limit": r.randint(1, 50)}}
        if roll == 1:
            rid = r.choice(self.ids)
            return {"action": "get_record", "args": {"table": r.choice(TABLES), "id": rid}}
        if roll == 2 and self.docs:
            return {"action": "get_document", "args": {"doc_id": r.choice(self.docs)}}
        a, b = (r.choice(self.amounts) if self.amounts else 1), r.randint(1, 1000)
        return {"action": "calculator", "args": {"expression": f"{a} {r.choice('+-*/')} {b}"}}

    def _submit(self) -> Dict[str, Any]:
        r = self.rng
        findings = []
        pool = self.ids + self.docs
        for _ in range(r.randint(0, 3) if pool else 0):
            findings.append({"type": r.choice(TYPES), "affected_ids": r.sample(pool, min(len(pool), r.randint(1, 2))),
                             "claimed_amount_minor": r.choice(self.amounts) if self.amounts else 0,
                             "currency": r.choice(self.currencies), "evidence_ids": r.sample(pool, min(len(pool), 3)),
                             "explanation": "random guess"})
        return {"action": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": findings}}}
