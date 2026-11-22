"""Evaluator: findings validation, one-to-one matching, metrics and terminal reward.

Published rules (also in policy.json -> matching / reward):

* Structure first: findings.schema.json (types, required fields, at most 30 findings, at most
  10 evidence ids per finding counted before any de-duplication, explanation at most 1000
  Unicode code points, claimed_amount_minor a JSON integer and not a boolean). A structural error
  makes the whole submission invalid: reward -1. ``episode_id`` must equal the episode's id.
* No normalisation of values: ids, type and currency are compared as exact strings (case and
  whitespace matter, no prefix stripping, no doc_id <-> invoice_id aliasing). ``affected_ids``
  and ``evidence_ids`` are sets: order and repeats inside one list do not matter.
* A finding is *eligible* for an issue iff: same type; set(affected_ids) == canonical set;
  same currency; |claimed - expected| <= tolerance_minor of that issue (published in truth; 0 for
  every generated issue type, see policy.json -> matching.amount_tolerance); every evidence id is
  an existing public record id (row_id, doc_id, invoice_id, payment_id or FX:<date>:<SRC>:<TGT>);
  the evidence set contains at least one complete acceptable evidence set.
* Matching: maximum-cardinality one-to-one bipartite matching. Among all maximum matchings the
  chosen one is the lexicographically smallest sequence of pairs (issue_id, finding index from 0),
  with issue_ids sorted by Unicode code points. TP/FP/FN never depend on the order of findings.
* TP = matched pairs, FP = findings - TP (unmatched or repeated), FN = issues - TP.
  P = TP/(TP+FP), R = TP/(TP+FN) (0 when the denominator is 0). F1 = 2TP/(2TP+FP+FN);
  F1 = 1 when there are no issues and no findings; 0 when there are no issues but findings.
* reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1), computed with exact fractions.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, FrozenSet, List, Optional

from .schema_lite import check as schema_check
from .version import STEP_COST


def fx_record_id(r: Dict[str, Any]) -> str:
    return f"FX:{r['date']}:{r['source_currency']}:{r['target_currency']}"


@dataclass(frozen=True)
class NormFinding:
    index: int
    type: str
    affected: FrozenSet[str]
    amount: int
    currency: str
    evidence: FrozenSet[str]
    evidence_resolves: bool


class Grader:
    def __init__(self, public: Dict[str, Any], truth: Dict[str, Any]) -> None:
        self.episode_id = public["manifest"]["episode_id"]
        self.accepted_episode_ids = {self.episode_id, truth.get("episode_id", self.episode_id)}
        self.issues = sorted(truth["issues"], key=lambda i: i["issue_id"])
        # doc_id <-> business id maps are kept for diagnostics only; matching never uses them.
        self.doc_by_biz: Dict[str, str] = {}
        for inv in public["invoices"]:
            self.doc_by_biz[inv["invoice_id"]] = inv["doc_id"]
        for p in public["payments"]:
            self.doc_by_biz[p["payment_id"]] = p["doc_id"]
        self.record_ids = set(self.doc_by_biz) | set(self.doc_by_biz.values())
        self.record_ids.update(r["row_id"] for r in public["ledger"])
        self.record_ids.update(fx_record_id(r) for r in public["fx"])
        self._canon = [
            {"type": i["type"], "affected": frozenset(i["affected_ids"]),
             "amount": int(i["expected_amount_minor"]), "currency": i["currency"],
             "tolerance": int(i.get("tolerance_minor", 0) or 0),
             "sets": [frozenset(s) for s in i["evidence_sets"]]}
            for i in self.issues]

    # --------------------------------------------------------------- validation
    def validate(self, payload: Any) -> Optional[str]:
        errors = schema_check(payload, "findings")
        if errors:
            return "; ".join(errors[:3])
        if payload["episode_id"] not in self.accepted_episode_ids:
            return "$.episode_id: does not match the current episode"
        return None

    def normalize(self, index: int, f: Dict[str, Any]) -> NormFinding:
        evidence = frozenset(f["evidence_ids"])
        return NormFinding(index, f["type"], frozenset(f["affected_ids"]), int(f["claimed_amount_minor"]),
                           f["currency"], evidence, all(e in self.record_ids for e in evidence))

    def eligible(self, nf: NormFinding, k: int) -> bool:
        c = self._canon[k]
        return (nf.type == c["type"] and nf.affected == c["affected"] and nf.currency == c["currency"]
                and abs(nf.amount - c["amount"]) <= c["tolerance"]
                and nf.evidence_resolves and any(s <= nf.evidence for s in c["sets"]))

    # ----------------------------------------------------------------- matching
    def match(self, findings: List[NormFinding]) -> Dict[int, int]:
        """Return {finding_index: issue_index} of the published deterministic maximum matching."""
        n_issues = len(self._canon)
        adj = [[fi for fi in range(len(findings)) if self.eligible(findings[fi], k)] for k in range(n_issues)]

        def max_matching(issues: List[int], banned: set) -> int:
            owner: Dict[int, int] = {}

            def augment(k: int, seen: set) -> bool:
                for fi in adj[k]:
                    if fi in banned or fi in seen:
                        continue
                    seen.add(fi)
                    if fi not in owner or augment(owner[fi], seen):
                        owner[fi] = k
                        return True
                return False

            return sum(augment(k, set()) for k in issues)

        # issues are already sorted by issue_id (Unicode code points)
        target = max_matching(list(range(n_issues)), set())
        chosen: Dict[int, int] = {}
        used: set = set()
        for k in range(n_issues):
            rest = list(range(k + 1, n_issues))
            need = target - len(chosen)
            if need == 0:
                break
            for fi in adj[k]:  # finding indices ascending
                if fi in used:
                    continue
                if 1 + max_matching(rest, used | {fi}) == need:
                    chosen[fi] = k
                    used.add(fi)
                    break
        return chosen

    # ------------------------------------------------------------------ grading
    def grade(self, payload: Any, inspection_steps: int, status: str = "submitted") -> Dict[str, Any]:
        n_issues = len(self.issues)
        result: Dict[str, Any] = {"episode_id": self.episode_id, "status": status, "error": None,
                                  "n_issues": n_issues, "inspection_steps": inspection_steps}
        if status == "submitted":
            err = self.validate(payload)
            if err is not None:
                result.update(status="invalid_submission", error=err, n_findings=None, tp=0, fp=0, fn=n_issues,
                              precision=0.0, recall=0.0, f1=0.0, reward=-1.0, matches=[], missed_issues=
                              [i["issue_id"] for i in self.issues])
                return result
            findings = [self.normalize(i, f) for i, f in enumerate(payload["findings"])]
        else:  # truncated (step or time limit): empty final answer
            findings = []
        pairs = self.match(findings)
        tp = len(pairs)
        fp = len(findings) - tp
        fn = n_issues - tp
        precision = Fraction(tp, tp + fp) if tp + fp else Fraction(0)
        recall = Fraction(tp, tp + fn) if tp + fn else Fraction(0)
        if n_issues == 0:
            f1 = Fraction(1) if not findings else Fraction(0)
        else:
            f1 = Fraction(2 * tp, 2 * tp + fp + fn)
        reward = terminal_reward(f1, inspection_steps)
        matched_issue_ids = {self.issues[k]["issue_id"] for k in pairs.values()}
        result.update(
            n_findings=len(findings), tp=tp, fp=fp, fn=fn, precision=float(precision), recall=float(recall),
            f1=float(f1), reward=float(reward),
            matches=[{"finding": fi, "issue_id": self.issues[k]["issue_id"]}
                     for fi, k in sorted(pairs.items(), key=lambda x: (x[1], x[0]))],
            missed_issues=[i["issue_id"] for i in self.issues if i["issue_id"] not in matched_issue_ids],
        )
        return result


def terminal_reward(f1: Fraction, inspection_steps: int) -> Fraction:
    r = 2 * Fraction(f1) - 1 - Fraction(STEP_COST) * inspection_steps
    return max(Fraction(-1), min(Fraction(1), r))
