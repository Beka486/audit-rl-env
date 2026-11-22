"""Oracle: evaluator-only. Reads truth and submits it with its first acceptable evidence set at step 1.

It checks that the maximum reward is reachable; it says nothing about agent ability.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import Policy


def oracle_findings(truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for issue in sorted(truth["issues"], key=lambda i: i["issue_id"]):
        evidence = issue["evidence_sets"][0]  # first acceptable set in truth order (stable under id remapping)
        out.append({"type": issue["type"], "affected_ids": list(issue["affected_ids"]),
                    "claimed_amount_minor": issue["expected_amount_minor"], "currency": issue["currency"],
                    "evidence_ids": list(evidence), "explanation": "oracle"})
    return out


class OracleAdapter(Policy):
    name = "oracle"
    evaluator_only = True

    def __init__(self, truth_provider=None) -> None:
        self.truth_provider = truth_provider

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.truth_provider is None:
            raise RuntimeError("oracle needs evaluator-side truth access")
        truth = self.truth_provider()
        return {"action": "submit", "args": {"findings": {"episode_id": observation["manifest"]["episode_id"],
                                                          "findings": oracle_findings(truth)}}}
