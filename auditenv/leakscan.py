"""Leak scanner: proves that agent-visible text contains nothing from the closed evaluator files.

Forbidden in any observation / trajectory / public file:
* every truth issue_id, every row_id that exists only in provenance (removed rows);
* evaluator-only key names (issue_id, evidence_sets, expected_amount_minor, tolerance_minor,
  interaction_id, provenance, truth, removed_ledger_rows, added_ledger_rows, added_documents);
* injection variant names and change reasons from provenance;
* the secret salt and filesystem paths of the episode directories.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

PRIVATE_KEYS = ["issue_id", "evidence_sets", "expected_amount_minor", "tolerance_minor", "interaction_id",
                "provenance", "truth", "removed_ledger_rows", "added_ledger_rows", "added_documents",
                "copy_of", "distractors", "secret_used"]
VARIANT_WORDS = ["document_wall_clock_date", "invoice_issue_date", "next_business_day", "previous_business_day",
                 "reverse_pair_rate",
                 "early_end_", "late_end_", "early_start_", "late_start_", "duplicate_misapplied",
                 "fx_partial_boundary"]
PATH_RE = re.compile(r"(/home/|/tmp/|/mnt/|[A-Za-z]:\\\\|private/|episodes/)")


def forbidden_tokens(episode, secret: str = "") -> List[str]:
    toks = [i["issue_id"] for i in episode.truth["issues"]]
    toks += [r["row_id"] for r in episode.provenance.get("removed_ledger_rows", [])]
    toks += PRIVATE_KEYS + VARIANT_WORDS
    toks += [i["description"] for i in episode.truth["issues"] if i.get("description")]
    for c in episode.provenance.get("changes", []):
        if c.get("reason"):
            toks.append(c["reason"])
    if secret:
        toks.append(secret)
    return toks


def scan_text(text: str, tokens: Iterable[str]) -> List[str]:
    hits = [t for t in tokens if t and t in text]
    if PATH_RE.search(text):
        hits.append("filesystem path")
    return hits


def scan_observations(observations: Iterable[Dict[str, Any]], episode, secret: str = "") -> List[str]:
    toks = forbidden_tokens(episode, secret)
    hits: List[str] = []
    for obs in observations:
        hits += scan_text(json.dumps(obs, ensure_ascii=False), toks)
    return sorted(set(hits))
