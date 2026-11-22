"""Benign transformations used by the paired checks: record permutation and ID remapping.

Neither changes the accounting content, so a correct answer maps one-to-one onto the
transformed episode and must earn the same reward (with the new IDs).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

from .episode import Episode
from .rng import Rng, derive_hex

LETTERS = "ABCDEFGHJKMNPQRSTVWXYZ"


def _new_id(old: str, rng: Rng) -> str:
    cut = max(old.rfind("-"), old.rfind("/"))
    if cut >= 0:
        head, tail = old[:cut + 1], old[cut + 1:]
    else:
        k = 0
        while k < len(old) and old[k].isalpha():
            k += 1
        if k >= len(old) or k > 4:
            k = 1
        head, tail = old[:k], old[k:]
    out = []
    for ch in tail:
        if ch.isdigit():
            out.append(str(rng.below(10)))
        elif ch.isalpha():
            out.append(LETTERS[rng.below(len(LETTERS))])
        else:
            out.append(ch)
    return head + "".join(out)


def apply_transform(ep: Episode, spec: Dict[str, Any]) -> Tuple[Episode, Dict[str, str]]:
    seed = spec["seed"]
    permute = spec.get("permute", True)
    remap = spec.get("remap_ids", True)
    rng = Rng.from_key("transform", ep.episode_id, seed)
    ep = copy.deepcopy(ep)
    mapping: Dict[str, str] = {}
    if remap:
        old_ids = []
        for r in ep.ledger + ep.provenance.get("removed_ledger_rows", []):
            old_ids.append(r["row_id"])
        for d in ep.invoices:
            old_ids += [d["doc_id"], d["invoice_id"], d["entity_id"]]
        for d in ep.payments:
            old_ids += [d["doc_id"], d["payment_id"]]
        taken = set(old_ids)
        for old in old_ids:
            if old in mapping:
                continue
            for _ in range(1000):
                new = _new_id(old, rng)
                if new not in taken:
                    break
            else:  # pragma: no cover
                raise RuntimeError("cannot remap id")
            taken.add(new)
            mapping[old] = new

    def m(x):
        return mapping.get(x, x) if x is not None else None

    def remap_row(r):
        for k in ("row_id", "entity_id", "invoice_id", "payment_id"):
            r[k] = m(r[k])

    new_eid = "EP-" + derive_hex("transformed", ep.episode_id, seed, permute, remap, n=12).upper()
    for r in ep.ledger:
        remap_row(r)
    for d in ep.invoices:
        for k in ("doc_id", "invoice_id", "entity_id"):
            d[k] = m(d[k])
    for d in ep.payments:
        d["doc_id"], d["payment_id"] = m(d["doc_id"]), m(d["payment_id"])
        if d["invoice_refs"] is not None:
            for ref in d["invoice_refs"]:
                ref["invoice_id"] = m(ref["invoice_id"])
    ep.manifest["episode_id"] = new_eid
    ep.truth["episode_id"] = new_eid
    for it in ep.truth["issues"]:
        it["affected_ids"] = sorted(m(a) for a in it["affected_ids"])
        it["evidence_sets"] = [[m(e) for e in s] for s in it["evidence_sets"]]  # order kept: ids only
    prov = ep.provenance
    prov["episode_id"] = new_eid
    for r in prov.get("removed_ledger_rows", []):
        remap_row(r)
    prov["removed_ledger_rows"] = sorted(prov.get("removed_ledger_rows", []), key=lambda r: r["row_id"])
    prov["added_ledger_rows"] = sorted(m(x) for x in prov.get("added_ledger_rows", []))
    prov["added_documents"] = sorted(m(x) for x in prov.get("added_documents", []))
    for c in prov.get("changes", []):
        if isinstance(c.get("journal"), dict):
            c["journal"]["document"] = m(c["journal"]["document"])
        if "documents" in c:
            c["documents"] = [m(x) for x in c["documents"]]
        if "copy_of" in c:
            c["copy_of"] = m(c["copy_of"])
    for d in prov.get("distractors", []):
        for k in ("invoice", "doc"):
            if k in d:
                d[k] = m(d[k])
        for k in ("payments", "invoices", "customers"):
            if k in d:
                d[k] = [m(x) for x in d[k]]
    prov.setdefault("transforms", []).append({"seed": seed, "permute": permute, "remap_ids": remap})
    if permute:
        rng.shuffle(ep.ledger)
        rng.shuffle(ep.invoices)
        rng.shuffle(ep.payments)
        rng.shuffle(ep.fx)
        for d in ep.payments:
            if d["invoice_refs"] and len(d["invoice_refs"]) > 1:
                rng.shuffle(d["invoice_refs"])
    # the transformed public records must still reconcile to the transformed truth
    from .checker import reconcile
    found, anomalies = reconcile(ep.public())
    want = sorted((i["type"], tuple(i["affected_ids"]), i["expected_amount_minor"], i["currency"]) for i in ep.truth["issues"])
    got = sorted((i["type"], tuple(i["affected_ids"]), i["amount_minor"], i["currency"]) for i in found)
    if anomalies or want != got:  # pragma: no cover - would be a transform bug
        raise RuntimeError("transform changed the accounting content")
    mapping["__episode_id__"] = new_eid
    return ep, mapping


def map_findings(findings: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """Translate a findings.json payload written for the original episode into the transformed IDs."""
    out = copy.deepcopy(findings)
    out["episode_id"] = mapping.get("__episode_id__", out["episode_id"])
    for f in out["findings"]:
        f["affected_ids"] = [mapping.get(a, a) for a in f["affected_ids"]]
        f["evidence_ids"] = [mapping.get(e, e) for e in f["evidence_ids"]]
    return out
