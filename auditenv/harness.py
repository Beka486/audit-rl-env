"""Evaluation harness: runs policies on splits, records trajectories, aggregates diagnostics.

The harness is evaluator-side. Policies only receive observations and return actions.
With ``isolated=True`` the environment and grader run in a separate process (server.py) and
grades are read from the evaluator log after the run.
"""
from __future__ import annotations

import importlib
import json
import os
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .env import AuditEnv
from .generator import generate_episode
from .policies import ALL_POLICIES
from .remote import RemoteAuditEnv, RemotePolicy

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "configs" / "splits"


def load_split(name: str) -> Dict[str, Any]:
    """Load a split by name (configs/splits/<name>.json) or by path to a split JSON file.

    For closed runs the secret salt can be kept out of the file: if the environment variable
    AUDITENV_SECRET is set it replaces the split's ``secret``.
    """
    path = Path(name)
    if not (path.suffix == ".json" and path.exists()):
        path = SPLITS_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as fh:
        split = json.load(fh)
    if os.environ.get("AUDITENV_SECRET"):
        split["secret"] = os.environ["AUDITENV_SECRET"]
    return split


def make_policy(name: str, truth_provider=None):
    """A built-in policy name, or ``package.module:ClassName`` for an external policy with
    ``reset(seed)`` and ``act(observation) -> action``."""
    if ":" in name:
        mod, _, cls_name = name.partition(":")
        return getattr(importlib.import_module(mod), cls_name)()
    cls = ALL_POLICIES[name]
    return cls(truth_provider) if name == "oracle" else cls()


def run_episode(env, policy_name: str, seed: int, config: Dict[str, Any], policy_seed: int,
                record: bool = True, split: str = "", policy=None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    obs, info = env.reset(seed=seed, config=config)
    episode_id = obs["manifest"]["episode_id"]
    traj: List[Dict[str, Any]] = []
    if record:
        traj.append({"episode_id": episode_id, "split": split, "policy": policy_name, "step": 0, "action": None,
                     "observation": obs, "reward": 0.0, "terminated": False, "truncated": False})
    done, step, ret = False, 0, 0.0
    last_action = None
    if policy_name == "oracle" and isinstance(env, RemoteAuditEnv):
        last_action, (obs, reward, term, trunc, info) = env.oracle_submit()
        step, ret, done = 1, reward, True
        if record:
            traj.append({"episode_id": episode_id, "split": split, "policy": policy_name, "step": 1,
                         "action": last_action, "observation": obs, "reward": reward, "terminated": term, "truncated": trunc})
    else:
        if policy is None:
            policy = make_policy(policy_name, (lambda: env.episode.truth) if policy_name == "oracle" else None)
        policy.reset(policy_seed)
        while not done:
            action = policy.act(obs)
            last_action = action
            obs, reward, term, trunc, info = env.step(action)
            step += 1
            ret += reward
            done = term or trunc
            if record:
                traj.append({"episode_id": episode_id, "split": split, "policy": policy_name, "step": step,
                             "action": action, "observation": obs, "reward": reward, "terminated": term, "truncated": trunc})
    return {"episode_id": episode_id, "policy": policy_name, "steps": step, "agent_return": ret,
            "seconds": time.perf_counter() - t0, "trajectory": traj, "final_action": last_action}


def run_suite(policy_name: str, split: Dict[str, Any], mode: str = "eval", isolated: bool = True,
              record: bool = False, policy_seed: int = 0) -> List[Dict[str, Any]]:
    episodes = split["episodes"]
    secret = split.get("secret", "")
    results: List[Dict[str, Any]] = []
    if isolated:
        with tempfile.TemporaryDirectory() as td:
            log_path = str(Path(td) / "evaluator.jsonl")
            env = RemoteAuditEnv(log_path, mode=mode, secret=secret, oracle=(policy_name == "oracle"))
            # three processes: policy (no secret, observations only) | harness | environment + evaluator
            agent = RemotePolicy(policy_name) if policy_name != "oracle" else None
            try:
                for i, e in enumerate(episodes):
                    results.append(run_episode(env, policy_name, e["seed"], e["config"], policy_seed + i, record,
                                               split.get("split", ""), policy=agent))
            finally:
                env.close()
                if agent is not None:
                    agent.close()
            grades = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines() if line]
    else:
        env = AuditEnv(mode=mode, secret=secret)
        grades = []
        for i, e in enumerate(episodes):
            results.append(run_episode(env, policy_name, e["seed"], e["config"], policy_seed + i, record,
                                       split.get("split", "")))
            grades.append(env.last_evaluation())
    if len(grades) != len(results):
        raise RuntimeError("evaluator log does not match the number of episodes")
    for r, g in zip(results, grades):
        if g["episode_id"] != r["episode_id"]:
            raise RuntimeError("evaluator log out of order")
        r["grade"] = g
    return results


# ----------------------------------------------------------------- diagnosis
def submitted_payload(action: Any) -> Optional[Dict[str, Any]]:
    """The findings object of a final action in any accepted form (object or JSON text)."""
    if isinstance(action, (str, bytes)):
        try:
            action = json.loads(action)
        except (ValueError, RecursionError):
            return None
    if not isinstance(action, dict) or action.get("action") != "submit":
        return None
    args = action.get("args")
    payload = args.get("findings") if isinstance(args, dict) else None
    return payload if isinstance(payload, dict) else None


def diagnose(episode, findings_payload: Optional[Dict[str, Any]], grade: Dict[str, Any]) -> Dict[str, Any]:
    """Measured failure analysis for one graded episode (evaluator-side)."""
    from .grader import Grader
    truth = episode.truth["issues"]
    out = {"fn_by_type": Counter(), "fp_reasons": Counter(), "tp_by_type": Counter()}
    by_id = {i["issue_id"]: i for i in truth}
    for iid in grade.get("missed_issues", []):
        it = by_id[iid]
        key = it["type"] + ("+interaction" if it["interaction_id"] else "")
        out["fn_by_type"][key] += 1
    matched = {m["finding"]: m["issue_id"] for m in grade.get("matches", [])}
    for iid in matched.values():
        out["tp_by_type"][by_id[iid]["type"]] += 1
    if grade["status"] == "invalid_submission":
        out["fp_reasons"]["invalid_submission"] += 1
        return out
    if not findings_payload or grade["status"] != "submitted":
        return out
    g = Grader(episode.public(), episode.truth)
    distractor_docs: Dict[str, str] = {}
    for d in episode.provenance.get("distractors", []):
        for k in ("invoice", "doc"):
            if k in d:
                distractor_docs[d[k]] = d["kind"]
        for k in ("payments", "invoices"):
            for x in d.get(k, []):
                distractor_docs[x] = d["kind"]
    for idx, f in enumerate(findings_payload["findings"]):
        if idx in matched:
            continue
        nf = g.normalize(idx, f)
        same = [k for k, c in enumerate(g._canon) if c["type"] == nf.type and c["affected"] == nf.affected]
        if same:
            k = same[0]
            c = g._canon[k]
            if g.issues[k]["issue_id"] in matched.values() and g.eligible(nf, k):
                reason = "repeated_finding"
            elif nf.currency != c["currency"]:
                reason = "wrong_currency"
            elif abs(nf.amount - c["amount"]) > c["tolerance"]:
                reason = "wrong_amount"
            else:
                reason = "insufficient_evidence"
        else:
            kinds = {distractor_docs.get(g.doc_by_biz.get(a, a)) for a in nf.affected} - {None}
            other_type = any(c["affected"] == nf.affected for c in g._canon)
            if other_type:
                reason = "wrong_type"
            elif kinds:
                reason = "fooled_by_distractor:" + sorted(kinds)[0]
            elif nf.type == "period_cutoff" and _tz_ambiguous(episode, nf.affected):
                reason = "fooled_by_timezone_rendering"
            elif nf.type == "payment_allocation" and _cross_currency(episode, nf.affected):
                reason = "compared_currencies_without_conversion"
            elif nf.type == "duplicate_invoice":
                reason = "same_amount_different_line_items"
            else:
                reason = "false_accusation"
        out["fp_reasons"][reason] += 1
    return out


def _tz_ambiguous(episode, affected) -> bool:
    from .timeutil import local_date, parse, parse_offset
    off = parse_offset(episode.manifest["timezone"])
    for d in episode.invoices + episode.payments:
        bid = d.get("invoice_id") or d.get("payment_id")
        if bid in affected:
            ts = d.get("issued_at") or d.get("timestamp")
            if parse(ts).date() != local_date(parse(ts), off):
                return True
    return False


def _cross_currency(episode, affected) -> bool:
    inv_ccy = {i["invoice_id"]: i["currency"] for i in episode.invoices}
    for p in episode.payments:
        if p["payment_id"] in affected and p["invoice_refs"]:
            return any(inv_ccy.get(r["invoice_id"]) != p["currency"] for r in p["invoice_refs"])
    return False


def summarize(results: List[Dict[str, Any]], episodes=None) -> Dict[str, Any]:
    grades = [r["grade"] for r in results]

    def mean(key):
        return statistics.mean(g[key] for g in grades) if grades else 0.0

    tp, fp, fn = (sum(g[k] for g in grades) for k in ("tp", "fp", "fn"))
    s = {
        "episodes": len(results),
        "precision_mean": mean("precision"), "recall_mean": mean("recall"), "f1_mean": mean("f1"),
        "micro_precision": tp / (tp + fp) if tp + fp else 0.0, "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
        "return_mean": mean("reward"), "return_min": min(g["reward"] for g in grades),
        "return_max": max(g["reward"] for g in grades),
        "steps_mean": statistics.mean(r["steps"] for r in results),
        "seconds_mean": statistics.mean(r["seconds"] for r in results),
        "seconds_max": max(r["seconds"] for r in results),
        "invalid_submissions": sum(g["status"] == "invalid_submission" for g in grades),
        "truncated": sum(g["status"] in ("step_limit", "time_limit") for g in grades),
        "agent_visible_return_mean": statistics.mean(r["agent_return"] for r in results),
    }
    if episodes is not None:
        fn_t, fp_r, tp_t = Counter(), Counter(), Counter()
        for r, ep in zip(results, episodes):
            d = diagnose(ep, submitted_payload(r.get("final_action")), r["grade"])
            fn_t.update(d["fn_by_type"])
            fp_r.update(d["fp_reasons"])
            tp_t.update(d["tp_by_type"])
        s["tp_by_type"] = dict(sorted(tp_t.items()))
        s["fn_by_type"] = dict(sorted(fn_t.items()))
        s["fp_reasons"] = dict(sorted(fp_r.items()))
    return s


def episodes_of(split: Dict[str, Any]):
    return [generate_episode(e["seed"], e["config"], secret=split.get("secret", "")) for e in split["episodes"]]


def by_group(results: List[Dict[str, Any]], split: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Mean F1 / return per episode group (clean / interacting / other)."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r, e in zip(results, split["episodes"]):
        n = e["config"].get("name", "")
        g = "clean" if "clean" in n else ("interacting" if ("interact" in n or "boundary" in n or "misapplied" in n) else "other")
        groups[g].append(r["grade"])
    return {g: {"episodes": len(v), "f1_mean": statistics.mean(x["f1"] for x in v),
                "return_mean": statistics.mean(x["reward"] for x in v)} for g, v in sorted(groups.items())}
