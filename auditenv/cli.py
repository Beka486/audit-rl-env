"""Command line: ``python -m auditenv <command>``.

  check          the single official verification command (tests + data + paired + isolation + report)
  generate       export split episodes (public/ + private/) with an index of hashes
  validate       independent data validation of exported episodes
  baselines      run the four baseline policies (+ scripted auditor) and print diagnostics
  trajectories   record trajectories.jsonl (open training mode, no oracle)
  evaluate       closed-mode run of any policy (built-in or module:Class) on any split file
  report         write reports/baseline_report.{json,md} with P/R/F1/return/steps/time and failure analysis
  demo           step-by-step run of the scripted auditor on one episode
  serve          run the out-of-process environment server (JSON lines on stdio)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

from .version import ENV_VERSION, GENERATOR_VERSION, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "episodes"
REPORTS = ROOT / "reports"
TRAJ = ROOT / "trajectories" / "trajectories.jsonl"
POLICIES = ["random", "reconciliation", "flag_all", "oracle", "scripted_auditor"]


def _split(name: str) -> Dict[str, Any]:
    from .harness import load_split
    try:
        return load_split(name)
    except FileNotFoundError:
        raise SystemExit(f"error: unknown split '{name}' (use a name from configs/splits or a path to a .json file)") from None


# ------------------------------------------------------------------ generate
def export_split(name: str, out_root: Path) -> Dict[str, Any]:
    from .generator import generate_episode
    split = _split(name)
    d = out_root / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    index = {"split": name, "env_version": ENV_VERSION, "generator_version": GENERATOR_VERSION,
             "schema_version": SCHEMA_VERSION, "config_file": f"configs/splits/{name}.json", "episodes": []}
    for i, e in enumerate(split["episodes"], start=1):
        ep = generate_episode(e["seed"], e["config"], secret=split.get("secret", ""))
        sub = d / f"{i:02d}_{ep.episode_id}"
        ep.write(sub)
        index["episodes"].append({"index": i, "dir": sub.name, "episode_id": ep.episode_id, "seed": e["seed"],
                                  "config_name": e["config"].get("name"), "public_sha256": ep.public_sha256(),
                                  "full_sha256": ep.full_sha256(), "ledger_rows": len(ep.ledger),
                                  "documents": ep.manifest["counts"]["documents"], "issues": len(ep.truth["issues"])})
    (d / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def cmd_generate(a) -> int:
    for name in (["train", "dev"] if a.split == "all" else [a.split]):
        idx = export_split(name, Path(a.out))
        print(f"{name}: {len(idx['episodes'])} episodes -> {Path(a.out) / name}")
    return 0


# ------------------------------------------------------------------ validate
def validate_root(root: Path) -> Dict[str, Any]:
    try:
        from .validate_data import validate_tree
    except ImportError:  # pragma: no cover
        validate_tree = None
    if validate_tree is not None:
        return validate_tree(str(root))
    from .checker import reconcile
    from .episode import Episode
    results = []
    for p in sorted(root.rglob("public")):
        ep = Episode.load(p.parent)
        found, an = reconcile(ep.public())
        results.append({"episode_id": ep.episode_id, "ok": not an})
    return {"episodes": len(results), "ok": sum(r["ok"] for r in results), "results": results}


def cmd_validate(a) -> int:
    s = validate_root(Path(a.root))
    print(f"validated {s['episodes']} episodes, ok: {s['ok']}")
    if not s["episodes"]:
        print(f"error: no episodes found under {a.root}", file=sys.stderr)
        return 1
    for r in s["results"]:
        if not r["ok"]:
            print("FAIL", r.get("episode_id"), r.get("errors", [])[:5])
    return 0 if s["ok"] == s["episodes"] else 1


# ----------------------------------------------------------------- baselines
def fmt_table(rows: List[Dict[str, Any]]) -> str:
    head = "| policy | episodes | P (mean) | R (mean) | F1 (mean) | micro P | micro R | return (mean) | return min..max | steps (mean) | time s (mean / max) | invalid | truncated |"
    sep = "|" + "---|" * 13
    lines = [head, sep]
    for r in rows:
        s = r["summary"]
        lines.append(f"| {r['policy']} | {s['episodes']} | {s['precision_mean']:.3f} | {s['recall_mean']:.3f} | "
                     f"{s['f1_mean']:.3f} | {s['micro_precision']:.3f} | {s['micro_recall']:.3f} | {s['return_mean']:.3f} | "
                     f"{s['return_min']:.3f}..{s['return_max']:.3f} | {s['steps_mean']:.1f} | "
                     f"{s['seconds_mean']:.3f} / {s['seconds_max']:.3f} | {s['invalid_submissions']} | {s['truncated']} |")
    return "\n".join(lines)


def run_baselines(split_name: str, isolated: bool = True, policies=POLICIES) -> Dict[str, Any]:
    from .harness import by_group, episodes_of, run_suite, summarize
    split = _split(split_name)
    eps = episodes_of(split)
    rows = []
    for pol in policies:
        res = run_suite(pol, split, mode="eval", isolated=isolated, record=False)
        rows.append({"policy": pol, "summary": summarize(res, eps), "groups": by_group(res, split),
                     "per_episode": [{"episode_id": r["episode_id"], "config": e["config"].get("name"),
                                      "issues": len(ep.truth["issues"]), "tp": r["grade"]["tp"], "fp": r["grade"]["fp"],
                                      "fn": r["grade"]["fn"], "f1": r["grade"]["f1"], "return": r["grade"]["reward"],
                                      "steps": r["steps"], "seconds": round(r["seconds"], 4)}
                                     for r, e, ep in zip(res, split["episodes"], eps)]})
    return {"split": split_name, "isolated": isolated, "rows": rows}


def cmd_baselines(a) -> int:
    out = run_baselines(a.split, isolated=not a.in_process)
    print(f"split={a.split} isolated={out['isolated']} (closed mode: rewards withheld from policies)")
    print(fmt_table(out["rows"]))
    return 0


def policy_load_error(name: str) -> str:
    """Load the policy in a guarded policy process, never in the evaluator's. Empty string if it loads."""
    from .remote import policy_env
    if name == "oracle":
        return ""
    try:
        p = subprocess.run([sys.executable, "-m", "auditenv.agent_runner", "--policy", name, "--check"],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=policy_env(), cwd=os.getcwd(), timeout=30)
    except subprocess.TimeoutExpired:
        return "loading took longer than 30 s"
    return "" if p.returncode == 0 else (p.stderr.strip().splitlines() or ["cannot load"])[-1]


def cmd_evaluate(a) -> int:
    for name in a.policy:
        err = policy_load_error(name)
        if err:
            print(f"error: policy '{name}' cannot be loaded: {err}", file=sys.stderr)
            return 2
    out = run_baselines(a.split, isolated=True, policies=a.policy)
    out["split"] = Path(a.split).name
    print(f"split={out['split']} closed mode, environment and evaluator in a separate process")
    print(fmt_table(out["rows"]))
    for r in out["rows"]:
        s = r["summary"]
        print(f"- {r['policy']}: FN by type {s.get('fn_by_type', {})}; FP reasons {s.get('fp_reasons', {})}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


# -------------------------------------------------------------- trajectories
def record_trajectories(path: Path, split_name: str = "train",
                        policies=("random", "reconciliation", "flag_all", "scripted_auditor")) -> Dict[str, Any]:
    from .harness import run_suite
    split = _split(split_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    h = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as fh:
        for pol in policies:
            res = run_suite(pol, split, mode="train", isolated=False, record=True)
            for r in res:
                for rec in r["trajectory"]:
                    line = json.dumps(rec, sort_keys=True, ensure_ascii=False)
                    fh.write(line + "\n")
                    h.update(line.encode("utf-8"))
                    n += 1
    return {"records": n, "sha256": h.hexdigest(), "split": split_name, "policies": list(policies), "mode": "train"}


def cmd_trajectories(a) -> int:
    info = record_trajectories(Path(a.out), a.split)
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {info['records']} trajectory records -> {a.out}")
    return 0


# ---------------------------------------------------------------- paired checks
def paired_checks(split_name: str = "holdout_sim") -> List[Dict[str, Any]]:
    """20 checks: (a) replay of seed+actions is exact; (b) permutation + ID remap preserves the
    oracle answer (mapped) and its reward, and the scripted auditor's reward."""
    from .env import AuditEnv
    from .generator import generate_episode
    from .grader import Grader
    from .policies import ScriptedAuditor, oracle_findings
    from .transforms import apply_transform, map_findings
    split = _split(split_name)
    out = []
    for k, e in enumerate(split["episodes"]):
        secret = split.get("secret", "")
        ep = generate_episode(e["seed"], e["config"], secret=secret)
        # (a) exact replay
        def play(env, conf):
            obs, _ = env.reset(seed=e["seed"], config=conf)
            pol = ScriptedAuditor()
            pol.reset(0)
            lines, done = [json.dumps(obs, sort_keys=True)], False
            while not done:
                act = pol.act(obs)
                obs, r, t, tr, info = env.step(act)
                lines.append(json.dumps([act, obs, r, t, tr, info], sort_keys=True))
                done = t or tr
            return hashlib.sha256("\n".join(lines).encode()).hexdigest(), env.last_evaluation()["reward"]
        h1, r1 = play(AuditEnv(mode="train", secret=secret), e["config"])
        h2, r2 = play(AuditEnv(mode="train", secret=secret), e["config"])
        # (b) permutation + id remap
        spec = {"seed": 1000 + k, "permute": True, "remap_ids": True}
        tep, mapping = apply_transform(ep, spec)
        payload = {"episode_id": ep.episode_id, "findings": oracle_findings(ep.truth)}
        g0, g1 = Grader(ep.public(), ep.truth), Grader(tep.public(), tep.truth)
        or0 = g0.grade(payload, 0)["reward"]
        or1 = g1.grade(map_findings(payload, mapping), 0)["reward"]
        mapped_truth = sorted((i["type"], tuple(sorted(mapping.get(a, a) for a in i["affected_ids"])),
                               i["expected_amount_minor"], i["currency"]) for i in ep.truth["issues"])
        new_truth = sorted((i["type"], tuple(i["affected_ids"]), i["expected_amount_minor"], i["currency"])
                           for i in tep.truth["issues"])
        tconf = dict(e["config"], transform=spec)
        _, rs1 = play(AuditEnv(mode="train", secret=secret), tconf)
        ok = (h1 == h2 and r1 == r2 and or0 == or1 == 1.0 and mapped_truth == new_truth and rs1 == r1)
        out.append({"episode": e["config"].get("name"), "replay_identical": h1 == h2 and r1 == r2,
                    "oracle_reward_original": or0, "oracle_reward_transformed": or1,
                    "truth_maps_onto_transformed": mapped_truth == new_truth,
                    "scripted_reward_original": r1, "scripted_reward_transformed": rs1, "ok": ok})
    return out


# ------------------------------------------------------------------- report
def boundary_attack_check(seeds=range(20)) -> Dict[str, Any]:
    """Measured failure and its fix: blind boundary guessing on cutoff-heavy books (4 cutoff errors),
    without (boundary_per_cutoff=0, the original generator) and with the fix (default 2)."""
    import statistics
    from .harness import run_suite
    zero = {"duplicate_invoice": 0, "payment_allocation": 0, "fx_conversion": 0, "period_cutoff": 4}
    out = {}
    for label, per in (("before_fix", 0), ("after_fix", 2)):
        cfg = {"config_version": "1.0", "name": f"cutoff-heavy-boundary_per_cutoff-{per}", "errors": zero,
               "interactions": {"fx_partial_boundary": 0, "duplicate_misapplied": 0},
               "distractors": {"boundary_per_cutoff": per}}
        split = {"split": "attack", "secret": "", "episodes": [{"seed": s, "config": cfg} for s in seeds]}
        res = run_suite("boundary_guess", split, mode="eval", isolated=True)
        rewards = [r["grade"]["reward"] for r in res]
        out[label] = {"boundary_per_cutoff": per, "episodes": len(rewards), "return_mean": statistics.mean(rewards),
                      "positive_returns": sum(r > 0 for r in rewards),
                      "precision_mean": statistics.mean(r["grade"]["precision"] for r in res),
                      "tp": sum(r["grade"]["tp"] for r in res), "fp": sum(r["grade"]["fp"] for r in res)}
    return out


def write_report(out_dir: Path = REPORTS) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = {"env_version": ENV_VERSION, "generator_version": GENERATOR_VERSION, "python": sys.version.split()[0],
            "note": "Closed mode: policies received reward 0 at every step; grades come from the evaluator log of "
                    "the out-of-process server. Timing is wall-clock on the machine that ran the report.",
            "splits": {}}
    for name in ("train", "dev", "holdout_sim"):
        data["splits"][name] = run_baselines(name, isolated=True)
    data["splits"]["judge_matrix"] = run_baselines("judge_matrix", isolated=True)
    data["paired_checks"] = paired_checks("holdout_sim")
    data["boundary_attack"] = boundary_attack_check()
    data["elapsed_s"] = round(time.time() - t0, 2)
    (out_dir / "baseline_report.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    md = ["# Baseline diagnostics", "",
          f"env {ENV_VERSION}, generator {GENERATOR_VERSION}, Python {data['python']}. Generated by "
          "`python -m auditenv report`.", "",
          "Closed-mode runs: environment and evaluator in a separate process; policies saw reward 0.0 and "
          "empty info at every step. Reward is the common terminal reward clip(2*F1 - 1 - 0.002*steps, -1, 1). "
          "P/R are per-episode means (0 when the denominator is 0, e.g. clean books with no findings), "
          "micro P/R pool TP/FP/FN over the split.", ""]
    for name, s in data["splits"].items():
        md += [f"## Split `{name}`", "", fmt_table(s["rows"]), "", "Per group (mean F1 / mean return):", ""]
        for r in s["rows"]:
            g = ", ".join(f"{k}: {v['f1_mean']:.2f} / {v['return_mean']:.2f} (n={v['episodes']})" for k, v in r["groups"].items())
            md.append(f"- {r['policy']}: {g}")
        md += ["", "Failure analysis (counts over the split):", ""]
        for r in s["rows"]:
            sm = r["summary"]
            md.append(f"- **{r['policy']}** - TP by type {sm.get('tp_by_type', {})}; FN by type {sm.get('fn_by_type', {})}; "
                      f"FP reasons {sm.get('fp_reasons', {})}")
        md.append("")
    ba = data["boundary_attack"]
    md += ["## Measured failure and fix: blind boundary guessing", "",
           "Policy `boundary_guess` (auditenv/policies/boundary_guess.py) reads the tables and flags the 6 documents "
           "closest to a period boundary as period_cutoff, without any reconciliation. 20 cutoff-heavy episodes "
           "(4 cutoff errors, other errors 0), closed mode:", "",
           "| generator | boundary_per_cutoff | return (mean) | episodes with return > 0 | P (mean) | TP / FP |",
           "|---|---|---|---|---|---|"]
    for label, v in ba.items():
        md.append(f"| {label.replace('_', ' ')} | {v['boundary_per_cutoff']} | {v['return_mean']:.3f} | "
                  f"{v['positive_returns']}/{v['episodes']} | {v['precision_mean']:.3f} | {v['tp']} / {v['fp']} |")
    md += ["", "Cause: cutoff errors were built with the same boundary documents as the 2-5 boundary distractors, so "
           "with 4 errors a blind guess had about 50% precision. Fix: 2 extra boundary distractors per cutoff error "
           "(config `distractors.boundary_per_cutoff`, default 2). Re-check: tests/test_redteam.py "
           "test_blind_boundary_guessing_is_not_profitable_on_cutoff_heavy_books.", ""]
    pc = data["paired_checks"]
    md += ["## Paired checks (holdout_sim)", "",
           f"{sum(p['ok'] for p in pc)}/{len(pc)} passed: exact replay of seed+actions, and permutation + ID remap "
           "keep the (mapped) oracle answer at reward 1.0 and the scripted auditor's reward unchanged.", ""]
    (out_dir / "baseline_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return data


def cmd_report(a) -> int:
    data = write_report(Path(a.out))
    print(f"report written to {a.out} in {data['elapsed_s']} s")
    return 0


# --------------------------------------------------------------------- demo
def demo(split_name: str = "dev", index: int = 5, out: Path | None = None, stream=sys.stdout) -> Dict[str, Any]:
    from .env import AuditEnv
    from .policies import ScriptedAuditor
    split = _split(split_name)
    if not 1 <= index <= len(split["episodes"]):
        raise SystemExit(f"error: --index must be between 1 and {len(split['episodes'])}")
    e = split["episodes"][index - 1]
    env = AuditEnv(mode="train", secret=split.get("secret", ""))
    obs, info = env.reset(seed=e["seed"], config=e["config"])
    lines = [f"# Demo: scripted auditor on {split_name} episode {index} ({obs['manifest']['episode_id']})", "",
             f"Task: {obs['task']}", ""]
    pol = ScriptedAuditor()
    pol.reset(0)
    done = False
    while not done:
        act = pol.act(obs)
        obs, reward, term, trunc, info = env.step(act)
        done = term or trunc
        a = act["action"]
        if a == "list_records":
            desc = f"list_records({act['args']['table']}, cursor={act['args']['cursor']}) -> {len(obs['result']['records'])} records"
        elif a == "get_document":
            desc = f"get_document({act['args']['doc_id']}) -> {len(obs['result']['document'].get('line_items') or [])} line items"
        elif a == "get_record":
            res = obs.get("result", {}).get("record", {})
            desc = f"get_record(fx, {act['args']['id']}) -> rate {res.get('rate')}"
        elif a == "submit":
            desc = f"submit({len(act['args']['findings']['findings'])} findings)"
        else:
            desc = a
        lines.append(f"- step {obs['steps_used']:>3}: {desc}; reward={reward}")
        if a == "submit":
            for f in act["args"]["findings"]["findings"]:
                lines.append(f"    - {f['type']} {f['affected_ids']} {f['claimed_amount_minor']} {f['currency']} "
                             f"evidence={f['evidence_ids']}")
    ev = env.last_evaluation()
    lines += ["", f"Result (open training mode): terminated={term} truncated={trunc} reward={reward}; "
                  f"TP={ev['tp']} FP={ev['fp']} FN={ev['fn']} P={ev['precision']:.2f} R={ev['recall']:.2f} F1={ev['f1']:.2f}"]
    text = "\n".join(lines) + "\n"
    stream.write(text)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return ev


def cmd_demo(a) -> int:
    demo(a.split, a.index, Path(a.out) if a.out else None)
    return 0


# -------------------------------------------------------------------- check
def cmd_check(a) -> int:
    t0 = time.time()
    results: List[tuple] = []

    def stage(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)

    print(f"auditenv check: env {ENV_VERSION}, generator {GENERATOR_VERSION}, Python {sys.version.split()[0]}", flush=True)
    # 1. unit tests
    if not a.skip_tests:
        buf = io.StringIO()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))

        def test_ids(node):
            if isinstance(node, unittest.TestSuite):
                return [i for child in node for i in test_ids(child)]
            return [node.id()]

        jury_ids = {i for i in test_ids(suite) if i.startswith("tests.test_judge_adapter.")}
        res = unittest.TextTestRunner(stream=buf, verbosity=1).run(suite)
        ok = res.wasSuccessful()
        stage("unit tests", ok, f"{res.testsRun} tests, {len(res.failures)} failures, {len(res.errors)} errors, "
                                f"{len(res.expectedFailures)} expected failures")
        if not ok:
            print(buf.getvalue()[-4000:])
        # 1b. the jury check definitions (blocks 1-4): reported from the same run, not executed twice
        failed_ids = {getattr(t, "test_case", t).id() for t, _ in res.failures + res.errors}
        stage("jury check definitions via judge_adapter.py: C01-C20 contract, D01-D20 matrix, 20 matcher/reward "
              "tests + M01-M12, 20 paired checks through episode_dir", bool(jury_ids) and not failed_ids & jury_ids,
              f"{len(jury_ids)} tests, {len(failed_ids & jury_ids)} failed")
    # 2. reproducibility of committed episodes
    with tempfile.TemporaryDirectory() as td:
        same = True
        detail = []
        for name in ("train", "dev"):
            idx = export_split(name, Path(td))
            committed = EPISODES / name / "index.json"
            if not committed.exists():
                same = False
                detail.append(f"{name}: no committed index")
                continue
            old = json.loads(committed.read_text(encoding="utf-8"))
            a1 = [(x["episode_id"], x["full_sha256"]) for x in old["episodes"]]
            a2 = [(x["episode_id"], x["full_sha256"]) for x in idx["episodes"]]
            if a1 != a2:
                same = False
                detail.append(f"{name}: regenerated episodes differ from committed ones")
            for x in old["episodes"]:
                from .episode import Episode
                disk = Episode.load(EPISODES / name / x["dir"])
                if disk.full_sha256() != x["full_sha256"]:
                    same = False
                    detail.append(f"{name}/{x['dir']}: files on disk differ from index")
        stage("episodes regenerate bit-identically from seeds+configs (10 train + 10 dev)", same, "; ".join(detail))
        # 3. data validation (committed + holdout-like)
        s1 = validate_root(EPISODES)
        export_split("holdout_sim", Path(td))
        s2 = validate_root(Path(td) / "holdout_sim")
        bad = [r for r in s1["results"] + s2["results"] if not r["ok"]]
        stage("data & truth validation (clean books balance and follow policy, declared discrepancies present, "
              "truth complete, evidence sufficient)", not bad,
              f"{s1['ok'] + s2['ok']}/{s1['episodes'] + s2['episodes']} episodes ok")
        for r in bad[:5]:
            print("   ", r.get("episode_id"), r.get("errors", [])[:3])
    # 4. paired checks
    pc = paired_checks("holdout_sim")
    stage("paired checks: replay exact, permutation + ID remap keeps oracle answer and reward",
          all(p["ok"] for p in pc), f"{sum(p['ok'] for p in pc)}/{len(pc)}")
    # 5. isolated baselines
    out = run_baselines("holdout_sim", isolated=True)
    rows = {r["policy"]: r["summary"] for r in out["rows"]}
    ok = (rows["oracle"]["return_min"] == 1.0 and rows["flag_all"]["return_mean"] < 0
          and rows["scripted_auditor"]["f1_mean"] > rows["reconciliation"]["f1_mean"]
          and all(v["agent_visible_return_mean"] == 0.0 for v in rows.values())
          and all(v["seconds_max"] < 60 for v in rows.values()))
    stage("closed-mode baselines in a separate evaluator process (oracle = 1.0, flag-all < 0, rewards withheld, <60 s/episode)", ok)
    print(fmt_table(out["rows"]))
    # 6. leak scan of committed trajectories
    from .episode import Episode
    from .leakscan import forbidden_tokens, scan_text
    hits = []
    if TRAJ.exists():
        eps = {}
        for idx_file in (EPISODES / "train" / "index.json",):
            idx = json.loads(idx_file.read_text(encoding="utf-8"))
            for x in idx["episodes"]:
                eps[x["episode_id"]] = Episode.load(EPISODES / "train" / x["dir"])
        toks = {k: forbidden_tokens(v) for k, v in eps.items()}
        n = 0
        with open(TRAJ, "r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                n += 1
                for k in ("episode_id", "step", "action", "observation", "reward", "terminated", "truncated"):
                    if k not in rec:
                        hits.append(f"record without {k}")
                h = scan_text(json.dumps(rec["observation"], ensure_ascii=False), toks.get(rec["episode_id"], []))
                hits += h
        stage("trajectories.jsonl: required fields present, no closed truth in observations", not hits and n > 0,
              f"{n} records scanned" + (f"; hits {sorted(set(hits))[:5]}" if hits else ""))
    else:
        stage("trajectories.jsonl present", False, "run `python -m auditenv trajectories`")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} stages passed in {time.time() - t0:.1f} s")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m auditenv", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--skip-tests", action="store_true")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("generate")
    p.add_argument("--split", default="all", choices=["all", "train", "dev", "holdout_sim"])
    p.add_argument("--out", default=str(EPISODES))
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("validate")
    p.add_argument("--root", default=str(EPISODES))
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("baselines")
    p.add_argument("--split", default="dev", help="train | dev | holdout_sim | path/to/split.json")
    p.add_argument("--in-process", action="store_true")
    p.set_defaults(fn=cmd_baselines)
    p = sub.add_parser("evaluate", help="closed-mode evaluation of any policy on any split file")
    p.add_argument("--split", required=True, help="split name or path/to/split.json (secret via AUDITENV_SECRET)")
    p.add_argument("--policy", action="append", required=True,
                   help="built-in name (random, reconciliation, flag_all, oracle, scripted_auditor) or module:Class")
    p.add_argument("--out", default=None, help="write the full JSON result here")
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("trajectories")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default=str(TRAJ))
    p.set_defaults(fn=cmd_trajectories)
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(REPORTS))
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("demo")
    p.add_argument("--split", default="dev")
    p.add_argument("--index", type=int, default=5)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("serve")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=lambda a: __import__("auditenv.server", fromlist=["main"]).main(a.rest))
    a = ap.parse_args(argv)
    return a.fn(a)
