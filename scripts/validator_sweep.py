"""Validator sweep: generate many episodes across configs and check each with the independent validator.

    python scripts/validator_sweep.py --per-config 200 --salt any-salt [--base 0] [--out sweep.json]

Configs: the default config, the 20 jury-matrix configs (configs/splits/judge_matrix.json) and
7 stress configs (12 errors, 4 foreign currencies, JPY/KWD pool, smallest and largest books,
reverse-pair FX only, 4 boundary distractors per cutoff error). Each episode is written to a
temporary directory and checked by ``auditenv/validate_data.py``, which does not import the
generator. Exit code 0 only if every episode generates and passes every validator check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STRESS = [
    ("stress-12-errors", {"errors": {"duplicate_invoice": 3, "payment_allocation": 3, "fx_conversion": 3, "period_cutoff": 3},
                          "interactions": {"fx_partial_boundary": 0, "duplicate_misapplied": 0}}),
    ("stress-4-foreign", {"currencies": {"foreign_count": [4, 4]}}),
    ("stress-jpy-kwd", {"currencies": {"pool": ["KZT", "JPY", "KWD", "USD"], "foreign_count": [3, 3]}}),
    ("stress-small", {"size": {"customers": [6, 6], "invoices": [40, 40]}}),
    ("stress-large", {"size": {"customers": [12, 12], "invoices": [75, 75]}}),
    ("stress-reverse-only", {"variants": {"fx_conversion": ["reverse_pair_rate"]}, "errors": {"fx_conversion": [2, 2]}}),
    ("stress-boundary4", {"distractors": {"boundary_per_cutoff": 4}, "errors": {"period_cutoff": [2, 2]}}),
]


def configs():
    out = [("default", {})]
    split = json.loads((ROOT / "configs" / "splits" / "judge_matrix.json").read_text(encoding="utf-8"))
    out += [(e["config"]["name"], e["config"]) for e in split["episodes"]]
    return out + STRESS


def one(job):
    name, cfg, seed, salt = job
    from auditenv.generator import generate_episode
    from auditenv.validate_data import validate_episode_dir
    try:
        ep = generate_episode(seed, cfg, secret=salt)
    except Exception as exc:  # a generation failure is a sweep failure, never hidden
        return {"name": name, "seed": seed, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
    with tempfile.TemporaryDirectory() as d:
        ep.write(d)
        r = validate_episode_dir(d)
    issues = ep.truth["issues"]
    return {"name": name, "seed": seed, "ok": r["ok"], "error": "; ".join(r["errors"][:3]),
            "attempt": ep.provenance.get("attempt", 0), "issues": len(issues),
            "evidence_sets": sum(len(i["evidence_sets"]) for i in issues),
            "variants": Counter(f'{i["type"]}:{i["variant"]}' for i in issues)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-config", type=int, default=50)
    ap.add_argument("--salt", default="sweep-salt")
    ap.add_argument("--base", type=int, default=0, help="first seed; config k uses seeds base + 1000*k + i")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfgs = configs()
    jobs = [(name, cfg, a.base + 1000 * k + i, a.salt) for k, (name, cfg) in enumerate(cfgs) for i in range(a.per_config)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, jobs, chunksize=8))
    good = [r for r in res if "issues" in r]
    variants = Counter()
    for r in good:
        variants.update(r["variants"])
    summary = {"configs": len(cfgs), "episodes": len(res), "ok": sum(1 for r in res if r["ok"]),
               "retried_builds": sum(1 for r in good if r["attempt"]), "issues": sum(r["issues"] for r in good),
               "evidence_sets_checked": sum(r["evidence_sets"] for r in good),
               "variants": dict(sorted(variants.items())), "seconds": round(time.time() - t0, 1),
               "failures": [{k: r[k] for k in ("name", "seed", "error")} for r in res if not r["ok"]][:20]}
    text = json.dumps(summary, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["ok"] == summary["episodes"] else 1


if __name__ == "__main__":
    sys.exit(main())
