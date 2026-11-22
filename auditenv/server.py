"""Out-of-process environment + evaluator speaking JSON lines on stdin/stdout.

The agent side (harness + policy) never holds truth: it sends ``reset``/``step`` requests and
receives observations. In ``eval`` mode rewards and diagnostics are withheld; grades are
appended to the evaluator log file named on the command line (not reachable via the protocol).
The secret salt for closed seeds is read from the AUDITENV_SECRET environment variable.

Requests:  {"op": "reset", "seed": int, "config": {...}, "mode": "train"|"eval" (optional)}
           {"op": "step", "action": {...}}
           {"op": "oracle_submit"}            (only if started with --oracle; evaluator use)
           {"op": "close"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="auditenv.server")
    ap.add_argument("--mode", choices=["train", "eval"], default="eval")
    ap.add_argument("--evaluator-log", required=True)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args(argv)
    secret = os.environ.pop("AUDITENV_SECRET", "")
    log_key = os.environ.pop("AUDITENV_LOG_KEY", "")
    from .env import AuditEnv
    from .logauth import seal
    from .policies.oracle import oracle_findings
    env = AuditEnv(mode=args.mode, secret=secret, time_limit_s=args.time_limit)
    out = sys.stdout
    sys.stdout = sys.stderr  # nothing but protocol lines may reach the pipe
    log = open(args.evaluator_log, "a", encoding="utf-8")
    sequence = 0

    def send(obj: Dict[str, Any]) -> None:
        out.write(json.dumps(obj) + "\n")  # ASCII on the wire: no locale dependence
        out.flush()

    def stepped(action: Any) -> Dict[str, Any]:
        nonlocal sequence
        before = len(env.evaluator_log)
        obs, reward, term, trunc, info = env.step(action)
        if len(env.evaluator_log) > before:  # a new grade (not a step after the end of the episode)
            ev = env.last_evaluation()
            if log_key:
                sequence += 1
                ev = seal(ev, sequence, log_key)
            log.write(json.dumps(ev, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())
        return {"ok": True, "observation": obs, "reward": reward, "terminated": term, "truncated": trunc, "info": info}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op") if isinstance(req, dict) else None
            if op == "reset":
                if req.get("mode") is not None:
                    env.set_mode(req["mode"])
                obs, info = env.reset(seed=req.get("seed"), config=req.get("config"))
                send({"ok": True, "observation": obs, "info": info})
            elif op == "step":
                send(stepped(req.get("action")))
            elif op == "oracle_submit":
                if not args.oracle:
                    send({"ok": False, "error": "oracle disabled"})
                    continue
                ep = env.episode
                action = {"action": "submit", "args": {"findings": {"episode_id": ep.episode_id,
                                                                    "findings": oracle_findings(ep.truth)}}}
                resp = stepped(action)
                resp["action"] = action
                send(resp)
            elif op == "close":
                send({"ok": True})
                break
            else:
                send({"ok": False, "error": "unknown op"})
        except Exception as exc:  # never echo internals (no traceback, no paths)
            send({"ok": False, "error": type(exc).__name__})
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
