"""Agent-side process: runs one policy and speaks JSON lines on stdin/stdout.

In a closed run the harness starts three roles in separate processes:

* this agent process - the policy; started with an environment that has no AUDITENV_SECRET and
  receives nothing but observations;
* the environment/evaluator server (server.py) - holds the episode, truth and grader;
* the harness - relays observations and actions between the two and reads grades from the
  evaluator log.

Requests:  {"op": "reset", "seed": int}           -> {"ok": true}
           {"op": "act", "observation": {...}}     -> {"ok": true, "action": {...}}
           {"op": "close"}
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict


def load_policy(name: str):
    """Built-in agent-side policy or ``module:Class``. Deliberately imports nothing evaluator-side."""
    if ":" in name:
        import importlib
        mod, _, cls_name = name.partition(":")
        return getattr(importlib.import_module(mod), cls_name)()
    from .policies import ALL_POLICIES
    return ALL_POLICIES[name]()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="auditenv.agent_runner")
    ap.add_argument("--policy", required=True, help="built-in policy name or module:Class")
    ap.add_argument("--check", action="store_true", help="only verify that the policy loads, then exit")
    args = ap.parse_args(argv)
    if args.policy == "oracle":
        raise SystemExit("the oracle is evaluator-only and cannot run in the agent process")
    if args.check:
        try:
            policy = load_policy(args.policy)
            if not (callable(getattr(policy, "reset", None)) and callable(getattr(policy, "act", None))):
                raise TypeError("a policy needs reset(seed) and act(observation)")
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}"[:300], file=sys.stderr)
            return 2
        return 0
    policy = load_policy(args.policy)
    out = sys.stdout
    sys.stdout = sys.stderr  # a policy that prints must not corrupt the protocol

    def send(obj: Dict[str, Any]) -> None:
        out.write(json.dumps(obj) + "\n")  # ASCII on the wire: no locale dependence
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op") if isinstance(req, dict) else None
            if op == "reset":
                policy.reset(int(req.get("seed", 0)))
                send({"ok": True})
            elif op == "act":
                action: Any = policy.act(req["observation"])
                send({"ok": True, "action": action})
            elif op == "close":
                send({"ok": True})
                break
            else:
                send({"ok": False, "error": "unknown op"})
        except Exception as exc:  # a crashing policy submits nothing; the harness records it
            send({"ok": False, "error": type(exc).__name__})
    return 0


if __name__ == "__main__":  # pragma: no cover
    from auditenv.policy_guard import install
    install()  # only in a dedicated process: the hook cannot be removed
    sys.exit(main())
