"""Shared helpers for the test-suite."""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

from auditenv.env import AuditEnv
from auditenv.policies import ALL_POLICIES, oracle_findings

ZERO_ERRORS = {"duplicate_invoice": 0, "payment_allocation": 0, "fx_conversion": 0, "period_cutoff": 0}
ZERO_INTER = {"fx_partial_boundary": 0, "duplicate_misapplied": 0}
CLEAN = {"errors": dict(ZERO_ERRORS), "interactions": dict(ZERO_INTER)}
ERRORS_ALL = {"errors": {"duplicate_invoice": 2, "payment_allocation": 2, "fx_conversion": 2, "period_cutoff": 2},
              "interactions": {"fx_partial_boundary": 1, "duplicate_misapplied": 1}}
SOME_ERRORS = {"errors": {"duplicate_invoice": 1, "payment_allocation": 1, "fx_conversion": 1, "period_cutoff": 1},
               "interactions": dict(ZERO_INTER)}


def only(error_type: str, n: int = 1) -> Dict[str, Any]:
    e = dict(ZERO_ERRORS)
    e[error_type] = n
    return {"errors": e, "interactions": dict(ZERO_INTER)}


def submit_action(episode_id: str, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"action": "submit", "args": {"findings": {"episode_id": episode_id, "findings": findings}}}


def run_policy(env: AuditEnv, name: str, seed: int, config: Dict[str, Any], policy_seed: int = 0):
    obs, info = env.reset(seed=seed, config=config)
    cls = ALL_POLICIES[name]
    pol = cls(lambda: env.episode.truth) if name == "oracle" else cls()
    pol.reset(policy_seed)
    trace = [obs]
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(pol.act(obs))
        trace.append({"obs": obs, "reward": r, "terminated": term, "truncated": trunc, "info": info})
        done = term or trunc
    return trace, env.last_evaluation()


def submit_after(env: AuditEnv, findings: List[Dict[str, Any]], explore_steps: int = 0):
    """Reset must already have happened. Spend ``explore_steps`` calculator steps, then submit."""
    for _ in range(explore_steps):
        env.step({"action": "calculator", "args": {"expression": "1+1"}})
    obs, r, term, trunc, info = env.step(submit_action(env.episode.episode_id, findings))
    return r, env.last_evaluation()


def oracle(env: AuditEnv) -> List[Dict[str, Any]]:
    return copy.deepcopy(oracle_findings(env.episode.truth))


def dumps(x: Any) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False)
