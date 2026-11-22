"""AuditEnv: Gymnasium-style, JSON-compatible environment.

    env = AuditEnv(mode="train")          # or "eval" (closed run)
    obs, info = env.reset(seed=7, config={})
    obs, reward, terminated, truncated, info = env.step({"action": "list_records",
                                                         "args": {"table": "ledger"}})

* Exploration reward is always 0. Only ``submit`` (terminated=True) or the step/time limit
  (truncated=True, empty final answer) ends the episode.
* mode="train": the terminal reward and aggregate metrics are returned to the policy.
* mode="eval":  the policy always receives reward 0.0 and empty info; the grade is kept by
  the evaluator (``evaluator_log`` here, or a file written by the out-of-process server).
  The submit observation only says whether the findings were structurally valid
  ({"status": "received"} or {"status": "invalid_submission", "error": <schema message>}),
  never whether they were correct.
* The environment object itself is harness-side. Policies only ever see observations.
"""
from __future__ import annotations

import copy
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .calculator import CalcError, evaluate
from .episode import Episode
from .generator import generate_episode
from .grader import Grader
from .policy_text import TOOLS, task_text
from .tools import RecordStore, ToolError
from .version import (DEFAULT_TIME_LIMIT_S, ENV_VERSION, MAX_ACTION_BYTES, MAX_ACTION_DEPTH, MAX_STEPS, PAGE_LIMIT,
                      SCHEMA_VERSION)

ACTIONS = {
    "list_records": ({"table"}, {"cursor", "limit"}),
    "get_record": ({"table", "id"}, set()),
    "get_document": ({"doc_id"}, set()),
    "calculator": ({"expression"}, set()),
    "submit": ({"findings"}, set()),
}
# Nesting limit of an action (version.MAX_ACTION_DEPTH), checked on the text before parsing, so
# that the result does not depend on the Python version's recursion limits (3.11 raises
# RecursionError near depth 1000, 3.12+ parses deeper). A valid action is at most 6 levels deep.
MAX_JSON_DEPTH = MAX_ACTION_DEPTH
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.S)
_JSON_BRACKET = re.compile(r"[\[\]{}]")


def _too_deep(text: str, limit: int = MAX_JSON_DEPTH) -> bool:
    depth = 0
    for m in _JSON_BRACKET.finditer(_JSON_STRING.sub('""', text)):
        if m.group() in "[{":
            depth += 1
            if depth > limit:
                return True
        else:
            depth -= 1
    return False


class ResetNeeded(RuntimeError):
    pass


def _json_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))


class AuditEnv:
    metadata = {"render_modes": [], "env_version": ENV_VERSION}

    def __init__(self, mode: str = "train", secret: Optional[str] = None,
                 time_limit_s: float = DEFAULT_TIME_LIMIT_S, clock: Optional[Callable[[], float]] = None) -> None:
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode
        self._secret = secret or ""
        self.time_limit_s = float(time_limit_s)
        self._clock = clock or time.monotonic
        self.evaluator_log: List[Dict[str, Any]] = []
        self._episode: Optional[Episode] = None
        self._done = True

    # ------------------------------------------------------------------ reset
    def set_mode(self, mode: str) -> None:
        """Harness-side switch between open training ("train") and closed ("eval") runs."""
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode

    def reset(self, seed: Optional[int] = None, config: Optional[Dict[str, Any]] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        config = copy.deepcopy(config) if config else {}
        if options and "config" in options and not config:
            config = copy.deepcopy(options["config"])
        if "episode_dir" in config:
            ep = Episode.load(config["episode_dir"])
            if config.get("transform"):
                from .transforms import apply_transform
                ep, _ = apply_transform(ep, config["transform"])
        else:
            ep = generate_episode(0 if seed is None else seed, config, secret=self._secret)
        self._load(ep)
        self._run_record = {"env_version": ENV_VERSION, "seed": seed, "config": copy.deepcopy(config)}
        obs = {
            "kind": "reset",
            "task": task_text(ep.manifest),
            "manifest": ep.manifest,
            "policy": ep.policy,
            "tools": TOOLS,
            "steps_used": 0,
            "steps_remaining": MAX_STEPS,
            "max_steps": MAX_STEPS,
        }
        info = {"env_version": ENV_VERSION, "schema_version": SCHEMA_VERSION, "mode": self.mode}
        return _json_copy(obs), info

    def _load(self, ep: Episode) -> None:
        self._episode = ep
        public = ep.public()
        self._store = RecordStore(public)
        self._grader = Grader(public, ep.truth)
        self._steps = 0
        self._done = False
        self._last_flags = (False, False)
        self._t0 = self._clock()

    # ------------------------------------------------------------------- step
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._episode is None:
            raise ResetNeeded("call reset() before step()")
        if self._done:
            term, trunc = self._last_flags
            obs = {"kind": "error", "action": None, "error": {"code": "episode_over",
                   "message": "The episode has ended; call reset()."},
                   "steps_used": self._steps, "steps_remaining": MAX_STEPS - self._steps}
            return obs, 0.0, term, trunc, {}
        if self._clock() - self._t0 > self.time_limit_s:
            self._steps += 1
            return self._finish(None, "time_limit", {"kind": "truncated", "reason": "time_limit"})
        self._steps += 1
        name, args, err = self._parse(action)
        if err is not None:
            return self._observe({"kind": "error", "action": name, "error": err})
        if name == "submit":
            return self._finish(args["findings"], "submitted", {"kind": "submitted"})
        try:
            if name == "list_records":
                result = self._store.list_records(args["table"], args.get("cursor"), args.get("limit", PAGE_LIMIT))
            elif name == "get_record":
                result = self._store.get_record(args["table"], args["id"])
            elif name == "get_document":
                result = self._store.get_document(args["doc_id"])
            else:
                result = {"expression": args["expression"], "result": evaluate(args["expression"])}
            return self._observe({"kind": "result", "action": name, "result": result})
        except ToolError as e:
            return self._observe({"kind": "error", "action": name, "error": {"code": e.code, "message": e.message}})
        except CalcError as e:
            return self._observe({"kind": "error", "action": name, "error": {"code": e.code, "message": e.message}})

    def _parse(self, action: Any):
        # Every action is reduced to plain JSON values first, so that no Python object (e.g. a str
        # subclass with a custom __eq__) can reach validation or matching.
        try:
            text = action if isinstance(action, (str, bytes)) else json.dumps(action, ensure_ascii=False)
        except (ValueError, TypeError, RecursionError):
            return None, None, {"code": "invalid_action", "message": "Action is not JSON-compatible."}
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8")
            except UnicodeDecodeError:
                return None, None, {"code": "invalid_action", "message": "Action is not valid JSON."}
        if len(text) > MAX_ACTION_BYTES:
            return None, None, {"code": "invalid_action", "message": "Action is too large."}
        if _too_deep(text):
            return None, None, {"code": "invalid_action",
                                "message": f"Action JSON is nested deeper than {MAX_JSON_DEPTH} levels."}
        try:
            action = json.loads(text)
        except (ValueError, TypeError, RecursionError):
            return None, None, {"code": "invalid_action", "message": "Action is not valid JSON."}
        if not isinstance(action, dict) or set(action) - {"action", "args"} or "action" not in action:
            return None, None, {"code": "invalid_action",
                                "message": "Action must be an object {\"action\": name, \"args\": {...}}."}
        name = action["action"]
        if not isinstance(name, str) or name not in ACTIONS:
            return None, None, {"code": "invalid_action", "message": "Unknown action. Actions: " + ", ".join(ACTIONS) + "."}
        args = action.get("args", {})
        if not isinstance(args, dict):
            if name == "submit":
                return name, {"findings": args}, None  # graded as an invalid submission
            return name, None, {"code": "invalid_arguments", "message": "args must be an object."}
        required, optional = ACTIONS[name]
        if name == "submit":
            if set(args) != {"findings"}:
                return name, {"findings": None}, None  # malformed submit: invalid structure, -1
            return name, args, None
        if set(args) - required - optional or not required <= set(args):
            return name, None, {"code": "invalid_arguments",
                                "message": f"Arguments for {name}: required {sorted(required)}, optional {sorted(optional)}."}
        return name, args, None

    def _observe(self, obs: Dict[str, Any]):
        obs["steps_used"] = self._steps
        obs["steps_remaining"] = MAX_STEPS - self._steps
        if self._steps >= MAX_STEPS:
            obs["episode_end"] = {"reason": "step_limit"}
            return self._finish(None, "step_limit", obs)
        return _json_copy(obs), 0.0, False, False, {}

    def _finish(self, findings: Any, how: str, obs: Dict[str, Any]):
        self._done = True
        if how == "submitted":
            grade = self._grader.grade(findings, inspection_steps=self._steps - 1, status="submitted")
            terminated, truncated = True, False
        else:
            grade = self._grader.grade(None, inspection_steps=self._steps, status=how)
            terminated, truncated = False, True
        grade["steps_used"] = self._steps
        grade["mode"] = self.mode
        grade.update(getattr(self, "_run_record", {}))
        self.evaluator_log.append(grade)
        self._last_flags = (terminated, truncated)
        obs["steps_used"] = self._steps
        obs["steps_remaining"] = MAX_STEPS - self._steps
        if self.mode == "train":
            if how == "submitted":
                obs["result"] = {"status": "graded" if grade["status"] == "submitted" else "invalid_submission"}
                if grade["error"]:
                    obs["result"]["error"] = grade["error"][:300]
            info = {"evaluation": {k: grade[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1",
                                                        "inspection_steps", "reward")}}
            reward = float(grade["reward"])
        else:
            if how == "submitted":
                if grade["status"] == "submitted":
                    obs["result"] = {"status": "received"}
                else:  # structure only (public schema), never correctness
                    obs["result"] = {"status": "invalid_submission", "error": (grade["error"] or "")[:300]}
            info = {}
            reward = 0.0
        return _json_copy(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------ harness API
    def last_evaluation(self) -> Optional[Dict[str, Any]]:
        return copy.deepcopy(self.evaluator_log[-1]) if self.evaluator_log else None

    @property
    def episode(self) -> Optional[Episode]:
        """Harness/evaluator-side access (oracle, reports). Never hand this to a policy."""
        return self._episode
