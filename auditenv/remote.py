"""Client for the out-of-process environment (see server.py). Same reset/step contract as AuditEnv."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


class RemoteError(RuntimeError):
    pass


class RemoteAuditEnv:
    def __init__(self, evaluator_log: str, mode: str = "eval", secret: Optional[str] = None,
                 oracle: bool = False, time_limit_s: float = 60.0) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.pop("AUDITENV_SECRET", None)
        if secret:
            env["AUDITENV_SECRET"] = secret
        cmd = [sys.executable, "-m", "auditenv.server", "--mode", mode, "--evaluator-log", evaluator_log,
               "--time-limit", str(time_limit_s)]
        if oracle:
            cmd.append("--oracle")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", env=env, cwd=str(ROOT))

    def _call(self, req: Dict[str, Any]) -> Dict[str, Any]:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RemoteError("environment server exited")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RemoteError(resp.get("error", "error"))
        return resp

    def reset(self, seed: Optional[int] = None, config: Optional[Dict[str, Any]] = None,
              mode: Optional[str] = None) -> Tuple[Dict, Dict]:
        req: Dict[str, Any] = {"op": "reset", "seed": seed, "config": config or {}}
        if mode is not None:
            req["mode"] = mode
        r = self._call(req)
        return r["observation"], r["info"]

    def step(self, action: Any):
        # The action crosses the process boundary as JSON text, serialised here inside a guard:
        # a non-JSON Python object or a structure too deep for this interpreter becomes a
        # bounded invalid_action error in the environment (step consumed), never an exception.
        if isinstance(action, bytes):
            action = action.decode("utf-8", errors="replace")
        if not isinstance(action, str):
            try:
                action = json.dumps(action, ensure_ascii=False)
            except (ValueError, TypeError, RecursionError):
                action = "<not json>"
        r = self._call({"op": "step", "action": action})
        return r["observation"], r["reward"], r["terminated"], r["truncated"], r["info"]

    def oracle_submit(self):
        r = self._call({"op": "oracle_submit"})
        return r["action"], (r["observation"], r["reward"], r["terminated"], r["truncated"], r["info"])

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._call({"op": "close"})
            except RemoteError:
                pass
            self.proc.wait(timeout=10)
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RemotePolicy:
    """Agent process wrapper (see agent_runner.py): the policy runs in its own process, started
    without AUDITENV_SECRET, and only ever receives observations. A crash of the policy ends the
    episode with an invalid submission (reward -1)."""

    CRASH_ACTION = {"action": "submit", "args": {"findings": None}}

    def __init__(self, policy: str) -> None:
        env = dict(os.environ)
        env.pop("AUDITENV_SECRET", None)
        paths = [str(ROOT), os.getcwd()] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        self.proc = subprocess.Popen([sys.executable, "-m", "auditenv.agent_runner", "--policy", policy],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", env=env, cwd=os.getcwd())
        self.crashes = 0

    def _call(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            assert self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            resp = json.loads(line) if line else None
        except (OSError, ValueError):
            resp = None
        if not resp or not resp.get("ok"):
            self.crashes += 1
            return None
        return resp

    def reset(self, seed: int) -> None:
        self._call({"op": "reset", "seed": seed})

    def act(self, observation: Dict[str, Any]) -> Any:
        resp = self._call({"op": "act", "observation": observation})
        return resp["action"] if resp else dict(self.CRASH_ACTION)

    def close(self) -> None:
        if self.proc.poll() is None:
            self._call({"op": "close"})
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self.proc.kill()
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
