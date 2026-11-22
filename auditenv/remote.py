"""Client for the out-of-process environment (see server.py). Same reset/step contract as AuditEnv."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


POLICY_GRACE_S = 1.0  # after the episode limit, so that the environment records time_limit itself


class RemoteError(RuntimeError):
    pass


def policy_env() -> Dict[str, str]:
    """Environment of a policy process: no salt, no log key and no evaluator log location."""
    env = dict(os.environ)
    for key in ("AUDITENV_SECRET", "AUDITENV_LOG_KEY", "AUDITENV_EVALUATOR_LOG"):
        env.pop(key, None)
    paths = [str(ROOT), os.getcwd()] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


class RemoteAuditEnv:
    def __init__(self, evaluator_log: str, mode: str = "eval", secret: Optional[str] = None,
                 oracle: bool = False, time_limit_s: float = 60.0, log_key: Optional[str] = None) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.pop("AUDITENV_SECRET", None)
        if secret:
            env["AUDITENV_SECRET"] = secret
        env.pop("AUDITENV_LOG_KEY", None)
        if log_key:
            env["AUDITENV_LOG_KEY"] = log_key
        cmd = [sys.executable, "-m", "auditenv.server", "--mode", mode, "--evaluator-log", evaluator_log,
               "--time-limit", str(time_limit_s)]
        if oracle:
            cmd.append("--oracle")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", env=env, cwd=str(ROOT))

    def _call(self, req: Dict[str, Any]) -> Dict[str, Any]:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(req) + "\n")  # ASCII on the wire: no locale dependence
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
    episode with an invalid submission (reward -1). A policy that does not answer before the
    episode time limit is killed, the episode is truncated with empty findings, and the next
    episode gets a fresh process."""

    CRASH_ACTION = {"action": "submit", "args": {"findings": None}}

    def __init__(self, policy: str, time_limit_s: float = 60.0) -> None:
        self.policy = policy
        self.time_limit_s = float(time_limit_s)
        self.crashes = 0
        self._deadline = time.monotonic() + self.time_limit_s + POLICY_GRACE_S
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen([sys.executable, "-m", "auditenv.agent_runner", "--policy", self.policy],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", env=policy_env(), cwd=os.getcwd())
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._reader = threading.Thread(target=self._read, args=(self.proc.stdout, self._lines), daemon=True)
        self._reader.start()

    @staticmethod
    def _read(stream, lines) -> None:
        try:
            for line in stream:
                lines.put(line)
        except (OSError, ValueError):
            pass
        lines.put("")

    def _stop(self) -> None:
        """Kill the process, wait for the reader and release the pipes."""
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass
        self._reader.join(timeout=5)
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream:
                    stream.close()
            except (OSError, ValueError):
                pass

    def _call(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        resp = None
        try:
            assert self.proc.stdin
            self.proc.stdin.write(json.dumps(req) + "\n")  # ASCII on the wire: no locale dependence
            self.proc.stdin.flush()
            line = self._lines.get(timeout=max(0.0, self._deadline - time.monotonic()))
            resp = json.loads(line) if line else None
        except queue.Empty:
            self.proc.kill()
        except (OSError, ValueError):
            pass
        if not resp or not resp.get("ok"):
            self.crashes += 1
            return None
        return resp

    def reset(self, seed: int) -> None:
        if self.proc.poll() is not None:
            self._stop()
            self._start()
        self._deadline = time.monotonic() + self.time_limit_s + POLICY_GRACE_S
        self._call({"op": "reset", "seed": seed})

    def act(self, observation: Dict[str, Any]) -> Any:
        resp = self._call({"op": "act", "observation": observation})
        return resp["action"] if resp else dict(self.CRASH_ACTION)

    def close(self) -> None:
        if self.proc.poll() is None:
            self._deadline = time.monotonic() + 5.0
            self._call({"op": "close"})
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        self._stop()
