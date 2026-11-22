"""Hostile policies for test_hardening (loaded as ``tests.hardening_policies:Class``)."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _blocked(fn):
    try:
        fn()
    except (PermissionError, ImportError):
        return True
    except Exception as exc:  # any other failure is not proof of the guard
        return f"other:{type(exc).__name__}"
    return False


def _socket():
    import socket
    socket.socket()


def _import_grader():
    import auditenv.grader  # noqa: F401


class Attacker:
    def reset(self, seed):
        pass

    def act(self, obs):
        truth = next(iter(sorted((ROOT / "tests").parent.glob("episodes/dev/01_*/private/truth.json"))), None)
        direct = ROOT / "episodes" / "dev"
        first = None
        for name in ("01_EP-31BA5EA6DAAC",):
            first = direct / name / "private" / "truth.json"
        seen = {
            "read_truth": _blocked(lambda: first.read_text(encoding="utf-8")),
            "list_episodes": _blocked(lambda: os.listdir(direct)),
            "import_grader": _blocked(_import_grader),
            "spawn": _blocked(lambda: subprocess.run(["python", "--version"], capture_output=True)),
            "socket": _blocked(_socket),
            "write_file": _blocked(lambda: Path("hardening-forged.jsonl").write_text("x", encoding="utf-8")),
            "glob_truth_found": truth is not None,
        }
        return {"action": "submit", "args": {"findings": {"episode_id": obs["manifest"]["episode_id"], "findings": [
            {"type": "period_cutoff", "affected_ids": ["PROBE"], "claimed_amount_minor": 0,
             "currency": obs["manifest"]["base_currency"], "evidence_ids": [], "explanation": json.dumps(seen)}]}}}


class HangOnce:
    """Hangs on the first episode only; a restarted process has a fresh marker."""

    def reset(self, seed):
        self.seed = seed

    def act(self, obs):
        if self.seed == 0:
            time.sleep(60)
        return {"action": "submit", "args": {"findings": {"episode_id": obs["manifest"]["episode_id"], "findings": []}}}


class StringSubmit:
    def reset(self, seed):
        pass

    def act(self, obs):
        return json.dumps({"action": "submit", "args": {"findings": {"episode_id": obs["manifest"]["episode_id"],
                                                                     "findings": []}}})


class ListArgsSubmit:
    def reset(self, seed):
        pass

    def act(self, obs):
        return {"action": "submit", "args": [1]}
