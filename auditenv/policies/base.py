from __future__ import annotations

from typing import Any, Dict, List, Optional


class Policy:
    """Observation -> action. ``reset`` receives only the policy's own seed (never the episode seed)."""

    name = "base"
    evaluator_only = False

    def reset(self, seed: int = 0) -> None:  # pragma: no cover - trivial
        pass

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class TableReader(Policy):
    """Pages through ``tables`` with list_records, then hands over to ``next_action``."""

    tables = ("invoices", "payments", "ledger")

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.manifest: Optional[Dict[str, Any]] = None
        self.policy: Optional[Dict[str, Any]] = None
        self.data: Dict[str, List[Dict[str, Any]]] = {t: [] for t in ("ledger", "invoices", "payments", "fx")}
        self.read_queue = list(self.tables)
        self.cursor: Optional[str] = None
        self.steps_remaining = 100

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self.steps_remaining = observation.get("steps_remaining", self.steps_remaining)
        kind = observation.get("kind")
        if kind == "reset":
            self.manifest = observation["manifest"]
            self.policy = observation["policy"]
        elif kind == "result":
            if observation["action"] == "list_records":
                res = observation["result"]
                self.data[res["table"]].extend(res["records"])
                self.cursor = res["next_cursor"]
                if self.cursor is None and self.read_queue and self.read_queue[0] == res["table"]:
                    self.read_queue.pop(0)
            else:
                self.on_result(observation["action"], observation["result"])
        elif kind == "error":
            if observation.get("action") == "list_records" and self.read_queue:
                self.read_queue.pop(0)
                self.cursor = None
            self.on_error(observation)
        if self.read_queue and self.steps_remaining > 1:
            return {"action": "list_records", "args": {"table": self.read_queue[0], "cursor": self.cursor, "limit": 50}}
        return self.next_action()

    def on_result(self, action: str, result: Dict[str, Any]) -> None:
        pass

    def on_error(self, observation: Dict[str, Any]) -> None:
        pass

    def next_action(self) -> Dict[str, Any]:
        raise NotImplementedError

    def submit(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"action": "submit", "args": {"findings": {"episode_id": self.manifest["episode_id"],
                                                          "findings": findings[:30]}}}
