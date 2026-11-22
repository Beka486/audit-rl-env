"""External policies used by test_agent_process (loaded as ``tests.probe_policies:Class``)."""
import os
import sys


class Probe:
    """Reports what the agent process can see, then submits nothing useful."""

    def reset(self, seed):
        self.seed = seed

    def act(self, obs):
        seen = {"secret": os.environ.get("AUDITENV_SECRET"), "pid": os.getpid(),
                "generator_loaded": "auditenv.generator" in sys.modules,
                "obs_keys": sorted(obs)}
        return {"action": "submit", "args": {"findings": {
            "episode_id": obs["manifest"]["episode_id"],
            "findings": [{"type": "period_cutoff", "affected_ids": ["PROBE"], "claimed_amount_minor": 0,
                          "currency": obs["manifest"]["base_currency"], "evidence_ids": [],
                          "explanation": repr(seen)}]}}}


class Crasher:
    def reset(self, seed):
        pass

    def act(self, obs):
        raise RuntimeError("policy bug")
