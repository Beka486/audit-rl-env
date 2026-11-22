"""Isolation and leakage: out-of-process evaluator, closed mode, no truth in observations or trajectories."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from auditenv.env import AuditEnv
from auditenv.episode import Episode
from auditenv.leakscan import forbidden_tokens, scan_observations, scan_text
from auditenv.remote import RemoteAuditEnv, RemoteError
from auditenv.schema_lite import check
from tests.helpers import ERRORS_ALL, SOME_ERRORS, dumps, run_policy, submit_action

ROOT = pathlib.Path(__file__).resolve().parent.parent


class OutOfProcess(unittest.TestCase):
    def test_remote_matches_in_process_and_keeps_grades_server_side(self):
        from auditenv.policies import ScriptedAuditor
        with tempfile.TemporaryDirectory() as td:
            log = os.path.join(td, "evaluator.jsonl")
            remote = RemoteAuditEnv(log, mode="eval")
            local = AuditEnv(mode="eval")
            try:
                for seed in (1, 2):
                    o1, _ = remote.reset(seed=seed, config=SOME_ERRORS)
                    o2, _ = local.reset(seed=seed, config=SOME_ERRORS)
                    self.assertEqual(dumps(o1), dumps(o2))
                    pol = ScriptedAuditor()
                    pol.reset(0)
                    done = False
                    while not done:
                        a = pol.act(o1)
                        o1, r1, t1, tr1, i1 = remote.step(a)
                        o2, r2, t2, tr2, i2 = local.step(a)
                        self.assertEqual(dumps([o1, r1, t1, tr1, i1]), dumps([o2, r2, t2, tr2, i2]))
                        self.assertEqual(r1, 0.0)
                        self.assertEqual(i1, {})
                        done = t1 or tr1
                    with self.assertRaises(RemoteError):
                        remote.oracle_submit()  # oracle is disabled unless the evaluator enables it
            finally:
                remote.close()
            grades = [json.loads(x) for x in pathlib.Path(log).read_text().splitlines()]
            self.assertEqual(len(grades), 2)
            self.assertTrue(all(g["f1"] == 1.0 for g in grades))
            self.assertEqual([g["reward"] for g in grades], [g["reward"] for g in local.evaluator_log])

    def test_server_errors_are_opaque(self):
        with tempfile.TemporaryDirectory() as td:
            env = RemoteAuditEnv(os.path.join(td, "log.jsonl"))
            try:
                with self.assertRaises(RemoteError) as cm:
                    env.reset(seed=1, config={"errors": {"fraud": 1}})
                self.assertEqual(str(cm.exception), "ConfigError")
                with self.assertRaises(RemoteError) as cm:
                    env.step({"action": "calculator", "args": {"expression": "1"}})
                self.assertEqual(str(cm.exception), "ResetNeeded")
            finally:
                env.close()

    def test_oracle_runs_evaluator_side(self):
        with tempfile.TemporaryDirectory() as td:
            log = os.path.join(td, "evaluator.jsonl")
            env = RemoteAuditEnv(log, mode="eval", oracle=True)
            try:
                env.reset(seed=4, config=ERRORS_ALL)
                action, (obs, r, t, tr, info) = env.oracle_submit()
                self.assertTrue(t)
                self.assertEqual(r, 0.0)
            finally:
                env.close()
            g = json.loads(pathlib.Path(log).read_text().splitlines()[0])
            self.assertEqual(g["reward"], 1.0)


class NoLeaks(unittest.TestCase):
    def test_observations_contain_no_closed_data(self):
        for mode in ("train", "eval"):
            for pol in ("scripted_auditor", "random", "flag_all", "reconciliation"):
                for seed in range(3):
                    env = AuditEnv(mode=mode, secret="s3cr3t-salt" if mode == "eval" else None)
                    trace, ev = run_policy(env, pol, seed, ERRORS_ALL, policy_seed=seed)
                    obs = [trace[0]] + [t["obs"] for t in trace[1:]] + [t["info"] for t in trace[1:]]
                    hits = scan_observations(obs, env.episode, "s3cr3t-salt" if mode == "eval" else "")
                    self.assertEqual(hits, [], (mode, pol, seed))
                    for t in trace[1:]:
                        self.assertEqual(check(t["obs"], "observation"), [])
                        if mode == "eval":
                            self.assertEqual(t["reward"], 0.0)
                            self.assertEqual(t["info"], {})

    def test_removed_rows_are_unreachable_and_indistinguishable(self):
        env = AuditEnv()
        env.reset(seed=1, config={"errors": {"duplicate_invoice": 0, "payment_allocation": 3, "fx_conversion": 0, "period_cutoff": 3},
                                  "interactions": {"fx_partial_boundary": 0, "duplicate_misapplied": 0}})
        removed = [r["row_id"] for r in env.episode.provenance["removed_ledger_rows"]]
        self.assertTrue(removed)
        msgs = set()
        for rid in removed + ["L-ZZZZZZZZ", "GL000000000"]:
            obs, *_ = env.step({"action": "get_record", "args": {"table": "ledger", "id": rid}})
            msgs.add(dumps(obs["error"]))
        self.assertEqual(len(msgs), 1)

    def test_manifest_and_policy_have_no_private_fields(self):
        ep = AuditEnv()
        obs, info = ep.reset(seed=3, config=ERRORS_ALL)
        text = dumps([obs, info])
        self.assertEqual(scan_text(text, forbidden_tokens(ep.episode)), [])
        for word in ("seed", "config_hash", "errors", "interactions", "distractor"):
            self.assertNotIn(f'"{word}"', text)

    def test_committed_trajectories(self):
        path = ROOT / "trajectories" / "trajectories.jsonl"
        if not path.exists():
            self.skipTest("trajectories not generated yet")
        idx = json.loads((ROOT / "episodes" / "train" / "index.json").read_text())
        eps = {x["episode_id"]: Episode.load(ROOT / "episodes" / "train" / x["dir"]) for x in idx["episodes"]}
        toks = {k: forbidden_tokens(v) for k, v in eps.items()}
        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                n += 1
                self.assertEqual(check(rec, "trajectory_record"), [])
                self.assertIn(rec["episode_id"], eps)
                self.assertEqual(scan_text(dumps(rec["observation"]), toks[rec["episode_id"]]), [])
                self.assertNotIn("oracle", rec.get("policy", ""))
        self.assertGreater(n, 100)

    def test_policies_never_receive_env(self):
        import inspect
        from auditenv.policies import ALL_POLICIES
        for name, cls in ALL_POLICIES.items():
            sig = inspect.signature(cls.act)
            self.assertEqual(list(sig.parameters)[1:], [list(sig.parameters)[1]])  # act(self, observation)


class Determinism(unittest.TestCase):
    def test_same_seed_config_actions_reproduce_everything(self):
        def trace(seed):
            env = AuditEnv()
            t, ev = run_policy(env, "random", seed, ERRORS_ALL, policy_seed=7)
            return dumps(t), dumps(ev), env.episode.full_sha256()
        self.assertEqual(trace(3), trace(3))
        self.assertNotEqual(trace(3)[2], trace(4)[2])

    def test_hash_randomisation_does_not_change_episodes(self):
        code = ("import sys; sys.path.insert(0, %r); from auditenv.generator import generate_episode; "
                "print(generate_episode(12, {}).full_sha256(), generate_episode(13, {'transform': {'seed': 2}}).full_sha256())") % str(ROOT)
        outs = set()
        for hs in ("0", "1", "12345"):
            envv = dict(os.environ, PYTHONHASHSEED=hs)
            outs.add(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=envv, check=True).stdout)
        self.assertEqual(len(outs), 1)

    def test_committed_episodes_match_generator(self):
        from auditenv.generator import generate_episode
        for split in ("train", "dev"):
            p = ROOT / "episodes" / split / "index.json"
            if not p.exists():
                self.skipTest("episodes not exported yet")
            idx = json.loads(p.read_text())
            doc = json.loads((ROOT / "configs" / "splits" / f"{split}.json").read_text())
            for x, e in zip(idx["episodes"], doc["episodes"]):
                ep = generate_episode(e["seed"], e["config"], secret=doc["secret"])
                self.assertEqual(ep.full_sha256(), x["full_sha256"])
                self.assertEqual(Episode.load(ROOT / "episodes" / split / x["dir"]).full_sha256(), x["full_sha256"])


if __name__ == "__main__":
    unittest.main()
