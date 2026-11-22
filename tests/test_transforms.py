"""Paired checks: permutation and ID remapping preserve the oracle answer and reward with new IDs."""
from __future__ import annotations

import unittest

from auditenv.checker import reconcile
from auditenv.env import AuditEnv
from auditenv.generator import generate_episode
from auditenv.grader import Grader
from auditenv.policies import oracle_findings
from auditenv.transforms import apply_transform, map_findings
from tests.helpers import ERRORS_ALL, run_policy


class Paired(unittest.TestCase):
    def test_permutation_and_id_remap(self):
        for seed in range(10):
            cfg = ERRORS_ALL if seed % 2 else {}
            ep = generate_episode(seed, cfg)
            tep, mp = apply_transform(ep, {"seed": seed + 100})
            # every id changed, the content did not
            self.assertTrue(set(r["row_id"] for r in ep.ledger).isdisjoint(r["row_id"] for r in tep.ledger))
            self.assertTrue(set(i["invoice_id"] for i in ep.invoices).isdisjoint(i["invoice_id"] for i in tep.invoices))
            self.assertNotEqual([r["row_id"] for r in tep.ledger], sorted(r["row_id"] for r in tep.ledger))
            self.assertNotEqual(tep.episode_id, ep.episode_id)
            mapped = sorted((i["type"], tuple(sorted(mp[a] for a in i["affected_ids"])), i["expected_amount_minor"], i["currency"])
                            for i in ep.truth["issues"])
            new = sorted((i["type"], tuple(i["affected_ids"]), i["expected_amount_minor"], i["currency"]) for i in tep.truth["issues"])
            self.assertEqual(mapped, new)
            payload = {"episode_id": ep.episode_id, "findings": oracle_findings(ep.truth)}
            r0 = Grader(ep.public(), ep.truth).grade(payload, 0)["reward"]
            r1 = Grader(tep.public(), tep.truth).grade(map_findings(payload, mp), 0)["reward"]
            self.assertEqual(r0, 1.0)
            self.assertEqual(r1, 1.0)
            # an unmapped answer does not transfer (old ids are gone)
            stale = Grader(tep.public(), tep.truth).grade(dict(payload, episode_id=tep.episode_id), 0)
            if ep.truth["issues"]:
                self.assertEqual(stale["tp"], 0)
            found, anomalies = reconcile(tep.public())
            self.assertEqual(anomalies, [])

    def test_scripted_auditor_reward_invariant(self):
        for seed in range(5):
            _, ev0 = run_policy(AuditEnv(), "scripted_auditor", seed, ERRORS_ALL)
            cfg = dict(ERRORS_ALL, transform={"seed": 7, "permute": True, "remap_ids": True})
            _, ev1 = run_policy(AuditEnv(), "scripted_auditor", seed, cfg)
            self.assertEqual(ev0["reward"], ev1["reward"])
            self.assertEqual((ev0["tp"], ev0["fp"], ev0["fn"]), (ev1["tp"], ev1["fp"], ev1["fn"]))

    def test_transform_from_disk(self):
        import tempfile
        ep = generate_episode(3, ERRORS_ALL)
        with tempfile.TemporaryDirectory() as td:
            ep.write(td)
            env = AuditEnv()
            obs, _ = env.reset(config={"episode_dir": td, "transform": {"seed": 5}})
            self.assertNotEqual(obs["manifest"]["episode_id"], ep.episode_id)
            from tests.helpers import submit_action
            o, r, t, tr, info = env.step(submit_action(env.episode.episode_id, oracle_findings(env.episode.truth)))
            self.assertEqual(r, 1.0)


if __name__ == "__main__":
    unittest.main()
