"""The policy runs in its own process without the secret salt and sees only observations."""
import ast
import os
import unittest

from auditenv.harness import load_split, run_suite


class AgentProcessTest(unittest.TestCase):
    def setUp(self):
        self.split = load_split("holdout_sim")
        self.split["episodes"] = self.split["episodes"][5:7]

    def test_external_policy_runs_in_separate_process_without_secret(self):
        self.assertTrue(self.split["secret"])
        os.environ["AUDITENV_SECRET"] = self.split["secret"]  # present in the harness process
        try:
            res = run_suite("tests.probe_policies:Probe", self.split, mode="eval", isolated=True, record=True)
        finally:
            del os.environ["AUDITENV_SECRET"]
        for r in res:
            seen = ast.literal_eval(r["final_action"]["args"]["findings"]["findings"][0]["explanation"])
            self.assertIsNone(seen["secret"])
            self.assertNotEqual(seen["pid"], os.getpid())
            self.assertFalse(seen["generator_loaded"])
            self.assertEqual(r["agent_return"], 0.0)       # closed mode: reward withheld
            self.assertEqual(r["grade"]["fp"], 1)          # graded by the evaluator process
            for rec in r["trajectory"]:
                self.assertNotIn("truth", rec["observation"])

    def test_crashing_policy_is_an_invalid_submission(self):
        res = run_suite("tests.probe_policies:Crasher", self.split, mode="eval", isolated=True)
        for r in res:
            self.assertEqual(r["grade"]["status"], "invalid_submission")
            self.assertEqual(r["grade"]["reward"], -1.0)

    def test_evaluate_command_on_a_split_file_with_secret_from_environment(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from auditenv.cli import main
        with tempfile.TemporaryDirectory() as td:
            split = {"split": "hidden", "config_version": "1.0",
                     "episodes": [{"seed": 4242, "config": {"config_version": "1.0", "name": "hidden-1"}}]}
            path = Path(td) / "hidden.json"
            path.write_text(json.dumps(split), encoding="utf-8")
            outs = {}
            for secret in ("salt-a", "salt-b"):
                os.environ["AUDITENV_SECRET"] = secret
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main(["evaluate", "--split", str(path), "--policy", "oracle",
                                               "--policy", "scripted_auditor", "--out", str(Path(td) / "r.json")]), 0)
                finally:
                    del os.environ["AUDITENV_SECRET"]
                outs[secret] = json.loads((Path(td) / "r.json").read_text(encoding="utf-8"))
            for out in outs.values():
                rows = {r["policy"]: r["summary"] for r in out["rows"]}
                self.assertEqual(rows["oracle"]["return_min"], 1.0)
                self.assertEqual(rows["scripted_auditor"]["agent_visible_return_mean"], 0.0)
            ids = [o["rows"][0]["per_episode"][0]["episode_id"] for o in outs.values()]
            self.assertNotEqual(ids[0], ids[1])  # the salt changes the episode

    def test_oracle_cannot_run_in_agent_process(self):
        from auditenv.agent_runner import main
        with self.assertRaises(SystemExit):
            main(["--policy", "oracle"])


if __name__ == "__main__":
    unittest.main()
