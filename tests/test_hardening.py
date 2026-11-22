"""Regression tests for defects found by the independent pre-release QA pass."""
import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import judge_adapter as ja
from auditenv.calculator import CalcError, evaluate
from auditenv.cli import main
from auditenv.env import AuditEnv
from auditenv.episode import Episode
from auditenv.harness import load_split, run_episode, run_suite
from auditenv.remote import RemoteAuditEnv, RemotePolicy

ROOT = Path(__file__).resolve().parent.parent
DEV = sorted(p for p in (ROOT / "episodes" / "dev").iterdir() if p.is_dir())


def quiet(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            return main(argv)
        except SystemExit as exc:
            return exc.code


def closed(d):
    return {"mode": "closed", "episode_dir": str(d), "team_config": {}}


def submit(d, findings=None):
    eid = json.loads((Path(d) / "public" / "manifest.json").read_text(encoding="utf-8"))["episode_id"]
    return {"name": "submit", "args": {"findings": {"episode_id": eid, "findings": findings or []}}}


class LogCase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.log = Path(self.td.name) / "evaluator.jsonl"
        patcher = mock.patch.dict(os.environ, {"AUDITENV_EVALUATOR_LOG": str(self.log)})
        patcher.start()
        self.addCleanup(patcher.stop)


class PolicyBoundary(unittest.TestCase):
    def test_policy_process_cannot_reach_closed_data_or_tools(self):
        split = load_split("dev")
        split["episodes"] = split["episodes"][:1]
        res = run_suite("tests.hardening_policies:Attacker", split, mode="eval", isolated=True, record=False)
        seen = json.loads(res[0]["final_action"]["args"]["findings"]["findings"][0]["explanation"])
        for key in ("read_truth", "list_episodes", "import_grader", "spawn", "socket", "write_file"):
            self.assertIs(seen[key], True, key)
        self.assertFalse(seen["glob_truth_found"])
        self.assertFalse((ROOT / "hardening-forged.jsonl").exists())

    def test_builtin_policies_still_work_under_the_guard(self):
        split = load_split("dev")
        split["episodes"] = split["episodes"][:2]
        for name in ("random", "reconciliation", "flag_all", "scripted_auditor"):
            res = run_suite(name, split, mode="eval", isolated=True)
            self.assertTrue(all(r["grade"]["status"] == "submitted" for r in res), name)

    def test_policy_process_does_not_inherit_log_location(self):
        from auditenv.remote import policy_env
        with mock.patch.dict(os.environ, {"AUDITENV_EVALUATOR_LOG": "x", "AUDITENV_LOG_KEY": "y", "AUDITENV_SECRET": "z"}):
            env = policy_env()
        for key in ("AUDITENV_EVALUATOR_LOG", "AUDITENV_LOG_KEY", "AUDITENV_SECRET"):
            self.assertNotIn(key, env)


class Timing(unittest.TestCase):
    def test_time_limit_counts_from_the_start_of_reset(self):
        now = [100.0]
        load = Episode.load

        def slow(*a, **k):
            now[0] += 5.0
            return load(*a, **k)

        env = AuditEnv(mode="eval", time_limit_s=60, clock=lambda: now[0])
        with mock.patch.object(Episode, "load", side_effect=slow):
            env.reset(1, {"episode_dir": str(DEV[0])})
        now[0] = 160.5  # 60.5 s after reset began, 55.5 s after loading finished
        obs, _, term, trunc, _ = env.step({"action": "calculator", "args": {"expression": "1+1"}})
        self.assertTrue(trunc and not term)
        self.assertEqual(obs["reason"], "time_limit")

    def test_hung_policy_is_stopped_counted_empty_and_replaced(self):
        split = load_split("dev")
        with tempfile.TemporaryDirectory() as td:
            log = str(Path(td) / "evaluator.jsonl")
            env = RemoteAuditEnv(log, mode="eval", time_limit_s=0.5)
            agent = RemotePolicy("tests.hardening_policies:HangOnce", time_limit_s=0.5)
            t0 = time.monotonic()
            try:
                for i, e in enumerate(split["episodes"][:2]):
                    run_episode(env, "hang", e["seed"], e["config"], i, record=False, policy=agent)
            finally:
                env.close()
                agent.close()
            self.assertLess(time.monotonic() - t0, 15)
            grades = [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(grades[0]["status"], "time_limit")
        self.assertEqual(grades[0]["n_findings"], 0)
        self.assertEqual(grades[1]["status"], "submitted")


class Encoding(LogCase):
    def test_lone_surrogate_is_a_bounded_error(self):
        env = ja.make_env()
        try:
            env.reset(1, closed(DEV[0]))
            obs, _, term, trunc, _ = env.step({"name": "calculator", "args": {"expression": "\ud800"}})
            self.assertEqual(obs["error"]["code"], "calculator_syntax")
            self.assertFalse(term or trunc)
        finally:
            env.close()

    def test_score_episode_with_surrogate_key_is_invalid_not_a_crash(self):
        eid = submit(DEV[0])["args"]["findings"]["episode_id"]
        self.assertEqual(ja.score_episode(str(DEV[0]), {"episode_id": eid, "findings": [], "\ud800": 1}, 0)["return"], -1.0)

    def test_cyrillic_explanation_survives_a_legacy_locale(self):
        d = next(p for p in DEV if json.loads((p / "private" / "truth.json").read_text(encoding="utf-8"))["issues"])
        answer = ja.oracle_findings(str(d))
        for f in answer["findings"]:
            f["explanation"] = "Расхождение " * 80
        with mock.patch.dict(os.environ, {"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp1251"}):
            env = ja.make_env()
            try:
                env.reset(1, closed(d))
                obs = env.step({"name": "submit", "args": {"findings": answer}})[0]
                self.assertEqual(obs["result"], {"accepted": True})
                self.assertEqual(env.last_evaluation()["f1"], 1.0)
            finally:
                env.close()


class EvaluatorLog(LogCase):
    def test_grade_belongs_to_its_own_environment(self):
        d = next(p for p in DEV if json.loads((p / "private" / "truth.json").read_text(encoding="utf-8"))["issues"])
        a, b = ja.make_env(), ja.make_env()
        try:
            a.reset(1, closed(d))
            a.step({"name": "submit", "args": {"findings": ja.oracle_findings(str(d))}})
            b.reset(1, closed(d))
            self.assertIsNone(b.last_evaluation())
            b.step(submit(d))
            self.assertGreater(a.last_evaluation()["reward"], 0.9)
            self.assertEqual(b.last_evaluation()["tp"], 0)
        finally:
            a.close()
            b.close()

    def test_forged_and_broken_lines_are_ignored(self):
        d = DEV[0]
        env = ja.make_env()
        try:
            env.reset(1, closed(d))
            env.step(submit(d))
            real = env.last_evaluation()
            with open(self.log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"episode_id": "x", "reward": 1.0, "forged": True}) + "\n")
                fh.write(json.dumps({"reward": 1.0, "log_auth": {"seq": 99, "mac": "0" * 64}}) + "\n")
                fh.write("{not json\n")
            self.assertEqual(env.last_evaluation(), real)
        finally:
            env.close()


class CommandLine(unittest.TestCase):
    def test_validate_of_nothing_fails(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(quiet(["validate", "--root", td]), 1)
            self.assertEqual(quiet(["validate", "--root", str(Path(td) / "missing")]), 1)

    def test_evaluate_rejects_a_policy_that_cannot_load(self):
        for name in ("nope", "os:path", "no.such.module:Agent"):
            self.assertEqual(quiet(["evaluate", "--split", "dev", "--policy", name]), 2, name)

    def test_demo_rejects_out_of_range_index(self):
        for index in ("0", "-3", "99"):
            self.assertNotEqual(quiet(["demo", "--split", "dev", f"--index={index}"]), 0, index)

    def test_unknown_split_is_a_clean_error(self):
        code = quiet(["baselines", "--split", "nope"])
        self.assertIsInstance(code, str)
        self.assertIn("nope", code)

    def test_json_string_and_malformed_submissions_are_summarized(self):
        split = load_split("dev")
        split["episodes"] = split["episodes"][:1]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "one.json"
            path.write_text(json.dumps(split), encoding="utf-8")
            self.assertEqual(quiet(["evaluate", "--split", str(path), "--policy", "tests.hardening_policies:StringSubmit",
                                    "--policy", "tests.hardening_policies:ListArgsSubmit"]), 0)


class Inputs(LogCase):
    def test_score_episode_rejects_impossible_step_counts(self):
        answer = ja.oracle_findings(str(DEV[0]))
        for bad in (-1, True):
            with self.assertRaises(ValueError):
                ja.score_episode(str(DEV[0]), answer, bad)

    def test_calculator_bounds_every_literal(self):
        with self.assertRaises(CalcError):
            evaluate("1" + "0" * 40 + " - 1" + "0" * 40)
        self.assertEqual(evaluate("999999999999999999999999999999 - 1"), "999999999999999999999999999998")


if __name__ == "__main__":
    unittest.main()
