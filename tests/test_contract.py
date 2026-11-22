"""reset/step contract, tools, limits, termination, invalid input (spec section 04)."""
from __future__ import annotations

import json
import unittest

from auditenv.env import AuditEnv, ResetNeeded
from auditenv.schema_lite import check
from auditenv.version import MAX_STEPS, PAGE_LIMIT
from tests.helpers import CLEAN, SOME_ERRORS, dumps, submit_action


class ResetContract(unittest.TestCase):
    def test_reset_returns_obs_and_info(self):
        env = AuditEnv()
        out = env.reset(seed=1, config={})
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        obs, info = out
        for k in ("task", "manifest", "policy", "tools", "steps_remaining"):
            self.assertIn(k, obs)
        self.assertEqual(obs["steps_remaining"], MAX_STEPS)
        self.assertEqual(obs["steps_used"], 0)
        self.assertEqual(json.loads(json.dumps(obs)), obs)
        self.assertEqual(json.loads(json.dumps(info)), info)
        self.assertEqual(check(obs, "observation"), [])
        self.assertEqual(check(obs["manifest"], "manifest"), [])
        m = obs["manifest"]
        for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone", "counts", "rounding"):
            self.assertIn(k, m)
        for k in ("rate_date", "pair", "publication"):
            self.assertIn(k, obs["policy"]["fx"])
        self.assertIn("period", obs["policy"])
        self.assertIn("rounding", obs["policy"])

    def test_step_before_reset_raises(self):
        with self.assertRaises(ResetNeeded):
            AuditEnv().step({"action": "calculator", "args": {"expression": "1"}})

    def test_reset_isolates_state(self):
        env = AuditEnv()
        env.reset(seed=1, config={})
        for _ in range(5):
            env.step({"action": "calculator", "args": {"expression": "2*3"}})
        env.step(submit_action(env.episode.episode_id, []))
        obs, _ = env.reset(seed=2, config={})
        self.assertEqual(obs["steps_used"], 0)
        o, r, t, tr, i = env.step({"action": "calculator", "args": {"expression": "1"}})
        self.assertEqual(o["steps_used"], 1)
        self.assertFalse(t or tr)
        # a fresh env with the same seed sees exactly the same first observation
        a, _ = AuditEnv().reset(seed=2, config={})
        self.assertEqual(dumps(a), dumps(obs))


class StepContract(unittest.TestCase):
    def setUp(self):
        self.env = AuditEnv()
        self.obs, _ = self.env.reset(seed=11, config=SOME_ERRORS)

    def step(self, action):
        out = self.env.step(action)
        self.assertEqual(len(out), 5)
        obs, reward, term, trunc, info = out
        self.assertEqual(json.loads(json.dumps(obs)), obs)
        self.assertEqual(check(obs, "observation"), [], obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(term, bool)
        self.assertIsInstance(trunc, bool)
        self.assertIsInstance(info, dict)
        return out

    def test_exploration_reward_is_zero(self):
        for a in ({"action": "list_records", "args": {"table": "ledger"}},
                  {"action": "get_record", "args": {"table": "fx", "id": "nope"}},
                  {"action": "calculator", "args": {"expression": "1/3"}},
                  {"action": "bogus"}):
            obs, r, t, tr, info = self.step(a)
            self.assertEqual(r, 0.0)
            self.assertFalse(t or tr)
            self.assertEqual(info, {})

    def test_pagination_covers_every_record_once_and_is_stable(self):
        for table, n in (("ledger", self.obs["manifest"]["counts"]["ledger_rows"]),
                         ("invoices", self.obs["manifest"]["counts"]["invoices"]),
                         ("payments", self.obs["manifest"]["counts"]["payments"])):
            seen, cursor, pages = [], None, []
            while True:
                obs, *_ = self.step({"action": "list_records", "args": {"table": table, "cursor": cursor, "limit": PAGE_LIMIT}})
                recs = obs["result"]["records"]
                self.assertLessEqual(len(recs), PAGE_LIMIT)
                pages.append((cursor, dumps(recs)))
                seen += recs
                cursor = obs["result"]["next_cursor"]
                if cursor is None:
                    break
                if obs["steps_remaining"] < 5:
                    self.skipTest("budget")
            self.assertEqual(len(seen), n)
            key = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id"}[table]
            self.assertEqual(len({r[key] for r in seen}), n)
            c, page = pages[-1]
            obs, *_ = self.step({"action": "list_records", "args": {"table": table, "cursor": c, "limit": PAGE_LIMIT}})
            self.assertEqual(dumps(obs["result"]["records"]), page)  # same cursor -> same page
            self.env.reset(seed=11, config=SOME_ERRORS)

    def test_limits_and_bad_arguments(self):
        for args, code in (({"table": "ledger", "limit": 51}, "invalid_arguments"),
                           ({"table": "ledger", "limit": 0}, "invalid_arguments"),
                           ({"table": "ledger", "limit": "5"}, "invalid_arguments"),
                           ({"table": "ledger", "limit": True}, "invalid_arguments"),
                           ({"table": "truth"}, "unknown_table"),
                           ({"table": "../private/truth.json"}, "unknown_table"),
                           ({"table": "ledger", "cursor": "invoices@0"}, "invalid_cursor"),
                           ({"table": "ledger", "cursor": "ledger@999999"}, "invalid_cursor"),
                           ({"table": "ledger", "cursor": 5}, "invalid_cursor"),
                           ({"table": "ledger", "extra": 1}, "invalid_arguments"),
                           ({}, "invalid_arguments")):
            before = self.env._steps
            obs, r, t, tr, _ = self.step({"action": "list_records", "args": args})
            self.assertEqual(obs["kind"], "error")
            self.assertEqual(obs["error"]["code"], code, args)
            self.assertLessEqual(len(obs["error"]["message"]), 300)
            self.assertEqual(self.env._steps, before + 1)  # consumes a step
            self.assertEqual(r, 0.0)

    def test_unknown_action_and_id(self):
        for action, code in (({"action": "shell", "args": {"cmd": "ls"}}, "invalid_action"),
                             ({"action": "sql", "args": {"q": "select"}}, "invalid_action"),
                             ("not json", "invalid_action"),
                             ([1, 2], "invalid_action"),
                             ({"args": {}}, "invalid_action"),
                             ({"action": "get_record", "args": {"table": "ledger", "id": "L-DOESNOTEXIST"}}, "unknown_id"),
                             ({"action": "get_document", "args": {"doc_id": "DOC-NOPE"}}, "unknown_id"),
                             ({"action": "get_record", "args": {"table": "ledger"}}, "invalid_arguments")):
            obs, r, t, tr, _ = self.step(action)
            self.assertEqual(obs["error"]["code"], code, action)

    def test_get_record_and_document(self):
        obs, *_ = self.step({"action": "list_records", "args": {"table": "invoices", "limit": 3}})
        inv = obs["result"]["records"][0]
        self.assertNotIn("line_items", inv)
        obs, *_ = self.step({"action": "get_record", "args": {"table": "invoices", "id": inv["invoice_id"]}})
        self.assertEqual(obs["result"]["record"], inv)
        obs, *_ = self.step({"action": "get_document", "args": {"doc_id": inv["doc_id"]}})
        doc = obs["result"]["document"]
        self.assertEqual(len(doc["line_items"]), inv["line_item_count"])
        self.assertEqual(check(doc, "invoice"), [])
        fx = self.env.episode.fx[0]
        fid = f"FX:{fx['date']}:{fx['source_currency']}:{fx['target_currency']}"
        obs, *_ = self.step({"action": "get_record", "args": {"table": "fx", "id": fid}})
        self.assertEqual(obs["result"]["record"]["rate"], fx["rate"])

    def test_returned_records_are_copies(self):
        obs, *_ = self.step({"action": "list_records", "args": {"table": "ledger", "limit": 1}})
        rid = obs["result"]["records"][0]["row_id"]
        obs["result"]["records"][0]["debit_minor"] = 10 ** 12
        obs2, *_ = self.step({"action": "get_record", "args": {"table": "ledger", "id": rid}})
        self.assertNotEqual(obs2["result"]["record"]["debit_minor"], 10 ** 12)

    def test_calculator(self):
        obs, *_ = self.step({"action": "calculator", "args": {"expression": "(1250.50 * 3) / 2 - 0.25"}})
        self.assertEqual(obs["result"]["result"], "1875.5")
        for bad in ("__import__('os')", "open('x')", "2**10", "1e3", "a+1", "1/0", "x" * 300, "().__class__", "٣+1"):
            obs, *_ = self.step({"action": "calculator", "args": {"expression": bad}})
            self.assertEqual(obs["kind"], "error", bad)
            self.assertTrue(obs["error"]["code"].startswith("calculator") or obs["error"]["code"] == "invalid_arguments")

    def test_step_limit_truncates_with_empty_answer(self):
        env = AuditEnv()
        env.reset(seed=3, config=SOME_ERRORS)
        for i in range(MAX_STEPS):
            obs, r, t, tr, info = env.step({"action": "calculator", "args": {"expression": "1"}})
            if i < MAX_STEPS - 1:
                self.assertFalse(t or tr)
        self.assertTrue(tr)
        self.assertFalse(t)
        self.assertEqual(obs["steps_remaining"], 0)
        ev = env.last_evaluation()
        self.assertEqual(ev["status"], "step_limit")
        self.assertEqual(ev["n_findings"], 0)
        self.assertEqual(r, -1.0)  # F1 = 0 on erroneous books, minus step cost, clipped

    def test_step_limit_on_clean_books(self):
        env = AuditEnv()
        env.reset(seed=3, config=CLEAN)
        for _ in range(MAX_STEPS):
            obs, r, t, tr, info = env.step({"action": "calculator", "args": {"expression": "1"}})
        self.assertTrue(tr)
        self.assertAlmostEqual(r, 2 * 1 - 1 - 0.002 * 100)

    def test_time_limit_truncates(self):
        t = [0.0]
        env = AuditEnv(time_limit_s=60, clock=lambda: t[0])
        env.reset(seed=3, config={})
        env.step({"action": "calculator", "args": {"expression": "1"}})
        t[0] = 61.0
        obs, r, term, trunc, info = env.step({"action": "list_records", "args": {"table": "ledger"}})
        self.assertTrue(trunc)
        self.assertFalse(term)
        self.assertEqual(obs["kind"], "truncated")
        self.assertEqual(obs["reason"], "time_limit")
        self.assertEqual(env.last_evaluation()["status"], "time_limit")

    def test_submit_terminates_and_resubmit_is_rejected(self):
        eid = self.env.episode.episode_id
        obs, r, t, tr, info = self.step(submit_action(eid, []))
        self.assertTrue(t)
        self.assertFalse(tr)
        n = len(self.env.evaluator_log)
        obs, r2, t2, tr2, info2 = self.step(submit_action(eid, []))
        self.assertEqual(obs["error"]["code"], "episode_over")
        self.assertEqual(r2, 0.0)
        self.assertEqual(len(self.env.evaluator_log), n)

    def test_invalid_submission_is_minus_one(self):
        eid = self.env.episode.episode_id
        for bad in ({"episode_id": eid},
                    {"episode_id": "EP-OTHER", "findings": []},
                    {"episode_id": eid, "findings": [{"type": "fraud"}]},
                    "findings",
                    None):
            env = AuditEnv()
            env.reset(seed=11, config=SOME_ERRORS)
            obs, r, t, tr, info = env.step({"action": "submit", "args": {"findings": bad}})
            self.assertTrue(t)
            self.assertEqual(r, -1.0)
            self.assertEqual(env.last_evaluation()["status"], "invalid_submission")
            self.assertTrue(env.last_evaluation()["error"])


class EvalMode(unittest.TestCase):
    def test_eval_mode_withholds_reward_and_diagnostics(self):
        env = AuditEnv(mode="eval")
        env.reset(seed=5, config=SOME_ERRORS)
        from auditenv.policies import oracle_findings
        obs, r, t, tr, info = env.step(submit_action(env.episode.episode_id, oracle_findings(env.episode.truth)))
        self.assertTrue(t)
        self.assertEqual(r, 0.0)
        self.assertEqual(info, {})
        self.assertEqual(obs["result"], {"status": "received"})
        self.assertEqual(env.last_evaluation()["reward"], 1.0)
        env.reset(seed=5, config=SOME_ERRORS)
        obs2, r2, *_ = env.step({"action": "submit", "args": {"findings": "garbage"}})
        self.assertEqual(r2, 0.0)
        # only structural validity (public schema) is reported, never the score
        self.assertEqual(obs2["result"]["status"], "invalid_submission")
        self.assertEqual(set(obs2["result"]), {"status", "error"})
        self.assertEqual(env.last_evaluation()["reward"], -1.0)


if __name__ == "__main__":
    unittest.main()
