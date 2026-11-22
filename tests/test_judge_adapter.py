"""Jury check definitions (Genesis FinTech, 20 Nov 2026) run against judge_adapter.py.

Block 1: contract tests C01-C20 through make_env() with episode_dir fixtures C, E, L and id U.
Block 2: the D01-D20 matrix (configs/splits/judge_matrix.json) is generated and independently validated.
Block 3: the 20 matcher/reward tests and the open examples M01-M12 through score_episode().
Block 4: 20 paired checks: the jury-style copy (permutation and/or one-to-one id replacement done on
         the files, truth and provenance corrected in the private part) is loaded through episode_dir.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import os
import random
import shutil
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path

import judge_adapter as ja
from auditenv.episode import Episode
from auditenv.harness import load_split
from auditenv.leakscan import forbidden_tokens, scan_text
from auditenv.validate_data import validate_episode_dir

CLEAN = {"config_version": "1.0", "name": "fixture-C",
         "errors": {"duplicate_invoice": 0, "payment_allocation": 0, "fx_conversion": 0, "period_cutoff": 0},
         "interactions": {"fx_partial_boundary": 0, "duplicate_misapplied": 0}}
ONE_ERROR = copy.deepcopy(CLEAN)
ONE_ERROR["name"] = "fixture-E"
ONE_ERROR["errors"]["payment_allocation"] = 1
SEED_C, SEED_E = 918273645, 918273646
EPS = 1e-9


def dumps(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


class JudgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="judge-")
        cls.tmp = Path(cls._tmp.name)
        cls._old_log = os.environ.get("AUDITENV_EVALUATOR_LOG")
        cls.log = cls.tmp / "evaluator.jsonl"
        os.environ["AUDITENV_EVALUATOR_LOG"] = str(cls.log)
        cls.C = cls.tmp / "C"
        cls.E = cls.tmp / "E"
        ja.generate_episode(SEED_C, CLEAN, str(cls.C))
        ja.generate_episode(SEED_E, ONE_ERROR, str(cls.E))
        cls.L = cls.C  # its ledger has >= 101 rows (checked in C05)

    @classmethod
    def tearDownClass(cls):
        if cls._old_log is None:
            os.environ.pop("AUDITENV_EVALUATOR_LOG", None)
        else:
            os.environ["AUDITENV_EVALUATOR_LOG"] = cls._old_log
        cls._tmp.cleanup()

    def log_lines(self):
        return [json.loads(l) for l in self.log.read_text(encoding="utf-8").splitlines()] if self.log.exists() else []


# =============================================================================== Block 1
class Block1Contract(JudgeBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.env = ja.make_env()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        super().tearDownClass()

    def closed(self, d):
        return {"mode": "closed", "episode_dir": str(d), "team_config": {}}

    def reset(self, d=None, seed=0):
        return self.env.reset(seed, self.closed(d or self.C))

    def step(self, name, **args):
        return self.env.step({"name": name, "args": args})

    def public(self, d):
        return Episode.load(d).public()

    def unknown_id(self, d):
        ep = Episode.load(d)
        text = dumps(ep.public())
        u = "U-0000-NOT-IN-FIXTURE"
        self.assertNotIn(u, text)
        return u

    def test_C01_reset_returns_json_pair(self):
        obs, info = self.reset()
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)
        self.assertEqual(json.loads(json.dumps([obs, info])), [obs, info])

    def test_C02_first_observation(self):
        obs, _ = self.reset()
        self.assertEqual(set(obs), {"task", "manifest", "policy", "remaining_steps"})
        self.assertEqual(obs["remaining_steps"], 100)
        self.assertIsInstance(obs["task"], str)
        text = dumps(obs)
        self.assertNotIn(str(SEED_C), text)
        self.assertEqual(scan_text(text, forbidden_tokens(Episode.load(self.C))), [])

    def test_C03_manifest_matches_data(self):
        obs, _ = self.reset()
        m = obs["manifest"]
        for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                  "counts", "rounding"):
            self.assertIn(k, m)
        pub = self.public(self.C)
        self.assertEqual(m["counts"]["ledger_rows"], len(pub["ledger"]))
        self.assertEqual(m["counts"]["invoices"], len(pub["invoices"]))
        self.assertEqual(m["counts"]["payments"], len(pub["payments"]))
        self.assertEqual(m["counts"]["documents"], len(pub["invoices"]) + len(pub["payments"]))
        self.assertEqual(m["counts"]["fx_rates"], len(pub["fx"]))

    def test_C04_step_tuple_types(self):
        self.reset()
        out = self.step("list_records", table="ledger", cursor=None, limit=10)
        self.assertEqual(len(out), 5)
        obs, reward, term, trunc, info = out
        self.assertIsInstance(term, bool)
        self.assertIsInstance(trunc, bool)
        self.assertEqual(json.loads(json.dumps(out)), list(out))
        self.assertEqual(set(obs), {"result", "error", "remaining_steps"})
        self.assertEqual(set(obs["result"]), {"records", "next_cursor"})
        self.assertEqual(obs["remaining_steps"], 99)

    def test_C05_pages_cover_every_row_once(self):
        pub = self.public(self.L)
        self.assertGreaterEqual(len(pub["ledger"]), 101)
        self.reset(self.L)
        seen, cursor, pages = [], None, 0
        while True:
            obs = self.step("list_records", table="ledger", cursor=cursor, limit=50)[0]
            self.assertLessEqual(len(obs["result"]["records"]), 50)
            seen += [r["row_id"] for r in obs["result"]["records"]]
            pages += 1
            cursor = obs["result"]["next_cursor"]
            if cursor is None:
                break
        self.assertGreaterEqual(pages, 3)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(sorted(seen), sorted(r["row_id"] for r in pub["ledger"]))

    def test_C06_same_cursor_same_page(self):
        self.reset(self.L)
        cur = self.step("list_records", table="ledger", cursor=None, limit=50)[0]["result"]["next_cursor"]
        a = self.step("list_records", table="ledger", cursor=cur, limit=50)[0]
        b = self.step("list_records", table="ledger", cursor=cur, limit=50)[0]
        self.assertEqual(a["result"], b["result"])
        self.assertEqual(a["remaining_steps"] - 1, b["remaining_steps"])

    def test_C07_get_record(self):
        r1 = self.public(self.C)["ledger"][3]
        self.reset()
        obs = self.step("get_record", table="ledger", id=r1["row_id"])[0]
        self.assertIsNone(obs["error"])
        self.assertEqual(obs["result"], r1)

    def test_C08_get_document(self):
        d1 = self.public(self.C)["invoices"][2]
        self.reset()
        obs = self.step("get_document", doc_id=d1["doc_id"])[0]
        self.assertEqual(obs["result"], d1)

    def test_C09_calculator_decimal(self):
        self.reset()
        obs = self.step("calculator", expression="0.1+0.2")[0]
        self.assertEqual(obs["result"], "0.3")

    def test_C10_calculator_rejects_code(self):
        self.reset()
        for expr in ("open('/etc/passwd').read()", "__import__('os').system('id')", "1+1; import os",
                     "lambda: 1", "().__class__", "'a'*10"):
            obs = self.step("calculator", expression=expr)[0]
            self.assertIsNone(obs["result"])
            self.assertIsNotNone(obs["error"])
            self.assertNotIn("root:", dumps(obs))

    def test_C11_C12_unknown_ids(self):
        u = self.unknown_id(self.C)
        _, _ = self.reset()
        o1, _, t1, tr1, _ = self.step("get_record", table="ledger", id=u)
        o2, _, t2, tr2, _ = self.step("get_document", doc_id=u)
        for o in (o1, o2):
            self.assertIsNone(o["result"])
            self.assertEqual(set(o["error"]), {"code", "message"})
        self.assertEqual((o1["remaining_steps"], o2["remaining_steps"]), (99, 98))
        self.assertFalse(t1 or tr1 or t2 or tr2)
        self.assertNotIn("line_items", dumps(o2))
        o3 = self.step("list_records", table="invoices", cursor=None, limit=1)[0]  # process still alive
        self.assertIsNone(o3["error"])

    def test_C13_unknown_action(self):
        self.reset()
        obs, reward, term, trunc, _ = self.env.step({"name": "delete_everything", "args": {}})
        self.assertIsNotNone(obs["error"])
        self.assertEqual(obs["remaining_steps"], 99)
        self.assertFalse(term or trunc)

    def test_C14_error_costs_a_step_and_keeps_access(self):
        r1 = self.public(self.C)["ledger"][0]
        self.reset()
        a = self.step("get_record", table="ledger", id=r1["row_id"])[0]
        b = self.step("get_record", table="ledger", id=self.unknown_id(self.C))[0]
        c = self.step("get_record", table="ledger", id=r1["row_id"])[0]
        self.assertEqual([a["remaining_steps"], b["remaining_steps"], c["remaining_steps"]], [99, 98, 97])
        self.assertEqual(a["result"], c["result"])

    def test_C15_submit_is_final(self):
        obs, _ = self.reset(self.E)
        findings = ja.oracle_findings(str(self.E))
        n = len(self.log_lines())
        o, r, term, trunc, info = self.step("submit", findings=findings)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(o["result"], {"accepted": True})
        fixed = self.log_lines()[-1]
        self.assertEqual(len(self.log_lines()), n + 1)
        for later in ({"name": "submit", "args": {"findings": {"episode_id": "x", "findings": []}}},
                      {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 5}}):
            o2 = self.env.step(later)[0]
            self.assertIsNotNone(o2["error"])
        self.assertEqual(len(self.log_lines()), n + 1)
        self.assertEqual(self.log_lines()[-1], fixed)
        self.assertEqual((fixed["tp"], fixed["fp"], fixed["fn"]), (1, 0, 0))

    def test_C16_submit_as_action_100(self):
        obs, _ = self.reset(self.E)
        for i in range(99):
            o, _, term, trunc, _ = self.step("calculator", expression="1+1")
            self.assertFalse(term or trunc)
        self.assertEqual(o["remaining_steps"], 1)
        o, r, term, trunc, _ = self.step("submit", findings=ja.oracle_findings(str(self.E)))
        self.assertEqual((term, trunc), (True, False))
        self.assertEqual(o["result"], {"accepted": True})
        g = self.log_lines()[-1]
        self.assertEqual(g["inspection_steps"], 99)
        self.assertAlmostEqual(g["reward"], 1 - 0.002 * 99, delta=EPS)

    def test_C17_step_limit_without_submit(self):
        self.reset(self.E)
        for i in range(100):
            o, r, term, trunc, _ = self.step("calculator", expression="2*3")
            if i < 99:
                self.assertFalse(term or trunc)
        self.assertEqual((term, trunc), (False, True))
        g = self.log_lines()[-1]
        self.assertEqual((g["status"], g["n_findings"], g["tp"]), ("step_limit", 0, 0))
        n = len(self.log_lines())
        o2 = self.step("submit", findings=ja.oracle_findings(str(self.E)))[0]
        self.assertIsNotNone(o2["error"])
        self.assertEqual(len(self.log_lines()), n)

    def test_C18_time_limit_includes_policy_time(self):
        env = ja.JudgeEnv(time_limit_s=0.5)  # same code path as the 60 s default, shortened for the test
        try:
            env.reset(0, self.closed(self.E))
            env.step({"name": "calculator", "args": {"expression": "1+1"}})
            time.sleep(0.7)  # a slow policy
            o, r, term, trunc, info = env.step({"name": "submit", "args": {"findings": ja.oracle_findings(str(self.E))}})
            self.assertEqual((term, trunc, r, info), (False, True, 0.0, {}))
            self.assertEqual(o["error"]["code"], "time_limit")
            g = env.last_evaluation()
            self.assertEqual((g["status"], g["n_findings"], g["tp"]), ("time_limit", 0, 0))
        finally:
            env.close()

    def test_C19_closed_mode_withholds_reward(self):
        self.reset(self.E)
        for args in ({"table": "ledger", "cursor": None, "limit": 50}, {"table": "payments", "cursor": None, "limit": 50}):
            o, r, term, trunc, info = self.step("list_records", **args)
            self.assertEqual((r, info), (0.0, {}))
            self.assertIsInstance(r, float)
        o, r, term, trunc, info = self.step("submit", findings=ja.oracle_findings(str(self.E)))
        self.assertEqual((r, info, term), (0.0, {}, True))
        self.assertEqual(o, {"result": {"accepted": True}, "error": None, "remaining_steps": 97})
        g = self.log_lines()[-1]
        self.assertAlmostEqual(g["reward"], 1 - 0.002 * 2, delta=EPS)
        # train mode: the terminal reward reaches the policy
        self.env.reset(0, {"mode": "train", "episode_dir": str(self.E)})
        o, r, term, *_ = self.step("submit", findings=ja.oracle_findings(str(self.E)))
        self.assertEqual(r, 1.0)

    def test_C20_no_private_data_in_any_channel(self):
        for d, seed in ((self.C, SEED_C), (self.E, SEED_E)):
            ep = Episode.load(d)
            toks = forbidden_tokens(ep) + [str(seed), str(d), str(self.tmp)]
            outs = [self.reset(d)]
            for table, key in (("ledger", "row_id"), ("invoices", "invoice_id"), ("payments", "payment_id"), ("fx", "fx_id")):
                cursor = None
                while True:
                    o = self.step("list_records", table=table, cursor=cursor, limit=50)
                    outs.append(o)
                    cursor = o[0]["result"]["next_cursor"]
                    if cursor is None or o[0]["remaining_steps"] < 40:
                        break
            for doc in [x["doc_id"] for x in ep.invoices[:8] + ep.payments[:8]]:
                outs.append(self.step("get_document", doc_id=doc))
            removed = [r["row_id"] for r in ep.provenance.get("removed_ledger_rows", [])]
            for rid in removed[:3] + ["../private/truth.json", "' OR 1=1 --", "/etc/passwd"]:
                outs.append(self.step("get_record", table="ledger", id=rid))
                outs.append(self.step("get_document", doc_id=rid))
            outs.append(self.env.step({"name": "list_records", "args": {"table": "../private/truth", "cursor": None}}))
            outs.append(self.step("calculator", expression="__import__('subprocess').call('ls')"))
            outs.append(self.step("submit", findings={"episode_id": ep.episode_id, "findings": []}))
            text = dumps(outs)
            self.assertEqual(scan_text(text, toks), [], d.name)
            for o in outs[1:]:
                self.assertEqual(o[1], 0.0)
                self.assertEqual(o[4], {})
            # removed rows answer exactly like random ids
            errs = {dumps(o[0]["error"]) for o in outs if o[0].get("error")}
            self.assertTrue(all("truth" not in e for e in errs))


# =============================================================================== Block 2
class Block2Matrix(unittest.TestCase):
    EXPECT = {  # name prefix -> multiset of truth types (+interaction flag)
        "D01": [], "D02": [], "D03": [], "D04": [], "D05": [],
        "D06": ["duplicate_invoice"], "D07": ["duplicate_invoice"], "D08": ["duplicate_invoice"] * 2,
        "D09": ["payment_allocation"], "D10": ["payment_allocation"], "D11": ["payment_allocation"] * 2,
        "D12": ["fx_conversion"], "D13": ["fx_conversion"],
        "D14": ["period_cutoff"], "D15": ["period_cutoff"] * 2,
        "D16": ["fx_conversion+i", "payment_allocation+i"], "D17": ["fx_conversion+i", "payment_allocation+i"],
        "D18": ["duplicate_invoice", "fx_conversion+i", "payment_allocation+i"],
        "D19": ["duplicate_invoice+i", "payment_allocation+i"],
        "D20": ["duplicate_invoice+i", "fx_conversion", "payment_allocation+i"],
    }
    DISTRACTOR = {"D02": "split_payments", "D03": "refund_pairs", "D04": "recurring_customers",
                  "D05": "boundary_documents"}

    def test_matrix_generates_and_validates(self):
        split = load_split("judge_matrix")
        self.assertEqual(len(split["episodes"]), 20)
        with tempfile.TemporaryDirectory() as td:
            variants = []
            for e in split["episodes"]:
                name = e["config"]["name"][:3]
                with self.subTest(episode=name):
                    d = Path(td) / name
                    ja.generate_episode(e["seed"], e["config"], str(d))
                    res = validate_episode_dir(d)
                    self.assertTrue(res["ok"], res["errors"][:5])
                    ep = Episode.load(d)
                    got = sorted(i["type"] + ("+i" if i["interaction_id"] else "") for i in ep.truth["issues"])
                    self.assertEqual(got, sorted(self.EXPECT[name]))
                    if name in self.DISTRACTOR:
                        kinds = {x["kind"] for x in ep.provenance["distractors"]}
                        self.assertIn(self.DISTRACTOR[name], kinds)
                    variants += [i["variant"] for i in ep.truth["issues"] if i["type"] == "payment_allocation"
                                 and name in ("D09", "D10", "D11")]
            self.assertIn("missing", variants)
            self.assertIn("partial", variants)


# =============================================================================== Block 3
class Block3Matcher(JudgeBase):
    """Common fixture: one issue A of a supported type T on one canonical record R1, 10000 KZT,
    evidence set {D1, D2}, tolerance 0 in this fixture."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        pub = Episode.load(cls.C).public()
        cls.eid = pub["manifest"]["episode_id"]
        cls.R = [p["payment_id"] for p in pub["payments"][:5]]
        cls.D1, cls.D2 = pub["invoices"][0]["doc_id"], pub["payments"][0]["doc_id"]
        cls.D3 = pub["ledger"][0]["row_id"]
        cls.U = "U-NOT-IN-THIS-FIXTURE"
        assert cls.U not in dumps(pub)
        cls.T = "payment_allocation"

    def fixture(self, issues, name):
        d = self.tmp / name
        if not d.exists():
            shutil.copytree(self.C / "public", d / "public")
            (d / "private").mkdir()
            shutil.copy(self.C / "private" / "provenance.json", d / "private" / "provenance.json")
        (d / "private" / "truth.json").write_text(json.dumps({"episode_id": self.eid, "issues": issues}), encoding="utf-8")
        return str(d)

    def issue(self, iid="A", affected=None, amount=10000, tau=0, sets=None, type_=None):
        return {"issue_id": iid, "type": type_ or self.T, "affected_ids": affected or [self.R[0]],
                "expected_amount_minor": amount, "currency": "KZT", "tolerance_minor": tau,
                "evidence_sets": sets or [[self.D1, self.D2]]}

    def F(self, **kw):
        f = {"type": self.T, "affected_ids": [self.R[0]], "claimed_amount_minor": 10000, "currency": "KZT",
             "evidence_ids": [self.D1, self.D2], "explanation": "short reason"}
        f.update(kw)
        return f

    def score(self, issues, findings, steps=10, name="fx3"):
        d = self.fixture(issues, name)
        return ja.score_episode(d, {"episode_id": self.eid, "findings": findings}, steps)

    def expect(self, out, tp, fp, fn, p, r, f1, ret):
        self.assertEqual((out["tp"], out["fp"], out["fn"]), (tp, fp, fn))
        for k, v in (("precision", p), ("recall", r), ("f1", f1), ("return", ret)):
            self.assertAlmostEqual(out[k], float(v), delta=EPS, msg=k)

    def test_01_correct(self):
        self.expect(self.score([self.issue()], [self.F()]), 1, 0, 0, 1, 1, 1, Fraction(98, 100))

    def test_02_empty_findings(self):
        self.expect(self.score([self.issue()], []), 0, 0, 1, 0, 0, 0, -1)

    def test_03_empty_truth_empty_findings(self):
        self.expect(self.score([], []), 0, 0, 0, 0, 0, 1, Fraction(98, 100))

    def test_04_empty_truth_false_finding(self):
        self.expect(self.score([], [self.F()]), 0, 1, 0, 0, 0, 0, -1)

    def test_05_two_identical(self):
        self.expect(self.score([self.issue()], [self.F(), self.F()]), 1, 1, 0, Fraction(1, 2), 1, Fraction(2, 3),
                    Fraction(47, 150))

    def test_06_to_11_single_field_wrong(self):
        wrong = {
            "06_type": self.F(type="duplicate_invoice"),
            "07_other_record": self.F(affected_ids=[self.R[1]]),
            "08_amount_10001": self.F(claimed_amount_minor=10001),
            "09_currency": self.F(currency="USD"),
            "10_only_D1": self.F(evidence_ids=[self.D1]),
            "11_missing_evidence_id": self.F(evidence_ids=[self.D1, self.U]),
        }
        for label, f in wrong.items():
            with self.subTest(label):
                self.expect(self.score([self.issue()], [f]), 0, 1, 1, 0, 0, 0, -1)

    def test_12_flag_all(self):
        types = ["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"]
        fs = [self.F(type=t, affected_ids=[r]) for r in self.R for t in types]
        self.assertEqual(len(fs), 20)
        self.expect(self.score([self.issue()], fs), 1, 19, 0, Fraction(1, 20), 1, Fraction(2, 21), Fraction(-871, 1050))

    def test_13_14_two_issues(self):
        b = self.issue("B", affected=[self.R[2]], type_="duplicate_invoice", amount=5000, sets=[[self.D3]])
        self.expect(self.score([self.issue(), b], [self.F()]), 1, 0, 1, 1, Fraction(1, 2), Fraction(2, 3),
                    Fraction(47, 150))
        false = self.F(affected_ids=[self.R[4]])
        self.expect(self.score([self.issue(), b], [self.F(), false]), 1, 1, 1, Fraction(1, 2), Fraction(1, 2),
                    Fraction(1, 2), Fraction(-2, 100))

    def test_15_16_17_step_cost(self):
        for steps, ret in ((0, 1), (50, Fraction(90, 100)), (100, Fraction(80, 100))):
            self.expect(self.score([self.issue()], [self.F()], steps=steps), 1, 0, 0, 1, 1, 1, ret)

    def test_18_19_20_structure_limits(self):
        bad = {
            "18_31_findings": {"episode_id": self.eid, "findings": [self.F() for _ in range(31)]},
            "19_11_evidence": {"episode_id": self.eid, "findings": [self.F(evidence_ids=[f"E{i}" for i in range(11)])]},
            "20_explanation_1001": {"episode_id": self.eid, "findings": [self.F(explanation="a" * 1001)]},
        }
        d = self.fixture([self.issue()], "fx3")
        for label, payload in bad.items():
            with self.subTest(label):
                out = ja.score_episode(d, payload, 10)
                self.assertEqual(out, {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None,
                                       "f1": None, "return": -1.0})
                last = self.log_lines()[-1]
                self.assertEqual(last["status"], "invalid_submission")
                self.assertTrue(last["error"])
        ok = ja.score_episode(d, {"episode_id": self.eid, "findings": [self.F(explanation="a" * 1000)]}, 10)
        self.assertEqual(ok["tp"], 1)
        # the same limits end an episode with -1 in the evaluator log
        env = ja.make_env()
        try:
            for label, payload in bad.items():
                with self.subTest("episode " + label):
                    env.reset(0, {"mode": "closed", "episode_dir": d})
                    o, r, term, trunc, info = env.step({"name": "submit", "args": {"findings": payload}})
                    self.assertEqual((r, term, trunc, info), (0.0, True, False, {}))
                    self.assertEqual(o["error"]["code"], "invalid_submission")
                    g = env.last_evaluation()
                    self.assertEqual((g["status"], g["reward"]), ("invalid_submission", -1.0))
        finally:
            env.close()

    # ------------------------------------------------------------------ M01-M12
    def m_issue(self, **kw):
        return self.issue(affected=[self.R[0], self.R[1]], **kw)

    def m(self, findings, issues=None, name="fxm"):
        return self.score(issues or [self.m_issue()], findings, name=name)

    def MF(self, **kw):
        kw.setdefault("affected_ids", [self.R[0], self.R[1]])
        return self.F(**kw)

    def test_M01_affected_is_a_set(self):
        self.assertEqual(self.m([self.MF(affected_ids=[self.R[1], self.R[0], self.R[0]])])["tp"], 1)

    def test_M02_extra_affected(self):
        out = self.m([self.MF(affected_ids=[self.R[0], self.R[1], self.R[2]])])
        self.assertEqual((out["fp"], out["fn"]), (1, 1))

    def test_M03_extra_existing_evidence(self):
        self.assertEqual(self.m([self.MF(evidence_ids=[self.D2, self.D1, self.D3])])["tp"], 1)

    def test_M04_unknown_evidence(self):
        out = self.m([self.MF(evidence_ids=[self.D1, self.D2, self.U])])
        self.assertEqual((out["tp"], out["fp"], out["fn"]), (0, 1, 1))

    def test_M05_eleven_evidence_with_repeats(self):
        out = self.m([self.MF(evidence_ids=[self.D2] + [self.D1] * 10)])
        self.assertEqual(out["return"], -1.0)
        self.assertIsNone(out["tp"])

    def test_M06_currency_exact(self):
        for cur in ("kzt", " KZT", "KZT "):
            out = self.m([self.MF(currency=cur)])
            self.assertEqual((out["tp"], out["fp"], out["fn"]), (0, 1, 1), cur)

    def test_M07_amount_exact(self):
        self.assertEqual(self.m([self.MF(claimed_amount_minor=10001)])["tp"], 0)

    def test_M08_published_tau_one(self):
        issues = [self.m_issue(tau=1)]
        self.assertEqual(self.m([self.MF(claimed_amount_minor=10001)], issues, name="fxm8")["tp"], 1)
        self.assertEqual(self.m([self.MF(claimed_amount_minor=9999)], issues, name="fxm8")["tp"], 1)
        self.assertEqual(self.m([self.MF(claimed_amount_minor=10002)], issues, name="fxm8")["tp"], 0)

    def test_M09_amount_type(self):
        for amt in ("10000", True, 10000.0):
            self.assertEqual(self.m([self.MF(claimed_amount_minor=amt)])["return"], -1.0, repr(amt))

    def two_issues(self):
        a = self.m_issue(iid="A", sets=[[self.D1, self.D2]])
        b = self.m_issue(iid="B", sets=[[self.D3]])
        return [a, b]

    def test_M10_maximum_matching_not_greedy(self):
        f0 = self.MF(evidence_ids=[self.D1, self.D2, self.D3])  # fits A and B
        f1 = self.MF(evidence_ids=[self.D1, self.D2])  # fits only A
        out = self.m([f0, f1], self.two_issues(), name="fxm10")
        self.assertEqual((out["tp"], out["fp"], out["fn"]), (2, 0, 0))
        self.assertEqual(self.log_lines()[-1]["matches"], [{"finding": 1, "issue_id": "A"}, {"finding": 0, "issue_id": "B"}])

    def test_M11_deterministic_tie_break(self):
        both = self.MF(evidence_ids=[self.D1, self.D2, self.D3])
        out = self.m([both, copy.deepcopy(both)], self.two_issues(), name="fxm11")
        self.assertEqual(out["tp"], 2)
        self.assertEqual(self.log_lines()[-1]["matches"], [{"finding": 0, "issue_id": "A"}, {"finding": 1, "issue_id": "B"}])
        only_a = self.MF()
        x = self.m([both, only_a], self.two_issues(), name="fxm11")
        y = self.m([only_a, both], self.two_issues(), name="fxm11")
        self.assertEqual(x, y)

    def test_M12_identical_findings(self):
        out = self.m([self.MF(), self.MF()])
        self.assertEqual((out["tp"], out["fp"], out["fn"]), (1, 1, 0))
        self.assertAlmostEqual(out["return"], 47 / 150, delta=EPS)


# =============================================================================== Block 4
def _read_csv(p: Path):
    rows = list(csv.reader(io.StringIO(p.read_text(encoding="utf-8"))))
    return rows[0], rows[1:]


def _write_csv(p: Path, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    p.write_text(buf.getvalue(), encoding="utf-8")


def jury_transform(src: Path, dst: Path, permute: bool, remap: bool, seed: int) -> dict:
    """An independent, jury-style transformation of a copy: permutes records and/or replaces ids
    one-to-one in all data and references (separate namespaces), and corrects truth and provenance."""
    rnd = random.Random(seed)
    shutil.copytree(src, dst)
    pub, priv = dst / "public", dst / "private"
    lh, ledger = _read_csv(pub / "ledger.csv")
    fh, fx = _read_csv(pub / "fx.csv")
    invoices = read_jsonl(pub / "invoices.jsonl")
    payments = read_jsonl(pub / "payments.jsonl")
    truth = json.loads((priv / "truth.json").read_text(encoding="utf-8"))
    prov = json.loads((priv / "provenance.json").read_text(encoding="utf-8"))
    col = {c: i for i, c in enumerate(lh)}
    mapping = {}
    if remap:
        spaces = {"row": set(), "doc": set(), "inv": set(), "pay": set(), "ent": set()}
        for r in ledger + [[x["row_id"]] for x in prov.get("removed_ledger_rows", [])]:
            spaces["row"].add(r[0])
        for r in ledger:
            if r[col["entity_id"]] != "null":
                spaces["ent"].add(r[col["entity_id"]])
        for d in invoices:
            spaces["doc"].add(d["doc_id"]); spaces["inv"].add(d["invoice_id"]); spaces["ent"].add(d["entity_id"])
        for d in payments:
            spaces["doc"].add(d["doc_id"]); spaces["pay"].add(d["payment_id"])
        prefix = {"row": "LR", "doc": "DOC", "inv": "INVOICE", "pay": "PAYMENT", "ent": "CUSTOMER"}
        for ns, ids in spaces.items():
            ids = sorted(ids)
            nums = list(range(10_000, 10_000 + len(ids)))
            rnd.shuffle(nums)
            for old, n in zip(ids, nums):
                mapping[old] = f"{prefix[ns]}_{n}"
        assert len(set(mapping.values())) == len(mapping)

    def m(x):
        return mapping.get(x, x) if isinstance(x, str) else x

    for r in ledger:
        for c in ("row_id", "entity_id", "invoice_id", "payment_id"):
            r[col[c]] = m(r[col[c]])
    for d in invoices:
        for k in ("doc_id", "invoice_id", "entity_id"):
            d[k] = m(d[k])
    for d in payments:
        d["doc_id"], d["payment_id"] = m(d["doc_id"]), m(d["payment_id"])
        for ref in d["invoice_refs"] or []:
            ref["invoice_id"] = m(ref["invoice_id"])
    for it in truth["issues"]:
        it["affected_ids"] = [m(a) for a in it["affected_ids"]]
        it["evidence_sets"] = [[m(e) for e in s] for s in it["evidence_sets"]]

    def deep(x):
        if isinstance(x, dict):
            return {k: deep(v) for k, v in x.items()}
        if isinstance(x, list):
            return [deep(v) for v in x]
        return m(x)

    prov = deep(prov)
    if permute:
        for rows in (ledger, fx, invoices, payments):
            rnd.shuffle(rows)
    _write_csv(pub / "ledger.csv", lh, ledger)
    _write_csv(pub / "fx.csv", fh, fx)
    (pub / "invoices.jsonl").write_text("".join(json.dumps(d) + "\n" for d in invoices), encoding="utf-8")
    (pub / "payments.jsonl").write_text("".join(json.dumps(d) + "\n" for d in payments), encoding="utf-8")
    (priv / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    (priv / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return mapping


def canon(findings, inverse=None):
    inv = inverse or {}
    f = lambda x: inv.get(x, x)
    return sorted((g["type"], tuple(sorted(f(a) for a in g["affected_ids"])), g["claimed_amount_minor"], g["currency"],
                   tuple(sorted(f(e) for e in g["evidence_ids"]))) for g in findings)


class Block4Pairs(JudgeBase):
    def play(self, env, d, actions):
        obs, info = env.reset(0, {"mode": "closed", "episode_dir": str(d)})
        trace = [obs, info]
        for a in actions:
            trace.append(env.step(a))
        return trace, env.last_evaluation()

    def test_twenty_pairs(self):
        split = load_split("judge_matrix")
        env = ja.make_env()
        try:
            for k, e in enumerate(split["episodes"]):
                name = e["config"]["name"][:3]
                permute, remap = (True, False) if k < 5 else (False, True) if k < 10 else (True, True)
                with self.subTest(pair=f"P{k + 1:02d}", episode=name, permute=permute, remap=remap):
                    src, dst = self.tmp / f"pair{k}", self.tmp / f"pair{k}-t"
                    ja.generate_episode(e["seed"], e["config"], str(src))
                    mapping = jury_transform(src, dst, permute, remap, seed=100 + k)
                    inverse = {v: key for key, v in mapping.items()}
                    o_src, o_dst = ja.oracle_findings(str(src)), ja.oracle_findings(str(dst))
                    # 1. exact replay on the original
                    probe = [{"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}},
                             {"name": "list_records", "args": {"table": "payments", "cursor": None, "limit": 20}},
                             {"name": "calculator", "args": {"expression": "12.5*4"}}]
                    script = probe + [{"name": "submit", "args": {"findings": o_src}}]
                    t1, g1 = self.play(env, src, script)
                    t2, g2 = self.play(env, src, script)
                    self.assertEqual(dumps(t1), dumps(t2))
                    self.assertEqual(g1, g2)
                    # 2. oracle with the new ids maps back onto the same answer
                    self.assertEqual(canon(o_dst["findings"], inverse), canon(o_src["findings"]))
                    # 3. same metrics and reward with the same number of inspection steps
                    s1 = ja.score_episode(str(src), o_src, 3)
                    s2 = ja.score_episode(str(dst), o_dst, 3)
                    self.assertEqual(s1, s2)
                    self.assertAlmostEqual(s1["return"], 1 - 0.002 * 3, delta=EPS)
                    _, g3 = self.play(env, dst, probe + [{"name": "submit", "args": {"findings": o_dst}}])
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1", "reward", "inspection_steps"):
                        self.assertEqual(g3[key], g1[key], key)
                    # a broken evidence set must not keep the reward
                    if o_dst["findings"]:
                        broken = copy.deepcopy(o_dst)
                        broken["findings"][0]["evidence_ids"] = broken["findings"][0]["evidence_ids"][:-1] or ["NOPE"]
                        self.assertLess(ja.score_episode(str(dst), broken, 3)["tp"], s2["tp"])
                    # 4. no new leak channel on the transformed copy
                    ep_t = Episode.load(dst)
                    trace, _ = self.play(env, dst, probe)
                    self.assertEqual(scan_text(dumps(trace), forbidden_tokens(ep_t) + [str(dst)]), [])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
