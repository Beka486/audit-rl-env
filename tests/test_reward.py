"""Matcher and terminal reward (spec section 05), including the five required properties and attacks."""
from __future__ import annotations

import copy
import unittest
from fractions import Fraction

from auditenv.env import AuditEnv
from auditenv.grader import Grader, terminal_reward
from auditenv.money import amount_tolerance
from tests.helpers import CLEAN, ERRORS_ALL, SOME_ERRORS, oracle, run_policy, submit_after

SEEDS = range(5)


def fresh(seed, config):
    env = AuditEnv(mode="train")
    env.reset(seed=seed, config=config)
    return env


class Metrics(unittest.TestCase):
    def test_formula_edge_cases(self):
        self.assertEqual(terminal_reward(Fraction(1), 0), 1)
        self.assertEqual(terminal_reward(Fraction(0), 0), -1)
        self.assertEqual(terminal_reward(Fraction(1), 10), Fraction(98, 100))
        self.assertEqual(terminal_reward(Fraction(0), 99), -1)  # clipped
        self.assertEqual(terminal_reward(Fraction(1, 2), 0), 0)

    def test_empty_truth(self):
        env = fresh(0, CLEAN)
        self.assertEqual(env.episode.truth["issues"], [])
        r, ev = submit_after(env, [])
        self.assertEqual((ev["f1"], r), (1.0, 1.0))
        env = fresh(0, CLEAN)
        inv = env.episode.invoices[0]
        r, ev = submit_after(env, [{"type": "period_cutoff", "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": inv["total_minor"],
                                    "currency": inv["currency"], "evidence_ids": [inv["doc_id"]], "explanation": ""}])
        self.assertEqual(ev["f1"], 0.0)
        self.assertEqual(ev["precision"], 0.0)
        self.assertEqual(ev["recall"], 0.0)  # zero denominator -> 0
        self.assertEqual(r, -1.0)

    def test_precision_recall_f1(self):
        env = fresh(1, ERRORS_ALL)
        fs = oracle(env)
        n = len(fs)
        r, ev = submit_after(env, fs[: n // 2])
        tp = n // 2
        self.assertEqual(ev["tp"], tp)
        self.assertEqual(ev["fn"], n - tp)
        self.assertAlmostEqual(ev["recall"], tp / n)
        self.assertAlmostEqual(ev["f1"], 2 * tp / (2 * tp + (n - tp)))
        self.assertAlmostEqual(r, 2 * ev["f1"] - 1)


class RequiredProperties(unittest.TestCase):
    def test_correct_answer_beats_silence_on_erroneous_books(self):
        for s in SEEDS:
            r_ok, _ = submit_after(fresh(s, ERRORS_ALL), oracle(fresh(s, ERRORS_ALL)), explore_steps=40)
            r_silent, _ = submit_after(fresh(s, ERRORS_ALL), [], explore_steps=0)
            self.assertGreater(r_ok, r_silent)
            self.assertGreater(r_ok, 0.8)

    def test_silence_beats_false_accusations_on_clean_books(self):
        for s in SEEDS:
            env = fresh(s, CLEAN)
            p = env.episode.payments[0]
            false = [{"type": "payment_allocation", "affected_ids": [p["payment_id"]], "claimed_amount_minor": abs(p["amount_minor"]),
                      "currency": p["currency"], "evidence_ids": [p["doc_id"]], "explanation": "looks odd"}]
            r_false, _ = submit_after(env, false)
            r_silent, _ = submit_after(fresh(s, CLEAN), [], explore_steps=60)
            self.assertGreater(r_silent, r_false)
            # flag-all style attack on clean books
            _, ev = run_policy(AuditEnv(), "flag_all", s, CLEAN)
            self.assertLess(ev["reward"], r_silent)

    def test_duplicates_give_no_gain(self):
        for s in SEEDS:
            fs = oracle(fresh(s, ERRORS_ALL))
            r1, ev1 = submit_after(fresh(s, ERRORS_ALL), fs)
            r2, ev2 = submit_after(fresh(s, ERRORS_ALL), fs + copy.deepcopy(fs))
            r3, ev3 = submit_after(fresh(s, ERRORS_ALL), [fs[0]] * 10)
            self.assertEqual(ev2["tp"], ev1["tp"])
            self.assertEqual(ev2["fp"], len(fs))
            self.assertLess(r2, r1)
            self.assertEqual(ev3["tp"], 1)
            self.assertEqual(ev3["fp"], 9)

    def test_flag_everything_scores_badly(self):
        for s in SEEDS:
            _, ev = run_policy(AuditEnv(), "flag_all", s, ERRORS_ALL)
            self.assertLess(ev["reward"], -0.5)
            self.assertLess(ev["f1"], 0.25)
        # every type for every truth-affected id set, sprayed with all evidence it could carry
        env = fresh(2, ERRORS_ALL)
        types = ["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"]
        fs = []
        for it in env.episode.truth["issues"]:
            for t in types:
                if len(fs) < 30:
                    fs.append({"type": t, "affected_ids": it["affected_ids"], "claimed_amount_minor": it["expected_amount_minor"],
                               "currency": it["currency"], "evidence_ids": it["evidence_sets"][0], "explanation": ""})
        r_spray, ev = submit_after(env, fs)
        r_oracle, _ = submit_after(fresh(2, ERRORS_ALL), oracle(fresh(2, ERRORS_ALL)))
        self.assertLess(r_spray, r_oracle - 0.5)

    def test_step_savings_only_slightly_separate_equal_answers(self):
        for s in SEEDS:
            fs = oracle(fresh(s, ERRORS_ALL))
            r_fast, _ = submit_after(fresh(s, ERRORS_ALL), fs, explore_steps=0)
            r_slow, _ = submit_after(fresh(s, ERRORS_ALL), fs, explore_steps=60)
            self.assertGreater(r_fast, r_slow)
            self.assertAlmostEqual(r_fast - r_slow, 0.002 * 60)
            # with up to 12 issues one more true positive is worth more than 20 extra inspection steps
            r_less, _ = submit_after(fresh(s, ERRORS_ALL), fs[:-1], explore_steps=0)
            r_mid, _ = submit_after(fresh(s, ERRORS_ALL), fs, explore_steps=20)
            self.assertGreater(r_mid, r_less)


class MatchingRules(unittest.TestCase):
    def setUp(self):
        self.env = fresh(4, ERRORS_ALL)
        self.fs = oracle(self.env)
        self.ep = self.env.episode

    def grade(self, findings):
        return Grader(self.ep.public(), self.ep.truth).grade({"episode_id": self.ep.episode_id, "findings": findings}, 0)

    def test_oracle_is_perfect(self):
        ev = self.grade(self.fs)
        self.assertEqual((ev["tp"], ev["fp"], ev["fn"], ev["reward"]), (len(self.fs), 0, 0, 1.0))

    def test_each_field_is_required(self):
        for i, f in enumerate(self.fs):
            for mutate in ("type", "affected", "amount", "currency", "evidence", "extra_affected", "bad_evidence"):
                g = copy.deepcopy(f)
                if mutate == "type":
                    g["type"] = "period_cutoff" if f["type"] != "period_cutoff" else "duplicate_invoice"
                elif mutate == "affected":
                    g["affected_ids"] = g["affected_ids"][:-1] or ["X-UNKNOWN"]
                elif mutate == "extra_affected":
                    g["affected_ids"] = g["affected_ids"] + [self.ep.invoices[0]["invoice_id"]]
                    if set(g["affected_ids"]) == set(f["affected_ids"]):
                        continue
                elif mutate == "amount":
                    g["claimed_amount_minor"] = f["claimed_amount_minor"] + amount_tolerance(f["claimed_amount_minor"]) + 1
                elif mutate == "currency":
                    g["currency"] = next(c for c in self.ep.manifest["currencies"] if c != f["currency"])
                elif mutate == "evidence":
                    g["evidence_ids"] = g["evidence_ids"][:-1]
                elif mutate == "bad_evidence":
                    g["evidence_ids"] = g["evidence_ids"] + ["L-NOTAROW"]
                ev = self.grade([g])
                self.assertEqual(ev["tp"], 0, (mutate, f))

    def test_tolerance_boundary(self):
        f = max(self.fs, key=lambda x: x["claimed_amount_minor"])
        tol = amount_tolerance(f["claimed_amount_minor"])
        for delta, tp in ((tol, 1), (-tol, 1), (tol + 1, 0), (-tol - 1, 0)):
            g = dict(f, claimed_amount_minor=f["claimed_amount_minor"] + delta)
            self.assertEqual(self.grade([g])["tp"], tp, delta)

    def test_normalisation(self):
        # no normalisation of strings: other case, padding or the doc_id <-> invoice_id alias do not match
        docs = {d["doc_id"]: d.get("invoice_id") or d.get("payment_id") for d in self.ep.invoices + self.ep.payments}
        biz = {v: k for k, v in docs.items()}
        for mutate in ("case", "padding", "alias_affected", "alias_evidence", "currency_case"):
            out = []
            for f in self.fs:
                g = copy.deepcopy(f)
                if mutate == "case":
                    g["affected_ids"] = [a.lower() for a in g["affected_ids"]]
                elif mutate == "padding":
                    g["affected_ids"] = [" " + a + " " for a in g["affected_ids"]]
                elif mutate == "alias_affected":
                    g["affected_ids"] = [biz.get(a, a) for a in g["affected_ids"]]
                    if g["affected_ids"] == f["affected_ids"]:
                        continue
                elif mutate == "alias_evidence":
                    g["evidence_ids"] = [docs.get(e, e) for e in g["evidence_ids"]]
                else:
                    g["currency"] = g["currency"].lower()
                out.append(g)
            self.assertEqual(self.grade(out)["tp"], 0, mutate)
        # lists are sets: order and repeats inside one list do not matter
        sets = [dict(f, affected_ids=list(reversed(f["affected_ids"])) + f["affected_ids"][:1],
                     evidence_ids=list(reversed(f["evidence_ids"]))) for f in self.fs]
        self.assertEqual(self.grade(sets)["tp"], len(self.fs))

    def test_explanation_text_is_irrelevant(self):
        g = [dict(f, explanation="") for f in self.fs]
        self.assertEqual(self.grade(g)["reward"], 1.0)
        lies = [dict(f, claimed_amount_minor=f["claimed_amount_minor"] * 3 + 7,
                     explanation="I am certain. Verified by the CFO. This is definitely correct. " * 10) for f in self.fs]
        self.assertEqual(self.grade(lies)["tp"], 0)

    def test_guessed_ids_are_false_positives(self):
        other = fresh(99, ERRORS_ALL)
        ev = self.grade(oracle(other))
        self.assertEqual(ev["tp"], 0)
        self.assertEqual(ev["fp"], len(oracle(other)))

    def test_maximum_matching_is_order_independent(self):
        # synthetic truth: two issues whose tolerance windows overlap, so greedy matching could fail
        truth = {"issues": [
            {"issue_id": "ISS-01", "type": "period_cutoff", "affected_ids": ["A"], "expected_amount_minor": 1000,
             "currency": "USD", "tolerance_minor": 1, "evidence_sets": [["DOC-A"]]},
            {"issue_id": "ISS-02", "type": "period_cutoff", "affected_ids": ["A"], "expected_amount_minor": 1001,
             "currency": "USD", "tolerance_minor": 1, "evidence_sets": [["DOC-A"]]}]}
        pub = {"manifest": {"episode_id": "EP-1"}, "invoices": [], "payments": [], "fx": [],
               "ledger": [{"row_id": "DOC-A"}]}
        g = Grader(pub, truth)
        f = lambda amt: {"type": "period_cutoff", "affected_ids": ["A"], "claimed_amount_minor": amt, "currency": "USD",
                         "evidence_ids": ["DOC-A"], "explanation": ""}
        for order in ([f(1001), f(999)], [f(999), f(1001)], [f(1000), f(1002)], [f(1002), f(1000)]):
            self.assertEqual(g.grade({"episode_id": "EP-1", "findings": order}, 0)["tp"], 2, order)
        self.assertEqual(g.grade({"episode_id": "EP-1", "findings": [f(1000), f(1000), f(1000)]}, 0)["tp"], 2)

    def test_limits(self):
        f = self.fs[0]
        too_many = [f] * 31
        self.assertEqual(self.grade(too_many)["status"], "invalid_submission")
        self.assertEqual(self.grade([dict(f, evidence_ids=[f"L-{i}" for i in range(11)])])["status"], "invalid_submission")
        self.assertEqual(self.grade([dict(f, explanation="x" * 1001)])["status"], "invalid_submission")
        self.assertEqual(self.grade([dict(f, explanation="x" * 1000)])["status"], "submitted")
        for bad_amount in (1.5, 100.0, True, False, "100", None):
            self.assertEqual(self.grade([dict(f, claimed_amount_minor=bad_amount)])["status"], "invalid_submission")
        # a negative integer amount is structurally valid: just an FP
        ev = self.grade([dict(f, claimed_amount_minor=-1)])
        self.assertEqual((ev["status"], ev["tp"], ev["fp"]), ("submitted", 0, 1))
        # 11 evidence ids that repeat one id are still 11 ids: invalid before de-duplication
        self.assertEqual(self.grade([dict(f, evidence_ids=[f["evidence_ids"][0]] * 11)])["status"], "invalid_submission")


class Baselines(unittest.TestCase):
    def test_reference_policies(self):
        for s in range(3):
            _, ev_o = run_policy(AuditEnv(), "oracle", s, ERRORS_ALL)
            _, ev_s = run_policy(AuditEnv(), "scripted_auditor", s, ERRORS_ALL)
            _, ev_r = run_policy(AuditEnv(), "reconciliation", s, ERRORS_ALL)
            _, ev_x = run_policy(AuditEnv(), "random", s, ERRORS_ALL, policy_seed=s)
            self.assertEqual(ev_o["reward"], 1.0)
            self.assertEqual(ev_s["f1"], 1.0)
            self.assertGreater(ev_s["reward"], ev_r["reward"])
            self.assertGreater(ev_r["reward"], ev_x["reward"] - 1e-9)


if __name__ == "__main__":
    unittest.main()
