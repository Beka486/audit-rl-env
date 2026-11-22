"""Generator: configurable error types, interactions, clean companies, distractors, formats, self-consistency."""
from __future__ import annotations

import csv
import io
import re
import unittest
from collections import Counter
from decimal import Decimal

from auditenv.checker import reconcile
from auditenv.config import ConfigError, normalize_config
from auditenv.episode import Episode
from auditenv.generator import generate_episode
from auditenv.rng import Rng
from auditenv.schema_lite import check
from tests.helpers import CLEAN, ZERO_ERRORS, ZERO_INTER, only

TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
TYPES = ["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"]


def truth_set(ep):
    return sorted((i["type"], tuple(i["affected_ids"]), i["expected_amount_minor"], i["currency"]) for i in ep.truth["issues"])


def found_set(pub):
    found, anomalies = reconcile(pub)
    return sorted((i["type"], tuple(i["affected_ids"]), i["amount_minor"], i["currency"]) for i in found), anomalies


class ErrorTypes(unittest.TestCase):
    def test_each_type_independently_configurable(self):
        for t in TYPES:
            for n in (1, 3):
                for seed in range(3):
                    ep = generate_episode(seed, only(t, n))
                    c = Counter(i["type"] for i in ep.truth["issues"])
                    self.assertEqual(dict(c), {t: n}, (t, n, seed))

    def test_interaction_scenarios(self):
        for seed in range(6):
            ep = generate_episode(seed, {"errors": dict(ZERO_ERRORS), "interactions": {"fx_partial_boundary": 1, "duplicate_misapplied": 0}})
            types = sorted(i["type"] for i in ep.truth["issues"])
            self.assertEqual(types, ["fx_conversion", "payment_allocation"])
            self.assertEqual(len({i["interaction_id"] for i in ep.truth["issues"]}), 1)
            fx = next(i for i in ep.truth["issues"] if i["type"] == "fx_conversion")
            al = next(i for i in ep.truth["issues"] if i["type"] == "payment_allocation")
            self.assertEqual(fx["affected_ids"], al["affected_ids"])  # same payment and invoice
            self.assertNotEqual(fx["currency"], al["currency"])        # payment vs invoice currency
            ep = generate_episode(seed, {"errors": dict(ZERO_ERRORS), "interactions": {"fx_partial_boundary": 0, "duplicate_misapplied": 1}})
            types = sorted(i["type"] for i in ep.truth["issues"])
            self.assertEqual(types, ["duplicate_invoice", "payment_allocation"])
            dup = next(i for i in ep.truth["issues"] if i["type"] == "duplicate_invoice")
            al = next(i for i in ep.truth["issues"] if i["type"] == "payment_allocation")
            self.assertTrue(set(dup["affected_ids"]) & set(al["affected_ids"]))

    def test_clean_companies_have_distractors_but_no_issues(self):
        for seed in range(8):
            ep = generate_episode(seed, CLEAN)
            self.assertEqual(ep.truth["issues"], [])
            kinds = {d["kind"] for d in ep.provenance["distractors"]}
            self.assertTrue({"split_payments", "boundary_documents", "cross_currency_payments", "recurring_customers"} <= kinds, kinds)
            refunds = [p for p in ep.payments if p["amount_minor"] < 0]
            unidentified = [p for p in ep.payments if p["invoice_refs"] is None]
            self.assertTrue(unidentified)
            found, anomalies = found_set(ep.public())
            self.assertEqual((found, anomalies), ([], []))
            del refunds

    def test_issue_count_and_sizes(self):
        for seed in range(25):
            ep = generate_episode(seed, {})
            self.assertLessEqual(len(ep.truth["issues"]), 12)
            self.assertTrue(200 <= len(ep.ledger) <= 2000, len(ep.ledger))
            self.assertTrue(20 <= ep.manifest["counts"]["documents"] <= 200)
            self.assertEqual(ep.manifest["counts"]["ledger_rows"], len(ep.ledger))
            self.assertEqual(ep.manifest["counts"]["invoices"], len(ep.invoices))
            self.assertEqual(ep.manifest["counts"]["payments"], len(ep.payments))


class Consistency(unittest.TestCase):
    """The public records reproduce truth exactly; the reconstructed clean books have zero discrepancies."""

    def check_episode(self, ep):
        got, anomalies = found_set(ep.public())
        self.assertEqual(anomalies, [])
        self.assertEqual(got, truth_set(ep))
        got_clean, anomalies_clean = found_set(ep.clean_public())
        self.assertEqual((got_clean, anomalies_clean), ([], []))

    def test_default_config_many_seeds(self):
        for seed in range(60):
            self.check_episode(generate_episode(seed, {}))

    def test_reverse_pair_rate_never_books_a_zero_amount(self):
        """Measured failure (validator sweep): KWD 169.004 paid on a JPY invoice, reverse-pair rate
        0.0017678258 -> 0 JPY, a ledger row with neither debit nor credit; generation failed for that
        seed. Fix: an FX variant is offered only if its booking is >= 1 minor unit, and a failed
        self-check retries the build instead of failing. Re-check: the seed and a JPY/KWD sweep."""
        cfg = {"config_version": "1.0", "name": "D13-fx-reverse-pair",  # configs/splits/judge_matrix.json
               "errors": {"duplicate_invoice": 0, "payment_allocation": 0, "fx_conversion": 1, "period_cutoff": 0},
               "interactions": {"fx_partial_boundary": 0, "duplicate_misapplied": 0},
               "variants": {"fx_conversion": ["reverse_pair_rate"]}}
        ep = generate_episode(913001, cfg, secret="sweep-salt")
        self.check_episode(ep)
        self.assertEqual([i["variant"] for i in ep.truth["issues"]], ["reverse_pair_rate"])
        cfg = {"errors": {"duplicate_invoice": 0, "payment_allocation": 0, "fx_conversion": 2, "period_cutoff": 0},
               "interactions": {"fx_partial_boundary": 0, "duplicate_misapplied": 0},
               "variants": {"fx_conversion": ["reverse_pair_rate"]},
               "currencies": {"base": None, "foreign_count": [2, 3], "pool": ["KZT", "JPY", "KWD", "USD"]}}
        # 51..227 failed before the fix (6 of 300 seeds of this config)
        for seed in [51, 69, 88, 124, 146, 227] + list(range(20)):
            ep = generate_episode(seed, cfg, secret="sweep-salt")
            self.check_episode(ep)
            for row in ep.ledger:
                self.assertTrue((row["debit_minor"] > 0) != (row["credit_minor"] > 0), row)

    def test_random_configs_fuzz(self):
        rng = Rng.from_key("fuzz")
        for k in range(40):
            errors = {t: rng.randint(0, 3) for t in TYPES}
            inter = {"fx_partial_boundary": rng.randint(0, 1), "duplicate_misapplied": rng.randint(0, 1)}
            while sum(errors.values()) + 2 * sum(inter.values()) > 12:
                t = rng.choice(TYPES)
                errors[t] = max(0, errors[t] - 1)
            cfg = {"errors": errors, "interactions": inter,
                   "size": {"customers": [3 + rng.randint(0, 5), 10 + rng.randint(0, 10)], "invoices": [20 + rng.randint(0, 30), 60 + rng.randint(0, 40)],
                            "prior_invoices": [0, rng.randint(0, 8)], "min_ledger_rows": 200, "max_ledger_rows": 2000, "max_documents": 200},
                   "timezone": rng.choice([None, "+05:00", "-05:00", "+05:30", "+09:00", "+00:00", "-03:00"]),
                   "period": rng.choice([None, "2026-02", "2025-12", "2026-11", "2024-02"]),
                   "currencies": {"base": None, "foreign_count": [1, rng.randint(1, 2)],
                                  "pool": rng.sample(["KZT", "USD", "EUR", "JPY", "KWD", "GBP", "CNY", "TRY"], rng.randint(3, 6))}}
            ep = generate_episode(1000 + k, cfg)
            self.check_episode(ep)
            c = Counter(i["type"] for i in ep.truth["issues"])
            want = Counter()
            for t, n in errors.items():
                want[t] += n
            want["fx_conversion"] += inter["fx_partial_boundary"]
            want["payment_allocation"] += inter["fx_partial_boundary"] + inter["duplicate_misapplied"]
            want["duplicate_invoice"] += inter["duplicate_misapplied"]
            self.assertEqual(+c, +want, cfg)


class Formats(unittest.TestCase):
    def test_value_formats_and_schemas(self):
        for seed in range(6):
            ep = generate_episode(seed, {})
            exps = {c: v["exponent"] for c, v in ep.manifest["currencies"].items()}
            self.assertEqual(check(ep.manifest, "manifest"), [])
            self.assertEqual(check(ep.truth, "truth"), [])
            self.assertEqual(check(ep.provenance, "provenance"), [])
            for r in ep.ledger:
                self.assertEqual(check(r, "ledger_row"), [])
                self.assertTrue(TS.match(r["timestamp"]))
                self.assertIn(r["currency"], exps)
            for d in ep.invoices:
                self.assertEqual(check(d, "invoice"), [])
            for d in ep.payments:
                self.assertEqual(check(d, "payment"), [])
            for r in ep.fx:
                self.assertEqual(check(r, "fx_row"), [])
                Decimal(r["rate"])
            files = ep.public_files()
            rows = list(csv.DictReader(io.StringIO(files["ledger.csv"])))
            self.assertEqual(list(rows[0].keys()), ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                                                    "entity_id", "invoice_id", "payment_id"])
            self.assertTrue(any(r["invoice_id"] == "null" for r in rows))  # explicit null
            self.assertNotIn(",,", files["ledger.csv"])
            self.assertEqual(list(csv.DictReader(io.StringIO(files["fx.csv"]))).__len__(), len(ep.fx))

    def test_variation_across_seeds(self):
        bases, tzs, styles, periods = set(), set(), set(), set()
        for seed in range(30):
            ep = generate_episode(seed, {})
            bases.add(ep.manifest["base_currency"])
            tzs.add(ep.manifest["timezone"])
            styles.add(re.sub(r"\d", "9", ep.invoices[0]["invoice_id"])[:4])
            periods.add(ep.manifest["period_start"][:7])
        self.assertGreater(len(bases), 4)
        self.assertGreater(len(tzs), 4)
        self.assertGreater(len(styles), 2)
        self.assertGreater(len(periods), 5)

    def test_roundtrip_through_files(self):
        import tempfile
        ep = generate_episode(7, {})
        with tempfile.TemporaryDirectory() as td:
            ep.write(td)
            back = Episode.load(td)
        self.assertEqual(back.public_sha256(), ep.public_sha256())
        self.assertEqual(back.full_sha256(), ep.full_sha256())


class Config(unittest.TestCase):
    def test_rejects_bad_configs(self):
        for bad in ({"nope": 1}, {"errors": {"fraud": 1}}, {"errors": {"duplicate_invoice": -1}},
                    {"errors": {"duplicate_invoice": [3, 1]}}, {"errors": {t: 4 for t in TYPES}},
                    {"timezone": "Asia/Almaty"}, {"period": "2026-13"}, {"config_version": "0.1"},
                    {"currencies": {"pool": ["USD"]}}, {"transform": {"seed": "x"}}, "string"):
            with self.assertRaises(ConfigError, msg=bad):
                normalize_config(bad)

    def test_config_versions(self):
        self.assertEqual(normalize_config({})["config_version"], "1.0")
        import json, pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        for p in (root / "configs" / "splits").glob("*.json"):
            doc = json.loads(p.read_text())
            self.assertEqual(check(doc, "split"), [], p)
            for e in doc["episodes"]:
                normalize_config(e["config"])

    def test_secret_salt_changes_data(self):
        a = generate_episode(5, {}, secret="")
        b = generate_episode(5, {}, secret="jury-secret")
        self.assertNotEqual(a.public_sha256(), b.public_sha256())
        self.assertNotEqual(a.episode_id, b.episode_id)
        self.assertIsNone(b.provenance["seed"])
        self.assertNotIn("seed", b.manifest)


if __name__ == "__main__":
    unittest.main()
