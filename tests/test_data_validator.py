"""Tests for the independent data validator (auditenv.validate_data).

The generator is used only as a black box to produce episodes; every property is checked by
``auditenv.validate_data``, which re-implements the specification on its own.

Run: python3.11 -m unittest tests.test_data_validator -v
"""
from __future__ import annotations

import ast
import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from auditenv.generator import generate_episode
from auditenv.validate_data import (CHECKS, convert_minor, find_episode_dirs, main, tolerance_for,
                                    validate_episode_dir, validate_tree, weekday_rate_date)

ZERO_ERRORS = {"duplicate_invoice": 0, "payment_allocation": 0, "fx_conversion": 0, "period_cutoff": 0}
ZERO_INTERACTIONS = {"fx_partial_boundary": 0, "duplicate_misapplied": 0}
REQUIRED_CHECKS = {"format", "clean_books_balanced", "clean_books_follow_policy", "declared_discrepancies_present",
                   "truth_complete", "evidence_sufficient", "truth_wellformed", "no_private_leak"}


def _single(kind: str) -> dict:
    errors = dict(ZERO_ERRORS)
    errors[kind] = 3
    return {"errors": errors, "interactions": dict(ZERO_INTERACTIONS)}


# (folder name, config, seeds) -> 30 episodes in total
PLAN = [
    ("default", {}, range(1, 9)),
    ("clean", {"errors": dict(ZERO_ERRORS), "interactions": dict(ZERO_INTERACTIONS)}, range(11, 15)),
    ("interactions", {"interactions": {"fx_partial_boundary": 1, "duplicate_misapplied": 1}}, range(21, 27)),
    ("duplicate_invoice", _single("duplicate_invoice"), range(31, 34)),
    ("payment_allocation", _single("payment_allocation"), range(41, 44)),
    ("fx_conversion", _single("fx_conversion"), range(51, 54)),
    ("period_cutoff", _single("period_cutoff"), range(61, 64)),
]
N_EPISODES = sum(len(seeds) for _, _, seeds in PLAN)


# --------------------------------------------------------------------------- file helpers
def read_ledger(ep_dir: Path):
    rows = list(csv.reader(io.StringIO((ep_dir / "public" / "ledger.csv").read_text(encoding="utf-8"))))
    return rows[0], rows[1:]


def write_ledger(ep_dir: Path, header, rows) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    (ep_dir / "public" / "ledger.csv").write_text(buf.getvalue(), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


# --------------------------------------------------------------------------- helpers under test
class HelperTest(unittest.TestCase):
    def test_conversion_rounds_half_even_only_at_the_end(self):
        self.assertEqual(convert_minor(125, 2, 1, "1"), 12)        # 12.5 -> 12
        self.assertEqual(convert_minor(135, 2, 1, "1"), 14)        # 13.5 -> 14
        self.assertEqual(convert_minor(100, 2, 0, "150.5"), 150)   # 1.00 USD -> 150.5 JPY -> 150
        self.assertEqual(convert_minor(100, 2, 0, "151.5"), 152)
        self.assertEqual(convert_minor(1, 0, 3, "0.0032505"), 3)   # JPY -> KWD (exponent 3)
        self.assertEqual(convert_minor(23732, 2, 2, "644.69"), 15299783)

    def test_weekday_rate_date(self):
        self.assertEqual(weekday_rate_date(date(2026, 11, 21)), date(2026, 11, 20))  # Saturday -> Friday
        self.assertEqual(weekday_rate_date(date(2026, 11, 22)), date(2026, 11, 20))  # Sunday -> Friday
        self.assertEqual(weekday_rate_date(date(2026, 11, 23)), date(2026, 11, 23))  # Monday stays

    def test_tolerance(self):
        for amount in (1, 999, 476780, -5000):  # published tau = 0 for every generated issue
            self.assertEqual(tolerance_for(amount), 0)

    def test_validator_is_independent_of_generator_and_checker(self):
        src = (Path(__file__).resolve().parent.parent / "auditenv" / "validate_data.py").read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add("." * node.level + (node.module or ""))
        self.assertFalse([m for m in imported if m.startswith(".") or m.startswith("auditenv")], imported)


# --------------------------------------------------------------------------- generated episodes
class GeneratedEpisodesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="auditenv-validate-")
        cls.root = Path(cls._tmp.name) / "episodes"
        for name, cfg, seeds in PLAN:
            for seed in seeds:
                generate_episode(seed, cfg).write(cls.root / name / f"seed{seed}")
        cls.summary = validate_tree(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_tree_discovery(self):
        self.assertEqual(len(find_episode_dirs(self.root)), N_EPISODES)
        self.assertEqual(self.summary["episodes"], N_EPISODES)

    def test_result_shape_and_required_checks(self):
        for r in self.summary["results"]:
            self.assertTrue({"episode_id", "ok", "checks", "errors"} <= set(r))
            self.assertTrue(REQUIRED_CHECKS <= set(r["checks"]), set(r["checks"]))
            self.assertEqual(set(r["checks"]), set(CHECKS))

    def test_every_episode_passes_every_check(self):
        for r in self.summary["results"]:
            with self.subTest(episode=r["path"]):
                failed = [c for c, ok in r["checks"].items() if not ok]
                self.assertEqual(failed, [], r["errors"][:10])
                self.assertTrue(r["ok"])
                self.assertEqual(r["errors"], [])
        self.assertEqual(self.summary["ok"], N_EPISODES)

    def test_plan_covers_all_issue_types_variants_and_interactions(self):
        types, variants, interactions = set(), set(), 0
        for d in find_episode_dirs(self.root):
            truth = read_json(d / "private" / "truth.json")
            for issue in truth["issues"]:
                types.add(issue["type"])
                variants.add((issue["type"], issue["variant"]))
                interactions += issue["interaction_id"] is not None
        self.assertEqual(types, {"duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"})
        for v in ("missing", "partial", "misapplied"):
            self.assertIn(("payment_allocation", v), variants)
        self.assertTrue(any(t == "period_cutoff" and v.startswith("late") for t, v in variants))
        self.assertTrue(any(t == "period_cutoff" and v.startswith("early") for t, v in variants))
        self.assertGreater(interactions, 0)

    def test_clean_config_declares_nothing(self):
        for d in find_episode_dirs(self.root / "clean"):
            self.assertEqual(read_json(d / "private" / "truth.json")["issues"], [])

    def test_cli_success(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn(f"{N_EPISODES}/{N_EPISODES} episodes valid", out.getvalue())


# --------------------------------------------------------------------------- corruption must be caught
class CorruptionDetectionTest(unittest.TestCase):
    """Each tampering must turn the relevant check red (and the episode not ok)."""

    CONFIG = {"errors": {"duplicate_invoice": 1, "payment_allocation": 1, "fx_conversion": 1, "period_cutoff": 1},
              "interactions": {"fx_partial_boundary": 1, "duplicate_misapplied": 0}}

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="auditenv-corrupt-")
        cls.tmp = Path(cls._tmp.name)
        cls.base = None
        for seed in range(700, 760):  # deterministic search for a host with every ingredient we need
            d = generate_episode(seed, cls.CONFIG).write(cls.tmp / f"base{seed}")
            prov = read_json(d / "private" / "provenance.json")
            truth = read_json(d / "private" / "truth.json")
            if prov["removed_ledger_rows"] and len(truth["issues"]) >= 4 and cls._untouched_invoice(d) is not None:
                cls.base = d
                break
        assert cls.base is not None, "no suitable base episode found"
        cls._n = 0

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def _untouched_invoice(d: Path):
        """An in-ledger invoice that no issue, evidence set or injected row refers to."""
        truth = read_json(d / "private" / "truth.json")
        prov = read_json(d / "private" / "provenance.json")
        mentioned = set()
        for issue in truth["issues"]:
            mentioned.update(issue["affected_ids"])
            for s in issue["evidence_sets"]:
                mentioned.update(s)
        header, rows = read_ledger(d)
        added = set(prov["added_ledger_rows"])
        invoices = {i["invoice_id"]: i for i in read_jsonl(d / "public" / "invoices.jsonl")}
        for r in rows:
            inv_id, pay_id = r[7], r[8]
            if pay_id == "null" and inv_id in invoices and inv_id not in mentioned \
                    and invoices[inv_id]["doc_id"] not in mentioned:
                journal = [x for x in rows if x[7] == inv_id and x[8] == "null"]
                if not any(x[0] in added or x[0] in mentioned for x in journal):
                    return inv_id
        return None

    def copy(self) -> Path:
        CorruptionDetectionTest._n += 1
        dst = self.tmp / f"case{self._n}"
        shutil.copytree(self.base, dst)
        return dst

    def assertFails(self, result, *checks):
        self.assertFalse(result["ok"])
        for c in checks:
            self.assertFalse(result["checks"][c], f"{c} should fail; errors: {result['errors'][:6]}")
            self.assertTrue(any(e.startswith(c + ":") for e in result["errors"]))

    def test_baseline_is_valid(self):
        r = validate_episode_dir(self.base)
        self.assertTrue(r["ok"], r["errors"][:10])

    def test_tampered_amount_of_a_clean_ledger_row(self):
        d = self.copy()
        inv_id = self._untouched_invoice(d)
        header, rows = read_ledger(d)
        row = next(r for r in rows if r[7] == inv_id and r[8] == "null" and r[2] == "1100-AR")
        row[3] = str(int(row[3]) + 100)
        write_ledger(d, header, rows)
        r = validate_episode_dir(d)
        self.assertFails(r, "clean_books_balanced", "clean_books_follow_policy", "truth_complete")
        self.assertTrue(r["checks"]["format"])

    def test_deleted_truth_issue(self):
        d = self.copy()
        truth = read_json(d / "private" / "truth.json")
        removed = truth["issues"].pop(0)
        write_json(d / "private" / "truth.json", truth)
        r = validate_episode_dir(d)
        self.assertFails(r, "truth_complete")
        self.assertTrue(any(removed["type"] in e and "undeclared" in e for e in r["errors"]))

    def test_changed_expected_amount(self):
        d = self.copy()
        truth = read_json(d / "private" / "truth.json")
        issue = truth["issues"][0]
        issue["expected_amount_minor"] += 7  # tolerance stays the published 0: truth remains well-formed
        write_json(d / "private" / "truth.json", truth)
        r = validate_episode_dir(d)
        self.assertFails(r, "declared_discrepancies_present", "evidence_sufficient", "truth_complete")
        self.assertTrue(r["checks"]["truth_wellformed"])

    def test_tolerance_other_than_published_tau(self):
        d = self.copy()
        truth = read_json(d / "private" / "truth.json")
        truth["issues"][0]["tolerance_minor"] = 5
        write_json(d / "private" / "truth.json", truth)
        self.assertFails(validate_episode_dir(d), "truth_wellformed")

    def test_evidence_set_points_at_wrong_row(self):
        d = self.copy()
        truth = read_json(d / "private" / "truth.json")
        header, rows = read_ledger(d)
        row_ids = {r[0] for r in rows}
        decoy = next(r[0] for r in rows if r[7] == self._untouched_invoice(d) and r[8] == "null")
        issue = next(i for i in truth["issues"] if any(x in row_ids for x in i["evidence_sets"][0]))
        ev = issue["evidence_sets"][0]
        ev[next(k for k, x in enumerate(ev) if x in row_ids)] = decoy
        issue["evidence_sets"][0] = sorted(ev)
        write_json(d / "private" / "truth.json", truth)
        r = validate_episode_dir(d)
        self.assertFails(r, "evidence_sufficient")
        self.assertTrue(r["checks"]["declared_discrepancies_present"])
        self.assertTrue(r["checks"]["truth_complete"])

    def test_evidence_id_that_does_not_exist(self):
        d = self.copy()
        truth = read_json(d / "private" / "truth.json")
        truth["issues"][0]["evidence_sets"][0][0] = "GL-DOES-NOT-EXIST"
        write_json(d / "private" / "truth.json", truth)
        self.assertFails(validate_episode_dir(d), "evidence_sufficient")

    def test_removed_ledger_row_dropped_from_provenance(self):
        d = self.copy()
        prov = read_json(d / "private" / "provenance.json")
        prov["removed_ledger_rows"].pop(0)
        write_json(d / "private" / "provenance.json", prov)
        r = validate_episode_dir(d)
        self.assertFails(r, "clean_books_balanced")
        self.assertTrue(r["checks"]["declared_discrepancies_present"])

    def test_added_document_dropped_from_provenance(self):
        d = self.copy()
        prov = read_json(d / "private" / "provenance.json")
        self.assertTrue(prov["added_documents"])
        prov["added_documents"] = []
        write_json(d / "private" / "provenance.json", prov)
        self.assertFails(validate_episode_dir(d), "clean_books_follow_policy")

    def test_issue_id_in_public_file(self):
        d = self.copy()
        truth = read_json(d / "private" / "truth.json")
        manifest = read_json(d / "public" / "manifest.json")
        manifest["company_name"] += " " + truth["issues"][0]["issue_id"]
        write_json(d / "public" / "manifest.json", manifest)
        r = validate_episode_dir(d)
        self.assertFails(r, "no_private_leak")
        self.assertTrue(r["checks"]["format"])

    def test_private_marker_in_public_file(self):
        d = self.copy()
        inv_id = self._untouched_invoice(d)
        invoices = read_jsonl(d / "public" / "invoices.jsonl")
        inv = next(i for i in invoices if i["invoice_id"] == inv_id)
        inv["line_items"][0]["sku"] = "see truth table"
        write_jsonl(d / "public" / "invoices.jsonl", invoices)
        self.assertFails(validate_episode_dir(d), "no_private_leak")

    def test_used_fx_rate_tampered(self):
        d = self.copy()
        truth = read_json(d / "private" / "truth.json")
        fx_ids = {x for i in truth["issues"] if i["type"] == "fx_conversion" for s in i["evidence_sets"] for x in s
                  if x.startswith("FX:")}
        target = sorted(fx_ids)[0].split(":")[1:]
        text = (d / "public" / "fx.csv").read_text(encoding="utf-8").splitlines()
        out = []
        for line in text:
            parts = line.split(",")
            if parts[:3] == target:
                parts[3] = format(Decimal(parts[3]) * Decimal("1.05"), "f")
            out.append(",".join(parts))
        (d / "public" / "fx.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
        self.assertFails(validate_episode_dir(d), "clean_books_follow_policy", "declared_discrepancies_present")

    def test_weekend_fx_rate(self):
        d = self.copy()
        lines = (d / "public" / "fx.csv").read_text(encoding="utf-8").splitlines()
        first = lines[1].split(",")
        day = date.fromisoformat(first[0])
        saturday = day + timedelta(days=(5 - day.weekday()) % 7)
        lines.append(",".join([saturday.isoformat(), first[1], first[2], first[3]]))
        (d / "public" / "fx.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest = read_json(d / "public" / "manifest.json")
        manifest["counts"]["fx_rates"] += 1  # keep counts right so only the calendar rule is violated
        write_json(d / "public" / "manifest.json", manifest)
        r = validate_episode_dir(d)
        self.assertFails(r, "format")
        self.assertTrue(any("weekend" in e for e in r["errors"]), r["errors"])

    def test_manifest_count_mismatch(self):
        d = self.copy()
        manifest = read_json(d / "public" / "manifest.json")
        manifest["counts"]["payments"] += 1
        write_json(d / "public" / "manifest.json", manifest)
        self.assertFails(validate_episode_dir(d), "format")

    def test_journal_deleted_entirely(self):
        d = self.copy()
        inv_id = self._untouched_invoice(d)
        header, rows = read_ledger(d)
        kept = [r for r in rows if not (r[7] == inv_id and r[8] == "null")]
        write_ledger(d, header, kept)
        manifest = read_json(d / "public" / "manifest.json")
        manifest["counts"]["ledger_rows"] = len(kept)
        write_json(d / "public" / "manifest.json", manifest)
        r = validate_episode_dir(d)
        self.assertFails(r, "truth_complete", "clean_books_follow_policy")
        self.assertTrue(any("period_cutoff" in e and inv_id in e for e in r["errors"]))

    def test_missing_public_file(self):
        d = self.copy()
        (d / "public" / "fx.csv").unlink()
        r = validate_episode_dir(d)
        self.assertFails(r, "format")
        self.assertFalse(any(r["checks"].values()))

    def test_cli_reports_failure(self):
        d = self.copy()
        truth = read_json(d / "private" / "truth.json")
        truth["issues"].pop()
        write_json(d / "private" / "truth.json", truth)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([str(d)])
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out.getvalue())


if __name__ == "__main__":
    unittest.main()
