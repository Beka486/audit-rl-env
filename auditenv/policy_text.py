"""Published contract texts: chart of accounts, policy.json, manifest.json, task and tool specs."""
from __future__ import annotations

from typing import Any, Dict, List

from . import money
from .version import (ENV_VERSION, GENERATOR_VERSION, MAX_ACTION_BYTES, MAX_ACTION_DEPTH, MAX_EVIDENCE_IDS,
                      MAX_EXPLANATION_CHARS, MAX_FINDINGS, MAX_STEPS, PAGE_LIMIT, SCHEMA_VERSION, STEP_COST)

ACCOUNTS = {
    "bank": "1000-BANK",
    "ar": "1100-AR",
    "fx_clearing": "1950-FX-CLEARING",
    "unapplied": "2400-UNAPPLIED-CASH",
    "revenue_services": "4000-REVENUE-SERVICES",
    "revenue_subscriptions": "4100-REVENUE-SUBSCRIPTIONS",
    "revenue_goods": "4200-REVENUE-GOODS",
}

FINDING_TYPES = {
    "duplicate_invoice": {
        "meaning": "The same invoice (same entity_id, currency, total and identical line_items) is booked twice.",
        "affected_ids": "both invoice_ids",
        "amount": "total_minor of one copy",
        "currency": "invoice currency",
        "evidence": "both invoice documents + one ledger row of each invoice's booking",
    },
    "payment_allocation": {
        "meaning": "A payment's invoice_refs allocation was not (fully) applied to the referenced invoice: "
                   "left as unapplied cash, applied only in part, or applied to a different invoice.",
        "affected_ids": "[payment_id, invoice_id of the referenced invoice]",
        "amount": "referenced allocation minus the part applied to that invoice, in payment currency",
        "currency": "payment currency",
        "evidence": "the payment document + one ledger row of that payment that shows the unapplied, partially "
                    "applied or misapplied amount",
    },
    "fx_conversion": {
        "meaning": "A cross-currency allocation was converted with a rate other than the policy rate (rate of "
                   "another date, or the rate of the reverse currency pair).",
        "affected_ids": "[payment_id, invoice_id]",
        "amount": "|booked invoice-currency amount - correct conversion of the applied payment-currency amount|",
        "currency": "invoice currency",
        "evidence": "the payment document + the fx record of the policy rate date + the ledger row with the booked "
                    "invoice-currency amount; if only part of the payment was applied, also the 1950-FX-CLEARING "
                    "row in payment currency that shows the applied part",
    },
    "period_cutoff": {
        "meaning": "A document is booked in this period although its timestamp is outside the period, or a "
                   "document inside the period is missing from this period's ledger.",
        "affected_ids": "[invoice_id] or [payment_id]",
        "amount": "|total_minor| of the invoice or |amount_minor| of the payment",
        "currency": "document currency",
        "evidence": "the document (+ one of its ledger rows if it is booked in this ledger)",
    },
}


def build_manifest(*, episode_id: str, company_name: str, base_currency: str, currencies: List[str],
                   period_start: str, period_end: str, timezone: str, counts: Dict[str, int]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "env_version": ENV_VERSION,
        "generator_version": GENERATOR_VERSION,
        "episode_id": episode_id,
        "company_name": company_name,
        "base_currency": base_currency,
        "currencies": {c: {"exponent": money.exponent(c)} for c in currencies},
        "period_start": period_start,
        "period_end": period_end,
        "timezone": timezone,
        "counts": counts,
        "rounding": {
            "money": "integers in minor units; minor = major * 10^exponent (see currencies)",
            "mode": "ROUND_HALF_EVEN",
            "fx_conversion": "target_minor = round_half_even(amount_minor * 10^-exp(source) * rate * 10^exp(target))",
            "amount_tolerance": "exact: a claimed amount matches if |claimed - expected| <= tau with tau = 0 minor "
                                "units for every finding type",
        },
        "files": {"ledger": "ledger.csv", "invoices": "invoices.jsonl", "payments": "payments.jsonl",
                  "fx": "fx.csv", "policy": "policy.json"},
        "null_policy": "missing references are explicit null (JSON null; the literal 'null' in CSV)",
    }


def build_policy(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "timezone": manifest["timezone"],
        "period": {
            "start": manifest["period_start"],
            "end": manifest["period_end"],
            "rule": "A document belongs to the period iff its timestamp (issued_at for invoices, timestamp for "
                    "payments), compared as an instant, lies in [start, end]. Ledger rows of the period carry "
                    "timestamps inside the period. Calendar dates are taken in the company timezone.",
        },
        "fx": {
            "rate_definition": "rate = units of target_currency per 1 unit of source_currency (major units)",
            "rate_date": "the calendar date of the payment timestamp in the company timezone; if no rate is "
                         "published for that date use the most recent earlier published date",
            "publication": "rates are published for every Monday-Friday date covered by fx.csv and never on "
                           "Saturday or Sunday",
            "pair": "use the row with source_currency = payment currency and target_currency = invoice currency; "
                    "never invert a rate",
            "record_id": "FX:<date>:<source_currency>:<target_currency>",
        },
        "rounding": {"mode": "ROUND_HALF_EVEN", "applies_to": "only the final converted amount",
                     "formula": manifest["rounding"]["fx_conversion"]},
        "accounts": dict(ACCOUNTS),
        "booking_rules": [
            "Invoice in the period: Dr 1100-AR total_minor; Cr the revenue_account of each line item amount_minor. "
            "Rows carry entity_id and invoice_id, payment_id null.",
            "Payment in the period (amount_minor > 0): Dr 1000-BANK amount_minor; for each invoice_refs allocation "
            "in the invoice currency: Cr 1100-AR with that invoice_id; any amount not covered by invoice_refs: "
            "Cr 2400-UNAPPLIED-CASH (invoice_id null).",
            "Allocation in another currency (single reference): Cr 1950-FX-CLEARING allocation (payment currency); "
            "Dr 1950-FX-CLEARING and Cr 1100-AR the converted amount (invoice currency), all with that invoice_id.",
            "Payment with invoice_refs = null (unidentified receipt): Dr 1000-BANK, Cr 2400-UNAPPLIED-CASH; "
            "entity_id null.",
            "Refund (amount_minor < 0, negative allocation): Cr 1000-BANK |amount|, Dr 1100-AR |amount|.",
            "Every booked journal balances per currency; rows of one payment share payment_id.",
        ],
        "finding_types": FINDING_TYPES,
        "matching": {
            "one_to_one": True,
            "true_positive_requires": ["same type (exact string)", "set(affected_ids) equals the canonical set",
                                       "|claimed_amount_minor - expected| <= tau", "same currency (exact string)",
                                       "set(evidence_ids) contains at least one complete acceptable evidence set",
                                       "every evidence id is an existing public record id"],
            "normalisation": "none: ids, type and currency are compared as exact strings (case and whitespace "
                             "matter, no aliases); affected_ids and evidence_ids are sets (order and repeats inside "
                             "one list do not matter); limits are checked on the raw lists before that",
            "record_ids": "row_id, doc_id, invoice_id, payment_id, FX:<date>:<source_currency>:<target_currency>",
            "canonical_affected_ids": {k: v["affected_ids"] for k, v in FINDING_TYPES.items()},
            "amount_tolerance": manifest["rounding"]["amount_tolerance"],
            "tau_minor": 0,
            "duplicates": "repeated findings are separate findings; once the issue is matched they are false positives",
            "tie_break": "maximum-cardinality matching; among maximum matchings the lexicographically smallest "
                         "sequence of (reference issue id, finding index from 0) pairs, reference issues ordered by "
                         "their ids in Unicode code-point order",
            "invalid_structure": "wrong value type, missing field, unknown field, more than 30 findings, more than 10 "
                                 "evidence ids in one finding (before de-duplication), explanation longer than 1000 "
                                 "code points, or an episode_id of another episode: the whole submission is invalid",
        },
        "reward": {
            "terminal": f"clip(2*F1 - 1 - {STEP_COST}*inspection_steps, -1, 1)",
            "inspection_steps": "all step calls before submit, including erroneous ones; submit counts toward the "
                                "100-step limit but not toward inspection_steps (all steps if the episode is truncated)",
            "f1": "2TP/(2TP+FP+FN); 1 if there are no issues and no findings; 0 if there are no issues but findings",
            "invalid_submission": -1,
            "exploration_reward": 0,
        },
        "limits": {"max_steps": MAX_STEPS, "page_limit": PAGE_LIMIT, "max_findings": MAX_FINDINGS,
                   "max_evidence_ids": MAX_EVIDENCE_IDS, "max_explanation_chars": MAX_EXPLANATION_CHARS,
                   "time_limit_s": 60, "max_action_bytes": MAX_ACTION_BYTES, "max_action_depth": MAX_ACTION_DEPTH},
    }


TOOLS = [
    {"action": "list_records", "args": {"table": "ledger|invoices|payments|fx", "cursor": "string|null (from next_cursor)",
                                         "limit": f"int 1..{PAGE_LIMIT} (default {PAGE_LIMIT})"},
     "returns": "records (invoices without line_items, with line_item_count), next_cursor, total"},
    {"action": "get_record", "args": {"table": "ledger|invoices|payments|fx",
                                      "id": "row_id | invoice_id | payment_id | FX:<date>:<SRC>:<TGT>"},
     "returns": "one record"},
    {"action": "get_document", "args": {"doc_id": "doc_id of an invoice or payment"},
     "returns": "the full document (invoices include line_items)"},
    {"action": "calculator", "args": {"expression": "arithmetic on decimal numbers: + - * / ( )"},
     "returns": "decimal string result"},
    {"action": "submit", "args": {"findings": "{episode_id, findings: [{type, affected_ids, claimed_amount_minor, "
                                              "currency, evidence_ids, explanation}]}"},
     "returns": "ends the episode"},
]


def task_text(manifest: Dict[str, Any]) -> str:
    return (
        f"You are auditing the accounts-receivable books of {manifest['company_name']} for the period "
        f"{manifest['period_start']} .. {manifest['period_end']} (company timezone {manifest['timezone']}, "
        f"base currency {manifest['base_currency']}). Records are read-only and reachable only through the tools. "
        "Find booking discrepancies of the types listed in policy.finding_types (there may be none) and submit them "
        "with supporting evidence ids. Amounts are integer minor units; never compare different currencies without "
        f"conversion. You have {MAX_STEPS} steps including submit; every step before submit costs a little reward."
    )
