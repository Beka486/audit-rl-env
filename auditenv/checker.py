"""Reference reconciliation over PUBLIC records only.

This is the published definition of every discrepancy type. It never reads truth or
provenance. The generator uses it as a self-check (its output must equal truth on every
episode and must be empty on the reconstructed clean books), and the data validator
re-runs it on exported episodes. ``anomalies`` lists anything that is not a declared
discrepancy type (unbalanced journals, dangling references, ...); a correct episode has none.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

from . import money
from .policy_text import ACCOUNTS
from .timeutil import local_date, parse, parse_offset, rate_date_for

BANK, AR, FXC, UNAPPLIED = ACCOUNTS["bank"], ACCOUNTS["ar"], ACCOUNTS["fx_clearing"], ACCOUNTS["unapplied"]
REVENUE = {ACCOUNTS["revenue_services"], ACCOUNTS["revenue_subscriptions"], ACCOUNTS["revenue_goods"]}
KNOWN_ACCOUNTS = set(ACCOUNTS.values())


def line_signature(inv: Dict[str, Any]) -> Tuple:
    lines = tuple(sorted((li["description"], li["quantity"], li["unit_price_minor"], li["amount_minor"],
                          li["revenue_account"]) for li in inv["line_items"]))
    return (inv["entity_id"], inv["currency"], inv["total_minor"], lines)


def reconcile(pub: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    m = pub["manifest"]
    offset = parse_offset(m["timezone"])
    start, end = parse(m["period_start"]), parse(m["period_end"])
    currencies = set(m["currencies"])

    def inside(ts: str) -> bool:
        return start <= parse(ts) <= end

    invoices = {d["invoice_id"]: d for d in pub["invoices"]}
    payments = {d["payment_id"]: d for d in pub["payments"]}
    fx = {(r["date"], r["source_currency"], r["target_currency"]): r["rate"] for r in pub["fx"]}
    anomalies: List[str] = []
    issues: List[Dict[str, Any]] = []

    if len(invoices) != len(pub["invoices"]) or len(payments) != len(pub["payments"]):
        anomalies.append("duplicate invoice_id or payment_id values")

    inv_j: Dict[str, List[dict]] = defaultdict(list)
    pay_j: Dict[str, List[dict]] = defaultdict(list)
    seen_rows = set()
    for r in pub["ledger"]:
        rid = r["row_id"]
        if rid in seen_rows:
            anomalies.append(f"duplicate row_id {rid}")
        seen_rows.add(rid)
        d, c = r["debit_minor"], r["credit_minor"]
        if not (isinstance(d, int) and isinstance(c, int) and d >= 0 and c >= 0 and (d == 0) != (c == 0)):
            anomalies.append(f"row {rid}: exactly one of debit/credit must be positive")
        if r["currency"] not in currencies:
            anomalies.append(f"row {rid}: currency not in manifest")
        if r["account"] not in KNOWN_ACCOUNTS:
            anomalies.append(f"row {rid}: unknown account")
        if not inside(r["timestamp"]):
            anomalies.append(f"row {rid}: timestamp outside the period")
        if r["invoice_id"] is not None and r["invoice_id"] not in invoices:
            anomalies.append(f"row {rid}: dangling invoice_id")
        if r["payment_id"] is not None:
            if r["payment_id"] not in payments:
                anomalies.append(f"row {rid}: dangling payment_id")
            pay_j[r["payment_id"]].append(r)
        elif r["invoice_id"] is not None:
            inv_j[r["invoice_id"]].append(r)
        else:
            anomalies.append(f"row {rid}: belongs to no document")

    for key, rows in list(inv_j.items()) + list(pay_j.items()):
        bal: Dict[str, int] = defaultdict(int)
        for r in rows:
            bal[r["currency"]] += r["debit_minor"] - r["credit_minor"]
        if any(v != 0 for v in bal.values()):
            anomalies.append(f"journal {key} does not balance per currency")
        if len({r["timestamp"] for r in rows}) != 1:
            anomalies.append(f"journal {key} rows carry different timestamps")

    # invoice journals must reproduce the document exactly
    for iid, rows in inv_j.items():
        inv = invoices.get(iid)
        if inv is None:
            continue
        ar = [r for r in rows if r["account"] == AR]
        rev = sorted((r["account"], r["credit_minor"]) for r in rows if r["account"] in REVENUE)
        want_rev = sorted((li["revenue_account"], li["amount_minor"]) for li in inv["line_items"])
        if (len(ar) != 1 or ar[0]["debit_minor"] != inv["total_minor"] or rev != want_rev
                or len(rows) != 1 + len(want_rev) or any(r["currency"] != inv["currency"] for r in rows)
                or any(r["entity_id"] != inv["entity_id"] for r in rows)):
            anomalies.append(f"invoice journal {iid} does not match its document")
        if sum(li["amount_minor"] for li in inv["line_items"]) != inv["total_minor"]:
            anomalies.append(f"invoice {iid}: line items do not sum to total")
        for li in inv["line_items"]:
            if li["quantity"] * li["unit_price_minor"] != li["amount_minor"]:
                anomalies.append(f"invoice {iid}: line amount != quantity * unit price")

    for pid, rows in pay_j.items():
        p = payments.get(pid)
        if p is None:
            continue
        bank = [r for r in rows if r["account"] == BANK]
        if len(bank) != 1 or bank[0]["currency"] != p["currency"]:
            anomalies.append(f"payment journal {pid}: needs one bank row in payment currency")
        elif p["amount_minor"] > 0 and bank[0]["debit_minor"] != p["amount_minor"]:
            anomalies.append(f"payment journal {pid}: bank debit != amount")
        elif p["amount_minor"] < 0 and bank[0]["credit_minor"] != -p["amount_minor"]:
            anomalies.append(f"payment journal {pid}: bank credit != refund amount")
        if p["invoice_refs"] is not None:
            if p["amount_minor"] >= 0 and sum(x["amount_minor"] for x in p["invoice_refs"]) > p["amount_minor"]:
                anomalies.append(f"payment {pid}: invoice_refs exceed the amount")
            for ref in p["invoice_refs"]:
                if ref["invoice_id"] not in invoices:
                    anomalies.append(f"payment {pid}: invoice_refs points to an unknown invoice")

    # ---------------------------------------------------------------- R4 cutoff
    for iid, inv in invoices.items():
        booked, ins = iid in inv_j, inside(inv["issued_at"])
        if booked != ins:
            issues.append({"type": "period_cutoff", "affected_ids": [iid], "amount_minor": abs(inv["total_minor"]),
                           "currency": inv["currency"]})
        elif booked and parse(inv_j[iid][0]["timestamp"]) != parse(inv["issued_at"]):
            anomalies.append(f"invoice {iid}: booked at a different instant than issued_at")
    for pid, p in payments.items():
        booked, ins = pid in pay_j, inside(p["timestamp"])
        if booked != ins:
            issues.append({"type": "period_cutoff", "affected_ids": [pid], "amount_minor": abs(p["amount_minor"]),
                           "currency": p["currency"]})
        elif booked and parse(pay_j[pid][0]["timestamp"]) != parse(p["timestamp"]):
            anomalies.append(f"payment {pid}: booked at a different instant than its timestamp")

    # ------------------------------------------------------------- R1 duplicates
    groups: Dict[Tuple, List[str]] = defaultdict(list)
    for iid in sorted(inv_j):
        if iid in invoices:
            groups[line_signature(invoices[iid])].append(iid)
    for sig, ids in groups.items():
        if len(ids) == 2:
            inv = invoices[ids[0]]
            issues.append({"type": "duplicate_invoice", "affected_ids": sorted(ids), "amount_minor": inv["total_minor"],
                           "currency": inv["currency"]})
        elif len(ids) > 2:
            anomalies.append(f"{len(ids)} identical invoices {ids}")

    # ------------------------------------------- R2 allocation and R3 conversion
    for pid, rows in pay_j.items():
        p = payments.get(pid)
        if p is None or p["invoice_refs"] is None:
            continue
        refs = p["invoice_refs"]
        referenced = {x["invoice_id"] for x in refs}
        shortfall_total = 0
        for ref in refs:
            iid, a = ref["invoice_id"], ref["amount_minor"]
            inv = invoices.get(iid)
            if inv is None:
                continue
            acct = AR if inv["currency"] == p["currency"] else FXC
            applied = sum(r["credit_minor"] - r["debit_minor"] for r in rows
                          if r["account"] == acct and r["invoice_id"] == iid and r["currency"] == p["currency"])
            short = a - applied
            if short > 0:
                shortfall_total += short
                issues.append({"type": "payment_allocation", "affected_ids": sorted([pid, iid]),
                               "amount_minor": short, "currency": p["currency"]})
            elif short < 0:
                anomalies.append(f"payment {pid}: more applied to {iid} than referenced")
            if inv["currency"] != p["currency"] and applied > 0:
                if len(refs) != 1:
                    anomalies.append(f"payment {pid}: cross-currency payments must have one reference")
                booked_ar = sum(r["credit_minor"] - r["debit_minor"] for r in rows
                                if r["account"] == AR and r["invoice_id"] == iid and r["currency"] == inv["currency"])
                booked_fxc = sum(r["debit_minor"] - r["credit_minor"] for r in rows
                                 if r["account"] == FXC and r["invoice_id"] == iid and r["currency"] == inv["currency"])
                if booked_ar != booked_fxc:
                    anomalies.append(f"payment {pid}: clearing and AR disagree")
                rd = rate_date_for(local_date(parse(p["timestamp"]), offset)).isoformat()
                rate = fx.get((rd, p["currency"], inv["currency"]))
                if rate is None:
                    anomalies.append(f"payment {pid}: policy rate {rd} {p['currency']}->{inv['currency']} not published")
                    continue
                good = money.convert(applied, p["currency"], inv["currency"], rate)
                if booked_ar != good:
                    issues.append({"type": "fx_conversion", "affected_ids": sorted([pid, iid]),
                                   "amount_minor": abs(booked_ar - good), "currency": inv["currency"]})
        # credits to invoices the remittance does not mention must be explained by a shortfall
        stray = sum(r["credit_minor"] - r["debit_minor"] for r in rows
                    if r["account"] in (AR, FXC) and r["currency"] == p["currency"] and r["invoice_id"] is not None
                    and r["invoice_id"] not in referenced)
        if stray > shortfall_total:
            anomalies.append(f"payment {pid}: applied to unreferenced invoices beyond its shortfall")
    for pid, rows in pay_j.items():
        p = payments.get(pid)
        if p is not None and p["invoice_refs"] is None:
            if any(r["account"] in (AR, FXC) for r in rows):
                anomalies.append(f"payment {pid}: unidentified receipt applied to an invoice")

    issues.sort(key=lambda x: (x["type"], x["affected_ids"]))
    return issues, anomalies
