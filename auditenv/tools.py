"""Read-only record store behind the agent tools.

Only public records are loaded here; the store has no reference to truth or provenance.
Every returned object is a fresh copy, so callers cannot mutate environment state.
Ids are looked up exactly (case and whitespace matter, no aliases): get_record uses the
table's key (ledger.row_id, invoices.invoice_id, payments.payment_id, fx.fx_id =
FX:<date>:<source_currency>:<target_currency>), get_document uses doc_id.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional

from .version import PAGE_LIMIT

TABLES = ("ledger", "invoices", "payments", "fx")
_CURSOR_RE = re.compile(r"\A(ledger|invoices|payments|fx)@([0-9]{1,7})\Z")


class ToolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def fx_id(r: Dict[str, Any]) -> str:
    return f"FX:{r['date']}:{r['source_currency']}:{r['target_currency']}"


class RecordStore:
    def __init__(self, public: Dict[str, Any]) -> None:
        headers = []
        for inv in public["invoices"]:
            h = {k: v for k, v in inv.items() if k != "line_items"}
            h["line_item_count"] = len(inv["line_items"])
            headers.append(h)
        fx_rows = []
        for r in public["fx"]:
            x = dict(r)
            x["fx_id"] = fx_id(r)
            fx_rows.append(x)
        self.tables: Dict[str, List[Dict[str, Any]]] = {
            "ledger": copy.deepcopy(public["ledger"]),
            "invoices": headers,
            "payments": copy.deepcopy(public["payments"]),
            "fx": fx_rows,
        }
        self.keys = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}
        self.index: Dict[str, Dict[str, int]] = {}
        for t, rows in self.tables.items():
            self.index[t] = {r[self.keys[t]]: i for i, r in enumerate(rows)}
        self.documents: Dict[str, Dict[str, Any]] = {}
        for d in list(public["invoices"]) + list(public["payments"]):
            self.documents[d["doc_id"]] = copy.deepcopy(d)

    @staticmethod
    def _table(table: Any) -> str:
        if not isinstance(table, str):
            raise ToolError("invalid_arguments", "table must be a string.")
        if table not in TABLES:
            raise ToolError("unknown_table", "Unknown table. Tables: ledger, invoices, payments, fx.")
        return table

    def list_records(self, table: Any, cursor: Any = None, limit: Any = PAGE_LIMIT) -> Dict[str, Any]:
        t = self._table(table)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= PAGE_LIMIT:
            raise ToolError("invalid_arguments", f"limit must be an integer between 1 and {PAGE_LIMIT}.")
        rows = self.tables[t]
        if cursor is None:
            offset = 0
        else:
            if not isinstance(cursor, str):
                raise ToolError("invalid_cursor", "cursor must be a string returned as next_cursor, or null.")
            m = _CURSOR_RE.match(cursor)
            if not m or m.group(1) != t or int(m.group(2)) > len(rows):
                raise ToolError("invalid_cursor", "cursor is not valid for this table.")
            offset = int(m.group(2))
        page = copy.deepcopy(rows[offset:offset + limit])
        nxt = offset + len(page)
        return {"table": t, "records": page, "next_cursor": f"{t}@{nxt}" if nxt < len(rows) else None,
                "total": len(rows), "offset": offset}

    def get_record(self, table: Any, rid: Any) -> Dict[str, Any]:
        t = self._table(table)
        if not isinstance(rid, str) or not rid or len(rid) > 256:
            raise ToolError("invalid_arguments", "id must be a non-empty string.")
        i = self.index[t].get(rid)
        if i is None:
            raise ToolError("unknown_id", "No record with this id in the table.")
        return {"table": t, "record": copy.deepcopy(self.tables[t][i])}

    def get_document(self, doc_id: Any) -> Dict[str, Any]:
        if not isinstance(doc_id, str) or not doc_id or len(doc_id) > 256:
            raise ToolError("invalid_arguments", "doc_id must be a non-empty string.")
        d = self.documents.get(doc_id)
        if d is None:
            raise ToolError("unknown_id", "No document with this doc_id.")
        return {"document": copy.deepcopy(d)}
