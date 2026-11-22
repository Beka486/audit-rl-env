"""Authenticated evaluator-log records.

The key is created by the jury-side client and handed only to the evaluator process, so a line
appended to a shared log by anyone else, or by another run, is ignored by ``last_evaluation``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Dict, Iterable, Optional

FIELD = "log_auth"


def _mac(record: Dict[str, Any], seq: int, key: str) -> str:
    body = {k: v for k, v in record.items() if k != FIELD}
    text = json.dumps([seq, body], sort_keys=True, separators=(",", ":"))
    return hmac.new(key.encode("utf-8"), text.encode("ascii"), hashlib.sha256).hexdigest()


def seal(record: Dict[str, Any], seq: int, key: str) -> Dict[str, Any]:
    return {**record, FIELD: {"seq": seq, "mac": _mac(record, seq, key)}}


def latest(lines: Iterable[str], key: str) -> Optional[Dict[str, Any]]:
    """The authentic record with the highest sequence number, without its auth field."""
    best, best_seq = None, -1
    for line in lines:
        try:
            rec = json.loads(line)
            auth = rec[FIELD]
            seq, mac = auth["seq"], auth["mac"]
        except (ValueError, TypeError, KeyError):
            continue
        if type(seq) is int and isinstance(mac, str) and seq > best_seq and hmac.compare_digest(mac, _mac(rec, seq, key)):
            best, best_seq = {k: v for k, v in rec.items() if k != FIELD}, seq
    return best
