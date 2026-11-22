"""ISO-8601 helpers. All timestamps carry an explicit UTC offset; offsets are fixed (no DST)."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")


def tz(offset_min: int) -> timezone:
    return timezone(timedelta(minutes=offset_min))


def offset_str(offset_min: int) -> str:
    sign = "+" if offset_min >= 0 else "-"
    m = abs(offset_min)
    return f"{sign}{m // 60:02d}:{m % 60:02d}"


def parse_offset(s: str) -> int:
    m = _OFFSET_RE.match(s)
    if not m:
        raise ValueError(f"bad UTC offset {s!r}")
    minutes = int(m.group(2)) * 60 + int(m.group(3))
    return minutes if m.group(1) == "+" else -minutes


def fmt(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("naive datetime")
    return dt.replace(microsecond=0).isoformat()


def render(instant_utc: datetime, offset_min: int) -> str:
    return fmt(instant_utc.astimezone(tz(offset_min)))


def parse(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp without offset: {s!r}")
    return dt


def local_date(instant: datetime, offset_min: int) -> date:
    return instant.astimezone(tz(offset_min)).date()


def wall_date(ts: str) -> date:
    """Calendar date as written in the timestamp (in its own offset)."""
    return parse(ts).date()


def is_business_day(d: date) -> bool:
    return d.weekday() < 5


def prev_business_day(d: date) -> date:
    d = d - timedelta(days=1)
    while not is_business_day(d):
        d -= timedelta(days=1)
    return d


def next_business_day(d: date) -> date:
    d = d + timedelta(days=1)
    while not is_business_day(d):
        d += timedelta(days=1)
    return d


def rate_date_for(d: date) -> date:
    """Policy: latest published (business) day on or before ``d``."""
    while not is_business_day(d):
        d -= timedelta(days=1)
    return d


def utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc)
