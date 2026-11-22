"""Episode configuration: defaults, merging, validation and sampling.

A config describes a *distribution* of episodes. Integer fields accept either an
int or an inclusive range ``[lo, hi]``; ranges are sampled per episode from the
seed, so new parameter combinations come from new seeds and new configs.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict

from .rng import Rng, derive_hex
from .version import CONFIG_VERSION

ERROR_TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")
INTERACTIONS = ("fx_partial_boundary", "duplicate_misapplied")
# Number of truth issues each interaction scenario contributes.
INTERACTION_ISSUES = {"fx_partial_boundary": 2, "duplicate_misapplied": 2}
DISTRACTORS = (
    "split_payments", "multi_invoice_payments", "refund_pairs", "recurring_customers",
    "unidentified_receipts", "on_account_payments", "boundary_documents",
    "cross_currency_payments", "legit_partial_payments", "boundary_per_cutoff",
)
CURRENCY_POOL = ["KZT", "USD", "EUR", "GBP", "CNY", "AED", "TRY", "INR", "JPY", "KWD", "UZS", "CHF"]
TIMEZONE_POOL = ["+05:00", "+06:00", "+03:00", "+01:00", "+00:00", "-05:00", "+05:30", "+09:00", "+04:00", "-03:00"]
MAX_ISSUES = 12
# Injection variants that a config may restrict (each episode samples uniformly from the allowed list).
VARIANTS = {
    "payment_allocation": ("missing", "partial"),
    "fx_conversion": ("next_business_day", "previous_business_day", "document_wall_clock_date",
                      "invoice_issue_date", "reverse_pair_rate"),
    "period_cutoff": ("early", "late"),
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "name": "default",
    "size": {
        "customers": [6, 12],
        "invoices": [40, 75],
        "prior_invoices": [3, 6],
        "min_ledger_rows": 200,
        "max_ledger_rows": 1200,
        "max_documents": 200,
    },
    "currencies": {"base": None, "foreign_count": [1, 3], "pool": CURRENCY_POOL},
    "timezone": None,
    "period": None,
    "errors": {"duplicate_invoice": [0, 2], "payment_allocation": [0, 2],
               "fx_conversion": [0, 2], "period_cutoff": [0, 2]},
    "interactions": {"fx_partial_boundary": [0, 1], "duplicate_misapplied": [0, 1]},
    "variants": {k: list(v) for k, v in VARIANTS.items()},
    "distractors": {
        "split_payments": [1, 4], "multi_invoice_payments": [1, 3], "refund_pairs": [0, 2],
        "recurring_customers": [1, 2], "unidentified_receipts": [1, 3],
        "on_account_payments": [0, 2], "boundary_documents": [2, 5],
        "cross_currency_payments": [2, 5], "legit_partial_payments": [1, 3],
        # extra boundary documents per period_cutoff error (fix for the blind boundary-guessing attack;
        # 0 reproduces the measured weakness, see DESIGN.md section 6)
        "boundary_per_cutoff": 2,
    },
    "transform": None,
}


class ConfigError(ValueError):
    pass


def _merge(base: Any, over: Any) -> Any:
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge(base[k], v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(over)


def _check_count(name: str, v: Any, lo_bound: int = 0, hi_bound: int = 10_000) -> None:
    if isinstance(v, bool):
        raise ConfigError(f"{name}: must be int or [lo, hi]")
    if isinstance(v, int):
        if not lo_bound <= v <= hi_bound:
            raise ConfigError(f"{name}: out of range [{lo_bound}, {hi_bound}]")
        return
    if (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v)):
        if not lo_bound <= v[0] <= v[1] <= hi_bound:
            raise ConfigError(f"{name}: need {lo_bound} <= lo <= hi <= {hi_bound}")
        return
    raise ConfigError(f"{name}: must be int or [lo, hi]")


def _hi(v: Any) -> int:
    return v if isinstance(v, int) else v[1]


def _lo(v: Any) -> int:
    return v if isinstance(v, int) else v[0]


def normalize_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge a user config over the defaults and validate it. Deterministic, no sampling."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ConfigError("config must be a JSON object")
    unknown = set(config) - set(DEFAULT_CONFIG) - {"episode_dir"}
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    cfg = _merge(DEFAULT_CONFIG, {k: v for k, v in config.items() if k != "episode_dir"})
    if cfg["config_version"] != CONFIG_VERSION:
        raise ConfigError(f"unsupported config_version {cfg['config_version']!r}")
    for section, keys in (("errors", ERROR_TYPES), ("interactions", INTERACTIONS), ("distractors", DISTRACTORS)):
        extra = set(cfg[section]) - set(keys)
        if extra:
            raise ConfigError(f"unknown {section}: {sorted(extra)}")
        for k in keys:
            _check_count(f"{section}.{k}", cfg[section][k], 0, 12 if section != "distractors" else 20)
    var = cfg["variants"]
    if not isinstance(var, dict) or set(var) - set(VARIANTS):
        raise ConfigError(f"variants: keys must be among {sorted(VARIANTS)}")
    for k, allowed in VARIANTS.items():
        v = var.get(k)
        if (not isinstance(v, list) or not v or len(set(v)) != len(v)
                or any(not isinstance(x, str) or x not in allowed for x in v)):
            raise ConfigError(f"variants.{k}: a non-empty list of distinct values from {list(allowed)}")
    size = cfg["size"]
    extra = set(size) - set(DEFAULT_CONFIG["size"])
    if extra:
        raise ConfigError(f"unknown size keys: {sorted(extra)}")
    _check_count("size.customers", size["customers"], 3, 40)
    _check_count("size.invoices", size["invoices"], 15, 160)
    _check_count("size.prior_invoices", size["prior_invoices"], 0, 20)
    for k in ("min_ledger_rows", "max_ledger_rows", "max_documents"):
        _check_count(f"size.{k}", size[k], 1, 5000)
    if size["min_ledger_rows"] > size["max_ledger_rows"]:
        raise ConfigError("size.min_ledger_rows > size.max_ledger_rows")
    cur = cfg["currencies"]
    if set(cur) - {"base", "foreign_count", "pool"}:
        raise ConfigError("unknown currencies keys")
    from .money import EXPONENTS
    pool = cur["pool"]
    if not isinstance(pool, list) or len(pool) < 2 or any(c not in EXPONENTS for c in pool) or len(set(pool)) != len(pool):
        raise ConfigError("currencies.pool must list >= 2 distinct supported currencies")
    if cur["base"] is not None and cur["base"] not in EXPONENTS:
        raise ConfigError("currencies.base not supported")
    _check_count("currencies.foreign_count", cur["foreign_count"], 1, 4)
    if _hi(cur["foreign_count"]) > len(pool) - 1:
        raise ConfigError("currencies.foreign_count larger than pool allows")
    if cfg["timezone"] is not None:
        from .timeutil import parse_offset
        try:
            off = parse_offset(cfg["timezone"])
        except (ValueError, TypeError):
            raise ConfigError("timezone must look like '+05:00'") from None
        if not -12 * 60 <= off <= 14 * 60:
            raise ConfigError("timezone offset out of range")
    if cfg["period"] is not None:
        p = cfg["period"]
        if not (isinstance(p, str) and len(p) == 7 and p[4] == "-" and p[:4].isdigit() and p[5:].isdigit()
                and 1 <= int(p[5:]) <= 12 and 2000 <= int(p[:4]) <= 2099):
            raise ConfigError("period must look like 'YYYY-MM'")
    max_issues = sum(_hi(cfg["errors"][k]) for k in ERROR_TYPES) + sum(
        INTERACTION_ISSUES[k] * _hi(cfg["interactions"][k]) for k in INTERACTIONS)
    min_issues = sum(_lo(cfg["errors"][k]) for k in ERROR_TYPES) + sum(
        INTERACTION_ISSUES[k] * _lo(cfg["interactions"][k]) for k in INTERACTIONS)
    if min_issues > MAX_ISSUES:
        raise ConfigError(f"config always exceeds {MAX_ISSUES} issues")
    del max_issues  # upper ranges are capped during sampling instead
    if cfg["transform"] is not None:
        t = cfg["transform"]
        if not isinstance(t, dict) or set(t) - {"seed", "permute", "remap_ids"} or not isinstance(t.get("seed"), int) \
                or isinstance(t.get("seed"), bool):
            raise ConfigError("transform must be {seed:int, permute?:bool, remap_ids?:bool}")
        for k in ("permute", "remap_ids"):
            if k in t and not isinstance(t[k], bool):
                raise ConfigError(f"transform.{k} must be bool")
    if not isinstance(cfg["name"], str) or len(cfg["name"]) > 80:
        raise ConfigError("name must be a short string")
    return cfg


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(cfg: Dict[str, Any]) -> str:
    return derive_hex("config", canonical_json(cfg), n=16)


def _sample(v: Any, rng: Rng) -> int:
    return v if isinstance(v, int) else rng.randint(v[0], v[1])


def sample_config(cfg: Dict[str, Any], rng: Rng) -> Dict[str, Any]:
    """Turn ranges into concrete numbers for one episode (recorded only in provenance)."""
    out: Dict[str, Any] = {"size": {}, "errors": {}, "interactions": {}, "distractors": {}}
    for k, v in cfg["size"].items():
        out["size"][k] = _sample(v, rng)
    out["foreign_count"] = _sample(cfg["currencies"]["foreign_count"], rng)
    budget = MAX_ISSUES
    order = list(INTERACTIONS) + list(ERROR_TYPES)

    def section_of(k: str) -> str:
        return "interactions" if k in INTERACTIONS else "errors"

    for idx, k in enumerate(order):
        per = INTERACTION_ISSUES.get(k, 1)
        lo = _lo(cfg[section_of(k)][k])
        reserved = sum(INTERACTION_ISSUES.get(j, 1) * _lo(cfg[section_of(j)][j]) for j in order[idx + 1:])
        n = _sample(cfg[section_of(k)][k], rng)
        n = max(lo, min(n, (budget - reserved) // per))
        out[section_of(k)][k] = n
        budget -= n * per
    for k in DISTRACTORS:
        out["distractors"][k] = _sample(cfg["distractors"][k], rng)
    return out
