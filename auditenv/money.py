"""Money and FX arithmetic.

* Amounts are integers in minor units; each currency has an exponent.
* FX rates are decimal strings: 1 unit of source = ``rate`` units of target.
* Conversion: target_minor = round_half_even(amount_minor * 10^-e_src * rate * 10^e_tgt).
  Only the final result is rounded.
"""
from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

# Fictional-use currency table (ISO-like codes, exponents as in ISO 4217).
EXPONENTS = {
    "KZT": 2, "USD": 2, "EUR": 2, "GBP": 2, "CNY": 2, "AED": 2, "TRY": 2,
    "INR": 2, "JPY": 0, "KWD": 3, "UZS": 2, "CHF": 2,
}

_CTX = Context(prec=60, rounding=ROUND_HALF_EVEN)


def exponent(currency: str) -> int:
    try:
        return EXPONENTS[currency]
    except KeyError:
        raise ValueError(f"unknown currency {currency!r}") from None


def convert(amount_minor: int, src: str, tgt: str, rate: str | Decimal) -> int:
    """Convert integer minor units of ``src`` into integer minor units of ``tgt``."""
    if isinstance(amount_minor, bool) or not isinstance(amount_minor, int):
        raise TypeError("amount_minor must be int")
    with localcontext(_CTX):
        r = Decimal(rate)
        value = Decimal(amount_minor).scaleb(-exponent(src)) * r
        value = value.scaleb(exponent(tgt))
        return int(value.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def quantize_rate(value: Decimal, significant: int = 8) -> str:
    """Round a positive rate to ``significant`` significant digits, plain notation."""
    with localcontext(_CTX):
        if value <= 0:
            raise ValueError("rate must be positive")
        q = Decimal(1).scaleb(value.adjusted() - (significant - 1))
        out = value.quantize(q, rounding=ROUND_HALF_EVEN)
        return format(out, "f")


def major_str(amount_minor: int, currency: str) -> str:
    e = exponent(currency)
    with localcontext(_CTX):
        return format(Decimal(amount_minor).scaleb(-e).quantize(Decimal(1).scaleb(-e)), "f")


TOLERANCE_MINOR = 0


def amount_tolerance(expected_minor: int) -> int:
    """Published amount tolerance (tau) of every generated issue type: 0 minor units.

    All expected amounts are exact integer results of the published booking and rounding
    rules, so no tolerance is needed. The grader reads tau per issue from truth.tolerance_minor.
    """
    del expected_minor
    return TOLERANCE_MINOR
