"""Safe arithmetic calculator: decimal numbers, + - * /, unary +/-, parentheses. Nothing else.

There is no eval(), no names, no attribute access, no exponent operator and no I/O. Input
length, token count, nesting depth and result magnitude are bounded.
"""
from __future__ import annotations

from decimal import Context, Decimal, DivisionByZero, InvalidOperation, Overflow, localcontext
from typing import List, Tuple

MAX_LEN = 256
MAX_TOKENS = 128
MAX_DEPTH = 24
MAX_ABS = Decimal(10) ** 30
_CTX = Context(prec=34, Emax=999, Emin=-999, traps=[DivisionByZero, InvalidOperation, Overflow])


class CalcError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _tokenize(expr: str) -> List[Tuple[str, str]]:
    toks: List[Tuple[str, str]] = []
    i, n = 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch in " \t":
            i += 1
            continue
        if ch in "+-*/()":
            toks.append(("op", ch))
            i += 1
            continue
        if ch.isdigit() or ch == ".":
            j = i
            dots = 0
            while j < n and (expr[j].isdigit() or expr[j] == "."):
                if expr[j] == ".":
                    dots += 1
                j += 1
            text = expr[i:j]
            if dots > 1 or text == "." or not all(c.isascii() for c in text):
                raise CalcError("calculator_syntax", "Malformed number.")
            toks.append(("num", text))
            i = j
            continue
        raise CalcError("calculator_syntax", "Only digits, '.', '+', '-', '*', '/', '(' , ')' and spaces are allowed.")
    if len(toks) > MAX_TOKENS:
        raise CalcError("calculator_limit", f"Expression has more than {MAX_TOKENS} tokens.")
    return toks


class _Parser:
    def __init__(self, toks: List[Tuple[str, str]]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def expr(self, depth: int) -> Decimal:
        if depth > MAX_DEPTH:
            raise CalcError("calculator_limit", "Expression nested too deeply.")
        v = self.term(depth)
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            rhs = self.term(depth)
            v = v + rhs if op == "+" else v - rhs
            _bound(v)
        return v

    def term(self, depth: int) -> Decimal:
        v = self.factor(depth)
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            rhs = self.factor(depth)
            if op == "/" and rhs == 0:
                raise CalcError("calculator_math", "Division by zero.")
            v = v * rhs if op == "*" else v / rhs
            _bound(v)
        return v

    def factor(self, depth: int) -> Decimal:
        t = self.take()
        if t is None:
            raise CalcError("calculator_syntax", "Unexpected end of expression.")
        if t == ("op", "+"):
            return self.factor(depth + 1)
        if t == ("op", "-"):
            return -self.factor(depth + 1)
        if t == ("op", "("):
            v = self.expr(depth + 1)
            if self.take() != ("op", ")"):
                raise CalcError("calculator_syntax", "Missing ')'.")
            return v
        if t[0] == "num":
            v = Decimal(t[1])
            _bound(v)
            return v
        raise CalcError("calculator_syntax", "Unexpected token.")


def _bound(v: Decimal) -> None:
    if abs(v) >= MAX_ABS:
        raise CalcError("calculator_limit", "Result magnitude too large.")


def format_decimal(v: Decimal) -> str:
    if v == 0:
        return "0"
    s = format(v.normalize(_CTX), "f")
    return s


def evaluate(expression: object) -> str:
    if not isinstance(expression, str):
        raise CalcError("invalid_arguments", "expression must be a string.")
    if len(expression) > MAX_LEN:
        raise CalcError("calculator_limit", f"Expression longer than {MAX_LEN} characters.")
    toks = _tokenize(expression)
    if not toks:
        raise CalcError("calculator_syntax", "Empty expression.")
    try:
        with localcontext(_CTX):
            p = _Parser(toks)
            v = p.expr(0)
            if p.peek() is not None:
                raise CalcError("calculator_syntax", "Unexpected trailing input.")
            _bound(v)
            return format_decimal(+v)
    except CalcError:
        raise
    except (ArithmeticError, RecursionError):
        raise CalcError("calculator_math", "Arithmetic error.") from None
