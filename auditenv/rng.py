"""Deterministic, platform-independent PRNG (SplitMix64).

Python's ``random`` module is deterministic too, but its algorithms are an
implementation detail that has changed between versions. SplitMix64 is a few
lines of integer arithmetic, so the generator produces bit-identical episodes
on every Python >= 3.11 and every OS. No floats are used for data; ``random()``
is only used for probability comparisons and is computed exactly.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")
MASK64 = (1 << 64) - 1
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def derive_int(*parts: object) -> int:
    data = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def derive_hex(*parts: object, n: int = 64) -> str:
    data = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:n]


class Rng:
    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & MASK64

    @classmethod
    def from_key(cls, *parts: object) -> "Rng":
        return cls(derive_int(*parts))

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def random(self) -> float:
        """Uniform float in [0, 1); exact (53-bit mantissa from integer)."""
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def below(self, n: int) -> int:
        if n <= 0:
            raise ValueError("below() needs n > 0")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = self.next_u64()
            if x < limit:
                return x % n

    def randint(self, a: int, b: int) -> int:
        if b < a:
            raise ValueError("randint(a, b) needs a <= b")
        return a + self.below(b - a + 1)

    def chance(self, p: float) -> bool:
        return self.random() < p

    def choice(self, seq: Sequence[T]) -> T:
        if not seq:
            raise ValueError("choice from empty sequence")
        return seq[self.below(len(seq))]

    def weighted(self, items: Sequence[T], weights: Sequence[int]) -> T:
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        x = self.below(total)
        acc = 0
        for item, w in zip(items, weights):
            acc += w
            if x < acc:
                return item
        return items[-1]

    def shuffle(self, lst: List[T]) -> None:
        for i in range(len(lst) - 1, 0, -1):
            j = self.below(i + 1)
            lst[i], lst[j] = lst[j], lst[i]

    def sample(self, seq: Iterable[T], k: int) -> List[T]:
        pool = list(seq)
        if k > len(pool):
            raise ValueError("sample larger than population")
        for i in range(k):
            j = i + self.below(len(pool) - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]

    def token(self, n: int, alphabet: str = CROCKFORD) -> str:
        return "".join(alphabet[self.below(len(alphabet))] for _ in range(n))

    def digits(self, n: int) -> str:
        return "".join(str(self.below(10)) for _ in range(n))
