"""splitmix64: a seeded integer PRNG with one defined output sequence on every machine."""

from __future__ import annotations

MASK64 = (1 << 64) - 1


class SplitMix64:
    """The splitmix64 generator (the same constants as the Rust test helper `Rng`)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self) -> float:
        """A uniform value in [0, 1) with 53 random bits (exact conversion)."""
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def normal(self) -> float:
        """An approximately normal value, mean 0, variance 1 (Irwin-Hall, 12 uniforms)."""
        s = 0.0
        for _ in range(12):
            s += self.uniform()
        return s - 6.0

    def below(self, n: int) -> int:
        return self.next_u64() % n

    def bytes(self, n: int) -> bytes:
        return bytes(self.next_u64() & 0xFF for _ in range(n))
