"""Python reference range coder and frequency tables, for measurement (plan §5, §5.1).

This coder obeys the requirements of plan §5. The Rust crate `taco-coder` and its ADR are the
authority for the published format. If they differ from this file, the crate is correct, and
this file must change.

Algorithm (64-bit LZMA-style range coder, all values are Python integers):

- State: `low` (64 bits and a carry, < 2^65), `range` (64 bits), a cache byte and a count of
  pending bytes. Start: low = 0, range = 2^64 - 1.
- Encode symbol (cum, freq) of a table with total T <= 2^32:
  r = range // T; low += r * cum;
  range = r * freq if cum + freq < T, else range - r * cum (the last symbol takes the rest).
- Renormalise: while range < 2^56: range <<= 8, shift out the top byte of low (LZMA
  `ShiftLow`, with carry propagation through pending 0xFF bytes). Thus range >= 2^56 and
  r >= 2^24 after each symbol.
- The first byte of an LZMA-style stream is always 0. The encoder does not write it.
- Flush: select the value in [low, low + range) with the most trailing zero bytes, shift out 9
  bytes, then remove the trailing zero bytes. The decoder reads zero bytes after the end of the
  input (§5 item 5), so the removed bytes are not necessary.
- Decode: code = the first 8 bytes. r = range // T; v = min(code // r, T - 1); the symbol is
  the token s with cum[s] <= v < cum[s + 1]; then the same update as the encoder, with
  code -= r * cum[s].

`from_weights` is plan §5.1 exactly, vectorised with NumPy for speed. `from_weights_ref` is the
same algorithm with Python integers (u128 semantics); a test compares them.
"""

from __future__ import annotations

import math

import numpy as np

FREQ_BITS = 32
TOTAL = 1 << FREQ_BITS
RANGE_BITS = 64
MASK64 = (1 << RANGE_BITS) - 1
TOP = 1 << 56
CARRY_TEST = 0xFF << 56
WEIGHT_SCALE_BITS = 32
LN2 = math.log(2.0)


# Frequency tables (§5.1) ------------------------------------------------------------------


def from_weights_ref(w: list[int], total: int = TOTAL) -> list[int]:
    """Plan §5.1 with Python integers. Slow; for tests."""
    v = len(w)
    if sum(w) == 0:
        w = [1] * v
    s = sum(w)
    f = [1 + (wi * (total - v)) // s for wi in w]
    best = max(range(v), key=lambda i: (w[i], -i))
    f[best] += total - sum(f)
    return f


def from_weights(w: np.ndarray, total: int = TOTAL) -> np.ndarray:
    """Plan §5.1 for a batch of weight rows [R, V] (uint64). Returns uint64 frequencies.

    INVARIANT: every weight is <= 2^32 and total <= 2^32, so w * (total - V) < 2^64 and the
    uint64 product is exact (the u128 of §5.1 is not necessary for this input range).
    """
    w = np.atleast_2d(w)
    if w.dtype != np.uint64:
        raise TypeError("weights must be uint64")
    if int(w.max()) > (1 << WEIGHT_SCALE_BITS):
        raise ValueError("weight above 2^32: the uint64 product can overflow")
    v = w.shape[1]
    w = w.copy()
    zero = w.sum(axis=1) == 0
    w[zero] = 1
    s = w.sum(axis=1, keepdims=True)
    f = np.uint64(1) + (w * np.uint64(total - v)) // s
    rem = np.uint64(total) - f.sum(axis=1)
    best = np.argmax(w, axis=1)  # NOTE: argmax returns the lowest index on a tie (§5.1 step 3).
    f[np.arange(len(f)), best] += rem
    return f


def weights_from_logits(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Float logits [R, V] -> (u64 weights [R, V], float64 log-sum-exp [R]).

    The scaling, in float64: e = exp(logit - max), s = sum(e), w = floor(e * (2^32 / s)).
    Thus w is p * 2^32 rounded down, with w <= 2^32. Tokens with p < 2^-32 get w = 0, and §5.1
    gives them the minimum frequency 1.
    """
    x = np.atleast_2d(logits).astype(np.float64)
    m = x.max(axis=1)
    x -= m[:, None]
    np.exp(x, out=x)
    s = x.sum(axis=1)
    x *= (float(1 << WEIGHT_SCALE_BITS) / s)[:, None]
    np.floor(x, out=x)
    return x.astype(np.uint64), m + np.log(s)


def tables_from_logits(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq uint64 [R, V], log-sum-exp [R]) for float logits [R, V]."""
    w, lse = weights_from_logits(logits)
    return from_weights(w), lse


def log2_prob(logit: float, lse: float) -> float:
    """log2 of the float64 softmax probability of one token: (logit - lse) / ln 2."""
    return (float(logit) - float(lse)) / LN2


def probabilities(logits: np.ndarray) -> np.ndarray:
    """Float64 softmax probabilities [R, V] (for drift measurements)."""
    x = np.atleast_2d(logits).astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=1, keepdims=True)
    return x


def cumulative(freq: np.ndarray) -> np.ndarray:
    """Return cum [V + 1] with cum[0] = 0 and cum[V] = TOTAL, for one table."""
    cum = np.zeros(len(freq) + 1, dtype=np.uint64)
    np.cumsum(freq, out=cum[1:])
    return cum


# Range coder (§5) -------------------------------------------------------------------------


class Encoder:
    """Range encoder. Call `encode` for each symbol, then `finish` one time."""

    def __init__(self) -> None:
        self.low = 0
        self.range = MASK64
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()
        self._first = True

    def _emit(self, b: int) -> None:
        if self._first:
            # INVARIANT: the first byte is always 0, because low + range never passes 2^64
            # at the scale of the first byte. The decoder does not read it.
            if b != 0:
                raise AssertionError("first byte of the stream is not 0")
            self._first = False
            return
        self.out.append(b)

    def _shift_low(self) -> None:
        if (self.low & MASK64) < CARRY_TEST or (self.low >> RANGE_BITS) != 0:
            carry = self.low >> RANGE_BITS
            temp = self.cache
            while True:
                self._emit((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (self.low >> 56) & 0xFF
        self.cache_size += 1
        self.low = (self.low & (TOP - 1)) << 8

    def encode(self, cum: int, freq: int, total: int = TOTAL) -> None:
        """Encode the symbol with interval [cum, cum + freq) of a table with this total."""
        if not (0 < freq and cum + freq <= total <= TOTAL):
            raise ValueError("bad symbol interval")
        r = self.range // total
        self.low += r * cum
        if cum + freq < total:
            self.range = r * freq
        else:
            self.range -= r * cum
        while self.range < TOP:
            self.range <<= 8
            self._shift_low()

    def finish(self) -> bytes:
        """Flush and return the stream. The encoder must not be used after this call."""
        end = self.low + self.range
        for k in range(9):
            unit = 1 << (RANGE_BITS - 8 * k)
            v = -(-self.low // unit) * unit
            if v < end:
                self.low = v
                break
        for _ in range(9):
            self._shift_low()
        out = bytes(self.out)
        return out.rstrip(b"\x00")


class Decoder:
    """Range decoder. Reads zero bytes after the end of the stream."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 8
        self.code = int.from_bytes(data[:8].ljust(8, b"\x00"), "big")
        self.range = MASK64
        self._r = 0

    def target(self, total: int = TOTAL) -> int:
        """Return the value v that selects the next symbol: cum[s] <= v < cum[s + 1]."""
        self._r = self.range // total
        return min(self.code // self._r, total - 1)

    def consume(self, cum: int, freq: int, total: int = TOTAL) -> None:
        """Remove the symbol that `target` selected. Same update as `Encoder.encode`."""
        r = self._r
        self.code -= r * cum
        if cum + freq < total:
            self.range = r * freq
        else:
            self.range -= r * cum
        while self.range < TOP:
            b = self.data[self.pos] if self.pos < len(self.data) else 0
            self.pos += 1
            self.code = ((self.code << 8) | b) & MASK64
            self.range <<= 8

    def decode(self, cum: np.ndarray, total: int = TOTAL) -> int:
        """Decode one symbol with the cumulative table cum [V + 1]."""
        v = self.target(total)
        s = int(np.searchsorted(cum, v, side="right")) - 1
        self.consume(int(cum[s]), int(cum[s + 1] - cum[s]), total)
        return s
