"""Round-trip tests for the Python range coder and frequency tables (plan §5, §5.1).

Run: uv run --project tools python -m baseline.test_coder
"""

from __future__ import annotations

import math
import random

import numpy as np

from baseline.rangecoder import TOTAL, Decoder, Encoder, cumulative, from_weights, from_weights_ref


def roundtrip(tables: list[list[int]], symbols: list[int], total: int) -> int:
    enc = Encoder()
    for t, s in zip(tables, symbols, strict=True):
        cum = sum(t[:s])
        enc.encode(cum, t[s], total)
    data = enc.finish()
    dec = Decoder(data)
    for t, s in zip(tables, symbols, strict=True):
        cum = np.zeros(len(t) + 1, dtype=np.uint64)
        np.cumsum(np.array(t, dtype=np.uint64), out=cum[1:])
        got = dec.decode(cum, total)
        assert got == s, f"decoded {got}, expected {s}"
    return len(data)


def random_table(rng: random.Random, v: int, total: int) -> list[int]:
    cuts = sorted(rng.sample(range(1, total), v - 1)) if v > 1 else []
    edges = [0, *cuts, total]
    return [edges[i + 1] - edges[i] for i in range(v)]


def main() -> None:
    rng = random.Random(5)
    n = 0
    # Random tables and totals up to 2^32, random symbols.
    for _ in range(300):
        total = rng.choice([TOTAL, 1 << rng.randrange(1, 33), rng.randrange(2, TOTAL + 1)])
        v = rng.randrange(1, min(total, 40) + 1)
        length = rng.randrange(0, 200)
        tables = [random_table(rng, v, total) for _ in range(length)]
        symbols = [rng.randrange(v) for _ in range(length)]
        roundtrip(tables, symbols, total)
        n += 1
    # Extreme tables: the coded symbol has frequency 1 of 2^32, or the last symbol.
    for _ in range(50):
        length = rng.randrange(1, 300)
        t = [1, TOTAL - 2, 1]
        roundtrip([t] * length, [rng.choice([0, 2]) for _ in range(length)], TOTAL)
        roundtrip([t] * length, [1] * length, TOTAL)
        n += 2
    # The size is near the information content.
    t = [1 << 30, 3 << 30]
    syms = [rng.randrange(2) for _ in range(4000)]
    size = roundtrip([t] * len(syms), syms, TOTAL)
    ideal = sum(-math.log2(t[s] / TOTAL) for s in syms) / 8
    assert size <= ideal + 2, (size, ideal)
    # Empty stream.
    assert Encoder().finish() == b""
    # §5.1: the vectorised table is the same as the integer reference, and sums to T.
    for _ in range(200):
        v = rng.randrange(1, 300)
        w = [rng.choice([0, 1, rng.randrange(1 << 32), 1 << 32]) for _ in range(v)]
        if rng.random() < 0.1:
            w = [0] * v
        f = from_weights(np.array([w], dtype=np.uint64))[0]
        assert f.tolist() == from_weights_ref(w), "vectorised table differs from reference"
        assert int(f.sum()) == TOTAL and int(f.min()) >= 1
        assert int(cumulative(f)[-1]) == TOTAL
        n += 1
    print(f"coder: {n} round trips and table checks passed; 4000 binary symbols: "
          f"{size} bytes, ideal {ideal:.1f} bytes")


if __name__ == "__main__":
    main()
