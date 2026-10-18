"""Plan §5.1 (`FreqTable::from_weights`) for many rows at a time, exact, with NumPy.

The quotient `floor(w * (T - V) / W)` needs about 96 bits. NumPy has no such integer, so the
function computes it in float64, and checks each result: the float64 value has a relative error
of less than 1e-15 (four correctly rounded operations), so the floor is certain if the floors of
`v * (1 - 1e-15)` and `v * (1 + 1e-15)` agree. The few uncertain entries use Python integers.
Thus the result is the exact table of `taco-coder`. `_self_test` compares it with
`baseline.rangecoder.from_weights_ref` (Python integers).
"""

from __future__ import annotations

import numpy as np

FREQ_BITS = 32
TOTAL = 1 << FREQ_BITS
EPS = 1e-15


def quotients(w: np.ndarray, spare: int, wsum: np.ndarray) -> np.ndarray:
    """floor(w * spare / wsum) for u64 rows w [r, V] and the row sums wsum [r] (< 2^64)."""
    v = w.astype(np.float64) * float(spare) / wsum.astype(np.float64)[:, None]
    lo = np.floor(v * (1.0 - EPS))
    hi = np.floor(v * (1.0 + EPS))
    q = lo.astype(np.int64)
    bad = np.nonzero(lo != hi)
    for r, i in zip(*bad, strict=True):
        q[r, i] = int(w[r, i]) * spare // int(wsum[r])
    return q


def entries(w: np.ndarray, targets: np.ndarray, total: int = TOTAL) -> np.ndarray:
    """The frequency of `targets[r]` in the table of each row `w[r]` (plan §5.1, exact).

    Rows with the sum 0 are not permitted here (HardShell always gives a weight >= 1).
    """
    r, nv = w.shape
    wsum_obj = [int(x) for x in w.sum(axis=1, dtype=np.uint64)]
    # INVARIANT: the callers give rows with sums below 2^63; check it with Python integers.
    for i in range(r):
        if wsum_obj[i] == 0 or wsum_obj[i] >= 1 << 63:
            raise ValueError("a row sum is 0 or too large for this tool")
    wsum = w.sum(axis=1, dtype=np.uint64)
    spare = total - nv
    q = quotients(w, spare, wsum)
    f = 1 + q
    rem = total - f.sum(axis=1)
    top = np.argmax(w, axis=1)  # the first maximum: the lowest id on a tie
    rows = np.arange(r)
    out = f[rows, targets].copy()
    out += np.where(top == targets, rem, 0)
    return out


def _self_test() -> None:
    from baseline.rangecoder import from_weights_ref

    rng = np.random.default_rng(3)
    for trial in range(40):
        nv = int(rng.integers(2, 300))
        kind = trial % 4
        if kind == 0:
            w = rng.integers(0, 1 << 40, (3, nv), dtype=np.uint64)
        elif kind == 1:
            w = rng.integers(0, 4, (3, nv), dtype=np.uint64)
            w[:, 0] = 1
        elif kind == 2:
            w = (rng.integers(0, 2, (3, nv), dtype=np.uint64) << np.uint64(55)) + np.uint64(1)
        else:
            w = rng.integers(1, 1 << 20, (3, nv), dtype=np.uint64) * np.uint64(1 << 30)
        t = rng.integers(0, nv, 3)
        got = entries(w, t)
        for r in range(3):
            ref = from_weights_ref([int(x) for x in w[r]])
            if ref[t[r]] != got[r]:
                raise SystemExit(f"trial {trial} row {r}: {got[r]} != {ref[t[r]]}")
    print("freq self test: 40 of 40 equal")


if __name__ == "__main__":
    _self_test()
