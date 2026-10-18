"""The token match model of ADR 0010, in Python integers: the mirror of `crates/taco-match`.

The model sees the tokens of one block. Before token `i`, it finds the longest earlier
occurrence of the context `h[i-N..i)` for the orders `N` (longest first), and predicts that the
token after that occurrence comes next. The match length is the confidence input of the mixer.

Rules (ADR 0010 section 2; `crates/taco-match/src/matcher.rs` is the reference):

1. Hash of order `N` at position `i` (`i >= N`):
   `H = sum over k in 1..=N of (h[i-k] + 1) * B^(k-1)` (mod 2^64), then
   `slot = ((H ^ (H >> 32)) * MIX mod 2^64) >> (64 - hash_bits)`.
2. One table for each order. `table[slot]` holds the last position `j` (with `j >= N`) that had
   that slot. Position `i` goes into each table after the look-up of step 3 for position `i`.
3. Before token `i`: if the match length `len` is less than the largest order, look up the
   orders `N > len`, longest first. A candidate `c` is accepted if the verified length `L`
   (the number of equal tokens before `c` and before `i`, at most `verify_max`) is at least `N`.
   Then `ptr = c` and `len = L`. The first accepted order wins.
4. The prediction is `h[ptr]` with the length `len`, if `len > 0`.
5. After token `i`: if `len > 0` and `h[ptr] == h[i]`, then `ptr += 1` and
   `len = min(len + 1, max_len)`. Otherwise `len = 0`.

`predict_block` gives the same predictions as the step-by-step rule. It finds the table
contents with a sort (the table holds the last earlier position with the same slot).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

B = 0x9E3779B97F4A7C15
MIX = 0xD6E8FEB86659FD93
M64 = (1 << 64) - 1


@dataclass(frozen=True)
class MatchParams:
    """The parameters of the match model (ADR 0010 section 2)."""

    orders: tuple[int, ...] = (32, 16, 8, 4)
    hash_bits: int = 20
    verify_max: int = 64
    max_len: int = 65535

    @property
    def key(self) -> str:
        return "o" + "-".join(map(str, self.orders)) + f"_h{self.hash_bits}"


def slots(h: np.ndarray, order: int, hash_bits: int) -> np.ndarray:
    """The slot of order `order` for each position `i` in `0..=len(h)`; -1 for `i < order`."""
    n = len(h)
    out = np.full(n + 1, -1, dtype=np.int64)
    if n < order:
        return out
    with np.errstate(over="ignore"):
        hu = h.astype(np.uint64) + np.uint64(1)
        acc = np.zeros(n + 1 - order, dtype=np.uint64)
        bk = np.uint64(1)
        for k in range(1, order + 1):
            # positions i = order..n; the token h[i-k]
            acc += hu[order - k : n + 1 - k] * bk
            bk = bk * np.uint64(B)
        x = (acc ^ (acc >> np.uint64(32))) * np.uint64(MIX)
    out[order:] = (x >> np.uint64(64 - hash_bits)).astype(np.int64)
    return out


def last_same(s: np.ndarray) -> np.ndarray:
    """For each position, the last earlier position with the same slot (>= 0), else -1."""
    n = len(s)
    out = np.full(n, -1, dtype=np.int64)
    valid = np.nonzero(s >= 0)[0]
    if len(valid) == 0:
        return out
    order = valid[np.argsort(s[valid], kind="stable")]
    sv = s[order]
    same = sv[1:] == sv[:-1]
    out[order[1:][same]] = order[:-1][same]
    return out


def common_suffix(h: list[int], c: int, i: int, limit: int) -> int:
    """The number of k in 1..=limit with h[c-k] == h[i-k], stopping at the first difference."""
    n = 0
    while n < limit and c - n - 1 >= 0 and h[c - n - 1] == h[i - n - 1]:
        n += 1
    return n


def predict_block(tokens: np.ndarray, p: MatchParams) -> tuple[np.ndarray, np.ndarray]:
    """The prediction before each token of one block: (token or -1, length or 0)."""
    n = len(tokens)
    h = tokens.tolist()
    cand = {o: last_same(slots(tokens, o, p.hash_bits)).tolist() for o in p.orders}
    top = p.orders[0]
    pred = np.full(n, -1, dtype=np.int64)
    plen = np.zeros(n, dtype=np.int64)
    ptr = 0
    ln = 0
    for i in range(n):
        if ln < top:
            for o in p.orders:
                if o <= ln or i < o:
                    continue
                c = cand[o][i]
                if c < 0:
                    continue
                v = common_suffix(h, c, i, p.verify_max)
                if v >= o:
                    ptr, ln = c, v
                    break
        if ln > 0:
            pred[i] = h[ptr]
            plen[i] = ln
            if h[ptr] == h[i]:
                ptr += 1
                ln = min(ln + 1, p.max_len)
            else:
                ln = 0
    return pred, plen


def predict_block_ref(tokens: list[int], p: MatchParams) -> tuple[list[int], list[int]]:
    """The same rule with explicit tables, step by step (slow; for tests)."""
    tables = {o: [-1] * (1 << p.hash_bits) for o in p.orders}
    h: list[int] = []
    pred, plen = [], []
    ptr = ln = 0

    def slot(i: int, o: int) -> int:
        acc = 0
        bk = 1
        for k in range(1, o + 1):
            acc = (acc + (h[i - k] + 1) * bk) & M64
            bk = (bk * B) & M64
        x = ((acc ^ (acc >> 32)) * MIX) & M64
        return x >> (64 - p.hash_bits)

    for t in tokens:
        i = len(h)
        if ln < p.orders[0]:
            for o in p.orders:
                if o <= ln or i < o:
                    continue
                c = tables[o][slot(i, o)]
                if c < 0:
                    continue
                v = common_suffix(h, c, i, p.verify_max)
                if v >= o:
                    ptr, ln = c, v
                    break
        for o in p.orders:
            if i >= o:
                tables[o][slot(i, o)] = i
        if ln > 0:
            pred.append(h[ptr])
            plen.append(ln)
            if h[ptr] == t:
                ptr += 1
                ln = min(ln + 1, p.max_len)
            else:
                ln = 0
        else:
            pred.append(-1)
            plen.append(0)
        h.append(t)
    return pred, plen


def predict_file(ids: np.ndarray, block_starts: np.ndarray, blocks_per_unit: int,
                 p: MatchParams) -> tuple[np.ndarray, np.ndarray]:
    """Predictions for a file whose block is `blocks_per_unit` 16 KiB token blocks."""
    n = len(ids)
    pred = np.full(n, -1, dtype=np.int64)
    plen = np.zeros(n, dtype=np.int64)
    nb = len(block_starts) - 1
    for b in range(0, nb, blocks_per_unit):
        s = int(block_starts[b])
        e = int(block_starts[min(b + blocks_per_unit, nb)])
        pr, pl = predict_block(ids[s:e], p)
        pred[s:e] = pr
        plen[s:e] = pl
    return pred, plen


def _self_test() -> None:
    rng = np.random.default_rng(7)
    for trial in range(30):
        n = int(rng.integers(0, 400))
        vocab = int(rng.integers(2, 12))
        base = rng.integers(0, vocab, n)
        if trial % 3 == 0 and n > 20:
            base[n // 2 :] = base[: n - n // 2]  # a long repeat
        p = MatchParams(orders=(8, 4, 2, 1), hash_bits=int(rng.integers(3, 12)), verify_max=16,
                        max_len=40)
        a = predict_block(base, p)
        b = predict_block_ref(base.tolist(), p)
        if a[0].tolist() != b[0] or a[1].tolist() != b[1]:
            raise SystemExit(f"trial {trial}: predict_block differs from the reference")
    print("matchmodel self test: 30 of 30 equal")


if __name__ == "__main__":
    _self_test()
