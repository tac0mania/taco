"""The integer mixer of ADR 0010, in Python integers: the mirror of `crates/taco-match`.

The mixer gets the u64 weights of HardShell and the prediction of the match model (a token `p`
and a match length `L`). It gives new u64 weights for `FreqTable::from_weights`:

1. `W = sum(w)` (exact). With no prediction, each weight becomes `max(w, 1)`. Stop.
2. `pg = clamp(floor(w[p] * 2^16 / W), 1, 2^16 - 1)`: the GPT-2 probability of `p`.
3. Inputs, in units of 1/256 nat: `x_g = stretch(pg)`, `x_m = stretch(hit[b] >> 6)`, and the
   bias `256`. `b = bucket(L)`. `hit[b]` is an adaptive probability (22 bits) that the
   prediction is correct.
4. The weight set is `c = 2 * b + a` (`a = 1` if `p` is the GPT-2 token with the largest
   weight, the lowest id on a tie).
   `dot = wg[c] * x_g + wm[c] * x_m + wb[c] * 256` (weights in units of 2^-16),
   `x = clamp(round(dot / 2^16), -4095, 4095)`, `q = squash(x)`: the final probability of `p`.
5. `R = max(W - w[p], 1)`, `m = 2^16 - q`, `k = max(0, bitlen(R) + 16 - 62)`.
   For `i != p`: `w'[i] = max(1, (w[i] * m) >> k)`. `w'[p] = max(1, (R * q) >> k)`.
   Thus `p` has the probability `q / 2^16`, and the other tokens share `m / 2^16` in the
   proportions of GPT-2. Every weight is at least 1 and at most 2^62.
6. After the actual token `a`: `y = (a == p)`. Update `hit[b]` (count-based rate) and the three
   weights of set `c` (`w += round(x * err * lr / 2^20)`, `err = y * 2^16 - q`, clamped to
   ±2^22). The rate is `max(lr_min, floor(lr * D / (n + D)))`, `n` = earlier updates of `c`.

The squash table is integer: `E[0] = 2^62`, `E[j + 1] = round(E[j] * C / 2^62)` with
`C = round(2^62 * exp(-1/256))`, and `squash(x) = round(2^78 / (2^62 + E[x]))` for `x >= 0`,
`squash(-x) = 2^16 - squash(x)`, clamped to `[1, 2^16 - 1]`. `stretch(p)` is the smallest `x`
with `squash(x) >= p`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C_EXP = 4593706758521714574  # round(2^62 * exp(-1/256)), 60-digit decimal arithmetic
X_MAX = 4095
P_ONE = 1 << 16
HIT_BITS = 22
N_BUCKETS = 21
W_LIMIT = 1 << 22


def bitlen(x: int) -> int:
    return x.bit_length()


def make_squash() -> list[int]:
    """squash(x) for x in -X_MAX..=X_MAX (index x + X_MAX)."""
    e = [1 << 62]
    for _ in range(X_MAX):
        e.append((e[-1] * C_EXP + (1 << 61)) >> 62)
    pos = []
    for x in range(X_MAX + 1):
        d = (1 << 62) + e[x]
        pos.append(((1 << 78) + d // 2) // d)
    out = [0] * (2 * X_MAX + 1)
    for x in range(X_MAX + 1):
        out[X_MAX + x] = min(max(pos[x], 1), P_ONE - 1)
        out[X_MAX - x] = min(max(P_ONE - pos[x], 1), P_ONE - 1)
    return out


SQUASH = make_squash()


def make_stretch() -> list[int]:
    """stretch(p) for p in 0..2^16: the smallest x with squash(x) >= p (X_MAX if none)."""
    out = [X_MAX] * P_ONE
    x = -X_MAX
    for p in range(P_ONE):
        while x <= X_MAX and SQUASH[x + X_MAX] < p:
            x += 1
        out[p] = min(x, X_MAX)
    return out


STRETCH = make_stretch()


def squash(x: int) -> int:
    return SQUASH[min(max(x, -X_MAX), X_MAX) + X_MAX]


def bucket(length: int) -> int:
    """The confidence bucket of a match length (>= 1): 0..14 for 1..15, then one per octave."""
    if length < 16:
        return length - 1
    return min(15 + bitlen(length) - 5, N_BUCKETS - 1)


@dataclass(frozen=True)
class MixerParams:
    """The constants of the mixer (ADR 0010 section 3)."""

    lr: int = 160  # learning rate: weight change = x * err * lr / 2^20
    hit_limit: int = 255  # the count limit of the adaptive hit probability
    hit_init: int = 1 << (HIT_BITS - 1)  # 0.5
    wg_init: int = 1 << 16  # 1.0: at the start, q is the GPT-2 probability
    wm_init: int = 1 << 16
    wb_init: int = 0
    lr_min: int = 16  # lr_eff = max(lr_min, lr * D / (n + D)); D = 0: a constant rate `lr`
    lr_decay: int = 64  # D
    agree: bool = True  # weight sets for each (bucket, p == GPT-2 top token)

    @property
    def key(self) -> str:
        return (f"lr{self.lr}_lim{self.hit_limit}_wm{self.wm_init}_min{self.lr_min}"
                f"_d{self.lr_decay}_a{int(self.agree)}")


@dataclass
class Mixer:
    """The state of the mixer for one block."""

    p: MixerParams = field(default_factory=MixerParams)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        p = self.p
        self.hit = [p.hit_init] * N_BUCKETS
        self.count = [0] * N_BUCKETS
        self.w = [[p.wg_init, p.wm_init, p.wb_init] for _ in range(2 * N_BUCKETS)]
        self.wn = [0] * (2 * N_BUCKETS)
        self.pending: tuple[int, int, tuple[int, int, int], int] | None = None

    def prob(self, wp: int, wsum: int, length: int, agree: bool = False) -> int:
        """Steps 2 to 4: q for the predicted token, and remember the inputs for `update`."""
        pg = (wp << 16) // wsum if wsum > 0 else 1
        pg = min(max(pg, 1), P_ONE - 1)
        b = bucket(length)
        xs = (STRETCH[pg], STRETCH[self.hit[b] >> (HIT_BITS - 16)], 256)
        c = b * 2 + (1 if (agree and self.p.agree) else 0)
        wv = self.w[c]
        dot = wv[0] * xs[0] + wv[1] * xs[1] + wv[2] * xs[2]
        x = (dot + (1 << 15)) >> 16
        q = squash(x)
        self.pending = (b, q, xs, c)
        return q

    def update(self, hit: bool) -> None:
        """Step 6. Call after `prob` with y = (actual == predicted)."""
        if self.pending is None:
            return
        b, q, xs, c = self.pending
        self.pending = None
        y = 1 if hit else 0

        err = (y << 16) - q
        wv = self.w[c]
        lr = self.p.lr
        if self.p.lr_decay:
            lr = max(self.p.lr_min, lr * self.p.lr_decay // (self.wn[c] + self.p.lr_decay))
            self.wn[c] = min(self.wn[c] + 1, 1 << 20)
        for i in range(3):
            d = (xs[i] * err * lr + (1 << 19)) >> 20
            wv[i] = min(max(wv[i] + d, -W_LIMIT), W_LIMIT)
        n = self.count[b]
        h = self.hit[b]
        if hit:
            h += (((1 << HIT_BITS) - h) * 2) // (2 * n + 3)
        else:
            h -= (h * 2) // (2 * n + 3)
        self.hit[b] = h
        self.count[b] = min(n + 1, self.p.hit_limit)


def mix_row(w: list[int], pred: int, q: int) -> list[int]:
    """Step 5 on one row (Python integers). `pred < 0` means no prediction."""
    if pred < 0:
        return [max(x, 1) for x in w]
    wsum = sum(w)
    r = max(wsum - w[pred], 1)
    m = P_ONE - q
    k = max(0, bitlen(r) + 16 - 62)
    out = [max(1, (x * m) >> k) for x in w]
    out[pred] = max(1, (r * q) >> k)
    return out


def mix_row_np(w: np.ndarray, pred: int, q: int) -> np.ndarray:
    """Step 5 with NumPy, for rows whose weights are below 2^47 (HardShell: below 2^41)."""
    if pred < 0:
        return np.maximum(w, np.uint64(1))
    if int(w.max()) >= 1 << 47:
        return np.array(mix_row([int(x) for x in w], pred, q), dtype=np.uint64)
    wsum = int(w.sum(dtype=np.uint64))
    r = max(wsum - int(w[pred]), 1)
    m = P_ONE - q
    k = max(0, bitlen(r) + 16 - 62)
    out = np.maximum((w * np.uint64(m)) >> np.uint64(k), np.uint64(1))
    out[pred] = max(1, (r * q) >> k)
    return out
