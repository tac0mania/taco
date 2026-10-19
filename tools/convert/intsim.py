"""A NumPy simulation of the HardShell GPT-2 forward pass, from the integer tensors of a model.

This tool checks the converter and finds the parts of the integer model that differ from the
float model. It follows the forward pass in `crates/hardshell/src/gpt2/mod.rs`, for a full
window at a time (integer arithmetic does not depend on the order, so the full-sequence pass
and the KV-cache pass give the same values).

Exactness: the matrix products, the rescales with a product below 2^62, the softmax and the
logits are exact. The LayerNorm statistics and the attention-output rescale use float64, so a
value can differ from HardShell by one unit in rare cases. Thus the result is near HardShell,
not the same bits. HardShell (`eval_bits`) gives the reference numbers.

NOTE: a measurement tool with floats, not the prediction path (CLAUDE.md §1 rule 1).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from convert import numerics as nm
from convert.hsmfile import HsmModel

R = nm.RESIDUAL_LIMIT
SCORE_LIMIT = 1 << 40


def rescale_exact(x: np.ndarray, m, s: int) -> np.ndarray:
    """floor((x * m + 2^(s-1)) / 2^s) for int64 x and m with |x * m| < 2^62."""
    x = np.asarray(x, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    if s == 0:
        return x * m
    if s > 62:
        return np.zeros(np.broadcast(x, m).shape, dtype=np.int64)
    return (x * m + (np.int64(1) << np.int64(s - 1))) >> np.int64(s)


def rescale_wide(x: np.ndarray, m, s: int) -> np.ndarray:
    """The rule of `rescale_exact` for |x| < 2^46 and 0 <= m < 2^31 (the i16 activations).

    m = mh * 2^16 + ml, so each product is below 2^62, and
    floor((x * m + 2^(s-1)) / 2^s) = floor((x * mh + floor((x * ml + 2^(s-1)) / 2^16)) / 2^(s-16))
    for s >= 16. Exact.
    """
    x = np.asarray(x, dtype=np.int64)
    m = np.asarray(m, dtype=np.int64)
    if s < 16 or s > 62:
        raise ValueError(f"rescale_wide: shift {s} is not in 16..=62")
    mh, ml = m >> np.int64(16), m & np.int64(0xFFFF)
    low = (x * ml + (np.int64(1) << np.int64(s - 1))) >> np.int64(16)
    return (x * mh + low) >> np.int64(s - 16)


def rescale_float(x: np.ndarray, m, s: int) -> np.ndarray:
    """The same rule in float64, for products that do not fit in int64 (near, not exact)."""
    return np.floor(np.asarray(x, dtype=np.float64) * (float(m) * nm.pow2(-s)) + 0.5)


class IntModel:
    """The tensors of an `HsmModel` in the form that the simulation uses."""

    def __init__(self, hsm: HsmModel) -> None:
        c = hsm.config
        self.n_layer, self.n_head, self.d = c["n_layer"], c["n_head"], c["d_model"]
        self.k, self.j = c["constants"]["ln_k"], c["constants"]["ln_j"]
        self.eps = c["constants"]["ln_eps"]
        self.p = c["constants"]["softmax_p"]
        self.lnf_max = 32767 if c["constants"].get("ln_f_bits") == 16 else 127
        self.beta_frac = c["constants"].get("ln_beta_frac", 0)
        self.t = {t.name: t for t in hsm.tensors}

    def arr(self, name: str) -> np.ndarray:
        return np.asarray(self.t[name].data)

    def resc(self, name: str) -> tuple[np.ndarray, int]:
        t = self.t[name]
        return np.asarray(t.data, dtype=np.int64), int(t.shift)


Hook = Callable[[str, np.ndarray], None]


def layernorm(im: IntModel, prefix: str, x: np.ndarray, qmax: int = 127) -> np.ndarray:
    g = im.arr(prefix + ".g").astype(np.float64)
    b = im.arr(prefix + ".b").astype(np.int64)
    m, s = im.resc(prefix + ".rescale")
    n = x.shape[1]
    mean = np.floor_divide(x.sum(axis=1), n)  # exact: |sum| < 2^47
    c = (x - mean[:, None]).astype(np.float64)
    var = np.floor((c * c).sum(axis=1) / n)
    root = np.floor(np.sqrt(var * nm.pow2(2 * im.j) + im.eps))
    inv = np.floor(nm.pow2(im.k) / np.maximum(root, 1.0))
    t = c * inv[:, None] * g[None, :] * (float(m[0]) * nm.pow2(-s))
    if im.beta_frac:
        # hsm version 2: beta has `beta_frac` more bits, added before the one rounding.
        y = np.floor(t + b * nm.pow2(-im.beta_frac) + 0.5)
        return np.clip(y, -qmax, qmax).astype(np.int64)
    return np.clip(np.floor(t + 0.5) + b, -qmax, qmax).astype(np.int64)


def linear(im: IntModel, prefix: str, a: np.ndarray) -> np.ndarray:
    """rescale(w . a + b) for each row, before the clamp. Exact.

    An i16 input (hsm version 2) gives |acc| < 2^46 (est. bound 2^16 * 2^7 * 2^15 + |b|): the
    float64 product is still exact (< 2^53), and `rescale_wide` keeps the rescale exact.
    """
    w = im.arr(prefix + ".w").astype(np.float64)
    acc = np.rint(a.astype(np.float64) @ w.T).astype(np.int64)  # exact: |acc| < 2^53
    m, s = im.resc(prefix + ".rescale")
    acc = acc + im.arr(prefix + ".b").astype(np.int64)
    if np.abs(acc).max(initial=0) >= 1 << 31:
        return rescale_wide(acc, m[None, :], s)
    return rescale_exact(acc, m[None, :], s)


def forward(im: IntModel, tokens: np.ndarray, hook: Hook | None = None) -> np.ndarray:
    """Return the u64 weights [T, V] of the window `tokens` (position 0 first)."""
    see = hook or (lambda site, x: None)
    t = len(tokens)
    d, nh = im.d, im.n_head
    hd = d // nh
    wte = im.arr("wte").astype(np.int64)
    em, es = im.resc("wte.rescale_embed")
    pm, ps = im.resc("wpe.rescale")
    wpe = im.arr("wpe").astype(np.int64)
    x = rescale_exact(wte[tokens], em[tokens][:, None], es) + rescale_exact(
        wpe[:t], pm[:t][:, None], ps)
    x = np.clip(x, -R, R)
    see("res", x)
    attn_exp = im.arr("attn_exp").astype(np.int64)
    mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    for li in range(im.n_layer):
        p = f"h.{li}"
        a = layernorm(im, p + ".ln_1", x)
        see(p + ".ln_1", a)
        qkv = np.clip(linear(im, p + ".attn.c_attn", a), -127, 127)
        q, k, v = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
        see(p + ".q", q)
        see(p + ".k", k)
        see(p + ".v", v)
        sm, ss = im.resc(p + ".attn.score.rescale")
        om, os_ = im.resc(p + ".attn.out.rescale")
        att = np.empty((t, d), dtype=np.int64)
        for h in range(nh):
            qh = q[:, h * hd : (h + 1) * hd].astype(np.float64)
            kh = k[:, h * hd : (h + 1) * hd].astype(np.float64)
            vh = v[:, h * hd : (h + 1) * hd].astype(np.float64)
            dot = np.rint(qh @ kh.T).astype(np.int64)
            sc = np.clip(rescale_exact(dot, sm[0], ss), -SCORE_LIMIT, SCORE_LIMIT)
            sc = np.where(mask, np.int64(-SCORE_LIMIT) * 4, sc)  # masked: weight 0 below
            mx = sc.max(axis=1, keepdims=True)
            e = attn_exp[np.clip(mx - sc, 0, len(attn_exp) - 1)]
            e = np.where(mask, 0, e)
            pr = np.floor_divide(e << np.int64(im.p), e.sum(axis=1, keepdims=True))
            acc = pr.astype(np.float64) @ vh  # exact: |acc| < 2^38
            att[:, h * hd : (h + 1) * hd] = np.clip(rescale_float(acc, om[0], os_), -127, 127)
        see(p + ".att", att)
        x = np.clip(x + linear(im, p + ".attn.c_proj", att), -R, R)
        see("res", x)
        a = layernorm(im, p + ".ln_2", x)
        see(p + ".ln_2", a)
        gel = im.arr(p + ".mlp.gelu").astype(np.int64)
        half = len(gel) // 2
        f = np.clip(linear(im, p + ".mlp.c_fc", a), -half, half - 1)
        see(p + ".fc", f)
        g = gel[f + half]
        see(p + ".gelu", g)
        x = np.clip(x + linear(im, p + ".mlp.c_proj", g), -R, R)
        see("res", x)
    a = layernorm(im, "ln_f", x, im.lnf_max)
    see("ln_f", a)
    acc = np.rint(a.astype(np.float64) @ wte.astype(np.float64).T).astype(np.int64)
    lm, ls = im.resc("wte.rescale_logits")
    resc = rescale_wide if np.abs(acc).max(initial=0) >= 1 << 31 else rescale_exact
    lg = np.clip(resc(acc, lm[None, :], ls), -SCORE_LIMIT, SCORE_LIMIT)
    lt = im.arr("logits_exp").astype(np.uint64)
    idx = np.clip(lg.max(axis=1, keepdims=True) - lg, 0, len(lt) - 1)
    return lt[idx]


def bits_of(weights: np.ndarray, targets: np.ndarray) -> tuple[float, int]:
    """The ideal size -log2(w_t / sum w) over tokens with w_t > 0, and the count of w_t = 0."""
    wt = weights[np.arange(len(targets)), targets].astype(np.float64)
    tot = weights.astype(np.float64).sum(axis=1)
    ok = wt > 0
    return float(-np.log2(wt[ok] / tot[ok]).sum()), int((~ok).sum())
