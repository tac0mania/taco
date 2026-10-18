"""A faster copy of `convert.intsim.forward`: the same integers, with float32 matrix products
where float32 is exact.

A product of `i8` values with `i8` values over `n` inputs has partial sums of magnitude at
most `127 * 127 * n`. For `n <= 1,040` this is less than `2^24`, so float32 adds the integers
exactly. Thus the products over the width 768 (c_attn, attn c_proj, c_fc, the logits) and over
the head width 64 (the scores) use float32. The MLP c_proj (width 3,072) uses three float32
products of width 1,024 and an int64 sum. The other steps are the steps of `convert.intsim`.

`check` compares the output with `convert.intsim.forward` bit for bit.

NOTE: a measurement tool with floats, not the prediction path (CLAUDE.md §1 rule 1).
"""

from __future__ import annotations

import numpy as np
from convert import intsim
from convert.intsim import SCORE_LIMIT, IntModel, layernorm, rescale_exact, rescale_float

R = intsim.R
F32_EXACT_WIDTH = 1024  # 127 * 127 * 1024 = 16,516,096 < 2^24 = 16,777,216


class FastModel:
    """The tensors of an `IntModel` as float32 copies for exact products."""

    def __init__(self, im: IntModel) -> None:
        self.im = im
        self.w32: dict[str, np.ndarray] = {}
        for li in range(im.n_layer):
            for site in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
                name = f"h.{li}.{site}.w"
                w = im.arr(name)
                if int(np.abs(w.astype(np.int64)).max()) > 127:
                    raise ValueError(f"{name} is not i8")
                self.w32[name] = np.ascontiguousarray(w.astype(np.float32).T)
        self.wte = im.arr("wte").astype(np.int64)
        self.wte32_t = np.ascontiguousarray(im.arr("wte").astype(np.float32).T)


def _dot_exact(a: np.ndarray, wt32: np.ndarray) -> np.ndarray:
    """a [t, n] (|a| <= 127) times wt32 [n, out] (|w| <= 127), exact, as int64."""
    n = a.shape[1]
    a32 = a.astype(np.float32)
    if n <= F32_EXACT_WIDTH:
        return (a32 @ wt32).astype(np.int64)
    acc = np.zeros((a.shape[0], wt32.shape[1]), dtype=np.int64)
    for s in range(0, n, F32_EXACT_WIDTH):
        acc += (a32[:, s : s + F32_EXACT_WIDTH] @ wt32[s : s + F32_EXACT_WIDTH]).astype(np.int64)
    return acc


def _linear(fm: FastModel, prefix: str, a: np.ndarray) -> np.ndarray:
    im = fm.im
    if int(np.abs(a).max(initial=0)) > 127:
        raise ValueError(f"{prefix}: the input is not i8")
    acc = _dot_exact(a, fm.w32[prefix + ".w"])
    m, s = im.resc(prefix + ".rescale")
    return rescale_exact(acc + im.arr(prefix + ".b").astype(np.int64), m[None, :], s)


def forward(fm: FastModel, tokens: np.ndarray) -> np.ndarray:
    """Return the u64 weights [T, V] of the window `tokens` (position 0 first)."""
    return weights_of(fm, final_ln(fm, tokens))


def forward_chunks(fm: FastModel, tokens: np.ndarray, rows: int = 128):
    """Yield (first row, u64 weights [rows, V]) for the window, to limit the memory."""
    a = final_ln(fm, tokens)
    for r in range(0, len(a), rows):
        yield r, weights_of(fm, a[r : r + rows])


def weights_of(fm: FastModel, a: np.ndarray) -> np.ndarray:
    """The u64 weights of the rows `a` of the ln_f output (plan §7.6, ADR 0004 section 5)."""
    im = fm.im
    acc = _dot_exact(a, fm.wte32_t)
    lm, ls = im.resc("wte.rescale_logits")
    lg = np.clip(rescale_exact(acc, lm[None, :], ls), -SCORE_LIMIT, SCORE_LIMIT)
    lt = im.arr("logits_exp").astype(np.uint64)
    idx = np.clip(lg.max(axis=1, keepdims=True) - lg, 0, len(lt) - 1)
    return lt[idx]


def final_ln(fm: FastModel, tokens: np.ndarray) -> np.ndarray:
    """The ln_f output [T, d] (i8 values) of the window `tokens`."""
    im = fm.im
    t = len(tokens)
    d, nh = im.d, im.n_head
    hd = d // nh
    em, es = im.resc("wte.rescale_embed")
    pm, ps = im.resc("wpe.rescale")
    wpe = im.arr("wpe").astype(np.int64)
    x = rescale_exact(fm.wte[tokens], em[tokens][:, None], es) + rescale_exact(
        wpe[:t], pm[:t][:, None], ps)
    x = np.clip(x, -R, R)
    attn_exp = im.arr("attn_exp").astype(np.int64)
    mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    for li in range(im.n_layer):
        p = f"h.{li}"
        a = layernorm(im, p + ".ln_1", x)
        qkv = np.clip(_linear(fm, p + ".attn.c_attn", a), -127, 127)
        q, k, v = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
        sm, ss = im.resc(p + ".attn.score.rescale")
        om, os_ = im.resc(p + ".attn.out.rescale")
        att = np.empty((t, d), dtype=np.int64)
        for h in range(nh):
            qh = q[:, h * hd : (h + 1) * hd].astype(np.float32)
            kh = k[:, h * hd : (h + 1) * hd].astype(np.float32)
            vh = v[:, h * hd : (h + 1) * hd].astype(np.float64)
            dot = (qh @ kh.T).astype(np.int64)  # exact: 127 * 127 * 64 < 2^24
            sc = np.clip(rescale_exact(dot, sm[0], ss), -SCORE_LIMIT, SCORE_LIMIT)
            sc = np.where(mask, np.int64(-SCORE_LIMIT) * 4, sc)
            mx = sc.max(axis=1, keepdims=True)
            e = attn_exp[np.clip(mx - sc, 0, len(attn_exp) - 1)]
            e = np.where(mask, 0, e)
            pr = np.floor_divide(e << np.int64(im.p), e.sum(axis=1, keepdims=True))
            acc = pr.astype(np.float64) @ vh  # exact: |acc| < 2^38 (as convert.intsim)
            att[:, h * hd : (h + 1) * hd] = np.clip(rescale_float(acc, om[0], os_), -127, 127)
        x = np.clip(x + _linear(fm, p + ".attn.c_proj", att), -R, R)
        a = layernorm(im, p + ".ln_2", x)
        gel = im.arr(p + ".mlp.gelu").astype(np.int64)
        half = len(gel) // 2
        f = np.clip(_linear(fm, p + ".mlp.c_fc", a), -half, half - 1)
        g = gel[f + half]
        x = np.clip(x + _linear(fm, p + ".mlp.c_proj", g), -R, R)
    return layernorm(im, "ln_f", x)


def load() -> FastModel:
    """Build the `clip` model in memory (as `convert.gpt2`) and return its fast form."""
    from baseline._paths import MODEL_DIR
    from baseline.bpe import Tokenizer
    from convert import recipe as rc
    from convert.build import build
    from convert.calibrate import STATS_PATH
    from convert.gpt2 import vocabulary_and_merges
    from convert.model import load_gpt2

    tok = Tokenizer()
    vocab, merges = vocabulary_and_merges(tok)
    sc = rc.scales(rc.load_stats(STATS_PATH), rc.RECIPES["clip"])
    hsm = build(load_gpt2(MODEL_DIR / "model.safetensors"), sc, vocab, merges)
    return FastModel(IntModel(hsm))


def check() -> None:
    """Compare `forward` with `convert.intsim.forward` on two windows, bit for bit."""
    import time

    fm = load()
    rng = np.random.default_rng(20261018)
    for n in (1024, 300):
        toks = rng.integers(0, 50256, n)
        t0 = time.time()
        a = intsim.forward(fm.im, toks)
        t1 = time.time()
        b = np.concatenate([w for _, w in forward_chunks(fm, toks, 200)])
        t2 = time.time()
        same = bool(np.array_equal(a, b))
        print(f"window {n}: equal {same}, intsim {t1 - t0:.1f} s, fastsim {t2 - t1:.1f} s",
              flush=True)
        if not same:
            raise SystemExit("fastsim differs from intsim")


if __name__ == "__main__":
    check()
