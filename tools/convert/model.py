"""A float GPT-2 for the converter: weights in [out, in] form, a full-sequence forward pass
with observation points for calibration, and optional fake quantisation for measurement.

The mathematics is the same as `baseline.gpt2_np`. This module adds the points where HardShell
has an integer activation (the "sites"), so that the calibration can measure each one and the
quality measurement can quantise each one alone.

Sites (one name for each activation tensor of HardShell, plan §7.1):

- `res`: the residual stream after the embedding and after each addition (i32).
- `h.{l}.ln_1`, `h.{l}.ln_2`, `ln_f`: LayerNorm outputs (i8).
- `h.{l}.q`, `h.{l}.k`, `h.{l}.v`: attention projections (i8).
- `h.{l}.att`: the attention output before `c_proj` (i8).
- `h.{l}.fc`: the GELU input (i16).
- `h.{l}.gelu`: the GELU output (i8).

NOTE: this is a measurement tool with floats, not the prediction path (CLAUDE.md §1 rule 1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from convert.numerics import gelu as det_gelu

LN_EPS = 1e-5


@dataclass
class FBlock:
    """The float weights of one block. Matrices are [out, in]."""

    ln1_g: np.ndarray
    ln1_b: np.ndarray
    attn_w: np.ndarray
    attn_b: np.ndarray
    proj_w: np.ndarray
    proj_b: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray
    fc_w: np.ndarray
    fc_b: np.ndarray
    mlp_w: np.ndarray
    mlp_b: np.ndarray


@dataclass
class FModel:
    """A float GPT-2 model. `wte` is tied: the embedding and the output projection."""

    wte: np.ndarray
    wpe: np.ndarray
    blocks: list[FBlock]
    lnf_g: np.ndarray
    lnf_b: np.ndarray
    n_head: int
    eot: int

    @property
    def d_model(self) -> int:
        return self.wte.shape[1]

    @property
    def d_ff(self) -> int:
        return self.blocks[0].fc_w.shape[0]

    @property
    def vocab(self) -> int:
        return self.wte.shape[0]

    @property
    def n_ctx(self) -> int:
        return self.wpe.shape[0]


def load_gpt2(path: Path, dtype: type = np.float32, n_head: int = 12) -> FModel:
    """Load a GPT-2-architecture model from a safetensors file. Conv1D weights become [out, in].

    The file can be the published GPT-2 small file (names `h.0.ln_1.weight`, ...) or a file
    that `transformers` writes (the same names after `transformer.`). The number of blocks is
    the number of `h.{i}.ln_1.weight` tensors. If the file has `lm_head.weight`, it must be
    equal to `wte.weight`, because the model file stores one tied matrix.
    """
    from safetensors.numpy import safe_open

    with safe_open(str(path), "np") as f:
        keys = set(f.keys())
        pre = "transformer." if "transformer.wte.weight" in keys else ""

        def get(name: str) -> np.ndarray:
            return np.ascontiguousarray(f.get_tensor(pre + name), dtype=dtype)

        def get_t(name: str) -> np.ndarray:
            return np.ascontiguousarray(f.get_tensor(pre + name).T, dtype=dtype)

        n_layer = 0
        while f"{pre}h.{n_layer}.ln_1.weight" in keys:
            n_layer += 1
        if "lm_head.weight" in keys and not np.array_equal(
            f.get_tensor("lm_head.weight"), f.get_tensor(pre + "wte.weight")
        ):
            raise ValueError("lm_head.weight is not wte.weight: the model is not tied")
        blocks = []
        for i in range(n_layer):
            p = f"h.{i}."
            blocks.append(
                FBlock(
                    get(p + "ln_1.weight"), get(p + "ln_1.bias"),
                    get_t(p + "attn.c_attn.weight"), get(p + "attn.c_attn.bias"),
                    get_t(p + "attn.c_proj.weight"), get(p + "attn.c_proj.bias"),
                    get(p + "ln_2.weight"), get(p + "ln_2.bias"),
                    get_t(p + "mlp.c_fc.weight"), get(p + "mlp.c_fc.bias"),
                    get_t(p + "mlp.c_proj.weight"), get(p + "mlp.c_proj.bias"),
                )
            )
        return FModel(
            wte=get("wte.weight"), wpe=get("wpe.weight"), blocks=blocks,
            lnf_g=get("ln_f.weight"), lnf_b=get("ln_f.bias"), n_head=n_head, eot=50256,
        )


Hook = Callable[[str, np.ndarray], np.ndarray | None]


def _ln(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    xc = x - mean
    var = (xc * xc).mean(axis=-1, keepdims=True)
    return xc / np.sqrt(var + x.dtype.type(LN_EPS)) * g + b


def _gelu_fast(x: np.ndarray) -> np.ndarray:
    c = x.dtype.type(np.sqrt(2.0 / np.pi))
    return x.dtype.type(0.5) * x * (1 + np.tanh(c * (x + x.dtype.type(0.044715) * x**3)))


def forward(
    m: FModel, tokens: np.ndarray, hook: Hook | None = None, det_gelu_fn: bool = False
) -> np.ndarray:
    """Return the logits [T, V] of the full-sequence causal pass.

    `hook(site, x)` sees each site. If it returns an array, that array replaces `x` (fake
    quantisation). `det_gelu_fn` selects the GELU of `convert.numerics` (basic operations only)
    instead of np.tanh.
    """

    def h(site: str, x: np.ndarray) -> np.ndarray:
        if hook is None:
            return x
        y = hook(site, x)
        return x if y is None else y

    t = len(tokens)
    d = m.d_model
    nh = m.n_head
    hd = d // nh
    dt = m.wte.dtype.type
    gelu_fn = (lambda v: det_gelu(v).astype(v.dtype)) if det_gelu_fn else _gelu_fast
    x = h("res", m.wte[tokens] + m.wpe[:t])
    mask = np.triu(np.full((t, t), -np.inf, dtype=m.wte.dtype), k=1)
    scale = dt(1.0 / np.sqrt(hd))
    for li, b in enumerate(m.blocks):
        p = f"h.{li}."
        a = h(p + "ln_1", _ln(x, b.ln1_g, b.ln1_b))
        qkv = a @ b.attn_w.T + b.attn_b
        q = h(p + "q", qkv[:, :d])
        k = h(p + "k", qkv[:, d : 2 * d])
        v = h(p + "v", qkv[:, 2 * d :])
        qh = q.reshape(t, nh, hd).transpose(1, 0, 2)
        kh = k.reshape(t, nh, hd).transpose(1, 2, 0)
        vh = v.reshape(t, nh, hd).transpose(1, 0, 2)
        s = (qh @ kh) * scale + mask
        s = s - s.max(axis=-1, keepdims=True)
        e = np.exp(s)
        att = e / e.sum(axis=-1, keepdims=True)
        y = (att @ vh).transpose(1, 0, 2).reshape(t, d)
        y = h(p + "att", y)
        x = h("res", x + (y @ b.proj_w.T + b.proj_b))
        a = h(p + "ln_2", _ln(x, b.ln2_g, b.ln2_b))
        f = h(p + "fc", a @ b.fc_w.T + b.fc_b)
        g = h(p + "gelu", gelu_fn(f))
        x = h("res", x + (g @ b.mlp_w.T + b.mlp_b))
    a = h("ln_f", _ln(x, m.lnf_g, m.lnf_b))
    return a @ m.wte.T


def log2_probs_of(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """-log2 p(target) for each row, in float64."""
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return (lse - z[np.arange(len(targets)), targets]) / np.log(2.0)


def with_weights(m: FModel, f: Callable[[np.ndarray], np.ndarray]) -> FModel:
    """A copy of `m` with `f` applied to every matrix (wte, wpe and the four linear layers)."""
    blocks = [
        replace(b, attn_w=f(b.attn_w), proj_w=f(b.proj_w), fc_w=f(b.fc_w), mlp_w=f(b.mlp_w))
        for b in m.blocks
    ]
    return replace(m, wte=f(m.wte), wpe=f(m.wpe), blocks=blocks)
