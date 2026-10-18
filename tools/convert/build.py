"""Build the integer model: float weights and activation ranges to the tensors of a `.hsm` file.

The tensor names, types, shapes and order are the table in the module documentation of
`crates/hardshell/src/gpt2/mod.rs`. The rules for each value are in ADR 0004. The Rust test
builder `crates/hardshell/tests/common/builder.rs` uses the same rules.

For each linear layer with input scale `s_in`, weight row scale `w_r` and output scale
`s_out(r)`: the accumulator unit is `s_in * w_r`, the bias is `round(b / (s_in * w_r))`, and the
rescale ratio is `s_in * w_r / s_out(r)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from convert import numerics as nm
from convert.hsmfile import HsmModel, gpt2_config
from convert.model import FModel


@dataclass
class Scales:
    """The real value of one unit of each integer activation.

    `res` is the residual scale. `site[name]` is the scale of each other site of
    `convert.model`. A scale is `range / 127` for an i8 site, and `range / 32767` for the GELU
    input of a layer with an i16 GELU table.
    """

    res: float
    site: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_ranges(cls, res: float, ranges: dict[str, float],
                    gelu_bits: dict[int, int] | None = None) -> Scales:
        """Scales from ranges. `gelu_bits[l]` is 8 or 16 (the default) for layer `l`."""
        s = {}
        for k, r in ranges.items():
            qmax = nm.I8_MAX
            if k.endswith(".fc"):
                layer = int(k.split(".")[1])
                qmax = nm.I16_MAX if (gelu_bits or {}).get(layer, 16) == 16 else nm.I8_MAX
            s[k] = r / qmax
        return cls(res, s)


def _linear(out: HsmModel, prefix: str, w: np.ndarray, b: np.ndarray, s_in: float,
            s_out: np.ndarray, in_err: np.ndarray | None = None) -> None:
    """A linear layer. `in_err` is a known constant error of the input (float units, one value
    for each input channel); the bias subtracts its effect `W_q . in_err`."""
    q, ws = nm.quantise_rows(w)
    acc = s_in * ws
    b = b.astype(np.float64)
    if in_err is not None:
        b = b - (q.astype(np.float64) @ in_err) * ws
    out.add(f"{prefix}.w", "i8", q)
    out.add(f"{prefix}.b", "i32", nm.round_i32(b / acc))
    m, s = nm.quant_mult_rows(acc / s_out)
    out.add(f"{prefix}.rescale", "i32", m, s)


def _layernorm(out: HsmModel, prefix: str, g: np.ndarray, b: np.ndarray, s_out: float
               ) -> np.ndarray:
    """A LayerNorm. Returns the rounding error of beta, `round(b / s) * s - b`, in float units.

    HardShell adds beta after the rescale (plan §7.3 step 4), so the output has two roundings.
    The beta rounding is the same at every position: a constant error of up to half a unit in
    each channel. The next linear layer can remove it (`_linear`, `in_err`).
    """
    g = g.astype(np.float64)
    s_gamma = float(np.abs(g).max()) / nm.I16_MAX
    gq = np.rint(g / s_gamma).astype(np.int16)
    out.add(f"{prefix}.g", "i16", gq)
    bq = nm.round_i32(b.astype(np.float64) / s_out)
    out.add(f"{prefix}.b", "i32", bq)
    m, s = nm.layernorm_rescale(s_gamma, s_out)
    out.add(f"{prefix}.rescale", "i32", np.array([m]), s)
    return bq.astype(np.float64) * s_out - b.astype(np.float64)


def build(fm: FModel, sc: Scales, vocabulary: list[bytes], merges: list[tuple[int, int]],
          gelu_bits: dict[int, int] | None = None, beta_correction: bool = True) -> HsmModel:
    """Make all tensors of the model file. Cost: one pass over the weights.

    `gelu_bits[l]` selects the GELU table of layer `l`: 16 (65,536 entries, the default) or 8
    (256 entries). It must agree with the scales (`Scales.from_ranges`). `beta_correction`
    removes the constant error of the LayerNorm beta rounding in the next linear layer.
    """
    d, f, nh = fm.d_model, fm.d_ff, fm.n_head
    hd = d // nh
    delta = nm.pow2(nm.DELTA_LOG2)
    cfg = gpt2_config(
        n_layer=len(fm.blocks), n_head=nh, d_model=d, d_ff=f, vocab_size=fm.vocab,
        n_ctx=fm.n_ctx, eot_token=fm.eot, ln_k=nm.LN_K, ln_j=nm.LN_J,
        ln_eps=nm.ln_eps_units(sc.res), softmax_p=nm.SOFTMAX_P,
    )
    if len(vocabulary) != fm.vocab:
        raise ValueError("the vocabulary size is not the wte row count")
    out = HsmModel(cfg, vocabulary, merges)

    s_lnf = sc.site["ln_f"]
    wq, ws = nm.quantise_rows(fm.wte)
    out.add("wte", "i8", wq)
    m, s = nm.quant_mult_rows(ws / sc.res)
    out.add("wte.rescale_embed", "i32", m, s)
    m, s = nm.quant_mult_rows(ws * s_lnf / delta)
    out.add("wte.rescale_logits", "i32", m, s)
    pq, ps = nm.quantise_rows(fm.wpe)
    out.add("wpe", "i8", pq)
    m, s = nm.quant_mult_rows(ps / sc.res)
    out.add("wpe.rescale", "i32", m, s)

    for li, b in enumerate(fm.blocks):
        p = f"h.{li}"
        g = lambda name: sc.site[f"{p}.{name}"]  # noqa: E731
        e1 = _layernorm(out, f"{p}.ln_1", b.ln1_g, b.ln1_b, g("ln_1"))
        qkv_out = np.concatenate([np.full(d, g("q")), np.full(d, g("k")), np.full(d, g("v"))])
        _linear(out, f"{p}.attn.c_attn", b.attn_w, b.attn_b, g("ln_1"), qkv_out,
                e1 if beta_correction else None)
        mm, ss = nm.quant_mult(g("q") * g("k") / (np.sqrt(float(hd)) * delta))
        out.add(f"{p}.attn.score.rescale", "i32", np.array([mm]), ss)
        mm, ss = nm.quant_mult(g("v") / (nm.pow2(nm.SOFTMAX_P) * g("att")))
        out.add(f"{p}.attn.out.rescale", "i32", np.array([mm]), ss)
        _linear(out, f"{p}.attn.c_proj", b.proj_w, b.proj_b, g("att"), np.full(d, sc.res))
        e2 = _layernorm(out, f"{p}.ln_2", b.ln2_g, b.ln2_b, g("ln_2"))
        _linear(out, f"{p}.mlp.c_fc", b.fc_w, b.fc_b, g("ln_2"), np.full(f, g("fc")),
                e2 if beta_correction else None)
        bits = (gelu_bits or {}).get(li, 16)
        out.add(f"{p}.mlp.gelu", "i8", nm.gelu_table(bits, g("fc"), g("gelu")))
        _linear(out, f"{p}.mlp.c_proj", b.mlp_w, b.mlp_b, g("gelu"), np.full(d, sc.res))
    _layernorm(out, "ln_f", fm.lnf_g, fm.lnf_b, s_lnf)
    out.add("attn_exp", "i32", nm.attn_exp_table())
    out.add("logits_exp", "u64", nm.logits_exp_table())
    return out
