"""Build the integer model: float weights and activation ranges to the tensors of a `.hsm` file.

The tensor names, types, shapes and order are the table in the module documentation of
`crates/hardshell/src/gpt2/mod.rs`. The rules for each value are in ADR 0004. The Rust test
builder `crates/hardshell/tests/common/builder.rs` uses the same rules.

For each linear layer with input scale `s_in`, weight row scale `w_r` and output scale
`s_out(r)`: the accumulator unit is `s_in * w_r`, the bias is `R(b / (s_in * w_r))`, and the
rescale ratio is `s_in * w_r / s_out(r)`.

DETERMINISM (ADR 0014): every value is exact. The float weights become exact numbers
`m * 2^e` from their bits, every scale is an exact fraction, and every rounding is `R`, round
half to even (`convert.exact`). The tables come from `decimal` at 60 digits. Thus the same
inputs give the same file on every machine. This module does not run the float model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from convert import exact as ex
from convert import numerics as nm
from convert.hsmfile import HsmModel, gpt2_config
from convert.model import FModel


# The site kinds that can have an i16 output (hsm version 2, ADR 0004 addendum B).
WIDE_KINDS = frozenset({"gelu", "ln_f"})


@dataclass
class Scales:
    """The real value of one unit of each integer activation, as exact fractions.

    `res` is the residual scale. `site[name]` is the scale of each other site of
    `convert.model`. A scale is `range / 127` for an i8 site, and `range / 32767` for the GELU
    input of a layer with an i16 GELU table.
    """

    res: Fraction
    site: dict[str, Fraction] = field(default_factory=dict)

    wide: frozenset[str] = frozenset()

    @classmethod
    def from_ranges(cls, res: Fraction, ranges: dict[str, Fraction],
                    gelu_bits: dict[int, int] | None = None,
                    wide: frozenset[str] = frozenset()) -> Scales:
        """Scales from ranges. `gelu_bits[l]` is 8 or 16 (the default) for layer `l`.

        `wide` names the site kinds with an i16 output (`gelu`, `ln_f`; hsm version 2, ADR 0004
        addendum B). Their scale is `range / 32767`.
        """
        if not wide <= WIDE_KINDS:
            raise ValueError(f"only {sorted(WIDE_KINDS)} can be i16, not {sorted(wide)}")
        s = {}
        for k, r in ranges.items():
            qmax = nm.I8_MAX
            if k.endswith(".fc"):
                layer = int(k.split(".")[1])
                qmax = nm.I16_MAX if (gelu_bits or {}).get(layer, 16) == 16 else nm.I8_MAX
            elif (k == "ln_f" and "ln_f" in wide) or (k.endswith(".gelu") and "gelu" in wide):
                qmax = nm.I16_MAX
            s[k] = Fraction(r) / qmax
        return cls(Fraction(res), s, frozenset(wide))

    def floats(self) -> tuple[float, dict[str, float]]:
        """The scales as floats, for measurement tools and records only (not the converter)."""
        return float(self.res), {k: float(v) for k, v in self.site.items()}


def _i32(values: list[int], what: str) -> np.ndarray:
    if values and max(abs(v) for v in values) > nm.I32_MAX:
        raise ValueError(f"{what}: a value is out of the i32 range")
    return np.array(values, dtype=np.int32)


def _linear(out: HsmModel, prefix: str, w: np.ndarray, b: np.ndarray, s_in: Fraction,
            s_out: list[Fraction], in_err: list[Fraction] | None = None) -> None:
    """A linear layer. `in_err` is a known constant error of the input (one exact value for
    each input channel); the bias subtracts its effect `W_q . in_err`.

    The float converter of version 1 calculated `R((b - (W_q . in_err) * w_r) / (s_in * w_r))`.
    The exact form is the same number: `R(b / (s_in * w_r) - (W_q . in_err) / s_in)`.
    """
    q, ws = ex.quantise_rows(w)
    bf = ex.fractions_of(b)
    corr = ex.matvec(q, in_err) if in_err is not None else [Fraction(0)] * len(ws)
    bias = [ex.rhe(bf[r] / (s_in * ws[r]) - corr[r] / s_in) for r in range(len(ws))]
    out.add(f"{prefix}.w", "i8", q.astype(np.int8))
    out.add(f"{prefix}.b", "i32", _i32(bias, prefix + ".b"))
    m, s = ex.quant_mult_rows([s_in * ws[r] / s_out[r] for r in range(len(ws))])
    out.add(f"{prefix}.rescale", "i32", m, s)


def _layernorm(out: HsmModel, prefix: str, g: np.ndarray, b: np.ndarray, s_out: Fraction,
               frac: int = 0) -> list[Fraction]:
    """A LayerNorm. Returns the rounding error of beta, `R(b / s) * s - b`, exact.

    With `frac > 0` (hsm version 2, header constant `ln_beta_frac`), beta is stored in units of
    `s / 2^frac`, and HardShell adds it before the one rounding: the error is then at most
    `2^-(frac+1)` output units.

    HardShell adds beta after the rescale (plan §7.3 step 4), so the output has two roundings.
    The beta rounding is the same at every position: a constant error of up to half a unit in
    each channel. The next linear layer removes it (`_linear`, `in_err`): the beta correction
    of ADR 0004 section 7.
    """
    gq, gs = ex.quantise_rows(np.asarray(g).reshape(1, -1), nm.I16_MAX)
    s_gamma = gs[0]
    out.add(f"{prefix}.g", "i16", gq.reshape(-1).astype(np.int16))
    bf = ex.fractions_of(b)
    unit = s_out / (1 << frac)
    bq = [ex.rhe(x / unit) for x in bf]
    out.add(f"{prefix}.b", "i32", _i32(bq, prefix + ".b"))
    m, s = ex.quant_mult(s_gamma / (Fraction(1 << (nm.LN_K - nm.LN_J)) * s_out))
    out.add(f"{prefix}.rescale", "i32", np.array([m]), s)
    return [q * unit - x for q, x in zip(bq, bf, strict=True)]


def ln_eps_units(s_res: Fraction) -> int:
    """The LayerNorm eps in the units of var * 2^(2J), at least 1 (ADR 0004 section 2).

    `R(eps / s_res^2 * 2^(2J))` with the published GPT-2 eps = 1/100,000 exactly.
    """
    return max(1, ex.rhe(nm.LN_EPS_EXACT / (s_res * s_res) * (1 << (2 * nm.LN_J))))


def build(fm: FModel, sc: Scales, vocabulary: list[bytes], merges: list[tuple[int, int]],
          gelu_bits: dict[int, int] | None = None, beta_correction: bool = True,
          ln_beta_frac: int = 0) -> HsmModel:
    """Make all tensors of the model file. Cost: one pass over the weights.

    `gelu_bits[l]` selects the GELU table of layer `l`: 16 (65,536 entries, the default) or 8
    (256 entries). It must agree with the scales (`Scales.from_ranges`). `sc.wide` selects the
    i16 outputs of hsm version 2: an `i16` GELU table, and `ln_f_bits = 16` in the header.
    `beta_correction` removes the constant error of the LayerNorm beta rounding in the next
    linear layer. `ln_beta_frac > 0` (hsm version 2) stores each LayerNorm beta with that many
    more bits, added before the one rounding of the LayerNorm.
    """
    d, f, nh = fm.d_model, fm.d_ff, fm.n_head
    hd = d // nh
    delta = Fraction(1, 1 << -nm.DELTA_LOG2)
    cfg = gpt2_config(
        n_layer=len(fm.blocks), n_head=nh, d_model=d, d_ff=f, vocab_size=fm.vocab,
        n_ctx=fm.n_ctx, eot_token=fm.eot, ln_k=nm.LN_K, ln_j=nm.LN_J,
        ln_eps=ln_eps_units(sc.res), softmax_p=nm.SOFTMAX_P,
        ln_f_bits=16 if "ln_f" in sc.wide else None,
        ln_beta_frac=ln_beta_frac or None,
    )
    if len(vocabulary) != fm.vocab:
        raise ValueError("the vocabulary size is not the wte row count")
    out = HsmModel(cfg, vocabulary, merges)

    s_lnf = sc.site["ln_f"]
    wq, ws = ex.quantise_rows(fm.wte)
    out.add("wte", "i8", wq.astype(np.int8))
    m, s = ex.quant_mult_rows([x / sc.res for x in ws])
    out.add("wte.rescale_embed", "i32", m, s)
    m, s = ex.quant_mult_rows([x * s_lnf / delta for x in ws])
    out.add("wte.rescale_logits", "i32", m, s)
    del wq
    pq, ps = ex.quantise_rows(fm.wpe)
    out.add("wpe", "i8", pq.astype(np.int8))
    m, s = ex.quant_mult_rows([x / sc.res for x in ps])
    out.add("wpe.rescale", "i32", m, s)

    for li, b in enumerate(fm.blocks):
        p = f"h.{li}"
        g = lambda name: sc.site[f"{p}.{name}"]  # noqa: E731
        e1 = _layernorm(out, f"{p}.ln_1", b.ln1_g, b.ln1_b, g("ln_1"), ln_beta_frac)
        qkv_out = [g("q")] * d + [g("k")] * d + [g("v")] * d
        _linear(out, f"{p}.attn.c_attn", b.attn_w, b.attn_b, g("ln_1"), qkv_out,
                e1 if beta_correction else None)
        mm, ss = ex.quant_mult_over_sqrt(g("q") * g("k") / delta, hd)
        out.add(f"{p}.attn.score.rescale", "i32", np.array([mm]), ss)
        mm, ss = ex.quant_mult(g("v") / (Fraction(1 << nm.SOFTMAX_P) * g("att")))
        out.add(f"{p}.attn.out.rescale", "i32", np.array([mm]), ss)
        _linear(out, f"{p}.attn.c_proj", b.proj_w, b.proj_b, g("att"), [sc.res] * d)
        e2 = _layernorm(out, f"{p}.ln_2", b.ln2_g, b.ln2_b, g("ln_2"), ln_beta_frac)
        _linear(out, f"{p}.mlp.c_fc", b.fc_w, b.fc_b, g("ln_2"), [g("fc")] * f,
                e2 if beta_correction else None)
        bits = (gelu_bits or {}).get(li, 16)
        if "gelu" in sc.wide:
            out.add(f"{p}.mlp.gelu", "i16",
                    ex.gelu_table(bits, g("fc"), g("gelu"), out_max=nm.I16_MAX))
        else:
            out.add(f"{p}.mlp.gelu", "i8", ex.gelu_table(bits, g("fc"), g("gelu")))
        _linear(out, f"{p}.mlp.c_proj", b.mlp_w, b.mlp_b, g("gelu"), [sc.res] * d)
    _layernorm(out, "ln_f", fm.lnf_g, fm.lnf_b, s_lnf, ln_beta_frac)
    out.add("attn_exp", "i32",
            np.array(ex.exp_table(nm.ATTN_EXP_E, nm.TABLE_LEN, nm.DELTA_LOG2), dtype=np.int32))
    out.add("logits_exp", "u64",
            np.array(ex.exp_table(nm.LOGITS_EXP_E, nm.TABLE_LEN, nm.DELTA_LOG2), dtype=np.uint64))
    return out
