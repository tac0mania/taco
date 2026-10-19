"""Tests for the exact converter arithmetic (ADR 0014).

Run: uv run --project tools python -m convert.test_exact

The tests check the tie rule (ties, negative values, subnormals, zero, the largest |w|), the
exact value of float bits, and that the exact rules and the float rules of version 1
(`convert.numerics`) agree on random inputs away from the rounding borders.
"""

from __future__ import annotations

import sys
import time
from decimal import Decimal
from fractions import Fraction

import numpy as np

from convert import exact as ex
from convert import numerics as nm
from convert.rng import SplitMix64

F32_MAX = np.float32(np.finfo(np.float32).max)
F32_TINY = np.float32(np.finfo(np.float32).smallest_subnormal)
F64_TINY = np.float64(np.finfo(np.float64).smallest_subnormal)


def ref_quantise(w: np.ndarray, qmax: int) -> np.ndarray:
    """The reference: R(qmax * w / max|row|) with Fractions, one element at a time."""
    out = np.zeros(w.shape, dtype=np.int64)
    for i, row in enumerate(w):
        fr = [Fraction(float(x)) for x in row]
        mx = max(abs(x) for x in fr)
        for j, x in enumerate(fr):
            out[i, j] = ex.rhe(qmax * x / mx) if mx else 0
    return out


def log_uniform(rng: np.random.Generator, shape, lo: float, hi: float, dtype) -> np.ndarray:
    mag = np.exp(rng.uniform(np.log(lo), np.log(hi), shape))
    sign = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
    return (sign * mag).astype(dtype)


def test_rhe_ties() -> None:
    cases = {
        Fraction(1, 2): 0, Fraction(3, 2): 2, Fraction(5, 2): 2, Fraction(7, 2): 4,
        Fraction(-1, 2): 0, Fraction(-3, 2): -2, Fraction(-5, 2): -2, Fraction(-7, 2): -4,
        Fraction(0): 0, Fraction(1, 3): 0, Fraction(2, 3): 1, Fraction(-2, 3): -1,
        Fraction(253, 2): 126, Fraction(255, 2): 128, Fraction(-255, 2): -128,
    }
    for x, want in cases.items():
        assert ex.rhe(x) == want, (x, ex.rhe(x), want)
    num = np.array([1, 3, 5, 7, 2, 9, 0], dtype=np.int64)
    den = np.array([2, 2, 2, 2, 3, 4, 5], dtype=np.int64)
    assert ex.rhe_vec(num, den).tolist() == [0, 2, 2, 4, 1, 2, 0]


def test_float_parts() -> None:
    vals32 = np.array([0.0, -0.0, F32_TINY, -F32_TINY, F32_TINY * 7,
                       np.float32(1.1754942e-38), np.float32(1.17549435e-38), 1.0, -2.5,
                       0.1, F32_MAX, -F32_MAX], dtype=np.float32)
    rng = np.random.default_rng(1)
    vals32 = np.concatenate([vals32, log_uniform(rng, 2000, 1e-44, 3e38, np.float32)])
    got = ex.fractions_of(vals32)
    for g, v in zip(got, vals32, strict=True):
        assert g == Fraction(float(v)), (v, g)
    vals64 = np.concatenate([np.array([0.0, F64_TINY, -F64_TINY * 3, 2.2250738585072014e-308,
                                       1.7976931348623157e308]),
                             log_uniform(rng, 2000, 1e-320, 1e300, np.float64)])
    for g, v in zip(ex.fractions_of(vals64), vals64, strict=True):
        assert g == Fraction(float(v)), (v, g)
    for bad in (np.inf, -np.inf, np.nan):
        try:
            ex.float_parts(np.array([bad], dtype=np.float32))
        except ValueError:
            continue
        raise AssertionError(f"float_parts accepted {bad}")


def test_quantise_ties_and_edges() -> None:
    # 127 * w / 254 = w / 2: the ties 1.5, 2.5, 3.5 and their negative values.
    w = np.array([[254.0, 3.0, 5.0, 7.0, -3.0, -5.0, -7.0, 127.0, 1.0]], dtype=np.float32)
    q, s = ex.quantise_rows(w)
    assert q.tolist() == [[127, 2, 2, 4, -2, -2, -4, 64, 0]], q
    assert s == [Fraction(2)]
    # Zero: a row of zeros has the scale 1 and q = 0. -0.0 is zero.
    q, s = ex.quantise_rows(np.array([[0.0, -0.0, 0.0]], dtype=np.float32))
    assert q.tolist() == [[0, 0, 0]] and s == [Fraction(1)]
    # The largest |w|: f32 max and its half (an exact tie 63.5 -> 64).
    w = np.array([[F32_MAX, F32_MAX / np.float32(2), -F32_MAX, -F32_MAX / np.float32(2)]],
                 dtype=np.float32)
    q, s = ex.quantise_rows(w)
    assert q.tolist() == [[127, 64, -127, -64]], q
    assert s == [Fraction(float(F32_MAX)) / 127]
    # Subnormals: a row of subnormals, and a normal row with subnormal elements.
    t = F32_TINY
    w = np.array([[t * 254, t * 3, t * 5, -t * 7, t]], dtype=np.float32)
    q, _ = ex.quantise_rows(w)
    assert q.tolist() == [[127, 2, 2, -4, 0]], q
    w = np.array([[np.float32(1.1754944e-38) * 254, t * 3, np.float32(1e-38)]], dtype=np.float32)
    assert ex.quantise_rows(w)[0].tolist() == ref_quantise(w, 127).tolist()
    # The cut: elements 2^-7 to 2^-10 of the row maximum (R(.) goes to 0 near 2^-8).
    row = [1.0] + [2.0 ** -k * (1 + i / 8) for k in range(6, 12) for i in range(8)]
    w = np.array([row, [-x for x in row]], dtype=np.float32)
    assert ex.quantise_rows(w)[0].tolist() == ref_quantise(w, 127).tolist()
    w64 = np.array([row], dtype=np.float64)
    assert ex.quantise_rows(w64, 32767)[0].tolist() == ref_quantise(w64, 32767).tolist()


def test_quantise_matches_reference() -> None:
    rng = np.random.default_rng(2)
    for dtype, qmax in ((np.float32, 127), (np.float32, 32767), (np.float64, 127),
                        (np.float64, 32767)):
        w = log_uniform(rng, (12, 40), 1e-12, 10.0, dtype)
        w[3] = 0.0
        w[5, :10] = dtype(0.0)
        got = ex.quantise_rows(w, qmax)[0]
        assert got.tolist() == ref_quantise(w, qmax).tolist(), (dtype, qmax)


def test_exact_agrees_with_float_away_from_borders() -> None:
    """The exact rules and the version 1 float rules on random float32 weights."""
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((512, 768)) * 0.14).astype(np.float32)
    qe, se = ex.quantise_rows(w)
    qf, sf = nm.quantise_rows(w)
    # Distance of the exact value 127 * w / max to the nearest tie k + 1/2.
    mx = np.abs(w.astype(np.float64)).max(axis=1, keepdims=True)
    y = np.abs(127.0 * w.astype(np.float64) / mx)
    far = np.abs(y - np.floor(y) - 0.5) > 1e-9
    assert far.mean() > 0.999
    assert np.array_equal(qe[far], qf[far].astype(np.int64))
    for a, b in zip(se, sf, strict=True):
        assert abs(float(a) - b) <= 2e-16 * b
    # Rescale multipliers.
    ratios = np.exp(rng.uniform(-30.0, 10.0, 3000))
    mf, sfl = nm.quant_mult_rows(ratios)
    me, sex = ex.quant_mult_rows([Fraction(float(r)) for r in ratios])
    assert sex == sfl
    yy = ratios * 2.0**sex
    far = np.abs(yy - np.floor(yy) - 0.5) > 1e-4
    assert np.array_equal(me[far], mf[far])
    # Ranges and the residual rule.
    for x in np.exp(rng.uniform(-25.0, 9.0, 3000)):
        assert ex.round_range(Fraction(float(x))) == Fraction(nm.round_range(float(x)))
    for x in np.exp(rng.uniform(-6.0, 3.0, 300)):
        f = nm.round_range(float(x) / nm.pow2(12) / 2.0**0.5)
        assert ex.round_range_sq(Fraction(float(x)) ** 2 / (1 << 25)) == Fraction(f)
    # The grid points themselves: v rounds to v.
    for j in range(16):
        v = Fraction(ex.GRID[j], 1 << 52) * 8
        assert ex.round_range(v) == v == Fraction(nm.round_range(float(v)))


def test_tables_agree_with_float() -> None:
    assert ex.exp_table(30, 8192, -8) == nm.attn_exp_table().astype(np.int64).tolist()
    assert ex.exp_table(40, 8192, -8) == [int(x) for x in nm.logits_exp_table()]
    for bits, s_in, s_out in ((8, Fraction(13, 127), Fraction(3, 127)),
                              (16, Fraction(13, 32767), Fraction(3, 127))):
        got = ex.gelu_table(bits, s_in, s_out)
        assert np.array_equal(got, nm.gelu_table(bits, float(s_in), float(s_out)))
    # The negative half from the identity gelu(-v) = gelu(v) - v, against a direct value.
    ctx = ex.context()
    c = ctx.sqrt(ctx.divide(Decimal(2), ex._pi(ctx)))
    v = Decimal("1.37")
    diff = ctx.subtract(ex._gelu(-v, c, ctx), ctx.subtract(ex._gelu(v, c, ctx), v))
    assert abs(diff) < Decimal("1e-45"), diff


def test_quant_mult_over_sqrt() -> None:
    a = Fraction(12345, 678)
    assert ex.quant_mult_over_sqrt(a, 64) == ex.quant_mult(a / 8)
    ctx = ex.context()
    for d in (2, 3, 80, 96, 128):
        m, s = ex.quant_mult_over_sqrt(a, d)
        y = ctx.divide(ctx.multiply(Decimal(a.numerator), Decimal(2) ** s),
                       ctx.multiply(Decimal(a.denominator), ctx.sqrt(Decimal(d))))
        assert m == int(y.to_integral_value()), (d, m, y)
        assert 2**30 <= m < 2**31


def test_matvec() -> None:
    rng = np.random.default_rng(4)
    q = rng.integers(-127, 128, (7, 33))
    v = [Fraction(int(rng.integers(-2**62, 2**62)) * int(rng.integers(1, 2**40)),
                  2 ** int(rng.integers(0, 90)) * 127) for _ in range(33)]
    got = ex.matvec(q, v)
    want = [sum(int(q[r, c]) * v[c] for c in range(33)) for r in range(7)]
    assert got == want


def test_binary64_of_text() -> None:
    texts = ["0.1", "1e-5", "3055.42333984375", "5e-324", "2.4703282292062328e-324",
             "1.7976931348623157e308", "9007199254740993", "9007199254740995", "-0.0", "0"]
    rng = np.random.default_rng(5)
    texts += [repr(float(x)) for x in log_uniform(rng, 3000, 1e-300, 1e300, np.float64)]
    texts += [f"{x:.17e}" for x in rng.standard_normal(500)]
    for t in texts:
        assert ex.binary64_of_text(t) == Fraction(float(t)), t


def test_tiny_generator_is_ieee() -> None:
    """The tiny weights: the float generator gives the correctly rounded binary64 results."""
    a, b = SplitMix64(99), SplitMix64(99)
    for _ in range(300):
        s = Fraction(0)
        for _ in range(12):
            s = ex.binary64_round(s + Fraction(b.next_u64() >> 11, 1 << 53))
        n = ex.binary64_round(s - 6)
        exact = ex.binary64_round(1 + ex.binary64_round(Fraction(0.1) * n))
        got = 1.0 + 0.1 * a.normal()
        assert Fraction(got) == exact


def test_no_float_model_guard() -> None:
    from convert import model

    with ex.no_float_model():
        try:
            model.forward(None, None)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the guard did not refuse the float model")
    assert model.forward.__name__ == "forward"


def test_wide_gelu_table() -> None:
    """The i16 GELU table of hsm version 2 (ADR 0004 addendum B) against the float rule."""
    s_in, s_out = Fraction(13, 32767), Fraction(13, 32767)
    got = ex.gelu_table(16, s_in, s_out, out_max=ex.I16_MAX)
    assert got.dtype == np.int16 and len(got) == 65536
    x = np.arange(-32768, 32768, dtype=np.float64) * float(s_in)
    want = np.clip(np.rint(nm.gelu(x) / float(s_out)), -32767, 32767)
    assert np.array_equal(got.astype(np.float64), want)
    assert got.min() >= -32767 and got.max() <= 32767


def test_hsm_version_rule() -> None:
    """hsm version 2 if and only if the model has ln_f_bits or an i16 GELU table."""
    from convert.hsmfile import HsmModel, gpt2_config, version_of

    cfg = dict(n_layer=1, n_head=1, d_model=4, d_ff=8, vocab_size=2, n_ctx=4, eot_token=1,
               ln_k=48, ln_j=8, ln_eps=1, softmax_p=30)
    m = HsmModel(gpt2_config(**cfg), [b"a", b"b"])
    m.add("h.0.mlp.gelu", "i8", np.zeros(256, dtype=np.int8))
    assert version_of(m) == 1
    m = HsmModel(gpt2_config(**cfg, ln_f_bits=16), [b"a", b"b"])
    assert version_of(m) == 2
    m = HsmModel(gpt2_config(**cfg), [b"a", b"b"])
    m.add("h.0.mlp.gelu", "i16", np.zeros(256, dtype=np.int16))
    assert version_of(m) == 2
    m = HsmModel(gpt2_config(**cfg, ln_beta_frac=8), [b"a", b"b"])
    assert version_of(m) == 2 and m.config["constants"]["ln_beta_frac"] == 8


def test_selfcheck() -> None:
    from convert.selfcheck import Inputs, SelfCheckError, check

    inp = Inputs("gpt2", "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
                 "clip", "8f7731d98c363316488d515a523aa5b73123e59b35705bc744a2c55bf90d12c2")
    check(inp, "aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2",
          127237568, False)
    for sha, inputs in (("0" * 64, inp), ("aaa0", Inputs("x", "y", "clip", "z"))):
        try:
            check(inputs, sha, 1, False)
        except SelfCheckError:
            continue
        raise AssertionError("the self-check accepted a wrong hash")
    check(inp, "0" * 64, 1, True)  # --new-hash: no error


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t0 = time.time()
        t()
        print(f"ok {t.__name__} ({time.time() - t0:.1f} s)")
    print(f"{len(tests)} tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
