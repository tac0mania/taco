"""The constants of ADR 0004, and the float form of the converter rules of version 1.

The converter does not use the float functions of this module. It uses the exact rules of
`convert.exact` (ADR 0014). The float functions stay for two uses only:

- The measurement tools (`convert.model`, `convert.fq`, `convert.intsim`) use them with
  floats, for example the fake quantisation of `fq`.
- `convert.test_exact` compares the exact rules with these float rules on random inputs.

The float functions use only IEEE 754 basic operations, exact operations and np.rint, but the
result of a chain of float operations near a rounding border can depend on the order of the
operations. ADR 0014 thus requires exact arithmetic for the model file.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

# ADR 0004 constants.
LN_K = 48
LN_J = 8
SOFTMAX_P = 30
DELTA_LOG2 = -8  # One score unit and one logit unit: 2^-8 (ADR 0004 sections 4 and 5).
TABLE_LEN = 8192
ATTN_EXP_E = 30
LOGITS_EXP_E = 40
LN_EPS = 1e-5  # GPT-2 LayerNorm epsilon (published), float form for the float model.
LN_EPS_EXACT = Fraction(1, 100000)  # The same epsilon, exact, for the converter.
I8_MAX = 127
I16_MAX = 32767
I32_MAX = 2**31 - 1
RESIDUAL_LIMIT = 2**30 - 1


def pow2(k: int) -> float:
    """2^k, exact."""
    return float(np.ldexp(1.0, k))


LN2 = 0.6931471805599453  # ln 2, correctly rounded (published).


def det_exp(x: np.ndarray) -> np.ndarray:
    """exp(x) with basic operations only: range reduction by ln 2, then a Taylor series.

    The relative error is about 1e-16 (the Rust test helper `det_exp` uses the same method and
    measures 1e-14 or better against the platform exp).
    """
    x = np.asarray(x, dtype=np.float64)
    k = np.rint(x / LN2)
    r = x - k * LN2
    term = np.ones_like(r)
    total = np.ones_like(r)
    for i in range(1, 22):
        term = term * r / float(i)
        total = total + term
    out = np.ldexp(total, k.astype(np.int64))
    out = np.where(x < -700.0, 0.0, out)
    return np.where(x > 700.0, np.inf, out)


def det_tanh(x: np.ndarray) -> np.ndarray:
    """tanh(x) with basic operations only."""
    x = np.asarray(x, dtype=np.float64)
    xc = np.clip(x, -20.0, 20.0)
    e = det_exp(2.0 * xc)
    return (e - 1.0) / (e + 1.0)


SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2 / pi), correctly rounded (published).


def gelu(x: np.ndarray) -> np.ndarray:
    """GPT-2 GELU, the tanh form ("gelu_new")."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + det_tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


def quant_mult_rows(ratios: np.ndarray) -> tuple[np.ndarray, int]:
    """Rescale constants for positive ratios with one shared shift (ADR 0004 section 6).

    The shift `s` is the largest value up to 126 with `max(r) * 2^s < 2^31 - 1`, so the largest
    multiplier is in [2^30, 2^31). Each multiplier is `m = round(r * 2^s)` (half to even).
    """
    r = np.asarray(ratios, dtype=np.float64).reshape(-1)
    mx = float(r.max())
    if not (mx > 0.0 and mx < I32_MAX) or float(r.min()) < 0.0:
        raise ValueError(f"ratio out of range: min {float(r.min())}, max {mx}")
    s = 0
    while s < 126 and mx * pow2(s + 1) < I32_MAX:
        s += 1
    m = np.rint(r * pow2(s))
    if float(m.max()) > I32_MAX:
        raise ValueError("multiplier overflow")
    return m.astype(np.int32), s


def quant_mult(ratio: float) -> tuple[int, int]:
    """Rescale constants (m, s) for one positive ratio."""
    m, s = quant_mult_rows(np.array([ratio]))
    return int(m[0]), s


def quantise_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric i8 quantisation with one scale for each row (plan §7.1).

    `w` is [rows, cols]. The scale of a row is max|row| / 127 (1.0 for a row of zeros). The
    result is (q, scale): q = round(w / scale), clamped to [-127, 127].
    """
    w = np.asarray(w, dtype=np.float64)
    mx = np.abs(w).max(axis=1)
    scale = np.where(mx > 0.0, mx / I8_MAX, 1.0)
    q = np.clip(np.rint(w / scale[:, None]), -I8_MAX, I8_MAX).astype(np.int8)
    return q, scale


def round_i32(x: np.ndarray) -> np.ndarray:
    """Round to the nearest integer and check the i32 range."""
    q = np.rint(np.asarray(x, dtype=np.float64))
    if q.size and float(np.abs(q).max()) > I32_MAX:
        raise ValueError("value out of the i32 range")
    return q.astype(np.int32)


def exp_table(e: int, length: int) -> np.ndarray:
    """table[i] = round(2^e * exp(-i * 2^DELTA_LOG2)) as float64 integers (plan §7.5, §7.6)."""
    i = np.arange(length, dtype=np.float64)
    return np.rint(pow2(e) * det_exp(-i * pow2(DELTA_LOG2)))


def attn_exp_table() -> np.ndarray:
    """The softmax exponent table: i32, 8,192 entries (ADR 0004 section 4)."""
    return exp_table(ATTN_EXP_E, TABLE_LEN).astype(np.int32)


def logits_exp_table() -> np.ndarray:
    """The logit exponent table: u64, 8,192 entries (ADR 0004 section 5)."""
    return exp_table(LOGITS_EXP_E, TABLE_LEN).astype(np.uint64)


def gelu_table(bits: int, s_in: float, s_out: float) -> np.ndarray:
    """A GELU table for inputs of `bits` bits at scale s_in, with i8 outputs at scale s_out.

    The entry at index `x + 2^(bits-1)` is round(gelu(x * s_in) / s_out), clamped to ±127.
    """
    half = 1 << (bits - 1)
    x = np.arange(-half, half, dtype=np.float64) * s_in
    return np.clip(np.rint(gelu(x) / s_out), -I8_MAX, I8_MAX).astype(np.int8)


def ln_eps_units(s_res: float) -> int:
    """The LayerNorm eps in the units of var * 2^(2J), at least 1 (ADR 0004 section 2)."""
    return max(1, int(np.rint(LN_EPS / (s_res * s_res) * pow2(2 * LN_J))))


def layernorm_rescale(s_gamma: float, s_out: float) -> tuple[int, int]:
    """LayerNorm (m, s): `(x - mean) * inv` is the normalised value times 2^(K-J)."""
    return quant_mult(s_gamma / (pow2(LN_K - LN_J) * s_out))


def round_range(x: float, steps_per_octave: int = 16) -> float:
    """Round a positive calibration range up to the grid 2^(k / steps_per_octave).

    DETERMINISM: calibration runs a float forward pass, whose last bits depend on the machine
    (phase 1). A range on a coarse grid makes the same model file from slightly different
    calibration values. The grid points come from exact operations only (repeated sqrt of 2).
    """
    if not x > 0.0:
        raise ValueError(f"range must be positive: {x}")
    # 2^(1/16) by four square roots of 2: sqrt is correctly rounded.
    step = 2.0
    n = steps_per_octave
    while n > 1:
        step = float(np.sqrt(step))
        n //= 2
    _, e = np.frexp(x)  # x = f * 2^e with 0.5 <= f < 1, exact.
    v = pow2(int(e) - 1)
    while v < x:
        v *= step
    return v
