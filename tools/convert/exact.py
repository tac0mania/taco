"""Exact arithmetic for the converter (ADR 0014): one result for each input, on every machine.

The converter uses no floating-point arithmetic. This module supplies the parts:

- `float_parts`: the exact value `m * 2^e` of each float32 or float64 value, from its bits.
- `rhe` and `rhe_vec`: the tie rule. Every rounding of the converter uses it.
- `quantise_rows`: symmetric `i8` (or `i16`) quantisation with one scale for each row.
- `quant_mult_rows` and `quant_mult_over_sqrt`: the rescale constants `(m, s)`.
- `round_range` and `round_range_sq`: the grid of the activation ranges.
- `gelu_table`, `exp_table`: the tables, calculated with `decimal` at `PREC` digits.
- `binary64_of_text`: the value of a number in a statistics file.

The tie rule (ADR 0014): `R(x)` is the integer nearest to `x`. If `x` is exactly halfway
between two integers, `R(x)` is the even integer. This is "round half to even", the rule of
`np.rint` that the float converter of version 1 used.

NOTE: every decimal operation goes through `context()` (`ctx.abs`, `ctx.minus` too). A plain
`abs(d)` or `-d` rounds to the global context of 28 digits.

DETERMINISM: every value here is an exact rational number (Python `int` and `Fraction`, or
`int64` arrays with an `INVARIANT:` bound), except the tables. The tables use `decimal`, whose
operations (`+ - * /`, `sqrt`, `exp`) are correctly rounded by the General Decimal Arithmetic
specification. The converter stops when a table entry is nearer than `TIE_MARGIN` to a tie, so
that the rounding error of `decimal` cannot select the result.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from decimal import (
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
from fractions import Fraction
from math import isqrt, lcm

import numpy as np

I8_MAX = 127
I16_MAX = 32767
I32_MAX = 2**31 - 1

# --- The tie rule ---------------------------------------------------------------------------


def rhe(x: Fraction | int) -> int:
    """R(x): round half to even, exact. Cost: one integer division."""
    x = Fraction(x)
    q, r = divmod(x.numerator, x.denominator)  # floor; 0 <= r < denominator
    twice = 2 * r
    if twice > x.denominator or (twice == x.denominator and q & 1):
        q += 1
    return q


def rhe_vec(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """R(num / den) for int64 arrays with num >= 0 and den > 0.

    INVARIANT: 2 * den < 2^63 (the caller checks it), so no step overflows.
    """
    q = num // den
    twice = 2 * (num - q * den)
    up = (twice > den) | ((twice == den) & ((q & 1) == 1))
    return q + up


# --- Float bits to exact values -------------------------------------------------------------

_LAYOUT = {
    # dtype: (unsigned view, fraction bits, exponent bits, exponent of the subnormals)
    np.dtype(np.float32): (np.uint32, 23, 8, -149),
    np.dtype(np.float64): (np.uint64, 52, 11, -1074),
}


def mantissa_bits(dtype: np.dtype) -> int:
    """24 for float32, 53 for float64."""
    return _LAYOUT[np.dtype(dtype)][1] + 1


def float_parts(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(negative, m, e) with a = (-1)^negative * m * 2^e, exact, from the bits of `a`.

    `m` and `e` are int64. A normal value has 2^(p-1) <= m < 2^p (p = 24 or 53). A subnormal
    value and zero have the exponent of the subnormals. Inf and NaN are an error.
    """
    a = np.asarray(a)
    if a.dtype not in _LAYOUT:
        raise ValueError(f"float_parts: dtype {a.dtype} is not float32 or float64")
    view, fbits, ebits, emin = _LAYOUT[a.dtype]
    bits = np.ascontiguousarray(a).view(view)
    width = fbits + ebits + 1
    neg = (bits >> view(width - 1)) == 1
    ef = ((bits >> view(fbits)) & view((1 << ebits) - 1)).astype(np.int64)
    fr = (bits & view((1 << fbits) - 1)).astype(np.int64)
    if bool((ef == (1 << ebits) - 1).any()):
        raise ValueError("float_parts: the input has an inf or a NaN")
    normal = ef > 0
    m = np.where(normal, fr | (np.int64(1) << np.int64(fbits)), fr)
    e = np.where(normal, ef - 1 + emin, emin).astype(np.int64)
    return neg, m, e


def fraction_of(neg: bool, m: int, e: int) -> Fraction:
    """The exact value (-1)^neg * m * 2^e."""
    v = Fraction(int(m) << int(e)) if e >= 0 else Fraction(int(m), 1 << int(-e))
    return -v if neg else v


def fractions_of(a: np.ndarray) -> list[Fraction]:
    """The exact value of each element of a float32 or float64 array (row-major order)."""
    neg, m, e = float_parts(np.asarray(a).reshape(-1))
    return [fraction_of(n, mm, ee) for n, mm, ee in zip(neg.tolist(), m.tolist(), e.tolist(),
                                                         strict=True)]


def binary64_round(x: Fraction) -> Fraction:
    """The binary64 value nearest to x (ties to even), as an exact fraction."""
    x = Fraction(x)
    if x == 0:
        return x
    sign = -1 if x < 0 else 1
    a = abs(x)
    # t = floor(log2 a), exact.
    t = floor_log2(a)
    quantum_exp = max(t - 52, -1074)
    unit = Fraction(1 << quantum_exp) if quantum_exp >= 0 else Fraction(1, 1 << -quantum_exp)
    v = rhe(a / unit) * unit
    if v >= Fraction(2) ** 1024:
        raise ValueError("binary64_round: the value is too large for binary64")
    return sign * v


def binary64_of_text(text: str) -> Fraction:
    """The value of a decimal number in a TOML file: the nearest binary64 value, exact.

    This is the value that `float(text)` gives (IEEE 754 round to nearest). The statistics
    files hold `repr` texts of binary64 values, so this gives back the measured values.
    """
    return binary64_round(Fraction(text))


def load_toml_exact(path) -> dict:
    """Read a TOML file. Each float becomes its exact binary64 value (`binary64_of_text`)."""
    import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f, parse_float=binary64_of_text)


# --- Quantisation ---------------------------------------------------------------------------

ROW_CHUNK = 2048  # Rows in one vectorised step. PERF: bounds the memory to about 0.5 GB (est.).


def _abs_row_max(w: np.ndarray, m: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The (m, e) of max |x| of each row, exact.

    For values without the sign bit, the order of the bit patterns (as unsigned integers) is
    the order of the magnitudes (IEEE 754). Thus an integer argmax finds the maximum.
    """
    view, fbits, ebits, _ = _LAYOUT[w.dtype]
    mag = np.ascontiguousarray(w).view(view) & view((1 << (fbits + ebits)) - 1)
    idx = mag.argmax(axis=1)
    rows = np.arange(m.shape[0])
    return m[rows, idx], e[rows, idx]


def quantise_rows(w: np.ndarray, qmax: int = I8_MAX) -> tuple[np.ndarray, list[Fraction]]:
    """Symmetric quantisation with one scale for each row (plan §7.1), exact.

    `w` is a float32 or float64 matrix [rows, cols]. The scale of a row is
    `scale = max|row| / qmax` (exact), and 1 for a row of zeros. The result is
    `(q, scales)` with `q = R(w / scale)`, an int64 matrix in [-qmax, qmax].
    Cost: one pass over `w` with int64 arithmetic.
    """
    w = np.asarray(w)
    if w.ndim != 2:
        raise ValueError("quantise_rows: w must be a matrix")
    p = mantissa_bits(w.dtype)
    cut = 2 + int(qmax).bit_length()  # a shift of `cut` or more gives R(.) = 0 (see below)
    vector = p + cut <= 62
    q = np.zeros(w.shape, dtype=np.int64)
    scales: list[Fraction] = []
    for r0 in range(0, w.shape[0], ROW_CHUNK):
        part = np.ascontiguousarray(w[r0 : r0 + ROW_CHUNK])
        neg, m, e = float_parts(part)
        mx_m, mx_e = _abs_row_max(part, m, e)
        for mm, ee in zip(mx_m.tolist(), mx_e.tolist(), strict=True):
            scales.append(fraction_of(False, mm, ee) / qmax if mm else Fraction(1))
        # |w| / scale = qmax * m * 2^e / (mx_m * 2^mx_e) = qmax * m / (mx_m * 2^sh).
        sh = mx_e[:, None] - e
        if bool((sh < 0).any()):
            raise ValueError("quantise_rows: an exponent is above the row maximum")
        # INVARIANT: sh > 0 only when the row maximum is normal (mx_m >= 2^(p-1)). Then for
        # sh >= cut, qmax * m / (mx_m * 2^sh) < qmax * 2^(1 - sh) <= 1/2, so R(.) = 0.
        live = (sh < cut) & (m > 0)
        if vector:
            # INVARIANT: num < 2^(bits(qmax) + p) and 2 * den < 2^(p + cut) <= 2^62.
            shc = np.where(live, sh, 0)
            num = np.int64(qmax) * m
            den = mx_m[:, None] << shc
            den = np.where(den == 0, 1, den)
            mag = np.where(live, rhe_vec(num, den), 0)
        else:
            mag = np.zeros(m.shape, dtype=np.int64)
            for i, j in zip(*np.nonzero(live), strict=True):
                mag[i, j] = rhe(Fraction(qmax * int(m[i, j]), int(mx_m[i]) << int(sh[i, j])))
        if bool((mag > qmax).any()):
            raise ValueError("quantise_rows: a value is above qmax")
        q[r0 : r0 + ROW_CHUNK] = np.where(neg, -mag, mag)
    return q, scales


def matvec(q: np.ndarray, v: Sequence[Fraction]) -> list[Fraction]:
    """The exact product q . v of an integer matrix [rows, cols] and a vector of fractions.

    The vector becomes integers over one common denominator, in limbs of 24 bits. Each limb
    is one int64 matrix product. INVARIANT: |q| <= 2^15 and cols < 2^23, so every limb sum is
    below 2^62. Integer addition is exact, so the order of the sums has no effect.
    """
    q = np.asarray(q, dtype=np.int64)
    rows, cols = q.shape
    if len(v) != cols or cols >= 1 << 23 or (q.size and int(np.abs(q).max()) > 1 << 15):
        raise ValueError("matvec: shapes or bounds")
    den = 1
    for x in v:
        den = lcm(den, x.denominator)
    nums = [x.numerator * (den // x.denominator) for x in v]
    sign = np.array([-1 if n < 0 else 1 for n in nums], dtype=np.int64)
    mags = [abs(n) for n in nums]
    top = max((n.bit_length() for n in mags), default=0)
    total = [0] * rows
    for k in range(0, top, 24):
        limb = np.array([(n >> k) & 0xFFFFFF for n in mags], dtype=np.int64) * sign
        part = (q @ limb).tolist()
        for r in range(rows):
            total[r] += part[r] << k
    return [Fraction(t, den) for t in total]


# --- Rescale constants (ADR 0004 section 6) -------------------------------------------------


def quant_mult_rows(ratios: Sequence[Fraction]) -> tuple[np.ndarray, int]:
    """Rescale constants for positive ratios with one shared shift, exact.

    The shift `s` is the largest value up to 126 with `max(r) * 2^(s+1) < 2^31 - 1`, so the
    largest multiplier is in [2^30, 2^31). Each multiplier is `m = R(r * 2^s)`.
    """
    rs = [Fraction(r) for r in ratios]
    mx = max(rs)
    if not (0 < mx < I32_MAX) or min(rs) < 0:
        raise ValueError(f"ratio out of range: min {float(min(rs))}, max {float(mx)}")
    s = 0
    while s < 126 and mx * (1 << (s + 1)) < I32_MAX:
        s += 1
    m = [rhe(r * (1 << s)) for r in rs]
    if max(m) > I32_MAX:
        raise ValueError("multiplier overflow")
    return np.array(m, dtype=np.int32), s


def quant_mult(ratio: Fraction) -> tuple[int, int]:
    """Rescale constants (m, s) for one positive ratio."""
    m, s = quant_mult_rows([ratio])
    return int(m[0]), s


def quant_mult_over_sqrt(a: Fraction, d: int) -> tuple[int, int]:
    """Rescale constants (m, s) for the ratio a / sqrt(d), with a > 0 and d > 0, exact.

    If d is a square, the ratio is rational. If not, the ratio is irrational: it is never a
    tie, and the converter compares squares.
    """
    root = isqrt(d)
    if root * root == d:
        return quant_mult(Fraction(a) / root)
    y2 = Fraction(a) ** 2 / d  # the square of the ratio
    lim2 = I32_MAX * I32_MAX
    if not (0 < y2 < lim2):
        raise ValueError("ratio out of range")
    s = 0
    while s < 126 and y2 * (1 << (2 * (s + 1))) < lim2:
        s += 1
    t = y2 * (1 << (2 * s))  # (ratio * 2^s)^2
    n = isqrt(t.numerator // t.denominator)
    while Fraction((2 * n + 1) ** 2, 4) <= t:
        n += 1
    while n > 0 and Fraction((2 * n - 1) ** 2, 4) > t:
        n -= 1
    if n > I32_MAX:
        raise ValueError("multiplier overflow")
    return n, s


# --- The range grid -------------------------------------------------------------------------

# The grid points of one octave: GRID[j] / 2^52 for j = 0 to 17. These are the binary64 values
# that the float converter of version 1 calculated (2^(1/16) by four square roots of 2, then
# repeated multiplication). They are near 2^(j/16), not equal to it. The exact converter keeps
# them as fixed constants, so the scales of version 1 do not change (research note
# 2026-10-19-exact-conversion). NOTE: GRID[16] is below 2 (by 12 units of 2^-52), so a range in
# (GRID[16] * 2^(k-1), 2^k) goes to GRID[17]. Version 1 did the same.
GRID = (
    4503599627370496, 4702991017412879, 4911210218475898, 5128648071143991,
    5355712719992595, 5592830379701279, 5840446135085603, 6099024776549413,
    6369051672525768, 6651033680544130, 6945500098633942, 7253003658850442,
    7574121564787622, 7909456575025812, 8259638134547582, 8625323556245711,
    9007199254740980, 9405982034825746,
)


def _pow2(k: int) -> Fraction:
    return Fraction(1 << k) if k >= 0 else Fraction(1, 1 << -k)


def floor_log2(x: Fraction) -> int:
    """floor(log2 x) for x > 0, exact."""
    t = x.numerator.bit_length() - x.denominator.bit_length()
    if _pow2(t) > x:
        t -= 1
    return t


def round_range_sq(sq: Fraction) -> Fraction:
    """Round a positive range up to the grid. The argument is the square of the range.

    With the range x = sqrt(sq) = f * 2^e (0.5 <= f < 1), the result is the first value
    v = 2^(e-1) * GRID[j] / 2^52 (j = 0, 1, ...) with v >= x. The square permits an irrational
    range, for example `row_std_min / (2^12 * sqrt(2))` of the residual rule.
    """
    sq = Fraction(sq)
    if not sq > 0:
        raise ValueError(f"range must be positive: {sq}")
    e = floor_log2(sq) // 2 + 1  # 2^(2e-2) <= sq < 2^(2e)
    base = _pow2(e - 1 - 52)
    for g in GRID:
        v = base * g
        if v * v >= sq:
            return v
    raise ValueError("round_range: no grid point")  # unreachable: GRID[17]^2 > 4 * 2^104


def round_range(x: Fraction) -> Fraction:
    """Round a positive rational range up to the grid (`round_range_sq`)."""
    x = Fraction(x)
    if not x > 0:
        raise ValueError(f"range must be positive: {x}")
    return round_range_sq(x * x)


# --- Tables (plan §7.4 to §7.6) -------------------------------------------------------------

PREC = 50  # decimal digits for the tables
TIE_MARGIN = Decimal("1e-30")  # an entry nearer than this to a tie stops the converter
GELU_C = Decimal("0.044715")  # the GPT-2 GELU constant (published)
# pi to 70 decimals (published).
PI_70 = "3.1415926535897932384626433832795028841971693993751058209749445923078164"


def context() -> Context:
    """The one decimal context of the converter. It does not use the global context."""
    return Context(prec=PREC, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
                   traps=[InvalidOperation, DivisionByZero, Overflow])


def _pi(ctx: Context) -> Decimal:
    """pi to the precision of `ctx` (the series of the Python `decimal` documentation)."""
    c = Context(prec=ctx.prec + 10, rounding=ROUND_HALF_EVEN)
    three = Decimal(3)
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n + na, na + 8
        d, da = d + da, da + 32
        t = c.divide(c.multiply(t, n), d)
        s = c.add(s, t)
    pi = ctx.plus(s)
    if ctx.prec > 68 or ctx.abs(ctx.subtract(pi, Decimal(PI_70))) > ctx.power(Decimal(10), 2 - ctx.prec):
        raise ValueError("pi: the series does not agree with the published digits")
    return pi


def _dec(x: Fraction, ctx: Context) -> Decimal:
    return ctx.divide(Decimal(x.numerator), Decimal(x.denominator))


def _round_entry(y: Decimal, ctx: Context, what: str) -> int:
    """R(y), with the margin check."""
    fl = y.to_integral_value(rounding=ROUND_FLOOR, context=ctx)
    dist = ctx.abs(ctx.subtract(ctx.subtract(y, fl), Decimal("0.5")))
    if dist < TIE_MARGIN:
        raise ValueError(f"{what}: an entry is within {TIE_MARGIN} of a tie; increase PREC")
    return int(y.to_integral_value(rounding=ROUND_HALF_EVEN, context=ctx))


class TieStats:
    """The smallest distance to a tie of all table entries (for the research note)."""

    smallest: Decimal | None = None

    @classmethod
    def see(cls, y: Decimal, ctx: Context) -> None:
        fl = y.to_integral_value(rounding=ROUND_FLOOR, context=ctx)
        d = ctx.abs(ctx.subtract(ctx.subtract(y, fl), Decimal("0.5")))
        if cls.smallest is None or d < cls.smallest:
            cls.smallest = d


def exp_table(e: int, length: int, delta_log2: int) -> list[int]:
    """table[i] = R(2^e * exp(-i * 2^delta_log2)) (plan §7.5, §7.6).

    No entry is a tie: exp of a nonzero rational is irrational (Lindemann), and entry 0 is
    2^e exactly.
    """
    ctx = context()
    scale = Decimal(1 << e)
    step = _dec(_pow2(delta_log2), ctx)
    out = []
    for i in range(length):
        y = ctx.multiply(scale, ctx.exp(ctx.multiply(Decimal(-i), step)))
        if i:
            TieStats.see(y, ctx)
            out.append(_round_entry(y, ctx, "exp_table"))
        else:
            out.append(1 << e)
    return out


def _gelu(v: Decimal, c: Decimal, ctx: Context) -> Decimal:
    """0.5 * v * (1 + tanh(c * (v + 0.044715 * v^3))), the GPT-2 GELU ("gelu_new")."""
    v3 = ctx.multiply(ctx.multiply(v, v), v)
    w = ctx.multiply(c, ctx.add(v, ctx.multiply(GELU_C, v3)))
    # tanh(w) = 1 - 2 / (exp(2w) + 1) for w >= 0, and tanh(-w) = -tanh(w).
    aw = ctx.abs(w)
    t = ctx.subtract(Decimal(1), ctx.divide(Decimal(2), ctx.add(ctx.exp(ctx.multiply(2, aw)),
                                                                 Decimal(1))))
    if w < 0:
        t = ctx.minus(t)
    return ctx.multiply(ctx.multiply(Decimal("0.5"), v), ctx.add(Decimal(1), t))


def gelu_table(bits: int, s_in: Fraction, s_out: Fraction, out_max: int = I8_MAX) -> np.ndarray:
    """A GELU table for inputs of `bits` bits at scale s_in, with outputs at scale s_out.

    The entry at index `x + 2^(bits-1)` is clamp(R(gelu(x * s_in) / s_out), -q, q) with
    `q = out_max`: 127 for an i8 table, 32,767 for an i16 table (hsm version 2, ADR 0004
    addendum B). The identity gelu(-v) = gelu(v) - v gives the negative half from the positive
    half. Cost: one decimal `exp` for each positive x (32,767 for an i16 input).
    """
    if out_max not in (I8_MAX, I16_MAX):
        raise ValueError(f"out_max must be {I8_MAX} or {I16_MAX}")
    ctx = context()
    c = ctx.sqrt(ctx.divide(Decimal(2), _pi(ctx)))
    half = 1 << (bits - 1)
    out = np.zeros(2 * half, dtype=np.int8 if out_max == I8_MAX else np.int16)
    lim = Decimal(out_max + 1)
    num, den = Decimal(s_out.denominator), Decimal(s_out.numerator)

    def entry(y: Decimal) -> int:
        if ctx.abs(y) >= lim:
            return out_max if y > 0 else -out_max
        TieStats.see(y, ctx)
        return max(-out_max, min(out_max, _round_entry(y, ctx, "gelu_table")))

    for x in range(1, half + 1):
        v = _dec(Fraction(x) * s_in, ctx)
        gv = _gelu(v, c, ctx)
        if x < half:
            out[half + x] = entry(ctx.divide(ctx.multiply(gv, num), den))
        out[half - x] = entry(ctx.divide(ctx.multiply(ctx.subtract(gv, v), num), den))
    return out


# --- Guards ---------------------------------------------------------------------------------


@contextlib.contextmanager
def no_float_model() -> Iterator[None]:
    """Inside this block, a call of the float model (`convert.model.forward`) is an error.

    The converter reads the calibration statistics as data (ADR 0014 point 4). It never runs
    the float model. This guard makes the rule a check.
    """
    from convert import model

    saved = model.forward

    def refuse(*_a, **_k):
        raise RuntimeError("the converter must not run the float model (ADR 0014 point 4)")

    model.forward = refuse
    try:
        yield
    finally:
        model.forward = saved
