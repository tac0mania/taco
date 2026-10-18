//! The integer helpers of HardShell: rescale, clamp, floor division, square root and table
//! look-up. These are the only rounding and division rules in the prediction path.
//!
//! PLAN(§7.2): each rule in the table of §7.2 has exactly one function here. Other modules do
//! not round, divide signed values or saturate in a different way.
//!
//! DETERMINISM: every function here uses only integer operations with one defined result for
//! every input. No result depends on the machine, the build profile or the order of calls.

/// A divisor that is known to be more than zero.
///
/// `floor_div` takes a `Divisor`, so a division by zero or by a negative value cannot be
/// written. A `Divisor` is made only by [`Divisor::new`], which refuses `b <= 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Divisor(i64);

impl Divisor {
    /// The divisor 1.
    pub const MIN: Self = Self(1);

    /// Makes a divisor. Returns `None` if `b` is zero or negative. Cost: one comparison.
    #[must_use]
    pub const fn new(b: i64) -> Option<Self> {
        if b > 0 { Some(Self(b)) } else { None }
    }

    /// The value of the divisor. It is always more than zero.
    #[must_use]
    pub const fn get(self) -> i64 {
        self.0
    }
}

/// Floor division: the largest integer `q` with `q * b <= a`.
///
/// PLAN(§7.2): `floor_div(a, b)` for `b > 0` is `a.div_euclid(b)`.
///
/// DETERMINISM: for `b > 0`, Euclidean division and floor division are the same, and
/// `div_euclid` cannot overflow (the only overflow case, `i64::MIN / -1`, needs `b < 0`).
#[must_use]
pub const fn floor_div(a: i64, b: Divisor) -> i64 {
    // INVARIANT: `b.0 > 0` (the `Divisor` constructor checks it), so this cannot panic.
    a.div_euclid(b.0)
}

/// The largest shift that [`rescale`] calculates. For a larger shift the exact result is 0,
/// because `|x * m| <= 2^94 < 2^(s-1)`.
const RESCALE_MAX_SHIFT: u32 = 95;

/// Rescale with round half up: `floor((x * m + 2^(s-1)) / 2^s)`. For `s = 0` the result is
/// `x * m`.
///
/// PLAN(§7.2): `rescale(x, m, s) = (x * m + 2^(s-1)) >> s`, with an arithmetic shift.
///
/// The product is calculated in `i128`, because `|x * m| <= 2^94` does not fit in `i64`. The
/// result is exact for every `x`, `m` and `s`. Its magnitude is at most `2^94`. The caller
/// clamps it to a narrower type with [`clamp_wide`] or one of the `clamp_*` functions. Cost:
/// one 128-bit multiplication.
///
/// DETERMINISM: `>>` on a signed integer in Rust is an arithmetic shift, which is a floor
/// division by `2^s` on every machine.
#[must_use]
pub const fn rescale(x: i64, m: i32, s: u8) -> i128 {
    // INVARIANT: |x| <= 2^63 and |m| <= 2^31, so |x * m| <= 2^94 and no step overflows `i128`.
    let product = (x as i128) * (m as i128);
    shift_round(product, s, RESCALE_MAX_SHIFT)
}

/// The largest permitted `|x|` in [`rescale_wide`] is less than `2^95`.
pub const RESCALE_WIDE_INPUT_BITS: u32 = 95;

/// The largest shift that [`rescale_wide`] calculates. For a larger shift the exact result is
/// 0, because `|x * m| < 2^126 <= 2^(s-1)`.
const RESCALE_WIDE_MAX_SHIFT: u32 = 126;

/// Rescale with round half up for an `i128` input: `floor((x * m + 2^(s-1)) / 2^s)`.
///
/// This is the same rule as [`rescale`]. LayerNorm uses it, because its product
/// `(x - mean) * inv * gamma` does not fit in `i64` (see `layernorm`).
///
/// The result is exact if `|x| < 2^95`. Then `|x * m| < 2^126` and no step overflows. If the
/// caller breaks this bound, the multiplication wraps: the result is wrong but it is still
/// defined, and it is the same on every machine. Cost: one 128-bit multiplication.
#[must_use]
pub const fn rescale_wide(x: i128, m: i32, s: u8) -> i128 {
    // INVARIANT: the caller keeps |x| < 2^95 (see the doc comment). The wrapping operation
    // keeps the result defined, and equal on every machine, also when the caller breaks it.
    let product = x.wrapping_mul(m as i128);
    shift_round(product, s, RESCALE_WIDE_MAX_SHIFT)
}

/// `floor((p + 2^(s-1)) / 2^s)`. The caller gives `max_shift <= 126` such that
/// `|p| < 2^max_shift` and `|p| + 2^(max_shift-1) < 2^127`.
const fn shift_round(p: i128, s: u8, max_shift: u32) -> i128 {
    if s == 0 {
        return p;
    }
    let s = s as u32;
    // NOTE: for `s > max_shift`, `2^(s-1) > |p|`, so `0 <= p + 2^(s-1) < 2^s` and the exact
    // result is 0. This also keeps the shifts below in range for every `s` up to 255.
    if s > max_shift {
        return 0;
    }
    // INVARIANT: 1 <= s <= max_shift <= 126, so `1 << (s - 1)` fits in `i128`, and the caller
    // bound keeps `p + half` in range.
    let half = 1i128 << (s - 1);
    (p.wrapping_add(half)) >> s
}

/// Clamp `x` to `[lo, hi]`: `max(lo, min(x, hi))`.
///
/// PLAN(§7.2): this is the only saturating operation in the prediction path. If `lo > hi`, the
/// result is `lo`, so every input has one defined result.
#[must_use]
pub const fn clamp(x: i64, lo: i64, hi: i64) -> i64 {
    let upper = if x < hi { x } else { hi };
    if upper > lo { upper } else { lo }
}

/// Clamp an `i128` to `[lo, hi]` with the same rule as [`clamp`]. The result fits in `i64`.
#[must_use]
pub const fn clamp_wide(x: i128, lo: i64, hi: i64) -> i64 {
    let lo_wide = lo as i128;
    let hi_wide = hi as i128;
    let upper = if x < hi_wide { x } else { hi_wide };
    let clamped = if upper > lo_wide { upper } else { lo_wide };
    // INVARIANT: `clamped` is `lo`, `hi` or a value between them, so it fits in `i64`.
    #[allow(clippy::cast_possible_truncation)]
    let narrow = clamped as i64;
    narrow
}

/// The smallest value of an `i8` activation. Activations are symmetric: `-127..=127`.
pub const ACT_I8_MIN: i64 = -127;
/// The largest value of an `i8` activation.
pub const ACT_I8_MAX: i64 = 127;

/// Clamp to the symmetric `i8` activation range `[-127, 127]` and convert.
#[must_use]
pub const fn clamp_act_i8(x: i128) -> i8 {
    let c = clamp_wide(x, ACT_I8_MIN, ACT_I8_MAX);
    // INVARIANT: -127 <= c <= 127, so the conversion is exact.
    #[allow(clippy::cast_possible_truncation)]
    let narrow = c as i8;
    narrow
}

/// Clamp to the full `i16` range `[-32768, 32767]` and convert.
#[must_use]
pub const fn clamp_i16(x: i128) -> i16 {
    let c = clamp_wide(x, i16::MIN as i64, i16::MAX as i64);
    // INVARIANT: `c` is in the `i16` range, so the conversion is exact.
    #[allow(clippy::cast_possible_truncation)]
    let narrow = c as i16;
    narrow
}

/// Clamp to `[lo, hi]` with `i32` limits and convert.
#[must_use]
pub const fn clamp_i32(x: i128, lo: i32, hi: i32) -> i32 {
    let c = clamp_wide(x, lo as i64, hi as i64);
    // INVARIANT: `c` is `lo`, `hi` or between them, so it is in the `i32` range.
    #[allow(clippy::cast_possible_truncation)]
    let narrow = c as i32;
    narrow
}

/// Integer square root: `floor(sqrt(x))`.
///
/// PLAN(§7.2): calculated with integer operations only (the digit-by-digit method, two bits
/// of `x` in each round). Cost: at most 32 rounds.
#[must_use]
pub const fn isqrt_u64(x: u64) -> u64 {
    let r = isqrt_u128(x as u128);
    // INVARIANT: floor(sqrt(x)) < 2^32 for x < 2^64, so the conversion is exact.
    #[allow(clippy::cast_possible_truncation)]
    let narrow = r as u64;
    narrow
}

/// Integer square root of a `u128`: `floor(sqrt(x))`. Cost: at most 64 rounds.
#[must_use]
pub const fn isqrt_u128(x: u128) -> u128 {
    // The largest power of 4 that is not more than x.
    let mut bit: u128 = 1 << 126;
    while bit > x {
        bit >>= 2;
    }
    let mut rem = x;
    let mut res: u128 = 0;
    while bit != 0 {
        // INVARIANT: res < 2^64 and bit <= 2^126, so `res + bit` does not overflow.
        let trial = res + bit;
        if rem >= trial {
            rem -= trial;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    res
}

/// A look-up table with at least one entry. The index is clamped to the table range.
///
/// PLAN(§7.2): "The index is clamped to the table range before the look-up." Thus every
/// `i64` index gives one defined entry, and a look-up cannot go out of bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table<T> {
    entries: Box<[T]>,
    /// The entry at index 0. It is returned only on a path that the clamp makes impossible.
    first: T,
    /// `entries.len() - 1` as an `i64`.
    last_index: i64,
}

impl<T: Copy> Table<T> {
    /// Makes a table. Returns `None` if `entries` is empty, or has more than `2^62` entries.
    #[must_use]
    pub fn new(entries: Vec<T>) -> Option<Self> {
        let first = *entries.first()?;
        let last_index = i64::try_from(entries.len() - 1).ok()?;
        if last_index >= (1 << 62) {
            return None;
        }
        Some(Self { entries: entries.into_boxed_slice(), first, last_index })
    }

    /// The entry at `clamp(index, 0, len - 1)`. Cost: one comparison pair and one load.
    #[must_use]
    pub fn lookup(&self, index: i64) -> T {
        let i = clamp(index, 0, self.last_index);
        // INVARIANT: 0 <= i <= last_index < 2^62, so `i` fits in `usize` on 64-bit machines,
        // and `get` finds the entry. The fallback is never used.
        match usize::try_from(i) {
            Ok(u) => match self.entries.get(u) {
                Some(v) => *v,
                None => self.first,
            },
            Err(_) => self.first,
        }
    }

    /// The number of entries. It is at least 1.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Always `false`: a table has at least one entry. Present for API completeness.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The entries in index order.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescale_rounds_half_up() {
        // 3 * 1 / 2 = 1.5 -> 2; -3 / 2 = -1.5 -> -1 (half up, not half away from zero).
        assert_eq!(rescale(3, 1, 1), 2);
        assert_eq!(rescale(-3, 1, 1), -1);
        assert_eq!(rescale(-5, 1, 1), -2);
        assert_eq!(rescale(5, 1, 0), 5);
        assert_eq!(rescale(7, 3, 2), 5); // 21 / 4 = 5.25 -> 5
    }

    #[test]
    fn rescale_handles_extreme_inputs() {
        assert_eq!(rescale(i64::MIN, i32::MIN, 0), (i64::MIN as i128) * (i32::MIN as i128));
        assert_eq!(rescale(i64::MIN, i32::MIN, 94), 1);
        assert_eq!(rescale(i64::MAX, i32::MAX, 255), 0);
        assert_eq!(rescale(i64::MIN, i32::MAX, 200), 0);
        assert_eq!(rescale_wide(-(1i128 << 94), i32::MIN, 126), 1);
        assert_eq!(rescale_wide(-(1i128 << 94), i32::MIN, 127), 0);
        assert_eq!(rescale(i64::MIN, i32::MIN, 95), 1);
        assert_eq!(rescale(i64::MIN, i32::MIN, 96), 0);
        assert_eq!(rescale_wide(1i128 << 94, i32::MIN, 120), -(1i128 << 5));
    }

    #[test]
    fn clamp_is_defined_when_bounds_cross() {
        assert_eq!(clamp(5, 0, 3), 3);
        assert_eq!(clamp(-5, 0, 3), 0);
        assert_eq!(clamp(2, 0, 3), 2);
        assert_eq!(clamp(2, 4, 3), 4);
        assert_eq!(clamp_wide(i128::MAX, -1, 1), 1);
        assert_eq!(clamp_act_i8(-1000), -127);
        assert_eq!(clamp_i16(1 << 40), i16::MAX);
    }

    #[test]
    fn divisor_refuses_zero_and_negative() {
        assert!(Divisor::new(0).is_none());
        assert!(Divisor::new(-3).is_none());
        let d = Divisor::new(3).unwrap();
        assert_eq!(floor_div(-7, d), -3);
        assert_eq!(floor_div(7, d), 2);
        assert_eq!(floor_div(i64::MIN, Divisor::new(1).unwrap()), i64::MIN);
    }

    #[test]
    fn isqrt_is_exact_at_boundaries() {
        assert_eq!(isqrt_u64(0), 0);
        assert_eq!(isqrt_u64(1), 1);
        assert_eq!(isqrt_u64(3), 1);
        assert_eq!(isqrt_u64(4), 2);
        assert_eq!(isqrt_u64(u64::MAX), u64::from(u32::MAX));
        assert_eq!(isqrt_u128(u128::MAX), u128::from(u64::MAX));
        for r in [1u128, 2, 1000, 1 << 40, (1 << 63) - 1] {
            assert_eq!(isqrt_u128(r * r), r);
            assert_eq!(isqrt_u128(r * r - 1), r - 1);
        }
    }

    #[test]
    fn table_clamps_the_index() {
        let t = Table::new(vec![10u8, 20, 30]).unwrap();
        assert_eq!(t.lookup(-5), 10);
        assert_eq!(t.lookup(1), 20);
        assert_eq!(t.lookup(i64::MAX), 30);
        assert!(Table::<u8>::new(vec![]).is_none());
    }
}
