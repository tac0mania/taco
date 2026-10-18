//! Integer LayerNorm (plan §7.3).
//!
//! For a row `x` of width `n` (the residual stream, `i32`):
//!
//! 1. `mean = floor_div(sum(x), n)`.
//! 2. `var = floor(sum((x - mean)^2) / n)` (non-negative, so this is `floor_div`).
//! 3. `inv = floor(2^K / isqrt(var * 2^(2J) + eps))`.
//! 4. `y = clamp(rescale((x - mean) * inv * gamma, m, s) + beta, -127, 127)`, an `i8`.
//!
//! In real numbers, `(x - mean) * inv` is the normalised value times `2^(K-J)`. The converter
//! puts `gamma` (`i16`) at a scale of its choice, and selects `m` and `s` so that the result has
//! the scale of the `i8` output. `beta` is an `i32` at the output scale. `eps` is the float
//! epsilon in the units of `var * 2^(2J)`.
//!
//! PLAN(§7.3): the constants are a phase 2 decision; ADR 0004 records them and the measured
//! error.
//!
//! DETERMINISM: the sums are integer sums, so their order does not change them. The divisions
//! are `floor_div` and unsigned division, and the square root is `isqrt_u128`. Thus every
//! machine gives the same output bits.
//!
//! Bounds (they hold for every `i32` input, so the result is always exact):
//! - `|sum(x)| <= 2^31 * 2^16 = 2^47` with `n <= 2^16`: fits in `i64`.
//! - `|x - mean| < 2^32`, so `(x - mean)^2 < 2^64` and the sum is less than `2^80` (`u128`).
//! - `var < 2^64`, `var * 2^(2J) + eps < 2^96 + 2^62` with `J <= 16` (`u128`).
//! - `isqrt(...) >= 1` because `eps >= 1`, so `inv <= 2^K <= 2^48`.
//! - `|(x - mean) * inv * gamma| < 2^32 * 2^48 * 2^15 = 2^95`: the input bound of
//!   [`rescale_wide`].

use crate::error::{Error, Result};
use crate::int::{Divisor, clamp_act_i8, floor_div, isqrt_u128, rescale_wide};

/// The largest permitted `K`.
pub const MAX_K: u8 = 48;
/// The largest permitted `J`.
pub const MAX_J: u8 = 16;
/// The largest permitted `eps`: `2^62`.
pub const MAX_EPS: u64 = 1 << 62;
/// The largest permitted row width: `2^16`.
pub const MAX_WIDTH: usize = 1 << 16;

/// The constants `K`, `J` and `eps` of the integer LayerNorm. They are the same for every
/// LayerNorm of a model and are stated in the model file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LnConstants {
    k: u8,
    j: u8,
    eps: u64,
}

impl LnConstants {
    /// Validates the constants.
    ///
    /// # Errors
    /// [`Error::InvalidHeader`] if `k > 48`, `j > 16`, `eps == 0` or `eps > 2^62`.
    pub fn new(k: u8, j: u8, eps: u64) -> Result<Self> {
        let bad = |field: &str, reason| Err(Error::InvalidHeader { field: field.into(), reason });
        if k > MAX_K {
            return bad("constants.ln_k", "must be at most 48");
        }
        if j > MAX_J {
            return bad("constants.ln_j", "must be at most 16");
        }
        if eps == 0 || eps > MAX_EPS {
            return bad("constants.ln_eps", "must be in 1..=2^62");
        }
        Ok(Self { k, j, eps })
    }

    /// `K`: `inv = floor(2^K / isqrt(...))`.
    #[must_use]
    pub fn k(self) -> u8 {
        self.k
    }

    /// `J`: the variance is multiplied by `2^(2J)` before the square root.
    #[must_use]
    pub fn j(self) -> u8 {
        self.j
    }

    /// `eps` in the units of `var * 2^(2J)`.
    #[must_use]
    pub fn eps(self) -> u64 {
        self.eps
    }
}

/// The parameters of one LayerNorm: `gamma` (`i16`), `beta` (`i32`, output units) and the
/// rescale `(m, s)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerNorm {
    gamma: Box<[i16]>,
    beta: Box<[i32]>,
    m: i32,
    s: u8,
}

impl LayerNorm {
    /// Makes a LayerNorm.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `gamma` and `beta` have different lengths;
    /// [`Error::InvalidShape`] if the width is 0 or more than [`MAX_WIDTH`].
    pub fn new(gamma: Vec<i16>, beta: Vec<i32>, m: i32, s: u8) -> Result<Self> {
        if gamma.len() != beta.len() {
            return Err(Error::LengthMismatch { expected: gamma.len(), found: beta.len() });
        }
        if gamma.is_empty() || gamma.len() > MAX_WIDTH {
            return Err(Error::InvalidShape { reason: "LayerNorm width must be in 1..=2^16" });
        }
        Ok(Self { gamma: gamma.into_boxed_slice(), beta: beta.into_boxed_slice(), m, s })
    }

    /// The row width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.gamma.len()
    }

    /// Normalises the row `x` into `out`. Cost: three passes over the row, one integer square
    /// root and one `u64` division.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `x` or `out` does not have the width of the LayerNorm.
    pub fn apply(&self, c: LnConstants, x: &[i32], out: &mut [i8]) -> Result<()> {
        let n = self.gamma.len();
        if x.len() != n {
            return Err(Error::LengthMismatch { expected: n, found: x.len() });
        }
        if out.len() != n {
            return Err(Error::LengthMismatch { expected: n, found: out.len() });
        }
        // INVARIANT: 1 <= n <= 2^16 (checked in `new`), so these conversions are exact and the
        // divisor is positive.
        let n_i64 = i64::try_from(n).unwrap_or(1);
        let n_div = Divisor::new(n_i64).unwrap_or(Divisor::MIN);
        let n_u128 = n as u128;

        // INVARIANT: |sum| <= 2^47 (see the module bounds).
        let sum: i64 = x.iter().map(|&v| i64::from(v)).sum();
        let mean = floor_div(sum, n_div);

        // INVARIANT: each square is less than 2^64 and the sum is less than 2^80.
        let sum_sq: u128 = x
            .iter()
            .map(|&v| {
                let d = (i64::from(v) - mean).unsigned_abs();
                u128::from(d) * u128::from(d)
            })
            .sum();
        // NOTE: unsigned division of non-negative values is the floor division of §7.3 step 2.
        let var = sum_sq / n_u128;
        // INVARIANT: var < 2^64 and j <= 16, so var << 2j < 2^96, plus eps <= 2^62.
        let root = isqrt_u128((var << (2 * u32::from(c.j))) + u128::from(c.eps));
        // INVARIANT: root >= 1 because eps >= 1; root < 2^49.
        let inv = (1u128 << c.k) / root.max(1);
        let inv = i128::try_from(inv).unwrap_or(0);

        for (((o, &v), &g), &b) in out.iter_mut().zip(x).zip(&self.gamma).zip(&self.beta) {
            let centred = i128::from(i64::from(v) - mean);
            // INVARIANT: |centred * inv * g| < 2^95 (module bounds), the rescale_wide bound.
            let t = centred * inv * i128::from(g);
            *o = clamp_act_i8(rescale_wide(t, self.m, self.s) + i128::from(b));
        }
        Ok(())
    }
}
