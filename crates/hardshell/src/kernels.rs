//! Integer matrix-vector products: the reference kernels, and the fast kernels ([`Kernels`]).
//!
//! The free functions of this module are the reference: single-threaded scalar loops. A faster
//! path (SIMD, GPU) must give the same bits as these functions, and a test must prove it
//! (CLAUDE.md §1 rule 10). [`Kernels`] holds the fast paths: portable Rust, NEON `sdot` on
//! aarch64 and AVX2 on x86_64. `tests/fast_kernels.rs` compares each one with the reference.
//!
//! DETERMINISM: integer addition is associative and commutative, and no sum here overflows
//! (see the bounds). Thus a compiler can vectorise or reorder the sums, and the result stays
//! the same on every machine.

use crate::error::{Error, Result};
use crate::tensor::Matrix;

#[cfg(target_arch = "x86_64")]
mod avx2;
mod fast;
#[cfg(target_arch = "aarch64")]
mod neon;
mod portable;

pub use fast::{Kernels, MatRef, rescale_clamp};

/// The longest `i8 x i8` dot product with an `i32` accumulator.
///
/// PLAN(§7.1): bound. Each product `a * b` has `|a * b| <= 128 * 128 = 2^14`. A sum of at most
/// `2^16` products has a magnitude of at most `2^30 < 2^31`, so the `i32` cannot overflow. GPT-2
/// small needs 3,072 (published).
pub const MAX_DOT_LEN_I32: usize = 1 << 16;

/// The longest `i8 x i16` dot product with an `i64` accumulator.
///
/// Bound: `|a * b| <= 2^7 * 2^15 = 2^22`, so a sum of `2^40` products is at most `2^62`.
pub const MAX_DOT_LEN_I64: usize = 1 << 40;

/// The dot product of two `i8` vectors with an `i32` accumulator.
///
/// The caller keeps `a.len() <= MAX_DOT_LEN_I32`. If the lengths differ, the extra values of
/// the longer vector are not used. Cost: one pass.
#[must_use]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    // INVARIANT: at most MAX_DOT_LEN_I32 terms of magnitude <= 2^14, so |sum| <= 2^30 (the
    // public callers check the length; see MAX_DOT_LEN_I32).
    a.iter().zip(b).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum()
}

/// The dot product of an `i8` vector and an `i16` vector with an `i64` accumulator. Cost: one
/// pass.
#[must_use]
pub fn dot_i8_i16(a: &[i8], b: &[i16]) -> i64 {
    // INVARIANT: at most MAX_DOT_LEN_I64 terms of magnitude <= 2^22, so |sum| <= 2^62.
    a.iter().zip(b).map(|(&x, &y)| i64::from(i32::from(x) * i32::from(y))).sum()
}

fn check(rows: usize, cols: usize, x_len: usize, out_len: usize, max: usize) -> Result<()> {
    if cols > max {
        return Err(Error::DotTooLong { len: cols, max });
    }
    if x_len != cols {
        return Err(Error::LengthMismatch { expected: cols, found: x_len });
    }
    if out_len != rows {
        return Err(Error::LengthMismatch { expected: rows, found: out_len });
    }
    Ok(())
}

/// `out = w * x` for an `i8` matrix `[rows, cols]` and an `i8` vector, with `i32` accumulators.
///
/// PLAN(§7.1): `i32` accumulators for `i8 x i8` dot products. Cost: `rows * cols`
/// multiply-adds, one pass over `w`.
///
/// # Errors
/// [`Error::DotTooLong`] if `cols > MAX_DOT_LEN_I32`; [`Error::LengthMismatch`] if
/// `x.len() != cols` or `out.len() != rows`.
pub fn matvec_i8(w: &Matrix<i8>, x: &[i8], out: &mut [i32]) -> Result<()> {
    check(w.rows(), w.cols(), x.len(), out.len(), MAX_DOT_LEN_I32)?;
    for (o, row) in out.iter_mut().zip(w.row_iter()) {
        *o = dot_i8(row, x);
    }
    Ok(())
}

/// `out = w * x` for an `i8` matrix `[rows, cols]` and an `i16` vector, with `i64`
/// accumulators. This is the kernel for `i16` activations (plan §7.1). Cost: `rows * cols`
/// multiply-adds.
///
/// # Errors
/// [`Error::DotTooLong`] if `cols > MAX_DOT_LEN_I64`; [`Error::LengthMismatch`] if
/// `x.len() != cols` or `out.len() != rows`.
pub fn matvec_i8_i16(w: &Matrix<i8>, x: &[i16], out: &mut [i64]) -> Result<()> {
    check(w.rows(), w.cols(), x.len(), out.len(), MAX_DOT_LEN_I64)?;
    for (o, row) in out.iter_mut().zip(w.row_iter()) {
        *o = dot_i8_i16(row, x);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_i8_gives_exact_sums() {
        let w = Matrix::new(2, 3, vec![1i8, -2, 3, -128, -128, 127]).unwrap();
        let mut out = [0i32; 2];
        matvec_i8(&w, &[-128, 5, 2], &mut out).unwrap();
        assert_eq!(out, [-128 - 10 + 6, 16384 - 640 + 254]);
    }

    #[test]
    fn matvec_i8_accepts_the_worst_case_at_the_bound() {
        // All products are +2^14: the sum is exactly 2^30 and does not overflow.
        let n = MAX_DOT_LEN_I32;
        let w = Matrix::new(1, n, vec![-128i8; n]).unwrap();
        let mut out = [0i32; 1];
        matvec_i8(&w, &vec![-128i8; n], &mut out).unwrap();
        assert_eq!(out[0], 1 << 30);
    }

    #[test]
    fn matvec_refuses_bad_lengths() {
        let w = Matrix::new(2, 3, vec![0i8; 6]).unwrap();
        let mut out = [0i32; 2];
        assert!(matches!(matvec_i8(&w, &[0; 2], &mut out), Err(Error::LengthMismatch { .. })));
        assert!(matches!(matvec_i8(&w, &[0; 3], &mut out[..1]), Err(Error::LengthMismatch { .. })));
        let wide = Matrix::new(1, MAX_DOT_LEN_I32 + 1, vec![0i8; MAX_DOT_LEN_I32 + 1]).unwrap();
        let x = vec![0i8; MAX_DOT_LEN_I32 + 1];
        assert!(matches!(matvec_i8(&wide, &x, &mut out[..1]), Err(Error::DotTooLong { .. })));
    }

    #[test]
    fn matvec_i8_i16_gives_exact_sums() {
        let w = Matrix::new(1, 2, vec![-128i8, 127]).unwrap();
        let mut out = [0i64; 1];
        matvec_i8_i16(&w, &[-32768, 32767], &mut out).unwrap();
        assert_eq!(out[0], 128 * 32768 + 127 * 32767);
    }
}
