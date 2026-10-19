//! The portable fast kernels: safe Rust, written so that the compiler can vectorise them.
//!
//! INVARIANT: the compiler can split a sum into vector lanes; each lane sum is a subset of the
//! products of one dot product. The bound of the parent module (`fast`) applies, so no sum
//! overflows.

use super::fast::Shape;

/// The dot product of `a` and `b`, which have the same length. Cost: one pass.
///
/// NOTE: this is the expression of the reference `dot_i8`. The compiler vectorises it well
/// (measured: faster than a loop with 16 explicit lane sums on the M2).
pub(super) fn dot(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum()
}

/// `out[r][o] = w[o] . x[r]` for the checked `shape`. Each weight row stays in the L1 cache
/// while all rows of `x` use it. Cost: `rows * cols * n` multiply-adds.
pub(super) fn matmul(w: &[i8], x: &[i8], shape: Shape, out: &mut [i32]) {
    let Shape { rows, cols, .. } = shape;
    for o in 0..rows {
        let w_row = shape.w_row(w, o);
        for (x_row, out_row) in x.chunks_exact(cols).zip(out.chunks_exact_mut(rows)) {
            if let Some(slot) = out_row.get_mut(o) {
                *slot = dot(w_row, x_row);
            }
        }
    }
}

/// The number of `i8 x i16` products that one `i32` partial sum holds.
///
/// INVARIANT: each product has a magnitude of at most `2^7 * 2^15 = 2^22`, so a sum of `2^8`
/// products is at most `2^30 < 2^31`.
const WIDE_BLOCK: usize = 256;

/// The dot product of an `i8` and an `i16` vector of the same length, in `i64`: the same value
/// as the reference `dot_i8_i16`. Cost: one pass.
pub(super) fn dot_i8_i16(a: &[i8], b: &[i16]) -> i64 {
    a.chunks(WIDE_BLOCK)
        .zip(b.chunks(WIDE_BLOCK))
        .map(|(a, b)| {
            let part: i32 = a.iter().zip(b).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum();
            i64::from(part)
        })
        .sum()
}

/// `out[r][o] = w[o] . x[r]` for `i8` weights `[rows, cols]` and `i16` rows `x`. The caller
/// checks the shapes (`Kernels::matmul_i8_i16`). Cost: `rows * cols * n` multiply-adds.
pub(super) fn matmul_i8_i16(w: &[i8], x: &[i16], cols: usize, out: &mut [i64]) {
    let rows = w.len() / cols.max(1);
    for (o, w_row) in w.chunks_exact(cols).enumerate() {
        for (x_row, out_row) in x.chunks_exact(cols).zip(out.chunks_exact_mut(rows)) {
            if let Some(slot) = out_row.get_mut(o) {
                *slot = dot_i8_i16(w_row, x_row);
            }
        }
    }
}
