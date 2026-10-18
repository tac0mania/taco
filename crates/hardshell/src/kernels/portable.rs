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
