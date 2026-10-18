//! The op kernels of the GPT-2 forward pass: `embed`, `norm`, `linear` and `logits`. The
//! `attention` op is in its own module (`attention`). Each op works on `n` rows at a time (`n` positions of one block, or one position
//! of `n` blocks), and each row gives the same bits as the reference session
//! (`gpt2::reference`) for that row.
//!
//! NOTE: the ops are separate functions, so that a later graph interpreter can call them
//! (`docs/design/anymodel.md` §4). The forward pass in `engine` only composes them.
//!
//! DETERMINISM: the matrix products are exact (`kernels::Kernels`), and every rounding and
//! saturation is the rule of `int`, through `kernels::rescale_clamp` (the same results, proved
//! by a property test).

use super::model::{Linear, Model};
use super::{RESIDUAL_LIMIT as RES, SCORE_LIMIT};
use crate::error::{Error, Result};
use crate::gelu::GeluTable;
use crate::int::{clamp, clamp_i32, rescale};
use crate::kernels::{Kernels, rescale_clamp};
use crate::layernorm::{LayerNorm, LnConstants};
use crate::logits::{LogitTable, MAX_VOCAB, logits_to_weights};

/// `(acc[r] + b[r], m[r])` for each output row `r` of one input row.
fn terms<'a>(
    acc: &'a [i32],
    b: &'a [i32],
    m: &'a [i32],
) -> impl Iterator<Item = (i64, i32)> + use<'a> {
    acc.iter().zip(b).zip(m).map(|((&a, &b), &m)| (i64::from(a) + i64::from(b), m))
}

/// The kernels and the model that every op uses.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ctx<'m> {
    pub(crate) k: Kernels,
    pub(crate) model: &'m Model,
}

/// An internal length error. It cannot occur for a model that `Model::load` accepted.
pub(crate) fn internal(expected: usize, found: usize) -> Error {
    Error::LengthMismatch { expected, found }
}

/// A clamped value in `-127..=127` as an `i8`.
pub(crate) fn act_i8(v: i64) -> i8 {
    // INVARIANT: the callers clamp `v` to [-127, 127], so the conversion is exact.
    i8::try_from(v).unwrap_or(0)
}

/// `embed`: the token and position embeddings of one row into the residual row `x`.
///
/// # Errors
/// An internal length error if `tok` or `pos` is outside the model.
pub(crate) fn embed(model: &Model, tok: usize, pos: usize, x: &mut [i32]) -> Result<()> {
    let d = model.dims.d_model;
    let te = model.wte.row(tok).ok_or(internal(d, 0))?;
    let pe = model.wpe.row(pos).ok_or(internal(d, 0))?;
    let tm = model.embed.m.get(tok).copied().unwrap_or(0);
    let pm = model.pos.m.get(pos).copied().unwrap_or(0);
    for ((x, &a), &b) in x.iter_mut().zip(te).zip(pe) {
        let v = rescale(i64::from(a), tm, model.embed.s) + rescale(i64::from(b), pm, model.pos.s);
        *x = clamp_i32(v, -RES, RES);
    }
    Ok(())
}

/// `norm`: LayerNorm of each row of `x` (rows of `d` values) into `out`.
///
/// # Errors
/// The errors of [`LayerNorm::apply`].
pub(crate) fn norm(
    ln: &LayerNorm,
    c: LnConstants,
    d: usize,
    x: &[i32],
    out: &mut [i8],
) -> Result<()> {
    for (xr, or) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        ln.apply(c, xr, or)?;
    }
    Ok(())
}

/// What a `linear` op does with `rescale(W[r] . x + b[r], m[r], s)`.
pub(crate) enum LinearOut<'a> {
    /// Clamp to an `i8` activation (`-127..=127`).
    Act(&'a mut [i8]),
    /// Add to the residual stream and clamp to `[-R, R]`.
    Residual(&'a mut [i32]),
    /// Clamp to the input range of a GELU table, then look up the table.
    Gelu(&'a mut [i8], &'a GeluTable),
}

/// `linear`: `y[r] = rescale(W[r] . x + b[r], m[r], s)` for `n` input rows, then the output
/// step of `out`. `acc` is scratch space of at least `n * out_rows` values. Cost: one matrix
/// product (each weight read once for all `n` rows) and one rescale for each output.
///
/// # Errors
/// The errors of [`Kernels::matmul_i8`], or an internal length error.
pub(crate) fn linear(
    k: Kernels,
    lin: &Linear,
    x: &[i8],
    acc: &mut [i32],
    out: LinearOut<'_>,
) -> Result<()> {
    let (rows, cols) = (lin.w.rows(), lin.w.cols());
    let n = x.len() / cols;
    let have = acc.len();
    let acc = acc.get_mut(..n * rows).ok_or(internal(n * rows, have))?;
    k.matmul_i8(lin.w.data(), x, cols, acc)?;
    let (m, s) = (&lin.rescale.m, lin.rescale.s);
    // Each output row zips the accumulators with the biases and the multipliers. `Model::load`
    // checked that `b` and `m` have `rows` entries.
    match out {
        LinearOut::Act(y) => {
            let y = y.get_mut(..n * rows).ok_or(internal(n * rows, 0))?;
            for (yr, ar) in y.chunks_exact_mut(rows).zip(acc.chunks_exact(rows)) {
                for (o, (v, m)) in yr.iter_mut().zip(terms(ar, &lin.b, m)) {
                    *o = act_i8(rescale_clamp(v, m, s, -127, 127));
                }
            }
        }
        LinearOut::Residual(x) => {
            let x = x.get_mut(..n * rows).ok_or(internal(n * rows, 0))?;
            let (lim, res) = (1i64 << 40, i64::from(RES));
            for (xr, ar) in x.chunks_exact_mut(rows).zip(acc.chunks_exact(rows)) {
                for (xv, (v, m)) in xr.iter_mut().zip(terms(ar, &lin.b, m)) {
                    // INVARIANT: |x| <= R < 2^30. If |delta| > 2^40, then x + delta and
                    // x + clamp(delta, -2^40, 2^40) are both beyond R with the same sign, so
                    // the clamp to [-R, R] gives the same value as the reference
                    // `clamp_i32(x + rescale(..), -R, R)`.
                    let delta = rescale_clamp(v, m, s, -lim, lim);
                    let sum = clamp(i64::from(*xv) + delta, -res, res);
                    *xv = i32::try_from(sum).unwrap_or(0);
                }
            }
        }
        LinearOut::Gelu(g, table) => {
            let g = g.get_mut(..n * rows).ok_or(internal(n * rows, 0))?;
            let (lo, hi) = (table.min_input(), table.max_input());
            for (gr, ar) in g.chunks_exact_mut(rows).zip(acc.chunks_exact(rows)) {
                for (o, (v, m)) in gr.iter_mut().zip(terms(ar, &lin.b, m)) {
                    *o = table.apply(rescale_clamp(v, m, s, lo, hi));
                }
            }
        }
    }
    Ok(())
}

/// `logits`: the output weights of `n` rows of the final LayerNorm output `a`. The weights of
/// row `i` go to row `slots[i]` of `weights` (rows of `vocab` values). `acc` is scratch space
/// of at least `n * vocab` values, and `logits` holds one row of `vocab` values. Cost: one matrix product with `wte` (each weight
/// read once for all rows), one rescale and one table look-up for each logit.
///
/// # Errors
/// The errors of [`Kernels::matmul_i8`] and `logits_to_weights`, or an internal length error.
pub(crate) fn logits(
    cx: Ctx<'_>,
    a: &[i8],
    acc: &mut [i32],
    logits: &mut [i64],
    weights: &mut [u64],
    slots: impl Iterator<Item = usize>,
) -> Result<()> {
    let (k, model) = (cx.k, cx.model);
    let (vocab, d) = (model.dims.vocab, model.dims.d_model);
    let n = a.len() / d;
    let acc = acc.get_mut(..n * vocab).ok_or(internal(n * vocab, 0))?;
    k.matmul_i8(model.wte.data(), a, d, acc)?;
    let s = model.logit.s;
    for (ar, slot) in acc.chunks_exact(vocab).zip(slots) {
        let w = weights.get_mut(slot * vocab..(slot + 1) * vocab).ok_or(internal(slot, 0))?;
        for ((l, &v), &m) in logits.iter_mut().zip(ar).zip(model.logit.m.iter()) {
            *l = rescale_clamp(i64::from(v), m, s, -SCORE_LIMIT, SCORE_LIMIT);
        }
        weights_fast(logits, model.logits_exp.entries(), w)?;
    }
    Ok(())
}

/// The same result as `logits::logits_to_weights` (plan §7.6) for logits in
/// `[-2^40, 2^40]` and a table that `LogitTable::new` accepted: `w = table[min(max - l, len - 1)]`
/// with `0 <= max - l <= 2^41`. A property test compares the two.
///
/// # Errors
/// The errors of `logits_to_weights` for the same input.
pub(crate) fn weights_fast(logits: &[i64], table: &[u64], out: &mut [u64]) -> Result<()> {
    if logits.is_empty()
        || logits.len() > MAX_VOCAB
        || out.len() != logits.len()
        || table.is_empty()
    {
        // NOTE: the reference gives the exact error for these inputs.
        return logits_to_weights(logits, &LogitTable::new(table.to_vec())?, out);
    }
    let max = logits.iter().copied().max().unwrap_or(0);
    let last = table.len() - 1;
    for (o, &l) in out.iter_mut().zip(logits) {
        // INVARIANT: max >= l, so the difference is not negative (it fits: both are clamped).
        let d = usize::try_from(max.wrapping_sub(l)).unwrap_or(usize::MAX).min(last);
        *o = table.get(d).copied().unwrap_or(0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::weights_fast;
    use crate::logits::{LogitTable, logits_to_weights};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn weights_fast_equals_the_reference(
            len in 1usize..50,
            first in 1u64..=(1 << 40),
            logits in proptest::collection::vec(
                prop_oneof![-(1i64 << 40)..=(1i64 << 40), -40i64..40], 1..80),
        ) {
            let t: Vec<u64> = (0..len as u64).map(|i| first >> i.min(63)).collect();
            let mut want = vec![0u64; logits.len()];
            logits_to_weights(&logits, &LogitTable::new(t.clone()).unwrap(), &mut want).unwrap();
            let mut got = vec![0u64; logits.len()];
            weights_fast(&logits, &t, &mut got).unwrap();
            prop_assert_eq!(got, want);
        }
    }
}
