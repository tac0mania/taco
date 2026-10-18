//! The `attention` op: causal multi-head attention over the KV cache of one block, for a run
//! of rows at consecutive positions (one row on the step and batch paths, many rows on the
//! prefill path).
//!
//! Per head, the op is three exact integer steps:
//!
//! 1. `dot[i][j] = q_i . k_j` for all rows `i` of the run and all positions `j` up to the last
//!    row: one matrix product, so each key is read once for all rows.
//! 2. For each row: `score = clamp(rescale(dot), -2^40, 2^40)` and `p = softmax(score)` over
//!    its own positions `0..=pos_i` (plan §7.5), as in the reference.
//! 3. `acc[c] = sum_j p[j] * v[j][c]`. `p` has up to 31 bits, so the op writes each `p` as four
//!    signed base-256 digits, `p = d0 + 2^8 d1 + 2^16 d2 + 2^24 d3` with each `d` in
//!    `-128..=127` ([`digits`]). Then `acc[c] = sum_k 2^(8k) (d_k . v[.][c])`: four `i8` dot
//!    products for each channel, with the values cache in transposed form (`[channel, pos]`).
//!
//! DETERMINISM: every step is exact. The digits give `p` exactly; each digit dot product has
//! at most `n_ctx <= 2^16` terms of magnitude `<= 2^14`, so it fits in an `i32`
//! (`kernels::MAX_DOT_LEN_I32`); and `|acc| <= 2^37` in the `i64` sum. Thus `acc` is the same
//! integer as the `i64` loop of the reference session, and the output is the same bits.

use super::SCORE_LIMIT;
use super::model::{Model, Rescale};
use super::ops::{Ctx, act_i8, internal};
use crate::error::Result;
use crate::kernels::{MatRef, rescale_clamp};
use crate::softmax::{MAX_P, softmax};

/// The four balanced base-256 digits of `p`: `p = d0 + 2^8 d1 + 2^16 d2 + 2^24 d3`, each digit
/// in `-128..=127`. Exact for `0 <= p <= 2^30` (the range of a softmax weight with `P <= 30`).
///
/// Proof: each round takes the digit `d = ((p + 128) mod 256) - 128`, so `p - d` is a multiple
/// of 256, and `p' = (p - d) / 256` is exact. For `0 <= p <= 2^30`, the values `p'` are
/// `>= 0` and at most `2^22`, `2^14`, then `2^6`, so the last digit is `p'` itself and nothing
/// remains. A unit test checks the identity at the ends and on random values.
#[must_use]
pub(crate) fn digits(p: i64) -> [i8; 4] {
    let mut rest = p;
    let mut out = [0i8; 4];
    for d in &mut out {
        // INVARIANT: `(rest + 128) & 255` is in 0..=255, so the digit is in -128..=127.
        let digit = ((rest + 128) & 255) - 128;
        *d = i8::try_from(digit).unwrap_or(0);
        rest = (rest - digit) >> 8;
    }
    out
}

/// The KV cache of one block. The keys are `[n_layer, n_head, n_ctx, head_dim]`, so the keys
/// of one head are one contiguous matrix with a row for each position. The values are
/// transposed, `[n_layer, n_head, head_dim, n_ctx]`, so each channel of one head is one row.
#[derive(Debug)]
pub(crate) struct KvCache {
    k: Vec<i8>,
    vt: Vec<i8>,
}

impl KvCache {
    /// An empty cache. Cost: allocates `2 * n_layer * n_ctx * d_model` bytes.
    pub(crate) fn new(model: &Model) -> Self {
        let d = model.dims;
        let len = d.n_layer * d.n_ctx * d.d_model;
        Self { k: vec![0; len], vt: vec![0; len] }
    }
}

/// Scratch space of the `attention` op for runs of up to `cap` rows.
#[derive(Debug)]
pub(crate) struct AttnScratch {
    cap: usize,
    /// `[cap, head_dim]`: the queries of one head.
    q: Vec<i8>,
    /// `[cap, n_ctx]`: `q . k` of one head.
    dots: Vec<i32>,
    scores: Vec<i64>,
    probs: Vec<i64>,
    /// `[4 * cap, n_ctx]`: the digits of `p` of each row.
    digits: Vec<i8>,
    /// `[4 * cap, head_dim]`: the digit dot products.
    pv: Vec<i32>,
}

impl AttnScratch {
    /// Scratch space for runs of up to `cap` rows. Cost: about `9 * cap * n_ctx` bytes.
    pub(crate) fn new(model: &Model, cap: usize) -> Self {
        let d = model.dims;
        let cap = cap.max(1);
        Self {
            cap,
            q: vec![0; cap * d.head_dim],
            dots: vec![0; cap * d.n_ctx],
            scores: vec![0; d.n_ctx],
            probs: vec![0; d.n_ctx],
            digits: vec![0; 4 * cap * d.n_ctx],
            pv: vec![0; 4 * cap * d.head_dim],
        }
    }
}

/// A run of rows of one block: `count` rows at the positions `pos0..pos0 + count` of layer
/// `layer`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Run {
    pub(crate) layer: usize,
    pub(crate) pos0: usize,
    pub(crate) count: usize,
}

/// `attention` for a run of rows: writes `k` and `v` of each row of `qkv` (rows of
/// `3 * d_model` values) into the cache, then the causal attention of each row over the
/// positions up to its own, into `out` (rows of `d_model` values).
///
/// Cost: for each head, about `n * head_dim * (count + 4 * count)` multiply-adds in `i8` dot
/// products, where `n = pos0 + count`, plus one softmax for each row.
///
/// # Errors
/// The errors of `softmax` and the kernels, or an internal length error.
pub(crate) fn attention(
    cx: Ctx<'_>,
    cache: &mut KvCache,
    run: Run,
    qkv: &[i8],
    s: &mut AttnScratch,
    out: &mut [i8],
) -> Result<()> {
    let model = cx.model;
    let dims = model.dims;
    let (d, hd, ctx) = (dims.d_model, dims.head_dim, dims.n_ctx);
    let block = model.blocks.get(run.layer).ok_or(internal(dims.n_layer, run.layer))?;
    let n = run.pos0 + run.count;
    if run.count == 0 || run.count > s.cap || n > ctx {
        return Err(internal(s.cap, run.count));
    }
    if qkv.len() != run.count * 3 * d || out.len() != run.count * d {
        return Err(internal(run.count * 3 * d, qkv.len()));
    }
    let layer_base = run.layer * ctx * d;
    // 1. Write the keys and values of all rows of the run.
    for (i, row) in qkv.chunks_exact(3 * d).enumerate() {
        let pos = run.pos0 + i;
        let (k_new, v_new) = row.get(d..).unwrap_or(&[]).split_at(d);
        for (h, (kh, vh)) in k_new.chunks_exact(hd).zip(v_new.chunks_exact(hd)).enumerate() {
            let head = layer_base + h * ctx * hd;
            let at = head + pos * hd;
            cache.k.get_mut(at..at + hd).ok_or(internal(hd, 0))?.copy_from_slice(kh);
            for (c, &v) in vh.iter().enumerate() {
                *cache.vt.get_mut(head + c * ctx + pos).ok_or(internal(ctx, 0))? = v;
            }
        }
    }
    // 2. Each head, for all rows at once.
    for h in 0..dims.n_head {
        let head = layer_base + h * ctx * hd;
        let q = s.q.get_mut(..run.count * hd).ok_or(internal(hd, 0))?;
        for (qr, row) in q.chunks_exact_mut(hd).zip(qkv.chunks_exact(3 * d)) {
            qr.copy_from_slice(row.get(h * hd..(h + 1) * hd).ok_or(internal(hd, 0))?);
        }
        let keys = cache.k.get(head..head + n * hd).ok_or(internal(n * hd, 0))?;
        let dots = s.dots.get_mut(..run.count * n).ok_or(internal(n, 0))?;
        cx.k.matmul_i8(keys, q, hd, dots)?;
        let dig = s.digits.get_mut(..4 * run.count * n).ok_or(internal(n, 0))?;
        for (i, (dot_row, dig_rows)) in
            dots.chunks_exact(n).zip(dig.chunks_exact_mut(4 * n)).enumerate()
        {
            let len = run.pos0 + i + 1;
            let probs = probabilities(cx, block.score, dot_row, len, &mut s.scores, &mut s.probs)?;
            let (r0, rest) = dig_rows.split_at_mut(n);
            let (r1, rest) = rest.split_at_mut(n);
            let (r2, r3) = rest.split_at_mut(n);
            for (((p, d0), (d1, d2)), d3) in probs
                .iter()
                .zip(r0.iter_mut())
                .zip(r1.iter_mut().zip(r2.iter_mut()))
                .zip(r3.iter_mut())
            {
                [*d0, *d1, *d2, *d3] = digits(*p);
            }
            // The positions after the row get the digit 0, so they add nothing.
            for r in [r0, r1, r2, r3] {
                r.get_mut(len..).unwrap_or(&mut []).fill(0);
            }
        }
        let values = MatRef {
            data: cache.vt.get(head..head + hd * ctx).ok_or(internal(hd * ctx, 0))?,
            rows: hd,
            stride: ctx,
        };
        let pv = s.pv.get_mut(..4 * run.count * hd).ok_or(internal(hd, 0))?;
        cx.k.matmul_i8_strided(values, dig, n, pv)?;
        let Rescale { m, s: shift } = block.attn_out;
        for (pv4, out_row) in pv.chunks_exact(4 * hd).zip(out.chunks_exact_mut(d)) {
            let out_h = out_row.get_mut(h * hd..(h + 1) * hd).ok_or(internal(hd, 0))?;
            for (c, o) in out_h.iter_mut().enumerate() {
                let mut acc = 0i64;
                for k in (0..4).rev() {
                    // INVARIANT: |d_k . v| <= 2^30, so |acc| < 2^55 at every step.
                    acc = (acc << 8) + i64::from(pv4.get(k * hd + c).copied().unwrap_or(0));
                }
                *o = act_i8(rescale_clamp(acc, m, shift, -127, 127));
            }
        }
    }
    Ok(())
}

/// The softmax weights of one row over its first `len` positions (plan §7.5).
fn probabilities<'s>(
    cx: Ctx<'_>,
    score: Rescale,
    dots: &[i32],
    len: usize,
    scores: &mut [i64],
    probs: &'s mut [i64],
) -> Result<&'s [i64]> {
    let scores = scores.get_mut(..len).ok_or(internal(len, 0))?;
    for (sc, &dot) in scores.iter_mut().zip(dots) {
        *sc = rescale_clamp(i64::from(dot), score.m, score.s, -SCORE_LIMIT, SCORE_LIMIT);
    }
    let probs = probs.get_mut(..len).ok_or(internal(len, 0))?;
    softmax_fast(scores, cx.model.attn_exp.entries(), cx.model.softmax_p, probs)?;
    Ok(probs)
}

/// The same result as `softmax::softmax` (plan §7.5) for a table that `ExpTable::new`
/// accepted, with `i64` and `u64` arithmetic only. A property test compares the two.
///
/// The steps of the reference: `d = max - s` (here `0 <= d <= 2^41`, because every score is
/// in `[-2^40, 2^40]`), `e = table[min(d, len - 1)]`, `p = floor((e << P) / sum(e))`. Every `e`
/// is in `0..2^31` and `sum >= 1`, so the floor division of the reference is the unsigned
/// division here.
///
/// # Errors
/// The errors of `softmax::softmax` for the same input.
pub(crate) fn softmax_fast(scores: &[i64], table: &[i32], p: u8, out: &mut [i64]) -> Result<()> {
    if p > MAX_P || scores.is_empty() || out.len() < scores.len() || table.is_empty() {
        // NOTE: the reference gives the exact error for these inputs.
        return softmax(scores, &crate::softmax::ExpTable::new(table.to_vec())?, p, out);
    }
    let max = scores.iter().copied().max().unwrap_or(0);
    let last = table.len() - 1;
    let mut sum: u64 = 0;
    for (o, &s) in out.iter_mut().zip(scores) {
        // INVARIANT: max >= s, and both are within the score limit, so d fits in an i64.
        let d = usize::try_from(max.wrapping_sub(s)).unwrap_or(usize::MAX).min(last);
        let e = u64::try_from(table.get(d).copied().unwrap_or(0)).unwrap_or(0);
        *o = i64::try_from(e).unwrap_or(0);
        // INVARIANT: at most 2^31 entries below 2^31, so sum < 2^62.
        sum += e;
    }
    let sum = sum.max(1);
    for o in out.iter_mut().take(scores.len()) {
        // INVARIANT: e < 2^31 and p <= 30, so e << p < 2^61, and the quotient fits in i64.
        let e = u64::try_from(*o).unwrap_or(0);
        *o = i64::try_from((e << p) / sum).unwrap_or(0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{digits, softmax_fast};
    use crate::softmax::{ExpTable, softmax};
    use proptest::prelude::*;

    /// A valid exponent table: the first entry at least 1, not increasing, below 2^31.
    fn table() -> impl Strategy<Value = Vec<i32>> {
        proptest::collection::vec(0i32..1 << 20, 1..40).prop_map(|steps| {
            let mut v = Vec::with_capacity(steps.len());
            let mut e: i64 = (1 << 31) - 1;
            for s in steps {
                v.push(i32::try_from(e.max(0)).unwrap());
                e -= i64::from(s) * 97;
            }
            v[0] = v[0].max(1);
            v
        })
    }

    proptest! {
        #[test]
        fn softmax_fast_equals_the_reference(
            t in table(),
            scores in proptest::collection::vec(
                prop_oneof![-(1i64 << 40)..=(1i64 << 40), -50i64..50, Just(1i64 << 40)], 1..60),
            p in 0u8..=30,
        ) {
            let mut want = vec![0i64; scores.len()];
            softmax(&scores, &ExpTable::new(t.clone()).unwrap(), p, &mut want).unwrap();
            let mut got = vec![0i64; scores.len()];
            softmax_fast(&scores, &t, p, &mut got).unwrap();
            prop_assert_eq!(got, want);
        }
    }

    fn value(d: [i8; 4]) -> i64 {
        d.iter().rev().fold(0i64, |acc, &x| (acc << 8) + i64::from(x))
    }

    #[test]
    fn digits_are_exact_at_the_ends() {
        for p in [0i64, 1, 127, 128, 255, 256, (1 << 30) - 1, 1 << 30, 0x7f7f_7f7f >> 1] {
            assert_eq!(value(digits(p)), p, "p = {p}");
        }
    }

    proptest! {
        #[test]
        fn digits_are_exact(p in 0i64..=(1 << 30)) {
            prop_assert_eq!(value(digits(p)), p);
        }
    }
}
