//! The GPT-2 forward pass for a batch of rows, composed from the op kernels of `ops`.
//!
//! A row is one token at one position of one block (a "lane"). One call runs all layers for
//! all rows: each `linear` op reads its weights once for all rows. The rows of one call can be
//! many positions of one block (the prefill path of [`super::Session::step_many`]), one
//! position of many blocks (the batch path of [`super::Batch`]), or a mix.
//!
//! DETERMINISM: every op computes each row with the same integer rules as the reference
//! session, and no op mixes values of two rows, except through the KV cache of their own
//! block. Thus the weights of a row do not depend on the other rows of the call.

use super::Model;
use super::attention::{self, AttnScratch, KvCache, Run};
use super::ops::{self, Ctx, LinearOut, internal};
use crate::error::Result;
use crate::kernels::Kernels;

/// One row of a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    /// The index of the KV cache (the block) of the row.
    pub(crate) lane: usize,
    /// The position of the token in the context. The rows of one lane in one call have
    /// increasing positions.
    pub(crate) pos: usize,
    /// The token, already checked against the vocabulary.
    pub(crate) tok: usize,
    /// The output slot of the weights, or `None` if the caller does not need them (the
    /// `<|endoftext|>` row of a context restart).
    pub(crate) out: Option<usize>,
}

/// The work buffers for up to `cap` rows, and `slots` rows of output weights.
#[derive(Debug)]
pub(crate) struct Work {
    cap: usize,
    x: Vec<i32>,
    a: Vec<i8>,
    qkv: Vec<i8>,
    att: Vec<i8>,
    g: Vec<i8>,
    acc: Vec<i32>,
    /// The `i16` GELU outputs and the `i16` final LayerNorm outputs of a model of hsm format
    /// version 2 (ADR 0004 addendum B), and their `i64` accumulators. Empty for other models.
    g16: Vec<i16>,
    a16: Vec<i16>,
    acc64: Vec<i64>,
    attn: AttnScratch,
    logits: Vec<i64>,
    weights: Vec<u64>,
}

impl Work {
    /// Buffers for `cap >= 1` rows and `slots >= 1` output rows. Cost: about
    /// `cap * (4 * vocab + 9 * d_model + 2 * d_ff) + 8 * slots * vocab` bytes.
    pub(crate) fn new(model: &Model, cap: usize, slots: usize) -> Self {
        let d = model.dims;
        let (cap, slots) = (cap.max(1), slots.max(1));
        let acc_row = (3 * d.d_model).max(d.d_ff).max(d.vocab);
        let wide_gelu = model.blocks.iter().any(|b| b.gelu.is_wide());
        let wide_lnf = model.ln_f_bits == 16;
        let acc64_row = match (wide_lnf, wide_gelu) {
            (true, _) => d.vocab.max(d.d_model),
            (false, true) => d.d_model,
            (false, false) => 0,
        };
        Self {
            cap,
            x: vec![0; cap * d.d_model],
            a: vec![0; cap * d.d_model],
            qkv: vec![0; cap * 3 * d.d_model],
            att: vec![0; cap * d.d_model],
            g: vec![0; cap * d.d_ff],
            acc: vec![0; cap * acc_row],
            g16: vec![0; if wide_gelu { cap * d.d_ff } else { 0 }],
            a16: vec![0; if wide_lnf { cap * d.d_model } else { 0 }],
            acc64: vec![0; cap * acc64_row],
            attn: AttnScratch::new(model, cap),
            logits: vec![0; d.vocab],
            weights: vec![0; slots * d.vocab],
        }
    }

    /// The largest number of rows of one call of [`forward`].
    pub(crate) fn cap(&self) -> usize {
        self.cap
    }

    /// The weights of output slot `slot`, or `None` if there is no such slot.
    pub(crate) fn weights(&self, vocab: usize, slot: usize) -> Option<&[u64]> {
        self.weights.get(slot * vocab..(slot + 1) * vocab)
    }

    /// The weights of the output slots `0..n`.
    pub(crate) fn weights_prefix(&self, vocab: usize, n: usize) -> Option<&[u64]> {
        self.weights.get(..n * vocab)
    }
}

/// Runs the forward pass for `rows` (at most `work.cap()` rows) and writes the weights of each
/// row with an output slot. `caches[row.lane]` is the KV cache of each row. The caller keeps
/// the positions of the lanes.
///
/// # Errors
/// An internal length error if a row names a lane, a position or a slot that does not exist;
/// the errors of the ops.
pub(crate) fn forward(
    k: Kernels,
    model: &Model,
    caches: &mut [KvCache],
    rows: &[Row],
    work: &mut Work,
) -> Result<()> {
    let cx = Ctx { k, model };
    let dims = model.dims;
    let (d, f, n) = (dims.d_model, dims.d_ff, rows.len());
    if n > work.cap {
        return Err(internal(work.cap, n));
    }
    let x = work.x.get_mut(..n * d).ok_or(internal(n * d, 0))?;
    for (row, xr) in rows.iter().zip(x.chunks_exact_mut(d)) {
        ops::embed(model, row.tok, row.pos, xr)?;
    }
    let a = work.a.get_mut(..n * d).ok_or(internal(n * d, 0))?;
    let qkv = work.qkv.get_mut(..n * 3 * d).ok_or(internal(n * 3 * d, 0))?;
    let att = work.att.get_mut(..n * d).ok_or(internal(n * d, 0))?;
    let g = work.g.get_mut(..n * f).ok_or(internal(n * f, 0))?;
    for (layer, block) in model.blocks.iter().enumerate() {
        ops::norm(&block.ln_1, model.ln_consts, d, x, a)?;
        ops::linear(k, &block.c_attn, a, &mut work.acc, LinearOut::Act(qkv))?;
        let mut start = 0;
        for len in runs(rows) {
            let lane = rows.get(start).map_or(0, |r| r.lane);
            let pos0 = rows.get(start).map_or(0, |r| r.pos);
            let cache = caches.get_mut(lane).ok_or(internal(lane, 0))?;
            let q = qkv.get(start * 3 * d..(start + len) * 3 * d).ok_or(internal(len, 0))?;
            let o = att.get_mut(start * d..(start + len) * d).ok_or(internal(len, 0))?;
            let run = Run { layer, pos0, count: len };
            attention::attention(cx, cache, run, q, &mut work.attn, o)?;
            start += len;
        }
        ops::linear(k, &block.c_proj, att, &mut work.acc, LinearOut::Residual(x))?;
        ops::norm(&block.ln_2, model.ln_consts, d, x, a)?;
        if block.gelu.is_wide() {
            // hsm format version 2: i16 GELU outputs (ADR 0004 addendum B).
            let g16 = work.g16.get_mut(..n * f).ok_or(internal(n * f, 0))?;
            let out = LinearOut::GeluWide(g16, &block.gelu);
            ops::linear(k, &block.c_fc, a, &mut work.acc, out)?;
            ops::linear_wide_residual(k, &block.mlp_proj, g16, &mut work.acc64, x)?;
        } else {
            ops::linear(k, &block.c_fc, a, &mut work.acc, LinearOut::Gelu(g, &block.gelu))?;
            ops::linear(k, &block.mlp_proj, g, &mut work.acc, LinearOut::Residual(x))?;
        }
    }
    if model.ln_f_bits == 16 {
        // hsm format version 2: i16 outputs of the final LayerNorm (ADR 0004 addendum B). The
        // same steps as below, with i16 rows and i64 accumulators.
        let a16 = work.a16.get_mut(..n * d).ok_or(internal(n * d, 0))?;
        let mut packed = 0;
        for (row, xr) in rows.iter().zip(x.chunks_exact(d)) {
            if row.out.is_some() {
                let ar = a16.get_mut(packed * d..(packed + 1) * d).ok_or(internal(d, 0))?;
                model.ln_f.apply_i16(model.ln_consts, xr, ar)?;
                packed += 1;
            }
        }
        if packed == 0 {
            return Ok(());
        }
        let a16 = a16.get(..packed * d).ok_or(internal(packed * d, 0))?;
        let slots = rows.iter().filter_map(|r| r.out);
        let (acc, logits, weights) = (&mut work.acc64, &mut work.logits, &mut work.weights);
        return ops::logits_wide(cx, a16, acc, logits, weights, slots);
    }
    // The final LayerNorm and the logits, only for the rows with an output slot. Their rows
    // of `a` are packed at the front, in row order.
    let mut packed = 0;
    for (row, xr) in rows.iter().zip(x.chunks_exact(d)) {
        if row.out.is_some() {
            let ar = a.get_mut(packed * d..(packed + 1) * d).ok_or(internal(d, 0))?;
            model.ln_f.apply(model.ln_consts, xr, ar)?;
            packed += 1;
        }
    }
    if packed == 0 {
        return Ok(());
    }
    let a = a.get(..packed * d).ok_or(internal(packed * d, 0))?;
    let slots = rows.iter().filter_map(|r| r.out);
    ops::logits(cx, a, &mut work.acc, &mut work.logits, &mut work.weights, slots)
}

/// The lengths of the runs of `rows`: the longest groups of rows of one lane at consecutive
/// positions, in row order.
fn runs(rows: &[Row]) -> impl Iterator<Item = usize> + '_ {
    rows.chunk_by(|a, b| a.lane == b.lane && b.pos == a.pos + 1).map(<[Row]>::len)
}
