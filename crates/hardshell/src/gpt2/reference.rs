//! The reference GPT-2 session: one token at a time, with the scalar reference kernels.
//!
//! This is the code of the first HardShell session, unchanged except for the name. It is slow
//! (measured: 84.7 steps per second on one M2 thread, real `gpt2.hsm`, 1,024 positions). It is
//! the oracle: [`super::Session`] and [`super::Batch`] are the fast paths, and the tests in
//! `tests/fast_paths.rs` prove that they give the same weights as this session, bit for bit.
//! A GPU or other implementation can compare its output with it too (plan §11).

use super::Overlap;
use super::model::{Block, Gelu, Linear, Model, RowRescale};
use super::{RESIDUAL_LIMIT as RES, SCORE_LIMIT};
use crate::error::{Error, Result};
use crate::int::{clamp_act_i8, clamp_i32, clamp_wide, rescale};
use crate::kernels::{dot_i8, matvec_i8, matvec_i8_i16};
use crate::logits::logits_to_weights;
use crate::softmax::softmax;
use crate::types::TokenId;

/// The reference session: the KV cache, the position and the work buffers of one block.
///
/// It has the same API and gives the same weights as [`super::Session`] (DETERMINISM: every
/// step uses only the integer rules of `int`). Use it only to check other implementations.
#[derive(Debug)]
pub struct ReferenceSession<'m> {
    model: &'m Model,
    /// The overlap of a context restart (ADR 0015).
    overlap: Overlap,
    /// The number of tokens in the context.
    pos: usize,
    /// `[n_ctx]`: the token at each position of the context.
    tokens: Vec<usize>,
    /// `[n_layer, n_ctx, d_model]` keys and values.
    k_cache: Vec<i8>,
    v_cache: Vec<i8>,
    x: Vec<i32>,
    a: Vec<i8>,
    acc: Vec<i32>,
    qkv: Vec<i8>,
    att: Vec<i8>,
    g: Vec<i8>,
    /// The `i16` GELU outputs and the `i16` final LayerNorm outputs (format version 2), with
    /// their `i64` accumulators. Empty for a model without them.
    g16: Vec<i16>,
    a16: Vec<i16>,
    acc64: Vec<i64>,
    scores: Vec<i64>,
    probs: Vec<i64>,
    head_acc: Vec<i64>,
    logits: Vec<i64>,
    weights: Vec<u64>,
}

/// An internal length error. It cannot occur for a model that `Model::load` accepted.
fn internal(expected: usize, found: usize) -> Error {
    Error::LengthMismatch { expected, found }
}

/// `rescale(acc + b, m[r], s)` for row `r`, as an `i128`.
fn row_rescale(acc: i32, b: i32, r: &RowRescale, row: usize) -> i128 {
    let m = r.m.get(row).copied().unwrap_or(0);
    rescale(i64::from(acc) + i64::from(b), m, r.s)
}

impl<'m> ReferenceSession<'m> {
    /// Makes a session with an empty context and no context overlap. Cost: allocates the KV
    /// cache, `2 * n_layer * n_ctx * d_model` bytes.
    ///
    /// PERF: GPT-2 small needs 18.9 MB of cache (est.: 2 * 12 * 1,024 * 768 bytes). A model
    /// file with many layers and a long context can ask for much more memory than its own size;
    /// the decoder loads only model files with a known hash.
    #[must_use]
    pub fn new(model: &'m Model) -> Self {
        Self::with_overlap(model, Overlap::NONE)
    }

    /// Makes a session with an empty context and the context overlap `overlap` (ADR 0015).
    /// A session with an overlap that is not valid for `model` refuses every step
    /// ([`Model::check_overlap`]).
    #[must_use]
    pub fn with_overlap(model: &'m Model, overlap: Overlap) -> Self {
        let d = model.dims;
        let cache = d.n_layer * d.n_ctx * d.d_model;
        let acc_len = (3 * d.d_model).max(d.d_ff).max(d.vocab);
        let wide_gelu = model.blocks.iter().any(|b| b.gelu.is_wide());
        let wide_lnf = model.ln_f_bits == 16;
        let acc64_len = if wide_gelu || wide_lnf { d.d_model.max(d.vocab) } else { 0 };
        Self {
            model,
            overlap,
            pos: 0,
            tokens: vec![0; d.n_ctx],
            k_cache: vec![0; cache],
            v_cache: vec![0; cache],
            x: vec![0; d.d_model],
            a: vec![0; d.d_model],
            acc: vec![0; acc_len],
            qkv: vec![0; 3 * d.d_model],
            att: vec![0; d.d_model],
            g: vec![0; d.d_ff],
            g16: vec![0; if wide_gelu { d.d_ff } else { 0 }],
            a16: vec![0; if wide_lnf { d.d_model } else { 0 }],
            acc64: vec![0; acc64_len],
            scores: vec![0; d.n_ctx],
            probs: vec![0; d.n_ctx],
            head_acc: vec![0; d.head_dim],
            logits: vec![0; d.vocab],
            weights: vec![0; d.vocab],
        }
    }

    /// Empties the context. Cost: no memory is written; old cache entries are not read again.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// The number of tokens in the context.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Starts a block (plan §7.7): empties the context and runs `<|endoftext|>`. The result is
    /// the weights for the first token of the block.
    ///
    /// # Errors
    /// The errors of [`ReferenceSession::step`].
    pub fn start_block(&mut self) -> Result<&[u64]> {
        self.reset();
        self.step(self.model.eot_token())
    }

    /// Adds `token` to the context and runs the model. The result has one weight for each
    /// token of the vocabulary: the prediction for the next token (plan §7.6). The weight of
    /// the most likely token is at least 1, and the sum is at most `2^60`.
    ///
    /// If the context is full, the session starts again first: `<|endoftext|>` at position 0,
    /// the last `K` tokens at the positions `1..=K`, then `token` (ADR 0015). Cost: one forward
    /// pass, about `n_layer * 12 * d_model^2 + vocab * d_model` multiply-adds for GPT-2 shapes,
    /// plus the attention over the context; at a restart, also `K + 1` passes without logits.
    ///
    /// # Errors
    /// [`Error::TokenOutOfRange`] if `token` is not in the vocabulary;
    /// [`Error::InvalidOverlap`] if the overlap is not valid for the model.
    pub fn step(&mut self, token: TokenId) -> Result<&[u64]> {
        self.model.check_overlap(self.overlap)?;
        let tok = self.model.check_token(token)?;
        let n_ctx = self.model.dims.n_ctx;
        if self.pos >= n_ctx {
            // PLAN(§7.7): the context starts again with <|endoftext|> and the last K tokens.
            // The simple form: copy the kept tokens out, then run each one as a step.
            let k = self.overlap.tokens();
            let kept = self.tokens.get(n_ctx - k..n_ctx).ok_or(internal(k, 0))?.to_vec();
            self.reset();
            let eot = self.model.check_token(self.model.eot_token())?;
            self.forward(eot, false)?;
            for t in kept {
                self.forward(t, false)?;
            }
        }
        self.forward(tok, true)?;
        Ok(&self.weights)
    }

    /// One forward pass at position `self.pos`; then `pos` increases by one. With `logits`
    /// false, the pass stops after the blocks (a restart row, whose weights are not used).
    ///
    /// DETERMINISM: every value is an integer, every rounding is `int::rescale`, and every
    /// saturation is `int::clamp*`. The kernels may sum in any order (integer addition is
    /// associative and nothing overflows), so the weights are the same bits on every machine.
    fn forward(&mut self, tok: usize, logits: bool) -> Result<()> {
        let model = self.model;
        let t = self.pos;
        *self.tokens.get_mut(t).ok_or(internal(t, 0))? = tok;
        self.embed(tok, t)?;
        for (l, block) in model.blocks.iter().enumerate() {
            self.attention(block, l, t)?;
            self.mlp(block)?;
        }
        if !logits {
            self.pos = t + 1;
            return Ok(());
        }
        let vocab = model.dims.vocab;
        if model.ln_f_bits == 16 {
            // Format version 2: i16 outputs of ln_f and an i64 accumulator (ADR 0004 B).
            model.ln_f.apply_i16(model.ln_consts, &self.x, &mut self.a16)?;
            let acc = self.acc64.get_mut(..vocab).ok_or(internal(vocab, 0))?;
            matvec_i8_i16(&model.wte, &self.a16, acc)?;
            for (v, (l, &s)) in self.logits.iter_mut().zip(acc.iter()).enumerate() {
                let m = model.logit.m.get(v).copied().unwrap_or(0);
                *l = clamp_wide(rescale(s, m, model.logit.s), -SCORE_LIMIT, SCORE_LIMIT);
            }
        } else {
            model.ln_f.apply(model.ln_consts, &self.x, &mut self.a)?;
            let acc = self.acc.get_mut(..vocab).ok_or(internal(vocab, 0))?;
            matvec_i8(&model.wte, &self.a, acc)?;
            for (v, (l, &s)) in self.logits.iter_mut().zip(acc.iter()).enumerate() {
                let m = model.logit.m.get(v).copied().unwrap_or(0);
                *l = clamp_wide(rescale(i64::from(s), m, model.logit.s), -SCORE_LIMIT, SCORE_LIMIT);
            }
        }
        logits_to_weights(&self.logits, &model.logits_exp, &mut self.weights)?;
        self.pos = t + 1;
        Ok(())
    }

    /// Step 1 of the forward pass: token and position embeddings into the residual stream.
    fn embed(&mut self, tok: usize, t: usize) -> Result<()> {
        let model = self.model;
        let d = model.dims.d_model;
        let te = model.wte.row(tok).ok_or(internal(d, 0))?;
        let pe = model.wpe.row(t).ok_or(internal(d, 0))?;
        let tm = model.embed.m.get(tok).copied().unwrap_or(0);
        let pm = model.pos.m.get(t).copied().unwrap_or(0);
        for ((x, &a), &b) in self.x.iter_mut().zip(te).zip(pe) {
            let v =
                rescale(i64::from(a), tm, model.embed.s) + rescale(i64::from(b), pm, model.pos.s);
            *x = clamp_i32(v, -RES, RES);
        }
        Ok(())
    }

    /// `acc = lin.w * input`, then `x[r] = clamp(x[r] + rescale(acc[r] + b[r]), -R, R)`.
    fn add_to_residual(x: &mut [i32], acc: &mut [i32], lin: &Linear, input: &[i8]) -> Result<()> {
        matvec_i8(&lin.w, input, acc)?;
        for (r, (xv, (&s, &b))) in x.iter_mut().zip(acc.iter().zip(lin.b.iter())).enumerate() {
            let delta = row_rescale(s, b, &lin.rescale, r);
            *xv = clamp_i32(i128::from(*xv) + delta, -RES, RES);
        }
        Ok(())
    }

    /// Steps 2.1 to 2.4: LayerNorm, q, k, v, causal attention, output projection, residual.
    fn attention(&mut self, block: &Block, layer: usize, t: usize) -> Result<()> {
        let model = self.model;
        let dims = model.dims;
        let (d, hd) = (dims.d_model, dims.head_dim);
        block.ln_1.apply(model.ln_consts, &self.x, &mut self.a)?;
        let acc = self.acc.get_mut(..3 * d).ok_or(internal(3 * d, 0))?;
        matvec_i8(&block.c_attn.w, &self.a, acc)?;
        for (r, (o, (&s, &b))) in
            self.qkv.iter_mut().zip(acc.iter().zip(block.c_attn.b.iter())).enumerate()
        {
            *o = clamp_act_i8(row_rescale(s, b, &block.c_attn.rescale, r));
        }
        // Put k and v of this position into the cache.
        let layer_base = layer * dims.n_ctx * d;
        let at = layer_base + t * d;
        let (q, kv) = self.qkv.split_at(d);
        let (k, v) = kv.split_at(d);
        self.k_cache.get_mut(at..at + d).ok_or(internal(d, 0))?.copy_from_slice(k);
        self.v_cache.get_mut(at..at + d).ok_or(internal(d, 0))?.copy_from_slice(v);

        let n = t + 1;
        let keys = self.k_cache.get(layer_base..layer_base + n * d).ok_or(internal(n * d, 0))?;
        let values = self.v_cache.get(layer_base..layer_base + n * d).ok_or(internal(n * d, 0))?;
        let scores = self.scores.get_mut(..n).ok_or(internal(n, 0))?;
        let probs = self.probs.get_mut(..n).ok_or(internal(n, 0))?;
        for (h, (q_h, out_h)) in q.chunks_exact(hd).zip(self.att.chunks_exact_mut(hd)).enumerate() {
            let off = h * hd;
            for (sc, key) in scores.iter_mut().zip(keys.chunks_exact(d)) {
                let k_h = key.get(off..off + hd).ok_or(internal(hd, 0))?;
                let dot = dot_i8(q_h, k_h);
                *sc = clamp_wide(
                    rescale(i64::from(dot), block.score.m, block.score.s),
                    -SCORE_LIMIT,
                    SCORE_LIMIT,
                );
            }
            softmax(scores, &model.attn_exp, model.softmax_p, probs)?;
            self.head_acc.fill(0);
            for (&p, val) in probs.iter().zip(values.chunks_exact(d)) {
                let v_h = val.get(off..off + hd).ok_or(internal(hd, 0))?;
                for (acc, &vv) in self.head_acc.iter_mut().zip(v_h) {
                    // INVARIANT: 0 <= p <= 2^30, |v| <= 128 and sum(p) <= 2^30, so
                    // |acc| <= 2^37.
                    *acc += p * i64::from(vv);
                }
            }
            for (o, &a) in out_h.iter_mut().zip(self.head_acc.iter()) {
                *o = clamp_act_i8(rescale(a, block.attn_out.m, block.attn_out.s));
            }
        }
        let acc = self.acc.get_mut(..d).ok_or(internal(d, 0))?;
        Self::add_to_residual(&mut self.x, acc, &block.c_proj, &self.att)
    }

    /// Steps 2.5 to 2.7: LayerNorm, MLP with GELU, residual.
    fn mlp(&mut self, block: &Block) -> Result<()> {
        let model = self.model;
        let (d, f) = (model.dims.d_model, model.dims.d_ff);
        block.ln_2.apply(model.ln_consts, &self.x, &mut self.a)?;
        let acc = self.acc.get_mut(..f).ok_or(internal(f, 0))?;
        matvec_i8(&block.c_fc.w, &self.a, acc)?;
        let (lo, hi) = block.gelu.input_range();
        let inputs = acc
            .iter()
            .zip(block.c_fc.b.iter())
            .enumerate()
            .map(|(r, (&s, &b))| clamp_wide(row_rescale(s, b, &block.c_fc.rescale, r), lo, hi));
        match &block.gelu {
            Gelu::Narrow(table) => {
                for (o, x) in self.g.iter_mut().zip(inputs) {
                    *o = table.apply(x);
                }
                let acc = self.acc.get_mut(..d).ok_or(internal(d, 0))?;
                Self::add_to_residual(&mut self.x, acc, &block.mlp_proj, &self.g)
            }
            Gelu::Wide(table) => {
                // Format version 2: i16 GELU outputs and an i64 accumulator (ADR 0004 B).
                for (o, x) in self.g16.iter_mut().zip(inputs) {
                    *o = table.apply(x);
                }
                let acc = self.acc64.get_mut(..d).ok_or(internal(d, 0))?;
                matvec_i8_i16(&block.mlp_proj.w, &self.g16, acc)?;
                let lin = &block.mlp_proj;
                for (r, (xv, (&s, &b))) in
                    self.x.iter_mut().zip(acc.iter().zip(lin.b.iter())).enumerate()
                {
                    // INVARIANT: |s| <= d_ff * 2^22 <= 2^38 (MAX_DOT_LEN_I32 bounds d_ff), so
                    // s + b does not overflow.
                    let m = lin.rescale.m.get(r).copied().unwrap_or(0);
                    let delta = rescale(s + i64::from(b), m, lin.rescale.s);
                    *xv = clamp_i32(i128::from(*xv) + delta, -RES, RES);
                }
                Ok(())
            }
        }
    }
}
