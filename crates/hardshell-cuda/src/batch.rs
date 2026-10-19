//! Many independent GPT-2 sessions on one CUDA device, and prefill of many positions at once.
//!
//! A batch has `slots` sessions; each slot has its own KV cache and position. One call runs a
//! list of (slot, token) inputs. A slot can occur many times in the list: its tokens then run
//! at consecutive positions in one pass (prefill), so one read of the weights serves all rows.
//! Each row gives the weights that `hardshell::gpt2::Session::step` gives for the same history,
//! bit for bit.
//!
//! The host composes the op kernels into the forward pass of the `hardshell::gpt2` module
//! documentation. The residual steps 2.4 and 2.7 are a `linear` op with an i32 output and an
//! `add` op: `clamp(x + clamp(delta, i32), -R, R)` equals `clamp(x + delta, -R, R)`, because
//! `|x| <= R < 2^30`, so a delta outside the i32 range gives the limit in both forms.

use std::cell::RefCell;
use std::collections::HashMap;

use hardshell::TokenId;
use hardshell::gpt2::{Overlap, StreamState};

use crate::device::{bytes_u64, u32_bytes, u32_of};
use crate::driver::{Arg, DevBuf, Graph};
use crate::error::{Error, Result};
use crate::kernels::padded_rows;
use crate::model::CudaModel;
use crate::ops::MatKernel;

mod fused;
mod mixed;
mod plan;
mod record;
mod stream;

pub(crate) use mixed::Mixing;

/// A question to the coder op for one input (see [`CudaBatch::run_coded`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoderQuery {
    /// The interval of this token (the encoder).
    Encode(TokenId),
    /// The symbol whose interval holds this target, `target < 2^freq_bits` (the decoder).
    Decode(u64),
}

/// The interval of one symbol in the frequency table of `taco_coder::FreqTable::from_weights`:
/// the values that the range coder needs. The range coder defines these values, so the struct
/// does not grow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoderInterval {
    /// The symbol.
    pub symbol: TokenId,
    /// The sum of the frequencies of the lower symbols.
    pub start: u64,
    /// The frequency of the symbol, at least 1.
    pub freq: u64,
    /// The symbol is the last symbol of the table.
    pub is_last: bool,
}

/// What a pass gives back.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Out {
    /// Nothing; wait for the GPU.
    Nothing,
    /// The weights of each row.
    Weights,
    /// The coder interval of each row (with `freq_bits`).
    Intervals(u32),
    /// The mixer statistics of each row, then (after the host step) the coder interval of the
    /// mixed weights of each row (with `freq_bits`). See `mixed`.
    Mixed(u32),
}

/// What a call asks of each pass: the output, the coder query words of the inputs, and for
/// [`Out::Mixed`] the predictions and the host step of the mixer.
struct Ask<'q, 'm> {
    out: Out,
    queries: &'q [(u32, u32)],
    mixing: Option<&'q mut Mixing<'m>>,
}

impl<'q> Ask<'q, '_> {
    /// A call with no mixer.
    fn new(out: Out, queries: &'q [(u32, u32)]) -> Self {
        Self { out, queries, mixing: None }
    }
}

/// One row of a pass: a token of a slot at a position. `input` is the index of the input that
/// the row answers; a restart row (`<|endoftext|>` at position 0) answers no input.
#[derive(Clone, Copy)]
struct Row {
    slot: usize,
    token: TokenId,
    pos: usize,
    input: Option<usize>,
}

/// The output of a `linear` op: packed i8 or i32 values, with the row stride.
#[derive(Clone, Copy)]
enum LinOut<'b> {
    I8(&'b DevBuf, u32),
    I32(&'b DevBuf, u32),
}

/// The work buffers of a batch.
struct Bufs {
    rows: DevBuf,
    x: DevBuf,
    a: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    g: DevBuf,
    acc: DevBuf,
    tmp: DevBuf,
    sc: DevBuf,
    logit: DevBuf,
    weights: DevBuf,
    queries: DevBuf,
    intervals: DevBuf,
    /// The compact logits of the fused path: u16 differences and the tile maxima (i64).
    d16: DevBuf,
    tmax: DevBuf,
    stamps: DevBuf,
    /// The row groups of `attend_rows`: (first row, count) for each group.
    groups: DevBuf,
    /// The mix record of each row (8 u32 words, `kernels/hardshell.cu`).
    mix: DevBuf,
    /// The mixer statistics of each row: W, w[p] and the top token (u64).
    stats: DevBuf,
    /// The intervals of GPT-2 alone of a dual mixed pass (symbol, start, freq).
    plain: DevBuf,
}

/// Many sessions of one model on one CUDA device.
pub struct CudaBatch<'m> {
    model: &'m CudaModel<'m>,
    slots: usize,
    max_rows: usize,
    pos: Vec<usize>,
    /// The context overlap of every slot (ADR 0015).
    overlap: Overlap,
    /// `[slots, n_ctx]`: the token at each position of the context of each slot, for the
    /// overlap of a restart.
    hist: Vec<TokenId>,
    /// The inputs of each slot, for a truncation (ADR 0017, `hardshell::gpt2::StreamState`).
    streams: Vec<StreamState>,
    /// The history of the full context of each slot before its last restart (empty before
    /// the first restart), for a truncation before the restart.
    prev: Vec<Vec<TokenId>>,
    /// `[slots, vocab]`: the weights of the last [`CudaBatch::step`] of each slot.
    host: Vec<u64>,
    b: Bufs,
    kv: Vec<(DevBuf, DevBuf)>,
    tmpn: u32,
    /// The most split-K partials that fit in `acc`.
    max_splits: u32,
    /// The recorded graphs of the pass shapes that ran.
    graphs: RefCell<HashMap<GraphKey, Graph>>,
    bytes: u64,
    /// Run one kernel for each op (the reference structure) instead of the fused kernels.
    op_level: bool,
}

impl std::fmt::Debug for CudaBatch<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBatch")
            .field("slots", &self.slots)
            .field("max_rows", &self.max_rows)
            .finish_non_exhaustive()
    }
}

/// GPU timer marks of a profiled pass: one `stamp` kernel after each op.
struct Stamps<'a> {
    buf: &'a DevBuf,
    labels: Vec<&'static str>,
}

/// Adds a timer mark named `name` after the kernels before it, if `stamps` is set.
fn mark(
    dev: &crate::device::CudaDevice,
    stamps: &mut Option<Stamps<'_>>,
    name: &'static str,
) -> Result<()> {
    if let Some(st) = stamps.as_mut()
        && st.labels.len() < MAX_STAMPS
    {
        let i = u32_of(st.labels.len(), "too many marks")?;
        dev.launch(&dev.k.stamp, (1, 1), 1, &[Arg::Ptr(st.buf.ptr()), Arg::U32(i)])?;
        st.labels.push(name);
    }
    Ok(())
}

/// The columns of one tile of the logits product with the compact epilogue (`BN` of the
/// tensor-core kernels).
const LOGIT_TILE: u32 = 128;

/// The most tiles of compact logits that `coder_d16` keeps in shared memory (`CODER_TILES`).
const CODER_TILES: u32 = 2048;

/// The block size of the `layer_norm` and `residual_norm` kernels (`NORM_T` in
/// kernels/hardshell.cu): one block for each row.
const NORM_THREADS: u32 = 128;

/// The block size of the `attend` kernel (`ATT_T` in kernels/hardshell.cu).
const ATTEND_THREADS: u32 = 128;

/// The most rows of a group of `attend_rows` (`ATG_ROWS` in kernels/hardshell.cu).
const ATTEND_GROUP_ROWS: u32 = 16;

/// The longest context whose scores `attend` keeps in shared memory (`ATT_CTX`); a longer one
/// uses 12 bytes of `sc` for each position, row and head.
const ATTEND_SMEM_CTX: u32 = 1024;

/// The most timer marks of one profiled pass.
const MAX_STAMPS: usize = 1024;

/// The key of a recorded graph: the number of rows, the output (0: no coder op; `MIXED_GRAPH`:
/// the mixer statistics; otherwise `freq_bits` of the coder op), and the number of row groups of
/// the attention (0: one block for each row).
type GraphKey = (u32, u32, u32);

/// The output part of the key of a graph that ends with the mixer statistics (above 32).
const MIXED_GRAPH: u32 = 64;

impl<'m> CudaBatch<'m> {
    /// Makes a batch of `slots` sessions with empty contexts, and work buffers for passes of
    /// up to `max_rows` rows. Cost: the KV cache (`2 * n_layer * n_ctx * d_model` bytes for each
    /// slot) and about `20 * vocab` bytes for each row (`24 * vocab` for a model of model-file
    /// version 2, whose `acc` holds two operand rows for each row; est. from the buffer sizes).
    ///
    /// # Errors
    /// [`Error::Unsupported`] if `slots` or `max_rows` is 0; [`Error::Cuda`] if the device has
    /// not enough memory. A `max_rows` of 1 becomes 2, because a restart needs two rows.
    pub fn new(model: &'m CudaModel<'m>, slots: usize, max_rows: usize) -> Result<Self> {
        if slots == 0 || max_rows == 0 {
            return Err(Error::Unsupported {
                reason: "a batch needs at least one slot and one row",
            });
        }
        let dev = model.dev;
        let d = model.dims;
        // NOTE: a restart needs two rows (<|endoftext|> and the token) in one pass.
        let max_rows = max_rows.max(2);
        let (r, rp) = (max_rows, padded_rows(max_rows));
        // NOTE: an i16 activation (model-file version 2) is two i8 operand rows (the planes of
        // `wide_dot` in kernels/hardshell.cu): `a` (ln_f), `g` (GELU) and `acc` hold two times
        // the rows for such a model.
        let gelu_wide = model.layers.iter().any(|l| l.gelu_wide);
        let rp_a = if model.ln_f_wide { padded_rows(2 * max_rows) } else { rp };
        let rp_g = if gelu_wide { padded_rows(2 * max_rows) } else { rp };
        let acc_rows = if gelu_wide || model.ln_f_wide { 2 * r } else { r };
        let accn = (3 * d.d).max(d.d_ff).max(d.vocab);
        let max_splits = (accn / (3 * d.d).max(d.d_ff).max(1)).clamp(1, 16);
        let tmpn = d.d.max(d.d_ff);
        let us = |x: u32| x as usize;
        let ctx4 = d.n_ctx.next_multiple_of(4);
        let sizes = [
            16 * r,
            4 * r * us(d.d),
            4 * rp_a * us(d.dw),
            r * 3 * us(d.d),
            4 * rp * us(d.dw),
            4 * rp_g * us(d.fw),
            4 * acc_rows * us(accn),
            4 * r * us(tmpn),
            if ctx4 > ATTEND_SMEM_CTX { 12 * r * us(d.n_head) * us(ctx4) } else { 0 },
            8 * r * us(d.vocab),
            8 * r * us(d.vocab),
            16 * r,
            24 * r,
            8 * MAX_STAMPS,
            2 * r * us(d.vocab),
            8 * r * us(d.vocab.div_ceil(LOGIT_TILE)),
            8 * r,
            32 * r,
            24 * r,
            24 * r,
        ];
        let mut it = sizes.iter().map(|&s| dev.zeroed(s));
        let mut next = || it.next().unwrap_or_else(|| dev.zeroed(0));
        let b = Bufs {
            rows: next()?,
            x: next()?,
            a: next()?,
            qkv: next()?,
            att: next()?,
            g: next()?,
            acc: next()?,
            tmp: next()?,
            sc: next()?,
            logit: next()?,
            weights: next()?,
            queries: next()?,
            intervals: next()?,
            stamps: next()?,
            d16: next()?,
            tmax: next()?,
            groups: next()?,
            mix: next()?,
            stats: next()?,
            plain: next()?,
        };
        let cache = slots * us(ctx4) * us(d.d);
        let kv = (0..d.n_layer)
            .map(|_| Ok((dev.zeroed(cache)?, dev.zeroed(cache)?)))
            .collect::<Result<Vec<_>>>()?;
        let bytes =
            sizes.iter().map(|&s| s as u64).sum::<u64>() + 2 * u64::from(d.n_layer) * cache as u64;
        Ok(Self {
            model,
            slots,
            max_rows,
            pos: vec![0; slots],
            overlap: Overlap::NONE,
            hist: vec![TokenId::new(0); slots * model.n_ctx()],
            streams: vec![StreamState::new(); slots],
            prev: vec![Vec::new(); slots],
            host: vec![0; slots * model.vocab_size()],
            b,
            kv,
            tmpn,
            max_splits,
            graphs: RefCell::new(HashMap::new()),
            bytes,
            op_level: false,
        })
    }

    /// Selects the op-level path (one kernel for each op, ADR 0013) or the fused path (the
    /// default: fused kernels that a test proves equal to the op-level path). Both give the same
    /// bits; the op-level path is for tests and measurements. Cost: the recorded graphs are
    /// dropped.
    pub fn set_op_level(&mut self, op_level: bool) {
        if self.op_level != op_level {
            self.op_level = op_level;
            self.graphs.borrow_mut().clear();
        }
    }

    /// Sets the context overlap of every slot (ADR 0015): a slot whose context is full starts
    /// again with `<|endoftext|>` and its last `K` tokens, as `hardshell::gpt2::Session` does.
    /// Call it before the first pass. The `K + 1` restart rows run in passes of at most
    /// [`CudaBatch::max_rows`] rows.
    ///
    /// PERF: a restart row runs the full pass, also the logits, whose result is not used (est.
    /// 31 % of the multiply-adds of a GPT-2 small row: `vocab * d` of about
    /// `12 * 12 * d^2 + vocab * d`).
    ///
    /// # Errors
    /// [`Error::Model`] with `InvalidOverlap` if `K > n_ctx - 2`.
    pub fn set_overlap(&mut self, overlap: Overlap) -> Result<()> {
        let n_ctx = self.model.n_ctx();
        if overlap.tokens().saturating_add(2) > n_ctx {
            return Err(Error::Model(hardshell::Error::InvalidOverlap {
                overlap: overlap.get(),
                n_ctx,
            }));
        }
        self.overlap = overlap;
        Ok(())
    }

    /// The context overlap of the slots (ADR 0015).
    #[must_use]
    pub fn overlap(&self) -> Overlap {
        self.overlap
    }

    /// The number of slots.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The most rows of one pass.
    #[must_use]
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// The bytes of GPU memory that this batch allocated (KV caches and work buffers).
    #[must_use]
    pub fn gpu_bytes(&self) -> u64 {
        self.bytes
    }

    /// The number of tokens in the context of `slot` (0 for a slot that is not valid).
    #[must_use]
    pub fn position(&self, slot: usize) -> usize {
        self.pos.get(slot).copied().unwrap_or(0)
    }

    /// Empties the context of `slot`.
    pub fn reset(&mut self, slot: usize) {
        if let Some(p) = self.pos.get_mut(slot) {
            *p = 0;
        }
        if let Some(s) = self.streams.get_mut(slot) {
            s.clear();
        }
    }

    /// The weights of the last [`CudaBatch::step`] that included `slot`.
    #[must_use]
    pub fn weights(&self, slot: usize) -> Option<&[u64]> {
        let v = self.model.vocab_size();
        self.host.get(slot * v..(slot + 1) * v)
    }

    /// Adds one token to each listed slot (each slot at most once) and keeps the weights of
    /// each slot for [`CudaBatch::weights`]. Cost: one pass; one read back of
    /// `inputs.len() * vocab * 8` bytes.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::run_weights`]; [`Error::BadSlot`] if a slot occurs two times.
    pub fn step(&mut self, inputs: &[(usize, TokenId)]) -> Result<()> {
        let mut seen = vec![false; self.slots];
        for &(s, _) in inputs {
            match seen.get_mut(s) {
                Some(x) if !*x => *x = true,
                _ => return Err(Error::BadSlot { slot: s, slots: self.slots }),
            }
        }
        let w = self.run_weights(inputs)?;
        let v = self.model.vocab_size();
        for (&(s, _), src) in inputs.iter().zip(w.chunks(v)) {
            if let Some(dst) = self.host.get_mut(s * v..(s + 1) * v) {
                dst.copy_from_slice(src);
            }
        }
        Ok(())
    }

    /// Runs the inputs and gives the weights of each input, in input order (`inputs.len()`
    /// times `vocab` values). A slot can occur many times: its tokens run at consecutive
    /// positions (prefill). A slot whose context is full starts again first, as on the CPU
    /// (plan §7.7, ADR 0015). Cost: one pass for each `max_rows` rows (and more at a restart:
    /// `K + 1` more rows).
    ///
    /// # Errors
    /// [`Error::BadSlot`] if a slot is not valid; [`Error::Model`] with `TokenOutOfRange` if a
    /// token is not in the vocabulary; [`Error::Cuda`] on a driver error. After an error, no
    /// position changed.
    pub fn run_weights(&mut self, inputs: &[(usize, TokenId)]) -> Result<Vec<u64>> {
        let mut out = vec![0u64; inputs.len() * self.model.vocab_size()];
        self.run(inputs, Ask::new(Out::Weights, &[]), &mut out, None)?;
        Ok(out)
    }

    /// Runs the inputs as [`CudaBatch::run_weights`] does, then makes the coder interval of
    /// each input on the GPU and reads back only the intervals (24 bytes for each input). The
    /// interval is the interval of `taco_coder::FreqTable::from_weights(weights, freq_bits)`
    /// (plan §5.1): for [`CoderQuery::Encode`], of the given token; for [`CoderQuery::Decode`],
    /// of the symbol whose interval holds the target. `queries[i]` belongs to `inputs[i]`.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::run_weights`]; [`Error::LengthMismatch`] if the lengths
    /// differ; [`Error::Unsupported`] if `freq_bits` is not in `1..=32`, `2^freq_bits` is less
    /// than the vocabulary, or a token or target is out of range.
    pub fn run_coded(
        &mut self,
        inputs: &[(usize, TokenId)],
        queries: &[CoderQuery],
        freq_bits: u32,
    ) -> Result<Vec<CoderInterval>> {
        if queries.len() != inputs.len() {
            return Err(Error::LengthMismatch { expected: inputs.len(), found: queries.len() });
        }
        let q = coder_words(queries, self.model.dims.vocab, freq_bits)?;
        let mut raw = vec![0u64; 3 * inputs.len()];
        self.run(inputs, Ask::new(Out::Intervals(freq_bits), &q), &mut raw, None)?;
        Ok(intervals_of(&raw, self.model.dims.vocab))
    }

    /// Runs the inputs and waits, with no read back. For measurements.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::run_weights`].
    pub fn run_without_readback(&mut self, inputs: &[(usize, TokenId)]) -> Result<()> {
        self.run(inputs, Ask::new(Out::Nothing, &[]), &mut [], None)
    }

    /// Runs the inputs as [`CudaBatch::run_coded`] does (with encode queries of token 0 and
    /// 32 bits), in a graph that is not kept, with a GPU timer mark after each kernel, and reads nothing
    /// back. Gives the GPU nanoseconds of each op name (summed over the layers and the passes),
    /// in pass order. For measurements.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::run_weights`].
    pub fn profile(&mut self, inputs: &[(usize, TokenId)]) -> Result<Vec<(&'static str, u64)>> {
        let mut prof = Vec::new();
        let queries = vec![(0, 0); inputs.len()];
        self.run(inputs, Ask::new(Out::Intervals(32), &queries), &mut [], Some(&mut prof))?;
        Ok(prof)
    }

    /// Validates, plans the passes, runs them, and commits the positions.
    fn run(
        &mut self,
        inputs: &[(usize, TokenId)],
        ask: Ask<'_, '_>,
        sink: &mut [u64],
        mut prof: Option<&mut Vec<(&'static str, u64)>>,
    ) -> Result<()> {
        let Ask { out, queries, mut mixing } = ask;
        let vocab = self.model.dims.vocab;
        for &(slot, tok) in inputs {
            if slot >= self.slots {
                return Err(Error::BadSlot { slot, slots: self.slots });
            }
            if tok.get() >= vocab {
                return Err(Error::Model(hardshell::Error::TokenOutOfRange {
                    token: tok.get(),
                    vocab_size: vocab,
                }));
            }
        }
        let plan = self.plan(inputs);
        for pass in &plan.passes {
            if let (Out::Mixed(_), Some(m)) = (out, mixing.as_deref()) {
                self.write_predictions(pass, m)?;
            }
            if let Out::Intervals(_) | Out::Mixed(_) = out {
                let mut words = Vec::with_capacity(4 * pass.len());
                for row in pass {
                    let (kind, v) =
                        row.input.and_then(|i| queries.get(i).copied()).unwrap_or((0, 0));
                    words.extend([kind, v, 0, 0]);
                }
                self.model.dev.ctx.write(&self.b.queries, 0, &u32_bytes(&words))?;
            }
            self.forward(pass, out, prof.as_deref_mut())?;
            if let (Out::Mixed(bits), Some(m)) = (out, mixing.as_deref_mut()) {
                self.mix_pass(pass, bits, m)?;
            }
            self.collect(pass, out, sink)?;
        }
        self.commit(plan);
        Ok(())
    }

    /// Copies the results of a pass into `sink`.
    fn collect(&self, pass: &[Row], out: Out, sink: &mut [u64]) -> Result<()> {
        let ctx = &self.model.dev.ctx;
        let (per_row, buf) = match out {
            Out::Nothing => return ctx.synchronize(),
            Out::Weights => (self.model.vocab_size(), &self.b.weights),
            Out::Intervals(_) | Out::Mixed(_) => (3, &self.b.intervals),
        };
        let mut bytes = vec![0u8; 8 * per_row * pass.len()];
        ctx.read(buf, &mut bytes)?;
        let vals = bytes_u64(&bytes);
        for (r, row) in pass.iter().enumerate() {
            if let Some(i) = row.input
                && let (Some(dst), Some(src)) = (
                    sink.get_mut(i * per_row..(i + 1) * per_row),
                    vals.get(r * per_row..(r + 1) * per_row),
                )
            {
                dst.copy_from_slice(src);
            }
        }
        Ok(())
    }

    /// One forward pass over `pass`: writes the rows, then replays the recorded graph of this
    /// shape (it records the graph the first time). With `prof`, it launches the kernels
    /// directly, with timer marks, and adds the op times to `prof`.
    fn forward(
        &self,
        pass: &[Row],
        out: Out,
        prof: Option<&mut Vec<(&'static str, u64)>>,
    ) -> Result<()> {
        let dev = self.model.dev;
        let n = u32_of(pass.len(), "too many rows")?;
        let mut info = Vec::with_capacity(4 * pass.len());
        for r in pass {
            info.extend([u32_of(r.slot, "slot")?, r.token.get(), u32_of(r.pos, "position")?, 0]);
        }
        dev.ctx.write(&self.b.rows, 0, &u32_bytes(&info))?;
        let groups = match self.att_groups(pass) {
            Some(words) => {
                dev.ctx.write(&self.b.groups, 0, &u32_bytes(&words))?;
                u32_of(words.len() / 2, "too many groups")?
            }
            None => 0,
        };
        let kernel = MatKernel::for_rows(pass.len());
        if let Some(prof) = prof {
            let mut st = Some(Stamps { buf: &self.b.stamps, labels: Vec::new() });
            // NOTE: the marks run in a graph, as a normal pass does, so the gaps between the
            // kernels are the gaps of a replayed graph.
            dev.ctx.begin_capture()?;
            let recorded = self.record_path(n, kernel, out, groups, &mut st);
            let graph = dev.ctx.end_capture();
            recorded?;
            let graph = graph?;
            dev.ctx.launch_graph(&graph)?;
            let labels = st.map(|s| s.labels).unwrap_or_default();
            let mut bytes = vec![0u8; 8 * labels.len()];
            dev.ctx.read(&self.b.stamps, &mut bytes)?;
            let t = bytes_u64(&bytes);
            for (i, &label) in labels.iter().enumerate().skip(1) {
                let ns = t
                    .get(i)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(t.get(i - 1).copied().unwrap_or(0));
                match prof.iter_mut().find(|(l, _)| *l == label) {
                    Some(e) => e.1 += ns,
                    None => prof.push((label, ns)),
                }
            }
            return Ok(());
        }
        let shape = match out {
            Out::Intervals(bits) => bits,
            Out::Mixed(_) => MIXED_GRAPH,
            Out::Nothing | Out::Weights => 0,
        };
        let key: GraphKey = (n, shape, groups);
        let mut graphs = self.graphs.borrow_mut();
        let graph = match graphs.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                dev.ctx.begin_capture()?;
                let recorded = self.record_path(n, kernel, out, groups, &mut None);
                // NOTE: end the capture also after an error, so the stream leaves capture mode.
                let graph = dev.ctx.end_capture();
                recorded?;
                e.insert(graph?)
            }
        };
        dev.ctx.launch_graph(graph)
    }
}

/// The query words of the coder op (kind, value) for `queries`.
///
/// # Errors
/// [`Error::Unsupported`] if `freq_bits` is not in `1..=32`, `2^freq_bits` is less than the
/// vocabulary, or a token or target is out of range.
pub(crate) fn coder_words(
    queries: &[CoderQuery],
    vocab: u32,
    freq_bits: u32,
) -> Result<Vec<(u32, u32)>> {
    if freq_bits == 0 || freq_bits > 32 || u64::from(vocab) > 1u64 << freq_bits {
        return Err(Error::Unsupported {
            reason: "freq_bits must be in 1..=32 and 2^freq_bits >= vocab",
        });
    }
    let total = 1u64 << freq_bits;
    queries
        .iter()
        .map(|query| match *query {
            CoderQuery::Encode(t) if t.get() < vocab => Ok((0, t.get())),
            CoderQuery::Decode(x) if x < total => Ok((1, u32::try_from(x).unwrap_or(u32::MAX))),
            _ => Err(Error::Unsupported { reason: "a coder token or target is out of range" }),
        })
        .collect()
}

/// The intervals of the raw output of the coder op: symbol, start and freq for each input.
pub(crate) fn intervals_of(raw: &[u64], vocab: u32) -> Vec<CoderInterval> {
    raw.as_chunks::<3>()
        .0
        .iter()
        .map(|w| {
            let sym = u32::try_from(w[0]).unwrap_or(u32::MAX);
            CoderInterval {
                symbol: TokenId::new(sym),
                start: w[1],
                freq: w[2],
                is_last: sym.checked_add(1) == Some(vocab),
            }
        })
        .collect()
}
