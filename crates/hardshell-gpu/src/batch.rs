//! Many independent GPT-2 sessions that step together on the GPU.
//!
//! A batch has `slots` sessions. Each slot has its own KV cache, position and history. One step
//! runs one token for each slot in the step, so one read of the weights serves all of them.
//! Each slot gives the same weights as a `hardshell::gpt2::Session` with the same tokens and
//! the same context overlap (ADR 0015), bit for bit.
//!
//! The host composes the op kernels into the forward pass of the `hardshell::gpt2` module
//! documentation. The residual steps 2.4 and 2.7 are a `linear` op with an i32 output and an
//! `add` op: `clamp(x + clamp(delta, i32), -R, R)` equals `clamp(x + delta, -R, R)`, because
//! `|x| <= R < 2^30`, so a delta outside the i32 range gives the limit in both forms.

mod context;
mod plan;
mod record;
mod rows;

use hardshell::TokenId;
use hardshell::gpt2::{Overlap, StreamState};

use crate::device::bytes_to_words;
use crate::error::{Error, Result};
use crate::model::GpuModel;
use crate::ops::{MatKernel, u32_of};
use context::{History, RESTART_ROWS, Row};
use plan::Plan;
use record::{MAX_STAMPS, Mode, Rec, sum_stamps};

/// A question to the coder op for one row (see [`GpuBatch::step_coded`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoderQuery {
    /// The interval of this token (the encoder).
    Encode(TokenId),
    /// The symbol whose interval holds this target, `target < 2^freq_bits` (the decoder).
    Decode(u64),
}

/// The interval of one symbol in the frequency table: the values that the range coder needs.
/// The range coder defines these four values, so the struct does not grow.
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

/// Many sessions of one model on the GPU.
pub struct GpuBatch<'m> {
    model: &'m GpuModel,
    slots: usize,
    /// The most rows of one forward pass: `slots`, or at least `RESTART_ROWS` with an
    /// overlap.
    max_rows: usize,
    overlap: Overlap,
    pos: Vec<usize>,
    /// The tokens of the context of each slot, for the overlap of a restart.
    hist: Vec<History>,
    /// The inputs of each slot, for a truncation (ADR 0017, `hardshell::gpt2::StreamState`).
    streams: Vec<StreamState>,
    /// The history of the full context of each slot before its last restart (empty before the
    /// first restart), for a truncation before the restart.
    prev: Vec<Vec<TokenId>>,
    /// `[slots, vocab]` weights of the last step of each slot.
    host: Vec<u64>,
    plan: Plan,
    /// The words of the coder intervals of the last `step_coded`.
    last_intervals: Vec<u32>,
    /// The words of the weights of the last pass of [`GpuBatch::run_rows`].
    last_rows: Vec<u32>,
}

impl std::fmt::Debug for GpuBatch<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBatch")
            .field("slots", &self.slots)
            .field("overlap", &self.overlap)
            .finish_non_exhaustive()
    }
}

impl<'m> GpuBatch<'m> {
    /// Makes a batch of `slots` sessions with empty contexts and no context overlap. Cost:
    /// allocates the KV cache (`2 * n_layer * n_ctx * d_model` bytes for each slot) and the
    /// work buffers, and makes the bind groups one time.
    ///
    /// # Errors
    /// [`Error::DeviceLimit`] if a buffer is larger than one binding of the device or the grid
    /// is too large; [`Error::Gpu`] on a GPU error.
    pub fn new(model: &'m GpuModel, slots: usize) -> Result<Self> {
        Self::with_overlap(model, slots, Overlap::NONE)
    }

    /// Makes a batch of `slots` sessions with empty contexts and the context overlap `overlap`
    /// (ADR 0015) for every slot: a slot whose context is full starts again with
    /// `<|endoftext|>` and its last `K` tokens, as `hardshell::gpt2::Session` with
    /// `SessionConfig::with_overlap` does. With an overlap, a pass has room for at least 32
    /// rows, so the `K + 1` rows of a restart run in few passes. For a batch of fewer than 32
    /// slots this costs the work buffers of the extra rows (for GPT-2 small about 1.1 MB for
    /// each row, est. from the buffer sizes).
    ///
    /// PERF: a restart pass stops after the last block: the final norm and the logits of a
    /// restart row are not computed, because their weights are not used.
    ///
    /// # Errors
    /// [`Error::Model`] with `InvalidOverlap` if `K > n_ctx - 2`; the errors of
    /// [`GpuBatch::new`].
    pub fn with_overlap(model: &'m GpuModel, slots: usize, overlap: Overlap) -> Result<Self> {
        let rows = if overlap.is_none() { slots.max(1) } else { slots.max(RESTART_ROWS) };
        Self::with_rows(model, slots, overlap, rows)
    }

    /// The same as [`GpuBatch::with_overlap`], with work buffers for passes of up to
    /// `max_rows` rows (at least the rows of [`GpuBatch::with_overlap`]): room for the draft
    /// rows of many slots in one pass of [`GpuBatch::run_rows`] (ADR 0017). Cost: about
    /// 1.1 MB of GPU memory for each row of GPT-2 small (est. from the buffer sizes).
    ///
    /// # Errors
    /// The errors of [`GpuBatch::with_overlap`].
    pub fn with_rows(
        model: &'m GpuModel,
        slots: usize,
        overlap: Overlap,
        max_rows: usize,
    ) -> Result<Self> {
        let n_ctx = model.n_ctx();
        if overlap.tokens().saturating_add(2) > n_ctx {
            return Err(Error::Model(hardshell::Error::InvalidOverlap {
                overlap: overlap.get(),
                n_ctx,
            }));
        }
        let least = if overlap.is_none() { slots.max(1) } else { slots.max(RESTART_ROWS) };
        let max_rows = max_rows.max(least);
        let plan = Plan::new(model, slots, max_rows)?;
        Ok(Self {
            model,
            slots,
            max_rows,
            overlap,
            pos: vec![0; slots],
            hist: (0..slots).map(|_| History::new(n_ctx)).collect(),
            streams: vec![StreamState::new(); slots],
            prev: vec![Vec::new(); slots],
            host: vec![0; slots * model.vocab_size()],
            plan,
            last_intervals: Vec::new(),
            last_rows: Vec::new(),
        })
    }

    /// The model of this batch.
    #[must_use]
    pub fn model(&self) -> &'m GpuModel {
        self.model
    }

    /// The number of slots.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The context overlap of the slots (ADR 0015).
    #[must_use]
    pub fn overlap(&self) -> Overlap {
        self.overlap
    }

    /// The bytes of GPU memory that this batch uses: KV caches, work buffers and the read-back
    /// buffer (measured from the allocations).
    #[must_use]
    pub fn gpu_bytes(&self) -> u64 {
        self.plan.bytes
    }

    /// The number of tokens in the context of `slot` (0 for a slot that is not valid).
    #[must_use]
    pub fn position(&self, slot: usize) -> usize {
        self.pos.get(slot).copied().unwrap_or(0)
    }

    /// Empties the context of `slot`. Cost: none on the GPU; old cache entries are not read
    /// again.
    pub fn reset(&mut self, slot: usize) {
        if let Some(p) = self.pos.get_mut(slot) {
            *p = 0;
        }
        if let Some(s) = self.streams.get_mut(slot) {
            s.clear();
        }
    }

    /// The weights of the last step that included `slot`: one for each token of the
    /// vocabulary. `None` if the slot is not valid.
    #[must_use]
    pub fn weights(&self, slot: usize) -> Option<&[u64]> {
        let v = self.model.vocab_size();
        self.host.get(slot * v..(slot + 1) * v)
    }

    /// Adds one token to each listed slot and runs the model for all of them in one pass. The
    /// weights of each slot then equal the result of `hardshell::gpt2::Session::step` with the
    /// same history and overlap. A slot whose context is full starts again first, with the last
    /// `K` tokens of its block, as on the CPU (plan §7.7, ADR 0015). Cost: one forward pass for
    /// the listed slots, one read back of `inputs.len() * vocab * 8` bytes; a restart adds
    /// `K + 1` rows, in passes of up to `max(slots, 32)` rows (`slots` rows if `K = 0`).
    ///
    /// # Errors
    /// [`Error::BadSlot`] if a slot is not valid or is listed two times;
    /// [`Error::Model`] with `TokenOutOfRange` if a token is not in the vocabulary;
    /// [`Error::Gpu`] on a GPU error. After an error, no slot changed.
    pub fn step(&mut self, inputs: &[(usize, TokenId)]) -> Result<()> {
        self.step_mode(inputs, Mode::Read).map(|_| ())
    }

    fn step_mode(
        &mut self,
        inputs: &[(usize, TokenId)],
        mode: Mode,
    ) -> Result<Option<Vec<(&'static str, u64)>>> {
        let vocab = self.model.dims.vocab;
        let mut seen = vec![false; self.slots];
        for &(slot, tok) in inputs {
            match seen.get_mut(slot) {
                Some(s) if !*s => *s = true,
                _ => return Err(Error::BadSlot { slot, slots: self.slots }),
            }
            if tok.get() >= vocab {
                return Err(Error::Model(hardshell::Error::TokenOutOfRange {
                    token: tok.get(),
                    vocab_size: vocab,
                }));
            }
        }
        if inputs.is_empty() {
            return Ok(None);
        }
        let n_ctx = self.model.n_ctx();
        let eot = self.model.eot;
        let k = self.overlap.tokens();
        let mut restart: Vec<Row> = Vec::new();
        let mut restarted = Vec::new();
        let mut rows: Vec<Row> = Vec::with_capacity(inputs.len());
        for &(s, t) in inputs {
            let mut pos = self.position(s);
            if pos >= n_ctx
                && let Some(h) = self.hist.get(s)
            {
                // PLAN(§7.7): the context of this slot starts again, with its last `k` tokens
                // (ADR 0015).
                h.restart_rows(s, k, eot, &mut restart);
                restarted.push(s);
                pos = k + 1;
            }
            rows.push((s, t, pos));
        }
        // NOTE: a pass boundary can fall between the rows of a restart; the earlier pass writes
        // the cache at the lower positions before the later pass reads them (queue order).
        for group in restart.chunks(self.max_rows) {
            self.forward(group, Mode::Restart)?;
        }
        let profile = self.forward(&rows, mode)?;
        // The histories and positions change only after all passes, so an error changes no
        // slot: a slot that restarted runs its restart again at the next step.
        for &s in &restarted {
            if let (Some(h), Some(prev), Some(st)) =
                (self.hist.get_mut(s), self.prev.get_mut(s), self.streams.get_mut(s))
            {
                prev.clear();
                prev.extend_from_slice(h.tokens());
                h.restart(k, eot);
                st.restart(k);
            }
        }
        for &(s, t, p) in &rows {
            if let (Some(h), Some(q), Some(st)) =
                (self.hist.get_mut(s), self.pos.get_mut(s), self.streams.get_mut(s))
            {
                h.set(p, t);
                st.input(p);
                *q = p + 1;
            }
        }
        Ok(profile)
    }

    /// Runs the step as [`GpuBatch::step`] does, then makes the coder interval of each input on
    /// the GPU and reads back only the intervals (16 bytes for each input, not the weights).
    /// `queries[i]` belongs to `inputs[i]`. The interval is the interval of
    /// `taco_coder::FreqTable::from_weights(weights, freq_bits)` (plan §5.1): for
    /// [`CoderQuery::Encode`], of the given token; for [`CoderQuery::Decode`], of the symbol
    /// whose interval holds the target, as the decoder finds it. [`GpuBatch::weights`] keeps
    /// the old values. Cost: one forward pass and one coder pass (three passes over the
    /// weights of each row on the GPU).
    ///
    /// # Errors
    /// The errors of [`GpuBatch::step`]; [`Error::LengthMismatch`] if the lengths differ;
    /// [`Error::Unsupported`] if `freq_bits` is not in `1..=32` or the vocabulary is larger
    /// than `2^freq_bits`, or a token or target is out of range.
    pub fn step_coded(
        &mut self,
        inputs: &[(usize, TokenId)],
        queries: &[CoderQuery],
        freq_bits: u32,
    ) -> Result<Vec<CoderInterval>> {
        if queries.len() != inputs.len() {
            return Err(Error::LengthMismatch { expected: inputs.len(), found: queries.len() });
        }
        let vocab = self.model.dims.vocab;
        if freq_bits == 0 || freq_bits > 32 || u64::from(vocab) > 1u64 << freq_bits {
            return Err(Error::Unsupported {
                reason: "freq_bits must be in 1..=32 and 2^freq_bits >= vocab",
            });
        }
        let total = 1u64 << freq_bits;
        let mut words = Vec::with_capacity(4 * queries.len());
        for q in queries {
            let (kind, v) = match *q {
                CoderQuery::Encode(t) if t.get() < vocab => (0, t.get()),
                CoderQuery::Decode(x) if x < total => (1, u32::try_from(x).unwrap_or(u32::MAX)),
                _ => {
                    return Err(Error::Unsupported {
                        reason: "a coder token or target is out of range",
                    });
                }
            };
            words.extend([kind, v, 0, 0]);
        }
        // INVARIANT: 1 <= vocab <= total <= 2^32, so T - V < 2^32.
        let spare = u32::try_from(total - u64::from(vocab)).unwrap_or(u32::MAX);
        let (t_lo, t_hi) = ((total & 0xffff_ffff) as u32, (total >> 32) as u32);
        self.model.dev.write_words(&self.plan.io.freq_params, &[vocab, t_lo, t_hi, spare]);
        self.model.dev.write_words(&self.plan.io.queries, &words);
        self.step_mode(inputs, Mode::Intervals)?;
        let iv = &self.last_intervals;
        Ok(iv
            .as_chunks::<4>()
            .0
            .iter()
            .take(inputs.len())
            .map(|w| CoderInterval {
                symbol: TokenId::new(w[0]),
                start: u64::from(w[1]),
                freq: u64::from(w[2]) | (u64::from(w[3]) << 32),
                is_last: w[0] + 1 == vocab,
            })
            .collect())
    }

    /// Runs the step as [`GpuBatch::step`] does, but does not copy the weights to the host:
    /// [`GpuBatch::weights`] keeps the old values. It waits until the GPU finishes. This is for
    /// measurements (the cost of the read back).
    ///
    /// # Errors
    /// The errors of [`GpuBatch::step`].
    pub fn step_without_readback(&mut self, inputs: &[(usize, TokenId)]) -> Result<()> {
        self.step_mode(inputs, Mode::Wait).map(|_| ())
    }

    /// Runs the step as [`GpuBatch::step`] does, and measures each op with GPU timestamps.
    /// Gives `(op name, nanoseconds)` in pass order, summed over the layers. `None` if the
    /// device has no timestamps.
    ///
    /// # Errors
    /// The errors of [`GpuBatch::step`].
    pub fn profile_step(
        &mut self,
        inputs: &[(usize, TokenId)],
    ) -> Result<Option<Vec<(&'static str, u64)>>> {
        if !self.model.dev.has_timestamps() {
            return Ok(None);
        }
        self.step_mode(inputs, Mode::Profile)
    }

    /// One forward pass for `rows` (slot, token, position).
    fn forward(&mut self, rows: &[Row], mode: Mode) -> Result<Option<Vec<(&'static str, u64)>>> {
        let model = self.model;
        let dev = &model.dev;
        let dm = model.dims;
        let n = u32_of(rows.len(), "rows")?;
        let mut info = vec![n, 0, 0, 0];
        for &(s, t, p) in rows {
            info.extend([u32_of(s, "slot")?, t.get(), u32_of(p, "position")?, 0]);
        }
        dev.write_words(&self.plan.io.rows, &info);
        let kernel = MatKernel::for_rows(rows.len());
        let k = &model.k;
        let p = &self.plan;
        let queries = (mode == Mode::Profile).then(|| {
            dev.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profile"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_STAMPS,
            })
        });
        let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let labels;
        {
            let pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let mut r = Rec {
                pass,
                qs: queries.as_ref(),
                labels: Vec::new(),
                mm: (&k.matvec, &k.gemm),
                kernel,
                n,
            };
            r.stamp("start");
            r.run("embed", &k.embed, &p.embed, dm.d.div_ceil(64), n);
            for (l, lw) in p.layers.iter().zip(&model.layers) {
                r.run("norm", &k.norm, &l.ln_1, n, 1);
                r.matmul("matmul qkv", &l.qkv_mm, lw.c_attn.rows);
                r.run("linear out", &k.linear_i8, &l.qkv_out, (3 * dm.d).div_ceil(256), n);
                r.run("attention kv", &k.kv_store, &l.kv, (dm.d / 4).div_ceil(64), n);
                r.run("attention", &k.attend, &l.attn, dm.n_head, n);
                r.matmul("matmul proj", &l.proj_mm, lw.c_proj.rows);
                r.run("linear out", &k.linear_i32, &l.proj_out, dm.d.div_ceil(64), n);
                r.run("add", &k.add, &l.add, dm.d.div_ceil(64), n);
                r.run("norm", &k.norm, &l.ln_2, n, 1);
                r.matmul("matmul fc", &l.fc_mm, lw.c_fc.rows);
                r.run("linear out", &k.linear_i32, &l.fc_out, dm.d_ff.div_ceil(64), n);
                r.run("table", &k.table, &l.gelu, dm.d_ff.div_ceil(256), n);
                r.matmul("matmul mlp", &l.mlp_mm, lw.mlp_proj.rows);
                r.run("linear out", &k.linear_i32, &l.mlp_out, dm.d.div_ceil(64), n);
                r.run("add", &k.add, &l.add, dm.d.div_ceil(64), n);
            }
            if mode != Mode::Restart {
                r.run("norm", &k.norm, &p.ln_f, n, 1);
                r.matmul("matmul logits", &p.logits_mm, dm.vocab);
                r.run("logits", &k.logits_scale, &p.logits_scale, dm.vocab.div_ceil(64), n);
                r.run("logits", &k.logits_max, &p.logits_max, n, 1);
                r.run("logits", &k.logits_weights, &p.logits_weights, dm.vocab.div_ceil(64), n);
            }
            if mode == Mode::Intervals {
                r.run("coder interval", &k.coder_interval, &p.coder, n, 1);
            }
            labels = r.labels;
        }
        match mode {
            Mode::Restart => {
                dev.queue.submit([enc.finish()]);
                dev.check()?;
                Ok(None)
            }
            Mode::Wait => {
                dev.finish_and_wait(enc)?;
                Ok(None)
            }
            Mode::Rows => {
                let bytes = (rows.len() * model.vocab_size() * 8) as u64;
                let mut out = Vec::new();
                dev.finish_and_read(
                    enc,
                    &self.plan.io.weights,
                    &self.plan.io.staging,
                    bytes,
                    &mut out,
                )?;
                self.last_rows = bytes_to_words(&out);
                Ok(None)
            }
            Mode::Read => {
                let bytes = (rows.len() * model.vocab_size() * 8) as u64;
                let mut out = Vec::new();
                dev.finish_and_read(
                    enc,
                    &self.plan.io.weights,
                    &self.plan.io.staging,
                    bytes,
                    &mut out,
                )?;
                self.copy_weights(rows, &out);
                Ok(None)
            }
            Mode::Intervals => {
                let bytes = 16 * u64::from(n);
                let mut out = Vec::new();
                dev.finish_and_read(
                    enc,
                    &self.plan.io.intervals,
                    &self.plan.io.interval_staging,
                    bytes,
                    &mut out,
                )?;
                self.last_intervals = bytes_to_words(&out);
                Ok(None)
            }
            Mode::Profile => {
                let Some(qs) = queries else { return Ok(None) };
                let count = u32_of(labels.len(), "timestamps")?;
                let ts_bytes = 8 * u64::from(count);
                let resolve = dev.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamps"),
                    size: ts_bytes.div_ceil(256) * 256,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                enc.resolve_query_set(&qs, 0..count, &resolve, 0);
                let ts_staging = dev.staging("timestamps read", ts_bytes);
                let bytes = (rows.len() * model.vocab_size() * 8) as u64;
                let mut outs = [Vec::new(), Vec::new()];
                dev.finish_and_read_all(
                    enc,
                    &[
                        (&self.plan.io.weights, &self.plan.io.staging, bytes),
                        (&resolve, &ts_staging, ts_bytes),
                    ],
                    &mut outs,
                )?;
                self.copy_weights(rows, &outs[0]);
                let ps = u128::from(dev.timestamp_period_ps());
                let ns = sum_stamps(&labels, &outs[1])
                    .into_iter()
                    .map(|(l, t)| (l, u64::try_from(u128::from(t) * ps / 1000).unwrap_or(u64::MAX)))
                    .collect();
                Ok(Some(ns))
            }
        }
    }

    fn copy_weights(&mut self, rows: &[Row], out: &[u8]) {
        let vocab = self.model.vocab_size();
        let words = bytes_to_words(out);
        for (r, &(slot, _, _)) in rows.iter().enumerate() {
            let src = words.get(r * vocab * 2..(r + 1) * vocab * 2).unwrap_or(&[]);
            if let Some(dst) = self.host.get_mut(slot * vocab..(slot + 1) * vocab) {
                for (d, pair) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                    *d = u64::from(pair[0]) | (u64::from(pair[1]) << 32);
                }
            }
        }
    }
}
