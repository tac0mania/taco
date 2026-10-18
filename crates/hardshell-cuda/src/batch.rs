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

use crate::device::{bytes_u64, u32_bytes, u32_of};
use crate::driver::{Arg, DevBuf, Graph};
use crate::error::{Error, Result};
use crate::kernels::padded_rows;
use crate::model::{CudaModel, LinearW, NormW};
use crate::ops::{Mat, MatKernel, launch_matmul, splits_for};

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
    stamps: DevBuf,
}

/// Many sessions of one model on one CUDA device.
pub struct CudaBatch<'m> {
    model: &'m CudaModel<'m>,
    slots: usize,
    max_rows: usize,
    pos: Vec<usize>,
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

/// The block size of the `attend` kernel (`ATT_T` in kernels/hardshell.cu).
const ATTEND_THREADS: u32 = 128;

/// The most timer marks of one profiled pass.
const MAX_STAMPS: usize = 1024;

/// The key of a recorded graph: the number of rows and the output (0: no coder op; otherwise
/// `freq_bits` of the coder op).
type GraphKey = (u32, u32);

impl<'m> CudaBatch<'m> {
    /// Makes a batch of `slots` sessions with empty contexts, and work buffers for passes of
    /// up to `max_rows` rows. Cost: the KV cache (`2 * n_layer * n_ctx * d_model` bytes for each
    /// slot) and about `20 * vocab` bytes for each row.
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
        let accn = (3 * d.d).max(d.d_ff).max(d.vocab);
        let max_splits = (accn / (3 * d.d).max(d.d_ff).max(1)).clamp(1, 16);
        let tmpn = d.d.max(d.d_ff);
        let us = |x: u32| x as usize;
        let sizes = [
            16 * r,
            4 * r * us(d.d),
            4 * rp * us(d.dw),
            r * 3 * us(d.d),
            4 * rp * us(d.dw),
            4 * rp * us(d.fw),
            4 * r * us(accn),
            4 * r * us(tmpn),
            8 * r * us(d.n_head) * us(d.n_ctx),
            8 * r * us(d.vocab),
            8 * r * us(d.vocab),
            16 * r,
            24 * r,
            8 * MAX_STAMPS,
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
        };
        let cache = slots * us(d.n_ctx) * us(d.d);
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
            host: vec![0; slots * model.vocab_size()],
            b,
            kv,
            tmpn,
            max_splits,
            graphs: RefCell::new(HashMap::new()),
            bytes,
        })
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
    /// (plan §7.7). Cost: one pass for each `max_rows` rows (and one more at a restart).
    ///
    /// # Errors
    /// [`Error::BadSlot`] if a slot is not valid; [`Error::Model`] with `TokenOutOfRange` if a
    /// token is not in the vocabulary; [`Error::Cuda`] on a driver error. After an error, no
    /// position changed.
    pub fn run_weights(&mut self, inputs: &[(usize, TokenId)]) -> Result<Vec<u64>> {
        let mut out = vec![0u64; inputs.len() * self.model.vocab_size()];
        self.run(inputs, Out::Weights, &mut out, &[], None)?;
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
        let vocab = self.model.dims.vocab;
        if freq_bits == 0 || freq_bits > 32 || u64::from(vocab) > 1u64 << freq_bits {
            return Err(Error::Unsupported {
                reason: "freq_bits must be in 1..=32 and 2^freq_bits >= vocab",
            });
        }
        let total = 1u64 << freq_bits;
        let mut q = Vec::with_capacity(queries.len());
        for query in queries {
            q.push(match *query {
                CoderQuery::Encode(t) if t.get() < vocab => (0, t.get()),
                CoderQuery::Decode(x) if x < total => (1, u32::try_from(x).unwrap_or(u32::MAX)),
                _ => {
                    return Err(Error::Unsupported {
                        reason: "a coder token or target is out of range",
                    });
                }
            });
        }
        let mut raw = vec![0u64; 3 * inputs.len()];
        self.run(inputs, Out::Intervals(freq_bits), &mut raw, &q, None)?;
        Ok(raw
            .as_chunks::<3>()
            .0
            .iter()
            .map(|w| {
                let sym = u32::try_from(w[0]).unwrap_or(u32::MAX);
                CoderInterval {
                    symbol: TokenId::new(sym),
                    start: w[1],
                    freq: w[2],
                    is_last: sym + 1 == vocab,
                }
            })
            .collect())
    }

    /// Runs the inputs and waits, with no read back. For measurements.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::run_weights`].
    pub fn run_without_readback(&mut self, inputs: &[(usize, TokenId)]) -> Result<()> {
        self.run(inputs, Out::Nothing, &mut [], &[], None)
    }

    /// Runs the inputs as [`CudaBatch::run_coded`] does (with encode queries of token 0 and
    /// 32 bits), without graphs, with a GPU timer mark after each kernel, and reads nothing
    /// back. Gives the GPU nanoseconds of each op name (summed over the layers and the passes),
    /// in pass order. For measurements.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::run_weights`].
    pub fn profile(&mut self, inputs: &[(usize, TokenId)]) -> Result<Vec<(&'static str, u64)>> {
        let mut prof = Vec::new();
        let queries = vec![(0, 0); inputs.len()];
        self.run(inputs, Out::Intervals(32), &mut [], &queries, Some(&mut prof))?;
        Ok(prof)
    }

    /// Validates, plans the passes, runs them, and commits the positions.
    fn run(
        &mut self,
        inputs: &[(usize, TokenId)],
        out: Out,
        sink: &mut [u64],
        queries: &[(u32, u32)],
        mut prof: Option<&mut Vec<(&'static str, u64)>>,
    ) -> Result<()> {
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
        let (passes, pos) = self.plan(inputs);
        for pass in &passes {
            if let Out::Intervals(_) = out {
                let mut words = Vec::with_capacity(4 * pass.len());
                for row in pass {
                    let (kind, v) =
                        row.input.and_then(|i| queries.get(i).copied()).unwrap_or((0, 0));
                    words.extend([kind, v, 0, 0]);
                }
                self.model.dev.ctx.write(&self.b.queries, 0, &u32_bytes(&words))?;
            }
            self.forward(pass, out, prof.as_deref_mut())?;
            self.collect(pass, out, sink)?;
        }
        self.pos = pos;
        Ok(())
    }

    /// Splits the inputs into passes of at most `max_rows` rows. A restart of a slot starts a
    /// new pass if the slot already has rows in the current pass, because the restart writes
    /// cache positions that those rows read.
    fn plan(&self, inputs: &[(usize, TokenId)]) -> (Vec<Vec<Row>>, Vec<usize>) {
        let n_ctx = self.model.n_ctx();
        let eot = self.model.eot_token();
        let mut pos = self.pos.clone();
        let mut passes = Vec::new();
        let mut cur: Vec<Row> = Vec::new();
        let mut in_pass = vec![false; self.slots];
        for (i, &(slot, token)) in inputs.iter().enumerate() {
            let full = pos.get(slot).is_some_and(|&p| p >= n_ctx);
            let restart_here = full && in_pass.get(slot).copied().unwrap_or(false);
            if restart_here || cur.len() + usize::from(full) >= self.max_rows {
                passes.push(std::mem::take(&mut cur));
                in_pass.iter_mut().for_each(|x| *x = false);
            }
            if full {
                // PLAN(§7.7): the context starts again with <|endoftext|> at position 0.
                cur.push(Row { slot, token: eot, pos: 0, input: None });
                if let Some(p) = pos.get_mut(slot) {
                    *p = 1;
                }
            }
            let p = pos.get(slot).copied().unwrap_or(0);
            cur.push(Row { slot, token, pos: p, input: Some(i) });
            if let Some(p) = pos.get_mut(slot) {
                *p += 1;
            }
            if let Some(x) = in_pass.get_mut(slot) {
                *x = true;
            }
        }
        if !cur.is_empty() {
            passes.push(cur);
        }
        (passes, pos)
    }

    /// Copies the results of a pass into `sink`.
    fn collect(&self, pass: &[Row], out: Out, sink: &mut [u64]) -> Result<()> {
        let ctx = &self.model.dev.ctx;
        let (per_row, buf) = match out {
            Out::Nothing => return ctx.synchronize(),
            Out::Weights => (self.model.vocab_size(), &self.b.weights),
            Out::Intervals(_) => (3, &self.b.intervals),
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
        let kernel = MatKernel::for_rows(pass.len());
        if let Some(prof) = prof {
            let mut st = Some(Stamps { buf: &self.b.stamps, labels: Vec::new() });
            self.record(n, kernel, out, &mut st)?;
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
        let key: GraphKey = (n, if let Out::Intervals(bits) = out { bits } else { 0 });
        let mut graphs = self.graphs.borrow_mut();
        let graph = match graphs.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                dev.ctx.begin_capture()?;
                let recorded = self.record(n, kernel, out, &mut None);
                // NOTE: end the capture also after an error, so the stream leaves capture mode.
                let graph = dev.ctx.end_capture();
                recorded?;
                e.insert(graph?)
            }
        };
        dev.ctx.launch_graph(graph)
    }

    /// Launches all kernels of a pass of `n` rows (the rows are already on the GPU).
    fn record(
        &self,
        n: u32,
        kernel: MatKernel,
        out: Out,
        stamps: &mut Option<Stamps<'_>>,
    ) -> Result<()> {
        let model = self.model;
        let dev = model.dev;
        let d = model.dims;
        let mut op = |name: &'static str| -> Result<()> {
            if let Some(st) = stamps.as_mut()
                && st.labels.len() < MAX_STAMPS
            {
                let i = u32_of(st.labels.len(), "too many marks")?;
                dev.launch(&dev.k.stamp, (1, 1), 1, &[Arg::Ptr(st.buf.ptr()), Arg::U32(i)])?;
                st.labels.push(name);
            }
            Ok(())
        };
        op("start")?;
        let b = &self.b;
        let k = &dev.k;
        let tmpn = self.tmpn;
        let ptr = |x: &DevBuf| Arg::Ptr(x.ptr());
        dev.launch(
            &k.embed,
            (d.d.div_ceil(128), n),
            128,
            &[
                ptr(&model.wte.w),
                ptr(&model.embed_m),
                ptr(&model.wpe),
                ptr(&model.pos_m),
                ptr(&b.rows),
                ptr(&b.x),
                Arg::U32(d.d),
                Arg::U32(4 * d.dw),
                Arg::U32(model.embed_s),
                Arg::U32(model.pos_s),
            ],
        )?;
        op("embed")?;
        for (l, (kc, vc)) in model.layers.iter().zip(&self.kv) {
            self.norm(&l.ln_1, n)?;
            op("norm")?;
            let z = self.matmul(kernel, &l.c_attn, &b.a, n, true)?;
            op("matmul qkv")?;
            self.linear_out(&l.c_attn, LinOut::I8(&b.qkv, 3 * d.d), (-127, 127), n, z)?;
            op("linear out")?;
            dev.launch(
                &k.kv_store,
                (d.d.div_ceil(128), n),
                128,
                &[
                    ptr(&b.qkv),
                    ptr(kc),
                    ptr(vc),
                    ptr(&b.rows),
                    Arg::U32(d.d),
                    Arg::U32(3 * d.d),
                    Arg::U32(d.n_ctx),
                    Arg::U32(d.head_dim),
                ],
            )?;
            op("attention kv")?;
            dev.launch(
                &k.attend,
                (d.n_head, n),
                ATTEND_THREADS,
                &[
                    ptr(&b.qkv),
                    ptr(kc),
                    ptr(vc),
                    ptr(&b.sc),
                    ptr(&model.attn_exp),
                    ptr(&b.rows),
                    ptr(&b.att),
                    Arg::U32(d.d),
                    Arg::U32(3 * d.d),
                    Arg::U32(d.head_dim),
                    Arg::U32(d.n_ctx),
                    Arg::U32(d.n_head),
                    Arg::I32(l.score.0),
                    Arg::U32(l.score.1),
                    Arg::I32(l.attn_out.0),
                    Arg::U32(l.attn_out.1),
                    Arg::U32(model.softmax_p),
                    Arg::U32(model.attn_len),
                    Arg::U32(4 * d.dw),
                ],
            )?;
            op("attention")?;
            let z = self.matmul(kernel, &l.c_proj, &b.att, n, true)?;
            op("matmul proj")?;
            self.linear_out(&l.c_proj, LinOut::I32(&b.tmp, tmpn), (i32::MIN, i32::MAX), n, z)?;
            op("linear out")?;
            self.add(n)?;
            op("add")?;
            self.norm(&l.ln_2, n)?;
            op("norm")?;
            let z = self.matmul(kernel, &l.c_fc, &b.a, n, true)?;
            op("matmul fc")?;
            self.linear_out(
                &l.c_fc,
                LinOut::I32(&b.tmp, tmpn),
                (-l.gelu_offset, l.gelu_offset - 1),
                n,
                z,
            )?;
            op("linear out")?;
            dev.launch(
                &k.table_lookup,
                (d.d_ff.div_ceil(128), n),
                128,
                &[
                    ptr(&b.tmp),
                    ptr(&l.gelu),
                    ptr(&b.g),
                    Arg::U32(d.d_ff),
                    Arg::U32(tmpn),
                    Arg::U32(4 * d.fw),
                    Arg::I32(l.gelu_offset),
                    Arg::I32(l.gelu_len),
                ],
            )?;
            op("table")?;
            let z = self.matmul(kernel, &l.mlp_proj, &b.g, n, true)?;
            op("matmul mlp")?;
            self.linear_out(&l.mlp_proj, LinOut::I32(&b.tmp, tmpn), (i32::MIN, i32::MAX), n, z)?;
            op("linear out")?;
            self.add(n)?;
            op("add")?;
        }
        self.norm(&model.ln_f, n)?;
        op("norm")?;
        self.matmul(kernel, &model.wte, &b.a, n, false)?;
        op("matmul logits")?;
        dev.launch(
            &k.logits_scale,
            (d.vocab.div_ceil(128), n),
            128,
            &[
                ptr(&b.acc),
                ptr(&model.wte.m),
                ptr(&b.logit),
                Arg::U32(d.vocab),
                Arg::U32(d.vocab),
                Arg::U32(model.wte.shift),
            ],
        )?;
        op("logits")?;
        dev.launch(
            &k.logits_weights,
            (n, 1),
            256,
            &[
                ptr(&b.logit),
                ptr(&model.logits_exp),
                ptr(&b.weights),
                Arg::U32(d.vocab),
                Arg::U32(model.logits_len),
            ],
        )?;
        op("logits")?;
        if let Out::Intervals(bits) = out {
            dev.launch(
                &k.coder_interval,
                (n, 1),
                256,
                &[
                    ptr(&b.weights),
                    ptr(&b.queries),
                    ptr(&b.intervals),
                    Arg::U32(d.vocab),
                    Arg::U64(1u64 << bits),
                ],
            )?;
            op("coder interval")?;
        }
        Ok(())
    }

    fn norm(&self, nw: &NormW, n: u32) -> Result<()> {
        let m = self.model;
        let (k, j, eps) = m.ln;
        m.dev.launch(
            &m.dev.k.layer_norm,
            (n, 1),
            256,
            &[
                Arg::Ptr(self.b.x.ptr()),
                Arg::Ptr(nw.g.ptr()),
                Arg::Ptr(nw.b.ptr()),
                Arg::Ptr(self.b.a.ptr()),
                Arg::U32(m.dims.d),
                Arg::U32(4 * m.dims.dw),
                Arg::U32(k),
                Arg::U32(j),
                Arg::U64(eps),
                Arg::I32(nw.m),
                Arg::U32(nw.s),
            ],
        )
    }

    /// The product of a linear layer into `acc`: rows of `lin.rows` values, and with `split`,
    /// split-K partials `n * lin.rows` values apart. Gives the number of partials.
    ///
    /// INVARIANT: at most 16 partials of `n * lin.rows <= max_rows * max(3d, d_ff)` values fit
    /// in `acc` (`max_rows * max(3d, d_ff, vocab)` values); `new` checks it.
    fn matmul(
        &self,
        kernel: MatKernel,
        lin: &LinearW,
        a: &DevBuf,
        n: u32,
        split: bool,
    ) -> Result<u32> {
        let splits = if split && kernel == MatKernel::TensorCore {
            splits_for(lin.rows, lin.kw, n).min(self.max_splits)
        } else {
            1
        };
        let mat = Mat {
            w: &lin.w,
            a,
            out: &self.b.acc,
            out_stride: lin.rows,
            n: lin.rows,
            kw: lin.kw,
            m: n,
            splits,
        };
        launch_matmul(self.model.dev, kernel, &mat)?;
        Ok(splits)
    }

    fn linear_out(
        &self,
        lin: &LinearW,
        out: LinOut<'_>,
        range: (i32, i32),
        n: u32,
        splits: u32,
    ) -> Result<()> {
        let m = self.model;
        let (o8, o32, stride) = match out {
            LinOut::I8(b, s) => (b.ptr(), 0, s),
            LinOut::I32(b, s) => (0, b.ptr(), s),
        };
        m.dev.launch(
            &m.dev.k.linear_out,
            (lin.rows.div_ceil(128), n),
            128,
            &[
                Arg::Ptr(self.b.acc.ptr()),
                Arg::Ptr(lin.b.ptr()),
                Arg::Ptr(lin.m.ptr()),
                Arg::Ptr(o8),
                Arg::Ptr(o32),
                Arg::U32(lin.rows),
                Arg::U32(lin.rows),
                Arg::U32(stride),
                Arg::U32(lin.shift),
                Arg::I32(range.0),
                Arg::I32(range.1),
                Arg::U32(splits),
                Arg::U32(n * lin.rows),
            ],
        )
    }

    fn add(&self, n: u32) -> Result<()> {
        let m = self.model;
        m.dev.launch(
            &m.dev.k.add_clamp,
            (m.dims.d.div_ceil(128), n),
            128,
            &[
                Arg::Ptr(self.b.x.ptr()),
                Arg::Ptr(self.b.tmp.ptr()),
                Arg::U32(m.dims.d),
                Arg::U32(self.tmpn),
            ],
        )
    }
}
