//! Many independent GPT-2 sessions that step together on the GPU.
//!
//! A batch has `slots` sessions. Each slot has its own KV cache and position. One step runs one
//! token for each slot in the step, so one read of the weights serves all of them. Each slot
//! gives the same weights as a `hardshell::gpt2::Session` with the same tokens, bit for bit.
//!
//! The host composes the op kernels into the forward pass of the `hardshell::gpt2` module
//! documentation. The residual steps 2.4 and 2.7 are a `linear` op with an i32 output and an
//! `add` op: `clamp(x + clamp(delta, i32), -R, R)` equals `clamp(x + delta, -R, R)`, because
//! `|x| <= R < 2^30`, so a delta outside the i32 range gives the limit in both forms.

use hardshell::TokenId;

use crate::device::{GpuDevice, bytes_to_words};
use crate::error::{Error, Result};
use crate::kernels::stride_words;
use crate::model::{GpuModel, LinearW, NormW};
use crate::ops::{MatKernel, MmGroup, check_grid, record_matmul, u32_of};

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

/// The bind groups of one block.
struct LayerPlan {
    ln_1: wgpu::BindGroup,
    qkv_mm: MmGroup,
    qkv_out: wgpu::BindGroup,
    kv: wgpu::BindGroup,
    attn: wgpu::BindGroup,
    proj_mm: MmGroup,
    proj_out: wgpu::BindGroup,
    add: wgpu::BindGroup,
    ln_2: wgpu::BindGroup,
    fc_mm: MmGroup,
    fc_out: wgpu::BindGroup,
    gelu: wgpu::BindGroup,
    mlp_mm: MmGroup,
    mlp_out: wgpu::BindGroup,
}

/// The buffers of a batch that the host reads or writes.
struct Io {
    rows: wgpu::Buffer,
    weights: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// One coder query for each row: kind (0 encode, 1 decode), token or target.
    queries: wgpu::Buffer,
    /// One coder interval for each row: symbol, start, freq (two words).
    intervals: wgpu::Buffer,
    interval_staging: wgpu::Buffer,
    /// The `FreqParams` of the coder op.
    freq_params: wgpu::Buffer,
}

/// Many sessions of one model on the GPU.
pub struct GpuBatch<'m> {
    model: &'m GpuModel,
    slots: usize,
    pos: Vec<usize>,
    /// `[slots, vocab]` weights of the last step of each slot.
    host: Vec<u64>,
    io: Io,
    embed: wgpu::BindGroup,
    layers: Vec<LayerPlan>,
    ln_f: wgpu::BindGroup,
    logits_mm: MmGroup,
    logits_scale: wgpu::BindGroup,
    logits_max: wgpu::BindGroup,
    logits_weights: wgpu::BindGroup,
    coder: wgpu::BindGroup,
    gpu_bytes: u64,
    /// The words of the coder intervals of the last `step_coded`.
    last_intervals: Vec<u32>,
    /// Keeps the buffers that only bind groups use.
    _keep: Vec<wgpu::Buffer>,
}

impl std::fmt::Debug for GpuBatch<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBatch").field("slots", &self.slots).finish_non_exhaustive()
    }
}

/// Widths and strides of the work buffers.
#[derive(Clone, Copy)]
struct Shape {
    d: u32,
    dw: u32,
    fw: u32,
    qkvw: u32,
    accn: u32,
    tmpn: u32,
}

/// Makes buffers and bind groups; counts the bytes.
struct Builder<'a> {
    dev: &'a GpuDevice,
    keep: Vec<wgpu::Buffer>,
    bytes: u64,
}

impl Builder<'_> {
    fn zeroed(&mut self, label: &'static str, bytes: u64) -> Result<wgpu::Buffer> {
        self.bytes += bytes;
        self.dev.zeroed(label, bytes)
    }

    fn uniform(&mut self, words: &[u32]) -> wgpu::Buffer {
        let b = self.dev.uniform("params", words);
        self.keep.push(b.clone());
        b
    }
}

/// The work buffers of the forward pass.
struct Work {
    x: wgpu::Buffer,
    a: wgpu::Buffer,
    qkv: wgpu::Buffer,
    att: wgpu::Buffer,
    g: wgpu::Buffer,
    acc: wgpu::Buffer,
    tmp: wgpu::Buffer,
    sc: wgpu::Buffer,
    logit: wgpu::Buffer,
    row_max: wgpu::Buffer,
}

impl<'m> GpuBatch<'m> {
    /// Makes a batch of `slots` sessions with empty contexts. Cost: allocates the KV cache
    /// (`2 * n_layer * n_ctx * d_model` bytes for each slot) and the work buffers, and makes
    /// the bind groups one time.
    ///
    /// # Errors
    /// [`Error::DeviceLimit`] if a buffer is larger than one binding of the device or the grid
    /// is too large; [`Error::Gpu`] on a GPU error.
    pub fn new(model: &'m GpuModel, slots: usize) -> Result<Self> {
        let dev = &model.dev;
        let dm = model.dims;
        let b = u64::from(u32_of(slots.max(1), "slots")?);
        check_grid(dev, b)?;
        let s = Shape {
            d: dm.d,
            dw: dm.dw,
            fw: dm.fw,
            qkvw: u32_of(stride_words(3 * dm.d as usize), "width")?,
            accn: (3 * dm.d).max(dm.d_ff).max(dm.vocab),
            tmpn: dm.d.max(dm.d_ff),
        };
        let mut bl = Builder { dev, keep: Vec::new(), bytes: 0 };
        let w = Work {
            x: bl.zeroed("x", 4 * b * u64::from(s.d))?,
            a: bl.zeroed("a", 4 * b * u64::from(s.dw))?,
            qkv: bl.zeroed("qkv", 4 * b * u64::from(s.qkvw))?,
            att: bl.zeroed("att", 4 * b * u64::from(s.dw))?,
            g: bl.zeroed("g", 4 * b * u64::from(s.fw))?,
            acc: bl.zeroed("acc", 4 * b * u64::from(s.accn))?,
            tmp: bl.zeroed("tmp", 4 * b * u64::from(s.tmpn))?,
            sc: bl.zeroed("scores", 8 * b * u64::from(dm.n_head) * u64::from(dm.n_ctx))?,
            logit: bl.zeroed("logit", 8 * b * u64::from(dm.vocab))?,
            row_max: bl.zeroed("row max", 8 * b)?,
        };
        let weights_bytes = 8 * b * u64::from(dm.vocab);
        let io = Io {
            rows: bl.zeroed("rows", 16 + 16 * b)?,
            weights: bl.zeroed("weights", weights_bytes)?,
            staging: dev.staging("weights read", weights_bytes),
            queries: bl.zeroed("coder queries", 16 * b)?,
            intervals: bl.zeroed("coder intervals", 16 * b)?,
            interval_staging: dev.staging("coder intervals read", 16 * b),
            freq_params: dev.uniform("freq params", &[0, 0, 0, 0]),
        };
        let k = &model.k;
        let embed = dev.bind(
            &k.embed,
            &[
                (0, &model.wte),
                (1, &model.embed_m),
                (2, &model.wpe),
                (3, &model.pos_m),
                (4, &bl.uniform(&[s.d, s.dw, model.embed_s, model.pos_s, s.d, 0, 0, 0])),
                (5, &io.rows),
                (6, &w.x),
            ],
        );
        let mut layers = Vec::with_capacity(model.layers.len());
        for l in &model.layers {
            let cache = b * u64::from(dm.n_ctx) * u64::from(s.d);
            let kc = bl.zeroed("k cache", cache)?;
            let vc = bl.zeroed("v cache", cache)?;
            let attn_params = bl.uniform(&[
                s.d / 4,
                s.qkvw,
                dm.head_dim / 4,
                dm.n_ctx,
                l.score.0.cast_unsigned(),
                l.score.1,
                l.attn_out.0.cast_unsigned(),
                l.attn_out.1,
                model.softmax_p,
                model.attn_len,
                s.dw,
                dm.n_head,
            ]);
            let attn = dev.bind(
                &k.attend,
                &[
                    (0, &w.qkv),
                    (1, &kc),
                    (2, &vc),
                    (3, &w.sc),
                    (4, &model.attn_exp),
                    (5, &attn_params),
                    (6, &io.rows),
                    (7, &w.att),
                ],
            );
            let kv = dev.bind(
                &k.kv_store,
                &[(0, &w.qkv), (1, &kc), (2, &vc), (5, &attn_params), (6, &io.rows)],
            );
            let off = i32::try_from(l.gelu_offset)
                .map_err(|_| Error::Unsupported { reason: "a GELU table is too long" })?;
            let (gelu_lo, gelu_hi) = (-off, off - 1);
            layers.push(LayerPlan {
                ln_1: norm_group(&mut bl, model, &l.ln_1, &w.x, &w.a, &io.rows),
                qkv_mm: mm_group(&mut bl, model, &l.c_attn, &w.a, (&w.acc, s.accn), &io.rows),
                qkv_out: lin_group(
                    &mut bl,
                    model,
                    &l.c_attn,
                    &LinIo {
                        acc: (&w.acc, s.accn),
                        out: (&w.qkv, s.qkvw),
                        range: (-127, 127),
                        packed: true,
                    },
                    &io.rows,
                ),
                kv,
                attn,
                proj_mm: mm_group(&mut bl, model, &l.c_proj, &w.att, (&w.acc, s.accn), &io.rows),
                proj_out: lin_group(
                    &mut bl,
                    model,
                    &l.c_proj,
                    &LinIo {
                        acc: (&w.acc, s.accn),
                        out: (&w.tmp, s.tmpn),
                        range: (i32::MIN, i32::MAX),
                        packed: false,
                    },
                    &io.rows,
                ),
                add: add_group(&mut bl, model, s, &w, &io.rows),
                ln_2: norm_group(&mut bl, model, &l.ln_2, &w.x, &w.a, &io.rows),
                fc_mm: mm_group(&mut bl, model, &l.c_fc, &w.a, (&w.acc, s.accn), &io.rows),
                fc_out: lin_group(
                    &mut bl,
                    model,
                    &l.c_fc,
                    &LinIo {
                        acc: (&w.acc, s.accn),
                        out: (&w.tmp, s.tmpn),
                        range: (gelu_lo, gelu_hi),
                        packed: false,
                    },
                    &io.rows,
                ),
                gelu: dev.bind(
                    &k.table,
                    &[
                        (0, &w.tmp),
                        (1, &l.gelu),
                        (
                            2,
                            &bl.uniform(&[
                                dm.d_ff,
                                s.tmpn,
                                s.fw,
                                l.gelu_offset,
                                l.gelu_len,
                                0,
                                0,
                                0,
                            ]),
                        ),
                        (3, &io.rows),
                        (4, &w.g),
                    ],
                ),
                mlp_mm: mm_group(&mut bl, model, &l.mlp_proj, &w.g, (&w.acc, s.accn), &io.rows),
                mlp_out: lin_group(
                    &mut bl,
                    model,
                    &l.mlp_proj,
                    &LinIo {
                        acc: (&w.acc, s.accn),
                        out: (&w.tmp, s.tmpn),
                        range: (i32::MIN, i32::MAX),
                        packed: false,
                    },
                    &io.rows,
                ),
            });
            bl.keep.push(kc);
            bl.keep.push(vc);
        }
        let ln_f = norm_group(&mut bl, model, &model.ln_f, &w.x, &w.a, &io.rows);
        let logits_lin = LinearW {
            w: model.wte.clone(),
            b: model.logit_m.clone(),
            m: model.logit_m.clone(),
            shift: model.logit_s,
            rows: dm.vocab,
            kw: s.dw,
        };
        let logits_mm = mm_group(&mut bl, model, &logits_lin, &w.a, (&w.acc, s.accn), &io.rows);
        let lp = bl.uniform(&[dm.vocab, s.accn, model.logit_s, model.logits_len]);
        let logits_scale = dev.bind(
            &k.logits_scale,
            &[(0, &w.acc), (1, &model.logit_m), (2, &lp), (3, &io.rows), (4, &w.logit)],
        );
        let logits_max =
            dev.bind(&k.logits_max, &[(2, &lp), (3, &io.rows), (4, &w.logit), (5, &w.row_max)]);
        let logits_weights = dev.bind(
            &k.logits_weights,
            &[
                (2, &lp),
                (3, &io.rows),
                (4, &w.logit),
                (5, &w.row_max),
                (6, &model.logits_exp),
                (7, &io.weights),
            ],
        );
        let coder = dev.bind(
            &k.coder_interval,
            &[
                (0, &io.weights),
                (1, &io.queries),
                (2, &io.freq_params),
                (3, &io.rows),
                (4, &io.intervals),
            ],
        );
        bl.keep.extend([w.x, w.a, w.qkv, w.att, w.g, w.acc, w.tmp, w.sc, w.logit, w.row_max]);
        dev.check()?;
        Ok(Self {
            model,
            slots,
            pos: vec![0; slots],
            host: vec![0; slots * model.vocab_size()],
            io,
            embed,
            layers,
            ln_f,
            logits_mm,
            logits_scale,
            logits_max,
            logits_weights,
            coder,
            last_intervals: Vec::new(),
            gpu_bytes: bl.bytes + weights_bytes,
            _keep: bl.keep,
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

    /// The bytes of GPU memory that this batch uses: KV caches, work buffers and the read-back
    /// buffer (measured from the allocations).
    #[must_use]
    pub fn gpu_bytes(&self) -> u64 {
        self.gpu_bytes
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
    /// same history. A slot whose context is full starts again first, as on the CPU (plan
    /// §7.7). Cost: one forward pass for the listed slots (two if a slot starts again), one read
    /// back of `inputs.len() * vocab * 8` bytes.
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
        let restart: Vec<(usize, TokenId, usize)> = inputs
            .iter()
            .filter(|(s, _)| self.position(*s) >= n_ctx)
            .map(|&(s, _)| (s, eot, 0))
            .collect();
        if !restart.is_empty() {
            // PLAN(§7.7): the context starts again with <|endoftext|> at position 0.
            self.forward(&restart, Mode::Submit)?;
            for &(s, _, _) in &restart {
                if let Some(p) = self.pos.get_mut(s) {
                    *p = 1;
                }
            }
        }
        let rows: Vec<(usize, TokenId, usize)> =
            inputs.iter().map(|&(s, t)| (s, t, self.position(s))).collect();
        let profile = self.forward(&rows, mode)?;
        for &(s, _) in inputs {
            if let Some(p) = self.pos.get_mut(s) {
                *p += 1;
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
        self.model.dev.write_words(&self.io.freq_params, &[vocab, t_lo, t_hi, spare]);
        self.model.dev.write_words(&self.io.queries, &words);
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
    fn forward(
        &mut self,
        rows: &[(usize, TokenId, usize)],
        mode: Mode,
    ) -> Result<Option<Vec<(&'static str, u64)>>> {
        let model = self.model;
        let dev = &model.dev;
        let dm = model.dims;
        let n = u32_of(rows.len(), "rows")?;
        let mut info = vec![n, 0, 0, 0];
        for &(s, t, p) in rows {
            info.extend([u32_of(s, "slot")?, t.get(), u32_of(p, "position")?, 0]);
        }
        dev.write_words(&self.io.rows, &info);
        let kernel = MatKernel::for_rows(rows.len());
        let k = &model.k;
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
            r.run("embed", &k.embed, &self.embed, dm.d.div_ceil(64), n);
            for (l, lw) in self.layers.iter().zip(&model.layers) {
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
            r.run("norm", &k.norm, &self.ln_f, n, 1);
            r.matmul("matmul logits", &self.logits_mm, dm.vocab);
            r.run("logits", &k.logits_scale, &self.logits_scale, dm.vocab.div_ceil(64), n);
            r.run("logits", &k.logits_max, &self.logits_max, n, 1);
            r.run("logits", &k.logits_weights, &self.logits_weights, dm.vocab.div_ceil(64), n);
            if mode == Mode::Intervals {
                r.run("coder interval", &k.coder_interval, &self.coder, n, 1);
            }
            labels = r.labels;
        }
        match mode {
            Mode::Submit => {
                dev.queue.submit([enc.finish()]);
                dev.check()?;
                Ok(None)
            }
            Mode::Wait => {
                dev.finish_and_wait(enc)?;
                Ok(None)
            }
            Mode::Read => {
                let bytes = (rows.len() * model.vocab_size() * 8) as u64;
                let mut out = Vec::new();
                dev.finish_and_read(enc, &self.io.weights, &self.io.staging, bytes, &mut out)?;
                self.copy_weights(rows, &out);
                Ok(None)
            }
            Mode::Intervals => {
                let bytes = 16 * u64::from(n);
                let mut out = Vec::new();
                dev.finish_and_read(
                    enc,
                    &self.io.intervals,
                    &self.io.interval_staging,
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
                        (&self.io.weights, &self.io.staging, bytes),
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

    fn copy_weights(&mut self, rows: &[(usize, TokenId, usize)], out: &[u8]) {
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

/// What a forward pass does after the kernels.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Submit only (the restart pass: its weights are not used).
    Submit,
    /// Submit and wait, no read back.
    Wait,
    /// Read the weights back.
    Read,
    /// Read the weights and the timestamps back.
    Profile,
    /// Run the coder op and read only the intervals back.
    Intervals,
}

/// The most timestamps of one pass: one for each dispatch, and the start.
const MAX_STAMPS: u32 = 4096;

/// Records dispatches, with a timestamp after each one when profiling.
struct Rec<'a, 'p> {
    pass: wgpu::ComputePass<'p>,
    qs: Option<&'a wgpu::QuerySet>,
    labels: Vec<&'static str>,
    mm: (&'a wgpu::ComputePipeline, &'a wgpu::ComputePipeline),
    kernel: MatKernel,
    n: u32,
}

impl Rec<'_, '_> {
    fn stamp(&mut self, label: &'static str) {
        if let Some(qs) = self.qs
            && let Ok(i) = u32::try_from(self.labels.len())
            && i < MAX_STAMPS
        {
            self.pass.write_timestamp(qs, i);
            self.labels.push(label);
        }
    }

    fn run(
        &mut self,
        label: &'static str,
        pipe: &wgpu::ComputePipeline,
        g: &wgpu::BindGroup,
        x: u32,
        y: u32,
    ) {
        dispatch(&mut self.pass, pipe, g, x, y);
        self.stamp(label);
    }

    fn matmul(&mut self, label: &'static str, g: &MmGroup, rows_of_w: u32) {
        record_matmul(&mut self.pass, self.mm, g, self.kernel, rows_of_w, self.n);
        self.stamp(label);
    }
}

/// The ticks between each timestamp and the one before, summed for each label (first-seen
/// order).
fn sum_stamps(labels: &[&'static str], bytes: &[u8]) -> Vec<(&'static str, u64)> {
    let words = bytes_to_words(bytes);
    let ticks: Vec<u64> = words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u64::from(p[0]) | (u64::from(p[1]) << 32))
        .collect();
    let mut out: Vec<(&'static str, u64)> = Vec::new();
    for (i, label) in labels.iter().enumerate().skip(1) {
        let d = ticks
            .get(i)
            .copied()
            .unwrap_or(0)
            .saturating_sub(ticks.get(i - 1).copied().unwrap_or(0));
        match out.iter_mut().find(|(l, _)| l == label) {
            Some(e) => e.1 += d,
            None => out.push((label, d)),
        }
    }
    out
}

fn dispatch(
    p: &mut wgpu::ComputePass<'_>,
    pipe: &wgpu::ComputePipeline,
    g: &wgpu::BindGroup,
    x: u32,
    y: u32,
) {
    p.set_pipeline(pipe);
    p.set_bind_group(0, g, &[]);
    p.dispatch_workgroups(x, y, 1);
}

fn norm_group(
    bl: &mut Builder<'_>,
    model: &GpuModel,
    n: &NormW,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    rows: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let dm = model.dims;
    let (k, j, eps) = model.ln;
    let params = bl.uniform(&[
        dm.d,
        dm.d,
        dm.dw,
        k,
        j,
        (eps & 0xffff_ffff) as u32,
        (eps >> 32) as u32,
        n.m.cast_unsigned(),
        n.s,
        0,
        0,
        0,
    ]);
    bl.dev.bind(&model.k.norm, &[(0, x), (1, &n.g), (2, &n.b), (3, &params), (4, rows), (5, out)])
}

fn mm_group(
    bl: &mut Builder<'_>,
    model: &GpuModel,
    lin: &LinearW,
    a: &wgpu::Buffer,
    out: (&wgpu::Buffer, u32),
    rows: &wgpu::Buffer,
) -> MmGroup {
    let (out, out_stride) = out;
    let params = bl.uniform(&[lin.rows, lin.kw, out_stride, 0, 1, 0, 0, 0]);
    MmGroup::new(
        bl.dev,
        (&model.k.matvec, &model.k.gemm),
        &[(0, &lin.w), (1, a), (2, out), (3, &params), (4, rows)],
    )
}

/// The buffers and the output rule of one `linear` output op.
struct LinIo<'b> {
    /// The accumulators and their row stride.
    acc: (&'b wgpu::Buffer, u32),
    /// The output and its row stride (words if packed, values if not).
    out: (&'b wgpu::Buffer, u32),
    /// The clamp range.
    range: (i32, i32),
    /// Packed i8 output (`linear_i8`) or i32 output (`linear_i32`).
    packed: bool,
}

fn lin_group(
    bl: &mut Builder<'_>,
    model: &GpuModel,
    lin: &LinearW,
    io: &LinIo<'_>,
    rows: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let params = bl.uniform(&[
        lin.rows,
        io.acc.1,
        io.out.1,
        lin.shift,
        io.range.0.cast_unsigned(),
        io.range.1.cast_unsigned(),
        1,
        0,
    ]);
    let pipe = if io.packed { &model.k.linear_i8 } else { &model.k.linear_i32 };
    bl.dev.bind(
        pipe,
        &[(0, io.acc.0), (1, &lin.b), (2, &lin.m), (3, &params), (4, rows), (5, io.out.0)],
    )
}

fn add_group(
    bl: &mut Builder<'_>,
    model: &GpuModel,
    s: Shape,
    w: &Work,
    rows: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let params = bl.uniform(&[
        s.d,
        s.d,
        s.tmpn,
        0,
        (-hardshell::gpt2::RESIDUAL_LIMIT).cast_unsigned(),
        hardshell::gpt2::RESIDUAL_LIMIT.cast_unsigned(),
        0,
        0,
    ]);
    bl.dev.bind(&model.k.add, &[(0, &w.x), (1, &w.tmp), (2, &params), (3, rows)])
}
