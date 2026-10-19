//! The GPU memory of a batch: the KV caches, the work buffers, and the bind groups of every op
//! of the forward pass. A [`Plan`] is made one time, so a step only writes the rows and
//! dispatches.
//!
//! The KV caches have one region for each slot. The work buffers have one region for each row
//! of a pass. A batch with a context overlap has room for more rows than slots, so the
//! `K + 1` rows of a restart run in few passes (ADR 0015).

use crate::device::GpuDevice;
use crate::error::{Error, Result};
use crate::kernels::stride_words;
use crate::model::{GpuModel, LinearW, NormW};
use crate::ops::{MmGroup, check_grid, u32_of};

/// The bind groups of one block.
pub(super) struct LayerPlan {
    pub(super) ln_1: wgpu::BindGroup,
    pub(super) qkv_mm: MmGroup,
    pub(super) qkv_out: wgpu::BindGroup,
    pub(super) kv: wgpu::BindGroup,
    pub(super) attn: wgpu::BindGroup,
    pub(super) proj_mm: MmGroup,
    pub(super) proj_out: wgpu::BindGroup,
    pub(super) add: wgpu::BindGroup,
    pub(super) ln_2: wgpu::BindGroup,
    pub(super) fc_mm: MmGroup,
    pub(super) fc_out: wgpu::BindGroup,
    pub(super) gelu: wgpu::BindGroup,
    pub(super) mlp_mm: MmGroup,
    pub(super) mlp_out: wgpu::BindGroup,
}

/// The buffers of a batch that the host reads or writes.
pub(super) struct Io {
    pub(super) rows: wgpu::Buffer,
    pub(super) weights: wgpu::Buffer,
    pub(super) staging: wgpu::Buffer,
    /// One coder query for each row: kind (0 encode, 1 decode), token or target.
    pub(super) queries: wgpu::Buffer,
    /// One coder interval for each row: symbol, start, freq (two words).
    pub(super) intervals: wgpu::Buffer,
    pub(super) interval_staging: wgpu::Buffer,
    /// The `FreqParams` of the coder op.
    pub(super) freq_params: wgpu::Buffer,
}

/// The buffers and the bind groups of a batch.
pub(super) struct Plan {
    pub(super) io: Io,
    pub(super) embed: wgpu::BindGroup,
    pub(super) layers: Vec<LayerPlan>,
    pub(super) ln_f: wgpu::BindGroup,
    pub(super) logits_mm: MmGroup,
    pub(super) logits_scale: wgpu::BindGroup,
    pub(super) logits_max: wgpu::BindGroup,
    pub(super) logits_weights: wgpu::BindGroup,
    pub(super) coder: wgpu::BindGroup,
    /// The bytes of GPU memory of the plan (from the allocations).
    pub(super) bytes: u64,
    /// Keeps the buffers that only bind groups use.
    _keep: Vec<wgpu::Buffer>,
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

impl Plan {
    /// Makes the KV caches of `slots` slots, the work buffers of passes of up to `rows` rows,
    /// and the bind groups. The caller gives `1 <= slots <= rows`.
    ///
    /// # Errors
    /// [`Error::DeviceLimit`] if a buffer is larger than one binding of the device or the grid
    /// is too large; [`Error::Gpu`] on a GPU error.
    pub(super) fn new(model: &GpuModel, slots: usize, rows: usize) -> Result<Self> {
        let dev = &model.dev;
        let dm = model.dims;
        let b = u64::from(u32_of(rows.max(1), "rows")?);
        let slots = u64::from(u32_of(slots.max(1), "slots")?);
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
            let cache = slots * u64::from(dm.n_ctx) * u64::from(s.d);
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
            io,
            embed,
            layers,
            ln_f,
            logits_mm,
            logits_scale,
            logits_max,
            logits_weights,
            coder,
            bytes: bl.bytes + weights_bytes,
            _keep: bl.keep,
        })
    }
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
