//! The op-level path of a pass: one kernel for each op (ADR 0013), and the launches that both
//! paths share.

use super::mixed::MixSrc;
use super::{
    ATTEND_GROUP_ROWS, ATTEND_THREADS, CudaBatch, LinOut, NORM_THREADS, Out, Row, Stamps, mark,
};
use crate::driver::{Arg, DevBuf};
use crate::error::Result;
use crate::model::{LayerW, LinearW, NormW};
use crate::ops::{Mat, MatKernel, launch_matmul, plan_product};

impl CudaBatch<'_> {
    /// Launches the kernels of a pass on the selected path.
    pub(super) fn record_path(
        &self,
        n: u32,
        kernel: MatKernel,
        out: Out,
        groups: u32,
        stamps: &mut Option<Stamps<'_>>,
    ) -> Result<()> {
        if self.op_level {
            self.record(n, kernel, out, groups, stamps)
        } else {
            self.record_fused(n, kernel, out, groups, stamps)
        }
    }

    /// Launches all kernels of a pass of `n` rows (the rows are already on the GPU): one kernel
    /// for each op (ADR 0013), the reference structure of the fused path. `groups` is the number
    /// of row groups of the attention (0: one block for each row).
    pub(super) fn record(
        &self,
        n: u32,
        kernel: MatKernel,
        out: Out,
        groups: u32,
        stamps: &mut Option<Stamps<'_>>,
    ) -> Result<()> {
        let model = self.model;
        let dev = model.dev;
        let d = model.dims;
        let mut op = |name: &'static str| mark(dev, stamps, name);
        op("start")?;
        let b = &self.b;
        let k = &dev.k;
        let tmpn = self.tmpn;
        let ptr = |x: &DevBuf| Arg::Ptr(x.ptr());
        self.embed(n)?;
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
                    Arg::U32(d.n_ctx.next_multiple_of(4)),
                    Arg::U32(d.head_dim),
                ],
            )?;
            op("attention kv")?;
            self.attend(l, kc, vc, n, groups)?;
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
        self.logits_scale(n)?;
        op("logits")?;
        self.logits_weights(n)?;
        op("logits")?;
        if let Out::Intervals(bits) = out {
            self.coder_interval(n, bits, &self.b.intervals)?;
            op("coder interval")?;
        }
        if let Out::Mixed(_) = out {
            self.mix_stats(MixSrc::Weights, n)?;
            op("mix stats")?;
        }
        Ok(())
    }

    pub(super) fn embed(&self, n: u32) -> Result<()> {
        let model = self.model;
        let (dev, d, b) = (model.dev, model.dims, &self.b);
        let ptr = |x: &DevBuf| Arg::Ptr(x.ptr());
        dev.launch(
            &dev.k.embed,
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
        )
    }

    /// The row groups of a pass for `attend_rows`, as (first row, count) words: runs of up to
    /// `ATTEND_GROUP_ROWS` rows of one slot at consecutive positions. `None` if the head size does
    /// not suit the kernel (a multiple of 32, at most 128), or if the groups have fewer than two
    /// rows on average (decode): then `attend` runs one block for each row.
    pub(super) fn att_groups(&self, pass: &[Row]) -> Option<Vec<u32>> {
        let hd = self.model.dims.head_dim;
        if !hd.is_multiple_of(32) || hd > 128 {
            return None;
        }
        let mut words: Vec<u32> = Vec::new();
        let mut prev: Option<&Row> = None;
        for (i, row) in pass.iter().enumerate() {
            let joins = prev.is_some_and(|p| p.slot == row.slot && p.pos + 1 == row.pos)
                && words.last().is_some_and(|&c| c < ATTEND_GROUP_ROWS);
            if joins {
                if let Some(c) = words.last_mut() {
                    *c += 1;
                }
            } else {
                words.extend([u32::try_from(i).ok()?, 1]);
            }
            prev = Some(row);
        }
        // PERF: the group kernel pays three passes over the keys for each group; for groups of
        // one row (decode), `attend` is faster (est.).
        (words.len() <= pass.len()).then_some(words)
    }

    pub(super) fn attend(
        &self,
        l: &LayerW,
        kc: &DevBuf,
        vc: &DevBuf,
        n: u32,
        groups: u32,
    ) -> Result<()> {
        let model = self.model;
        let (dev, d, b) = (model.dev, model.dims, &self.b);
        let ptr = |x: &DevBuf| Arg::Ptr(x.ptr());
        if groups > 0 {
            return dev.launch(
                &dev.k.attend_rows,
                (d.n_head, groups),
                ATTEND_THREADS,
                &[
                    ptr(&b.qkv),
                    ptr(kc),
                    ptr(vc),
                    ptr(&b.groups),
                    ptr(&model.attn_exp),
                    ptr(&b.rows),
                    ptr(&b.att),
                    Arg::U32(3 * d.d),
                    Arg::U32(d.head_dim),
                    Arg::U32(d.n_ctx.next_multiple_of(4)),
                    Arg::U32(d.n_head),
                    Arg::I32(l.score.0),
                    Arg::U32(l.score.1),
                    Arg::I32(l.attn_out.0),
                    Arg::U32(l.attn_out.1),
                    Arg::U32(model.softmax_p),
                    Arg::U32(model.attn_len),
                    Arg::U32(4 * d.dw),
                ],
            );
        }
        dev.launch(
            &dev.k.attend,
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
                Arg::U32(d.n_ctx.next_multiple_of(4)),
                Arg::U32(d.n_head),
                Arg::I32(l.score.0),
                Arg::U32(l.score.1),
                Arg::I32(l.attn_out.0),
                Arg::U32(l.attn_out.1),
                Arg::U32(model.softmax_p),
                Arg::U32(model.attn_len),
                Arg::U32(4 * d.dw),
            ],
        )
    }

    pub(super) fn logits_scale(&self, n: u32) -> Result<()> {
        let model = self.model;
        let (dev, d, b) = (model.dev, model.dims, &self.b);
        dev.launch(
            &dev.k.logits_scale,
            (d.vocab.div_ceil(128), n),
            128,
            &[
                Arg::Ptr(b.acc.ptr()),
                Arg::Ptr(model.wte.m.ptr()),
                Arg::Ptr(b.logit.ptr()),
                Arg::U32(d.vocab),
                Arg::U32(d.vocab),
                Arg::U32(model.wte.shift),
            ],
        )
    }

    pub(super) fn logits_weights(&self, n: u32) -> Result<()> {
        let model = self.model;
        let (dev, d, b) = (model.dev, model.dims, &self.b);
        dev.launch(
            &dev.k.logits_weights,
            (n, 1),
            256,
            &[
                Arg::Ptr(b.logit.ptr()),
                Arg::Ptr(model.logits_exp.ptr()),
                Arg::Ptr(b.weights.ptr()),
                Arg::U32(d.vocab),
                Arg::U32(model.logits_len),
            ],
        )
    }

    /// The coder op of `n` rows on the weights of `logits_weights`, into `out`.
    pub(super) fn coder_interval(&self, n: u32, bits: u32, out: &DevBuf) -> Result<()> {
        let model = self.model;
        let (dev, d, b) = (model.dev, model.dims, &self.b);
        dev.launch(
            &dev.k.coder_interval,
            (n, 1),
            256,
            &[
                Arg::Ptr(b.weights.ptr()),
                Arg::Ptr(b.queries.ptr()),
                Arg::Ptr(out.ptr()),
                Arg::U32(d.vocab),
                Arg::U64(1u64 << bits),
            ],
        )
    }

    pub(super) fn norm(&self, nw: &NormW, n: u32) -> Result<()> {
        let m = self.model;
        let (k, j, eps) = m.ln;
        m.dev.launch(
            &m.dev.k.layer_norm,
            (n, 1),
            NORM_THREADS,
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
    pub(super) fn matmul(
        &self,
        kernel: MatKernel,
        lin: &LinearW,
        a: &DevBuf,
        n: u32,
        split: bool,
    ) -> Result<u32> {
        // NOTE: the products of the layers take the measured plan; the logits product (no
        // split) takes the kernel of the pass, which the compact logits need.
        let (kernel, splits) = if split {
            let (k, z) = plan_product(lin.rows, lin.kw, n);
            (k, z.min(self.max_splits))
        } else {
            (kernel, 1)
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

    pub(super) fn linear_out(
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

    pub(super) fn add(&self, n: u32) -> Result<()> {
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
