//! The fused path of a pass: fewer kernels than the op-level path, the same bits.
//!
//! Each fused kernel computes, in one launch, the expressions of two or three op kernels in the
//! same order (`kernels/hardshell.cu`):
//!
//! - `linear_qkv`: `linear_out` of `c_attn` and `kv_store`.
//! - `residual_norm`: `linear_out` of `c_proj` or `mlp.c_proj` (i32), `add_clamp`, and the next
//!   `layer_norm` (`ln_2`, the `ln_1` of the next block, or `ln_f`).
//! - `linear_gelu`: `linear_out` of `c_fc` and `table_lookup`.
//! - `coder_logits`: `logits_weights` and `coder_interval` (only when the pass gives intervals).
//! - `gemm_tc64_logits`, `gemm_tc128_logits` with `coder_d16`: the logits product,
//!   `logits_scale`, `logits_weights` and `coder_interval`, with compact logits (u16 differences to
//!   the max of each 128-column tile) between the two kernels.
//! - The mixed coder op (`batch::mixed`): `mix_stats_logits` or `mix_stats_d16`, then
//!   `coder_mixed_logits` or `coder_mixed_d16`, on the same two sources.
//!
//! ADR 0013 permits a fused kernel only with a test that proves it equal to the op-level path:
//! `tests/fused.rs` runs both paths on the same inputs and compares every weight and interval.

use super::mixed::MixSrc;
use super::{CODER_TILES, CudaBatch, LOGIT_TILE, NORM_THREADS, Out, Stamps, mark};
use crate::driver::{Arg, DevBuf};
use crate::error::Result;
use crate::model::{LayerW, LinearW, NormW};
use crate::ops::MatKernel;

impl CudaBatch<'_> {
    /// Launches the kernels of a pass of `n` rows on the fused path.
    pub(super) fn record_fused(
        &self,
        n: u32,
        kernel: MatKernel,
        out: Out,
        groups: u32,
        stamps: &mut Option<Stamps<'_>>,
    ) -> Result<()> {
        let model = self.model;
        let dev = model.dev;
        let b = &self.b;
        let mut op = |name: &'static str| mark(dev, stamps, name);
        op("start")?;
        self.embed(n)?;
        op("embed")?;
        let Some(first) = model.layers.first() else {
            return Err(crate::error::Error::Unsupported { reason: "a model needs a block" });
        };
        self.norm(&first.ln_1, n)?;
        op("norm")?;
        for (i, (l, (kc, vc))) in model.layers.iter().zip(&self.kv).enumerate() {
            let z = self.matmul(kernel, &l.c_attn, &b.a, n, true)?;
            op("matmul qkv")?;
            self.linear_qkv(&l.c_attn, kc, vc, n, z)?;
            op("linear qkv")?;
            self.attend(l, kc, vc, n, groups)?;
            op("attention")?;
            let z = self.matmul(kernel, &l.c_proj, &b.att, n, true)?;
            op("matmul proj")?;
            self.residual_norm(&l.c_proj, &l.ln_2, n, z)?;
            op("residual norm")?;
            let z = self.matmul(kernel, &l.c_fc, &b.a, n, true)?;
            op("matmul fc")?;
            self.linear_gelu(l, n, z)?;
            op("linear gelu")?;
            let z = self.matmul(kernel, &l.mlp_proj, &b.g, n, true)?;
            op("matmul mlp")?;
            let next = model.layers.get(i + 1).map_or(&model.ln_f, |x| &x.ln_1);
            self.residual_norm(&l.mlp_proj, next, n, z)?;
            op("residual norm")?;
        }
        if let Out::Intervals(_) | Out::Mixed(_) = out
            && self.compact_logits(kernel, n)?
        {
            op("matmul logits")?;
            if let Out::Intervals(bits) = out {
                self.coder_d16(n, bits, &self.b.intervals)?;
                op("coder interval")?;
            } else {
                self.mix_stats(MixSrc::D16, n)?;
                op("mix stats")?;
            }
            return Ok(());
        }
        self.matmul(kernel, &model.wte, &b.a, n, false)?;
        op("matmul logits")?;
        self.logits_scale(n)?;
        op("logits")?;
        match out {
            Out::Intervals(bits) => {
                self.coder_logits(n, bits, &self.b.intervals)?;
                op("coder interval")?;
            }
            Out::Mixed(_) => {
                self.mix_stats(MixSrc::Logits, n)?;
                op("mix stats")?;
            }
            Out::Nothing | Out::Weights => {
                self.logits_weights(n)?;
                op("logits")?;
            }
        }
        Ok(())
    }

    /// The logits product with the compact epilogue (`gemm_tc*_logits`: the tile maxima and the
    /// u16 differences), for the tensor-core kernels with 128-column tiles, a logits table of at
    /// most 2^16 entries, and at most `CODER_TILES` tiles. Gives `false` (and launches nothing)
    /// otherwise.
    fn compact_logits(&self, kernel: MatKernel, n: u32) -> Result<bool> {
        let m = self.model;
        let Some(f) = self.compact_kernel(kernel) else { return Ok(false) };
        let (tm, tn, _) = kernel.tile();
        let wte = &m.wte;
        let tiles = wte.rows.div_ceil(tn);
        m.dev.ctx.launch(
            f,
            (n.div_ceil(tm), tiles, 1),
            (256, 1, 1),
            &[
                Arg::Ptr(wte.w.ptr()),
                Arg::Ptr(self.b.a.ptr()),
                Arg::Ptr(wte.m.ptr()),
                Arg::Ptr(self.b.d16.ptr()),
                Arg::Ptr(self.b.tmax.ptr()),
                Arg::U32(wte.rows),
                Arg::U32(wte.kw),
                Arg::U32(m.dims.vocab),
                Arg::U32(n),
                Arg::U32(wte.shift),
                Arg::U32(m.logits_len.saturating_sub(1)),
            ],
        )?;
        Ok(true)
    }

    /// The logits kernel with the compact epilogue for `kernel` products, if the pass can use
    /// it (the conditions of [`CudaBatch::compact_logits`]).
    pub(super) fn compact_kernel(&self, kernel: MatKernel) -> Option<&crate::driver::Kernel> {
        let m = self.model;
        let k = &m.dev.k;
        let f = match kernel {
            MatKernel::TensorCore64 => &k.gemm_tc64_logits,
            MatKernel::TensorCore128 => &k.gemm_tc128_logits,
            _ => return None,
        };
        let (_, tn, _) = kernel.tile();
        let tiles = m.wte.rows.div_ceil(tn);
        (tn == LOGIT_TILE && m.logits_len <= 1 << 16 && tiles <= CODER_TILES).then_some(f)
    }

    /// The fused coder op of `n` rows on the compact logits, into `out`.
    pub(super) fn coder_d16(&self, n: u32, bits: u32, out: &DevBuf) -> Result<()> {
        let m = self.model;
        let b = &self.b;
        m.dev.launch(
            &m.dev.k.coder_d16,
            (n, 1),
            256,
            &[
                Arg::Ptr(b.d16.ptr()),
                Arg::Ptr(b.tmax.ptr()),
                Arg::Ptr(m.logits_exp.ptr()),
                Arg::Ptr(b.queries.ptr()),
                Arg::Ptr(out.ptr()),
                Arg::U32(m.dims.vocab),
                Arg::U32(m.dims.vocab.div_ceil(LOGIT_TILE)),
                Arg::U32(m.logits_len),
                Arg::U64(1u64 << bits),
            ],
        )
    }

    fn linear_qkv(
        &self,
        lin: &LinearW,
        kc: &DevBuf,
        vc: &DevBuf,
        n: u32,
        splits: u32,
    ) -> Result<()> {
        let m = self.model;
        let d = m.dims;
        m.dev.launch(
            &m.dev.k.linear_qkv,
            (lin.rows.div_ceil(128), n),
            128,
            &[
                Arg::Ptr(self.b.acc.ptr()),
                Arg::Ptr(lin.b.ptr()),
                Arg::Ptr(lin.m.ptr()),
                Arg::Ptr(self.b.qkv.ptr()),
                Arg::Ptr(kc.ptr()),
                Arg::Ptr(vc.ptr()),
                Arg::Ptr(self.b.rows.ptr()),
                Arg::U32(d.d),
                Arg::U32(3 * d.d),
                Arg::U32(lin.shift),
                Arg::U32(splits),
                Arg::U32(n * lin.rows),
                Arg::U32(d.n_ctx.next_multiple_of(4)),
                Arg::U32(d.head_dim),
            ],
        )
    }

    fn residual_norm(&self, lin: &LinearW, nw: &NormW, n: u32, splits: u32) -> Result<()> {
        let m = self.model;
        let (k, j, eps) = m.ln;
        m.dev.launch(
            &m.dev.k.residual_norm,
            (n, 1),
            NORM_THREADS,
            &[
                Arg::Ptr(self.b.acc.ptr()),
                Arg::Ptr(lin.b.ptr()),
                Arg::Ptr(lin.m.ptr()),
                Arg::U32(lin.shift),
                Arg::U32(splits),
                Arg::U32(n * lin.rows),
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

    fn linear_gelu(&self, l: &LayerW, n: u32, splits: u32) -> Result<()> {
        let m = self.model;
        let lin = &l.c_fc;
        m.dev.launch(
            &m.dev.k.linear_gelu,
            (lin.rows.div_ceil(128), n),
            128,
            &[
                Arg::Ptr(self.b.acc.ptr()),
                Arg::Ptr(lin.b.ptr()),
                Arg::Ptr(lin.m.ptr()),
                Arg::Ptr(l.gelu.ptr()),
                Arg::Ptr(self.b.g.ptr()),
                Arg::U32(lin.rows),
                Arg::U32(4 * m.dims.fw),
                Arg::U32(lin.shift),
                Arg::I32(-l.gelu_offset),
                Arg::I32(l.gelu_offset - 1),
                Arg::I32(l.gelu_offset),
                Arg::I32(l.gelu_len),
                Arg::U32(splits),
                Arg::U32(n * lin.rows),
            ],
        )
    }

    /// The fused coder op of `n` rows on the i64 logits, into `out`.
    pub(super) fn coder_logits(&self, n: u32, bits: u32, out: &DevBuf) -> Result<()> {
        let m = self.model;
        let b = &self.b;
        m.dev.launch(
            &m.dev.k.coder_logits,
            (n, 1),
            256,
            &[
                Arg::Ptr(b.logit.ptr()),
                Arg::Ptr(m.logits_exp.ptr()),
                Arg::Ptr(b.queries.ptr()),
                Arg::Ptr(out.ptr()),
                Arg::U32(m.dims.vocab),
                Arg::U32(m.logits_len),
                Arg::U64(1u64 << bits),
            ],
        )
    }
}
