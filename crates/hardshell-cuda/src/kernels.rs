//! The compiled kernels: the PTX that `tools/cuda/build_ptx.py` makes from
//! `kernels/hardshell.cu`, embedded in the crate.
//!
//! PLAN(§7): each kernel reproduces one rule of hardshell exactly; the op set follows
//! `docs/design/anymodel.md` §4. The host composes them into the GPT-2 step (`batch`).

use crate::driver::{Context, Kernel};
use crate::error::Result;

/// The PTX module (compute_80). The driver compiles it for the GPU when it loads.
pub(crate) const PTX: &str = include_str!("../kernels/hardshell.ptx");

/// The CUDA C source of the PTX. A test checks that the PTX names its SHA-256.
pub const SOURCE: &str = include_str!("../kernels/hardshell.cu");

/// The header line of the PTX that names the SHA-256 of the source.
pub const PTX_SOURCE_LINE: &str = "// source-sha256: ";

/// The PTX that the crate embeds.
#[must_use]
pub fn ptx() -> &'static str {
    PTX
}

/// Row strides of `i8` matrix operands are a multiple of this many 4-byte words (64 bytes).
pub(crate) const K_ALIGN_WORDS: usize = 16;
/// Matrix operand buffers hold their rows rounded up to a multiple of this.
pub(crate) const ROW_ALIGN: usize = 64;

/// The kernel handles.
pub(crate) struct Kernels {
    pub(crate) gemm_tiled: Kernel,
    pub(crate) gemm_tc128: Kernel,
    pub(crate) gemm_tc64: Kernel,
    pub(crate) gemm_tc128_logits: Kernel,
    pub(crate) gemm_tc64_logits: Kernel,
    pub(crate) matvec_dp4a: Kernel,
    pub(crate) embed: Kernel,
    pub(crate) layer_norm: Kernel,
    pub(crate) linear_out: Kernel,
    pub(crate) add_clamp: Kernel,
    pub(crate) table_lookup: Kernel,
    pub(crate) kv_store: Kernel,
    pub(crate) attend: Kernel,
    pub(crate) attend_rows: Kernel,
    pub(crate) logits_scale: Kernel,
    pub(crate) logits_weights: Kernel,
    pub(crate) coder_interval: Kernel,
    pub(crate) coder_logits: Kernel,
    pub(crate) coder_d16: Kernel,
    pub(crate) mix_stats_weights: Kernel,
    pub(crate) mix_stats_logits: Kernel,
    pub(crate) mix_stats_d16: Kernel,
    pub(crate) coder_mixed_weights: Kernel,
    pub(crate) coder_mixed_logits: Kernel,
    pub(crate) coder_mixed_d16: Kernel,
    pub(crate) residual_norm: Kernel,
    pub(crate) linear_gelu: Kernel,
    pub(crate) linear_qkv: Kernel,
    pub(crate) stamp: Kernel,
    pub(crate) read_bandwidth: Kernel,
}

impl Kernels {
    pub(crate) fn new(ctx: &Context) -> Result<Self> {
        Ok(Self {
            gemm_tiled: ctx.function("gemm_tiled")?,
            gemm_tc128: ctx.function("gemm_tc128")?,
            gemm_tc64: ctx.function("gemm_tc64")?,
            gemm_tc128_logits: ctx.function("gemm_tc128_logits")?,
            gemm_tc64_logits: ctx.function("gemm_tc64_logits")?,
            matvec_dp4a: ctx.function("matvec_dp4a")?,
            embed: ctx.function("embed")?,
            layer_norm: ctx.function("layer_norm")?,
            linear_out: ctx.function("linear_out")?,
            add_clamp: ctx.function("add_clamp")?,
            table_lookup: ctx.function("table_lookup")?,
            kv_store: ctx.function("kv_store")?,
            attend: ctx.function("attend")?,
            attend_rows: ctx.function("attend_rows")?,
            logits_scale: ctx.function("logits_scale")?,
            logits_weights: ctx.function("logits_weights")?,
            coder_interval: ctx.function("coder_interval")?,
            coder_logits: ctx.function("coder_logits")?,
            coder_d16: ctx.function("coder_d16")?,
            mix_stats_weights: ctx.function("mix_stats_weights")?,
            mix_stats_logits: ctx.function("mix_stats_logits")?,
            mix_stats_d16: ctx.function("mix_stats_d16")?,
            coder_mixed_weights: ctx.function("coder_mixed_weights")?,
            coder_mixed_logits: ctx.function("coder_mixed_logits")?,
            coder_mixed_d16: ctx.function("coder_mixed_d16")?,
            residual_norm: ctx.function("residual_norm")?,
            linear_gelu: ctx.function("linear_gelu")?,
            linear_qkv: ctx.function("linear_qkv")?,
            stamp: ctx.function("stamp")?,
            read_bandwidth: ctx.function("read_bandwidth")?,
        })
    }
}

impl Kernels {
    /// Every kernel with its name.
    pub(crate) fn all(&self) -> Vec<&Kernel> {
        vec![
            &self.gemm_tiled,
            &self.gemm_tc128,
            &self.gemm_tc64,
            &self.gemm_tc128_logits,
            &self.gemm_tc64_logits,
            &self.matvec_dp4a,
            &self.embed,
            &self.layer_norm,
            &self.linear_out,
            &self.add_clamp,
            &self.table_lookup,
            &self.kv_store,
            &self.attend,
            &self.attend_rows,
            &self.logits_scale,
            &self.logits_weights,
            &self.coder_interval,
            &self.coder_logits,
            &self.coder_d16,
            &self.mix_stats_weights,
            &self.mix_stats_logits,
            &self.mix_stats_d16,
            &self.coder_mixed_weights,
            &self.coder_mixed_logits,
            &self.coder_mixed_d16,
            &self.residual_norm,
            &self.linear_gelu,
            &self.linear_qkv,
            &self.stamp,
            &self.read_bandwidth,
        ]
    }
}

/// The row stride in words for `cols` i8 values: a multiple of 16 words.
pub(crate) fn stride_words(cols: usize) -> usize {
    cols.div_ceil(4).div_ceil(K_ALIGN_WORDS) * K_ALIGN_WORDS
}

/// `rows` rounded up to a multiple of [`ROW_ALIGN`].
pub(crate) fn padded_rows(rows: usize) -> usize {
    rows.div_ceil(ROW_ALIGN) * ROW_ALIGN
}

/// Copies `rows` rows of `cols` i8 values into rows of `stride` bytes (zero padding).
pub(crate) fn pad_rows(data: &[i8], rows: usize, cols: usize, stride: usize) -> Vec<u8> {
    let mut out = vec![0u8; padded_rows(rows) * stride];
    for (dst, src) in out.chunks_mut(stride).zip(data.chunks(cols.max(1)).take(rows)) {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = s.to_le_bytes()[0];
        }
    }
    out
}
