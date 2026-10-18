//! The compute pipelines: one WGSL module for each op, compiled one time for each device.
//!
//! PLAN(§7): each op reproduces one rule of hardshell exactly. The ops follow the op set of
//! `docs/design/anymodel.md` §4 (embed, norm, linear, add, table, attention, logits). The host
//! composes them into the GPT-2 step (`batch`), so a later model graph can reuse them.

use crate::device::GpuDevice;
use crate::error::Result;

const INT: &str = include_str!("shaders/int.wgsl");
const COMMON: &str = include_str!("shaders/common.wgsl");
pub(crate) const MATMUL: &str = include_str!("shaders/matmul.wgsl");
pub(crate) const SELFTEST: &str = include_str!("shaders/selftest.wgsl");
const EMBED: &str = include_str!("shaders/embed.wgsl");
const NORM: &str = include_str!("shaders/norm.wgsl");
const LINEAR: &str = include_str!("shaders/linear.wgsl");
const ADD: &str = include_str!("shaders/add.wgsl");
const TABLE: &str = include_str!("shaders/table.wgsl");
const ATTENTION: &str = include_str!("shaders/attention.wgsl");
const LOGITS: &str = include_str!("shaders/logits.wgsl");
const FREQ: &str = include_str!("shaders/freq.wgsl");

/// All shader sources, for the test that refuses floats in them.
pub const SOURCES: [(&str, &str); 12] = [
    ("int.wgsl", INT),
    ("common.wgsl", COMMON),
    ("matmul.wgsl", MATMUL),
    ("selftest.wgsl", SELFTEST),
    ("embed.wgsl", EMBED),
    ("norm.wgsl", NORM),
    ("linear.wgsl", LINEAR),
    ("add.wgsl", ADD),
    ("table.wgsl", TABLE),
    ("attention.wgsl", ATTENTION),
    ("logits.wgsl", LOGITS),
    ("freq.wgsl", FREQ),
];

/// Rows of `a` that the `matvec` kernel takes in one pass. More rows use `gemm`.
pub(crate) const MATVEC_MAX_ROWS: u32 = 8;
/// The tile of the `gemm` kernel.
pub(crate) const GEMM_TILE: u32 = 64;
/// Row strides of packed `i8` operands are a multiple of this many words (64 bytes).
pub(crate) const K_ALIGN_WORDS: usize = 16;

/// The compiled pipelines of one device.
pub(crate) struct Kernels {
    pub(crate) matvec: wgpu::ComputePipeline,
    pub(crate) gemm: wgpu::ComputePipeline,
    pub(crate) embed: wgpu::ComputePipeline,
    pub(crate) norm: wgpu::ComputePipeline,
    pub(crate) linear_i8: wgpu::ComputePipeline,
    pub(crate) linear_i32: wgpu::ComputePipeline,
    pub(crate) add: wgpu::ComputePipeline,
    pub(crate) table: wgpu::ComputePipeline,
    pub(crate) kv_store: wgpu::ComputePipeline,
    pub(crate) attend: wgpu::ComputePipeline,
    pub(crate) logits_scale: wgpu::ComputePipeline,
    pub(crate) logits_max: wgpu::ComputePipeline,
    pub(crate) logits_weights: wgpu::ComputePipeline,
    pub(crate) coder_interval: wgpu::ComputePipeline,
}

/// Compiles `src` with the shared parts and gives the pipeline of `entry`.
pub(crate) fn pipeline(dev: &GpuDevice, src: &str, entry: &str) -> Result<wgpu::ComputePipeline> {
    let m = dev.module(entry, &[INT, COMMON, src])?;
    dev.pipeline(&m, entry)
}

impl Kernels {
    /// Compiles every op. Cost: the driver compiles 12 small shaders (est. well below 1 s).
    pub(crate) fn new(dev: &GpuDevice) -> Result<Self> {
        let mm = dev.module("matmul", &[INT, COMMON, MATMUL])?;
        let attn = dev.module("attention", &[INT, COMMON, ATTENTION])?;
        let lin = dev.module("linear", &[INT, COMMON, LINEAR])?;
        let lg = dev.module("logits", &[INT, COMMON, LOGITS])?;
        Ok(Self {
            matvec: dev.pipeline(&mm, "matvec")?,
            gemm: dev.pipeline(&mm, "gemm")?,
            embed: pipeline(dev, EMBED, "embed")?,
            norm: pipeline(dev, NORM, "norm")?,
            linear_i8: dev.pipeline(&lin, "linear_i8")?,
            linear_i32: dev.pipeline(&lin, "linear_i32")?,
            add: pipeline(dev, ADD, "add")?,
            table: pipeline(dev, TABLE, "table")?,
            kv_store: dev.pipeline(&attn, "kv_store")?,
            attend: dev.pipeline(&attn, "attend")?,
            logits_scale: dev.pipeline(&lg, "logits_scale")?,
            logits_max: dev.pipeline(&lg, "logits_max")?,
            logits_weights: dev.pipeline(&lg, "logits_weights")?,
            coder_interval: pipeline(dev, FREQ, "coder_interval")?,
        })
    }
}

/// Packs `rows` rows of `cols` i8 values into words, with a row stride of `kw` words. Byte `k`
/// of a word is column `4w + k`; the padding is zero.
pub(crate) fn pack_i8(data: &[i8], rows: usize, cols: usize, kw: usize) -> Vec<u32> {
    let mut out = vec![0u32; rows * kw];
    for (r, row) in data.chunks(cols.max(1)).take(rows).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            let w = r * kw + c / 4;
            if let Some(word) = out.get_mut(w) {
                *word |= u32::from(v.to_le_bytes()[0]) << (8 * (c % 4));
            }
        }
    }
    out
}

/// The row stride in words for `cols` packed i8 values: a multiple of 16 words.
pub(crate) fn stride_words(cols: usize) -> usize {
    cols.div_ceil(4).div_ceil(K_ALIGN_WORDS) * K_ALIGN_WORDS
}
