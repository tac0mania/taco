//! Single ops on the GPU, for tests and measurements: the i8 matrix product with each kernel,
//! and the i8 x i16 product of model-file version 2.

use hardshell::tensor::Matrix;

use crate::device::{CudaDevice, bytes_i32, bytes_u64, i32_bytes, u32_of};
use crate::driver::{Arg, DevBuf};
use crate::error::{Error, Result};
use crate::kernels::{pad_rows, planes_of, row_sums, stride_words};

/// The kernel of a matrix product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatKernel {
    /// int8 tensor cores (`mma.sync` m16n8k32), 64 x 64 tiles in shared memory, split-K.
    TensorCore,
    /// int8 tensor cores, 64 (rows of `a`) x 128 (rows of `w`) tiles, `ldmatrix`, 4 stages of
    /// `cp.async`, split-K.
    TensorCore64,
    /// int8 tensor cores, 128 x 128 tiles, `ldmatrix`, 3 stages of `cp.async`, split-K.
    TensorCore128,
    /// `dp4a`, one warp for each weight row, up to 8 activation rows in one pass.
    Dp4a,
}

impl MatKernel {
    /// The kernel that a product with `rows` activation rows uses.
    #[must_use]
    pub fn for_rows(rows: usize) -> Self {
        match rows {
            0..=8 => Self::Dp4a,
            9..=64 => Self::TensorCore64,
            _ => Self::TensorCore128,
        }
    }

    /// The tile (rows of `a`, rows of `w`) of a tensor-core kernel, and the number of blocks
    /// that keeps the GPU busy (est.: the blocks that fit on the 30 SMs of the RTX 3060 at the
    /// same time).
    pub(crate) fn tile(self) -> (u32, u32, u32) {
        match self {
            Self::TensorCore => (64, 64, 120),
            Self::TensorCore64 => (64, 128, 60),
            Self::TensorCore128 => (128, 128, 60),
            Self::Dp4a => (8, 8, 1),
        }
    }

    /// `true` for the tensor-core kernels, which can split k.
    fn splits_k(self) -> bool {
        self != Self::Dp4a
    }
}

/// The operands of one matrix product `out = a * w^T`: `w` has `n` rows, `a` has `m` rows,
/// both with `kw` words in each row, in buffers of at least `padded_rows` rows. With split-K
/// (`splits > 1`, tensor cores only), partial `z` is at `out + z * m * out_stride` values; the
/// output op adds the partials.
pub(crate) struct Mat<'b> {
    pub(crate) w: &'b DevBuf,
    pub(crate) a: &'b DevBuf,
    pub(crate) out: &'b DevBuf,
    pub(crate) out_stride: u32,
    pub(crate) n: u32,
    pub(crate) kw: u32,
    pub(crate) m: u32,
    pub(crate) splits: u32,
}

/// The split-K factor for a product with `kernel`: enough blocks for the GPU, at most one stage
/// (64 bytes of k) for each split, and at most 16 splits.
pub(crate) fn splits_for(kernel: MatKernel, n: u32, kw: u32, m: u32) -> u32 {
    if !kernel.splits_k() {
        return 1;
    }
    let (tm, tn, target) = kernel.tile();
    let blocks = m.div_ceil(tm).max(1) * n.div_ceil(tn).max(1);
    target.div_ceil(blocks).clamp(1, (kw / 16).clamp(1, 16))
}

/// The kernel and the split-K factor of a product of a linear layer: `w` has `n` rows of `kw`
/// words, `a` has `m` rows. For the shapes of GPT-2 small (`c_attn`, `attn.c_proj`, `c_fc`,
/// `mlp.c_proj`) it takes the fastest choice of `gpt2 gemm` for the nearest measured `m`; for
/// other shapes, [`MatKernel::for_rows`] and [`splits_for`]. The choice changes no bit.
///
/// PERF: the table is the median of three runs of `gpt2 gemm 32 64 128 224 256 512 1024` on the
/// RTX 3060 Laptop (measured 2026-10-19, `docs/research/2026-10-18-gpu-backend.md` section 7).
/// The rule before it (enough blocks for 60 slots) was 25% to 40% slower at 128 and 224 rows.
pub(crate) fn plan_product(n: u32, kw: u32, m: u32) -> (MatKernel, u32) {
    use MatKernel::{TensorCore as A, TensorCore64 as B, TensorCore128 as C};
    // (n, kw): c_attn, attn.c_proj, c_fc, mlp.c_proj of width 768.
    let shape = match (n, kw) {
        (2304, 192) => Some(0),
        (768, 192) => Some(1),
        (3072, 192) => Some(2),
        (768, 768) => Some(3),
        _ => None,
    };
    let table: [(u32, [(MatKernel, u32); 4]); 6] = [
        (48, [(B, 2), (A, 2), (A, 1), (A, 4)]),
        (96, [(B, 2), (B, 2), (C, 1), (B, 4)]),
        (176, [(C, 2), (C, 3), (A, 1), (A, 2)]),
        (384, [(C, 1), (B, 1), (B, 1), (B, 2)]),
        (768, [(B, 1), (C, 1), (A, 1), (B, 1)]),
        (u32::MAX, [(C, 1), (B, 1), (B, 1), (C, 1)]),
    ];
    let measured = shape
        .filter(|_| m > 8)
        .and_then(|i| table.iter().find(|(top, _)| m <= *top).and_then(|r| r.1.get(i)));
    if let Some(&choice) = measured {
        return choice;
    }
    let kernel = MatKernel::for_rows(m as usize);
    (kernel, splits_for(kernel, n, kw, m))
}

/// Launches one matrix product.
pub(crate) fn launch_matmul(dev: &CudaDevice, kernel: MatKernel, p: &Mat<'_>) -> Result<()> {
    let (n, m) = (p.n, p.m);
    let args = [
        Arg::Ptr(p.w.ptr()),
        Arg::Ptr(p.a.ptr()),
        Arg::Ptr(p.out.ptr()),
        Arg::U32(n),
        Arg::U32(p.kw),
        Arg::U32(p.out_stride),
        Arg::U32(m),
        Arg::U32(m * p.out_stride),
    ];
    let (tm, tn, _) = kernel.tile();
    let grid = (m.div_ceil(tm), n.div_ceil(tn), p.splits.max(1));
    match kernel {
        MatKernel::TensorCore => dev.ctx.launch(&dev.k.gemm_tiled, grid, (128, 1, 1), &args),
        MatKernel::TensorCore64 => dev.ctx.launch(&dev.k.gemm_tc64, grid, (256, 1, 1), &args),
        MatKernel::TensorCore128 => dev.ctx.launch(&dev.k.gemm_tc128, grid, (256, 1, 1), &args),
        MatKernel::Dp4a => {
            dev.launch(&dev.k.matvec_dp4a, (n.div_ceil(8), m.div_ceil(8)), 256, &args[..7])
        }
    }
}

/// `out[i][r] = w[r] . a[i]` for each of the `m` rows of `a` (row-major, `w.cols()` values in
/// each row): the result of `hardshell::kernels::matvec_i8` for each row. The tensor-core
/// kernel uses the split-K factor of the model step. Cost: one upload, one product, one read
/// back.
///
/// # Errors
/// [`Error::LengthMismatch`] if `a.len() != m * w.cols()`; [`Error::Model`] if the rows are
/// longer than `MAX_DOT_LEN_I32`; [`Error::Cuda`] on a driver error.
pub fn matmul_i8(
    dev: &CudaDevice,
    w: &Matrix<i8>,
    a: &[i8],
    m: usize,
    kernel: MatKernel,
) -> Result<Vec<i32>> {
    let kw = u32_of(stride_words(w.cols()), "rows too long")?;
    let splits =
        splits_for(kernel, u32_of(w.rows(), "too many rows")?, kw, u32_of(m, "too many rows")?);
    matmul_i8_split(dev, w, a, m, kernel, splits)
}

/// As [`matmul_i8`], with `splits` partial sums over k (tensor cores only; 1 for dp4a). The
/// host adds the partials, as the output op does on the GPU. For tests of the split.
///
/// # Errors
/// The errors of [`matmul_i8`]; [`Error::Unsupported`] if `splits` is 0, more than 16, or more
/// than 1 for dp4a.
pub fn matmul_i8_split(
    dev: &CudaDevice,
    w: &Matrix<i8>,
    a: &[i8],
    m: usize,
    kernel: MatKernel,
    splits: u32,
) -> Result<Vec<i32>> {
    if splits == 0 || splits > 16 || (kernel == MatKernel::Dp4a && splits != 1) {
        return Err(Error::Unsupported { reason: "splits must be in 1..=16, and 1 for dp4a" });
    }
    let (n, k) = (w.rows(), w.cols());
    if a.len() != m * k {
        return Err(Error::LengthMismatch { expected: m * k, found: a.len() });
    }
    if k > hardshell::kernels::MAX_DOT_LEN_I32 {
        return Err(Error::Model(hardshell::Error::DotTooLong {
            len: k,
            max: hardshell::kernels::MAX_DOT_LEN_I32,
        }));
    }
    let kw = stride_words(k);
    let wb = dev.upload(&pad_rows(w.data(), n, k, 4 * kw))?;
    let ab = dev.upload(&pad_rows(a, m, k, 4 * kw))?;
    let parts = splits as usize;
    let ob = dev.zeroed(4 * m * n * parts)?;
    let n32 = u32_of(n, "too many rows")?;
    let mat = Mat {
        w: &wb,
        a: &ab,
        out: &ob,
        out_stride: n32,
        n: n32,
        kw: u32_of(kw, "rows too long")?,
        m: u32_of(m, "too many rows")?,
        splits,
    };
    launch_matmul(dev, kernel, &mat)?;
    let mut out = vec![0u8; 4 * m * n * parts];
    dev.ctx.read(&ob, &mut out)?;
    let vals = bytes_i32(&out);
    let mut sum = vec![0i64; m * n];
    for part in vals.chunks(m * n) {
        for (s, &v) in sum.iter_mut().zip(part) {
            *s += i64::from(v);
        }
    }
    // INVARIANT: each total is a dot product below 2^30 in magnitude.
    Ok(sum.into_iter().map(|v| i32::try_from(v).unwrap_or(i32::MAX)).collect())
}

/// `out[i][r] = w[r] . a[i]` for `m` rows of `i16` activations (`w.cols()` values in each row):
/// the result of `hardshell::kernels::matvec_i8_i16` for each row (ADR 0004 addendum B). The
/// GPU path of the model: the two i8 planes of each row (`plane_store`), one i8 product of `2m`
/// rows with `kernel` and `splits` partials, and the exact sum of `wide_dot` (the `wide_sum`
/// kernel). Cost: one upload, one product, one read back.
///
/// # Errors
/// The errors of [`matmul_i8_split`].
pub fn matmul_i8_i16(
    dev: &CudaDevice,
    w: &Matrix<i8>,
    a: &[i16],
    m: usize,
    kernel: MatKernel,
    splits: u32,
) -> Result<Vec<i64>> {
    if splits == 0 || splits > 16 || (kernel == MatKernel::Dp4a && splits != 1) {
        return Err(Error::Unsupported { reason: "splits must be in 1..=16, and 1 for dp4a" });
    }
    let (n, k) = (w.rows(), w.cols());
    if a.len() != m * k {
        return Err(Error::LengthMismatch { expected: m * k, found: a.len() });
    }
    if k > hardshell::kernels::MAX_DOT_LEN_I32 {
        return Err(Error::Model(hardshell::Error::DotTooLong {
            len: k,
            max: hardshell::kernels::MAX_DOT_LEN_I32,
        }));
    }
    let kw = stride_words(k);
    let wb = dev.upload(&pad_rows(w.data(), n, k, 4 * kw))?;
    let ab = dev.upload(&pad_rows(&planes_of(a, m, k), 2 * m, k, 4 * kw))?;
    let sums = dev.upload(&i32_bytes(&row_sums(w.data(), n, k)))?;
    let bias = dev.zeroed(4 * n.max(1))?;
    let ob = dev.zeroed(4 * 2 * m * n * splits as usize)?;
    let out = dev.zeroed(8 * m * n)?;
    let (n32, m32) = (u32_of(n, "too many rows")?, u32_of(m, "too many rows")?);
    let mat = Mat {
        w: &wb,
        a: &ab,
        out: &ob,
        out_stride: n32,
        n: n32,
        kw: u32_of(kw, "rows too long")?,
        m: 2 * m32,
        splits,
    };
    launch_matmul(dev, kernel, &mat)?;
    let args = [
        Arg::Ptr(ob.ptr()),
        Arg::Ptr(bias.ptr()),
        Arg::Ptr(sums.ptr()),
        Arg::Ptr(out.ptr()),
        Arg::U32(n32),
        Arg::U32(splits),
        Arg::U32(2 * m32 * n32),
    ];
    dev.launch(&dev.k.wide_sum, (n32.div_ceil(128), m32), 128, &args)?;
    let mut bytes = vec![0u8; 8 * m * n];
    dev.ctx.read(&out, &mut bytes)?;
    // NOTE: the u64 words are the two's complement of the i64 sums.
    Ok(bytes_u64(&bytes).into_iter().map(u64::cast_signed).collect())
}

/// The GPU time of `reps` products `out = a * w^T` (`w`: `n x k`, `a`: `m x k`, zero values) with
/// `kernel` and `splits` partials, in nanoseconds of the GPU timer (a `stamp` kernel before and
/// after; no host clock). For measurements of the tile and split choices.
///
/// # Errors
/// [`Error::Unsupported`] for a bad split (see [`matmul_i8_split`]) or a shape that does not fit
/// in `u32`; [`Error::Cuda`] on a driver error (for example, not enough memory).
pub fn matmul_gpu_ns(
    dev: &CudaDevice,
    (n, k, m): (usize, usize, usize),
    kernel: MatKernel,
    splits: u32,
    reps: u32,
) -> Result<u64> {
    if splits == 0 || splits > 16 || (kernel == MatKernel::Dp4a && splits != 1) {
        return Err(Error::Unsupported { reason: "splits must be in 1..=16, and 1 for dp4a" });
    }
    let kw = stride_words(k);
    let wb = dev.zeroed(4 * kw * crate::kernels::padded_rows(n))?;
    let ab = dev.zeroed(4 * kw * crate::kernels::padded_rows(m))?;
    let ob = dev.zeroed(4 * m * n * splits as usize)?;
    let stamps = dev.zeroed(16)?;
    let n32 = u32_of(n, "too many rows")?;
    let mat = Mat {
        w: &wb,
        a: &ab,
        out: &ob,
        out_stride: n32,
        n: n32,
        kw: u32_of(kw, "rows too long")?,
        m: u32_of(m, "too many rows")?,
        splits,
    };
    launch_matmul(dev, kernel, &mat)?;
    dev.launch(&dev.k.stamp, (1, 1), 1, &[Arg::Ptr(stamps.ptr()), Arg::U32(0)])?;
    for _ in 0..reps {
        launch_matmul(dev, kernel, &mat)?;
    }
    dev.launch(&dev.k.stamp, (1, 1), 1, &[Arg::Ptr(stamps.ptr()), Arg::U32(1)])?;
    let mut out = [0u8; 16];
    dev.ctx.read(&stamps, &mut out)?;
    let t = crate::device::bytes_u64(&out);
    Ok(t.get(1).copied().unwrap_or(0).saturating_sub(t.first().copied().unwrap_or(0)))
}

/// The GPU time of `reps` reads of a buffer of `bytes` bytes (a multiple of 16) with a simple
/// streaming kernel, in nanoseconds of the GPU timer: the memory bandwidth that a kernel can
/// reach. For measurements.
///
/// # Errors
/// [`Error::Cuda`] on a driver error (for example, not enough memory).
pub fn read_gpu_ns(dev: &CudaDevice, bytes: usize, reps: u32) -> Result<u64> {
    let buf = dev.zeroed(bytes)?;
    let (blocks, threads) = (30 * 16, 256u32);
    let out = dev.zeroed(4 * blocks * threads as usize)?;
    let stamps = dev.zeroed(16)?;
    let args = [Arg::Ptr(buf.ptr()), Arg::U64((bytes / 16) as u64), Arg::Ptr(out.ptr())];
    let grid = (u32_of(blocks, "too many blocks")?, 1);
    dev.launch(&dev.k.read_bandwidth, grid, threads, &args)?;
    dev.launch(&dev.k.stamp, (1, 1), 1, &[Arg::Ptr(stamps.ptr()), Arg::U32(0)])?;
    for _ in 0..reps {
        dev.launch(&dev.k.read_bandwidth, grid, threads, &args)?;
    }
    dev.launch(&dev.k.stamp, (1, 1), 1, &[Arg::Ptr(stamps.ptr()), Arg::U32(1)])?;
    let mut t = [0u8; 16];
    dev.ctx.read(&stamps, &mut t)?;
    let t = crate::device::bytes_u64(&t);
    Ok(t.get(1).copied().unwrap_or(0).saturating_sub(t.first().copied().unwrap_or(0)))
}
