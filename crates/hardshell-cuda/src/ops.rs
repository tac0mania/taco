//! Single ops on the GPU, for tests and measurements: the i8 matrix product with each kernel.

use hardshell::tensor::Matrix;

use crate::device::{CudaDevice, bytes_i32, u32_of};
use crate::driver::{Arg, DevBuf};
use crate::error::{Error, Result};
use crate::kernels::{pad_rows, stride_words};

/// The kernel of a matrix product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatKernel {
    /// int8 tensor cores (`mma.sync` m16n8k32), 64 x 64 tiles in shared memory, split-K.
    TensorCore,
    /// `dp4a`, one warp for each weight row, up to 8 activation rows in one pass.
    Dp4a,
}

impl MatKernel {
    /// The kernel that a product with `rows` activation rows uses.
    #[must_use]
    pub fn for_rows(rows: usize) -> Self {
        if rows <= 8 { Self::Dp4a } else { Self::TensorCore }
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

/// The number of blocks that keeps the GPU busy (est.: 4 blocks on each of 30 SMs).
const TARGET_BLOCKS: u32 = 120;

/// The split-K factor for a product: enough blocks for the GPU, at most one stage (64 bytes of
/// k) for each split, and at most 16 splits.
pub(crate) fn splits_for(n: u32, kw: u32, m: u32) -> u32 {
    let blocks = m.div_ceil(64).max(1) * n.div_ceil(64).max(1);
    TARGET_BLOCKS.div_ceil(blocks).clamp(1, (kw / 16).clamp(1, 16))
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
    match kernel {
        MatKernel::TensorCore => dev.ctx.launch(
            &dev.k.gemm_tiled,
            (m.div_ceil(64), n.div_ceil(64), p.splits.max(1)),
            (128, 1, 1),
            &args,
        ),
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
    let splits = match kernel {
        MatKernel::TensorCore => {
            splits_for(u32_of(w.rows(), "too many rows")?, kw, u32_of(m, "too many rows")?)
        }
        MatKernel::Dp4a => 1,
    };
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
