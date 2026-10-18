//! Single ops on the GPU, for tests and measurements: the integer helpers and the i8 matrix
//! product. The GPT-2 step (`batch`) uses the same kernels.

use hardshell::tensor::Matrix;

use crate::device::{GpuDevice, bytes_to_words, words_to_bytes};
use crate::error::{Error, Result};
use crate::kernels::{self, GEMM_TILE, MATVEC_MAX_ROWS, pack_i8, stride_words};

/// The kernel of a matrix product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatKernel {
    /// One pass over the weights for each 8 rows of activations (for small batches).
    Matvec,
    /// 64 x 64 tiles (for large batches).
    Gemm,
}

impl MatKernel {
    /// The kernel that a product with `rows` activation rows uses.
    #[must_use]
    pub fn for_rows(rows: usize) -> Self {
        if rows <= MATVEC_MAX_ROWS as usize { Self::Matvec } else { Self::Gemm }
    }
}

/// An integer helper of `int.wgsl`, for the self-test of a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntOp {
    /// `hardshell::int::rescale(x, m, s)`. Input words: x (2), m, s. Output: i128 (4).
    Rescale = 0,
    /// `floor_div(a, n)` for `1 <= n <= 2^16`. Input: a (2), n. Output: i64 (2).
    FloorDivSmall = 1,
    /// `isqrt_u128(x)`. Input: x (4). Output: u128 (4).
    Isqrt = 2,
    /// `n / d` for u64 values, `d >= 1`. Input: n (2), d (2). Output: u64 (2).
    DivU64 = 3,
    /// `clamp_wide(x, lo, hi)`. Input: x (4), lo (2), hi (2). Output: i64 (2).
    ClampWide = 4,
    /// The full product of two u64 values. Input: a (2), b (2). Output: u128 (4).
    MulU64 = 5,
    /// The rounding of `rescale_wide`: `shift_round(p, s, 126)`. Input: p (4), s. Output: i128.
    ShiftRound = 6,
    /// `x / n` for a u128 below 2^96 and `1 <= n <= 2^16`. Input: x (4), n. Output: u128.
    DivSmallU128 = 7,
    /// The packed dot product of four signed bytes. Input: a, b. Output: i32 (1).
    Dot4 = 8,
}

/// Single ops on one device.
pub struct GpuOps<'d> {
    dev: &'d GpuDevice,
    matvec: wgpu::ComputePipeline,
    gemm: wgpu::ComputePipeline,
    selftest: wgpu::ComputePipeline,
}

impl std::fmt::Debug for GpuOps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuOps").field("device", self.dev).finish_non_exhaustive()
    }
}

/// The bind groups of one matrix product: one for each kernel, because each pipeline has its
/// own layout.
pub(crate) struct MmGroup {
    pub(crate) matvec: wgpu::BindGroup,
    pub(crate) gemm: wgpu::BindGroup,
}

impl MmGroup {
    /// Binds `buffers` for both kernels.
    pub(crate) fn new(
        dev: &GpuDevice,
        pipelines: (&wgpu::ComputePipeline, &wgpu::ComputePipeline),
        buffers: &[(u32, &wgpu::Buffer)],
    ) -> Self {
        Self { matvec: dev.bind(pipelines.0, buffers), gemm: dev.bind(pipelines.1, buffers) }
    }
}

/// Records one matrix product into `pass`. `n` is the number of rows of `w`, `m` the number of
/// rows of `a`.
pub(crate) fn record_matmul(
    pass: &mut wgpu::ComputePass<'_>,
    pipelines: (&wgpu::ComputePipeline, &wgpu::ComputePipeline),
    group: &MmGroup,
    kernel: MatKernel,
    n: u32,
    m: u32,
) {
    let (matvec, gemm) = pipelines;
    match kernel {
        MatKernel::Matvec => {
            pass.set_pipeline(matvec);
            pass.set_bind_group(0, &group.matvec, &[]);
            pass.dispatch_workgroups(n.div_ceil(MATVEC_MAX_ROWS), m.div_ceil(MATVEC_MAX_ROWS), 1);
        }
        MatKernel::Gemm => {
            pass.set_pipeline(gemm);
            pass.set_bind_group(0, &group.gemm, &[]);
            pass.dispatch_workgroups(n.div_ceil(GEMM_TILE), m.div_ceil(GEMM_TILE), 1);
        }
    }
}

/// Checks that a grid of `groups` workgroups fits in one dimension of the device.
pub(crate) fn check_grid(dev: &GpuDevice, groups: u64) -> Result<()> {
    let limit = u64::from(dev.max_workgroups());
    if groups > limit {
        return Err(Error::DeviceLimit { what: "a dispatch", needed: groups, limit });
    }
    Ok(())
}

pub(crate) fn u32_of(x: usize, what: &'static str) -> Result<u32> {
    u32::try_from(x).map_err(|_| Error::DeviceLimit {
        what,
        needed: x as u64,
        limit: u64::from(u32::MAX),
    })
}

impl<'d> GpuOps<'d> {
    /// Compiles the kernels of this module for `dev`.
    ///
    /// # Errors
    /// [`Error::Gpu`] if the device refuses a shader.
    pub fn new(dev: &'d GpuDevice) -> Result<Self> {
        let mm = kernels::pipeline(dev, kernels::MATMUL, "matvec")?;
        let gm = kernels::pipeline(dev, kernels::MATMUL, "gemm")?;
        let st = kernels::pipeline(dev, kernels::SELFTEST, "selftest")?;
        Ok(Self { dev, matvec: mm, gemm: gm, selftest: st })
    }

    /// `out[i][r] = w[r] . a[i]` for each of the `m` rows of `a` (row-major, `w.cols()` values in
    /// each row): the result of `hardshell::kernels::matvec_i8` for each row. `kernel` selects
    /// the kernel; `None` selects it from `m`. Cost: one upload, one product, one read back.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `a.len() != m * w.cols()`; [`Error::DeviceLimit`] if a
    /// buffer or the grid is too large; [`Error::Gpu`] on a GPU error.
    pub fn matmul_i8(
        &self,
        w: &Matrix<i8>,
        a: &[i8],
        m: usize,
        kernel: Option<MatKernel>,
    ) -> Result<Vec<i32>> {
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
        let kernel = kernel.unwrap_or_else(|| MatKernel::for_rows(m));
        let kw = stride_words(k);
        let (n32, m32, kw32) = (u32_of(n, "rows")?, u32_of(m, "rows")?, u32_of(kw, "columns")?);
        check_grid(self.dev, u64::from(n32.div_ceil(MATVEC_MAX_ROWS)))?;
        check_grid(self.dev, u64::from(m32.div_ceil(MATVEC_MAX_ROWS)))?;
        let dev = self.dev;
        let wb = dev.storage("w", &words_to_bytes(&pack_i8(w.data(), n, k, kw)))?;
        let ab = dev.storage("a", &words_to_bytes(&pack_i8(a, m, k, kw)))?;
        let out_bytes = (m * n * 4) as u64;
        let ob = dev.zeroed("out", out_bytes)?;
        let params = dev.uniform("mat", &[n32, kw32, n32, m32, 0, 0, 0, 0]);
        let rows = dev.storage("rows", &[0u8; 32])?;
        let group = MmGroup::new(
            dev,
            (&self.matvec, &self.gemm),
            &[(0, &wb), (1, &ab), (2, &ob), (3, &params), (4, &rows)],
        );
        let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            record_matmul(&mut pass, (&self.matvec, &self.gemm), &group, kernel, n32, m32);
        }
        let staging = dev.staging("out read", out_bytes);
        let mut bytes = Vec::new();
        dev.finish_and_read(enc, &ob, &staging, out_bytes, &mut bytes)?;
        Ok(bytes_to_words(&bytes).into_iter().map(|w| w.cast_signed()).collect())
    }

    /// Runs `op` on each case (8 input words) and gives 4 output words for each case. The
    /// tests compare them with `hardshell::int`.
    ///
    /// # Errors
    /// [`Error::DeviceLimit`] if there are too many cases; [`Error::Gpu`] on a GPU error.
    pub fn int_op(&self, op: IntOp, cases: &[[u32; 8]]) -> Result<Vec<[u32; 4]>> {
        let dev = self.dev;
        let count = u32_of(cases.len(), "cases")?;
        check_grid(dev, u64::from(count.div_ceil(64)))?;
        let input: Vec<u32> = cases.iter().flatten().copied().collect();
        let ib = dev.storage("in", &words_to_bytes(&input))?;
        let out_bytes = (cases.len() * 16) as u64;
        let ob = dev.zeroed("out", out_bytes)?;
        let params = dev.uniform("test", &[op as u32, count, 0, 0]);
        let group = dev.bind(&self.selftest, &[(0, &ib), (1, &ob), (2, &params)]);
        let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.selftest);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        let staging = dev.staging("out read", out_bytes);
        let mut bytes = Vec::new();
        dev.finish_and_read(enc, &ob, &staging, out_bytes, &mut bytes)?;
        let words = bytes_to_words(&bytes);
        Ok(words.as_chunks::<4>().0.to_vec())
    }
}
