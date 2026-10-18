//! A CUDA device with the HardShell kernels loaded.

use crate::driver::{Arg, Context, DevBuf, Kernel};
use crate::error::{Error, Result};
use crate::kernels::{Kernels, PTX};

/// One NVIDIA GPU, its primary context, and the HardShell module.
///
/// The value uses the context of the calling thread, so it is not `Send`: use it on one thread.
pub struct CudaDevice {
    pub(crate) ctx: Context,
    pub(crate) k: Kernels,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice").field("name", &self.ctx.name()).finish_non_exhaustive()
    }
}

impl CudaDevice {
    /// Opens CUDA device `ordinal` and loads the kernels. The driver compiles the embedded PTX
    /// for the GPU (some hundred milliseconds the first time; the driver keeps a cache).
    ///
    /// # Errors
    /// [`Error::NoDevice`] if there is no driver or no such device; [`Error::Unsupported`] if
    /// the compute capability is below 8.0 (the kernels use int8 `mma.sync` m16n8k32);
    /// [`Error::Cuda`] if a driver call fails.
    pub fn open(ordinal: usize) -> Result<Self> {
        let ctx = Context::open(ordinal, PTX)?;
        if ctx.compute_capability().0 < 8 {
            return Err(Error::Unsupported {
                reason: "the GPU needs compute capability 8.0 or more",
            });
        }
        let k = Kernels::new(&ctx)?;
        Ok(Self { ctx, k })
    }

    /// The device name, for example "NVIDIA GeForce RTX 3060 Laptop GPU".
    #[must_use]
    pub fn name(&self) -> &str {
        self.ctx.name()
    }

    /// The compute capability (major, minor).
    #[must_use]
    pub fn compute_capability(&self) -> (i32, i32) {
        self.ctx.compute_capability()
    }

    /// Free and total device memory, in bytes.
    ///
    /// # Errors
    /// [`Error::Cuda`] if the driver call fails.
    pub fn memory(&self) -> Result<(usize, usize)> {
        self.ctx.mem_info()
    }

    pub(crate) fn launch(
        &self,
        k: &Kernel,
        grid: (u32, u32),
        block: u32,
        args: &[Arg],
    ) -> Result<()> {
        self.ctx.launch(k, (grid.0, grid.1, 1), (block, 1, 1), args)
    }

    pub(crate) fn zeroed(&self, bytes: usize) -> Result<DevBuf> {
        self.ctx.zeroed(bytes)
    }

    pub(crate) fn upload(&self, bytes: &[u8]) -> Result<DevBuf> {
        self.ctx.upload(bytes, bytes.len())
    }
}

/// Little-endian bytes of i32 values.
pub(crate) fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Little-endian bytes of u32 values.
pub(crate) fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Little-endian bytes of u64 values.
pub(crate) fn u64_bytes(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// u64 values of little-endian bytes.
pub(crate) fn bytes_u64(b: &[u8]) -> Vec<u64> {
    b.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect()
}

/// i32 values of little-endian bytes.
pub(crate) fn bytes_i32(b: &[u8]) -> Vec<i32> {
    b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect()
}

/// `x` as u32, or an error that names `what`.
pub(crate) fn u32_of(x: usize, what: &'static str) -> Result<u32> {
    u32::try_from(x).map_err(|_| Error::Unsupported { reason: what })
}
