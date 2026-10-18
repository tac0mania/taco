//! HardShell on NVIDIA GPUs: the integer GPT-2 of `hardshell` with int8 tensor cores.
//!
//! The CUDA path gives the same weights as `hardshell::gpt2::Session`, bit for bit (CLAUDE.md
//! §1 rule 10). The kernels are CUDA C, compiled to PTX by `tools/cuda/build_ptx.py` before
//! publication and embedded in this crate; at run time the GPU driver (`libcuda`) loads them.
//! A user needs only the NVIDIA driver (ADR 0008). On a machine without the driver, every
//! constructor gives [`Error::NoDevice`].
//!
//! The kernels are one for each op of the op set (`docs/design/anymodel.md` §4): `embed`,
//! `norm`, `linear` (int8 tensor cores or dp4a, then the output rescale), `add`, `table`,
//! `attention`, `logits`, and the coder interval. The host composes them into the GPT-2 step.
//!
//! The main types:
//! - [`CudaDevice`]: one GPU with the kernels loaded.
//! - [`CudaModel`]: a GPT-2 model file in GPU memory.
//! - [`CudaBatch`]: many independent sessions, and prefill of many positions in one pass.
//! - [`CudaSession`]: one session, with the API shape of `hardshell::gpt2::Session`.
#![deny(clippy::float_arithmetic)]

mod batch;
mod device;
mod driver;
mod error;
mod kernels;
mod model;
mod ops;
mod session;

pub use batch::{CoderInterval, CoderQuery, CudaBatch};
pub use device::CudaDevice;
pub use error::{Error, Result};
pub use kernels::{PTX_SOURCE_LINE, SOURCE as KERNEL_SOURCE, ptx};
pub use model::CudaModel;
pub use ops::{MatKernel, matmul_gpu_ns, matmul_i8, matmul_i8_split, read_gpu_ns};
pub use session::CudaSession;
