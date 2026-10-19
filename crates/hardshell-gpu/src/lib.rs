//! HardShell on a GPU: the integer GPT-2 of `hardshell` in portable WGSL compute shaders.
//!
//! The GPU path gives the same weights as `hardshell::gpt2::Session`, bit for bit (CLAUDE.md §1
//! rule 10). It runs through wgpu on Vulkan, Metal, DX12 or OpenGL, so a user needs only the
//! GPU driver (ADR 0008). WGSL has only 32-bit integers; the shaders emulate the 64-bit and
//! 128-bit rules of `hardshell::int` with 32-bit words, whose wrapping arithmetic WGSL defines.
//!
//! The kernels are one for each op of the op set (`docs/design/anymodel.md` §4): `embed`,
//! `norm`, `linear`, `add`, `table`, `attention` and `logits`. The host composes them into the
//! GPT-2 step.
//!
//! The main types:
//! - [`GpuDevice`]: an adapter and its device.
//! - [`GpuModel`]: a GPT-2 model file in GPU memory.
//! - [`GpuBatch`]: many independent sessions that step together, so one read of the weights
//!   serves many tokens. A batch has a context overlap `K` (ADR 0015), as the CPU paths.
//! - [`GpuSession`]: one session, with the API shape of `hardshell::gpt2::Session`.
//! - [`GpuOps`]: single ops, for tests and measurements.
#![deny(clippy::float_arithmetic)]

mod batch;
mod block_on;
mod device;
mod error;
mod kernels;
mod model;
mod ops;
mod session;

pub use batch::{CoderInterval, CoderQuery, GpuBatch};
pub use device::{AdapterChoice, AdapterSummary, GpuDevice, adapters};
pub use error::{Error, Result};
pub use kernels::SOURCES as SHADER_SOURCES;
pub use model::GpuModel;
pub use ops::{GpuOps, IntOp, MatKernel};
pub use session::GpuSession;
