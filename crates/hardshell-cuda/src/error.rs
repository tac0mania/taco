//! The errors of the CUDA backend.

/// A result with the error of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// An error of the CUDA backend. Each variant is a thing that a caller can act on.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No CUDA driver or no CUDA device. The caller can use another backend.
    #[error("no CUDA device: {reason}")]
    NoDevice {
        /// What is missing.
        reason: String,
    },
    /// A driver call failed.
    #[error("{call} failed: {name} ({code})")]
    Cuda {
        /// The driver function or the kernel.
        call: &'static str,
        /// The `CUresult` code.
        code: i32,
        /// The name of the code.
        name: String,
    },
    /// The device, the model or the request has a shape that the kernels do not support.
    #[error("not supported: {reason}")]
    Unsupported {
        /// Why.
        reason: &'static str,
    },
    /// The model file or a token is not valid (an error of the CPU reference crate).
    #[error(transparent)]
    Model(#[from] hardshell::Error),
    /// A slot index is not in the batch.
    #[error("slot {slot} is not valid in a batch of {slots} slots")]
    BadSlot {
        /// The slot.
        slot: usize,
        /// The number of slots in the batch.
        slots: usize,
    },
    /// The lengths of two arguments do not agree.
    #[error("length mismatch: expected {expected}, found {found}")]
    LengthMismatch {
        /// The length that was expected.
        expected: usize,
        /// The length that was found.
        found: usize,
    },
}
