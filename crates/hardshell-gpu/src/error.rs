//! The errors of the GPU backend.

/// A result with the error of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// An error of the GPU backend. Each variant is a thing that a caller can act on.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No GPU adapter agrees with the choice. The caller can use the CPU reference path.
    #[error("no GPU adapter found ({choice})")]
    NoAdapter {
        /// The choice that found nothing.
        choice: String,
    },
    /// The adapter refused to make a device.
    #[error("the GPU adapter refused the device: {reason}")]
    Device {
        /// The message of the driver or of wgpu.
        reason: String,
    },
    /// The GPU reported an error (validation, memory or internal) during the work.
    #[error("the GPU reported an error: {reason}")]
    Gpu {
        /// The message of the driver or of wgpu.
        reason: String,
    },
    /// The model or the batch needs more than a limit of the device.
    #[error("{what} needs {needed} but the device permits {limit}")]
    DeviceLimit {
        /// The thing that is too large.
        what: &'static str,
        /// The size that is necessary.
        needed: u64,
        /// The limit of the device.
        limit: u64,
    },
    /// The model has a shape that these kernels do not support.
    #[error("the GPU kernels do not support this model: {reason}")]
    Unsupported {
        /// Why.
        reason: &'static str,
    },
    /// The model file or a token is not valid (an error of the CPU reference crate).
    #[error(transparent)]
    Model(#[from] hardshell::Error),
    /// A slot index is not in the batch, or occurs two times in one step.
    #[error("slot {slot} is not valid in a batch of {slots} slots, or occurs two times")]
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
