//! HardShell: the integer-only inference engine of Taco, and the reader and writer of the
//! `.hsm` model file.
//!
//! HardShell changes a context of token ids into one non-negative `u64` weight for each token
//! of the vocabulary. The `taco-coder` crate changes the weights into a frequency table. All
//! arithmetic is integer arithmetic, so every machine calculates the same weights, bit for bit.
//!
//! PLAN(§7): the numerics. PLAN(§8): the model file.
//!
//! This is a kernel crate: no floats, no clock, no randomness, no global state, and no I/O
//! except reads from a byte slice that the caller gives.
#![deny(clippy::float_arithmetic)]

pub mod error;
pub mod gelu;
pub mod int;
pub mod kernels;
pub mod layernorm;
pub mod logits;
pub mod softmax;
pub mod tensor;
pub mod types;

pub use error::{Error, Result};
pub use types::{ModelHash, TokenId};
