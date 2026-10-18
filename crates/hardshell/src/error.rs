//! The error type of HardShell.
//!
//! Errors carry names, offsets, lengths and ids. They never carry the data that Taco
//! compresses.

/// Everything that can fail in HardShell. Each variant is one thing that a caller can act on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A tensor shape is not valid: no dimensions, too many dimensions, a zero dimension, or
    /// an element count that does not fit in memory.
    #[error("invalid tensor shape: {reason}")]
    InvalidShape {
        /// What is wrong.
        reason: &'static str,
    },
    /// A buffer has a different length than its shape or the operation needs.
    #[error("length mismatch: expected {expected}, found {found}")]
    LengthMismatch {
        /// The length that the shape or the operation needs.
        expected: usize,
        /// The length of the buffer.
        found: usize,
    },
    /// A dot product is longer than the bound that keeps its accumulator from overflow.
    #[error("dot product of length {len} is longer than the accumulator bound {max}")]
    DotTooLong {
        /// The length of the dot product.
        len: usize,
        /// The largest length that the accumulator permits.
        max: usize,
    },
    /// The file does not start with the `.hsm` magic bytes.
    #[error("the file does not start with the .hsm magic bytes")]
    BadMagic,
    /// The file has a format version that this reader does not know.
    #[error("unsupported .hsm version {version}")]
    UnsupportedVersion {
        /// The version in the file.
        version: u32,
    },
    /// A part of the file is outside the file.
    #[error("truncated file: {needed} bytes needed at offset {offset}, the file has {file_len}")]
    Truncated {
        /// The start of the part.
        offset: u64,
        /// The length of the part.
        needed: u64,
        /// The length of the file.
        file_len: u64,
    },
    /// The JSON header is not valid UTF-8.
    #[error("the .hsm header is not valid UTF-8")]
    HeaderNotUtf8,
    /// The JSON header does not parse into the schema. This includes a number with a fraction
    /// or an exponent, an unknown field, a missing field and a duplicate field.
    #[error("the .hsm header JSON is not valid at line {line}, column {column}: {message}")]
    HeaderJson {
        /// The line of the error in the header.
        line: usize,
        /// The column of the error in the header.
        column: usize,
        /// The parser message.
        message: String,
    },
    /// A header field has a value that is not permitted.
    #[error("invalid header field `{field}`: {reason}")]
    InvalidHeader {
        /// The field or the tensor name.
        field: String,
        /// What is wrong.
        reason: &'static str,
    },
    /// A section is not at the position that the canonical layout gives.
    #[error("invalid layout of `{name}`: {reason}")]
    Layout {
        /// The section or the tensor name.
        name: String,
        /// What is wrong.
        reason: &'static str,
    },
    /// A padding byte between sections is not zero.
    #[error("non-zero padding byte at offset {offset}")]
    NonZeroPadding {
        /// The offset of the first such byte.
        offset: u64,
    },
    /// The vocabulary section is not valid.
    #[error("invalid vocabulary at token {token}: {reason}")]
    Vocabulary {
        /// The token id where the problem starts.
        token: u32,
        /// What is wrong.
        reason: &'static str,
    },
    /// The model needs a tensor that the file does not have.
    #[error("missing tensor `{name}`")]
    MissingTensor {
        /// The tensor name.
        name: String,
    },
    /// The file has a tensor that the model does not use, or the same name two times.
    #[error("unexpected tensor `{name}`: {reason}")]
    UnexpectedTensor {
        /// The tensor name.
        name: String,
        /// What is wrong.
        reason: &'static str,
    },
    /// A tensor has a type, a shape, a rescale shift or values that the model does not permit.
    #[error("tensor `{name}` does not match the model: {reason}")]
    TensorMismatch {
        /// The tensor name.
        name: String,
        /// What is wrong.
        reason: &'static str,
    },
    /// A token id is not in the vocabulary.
    #[error("token id {token} is not less than the vocabulary size {vocab_size}")]
    TokenOutOfRange {
        /// The token id.
        token: u32,
        /// The vocabulary size.
        vocab_size: u32,
    },
    /// The writer cannot make the file, because a part is too large for the format.
    #[error("cannot write the .hsm file: {reason}")]
    TooLarge {
        /// What is too large.
        reason: &'static str,
    },
}

/// The result type of HardShell.
pub type Result<T> = core::result::Result<T, Error>;
