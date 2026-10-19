//! The error type of the library: one variant for each fault that a caller can act on.
//!
//! Errors carry offsets, lengths, block indices, ids and hashes. They never carry the data
//! that Taco compresses.

use taco_format::{Checksum, ModelHash};

use crate::backend::Backend;
use crate::tokenizer::MergeFault;

/// A fault in `fold`, `unfold` or the tokenizer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TacoError {
    /// HardShell refused the model file or a token.
    #[error("model: {0}")]
    Model(#[from] hardshell::Error),
    /// The range coder refused a table or a symbol. This is an internal fault.
    #[error("coder: {0}")]
    Coder(#[from] taco_coder::CoderError),
    /// The match model or the mixer refused its input (ADR 0010). This is an internal fault.
    #[error("mixer: {0}")]
    Match(#[from] taco_match::MatchError),
    /// The container is not valid, or the writer refused the blocks.
    #[error("container: {0}")]
    Format(#[from] taco_format::FormatError),
    /// The container needs a model, and the caller gave none.
    #[error("the container needs the model {container}, and no model was given")]
    ModelRequired {
        /// The model hash in the container.
        container: ModelHash,
    },
    /// The container needs a different model than the loaded model (plan §3 rule 3).
    #[error("the container needs the model {container}, the loaded model is {loaded}")]
    ModelHashMismatch {
        /// The model hash in the container.
        container: ModelHash,
        /// The hash of the loaded model.
        loaded: ModelHash,
    },
    /// The restored data does not have the checksum in the header (plan §3 rule 4).
    #[error("checksum mismatch: the header has {expected}, the restored data has {found}")]
    ChecksumMismatch {
        /// The checksum in the header.
        expected: Checksum,
        /// The checksum of the restored data.
        found: Checksum,
    },
    /// A coded or mixed block does not decode to its original length. The block is damaged.
    #[error("coded block {block} is damaged: {reason}")]
    CorruptBlock {
        /// The index of the block.
        block: u32,
        /// What is wrong.
        reason: &'static str,
    },
    /// The original length in the header is more than the limit that the caller set.
    #[error("the container restores {original_len} bytes, more than the limit {limit}")]
    OutputTooLarge {
        /// The original length in the header.
        original_len: u64,
        /// The limit of the caller.
        limit: u64,
    },
    /// The tokenizer was made for a vocabulary of a different size than the model.
    #[error("the tokenizer has {tokenizer} tokens, the model has {model}")]
    TokenizerMismatch {
        /// The vocabulary size of the tokenizer.
        tokenizer: usize,
        /// The vocabulary size of the model.
        model: usize,
    },
    /// The tokens of a block do not give the bytes of the block with the vocabulary of the
    /// model. The tokenizer and the model do not agree. Nothing was written.
    #[error("block {block}: the tokens do not restore the block with the model vocabulary")]
    TokenizerRoundTrip {
        /// The index of the block.
        block: u32,
    },
    /// A byte value has no single-byte token, or more than one.
    #[error("the vocabulary has no single token for the byte {byte:#04x}")]
    MissingByteToken {
        /// The byte value.
        byte: u8,
    },
    /// A BPE merge is not valid.
    #[error("merge {rank} is not valid: {fault}")]
    InvalidMerge {
        /// The rank of the merge (its index in the list).
        rank: u32,
        /// What is wrong.
        fault: MergeFault,
    },
    /// The vocabulary or the merge list has more than `u32::MAX` entries.
    #[error("the vocabulary or the merge list has more than u32::MAX entries")]
    VocabularyTooLarge,
    /// The input needs more blocks than the container can hold, or a length does not fit.
    #[error("the input is too large: {reason}")]
    InputTooLarge {
        /// What is too large.
        reason: &'static str,
    },
    /// A GPU backend failed: the driver refused a call, the device has not enough memory, or
    /// the device does not give the bits of the CPU (its self-test failed). The caller can use
    /// another backend (ADR 0016).
    #[error("backend {backend}: {reason}")]
    Backend {
        /// The backend.
        backend: Backend,
        /// What failed, as the driver or the self-test says it.
        reason: String,
    },
    /// A worker thread stopped without a result.
    #[error("a worker thread stopped without a result")]
    WorkerFailed,
}

/// The result type of the library.
pub type Result<T> = core::result::Result<T, TacoError>;
