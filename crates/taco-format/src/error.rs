//! The error type of the crate: one variant for each fault that a caller can act on.
//!
//! Errors carry offsets, lengths, block indices and hashes. They never carry the data that
//! Taco compresses.

use crate::types::{Checksum, ModelHash};

/// A fault in a container, or in the parts that a caller gives to the writer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FormatError {
    /// The input does not start with the magic bytes `"TACO"`.
    #[error("the input does not start with the .taco magic bytes")]
    BadMagic,
    /// The container has a format version that this reader does not know.
    #[error("unsupported .taco version {version}")]
    UnsupportedVersion {
        /// The version in the container.
        version: u16,
    },
    /// The flags field has a bit that the version of the container does not define. Versions 1
    /// and 2 define no flag.
    #[error("unknown .taco flags {flags:#06x}")]
    UnknownFlags {
        /// The flags in the container.
        flags: u16,
    },
    /// The input ends before the end of a fixed part (the header or the block table).
    #[error("truncated {part}: {needed} bytes needed, the input has {found}")]
    Truncated {
        /// The part that is not complete.
        part: &'static str,
        /// The length of the input that the part needs.
        needed: u64,
        /// The length of the input.
        found: u64,
    },
    /// The input does not end at the end of the last block: bytes are missing or extra.
    #[error(
        "the container length is {found} bytes, the header and the block table give {expected}"
    )]
    WrongTotalLength {
        /// The length that the header and the block table give.
        expected: u64,
        /// The length of the input.
        found: u64,
    },
    /// The block size is 0.
    #[error("the block size is 0")]
    ZeroBlockSize,
    /// The block size is more than [`crate::MAX_BLOCK_SIZE`] (64 MiB), the largest block size
    /// of versions 1 and 2 (ADR 0011, addendum "Version 2 is frozen", §4).
    #[error("the block size {block_size} is more than the maximum {max}")]
    BlockSizeTooLarge {
        /// The block size of the header or of the writer.
        block_size: u32,
        /// The maximum.
        max: u32,
    },
    /// The block count is not `ceil(original length / block size)`.
    #[error(
        "{found} blocks for {original_len} bytes in blocks of {block_size}: expected {expected}"
    )]
    WrongBlockCount {
        /// The original length.
        original_len: u64,
        /// The block size.
        block_size: u32,
        /// `ceil(original_len / block_size)`.
        expected: u64,
        /// The block count in the container, or the number of blocks that the caller gave.
        found: u64,
    },
    /// The data needs more blocks than a `u32` block count can hold.
    #[error("{original_len} bytes in blocks of {block_size} need {blocks} blocks, more than u32")]
    TooManyBlocks {
        /// The original length.
        original_len: u64,
        /// The block size.
        block_size: u32,
        /// The number of blocks that the data needs.
        blocks: u64,
    },
    /// A kind byte in the block table is not a kind of the version of the container: 0 (raw)
    /// or 1 (coded) in version 1, and also 2 (zstd) and 3 (mixed) in version 2.
    #[error("block {block} has the unknown kind {kind}")]
    UnknownBlockKind {
        /// The index of the block.
        block: u32,
        /// The kind byte.
        kind: u8,
    },
    /// A raw block does not store exactly its original length.
    #[error("raw block {block} stores {stored_len} bytes, its original length is {original_len}")]
    RawLengthMismatch {
        /// The index of the block.
        block: u32,
        /// The original length of the block.
        original_len: u32,
        /// The stored length of the block.
        stored_len: u64,
    },
    /// A coded block or a mixed block is not smaller than its original bytes. Plan §3 rule 6
    /// says that such a block is stored raw.
    #[error(
        "coded block {block} stores {stored_len} bytes, not less than its original {original_len}"
    )]
    CodedNotSmaller {
        /// The index of the block.
        block: u32,
        /// The original length of the block.
        original_len: u32,
        /// The stored length of the block.
        stored_len: u64,
    },
    /// A zstd block is not smaller than its original bytes. Such a block is stored raw.
    #[error(
        "zstd block {block} stores {stored_len} bytes, not less than its original {original_len}"
    )]
    ZstdNotSmaller {
        /// The index of the block.
        block: u32,
        /// The original length of the block.
        original_len: u32,
        /// The stored length of the block.
        stored_len: u64,
    },
    /// The stored bytes of a zstd block do not obey ADR 0011, or the zstd decoder refused
    /// them. The reason is a fixed text or the error name of the zstd library.
    #[error("zstd block {block} is not valid: {reason}")]
    CorruptZstdBlock {
        /// The index of the block.
        block: u32,
        /// Why the block is not valid.
        reason: &'static str,
    },
    /// The zstd library failed for a reason that is not the data, for example no memory.
    #[error("the zstd library failed: {reason}")]
    Zstd {
        /// The error name of the zstd library, or a fixed text.
        reason: &'static str,
    },
    /// A zstd function received a block of another kind.
    #[error("block {block} is not a zstd block")]
    NotZstd {
        /// The index of the block.
        block: u32,
    },
    /// The profile byte of a version 2 header is not 0 (text) or 1 (logs).
    #[error("unknown .taco profile {profile}")]
    UnknownProfile {
        /// The profile byte.
        profile: u8,
    },
    /// The coding parts field of a version 2 header has a bit that this reader does not
    /// define. ADR 0010 defines bit 0 (the match model and the mixer) and ADR 0015 defines bit
    /// 1 (the context overlap). The other bits are reserved.
    #[error("unknown .taco coding parts {parts:#06x}")]
    UnknownCodingParts {
        /// The coding parts field.
        parts: u16,
    },
    /// The coding parts have the match model and the mixer, and byte 0 of the coding
    /// parameters is not a mixer rule version that this reader knows (ADR 0010: version 1).
    #[error("unknown mixer rule version {rule}")]
    UnknownMixerRule {
        /// The rule version byte.
        rule: u8,
    },
    /// The coding parts of the header are not permitted with its profile. The profile `text`
    /// has no match model and mixer (ADR 0012).
    #[error("the profile {profile} does not permit the coding parts {parts:#06x}")]
    CodingPartsNotInProfile {
        /// The profile byte.
        profile: u8,
        /// The coding parts field.
        parts: u16,
    },
    /// The coding parts have the context overlap (bit 1), and `K` in bytes 2 and 3 of the
    /// coding parameters is 0. A container without an overlap clears bit 1 (ADR 0015 §4).
    #[error("the coding parts have the context overlap, and its K is 0")]
    ZeroContextOverlap,
    /// A mixed block (kind 3) is in a container whose coding parts do not have the match model
    /// and the mixer, or the writer received a mixed block for such a container.
    #[error("block {block} is mixed, but the container has no match model and mixer")]
    MixedWithoutCodingPart {
        /// The index of the block.
        block: u32,
    },
    /// A reserved field of a version 2 header is not zero.
    #[error("the reserved field {field} of the header is not zero")]
    ReservedNotZero {
        /// The name of the field.
        field: &'static str,
    },
    /// A container in the raw-only mode (model hash all zero) has a coded block or a mixed
    /// block.
    #[error("block {block} is coded, but the container is in the raw-only mode")]
    CodedBlockInRawOnly {
        /// The index of the block.
        block: u32,
    },
    /// A length or an offset does not fit in `u64` or `usize`.
    #[error("the length of {part} does not fit in the integer type")]
    LengthOverflow {
        /// The part whose length is too large.
        part: &'static str,
    },
    /// A stream gave a block with a wrong number of stored bytes.
    #[error("block {block} has {found} stored bytes, the block table gives {expected}")]
    StoredLengthMismatch {
        /// The index of the block.
        block: u32,
        /// The stored length in the block table.
        expected: u32,
        /// The number of bytes that the caller gave.
        found: u64,
    },
    /// The stream writer received a block of 0 bytes, a block longer than the block size, or a
    /// block after a short block. Only the last block can be shorter than the block size.
    #[error("block {block} has {found} bytes; the block size is {block_size}")]
    BlockLengthMismatch {
        /// The index of the block.
        block: u32,
        /// The block size.
        block_size: u32,
        /// The length of the block.
        found: u64,
    },
    /// The stream writer received a different number of bytes than the caller stated before
    /// the start. For example, the input file changed during the fold.
    #[error("the input has {found} bytes, {expected} were expected")]
    InputLengthMismatch {
        /// The length that the caller stated.
        expected: u64,
        /// The length that the writer received.
        found: u64,
    },
    /// The restored data has a different length than the header states.
    #[error("the restored data has {found} bytes, the header states {expected}")]
    RestoredLengthMismatch {
        /// The original length in the header.
        expected: u64,
        /// The length of the restored data.
        found: u64,
    },
    /// The SHA-256 of the restored data is not the checksum in the header (plan §3 rule 4).
    #[error("checksum mismatch: the header has {expected}, the restored data has {found}")]
    ChecksumMismatch {
        /// The checksum in the header.
        expected: Checksum,
        /// The checksum of the restored data.
        found: Checksum,
    },
    /// The container needs a different model than the loaded model (plan §3 rule 3).
    #[error("model hash mismatch: the container needs {container}, the loaded model is {loaded}")]
    ModelHashMismatch {
        /// The model hash in the container.
        container: ModelHash,
        /// The hash of the loaded model.
        loaded: ModelHash,
    },
}

/// The result type of the crate.
pub type Result<T> = core::result::Result<T, FormatError>;
