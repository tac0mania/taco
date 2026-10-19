//! The `.taco` container of Taco, format versions 1 and 2: header, block table, checksum, raw
//! fallback and zstd blocks.
//!
//! PLAN(§9): this crate lays out and validates the bytes of a container. It does not run the
//! model or the coder. The caller codes each block and gives the result to
//! [`Container::build`]. [`Container::parse`] checks untrusted bytes and gives the stored bytes
//! of each block as slices of the input, with no copy. ADR 0006 records the rules of version 1
//! that plan §9 does not state. ADR 0011 records version 2: the zstd block kind and the header
//! fields of ADR 0010 and ADR 0012. ADR 0010 gives the coding parts their meaning: the match
//! model and the mixer, and the mixed block kind. ADR 0015 adds the context overlap.
//!
//! The writer writes version 2 only. The reader reads version 1 and version 2.
//!
//! # Layout
//!
//! All integers are little-endian. The header is [`HEADER_LEN_V1`] (88) bytes in version 1
//! and [`HEADER_LEN`] (104) bytes in version 2. Each entry of the block table is
//! [`TABLE_ENTRY_LEN`] (5) bytes.
//!
//! | Offset | Size | Field | Rule |
//! |---|---|---|---|
//! | 0 | 4 | magic | `"TACO"` |
//! | 4 | 2 | version | `u16`, 1 or 2. This position is the same in every version. |
//! | 6 | 2 | flags | `u16`, 0. Versions 1 and 2 define no flag. |
//! | 8 | 32 | model hash | SHA-256 of the model file. All zero: the raw-only mode. |
//! | 40 | 8 | original length | `u64`, the number of bytes of the original data. |
//! | 48 | 32 | checksum | SHA-256 of the original data. |
//! | 80 | 4 | block size | `u32`, at least 1, at most [`MAX_BLOCK_SIZE`] (64 MiB). |
//! | 84 | 4 | block count | `u32`, exactly `ceil(original length / block size)`. |
//! | 88 | 1 | profile | Version 2 only. `u8`: 0 text, 1 logs (ADR 0012). |
//! | 89 | 1 | reserved | Version 2 only. 0. |
//! | 90 | 2 | coding parts | Version 2 only. `u16` bit set. Bit 0: the match model and the mixer (ADR 0010). Bit 1: the context overlap (ADR 0015). Other bits 0. |
//! | 92 | 12 | coding parameters | Version 2 only. Byte 0: the mixer rule version (1 or 2, ADR 0017) if bit 0 is set, else 0. Byte 1: 0. Bytes 2 and 3: the overlap `K` (`u16`, 1 or more) if bit 1 is set, else 0. Bytes 4 to 11: 0. |
//! | H | 5 × count | block table | H = 88 (version 1) or 104 (version 2). For each block: kind (`u8`), stored length (`u32`). |
//! | H + 5 × count | sum of stored lengths | blocks | The stored bytes of each block, in table order, with no gap. |
//!
//! The block kinds are 0 (raw) and 1 (coded) in version 1, and also 2 (zstd) and 3 (mixed:
//! coded with the match model and the mixer) in version 2.
//! Block `i` holds the original bytes `[i × block size, min((i + 1) × block size, original
//! length))`. Thus each block has `block size` bytes, and the last block can have fewer.
//!
//! # Rules that the reader checks
//!
//! 1. The magic, then the version, then the flags. A reader refuses an unknown version with
//!    [`FormatError::UnsupportedVersion`] before it reads other fields.
//! 2. Version 2: the profile is 0 or 1. The reserved byte is 0. The coding parts have only
//!    bits 0 and 1, and the profile `text` does not have bit 0. With bit 0, byte 0 of the
//!    coding parameters is a known mixer rule version (1 or 2); without it, byte 0 is 0. With bit
//!    1, bytes 2 and 3 hold `K >= 1`; without it, they are 0. Byte 1 and bytes 4 to 11 of the
//!    coding parameters are 0. A reader that does not know a bit refuses the container, so it
//!    never decodes with the wrong rule.
//! 3. The block size is at least 1 and at most [`MAX_BLOCK_SIZE`] (64 MiB, both versions; ADR 0011,
//!    addendum §4). The block count is `ceil(original length / block size)`.
//!    Thus empty data has 0 blocks and a container of exactly the header.
//! 4. A raw block stores exactly its original bytes: its stored length equals its original
//!    length.
//! 5. A coded block, a mixed block and a zstd block are smaller than their original bytes
//!    (plan §3 rule 6). A coded or mixed block of 0 bytes is permitted.
//! 6. A container in the raw-only mode (model hash all zero) has no coded or mixed block. It
//!    can have zstd blocks: they need no model.
//! 7. A version 1 container has no zstd block and no mixed block.
//! 8. A mixed block is permitted only if the coding parts have bit 0.
//! 9. The input ends exactly at the end of the last block: no missing byte and no extra byte.
//! 10. All sums use checked arithmetic. The parser allocates one entry for each block only
//!     after it checks that the block table is inside the input.
//!
//! Thus a version 2 container is at most `104 + 5 × block count + original length` bytes long
//! (plan §3 rule 6, [`worst_case_len`]).
//!
//! The parser does not check the checksum or the model hash: it does not have the original
//! data or the model. The caller uses [`Header::verify_model_hash`] before it decodes, and
//! [`Header::verify_checksum`] after it restores the data (plan §3 rules 3 and 4). The parser
//! does not decode a zstd block. [`BlockRef::decode_zstd_into`] decodes it, with the checks of
//! ADR 0011 §5.
//!
//! # Cost
//!
//! - [`Container::build`]: one SHA-256 pass over the original data and one copy of each
//!   stored block. One allocation of the exact output length, and one of one small entry for
//!   each block.
//! - [`Container::parse`]: two passes over the block table. One allocation of one small entry
//!   for each block. No copy of block data.
//! - [`Header::verify_checksum`]: one SHA-256 pass over the restored data.
//! - [`compress_zstd`] and [`BlockRef::decode_zstd_into`]: one zstd pass over one block.

// NOTE: this is not a kernel crate, but it has no floats. The lint keeps it so.
#![deny(clippy::float_arithmetic)]

mod coding;
mod error;
mod header;
mod layout;
mod reader;
mod stream;
mod types;
mod writer;
mod zstd_block;

pub use coding::{CodingParts, ContextOverlap, MixerRule};
pub use error::{FormatError, Result};
pub use header::Header;
pub use layout::worst_case_len;
pub use reader::{BlockRef, Container};
pub use stream::{ChecksumBuilder, StreamWriter, TableEntry, TableReader};
pub use types::{BlockKind, BlockSize, Checksum, ModelHash, Profile};
pub use writer::{Block, ContainerParams};
pub use zstd_block::{
    ZSTD_LEVEL, ZSTD_LIBRARY_VERSION, ZSTD_WINDOW_LOG_MAX, compress_zstd, zstd_library_version,
};

/// The first 4 bytes of every container.
pub const MAGIC: [u8; 4] = *b"TACO";

/// The format version that this crate writes. The reader also reads [`VERSION_1`].
pub const VERSION: u16 = 2;

/// The first format version (ADR 0006). The reader reads it. The writer does not write it.
pub const VERSION_1: u16 = 1;

/// The length of the header of version 2, the version that this crate writes, in bytes.
pub const HEADER_LEN: usize = 104;

/// The length of the header of version 1, in bytes.
pub const HEADER_LEN_V1: usize = 88;

/// The largest block size of versions 1 and 2: 64 MiB (ADR 0011, addendum "Version 2 is
/// frozen", §4). The reader refuses a header with a larger block size, and the writer refuses
/// to write one ([`FormatError::BlockSizeTooLarge`]).
///
/// The block size bounds the work and the memory of one block: a coded block has at most one
/// token for each original byte, and the decoder holds at most the stored and the restored
/// bytes of a block. NOTE: the largest block size that a measurement used is 1 MiB (the logs
/// profile, ADR 0012); 64 MiB keeps a factor of 64 above it.
// NOTE: the compiler evaluates this constant. A panic here is a build error, not a run-time
// panic, and 64 MiB is not 0.
#[allow(clippy::panic)]
pub const MAX_BLOCK_SIZE: BlockSize = {
    let Ok(b) = BlockSize::new(64 * 1024 * 1024) else { panic!("the block size is not 0") };
    b
};

/// The length of one entry of the block table, in bytes: the kind (1) and the stored length
/// (4).
pub const TABLE_ENTRY_LEN: usize = 5;
