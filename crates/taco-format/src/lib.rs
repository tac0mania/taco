//! The `.taco` container of Taco, format versions 1 and 2: header, block table, checksum, raw
//! fallback and zstd blocks.
//!
//! PLAN(§9): this crate lays out and validates the bytes of a container. It does not run the
//! model or the coder. The caller codes each block and gives the result to
//! [`Container::build`]. [`Container::parse`] checks untrusted bytes and gives the stored bytes
//! of each block as slices of the input, with no copy. ADR 0006 records the rules of version 1
//! that plan §9 does not state. ADR 0011 records version 2: the zstd block kind and the header
//! fields of ADR 0010 and ADR 0012.
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
//! | 80 | 4 | block size | `u32`, at least 1. |
//! | 84 | 4 | block count | `u32`, exactly `ceil(original length / block size)`. |
//! | 88 | 1 | profile | Version 2 only. `u8`: 0 text, 1 logs (ADR 0012). |
//! | 89 | 1 | reserved | Version 2 only. 0. |
//! | 90 | 2 | coding parts | Version 2 only. `u16`, 0. Reserved for ADR 0010. |
//! | 92 | 12 | coding parameters | Version 2 only. All 0. Reserved for ADR 0010. |
//! | H | 5 × count | block table | H = 88 (version 1) or 104 (version 2). For each block: kind (`u8`), stored length (`u32`). |
//! | H + 5 × count | sum of stored lengths | blocks | The stored bytes of each block, in table order, with no gap. |
//!
//! The block kinds are 0 (raw) and 1 (coded) in version 1, and also 2 (zstd) in version 2.
//! Block `i` holds the original bytes `[i × block size, min((i + 1) × block size, original
//! length))`. Thus each block has `block size` bytes, and the last block can have fewer.
//!
//! # Rules that the reader checks
//!
//! 1. The magic, then the version, then the flags. A reader refuses an unknown version with
//!    [`FormatError::UnsupportedVersion`] before it reads other fields.
//! 2. Version 2: the profile is 0 or 1. The reserved byte, the coding parts and the coding
//!    parameters are 0.
//! 3. The block size is at least 1. The block count is `ceil(original length / block size)`.
//!    Thus empty data has 0 blocks and a container of exactly the header.
//! 4. A raw block stores exactly its original bytes: its stored length equals its original
//!    length.
//! 5. A coded block and a zstd block are smaller than their original bytes (plan §3 rule 6).
//!    A coded block of 0 bytes is permitted.
//! 6. A container in the raw-only mode (model hash all zero) has no coded block. It can have
//!    zstd blocks: they need no model.
//! 7. A version 1 container has no zstd block.
//! 8. The input ends exactly at the end of the last block: no missing byte and no extra byte.
//! 9. All sums use checked arithmetic. The parser allocates one entry for each block only
//!    after it checks that the block table is inside the input.
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

mod error;
mod header;
mod layout;
mod reader;
mod types;
mod writer;
mod zstd_block;

pub use error::{FormatError, Result};
pub use header::Header;
pub use layout::worst_case_len;
pub use reader::{BlockRef, Container};
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

/// The length of one entry of the block table, in bytes: the kind (1) and the stored length
/// (4).
pub const TABLE_ENTRY_LEN: usize = 5;
