//! The `.taco` container of Taco, format version 1: header, block table, checksum and raw
//! fallback.
//!
//! PLAN(§9): this crate lays out and validates the bytes of a container. It does not run the
//! model or the coder. The caller codes each block and gives the result to
//! [`Container::build`]. [`Container::parse`] checks untrusted bytes and gives the stored bytes
//! of each block as slices of the input, with no copy. ADR 0006 records the rules that plan §9
//! does not state.
//!
//! # Layout (version 1)
//!
//! All integers are little-endian. The header is [`HEADER_LEN`] (88) bytes. Each entry of the
//! block table is [`TABLE_ENTRY_LEN`] (5) bytes.
//!
//! | Offset | Size | Field | Rule |
//! |---|---|---|---|
//! | 0 | 4 | magic | `"TACO"` |
//! | 4 | 2 | version | `u16`, 1. This position is the same in every version. |
//! | 6 | 2 | flags | `u16`, 0. Version 1 defines no flag. |
//! | 8 | 32 | model hash | SHA-256 of the model file. All zero: the raw-only mode. |
//! | 40 | 8 | original length | `u64`, the number of bytes of the original data. |
//! | 48 | 32 | checksum | SHA-256 of the original data. |
//! | 80 | 4 | block size | `u32`, at least 1. |
//! | 84 | 4 | block count | `u32`, exactly `ceil(original length / block size)`. |
//! | 88 | 5 × count | block table | For each block: kind (`u8`: 0 raw, 1 coded), stored length (`u32`). |
//! | 88 + 5 × count | sum of stored lengths | blocks | The stored bytes of each block, in table order, with no gap. |
//!
//! Block `i` holds the original bytes `[i × block size, min((i + 1) × block size, original
//! length))`. Thus each block has `block size` bytes, and the last block can have fewer.
//!
//! # Rules that the reader checks
//!
//! 1. The magic, then the version, then the flags. A reader refuses an unknown version with
//!    [`FormatError::UnsupportedVersion`] before it reads other fields.
//! 2. The block size is at least 1. The block count is `ceil(original length / block size)`.
//!    Thus empty data has 0 blocks and a container of exactly 88 bytes.
//! 3. A raw block stores exactly its original bytes: its stored length equals its original
//!    length.
//! 4. A coded block is smaller than its original bytes (plan §3 rule 6). A coded block of 0
//!    bytes is permitted.
//! 5. A container in the raw-only mode (model hash all zero) has no coded block.
//! 6. The input ends exactly at the end of the last block: no missing byte and no extra byte.
//! 7. All sums use checked arithmetic. The parser allocates one entry for each block only
//!    after it checks that the block table is inside the input.
//!
//! Thus a container is at most `88 + 5 × block count + original length` bytes long (plan §3
//! rule 6, [`worst_case_len`]).
//!
//! The parser does not check the checksum or the model hash: it does not have the original
//! data or the model. The caller uses [`Header::verify_model_hash`] before it decodes, and
//! [`Header::verify_checksum`] after it restores the data (plan §3 rules 3 and 4).
//!
//! # Cost
//!
//! - [`Container::build`]: one SHA-256 pass over the original data and one copy of each
//!   stored block. One allocation of the exact output length, and one of one small entry for
//!   each block.
//! - [`Container::parse`]: two passes over the block table. One allocation of one small entry
//!   for each block. No copy of block data.
//! - [`Header::verify_checksum`]: one SHA-256 pass over the restored data.

// NOTE: this is not a kernel crate, but it has no floats. The lint keeps it so.
#![deny(clippy::float_arithmetic)]

mod error;
mod layout;
mod reader;
mod types;
mod writer;

pub use error::{FormatError, Result};
pub use layout::worst_case_len;
pub use reader::{BlockRef, Container, Header};
pub use types::{BlockKind, BlockSize, Checksum, ModelHash};
pub use writer::{Block, ContainerParams};

/// The first 4 bytes of every container.
pub const MAGIC: [u8; 4] = *b"TACO";

/// The format version that this crate writes and reads.
pub const VERSION: u16 = 1;

/// The length of the fixed header of version 1, in bytes.
pub const HEADER_LEN: usize = 88;

/// The length of one entry of the block table, in bytes: the kind (1) and the stored length
/// (4).
pub const TABLE_ENTRY_LEN: usize = 5;
