//! The writer: it lays out the header, the block table and the stored bytes of each block.
//!
//! The caller codes the blocks. The writer does not run the model or the coder. It applies the
//! same rules as the reader, so every container that it writes parses again.

use crate::error::{FormatError, Result};
use crate::layout::{OriginalLens, block_count, header_and_table_len};
use crate::reader::Container;
use crate::types::{BlockKind, BlockSize, Checksum, ModelHash};
use crate::{MAGIC, VERSION};

/// The parts of the header that the caller selects. The writer calculates the other fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContainerParams {
    /// The hash of the model that coded the blocks, or [`ModelHash::RAW_ONLY`].
    pub model_hash: ModelHash,
    /// The number of original bytes in each block.
    pub block_size: BlockSize,
}

impl ContainerParams {
    /// Makes the parameters of a container.
    #[must_use]
    pub const fn new(model_hash: ModelHash, block_size: BlockSize) -> Self {
        Self { model_hash, block_size }
    }
}

/// The result of the coder for one block, as the caller gives it to [`Container::build`].
///
/// A raw block has no bytes here: the writer copies its original bytes. Thus a raw block always
/// stores exactly its original bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Block {
    /// Store the original bytes of the block.
    Raw,
    /// Store these bytes, the output of the range coder for the block.
    Coded(Vec<u8>),
}

impl Block {
    /// Selects the stored form of a block (plan §3 rule 6): [`Block::Coded`] if `coded` is
    /// smaller than the `original_len` bytes of the block, else [`Block::Raw`].
    #[must_use]
    pub fn choose(coded: Vec<u8>, original_len: usize) -> Self {
        if coded.len() < original_len { Self::Coded(coded) } else { Self::Raw }
    }

    /// The kind byte that the block table stores for this block.
    #[must_use]
    pub const fn kind(&self) -> BlockKind {
        match self {
            Self::Raw => BlockKind::Raw,
            Self::Coded(_) => BlockKind::Coded,
        }
    }
}

/// `x` as a `u64`, or a [`FormatError::LengthOverflow`] for `part`.
fn to_u64(x: usize, part: &'static str) -> Result<u64> {
    u64::try_from(x).map_err(|_| FormatError::LengthOverflow { part })
}

/// The stored bytes of one block, after the checks.
struct Stored<'a> {
    kind: BlockKind,
    bytes: &'a [u8],
    /// INVARIANT: `bytes.len()`, at most the block size, so it fits in `u32`.
    len: u32,
}

/// Checks one block against its original bytes and gives the bytes to store.
fn stored<'a>(
    index: u32,
    block: &'a Block,
    original: &'a [u8],
    original_len: u32,
    raw_only: bool,
) -> Result<Stored<'a>> {
    match block {
        Block::Raw => Ok(Stored { kind: BlockKind::Raw, bytes: original, len: original_len }),
        Block::Coded(coded) => {
            if raw_only {
                return Err(FormatError::CodedBlockInRawOnly { block: index });
            }
            let stored_len = to_u64(coded.len(), "coded block")?;
            match u32::try_from(coded.len()) {
                Ok(len) if len < original_len => {
                    Ok(Stored { kind: BlockKind::Coded, bytes: coded, len })
                }
                _ => Err(FormatError::CodedNotSmaller { block: index, original_len, stored_len }),
            }
        }
    }
}

impl Container<'_> {
    /// Writes a container for `original`, with one [`Block`] for each block of
    /// `params.block_size` bytes (the last block can be shorter).
    ///
    /// Use [`BlockSize::split`] to get the original bytes of each block, and [`Block::choose`]
    /// to select raw or coded for each block.
    ///
    /// Cost: one SHA-256 pass over `original` and one copy of each stored block, into one
    /// allocation of the exact output length. One more allocation holds one small entry for
    /// each block.
    ///
    /// # Errors
    ///
    /// - [`FormatError::TooManyBlocks`] if `original` needs more than `u32::MAX` blocks.
    /// - [`FormatError::WrongBlockCount`] if `blocks` does not have one entry for each block.
    /// - [`FormatError::CodedBlockInRawOnly`] if a block is coded and the model hash is
    ///   [`ModelHash::RAW_ONLY`].
    /// - [`FormatError::CodedNotSmaller`] if a coded block is not smaller than its original
    ///   bytes. Use [`Block::choose`] to store such a block raw.
    /// - [`FormatError::LengthOverflow`] if the container length does not fit in memory.
    pub fn build(params: &ContainerParams, original: &[u8], blocks: &[Block]) -> Result<Vec<u8>> {
        let original_len = to_u64(original.len(), "original data")?;
        let count = block_count(original_len, params.block_size)?;
        let found = to_u64(blocks.len(), "block list")?;
        if found != u64::from(count) {
            return Err(FormatError::WrongBlockCount {
                original_len,
                block_size: params.block_size.get(),
                expected: u64::from(count),
                found,
            });
        }

        let raw_only = params.model_hash.is_raw_only();
        let mut parts = Vec::with_capacity(blocks.len());
        let mut total = header_and_table_len(u64::from(count))?;
        let lens = OriginalLens::new(original_len, params.block_size);
        let pieces = params.block_size.split(original);
        for (index, ((block, piece), len)) in (0..count).zip(blocks.iter().zip(pieces).zip(lens)) {
            let part = stored(index, block, piece, len, raw_only)?;
            total = total
                .checked_add(u64::from(part.len))
                .ok_or(FormatError::LengthOverflow { part: "container" })?;
            parts.push(part);
        }
        let total = usize::try_from(total)
            .map_err(|_| FormatError::LengthOverflow { part: "container" })?;

        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(params.model_hash.as_bytes());
        out.extend_from_slice(&original_len.to_le_bytes());
        out.extend_from_slice(Checksum::of(original).as_bytes());
        out.extend_from_slice(&params.block_size.get().to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        for part in &parts {
            out.push(part.kind.as_u8());
            out.extend_from_slice(&part.len.to_le_bytes());
        }
        for part in &parts {
            out.extend_from_slice(part.bytes);
        }
        // NOTE: `total` is exact, so the vector does not grow. The tests check the length.
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bs(n: u32) -> BlockSize {
        BlockSize::new(n).unwrap()
    }

    fn model() -> ModelHash {
        ModelHash::from_bytes([0x11; 32])
    }

    #[test]
    fn choose_stores_raw_when_coded_is_not_smaller() {
        assert_eq!(Block::choose(vec![1, 2], 3), Block::Coded(vec![1, 2]));
        assert_eq!(Block::choose(vec![1, 2, 3], 3), Block::Raw);
        assert_eq!(Block::choose(vec![1, 2, 3, 4], 3), Block::Raw);
        assert_eq!(Block::choose(vec![], 1), Block::Coded(vec![]));
    }

    #[test]
    fn every_field_is_at_its_offset() {
        let data = b"abcdefghij";
        let blocks = [Block::Raw, Block::Coded(vec![0xC0, 0xDE]), Block::Raw];
        let out = Container::build(&ContainerParams::new(model(), bs(4)), data, &blocks).unwrap();
        assert_eq!(&out[0..4], b"TACO");
        assert_eq!(&out[4..6], &[1, 0]);
        assert_eq!(&out[6..8], &[0, 0]);
        assert_eq!(&out[8..40], &[0x11; 32]);
        assert_eq!(&out[40..48], &10u64.to_le_bytes());
        assert_eq!(&out[48..80], Checksum::of(data).as_bytes());
        assert_eq!(&out[80..84], &4u32.to_le_bytes());
        assert_eq!(&out[84..88], &3u32.to_le_bytes());
        assert_eq!(&out[88..93], &[0, 4, 0, 0, 0]);
        assert_eq!(&out[93..98], &[1, 2, 0, 0, 0]);
        assert_eq!(&out[98..103], &[0, 2, 0, 0, 0]);
        assert_eq!(&out[103..107], b"abcd");
        assert_eq!(&out[107..109], &[0xC0, 0xDE]);
        assert_eq!(&out[109..], b"ij");
    }

    #[test]
    fn empty_data_gives_the_header_only() {
        let out =
            Container::build(&ContainerParams::new(ModelHash::RAW_ONLY, bs(7)), &[], &[]).unwrap();
        assert_eq!(out.len(), 88);
        assert_eq!(&out[40..48], &[0; 8]);
        assert_eq!(&out[48..80], Checksum::of(&[]).as_bytes());
        assert_eq!(&out[84..88], &[0; 4]);
    }

    #[test]
    fn build_refuses_a_wrong_number_of_blocks() {
        let p = ContainerParams::new(model(), bs(4));
        assert_eq!(
            Container::build(&p, b"abcde", &[Block::Raw]),
            Err(FormatError::WrongBlockCount {
                original_len: 5,
                block_size: 4,
                expected: 2,
                found: 1
            })
        );
        assert!(Container::build(&p, b"", &[Block::Raw]).is_err());
    }

    #[test]
    fn build_refuses_a_coded_block_that_is_not_smaller() {
        let p = ContainerParams::new(model(), bs(4));
        assert_eq!(
            Container::build(&p, b"abcde", &[Block::Raw, Block::Coded(vec![9])]),
            Err(FormatError::CodedNotSmaller { block: 1, original_len: 1, stored_len: 1 })
        );
    }

    #[test]
    fn build_refuses_a_coded_block_in_the_raw_only_mode() {
        let p = ContainerParams::new(ModelHash::RAW_ONLY, bs(4));
        assert_eq!(
            Container::build(&p, b"abcd", &[Block::Coded(vec![])]),
            Err(FormatError::CodedBlockInRawOnly { block: 0 })
        );
    }
}
