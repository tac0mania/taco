//! The reader: it checks untrusted container bytes and gives the stored bytes of each block as
//! slices of the input.
//!
//! The rules are in the crate documentation. The reader allocates one [`BlockRef`] for each
//! block only after it checks that the block table is inside the input. Thus a header field
//! cannot cause an allocation larger than the input.

use crate::TABLE_ENTRY_LEN;
use crate::error::{FormatError, Result};
use crate::header::Header;
use crate::layout::{OriginalLens, header_and_table_len};
use crate::types::BlockKind;
use crate::zstd_block;

/// One block of a parsed container: its kind, its original length and its stored bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockRef<'a> {
    index: u32,
    kind: BlockKind,
    original_offset: u64,
    original_len: u32,
    stored: &'a [u8],
}

impl<'a> BlockRef<'a> {
    /// The index of the block in the container, from 0.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// How the block stores its bytes.
    #[must_use]
    pub const fn kind(&self) -> BlockKind {
        self.kind
    }

    /// The position of the first byte of the block in the original data:
    /// `index * block_size`.
    #[must_use]
    pub const fn original_offset(&self) -> u64 {
        self.original_offset
    }

    /// The number of original bytes of the block: the block size, or less for the last
    /// block. It is at least 1.
    #[must_use]
    pub const fn original_len(&self) -> u32 {
        self.original_len
    }

    /// The stored bytes, a slice of the container. For a raw block, these are the original
    /// bytes. For a coded block and a mixed block, these are the payload for the range decoder
    /// (ADR 0007). For a zstd block, these are one zstd frame. A coded, mixed or zstd block has
    /// fewer stored bytes than original bytes.
    #[must_use]
    pub const fn stored(&self) -> &'a [u8] {
        self.stored
    }

    /// Decodes a zstd block and appends exactly [`BlockRef::original_len`] bytes to `out`. On
    /// an error, `out` keeps its old length. The rules are in ADR 0011 §5.
    ///
    /// The memory is bounded: the zstd window is at most 8 MiB, and `out` grows only with the
    /// bytes that the frame really gives, up to the original length of the block. Thus a
    /// damaged header cannot make the decoder allocate a large buffer from a few bytes.
    ///
    /// Cost: one zstd decode of the block and one copy of the output.
    ///
    /// # Errors
    ///
    /// - [`FormatError::NotZstd`] if the block is not a zstd block.
    /// - [`FormatError::CorruptZstdBlock`] if the frame is not one zstd frame with the
    ///   content size of the block, if the zstd decoder refuses it, or if it gives more or
    ///   fewer bytes than the original length.
    /// - [`FormatError::Zstd`] if the zstd library fails, for example with no memory.
    pub fn decode_zstd_into(&self, out: &mut Vec<u8>) -> Result<()> {
        if self.kind != BlockKind::Zstd {
            return Err(FormatError::NotZstd { block: self.index });
        }
        zstd_block::decode_into(self.stored, self.original_len, self.index, out)
    }
}

/// A valid container of version 1 or version 2: its header and its blocks. The blocks borrow
/// the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Container<'a> {
    header: Header,
    blocks: Vec<BlockRef<'a>>,
}

/// `x` as a `usize`, or a [`FormatError::LengthOverflow`] for `part`.
fn to_usize(x: u64, part: &'static str) -> Result<usize> {
    usize::try_from(x).map_err(|_| FormatError::LengthOverflow { part })
}

/// Decodes and checks one entry of the block table of a container of `header.version()`.
fn entry(
    index: u32,
    bytes: &[u8; TABLE_ENTRY_LEN],
    original_len: u32,
    header: &Header,
) -> Result<(BlockKind, u32)> {
    let [kind, l0, l1, l2, l3] = *bytes;
    let stored_len = u32::from_le_bytes([l0, l1, l2, l3]);
    let raw_only = !header.needs_model();
    let kind = BlockKind::from_u8(kind)
        .filter(|k| k.min_version() <= header.version())
        .ok_or(FormatError::UnknownBlockKind { block: index, kind })?;
    match kind {
        BlockKind::Raw if stored_len != original_len => Err(FormatError::RawLengthMismatch {
            block: index,
            original_len,
            stored_len: u64::from(stored_len),
        }),
        BlockKind::Coded if raw_only => Err(FormatError::CodedBlockInRawOnly { block: index }),
        BlockKind::Coded if stored_len >= original_len => Err(FormatError::CodedNotSmaller {
            block: index,
            original_len,
            stored_len: u64::from(stored_len),
        }),
        BlockKind::Zstd if stored_len >= original_len => Err(FormatError::ZstdNotSmaller {
            block: index,
            original_len,
            stored_len: u64::from(stored_len),
        }),
        BlockKind::Mixed if header.coding_parts().mixer().is_none() => {
            Err(FormatError::MixedWithoutCodingPart { block: index })
        }
        BlockKind::Mixed if raw_only => Err(FormatError::CodedBlockInRawOnly { block: index }),
        BlockKind::Mixed if stored_len >= original_len => Err(FormatError::CodedNotSmaller {
            block: index,
            original_len,
            stored_len: u64::from(stored_len),
        }),
        BlockKind::Raw | BlockKind::Coded | BlockKind::Zstd | BlockKind::Mixed => {
            Ok((kind, stored_len))
        }
    }
}

impl<'a> Container<'a> {
    /// Parses and checks a container. The checks are in the crate documentation. The blocks
    /// are slices of `bytes`: the function copies no block data.
    ///
    /// The function does not verify the checksum or the model hash. Use
    /// [`Header::verify_model_hash`] and [`Header::verify_checksum`].
    ///
    /// Cost: two passes over the block table, and one allocation of one [`BlockRef`] for each
    /// block. The block count is at most `bytes.len() / 5`.
    ///
    /// # Errors
    ///
    /// - [`FormatError::BadMagic`], [`FormatError::UnsupportedVersion`] and
    ///   [`FormatError::UnknownFlags`] for an input that is not a version 1 or version 2
    ///   container.
    /// - [`FormatError::UnknownProfile`], [`FormatError::UnknownCodingParts`],
    ///   [`FormatError::UnknownMixerRule`], [`FormatError::CodingPartsNotInProfile`] and
    ///   [`FormatError::ReservedNotZero`] for a bad field of a version 2 header.
    /// - [`FormatError::Truncated`] if the header or the block table is not complete.
    /// - [`FormatError::ZeroBlockSize`] and [`FormatError::WrongBlockCount`] for a header that
    ///   does not agree with itself.
    /// - [`FormatError::UnknownBlockKind`] (also kinds 2 and 3 in version 1),
    ///   [`FormatError::RawLengthMismatch`], [`FormatError::CodedNotSmaller`],
    ///   [`FormatError::ZstdNotSmaller`], [`FormatError::CodedBlockInRawOnly`] and
    ///   [`FormatError::MixedWithoutCodingPart`] for a bad entry of the block table.
    /// - [`FormatError::WrongTotalLength`] if the input does not end at the end of the last
    ///   block.
    /// - [`FormatError::LengthOverflow`] if a length does not fit in `u64` or `usize`.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let found = u64::try_from(bytes.len())
            .map_err(|_| FormatError::LengthOverflow { part: "container" })?;
        let (header, rest) = Header::parse(bytes, found)?;
        let header_len = u64::try_from(header.byte_len())
            .map_err(|_| FormatError::LengthOverflow { part: "header" })?;

        let table_end = header_and_table_len(header_len, u64::from(header.block_count()))?;
        if table_end > found {
            return Err(FormatError::Truncated { part: "block table", needed: table_end, found });
        }
        // INVARIANT: table_end <= found, so the subtraction cannot wrap.
        let table_len = to_usize(table_end - header_len, "block table")?;
        let (table, mut payload) = rest
            .split_at_checked(table_len)
            .ok_or(FormatError::Truncated { part: "block table", needed: table_end, found })?;
        let (table, _) = table.as_chunks::<TABLE_ENTRY_LEN>();
        let lens = || OriginalLens::new(header.original_len(), header.block_size());

        // Pass 1: check each entry and add the stored lengths, before any allocation.
        let mut expected = table_end;
        for (index, (e, len)) in (0..header.block_count()).zip(table.iter().zip(lens())) {
            let (_, stored_len) = entry(index, e, len, &header)?;
            expected = expected
                .checked_add(u64::from(stored_len))
                .ok_or(FormatError::LengthOverflow { part: "container" })?;
        }
        if expected != found {
            return Err(FormatError::WrongTotalLength { expected, found });
        }

        // Pass 2: cut the payload into blocks. The table is inside the input, so the
        // allocation is at most bytes.len() / 5 entries.
        let mut blocks = Vec::with_capacity(table.len());
        let block_size = u64::from(header.block_size().get());
        let mut original_offset = 0u64;
        for (index, (e, len)) in (0..header.block_count()).zip(table.iter().zip(lens())) {
            let (kind, stored_len) = entry(index, e, len, &header)?;
            let (stored, tail) = payload
                .split_at_checked(to_usize(u64::from(stored_len), "block")?)
                .ok_or(FormatError::WrongTotalLength { expected, found })?;
            payload = tail;
            blocks.push(BlockRef { index, kind, original_offset, original_len: len, stored });
            // INVARIANT: the block count and the block size are less than 2^32, so
            // block_count * block_size < 2^64 and checked_add does not fail.
            original_offset = original_offset
                .checked_add(block_size)
                .ok_or(FormatError::LengthOverflow { part: "original data" })?;
        }
        Ok(Self { header, blocks })
    }

    /// The header.
    #[must_use]
    pub const fn header(&self) -> &Header {
        &self.header
    }

    /// The blocks, in order. There are [`Header::block_count`] of them.
    #[must_use]
    pub fn blocks(&self) -> &[BlockRef<'a>] {
        &self.blocks
    }
}
