//! The container in parts: a writer that takes one block at a time, and a reader that checks
//! one table entry at a time. Neither does I/O. The caller moves the bytes (ADR 0018).
//!
//! A version 2 container cannot be written in one pass: the header holds the original length,
//! the checksum and the block count, and the block table comes before the first stored block
//! (ADR 0018). Thus [`StreamWriter`] gives the table entry and the stored bytes of each block,
//! and the header at the end. The caller puts them at their offsets: it seeks back, or it
//! keeps the table and the stored bytes in temporary files. The bytes are the bytes of
//! [`crate::Container::build`] for the same blocks.
//!
//! A reader can read a container in one pass: the header ([`Header::from_bytes`]), the table
//! ([`TableReader`]), then the stored bytes of each block ([`TableEntry::block_ref`]). The
//! checks are the checks of [`crate::Container::parse`].

use sha2::{Digest, Sha256};

use crate::coding::CodingParts;
use crate::error::{FormatError, Result};
use crate::header::{Header, check_profile};
use crate::layout::{HEADER_LEN_U64, block_count, expected_block_count, header_and_table_len};
use crate::reader::{BlockRef, entry};
use crate::types::{BlockKind, Checksum, Profile};
use crate::writer::{Block, ContainerParams, stored};
use crate::{HEADER_LEN, TABLE_ENTRY_LEN};

/// The SHA-256 of data that arrives in parts. [`ChecksumBuilder::finish`] gives the same value
/// as [`Checksum::of`] of all parts together.
#[derive(Clone, Debug, Default)]
pub struct ChecksumBuilder(Sha256);

impl ChecksumBuilder {
    /// A builder with no data.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `data` after the data that the builder has. Cost: one SHA-256 pass over `data`.
    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    /// The checksum of all data that the builder received.
    #[must_use]
    pub fn finish(self) -> Checksum {
        Checksum::from_bytes(self.0.finalize().into())
    }
}

/// One checked entry of the block table: the kind and the stored length of a block, with its
/// index and its place in the original data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableEntry {
    index: u32,
    kind: BlockKind,
    /// INVARIANT: the reader or the writer checked it against the kind and the original length.
    stored_len: u32,
    original_len: u32,
    original_offset: u64,
}

impl TableEntry {
    /// The index of the block, from 0.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// The kind of the block.
    #[must_use]
    pub const fn kind(&self) -> BlockKind {
        self.kind
    }

    /// The number of stored bytes of the block.
    #[must_use]
    pub const fn stored_len(&self) -> u32 {
        self.stored_len
    }

    /// The number of original bytes of the block: at least 1, at most the block size.
    #[must_use]
    pub const fn original_len(&self) -> u32 {
        self.original_len
    }

    /// The position of the first original byte of the block.
    #[must_use]
    pub const fn original_offset(&self) -> u64 {
        self.original_offset
    }

    /// The 5 bytes of the entry in the block table: the kind, then the stored length
    /// (little-endian `u32`).
    #[must_use]
    pub const fn to_bytes(&self) -> [u8; TABLE_ENTRY_LEN] {
        let [a, b, c, d] = self.stored_len.to_le_bytes();
        [self.kind.as_u8(), a, b, c, d]
    }

    /// The block of this entry with its stored bytes. No copy.
    ///
    /// # Errors
    ///
    /// [`FormatError::StoredLengthMismatch`] if `stored` does not have the stored length of
    /// the entry.
    pub fn block_ref<'a>(&self, stored: &'a [u8]) -> Result<BlockRef<'a>> {
        if u64::try_from(stored.len()).ok() != Some(u64::from(self.stored_len)) {
            return Err(FormatError::StoredLengthMismatch {
                block: self.index,
                expected: self.stored_len,
                found: u64::try_from(stored.len()).unwrap_or(u64::MAX),
            });
        }
        Ok(BlockRef::from_parts(
            self.index,
            self.kind,
            self.original_offset,
            self.original_len,
            stored,
        ))
    }
}

/// Writes a version 2 container one block at a time, with no I/O (ADR 0018).
///
/// The caller gives each block in order with [`StreamWriter::push`]. The writer checks the
/// block with the rules of [`crate::Container::build`], adds its original bytes to the
/// checksum, and gives the table entry and the stored bytes. At the end,
/// [`StreamWriter::finish`] gives the header. The caller writes the header at offset 0, the
/// table entries from offset [`HEADER_LEN`], and the stored bytes after the table, in order.
///
/// Memory: a fixed size. The writer keeps no block.
#[derive(Clone, Debug)]
pub struct StreamWriter {
    params: ContainerParams,
    expected_len: Option<u64>,
    blocks: u32,
    original_len: u64,
    stored_total: u64,
    /// The index of a block that was shorter than the block size. Only the last block can be.
    short_block: Option<u32>,
    /// The index of the first mixed block, if one was pushed.
    first_mixed: Option<u32>,
    checksum: ChecksumBuilder,
}

impl StreamWriter {
    /// A writer for blocks of `params.block_size` with the model hash of `params`.
    /// `expected_len` is the original length if the caller knows it before the start (a
    /// file), else `None` (a pipe). The profile and the coding parts of `params` are the ones
    /// that the blocks use. [`StreamWriter::finish`] can still change them (`auto`).
    ///
    /// # Errors
    ///
    /// - [`FormatError::TooManyBlocks`] if `expected_len` needs more than `u32::MAX` blocks.
    /// - [`FormatError::CodingPartsNotInProfile`] if the profile does not permit the parts.
    pub fn new(params: ContainerParams, expected_len: Option<u64>) -> Result<Self> {
        check_profile(params.profile, params.coding)?;
        params.block_size.check_max()?;
        if let Some(len) = expected_len {
            block_count(len, params.block_size)?;
        }
        Ok(Self {
            params,
            expected_len,
            blocks: 0,
            original_len: 0,
            stored_total: 0,
            short_block: None,
            first_mixed: None,
            checksum: ChecksumBuilder::new(),
        })
    }

    /// The number of bytes of the header and the table for `original_len` bytes in blocks of
    /// the block size: the offset of the first stored byte.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooManyBlocks`] or [`FormatError::LengthOverflow`].
    pub fn data_offset(&self, original_len: u64) -> Result<u64> {
        let count = block_count(original_len, self.params.block_size)?;
        header_and_table_len(HEADER_LEN_U64, u64::from(count))
    }

    /// Checks the next block and gives its table entry and its stored bytes. `original` is the
    /// original bytes of the block: the block size, or fewer for the last block.
    ///
    /// Cost: one SHA-256 pass over `original`.
    ///
    /// # Errors
    ///
    /// - [`FormatError::BlockLengthMismatch`] if `original` is empty, longer than the block
    ///   size, or comes after a block that was shorter than the block size.
    /// - [`FormatError::InputLengthMismatch`] if the data is longer than `expected_len`.
    /// - [`FormatError::TooManyBlocks`] if this is block `u32::MAX`.
    /// - The block errors of [`crate::Container::build`].
    pub fn push<'a>(
        &mut self,
        original: &'a [u8],
        block: &'a Block,
    ) -> Result<(TableEntry, &'a [u8])> {
        let index = self.blocks;
        let size = self.params.block_size.get();
        let len = u32::try_from(original.len()).ok().filter(|&l| l >= 1 && l <= size);
        let (Some(len), None) = (len, self.short_block) else {
            return Err(FormatError::BlockLengthMismatch {
                block: index,
                block_size: size,
                found: u64::try_from(original.len()).unwrap_or(u64::MAX),
            });
        };
        let new_len = self
            .original_len
            .checked_add(u64::from(len))
            .ok_or(FormatError::LengthOverflow { part: "original data" })?;
        if let Some(expected) = self.expected_len
            && new_len > expected
        {
            return Err(FormatError::InputLengthMismatch { expected, found: new_len });
        }
        let next = index.checked_add(1).ok_or(FormatError::TooManyBlocks {
            original_len: new_len,
            block_size: size,
            blocks: u64::from(index) + 1,
        })?;
        let part = stored(index, block, original, len, &self.params)?;
        self.stored_total = self
            .stored_total
            .checked_add(u64::from(part.len))
            .ok_or(FormatError::LengthOverflow { part: "container" })?;
        if len < size {
            self.short_block = Some(index);
        }
        if part.kind == BlockKind::Mixed && self.first_mixed.is_none() {
            self.first_mixed = Some(index);
        }
        self.checksum.update(original);
        let entry = TableEntry {
            index,
            kind: part.kind,
            stored_len: part.len,
            original_len: len,
            original_offset: self.original_len,
        };
        self.original_len = new_len;
        self.blocks = next;
        Ok((entry, part.bytes))
    }

    /// The number of original bytes that the writer received.
    #[must_use]
    pub const fn original_len(&self) -> u64 {
        self.original_len
    }

    /// The number of blocks that the writer received.
    #[must_use]
    pub const fn blocks(&self) -> u32 {
        self.blocks
    }

    /// Ends the container with `profile` and `coding` and gives the header and its
    /// [`HEADER_LEN`] bytes, and the full container length.
    ///
    /// # Errors
    ///
    /// - [`FormatError::InputLengthMismatch`] if the data is not `expected_len` bytes.
    /// - [`FormatError::CodingPartsNotInProfile`] if the profile does not permit the parts.
    /// - [`FormatError::MixedWithoutCodingPart`] if a block is mixed and `coding` has no
    ///   mixer.
    /// - [`FormatError::LengthOverflow`] if the container length does not fit in `u64`.
    pub fn finish(
        self,
        profile: Profile,
        coding: CodingParts,
    ) -> Result<(Header, [u8; HEADER_LEN], u64)> {
        if let Some(expected) = self.expected_len
            && expected != self.original_len
        {
            return Err(FormatError::InputLengthMismatch { expected, found: self.original_len });
        }
        check_profile(profile, coding)?;
        if let (Some(block), None) = (self.first_mixed, coding.mixer()) {
            return Err(FormatError::MixedWithoutCodingPart { block });
        }
        let params = self.params.with_profile(profile).with_coding(coding);
        let header =
            Header::new_v2(&params, self.original_len, self.checksum.finish(), self.blocks);
        let total = header_and_table_len(HEADER_LEN_U64, u64::from(self.blocks))?
            .checked_add(self.stored_total)
            .ok_or(FormatError::LengthOverflow { part: "container" })?;
        Ok((header, header.to_bytes_v2()?, total))
    }
}

/// Checks the entries of a block table one at a time, in order, with the rules of
/// [`crate::Container::parse`]. It also adds the stored lengths, so the caller can compare the
/// container length with the input.
///
/// Memory: a fixed size. The reader keeps no entry.
#[derive(Clone, Debug)]
pub struct TableReader {
    header: Header,
    next: u32,
    original_offset: u64,
    container_len: u64,
}

impl TableReader {
    /// A reader for the table of a container with `header`.
    ///
    /// # Errors
    ///
    /// [`FormatError::LengthOverflow`] if the header and the table do not fit in `u64`.
    pub fn new(header: &Header) -> Result<Self> {
        let header_len = u64::try_from(header.byte_len())
            .map_err(|_| FormatError::LengthOverflow { part: "header" })?;
        let container_len = header_and_table_len(header_len, u64::from(header.block_count()))?;
        Ok(Self { header: *header, next: 0, original_offset: 0, container_len })
    }

    /// The header and the table length: the offset of the first stored byte.
    ///
    /// # Errors
    ///
    /// [`FormatError::LengthOverflow`] (not on a valid header).
    pub fn data_offset(&self) -> Result<u64> {
        let header_len = u64::try_from(self.header.byte_len())
            .map_err(|_| FormatError::LengthOverflow { part: "header" })?;
        header_and_table_len(header_len, u64::from(self.header.block_count()))
    }

    /// The number of entries that the table has after the entries already read.
    #[must_use]
    pub const fn remaining(&self) -> u32 {
        self.header.block_count().saturating_sub(self.next)
    }

    /// The container length that the header and the entries read so far give. After the last
    /// entry, this is the exact length of a valid container.
    #[must_use]
    pub const fn container_len(&self) -> u64 {
        self.container_len
    }

    /// Checks the next entry of the table.
    ///
    /// # Errors
    ///
    /// - [`FormatError::WrongBlockCount`] if all entries were already read.
    /// - The entry errors of [`crate::Container::parse`].
    /// - [`FormatError::LengthOverflow`] if the container length does not fit in `u64`.
    pub fn next_entry(&mut self, bytes: &[u8; TABLE_ENTRY_LEN]) -> Result<TableEntry> {
        let index = self.next;
        let count = self.header.block_count();
        let block_size = self.header.block_size();
        if index >= count {
            return Err(FormatError::WrongBlockCount {
                original_len: self.header.original_len(),
                block_size: block_size.get(),
                expected: expected_block_count(self.header.original_len(), block_size),
                found: u64::from(index) + 1,
            });
        }
        // INVARIANT: index < count = ceil(original_len / block_size), so the offset is less
        // than original_len, and the rest is at least 1.
        let rest = self.header.original_len().saturating_sub(self.original_offset);
        let original_len =
            u32::try_from(rest).map_or(block_size.get(), |r| r.min(block_size.get()));
        let (kind, stored_len) = entry(index, bytes, original_len, &self.header)?;
        self.container_len = self
            .container_len
            .checked_add(u64::from(stored_len))
            .ok_or(FormatError::LengthOverflow { part: "container" })?;
        let e = TableEntry {
            index,
            kind,
            stored_len,
            original_len,
            original_offset: self.original_offset,
        };
        self.original_offset = self.original_offset.saturating_add(u64::from(original_len));
        self.next = index.saturating_add(1);
        Ok(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockSize, ContextOverlap, MixerRule, ModelHash};
    use crate::{Container, compress_zstd};

    fn bs(n: u32) -> BlockSize {
        BlockSize::new(n).unwrap()
    }

    /// Lays out the parts of a stream writer as the caller does.
    fn assemble(params: ContainerParams, data: &[u8], blocks: &[Block], known: bool) -> Vec<u8> {
        let expected = known.then_some(data.len() as u64);
        let mut w = StreamWriter::new(params, expected).unwrap();
        let (mut table, mut stored_bytes) = (Vec::new(), Vec::new());
        for (piece, block) in params.block_size.split(data).zip(blocks) {
            let (e, s) = w.push(piece, block).unwrap();
            table.extend_from_slice(&e.to_bytes());
            stored_bytes.extend_from_slice(s);
        }
        let (_, header, total) = w.finish(params.profile, params.coding).unwrap();
        let mut out = header.to_vec();
        out.extend_from_slice(&table);
        out.extend_from_slice(&stored_bytes);
        assert_eq!(out.len() as u64, total);
        out
    }

    #[test]
    fn stream_writer_gives_the_bytes_of_build() {
        let data: Vec<u8> = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabc".repeat(10);
        let z = compress_zstd(&data[100..200]).unwrap();
        let blocks =
            [Block::Raw, Block::Zstd(z), Block::Coded(vec![1, 2, 3]), Block::Mixed(vec![9])];
        let coding = CodingParts::with_mixer(MixerRule::V1).with_overlap(ContextOverlap::new(7));
        let params = ContainerParams::new(ModelHash::from_bytes([3; 32]), bs(100))
            .with_profile(Profile::Logs)
            .with_coding(coding);
        let blocks: Vec<Block> =
            blocks.into_iter().chain(core::iter::repeat_n(Block::Raw, 1)).collect();
        let built = Container::build(&params, &data, &blocks).unwrap();
        assert_eq!(assemble(params, &data, &blocks, true), built);
        assert_eq!(assemble(params, &data, &blocks, false), built);
    }

    #[test]
    fn stream_writer_of_empty_data_gives_the_header_only() {
        let params = ContainerParams::new(ModelHash::RAW_ONLY, bs(65_536));
        let built = Container::build(&params, &[], &[]).unwrap();
        assert_eq!(assemble(params, &[], &[], true), built);
        assert_eq!(built.len(), HEADER_LEN);
    }

    #[test]
    fn stream_writer_refuses_bad_block_lengths() {
        let params = ContainerParams::new(ModelHash::RAW_ONLY, bs(4));
        let mut w = StreamWriter::new(params, None).unwrap();
        assert!(matches!(w.push(b"", &Block::Raw), Err(FormatError::BlockLengthMismatch { .. })));
        assert!(matches!(
            w.push(b"abcde", &Block::Raw),
            Err(FormatError::BlockLengthMismatch { .. })
        ));
        w.push(b"ab", &Block::Raw).unwrap();
        assert!(matches!(
            w.push(b"cd", &Block::Raw),
            Err(FormatError::BlockLengthMismatch { block: 1, .. })
        ));
    }

    #[test]
    fn stream_writer_refuses_a_length_that_is_not_the_expected_length() {
        let params = ContainerParams::new(ModelHash::RAW_ONLY, bs(4));
        let mut w = StreamWriter::new(params, Some(6)).unwrap();
        w.push(b"abcd", &Block::Raw).unwrap();
        assert_eq!(
            w.clone().finish(Profile::Text, CodingParts::NONE).map(|_| ()),
            Err(FormatError::InputLengthMismatch { expected: 6, found: 4 })
        );
        assert_eq!(
            w.push(b"efg", &Block::Raw).map(|_| ()),
            Err(FormatError::InputLengthMismatch { expected: 6, found: 7 })
        );
        let mut w = StreamWriter::new(params, Some(6)).unwrap();
        assert!(StreamWriter::new(params, Some(u64::MAX)).is_err());
        w.push(b"abcd", &Block::Raw).unwrap();
        w.push(b"ef", &Block::Raw).unwrap();
        assert!(w.finish(Profile::Text, CodingParts::NONE).is_ok());
    }

    #[test]
    fn finish_refuses_mixed_blocks_without_the_mixer() {
        let coding = CodingParts::with_mixer(MixerRule::V1);
        let params = ContainerParams::new(ModelHash::from_bytes([1; 32]), bs(4))
            .with_profile(Profile::Logs)
            .with_coding(coding);
        let mut w = StreamWriter::new(params, None).unwrap();
        let m = Block::Mixed(vec![1]);
        w.push(b"abcd", &m).unwrap();
        assert_eq!(
            w.clone().finish(Profile::Text, CodingParts::NONE).map(|_| ()),
            Err(FormatError::MixedWithoutCodingPart { block: 0 })
        );
        assert_eq!(
            w.clone().finish(Profile::Text, coding).map(|_| ()),
            Err(FormatError::CodingPartsNotInProfile { profile: 0, parts: 1 })
        );
        assert!(w.finish(Profile::Logs, coding).is_ok());
    }

    #[test]
    fn table_reader_gives_the_blocks_of_parse() {
        let data: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        let blocks = vec![Block::Raw, Block::Coded(vec![5; 10]), Block::Raw, Block::Coded(vec![])];
        let params = ContainerParams::new(ModelHash::from_bytes([2; 32]), bs(300));
        let bytes = Container::build(&params, &data, &blocks).unwrap();
        let parsed = Container::parse(&bytes).unwrap();
        let header = Header::from_bytes(&bytes).unwrap();
        assert_eq!(header, *parsed.header());
        assert_eq!(Header::len_from_prefix(&bytes[..6]).unwrap(), HEADER_LEN);
        let mut r = TableReader::new(&header).unwrap();
        let mut at = usize::try_from(r.data_offset().unwrap()).unwrap();
        let mut table = &bytes[HEADER_LEN..];
        for b in parsed.blocks() {
            let (e, rest) = table.split_first_chunk::<5>().unwrap();
            table = rest;
            let e = r.next_entry(e).unwrap();
            let len = e.stored_len() as usize;
            assert_eq!(e.block_ref(&bytes[at..at + len]).unwrap(), *b);
            assert!(e.block_ref(&vec![0; len + 1]).is_err());
            at += len;
        }
        assert_eq!(r.remaining(), 0);
        assert_eq!(r.container_len(), bytes.len() as u64);
        assert!(matches!(r.next_entry(&[0; 5]), Err(FormatError::WrongBlockCount { .. })));
    }

    #[test]
    fn table_reader_refuses_a_bad_entry() {
        let params = ContainerParams::new(ModelHash::RAW_ONLY, bs(4));
        let bytes = Container::build(&params, b"abcdef", &[Block::Raw, Block::Raw]).unwrap();
        let header = Header::from_bytes(&bytes).unwrap();
        let mut r = TableReader::new(&header).unwrap();
        // A coded block in the raw-only mode.
        assert_eq!(
            r.next_entry(&[1, 1, 0, 0, 0]),
            Err(FormatError::CodedBlockInRawOnly { block: 0 })
        );
        assert_eq!(
            r.next_entry(&[0, 3, 0, 0, 0]),
            Err(FormatError::RawLengthMismatch { block: 0, original_len: 4, stored_len: 3 })
        );
    }

    #[test]
    fn header_prefix_gives_the_length_of_each_version() {
        assert_eq!(Header::len_from_prefix(b"TACO\x01\x00").unwrap(), 88);
        assert_eq!(Header::len_from_prefix(b"TACO\x02\x00").unwrap(), 104);
        assert_eq!(
            Header::len_from_prefix(b"TACO\x03\x00"),
            Err(FormatError::UnsupportedVersion { version: 3 })
        );
        assert_eq!(Header::len_from_prefix(b"ZACO\x02\x00"), Err(FormatError::BadMagic));
        assert!(matches!(Header::len_from_prefix(b"TAC"), Err(FormatError::Truncated { .. })));
    }

    #[test]
    fn checksum_builder_equals_the_checksum_of_all_parts() {
        let mut b = ChecksumBuilder::new();
        b.update(b"hello ");
        b.update(b"");
        b.update(b"world");
        assert_eq!(b.finish(), Checksum::of(b"hello world"));
        assert_eq!(ChecksumBuilder::new().finish(), Checksum::of(&[]));
    }
}
