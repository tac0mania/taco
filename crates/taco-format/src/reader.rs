//! The reader: it checks untrusted container bytes and gives the stored bytes of each block as
//! slices of the input.
//!
//! The rules are in the crate documentation. The reader allocates one [`BlockRef`] for each
//! block only after it checks that the block table is inside the input. Thus a header field
//! cannot cause an allocation larger than the input.

use crate::error::{FormatError, Result};
use crate::layout::{HEADER_LEN_U64, OriginalLens, expected_block_count, header_and_table_len};
use crate::types::{BlockKind, BlockSize, Checksum, ModelHash};
use crate::{HEADER_LEN, MAGIC, TABLE_ENTRY_LEN, VERSION};

/// The fields of a valid version 1 header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    model_hash: ModelHash,
    original_len: u64,
    checksum: Checksum,
    block_size: BlockSize,
    /// INVARIANT: `ceil(original_len / block_size)`.
    block_count: u32,
}

impl Header {
    /// The format version of the container. This crate reads version [`VERSION`] only.
    #[must_use]
    pub const fn version(&self) -> u16 {
        VERSION
    }

    /// The hash of the model that coded the blocks, or [`ModelHash::RAW_ONLY`].
    #[must_use]
    pub const fn model_hash(&self) -> ModelHash {
        self.model_hash
    }

    /// True if the decoder needs a model: the model hash is not [`ModelHash::RAW_ONLY`].
    ///
    /// A container with a model hash needs that model even if all its blocks are raw (plan §3
    /// rule 3).
    #[must_use]
    pub fn needs_model(&self) -> bool {
        !self.model_hash.is_raw_only()
    }

    /// The number of bytes of the original data. The parser does not limit this value. A
    /// coded block of a few bytes can hold many original bytes, so do not allocate this length
    /// before the blocks decode.
    #[must_use]
    pub const fn original_len(&self) -> u64 {
        self.original_len
    }

    /// The SHA-256 of the original data.
    #[must_use]
    pub const fn checksum(&self) -> Checksum {
        self.checksum
    }

    /// The number of original bytes in each block. The last block can be shorter.
    #[must_use]
    pub const fn block_size(&self) -> BlockSize {
        self.block_size
    }

    /// The number of blocks: `ceil(original_len / block_size)`.
    #[must_use]
    pub const fn block_count(&self) -> u32 {
        self.block_count
    }

    /// Refuses a container that needs a different model than the loaded model (plan §3
    /// rule 3). A container in the raw-only mode needs no model, so it passes with any
    /// `loaded` hash. Call this before the blocks decode. Cost: 64 byte compares.
    ///
    /// # Errors
    ///
    /// [`FormatError::ModelHashMismatch`], with both hashes, if the hashes are different.
    pub fn verify_model_hash(&self, loaded: &ModelHash) -> Result<()> {
        if self.model_hash.is_raw_only() || self.model_hash == *loaded {
            Ok(())
        } else {
            Err(FormatError::ModelHashMismatch { container: self.model_hash, loaded: *loaded })
        }
    }

    /// Verifies the restored data against the original length and the checksum (plan §3
    /// rule 4). A failure is an error, never a silent wrong output. Cost: one SHA-256 pass
    /// over `restored`.
    ///
    /// # Errors
    ///
    /// - [`FormatError::RestoredLengthMismatch`] if `restored` does not have the original
    ///   length. The function does not calculate the SHA-256 then.
    /// - [`FormatError::ChecksumMismatch`] if the SHA-256 of `restored` is different.
    pub fn verify_checksum(&self, restored: &[u8]) -> Result<()> {
        let found = u64::try_from(restored.len())
            .map_err(|_| FormatError::LengthOverflow { part: "restored data" })?;
        if found != self.original_len {
            return Err(FormatError::RestoredLengthMismatch { expected: self.original_len, found });
        }
        let found = Checksum::of(restored);
        if found == self.checksum {
            Ok(())
        } else {
            Err(FormatError::ChecksumMismatch { expected: self.checksum, found })
        }
    }
}

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
    /// bytes. For a coded block, these are the bytes for the range decoder, and they are
    /// fewer than the original bytes.
    #[must_use]
    pub const fn stored(&self) -> &'a [u8] {
        self.stored
    }
}

/// A valid version 1 container: its header and its blocks. The blocks borrow the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Container<'a> {
    header: Header,
    blocks: Vec<BlockRef<'a>>,
}

/// `x` as a `usize`, or a [`FormatError::LengthOverflow`] for `part`.
fn to_usize(x: u64, part: &'static str) -> Result<usize> {
    usize::try_from(x).map_err(|_| FormatError::LengthOverflow { part })
}

/// Takes the first `N` bytes of `rest`. The caller checked the length of the header, so this
/// does not fail on a valid path. It returns an error and does not panic if it does.
fn take<const N: usize>(rest: &mut &[u8]) -> Result<[u8; N]> {
    let (head, tail) = rest
        .split_first_chunk::<N>()
        .ok_or(FormatError::LengthOverflow { part: "header field" })?;
    *rest = tail;
    Ok(*head)
}

/// Reads and checks the 88 bytes of the header. The magic and the version are already
/// checked.
fn read_header(bytes: &[u8; HEADER_LEN]) -> Result<Header> {
    let mut rest: &[u8] = bytes;
    let _magic: [u8; 4] = take(&mut rest)?;
    let _version: [u8; 2] = take(&mut rest)?;
    let flags = u16::from_le_bytes(take(&mut rest)?);
    if flags != 0 {
        return Err(FormatError::UnknownFlags { flags });
    }
    let model_hash = ModelHash::from_bytes(take(&mut rest)?);
    let original_len = u64::from_le_bytes(take(&mut rest)?);
    let checksum = Checksum::from_bytes(take(&mut rest)?);
    let block_size = BlockSize::new(u32::from_le_bytes(take(&mut rest)?))?;
    let block_count = u32::from_le_bytes(take(&mut rest)?);
    let expected = expected_block_count(original_len, block_size);
    if expected != u64::from(block_count) {
        return Err(FormatError::WrongBlockCount {
            original_len,
            block_size: block_size.get(),
            expected,
            found: u64::from(block_count),
        });
    }
    Ok(Header { model_hash, original_len, checksum, block_size, block_count })
}

/// Checks the magic and the version. A reader refuses an unknown version before it reads any
/// other field, because another version can have another header.
fn check_magic_and_version(bytes: &[u8], found: u64) -> Result<()> {
    let truncated = FormatError::Truncated { part: "header", needed: HEADER_LEN_U64, found };
    match bytes.first_chunk::<4>() {
        Some(magic) if *magic == MAGIC => {}
        Some(_) => return Err(FormatError::BadMagic),
        None if MAGIC.starts_with(bytes) => return Err(truncated),
        None => return Err(FormatError::BadMagic),
    }
    let version = bytes.get(4..6).and_then(|v| v.first_chunk::<2>()).ok_or(truncated)?;
    let version = u16::from_le_bytes(*version);
    if version == VERSION { Ok(()) } else { Err(FormatError::UnsupportedVersion { version }) }
}

/// Decodes and checks one entry of the block table.
fn entry(
    index: u32,
    bytes: &[u8; TABLE_ENTRY_LEN],
    original_len: u32,
    raw_only: bool,
) -> Result<(BlockKind, u32)> {
    let [kind, l0, l1, l2, l3] = *bytes;
    let stored_len = u32::from_le_bytes([l0, l1, l2, l3]);
    let kind =
        BlockKind::from_u8(kind).ok_or(FormatError::UnknownBlockKind { block: index, kind })?;
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
        BlockKind::Raw | BlockKind::Coded => Ok((kind, stored_len)),
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
    ///   [`FormatError::UnknownFlags`] for an input that is not a version 1 container.
    /// - [`FormatError::Truncated`] if the header or the block table is not complete.
    /// - [`FormatError::ZeroBlockSize`] and [`FormatError::WrongBlockCount`] for a header that
    ///   does not agree with itself.
    /// - [`FormatError::UnknownBlockKind`], [`FormatError::RawLengthMismatch`],
    ///   [`FormatError::CodedNotSmaller`] and [`FormatError::CodedBlockInRawOnly`] for a bad
    ///   entry of the block table.
    /// - [`FormatError::WrongTotalLength`] if the input does not end at the end of the last
    ///   block.
    /// - [`FormatError::LengthOverflow`] if a length does not fit in `u64` or `usize`.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let found = u64::try_from(bytes.len())
            .map_err(|_| FormatError::LengthOverflow { part: "container" })?;
        check_magic_and_version(bytes, found)?;
        let (header, rest) = bytes
            .split_first_chunk::<HEADER_LEN>()
            .ok_or(FormatError::Truncated { part: "header", needed: HEADER_LEN_U64, found })?;
        let header = read_header(header)?;

        let table_end = header_and_table_len(u64::from(header.block_count))?;
        if table_end > found {
            return Err(FormatError::Truncated { part: "block table", needed: table_end, found });
        }
        // INVARIANT: table_end <= found, so the subtraction cannot wrap.
        let table_len = to_usize(table_end - HEADER_LEN_U64, "block table")?;
        let (table, mut payload) = rest
            .split_at_checked(table_len)
            .ok_or(FormatError::Truncated { part: "block table", needed: table_end, found })?;
        let (table, _) = table.as_chunks::<TABLE_ENTRY_LEN>();
        let raw_only = header.model_hash.is_raw_only();
        let lens = || OriginalLens::new(header.original_len, header.block_size);

        // Pass 1: check each entry and add the stored lengths, before any allocation.
        let mut expected = table_end;
        for (index, (e, len)) in (0..header.block_count).zip(table.iter().zip(lens())) {
            let (_, stored_len) = entry(index, e, len, raw_only)?;
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
        let block_size = u64::from(header.block_size.get());
        let mut original_offset = 0u64;
        for (index, (e, len)) in (0..header.block_count).zip(table.iter().zip(lens())) {
            let (kind, stored_len) = entry(index, e, len, raw_only)?;
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
