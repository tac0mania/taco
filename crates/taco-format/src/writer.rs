//! The writer: it lays out the header, the block table and the stored bytes of each block,
//! and it selects the stored form of a block (raw, coded or zstd).
//!
//! The caller codes the blocks. The writer does not run the model or the coder. It applies the
//! same rules as the reader, so every container that it writes parses again. The writer
//! writes version 2 only (ADR 0011 §3).

use crate::error::{FormatError, Result};
use crate::header::Header;
use crate::layout::{HEADER_LEN_U64, OriginalLens, block_count, header_and_table_len};
use crate::reader::Container;
use crate::types::{BlockKind, BlockSize, Checksum, ModelHash, Profile};
use crate::zstd_block::{check_frame, compress_zstd};

/// The parts of the header that the caller selects. The writer calculates the other fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContainerParams {
    /// The hash of the model that coded the blocks, or [`ModelHash::RAW_ONLY`].
    pub model_hash: ModelHash,
    /// The number of original bytes in each block.
    pub block_size: BlockSize,
    /// The profile of the container (ADR 0012). The default is [`Profile::Text`].
    pub profile: Profile,
}

impl ContainerParams {
    /// Makes the parameters of a container with the profile [`Profile::Text`].
    #[must_use]
    pub const fn new(model_hash: ModelHash, block_size: BlockSize) -> Self {
        Self { model_hash, block_size, profile: Profile::Text }
    }

    /// The same parameters with another profile.
    #[must_use]
    pub const fn with_profile(self, profile: Profile) -> Self {
        Self { profile, ..self }
    }
}

/// The stored form of one block, as the caller gives it to [`Container::build`].
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
    /// Store these bytes, one zstd frame of the original bytes ([`compress_zstd`]).
    Zstd(Vec<u8>),
}

impl Block {
    /// Selects raw or coded (plan §3 rule 6): [`Block::Coded`] if `coded` is smaller than the
    /// `original_len` bytes of the block, else [`Block::Raw`]. This is
    /// [`Block::choose_smallest`] with no zstd candidate.
    #[must_use]
    pub fn choose(coded: Vec<u8>, original_len: usize) -> Self {
        Self::choose_smallest(original_len, Some(coded), None)
    }

    /// Selects the smallest stored form of a block of `original_len` bytes (ADR 0011 §4).
    ///
    /// The candidates are raw (`original_len` bytes), `coded` and `zstd`. `None` means that the
    /// candidate does not exist, for example because the encoder stopped early. The smallest
    /// stored length wins. On a tie, the order is raw, then zstd, then coded: a block is
    /// coded or zstd only if it is smaller than raw, and zstd wins a tie with coded because it
    /// decodes without the model. The rule depends only on the lengths, so every machine
    /// selects the same form. Cost: two compares.
    #[must_use]
    pub fn choose_smallest(
        original_len: usize,
        coded: Option<Vec<u8>>,
        zstd: Option<Vec<u8>>,
    ) -> Self {
        let mut best = Self::Raw;
        let mut best_len = original_len;
        if let Some(z) = zstd.filter(|z| z.len() < best_len) {
            best_len = z.len();
            best = Self::Zstd(z);
        }
        if let Some(c) = coded.filter(|c| c.len() < best_len) {
            best = Self::Coded(c);
        }
        best
    }

    /// Adds the zstd candidate to a block that is already raw or coded, and selects the
    /// smallest form with [`Block::choose_smallest`]. `original` is the original bytes of the
    /// block. A zstd block stays as it is.
    ///
    /// Cost: one [`compress_zstd`] of `original`.
    ///
    /// # Errors
    ///
    /// [`FormatError::Zstd`] if the zstd library fails.
    pub fn with_zstd(self, original: &[u8]) -> Result<Self> {
        let coded = match self {
            Self::Raw => None,
            Self::Coded(c) => Some(c),
            Self::Zstd(z) => return Ok(Self::Zstd(z)),
        };
        Ok(Self::choose_smallest(original.len(), coded, Some(compress_zstd(original)?)))
    }

    /// The kind byte that the block table stores for this block.
    #[must_use]
    pub const fn kind(&self) -> BlockKind {
        match self {
            Self::Raw => BlockKind::Raw,
            Self::Coded(_) => BlockKind::Coded,
            Self::Zstd(_) => BlockKind::Zstd,
        }
    }

    /// The number of bytes that the container stores for this block, if the block has
    /// `original_len` original bytes.
    #[must_use]
    pub fn stored_len(&self, original_len: usize) -> usize {
        match self {
            Self::Raw => original_len,
            Self::Coded(b) | Self::Zstd(b) => b.len(),
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
        Block::Zstd(frame) => {
            let stored_len = to_u64(frame.len(), "zstd block")?;
            match u32::try_from(frame.len()) {
                Ok(len) if len < original_len => {
                    check_frame(frame, original_len, index)?;
                    Ok(Stored { kind: BlockKind::Zstd, bytes: frame, len })
                }
                _ => Err(FormatError::ZstdNotSmaller { block: index, original_len, stored_len }),
            }
        }
    }
}

impl Container<'_> {
    /// Writes a version 2 container for `original`, with one [`Block`] for each block of
    /// `params.block_size` bytes (the last block can be shorter).
    ///
    /// Use [`BlockSize::split`] to get the original bytes of each block, and
    /// [`Block::choose_smallest`] or [`Block::with_zstd`] to select the form of each block.
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
    /// - [`FormatError::CodedNotSmaller`] and [`FormatError::ZstdNotSmaller`] if a coded or
    ///   zstd block is not smaller than its original bytes. Use [`Block::choose_smallest`] to
    ///   store such a block raw.
    /// - [`FormatError::CorruptZstdBlock`] if a zstd block is not one zstd frame with the
    ///   original length as its content size. The writer does not decode the frame.
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
        let mut total = header_and_table_len(HEADER_LEN_U64, u64::from(count))?;
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

        let header = Header::new_v2(
            params.profile,
            params.model_hash,
            original_len,
            Checksum::of(original),
            params.block_size,
            count,
        );
        let mut out = Vec::with_capacity(total);
        header.write(&mut out);
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
    fn choose_smallest_takes_the_smallest_with_ties_to_raw_then_zstd() {
        let v = |n: usize| vec![0xAA; n];
        // Raw wins every tie.
        assert_eq!(Block::choose_smallest(3, Some(v(3)), Some(v(3))), Block::Raw);
        assert_eq!(Block::choose_smallest(3, None, None), Block::Raw);
        assert_eq!(Block::choose_smallest(3, Some(v(4)), Some(v(5))), Block::Raw);
        // The smallest wins.
        assert_eq!(Block::choose_smallest(9, Some(v(4)), Some(v(5))), Block::Coded(v(4)));
        assert_eq!(Block::choose_smallest(9, Some(v(6)), Some(v(5))), Block::Zstd(v(5)));
        assert_eq!(Block::choose_smallest(9, None, Some(v(8))), Block::Zstd(v(8)));
        assert_eq!(Block::choose_smallest(9, Some(v(8)), None), Block::Coded(v(8)));
        // Zstd wins a tie with coded.
        assert_eq!(Block::choose_smallest(9, Some(v(5)), Some(v(5))), Block::Zstd(v(5)));
        assert_eq!(Block::choose_smallest(1, Some(v(0)), Some(v(0))), Block::Zstd(v(0)));
    }

    #[test]
    fn with_zstd_adds_the_zstd_candidate() {
        let text = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc".repeat(4);
        let z = compress_zstd(&text).unwrap();
        assert!(z.len() < 40, "{}", z.len());
        assert_eq!(Block::Raw.with_zstd(&text).unwrap(), Block::Zstd(z.clone()));
        assert_eq!(Block::Coded(vec![1; 3]).with_zstd(&text).unwrap(), Block::Coded(vec![1; 3]));
        assert_eq!(Block::Coded(vec![1; 39]).with_zstd(&text).unwrap(), Block::Zstd(z.clone()));
        assert_eq!(Block::Zstd(vec![5]).with_zstd(&text).unwrap(), Block::Zstd(vec![5]));
        // zstd cannot make 3 bytes smaller: the frame header alone is longer.
        assert_eq!(Block::Raw.with_zstd(b"abc").unwrap(), Block::Raw);
        assert_eq!(
            Block::Zstd(z).stored_len(240),
            Block::Zstd(compress_zstd(&text).unwrap()).stored_len(0)
        );
        assert_eq!(Block::Raw.stored_len(240), 240);
    }

    #[test]
    fn every_field_is_at_its_offset() {
        let data = b"abcdefghij";
        let blocks = [Block::Raw, Block::Coded(vec![0xC0, 0xDE]), Block::Raw];
        let params = ContainerParams::new(model(), bs(4)).with_profile(Profile::Logs);
        let out = Container::build(&params, data, &blocks).unwrap();
        assert_eq!(&out[0..4], b"TACO");
        assert_eq!(&out[4..6], &[2, 0]);
        assert_eq!(&out[6..8], &[0, 0]);
        assert_eq!(&out[8..40], &[0x11; 32]);
        assert_eq!(&out[40..48], &10u64.to_le_bytes());
        assert_eq!(&out[48..80], Checksum::of(data).as_bytes());
        assert_eq!(&out[80..84], &4u32.to_le_bytes());
        assert_eq!(&out[84..88], &3u32.to_le_bytes());
        assert_eq!(out[88], 1, "profile");
        assert_eq!(&out[89..104], &[0; 15], "reserved byte, coding parts, coding parameters");
        assert_eq!(&out[104..109], &[0, 4, 0, 0, 0]);
        assert_eq!(&out[109..114], &[1, 2, 0, 0, 0]);
        assert_eq!(&out[114..119], &[0, 2, 0, 0, 0]);
        assert_eq!(&out[119..123], b"abcd");
        assert_eq!(&out[123..125], &[0xC0, 0xDE]);
        assert_eq!(&out[125..], b"ij");
    }

    #[test]
    fn empty_data_gives_the_header_only() {
        let out =
            Container::build(&ContainerParams::new(ModelHash::RAW_ONLY, bs(7)), &[], &[]).unwrap();
        assert_eq!(out.len(), 104);
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
    fn build_checks_a_zstd_block() {
        let data = vec![b'z'; 100];
        let frame = compress_zstd(&data).unwrap();
        let p = ContainerParams::new(ModelHash::RAW_ONLY, bs(100));
        let out = Container::build(&p, &data, &[Block::Zstd(frame.clone())]).unwrap();
        assert_eq!(out.len(), 104 + 5 + frame.len());
        assert_eq!(&out[104..105], &[2]);
        assert_eq!(
            Container::build(&p, &data, &[Block::Zstd(vec![0; 100])]),
            Err(FormatError::ZstdNotSmaller { block: 0, original_len: 100, stored_len: 100 })
        );
        assert!(matches!(
            Container::build(&p, &data, &[Block::Zstd(vec![0; 10])]),
            Err(FormatError::CorruptZstdBlock { block: 0, .. })
        ));
        // The frame holds 100 bytes, the block 99.
        assert!(matches!(
            Container::build(&p, &data[..99], &[Block::Zstd(frame)]),
            Err(FormatError::CorruptZstdBlock { block: 0, .. })
        ));
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
