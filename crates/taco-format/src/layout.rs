//! The arithmetic of the layout: the block count, the original length of each block and the
//! container length. All of it is checked arithmetic.

use crate::error::{FormatError, Result};
use crate::types::BlockSize;
use crate::{HEADER_LEN, HEADER_LEN_V1, TABLE_ENTRY_LEN, VERSION_1};

/// `HEADER_LEN` (version 2) as a `u64`. INVARIANT: 104 fits.
pub(crate) const HEADER_LEN_U64: u64 = HEADER_LEN as u64;

/// `HEADER_LEN_V1` as a `u64`. INVARIANT: 88 fits.
pub(crate) const HEADER_LEN_V1_U64: u64 = HEADER_LEN_V1 as u64;

/// The header length of a version that this crate reads: 88 bytes for version 1, 104 bytes
/// for version 2.
pub(crate) const fn header_len(version: u16) -> usize {
    if version == VERSION_1 { HEADER_LEN_V1 } else { HEADER_LEN }
}

/// `TABLE_ENTRY_LEN` as a `u64`. INVARIANT: 5 fits.
pub(crate) const TABLE_ENTRY_LEN_U64: u64 = TABLE_ENTRY_LEN as u64;

/// `ceil(original_len / block_size)` as a `u64`. This cannot overflow: the result is at most
/// `original_len`.
pub(crate) fn expected_block_count(original_len: u64, block_size: BlockSize) -> u64 {
    original_len.div_ceil(u64::from(block_size.get()))
}

/// The number of blocks of `original_len` bytes in blocks of `block_size`.
///
/// # Errors
///
/// [`FormatError::TooManyBlocks`] if the count does not fit in the `u32` field.
pub(crate) fn block_count(original_len: u64, block_size: BlockSize) -> Result<u32> {
    let blocks = expected_block_count(original_len, block_size);
    u32::try_from(blocks).map_err(|_| FormatError::TooManyBlocks {
        original_len,
        block_size: block_size.get(),
        blocks,
    })
}

/// The original lengths of the blocks, in order: `block_size` each, the last one shorter.
///
/// The iterator gives `ceil(original_len / block_size)` values and stops. Each value is in
/// `1..=block_size`.
#[derive(Clone, Debug)]
pub(crate) struct OriginalLens {
    remaining: u64,
    block_size: u32,
}

impl OriginalLens {
    /// The lengths for `original_len` bytes in blocks of `block_size`.
    pub(crate) fn new(original_len: u64, block_size: BlockSize) -> Self {
        Self { remaining: original_len, block_size: block_size.get() }
    }
}

impl Iterator for OriginalLens {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.remaining == 0 {
            return None;
        }
        // INVARIANT: the value is at most block_size, so it fits in u32, and at most
        // remaining, so the subtraction cannot wrap.
        let len = u32::try_from(self.remaining).map_or(self.block_size, |r| r.min(self.block_size));
        self.remaining -= u64::from(len);
        Some(len)
    }
}

/// The length of a header of `header_len` bytes and a block table of `blocks` entries:
/// `header_len + 5 * blocks`.
///
/// # Errors
///
/// [`FormatError::LengthOverflow`] if the length does not fit in `u64`. With a `u32` block
/// count this cannot occur, but the function does not assume it.
pub(crate) fn header_and_table_len(header_len: u64, blocks: u64) -> Result<u64> {
    blocks
        .checked_mul(TABLE_ENTRY_LEN_U64)
        .and_then(|t| t.checked_add(header_len))
        .ok_or(FormatError::LengthOverflow { part: "block table" })
}

/// The largest container that this crate writes for `original_len` bytes in blocks of
/// `block_size`: every block raw, so `104 + 5 * block count + original_len` bytes (plan §9,
/// plan §3 rule 6). A coded or zstd block is smaller than its original bytes, so it does not
/// make the container larger. A version 1 container is 16 bytes smaller.
///
/// A caller can use it to bound an output buffer before it codes the blocks. Cost: a few
/// integer operations.
///
/// # Errors
///
/// - [`FormatError::TooManyBlocks`] if the block count does not fit in `u32`.
/// - [`FormatError::LengthOverflow`] if the length does not fit in `u64`.
pub fn worst_case_len(original_len: u64, block_size: BlockSize) -> Result<u64> {
    let blocks = block_count(original_len, block_size)?;
    header_and_table_len(HEADER_LEN_U64, u64::from(blocks))?
        .checked_add(original_len)
        .ok_or(FormatError::LengthOverflow { part: "container" })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bs(n: u32) -> BlockSize {
        BlockSize::new(n).unwrap()
    }

    #[test]
    fn header_length_depends_on_the_version() {
        assert_eq!(header_len(1), 88);
        assert_eq!(header_len(2), 104);
        assert_eq!(header_and_table_len(88, 3).unwrap(), 103);
        assert_eq!(header_and_table_len(104, 3).unwrap(), 119);
        assert!(header_and_table_len(104, u64::MAX).is_err());
    }

    #[test]
    fn block_count_rounds_up() {
        assert_eq!(block_count(0, bs(4)).unwrap(), 0);
        assert_eq!(block_count(1, bs(4)).unwrap(), 1);
        assert_eq!(block_count(4, bs(4)).unwrap(), 1);
        assert_eq!(block_count(5, bs(4)).unwrap(), 2);
        let max = u64::from(u32::MAX);
        assert_eq!(block_count(max * max, bs(u32::MAX)).unwrap(), u32::MAX);
    }

    #[test]
    fn block_count_refuses_more_than_u32() {
        let len = u64::from(u32::MAX) + 1;
        assert_eq!(
            block_count(len, bs(1)),
            Err(FormatError::TooManyBlocks { original_len: len, block_size: 1, blocks: len })
        );
        assert_eq!(block_count(u64::from(u32::MAX), bs(1)).unwrap(), u32::MAX);
    }

    #[test]
    fn original_lens_give_a_short_last_block() {
        assert_eq!(OriginalLens::new(10, bs(4)).collect::<Vec<_>>(), [4, 4, 2]);
        assert_eq!(OriginalLens::new(8, bs(4)).collect::<Vec<_>>(), [4, 4]);
        assert_eq!(OriginalLens::new(0, bs(4)).count(), 0);
        assert_eq!(OriginalLens::new(3, bs(u32::MAX)).collect::<Vec<_>>(), [3]);
        let huge = u64::from(u32::MAX) * 2 + 7;
        assert_eq!(
            OriginalLens::new(huge, bs(u32::MAX)).collect::<Vec<_>>(),
            [u32::MAX, u32::MAX, 7]
        );
    }

    #[test]
    fn worst_case_is_header_plus_five_bytes_per_block() {
        assert_eq!(worst_case_len(0, bs(1)).unwrap(), 104);
        assert_eq!(worst_case_len(1, bs(1)).unwrap(), 104 + 5 + 1);
        assert_eq!(worst_case_len(1000, bs(64)).unwrap(), 104 + 5 * 16 + 1000);
        assert!(matches!(
            worst_case_len(u64::MAX, bs(u32::MAX)),
            Err(FormatError::TooManyBlocks { .. })
        ));
        // u32::MAX blocks of u32::MAX bytes: the data fits in u64, the container does not.
        let max = u64::from(u32::MAX);
        assert_eq!(
            worst_case_len(max * max, bs(u32::MAX)),
            Err(FormatError::LengthOverflow { part: "container" })
        );
    }
}
