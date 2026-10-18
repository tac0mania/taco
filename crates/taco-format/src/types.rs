//! Newtypes for the values of the header: hashes, the block size, the block kind and the
//! profile.

use core::fmt;
use core::num::NonZeroU32;

use sha2::{Digest, Sha256};

use crate::error::{FormatError, Result};

/// Writes 32 bytes as lower-case hexadecimal, 64 characters.
fn write_hex(f: &mut fmt::Formatter<'_>, bytes: &[u8; 32]) -> fmt::Result {
    for b in bytes {
        write!(f, "{b:02x}")?;
    }
    Ok(())
}

/// The SHA-256 of a model file (plan §2, §8), as the container stores it.
///
/// All zero is the raw-only mode: the container needs no model. `hardshell` has its own
/// `ModelHash` with the same bytes. This crate does not depend on `hardshell`, so the `taco`
/// crate converts between the two with `from_bytes` and `as_bytes`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelHash([u8; 32]);

impl ModelHash {
    /// The model hash of the raw-only mode: 32 zero bytes. Every block is raw.
    pub const RAW_ONLY: Self = Self([0; 32]);

    /// Makes a model hash from its 32 bytes.
    #[must_use]
    #[inline]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The 32 bytes of the hash.
    #[must_use]
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// True if this is [`ModelHash::RAW_ONLY`]. Cost: 32 byte compares.
    #[must_use]
    pub fn is_raw_only(&self) -> bool {
        self.0 == [0; 32]
    }
}

impl From<[u8; 32]> for ModelHash {
    #[inline]
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl fmt::Display for ModelHash {
    /// Lower-case hexadecimal, 64 characters.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(f, &self.0)
    }
}

impl fmt::Debug for ModelHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelHash({self})")
    }
}

/// The SHA-256 of the original data (plan §3 rule 4).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Checksum([u8; 32]);

impl Checksum {
    /// Calculates the checksum of `data`. Cost: one SHA-256 pass over `data`.
    #[must_use]
    pub fn of(data: &[u8]) -> Self {
        Self(Sha256::digest(data).into())
    }

    /// Makes a checksum from its 32 bytes.
    #[must_use]
    #[inline]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The 32 bytes of the checksum.
    #[must_use]
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for Checksum {
    /// Lower-case hexadecimal, 64 characters.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(f, &self.0)
    }
}

impl fmt::Debug for Checksum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Checksum({self})")
    }
}

/// The number of original bytes in each block: at least 1, at most `u32::MAX`. The last block
/// can be shorter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockSize(NonZeroU32);

impl BlockSize {
    /// Makes a block size.
    ///
    /// # Errors
    ///
    /// [`FormatError::ZeroBlockSize`] if `bytes` is 0.
    #[inline]
    pub const fn new(bytes: u32) -> Result<Self> {
        match NonZeroU32::new(bytes) {
            Some(n) => Ok(Self(n)),
            None => Err(FormatError::ZeroBlockSize),
        }
    }

    /// The block size in bytes. It is at least 1.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// The block size as a `usize`. On a target where `usize` has fewer than 32 bits and the
    /// size does not fit, this is `usize::MAX`: such a block covers all data that can be in
    /// memory.
    #[must_use]
    #[inline]
    pub fn as_usize(self) -> usize {
        usize::try_from(self.0.get()).unwrap_or(usize::MAX)
    }

    /// Divides `data` into blocks of this size, the last block shorter. The blocks are the
    /// same as the blocks of a container with this block size. Cost: none until iteration.
    pub fn split(self, data: &[u8]) -> core::slice::Chunks<'_, u8> {
        data.chunks(self.as_usize())
    }
}

/// How a block stores its bytes: the `kind` byte of the block table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum BlockKind {
    /// The stored bytes are the original bytes. Kind byte 0.
    Raw = 0,
    /// The stored bytes are the output of the range coder. Kind byte 1.
    Coded = 1,
    /// The stored bytes are one zstd frame (RFC 8878) of the original bytes. Kind byte 2.
    /// Version 2 and later only (ADR 0011).
    Zstd = 2,
}

impl BlockKind {
    /// The kind byte in the block table.
    #[must_use]
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// The kind of a kind byte, or `None` if no version defines the byte. The reader also
    /// checks that the version of the container permits the kind ([`BlockKind::min_version`]).
    #[must_use]
    #[inline]
    pub const fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Raw),
            1 => Some(Self::Coded),
            2 => Some(Self::Zstd),
            _ => None,
        }
    }

    /// The first container version that permits this kind: 1 for raw and coded, 2 for zstd.
    #[must_use]
    #[inline]
    pub const fn min_version(self) -> u16 {
        match self {
            Self::Raw | Self::Coded => 1,
            Self::Zstd => 2,
        }
    }

    /// True if a block of this kind needs the model of the container to decode. Only a coded
    /// block needs it.
    #[must_use]
    #[inline]
    pub const fn needs_model(self) -> bool {
        matches!(self, Self::Coded)
    }
}

/// The profile of a container (ADR 0012): the kind of data that the encoder prepared the
/// container for. The header of version 2 stores it. A version 1 container has the profile
/// [`Profile::Text`].
///
/// The decoder does not select a model or coding parts from the profile. The model hash and the
/// coding parts of the header do that (ADR 0010, not accepted yet).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum Profile {
    /// Prose, documents, HTML and source code. Profile byte 0.
    #[default]
    Text = 0,
    /// Logs, JSON lines, CSV and telemetry. Profile byte 1.
    Logs = 1,
}

impl Profile {
    /// The profile byte in the header of version 2.
    #[must_use]
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// The profile of a profile byte, or `None` if version 2 does not define the byte.
    #[must_use]
    #[inline]
    pub const fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Text),
            1 => Some(Self::Logs),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_only_hash_is_all_zero() {
        assert!(ModelHash::RAW_ONLY.is_raw_only());
        assert_eq!(ModelHash::RAW_ONLY.as_bytes(), &[0; 32]);
        let mut b = [0; 32];
        b[31] = 1;
        assert!(!ModelHash::from_bytes(b).is_raw_only());
    }

    #[test]
    fn hashes_display_as_lower_case_hex() {
        let mut b = [0; 32];
        b[0] = 0xAB;
        b[31] = 0x01;
        let s = ModelHash::from_bytes(b).to_string();
        assert_eq!(s.len(), 64);
        assert!(s.starts_with("ab00"));
        assert!(s.ends_with("01"));
        assert_eq!(Checksum::from_bytes(b).to_string(), s);
        assert_eq!(format!("{:?}", ModelHash::from_bytes(b)), format!("ModelHash({s})"));
    }

    #[test]
    fn checksum_of_empty_data_is_the_sha256_of_nothing() {
        assert_eq!(
            Checksum::of(&[]).to_string(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn block_size_refuses_zero() {
        assert_eq!(BlockSize::new(0), Err(FormatError::ZeroBlockSize));
        assert_eq!(BlockSize::new(1).unwrap().get(), 1);
        assert_eq!(BlockSize::new(u32::MAX).unwrap().get(), u32::MAX);
    }

    #[test]
    fn split_gives_a_short_last_block() {
        let bs = BlockSize::new(3).unwrap();
        let lens: Vec<usize> = bs.split(&[0; 7]).map(<[u8]>::len).collect();
        assert_eq!(lens, [3, 3, 1]);
        assert_eq!(bs.split(&[]).count(), 0);
    }

    #[test]
    fn block_kind_bytes_are_zero_one_and_two_only() {
        assert_eq!(BlockKind::Raw.as_u8(), 0);
        assert_eq!(BlockKind::Coded.as_u8(), 1);
        assert_eq!(BlockKind::Zstd.as_u8(), 2);
        for b in 0..=255u8 {
            let k = BlockKind::from_u8(b);
            match b {
                0 => assert_eq!(k, Some(BlockKind::Raw)),
                1 => assert_eq!(k, Some(BlockKind::Coded)),
                2 => assert_eq!(k, Some(BlockKind::Zstd)),
                _ => assert_eq!(k, None),
            }
        }
        assert_eq!(BlockKind::Raw.min_version(), 1);
        assert_eq!(BlockKind::Coded.min_version(), 1);
        assert_eq!(BlockKind::Zstd.min_version(), 2);
        assert!(BlockKind::Coded.needs_model());
        assert!(!BlockKind::Raw.needs_model() && !BlockKind::Zstd.needs_model());
    }

    #[test]
    fn profile_bytes_are_zero_and_one_only() {
        assert_eq!(Profile::default(), Profile::Text);
        assert_eq!(Profile::Text.as_u8(), 0);
        assert_eq!(Profile::Logs.as_u8(), 1);
        for b in 0..=255u8 {
            let p = Profile::from_u8(b);
            match b {
                0 => assert_eq!(p, Some(Profile::Text)),
                1 => assert_eq!(p, Some(Profile::Logs)),
                _ => assert_eq!(p, None),
            }
        }
    }
}
