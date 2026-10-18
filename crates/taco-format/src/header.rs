//! The header of a container: its fields, the reader of version 1 and version 2, the writer of
//! version 2, and the two checks that the caller does (model hash and checksum).
//!
//! The first 88 bytes have the same layout in both versions. Version 2 adds 16 bytes (ADR
//! 0011 §2): the profile (ADR 0012) and the fields that ADR 0010 reserves for the coding parts
//! of the `logs` profile.

use crate::error::{FormatError, Result};
use crate::layout::{HEADER_LEN_V1_U64, expected_block_count, header_len};
use crate::types::{BlockSize, Checksum, ModelHash, Profile};
use crate::{HEADER_LEN, HEADER_LEN_V1, MAGIC, VERSION, VERSION_1};

/// The length of the coding parameters field of the version 2 header, in bytes.
const CODING_PARAMS_LEN: usize = 12;

/// The fields of a valid header of version 1 or version 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// INVARIANT: [`VERSION_1`] or [`VERSION`].
    version: u16,
    profile: Profile,
    model_hash: ModelHash,
    original_len: u64,
    checksum: Checksum,
    block_size: BlockSize,
    /// INVARIANT: `ceil(original_len / block_size)`.
    block_count: u32,
}

impl Header {
    /// The header of version [`VERSION`] that the writer makes. The caller checked that
    /// `block_count` is `ceil(original_len / block_size)`.
    pub(crate) const fn new_v2(
        profile: Profile,
        model_hash: ModelHash,
        original_len: u64,
        checksum: Checksum,
        block_size: BlockSize,
        block_count: u32,
    ) -> Self {
        Self {
            version: VERSION,
            profile,
            model_hash,
            original_len,
            checksum,
            block_size,
            block_count,
        }
    }

    /// The format version of the container: 1 or 2.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// The length of the header in bytes: 88 for version 1, 104 for version 2. The block
    /// table starts at this offset.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        header_len(self.version)
    }

    /// The profile of the container (ADR 0012). A version 1 container has the profile
    /// [`Profile::Text`].
    #[must_use]
    pub const fn profile(&self) -> Profile {
        self.profile
    }

    /// The hash of the model that coded the blocks, or [`ModelHash::RAW_ONLY`].
    #[must_use]
    pub const fn model_hash(&self) -> ModelHash {
        self.model_hash
    }

    /// True if the decoder needs a model: the model hash is not [`ModelHash::RAW_ONLY`].
    ///
    /// A container with a model hash needs that model even if all its blocks are raw or zstd
    /// (plan §3 rule 3).
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

    /// Appends the header bytes to `out`: 88 bytes for version 1, 104 bytes for version 2.
    /// The flags, the coding parts and the reserved fields are zero.
    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(self.model_hash.as_bytes());
        out.extend_from_slice(&self.original_len.to_le_bytes());
        out.extend_from_slice(self.checksum.as_bytes());
        out.extend_from_slice(&self.block_size.get().to_le_bytes());
        out.extend_from_slice(&self.block_count.to_le_bytes());
        if self.version != VERSION_1 {
            out.push(self.profile.as_u8());
            out.push(0);
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&[0; CODING_PARAMS_LEN]);
        }
    }

    /// Parses and checks the header at the start of `bytes`. Gives the header and the bytes
    /// after it.
    ///
    /// # Errors
    ///
    /// The header errors of [`crate::Container::parse`].
    pub(crate) fn parse(bytes: &[u8], found: u64) -> Result<(Self, &[u8])> {
        let version = check_magic_and_version(bytes, found)?;
        if version == VERSION_1 {
            let (head, rest) = split_header::<HEADER_LEN_V1>(bytes, found)?;
            Ok((read_common(version, head)?, rest))
        } else {
            let (head, rest) = split_header::<HEADER_LEN>(bytes, found)?;
            let (common, section) = head
                .split_first_chunk::<HEADER_LEN_V1>()
                .ok_or(FormatError::LengthOverflow { part: "header field" })?;
            let mut header = read_common(version, common)?;
            header.profile = read_v2_section(section)?;
            Ok((header, rest))
        }
    }
}

/// Splits the first `N` bytes of `bytes` off, or gives a [`FormatError::Truncated`] error for
/// a header of `N` bytes.
fn split_header<const N: usize>(bytes: &[u8], found: u64) -> Result<(&[u8; N], &[u8])> {
    bytes.split_first_chunk::<N>().ok_or(FormatError::Truncated {
        part: "header",
        // INVARIANT: N is 88 or 104, so it fits.
        needed: N as u64,
        found,
    })
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

/// Reads and checks the 88 bytes that version 1 and version 2 share. The magic and the version
/// are already checked. The profile is [`Profile::Text`]; version 2 reads its own.
fn read_common(version: u16, bytes: &[u8; HEADER_LEN_V1]) -> Result<Header> {
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
    Ok(Header {
        version,
        profile: Profile::Text,
        model_hash,
        original_len,
        checksum,
        block_size,
        block_count,
    })
}

/// Reads and checks the 16 bytes that version 2 adds (ADR 0011 §2): the profile, a reserved
/// byte, the coding parts and the coding parameters. Gives the profile.
fn read_v2_section(bytes: &[u8]) -> Result<Profile> {
    let mut rest = bytes;
    let [profile] = take::<1>(&mut rest)?;
    let profile = Profile::from_u8(profile).ok_or(FormatError::UnknownProfile { profile })?;
    let [reserved] = take::<1>(&mut rest)?;
    if reserved != 0 {
        return Err(FormatError::ReservedNotZero { field: "reserved byte 89" });
    }
    // NOTE: ADR 0010 (not accepted yet) defines the bits of the coding parts and the coding
    // parameters. Until then, the reader refuses every bit that is not zero.
    let parts = u16::from_le_bytes(take(&mut rest)?);
    if parts != 0 {
        return Err(FormatError::UnknownCodingParts { parts });
    }
    let params: [u8; CODING_PARAMS_LEN] = take(&mut rest)?;
    if params != [0; CODING_PARAMS_LEN] {
        return Err(FormatError::ReservedNotZero { field: "coding parameters" });
    }
    Ok(profile)
}

/// Checks the magic and the version, and gives the version. A reader refuses an unknown
/// version before it reads any other field, because another version can have another header.
fn check_magic_and_version(bytes: &[u8], found: u64) -> Result<u16> {
    // The version is not known yet: the shortest header (version 1) is the need.
    let truncated = FormatError::Truncated { part: "header", needed: HEADER_LEN_V1_U64, found };
    match bytes.first_chunk::<4>() {
        Some(magic) if *magic == MAGIC => {}
        Some(_) => return Err(FormatError::BadMagic),
        None if MAGIC.starts_with(bytes) => return Err(truncated),
        None => return Err(FormatError::BadMagic),
    }
    let version = bytes.get(4..6).and_then(|v| v.first_chunk::<2>()).ok_or(truncated)?;
    match u16::from_le_bytes(*version) {
        v @ (VERSION_1 | VERSION) => Ok(v),
        version => Err(FormatError::UnsupportedVersion { version }),
    }
}
