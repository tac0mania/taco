//! Helpers for the integration tests: an integer PRNG, a toy block coder, a full fold and
//! unfold over the container (raw, coded and zstd blocks), and a builder for hand-made (also
//! invalid) containers of version 1 and version 2.

#![allow(dead_code, reason = "each test file uses a different part")]
#![allow(clippy::unwrap_used, reason = "test code: a failure must stop the test")]

use taco_format::{
    Block, BlockKind, BlockSize, Container, ContainerParams, FormatError, ModelHash,
};

/// The splitmix64 generator (Steele, Lea, Flood, 2014). Integer only, so it is the same on
/// every machine.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A block size, for tests.
pub(crate) fn bs(n: u32) -> BlockSize {
    BlockSize::new(n).unwrap()
}

/// The hash of the "model" of the tests.
pub(crate) const MODEL: ModelHash = ModelHash::from_bytes([0x5A; 32]);

/// A toy coder in place of the model and the range coder: run-length pairs (count 1..=255,
/// value). It makes runs smaller and other data larger, so both block kinds occur.
pub(crate) fn toy_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let v = data[i];
        let mut n = 1;
        while i + n < data.len() && data[i + n] == v && n < 255 {
            n += 1;
        }
        out.push(u8::try_from(n).unwrap());
        out.push(v);
        i += n;
    }
    out
}

/// The decoder of [`toy_encode`]. It refuses a count of 0, an odd length and an output that
/// does not have `original_len` bytes, as a real decoder refuses the wrong symbol count.
pub(crate) fn toy_decode(stored: &[u8], original_len: usize) -> Option<Vec<u8>> {
    let (pairs, odd) = stored.as_chunks::<2>();
    if !odd.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for &[n, v] in pairs {
        if n == 0 || out.len() + usize::from(n) > original_len {
            return None;
        }
        out.extend(std::iter::repeat_n(v, usize::from(n)));
    }
    (out.len() == original_len).then_some(out)
}

/// Compresses `data` with the toy coder, zstd and the raw fallback: the smallest form of each
/// block (ADR 0011 §4). In the raw-only mode, the candidates are raw and zstd.
pub(crate) fn fold(data: &[u8], block_size: BlockSize, model: ModelHash) -> Vec<u8> {
    let blocks: Vec<Block> = block_size
        .split(data)
        .map(|b| {
            let coded = (!model.is_raw_only()).then(|| toy_encode(b));
            let zstd = taco_format::compress_zstd(b).unwrap();
            Block::choose_smallest(b.len(), coded, Some(zstd))
        })
        .collect();
    Container::build(&ContainerParams::new(model, block_size), data, &blocks).unwrap()
}

/// Phrases of a small vocabulary: zstd makes them smaller, and the toy coder does not.
pub(crate) fn phrases(seed: u64, len: usize) -> Vec<u8> {
    const WORDS: [&[u8]; 8] =
        [b"error ", b"info ", b"user=42 ", b"GET /index.html ", b"200 ", b"ms ", b"\n", b"ok "];
    let mut s = seed;
    let mut out = Vec::with_capacity(len + 16);
    while out.len() < len {
        out.extend_from_slice(WORDS[usize::try_from(splitmix64(&mut s) % 8).unwrap()]);
    }
    out.truncate(len);
    out
}

/// Why an unfold failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UnfoldError {
    /// The container or a check refused the input.
    Format(FormatError),
    /// The toy decoder refused a coded block.
    Decode(u32),
}

impl From<FormatError> for UnfoldError {
    fn from(e: FormatError) -> Self {
        Self::Format(e)
    }
}

/// Decompresses as the `taco` crate will: parse, check the model hash, decode each block,
/// verify the checksum.
pub(crate) fn unfold(bytes: &[u8], loaded: ModelHash) -> Result<Vec<u8>, UnfoldError> {
    let c = Container::parse(bytes)?;
    c.header().verify_model_hash(&loaded)?;
    let mut out = Vec::new();
    for b in c.blocks() {
        let len = usize::try_from(b.original_len()).unwrap();
        match b.kind() {
            BlockKind::Raw => out.extend_from_slice(b.stored()),
            BlockKind::Coded => {
                out.extend(toy_decode(b.stored(), len).ok_or(UnfoldError::Decode(b.index()))?);
            }
            BlockKind::Zstd => b.decode_zstd_into(&mut out)?,
            _ => unreachable!("version 2 has three kinds"),
        }
    }
    c.header().verify_checksum(&out)?;
    Ok(out)
}

/// The fields of a hand-made container. The default is a valid version 1 container of 0
/// blocks. [`Craft::v2`] makes it a version 2 container.
#[derive(Clone, Debug)]
pub(crate) struct Craft {
    pub(crate) magic: [u8; 4],
    pub(crate) version: u16,
    pub(crate) flags: u16,
    pub(crate) model_hash: [u8; 32],
    pub(crate) original_len: u64,
    pub(crate) checksum: [u8; 32],
    pub(crate) block_size: u32,
    pub(crate) block_count: u32,
    /// The bytes after the block count and before the block table: empty in version 1, the
    /// 16 bytes of the profile section in version 2.
    pub(crate) section: Vec<u8>,
    pub(crate) entries: Vec<(u8, u32)>,
    pub(crate) payload: Vec<u8>,
}

impl Default for Craft {
    fn default() -> Self {
        Self {
            magic: *b"TACO",
            version: 1,
            flags: 0,
            model_hash: *MODEL.as_bytes(),
            original_len: 0,
            checksum: [0; 32],
            block_size: 16,
            block_count: 0,
            section: Vec::new(),
            entries: Vec::new(),
            payload: Vec::new(),
        }
    }
}

impl Craft {
    /// The same container as version 2, with the profile `profile` and all reserved fields 0.
    pub(crate) fn v2(self, profile: u8) -> Self {
        let mut section = vec![0; 16];
        section[0] = profile;
        Self { version: 2, section, ..self }
    }

    /// The bytes of the container, with no checks.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.model_hash);
        out.extend_from_slice(&self.original_len.to_le_bytes());
        out.extend_from_slice(&self.checksum);
        out.extend_from_slice(&self.block_size.to_le_bytes());
        out.extend_from_slice(&self.block_count.to_le_bytes());
        out.extend_from_slice(&self.section);
        for &(kind, len) in &self.entries {
            out.push(kind);
            out.extend_from_slice(&len.to_le_bytes());
        }
        out.extend_from_slice(&self.payload);
        out
    }
}
