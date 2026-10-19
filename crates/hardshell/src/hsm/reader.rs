//! The `.hsm` reader: it checks the full file and gives owned, validated parts.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::header::{Header, ModelConfig, TensorEntry, V2Constants};
use super::merges::{self, MergePair};
use super::{
    MAGIC, MAX_VOCAB, PREAMBLE_LEN, VERSION_1, VERSION_2, align64, is_valid_name, required_version,
};
use crate::error::{Error, Result};
use crate::layernorm::{LnConstants, MAX_BETA_FRAC};
use crate::softmax::MAX_P;
use crate::tensor::{DType, Element, Tensor};
use crate::types::{ModelHash, TokenId};

/// The values of one tensor, with their type.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorData {
    /// `i8` values.
    I8(Tensor<i8>),
    /// `i16` values.
    I16(Tensor<i16>),
    /// `i32` values.
    I32(Tensor<i32>),
    /// `u64` values.
    U64(Tensor<u64>),
}

impl TensorData {
    /// The element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::U64(_) => DType::U64,
        }
    }

    /// The dimensions.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::I8(t) => t.shape().dims(),
            Self::I16(t) => t.shape().dims(),
            Self::I32(t) => t.shape().dims(),
            Self::U64(t) => t.shape().dims(),
        }
    }

    /// Appends the values as little-endian bytes.
    pub(crate) fn write_le(&self, out: &mut Vec<u8>) {
        fn put<T: Element>(t: &Tensor<T>, out: &mut Vec<u8>) {
            for &v in t.data() {
                v.write_le(out);
            }
        }
        match self {
            Self::I8(t) => put(t, out),
            Self::I16(t) => put(t, out),
            Self::I32(t) => put(t, out),
            Self::U64(t) => put(t, out),
        }
    }

    /// The number of bytes of the values.
    pub(crate) fn byte_len(&self) -> Option<u64> {
        let numel: usize = self.dims().iter().product();
        u64::try_from(numel.checked_mul(self.dtype().size())?).ok()
    }
}

/// One tensor of the file: its name, its values and its rescale shift (if it holds rescale
/// multipliers).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredTensor {
    name: String,
    data: TensorData,
    shift: Option<u8>,
}

impl StoredTensor {
    /// The name in the header.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The values.
    #[must_use]
    pub fn data(&self) -> &TensorData {
        &self.data
    }

    /// The rescale shift `s`, present only on an `i32` tensor of multipliers.
    #[must_use]
    pub fn shift(&self) -> Option<u8> {
        self.shift
    }

    /// Takes the values out.
    #[must_use]
    pub fn into_data(self) -> TensorData {
        self.data
    }
}

/// The vocabulary: the bytes of each token id, and the BPE merges of the encoder if the file
/// has them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vocabulary {
    bytes: Vec<u8>,
    /// `ends[i]` is the end of token `i` in `bytes`; token `i` starts at `ends[i - 1]` (or 0).
    ends: Vec<usize>,
    /// The merges in rank order, if the file has a merges section.
    merges: Option<Box<[MergePair]>>,
}

impl Vocabulary {
    /// The number of tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    /// `true` if the vocabulary has no tokens. A file that the reader accepts has at least one.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// The bytes of a token, or `None` if the id is not in the vocabulary. Cost: no copy.
    #[must_use]
    pub fn token(&self, id: TokenId) -> Option<&[u8]> {
        let i = usize::try_from(id.get()).ok()?;
        let end = *self.ends.get(i)?;
        let start = match i.checked_sub(1) {
            Some(p) => *self.ends.get(p)?,
            None => 0,
        };
        self.bytes.get(start..end)
    }

    /// The BPE merges in rank order (rank 0 first), or `None` if the file has no merges
    /// section. Only the encoder uses them; the decoder needs only [`Vocabulary::token`].
    #[must_use]
    pub fn merges(&self) -> Option<&[MergePair]> {
        self.merges.as_deref()
    }

    /// Parses the vocabulary section: exactly `count` entries that fill `section`.
    fn parse(section: &[u8], count: u32) -> Result<Self> {
        let mut ends = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        let mut bytes = Vec::with_capacity(section.len());
        let mut pos = 0usize;
        for token in 0..count {
            let bad = |reason| Error::Vocabulary { token, reason };
            let len_bytes =
                section.get(pos..pos + 2).ok_or(bad("the length is outside the section"))?;
            let len = usize::from(u16::from_le_bytes(le_bytes(len_bytes)));
            pos += 2;
            let tok =
                section.get(pos..pos + len).ok_or(bad("the bytes are outside the section"))?;
            bytes.extend_from_slice(tok);
            ends.push(bytes.len());
            pos += len;
        }
        if pos != section.len() {
            return Err(Error::Vocabulary { token: count, reason: "the section has extra bytes" });
        }
        Ok(Self { bytes, ends, merges: None })
    }

    /// Encodes tokens in the format of the vocabulary section.
    pub(crate) fn encode(tokens: &[Vec<u8>]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for t in tokens {
            let len = u16::try_from(t.len())
                .map_err(|_| Error::TooLarge { reason: "a token has more than 65,535 bytes" })?;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(t);
        }
        Ok(out)
    }
}

/// A parsed and validated `.hsm` file. All tensors are owned copies.
#[derive(Clone, Debug)]
pub struct HsmFile {
    version: u32,
    config: ModelConfig,
    ln_f_bits: u8,
    ln_beta_frac: u8,
    vocabulary: Vocabulary,
    tensors: BTreeMap<String, StoredTensor>,
    order: Vec<String>,
    hash: ModelHash,
}

/// The first `N` bytes of `b` as an array (missing bytes are zero; callers give `N` bytes).
fn le_bytes<const N: usize>(b: &[u8]) -> [u8; N] {
    let mut a = [0u8; N];
    for (d, s) in a.iter_mut().zip(b) {
        *d = *s;
    }
    a
}

fn slice(bytes: &[u8], offset: u64, len: u64) -> Result<&[u8]> {
    let file_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let trunc = Error::Truncated { offset, needed: len, file_len };
    let end = offset.checked_add(len).ok_or(trunc.clone())?;
    if end > file_len {
        return Err(trunc);
    }
    let (s, e) = (usize::try_from(offset), usize::try_from(end));
    match (s, e) {
        (Ok(s), Ok(e)) => bytes.get(s..e).ok_or(trunc),
        _ => Err(trunc),
    }
}

/// Checks that the bytes from `from` to `to` are zero.
fn check_padding(bytes: &[u8], from: u64, to: u64) -> Result<()> {
    let pad = slice(bytes, from, to.saturating_sub(from))?;
    match pad.iter().position(|&b| b != 0) {
        Some(i) => Err(Error::NonZeroPadding { offset: from + u64::try_from(i).unwrap_or(0) }),
        None => Ok(()),
    }
}

// DETERMINISM: values are decoded from little-endian bytes with `from_le_bytes`, so the result
// does not depend on the byte order of the machine.
fn decode<T: Element>(dims: &[usize], raw: &[u8]) -> Result<Tensor<T>> {
    let data: Vec<T> = raw.chunks_exact(T::DTYPE.size()).map(T::read_le).collect();
    Tensor::new(dims, data)
}

fn check_config(c: &ModelConfig) -> Result<()> {
    let bad = |field: &str, reason| Err(Error::InvalidHeader { field: field.into(), reason });
    if !is_valid_name(&c.architecture) {
        return bad("model.architecture", "must have 1 to 128 characters from a-z 0-9 _ .");
    }
    if c.vocab_size == 0 || c.vocab_size > MAX_VOCAB {
        return bad("model.vocab_size", "must be in 1..=2^20");
    }
    if c.eot_token >= c.vocab_size {
        return bad("model.eot_token", "must be less than vocab_size");
    }
    LnConstants::new(c.constants.ln_k, c.constants.ln_j, c.constants.ln_eps)?;
    if c.constants.softmax_p > MAX_P {
        return bad("model.constants.softmax_p", "must be at most 30");
    }
    Ok(())
}

/// Checks the format version rules (module documentation, "Format version 2").
fn check_version(version: u32, v2: V2Constants, tensors: &[TensorEntry]) -> Result<()> {
    if v2.ln_f_bits.is_some_and(|b| b != 16) {
        return Err(Error::InvalidHeader {
            field: "model.constants.ln_f_bits".into(),
            reason: "must be 16 (omit the field for 8)",
        });
    }
    if v2.ln_beta_frac.is_some_and(|f| f == 0 || f > MAX_BETA_FRAC) {
        return Err(Error::InvalidHeader {
            field: "model.constants.ln_beta_frac".into(),
            reason: "must be in 1..=16 (omit the field for 0)",
        });
    }
    let entries = tensors.iter().map(|e| (e.name.as_str(), e.dtype.as_str()));
    if required_version(v2.any(), entries) != version {
        return Err(Error::InvalidHeader {
            field: "version".into(),
            reason: "a file is version 2 if and only if it has a version 2 constant or an i16 \
                     GELU table",
        });
    }
    Ok(())
}

fn read_tensor(bytes: &[u8], e: &TensorEntry, expected_offset: u64) -> Result<StoredTensor> {
    let mismatch = |reason| Error::TensorMismatch { name: e.name.clone(), reason };
    let dtype = DType::from_name(&e.dtype).ok_or(mismatch("unknown dtype"))?;
    let dims: Vec<usize> = e
        .shape
        .iter()
        .map(|&d| usize::try_from(d))
        .collect::<core::result::Result<_, _>>()
        .map_err(|_| mismatch("a dimension does not fit in usize"))?;
    let shape = crate::tensor::Shape::new(&dims)?;
    let want_len = shape
        .numel()
        .checked_mul(dtype.size())
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(mismatch("the byte length overflows"))?;
    if e.length != want_len {
        return Err(mismatch("length is not product(shape) * dtype size"));
    }
    if e.shift.is_some() && dtype != DType::I32 {
        return Err(mismatch("only an i32 tensor of rescale multipliers has a shift"));
    }
    if e.offset != expected_offset {
        return Err(Error::Layout {
            name: e.name.clone(),
            reason: "the offset is not align64 of the end of the section before",
        });
    }
    let raw = slice(bytes, e.offset, e.length)?;
    let data = match dtype {
        DType::I8 => TensorData::I8(decode(&dims, raw)?),
        DType::I16 => TensorData::I16(decode(&dims, raw)?),
        DType::I32 => TensorData::I32(decode(&dims, raw)?),
        DType::U64 => TensorData::U64(decode(&dims, raw)?),
    };
    Ok(StoredTensor { name: e.name.clone(), data, shift: e.shift })
}

impl HsmFile {
    /// Parses and validates a full `.hsm` file (format version 1 or 2).
    ///
    /// Cost: one pass over the file to copy the tensors, and one SHA-256 pass for the model
    /// hash. The result holds a copy of every tensor.
    ///
    /// # Errors
    /// An error for each rule in the module documentation: [`Error::BadMagic`],
    /// [`Error::UnsupportedVersion`], [`Error::Truncated`], [`Error::HeaderNotUtf8`],
    /// [`Error::HeaderJson`], [`Error::InvalidHeader`], [`Error::Layout`],
    /// [`Error::NonZeroPadding`], [`Error::Vocabulary`], [`Error::TensorMismatch`],
    /// [`Error::UnexpectedTensor`] (a duplicate name) and [`Error::InvalidShape`].
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let pre = slice(bytes, 0, PREAMBLE_LEN)?;
        if pre.get(0..8) != Some(&MAGIC[..]) {
            return Err(Error::BadMagic);
        }
        let version = u32::from_le_bytes(le_bytes(pre.get(8..12).unwrap_or(&[])));
        if version != VERSION_1 && version != VERSION_2 {
            return Err(Error::UnsupportedVersion { version });
        }
        let header_len = u64::from(u32::from_le_bytes(le_bytes(pre.get(12..16).unwrap_or(&[]))));
        let text = core::str::from_utf8(slice(bytes, PREAMBLE_LEN, header_len)?)
            .map_err(|_| Error::HeaderNotUtf8)?;
        let header: Header = serde_json::from_str(text).map_err(|e| Error::HeaderJson {
            line: e.line(),
            column: e.column(),
            message: e.to_string(),
        })?;
        let (config, v2) = header.model.into_parts();
        check_config(&config)?;
        check_version(version, v2, &header.tensors)?;
        let (ln_f_bits, ln_beta_frac) = (v2.ln_f_bits.unwrap_or(8), v2.ln_beta_frac.unwrap_or(0));

        let header_end = PREAMBLE_LEN + header_len;
        let layout_err = |name: &str, reason| Error::Layout { name: name.into(), reason };
        let vocab_offset = align64(header_end).ok_or(layout_err("vocabulary", "overflow"))?;
        if header.vocabulary.offset != vocab_offset {
            return Err(layout_err("vocabulary", "the offset is not align64(16 + header_len)"));
        }
        check_padding(bytes, header_end, vocab_offset)?;
        let section = slice(bytes, vocab_offset, header.vocabulary.length)?;
        let mut vocabulary = Vocabulary::parse(section, config.vocab_size)?;

        let mut cursor = vocab_offset + header.vocabulary.length;
        if let Some(m) = header.merges {
            let offset = align64(cursor).ok_or(layout_err("merges", "overflow"))?;
            if m.offset != offset {
                return Err(layout_err(
                    "merges",
                    "the offset is not align64 of the vocabulary end",
                ));
            }
            check_padding(bytes, cursor, offset.min(u64::try_from(bytes.len()).unwrap_or(0)))?;
            let section = slice(bytes, m.offset, m.length)?;
            vocabulary.merges = Some(merges::parse(section, &vocabulary)?);
            cursor = offset + m.length;
        }
        let mut tensors = BTreeMap::new();
        let mut order = Vec::with_capacity(header.tensors.len());
        for e in &header.tensors {
            if !is_valid_name(&e.name) {
                return Err(Error::InvalidHeader {
                    field: "tensors[].name".into(),
                    reason: "must have 1 to 128 characters from a-z 0-9 _ .",
                });
            }
            if tensors.contains_key(&e.name) {
                return Err(Error::UnexpectedTensor {
                    name: e.name.clone(),
                    reason: "the name occurs two times",
                });
            }
            let offset = align64(cursor).ok_or(layout_err(&e.name, "overflow"))?;
            check_padding(bytes, cursor, offset.min(u64::try_from(bytes.len()).unwrap_or(0)))?;
            let t = read_tensor(bytes, e, offset)?;
            cursor = offset + e.length;
            order.push(e.name.clone());
            tensors.insert(e.name.clone(), t);
        }
        if u64::try_from(bytes.len()).ok() != Some(cursor) {
            return Err(layout_err("file", "the file does not end at the end of the last section"));
        }
        let hash = ModelHash::from_bytes(Sha256::digest(bytes).into());
        Ok(Self { version, config, ln_f_bits, ln_beta_frac, vocabulary, tensors, order, hash })
    }

    /// The model description of the header.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The format version of the file: [`super::VERSION_1`] or [`super::VERSION_2`].
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The output bits of the final LayerNorm: 8 (version 1, and version 2 without the header
    /// constant `ln_f_bits`) or 16 (version 2 with `ln_f_bits = 16`).
    #[must_use]
    pub fn ln_f_bits(&self) -> u8 {
        self.ln_f_bits
    }

    /// The extra bits of every LayerNorm beta: 0 (version 1, and version 2 without the header
    /// constant `ln_beta_frac`) or 1 to 16 (`LayerNorm::with_beta_frac`).
    #[must_use]
    pub fn ln_beta_frac(&self) -> u8 {
        self.ln_beta_frac
    }

    /// The vocabulary.
    #[must_use]
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// The BPE merges in rank order, or `None` if the file has no merges section. The same as
    /// `self.vocabulary().merges()`.
    #[must_use]
    pub fn merges(&self) -> Option<&[MergePair]> {
        self.vocabulary.merges()
    }

    /// The SHA-256 of the full file.
    #[must_use]
    pub fn hash(&self) -> ModelHash {
        self.hash
    }

    /// The tensor names in file order (also the names that were taken out).
    #[must_use]
    pub fn tensor_names(&self) -> &[String] {
        &self.order
    }

    /// The tensor with this name, if the file has it and it was not taken out.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&StoredTensor> {
        self.tensors.get(name)
    }

    /// Takes the tensor with this name out of the file, without a copy.
    pub fn take_tensor(&mut self, name: &str) -> Option<StoredTensor> {
        self.tensors.remove(name)
    }

    /// The names of the tensors that were not taken out, in name order.
    pub fn remaining_tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Takes the vocabulary out of the file.
    #[must_use]
    pub fn into_vocabulary(self) -> Vocabulary {
        self.vocabulary
    }
}
