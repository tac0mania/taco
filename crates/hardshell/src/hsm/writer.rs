//! The `.hsm` writer: the reference for the layout of format version 1.

use super::header::{Header, ModelConfig, Section, TensorEntry};
use super::reader::{HsmFile, TensorData, Vocabulary};
use super::{MAGIC, PREAMBLE_LEN, VERSION, align64};
use crate::error::{Error, Result};

/// Builds a `.hsm` file in memory.
///
/// Add the vocabulary and the tensors, then call [`HsmWriter::finish`]. The tensors are written
/// in the order in which they were added.
#[derive(Clone, Debug)]
#[must_use]
pub struct HsmWriter {
    config: ModelConfig,
    vocabulary: Vec<Vec<u8>>,
    tensors: Vec<(String, TensorData, Option<u8>)>,
}

impl HsmWriter {
    /// Starts a file for this model description.
    pub fn new(config: ModelConfig) -> Self {
        Self { config, vocabulary: Vec::new(), tensors: Vec::new() }
    }

    /// Sets the vocabulary: the bytes of each token, in token-id order.
    pub fn set_vocabulary(&mut self, tokens: Vec<Vec<u8>>) {
        self.vocabulary = tokens;
    }

    /// Adds a tensor. `shift` is the rescale shift of an `i32` tensor of multipliers, else
    /// `None`.
    pub fn add_tensor(&mut self, name: impl Into<String>, data: TensorData, shift: Option<u8>) {
        self.tensors.push((name.into(), data, shift));
    }

    /// Lays out and writes the file, then reads it back with [`HsmFile::parse`] to check it.
    ///
    /// Cost: one copy of every tensor into the output, and one full read of the output.
    ///
    /// # Errors
    /// [`Error::TooLarge`] if a part does not fit in the format, or any error of
    /// [`HsmFile::parse`] if the file breaks a rule (for example a bad name or a wrong
    /// vocabulary size).
    pub fn finish(&self) -> Result<Vec<u8>> {
        let vocab = Vocabulary::encode(&self.vocabulary)?;
        let vocab_len = u64::try_from(vocab.len()).map_err(|_| too_large("vocabulary"))?;
        let mut lens = Vec::with_capacity(self.tensors.len());
        for (_, data, _) in &self.tensors {
            lens.push(data.byte_len().ok_or(too_large("tensor"))?);
        }

        // The offsets depend on the header length; repeat until the header fits (module docs).
        let mut header_len: u64 = 0;
        let (json, header) = loop {
            let header = self.header(header_len, vocab_len, &lens)?;
            let json = serde_json::to_vec(&header).map_err(|_| too_large("header JSON"))?;
            let n = u64::try_from(json.len()).map_err(|_| too_large("header JSON"))?;
            if n <= header_len {
                break (json, header);
            }
            header_len = n;
        };
        let header_len_u32 = u32::try_from(header_len).map_err(|_| too_large("header"))?;

        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&header_len_u32.to_le_bytes());
        out.extend_from_slice(&json);
        pad_to(&mut out, PREAMBLE_LEN + header_len, b' ')?;
        pad_to(&mut out, header.vocabulary.offset, 0)?;
        out.extend_from_slice(&vocab);
        for ((_, data, _), entry) in self.tensors.iter().zip(&header.tensors) {
            pad_to(&mut out, entry.offset, 0)?;
            data.write_le(&mut out);
        }
        HsmFile::parse(&out)?;
        Ok(out)
    }

    fn header(&self, header_len: u64, vocab_len: u64, lens: &[u64]) -> Result<Header> {
        let vocab_offset = align64(PREAMBLE_LEN + header_len).ok_or(too_large("layout"))?;
        let mut cursor = vocab_offset.checked_add(vocab_len).ok_or(too_large("layout"))?;
        let mut tensors = Vec::with_capacity(self.tensors.len());
        for ((name, data, shift), &length) in self.tensors.iter().zip(lens) {
            let offset = align64(cursor).ok_or(too_large("layout"))?;
            let shape = data.dims().iter().map(|&d| u64::try_from(d).unwrap_or(u64::MAX)).collect();
            tensors.push(TensorEntry {
                name: name.clone(),
                dtype: data.dtype().name().into(),
                shape,
                offset,
                length,
                shift: *shift,
            });
            cursor = offset.checked_add(length).ok_or(too_large("layout"))?;
        }
        Ok(Header {
            model: self.config.clone(),
            vocabulary: Section { offset: vocab_offset, length: vocab_len },
            tensors,
        })
    }
}

fn too_large(reason: &'static str) -> Error {
    Error::TooLarge { reason }
}

/// Appends `fill` bytes until `out` has `len` bytes.
fn pad_to(out: &mut Vec<u8>, len: u64, fill: u8) -> Result<()> {
    let len = usize::try_from(len).map_err(|_| too_large("file"))?;
    if out.len() < len {
        out.resize(len, fill);
    }
    Ok(())
}
