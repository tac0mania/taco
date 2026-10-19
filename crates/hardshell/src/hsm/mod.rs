//! The `.hsm` model file, format versions 1 and 2: the reader and the writer (plan §8).
//!
//! The reader checks every part of the file before it gives a value to the caller. The writer
//! is the reference for the layout: the tests and the Python converter make files that the
//! reader accepts.
//!
//! # Layout
//!
//! All integers are little-endian. The file has these sections, in this order:
//!
//! | Offset | Length | Contents |
//! |---|---|---|
//! | 0 | 8 | magic `"HSHLMDL\0"` |
//! | 8 | 4 | version, `u32`, 1 or 2 (see "Format version 2") |
//! | 12 | 4 | `header_len`, `u32` |
//! | 16 | `header_len` | the JSON header, UTF-8 |
//! | `align64(16 + header_len)` | `vocabulary.length` | the vocabulary |
//! | `align64(end of the vocabulary)` | `merges.length` | the BPE merges (optional) |
//! | `align64(end of the section before)` | `length` | each tensor, in header order |
//!
//! `align64(x)` is the smallest multiple of 64 that is not less than `x`. The bytes between
//! sections are zero. The file ends at the end of the last tensor (or of the vocabulary, if
//! there are no tensors). The reader refuses every other layout, so one model has exactly one
//! file, and thus one model hash.
//!
//! NOTE: plan §8 says "all sections start at a multiple of 64 bytes". The fixed part and the
//! header start at 0 and 16; the vocabulary and every tensor start at a multiple of 64.
//!
//! The vocabulary has, for each token id from 0 to `vocab_size - 1`: a `u16` length, then the
//! bytes of the token. The section has no other bytes.
//!
//! The merges section is optional. It holds the BPE merges of the encoder in rank order (rank 0
//! first): for each merge, the `u32` id of the left token, then the `u32` id of the right token,
//! so `length = 8 * count`. If the header has no `merges` field, the file has no merges
//! section, and the tensors follow the vocabulary. An empty merges section is not permitted
//! (omit the field), so one model has one file. The reader checks that each id is in the
//! vocabulary, that each part has at least one byte, that no pair occurs two times, and that
//! the joined bytes of each pair are the bytes of exactly one token. Only the encoder uses the
//! merges; the decoder does not need them. The model hash covers them, because it covers the
//! full file (ADR 0004, addendum A).
//!
//! A tensor holds its values in row-major order, `dtype.size()` bytes for each value, so
//! `length = product(shape) * size`.
//!
//! # The JSON header
//!
//! The header is one JSON object. Every number is a non-negative integer: a number with a
//! fraction (`1.0`) or an exponent (`1e3`) is an error, and so is an unknown field, a missing
//! field or a duplicate field. Names and the architecture use only `a-z`, `0-9`, `_` and `.`,
//! with 1 to 128 characters.
//!
//! ```json
//! {
//!   "model": {
//!     "architecture": "gpt2",
//!     "n_layer": 2, "n_head": 4, "d_model": 64, "d_ff": 256,
//!     "vocab_size": 300, "n_ctx": 64, "eot_token": 299,
//!     "constants": {"ln_k": 48, "ln_j": 8, "ln_eps": 1234, "softmax_p": 30}
//!   },
//!   "vocabulary": {"offset": 4096, "length": 1234},
//!   "merges": {"offset": 5376, "length": 16},
//!   "tensors": [
//!     {"name": "wte", "dtype": "i8", "shape": [300, 64], "offset": 5440, "length": 19200},
//!     {"name": "wte.rescale_embed", "dtype": "i32", "shape": [300], "offset": 24640,
//!      "length": 1200, "shift": 27}
//!   ]
//! }
//! ```
//!
//! | Field | Type | Rule |
//! |---|---|---|
//! | `model.architecture` | string | the architecture; `gpt2` is the only one in version 1 |
//! | `model.n_layer`, `n_head`, `d_model`, `d_ff`, `n_ctx` | `u32` | the architecture checks them |
//! | `model.vocab_size` | `u32` | 1 to 2^20 |
//! | `model.eot_token` | `u32` | less than `vocab_size` |
//! | `model.constants.ln_k` | `u8` | LayerNorm `K`, at most 48 |
//! | `model.constants.ln_j` | `u8` | LayerNorm `J`, at most 16 |
//! | `model.constants.ln_eps` | `u64` | LayerNorm `eps`, 1 to 2^62 |
//! | `model.constants.softmax_p` | `u8` | softmax `P`, at most 30 |
//! | `model.constants.ln_f_bits` | `u8`, optional, version 2 only | 16: the final LayerNorm gives `i16` outputs; omit the field for `i8` |
//! | `model.constants.ln_beta_frac` | `u8`, optional, version 2 only | 1 to 16: every LayerNorm beta has this many more bits and is added before the rounding; omit the field for 0 |
//! | `vocabulary.offset`, `.length` | `u64` | the vocabulary section |
//! | `merges.offset`, `.length` | `u64`, optional | the merges section; omit the field for no merges |
//! | `tensors[].name` | string | unique |
//! | `tensors[].dtype` | string | `i8`, `i16`, `i32` or `u64` |
//! | `tensors[].shape` | array of `u64` | 1 to 4 dimensions, each at least 1 |
//! | `tensors[].offset`, `.length` | `u64` | the tensor data |
//! | `tensors[].shift` | `u8`, optional | only on an `i32` tensor: the tensor holds rescale multipliers `m`, and `shift` is the shift `s` (plan §7.1, §7.2) |
//!
//! A rescale `(m, s)` (plan §7.2) is an `i32` tensor of multipliers with a `shift`. It has
//! shape `[1]` for one multiplier, or one multiplier for each output row. The architecture
//! module (`gpt2`) gives the names of the tensors that it needs.
//!
//! # Format version 2
//!
//! Version 2 (ADR 0004 addendum B) permits `i16` outputs at two activation sites and a finer
//! LayerNorm beta, for models that lose too much with the rules of version 1. It adds exactly
//! three things to version 1:
//!
//! 1. The optional header constant `model.constants.ln_f_bits`. Its only permitted value is 16.
//! 2. The optional header constant `model.constants.ln_beta_frac`, from 1 to 16.
//! 3. A GELU table (a tensor whose name ends with `.mlp.gelu`) with the dtype `i16`.
//!
//! All other rules are the rules of version 1. A file is version 2 if and only if it uses one
//! of the three things ([`required_version`]). The reader refuses a file whose version is not
//! the required version, and the writer writes the required version. Thus one model still has
//! exactly one file, and every version 1 file keeps its bytes and its meaning.
//!
//! # The writer
//!
//! The writer serialises the header with `serde_json` in compact form (no spaces), with the
//! fields in the order of this schema. The offsets depend on the header length, so the writer
//! repeats: it calculates the layout for a header length, serialises the header, and stops when
//! the header fits. A header that is shorter than the length that the layout uses gets spaces
//! (0x20) at the end. JSON permits these spaces, and the reader accepts them.

mod header;
mod merges;
mod reader;
mod writer;

pub use header::{Constants, ModelConfig};
pub use merges::{MERGE_PAIR_LEN, MergePair};
pub use reader::{HsmFile, StoredTensor, TensorData, Vocabulary};
pub use writer::HsmWriter;

/// The magic bytes at the start of a `.hsm` file.
pub const MAGIC: [u8; 8] = *b"HSHLMDL\0";
/// The newest format version that this module reads and writes.
pub const VERSION: u32 = 2;
/// Format version 1: `i8` outputs at every activation site.
pub const VERSION_1: u32 = 1;
/// Format version 2: version 1, and `i16` outputs at the GELU tables and the final LayerNorm.
pub const VERSION_2: u32 = 2;
/// The alignment of the vocabulary and of every tensor, in bytes.
pub const ALIGN: u64 = 64;
/// The length of the fixed part before the header: magic, version and header length.
pub const PREAMBLE_LEN: u64 = 16;
/// The largest vocabulary: `2^20` tokens (the bound of `logits`).
pub const MAX_VOCAB: u32 = 1 << 20;
/// The longest tensor name or architecture name.
pub const MAX_NAME_LEN: usize = 128;

/// The smallest multiple of [`ALIGN`] that is not less than `x`, or `None` on overflow.
pub(crate) fn align64(x: u64) -> Option<u64> {
    x.checked_add(ALIGN - 1).map(|v| v & !(ALIGN - 1))
}

/// `true` if `name` has 1 to 128 characters from `a-z`, `0-9`, `_` and `.`.
pub(crate) fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME_LEN
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'.')
}

/// The format version that a file with these parts must have (see "Format version 2"): 2 if
/// the header has a constant of version 2 (`has_v2_constant`) or a tensor `*.mlp.gelu` has the
/// dtype `i16`, else 1. `tensors` gives the name and the dtype name of each tensor.
#[must_use]
pub fn required_version<'a>(
    has_v2_constant: bool,
    mut tensors: impl Iterator<Item = (&'a str, &'a str)>,
) -> u32 {
    let wide_gelu = tensors.any(|(name, dtype)| name.ends_with(".mlp.gelu") && dtype == "i16");
    if has_v2_constant || wide_gelu { VERSION_2 } else { VERSION_1 }
}
