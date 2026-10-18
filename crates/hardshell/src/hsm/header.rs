//! The JSON header of the `.hsm` file: the serde schema.
//!
//! Every number in the schema is an integer type. serde_json refuses a number with a fraction
//! or an exponent for an integer field, and `deny_unknown_fields` refuses any other field. Thus
//! a header with a float in it does not parse (plan §8).

use serde::{Deserialize, Serialize};

/// The model description in the header: architecture, dimensions and constants.
///
/// The fields are fixed for format version 1. A new field needs a new format version
/// (CLAUDE.md §1 rule 8), so this struct is not `#[non_exhaustive]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// The architecture name, for example `"gpt2"`.
    pub architecture: String,
    /// The number of transformer blocks.
    pub n_layer: u32,
    /// The number of attention heads in each block.
    pub n_head: u32,
    /// The width of the residual stream.
    pub d_model: u32,
    /// The width of the MLP hidden layer.
    pub d_ff: u32,
    /// The number of tokens in the vocabulary.
    pub vocab_size: u32,
    /// The context length: the largest number of positions.
    pub n_ctx: u32,
    /// The id of the `<|endoftext|>` token.
    pub eot_token: u32,
    /// The numerics constants (plan §7.3, §7.5).
    pub constants: Constants,
}

/// The numerics constants of the model (plan §7.3, §7.5, ADR 0004).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constants {
    /// LayerNorm `K`.
    pub ln_k: u8,
    /// LayerNorm `J`.
    pub ln_j: u8,
    /// LayerNorm `eps` in the units of `var * 2^(2J)`.
    pub ln_eps: u64,
    /// Softmax `P`.
    pub softmax_p: u8,
}

/// The full header as it is in the file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Header {
    pub(crate) model: ModelConfig,
    pub(crate) vocabulary: Section,
    /// The BPE merges for the encoder (optional; see the module documentation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) merges: Option<Section>,
    pub(crate) tensors: Vec<TensorEntry>,
}

/// The position of a section in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Section {
    pub(crate) offset: u64,
    pub(crate) length: u64,
}

/// One tensor or table in the header.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorEntry {
    pub(crate) name: String,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) offset: u64,
    pub(crate) length: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) shift: Option<u8>,
}
