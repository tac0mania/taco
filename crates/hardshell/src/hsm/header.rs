//! The JSON header of the `.hsm` file: the serde schema.
//!
//! Every number in the schema is an integer type. serde_json refuses a number with a fraction
//! or an exponent for an integer field, and `deny_unknown_fields` refuses any other field. Thus
//! a header with a float in it does not parse (plan §8).

use serde::{Deserialize, Serialize};

/// The model description in the header: architecture, dimensions and constants.
///
/// The fields are fixed for format version 1. A new field needs a new format version
/// (CLAUDE.md §1 rule 8), so this struct is not `#[non_exhaustive]`. The fields that format
/// version 2 adds (`ln_f_bits`, `ln_beta_frac`) are not in this struct: see `HsmFile`.
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

/// The `model` object as it is in the file: [`ModelConfig`] and, in format version 2, the
/// optional constants `ln_f_bits` and `ln_beta_frac`.
///
/// NOTE: the public [`ModelConfig`] and [`Constants`] keep the fields of format version 1, so
/// that code which builds them stays valid. `HsmFile` and `HsmWriter` give the fields of
/// version 2.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelJson {
    architecture: String,
    n_layer: u32,
    n_head: u32,
    d_model: u32,
    d_ff: u32,
    vocab_size: u32,
    n_ctx: u32,
    eot_token: u32,
    constants: ConstantsJson,
}

/// The `constants` object as it is in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConstantsJson {
    ln_k: u8,
    ln_j: u8,
    ln_eps: u64,
    softmax_p: u8,
    /// Format version 2 only: 16 makes the output of the final LayerNorm `i16`. The field is
    /// absent for an `i8` output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ln_f_bits: Option<u8>,
    /// Format version 2 only: every LayerNorm beta has this many more bits and is added before
    /// the one rounding (`LayerNorm::with_beta_frac`). The field is absent for 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ln_beta_frac: Option<u8>,
}

/// The header constants of format version 2 (absent in version 1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct V2Constants {
    /// `model.constants.ln_f_bits`.
    pub(crate) ln_f_bits: Option<u8>,
    /// `model.constants.ln_beta_frac`.
    pub(crate) ln_beta_frac: Option<u8>,
}

impl V2Constants {
    /// `true` if the header has a constant of version 2.
    pub(crate) fn any(self) -> bool {
        self.ln_f_bits.is_some() || self.ln_beta_frac.is_some()
    }
}

impl ModelJson {
    /// The header object for `config` and the optional `ln_f_bits`.
    pub(crate) fn new(config: &ModelConfig, v2: V2Constants) -> Self {
        let c = config.constants;
        Self {
            architecture: config.architecture.clone(),
            n_layer: config.n_layer,
            n_head: config.n_head,
            d_model: config.d_model,
            d_ff: config.d_ff,
            vocab_size: config.vocab_size,
            n_ctx: config.n_ctx,
            eot_token: config.eot_token,
            constants: ConstantsJson {
                ln_k: c.ln_k,
                ln_j: c.ln_j,
                ln_eps: c.ln_eps,
                softmax_p: c.softmax_p,
                ln_f_bits: v2.ln_f_bits,
                ln_beta_frac: v2.ln_beta_frac,
            },
        }
    }

    /// The public model description and the constants of version 2.
    pub(crate) fn into_parts(self) -> (ModelConfig, V2Constants) {
        let c = self.constants;
        let config = ModelConfig {
            architecture: self.architecture,
            n_layer: self.n_layer,
            n_head: self.n_head,
            d_model: self.d_model,
            d_ff: self.d_ff,
            vocab_size: self.vocab_size,
            n_ctx: self.n_ctx,
            eot_token: self.eot_token,
            constants: Constants {
                ln_k: c.ln_k,
                ln_j: c.ln_j,
                ln_eps: c.ln_eps,
                softmax_p: c.softmax_p,
            },
        };
        (config, V2Constants { ln_f_bits: c.ln_f_bits, ln_beta_frac: c.ln_beta_frac })
    }
}

/// The full header as it is in the file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Header {
    pub(crate) model: ModelJson,
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
