//! The GPT-2 model: load and validate all tensors from a `.hsm` file.

use super::{MAX_CTX, MAX_LAYERS};
use crate::error::{Error, Result};
use crate::gelu::GeluTable;
use crate::hsm::{HsmFile, StoredTensor, TensorData, Vocabulary};
use crate::kernels::MAX_DOT_LEN_I32;
use crate::layernorm::{LayerNorm, LnConstants, MAX_WIDTH};
use crate::logits::LogitTable;
use crate::softmax::ExpTable;
use crate::tensor::{DType, Matrix};
use crate::types::{ModelHash, TokenId};

/// One rescale `(m, s)` with a single multiplier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rescale {
    pub(crate) m: i32,
    pub(crate) s: u8,
}

/// Rescale multipliers for each row, with one shift.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowRescale {
    pub(crate) m: Box<[i32]>,
    pub(crate) s: u8,
}

/// A linear layer: `i8` weights `[out, in]`, `i32` bias and a rescale for each output row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Linear {
    pub(crate) w: Matrix<i8>,
    pub(crate) b: Box<[i32]>,
    pub(crate) rescale: RowRescale,
}

/// The GELU table of one block: `i8` outputs (format version 1), or `i16` outputs (format
/// version 2, ADR 0004 addendum B). The dtype of the tensor `h.{l}.mlp.gelu` selects it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Gelu {
    /// `i8` outputs.
    Narrow(GeluTable<i8>),
    /// `i16` outputs.
    Wide(GeluTable<i16>),
}

impl Gelu {
    /// The input range of the table: `(min, max)`.
    pub(crate) fn input_range(&self) -> (i64, i64) {
        match self {
            Self::Narrow(t) => (t.min_input(), t.max_input()),
            Self::Wide(t) => (t.min_input(), t.max_input()),
        }
    }

    /// `true` for a table with `i16` outputs.
    pub(crate) fn is_wide(&self) -> bool {
        matches!(self, Self::Wide(_))
    }
}

/// One transformer block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    pub(crate) ln_1: LayerNorm,
    pub(crate) c_attn: Linear,
    pub(crate) score: Rescale,
    pub(crate) attn_out: Rescale,
    pub(crate) c_proj: Linear,
    pub(crate) ln_2: LayerNorm,
    pub(crate) c_fc: Linear,
    pub(crate) gelu: Gelu,
    pub(crate) mlp_proj: Linear,
}

/// The dimensions of a model, as `usize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dims {
    pub(crate) n_layer: usize,
    pub(crate) n_head: usize,
    pub(crate) d_model: usize,
    pub(crate) d_ff: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab: usize,
    pub(crate) n_ctx: usize,
}

/// A loaded GPT-2 model. All tensors are validated, so a [`super::Session`] cannot index out of
/// bounds.
#[derive(Clone, Debug)]
pub struct Model {
    pub(crate) dims: Dims,
    pub(crate) eot: TokenId,
    pub(crate) ln_consts: LnConstants,
    pub(crate) softmax_p: u8,
    pub(crate) wte: Matrix<i8>,
    pub(crate) embed: RowRescale,
    pub(crate) logit: RowRescale,
    pub(crate) wpe: Matrix<i8>,
    pub(crate) pos: RowRescale,
    pub(crate) blocks: Vec<Block>,
    pub(crate) ln_f: LayerNorm,
    /// The output bits of `ln_f`: 8, or 16 (format version 2).
    pub(crate) ln_f_bits: u8,
    /// The extra bits of every LayerNorm beta: 0, or 1 to 16 (format version 2).
    pub(crate) ln_beta_frac: u8,
    pub(crate) attn_exp: ExpTable,
    pub(crate) logits_exp: LogitTable,
    vocabulary: Vocabulary,
    hash: ModelHash,
}

fn usize_of(field: &str, v: u32) -> Result<usize> {
    usize::try_from(v)
        .map_err(|_| Error::InvalidHeader { field: field.into(), reason: "does not fit in usize" })
}

fn check_dims(f: &HsmFile) -> Result<Dims> {
    let c = f.config();
    let bad = |field: &str, reason| Err(Error::InvalidHeader { field: field.into(), reason });
    if c.architecture != "gpt2" {
        return bad("model.architecture", "this loader reads only \"gpt2\"");
    }
    if c.n_layer == 0 || c.n_layer > MAX_LAYERS {
        return bad("model.n_layer", "must be in 1..=1024");
    }
    let d_model = usize_of("model.d_model", c.d_model)?;
    let d_ff = usize_of("model.d_ff", c.d_ff)?;
    let n_head = usize_of("model.n_head", c.n_head)?;
    // INVARIANT: these bounds keep every dot product within MAX_DOT_LEN_I32 and every
    // LayerNorm within MAX_WIDTH.
    if d_model == 0 || d_model > MAX_WIDTH || d_model > MAX_DOT_LEN_I32 {
        return bad("model.d_model", "must be in 1..=65536");
    }
    if d_ff == 0 || d_ff > MAX_DOT_LEN_I32 {
        return bad("model.d_ff", "must be in 1..=65536");
    }
    if n_head == 0 || !d_model.is_multiple_of(n_head) {
        return bad("model.n_head", "must be at least 1 and divide d_model");
    }
    if c.n_ctx < 2 || c.n_ctx > MAX_CTX {
        return bad("model.n_ctx", "must be in 2..=65536 (a restart needs two positions)");
    }
    Ok(Dims {
        n_layer: usize_of("model.n_layer", c.n_layer)?,
        n_head,
        d_model,
        d_ff,
        head_dim: d_model / n_head,
        vocab: usize_of("model.vocab_size", c.vocab_size)?,
        n_ctx: usize_of("model.n_ctx", c.n_ctx)?,
    })
}

/// Takes tensors out of the file by name and checks their type, shape and shift.
struct Loader {
    file: HsmFile,
}

impl Loader {
    fn take(&mut self, name: &str, dims: &[usize], shift: bool) -> Result<StoredTensor> {
        let t = self.file.take_tensor(name).ok_or(Error::MissingTensor { name: name.into() })?;
        let mismatch = |reason| Err(Error::TensorMismatch { name: name.into(), reason });
        if t.data().dims() != dims {
            return mismatch("the shape is not the shape that the model needs");
        }
        match (shift, t.shift()) {
            (true, None) => return mismatch("a rescale tensor needs a shift"),
            (false, Some(_)) => return mismatch("only a rescale tensor has a shift"),
            _ => {}
        }
        Ok(t)
    }

    fn wrong_type(name: &str) -> Error {
        Error::TensorMismatch {
            name: name.into(),
            reason: "the dtype is not the dtype that the model needs",
        }
    }

    fn i8_matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<Matrix<i8>> {
        match self.take(name, &[rows, cols], false)?.into_data() {
            TensorData::I8(t) => Matrix::from_tensor(t),
            _ => Err(Self::wrong_type(name)),
        }
    }

    fn i8_vec(&mut self, name: &str, len: usize) -> Result<Vec<i8>> {
        match self.take(name, &[len], false)?.into_data() {
            TensorData::I8(t) => Ok(t.into_data()),
            _ => Err(Self::wrong_type(name)),
        }
    }

    fn i16_vec(&mut self, name: &str, len: usize) -> Result<Vec<i16>> {
        match self.take(name, &[len], false)?.into_data() {
            TensorData::I16(t) => Ok(t.into_data()),
            _ => Err(Self::wrong_type(name)),
        }
    }

    fn i32_vec(&mut self, name: &str, len: usize) -> Result<Vec<i32>> {
        match self.take(name, &[len], false)?.into_data() {
            TensorData::I32(t) => Ok(t.into_data()),
            _ => Err(Self::wrong_type(name)),
        }
    }

    fn rows_rescale(&mut self, name: &str, len: usize) -> Result<RowRescale> {
        let t = self.take(name, &[len], true)?;
        let s = t.shift().unwrap_or(0);
        match t.into_data() {
            TensorData::I32(t) => Ok(RowRescale { m: t.into_data().into_boxed_slice(), s }),
            _ => Err(Self::wrong_type(name)),
        }
    }

    fn rescale(&mut self, name: &str) -> Result<Rescale> {
        let r = self.rows_rescale(name, 1)?;
        Ok(Rescale { m: r.m.first().copied().unwrap_or(0), s: r.s })
    }

    fn linear(&mut self, prefix: &str, rows: usize, cols: usize) -> Result<Linear> {
        Ok(Linear {
            w: self.i8_matrix(&format!("{prefix}.w"), rows, cols)?,
            b: self.i32_vec(&format!("{prefix}.b"), rows)?.into_boxed_slice(),
            rescale: self.rows_rescale(&format!("{prefix}.rescale"), rows)?,
        })
    }

    fn layernorm(&mut self, prefix: &str, d: usize) -> Result<LayerNorm> {
        let g = self.i16_vec(&format!("{prefix}.g"), d)?;
        let b = self.i32_vec(&format!("{prefix}.b"), d)?;
        let r = self.rescale(&format!("{prefix}.rescale"))?;
        LayerNorm::with_beta_frac(g, b, r.m, r.s, self.file.ln_beta_frac())
    }

    fn gelu(&mut self, name: &str) -> Result<Gelu> {
        let t = self.file.tensor(name).ok_or(Error::MissingTensor { name: name.into() })?;
        let len = t.data().dims().first().copied().unwrap_or(0);
        if len != 256 && len != 65_536 {
            return Err(Error::TensorMismatch {
                name: name.into(),
                reason: "a GELU table has 256 or 65536 entries",
            });
        }
        // NOTE: the reader permits an i16 GELU table only in a version 2 file.
        match t.data().dtype() {
            DType::I16 => Ok(Gelu::Wide(GeluTable::new(self.i16_vec(name, len)?)?)),
            _ => Ok(Gelu::Narrow(GeluTable::new(self.i8_vec(name, len)?)?)),
        }
    }

    fn table_len(&self, name: &str) -> Result<usize> {
        let t = self.file.tensor(name).ok_or(Error::MissingTensor { name: name.into() })?;
        match t.data().dims() {
            [n] => Ok(*n),
            _ => Err(Error::TensorMismatch {
                name: name.into(),
                reason: "a table has one dimension",
            }),
        }
    }

    fn block(&mut self, l: usize, d: &Dims) -> Result<Block> {
        let p = format!("h.{l}");
        Ok(Block {
            ln_1: self.layernorm(&format!("{p}.ln_1"), d.d_model)?,
            c_attn: self.linear(&format!("{p}.attn.c_attn"), 3 * d.d_model, d.d_model)?,
            score: self.rescale(&format!("{p}.attn.score.rescale"))?,
            attn_out: self.rescale(&format!("{p}.attn.out.rescale"))?,
            c_proj: self.linear(&format!("{p}.attn.c_proj"), d.d_model, d.d_model)?,
            ln_2: self.layernorm(&format!("{p}.ln_2"), d.d_model)?,
            c_fc: self.linear(&format!("{p}.mlp.c_fc"), d.d_ff, d.d_model)?,
            gelu: self.gelu(&format!("{p}.mlp.gelu"))?,
            mlp_proj: self.linear(&format!("{p}.mlp.c_proj"), d.d_model, d.d_ff)?,
        })
    }
}

impl Model {
    /// Loads a GPT-2 model from the bytes of a `.hsm` file.
    ///
    /// Cost: one pass over the file (copy and SHA-256). The model holds a copy of all weights:
    /// about the file size in memory.
    ///
    /// # Errors
    /// Every error of [`HsmFile::parse`]; [`Error::InvalidHeader`] if the architecture is not
    /// `gpt2` or a dimension is out of range; [`Error::MissingTensor`],
    /// [`Error::TensorMismatch`] or [`Error::UnexpectedTensor`] if the tensors do not agree
    /// with the table in the module documentation.
    pub fn load(bytes: &[u8]) -> Result<Self> {
        let file = HsmFile::parse(bytes)?;
        let d = check_dims(&file)?;
        let c = file.config().clone();
        let (ln_f_bits, ln_beta_frac) = (file.ln_f_bits(), file.ln_beta_frac());
        let ln_consts = LnConstants::new(c.constants.ln_k, c.constants.ln_j, c.constants.ln_eps)?;
        let mut ld = Loader { file };
        let wte = ld.i8_matrix("wte", d.vocab, d.d_model)?;
        let embed = ld.rows_rescale("wte.rescale_embed", d.vocab)?;
        let logit = ld.rows_rescale("wte.rescale_logits", d.vocab)?;
        let wpe = ld.i8_matrix("wpe", d.n_ctx, d.d_model)?;
        let pos = ld.rows_rescale("wpe.rescale", d.n_ctx)?;
        let mut blocks = Vec::with_capacity(d.n_layer);
        for l in 0..d.n_layer {
            blocks.push(ld.block(l, &d)?);
        }
        let ln_f = ld.layernorm("ln_f", d.d_model)?;
        let attn_len = ld.table_len("attn_exp")?;
        let attn_exp = ExpTable::new(ld.i32_vec("attn_exp", attn_len)?)?;
        let logits_len = ld.table_len("logits_exp")?;
        let logits_exp = match ld.take("logits_exp", &[logits_len], false)?.into_data() {
            TensorData::U64(t) => LogitTable::new(t.into_data())?,
            _ => return Err(Loader::wrong_type("logits_exp")),
        };
        if let Some(extra) = ld.file.remaining_tensor_names().next() {
            return Err(Error::UnexpectedTensor {
                name: extra.into(),
                reason: "the gpt2 model does not use this tensor",
            });
        }
        let hash = ld.file.hash();
        Ok(Self {
            dims: d,
            eot: TokenId::new(c.eot_token),
            ln_consts,
            softmax_p: c.constants.softmax_p,
            wte,
            embed,
            logit,
            wpe,
            pos,
            blocks,
            ln_f,
            ln_f_bits,
            ln_beta_frac,
            attn_exp,
            logits_exp,
            vocabulary: ld.file.into_vocabulary(),
            hash,
        })
    }

    /// The number of tokens in the vocabulary: the length of each weight vector.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.dims.vocab
    }

    /// The context length.
    #[must_use]
    pub fn n_ctx(&self) -> usize {
        self.dims.n_ctx
    }

    /// The `<|endoftext|>` token.
    #[must_use]
    pub fn eot_token(&self) -> TokenId {
        self.eot
    }

    /// The vocabulary of the model file.
    #[must_use]
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// The output bits of the final LayerNorm: 8, or 16 for a model file of format version 2
    /// with `ln_f_bits = 16` (ADR 0004 addendum B).
    #[must_use]
    pub fn ln_f_bits(&self) -> u8 {
        self.ln_f_bits
    }

    /// `true` if the model uses a rule of format version 2 (ADR 0004 addendum B): an `i16`
    /// GELU table, `ln_f_bits = 16`, or `ln_beta_frac`. A backend that does not implement
    /// these rules must refuse such a model.
    #[must_use]
    pub fn is_wide(&self) -> bool {
        self.ln_f_bits == 16
            || self.ln_beta_frac > 0
            || self.blocks.iter().any(|b| b.gelu.is_wide())
    }

    /// The SHA-256 of the model file.
    #[must_use]
    pub fn hash(&self) -> ModelHash {
        self.hash
    }

    /// Checks that `token` is in the vocabulary, and gives its row index.
    pub(crate) fn check_token(&self, token: TokenId) -> Result<usize> {
        let i = usize::try_from(token.get()).unwrap_or(usize::MAX);
        if i < self.dims.vocab {
            Ok(i)
        } else {
            Err(Error::TokenOutOfRange {
                token: token.get(),
                vocab_size: u32::try_from(self.dims.vocab).unwrap_or(u32::MAX),
            })
        }
    }
}
