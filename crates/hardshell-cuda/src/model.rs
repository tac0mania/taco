//! A GPT-2 model file in GPU memory.
//!
//! `CudaModel::load` first loads the file with `hardshell::gpt2::Model::load`, so the CUDA path
//! accepts exactly the files that the CPU reference accepts. Then it reads the tensors again by
//! name (the table in the `hardshell::gpt2` module documentation) and copies them to the GPU.

use hardshell::hsm::{HsmFile, TensorData};
use hardshell::{ModelHash, TokenId};

use crate::device::{CudaDevice, i32_bytes, u32_of, u64_bytes};
use crate::driver::DevBuf;
use crate::error::{Error, Result};
use crate::kernels::{pad_rows, stride_words};

/// A linear layer: `i8` weights (rows padded to 64, `kw` words in each row), bias, multipliers.
pub(crate) struct LinearW {
    pub(crate) w: DevBuf,
    pub(crate) b: DevBuf,
    pub(crate) m: DevBuf,
    pub(crate) shift: u32,
    pub(crate) rows: u32,
    pub(crate) kw: u32,
}

/// One LayerNorm: gamma and beta as i32 values, and the rescale.
pub(crate) struct NormW {
    pub(crate) g: DevBuf,
    pub(crate) b: DevBuf,
    pub(crate) m: i32,
    pub(crate) s: u32,
}

/// One transformer block.
pub(crate) struct LayerW {
    pub(crate) ln_1: NormW,
    pub(crate) c_attn: LinearW,
    pub(crate) score: (i32, u32),
    pub(crate) attn_out: (i32, u32),
    pub(crate) c_proj: LinearW,
    pub(crate) ln_2: NormW,
    pub(crate) c_fc: LinearW,
    pub(crate) gelu: DevBuf,
    pub(crate) gelu_offset: i32,
    pub(crate) gelu_len: i32,
    pub(crate) mlp_proj: LinearW,
}

/// The dimensions of the model, as `u32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dims {
    pub(crate) n_layer: u32,
    pub(crate) n_head: u32,
    pub(crate) d: u32,
    pub(crate) d_ff: u32,
    pub(crate) head_dim: u32,
    pub(crate) vocab: u32,
    pub(crate) n_ctx: u32,
    /// Words in each packed row of width `d` and `d_ff` (multiples of 16).
    pub(crate) dw: u32,
    pub(crate) fw: u32,
}

/// A GPT-2 model in the memory of one CUDA device.
pub struct CudaModel<'d> {
    pub(crate) dev: &'d CudaDevice,
    pub(crate) dims: Dims,
    eot: TokenId,
    hash: ModelHash,
    pub(crate) ln: (u32, u32, u64),
    pub(crate) softmax_p: u32,
    pub(crate) wte: LinearW,
    pub(crate) embed_m: DevBuf,
    pub(crate) embed_s: u32,
    pub(crate) wpe: DevBuf,
    pub(crate) pos_m: DevBuf,
    pub(crate) pos_s: u32,
    pub(crate) layers: Vec<LayerW>,
    pub(crate) ln_f: NormW,
    pub(crate) attn_exp: DevBuf,
    pub(crate) attn_len: u32,
    pub(crate) logits_exp: DevBuf,
    pub(crate) logits_len: u32,
    bytes: u64,
}

impl std::fmt::Debug for CudaModel<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaModel").field("hash", &self.hash).finish_non_exhaustive()
    }
}

fn missing(name: &str) -> Error {
    Error::Model(hardshell::Error::MissingTensor { name: name.into() })
}

/// Reads tensors by name and copies them to the GPU. The CPU loader already checked every name,
/// type and shape, so a failure here is an internal error.
struct Uploader<'a> {
    dev: &'a CudaDevice,
    file: &'a HsmFile,
    bytes: u64,
}

impl Uploader<'_> {
    fn data(&self, name: &str) -> Result<&TensorData> {
        self.file.tensor(name).map(|t| t.data()).ok_or_else(|| missing(name))
    }

    fn shift(&self, name: &str) -> Result<u32> {
        let t = self.file.tensor(name).ok_or_else(|| missing(name))?;
        Ok(u32::from(t.shift().unwrap_or(0)))
    }

    fn put(&mut self, bytes: &[u8]) -> Result<DevBuf> {
        self.bytes += bytes.len() as u64;
        self.dev.upload(bytes)
    }

    fn i32s(&self, name: &str) -> Result<Vec<i32>> {
        match self.data(name)? {
            TensorData::I32(t) => Ok(t.data().to_vec()),
            _ => Err(missing(name)),
        }
    }

    fn i32_buf(&mut self, name: &str) -> Result<DevBuf> {
        let v = self.i32s(name)?;
        self.put(&i32_bytes(&v))
    }

    fn rescale1(&self, name: &str) -> Result<(i32, u32)> {
        Ok((self.i32s(name)?.first().copied().unwrap_or(0), self.shift(name)?))
    }

    /// An `i8 [rows, cols]` matrix: rows of `stride_words(cols)` words, rows padded to 64.
    fn matrix(&mut self, name: &str) -> Result<(DevBuf, u32, u32)> {
        let (rows, cols, data) = match self.data(name)? {
            TensorData::I8(t) => match t.shape().dims() {
                [r, c] => (*r, *c, t.data()),
                _ => return Err(missing(name)),
            },
            _ => return Err(missing(name)),
        };
        let kw = stride_words(cols);
        let bytes = pad_rows(data, rows, cols, 4 * kw);
        Ok((self.put(&bytes)?, u32_of(rows, "too many rows")?, u32_of(kw, "rows too long")?))
    }

    fn linear(&mut self, prefix: &str) -> Result<LinearW> {
        let (w, rows, kw) = self.matrix(&format!("{prefix}.w"))?;
        Ok(LinearW {
            w,
            b: self.i32_buf(&format!("{prefix}.b"))?,
            m: self.i32_buf(&format!("{prefix}.rescale"))?,
            shift: self.shift(&format!("{prefix}.rescale"))?,
            rows,
            kw,
        })
    }

    fn norm(&mut self, prefix: &str) -> Result<NormW> {
        let gname = format!("{prefix}.g");
        let g: Vec<i32> = match self.data(&gname)? {
            TensorData::I16(t) => t.data().iter().map(|&x| i32::from(x)).collect(),
            _ => return Err(missing(&gname)),
        };
        let (m, s) = self.rescale1(&format!("{prefix}.rescale"))?;
        Ok(NormW { g: self.put(&i32_bytes(&g))?, b: self.i32_buf(&format!("{prefix}.b"))?, m, s })
    }

    fn layer(&mut self, l: usize) -> Result<LayerW> {
        let p = format!("h.{l}");
        let gname = format!("{p}.mlp.gelu");
        let gelu: Vec<u8> = match self.data(&gname)? {
            TensorData::I8(t) => t.data().iter().map(|x| x.to_le_bytes()[0]).collect(),
            _ => return Err(missing(&gname)),
        };
        let gelu_len = i32::try_from(gelu.len()).map_err(|_| missing(&gname))?;
        Ok(LayerW {
            ln_1: self.norm(&format!("{p}.ln_1"))?,
            c_attn: self.linear(&format!("{p}.attn.c_attn"))?,
            score: self.rescale1(&format!("{p}.attn.score.rescale"))?,
            attn_out: self.rescale1(&format!("{p}.attn.out.rescale"))?,
            c_proj: self.linear(&format!("{p}.attn.c_proj"))?,
            ln_2: self.norm(&format!("{p}.ln_2"))?,
            c_fc: self.linear(&format!("{p}.mlp.c_fc"))?,
            gelu: self.put(&gelu)?,
            gelu_offset: gelu_len / 2,
            gelu_len,
            mlp_proj: self.linear(&format!("{p}.mlp.c_proj"))?,
        })
    }
}

impl<'d> CudaModel<'d> {
    /// Loads a GPT-2 model file and copies its weights to `dev`. Cost: the CPU load (one
    /// SHA-256 of the file), a second parse, and one upload of about the file size.
    ///
    /// # Errors
    /// Every error of `hardshell::gpt2::Model::load` (as [`Error::Model`]);
    /// [`Error::Unsupported`] for a shape that the kernels do not support; [`Error::Cuda`] on a
    /// driver error (for example, not enough memory).
    pub fn load(dev: &'d CudaDevice, bytes: &[u8]) -> Result<Self> {
        let cpu = hardshell::gpt2::Model::load(bytes)?;
        // DETERMINISM: these kernels implement only the rules of model-file version 1. A model
        // with a version-2 rule (ADR 0004 addendum B: i16 GELU, ln_f_bits, ln_beta_frac) would
        // give different bits here, so it is refused instead of run.
        if cpu.is_wide() {
            return Err(Error::Unsupported {
                reason: "the model uses model-file version 2 rules (i16 GELU, ln_f_bits, ln_beta_frac)",
            });
        }
        let (eot, hash) = (cpu.eot_token(), cpu.hash());
        drop(cpu);
        let file = HsmFile::parse(bytes)?;
        let c = file.config().clone();
        let head_dim = c.d_model / c.n_head.max(1);
        if !c.d_model.is_multiple_of(4) || !head_dim.is_multiple_of(4) || head_dim > 256 {
            return Err(Error::Unsupported {
                reason: "d_model and head_dim must be multiples of 4, head_dim at most 256",
            });
        }
        let dims = Dims {
            n_layer: c.n_layer,
            n_head: c.n_head,
            d: c.d_model,
            d_ff: c.d_ff,
            head_dim,
            vocab: c.vocab_size,
            n_ctx: c.n_ctx,
            dw: u32_of(stride_words(c.d_model as usize), "d_model too large")?,
            fw: u32_of(stride_words(c.d_ff as usize), "d_ff too large")?,
        };
        let mut up = Uploader { dev, file: &file, bytes: 0 };
        let (w, rows, kw) = up.matrix("wte")?;
        let logit_m = up.i32_buf("wte.rescale_logits")?;
        let wte = LinearW {
            w,
            b: up.put(&[0u8; 4])?,
            m: logit_m,
            shift: up.shift("wte.rescale_logits")?,
            rows,
            kw,
        };
        let embed_m = up.i32_buf("wte.rescale_embed")?;
        let embed_s = up.shift("wte.rescale_embed")?;
        let (wpe, _, _) = up.matrix("wpe")?;
        let pos_m = up.i32_buf("wpe.rescale")?;
        let pos_s = up.shift("wpe.rescale")?;
        let layers = (0..c.n_layer as usize).map(|l| up.layer(l)).collect::<Result<Vec<_>>>()?;
        let ln_f = up.norm("ln_f")?;
        let attn = up.i32s("attn_exp")?;
        let attn_len = u32_of(attn.len(), "table too long")?;
        let attn_exp = up.put(&i32_bytes(&attn))?;
        let lexp: Vec<u64> = match up.data("logits_exp")? {
            TensorData::U64(t) => t.data().to_vec(),
            _ => return Err(missing("logits_exp")),
        };
        let logits_len = u32_of(lexp.len(), "table too long")?;
        let logits_exp = up.put(&u64_bytes(&lexp))?;
        let bytes = up.bytes;
        Ok(Self {
            dev,
            dims,
            eot,
            hash,
            ln: (u32::from(c.constants.ln_k), u32::from(c.constants.ln_j), c.constants.ln_eps),
            softmax_p: u32::from(c.constants.softmax_p),
            wte,
            embed_m,
            embed_s,
            wpe,
            pos_m,
            pos_s,
            layers,
            ln_f,
            attn_exp,
            attn_len,
            logits_exp,
            logits_len,
            bytes,
        })
    }

    /// The SHA-256 of the model file.
    #[must_use]
    pub fn hash(&self) -> ModelHash {
        self.hash
    }

    /// The number of tokens in the vocabulary.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.dims.vocab as usize
    }

    /// The context length.
    #[must_use]
    pub fn n_ctx(&self) -> usize {
        self.dims.n_ctx as usize
    }

    /// The `<|endoftext|>` token.
    #[must_use]
    pub fn eot_token(&self) -> TokenId {
        self.eot
    }

    /// The device of the model.
    #[must_use]
    pub fn device(&self) -> &'d CudaDevice {
        self.dev
    }

    /// The bytes of GPU memory that the weights use (measured from the uploads).
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.bytes
    }

    /// The KV cache bytes of one slot: `2 * n_layer * n_ctx * d_model`.
    #[must_use]
    pub fn cache_bytes_per_slot(&self) -> u64 {
        let d = self.dims;
        2 * u64::from(d.n_layer) * u64::from(d.n_ctx) * u64::from(d.d)
    }
}
