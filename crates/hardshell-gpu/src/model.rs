//! A GPT-2 model file in GPU memory.
//!
//! `GpuModel::load` first loads the file with `hardshell::gpt2::Model::load`, so the GPU path
//! accepts exactly the files that the CPU reference accepts. Then it reads the tensors again by
//! name (the table in the `hardshell::gpt2` module documentation) and copies them to the GPU.

use hardshell::hsm::{HsmFile, TensorData};
use hardshell::{ModelHash, TokenId};

use crate::device::{GpuDevice, words_to_bytes};
use crate::error::{Error, Result};
use crate::kernels::{Kernels, pack_i8, stride_words};

/// A linear layer: packed `i8` weights `[rows, kw words]`, bias and multipliers.
pub(crate) struct LinearW {
    pub(crate) w: wgpu::Buffer,
    pub(crate) b: wgpu::Buffer,
    pub(crate) m: wgpu::Buffer,
    pub(crate) shift: u32,
    pub(crate) rows: u32,
    pub(crate) kw: u32,
}

/// One LayerNorm: gamma and beta as i32 values, and the rescale.
pub(crate) struct NormW {
    pub(crate) g: wgpu::Buffer,
    pub(crate) b: wgpu::Buffer,
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
    pub(crate) gelu: wgpu::Buffer,
    /// `2^(b-1)`: the index of the GELU input 0.
    pub(crate) gelu_offset: u32,
    pub(crate) gelu_len: u32,
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
    /// Row strides (words) of packed vectors of width `d` and `d_ff`.
    pub(crate) dw: u32,
    pub(crate) fw: u32,
}

/// A GPT-2 model in GPU memory, with its compiled kernels.
pub struct GpuModel {
    pub(crate) dev: GpuDevice,
    pub(crate) k: Kernels,
    pub(crate) dims: Dims,
    pub(crate) eot: TokenId,
    hash: ModelHash,
    pub(crate) ln: (u32, u32, u64),
    pub(crate) softmax_p: u32,
    /// `wte`, packed with the stride `dw`: the embedding and the output matrix.
    pub(crate) wte: wgpu::Buffer,
    pub(crate) embed_m: wgpu::Buffer,
    pub(crate) embed_s: u32,
    pub(crate) logit_m: wgpu::Buffer,
    pub(crate) logit_s: u32,
    pub(crate) wpe: wgpu::Buffer,
    pub(crate) pos_m: wgpu::Buffer,
    pub(crate) pos_s: u32,
    pub(crate) layers: Vec<LayerW>,
    pub(crate) ln_f: NormW,
    pub(crate) attn_exp: wgpu::Buffer,
    pub(crate) attn_len: u32,
    pub(crate) logits_exp: wgpu::Buffer,
    pub(crate) logits_len: u32,
    gpu_bytes: u64,
}

impl std::fmt::Debug for GpuModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuModel")
            .field("device", &self.dev)
            .field("hash", &self.hash)
            .finish_non_exhaustive()
    }
}

/// Reads tensors by name and copies them to the GPU. The CPU loader already checked every
/// name, type and shape, so a failure here is an internal error.
struct Uploader<'a> {
    dev: &'a GpuDevice,
    file: &'a HsmFile,
    bytes: u64,
}

fn missing(name: &str) -> Error {
    Error::Model(hardshell::Error::MissingTensor { name: name.into() })
}

fn i32_words(v: &[i32]) -> Vec<u32> {
    v.iter().map(|x| x.cast_unsigned()).collect()
}

fn u32_of(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| Error::Unsupported { reason: "a dimension is 2^32 or more" })
}

impl Uploader<'_> {
    fn data(&self, name: &str) -> Result<&TensorData> {
        self.file.tensor(name).map(|t| t.data()).ok_or_else(|| missing(name))
    }

    fn shift(&self, name: &str) -> Result<u32> {
        let t = self.file.tensor(name).ok_or_else(|| missing(name))?;
        Ok(u32::from(t.shift().unwrap_or(0)))
    }

    fn put(&mut self, name: &str, words: &[u32]) -> Result<wgpu::Buffer> {
        let bytes = words_to_bytes(words);
        self.bytes += bytes.len() as u64;
        self.dev.storage(name, &bytes)
    }

    fn i32s(&self, name: &str) -> Result<Vec<i32>> {
        match self.data(name)? {
            TensorData::I32(t) => Ok(t.data().to_vec()),
            _ => Err(missing(name)),
        }
    }

    fn i32_buf(&mut self, name: &str) -> Result<wgpu::Buffer> {
        let v = self.i32s(name)?;
        self.put(name, &i32_words(&v))
    }

    fn rescale1(&self, name: &str) -> Result<(i32, u32)> {
        let m = self.i32s(name)?.first().copied().unwrap_or(0);
        Ok((m, self.shift(name)?))
    }

    /// An `i8 [rows, cols]` matrix, packed with the stride `stride_words(cols)`.
    fn packed(&mut self, name: &str) -> Result<(wgpu::Buffer, u32, u32)> {
        let (rows, cols, data) = match self.data(name)? {
            TensorData::I8(t) => match t.shape().dims() {
                [r, c] => (*r, *c, t.data()),
                _ => return Err(missing(name)),
            },
            _ => return Err(missing(name)),
        };
        let kw = stride_words(cols);
        let words = pack_i8(data, rows, cols, kw);
        Ok((self.put(name, &words)?, u32_of(rows)?, u32_of(kw)?))
    }

    fn linear(&mut self, prefix: &str) -> Result<LinearW> {
        let (w, rows, kw) = self.packed(&format!("{prefix}.w"))?;
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
        let g: Vec<u32> = match self.data(&gname)? {
            TensorData::I16(t) => t.data().iter().map(|&x| i32::from(x).cast_unsigned()).collect(),
            _ => return Err(missing(&gname)),
        };
        let (m, s) = self.rescale1(&format!("{prefix}.rescale"))?;
        Ok(NormW { g: self.put(&gname, &g)?, b: self.i32_buf(&format!("{prefix}.b"))?, m, s })
    }

    fn layer(&mut self, l: usize) -> Result<LayerW> {
        let p = format!("h.{l}");
        let gname = format!("{p}.mlp.gelu");
        let gelu: Vec<u32> = match self.data(&gname)? {
            TensorData::I8(t) => t.data().iter().map(|&x| i32::from(x).cast_unsigned()).collect(),
            _ => return Err(missing(&gname)),
        };
        let gelu_len = u32_of(gelu.len())?;
        Ok(LayerW {
            ln_1: self.norm(&format!("{p}.ln_1"))?,
            c_attn: self.linear(&format!("{p}.attn.c_attn"))?,
            score: self.rescale1(&format!("{p}.attn.score.rescale"))?,
            attn_out: self.rescale1(&format!("{p}.attn.out.rescale"))?,
            c_proj: self.linear(&format!("{p}.attn.c_proj"))?,
            ln_2: self.norm(&format!("{p}.ln_2"))?,
            c_fc: self.linear(&format!("{p}.mlp.c_fc"))?,
            gelu: self.put(&gname, &gelu)?,
            gelu_offset: gelu_len / 2,
            gelu_len,
            mlp_proj: self.linear(&format!("{p}.mlp.c_proj"))?,
        })
    }
}

fn check_supported(d: u32, d_ff: u32, head_dim: u32) -> Result<()> {
    if !d.is_multiple_of(4) || !d_ff.is_multiple_of(4) {
        return Err(Error::Unsupported { reason: "d_model and d_ff must be multiples of 4" });
    }
    if !head_dim.is_multiple_of(4) || head_dim > 256 {
        return Err(Error::Unsupported { reason: "head_dim must be a multiple of 4, at most 256" });
    }
    Ok(())
}

impl GpuModel {
    /// Loads a GPT-2 model file and copies its weights to the GPU of `dev`. Cost: the CPU
    /// load (one SHA-256 of the file), a second parse, and one upload of about the file size.
    ///
    /// # Errors
    /// Every error of `hardshell::gpt2::Model::load` (as [`Error::Model`]);
    /// [`Error::Unsupported`] for a shape that the kernels do not support;
    /// [`Error::DeviceLimit`] if a tensor does not fit in one binding; [`Error::Gpu`] on a GPU
    /// error.
    pub fn load(dev: &GpuDevice, bytes: &[u8]) -> Result<Self> {
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
        check_supported(c.d_model, c.d_ff, head_dim)?;
        let dims = Dims {
            n_layer: c.n_layer,
            n_head: c.n_head,
            d: c.d_model,
            d_ff: c.d_ff,
            head_dim,
            vocab: c.vocab_size,
            n_ctx: c.n_ctx,
            dw: u32_of(stride_words(c.d_model as usize))?,
            fw: u32_of(stride_words(c.d_ff as usize))?,
        };
        let k = Kernels::new(dev)?;
        let mut up = Uploader { dev, file: &file, bytes: 0 };
        let (wte, _, _) = up.packed("wte")?;
        let embed_m = up.i32_buf("wte.rescale_embed")?;
        let embed_s = up.shift("wte.rescale_embed")?;
        let logit_m = up.i32_buf("wte.rescale_logits")?;
        let logit_s = up.shift("wte.rescale_logits")?;
        let (wpe, _, _) = up.packed("wpe")?;
        let pos_m = up.i32_buf("wpe.rescale")?;
        let pos_s = up.shift("wpe.rescale")?;
        let layers = (0..c.n_layer as usize).map(|l| up.layer(l)).collect::<Result<Vec<_>>>()?;
        let ln_f = up.norm("ln_f")?;
        let attn = up.i32s("attn_exp")?;
        let attn_len = u32_of(attn.len())?;
        let attn_exp = up.put("attn_exp", &i32_words(&attn))?;
        let lexp: Vec<u32> = match up.data("logits_exp")? {
            TensorData::U64(t) => t
                .data()
                .iter()
                .flat_map(|&x| [(x & 0xffff_ffff) as u32, (x >> 32) as u32])
                .collect(),
            _ => return Err(missing("logits_exp")),
        };
        let logits_len = u32_of(lexp.len() / 2)?;
        let logits_exp = up.put("logits_exp", &lexp)?;
        let gpu_bytes = up.bytes;
        dev.check()?;
        Ok(Self {
            dev: dev.clone(),
            k,
            dims,
            eot,
            hash,
            ln: (u32::from(c.constants.ln_k), u32::from(c.constants.ln_j), c.constants.ln_eps),
            softmax_p: u32::from(c.constants.softmax_p),
            wte,
            embed_m,
            embed_s,
            logit_m,
            logit_s,
            wpe,
            pos_m,
            pos_s,
            layers,
            ln_f,
            attn_exp,
            attn_len,
            logits_exp,
            logits_len,
            gpu_bytes,
        })
    }

    /// The SHA-256 of the model file.
    #[must_use]
    pub fn hash(&self) -> ModelHash {
        self.hash
    }

    /// The number of tokens in the vocabulary: the length of each weight vector.
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
    pub fn device(&self) -> &GpuDevice {
        &self.dev
    }

    /// The bytes of GPU memory that the weights use (measured from the uploads).
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.gpu_bytes
    }

    /// The bytes of GPU memory that one slot of a batch uses: the KV cache, plus the work
    /// buffers of one row (est. from the buffer sizes in `batch`).
    #[must_use]
    pub fn slot_bytes(&self) -> u64 {
        let d = self.dims;
        let cache = 2 * u64::from(d.n_layer) * u64::from(d.n_ctx) * u64::from(d.d);
        let scores = 8 * u64::from(d.n_head) * u64::from(d.n_ctx);
        let rows = 4 * (3 * u64::from(d.dw) + u64::from(d.fw) + 2 * u64::from(d.d))
            + 4 * u64::from(d.d_ff.max(3 * d.d).max(d.vocab))
            + 4 * u64::from(d.d_ff)
            + 3 * 8 * u64::from(d.vocab);
        cache + scores + rows
    }
}
