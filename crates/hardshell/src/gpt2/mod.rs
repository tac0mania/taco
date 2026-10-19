//! GPT-2 in integers: token and learned position embeddings, pre-LayerNorm blocks with causal
//! multi-head attention and a KV cache, a GELU MLP, a final LayerNorm, and the output logits
//! with the tied token embedding.
//!
//! PLAN(§6): GPT-2 small is the phase 2 model. This module is generic over the dimensions in
//! the model file, so the tiny test model and GPT-2 small use the same code.
//!
//! # The forward pass for one token at position `t`
//!
//! The residual stream `x` is `i32` (ADR 0004 section 1). `R = 2^30 - 1` is the residual limit.
//! "rescale" is `int::rescale` with the multiplier of the row and the shift of the tensor.
//!
//! 1. `x[j] = clamp(rescale(wte[tok][j], embed_m[tok]) + rescale(wpe[t][j], pos_m[t]), -R, R)`.
//! 2. For each block:
//!    1. `a = ln_1(x)` (`i8`).
//!    2. `qkv[r] = clamp(rescale(c_attn.w[r] . a + c_attn.b[r], c_attn.m[r]), -127, 127)`.
//!       Rows `0..d` are `q`, `d..2d` are `k`, `2d..3d` are `v`. `k` and `v` go into the cache
//!       at position `t`.
//!    3. For each head `h` and each position `j <= t` (causal):
//!       `score[j] = clamp(rescale(q_h . k_h[j], score_m), -2^40, 2^40)`. Then
//!       `p = softmax(score)` (`softmax` module, plan §7.5), and
//!       `att_h[c] = clamp(rescale(sum_j p[j] * v_h[j][c], out_m), -127, 127)`.
//!    4. `x[r] = clamp(x[r] + rescale(c_proj.w[r] . att + c_proj.b[r], c_proj.m[r]), -R, R)`.
//!    5. `a = ln_2(x)`.
//!    6. `g[r] = gelu(clamp(rescale(c_fc.w[r] . a + c_fc.b[r], c_fc.m[r]), lo, hi))`, where
//!       `lo..=hi` is the input range of the GELU table.
//!    7. `x[r] = clamp(x[r] + rescale(mlp_proj.w[r] . g + mlp_proj.b[r], mlp_proj.m[r]), -R, R)`.
//! 3. `a = ln_f(x)`.
//! 4. `logit[v] = clamp(rescale(wte[v] . a, logit_m[v]), -2^40, 2^40)`.
//!
//! A model file of format version 2 (ADR 0004 addendum B) can widen two activations to `i16`
//! (`-32767..=32767`): the output `g` of an `i16` GELU table (step 2.6), and the output `a` of
//! `ln_f` with the header constant `ln_f_bits = 16` (step 3). Then the matrix product that
//! reads it (step 2.7 or step 4) uses `i64` accumulators. With the header constant
//! `ln_beta_frac`, every LayerNorm adds its beta before its one rounding
//! (`LayerNorm::with_beta_frac`). All other steps do not change.
//! 5. `weight[v] = logits_exp[clamp(max(logit) - logit[v], 0, len - 1)]` (plan §7.6).
//!
//! # Tensor names
//!
//! `{l}` is the block index, from 0. `d` is `d_model`, `f` is `d_ff`, `V` is `vocab_size`,
//! `C` is `n_ctx`. A "rescale" tensor is `i32` and has a `shift` in the header. The converter
//! writes the tensors in this order; the hash of the file depends on the order.
//!
//! | Name | Type | Shape | Meaning |
//! |---|---|---|---|
//! | `wte` | `i8` | `[V, d]` | token embedding, also the output projection (tied) |
//! | `wte.rescale_embed` | rescale | `[V]` | `wte` row to residual units |
//! | `wte.rescale_logits` | rescale | `[V]` | `wte[v] . a` to logit units |
//! | `wpe` | `i8` | `[C, d]` | position embedding |
//! | `wpe.rescale` | rescale | `[C]` | `wpe` row to residual units |
//! | `h.{l}.ln_1.g` | `i16` | `[d]` | LayerNorm gamma |
//! | `h.{l}.ln_1.b` | `i32` | `[d]` | LayerNorm beta, output units |
//! | `h.{l}.ln_1.rescale` | rescale | `[1]` | LayerNorm `(m, s)` |
//! | `h.{l}.attn.c_attn.w` | `i8` | `[3d, d]` | q, k, v projection, `[out, in]` |
//! | `h.{l}.attn.c_attn.b` | `i32` | `[3d]` | bias, accumulator units |
//! | `h.{l}.attn.c_attn.rescale` | rescale | `[3d]` | accumulator to `i8` q, k, v |
//! | `h.{l}.attn.score.rescale` | rescale | `[1]` | `q . k` to score units |
//! | `h.{l}.attn.out.rescale` | rescale | `[1]` | `sum p v` to `i8` |
//! | `h.{l}.attn.c_proj.w` | `i8` | `[d, d]` | attention output projection |
//! | `h.{l}.attn.c_proj.b` | `i32` | `[d]` | bias |
//! | `h.{l}.attn.c_proj.rescale` | rescale | `[d]` | accumulator to residual units |
//! | `h.{l}.ln_2.g`, `.b`, `.rescale` | | | as `ln_1` |
//! | `h.{l}.mlp.c_fc.w` | `i8` | `[f, d]` | MLP input projection |
//! | `h.{l}.mlp.c_fc.b` | `i32` | `[f]` | bias |
//! | `h.{l}.mlp.c_fc.rescale` | rescale | `[f]` | accumulator to GELU input units |
//! | `h.{l}.mlp.gelu` | `i8` (or `i16`, version 2) | `[256]` or `[65536]` | GELU table (plan §7.4) |
//! | `h.{l}.mlp.c_proj.w` | `i8` | `[d, f]` | MLP output projection |
//! | `h.{l}.mlp.c_proj.b` | `i32` | `[d]` | bias |
//! | `h.{l}.mlp.c_proj.rescale` | rescale | `[d]` | accumulator to residual units |
//! | `ln_f.g`, `ln_f.b`, `ln_f.rescale` | | | final LayerNorm |
//! | `attn_exp` | `i32` | `[L]` | softmax exponent table (plan §7.5) |
//! | `logits_exp` | `u64` | `[L]` | logit exponent table (plan §7.6) |
//!
//! The model file has no other tensors. [`Model::load`] refuses a missing tensor, an extra
//! tensor, a wrong type, a wrong shape and a rescale tensor without a shift.
//!
//! # Paths
//!
//! - [`ReferenceSession`]: the reference, one token at a time with the scalar kernels.
//! - [`Session`]: the same API, with the fast kernels, and [`Session::step_many`] for many
//!   known tokens at a time (the prefill path of the encoder).
//! - [`Batch`]: many sessions in lockstep (the batch path of the decoder).
//!
//! All paths give the same weights, bit for bit. They are built from the op kernels of the
//! private module `ops` (`embed`, `norm`, `linear`, `attention`, `logits`).
//!
//! # Context (plan §7.7)
//!
//! A block starts with an empty context and the `<|endoftext|>` token: call
//! [`Session::start_block`]. When the context is full (`n_ctx` tokens) and the caller gives one
//! more token, the session starts again: it empties the context, runs `<|endoftext|>` at
//! position 0, then the last `K` tokens of the block at the positions `1..=K` ([`Overlap`],
//! ADR 0015), then the token at position `K + 1`. With the default `K = 0` this is the rule of
//! ADR 0004 section 9. The encoder and the decoder use the same `K` (the container stores
//! it), so they stay in step.

mod attention;
mod batch;
mod context;
mod engine;
mod model;
mod ops;
mod reference;
mod session;
mod stream;

pub use batch::{Batch, BatchInput};
pub use context::Overlap;
pub use model::Model;
pub use reference::ReferenceSession;
pub use session::{Session, SessionConfig, StepWeights};
pub use stream::{StreamState, Truncation};

/// The residual limit `R = 2^30 - 1` (ADR 0004 section 1).
pub const RESIDUAL_LIMIT: i32 = (1 << 30) - 1;

/// The limit of a score or a logit after its rescale: `2^40`. It keeps `max - x` in range.
pub const SCORE_LIMIT: i64 = 1 << 40;

/// The largest number of blocks.
pub const MAX_LAYERS: u32 = 1024;
/// The largest context length.
pub const MAX_CTX: u32 = 1 << 16;
