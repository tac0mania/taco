# Design: any model (a model graph and an automatic converter)

Status: draft 1, 2026-10-18. Author: the anymodel design agent. Owner: Rohit. This document
obeys `docs/STYLE.md`. It changes no code. ADR 0009 (proposed) records the format decision.

Labels: "published" means a source states the number (§14 lists the sources). "measured" means
a Taco measurement of 2026-10-18. "est." means an estimate of this agent. Each estimate gives
its base.

## §1 The question and the answer in short

The owner asks three questions:

1. Can one general tool convert (or train) many LLMs from float to integers, with no new code
   for each model?
2. How long does a conversion take?
3. Does it run on the GPU or on the CPU?

The answers:

1. **Mostly yes, but not with no code at all.** A model that uses an operation that HardShell
   does not have always needs new engine code. A fixed set of 10 operations (§4) covers the
   main open decoder families on Hugging Face. These are GPT-2, GPT-J, GPT-NeoX/Pythia, Llama 2
   and 3, Mistral, Qwen 2 and 3, Gemma 1 to 3, Phi 2 to 4, SmolLM and OLMo 2. Inside a
   family, a new checkpoint needs no code. A new family needs a short architecture map in Python (about 100
   lines, est.). Mixture-of-experts and state-space models need new operations (§3.5).
2. **HardShell becomes an interpreter.** The `.hsm` file (format version 2) holds a graph of
   operations (§5). Every element-wise activation becomes a look-up table in the file, so a
   new activation needs no engine code. Version 1 files (GPT-2) stay readable.
3. **The converter searches the recipe itself.** It measures the loss on calibration windows
   inside a fixed budget (§6). For each site, it selects the clip statistic, the type (`i8` or
   `i16`) and the corrections. The manual diagnosis of GPT-2 (the `clip` recipe
   and the beta correction) becomes the first test of the search.
4. **Time and hardware (§7).** The search and an optional fine-tune run on the devbox GPU. The
   final build of the model file runs on the CPU, so that every machine makes the same file.
   On devbox, a 1B model converts in about 1 hour (est.). A 3B model needs 2 to 3 hours
   (est.), and a 7B model 4 to 6 hours (est.). The HardShell verification takes about as long
   as the search.
5. **Recommendation (§12).** Do the float survey and the automatic search on GPT-2 now. They
   touch only `tools/`. Build the graph interpreter after the SIMD and CUDA work lands. Ask
   those agents now to keep their kernels at the level of one operation.

## §2 Terms

This document uses the terms of plan §2, and these terms:

| Term | Meaning |
|---|---|
| Layer | One transformer layer (GPT-2 code calls it a "block"; here "block" is only the container block of plan §9). |
| Op | One operation of the model graph, for example `linear` or `norm`. |
| Value | A named integer vector that one op writes and other ops read, in one step. |
| Graph | The ordered list of ops that HardShell runs for each token. |
| Site | A value that has a calibrated scale. The recipe selects its range and its type. |
| Recipe | The discrete choices for each site: range statistic, type, corrections. |
| Architecture map | A Python file that changes one Hugging Face model family to a graph. |
| Family | Models with one architecture map, for example all `LlamaForCausalLM` checkpoints. |
| Step | One HardShell forward pass for one token. |

## §3 What can be generic and what cannot

### §3.1 The families

The table gives the facts from the `config.json` of each model and from the Hugging Face
`transformers` modeling code (published, read 2026-10-18; sources in §14). All models are
decoder-only transformers with causal attention and a KV cache.

| Family (model_type) | Example | Norm | Activation, MLP | Positions | Attention | Other |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2 small, 124M | LayerNorm, pre | `gelu_new`, 2 matrices | learned | MHA | tied embedding, biases everywhere |
| gptj | GPT-J 6B | LayerNorm, one per layer | `gelu_new`, 2 matrices | RoPE, `rotary_dim` 64 of 256, pairs interleaved | MHA, no q/k/v bias | parallel residual, `lm_head` bias |
| gpt_neox | Pythia 1B | LayerNorm, two per layer | `gelu`, 2 matrices | RoPE, `rotary_pct` 0.25, halves | MHA, biases | parallel residual (`use_parallel_residual`) |
| llama | Llama 2 7B, Llama 3.2 1B/3B, SmolLM2 | RMSNorm, pre | SiLU, SwiGLU (3 matrices) | RoPE; Llama 3 `rope_scaling` type `llama3` | GQA (Llama 3.2 1B: 32 q heads, 8 kv heads) | no biases; Llama 3.2 and SmolLM2 tie the embedding |
| mistral | Mistral 7B v0.1 | RMSNorm | SwiGLU | RoPE, base 10,000 | GQA 32/8, `sliding_window` 4,096 | no biases |
| qwen2 | Qwen2.5 1.5B | RMSNorm, eps 1e-6 | SwiGLU | RoPE, base 1,000,000 | GQA 12/2, **bias on q, k, v** | tied embedding |
| qwen3 | Qwen3 1.7B, 4B | RMSNorm | SwiGLU | RoPE, base 1,000,000 | GQA, **RMSNorm of q and k for each head** (QK-norm) | `head_dim` 128 set apart from the width |
| olmo2 | OLMo 2 1B, 7B | RMSNorm **after** each sublayer, before the residual add | SwiGLU | RoPE, base 500,000 | QK-norm on the full q and k width | no pre-norm |
| gemma | Gemma 2B | RMSNorm with `(1 + w)` | GeGLU (`gelu`) | RoPE | MQA (1 kv head), `head_dim` 256 | embedding times `sqrt(d)` |
| gemma2 | Gemma 2 2B | RMSNorm `(1 + w)`, pre **and** post each sublayer (4 norms) | GeGLU (`gelu_pytorch_tanh`) | RoPE | GQA 8/4, sliding window 4,096 on alternate layers, score scale `query_pre_attn_scalar` | **soft-cap** of scores (50) and of logits (30) with `tanh` |
| gemma3_text | Gemma 3 1B | as Gemma 2, with QK-norm | GeGLU | RoPE, base 1,000,000 global, 10,000 local | 5 local layers (window 512) for each global layer | no soft-cap (`null`) |
| phi | Phi-2 | LayerNorm, one per layer | `gelu_new` | RoPE, `partial_rotary_factor` 0.4 | MHA, biases | parallel residual, `lm_head` bias |
| phi3 | Phi-3 mini, Phi-4 | RMSNorm | SwiGLU, `gate_up_proj` in one matrix | RoPE (LongRoPE in 128k variants) | `qkv_proj` in one matrix; Phi-4 GQA 40/10 | Phi-3 mini config has `sliding_window` 2,047 |

### §3.2 The differences, and how each one becomes generic

| Difference | Generic form | Engine code? |
|---|---|---|
| LayerNorm or RMSNorm | One `norm` op with a mode: `layer` (subtract the mean) or `rms` (no mean). | One small addition to the LayerNorm kernel |
| Gemma `(1 + w)`, embedding times `sqrt(d)`, `query_pre_attn_scalar`, `1/sqrt(head_dim)` | The converter puts each constant into a `gamma` or a rescale. | No |
| GELU (erf, tanh, `gelu_new`), SiLU, ReLU, soft-cap `c * tanh(x / c)` | A `table` op over the integer input (§3.3). | No |
| SwiGLU and GeGLU: `act(g) * u` | A `table` op for `act(g)`, then a `mul` op. | The `mul` op, one time |
| Learned positions | An `embed` op with a position table (GPT-2). | No (version 1 has it) |
| RoPE: base, `llama3` scaling, LongRoPE, partial rotary | The converter calculates the cosine and sine table for each position up to `n_ctx`. One `rope` op reads it. | The `rope` op, one time |
| RoPE pairs: halves (`rotate_half`) or interleaved (GPT-J) | The converter permutes the rows of `W_q` and `W_k` to one layout. The dot product `q . k` does not change when q and k have the same permutation. | No |
| Partial rotary | The converter permutes the rotated channels to the start. `rope` has a `rotary_dim`. | No |
| MHA, GQA, MQA | The `attention` op has `n_head` and `n_kv_head`. Query head `h` reads kv head `h / (n_head / n_kv_head)`, as `repeat_kv` in transformers. | Part of `attention`, one time |
| Sliding window, local and global layers | An integer `window` on each `attention` op (0 means the full context). | Part of `attention`, one time |
| Score soft-cap (Gemma 2) | An optional table inside `attention`, applied to each score. | Part of `attention`, one time |
| Logit soft-cap (Gemma 2) | An optional table inside `logits`. | Part of `logits`, one time |
| QK-norm for each head (Qwen3, Gemma 3) or full width (OLMo 2) | A `norm` op with a `group` length (`head_dim` or the full width). | Part of `norm` |
| Biases or no biases | Each `linear` op has an optional bias. | No |
| Fused `qkv_proj`, `gate_up_proj` | The converter divides the matrix into separate `linear` ops. The bits do not change, because each row is independent. | No |
| Tied or separate output matrix | The `logits` op names a tensor. It can be the embedding tensor. | No |
| Pre-norm, post-norm (OLMo 2), pre and post (Gemma 2), parallel residual (GPT-J, NeoX, Phi-2) | Different wiring of the same ops in the graph. | No |
| `lm_head` bias (GPT-J, Phi-2) | An optional bias on `logits`. | No |

### §3.3 Key idea: every element-wise activation is a table

An activation `y = f(x)` of one integer input `x` has a finite set of inputs. If `x` is `i16`,
the set has 65,536 values. Thus `f` is exactly a table of 65,536 entries. HardShell clamps the
index and reads the entry (ADR 0004 section 3 already does this for GELU). The evaluation:

- **It covers all unary activations.** GELU in each form, SiLU, sigmoid, ReLU, squared ReLU and
  the soft-cap are tables. The table also contains the input scale and the output scale.
- **It gives determinism with no work.** HardShell does no arithmetic, only a look-up. The
  float work happens one time in the converter.
- **It is accurate.** The `i16` GELU table has a maximum error of 0.502 LSB of the output
  (measured, ADR 0004). The error comes from the output rounding only.
- **It is cheap.** A table of 65,536 `i16` entries uses 128 KiB. It fits in the L2 cache.
- **It does not cover functions of two inputs.** The gate product of SwiGLU needs a `mul` op.
  Softmax and the norms need their own ops, because they use a full row.
- **It needs a bounded input.** An `i32` input needs a clamp to a range of at most `2^20`
  entries (est. limit, 4 MiB for `i32` entries). The logit soft-cap of Gemma 2 needs about
  61,000 entries at the logit unit `2^-8` (est.). The base: `tanh` is flat after 4 times the
  cap, so the table has `2 * 4 * 30 * 256` entries.
- **The converter needs a deterministic float function.** The table must be the same on every
  machine. The converter uses only IEEE 754 basic operations (`convert.numerics`), not the
  platform `libm`. A new function (for example `erf`) needs a few lines there, not in
  HardShell.

### §3.4 Tokenizers

Plan §3 rule 5 says that any byte string can be tokenized and restored. The `tokenizer.json`
files give two kinds (published, read 2026-10-18):

| Kind | Models | Facts | Rule for Taco |
|---|---|---|---|
| Byte-level BPE | GPT-2, Llama 3, Qwen 2/3, OLMo 2, SmolLM2 | `ByteLevel` pre-tokenizer, `byte_fallback` false; all 256 bytes are base tokens | As GPT-2 today. The merges are pairs of token ids. |
| BPE with byte fallback (SentencePiece style) | Llama 2, Mistral, Gemma 1/2, Phi-3 mini | `byte_fallback` true; tokens `<0x00>` to `<0xFF>`; `▁` (U+2581) means a space | The converter writes the bytes of each token: `▁` becomes 0x20, `<0xNN>` becomes the byte. The encoder starts from the characters that have a token, and uses byte tokens for every other byte. |

Rules that keep the round trip exact:

1. **The decoder uses only the vocabulary bytes.** This does not change (plan §10). A token
   id always gives the same bytes.
2. **The encoder skips the normalizer.** Qwen3 has an `NFC` normalizer (published). NFC changes
   bytes, so Taco never applies it. Llama 2 prepends `▁` to the text. Taco does not prepend it.
3. **An ambiguous character uses byte tokens.** A literal `▁` in the input becomes three byte
   tokens (`E2 96 81`), never the space token. Thus decoding gives the input bytes.
4. **Two tokens can have the same bytes.** For example `<0x41>` and `A`. The decoder does not
   care. The encoder selects the lower id. The duplicate takes a little probability mass, so
   the ratio is a little worse (not measured).
5. **The pre-tokenizer is a quality choice, not a correctness rule.** Llama 3 and Qwen use a
   regex that differs from GPT-2. A different regex gives valid tokens, but the model sees
   unusual splits, so the bits increase. The model file names the regex (`gpt2`, `llama3`,
   `none`). Each regex is code in the `taco` crate, not in HardShell.
6. **The start token is a field.** GPT-2 starts a block with `<|endoftext|>`. Llama, Gemma and
   Phi expect a BOS token. Qwen has no BOS, so it uses `<|endoftext|>` (id 151,643). The
   header field `start_token` replaces `eot_token`.

Unigram SentencePiece models (for example T5) are out of scope. No family in §3.1 uses one in
its `tokenizer.json` (published, the 8 files that this agent read).

### §3.5 Out of scope, or later

| Kind | Examples | Why | When |
|---|---|---|---|
| Mixture of experts | Mixtral, Qwen3-MoE, OLMoE, DeepSeek | A router selects the top-k experts. Integer top-k is deterministic (ties to the lower id), but the files are large (Mixtral 8x7B: about 47B parameters, published). | A `moe` op, its own ADR, after a dense 1B model works |
| State-space and recurrent models | Mamba, RWKV | A recurrent state with decays and a selective scan. Each needs new ops and its own error analysis, because the state accumulates rounding error over the full block. | Research only |
| ALiBi attention bias | BLOOM, MPT | A bias for each distance. It is easy to add to `attention` as a table. | When a model needs it |
| Encoder-decoder, multimodal | T5, LLaVA | Not a next-token model of bytes. | Not planned |

## §4 The op set

Each op has exact integer semantics, written in the same way as the GPT-2 forward pass in
`crates/hardshell/src/gpt2/mod.rs`. Each op uses only the rules of plan §7.2 (`hardshell::int`).
The loader checks every bound that an op needs, so no model file can cause an overflow.

| Op | Inputs | Output | Semantics in short | New code? |
|---|---|---|---|---|
| `embed` | `token`, and `pos` optional | `i32` residual | `clamp(rescale(E[token]) + rescale(P[pos]), -R, R)`, one multiplier for each row | No (GPT-2 step 1) |
| `norm` | `i32` | `i8`, `i16` or `i32` | Mode `layer` or `rms`. Groups of `group` channels. `gamma` (`i16`) and `beta` (`i32`, optional). Constants `K`, `J`, `eps` as plan §7.3. | `rms` mode, groups, wider outputs |
| `linear` | `i8` or `i16` | `i8`, `i16` or `i32` | `clamp(rescale(W[r] . x + b[r], m[r], s))`. `i8` weights, one multiplier for each row. Bias optional. | `i16` input exists as a kernel (`matvec_i8_i16`) |
| `add` | two `i32` | `i32` | `clamp(a + b, -R, R)` | No |
| `table` | any integer | `i8`, `i16` or `i32` | `T[clamp(x + offset, 0, len - 1)]` | Generalises GELU |
| `mul` | two `i8` or `i16` | `i8` or `i16` | `clamp(rescale(a[c] * b[c], m, s))` | Yes, small |
| `rope` | `i8` or `i16` | `i8` or `i16` | For each pair `(c, c + rotary_dim / 2)` of each head: `x1 cos - x2 sin` and `x2 cos + x1 sin`, from an `i16` table at row `pos`, then one rescale. Other channels: the same rescale only. | Yes |
| `attention` | `q`, `k`, `v` | `i8` or `i16` | Writes `k` and `v` into its cache at `pos`. Causal, `n_head`, `n_kv_head`, `window`. Scores, optional soft-cap table, softmax (plan §7.5), output rescale. | GQA, window, soft-cap, `i16` scores |
| `logits` | `i8` or `i16` | `u64` weights | `W . x + b` (bias optional), rescale to the logit unit, optional soft-cap table, exponent table (plan §7.6) | Bias and soft-cap |
| `hadamard` (later) | `i8` or `i16` | `i8` or `i16` | An integer Walsh-Hadamard transform of blocks of `2^k` channels (additions and subtractions only), then a rescale | Yes, only for online rotations (§9.2) |

NOTE: `rope` with an `i8` input loses precision, because the rotation mixes two channels with
different values. The recipe search (§6.6) can select `i16` for `q` and `k`. Then the score dot
product is `i16 x i16` with an `i64` accumulator. HardShell needs that kernel.

## §5 The model graph in the `.hsm` file (format version 2)

### §5.1 Principles

1. **The file describes the computation.** HardShell does not know a model family. It runs
   the ops of the graph in file order for each step.
2. **One graph has one encoding.** The graph lists each op of each layer. It has no loops and
   no templates. Thus the writer has one way to write a graph. One model has one file and one
   model hash (ADR 0004 section 8 rule 3).
3. **Every value is written one time in a step.** A value has a name, a type and a length. The
   interpreter can reuse buffers, but the reuse cannot change a result.
4. **Unknown is an error.** An unknown op, field, type or name is an error, as in version 1.
   All numbers stay non-negative integers. A negative constant uses an `offset` field.
5. **The ops do not carry scales.** The converter puts all scales into rescales and tables.
   The file has no float.

### §5.2 Layout

The fixed part, the alignment, the vocabulary and the merges section do not change. The
version field becomes 2. The JSON header changes:

```json
{
  "model": {
    "architecture": "graph",
    "vocab_size": 49152, "n_ctx": 2048, "start_token": 0,
    "tokenizer": {"kind": "byte_bpe", "pretokenizer": "gpt2"},
    "constants": {"softmax_p": 30}
  },
  "vocabulary": {"offset": 4096, "length": 312345},
  "merges": {"offset": 316480, "length": 391200},
  "values": [
    {"name": "x.0", "dtype": "i32", "len": 576},
    {"name": "l.0.a", "dtype": "i8", "len": 576}
  ],
  "caches": [
    {"name": "l.0.kv", "n_kv_head": 3, "head_dim": 64, "dtype": "i8", "window": 0}
  ],
  "graph": [
    {"op": "embed", "out": "x.0", "w": "tok_emb", "rescale": "tok_emb.rescale"},
    {"op": "norm", "mode": "rms", "in": "x.0", "out": "l.0.a", "group": 576,
     "gamma": "l.0.ln_1.g", "rescale": "l.0.ln_1.rescale",
     "ln_k": 48, "ln_j": 8, "ln_eps": 2366912989}
  ],
  "tensors": [
    {"name": "tok_emb", "dtype": "i8", "shape": [49152, 576], "offset": 708000,
     "length": 28311552}
  ]
}
```

Rules:

1. `values` lists every value with its type and its length. `token` and `pos` are built-in
   inputs. The last op is `logits`. It writes the built-in output `weights`.
2. Each op entry has a fixed set of fields for its kind. A field names a value, a cache or a
   tensor, or holds an integer attribute.
3. A rescale is a tensor with a `shift`, as in version 1. A table is an ordinary tensor. Two
   ops can name the same tensor (for example a tied embedding, or one soft-cap table for all
   layers).
4. The tensors follow in the order of their first use in the graph. Thus the layout stays
   canonical.
5. The constants `ln_k`, `ln_j` and `ln_eps` move into each `norm` op, because QK-norm and the
   layer norms can need different values.

### §5.3 How tables and rescales attach to ops

| Op | Tensors that it names |
|---|---|
| `embed` | `w` (`i8 [V, d]`), `rescale` (`[V]`), optional `pos_w` and `pos_rescale` |
| `norm` | `gamma` (`i16 [group]`), optional `beta` (`i32 [n]`), `rescale` (`[1]`) |
| `linear` | `w` (`i8 [out, in]`), optional `b` (`i32 [out]`), `rescale` (`[out]`) |
| `table` | `t` (`[len]`), with the attribute `offset` |
| `mul` | `rescale` (`[1]`) |
| `rope` | `t` (`i16 [n_ctx, rotary_dim]`: cosines, then sines), `rescale` (`[1]`) |
| `attention` | `score_rescale` (`[1]`), `exp` (the `attn_exp` table), optional `cap` (table) with `cap_offset`, `out_rescale` (`[1]`) |
| `logits` | `w` (`i8 [V, d]`), optional `b`, `rescale` (`[V]`), optional `cap` with `cap_offset`, `exp` (the `logits_exp` table) |

### §5.4 The KV cache

Each `attention` op names one entry of `caches`. The entry gives `n_kv_head`, `head_dim`, the
type and the `window`. The session allocates `2 * n_ctx * n_kv_head * head_dim` entries for
each cache before the first step. The memory is thus known from the header, before any step.

The semantics of a window `w > 0`: at position `t`, the op reads the positions
`max(0, t - w + 1)` to `t`. A ring buffer is permitted, because the semantics name positions,
not memory addresses. The cache holds `k` after `rope`, as transformers does. The context
restart of plan §7.7 empties all caches.

A later option (not in version 2): a constant start prefix. Position 0 always holds the start
token, so its `k` and `v` are the same in every block. The converter can calculate them in
float and store them as tensors. Then the integer path never runs position 0. GPT-2 has its
largest residual value at position 0 (3,055 against 463 elsewhere, measured), so this can
give a finer residual scale. The gain is not measured.

### §5.5 Determinism and the conformance vectors

1. Each op has one integer result, from the rules in `hardshell::int`. The order of the ops
   is the file order. Thus the graph has one result on every machine.
2. A fast path (SIMD, CUDA) implements ops, or fuses ops. A test compares it with the
   reference interpreter and fails on any difference (CLAUDE.md §1 rule 10).
3. The existing vectors stay valid. `conformance/tiny.hsm` (version 1, SHA-256
   `eb78b89a…18f8`) must give the same output hash (`d4063522…0078`) through the interpreter.
4. New vectors: `tools/convert/tiny.py` makes one tiny graph model for each feature set,
   from a fixed seed. Each model has a hash of each output. `scripts/xcheck` runs all of them
   on the Mac and on devbox. The models:
   - tiny-llama: RMSNorm, RoPE, GQA, SwiGLU, tied embedding.
   - tiny-gemma2: post norms, GeGLU, soft-caps, alternate windows.
   - tiny-qwen3: QK-norm for each head, q/k/v bias.
   - tiny-phi2: parallel residual, partial rotary, `lm_head` bias.
5. The graph is part of the model hash, because the hash covers the full file.

### §5.6 Version 1 files stay readable

CLAUDE.md §1 rule 8 needs a reader for the old version and a test. The proposal:

1. The reader of version 1 changes the GPT-2 tensor table into the equivalent graph in
   memory. The `c_attn` matrix becomes three `linear` ops on row ranges. Each row is
   independent, so the bits do not change.
2. The interpreter runs both versions. There is one engine, not a switch between an old and a
   new implementation (CLAUDE.md §8).
3. A test runs `conformance/tiny.hsm` and `gpt2.hsm` (version 1) through the interpreter and
   compares the hashes with the golden hashes of `HANDOFF.md` §2.
4. When the test passes, the `gpt2` forward-pass module goes away. Its documentation becomes
   the documentation of the version 1 graph.

Every op semantics in §4 must contain the GPT-2 semantics exactly. Example: GPT-2 adds the two
embedding rescales before the clamp (step 1 of `gpt2/mod.rs`). Thus `embed` has the optional
position table inside it, not as a separate `add`.

## §6 The automatic converter

### §6.1 Overview

```
config.json + *.safetensors + tokenizer.json   (Hugging Face; GGUF later)
  -> import: architecture map -> float graph (Python, the same ops in float)
  -> check: float graph against transformers on 2 windows
  -> transforms that keep the float function (fold constants, optional SmoothQuant, rotation)
  -> calibration, one layer at a time -> calib_stats.toml (committed)
  -> recipe search on the GPU (fake quantisation, fixed budget) -> recipe.json (committed)
  -> build on the CPU (NumPy, deterministic) -> model.hsm and its SHA-256
  -> verify: integer simulation and HardShell on held-out windows
  -> optional: quantisation-aware fine-tune, if the loss target fails
```

### §6.2 Import

1. **Read** `config.json`, the safetensors index and `tokenizer.json`. Open the safetensors
   files with memory maps. Read one tensor at a time and change it to float32 only when the
   converter needs it. Thus a 7B model converts on a machine with 8 GB.
2. **Select the architecture map** from `model_type`. The map gives the graph of one layer,
   the names of the Hugging Face tensors and the folds (§3.2). A map has about 50 to 150 lines
   of Python (est., from the size of `tools/convert/model.py`, 190 lines for GPT-2). A map
   refuses a config field that it does not know, for example a `rope_scaling` type.
3. **Check the map.** Run the float graph and the transformers model on 2 windows of 1,024
   tokens. The largest difference of a logit must be less than `1e-3` in float32 (est. limit).
   This check finds a wrong map before any quantisation work. It needs `torch` and
   `transformers` in `tools/` (a dependency question for the lead).
4. **GGUF (later).** GGUF files store the architecture and the tokenizer in metadata keys
   (`general.architecture`, `llama.block_count` and others, published). A GGUF file with
   `F16`, `BF16` or `Q8_0` weights can feed the same import. A `Q4_K_M` file (for example
   Nanbeige on devbox) is already quantised: a second quantisation adds error. Use it only
   when no float weights exist.

NOTE: an automatic trace of the modeling code (`torch.export`) is an alternative to the maps.
It gives a graph for any model, but the graph contains mask code, cache code and fused kernels.
The pattern match to HardShell ops then becomes the hard part. The maps are simpler for 12
families, and the check of step 3 makes them safe.

### §6.3 Transforms that keep the float function

These transforms change the weights, not the function in float. The float check of §6.2 step
3 runs again after them.

1. **Folds.** Gemma `(1 + w)` into `gamma`, `sqrt(d)` into the embedding rescale, the score
   scale into `score_rescale`, RoPE layouts into row permutations.
2. **SmoothQuant (optional, searched).** Divide each input channel of a linear layer by `s_c`
   and multiply the column `c` of the weight by `s_c`. Fold `1 / s_c` into the `gamma` of the
   norm before. This moves the outliers from the activations into the weights. The strength
   `alpha` is a search choice (published default 0.5).
3. **Rotation (optional, §9.2).** Multiply the residual stream by an orthogonal matrix `Q` and
   fold `Q` into all weights that read or write the residual. RMSNorm does not change, because
   `||Qx|| = ||x||`. A LayerNorm first becomes an RMSNorm (the SliceGPT method). No engine
   change is necessary.

### §6.4 Calibration

1. The calibration set is a committed list of windows with URLs and SHA-256 values, as
   `tools/convert/calibration.toml` today (11 windows, 11,264 tokens). For a large model, 32
   to 64 windows (est.) give more stable statistics. No window is in the evaluation set.
2. **One layer at a time.** Load the weights of one layer, run all windows through it, record
   the statistics, keep the output, free the layer. The memory need is one layer plus the
   hidden states (11,264 tokens x 4,096 x 4 bytes = 185 MB for a 7B model, est. calculation).
   This method is published practice (GPTQ quantises one layer at a time).
3. The statistics are the ones of `convert.calibrate` today. For each site: the largest
   value, high quantiles of `|x|`, and quantiles over the channels of each channel maximum.
   For each norm input: the row standard deviation.
4. The output `calib_stats.toml` is committed. The build reads only this file and the weights,
   so a GPU difference in the last float bits cannot change the model file. Each range is
   rounded to the grid `2^(k/16)` (as today).

### §6.5 Weight quantisation

1. The default is the current rule: `i8`, symmetric, one scale for each output row, round to
   nearest.
2. **GPTQ (optional, searched for each layer).** GPTQ rounds the weights of a row in order.
   It corrects the later weights for the error, with the Hessian of the calibration inputs.
   It quantises 175B parameters in about 4 GPU hours (published). At `i8` the gain is small (est.), so the
   search tries it only if the weights alone lose more than a limit.
3. **Bias correction (always).** This is the general form of the GPT-2 beta correction. When a
   value has a known constant error for each channel (for example the rounded `beta` of a
   norm), the converter subtracts `W_q . err` from the bias of the next `linear` op. If the op
   has no bias (Llama), the converter adds one. An integer bias costs one addition for each
   row. The data-free quantisation paper (Nagel et al., 2019) describes the same correction
   for the expected weight error.

### §6.6 The recipe search

**The choices for each site:**

| Choice | Candidates |
|---|---|
| Range statistic | `max`, `q99999`, `q9999`, `chan_q99`, `chan_q90`, `chan_q50`, and the range with the smallest mean squared error |
| Type | `i8` or `i16` (for `q`, `k`, the GLU product, the down-projection input, the norm outputs) |
| SmoothQuant `alpha` | off, 0.3, 0.5, 0.7 for each norm and linear pair |
| GPTQ | off or on, for each layer |

The bias correction is always on. It is a rule, not a choice.

**The objective.** The mean KL divergence of the next-token distribution against the float
model, in bits per byte, on the calibration windows. The KL is the extra size against the
float model, and it has less noise than the size itself. The search reports bits per byte too.

**The cost.** Each `i16` site costs speed and memory. The search adds a cost term:
`objective + lambda * (number of i16 sites) / (number of sites)`. The owner sets `lambda`, or a
hard limit on the number of `i16` sites.

**The method.** Fake quantisation in PyTorch on the GPU, as `convert.fq` today, for the full
model:

1. **Sensitivity.** For each site kind (about 10 kinds), quantise that kind only and measure
   the KL. This needs about 10 passes.
2. **Kinds.** For each kind, in the order of sensitivity, try each candidate for all layers
   together and keep the best. This needs about 50 passes (est.).
3. **Layers.** For the 20 most sensitive single sites (one layer, one kind), try the
   candidates for that site alone. This needs about 80 passes (est.).
4. **Confirm.** Build the integer model with the selected recipe. Measure it with the integer
   simulation (`convert.intsim`, generalised for the graph ops) on the calibration windows.
   The simulation agrees with HardShell to 1e-6 bits on a window (measured for GPT-2).

The budget is a fixed number of passes (default 150, est.) and a fixed set of windows. The
search stops at the budget or when a full round gives no gain. The output is `recipe.json`:
the choices only, with the measured objective. It is committed.

**The first test.** Run the search on GPT-2 with no manual input. It must find a recipe that
is at least as good as `clip` with the beta correction on the calibration windows. That
recipe gives 2.594 bits per byte (measured). Then measure it on the 19 held-out windows (`clip`: 3.067, measured).

NOTE: the GPT-2 result shows a trap. The `clip` saturation made the integer model better than
the float model on held-out files (−9.3 %, measured), because it removed very small
probabilities. A search that minimises bits per byte can thus prefer a model that is not near
the float model. The KL objective prefers the float model. The owner selects the objective;
this design proposes the KL for the search and bits per byte for the report.

### §6.7 Build and verify

1. The build reads the weights, `calib_stats.toml` and `recipe.json`. It writes the model file
   with NumPy on the CPU. Two machines must give the same SHA-256 (as for `gpt2.hsm` today).
   GPTQ and SmoothQuant must also be deterministic on the CPU, or their results (the integer
   weights) must be published.
2. HardShell runs the held-out windows. The report gives the loss against float for each file,
   as `docs/research/2026-10-18-integer-gpt2.md` §6.

### §6.8 Quantisation-aware fine-tune (optional)

Use it only when the search does not reach the loss target. The owner sets the target, for
example +1 % bits per byte on held-out text.

1. **Method.** Keep the integer model of the search frozen: the `i8` weights with their row
   scales. Add LoRA adapters (rank 16 to 64, est.) in bf16 on each linear layer. Train with the
   fake quantisation of the recipe in the forward pass and the straight-through estimator for
   the rounding. The clip ranges can also learn (published: LLM-QAT, EfficientQAT).
2. **Merge.** Add the adapter to the float weight, then quantise again. The merge can move a
   weight to a different integer. QA-LoRA (published) constrains the adapter so that the merge
   stays on the grid. Measure the loss after the merge, not before.
3. **Data.** General text with a known licence, about 20M tokens (est.), with no file of the
   evaluation set.
4. **Memory.** The frozen base is the `i8` model: 1.2 GB for 1B and 3.2 GB for 3B (est.). Both
   fit the 6 GB GPU of devbox with gradient checkpoints. A 7B base (7.2 GB) does not fit.
5. **Publication.** A fine-tuned model file cannot be rebuilt from the Hugging Face weights.
   Thus Taco must publish the model file or the adapter. The licence must permit it (§9.4).

## §7 Time and hardware estimates

### §7.1 Machines

| Machine | Facts |
|---|---|
| Mac | Apple M2, 8 GB unified memory (measured config). CPU NumPy ran the GPT-2 float forward pass at about 61 GFLOP/s (est., from 46 s for 11,264 tokens x 2 x 124M). |
| devbox | Ryzen 7 5800H, 8 cores; Linux has 12 GB RAM (devbox README); RTX 3060 Laptop, 6 GB (measured config). This design assumes 10 TFLOP/s of usable fp16 throughput (est.; the published peak depends on the laptop power limit). |

### §7.2 Model sizes

| Model | Parameters | bf16 weights | `i8` model file | Fits the 3060 in bf16? | Fits the Mac in `i8`? |
|---|---:|---:|---:|---|---|
| GPT-2 small | 124M (published) | 0.25 GB | 127 MB (measured) | Yes | Yes |
| 1B (Llama 3.2 1B) | 1.24B (published) | 2.5 GB | 1.2 GB (est.) | Yes | Yes |
| 3B (Llama 3.2 3B) | 3.21B (published) | 6.4 GB | 3.2 GB (est.) | **No**: one layer at a time | Yes, with little margin |
| 7B (Mistral 7B) | 7.24B (published) | 14.5 GB | 7.2 GB (est.) | **No**: one layer at a time | **No** (8 GB with the OS) |

A 7B model also does not fit the 12 GB RAM of devbox in bf16. The converter reads one layer at
a time from the memory-mapped files (§6.2).

### §7.3 Conversion time

All numbers are est. GPT-2 calibration and build are measured. The bases: the GPT-2
measurements, FLOPs `= 2 x parameters x tokens`, and the throughputs of §7.1. "Search" is 150
fake-quantisation passes over 11 windows, with a factor of 2 for the fake quantisation.

| Stage | Where | GPT-2 small | 1B | 3B | 7B |
|---|---|---:|---:|---:|---:|
| Import and float check | CPU or GPU | 1 min | 2 min | 4 min | 10 min |
| Calibration, 11 windows | Mac CPU | 46 s (measured) | 8 min | 20 min | 45 min |
| Calibration, 11 windows | devbox GPU, one layer at a time | 10 s | 1 min | 2 min | 5 min |
| Recipe search, 150 passes | devbox GPU | 5 min | 15 min | 40 min | 1.5 h |
| Recipe search, 150 passes | Mac CPU | 4 h | 1.5 days | not practical | not practical |
| GPTQ for all layers (optional) | devbox GPU | 1 min | 5 min | 15 min | 30 min |
| Build of the model file | CPU | 2.6 s (measured) | 30 s | 1.5 min | 4 min |
| Integer simulation of the recipe, 11 windows | devbox CPU | 1 min | 10 min | 25 min | 1 h |
| HardShell, 19 held-out windows (19,456 steps), scalar, 6 threads | CPU | 2 min (measured, Mac) | 20 min | 55 min | 2 h (devbox only) |
| QAT, 20M tokens (optional) | devbox GPU | 30 min | 2 to 4 h | 6 to 9 h | not on devbox |
| **Total without QAT** | | **about 15 min** | **about 1 h** | **2 to 3 h** | **4 to 6 h** |

The HardShell time assumes the measured GPT-2 speed (156 steps per second with 6 threads)
divided by the ratio of the weight bytes. The SIMD work (phase 4) makes it shorter.

### §7.4 What runs where

| Work | CPU or GPU | Reason |
|---|---|---|
| Import, maps, float check | Either | Small |
| Calibration | GPU (CPU is possible) | Many float passes; the result is committed, so the device does not matter |
| Recipe search | GPU | 150 passes; the CPU is too slow for 1B and more |
| GPTQ, SmoothQuant, rotations | GPU, or CPU if the result must repeat | The build must give one file |
| Build of the model file | **CPU only** | Deterministic NumPy; two machines must give one SHA-256 |
| Tables (activations, RoPE, exponent) | **CPU only** | IEEE basic operations only (`convert.numerics`) |
| QAT | GPU | Training |
| HardShell verification | CPU now, GPU after phase 4 | The prediction path |

## §8 The effort to build the generic system

Estimates in agent days (one agent, one working day with tests and documents). All est.

| Part | Work | Agent days |
|---|---|---:|
| A. Float survey | Phase 0 scripts with transformers: float bits per byte of 4 to 6 candidate models on the phase 0 corpus | 1 |
| B. Format version 2 | ADR, header schema, reader, validator, writer (Rust and Python), version 1 lowering, golden-hash test | 4 to 6 |
| C. New ops | `rms` norm, groups, `table`, `mul`, `rope`, GQA, window, soft-caps, `i16` scores; property tests; tiny models of §5.5 | 4 to 6 |
| D. Converter import | Hugging Face reader, 12 architecture maps, float graph, check against transformers, one layer at a time | 4 to 6 |
| E. Search | Generic fake quantisation and integer simulation, sensitivity, search, report | 4 to 6 |
| F. Tokenizers | Byte-fallback BPE encoder, Llama 3 regex, `start_token`, round-trip tests for each family | 2 to 3 |
| G. QAT | LoRA with fake quantisation, merge, measurement | 3 to 5 |
| H. Rotations | Offline rotation (R1, R2); later the `hadamard` op | 2 to 4 |
| **A to F (the core)** | | **19 to 28** |
| **A to H** | | **24 to 37** |

## §9 Risks

### §9.1 Activation outliers in modern LLMs

- LLM.int8 found systematic outlier features in large models. Its method keeps them in a
  16-bit product, for less than 0.1 % of the values (published). HardShell has no float
  product, so the equivalent is an `i16` site.
- Massive activations: a few activations are about 100,000 times larger than the others, and
  they act as bias terms (published, Sun et al., 2024). They occur at the first token and at
  delimiter tokens. GPT-2 shows the same effect at position 0 (measured, §5.4).
- Llama 3 loses more under low-bit quantisation than Llama 2 (published, Huang et al., 2024).
  At 8 bits the published losses are small, but HardShell also quantises every activation and
  every non-linear function.
- SmoothQuant moves the outliers into the weights offline (published: W8A8 on many families).
  AWQ protects the salient weight channels (published; weight-only, so less relevant here).
- I-LLM shows integer-only inference of Llama models with integer softmax and norms (published,
  W4A4 and W6A6 results). It uses dynamic integer scales for each token. HardShell uses static
  scales. Dynamic scales are deterministic too, but they need a new op. This is a fallback if
  static scales fail.

Mitigation: the search of §6.6 (`i16` at sensitive sites), SmoothQuant, rotations, and QAT.

### §9.2 Can rotations stay offline?

QuaRot and SpinQuant multiply the hidden states by orthogonal matrices to remove outliers
(published). The question is which rotations need work at run time.

| Rotation | Place | Offline? |
|---|---|---|
| R1 | The residual stream | Yes. Fold `Q` into the embedding, the input of each layer and the output of each layer. RMSNorm does not change. |
| R2 | Each head, between `v` and the output projection | Yes. Fold into `W_v` and `W_o`. |
| R3 | `q` and `k` after RoPE (for KV cache quantisation) | No. RoPE is between the weights and the cache. Needs `hadamard`. |
| R4 | The input of the down projection (after the GLU product) | No. The GLU product is not linear. Needs `hadamard`. |

QuaRot runs its online Hadamard transforms in FP32 (published). HardShell can do them exactly
in integers: a Walsh-Hadamard transform uses only additions and subtractions, and the sum
grows by `log2(n)` bits only. Thus the `hadamard` op keeps the prediction path integer-only.
A width that is not a power of 2 needs a Hadamard matrix of another size, as in QuaRot. An
example is 11,008 = 2^8 x 43. QuaRot also reports 8-bit Llama 2 models with no loss and no
calibration data, with round to nearest (published). Thus R1 and R2 are the first
choice for an outlier problem. R4 is the second choice.

### §9.3 Speed of large models

HardShell runs GPT-2 small at 55 steps per second on one thread (measured). `taco fold` gives
about 0.5 KB/s with 6 threads (measured). The cost of a step is mostly one read of all
weights. Thus a 1B model is about 10 times slower (est.: 0.05 KB/s, 20 s for each KB). A 7B model is
about 60 times slower (est.). Each container block is independent, so a step can
process many blocks together. Then the weights are read one time for all blocks, and the
matrix-vector product becomes a matrix product. This helps the CPU and the GPU. Without the
GPU path of phase 4, a model of 1B or more is too slow for a user (est.).

The Mac has 8 GB. A 7B `i8` model (7.2 GB) does not fit, so the cross-machine check of plan
§11.3 cannot run on the Mac. Thus 3B is the largest model that both machines can check
(est.).

### §9.4 File sizes and licences

| Model | Licence (Hugging Face card, published) | Note |
|---|---|---|
| GPT-2 | MIT | |
| GPT-J, Pythia | Apache 2.0 | |
| Mistral 7B v0.1 | Apache 2.0 | |
| Qwen2.5 1.5B, Qwen3 1.7B and 4B | Apache 2.0 | Qwen2.5 3B has the licence "other" (a Qwen licence). Read it first. |
| SmolLM2, OLMo 2 | Apache 2.0 | Open training data (OLMo 2) |
| Phi-2, Phi-3 mini | MIT | |
| Llama 2, Llama 3.2 | Llama community licences, gated | Redistribution of derivatives has conditions. |
| Gemma 1 to 3 | Gemma terms, gated | A use policy applies to derivatives. |

A user must have the model file to decode. A file of 1.2 GB to 7.2 GB (est.) is a large
download for a compressor. Plan §11.1 makes the GPT-2 file from the published weights with a
hash check. This works for a large model only if the build is deterministic and the weights
stay available at the same SHA-256. A gated model needs each user to accept the licence.

### §9.5 Other risks

- **The map is wrong.** The float check of §6.2 step 3 finds it.
- **The search overfits the calibration set.** The evaluation set stays held out. The report
  gives held-out numbers only.
- **The interpreter is slower than the fixed GPT-2 code.** The overhead is about 15 ops for each
  layer for each step. The matrix products dominate. Measure it with the GPT-2 benchmark
  (41.2 steps per second today, measured). The gate permits at most 3 % loss (est.).
- **Conflict with the SIMD and CUDA agents.** They edit `hardshell` now. A fused GPT-2 CUDA
  kernel needs rework for the graph. Kernels for each op (matrix-vector product, norm,
  attention) move to the interpreter with no change.
- **Large vocabularies.** Llama 3 has 128,256 tokens and Gemma 3 has 262,144 tokens
  (published). The output matrix is 21 % of Llama 3.2 1B (est.: 262M of 1.24B). The limit
  `vocab_size <= 2^20` of ADR 0004 still holds.

## §10 Fit with the plan

- ADR 0005 says that version 1 uses GPT-2 only, and that other models are phase 6. This design
  does not change that. The graph format is format version 2 of the model file.
- Plan §13 questions 1 and 2 (accuracy of integer softmax, LayerNorm and `i8`) get new forms
  for each family. The search of §6.6 answers them by measurement.
- CLAUDE.md §1 rules 1, 2, 3, 7, 8 and 10 apply without change. Rule 9 ("measure before you
  tune") is the reason for the automatic search: each choice in `recipe.json` has its
  measurement.

## §11 Phased plan

Each phase ends at a gate. All estimates are est. The phases follow plan §12 and do not delay
version 1.

| Phase | Work | Gate | Agent days | When |
|---|---|---|---:|---|
| M0 | Float survey (part A): float bits per byte of GPT-2, SmolLM2 360M and 1.7B, Llama 3.2 1B, Qwen3 1.7B, on the phase 0 corpus | A candidate model is at least 10 % smaller than GPT-2 on the 16 text files. If none is, stop here. | 1 | Now (devbox GPU, tools only) |
| M1 | Automatic search on GPT-2 (part E for GPT-2, in `tools/convert`, version 1 file) | The search, with no manual input, gives at most the bits of `clip` on the calibration windows | 3 to 4 | Now, in parallel with phase 3 |
| M2 | Format version 2 and the interpreter (part B), GPT-2 only | `tiny.hsm` and `gpt2.hsm` (version 1) give the golden hashes. The speed loss is at most 3 %. The Mac and devbox agree. | 4 to 6 | After the SIMD and CUDA kernels land (phase 4) |
| M3 | New ops and tiny models (part C), tokenizers (part F) | Every tiny model has the same hashes on both machines. Round trips of all plan §11.2 classes for each tokenizer kind. | 6 to 9 | After M2 |
| M4 | Converter import for the llama family (part D), first model: the M0 winner at 1B or less | Float check passes. Integer loss against float is at most +2 % on held-out text, or QAT starts. | 4 to 6 | After M3 |
| M5 | Other families (maps), QAT (part G), rotations (part H) as the measurements need | Each family: float check and loss gate | 2 to 10 | Phase 6 |

The total to the first 1B model in Taco (M0 to M4): 18 to 26 agent days (est.).

## §12 Recommendation

1. **Now: M0 and M1.** They touch only `tools/` and do not conflict with the `simd` and `cuda`
   agents. M0 answers the main question first: is a larger model worth the cost? M1 removes
   the manual diagnosis that the owner named, and it is the base of every later conversion.
2. **Now: one request to the `cuda` and `simd` agents.** Keep kernels at the level of one op
   (matrix-vector product, norm, softmax, attention for one head). Do not write one fused
   GPT-2 kernel. Then the interpreter reuses the kernels.
3. **After the GPU and logs work: M2 to M4.** The interpreter replaces the code that the speed
   work changes now. A model of 1B or more is also too slow without the GPU path (§9.3). So
   the interpreter gives value only after phase 4.
4. **Target 1B to 3B first, not 7B.** 3B is the largest model that both machines can check
   (§9.3). 7B needs more memory than the Mac has.

## §13 Open questions for the owner

1. Which objective does the search use: KL against float (this design) or bits per byte?
2. What loss target starts QAT: +1 %, +2 % or another value?
3. Is a model file of 1.2 GB to 3.2 GB acceptable for a user download?
4. Can `tools/` depend on `torch` and `transformers` (for the float check and the search)?
5. Is a gated licence (Llama, Gemma) acceptable, or only Apache 2.0 and MIT models?

## §14 Sources

Model configs and code (published, read 2026-10-18):

- Configs: `https://huggingface.co/<repo>/resolve/main/config.json` for openai-community/gpt2,
  EleutherAI/gpt-j-6b, EleutherAI/pythia-1b, unsloth/llama-2-7b, unsloth/Llama-3.2-1B,
  unsloth/Llama-3.2-3B, mistralai/Mistral-7B-v0.1, Qwen/Qwen2.5-1.5B, Qwen/Qwen3-1.7B,
  Qwen/Qwen3-4B, unsloth/gemma-2b, unsloth/gemma-2-2b, unsloth/gemma-3-1b-pt, microsoft/phi-2,
  microsoft/Phi-3-mini-4k-instruct, microsoft/phi-4, HuggingFaceTB/SmolLM2-135M and 1.7B,
  allenai/OLMo-2-0425-1B, allenai/OLMo-2-1124-7B. (The unsloth repositories are ungated
  copies of the gated Meta and Google configs.)
- Tokenizers: `tokenizer.json` of 8 of these repositories, at the same URL form.
- Licences and parameter counts: `https://huggingface.co/api/models/<repo>`.
- Modeling code:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py,
  .../qwen3/modeling_qwen3.py, .../gemma2/modeling_gemma2.py, .../olmo2/modeling_olmo2.py,
  .../phi/modeling_phi.py, .../gptj/modeling_gptj.py.
- GGUF: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

Papers (published):

- LLM.int8: Dettmers et al., 2022. https://arxiv.org/abs/2208.07339
- SmoothQuant: Xiao et al., 2022. https://arxiv.org/abs/2211.10438
- GPTQ: Frantar et al., 2022. https://arxiv.org/abs/2210.17323
- AWQ: Lin et al., 2023. https://arxiv.org/abs/2306.00978
- Massive activations: Sun et al., 2024. https://arxiv.org/abs/2402.17762
- QuaRot: Ashkboos et al., 2024. https://arxiv.org/abs/2404.00456
- SpinQuant: Liu et al., 2024. https://arxiv.org/abs/2405.16406
- SliceGPT: Ashkboos et al., 2024. https://arxiv.org/abs/2401.15024
- I-LLM: Hu et al., 2024. https://arxiv.org/abs/2405.17849
- I-BERT: Kim et al., 2021. https://arxiv.org/abs/2101.01321
- Low-bit Llama 3: Huang et al., 2024. https://arxiv.org/abs/2404.14047
- Data-free quantisation (bias correction): Nagel et al., 2019. https://arxiv.org/abs/1906.04721
- LLM-QAT: Liu et al., 2023. https://arxiv.org/abs/2305.17888
- EfficientQAT: Chen et al., 2024. https://arxiv.org/abs/2407.11062
- QA-LoRA: Xu et al., 2023. https://arxiv.org/abs/2309.14717

Taco measurements: `HANDOFF.md` §2, ADR 0004, `docs/research/2026-10-18-integer-gpt2.md`.
