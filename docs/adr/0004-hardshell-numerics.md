# ADR 0004: HardShell numerics constants

Date: 2026-10-18. Status: accepted (proposed by the hardshell agent).

## Context

Plan §7.3 says that the LayerNorm constants are a phase 2 decision, recorded in an ADR with the
measurement. Plan §7.4 to §7.6 also leave the sizes of the tables and the precision `P` open.
This ADR records the values that HardShell and the converter use. The tests in
`crates/hardshell/tests/numerics.rs` measure each value against a float reference. Run
`cargo test -p hardshell --release --test numerics -- --nocapture` to see the numbers again.

All numbers below are measured on the Mac (ARM), 2026-10-18, unless a label says otherwise.

## Decision

### 1. The residual stream

- The residual stream is `i32`, one value for each channel, with one scale for the whole
  model. This is not an `i8` activation: it is not the input of a matrix product.
- HardShell clamps the residual to `[-(2^30 - 1), 2^30 - 1]` after each addition. This bound
  keeps all LayerNorm products exact (see `src/layernorm.rs`).
- The converter selects the residual scale so that a typical row has a standard deviation of
  `2^12` units or more. For GPT-2 this puts the largest calibration value at about `2^27.45`
  (measured), still inside the clamp. The first rule of this ADR also asked for a largest value
  near `2^24`. GPT-2 cannot meet both rules, so the standard-deviation rule has priority
  (changed 2026-10-18 after the convert measurements).

### 2. LayerNorm (plan §7.3)

| Constant | Value | Reason |
|---|---|---|
| `K` | 48 | The largest value that keeps `(x - mean) * inv * gamma` below `2^95` for every `i32` input. |
| `J` | 8 | Enough bits for `isqrt` at a small standard deviation, with `K - J = 40` bits for `inv`. |
| `eps` | `round(1e-5 / S_res^2 * 2^(2J))`, at least 1 | The GPT-2 epsilon in the units of `var * 2^(2J)`. |
| `gamma` | `i16`, one scale for each LayerNorm | |
| `beta` | `i32`, at the output scale | |
| Output | `i8`, clamped to `[-127, 127]` | |

Measured error, width 768, two outlier channels, against a float LayerNorm with the same
`gamma` and `beta`. The error is in units of the last bit (LSB) of the `i8` output. The best
possible error is 0.5 LSB (the output is rounded).

| Standard deviation (units) | K=48, J=8 | K=40, J=8 | K=32, J=0 | K=48, J=0 |
|---|---|---|---|---|
| 2^4 | 0.991 | 0.964 | 3.399 | 2.616 |
| 2^8 | 0.524 | 0.530 | 0.545 | 0.665 |
| 2^12 | 0.501 | 0.501 | 0.501 | 0.501 |
| 2^16 | 0.500 | 0.500 | 0.500 | 0.500 |
| 2^20 | 0.500 | 0.503 | 0.502 | 0.500 |
| 2^24 | 0.500 | 1.000 | 1.000 | 0.500 |

With K=48 and J=8, the rounding differs from the float result for 0.085% of the outputs at
`2^12` and for none at `2^16` to `2^24`. The error at `2^4` comes from step 1 of §7.3: the mean
is an integer, so it can be wrong by one unit. Thus the converter must keep the standard
deviation at `2^12` units or more (section 1).

### 3. GELU (plan §7.4)

- The input is `i16`: a table of 65,536 `i8` entries for each layer. The model file permits a
  table of 256 entries (`i8` input) too. The length of the table selects the input type.
- Measured, input range ±10, output scale 10/127: the `i16` table has a maximum error of 0.502
  LSB (RMS 0.227 LSB). The `i8` table has a maximum error of 1.008 LSB (RMS 0.323 LSB).
- The `i16` table costs 64 KiB for each layer, 768 KiB for GPT-2 small (est.).

### 4. Attention softmax (plan §7.5)

| Constant | Value |
|---|---|
| Score unit `delta` | `2^-8` (the converter puts it into the score rescale) |
| Table | `attn_exp[i] = round(2^30 * exp(-i * 2^-8))`, `i32`, 8,192 entries |
| `P` | 30 |

Measured, scores N(0, 3^2), 200 rows for each row length `n`:

| E, P, delta, length | n = 16 | n = 128 | n = 1,024 |
|---|---|---|---|
| 30, 24, 2^-4, 512 | 1.1e-4 bits | 1.8e-4 bits | 2.5e-4 bits |
| 30, 24, 2^-6, 2,048 | 8.0e-6 bits | 1.7e-5 bits | 5.6e-5 bits |
| 30, 24, 2^-8, 8,192 | 1.2e-6 bits | 6.2e-6 bits | 4.3e-5 bits |
| **30, 30, 2^-8, 8,192** | **4.8e-7 bits** | **8.0e-7 bits** | **1.5e-6 bits** |
| 30, 30, 2^-10, 32,768 | 4.0e-8 bits | 1.3e-7 bits | 7.4e-7 bits |

The numbers are the mean KL divergence of a row against the float softmax. With the chosen
values, the largest error of one probability is 6.9e-4. This error comes from the score unit.
`P = 24` loses mass in the floor division for long rows, so `P = 30` is the choice.
Bound: `e < 2^31` and `P <= 30`, so `e * 2^P < 2^61`.

### 5. Output logits (plan §7.6)

| Constant | Value |
|---|---|
| Logit unit `delta` | `2^-8` (the converter puts it into the logit rescale) |
| Table | `logits_exp[i] = round(2^40 * exp(-i * 2^-8))`, `u64`, 8,192 entries |

Measured, vocabulary 50,257, logits N(0, 3^2) and 5 peaks, with a minimum probability of
`2^-32` as in the coder: the mean KL divergence against the float softmax is 1.6e-4 bits for
each token at `delta = 2^-4`, 1.1e-5 bits at `2^-6` and 5.9e-7 bits at `2^-8`.

Bound: every entry is at most `2^40` and the vocabulary has at most `2^20` tokens, so the sum of
the weights is at most `2^60`. HardShell refuses a table that breaks this bound.

### 6. Rescale constants

- Each rescale has a multiplier `m` (`i32`) and a shift `s` (`u8`). The converter selects `s`
  so that the largest `m` of the tensor is in `[2^30, 2^31)`.
- A linear layer has one `m` for each output row and one `s` for the layer, because the weight
  scale is different for each row (plan §7.1).

### 7. Activations

- An `i8` activation is symmetric: HardShell clamps it to `[-127, 127]`.
- The converter selects each activation scale from calibration with a recipe for each site. The
  published recipe is `clip` (`tools/convert/recipe.py`). For most sites the largest calibration
  value maps to 127 (`i8`) or to 32,767 (the `i16` GELU input). For `ln_f` and the GELU outputs,
  the range comes from the typical channel, and the few outlier channels saturate at the clamp.
- The first rule ("the largest value maps to 127" at every site) failed for GPT-2: +67% bits on
  the held-out windows (measured). Outlier channels reach 295 while a typical channel used one
  or two levels. The `clip` recipe gives +0.2% against float on the held-out text files
  (measured, `docs/research/2026-10-18-integer-gpt2.md`).
- **Beta correction.** HardShell adds the LayerNorm `beta` after the rounding step, so each
  channel has a constant error. The converter removes that error through the bias of the next
  linear layer. This is a converter rule: every converter for a HardShell model must apply it.

### 8. The model file (plan §8)

Plan §8 does not give the exact header schema. HardShell defines it. The full text is in the
module documentation of `crates/hardshell/src/hsm/mod.rs` (layout and JSON schema) and
`crates/hardshell/src/gpt2/mod.rs` (the forward pass and the tensor names). The Python
converter must obey these two documents. The writer `hardshell::hsm::HsmWriter` is the
reference implementation.

The main rules:

1. The fixed part is 16 bytes: the magic, the version (1) and `header_len`. The JSON header
   follows at offset 16.
2. The vocabulary and every tensor start at a multiple of 64 bytes. Plan §8 says "all
   sections"; the fixed part and the header start at 0 and 16.
3. The layout is canonical: the vocabulary is at `align64(16 + header_len)`, each tensor is at
   `align64` of the end of the section before, in header order, and the file ends at the end of
   the last tensor. All gap bytes are zero. The header can end with spaces. Thus one model has
   exactly one file and one model hash.
4. The header has three fields: `model` (architecture, `n_layer`, `n_head`, `d_model`, `d_ff`,
   `vocab_size`, `n_ctx`, `eot_token`, and `constants` with `ln_k`, `ln_j`, `ln_eps`,
   `softmax_p`), `vocabulary` (`offset`, `length`) and `tensors` (`name`, `dtype`, `shape`,
   `offset`, `length`, and `shift` on a rescale tensor).
5. Every number is an integer. A fraction, an exponent, an unknown field, a missing field or a
   duplicate field is an error.
6. A rescale `(m, s)` is an `i32` tensor of multipliers with a `shift`. Plan §8 puts `(m, s)`
   in the JSON; HardShell puts `s` in the JSON and `m` in the data, because a linear layer has
   one `m` for each output row.
7. Plan §8 names the constants K, J and P. The file also needs `ln_eps`, `d_ff` and
   `eot_token`, so the header has them.

### 9. Context restart (plan §7.7)

When the context has `n_ctx` tokens and the caller gives one more token, the session empties
the context, runs `<|endoftext|>` at position 0, and runs the token at position 1.

## Consequences

- The converter must obey the bounds in sections 1 and 2. If the residual is too small, the
  LayerNorm error increases (measured above).
- These numbers measure each operation alone. The error of the full model is a phase 3
  measurement on GPT-2. On a random tiny model (2 layers, width 64, vocabulary 300), the mean
  KL divergence against the float model with the same `i8` weights is 0.006 to 0.009 bits for
  each token, and the most likely token agrees for 93.8% to 96.1% of the positions (measured,
  3 seeds, `crates/hardshell/tests/gpt2.rs`). A random model is not GPT-2, so these numbers do
  not predict the GPT-2 loss.
- A change to a constant changes the model file and the compressed bytes. It needs new
  conformance vectors.

## Addendum A: the BPE merges section (2026-10-18)

Status: accepted (proposed by the convert agent). Format version 1 is not frozen, so the
version stays 1.

### Context

The Rust tokenizer needs the GPT-2 BPE merge ranks (plan §10). The model hash must identify
the tokenizer (plan §8). Thus the merges go into the model file and not into a separate file.

### Decision

1. The header has an optional field `merges` with `offset` and `length`, between
   `vocabulary` and `tensors`.
2. The merges section starts at `align64` of the end of the vocabulary. The tensors follow at
   `align64` of the end of the merges section. Gap bytes are zero.
3. The section holds the merges in rank order (rank 0 first). Each merge is two `u32` values,
   little-endian: the id of the left token, then the id of the right token. Thus
   `length = 8 * count`.
4. A file with no merges omits the field. An empty section (`length` 0) is an error, so one
   model has exactly one file.
5. The reader refuses an id that is not in the vocabulary, a part with no bytes, a pair that
   occurs two times, and a pair whose joined bytes are not the bytes of exactly one token. The
   reader gives the merged token with each pair (`hardshell::hsm::MergePair`).
6. `HsmFile::merges()` and `Vocabulary::merges()` give the merges, read-only, in rank order.

### Consequences

- **The model hash covers the merges**, because the hash is the SHA-256 of the full file. A
  different merge list gives a different model hash.
- **The decoder does not need the merges.** The decoder changes token ids to bytes with the
  vocabulary only (plan §10). Only the encoder reads the merges.
- A file without the field is the same file as before this addendum. Thus the files that the
  tests made before this change still load.
- For GPT-2, merge rank `i` makes token `256 + i` (measured: all 50,000 merges). The file
  stores 400,000 bytes of merges.
