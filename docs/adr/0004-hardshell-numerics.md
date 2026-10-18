# ADR 0004: HardShell numerics constants

Date: 2026-10-18. Status: proposed (hardshell agent). The lead accepts it.

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
- The converter selects the residual scale so that the largest calibration value is near
  `2^24` and a typical row has a standard deviation of `2^12` units or more.

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

## Consequences

- The converter must obey the bounds in sections 1 and 2. If the residual is too small, the
  LayerNorm error increases (measured above).
- These numbers measure each operation alone. The error of the full model is a phase 3
  measurement on GPT-2. The error on a random tiny model is in the report of the hardshell
  agent.
- A change to a constant changes the model file and the compressed bytes. It needs new
  conformance vectors.
