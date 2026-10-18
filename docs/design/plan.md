# Taco: design plan

Status: draft 1, 2026-10-18. Owner: Rohit. This plan obeys `docs/STYLE.md`.

## §1 Goal

Taco is a lossless compressor that uses a language model. The goal has three parts:

1. Make files smaller than general compressors (gzip, bzip2, xz, zstd) make them.
2. Restore every file exactly, byte for byte.
3. Decompress on any machine. A file that a Mac compresses decompresses on a Linux PC with an
   NVIDIA GPU, and the reverse.

Part 3 is the new work. Language-model compressors exist, but most of them decode only on the
same hardware and software that encoded (see `docs/research/2026-10-18-verification.md`).
Taco removes this limit. HardShell, its engine, uses only integer arithmetic.

## §2 Terms

| Term | Meaning |
|---|---|
| Token | One unit of the model vocabulary. GPT-2 has 50,257 tokens. |
| Frequency table | One integer for each token. The coder uses it to divide its range. |
| Prediction path | All code from the model weights to the frequency table. |
| Model file | The published file that holds the integer weights, the tables and the vocabulary. |
| Model hash | The SHA-256 of the model file. |
| Container | The `.taco` file: header, blocks and checksum. |
| Block | A part of the input that Taco codes independently of other parts. |
| Conformance vector | An input with the exact outputs that a correct implementation must give. |
| Lane | A private build directory on devbox for one agent or task. |

## §3 Rules

1. No floating-point arithmetic in the prediction path.
2. Every integer operation has one defined result on every machine (§7.2).
3. The model file is the model. The decoder never converts or quantizes a model.
4. Every container stores a checksum of the original data. Decompression verifies it.
5. Every token has a frequency of 1 or more. Any byte string can be tokenized.
6. A block that does not become smaller is stored raw.
7. Conformance vectors define correct behaviour.
8. The model file and the container have format versions.
9. Measure before you tune.
10. One engine. A fast path gives the same bits as the reference path.

`CLAUDE.md` §1 gives the full text of each rule.

## §4 Data flow

```
fold (compress):
  input bytes -> blocks -> tokenizer -> token ids
  for each token: HardShell(context) -> integer logits -> frequency table -> range coder
  -> container (header, model hash, checksum, blocks)

unfold (decompress):
  container -> check model hash -> for each block:
  HardShell(context) -> frequency table -> range decoder -> token id -> bytes -> append
  -> verify checksum -> output
```

The decoder never tokenizes. It changes token ids to bytes with the vocabulary only. Thus the
encoder can change its tokenization later, and old files stay readable.

## §5 Range coder (`taco-coder`)

The coder is a range coder with byte output and carry propagation, in the style of the LZMA
range coder. Its state is wider, so that a frequency table can have a large total.

Requirements:

1. The total of a frequency table is at most 2^32. With 50,257 tokens and a minimum frequency
   of 1, the minimum costs less than 0.002% of the probability mass (est.).
2. After renormalisation, the range is at least 2^56. Thus `range / total` is at least 2^24,
   and the loss from integer division is less than 2^-24 of a symbol (est.).
3. `low` holds 64 bits and a carry. A `u128` is permitted.
4. The encoder and the decoder use the same division and the same rule for the last symbol.
5. `flush` writes enough bytes that the decoder decodes all symbols. The decoder reads zero
   bytes after the end of the input.
6. The coder also codes binary decisions with a 12-bit probability, for small models and tests.
7. Property tests: any frequency table that obeys the limits, and any symbol sequence, come back
   exactly.

The coder records its exact algorithm in its crate documentation and in an ADR. A change to the
algorithm changes the compressed bytes, so it needs a new container version (§9).

### §5.1 Frequency tables (`taco-coder::freq`)

The conversion from model weights to frequencies is part of the coder crate. The input is one
non-negative `u64` weight for each token. HardShell makes the weights from the logits with an
exponent table (§7.6). The output is a table with total `T = 2^FREQ_BITS`.

Algorithm:

1. Let `V` be the number of tokens and `W` the sum of the weights, calculated in `u128` (a
   `u64` sum can overflow). If `W` is 0, all weights are 1.
2. `f[i] = 1 + floor(w[i] * (T - V) / W)`. Calculate the product in `u128`.
3. The sum of `f` is at most `T`. Add the remainder `T - sum(f)` to the token with the largest
   weight. If two tokens have the same weight, use the lower token id.

The steps use only integer operations, so every machine makes the same table.

## §6 Model choice

Phase 2 uses GPT-2 small (124M parameters, 12 layers, width 768, context 1,024). Reasons:

- The weights are open (MIT licence) and small (548 MB as float32, about 124 MB as int8).
- Its tokenizer is byte-level BPE. Every byte has a token, so any input can be tokenized.
- It uses learned position embeddings, LayerNorm and GELU. It has no rotary embeddings.
- Published integer-only methods (I-BERT) cover all its operations.

GPT-2 small is the only model of version 1 (ADR 0005). Rohit decided this on 2026-10-18.
Nanbeige4.2-3B (on devbox) is not in the plan. Other models (a small model for one data type,
a larger model) are phase 6 work, after version 1 is released.

## §7 HardShell numerics

### §7.1 Number formats

- Weights: `i8`, symmetric, with one scale for each output row.
- Residual stream: `i32`, clamped to ±(2^30 − 1) (ADR 0004).
- Activations: `i8` (symmetric, −127 to 127) with one static scale for each tensor. A tool calibrates the scales on a
  calibration corpus before publication. If phase 3 shows too much loss, activations change to
  `i16` with `i64` accumulation. The model file states the type of each tensor.
- Accumulators: `i32` for `i8 x i8` dot products. Bound: 127 x 128 x 3,072 < 2^26, so an
  `i32` cannot overflow. Use `i64` for other products.
- Scales are not floats in the model file. Each scale is an integer multiplier `m` (`i32`) and a
  shift `s` (`u8`). The tool calculates them one time.

### §7.2 Rounding and division rules

These rules are the only permitted rules. Each one has one function in `hardshell::int`.

| Operation | Rule |
|---|---|
| Rescale | `rescale(x: i64, m: i32, s: u8) = (x * m + 2^(s-1)) >> s`, with an arithmetic shift. This rounds half up. The product is in `i128` if a bound does not show that `i64` is sufficient. |
| Clamp | `clamp(x, lo, hi)`. This is the only saturating operation. |
| Division | `floor_div(a, b)` for `b > 0` is `a.div_euclid(b)`. Never use `/` on signed values. |
| Square root | `isqrt(x) = floor(sqrt(x))`, calculated with integers only. |
| Table look-up | The index is clamped to the table range before the look-up. |

### §7.3 LayerNorm

For a row `x` of width `n`:

1. `mean = floor_div(sum(x), n)`.
2. `var = floor_div(sum((x - mean)^2), n)`.
3. `inv = floor_div(2^K, isqrt(var * 2^(2J) + eps_q))`, with `K` and `J` stated in the model file.
4. `y = rescale((x - mean) * inv * gamma_q, m, s) + beta_q`.

The exact constants are a phase 2 decision. Record them in an ADR with the measurement.

### §7.4 GELU

The input is an `i8` (or `i16`) value. GELU is a table with one entry for each input value: 256
entries for `i8`, 65,536 for `i16`. The tool calculates the table with floats one time. The
table is data in the model file.

### §7.5 Attention softmax

1. Scores are `i32` dot products of `q` and `k`, then rescaled.
2. Subtract the maximum of the row.
3. Look up `exp` in a table indexed by the difference. Clamp the index at the lower end.
4. `p[j] = floor_div(e[j] * 2^P, sum(e))`, with `P` stated in the model file.
5. The output is `rescale(sum(p[j] * v[j]), m, s)`.

### §7.6 Output logits

The final logits are integers. HardShell changes them to weights with the same kind of exponent
table as §7.5, then gives the weights to `taco-coder::freq`.

### §7.7 Context

A block starts with an empty context and the `<|endoftext|>` token. When the context is full
(1,024 tokens for GPT-2), the context starts again: `<|endoftext|>` at position 0, then the
next token at position 1. A sliding window is a phase 3 measurement.

## §8 Model file (`.hsm`, version 1)

All integers are little-endian. The header starts at byte 16. The vocabulary and each tensor
start at a multiple of 64 bytes. ADR 0004 gives the exact layout and the JSON schema.

```
magic        8 bytes   "HSHLMDL\0"
version      u32       1
header_len   u32
header       JSON, UTF-8, header_len bytes:
             architecture, layer count, head count, width, vocabulary size, context length,
             constants (K, J, P), and a list of tensors and tables:
             name, type, shape, offset, length, rescale (m, s) where it applies
vocabulary   for each token id: u16 length, then the bytes of the token
data         tensors and tables at the stated offsets
```

The model hash is the SHA-256 of the full file. The model file contains the vocabulary, so the
model hash also identifies the tokenizer. The JSON header is parsed with no floats: a number
with a fraction or an exponent is an error.

## §9 Container (`.taco`, version 1)

```
magic          4 bytes   "TACO"
version        u16       1
flags          u16       0
model hash     32 bytes  SHA-256 of the model file (all zero for the raw-only mode)
original len   u64
checksum       32 bytes  SHA-256 of the original data
block size     u32       input bytes in each block (the last block can be shorter)
block count    u32
block table    for each block: kind (u8: 0 = raw, 1 = coded), stored length (u32)
blocks         the stored bytes of each block
```

- Each block is independent. Thus a damaged block does not damage other blocks, and blocks can
  be coded in parallel.
- A block is stored raw if the coded form is not smaller.
- The worst-case growth is the header (88 bytes) plus 5 bytes for each block.
- The decoder verifies the checksum. A failed check is an error.

## §10 Tokenizer

- Encoder: GPT-2 byte-level BPE. A byte sequence that is not valid UTF-8 uses the single-byte
  tokens. The output token sequence always decodes to the exact input bytes. A test proves this
  for random bytes.
- Decoder: it changes each token id to its bytes with the vocabulary in the model file. It does
  no other work.
- The tokenization must be deterministic, so that the compressed bytes are the same on every
  machine. It does not need to be the canonical GPT-2 tokenization.

## §11 Conformance

### §11.1 Vectors

`conformance/` holds:

- A tiny test model (`tiny.hsm`, about 100 KB) that a tool makes from a fixed seed. It is
  committed, so the tests do not need a download.
- For each input: the SHA-256 of the logits at each step, of the frequency tables, and of the
  container.
- For GPT-2: the model hash and the container hashes. The model file is not committed. A tool
  makes it from the published weights, and the test checks its hash.

### §11.2 Input classes

Empty input, 1 byte, English prose, other languages (Hindi, Chinese, Arabic), source code, JSON,
log lines, repeated text (`dasdasdasdada`), random letters, random bytes, all 256 byte values, a
large mixed file, and fuzz inputs.

### §11.3 Cross-machine check

`scripts/xcheck` runs the conformance suite on the Mac (ARM) and on devbox (x86), and compares
all hashes. A GPU implementation joins the same check.

## §12 Phases

Each phase ends at a gate. Estimates are in working sessions with agents, not in calendar
weeks. All estimates are `est.`.

| Phase | Work | Gate | Estimate |
|---|---|---|---|
| 0 | Float baseline: GPT-2 in NumPy, real compression sizes against gzip, bzip2, xz, zstd | Bits per byte on the quality corpus are recorded | 1 day |
| 1 | Float drift: the same model on the Mac and on devbox. Count different predictions, and decode across machines | The failure is measured and recorded | 1 day |
| 2 | Reference implementation: coder, container, HardShell on CPU, tokenizer, `taco fold` and `taco unfold` | Round trip of all §11.2 classes. Same hashes on the Mac and on devbox | 2 to 3 weeks |
| 3 | Quality: measure integer against float loss. Calibration, `i16` activations, short fine-tune if necessary | Integer loss is less than a target that phase 0 sets | 1 to 2 weeks |
| 4 | Speed: SIMD on CPU, CUDA kernels on devbox, blocks in parallel | A GPU path with the same bits as the CPU path. A speed target that phase 2 sets | 1 to 3 months |
| 5 | Product: format version 1 frozen, benchmark page, release | A public benchmark | 2 to 4 weeks |
| 6 | After version 1: other models (logs, code, larger models) and other data types (images, audio) | Each one has its own plan | open |

## §13 Open questions

1. Can integer softmax and LayerNorm stay accurate enough? Phase 3 measures it.
2. Is `i8` activation sufficient for GPT-2, or are there outlier channels? Phase 3 measures it.
3. What block size gives the best balance of ratio and parallel speed? Phase 2 measures it.
4. What speed is useful for a first user (log storage, for example)? Phase 4 decides it.
