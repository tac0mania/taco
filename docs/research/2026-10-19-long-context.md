# Research: long context with a restart overlap (2026-10-19)

This note measures the context rule of ADR 0015 (proposed): when the GPT-2 context is full,
the context starts again with `<|endoftext|>` and the last `K` tokens of the block. It gives
bits per byte for `K` from 0 to 768 and block sizes from 16 KiB to 1 MiB, the compute cost,
and the evidence that every path gives the same bits. All numbers are measured, unless a
number has a different label.

## 1. Results in short

- **The overlap makes every input smaller.** With `K = 512` and blocks of 64 KiB: English
  text 1.8 % to 2.0 % smaller, cp.html 6.0 %, Chinese 4.0 %, the Apache log 20.6 % (against
  `K = 0` with the same blocks).
- **For English text the gain stops at `K` = 384 to 512.** `K = 768` is 0.2 % to 0.4 % worse
  than `K = 512` on alice29 and lcet10. The float GPT-2 also stops there (section 4.3), but
  the integer engine loses a little more at late positions. HTML, Chinese and the log become
  smaller up to `K = 768`.
- **A larger block alone gives almost nothing.** With `K = 0`, blocks of 1 MiB are only 0.2 %
  to 1.3 % smaller than blocks of 16 KiB, because the context restarts every 1,024 tokens. With
  `K = 512`, blocks of 64 KiB are 0.4 % (text) to 2.9 % (log) smaller than blocks of 16 KiB.
  Blocks of 256 KiB and 1 MiB add 0.0 % to 1.3 % more.
- **The cost is exact and known.** A restart runs `K + 1` rows for `n_ctx - 1 - K` new
  tokens: 34 % more rows with `K = 256`, 100 % with `K = 512`, 302 % with `K = 768` (long
  blocks). The time grows less than the rows. With `K = 512`: the CPU step path (the decoder)
  takes 24 % more time (NOTE: one thread, `cp.html`, Mac. ADR 0012 gives +39 % for the CPU
  unfold of a whole log at 16 KiB blocks with 6 threads on devbox; the setups differ), the CPU prefill path (the encoder) 41 % more, the GPU window path 83 %
  to 100 % more.
- **Every path gives the same bits.** The oracle, the CPU step, prefill and batch paths (NEON
  on the Mac, AVX2 on devbox) and the CUDA decode and prefill paths (RTX 3060) give the same
  weights for real GPT-2 with `K = 512` (SHA-256 `09b99a92…d452`). The golden hashes for
  `K = 0` did not change.
- **Recommendation:** `K = 512` and blocks of 64 KiB as the default (section 7).

## 2. The rule and why it is exact

GPT-2 uses absolute positions. Thus a KV cache cannot move to the left, and a true sliding
window needs a new prefill of 1,023 rows for each token (est.). The rule of ADR 0015 is a
sliding window with a step of `n_ctx - 1 - K` tokens:

1. A block starts with `<|endoftext|>` at position 0.
2. When the context holds 1,024 tokens and a new token comes, the session empties the context.
   It runs `<|endoftext|>` at position 0 and the last `K` tokens at the positions 1 to `K`.
   These rows give no weights.
3. The new token runs at position `K + 1`.

The rule depends only on the tokens of the block and on `K`. The restart rows are normal
forward rows. Thus each path computes them with the same integer rules (plan §3 rule 10).
`K = 0` is the old rule of ADR 0004 section 9.

One block with overlap `K` is equal to a list of independent windows (the test
`the_reference_obeys_the_definition` proves this on three model shapes). The measurement uses
this: the GPU runs each window as an independent job in its own slot. The CUDA restart path
(`--native`) gave the same frequency digest as the windows on cp.html for `K = 0` and
`K = 512`.

## 3. Setup

| Item | Value |
|---|---|
| Model | `gpt2.hsm`, SHA-256 `aaa03eb9…a5c2` (format version 1) |
| GPU | devbox, NVIDIA RTX 3060 Laptop, 6 GB, driver 610.74. The power limit is about 45 W: the GPU log shows 412 to 1,950 MHz. Another agent used the GPU at the same time for part of the run. |
| Tool | `crates/hardshell-cuda/examples/eval_overlap.rs` (32 slots, 16 inputs of each slot in each call). It codes each token with the table of plan §5.1 (`FREQ_BITS = 32`, the GPU coder op) and adds `32 - log2(f)`. |
| CPU check | `crates/hardshell/examples/eval_bits.rs --digest --overlap K` on the Mac (M2, NEON) |
| Windows | `tools/longctx/make_windows.py`: each input is cut into blocks of B bytes, and each block is tokenized alone (ADR 0007 §2.1). The files are in `tools/data/longctx/windows/` (ignored by git). |
| Raw results | `tools/data/longctx/overlap.tsv` (pass 1), `overlap2.tsv` (pass 2, with the bits by position), `float_alice29.log`, `gpu.log` (ignored by git; also on devbox in `~/scratch/ctx/`) |

The values are the table size of the tokens, without the container and the coder bytes. The
coder adds about 0.02 % (plan §5.1 measurement). The `logs` profile (match model and mixer,
ADR 0010) is not in this measurement.

| Input | Bytes | Tokens | xz -9e | zstd -19 |
|---|---:|---:|---:|---:|
| alice29.txt (whole) | 152,089 | 47,022 | 2.553 | 2.589 |
| lcet10.txt (whole) | 426,754 | 106,844 | 2.240 | 2.275 |
| cp.html (whole) | 24,603 | 10,666 | 2.488 | 2.509 |
| apache_access.log, first 1 MiB | 1,048,576 | 480,368 | 0.452 | 0.482 |
| pg24264.txt (Chinese), first 512 KiB | 524,288 | 400,215 | 3.042 | 3.090 |

The xz and zstd values of the first three files come from the benchmark note of 2026-10-19
(xz 5.4.5, zstd 1.5.5). The last two are from the Mac (xz 5.8.3, zstd 1.5.7). All are bits per
byte.

## 4. Bits per byte

### 4.1 All block sizes, `K` in {0, 256, 512, 768}

The value in parentheses is the change against `K = 0` with the same block size. A file that
is shorter than a block size has one block; the table gives its row only once.

| Input | Block | K = 0 | K = 256 | K = 512 | K = 768 |
|---|---|---:|---:|---:|---:|
| cp.html | 16 KiB | 1.514 | 1.450 (-4.2 %) | 1.429 (-5.6 %) | 1.420 (-6.2 %) |
| cp.html | 64 KiB | 1.505 | 1.426 (-5.3 %) | 1.415 (-6.0 %) | 1.400 (-7.0 %) |
| alice29 | 16 KiB | 2.039 | 2.007 (-1.6 %) | 2.003 (-1.8 %) | 2.009 (-1.5 %) |
| alice29 | 64 KiB | 2.037 | 1.996 (-2.0 %) | 1.995 (-2.0 %) | 2.000 (-1.8 %) |
| alice29 | 256 KiB | 2.032 | 1.995 (-1.8 %) | 1.992 (-2.0 %) | 1.999 (-1.6 %) |
| lcet10 | 16 KiB | 1.712 | 1.689 (-1.3 %) | 1.682 (-1.7 %) | 1.687 (-1.5 %) |
| lcet10 | 64 KiB | 1.709 | 1.682 (-1.6 %) | 1.677 (-1.8 %) | 1.680 (-1.7 %) |
| lcet10 | 256 KiB | 1.711 | 1.681 (-1.7 %) | 1.675 (-2.1 %) | 1.679 (-1.9 %) |
| lcet10 | 1 MiB | 1.709 | 1.683 (-1.5 %) | 1.675 (-2.0 %) | 1.679 (-1.8 %) |
| Chinese 512 KiB | 16 KiB | 3.695 | 3.578 (-3.2 %) | 3.550 (-3.9 %) | 3.535 (-4.3 %) |
| Chinese 512 KiB | 64 KiB | 3.688 | 3.567 (-3.3 %) | 3.539 (-4.0 %) | 3.523 (-4.5 %) |
| Chinese 512 KiB | 256 KiB | 3.688 | 3.567 (-3.3 %) | 3.536 (-4.1 %) | 3.519 (-4.6 %) |
| Chinese 512 KiB | 1 MiB | 3.687 | 3.567 (-3.3 %) | 3.538 (-4.0 %) | 3.519 (-4.6 %) |
| Apache log 1 MiB | 16 KiB | 0.946 | 0.783 (-17.2 %) | 0.764 (-19.2 %) | 0.747 (-21.0 %) |
| Apache log 1 MiB | 64 KiB | 0.934 | 0.763 (-18.4 %) | 0.742 (-20.6 %) | 0.723 (-22.6 %) |
| Apache log 1 MiB | 256 KiB | 0.934 | 0.759 (-18.7 %) | 0.736 (-21.2 %) | 0.718 (-23.1 %) |
| Apache log 1 MiB | 1 MiB | 0.934 | 0.756 (-19.1 %) | 0.732 (-21.6 %) | 0.717 (-23.2 %) |

Totals (all bytes of the five inputs together; text = alice29, lcet10 and cp.html). A file that
is shorter than the block size counts with its value for the whole file:

All inputs:

| Block | K = 0 | K = 256 | K = 512 | K = 768 |
|---|---:|---:|---:|---:|
| 16 KiB | 1.8413 | 1.7273 | 1.7093 | 1.6987 |
| 64 KiB | 1.8331 | 1.7122 | 1.6943 | 1.6822 |
| 256 KiB | 1.8329 | 1.7102 | 1.6904 | 1.6786 |
| 1 MiB | 1.8323 | 1.7088 | 1.6889 | 1.6781 |

Text:

| Block | K = 0 | K = 256 | K = 512 | K = 768 |
|---|---:|---:|---:|---:|
| 16 KiB | 1.7865 | 1.7596 | 1.7529 | 1.7570 |
| 64 KiB | 1.7830 | 1.7507 | 1.7465 | 1.7495 |
| 256 KiB | 1.7833 | 1.7500 | 1.7444 | 1.7481 |
| 1 MiB | 1.7824 | 1.7511 | 1.7442 | 1.7480 |

### 4.2 More values of `K` (one block size)

alice29 and cp.html are one block (the whole file). lcet10, the Chinese text and the log use
blocks of 64 KiB.

| Input | K = 0 | K = 128 | K = 384 | K = 512 | K = 640 | K = 768 |
|---|---:|---:|---:|---:|---:|---:|
| alice29 | 2.032 | 2.002 (-1.5 %) | 1.991 (-2.0 %) | 1.992 (-2.0 %) | 1.994 (-1.9 %) | 1.999 (-1.6 %) |
| lcet10 | 1.709 | 1.685 (-1.4 %) | 1.677 (-1.8 %) | 1.677 (-1.8 %) | 1.680 (-1.7 %) | 1.680 (-1.7 %) |
| cp.html | 1.505 | 1.450 (-3.7 %) | 1.420 (-5.7 %) | 1.415 (-6.0 %) | 1.406 (-6.6 %) | 1.400 (-7.0 %) |
| Chinese 512 KiB | 3.688 | 3.596 (-2.5 %) | 3.553 (-3.7 %) | 3.539 (-4.0 %) | 3.530 (-4.3 %) | 3.523 (-4.5 %) |
| Apache log 1 MiB | 0.934 | 0.790 (-15.4 %) | 0.751 (-19.6 %) | 0.742 (-20.6 %) | 0.733 (-21.5 %) | 0.723 (-22.6 %) |
| Total, text | 1.7819 | 1.7554 | 1.7457 | 1.7457 | 1.7476 | 1.7491 |
| Total, all | 1.8328 | 1.7338 | 1.7020 | 1.6941 | 1.6882 | 1.6821 |

### 4.3 Float GPT-2 against HardShell

The question: is the loss of `K = 768` on English text a property of GPT-2 or of the integer
engine? `tools/longctx/float_overlap.py` runs the float GPT-2 of `tools/baseline` (NumPy,
float32) on the same windows of alice29 (the whole file, one block). "Table" is the size with
a table of plan §5.1, `32 - log2(1 + p * (2^32 - V))`. The ideal float value is larger,
because the float model gives a few tokens almost no probability, and the table floor caps
their cost.

| K | Float, ideal | Float, table | HardShell, table | HardShell against float (table) |
|---:|---:|---:|---:|---:|
| 0 | 2.0804 | 1.9978 | 2.0321 | +1.7 % |
| 384 | 2.0194 | 1.9542 (−2.2 %) | 1.9905 (−2.0 %) | +1.9 % |
| 512 | 2.0183 | 1.9542 (−2.2 %) | 1.9919 (−2.0 %) | +1.9 % |
| 768 | 2.0169 | 1.9560 (−2.1 %) | 1.9988 (−1.6 %) | +2.2 % |

The float model also stops at `K` = 384 to 512 on this text. Thus the stop is mostly a
property of GPT-2. But the integer engine loses a little more at late positions: its distance
to the float model grows from 1.7 % (`K = 0`) to 2.2 % (`K = 768`). With `K = 768` every coded
token is at a position of 769 or more. This is information for the numerics work (ADR 0004
addendum B, the calib agent): the error of the integer engine grows with the length of the
context (measured on one file only).

## 5. Compute cost

### 5.1 Rows

A block of `n` tokens with `n > 1,024` has `ceil((n - 1,024) / (1,023 - K))` restarts of `K + 1`
rows each. For a long block the extra rows are `(K + 1) / (1,023 - K)` of the tokens:

| K | 0 | 128 | 256 | 384 | 512 | 640 | 768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Extra rows (long block, exact) | 0.1 % | 14.4 % | 33.5 % | 60.3 % | 100.4 % | 167.4 % | 301.6 % |
| Extra rows, lcet10 in blocks of 64 KiB (measured count) | 0.1 % | 13.8 % | 32 % | 57.7 % | 95.1 % | 158.4 % | 285.0 % |

A block of 16 KiB has fewer restarts for each token, so its extra rows are 5 % to 20 % fewer
(measured, section 4.1 inputs).

### 5.2 Time

| Path | Input | K = 0 | K = 512 | Change |
|---|---|---:|---:|---:|
| CPU step path (`Session`, chunk 1: the decoder), Mac M2, 1 thread | cp.html, 10,666 tokens | 78.2 s | 97.3 s | +24 % |
| CPU prefill path (chunk 32: the encoder), Mac M2, 1 thread | cp.html | 49.5 s | 69.7 s | +41 % |
| GPU windows (prefill), RTX 3060 | lcet10, 64 KiB blocks | 5.8 s | 11.5 s | +98 % |
| GPU windows (prefill), RTX 3060 | Chinese, 64 KiB blocks | 22.6 s | 45.0 s | +99 % |

The CPU paths do not compute the logits of a restart row (about 31 % of the multiply-adds of a
row, est.). The CPU runs the restart rows in groups of at least 32 rows, so it reads each
weight once for each group. Thus the CPU time grows less than the rows. The GPU computes the
logits of every row, so its time grows with the rows. The GPU gave about 18,000 rows per
second for all `K` (measured, for example 400,215 tokens in 22.4 s).

NOTE: the Mac ran other work (load average 1.9 to 4.5). The GPU was power limited and shared
for part of the run. The row counts are exact; the times are measured once.

## 6. Bit-exactness evidence

| Check | Where | Result |
|---|---|---|
| The rule equals the definition (fresh sessions, window by window), 3 model shapes, 6 values of `K` up to `n_ctx - 2` | Mac, devbox | pass |
| Step, prefill (10 chunk sizes) and batch paths equal the oracle, every kernel selection, 3 or more restarts | Mac (NEON, portable), devbox (AVX2, portable) | pass |
| Golden overlap vector (tiny model, 300 tokens, `K = 32`) `5a799282…4a83` | Mac, devbox, RTX 3060 (step and prefill) | equal |
| `K = 0` equals the old rule; all golden hashes of `hardshell` and `hardshell-cuda` unchanged | Mac, devbox, RTX 3060 | pass |
| CUDA batch against CPU sessions: 4 slots, `K` 0 to 62, passes of 16, 37 and 5 rows, restarts that cross passes, attention kernels of head size 12 and 64 | RTX 3060 | pass |
| CUDA coder intervals with `K = 40` equal `taco_coder::FreqTable` | RTX 3060 | pass |
| Real GPT-2, 2,100 tokens, `K = 512`: oracle, step, prefill and batch | Mac NEON, devbox AVX2 | digest `09b99a923377b260…d452` on both |
| Real GPT-2, the same tokens: CUDA decode and prefill against the CPU | RTX 3060 | equal |
| Real GPT-2, cp.html, frequency digest of every coded token, `K = 0` and `K = 512` | Mac NEON (step and prefill) and RTX 3060 (windows and restart path) | `c34e4969…` and `1458a3bb…` on all |
| Overlap on a model of hsm format version 2 (i16 GELU and ln_f, ADR 0004 addendum B) | Mac | pass |

## 7. Recommendation

1. **Overlap `K = 512`.** It gives the largest gain on English text (with `K = 384`), and 0.4 %
   (Chinese) to 1.2 % (log) more than `K = 384`. It costs 100 % more rows, but only
   24 % more time in the CPU decoder (measured). `K = 768` costs 4 times the rows and is
   worse on English text.
2. **Block size 64 KiB.** With `K = 512` it is 0.4 % (text) to 2.9 % (log) smaller than 16
   KiB. Larger blocks add at most 1.3 % (the log) and 0.1 % on text. Larger blocks also reduce the parallel work of the
   decoder: the CPU decoder runs up to 8 blocks in lockstep on each thread, and a GPU decoder
   needs many blocks for many slots. A 1 MiB file has 16 blocks of 64 KiB.
3. If decode speed is more important than size, use `K = 384` (58 % more rows): it loses 0 %
   on English text and 0.4 % to 1.2 % on the Chinese text and the log against `K = 512`.

These values change the compressed bytes. They need the container field of ADR 0015 section 4
and new conformance vectors (`crates/hardshell/CONTEXT-INTEGRATION.md`).

## 8. Open questions

- The `logs` profile mixes the match model with GPT-2. The match model already predicts many
  long repeats, so the gain of the overlap on logs with the mixer is not known. Measure it
  after the integration (`taco fold --profile logs` with `K = 512`).
- The encoder knows all tokens, so it can run the windows of a block in parallel (section 2).
  The decoder cannot, because it needs the tokens of the window before. On a GPU, the decoder
  needs many blocks in parallel.
- A new model with a longer context or relative positions (anymodel survey) makes the restart
  less frequent. The rule stays the same.
