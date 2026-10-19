# ADR 0015: the context restart keeps the last K tokens (context overlap)

Date: 2026-10-19. Status: proposed (by the context agent). Plan: §7.7, §9, §11. Changes: ADR
0004 section 9 (the restart rule), ADR 0007 section 3.3. Research note:
`docs/research/2026-10-19-long-context.md`. Integration: `crates/hardshell/CONTEXT-INTEGRATION.md`.

## Context

GPT-2 has a context of 1,024 tokens. ADR 0004 section 9 says: when the context is full, the
session empties it, runs `<|endoftext|>` at position 0, and runs the next token at position 1.
Thus the first tokens after a restart have almost no context. Their cost is high: on whole
files, Taco does not become better with the length of the file (benchmark note of 2026-10-19,
section 7.2). The plan (§7.7) names a sliding window as a phase 3 measurement.

GPT-2 uses absolute positions. A token at position `p` gives other keys and values than the
same token at position `p - 1`. Thus a KV cache cannot move to the left. A true sliding window
needs a full new prefill for each token (est. 1,023 rows for each token). A restart that keeps
the last `K` tokens gives most of the gain for a part of that cost.

## Decision

### 1. The rule

A block has an overlap `K` (an integer, `0 <= K <= n_ctx - 2`). A block starts as before:
`<|endoftext|>` at position 0. When the context holds `n_ctx` tokens and the caller gives one
more token `t`:

1. The session takes the last `K` tokens of the block before `t`. These are the tokens at the
   positions `n_ctx - K` to `n_ctx - 1` of the full context.
2. The session empties the context. It runs `<|endoftext|>` at position 0 and the `K` kept
   tokens at the positions 1 to `K`. These rows give no weights.
3. The session runs `t` at position `K + 1`. Its weights are the result of the step.

`K = 0` is the rule of ADR 0004 section 9. Every container before this ADR uses `K = 0`.

### 2. Equivalence (the definition that the tests use)

With overlap `K`, the weights after a restart are the weights of a fresh session that ran
`<|endoftext|>`, the `K` kept tokens, then `t`. Thus a block with overlap `K` is a list of
independent windows: window 0 holds the first `n_ctx - 1` tokens; each later window holds
`<|endoftext|>`, the `K` tokens before it, and `n_ctx - 1 - K` new tokens. A GPU can run the
windows of one block in parallel slots, and the encoder (which knows all tokens) can too.

### 3. Implementation (HardShell)

1. `hardshell::gpt2::Overlap` (a `u16` newtype) holds `K`. `Model::check_overlap` refuses
   `K > n_ctx - 2` with `Error::InvalidOverlap`.
2. Every path has the rule: `ReferenceSession::with_overlap` (the oracle, ADR 0013),
   `SessionConfig::with_overlap` (step and prefill), `Batch::with_overlap` (lanes), and in
   `hardshell-cuda` `CudaSession::with_overlap` and `CudaBatch::set_overlap`.
3. The restart rows are normal forward rows. The CPU paths do not compute the logits of a
   restart row, and they run the restart rows in groups of at least 32 rows (also a session
   with a chunk of 1). The CUDA path computes the logits and does not use them. These are
   speed choices only; the weights are the same.
4. The default is `Overlap::NONE`. Thus every caller that does not set an overlap gives the
   same bits as before, and the golden hashes do not change.

### 4. The container

The container stores `K` for each container (not for each block). The decoder uses the stored
`K`. A decoder that does not know the field refuses the container. The integration note gives
the proposed field: a coding part bit of the version 2 header with `K` as a `u16` in the coding
parameters. The lead decides the field and the version (plan §3 rule 8).

### 5. The default values

The research note (section 7) recommends the defaults of the encoder:

1. `K = 512` (half of the GPT-2 context). It gives the largest gain on English text (with
   `K = 384`) and 0.4 % to 1.2 % more than `K = 384` on Chinese text and on a log. It costs
   100 % more rows; the CPU decoder takes 24 % more time (measured).
2. Block size 64 KiB (from 16 KiB, ADR 0007 §6). With `K = 512` it is 0.4 % (text) to 2.9 %
   (log) smaller than 16 KiB. Larger blocks add at most 1.3 % and reduce the parallel work of
   the decoder.

The decoder does not depend on these defaults: it reads `K` and the block size from the
container. The lead decides the defaults (they change the compressed bytes and the
conformance vectors).

### 6. Conformance vector

The golden tiny model of `crates/hardshell/tests/gpt2.rs`, 300 tokens, `K = 32` (half of the
context of 64 tokens, 8 restarts): SHA-256 of all weights
`5a79928200ad35bb7e7a1be2a0a35e05dcecc001a89cf3cf0765dfc7e5304a83` (measured: Mac NEON,
devbox AVX2, RTX 3060).

## Measurements

Real GPT-2 (`gpt2.hsm` `aaa03eb9…a5c2`), table bits per byte (plan §5.1), RTX 3060 (measured,
`docs/research/2026-10-19-long-context.md`). The change is against `K = 0` with the same
blocks.

| Input | Blocks | K = 0 | K = 256 | K = 512 | K = 768 |
|---|---|---:|---:|---:|---:|
| alice29.txt (152,089 bytes) | 64 KiB | 2.037 | 1.996 (−2.0 %) | 1.995 (−2.0 %) | 2.000 (−1.8 %) |
| lcet10.txt (426,754 bytes) | 64 KiB | 1.709 | 1.682 (−1.6 %) | 1.677 (−1.8 %) | 1.680 (−1.7 %) |
| cp.html (24,603 bytes) | 64 KiB | 1.505 | 1.426 (−5.3 %) | 1.415 (−6.0 %) | 1.400 (−7.0 %) |
| Chinese text, 512 KiB | 64 KiB | 3.688 | 3.567 (−3.3 %) | 3.539 (−4.0 %) | 3.523 (−4.5 %) |
| Apache log, 1 MiB | 64 KiB | 0.934 | 0.763 (−18.4 %) | 0.742 (−20.6 %) | 0.723 (−22.6 %) |
| All five, 16 KiB blocks | 16 KiB | 1.841 | 1.727 | 1.709 | 1.699 |
| All five, 64 KiB blocks | 64 KiB | 1.833 | 1.712 | 1.694 | 1.682 |
| Extra rows (long block) | | 0.1 % | 33.5 % | 100.4 % | 301.6 % |

Bit-exactness: the oracle, the CPU step, prefill and batch paths (Mac NEON, devbox AVX2) and
the CUDA decode and prefill paths give the same weights for real GPT-2 with `K = 512` (2,100
tokens, SHA-256 `09b99a923377b260b83d676c8799d4cc07f96f2996bced7471128d4d561bd452`).

## Consequences

- A block with `K > 0` costs more compute: `K + 1` restart rows for each `n_ctx - 1 - K` new
  tokens. For GPT-2 this is 34 % more rows with `K = 256`, 100 % with `K = 512` and 302 % with
  `K = 768` (exact counts, over a long block).
- A change of `K` changes the compressed bytes after the first restart. The encoder and the
  decoder must use the same `K`. A wrong `K` gives wrong tokens, and the checksum finds them.
- The restart rule depends only on the tokens of the block and on `K`. Thus every machine and
  every path computes the same weights (tests in `crates/hardshell/tests/context.rs` and
  `crates/hardshell-cuda/tests/overlap.rs`).
- The match model of the logs profile (ADR 0010) sees the whole block. It does not depend on
  the context rule.
