# ADR 0017: speculative decoding and the skip rule (mixer rule version 2)

Date: 2026-10-19. Status: proposed (by the spec agent). Plan: §3, §4, §5.1, §7.7, §9, §11.
Related: ADR 0010 §8 (the first notes on both ideas), ADR 0011 §2 (the coding parameters),
ADR 0013 (the oracle), ADR 0015 (context overlap), ADR 0016 (the coded GPU path). Research
note: `docs/research/2026-10-19-speculative.md`.

## Context

`taco unfold` decodes one block token by token. Each token needs the HardShell weights of
the token before it. Thus one block costs one HardShell step for each token, and a small
batch step takes about 2.6 ms on the RTX 3060 (measured, `2026-10-19-full-power.md` §3). A
16 KiB block of English needs about 10 to 12 s (est.).

Logs repeat. The match model of the profile `logs` (ADR 0010) predicts 57 % to 96 % of the
tokens of 18 logs correctly, in long runs (measured, `2026-10-18-logs.md`). ADR 0010 §8 names
two ways to use this in the decoder:

1. **Speculative decoding.** Run HardShell on the draft of the match model in one batched
   pass, then decode the tokens in order and drop the rows after the first wrong draft token.
   No bit changes.
2. **A skip rule.** Code a very likely token without the GPT-2 weights. Then the decoder does
   not wait for HardShell for that token. This changes the bits, so it is a new mixer rule
   version.

## Decision

### 1. The rules that every part obeys

1. The bits come from the sequential rules only. A draft is a guess. The decoder uses a row of
   a draft pass only if every input before it in the pass is the decoded token.
2. Every backend gives each row of a draft pass the weights of single steps, bit for bit
   (CLAUDE.md §1 rule 10). A draft pass is a prefill pass of one stream: the rows of a stream
   run at consecutive positions, with the context rule of ADR 0015 at a full context.
3. A truncation of a stream to its first `len` inputs gives the state of a stream that never
   ran the later inputs, bit for bit (section 2).
4. The match model and the mixer see only the decoded tokens, in the order of ADR 0010 §4. A
   speculative state change of the mixer (section 3.3) is undone before the next decoded
   token.

### 2. The engine: the draft pass and the truncation (no format change)

An **input** is a row that the caller gives: the `<|endoftext|>` row of a block start, or a
token. The **stream length** is the number of inputs since the last reset. The rows of a
context restart are not inputs.

`hardshell::gpt2::StreamState` holds the rules of a truncation. Every backend uses it, so the
rules are in one place:

1. Inside the current context, a truncation moves the position back only. Causal attention
   makes the cache entries at the lower positions independent of the later rows.
2. A truncation before the last context restart takes the kept history of the context before
   the restart (`n_ctx` tokens, kept at the restart). It runs the rows of that history again
   at the positions that the new context wrote (rows without weights). Then the old context
   is back, bit for bit. A forward row depends only on its token, its position and the rows
   before it.
3. A truncation before the restart before the last one is an error (`BadTruncate`). A pass
   of at most `n_ctx - K - 1` rows of one stream never needs it.
4. `room()` gives the inputs that a stream can take before its next restart. A draft of at
   most `room()` rows never needs the rebuild of rule 2.

The API (all bit-equal to `ReferenceSession`, tests in section 6):

| Backend | Draft pass | Truncation |
|---|---|---|
| CPU `Session` | `step_many(&[known, draft...])` | `truncate(len)`, `stream_len()`, `room()` |
| CPU `Batch` | `run_rows(&[(lane, token)])`, a lane many times | `truncate(lane, len)`, `stream_len(lane)`, `room(lane)` |
| `hardshell-cuda` `CudaBatch` | `run_weights`, `run_coded`, `run_coded_mixed` with a slot many times (prefill rows) | `truncate(slot, len)`, `stream_len(slot)`, `room(slot)` |
| `hardshell-gpu` `GpuBatch` | `run_rows`, `run_rows_coded` (new); `GpuBatch::with_rows` for passes of many rows | `truncate(slot, len)`, `stream_len(slot)`, `room(slot)` |

The CUDA and wgpu sessions (`CudaSession`, `GpuSession`) also have `truncate` and
`stream_len`.

### 3. The decoder protocol (for `crates/taco`)

For one block at the stream length `n` (the tokens `t[0..i]` are decoded; `i = n - 1`):

1. **The draft.** `MatchModel::speculate(k - 1)` gives the predictions of the next positions
   if each prediction is correct. These are the exact mixer inputs of the draft positions.
   Draft tokens `d[0..]` are their tokens. `MatchModel::draft` can differ (ADR 0010 §8).
2. **The pass.** The inputs are `t[i-1]` (or `<|endoftext|>` at the block start) and
   `d[0..k-1]`. Row `j` gives the weights for position `i + j`. Row `j` is valid if
   `t[i+r] = d[r]` for all `r < j`.
3. **Decode in order.** For `j = 0, 1, ...`: decode `t[i+j]` with the weights of row `j` and
   the normal calls of ADR 0010 §4. Stop after the first `j` with `t[i+j] != d[j]`, or after
   the last row. Then `truncate(n + j)`: the stream keeps the valid inputs.
4. **On a GPU with the coder op**, the target of row `j > 0` is known only after row `j - 1`.
   Ask `Decode(target)` for row 0 and `Encode(d[j])` for row `j > 0`. Row `j` gives `d[j]`
   exactly when the decoder target is inside the interval of `d[j]`. At the first row where
   it is not, the decoder needs the symbol of that row: run its input again as row 0 of the
   next pass (a "redecode" of the same row in the pass is a later improvement, section 7).
5. **The mixer on a GPU** (`run_coded_mixed`) needs `q` of every draft row in the pass, before
   the tokens are known. Call `Mixer::checkpoint()` before the pass. In the host step of row
   `j`: `skip()` or `mix_stats()`, then `checkpoint()` (keep it as `cp[j]`), then
   `update(d[j])`. After the pass, at the first row `j` with `t[i+j] != d[j]`:
   `rollback(&cp[j])`, then `update(t[i+j])`. Then `commit()`. The CPU path has the full
   weights of every row, so it needs no checkpoint: it mixes row `j` after the update of row
   `j - 1`.

Rules 1 to 3 change no bit of any container. Speculative decoding needs no format change.

### 4. The skip rule: mixer rule version 2

`taco-match` has the rules (`src/skip.rs`), `tools/logs/skip.py` has a Python copy with the
same golden decisions (section 6). The mixer gets a context table of 2^16 adaptive hit
probabilities (22 bits, start 2^21, count 0). At a position with the match prediction
`(p, L)` and the bucket `b = bucket(L)`:

1. The slot is `((key + 1) * 0x9E3779B97F4A7C15 mod 2^64) >> 48` with `key = p * 21 + b`.
2. `Mixer::skip` skips GPT-2 if the entry has the count 2 or more and the hit probability
   `h >= 3,774,873` (0.9 · 2^22). Every bucket can skip. The entry depends only on earlier
   tokens, so the encoder and the decoder make the same decision.
3. A skipped position codes a **flag** first: a table of two symbols with the weights `q_s`
   (symbol 0, "the token is `p`") and `2^16 - q_s` (symbol 1), `q_s = clamp(h >> 6, 1, 65535)`,
   `FreqTable::from_weights(.., 32)`.
4. After symbol 1, the position codes the token in the table of the GPT-2 weights with the
   weight of `p` set to 0 (`skip_miss_weights`). Only this case needs the GPT-2 weights.
5. A position that the rule does not skip uses mixer rule version 1 (`mix` or `mix_stats`).
6. After the token, the entry of the slot and `hit[b]` move with the hit rule of ADR 0010 §3
   rule 6, at every position with a prediction. A mixed position also updates its weight set.

`MixerParams::V2` and `SkipParams::V2` hold the constants. `taco_match::rule_constants(v)`
gives the constants of the rule versions 1 and 2. Memory: 512 KiB more for each mixer.

The rule is also a better model. The entry of (`p`, bucket) learns that template tokens are
almost always correct and field tokens are not. Thus the table cost falls on every one of the
18 logs (section "Measurements").

### 5. The format impact (a request to the owners of `taco-format` and `taco`)

1. Byte 0 of the coding parameters (offset 92, ADR 0010 §5 rule 2, ADR 0011 §2) gets the
   value 2: mixer rule version 2. A reader knows the versions 1 and 2 and refuses others
   (`UnknownMixerRule`). A reader before this ADR refuses version 2 with that typed error. It
   never decodes a version 2 block with version 1 rules.
2. The block kind 3 ("mixed") does not change. Its range coder bytes hold, for each skipped
   position, the flag symbol and, after symbol 1, the token symbol (section 4). The token
   count of the payload counts tokens, not symbols.
3. The encoder of the profile `logs` writes rule version 2 (a decision of the lead: the default
   changes the bytes and the golden container `7262257e…55ec`). It still codes each block
   with GPT-2 alone and mixed and keeps the smaller form (ADR 0010 §5 rule 7).
4. Version 1 containers decode as before: `MixerParams::V1` has no skip rule, `skip()` gives
   `None`, and the golden hashes of rule version 1 do not change.
5. New conformance vectors: the rule 2 decisions of `crates/taco-match/tests/golden.rs`
   (below), and a rule 2 container vector of `taco-format` and `taco` (not made yet).

### 6. Proof

- `crates/hardshell/tests/speculative.rs` (7 tests) and 4 unit tests of `gpt2::stream`:
  random draft lengths and correct prefixes, truncations to all valid rows and to fewer rows,
  3 model shapes, `K` from 0 to `n_ctx - 2`, 4 or more restarts, every kernel selection, a
  wide model (hsm version 2), a batch with the drafts of many lanes in one call, truncations
  across a restart and the error two restarts back. Every used row equals the oracle.
- `crates/hardshell-cuda/tests/speculative.rs` (4 tests and 1 ignored): the same against the
  CPU, for the weights and the coder op, with passes of 2 to 16 rows. Real GPT-2
  (`gpt2.hsm` `aaa03eb9…a5c2`), 2 slots, 3,400 tokens of the evaluation windows, drafts of up
  to 16 rows, `K = 0` and `K = 512`: every used row has the SHA-256 of the CPU row (measured,
  RTX 3060).
- `crates/hardshell-gpu/tests/speculative.rs` (4 tests and 1 ignored): the same for wgpu.
- `crates/taco-match/tests/spec.rs` (10 tests) and `tests/golden.rs`: `mix_stats` then
  `update` equals `mix` then `update`; `speculate` equals `observe` of each prediction; a
  rollback gives the state of its checkpoint; rule 2 decisions depend only on earlier tokens.
  Golden decisions of rule 2 on the synthetic log of ADR 0010 §6 (`u8` kind and `u32` q for
  each position): `f6f7755b…930f` (239 skips) with `SkipParams::V2`, `8781e104…f212` (453
  skips) with a table of 2^6 entries. The Python copy gives the same values (measured).
- The golden hashes do not change: HardShell `d4063522…0078`, the overlap vector
  `5a799282…4a83`, the taco-match rule 1 vectors.

### 7. Not done (later work)

- The wiring in `taco unfold` and `taco fold` (section 3, section 5): a request to the owner
  of `crates/taco`.
- A "redecode" op: the coder op of one row of the last pass, with a new target, without a new
  forward pass (section 3 rule 4). It saves one pass at the end of each hit run.
- A direct interval of the flag table and of a skipped hit, without `from_weights` of 2
  symbols (est. small).
- The skip rule on the GPU mixed coder op: the host decides the skip before the pass, so the
  GPU computes the row for the context only. The coder interval of a skipped row comes from
  the host.

## Measurements

The size cost was measured with the simulation of `tools/logs/skip.py` on the facts of
`logs.gpt2pass` (the exact integer GPT-2 weights of each token, `gpt2.hsm`, GPT-2 context
restarts at each 16 KiB and each 1,024 tokens, `K = 0`). The cost is the approximate table
cost of `logs.sim` (within 0.0004 bpb of the exact cost, `2026-10-18-logs.md`). It is not a
real container. The input is the 18 logs of `tools/logs/logs.toml` (6,556,027 bytes,
2,798,749 tokens). All values are measured on devbox, 2026-10-19.

| Skip context | Limit `h` | Blocks | Rule 1 | Rule 2 | Change | Tokens skipped | Skip misses |
|---|---|---|---:|---:|---:|---:|---:|
| bucket only | 0.98 | 64 KiB | 0.4954 bpb | 0.4961 | +0.16 % | 11.3 % | 1.46 % |
| token × bucket, 2^16 | 0.95 | 64 KiB | 0.4954 | 0.4740 | −4.31 % | 65.7 % | 0.55 % |
| **token × bucket, 2^16 (rule 2)** | **0.90** | **64 KiB** | **0.4954** | **0.4716** | **−4.79 %** | **73.1 %** | **1.00 %** |
| token × bucket, 2^16 | 0.85 | 64 KiB | 0.4954 | 0.4718 | −4.76 % | 74.4 % | 1.22 % |
| token × bucket, 2^18 | 0.90 | 64 KiB | 0.4954 | 0.4711 | −4.90 % | 73.1 % | 0.96 % |
| **token × bucket, 2^16 (rule 2)** | **0.90** | **1 MiB** | **0.4274** | **0.3963** | **−7.28 %** | **77.9 %** | **1.03 %** |
| **token × bucket, 2^16 (rule 2)** | **0.90** | **16 KiB** | **0.5919** | **0.5744** | **−2.95 %** | **64.9 %** | |

Rule 2 makes each of the 18 logs smaller: −0.4 % to −9.4 % (16 KiB blocks), −1.8 % to
−13.6 % (64 KiB) and −3.8 % to −15.8 % (1 MiB) (least: `apache_access.log`, `HPC_2k.log`;
most: `OpenSSH_2k.log`). A table of 2^16 entries costs 0.1 % more bits than 2^18 and uses
a quarter of the memory.

The sequential HardShell passes of `taco unfold` for one block (simulation of the protocol of
section 3 on the same facts; tokens for each pass, all 18 logs, 64 KiB blocks):

| Protocol | Draft 4 | Draft 8 | Draft 16 | Draft 32 |
|---|---:|---:|---:|---:|
| Speculative decoding (rule 1 bits) | 2.52 | 3.35 | 3.96 | 4.31 |
| Speculative decoding with a redecode op | 2.91 | 4.20 | 5.31 | 6.00 |
| Skip rule only (rule 2): 3.62 tokens for each pass | | | | |
| Skip rule and speculative decoding, with redecode | 4.90 | 5.46 | 5.89 | 6.19 |

The engine speed of a draft pass and the expected unfold gain are in the research note.

## Consequences

- Speculative decoding changes no bit. A decoder can use it on every container of the
  profile `logs`, also rule version 1.
- The skip rule changes the bytes of the profile `logs` and needs the reader changes of
  section 5. It makes the logs smaller and the decoder faster.
- Every new backend implements `truncate` with `StreamState` and passes the speculative tests.
- A truncation before a restart costs a rebuild of up to `K + 1 + d` rows. A decoder avoids
  it with drafts of at most `room()` rows.
