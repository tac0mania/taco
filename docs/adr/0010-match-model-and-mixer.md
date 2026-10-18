# ADR 0010: the match model and the mixer of the logs profile

Date: 2026-10-18. Status: accepted (proposed by the logs agent; the format rules of section 5
were completed by the profile agent on 2026-10-19). Plan: §3, §4, §5.1, §7.6, §7.7, §9.
Related: ADR 0006 and ADR 0007 (container and coded block), ADR 0011 (container version 2),
ADR 0012 (the profiles `text` and `logs`).

## Context

On logs, GPT-2 alone only ties xz. A Loghub HDFS log of 16 KiB gives 3,129 bytes with Taco,
3,164 with xz and 3,139 with zstd (measured). Logs repeat templates, ids and field values over
long distances. Most repeats are further away than the 1,024 tokens of the GPT-2 context, or
on the other side of a context restart (plan §7.7). A match model finds the longest earlier
occurrence of the current context and predicts that the token after it comes next. PAQ, lpaq
and cmix use match models of this kind.

The measurement is in `docs/research/2026-10-18-logs.md`. It is the source of each constant in
this ADR (CLAUDE.md §1 rule 9).

## Decision

### 1. Scope

1. The match model and the mixer are one coding part of the `logs` profile (ADR 0012). The
   `text` profile does not use them.
2. The part sits between HardShell and `FreqTable::from_weights`. HardShell does not change.
   The coder does not change. The part changes the u64 weights only.
3. The crate `taco-match` is the reference implementation. It is a kernel crate: integers
   only, no I/O, no clock, no global state. `tools/logs/matchmodel.py` and
   `tools/logs/mixer.py` are Python copies that give the same bits (section 6).
4. **Mixer rule version 1** is the name of the rules and the constants in sections 2 and 3. A
   change to any rule or constant is a new mixer rule version.

### 2. The match model (mixer rule version 1)

The model sees the tokens of one block. The history `h` holds the tokens that the model has
seen. Position `i` is the position of the next token.

| Constant | Value |
|---|---|
| Orders `N` (longest first) | 32, 16, 8, 4, 2, 1 tokens |
| Table index bits | 20 (one table of 2^20 `u32` entries for each order) |
| Largest verified length | 64 tokens |
| Largest match length | 65,535 |
| Hash multiplier `B` | `0x9E3779B97F4A7C15` |
| Mix multiplier | `0xD6E8FEB86659FD93` |

Rules:

1. Hash of order `N` at position `i >= N` (all operations modulo 2^64):
   `H = sum over k in 1..=N of (h[i-k] + 1) * B^(k-1)`. The slot is
   `((H xor (H >> 32)) * MIX) >> (64 - 20)`.
2. Each entry of a table holds the last position `j >= N` with that slot, stored as `j + 1`.
   The value 0 is an empty entry.
3. After the token at position `i - 1` is known:
   1. If the match length `len > 0` and `h[ptr]` equals the token: `ptr = ptr + 1` and
      `len = min(len + 1, 65535)`. Else `len = 0`.
   2. Append the token to `h`.
   3. If `len < 32`, look up each order `N > len` with `N <= i`, longest first. A candidate
      `c` is accepted if the verified length `L` is at least `N`. `L` is the count of `k` in
      `1..=64` with `h[c-k] = h[i-k]`. The count stops at the first difference or at the block
      start. For the first accepted order: `ptr = c` and `len = L`.
   4. Put position `i` into the table of each order `N <= i`.
4. The prediction is the token `h[ptr]` with the length `len`, if `len > 0`.

The verification makes a hash collision harmless: a candidate with a different context is
refused. Thus the table size changes the compressed bytes, but never the correctness.

### 3. The mixer (mixer rule version 1)

Probabilities are 16-bit (`p / 2^16`). The logistic domain is in units of 1/256 nat, in
`-4095..=4095`.

| Constant | Value |
|---|---|
| `squash` table | `E[0] = 2^62`, `E[j+1] = (E[j] * C + 2^61) >> 62`, `C = 4593706758521714574` = round(2^62 · exp(−1/256)). `squash(x) = (2^78 + d/2) / d` with `d = 2^62 + E[x]` for `x >= 0`, `squash(−x) = 2^16 − squash(x)`, clamped to `[1, 65535]`. |
| `stretch(p)` | the smallest `x` with `squash(x) >= p` (4095 if none) |
| Buckets of the length `L` | `L − 1` for `L < 16`; then 15 for 16 to 31, 16 for 32 to 63, and so on, at most 20 |
| Weight sets | 42: one for each bucket and each value of the agreement bit |
| Start weights | GPT-2 input 1.0, match input 1.0, bias 0 (units of 2^-16: 65,536, 65,536, 0) |
| Learning rate | `max(16, floor(160 · 64 / (n + 64)))`, `n` = earlier updates of the set (at most 2^20) |
| Hit probability | 22 bits, start 2^21; rate `2 / (2n + 3)` with `n` at most 255 |
| Weight limit | ±2^22 |

Rules, for the weights `w` and the prediction `(p, L)`:

1. `W = sum(w)` (`u128`). With no prediction, each weight becomes `max(w, 1)`. Stop.
2. `pg = clamp(floor(w[p] · 2^16 / W), 1, 65535)`. If `W = 0`, `pg = 1`.
3. `b = bucket(L)`. `a = 1` if `p` is the token with the largest weight (the lowest id on a
   tie), else 0. The weight set is `c = 2b + a`.
4. Inputs: `x_g = stretch(pg)`, `x_m = stretch(hit[b] >> 6)`, bias 256.
   `dot = wg[c] · x_g + wm[c] · x_m + wb[c] · 256`.
   `x = clamp((dot + 2^15) >> 16, −4095, 4095)` (arithmetic shift). `q = squash(x)`.
5. `R = max(W − w[p], 1)`, `m = 2^16 − q`, `k = max(0, bitlen(R) − 46)`. This `max` is the
   one saturating operation of the rule. For `i != p`: `w'[i] = max(1, (w[i] · m) >> k)`.
   Then `w'[p] = max(1, (R · q) >> k)`. Thus `p` has the probability `q / 2^16`, and the other
   tokens share `m / 2^16` in the proportions of GPT-2.
6. After the actual token, `y = 1` if it is `p`, else 0. `err = y · 2^16 − q`. Each weight of
   set `c` changes by `(x_input · err · lr + 2^19) >> 20`, then is clamped to ±2^22. Then
   `hit[b]` moves towards `y · 2^22` by `floor(2 · distance / (2n + 3))`, and `n` grows by 1 to
   at most 255.

Bounds: `w[i] <= R` for `i != p`, so each output weight is in `1..2^62`, and the sum is below
2^63. No token has the weight 0 (plan §3 rule 5). `|dot| < 2^36` and each weight change is
less than `2^36` before the shift, so `i64` is sufficient.

### 4. The order of the calls in a block

This replaces step 2 of ADR 0007 section 3 for a block that uses the part:

1. At the block start: `MatchModel::reset()` and `Mixer::reset()`. Blocks stay independent.
2. For each token `t[i]`: HardShell gives `w`. `Mixer::mix(w, MatchModel::predict())`. Make the
   table with `from_weights(w, 32)`. Code `t[i]`. Then `Mixer::update(t[i])` and
   `MatchModel::observe(t[i])`.

The decoder does the same calls in the same order with the decoded token.

### 5. The format

These rules are part of container version 2 (ADR 0011). There is no separate version. A
change to a rule of this section changes which files are valid, so it needs a new container
version.

1. **The coding part.** Bit 0 of the coding parts field (header offset 90, `u16`) is the
   coding part "match model and mixer". The other 15 bits are reserved. A reader refuses a
   reserved bit (`UnknownCodingParts`).
2. **The mixer rule version.** Byte 0 of the coding parameters (header offset 92, `u8`) is
   the mixer rule version when bit 0 is set. Version 1 is sections 2 and 3. The rule version
   fixes every constant, so the header has no free parameter. A reader refuses an unknown
   version, also 0 (`UnknownMixerRule`). When bit 0 is clear, byte 0 is 0. Bytes 1 to 11 of the
   coding parameters (offsets 93 to 103) are 0. A reader refuses another value
   (`ReservedNotZero`).
3. **The profile.** The profile `text` (byte 0) has no coding part. A reader refuses bit 0
   with the profile `text` (`CodingPartsNotInProfile`). The profile `logs` (byte 1) permits
   bit 0, and also no coding part (for example ADR 0011 vector C, in the raw-only mode). The
   decoder does not use the profile. It uses the coding parts and the block kinds.
4. **The per-block choice is a block kind, not a flag.** Block kind 3 is "mixed": a block
   coded with the match model and the mixer. Kind 1 (coded) stays "GPT-2 alone". Reasons:
   1. The decoder must know the choice for each block. The block table already has a kind
      byte for each block, so kind 3 costs no byte. A flag in the payload changes ADR 0007; a
      flag in the header cannot be different for each block.
   2. The encoder can code each block both ways and keep the smaller form (item 7). Thus a
      `logs` container is never larger than the `text` container of the same input and block
      size.
   3. A coded block in a `logs` container decodes with no match tables (24 MiB less memory).
5. **The rules of kind 3.** The stored bytes have the layout of a coded block
   (`varint(token count) || range coder bytes`, ADR 0007 §1). The decoder refuses a mixed
   block with the rules of ADR 0007 §4 (`CorruptBlock`). The parser refuses a mixed block:
   1. in a version 1 container (`UnknownBlockKind`);
   2. if the coding parts do not have bit 0 (`MixedWithoutCodingPart`);
   3. in the raw-only mode (`CodedBlockInRawOnly`);
   4. if it is not smaller than its original bytes (`CodedNotSmaller`). A mixed block of 0
      bytes is valid, as a coded block of 0 bytes (ADR 0006 §6.4).
   Thus the growth bound of ADR 0011 §8 does not change.
6. **Decoding a mixed block.** The decoder makes a new match model and a new mixer of the rule
   version of the header for each mixed block, and does the calls of section 4. The HardShell
   weights are the weights of a coded block (ADR 0007 §3). The mixer state belongs to the
   block, so the block can decode in any lane of a `Batch` (ADR 0013).
7. **The encoder (rules that change the bytes, not the decoder).**
   1. With the profile `logs`, the encoder codes each block with two range coders from the
      same HardShell weights: GPT-2 alone and mixed. Each coder has the stop rule of ADR 0007
      §5.2 with the stop length `min(original length, zstd length)`. The model stops when both
      coders have stopped.
   2. The selected form is the smallest of raw, zstd, coded and mixed. On a tie, the order is
      raw, zstd, coded, mixed (`Block::choose_smallest(…).or_mixed(…)`). Mixed wins only if it
      is strictly smaller than every other form. Coded wins a tie with mixed because it
      decodes without the part.
   3. A `logs` container has bit 0 and rule version 1 in its header, also if no block is
      mixed.
   4. The profile `text` codes GPT-2 alone. Its bytes are the bytes before this ADR (the
      golden container `0b0ba3f1…7c07` does not change).
8. **The profile `auto` (encoder only).** The encoder codes as `logs`. If no block is mixed, it
   writes the `text` container: the same blocks, profile 0, no coding part. Else it writes the
   `logs` container. Thus `auto` gives the smaller of the two containers, and `text` on a tie.
   The rule depends only on the input, the model and the block size, so it is the same on
   every machine and for every number of threads. A test checks both cases. The cost is the
   cost of `logs` for every input.
9. **Compatibility.** A version 1 reader refuses a version 2 container (ADR 0006 section 2). A
   version 2 reader reads version 1 containers with the rules of ADR 0007. A reader of version 2
   before this ADR refuses bit 0 (`UnknownCodingParts`) and kind 3 (`UnknownBlockKind`) with a
   typed error. It never decodes a mixed block as a coded block.
10. **Conformance vectors.**
    - The two golden hashes of `crates/taco-match/tests/golden.rs` (section 6).
    - Container vector D (`crates/taco-format/tests/golden.rs`): the data and model hash of
      vector B, profile `logs`, bit 0, rule version 1, blocks raw, mixed (7 fixed bytes),
      zstd, coded (0 bytes), mixed (0 bytes). 312 bytes, SHA-256
      `69578bfc50bee4fa9f8a0eb6f09421ce4a64b1c48ec01e811cad3622eee95af0`. A hand-made copy of
      the layout gives the same bytes.
    - The `logs` golden container of the `taco` crate (`crates/taco/tests/logs_profile.rs`,
      tiny model): 2,148 bytes in blocks of 512, kinds mixed, zstd, raw, zstd, coded.
      1,328 bytes (the `text` container of the same input: 1,339 bytes), SHA-256
      `7262257e6d2875ae393dd2474acc8a00dd1b8adf829eeef75b0ca719f73b55ec`.

### 6. Tests and equality of the Python copy

- `crates/taco-match`: 19 tests. Property tests: the match model obeys a second, simple
  implementation of section 2; the same input gives the same output; a reset gives the state of
  a new model and mixer; every output weight is in `1..2^62`; the predicted token gets the
  probability `q`; no prediction changes only the zero weights; a draft starts with the
  prediction and changes nothing.
- Golden vectors: the SHA-256 of the predictions and of all mixed weights for a synthetic log
  of 4,000 tokens, for the V1 tables and for tables of 2^6 entries (many collisions). The
  Python copy (`python -m logs.vectors`) gives the same four hashes and the same hashes of the
  `squash` and `stretch` tables (measured, 2026-10-18).

### 7. Cost for each token

| Part | Cost |
|---|---|
| Match model | one hash pass over the last 32 tokens (32 multiplications); while `len < 32`, at most 6 verifications of at most 64 tokens |
| Mixer, no prediction | one pass over the weights |
| Mixer, with a prediction | two passes over the weights; 112 µs for 50,257 weights (measured, `examples/cost.rs`, Mac M2, shared) |
| Reset | 0.28 ms: the 6 tables of 2^20 entries (measured, same program) |
| Memory | 24 MiB of tables and 4 bytes for each token of the block |

One HardShell step costs about 18 ms on one thread (measured, about 55 tokens/s). Thus the part
adds less than 1% to the time of a token (est.).

### 8. The skip rule and speculative decoding (later work)

- **Skip rule.** The mixer state (`hit[b]`, the weights of set `c`, the counts) depends only on
  past tokens. Thus a rule such as "skip HardShell for this token when bucket `b` has a hit
  probability above a limit" can use it on both sides. Two facts make the rule harder:
  1. Rule 3 needs `pg` and the agreement bit from the GPT-2 weights. A skipped token needs a
     second rule without GPT-2 (for example `q` from `hit[b]` only, and a fixed table for the
     other tokens). That is a new mixer rule version.
  2. HardShell must still add the skipped token to its context. It can add a run of skipped
     tokens in one batched step, so the skip saves the output projection (the largest matrix,
     768 × 50,257) for each skipped token, not the full step (est.).
- **Speculative decoding.** `MatchModel::draft(k)` gives the next `k` tokens of the current
  match without HardShell and without a change of state. The draft reads `h[ptr]`,
  `h[ptr+1]`, …, and it reads its own earlier tokens when the match overlaps the position (a
  repeat with a short period). The first token is the prediction. The decoder can run
  HardShell on the draft positions in one batched pass, then decode each token in order with
  the mixer and the coder, and stop at the first token that differs from the draft.
  - **No roll-back of the match model.** `draft` does not change the model. The model and the
    mixer see only the decoded tokens, through `observe` and `update`, in the order of
    section 4. Thus a rejected draft costs nothing in this part. Only the HardShell context
    must drop the positions after the first wrong token (a truncation of its KV cache).
  - A later prediction can differ from the draft: a correct token can make a longer match
    replace the current match (section 2, rule 3.3). Then the decoder uses the prediction,
    not the draft. The draft is a guess; the bits come from the sequential rules only.
  - The mixer is sequential, but it costs about 0.1 ms for each token. The batched HardShell
    pass must give the same bits as single steps (CLAUDE.md §1 rule 10).
  - Expected draft length (measured on 18 logs, 1 MiB blocks): 40 % to 94 % of the tokens are
    in runs of 4 or more correct predictions; the mean run length for a token is 7 to 454
    tokens (`logs.analyse.hit_runs`).

## Measurements

Exact table cost of the simulation of rule version 1, with the integer GPT-2 weights, plus an
estimate of the container (`docs/research/2026-10-18-logs.md`, measured):

| Data | Block | GPT-2 alone | GPT-2 + match | xz -9e | zstd -19 --long=27 |
|---|---|---:|---:|---:|---:|
| 16 Loghub files, 4,458,875 bytes | 16 KiB | 1.006 bpb | 0.658 bpb | 0.966 bpb | 0.958 bpb |
| 16 Loghub files | 1 MiB | 1.001 | 0.497 | 0.554 | 0.597 |
| Apache access log, 1 MiB | 1 MiB | 0.951 | 0.400 | 0.453 | 0.482 |
| nginx JSON lines, 1 MiB | 1 MiB | 0.441 | 0.170 | 0.173 | 0.193 |
| 14 text files (control) | 16 KiB | 2.229 | 2.009 | 3.044 | 3.018 |

The part makes every text file smaller (−2.1 % to −38.4 %). bzip2 -9 is smaller than Taco on
the JSON lines (0.157 bpb), and xz is smaller on OpenStack (random hex ids).

## Consequences

- The logs profile has new compressed bytes and new conformance vectors.
- A change to any constant of sections 2 and 3 is a new mixer rule version and a new golden
  vector. The tables are part of the format.
- The match model and the mixer use 24 MiB for each block that runs at the same time.
- `scripts/check-no-floats.sh` must add `taco-match` to its kernel list.
- The memory of the decoder grows by 24 MiB for each mixed block that decodes at the same time
  (up to 8 lanes for each thread). The tables are zeroed allocations, so only the used entries
  cost resident memory.
- The measurement keeps the GPT-2 context restarts at each 16 KiB step for all block sizes. A
  measurement of Taco with large blocks in the real code is still necessary.
