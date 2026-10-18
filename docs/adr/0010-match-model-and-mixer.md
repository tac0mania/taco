# ADR 0010: the match model and the mixer of the logs profile

Date: 2026-10-18. Status: proposed (by the logs agent). Plan: §3, §4, §5.1, §7.6, §7.7, §9.
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

1. The part uses container version 2 (ADR 0011, ADR 0012). There is no separate version.
2. The decoder needs these facts from the container, and no other facts:
   1. The profile is `logs`, and the coding-part flags of the header have the bit "match model
      and mixer".
   2. The mixer rule version, one `u8`. Version 1 is sections 2 and 3. The rule version fixes
      every constant, so the header has no free parameters. A reader refuses an unknown rule
      version with a typed error.
   3. For each coded block: whether the block uses the part (proposal below).
3. Proposal for ADR 0011: a block kind "coded with the match part", next to the block kind
   "coded" (GPT-2 alone). The encoder can then code a block both ways and keep the smaller
   form. The cost is a second table and a second range coder in the encoder only, about
   0.5 ms for each token (est., from the 0.53 ms of `from_weights` in ADR 0003). The decoder
   does one of the two. With this rule, a block is never larger than with GPT-2 alone.
4. A version 1 reader refuses a version 2 container (ADR 0006 section 2). A version 2 reader
   reads version 1 containers with the rules of ADR 0007.
5. New conformance vectors: the two golden hashes of `crates/taco-match/tests/golden.rs`, and
   a container vector of the `logs` profile when ADR 0011 is in the code.

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
- The measurement keeps the GPT-2 context restarts at each 16 KiB step for all block sizes. A
  measurement of Taco with large blocks in the real code is still necessary.
