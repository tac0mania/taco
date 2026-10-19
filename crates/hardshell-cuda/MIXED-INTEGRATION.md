# The mixed coder op: how `taco` uses it

Written 2026-10-19 by the gpumixer agent. Readers: the lead, or the agent that changes
`crates/taco/src/{gpu_fold,gpu_unfold,lanes,codec,mix}.rs` and `crates/taco-match`. The design
is form 1 ("two round trips") of ADR 0016 §5. When the integration is done, ADR 0016 §5 must
say "built" and give the measurements of section 7.

## 1. What the op does

The profile `logs` makes each table from the weights of the match model and the mixer (ADR
0010 §3). Before this op, the GPU sent all 50,257 weights of each token back (402 KB). Now
each token costs two read backs of 24 bytes:

1. The GPU runs the forward pass. A stats kernel reads the GPT-2 weights of each row from the
   logits that stay on the GPU. It gives `MixStats`: `W = sum(w)`, `w[p]` for the prediction
   `p` of the row, and `top` (the token with the largest weight, the lowest id on a tie, token
   0 if all weights are 0).
2. The host calls the mixer of the row with these values. The mixer gives `q`.
3. The host computes `R = max(W - w[p], 1)`, `m = 2^16 - q`, `k = max(0, bitlen(R) + 16 - 62)`
   and `w'[p] = max(1, (R * q) >> k)` (`src/mixed.rs`, `mix_record`). These are the
   expressions of `taco_match::Mixer::mix` (`apply`).
4. A mixed coder kernel makes the table of `w'[i] = max(1, (w[i] * m) >> k)` for `i != p` and
   `w'[p]`, and answers the query. A row without a prediction uses `max(1, w[i])` (rule 1). A
   plain row uses `w[i]` (the table of GPT-2 alone).

The match model and the mixer state stay on the host. They are small and sequential.

## 2. The API (`hardshell_cuda`)

| Item | Use |
|---|---|
| `CudaBatch::run_coded_mixed(inputs, queries, tables, freq_bits, mixer)` | Unfold. One call for each step. `tables[i]` is `MixInput::Plain` (a coded block) or `MixInput::Mixed(prediction)` (a mixed block). Gives the `CoderInterval` of each input. |
| `CudaBatch::run_coded_dual(inputs, tokens, predictions, freq_bits, mixer)` | Fold of `logs` and `auto`. Gives two intervals for each input: GPT-2 alone (first) and mixed (second), from the same forward pass (ADR 0010 §5 rule 7.1). |
| `mixer: FnMut(usize, MixStats) -> u32` | The host step. The batch calls it for each input with a prediction, in input order, across the passes. It returns `q` in `1..=65535`. |
| `MixStats { total, predicted, top }` | The values of pass 1 of `Mixer::mix`. The agreement bit is `top == p`. |
| `mixed_intervals_of_weights(dev, weights, …)` | The op on given weights. Tests only. |

After an error, no position of the batch changes, but the batch can have called `mixer`. Then
drop the mixer state of the call (treat it as a fault of the backend).

## 3. A request for `taco-match` (its owner)

`Mixer::mix` needs the full weights. The host step needs the same state change from the
statistics only. Add this function and make `mix` use it, so that one code path sets the state:

```rust
/// Rules 1 to 4 and the pending state of [`Mixer::mix`] from the statistics of the weights:
/// `total = sum(w)`, `predicted = w[p]`, and `agree` (`p` is the token with the largest weight,
/// the lowest id on a tie). Gives `q`, or `None` without a prediction. A later
/// [`Mixer::update`] gives the state that `mix` and `update` give.
pub fn mix_stats(
    &mut self,
    total: u128,
    predicted: u64,
    agree: bool,
    prediction: Option<Prediction>,
) -> Option<u32> {
    self.pending = None;
    let pred = prediction?;
    let q = self.probability(u128::from(predicted), total, pred.length, agree);
    self.pending = Some(Pending { token: pred.token, ..q });
    Some(q.q)
}
```

A property test must show: `mix_stats` with the values of `w`, then `update`, gives the same
next `peek` and the same golden hashes as `mix(w)`, then `update`. Until this exists,
`examples/mixed.rs` uses a short vector with the same `pg`, length bucket and agreement bit
(`mix_from_stats`, marked `HACK:`). Do not copy that into `taco`.

`taco::mix::MixState` then needs three calls instead of `mix` and `observe`:
`predict() -> Option<Prediction>` (from the match model), `mix_stats(stats) -> Option<u32>`
(the mixer), and `observe(token)` (`Mixer::update`, then `MatchModel::observe`).

## 4. Fold (`gpu_fold.rs`)

1. `FoldOptions::uses_gpu` becomes true for `logs` and `auto` on the CUDA engine. The wgpu
   engine has no mixed op: keep the CPU path for it.
2. **One slot for each block, not for each window.** The mixer is sequential over the whole
   block, so the windows of a block cannot run in parallel slots. The slot runs the rows of the
   block in order. The batch applies the context rule at each restart
   (`CudaBatch::set_overlap(K)` with the `K` of the container, ADR 0015).
3. The rows of a block: row 0 has the input `<|endoftext|>` and codes `t[0]`; row `j` has the
   input `t[j-1]` and codes `t[j]`. `tokens[i]` is the token that the row codes.
4. Each call gives each running slot up to 512 rows (prefill). Many blocks run in parallel
   slots (up to 64).
5. Before a call, advance the match model of each slot over the rows of the call:
   `predictions[i] = model.predict()`, then `model.observe(t[j])`. The match model depends
   only on the tokens.
6. In the host step for input `i` (block `b`, token `j`): `q = mixer.mix_stats(…)`, then at
   once `mixer.update(t[j])`. Fold knows the token, so the next row of the block sees the new
   state. A row without a prediction has no call, and its `update` does nothing on the CPU
   (no pending state), so skip it.
7. After the call: the first interval goes to the encoder of GPT-2 alone
   (`push_interval`), the second to the mixed encoder (a new `push_mixed_interval`). Each
   encoder has its stop rule (ADR 0007 §5.2). When both stopped, the slot leaves and the
   block stops.
8. The exact sequence is `gpu_encode` in `examples/mixed.rs`. It gives the CPU bytes of both
   encoders (section 6).

## 5. Unfold (`gpu_unfold.rs`)

1. One call for each step for all slots: coded and mixed blocks together (no read back of
   weights, no CPU threads for the mixer).
2. For each slot: `queries[i] = Decode(decoder.target())`. For a mixed block,
   `tables[i] = Mixed(state.predict())`; for a coded block, `tables[i] = Plain`.
3. The host step: `q = mixer.mix_stats(…)` only. The token is not known yet.
4. After the call, for each slot: `decode_interval(interval)`. For a mixed block, also
   `mixer.update(token)` and `model.observe(token)`, in this order.
5. `BlockDecoder` needs a mixed form of `next_interval`, and access to `predict` and
   `mix_stats` of its `MixState`.
6. The exact sequence is `gpu_decode` in `examples/mixed.rs`.

## 6. Proof that the bits are the CPU bits

- `tests/mixed.rs` (8 tests, pass on the RTX 3060): every interval and the statistics of
  random calls equal `FreqTable::from_weights(Mixer::mix(w))` on the CPU. The calls have decode
  and prefill rows, restarts, an overlap of 9, plain rows among mixed rows, and the dual op.
  They run on the op-level path, the dp4a path (i64 logits) and the tensor-core path (compact
  logits). The real `taco_match::Mixer` of each slot gives `q`. A written-out copy of rule 5 is
  checked against `Mixer::mix` at every row. The op on given weights covers zero weights,
  all-zero rows, one-hot rows (`R = 1`), weights above 2^48 (the u128 loop of the mixer), ties,
  `q` of 1, 2, 2^15, 65,534 and 65,535, a vocabulary of 50,257, and 9, 16 and 32 bits.
- `examples/mixed.rs equiv` with `gpt2.hsm` (`aaa03eb9…a5c2`), the first 16 KiB of 6 logs
  (Apache_2k, HDFS_2k, OpenSSH_2k, Linux_2k, apache_access, nginx_json; 39,499 tokens, 5 or more
  restarts each), K = 0 and K = 512: the bytes of both encoders of the GPU equal the CPU bytes
  for each file, and the GPU decodes all 12 blocks (6 mixed, 6 plain, in one batch) to the
  tokens. The CPU path is the calls of ADR 0010 §4 (Session, MatchModel, Mixer::mix,
  FreqTable, Encoder).
- No change to the plain paths: all 38 tests of `hardshell-cuda` pass (also `--release
  --include-ignored`), the golden vector `d4063522…0078`, and `examples/gpt2.rs equiv` gives 7 of
  7 windows equal with the published hashes and 2,643 of 2,643 intervals.

Why the values are equal (ADR 0010 §3):

- The statistics are pass 1 of `Mixer::mix`. GPT-2 weights are at most 2^40 and `vocab <= 2^20`,
  so the u64 sum equals the u128 sum. The stats kernel keeps the lowest id of a tie and gives
  token 0 when all weights are 0, as the loop of `mix`.
- `w[i] <= R` for `i != p`, so `(w[i] * m) >> k < 2^62`. The kernel uses a u128 product, so the
  value equals the u128 loop and the u64 loop of `apply`.
- The mixed weights are below 2^62 and their sum is below 2^63. The coder kernel (`coder_body`,
  now a template on the weight source) is `from_weights` for every weight sum below 2^64: its
  sum is u64, and `quot` and `quot_fast` are exact for every divisor below 2^64.

## 7. Speed (measured, devbox, RTX 3060 Laptop at its 45 W limit)

`examples/mixed.rs bench`, `gpt2.hsm`, tokens of `apache_access.log`, the timing lock held,
`taskset -c 0-5`, 2 s of GPU work before the timing. Another agent loaded the CPUs (load
average 10); no other process used the GPU. Two runs:

| Shape | `run_coded` (text) | `run_coded_mixed` | `run_coded_dual` | `run_weights` (read back) |
|---|---|---|---|---|
| prefill 1 slot x 512 rows | 24,442 to 25,232 | 18,667 to 19,518 | 17,504 to 18,356 | 651 to 668 |
| prefill 8 slots x 512 rows | 23,200 to 23,515 | 19,421 to 19,700 | 17,392 to 17,866 | 633 to 696 |
| decode, batch 8 | 2,076 to 2,128 | 2,461 to 2,474 | | 800 to 925 |
| decode, batch 32 | 11,653 to 11,823 | 10,928 to 10,996 | | 788 to 842 |
| decode, batch 64 | 18,575 to 18,701 | 13,850 to 14,232 | | 765 to 888 |
| decode, batch 128 | 20,055 to 20,711 | 16,875 to 17,293 | | 717 to 811 |
| decode, batch 224 | 19,623 to 20,644 | 18,362 to 18,540 | | 600 to 789 |

Tokens/s. Decode is positions 0 to 128 (a short context). An earlier run with a load average
of 5 gave 884 to 1,197 tokens/s for the read back, and the same values for the other columns.

- The mixed op gives 74 % to 94 % of the tokens/s of the text op. At batch 8 it gave more
  than the text op in each run (not explained). The extra cost is one more pass over the
  logits (the stats kernel), one more kernel launch, and one more wait for the GPU in each
  pass.
- Against the read back: 28 to 31 times in prefill; in decode 3 times at batch 8, 13 times at
  batch 32, and 23 to 31 times at batch 224.
- Fold of `logs` with `run_coded_dual`: about 17,500 tokens/s. The CPU fold of ADR 0016 gave
  0.85 KB/s (about 380 tokens/s at 2.3 bytes for each token, est.). The host mixer is not in
  these numbers: the bench calls `Mixer::peek` (O(1), about the cost of `mix_stats`, est.).
- The memory of a batch grows by 80 bytes for each row.
