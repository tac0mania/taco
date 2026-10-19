# Security: threats, limits and fuzz results

Written 2026-10-19 by the fuzz agent. Plan: §3, §8, §9, §10, §11. ADRs: 0003, 0004, 0006, 0007,
0011, 0015.

This document lists the inputs that an attacker controls, the behaviour that Taco must have for
each input, the limits that exist and the limits that are missing. It also records the fuzz
harnesses, the number of cases that ran, and each finding.

## 1. The required behaviour

For every input, Taco must obey these rules:

1. **No crash.** Library code gives a typed error. It does not panic, abort or read out of
   bounds (CLAUDE.md §3).
2. **No silent wrong output.** `unfold` gives exactly the original data, or an error (plan §3
   rule 4).
3. **Bounded memory.** An input of `n` bytes cannot make Taco allocate much more than a fixed
   factor of `n`, or more than a limit that the caller sets.
4. **Bounded time.** The work of `unfold` is bounded by a limit that the caller can set. A
   small input does not cause unbounded work.
5. **Short errors.** An error carries offsets, lengths, ids and hashes. It never carries the
   original data, and it never copies a large part of an input.

## 2. Inputs and what an attacker controls

| Input | Who makes it | What an attacker controls | Reader |
|---|---|---|---|
| T1. The container (`.taco`) | Any person who sends a file | Every byte: header, table, blocks, hashes | `taco_format::Container::parse`, then `taco::unfold` |
| T2. The original data | The user of `fold` (also data from other people) | Every byte: invalid UTF-8, long runs, any length | `taco::Tokenizer`, `taco::fold` |
| T3. The model file (`.hsm`) | A model publisher, or a person who sends a model | Every byte: header JSON, sections, tensor values | `hardshell::hsm::HsmFile::parse`, `hardshell::gpt2::Model::load` |
| T4. The coded bytes of a block | The same as T1 | The range decoder input | `taco_coder::Decoder` |
| T5. CLI arguments | The user | Paths, block size, overlap, threads, backend | `taco-cli/src/args.rs` |

NOTE: the container stores the SHA-256 of the model. The decoder refuses a container that needs
another model (plan §3 rule 3). This does not protect against a hostile model file: the user
selects the model file with `--model`, and the hash only tells which model the container needs.

## 3. Limits

### 3.1 Limits that exist

| Input | Limit | Where | Source |
|---|---|---|---|
| T1 | The block table is inside the input before the parser allocates one entry for each block. | `taco-format/src/reader.rs` | ADR 0006 §9 |
| T1 | All sums of lengths use checked arithmetic. | `taco-format` | ADR 0006 §9 |
| T1 | The input ends exactly at the end of the last block. | `taco-format` | ADR 0006 §8 |
| T1 | A coded, mixed or zstd block is smaller than its original bytes. | `taco-format` | ADR 0006 §6, ADR 0011 §3 |
| T1 | A zstd frame has one frame, a content size equal to the block length, and a window of at most 8 MiB. The decoder writes at most the block length. | `taco-format/src/zstd_block.rs` | ADR 0011 §5 |
| T1 | The token count is at most the block length. A token has at least 1 byte. The decoded bytes are at most the block length. | `taco/src/codec.rs` | ADR 0007 §4 |
| T1 | The overlap `K` is at most `n_ctx - 2`. | `hardshell::gpt2::Model::check_overlap` | ADR 0015 §4 |
| T1 | The caller can set `UnfoldOptions::max_output_len`. The check comes before any block decodes. | `taco/src/unfold.rs` | |
| T1 | The CPU decoder decodes groups of at most 8 blocks on each thread, and stops at the first damaged group. | `taco/src/unfold.rs` | |
| T3 | `vocab_size` is 1 to 2^20. `n_layer` is 1 to 1,024. `d_model` and `d_ff` are 1 to 65,536. `n_ctx` is 2 to 65,536. | `hsm/reader.rs`, `gpt2/model.rs` | ADR 0004 |
| T3 | A shape has 1 to 4 dimensions. Each dimension is at least 1. The element count does not overflow. | `hardshell/src/tensor.rs` | |
| T3 | Every section is inside the file at its canonical offset. The padding is zero. | `hsm/reader.rs` | plan §8 |
| T3 | The exponent tables are not empty, do not increase, and start at 1 or more. `logits_exp` entries are at most 2^40. | `softmax.rs`, `logits.rs` | ADR 0004 |
| T3 | A rescale shift of up to 255 has one defined result. | `hardshell/src/int.rs` | ADR 0004 |
| T4 | The decoder reads zero bytes after the end of its input. Each symbol reads at most 4 bytes. | `taco-coder/src/decoder.rs` | ADR 0003 |
| T2 | BPE costs `O(n log n)` for a piece of `n` bytes. The pre-tokenizer makes one pass. | `taco/src/tokenizer` | |

### 3.2 Limits that are missing

| Input | Missing limit | Effect | Finding |
|---|---|---|---|
| T1 | The CLI sets no `max_output_len`. `unfold` keeps all decoded blocks and then a copy, so it needs about 2 times the original length. | A container of 9.6 KB aborts the CLI with 450 MiB of address space (measured). Fixed (streaming output, commit d63e60a). | F1 |
| T1 | The GPU decoder makes the state of every coded and mixed block before the first block runs. | A container of 120 KB used 11 GB of resident memory. The OOM killer stopped it (measured). | F2 |
| T1 | No limit on the work of a coded block, other than its stated length. | A container of 112 bytes costs 20,000 model steps (measured), and up to 2^32 - 1 steps for each block. A block is at most 64 MiB (commit d63e60a). | F5 |
| T1 | The block list of the parser uses about 40 bytes for each table entry of 5 bytes. | Memory of about 8 times the container length (est.). Linear, so acceptable. | F7 |
| T3 | No limit on the KV cache relative to the model file. | A model file of a few MB can state `n_layer = 1,024` and `n_ctx = 65,536`. One session then reserves `2 * n_layer * n_ctx * d_model` bytes (1 GiB for `d_model = 8`, est.), for each lane and each thread. | F6 |
| T3 | The JSON error message had no length limit. | An error message of 100,160 bytes (measured). | F3, fixed |
| T3 | The vocabulary parser reserved `vocab_size` entries before it read the section. | 8 MiB from a file of a few hundred bytes (est.). | F4, fixed |

## 4. Fuzz harnesses

The harnesses use only `proptest` and the standard library. Each case comes from one `u64`
seed (splitmix64), so every failure is reproducible. Each file has a short proptest run that
runs in every `cargo test`, and a long run that is `#[ignore]`.

| Target | File | Mutations | Oracle |
|---|---|---|---|
| T4 range decoder | `crates/taco-coder/tests/fuzz_decoder.rs` | Random tables (1 to 50,257 symbols, every total, zero and huge weights), random and valid input with a tail; symbol, bit and GPU-path decodes | No panic; each symbol is in the table; `target` with `decode_interval` gives the state of `decode`; a wrong interval is refused with no state change; encoder output decodes |
| T1 container reader | `crates/taco-format/tests/fuzz_reader.rs` | Toy-coder containers (versions 1 and 2, raw-only mode); bit flips, truncation, extra bytes, length fields set to large values, table entries changed or swapped, hashes changed | A parsed container obeys the layout rules; unfold gives an error or the original data |
| T3 model file | `crates/hardshell/tests/fuzz_hsm.rs` | `conformance/tiny.hsm`: byte damage; JSON header edits with the sections moved to stay canonical (large numbers, strings, removed, added and duplicate fields, shapes, tensor lists, shifts, version 2 constants); tensor values set to extremes | No panic; a loaded model gives `vocab_size` weights; the weight bound holds (largest at least 1, sum at most 2^60); the reference, portable, SIMD, `step_many` and batch paths give the same weights, bit for bit; short errors |
| T1 unfold | `crates/taco/tests/fuzz_unfold.rs` | 40 real containers of `tiny.hsm` (raw, coded, zstd and mixed blocks; profiles text, logs and auto; overlap 0 to 62); header fields, length and count changed together, kinds, stored bytes with the lengths made to agree, blocks swapped, duplicated or removed, token counts, hashes, version 2 written as version 1, coding parts and `K`, zstd frame headers | An error or exactly the original data; short errors; the time of the slowest case |
| T2 round trip | `crates/taco/tests/fuzz_unfold.rs` | Random, invalid UTF-8, long runs of one class (letters, spaces, new lines, digits), Unicode, merged tokens; all profiles, block sizes 16 to 1,000, overlap 0 to 62 | `unfold(fold(x)) == x` |

Run a long run on devbox (CPU only), with a memory limit and a time limit:

```text
scripts/dx fuzz "ulimit -v 6000000 && TACO_FUZZ_CASES=5000000 TACO_FUZZ_SEED=1000 \
  timeout 3600 cargo test -p taco-coder --test fuzz_decoder -- --ignored --nocapture"
```

Use the same form for `-p taco-format --test fuzz_reader`, `-p hardshell --test fuzz_hsm` and
`-p taco --test fuzz_unfold`. Use the dev profile (overflow checks on). `TACO_FUZZ_TRACE=1`
prints each seed before its case: use it to find the case of an abort or a hang.

The regression tests of the findings are in `crates/hardshell/tests/security_hsm.rs`,
`crates/taco/tests/security_unfold.rs` and `crates/taco-cli/tests/security_cli.rs`.

## 5. Results

All runs on devbox (Linux x86, processors 8 to 13, 6 threads), dev profile, 2026-10-19, seeds
from 1,000, `ulimit -v 6000000`, except the Mac row (seeds from 500,000,000). The first hsm run
stopped on a fault of the harness (a JSON edit of a tensor entry that was not an object), not of
the reader. The harness was repaired and the 1,000,000 cases ran again.

| Target | Cases | Result |
|---|---|---|
| T4 range decoder | 5,000,000 garbage runs and 5,000,000 round trips (measured, 526 s) | 0 failures |
| T1 container reader | 5,000,000 (measured, 249 s) | 0 failures. 105,746 damaged containers gave the original data (for example a flip that does not change a zstd frame); all others gave an error. |
| T3 model file | 1,000,000 (measured, 2,726 s) | 0 failures. 625,665 refused by the reader, 56,454 refused by `Model::load`, 317,881 loaded; 12,242,201 steps gave equal weights on the reference, portable, AVX2, `step_many` and batch paths. |
| T3 model file, Mac (NEON) | 20,000 (measured, 150 s) | 0 failures. 6,375 loaded; 243,368 steps equal on the reference, portable, NEON, `step_many` and batch paths. |
| T1 unfold | 2,000,000 (measured, 552 s) | 0 failures. 180,155 gave the original data, 271,321 `ChecksumMismatch`, 44,016 `CorruptBlock`, 78,545 `CorruptZstdBlock`, 12,644 `InvalidOverlap`, 169 `OutputTooLarge`, and all other parser errors (22 kinds). Slowest case 1.04 s. |
| T2 round trip | 30,000 (measured, 601 s), about 5,000 for each of the 6 data classes | 0 failures. Slowest case 5.9 s (dev profile; 94 blocks of 16 bytes with the profile `auto`: one mixer state for each block). |

## 6. Findings

### F1: `taco unfold` has no output limit, and `unfold` keeps 2 copies (high, fixed)

- **Input.** A raw-only container with one zstd block of RLE blocks. Each 4 bytes of the frame
  give 128 KiB. A container of 9.6 KB states 300 MiB. A container of 131 KB states 4 GiB in
  one block. More blocks state more.
- **Effect.** The CLI gives no `max_output_len`. `unfold` keeps every decoded block, then copies
  all of them into one buffer. Thus it needs about 2 times the stated length, plus the growth
  of the block buffer. With `ulimit -v 460800` (450 MiB), `taco unfold` stopped with SIGABRT:
  "memory allocation of 536870912 bytes failed" (measured, devbox).
- **Where.** `crates/taco-cli/src/main.rs` (`UnfoldOptions::new()` with no limit),
  `crates/taco/src/unfold.rs` (`parts`, then `out`).
- **Test.** `taco-cli/tests/security_cli.rs`,
  `unfold_of_a_small_container_that_states_much_data_does_not_abort` (Linux).
  The library defence passes: `taco/tests/security_unfold.rs`,
  `output_limit_refuses_a_zstd_bomb_before_it_decodes`.
- **Fix.** Commit d63e60a (2026-10-19, ADR 0018) writes the output as a stream. The test of F1
  passes and is not ignored. The library test uses 16 zstd blocks of 64 MiB, because a reader
  now refuses a larger block.

### F2: the GPU decoder makes the state of every block before it starts (high, the gpucli agent repairs it)

- **Input.** A container of 120 KB with 20,000 mixed blocks of 2 bytes (stored byte `0x01`,
  profile `logs`, coding part bit 0, mixer rule 1).
- **Effect.** `decode_on_gpu` makes one `BlockDecoder` for each coded and mixed block before the
  first slot runs. Each mixed block gets a new `MixState`: the logistic tables (144 KiB) and
  24 MiB of zeroed match tables. On devbox the process grew to 11 GB of resident memory and 68
  GB of address space, and the OOM killer stopped it (measured). On the CPU the same container
  fails in less than 1 s, because a group has at most 8 blocks (measured).
- **Where.** `crates/taco/src/unfold.rs` (`decode_on_gpu`), `crates/taco/src/gpu_unfold.rs`.
- **Test.** `taco/tests/security_unfold.rs`, `many_mixed_blocks_on_a_gpu_use_bounded_memory`
  (`#[ignore]`, Linux, needs a GPU and a free timing lock). The CPU test
  `many_mixed_blocks_on_the_cpu_fail_fast` passes.
- **Fix.** Make the `MixState` (or the full decoder) when a job enters a slot. The lead sent
  F2 to the gpucli agent (2026-10-19).

### F3: the JSON error of the model reader copied the input (low, fixed)

- **Input.** A model header with an unknown field of 100,000 bytes, or a long string where the
  schema has a number.
- **Effect.** `Error::HeaderJson` held the serde message, which quotes the field. The message
  had 100,160 bytes (measured).
- **Fix.** `hsm/reader.rs` cuts the message to 160 bytes on a character boundary.
- **Test.** `hardshell/tests/security_hsm.rs` (2 tests). Both failed before the fix (messages
  of 100,160 and 70,119 bytes, measured) and pass after it.

### F4: the vocabulary parser reserved memory from a header field (low, fixed)

- **Input.** A model header with `vocab_size = 2^20` and a short vocabulary section.
- **Effect.** `Vocabulary::parse` reserved `vocab_size` entries (8 MiB) before it read the
  section (est.).
- **Fix.** The capacity is at most the section length divided by 2, because each entry has at
  least 2 bytes. No test can see a capacity without an allocator hook, so this fix has no
  regression test.

### F5: the work of a coded block is bounded only by its stated length (medium, bounded)

- **Input.** A coded block whose stored bytes are only the token count `N` (3 bytes for
  `N = 20,000`).
- **Effect.** The range decoder reads zero bytes after its end, so each step gives token 0 (one
  byte). The block decodes to `N` bytes, and only the checksum at the end finds the damage. A
  container of 112 bytes cost 20,000 steps of the tiny model, 1.8 s in the dev profile
  (measured). The format permits blocks of up to 2^32 - 1 bytes, so one block can cost
  2^32 - 1 GPT-2 steps (years of CPU time, est.). `max_output_len` limits the work, but the CLI
  does not set it (F1).
- **Test.** `taco/tests/security_unfold.rs`,
  `coded_block_of_three_bytes_costs_one_step_for_each_stated_byte` (passes; it records the
  behaviour).
- **Status.** Commit d63e60a makes a maximum block size of 64 MiB a format rule. Thus
  one block costs at most 2^26 model steps, and the total work is bounded by the stated length
  (`max_output_len`, F1). The work for each stored byte stays large: this is a property of a
  model-based coder.

### F6: a model file can state a KV cache much larger than itself (medium, open, not measured)

- **Input.** A model file with a narrow model (`d_model = 8`), `n_layer = 1,024` and
  `n_ctx = 65,536`. The tensors of such a file need a few MB (est.).
- **Effect.** `Session`, `ReferenceSession` and `Batch` reserve `2 * n_layer * n_ctx * d_model`
  bytes of KV cache for each lane: 1 GiB (est.). The decoder has 8 lanes on each thread. The
  pages fill as the context grows.
- **Where.** `crates/hardshell/src/gpt2/model.rs` (`check_dims`), outside the ownership of the
  fuzz agent.
- **Recommendation.** `Model::load` refuses a KV cache of more than a fixed factor of the file
  length or of an absolute limit. The lead selects the limit (long-context models need a
  large cache).

### F7: the parser block list is about 8 times the table (low, accepted)

`Container::parse` makes one `BlockRef` (40 bytes, est.) for each table entry (5 bytes). A coded
block of 0 stored bytes is valid in the parser, so a container of `n` bytes can give
`n / 5` entries. The memory is linear in the input, so it obeys rule 3 of section 1.

## 7. What is not covered

- The GPU backends (`hardshell-cuda`, `hardshell-gpu`) were not fuzzed. F2 was found by
  analysis and one run.
- The CLI argument parser (T5) was not fuzzed. It has unit tests.
- The harnesses use one model (`tiny.hsm`). GPT-2 small is too slow for millions of cases.
- No coverage tool was used (no new tools). The counts of results by error type show which
  checks the cases reached.
- `fold` on data larger than 2,000 bytes was not fuzzed.
