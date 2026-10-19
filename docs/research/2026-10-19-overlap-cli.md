# Research: the context overlap in `taco fold` and `taco unfold` (2026-10-19)

This note records the integration of the context overlap (ADR 0015) into the container, the
library and the command, and the measurements that set the defaults. "bpb" is bits per byte.
All numbers are measured, unless a number has a different label. The code is the working tree
of the overlap agent on 2026-10-19 (not committed when this note was written).

## 1. Results in short

- **The container stores `K`.** Bit 1 of the coding parts (offset 90) is the context overlap.
  Bytes 2 and 3 of the coding parameters (offsets 94 and 95) hold `K` as a little-endian
  `u16`. The version stays 2. A reader that does not know bit 1 refuses the container.
- **The defaults.** The profile `text`: `K = 0`, blocks of 16 KiB (no change). The profiles
  `logs` and `auto`: `K` = half of the model context (512 for GPT-2), blocks of 64 KiB.
  `taco fold --overlap K` selects any `K` from 0 to `n_ctx - 2`.
- **Logs, the new defaults against the old defaults:** 363,234 bytes against 489,113 bytes on
  the 18 logs (0.443 against 0.597 bpb, −25.7 %). Each of the 18 logs becomes smaller (−8.0 %
  to −39.9 %).
- **At equal `K = 512`, 64 KiB blocks are 18.3 % smaller than 16 KiB blocks** (363,234
  against 444,436 bytes). Each of the 18 logs is smaller with 64 KiB (−3.3 % to −30.7 %).
- **The cost.** On a log of 171,239 bytes with 6 threads: fold 2.6 times the time of the old
  defaults, unfold 3.6 times on the CPU and 2.8 times on CUDA (section 5). Most of it is less
  parallel work in a small file. For each byte, one fold thread takes 23 % more CPU time with
  64 KiB than with 16 KiB.
- **Byte identity.** CUDA and wgpu give the CPU container for every `K` that the tests use
  (0, 1, 12 and 22 on the tiny model; 512 on the xcheck inputs with GPT-2, section 6). wgpu
  has the overlap since commit `993c03b` (`GpuBatch::with_overlap`).
- **The old vectors do not change.** The text golden `0b0ba3f1…7c07` and the logs golden
  `7262257e…55ec` come out again with `--overlap 0`. The 9 text inputs of `scripts/xcheck` give
  the same `.taco` files as before.
- **`scripts/xcheck` passes** with a new input with `K = 512` (section 6).

## 2. The format (ADR 0011 §2, ADR 0015 §4)

| Field | Rule |
|---|---|
| Offset 90, coding parts (`u16`) | Bit 0: the match model and the mixer. Bit 1: the context overlap. Bits 2 to 15: 0 (`UnknownCodingParts`). |
| Offset 92, byte 0 of the coding parameters | The mixer rule version (1) with bit 0, else 0. |
| Offset 93, byte 1 | 0 (`ReservedNotZero`). |
| Offsets 94 and 95, bytes 2 and 3 | `K` (`u16`, little-endian) with bit 1. `K` must be 1 or more (`ZeroContextOverlap`). Without bit 1 both bytes are 0 (`ReservedNotZero`). |
| Offsets 96 to 103, bytes 4 to 11 | 0 (`ReservedNotZero`). |
| Profile `text` | Permits bit 1. Refuses bit 0 (`CodingPartsNotInProfile`). |
| `K` against the model | The decoder refuses `K > n_ctx - 2` before the first block (`InvalidOverlap`). The container does not know `n_ctx`. |

Each `K` has exactly one encoding. The reader of commit `2bdeac5` refuses bit 1 with
`UnknownCodingParts`, so it never decodes such a container with the rule of `K = 0`.

The tests:

1. Unit tests of `crates/taco-format/src/coding.rs`: the bytes of every part, the refusal of
   unknown bits, of `K = 0` with bit 1, and of `K` bytes without bit 1.
2. `tests/version2.rs`: hand-made containers with `K` in both profiles, with the mixer, and
   the refusals. The writer gives the hand-made bytes.
3. A property test (`tests/round_trip.rs`): random layouts with a random `K` round trip, and
   bytes 94 and 95 hold `K`.
4. A fuzz property (`tests/fuzz.rs`): any value of the coding fields is refused, or it parses
   to coding parts that the writer writes back as the same 14 bytes.
5. Golden vectors E (vector B with `K = 512`, `d29fb5c2…0408`) and F (vector D with `K =
   512`, `9555a3e6…b6c6`). Each is equal to its base vector except bytes 90, 94 and 95.

## 3. The library and the command

| Part | Change |
|---|---|
| `FoldOptions` | `overlap: Option<Overlap>` and `block_size: Option<BlockSize>`; `None` is the default of the profile. `with_overlap`, `effective_overlap(model)` (refuses `K > n_ctx - 2`), `effective_block_size()`. |
| `FoldProfile` | `default_overlap(model)` and `default_block_size()`. |
| CPU fold | `SessionConfig::with_overlap`. |
| CPU unfold | `Batch::with_overlap` with the `K` of the header. |
| CUDA fold (profile `text`) | The windows of ADR 0015 §2: `<|endoftext|>`, the `K` kept tokens, then `n_ctx - 1 - K` new tokens, in parallel slots. A unit test runs each window in a fresh CPU session and compares the weights with one session with the overlap (`K` = 0, 1, 2, 16, 32, 61, 62 on `conformance/tiny.hsm`). |
| CUDA unfold | `CudaBatch::set_overlap`; passes of 224 rows when `K > 0`, so a restart of 513 rows takes 3 passes. |
| wgpu fold (profile `text`) | The same windows as CUDA, one row of each slot in each step. |
| wgpu unfold | `GpuBatch::with_overlap` with the `K` of the header (commit `993c03b`). |
| Backend selection | Does not depend on `K`: every backend has the overlap. |
| `container_overlap(bytes)` | The `K` of a container (`Overlap::NONE` without the coding part). |
| `FoldStats::overlap` | The `K` of the container. |
| `taco fold --overlap K` | The command prints "blocks of N bytes, overlap K". |

`auto` codes as `logs`, so it has the defaults of `logs`. When no block is mixed, `auto` writes
the `text` container with the same blocks, and that container keeps `K`. Thus `taco fold`
with no option uses `K = 512` and 64 KiB blocks also for text files. This is a consequence of
the decision that the lead can change.

## 4. Logs: sizes

### 4.1 Method

1. **Data.** The 18 logs of the logs-profile note: the 16 Loghub files whole, and the first
   1 MiB of `apache_access.log` and of `nginx_json.log` (6,556,027 bytes in total).
2. **Model.** `tools/data/out/gpt2.hsm`, SHA-256 `aaa03eb9…a5c2`.
3. **Program.** A release build of the working tree. `taco fold --profile logs --backend cpu
   --threads 1` with `--block-size 16384 --overlap 0` (the old defaults), `--block-size 16384
   --overlap 512`, and `--block-size 65536 --overlap 512` (the new defaults). One job for each
   file and configuration, 6 jobs at a time on processors 8 to 13 of devbox, `nice 15`.
4. **Cross-machine.** 4 of the 18 files with 16 KiB and `K = 512` also ran on the Mac. The
   `.taco` files have the same SHA-256 on both machines.
5. **Check against the logs-profile note.** The old defaults give the 16 KiB sizes of that
   note (for example HDFS 45,163 bytes, 1.255 bpb in both).

### 4.2 Results

The sizes are bytes of the `.taco` file, with bpb in parentheses.

| File | Bytes | 16 KiB, K = 0 (old) | 16 KiB, K = 512 | 64 KiB, K = 512 (new) | K = 512 against K = 0 (16 KiB) | 64 KiB against 16 KiB (K = 512) | New against old |
|---|---:|---:|---:|---:|---:|---:|---:|
| Android_2k.log | 279,076 | 22,295 (0.639) | 20,964 (0.601) | 15,497 (0.444) | −6.0 % | −26.1 % | −30.5 % |
| Apache_2k.log | 171,239 | 6,046 (0.282) | 5,030 (0.235) | 3,993 (0.187) | −16.8 % | −20.6 % | −34.0 % |
| BGL_2k.log | 317,150 | 39,580 (0.998) | 34,843 (0.879) | 32,370 (0.817) | −12.0 % | −7.1 % | −18.2 % |
| Hadoop_2k.log | 384,948 | 17,137 (0.356) | 14,874 (0.309) | 10,304 (0.214) | −13.2 % | −30.7 % | −39.9 % |
| HDFS_2k.log | 287,848 | 45,163 (1.255) | 41,588 (1.156) | 39,268 (1.091) | −7.9 % | −5.6 % | −13.1 % |
| HealthApp_2k.log | 187,456 | 14,693 (0.627) | 12,719 (0.543) | 10,609 (0.453) | −13.4 % | −16.6 % | −27.8 % |
| HPC_2k.log | 151,178 | 16,073 (0.851) | 15,295 (0.809) | 14,788 (0.783) | −4.8 % | −3.3 % | −8.0 % |
| Linux_2k.log | 216,485 | 9,448 (0.349) | 8,638 (0.319) | 7,191 (0.266) | −8.6 % | −16.8 % | −23.9 % |
| Mac_2k.log | 319,414 | 50,111 (1.255) | 48,408 (1.212) | 36,739 (0.920) | −3.4 % | −24.1 % | −26.7 % |
| OpenSSH_2k.log | 225,216 | 9,420 (0.335) | 7,699 (0.273) | 6,017 (0.214) | −18.3 % | −21.8 % | −36.1 % |
| OpenStack_2k.log | 595,119 | 63,119 (0.848) | 58,475 (0.786) | 45,631 (0.613) | −7.4 % | −22.0 % | −27.7 % |
| Proxifier_2k.log | 236,962 | 15,286 (0.516) | 13,134 (0.443) | 11,612 (0.392) | −14.1 % | −11.6 % | −24.0 % |
| Spark_2k.log | 196,268 | 11,122 (0.453) | 9,921 (0.404) | 8,182 (0.334) | −10.8 % | −17.5 % | −26.4 % |
| Thunderbird_2k.log | 325,192 | 18,773 (0.462) | 15,701 (0.386) | 12,517 (0.308) | −16.4 % | −20.3 % | −33.3 % |
| Windows_2k.log | 285,433 | 10,941 (0.307) | 9,607 (0.269) | 7,898 (0.221) | −12.2 % | −17.8 % | −27.8 % |
| Zookeeper_2k.log | 279,891 | 16,898 (0.483) | 15,010 (0.429) | 12,876 (0.368) | −11.2 % | −14.2 % | −23.8 % |
| apache_access.log, 1 MiB | 1,048,576 | 90,780 (0.693) | 85,126 (0.649) | 66,860 (0.510) | −6.2 % | −21.5 % | −26.3 % |
| nginx_json.log, 1 MiB | 1,048,576 | 32,228 (0.246) | 27,404 (0.209) | 20,882 (0.159) | −15.0 % | −23.8 % | −35.2 % |
| **Loghub, 16 files** | 4,458,875 | 366,105 (0.657) | 331,906 (0.595) | 275,492 (0.494) | −9.3 % | −17.0 % | −24.8 % |
| **All 18** | 6,556,027 | 489,113 (0.597) | 444,436 (0.542) | 363,234 (0.443) | −9.1 % | −18.3 % | −25.7 % |

Findings:

1. **The block size has the larger effect.** At `K = 512`, 64 KiB is 18.3 % smaller than
   16 KiB. The overlap at 16 KiB gives 9.1 %. The match model and the mixer start again at
   each block, and they find more repeats in a larger block (est.; the table cost of ADR
   0015 gives only 2.9 % for 64 KiB against 16 KiB on one log with GPT-2 alone).
2. **The new defaults are near 1 MiB blocks with `K = 0`.** The logs-profile note gives
   0.489 bpb for the Loghub files with 1 MiB blocks and `K = 0` (one block for each of these
   files). The new defaults give 0.494 bpb with 3 to 10 blocks for each file, so the threads
   and the lanes have more parallel work.
3. **Against xz -9e on the whole file** (the logs-profile note, section 5): the Loghub files
   are 10.8 % smaller (0.494 against 0.554 bpb). Two files are larger than xz: Mac_2k.log
   (0.920 against 0.856 bpb) and OpenStack_2k.log (0.613 against 0.520 bpb). This
   comparison mixes block sizes: xz has the whole file, Taco has blocks of 64 KiB.

## 5. Speed

Warning: these are not benchmarks. devbox ran 6 size jobs on processors 8 to 13 at the same
time (load average about 6 on 14 processors). The jobs below ran on processors 0 to 5 with 6
threads, under the timing lock (`~/scratch/timing.lock`). The RTX 3060 Laptop had its 45 W
power limit for these runs; no other process used it (0 % utilization before and after).
Warning: Rohit removed the 45 W limit later on 2026-10-19 (the GPU now runs at up to 95 W).
All GPU times of this note are at 45 W. The times include
the model load and, for CUDA, the device start and the self-test (about 2.5 s, est. from
ADR 0016).

### 5.1 The logs profile, `Apache_2k.log` (171,239 bytes)

| Step | 16 KiB, K = 0 (old) | 16 KiB, K = 512 | 64 KiB, K = 512 (new) |
|---|---:|---:|---:|
| `.taco` | 6,046 B | 5,030 B | 3,993 B |
| fold (the CPU on every engine, ADR 0016 §5) | 119.0 s, 1.41 KB/s | 185.3 s, 0.90 KB/s | 306.8 s, 0.55 KB/s |
| unfold, the CPU | 274.7 s, 0.61 KB/s | 382.3 s, 0.44 KB/s | 985.7 s, 0.17 KB/s |
| unfold, CUDA (mixed blocks read back the weights) | 43.4 s, 3.85 KB/s | 85.5 s, 1.96 KB/s | 122.5 s, 1.37 KB/s |

The file has 11 blocks of 16 KiB but only 3 blocks of 64 KiB. Fold runs one thread for each
block, so 64 KiB uses 3 of the 6 threads. The CPU unfold puts the 3 blocks in the lanes of one
thread. The CUDA unfold has one slot for each block. A file of 384 KiB or more gives each of 6
threads a block (est.).

The CPU time for each byte, from the size runs of section 4 (one thread for each job, all 18
logs, the sum of the job times): 21,846 s (16 KiB, `K = 0`), 28,880 s (16 KiB, `K = 512`,
+32 %), 35,590 s (64 KiB, `K = 512`, +23 % against 16 KiB). The machine was shared, so these
sums are orders of size only.

### 5.2 The profile `text`, the first 64 KiB of `alice29.txt` (blocks of 16 KiB)

| Step | K = 0 | K = 512 |
|---|---:|---:|
| `.taco` (equal on CUDA and the CPU) | 16,701 B | 16,441 B (−1.6 %) |
| fold, CUDA | 3.4 s | 4.2 s |
| unfold, CUDA | 14.6 s | 16.1 s |
| fold, the CPU | 43.4 s | 69.5 s |
| unfold, the CPU | 173.1 s | 198.9 s |

The CUDA fold runs the windows with their `K + 1` restart rows in one pass, so `K = 512`
costs 24 % more time here. The CPU fold costs 60 % more. The default of `text` stays `K = 0`.

### 5.3 Whole files (the lead's run)

The lead ran whole files with the binary of the working tree (`~/bench/whole-gpu/results.tsv`
on devbox; the overlap working tree, not committed; `--backend auto`, CUDA, with the logs
defaults of that time: 16 KiB, `K = 512`). Examples: `Apache_2k.log`, profile `logs`: 5,030
bytes, fold 193.4 s (the CPU), unfold 84.3 s (CUDA). The same file, profile `text`: 10,059
bytes, fold 4.8 s, unfold 7.0 s (CUDA). All 30 round trips of that run were exact.

## 6. Byte identity and the cross-machine check

1. **`crates/taco/tests/backends.rs`** (the tiny model, `n_ctx = 24`): 10 inputs, 1 and 4
   threads, `K` = 0, 1, 12 and 22. On devbox, CUDA gives the CPU container, unfolds the CPU
   container, and the CPU unfolds the CUDA container, for every case. The goldens with `K =
   12` come out of CUDA. The same holds for wgpu on GL/D3D12 on devbox
   (`GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA HARDSHELL_GPU_ADAPTER=NVIDIA`)
   and for wgpu on Metal on the Mac (4 of 4 tests pass, 127 s). Warning: on devbox the test
   process with GL crashes (SIGSEGV) after the checks pass. The tree of HEAD (`dfee08c`,
   without this work) crashes in the same way, so the cause is the Mesa D3D12 driver and not
   the overlap (measured). The tests print one line for each backend after its checks.
2. **The xcheck inputs with GPT-2 and `K = 512`, CUDA and wgpu against the CPU** (devbox,
   section 6.1).
3. **`scripts/xcheck`** has a new input, `apache_32k.k512.log` (the first 32 KiB of
   `Apache_2k.log`, profile `text`, `--overlap 512`). The two `.logs.` inputs have the logs
   defaults, so they now have `K = 512`. The script checks that the three files have `K =
   512` in the header. Results: section 6.2.

### 6.1 CUDA and wgpu against the CPU on the xcheck inputs

devbox, `gpt2.hsm`, `--overlap 512`, 6 threads, the 12 inputs of `scripts/xcheck`: profile
`text` for each input, and also profile `logs` for the two `.logs.` inputs (14 cases). wgpu
ran on GL/D3D12 on the RTX 3060. For each case, the `.taco` files of CUDA and of wgpu are
equal to the CPU file, and CUDA and wgpu each restore the CPU file exactly: 14 of 14 cases, 0
failures. The text fold ran on the GPU (the summary names the backend). The logs fold ran on the
CPU on every engine (ADR 0016 §5), and the logs unfold ran on the GPU.

| Input | Profile | `.taco` with `K = 512` |
|---|---|---:|
| apache_32k.k512.log | text | 1,943 B |
| binary_4k.xls | text | 1,440 B |
| dasdas.txt | text | 116 B |
| empty.bin | text | 104 B |
| english_32k.txt | text | 8,365 B |
| english_4k.txt | text | 1,207 B |
| loghub_linux_64k.logs.log | text, logs | 4,108 B, 2,111 B |
| nginx_32k.logs.jsonl | text, logs | 1,285 B, 916 B |
| other_language_6k.txt | text | 2,811 B |
| page.html | text | 4,513 B |
| random_3000.bin | text | 3,109 B |
| server_8k.log | text | 1,361 B |

The run stopped once for the GPU benchmark of the lead (the timing lock) and continued after
it. The first 7 cases ran at the 45 W limit, the other 7 without it. The power limit does not
change the bits.

### 6.2 `scripts/xcheck`

Two runs, both PASS (Mac Apple M2 and devbox x86, `gpt2.hsm` `aaa03eb9…a5c2`):

1. With the logs blocks of 16 KiB (before the decision): 12 of 12 `.taco` files identical, 24
   of 24 round trips exact. The 9 text inputs gave the same files as the xcheck before this
   work.
2. With the final defaults (logs blocks of 64 KiB, `K = 512`): 12 of 12 identical, 24 of 24
   round trips exact, the 3 overlap inputs have `K = 512` in the header.

| Input | Bytes | `.taco` (run 2) |
|---|---:|---:|
| apache_32k.k512.log (text, `K = 512`) | 32,768 | 1,943 |
| loghub_linux_64k.logs.log (logs defaults) | 65,536 | 2,111 |
| nginx_32k.logs.jsonl (logs defaults) | 32,768 | 916 |
| The 9 text inputs (`K = 0`) | | unchanged |

## 7. Conformance vectors

| Vector | Length | SHA-256 |
|---|---:|---|
| Container vector E (vector B, `K = 512`, profile text) | 329 B | `d29fb5c2f32daf836657bb73a9f48d4b9eca966f08aec8c9571eacafdcf80408` |
| Container vector F (vector D, `K = 512`, profile logs) | 312 B | `9555a3e603c93f6f27943855b3faa47d1679200060fd07756ce29441ee75c6b6` |
| taco, text golden input, `--overlap 12` | 1,660 B | `d705d4b6971fe351f29d14cd5723d3b2a79ced03facae32910bc13d09f3ce340` |
| taco, logs golden input, blocks of 512, default `K` (12) | 1,329 B | `584b60fbd89d3f689cdd75161d43be973ef3c87294680fae1bd0d6f3f7066835` |
| taco, all defaults of `logs` (64 KiB, `K = 12`), 69,632 B of input | 40,241 B | `6cdc1edc5b963b5cebb224d93e824841013b6e668d2e6f3dd3b424d16a5ced77` |
| taco, text golden, `--overlap 0` (unchanged) | 1,660 B | `0b0ba3f1bc8579247476163489ed3155a7828916d27e92f3435e9a8e74157c07` |
| taco, logs golden, `--overlap 0` (unchanged) | 1,328 B | `7262257e6d2875ae393dd2474acc8a00dd1b8adf829eeef75b0ca719f73b55ec` |

The tiny model codes worse than zstd on blocks of 64 KiB, so both blocks of the all-defaults
vector are zstd blocks. That vector pins the header of the defaults. The vector with blocks
of 512 pins mixed and coded bytes with the default `K`.

## 8. Limits and open items

- **The GL driver of devbox crashes at exit** (section 6, also at HEAD). This is not a fault
  of the overlap, but the `backends` test fails on devbox with GL for this reason.
- **The logs fold runs on the CPU.** The GPU mixer of commit `dda9edd` (`run_coded_dual`,
  `run_coded_mixed`) is not used by `taco` yet. With it, the logs fold and unfold can run on
  CUDA, and the speed cost of section 5.1 changes.
- **`auto` uses the logs defaults** also for text files (section 3).
- **The speeds** come from one run on a shared machine (section 5).
- **ADR 0016 §3** has an update: the windows have `K`, and CUDA and wgpu both support it.
- **The CUDA slot count** uses 24 bytes of work buffers for each row and each vocabulary token
  (`CUDA_ROW_BYTES_PER_VOCAB`, est.), the larger estimate of the two model-file versions, so a
  wide model (version 2) does not overflow the GPU memory.

## 9. Commands

```
scripts/dx overlap scripts/check.sh
taco fold FILE -o FILE.taco --model tools/data/out/gpt2.hsm --profile logs --backend cpu --threads 1 --block-size 65536 --overlap 512
taco unfold FILE.taco -o FILE.back --model tools/data/out/gpt2.hsm --backend cuda --threads 6
scripts/xcheck
```

The runner scripts of this note (`run.py`, `speed.sh`, `gpucheck.sh`) are in the scratch
directory of the session and on devbox in `~/scratch/overlap/`.
