# Research: phase 1 part B, float drift between the Mac and devbox (2026-10-19)

This note records phase 1 part B (plan §12). The commands are in §10 of
`docs/research/2026-10-18-phase0-baseline.md`. The float NumPy GPT-2 of `tools/baseline`
computed the logits of the same 4 windows on the Mac (ARM) and on devbox (x86). The tool then
compared the frequency tables and tried decodes across the two machines. The measurements ran
on 2026-10-18. All numbers are measured, unless a number has a different label.

## 1. Results in short

- **No window decoded across the machines.** Full-sequence path: 0 of 4 windows in each
  direction. KV-cache path: 0 of 4 windows in each direction (measured).
- **Every table was different.** The tables differ in 4,096 of 4,096 steps on each path. The
  first different step is step 0 in every window. At step 0, the model input is only
  `<|endoftext|>`, so even one position gives different bits on the two machines (measured).
- **The decoder fails at once.** The first wrong byte is byte 0 or byte 2 of the window in all
  16 cross decodes (measured).
- **The differences are small.** The largest probability difference is 6.2e-5. The largest
  logit difference is 0.034. The coded stream of a window has the same size on both machines,
  or a size that is 1 byte different (measured). Thus the compression ratio does not change,
  but the decode fails.
- **The BLAS thread count changes the bits on devbox.** OpenBLAS with the default thread count
  against `OPENBLAS_NUM_THREADS=1`: the tables differ in 4,096 of 4,096 steps, and 0 of 4
  windows decode in each direction (measured). On the Mac, the same pair changed nothing
  (phase 1 part A). Thus a float decoder on x86 must also use the same thread count.
- **One path on one machine is repeatable.** Two dumps on devbox with the same settings gave
  the same tables in all 4,096 steps, and 4 of 4 windows decoded (measured, §7).
- These results agree with the expected result in §10 of the phase 0 note, with plan §1 and
  with ADR 0002. HardShell removes this failure: the phase 2 gate decoded 9 of 9 files across
  the same two machines (`docs/research/2026-10-18-cross-machine.md`).

## 2. Setup

| Item | Mac | devbox |
|---|---|---|
| CPU | Apple M2 (ARM), 8 GB | AMD Ryzen 7 5800H (x86, AVX2), WSL2, 12 GB |
| OS | macOS 26.5.2 | Ubuntu 24.04, Linux 6.18.54.1 (WSL2) |
| Python | 3.12.11 | 3.12.15 |
| NumPy | 2.5.4 | 2.5.4 |
| BLAS | Apple Accelerate | scipy-openblas 0.3.34 (DYNAMIC_ARCH, 64-bit integers) |
| Priority | `nice -n 10`, load average about 60 to 70 (other agents) | `taskset -c 10-13 nice -n 19`, load average about 15 |
| Threads | Default | Default (the process could use 4 processors), and 1 |

Both machines used the same model file (`model.safetensors`, SHA-256 in
`tools/baseline/manifest.toml`), the same tokenizer and the same tool code. The windows are the
first window of each of 4 files: `alice29.txt`, `option.rs`, `HDFS_2k.log` and `pg24264.txt`
(4 windows, 4,096 steps). The KV-cache path ran the 4 windows in one batch of 4 on both
machines. The dtype was float32 for the model. The tables use float64 for the softmax, then
plan §5.1 with `FREQ_BITS` = 32.

## 3. Commands

Each command ran from `tools/`. On devbox, each command ran through
`scripts/dx bench "cd tools && taskset -c 10-13 nice -n 19 ..."`.

1. On devbox and on the Mac, for `P` in `full` and `kv`:
   `python -m baseline.phase1 dump --path P --dtype f32 --files alice29.txt option.rs
   HDFS_2k.log pg24264.txt --out data/out/phase1/MACHINE-P-f32`.
2. On devbox: the same `full` command with `OPENBLAS_NUM_THREADS=1` and `--threads 1`, to
   `data/out/phase1/devbox-full-f32-t1`.
3. Copy the devbox directories to the Mac (`rsync`, 833 MB each).
4. On the Mac: `python -m baseline.phase1 compare A B --out data/out/phase1/cmp-NAME.json` for
   each pair in §4.

The comparisons (`cmp-*.json`) and the dump metadata (`*/meta.json`) are in
`tools/data/out/phase1/` on the Mac (ignored by git). The logit files (4.1 GB) were deleted
after the comparison, to save disk. The commands above make them again.

## 4. Comparison of the tables

| Pair (A against B) | Steps with a different table | First different step | Max abs probability difference | Max abs logit difference | Windows decoded A to B | Windows decoded B to A |
|---|---:|---|---:|---:|---:|---:|
| Mac full against devbox full | 4,096 of 4,096 (100 %) | 0 in every window | 6.0e-5 | 2.3e-2 | 0 of 4 | 0 of 4 |
| Mac KV (batch 4) against devbox KV (batch 4) | 4,096 of 4,096 (100 %) | 0 in every window | 6.2e-5 | 1.1e-2 | 0 of 4 | 0 of 4 |
| devbox full, default threads, against 1 thread | 4,096 of 4,096 (100 %) | 0 in every window | 1.1e-5 | 2.0e-2 | 0 of 4 | 0 of 4 |
| Mac full against devbox full, 1 thread | 4,096 of 4,096 (100 %) | 0 in every window | 5.9e-5 | 3.4e-2 | 0 of 4 | 0 of 4 |

All values are measured.

## 5. Each window

"First wrong byte" is the offset in the window of the first byte that the decoder gives wrong.
Byte 0 means that the first token was already wrong. "Stream" is the size of the coder stream
of the window that each machine encodes.

| Pair | File | Window bytes | Stream A (bytes) | Stream B (bytes) | First wrong byte, A to B | First wrong byte, B to A |
|---|---|---:|---:|---:|---:|---:|
| Mac full against devbox full | alice29.txt | 3,365 | 852 | 852 | 0 | 0 |
| | pg24264.txt | 1,443 | 669 | 669 | 0 | 0 |
| | option.rs | 2,929 | 415 | 415 | 2 | 0 |
| | HDFS_2k.log | 2,709 | 466 | 466 | 2 | 2 |
| Mac KV against devbox KV | alice29.txt | 3,365 | 852 | 852 | 0 | 0 |
| | pg24264.txt | 1,443 | 669 | 669 | 0 | 0 |
| | option.rs | 2,929 | 415 | 415 | 2 | 0 |
| | HDFS_2k.log | 2,709 | 466 | 467 | 2 | 2 |
| devbox threads, default against 1 | alice29.txt | 3,365 | 852 | 852 | 0 | 0 |
| | pg24264.txt | 1,443 | 669 | 669 | 0 | 0 |
| | option.rs | 2,929 | 415 | 415 | 0 | 2 |
| | HDFS_2k.log | 2,709 | 466 | 467 | 2 | 2 |

All values are measured.

## 6. Findings

- **The machine changes the result at step 0.** On one Mac, the full-sequence path and the
  KV-cache path gave the same table at step 0 (phase 1 part A). Across the two machines, step 0
  is already different. The cause is the matrix kernels: Accelerate and OpenBLAS sum the
  products in a different order (est.: the kernels are not examined).
- **The BLAS thread count is a real variable on x86.** With OpenBLAS, 1 thread and the default
  thread count give different tables in every step. Possibly OpenBLAS divides the inner
  dimension between threads, which changes the order of the sums (UNVERIFIED: the cause is not
  measured). On the Mac, Accelerate gave the same bits with 1 thread and with the default
  (part A). Thus part A was a weak test, as the phase 0 note said.
- **The size does not show the failure.** The streams of the two machines have the same size
  within 1 byte. A size test or a ratio test does not find this failure. Only a decode on the
  other machine finds it.
- **A float decoder needs the same machine, path, batch shape and thread count.** Parts A and B
  together show that each of these 4 variables breaks the decode.

## 7. Repeatability and sizes on devbox

Two more measurements on devbox (§10, steps 6 and 7 of the phase 0 note):

- **Repeat on devbox.** A second full-sequence dump on devbox, in a new process, with the same
  settings, gave the same tables in 4,096 of 4,096 steps. All 4 windows decoded in each
  direction (measured). Thus one path on one x86 machine is repeatable, as on the Mac.
- **Phase 0 on devbox, the same 4 files (32 KiB each, 61,301 tokens).** Command:
  `python -m baseline.phase0 --files alice29.txt option.rs HDFS_2k.log pg24264.txt --out
  data/out/phase1/devbox-phase0.json`.

| File | Coded bytes, Mac | Coded bytes, devbox | Float ideal bytes, Mac | Float ideal bytes, devbox | Same SHA-256 of the streams |
|---|---:|---:|---:|---:|---|
| alice29.txt | 8,194 | 8,193 | 8,548.42 | 8,548.42 | no |
| pg24264.txt | 15,267 | 15,268 | 15,281.57 | 15,281.57 | no |
| option.rs | 4,549 | 4,549 | 4,545.26 | 4,545.26 | no |
| HDFS_2k.log | 5,920 | 5,919 | 5,937.15 | 5,937.15 | no |

All values are measured. The coded sizes differ by 1 byte or less. The ideal sizes are the same
to 0.01 byte. But the coded streams are different on all 4 files.

On devbox, phase 0 also repeated two checks of the Mac. The full-sequence stream decoded with
the KV-cache path in 0 of 63 windows (the Mac: 1 of 270 windows for all 19 files). The
KV-cache path on both sides decoded 4 of 4 windows. The KV-cache tables differed from the
full-sequence tables in 1,024 of 1,024 steps in each of the 4 windows. On the Mac, step 0 was
the same (1,023 of 1,024). Thus on devbox even step 0 differs between the two paths (measured).

Timing on devbox (4 processors, `nice -n 19`, load average 14 to 47, measured): full-sequence
path 1,311 s for 61,301 tokens (21.4 ms for each token); KV encode 268 s and KV decode 86 s for
4 windows of 1,024 tokens. The Mac gave 2.7 ms for each token on the full-sequence path in
phase 0, with a lower load. The load was too different for a speed comparison.

## 8. Limits

- 4 windows only (4,096 steps). Part A used 19 windows on the Mac. The result (100 % of the
  tables differ) leaves no doubt for these 4 windows, but other files are not measured.
- The devbox default thread count was not read from OpenBLAS. The process could use 4
  processors (`taskset -c 10-13`), so the default was probably 4 threads (UNVERIFIED).
- float64 was not measured across the machines.
- A GPU float path is not measured. It is expected to differ again (est.).
- Other agents used both machines at the same time. The load changes the speed, not the bits:
  two runs of one path on one machine gave the same tables (the Mac in phase 0, devbox in §7).
