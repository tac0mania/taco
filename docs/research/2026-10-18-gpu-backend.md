# GPU backends: bit-exact integer GPT-2 on wgpu and CUDA

Date: 2026-10-18. Author: the cuda agent. Decision record: ADR 0008.

All numbers are measured on 2026-10-18 unless a label says otherwise. "Tokens/s" counts model
steps of all slots together. devbox was loaded by other work during most CUDA runs (load
average 37 to 46 on 14 processors, from Python jobs, `paq8px` and `eval_bits`); the CUDA path
uses little CPU, but the numbers can be lower than on an idle machine.

## 1. Summary

- Two backends give the CPU weights bit for bit: `crates/hardshell-gpu` (wgpu, WGSL) and
  `crates/hardshell-cuda` (CUDA, PTX). A user needs only the GPU driver.
- Real model `gpt2.hsm` (`aaa03eb9…a5c2`): 7 of 7 text windows (2,643 steps, one window with a
  context restart) give the CPU SHA-256 on both backends. On CUDA this holds for decode and for
  prefill. 2,643 of 2,643 coder intervals equal the CPU `FreqTable`.
- The golden tiny vector `d4063522…0078` comes out of both backends.
- **Speed gate (10,000 tokens/s, RTX 3060): PASS for decode at short context (18,565 tokens/s,
  batch 128) and for prefill (12,012 to 22,355 tokens/s). FAIL for decode over a full
  1,024-token window (7,229 tokens/s, batch 224).** Section 5 gives the reason.
- Update of 2026-10-19 (section 7): fused kernels, new tile plan, new attention kernels. Full
  window at batch 224: 8,691 to 9,195 tokens/s (HEAD: 6,803 to 7,252, same conditions). Still
  below 10,000. The GPU ran at a 45 W power limit (default 80 W), at 300 to 500 MHz.

## 2. Equivalence evidence

Per-window SHA-256 of all `u64` weights, in step order (start block, then one step for each
token). The text is tokenized with the longest vocabulary match.

| Window | Steps | CPU SHA-256 | wgpu (GL on D3D12) | CUDA decode | CUDA prefill |
|---|---|---|---|---|---|
| alice29.txt | 257 | `a9b37a6ced372c6b…809b` | equal | equal | equal |
| cp.html | 257 | `bdace25b0710f6b4…7463` | equal | equal | equal |
| fields.c | 257 | `3d36b3b38cc7c3ce…1074` | equal | equal | equal |
| HDFS_2k.log | 257 | `c25d926d5a6dd4ea…43cf` | equal | equal | equal |
| kennedy.xls | 257 | `5d40afa09bd2ff1d…8444` | equal | equal | equal |
| option.rs.txt | 257 | `5ffe8ac8b2adc0d5…4384` | equal | equal | equal |
| alice29.txt, 1,100 tokens | 1,101 | `b0de8f4ca55360b6…01ba` | equal | equal | equal |

Run it again: `scripts/dx cuda` and
`target/release/examples/gpt2 equiv tools/data/out/gpt2.hsm <files>` in each crate.

Test suites (`cargo test -p hardshell-gpu`, `cargo test -p hardshell-cuda`):

| GPU | Backend | Result |
|---|---|---|
| Apple M2 | wgpu, Metal | 12 of 12 pass |
| NVIDIA RTX 3060 Laptop | wgpu, OpenGL on D3D12 | 12 of 12 pass |
| llvmpipe (x86) | wgpu, Vulkan | 12 of 12 pass |
| AMD Radeon iGPU | wgpu, OpenGL 4.2 on D3D12 | fail: the process crashes in the Mesa driver at the first compute dispatch (the shaders compile) |
| NVIDIA RTX 3060 Laptop | CUDA | 11 of 11 pass |
| Mac (no CUDA) | CUDA | builds; 2 PTX checks pass; GPU tests skip with a message |

## 3. The portable path (wgpu)

| Batch | Tokens/s, positions 3-19 | Tokens/s, positions 512-528 | GPU memory |
|---|---|---|---|
| 1 | 106 | 108 | 149 MB |
| 8 | 751 | 675 | 292 MB |
| 32 | 1,311 to 1,388 | 1,203 to 1,208 | 782 MB |
| 128 | 1,138 to 1,166 | 966 to 974 | 2,740 MB |

These runs read back all weights (402 KB for each token). With the coder op on the GPU (16 bytes
for each token), batch 128 gave 2,164 tokens/s on a loaded machine.

Where the time goes at batch 128, position 530 (112.5 ms for each step, measured with GPU
timestamps): the read back takes 62 ms (0.83 GB/s), driver and submit 21 ms, and the kernels
28.8 ms (attention 10.4, matrix products 13.4, that is 1.2 TMAC/s). The 64-bit and 128-bit
emulation is cheap: norm, linear output and logits together take 3.8 ms.

## 4. The CUDA path

### 4.1 Decode

Each step runs one token for each slot, with the coder op (24 bytes back for each token).

| Batch | Tokens/s, positions 3-19 | Tokens/s, positions 512-528 | ms for each step (3-19) | GPU memory |
|---|---|---|---|---|
| 1 | 339 | 377 | 2.95 | 330 MB |
| 8 | 2,707 | 2,196 | 2.96 | 466 MB |
| 32 | 11,602 | 5,089 | 2.76 | 951 MB |
| 64 | 16,465 | 7,062 | 3.89 | 1,588 MB |
| 128 | **18,565** | 7,074 | 6.89 | 2,867 MB |
| 192 | 17,384 | 7,211 | 11.04 | 4,149 MB |
| 224 | 17,309 | 7,321 | 12.94 | 4,788 MB |
| 224, positions 0 to 1,023 | **7,229** (average) | | | 4,788 MB |

With all weights read back, every batch size gives only 262 to 548 tokens/s: the read back of
402 KB for each token dominates. Thus the coder op is necessary for speed.

Memory: the weights take 126.5 MB. The KV cache takes 18.9 MB for each slot
(`2 * 12 * 1,024 * 768` bytes). The work buffers take about 1 MB for each row. 224 slots use
4.8 GB of the 6.4 GB card (measured from the free memory before and after).

### 4.2 Prefill

All tokens of a block run in one pass (fold knows the tokens). Rows of the same slot are at
consecutive positions; the attention reads the keys that the pass wrote.

| Slots x tokens | Rows in a pass | Tokens/s |
|---|---|---|
| 1 x 512 | 512 | 22,355 |
| 8 x 512 | 512 | 16,018 |
| 8 x 1,024 | 1,024 | 12,012 |

### 4.3 Where the time goes

GPU time by op at batch 224, position 528 (GPU timer marks between the kernels, 30.7 ms in
total): attention 12.3, matrix products 7.6 (qkv 1.4, proj 0.7, fc 1.8, mlp 1.7, logits 2.0),
linear output 3.6, logits 2.6, norm 2.1, coder 1.1, others 1.3.

The steps that made the CUDA path faster (batch 128 or 192, positions 3-19, coder op):

| Change | Tokens/s |
|---|---|
| First version: mma.sync from global memory, one launch for each op | 7,997 |
| Shared-memory tiles with cp.async, split-K | 9,320 to 11,881 |
| CUDA graphs (one launch for each step), KV cache stored by head | 11,529 to 12,258 |
| Attention: scores in shared memory, exact reciprocal for p | 12,472 |
| Coder op: warps own chunks, warp scans, exact reciprocal quotient | 16,450 to 17,517 |
| Attention: byte planes of p with dp4a for the sum of p * v; i64 score rescale | 17,892 to 18,565 |

### 4.4 Other measured costs

- `taco_coder::FreqTable::from_weights` on the CPU: 142 µs for each table (devbox, 1 thread).
  At 18,000 tokens/s this needs 2.6 CPU cores; the coder op on the GPU takes 1.1 ms for 224
  tables.
- The CPU reference on the same machine: 2,643 steps in 109.9 s with 7 threads (loaded
  machine).

## 5. Why decode over a full window is below 10,000 tokens/s

At position `t`, each token reads the keys and values of `t` positions: `2 * 12 * t * 768`
bytes. At batch 224 and position 512 this is 2.1 GB for each step (est.). The memory of the
card gives about 288 to 336 GB/s (est.: 12 to 14 Gbps on 192 bits), so attention needs at
least 6.3 to 7.3 ms for each step (est.). The other ops take 18.4 ms now. Thus a full window
at batch 224 can reach about 9,000 tokens/s (est.) if attention reaches the memory limit.

Ways to 10,000 tokens/s over a full window (est., not measured):

1. Faster matrix products: they reach about 4 TMAC/s; the tensor cores permit more than 20.
2. Fuse the linear output into the matrix product, and the logit rescale into the weights op.
   Both are op-level now (`docs/design/anymodel.md`); a fused path needs its own bit-exact test.
3. A shorter context for each block (a phase 3 question): positions 0 to 511 halve the
   attention bytes.

For fold, prefill already passes the gate (12,012 to 22,355 tokens/s).

## 6. Proposed integration API (not wired into `crates/taco`)

The coder needs intervals, not weights. A proposed shape:

```rust
/// A predictor that runs many blocks in one pass and gives coder intervals.
pub trait IntervalPredictor {
    /// Runs `inputs` (slot, token); a slot can occur many times (prefill). `queries[i]` asks
    /// for the interval of a token (encode) or the symbol of a target (decode).
    fn run_coded(&mut self, inputs: &[(usize, TokenId)], queries: &[CoderQuery], freq_bits: u32)
        -> Result<Vec<CoderInterval>>;
    fn reset(&mut self, slot: usize);
}
```

`hardshell_cuda::CudaBatch` and `hardshell_gpu::GpuBatch` have this method now. A CPU version is
`hardshell::gpt2::Session` plus `FreqTable::from_weights`.

- **Fold.** For a group of blocks: one `run_coded` call with all tokens of each block (prefill)
  and `CoderQuery::Encode(next token)`. The encoder codes each interval.
- **Unfold.** Each block keeps its range decoder. For each step: each decoder gives its target
  value; one `run_coded` call with the last token of each block and `CoderQuery::Decode(target)`;
  each decoder takes its interval, and the symbol is the next token.
- `taco-coder` needs two small public functions for this: an encode with an interval
  (start, freq, is_last), and a decode in two parts (the target for `freq_bits`, then the
  update with an interval). The current `Encoder::encode` and `Decoder::decode` take a full
  `FreqTable`.
- Device choice: `hardshell_cuda::CudaDevice::open(0)`, else `hardshell_gpu::GpuDevice::open`,
  else the CPU. Each choice gives the same bytes. The container needs no flag for it.

## 7. Tuning of 2026-10-19 (kernels agent)

All numbers in this section are measured on devbox, RTX 3060 Laptop, on 2026-10-19 between
00:40 and 01:30 +04. "HEAD" is commit 74de31f. "Final" is the working tree with the changes
below. Each A/B pair ran back to back, with the same command (`gpt2 bench`, `gpt2 window`).

### 7.1 Conditions

- **GPU power limit: 45 W** (`nvidia-smi -q`: current power limit 45.00 W, default 80.00 W;
  "SW Power Cap" and "SW Thermal Slowdown" active). The laptop was on mains power (Windows:
  battery status 2, power online). Under load the GPU stayed at 44 to 45 W, and the SM clock
  fell to 300 to 500 MHz (mean of the busy samples, `nvidia-smi` every 0.5 s); the maximum is
  2,100 MHz. The memory clock stayed at 6,001 MHz. Idle power was 23 W.
- Memory bandwidth (`gpt2 bandwidth`, a streaming read): 274 to 280 GB/s.
- CPU load: 18 to 57 on 14 processors (other agents: `taco` benchmarks, HardShell evaluations).
  The timing runs used processors 0 to 5 and held `~/scratch/timing.lock`.
- No other process used the GPU during a timing run.
- Variation: about 2% between repeated runs of the same binary; the first run after a pause
  is faster (a cooler GPU).

### 7.2 Equivalence

The final tree gives the CPU weights bit for bit:

- `cargo test -p hardshell-cuda --release -- --include-ignored`: 22 of 22 pass (4 fused-path,
  14 GPT-2, 4 product and PTX tests). New tests: row-group attention with head size 64 and 128,
  runs of many slots in one pass, rows of more than 1,024 values in the norm.
- `gpt2 equiv` on `gpt2.hsm` (`aaa03eb9…a5c2`): 7 of 7 windows equal for fused decode, op-level
  decode and prefill (CPU SHA-256 values of section 2); 2,643 of 2,643 coder intervals equal.
- Golden tiny vector `d4063522…0078`: equal (decode and prefill).
- The PTX test: the committed PTX names the SHA-256 of the committed source.

### 7.3 Changes

| Change | Why it is exact |
|---|---|
| Fused kernels (previous agent): `linear_qkv` (linear output and KV store), `residual_norm` (linear output, add, norm), `linear_gelu`, logits product with compact u16 logits and `coder_d16` | The same expressions in one kernel; `tests/fused.rs` compares every weight and interval with the op-level path |
| Tensor-core tiles 64 x 128 and 128 x 128 with `ldmatrix` (previous agent) | Exact i32 sums; the tile order does not change an integer sum |
| A measured plan of kernel and split-K for each product shape (`ops::plan_product`, table of 7.5) | Each kernel and split gives the same exact sum |
| Decode attention keeps the i32 dots in shared memory (4 KB less); the row max is `max(score(max dot), score(min dot))` | The score is monotonic in the dot for each sign of the multiplier (the DETERMINISM note in the kernel) |
| `attend_rows`: prefill attention of groups of up to 16 rows of one slot, with `mma.sync` (s8 x s8 for the dots, u8 x s8 for the byte planes of p times the values), three passes over the keys | The same dots, row max, sum, p and plane sums as `attend`, as exact integer sums |
| The fused epilogues use an i64 rescale when \|x\| < 2^31 | Then \|x * m\| < 2^62, and the i64 form gives the i128 value |
| Norm: blocks of 128 threads (one wave for 224 rows), the row in registers, u64 forms of `isqrt` and of the variance division when the value is below 2^64 | The same functions; the product `centred * inv * (gamma * m)` is the same integer |

### 7.4 Results, HEAD against final

Tokens/s with the coder op. Two runs of each binary (in the order HEAD, final, HEAD, final, with
45 s pauses); the "full window" line also has 2 runs of `gpt2 window 224` of the same final binary.

| Measurement | HEAD | Final | Change |
|---|---|---|---|
| Decode batch 32, positions 3-19 | 11,291 to 11,500 | 8,124 to 12,368 | the low value is one slow run |
| Decode batch 32, positions 512-528 | 5,033 to 5,556 | 6,829 to 6,916 | +30% |
| Decode batch 128, positions 3-19 | 16,472 to 17,975 | 21,371 to 24,598 | +35% |
| Decode batch 128, positions 512-528 | 6,889 to 7,139 | 9,067 to 9,640 | +33% |
| Decode batch 224, positions 3-19 | 16,461 to 17,296 | 26,900 to 27,823 | +62% |
| Decode batch 224, positions 512-528 | 6,964 to 7,317 | 8,637 to 9,126 | +25% |
| **Decode batch 224, full window (0 to 1,023)** | **6,901 to 7,178** | **8,691 to 9,195** | **+27%** |
| Prefill 1 slot x 512 | 16,167 to 16,225 | 28,100 to 40,573 | +2.1x |
| Prefill 8 slots x 512 | 15,367 to 15,407 | 27,781 to 30,469 | +1.9x |
| Prefill 8 slots x 1,024 | 11,403 to 12,427 | 25,819 to 26,191 | +2.2x |

GPU time by op at batch 224, position 528 (ms, GPU timer marks, final tree): total 20.7 to 20.9;
attention 9.2 to 9.4, products 6.7 to 6.8 (qkv 1.3, proj 0.7, fc 1.5, mlp 1.3, logits 1.8 to
1.9), residual norm 1.8, linear qkv 0.8, linear gelu 0.8, coder 1.3. HEAD: total 28.9 to 31.5;
attention 11.5 to 12.4.

Prefill attention, 8 slots x 1,024 tokens: 214.8 ms (one block for each row, before
`attend_rows`) and 110 to 124 ms (`attend_rows`).

### 7.5 The plan of the products

`gpt2 gemm 32 64 128 224 256 512 1024`, three runs, median GPU time of each kernel and split
(A: 64 x 64 tiles, B: 64 x 128, C: 128 x 128; the number is the split-K factor). The runs used
zero operands, at 1,100 to 1,340 MHz, so the absolute times are lower than in a decode step.

| Rows | qkv | proj | fc | mlp |
|---|---|---|---|---|
| 32 | B2 14 µs | A2 13 | A1 19 | A4 19 |
| 64 | B2 21 | B2 14 | C1 19 | B4 20 |
| 128 | C2 29 | C3 18 | A1 26 | A2 26 |
| 224 | C1 36 | B1 20 | B1 33 | B2 33 |
| 512 | B1 56 | C1 26 | A1 64 | B1 56 |
| 1,024 | C1 97 | B1 40 | B1 126 | C1 99 |

The rule before (enough blocks for 60 slots, so 2 to 10 splits) took 25% to 40% more time at
128 and 224 rows. Split-K helps only when a product has few tiles and a long k (mlp). The
logits product keeps the 64 x 128 or 128 x 128 kernel, because the compact logits need tiles of
128 columns: 64 x 64 tiles are faster for the product alone (519 against 671 µs at 224 rows),
but the op-level logits path then costs about 2 ms more for each step (section 4.3).

Not kept: 12 instead of 8 blocks of `attend` on each SM (40 registers, no spill): 9,158 and
9,227 against 9,195 and 9,076 tokens/s over a full window; the difference is in the noise.

### 7.6 The gate

Full-window decode is 8,691 to 9,195 tokens/s at batch 224: **below 10,000**. At 45 W the GPU
is power-bound: a step at position 528 takes 20.7 ms of GPU time at about 450 MHz. Attention
reads 2.18 GB in 9.2 ms (237 GB/s, 85% of the measured read bandwidth). The other ops are
instruction-bound at the low clock.

Next steps (est., not measured):

1. Measure again at the default power limit (80 W). The SM clock is then not limited by power;
   the products, norms and the coder (11 ms of the 20.7 ms) are instruction-bound and become
   faster with the clock. This is a setting of the laptop (power mode), not of Taco.
2. Batch 256 (about 5.2 GB; the card has 5.37 GB free).
3. A 64 x 64 tile with the compact logits epilogue (about 0.15 ms for each step at 224 rows
   at the survey clock).
4. Skip the zero byte planes of p in decode attention: most p are below 2^24, so plane 3 is
   zero for most keys.
