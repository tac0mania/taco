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
