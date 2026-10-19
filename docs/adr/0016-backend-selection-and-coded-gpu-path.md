# ADR 0016: backend selection and the coded GPU path of fold and unfold

Date: 2026-10-19. Status: proposed (by the gpucli agent). Plan: §4, §5, §12 phase 4. Related:
ADR 0003 (coder), ADR 0007 (coded block), ADR 0008 (GPU backends), ADR 0010 (match model and
mixer), ADR 0013 (fast paths), ADR 0015 (context overlap). Product plan item 9.

## Context

`hardshell-cuda` and `hardshell-gpu` give the CPU weights bit for bit (ADR 0008). Before this
ADR, `taco fold` and `taco unfold` used only the CPU. The research note of 2026-10-18 (section 6)
proposed the integration: the GPU coder op gives the coder interval of each token, so only 24
bytes for each token come back, not 402 KB of weights.

The coder needs a small interface for this: an encode of an interval, and a decode in two parts
(the target, then the update). ADR 0003 must stay valid: the bytes cannot change.

## Decision

### 1. Selection (`--backend auto|cuda|gpu|cpu`)

1. `taco fold` and `taco unfold` take `--backend`. The default is `auto`.
2. `auto` tries the backends in this order and uses the first that works:
   1. CUDA: the driver library loads, device 0 has compute capability 8.0 or more, the model
      loads into GPU memory, and the self-test passes.
   2. wgpu: the best hardware adapter (discrete, then integrated, never a CPU adapter) is a
      discrete GPU, has compute shaders, the model loads, and the self-test passes.
   3. The CPU.

   `auto` does not use an integrated GPU, because the portable path has no prefill and the CPU
   fold is faster there (measured on the Apple M2, section "Measurements"). `--backend gpu`
   uses any hardware adapter.
3. An explicit `cuda` or `gpu` that does not work is an error (`TacoError::Backend`). `auto`
   writes the reasons in `Engine::notes`. The CLI prints them when it falls back to the CPU.
4. **The self-test** runs on the loaded model before each use of a GPU: 6 steps of a block on
   the GPU and on the CPU session. The full weights of step 1 must be equal, and the coder
   answers of every step (encode) and of three targets (decode: the first, one inside, the
   last) must be equal. A GPU that fails is not used. The test costs 6 CPU steps (about 0.1 s
   for GPT-2 small, est.).
5. The container does not record the backend. Every backend reads every container.
6. The binary builds and runs on every machine: `hardshell-cuda` loads `libcuda` at run time,
   and wgpu finds the drivers at run time.

The library API: `taco::with_engine(choice, model_file, model, |engine| ...)` opens the backend
and gives an `Engine`. `taco::fold_with(engine, ...)` and `taco::unfold_with(engine, ...)` run
on it. `taco::fold` and `taco::unfold` are the CPU engine. `FoldStats::backend` names the
backend that ran.

### 2. The coder interface (`taco-coder`)

1. `CodeInterval::new(start, freq, bits)` holds `[start, start + freq)` of a table with the
   total `T = 2^bits`. It refuses `freq = 0` and `start + freq > T`.
2. The symbol is the last symbol of its table exactly when `start + freq = T`. Reason: every
   start of a valid table is less than `T`, so a symbol that is not the last ends below `T`.
   Thus the three values are sufficient for the rule "the last symbol receives the remainder".
3. `Encoder::encode_interval(iv)` does the encoder steps of ADR 0003 with these values.
4. `Decoder::target(bits)` gives `t = min(code / r, T - 1)` and changes nothing.
   `Decoder::decode_interval(iv)` refuses an interval that does not hold `t`
   (`TargetNotInInterval`, the state does not change), else does decoder steps 4 to 6.
5. `FreqTable::code_interval(symbol)` and `FreqTable::symbol_at(target)` give the intervals of
   a table, for tests and for the self-test.

`crates/taco-coder/tests/interval.rs` proves that the interval path gives the bytes and the
decoder states of the table path: property tests (random weights with 0 and `u64::MAX`, 9 to
32 bits, the last symbol in a third of the cases) and 2,040 symbols of 50,257-symbol tables.
The golden coder vector `91676cdc…798f` does not change.

### 3. The coded GPU path of fold (profile `text`; `logs` and `auto` see section 5)

1. The CPU threads tokenize each block, check the tokens, and compress it with zstd.
2. Each block is divided into its context windows (ADR 0015 section 2 with `K = 0`): window
   0 holds tokens `0..n_ctx`. Window `w >= 1` starts with a new context (`<|endoftext|>`,
   whose weights are not used), then the token before its first token. It holds `n_ctx - 1`
   tokens. This is the restart rule of ADR 0004 section 9.
3. The windows run in parallel slots (64 at most). The CUDA backend runs all rows of a window in
   one call (prefill, passes of 512 rows). The wgpu backend runs one row of each slot in each
   step (it has no prefill).
4. The coder op answers `Encode(token)` with the interval of the token. The encoder of each
   block takes the intervals of its windows in token order.
5. The stop rule of ADR 0007 §5.2 applies as on the CPU. When both candidates of a block
   stopped, the windows of the block that did not start are skipped.

Update of 2026-10-19 (the overlap agent, ADR 0015): the windows have the overlap `K` of the
container: `<|endoftext|>`, the `K` kept tokens, and `n_ctx - 1 - K` new tokens
(`crates/taco/src/gpu_fold.rs`). CUDA and wgpu both support every `K`; unfold uses the overlap
of the backend batch (`CudaBatch::set_overlap`, `GpuBatch::with_overlap`), and the backend
starts a full context again with the last `K` tokens.

### 4. The coded GPU path of unfold

1. Raw and zstd blocks decode on the CPU threads.
2. Each coded block decodes in one slot (a mixed block too, section 5). All slots step
   together, one token for each slot.
3. For each step: each decoder gives its target (`Decoder::target`). The GPU runs the last
   token of each slot, and the coder op answers `Decode(target)` with the token whose interval
   holds the target. Each decoder takes the interval (`Decoder::decode_interval`).
4. The backend starts the context again when it is full, as the CPU session does.

### 5. The profile `logs`

The mixer changes the weights before the table: the predicted token gets the probability `q`,
and `q` depends on `W = sum(w)`, `w[p]` and the agreement bit `p = argmax(w)`, all from the
weights of that row (ADR 0010 §3). Thus the coder op of ADR 0008 cannot give the mixed interval.

**Decision for now (measured, see "Measurements"):**

1. **Fold.** The profiles `logs` and `auto` fold on the CPU on every engine
   (`FoldOptions::uses_gpu` is false). `FoldStats::backend` says so, and the CLI does not open
   a GPU for them. Reason: the encoder needs all weights of
   each token. On devbox the read back of the weights costs about 100 times the coder op, and
   the CUDA fold of 128 KiB of `apache_access.log` took 163.0 s against 141.4 s on the CPU.
2. **Unfold.** A mixed block decodes on the GPU: the GPU gives the full weights of each step,
   and CPU threads run the mixer, the table and the decoder (`BlockDecoder::next`, the code of
   the CPU path). The same unfold took 136.5 s with CUDA against 606.8 s on the CPU, because
   the CPU unfold has no prefill.

Both forms are exact by construction: they are the CPU code with the weights of the GPU.

**The exact GPU path (proposed, not built).** The read back costs about 100 times the coder op
(section "Measurements"), so the logs profile needs the mixer on the GPU. Two forms are exact:

1. **Two round trips.** A stats op after the weights op gives, for each row, `W` (u128),
   `w[p]` for the match prediction `p` of the row, and the agreement bit: 25 bytes for each row.
   The host runs the mixer from these values only (`pg`, the agreement bit, `q`, and the
   update after the token need no other weight; the match prediction depends only on the
   tokens). In fold, the host runs the mixer over all rows of a pass in order after one read
   back. A mixed coder op then makes `from_weights` of `w'[i] = max(1, (w[i] * m) >> k)` for
   `i != p` and `w'[p] = max(1, (R * q) >> k)` (ADR 0010 §3 rule 5), with `R = W - w[p]` and
   `k = max(0, bitlen(R) - 46)` on the GPU, and answers `Encode` or `Decode`.
2. **One pass.** The host gives each row its prediction `p` and the length bucket. A mixer
   kernel with one thread for each slot walks the rows of the slot in order: it reads the stats
   of the row, computes `q` with the `squash` and `stretch` tables of `taco-match` (uploaded as
   data), and updates the weight sets and `hit[b]` of the slot (about 1 KB of state for each
   slot). The mixed coder op follows. No host round trip is added.

Both forms need a step between the forward pass and the read back of `CudaBatch::run` (and of
`GpuBatch::step_coded`), so the weights stay on the GPU. That is a change to
`crates/hardshell-cuda/src/batch.rs`, `kernels/hardshell.cu` and its PTX, and to
`crates/hardshell-gpu`. The owners of those files make it. A test must compare every mixed
interval with `FreqTable::from_weights(MixState::mix(w))`. The `coder_body` kernel already
reads each weight through `WeightSrc::at`, so a mixed form of `at` and a stats pass are a
small change to it (est.). The mixed weights are below 2^62 and their sum is below 2^63 (ADR
0010 §3), so the u64 sum of `coder_body` stays exact. Its bound comment (weights of at most
2^40) and the quotient helpers must be checked for 62-bit weights.

### 6. Proof

`crates/taco/tests/backends.rs` runs on each GPU backend of the machine and skips a backend
that the machine does not have:

- 10 cases (empty, one byte, English with blocks of 7 and 700, a mixed file of 40,000 bytes,
  log lines, random bytes; profiles text, logs, auto), with 1 and 4 threads: the container of
  the GPU equals the CPU container byte for byte; the GPU unfolds the CPU container; the CPU
  unfolds the GPU container.
- The golden containers `0b0ba3f1…7c07` (text) and `7262257e…55ec` (logs) come out of each
  GPU backend.
- A GPU refuses a model file whose hash is not the hash of the loaded model.

Results (measured, 2026-10-19):

| Machine | Backend | `tests/backends.rs` | Golden `0b0ba3f1…`, `7262257e…` |
|---|---|---|---|
| devbox, RTX 3060 Laptop | CUDA | 4 of 4 pass | equal |
| devbox, RTX 3060 Laptop | wgpu, OpenGL on D3D12 (`GALLIUM_DRIVER=d3d12`, `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`) | 4 of 4 pass | equal |
| Mac, Apple M2 | wgpu, Metal | 4 of 4 pass | equal |

The 11 inputs of `scripts/xcheck` with `gpt2.hsm` (`aaa03eb9…a5c2`), `taco fold --profile
text|logs` (logs for the two `.logs.` inputs), 6 threads:

- devbox: the 11 `.taco` files of `--backend cuda` equal the files of `--backend cpu`, byte for
  byte. `taco unfold` restores all 11 inputs from them with `--backend cpu` and with
  `--backend cuda`.
- Mac: `--backend gpu` (Metal) gives the devbox CPU file for each of the 11 inputs, byte for
  byte, and unfolds each of them on Metal. Thus Metal (ARM) equals CUDA and the CPU (x86).

## Measurements

### Speed of `taco fold` and `taco unfold` (end to end)

Measured on 2026-10-19 on devbox: RTX 3060 Laptop at its **45 W power limit** (default 80 W;
the SM clock falls to 300 to 500 MHz under load, research note 2026-10-18 section 7.1), CUDA
and wgpu on OpenGL on D3D12. `gpt2.hsm` (`aaa03eb9…a5c2`), blocks of 16 KiB, `--threads 6`,
`taskset -c 0-5`, the timing lock held. Other agents loaded the CPUs (load average 19 to 34 on
14 processors); no other process used the GPU. The times include the model load (0.4 to 0.9
s), and for a GPU the device start, the upload and the self-test (about 2.5 s together, est.
from the empty input: 3.0 s with CUDA, 0.8 s on the CPU). KB is 1,024 bytes. Each `.taco` file
is the same on the three backends (SHA-256 prefix in the table), and each unfold restored the
input.

| Input | Profile | CUDA fold | CUDA unfold | wgpu (GL) fold | wgpu (GL) unfold | CPU fold | CPU unfold | `.taco` |
|---|---|---|---|---|---|---|---|---|
| alice29.txt, 152,089 B | text | 5.3 s, **28.1 KB/s** | 17.2 s, **8.6 KB/s** | 79.1 s, 1.9 KB/s | 182.1 s, 0.82 KB/s | 94.4 s, 1.6 KB/s | 393.3 s, 0.38 KB/s | `86febe60…` |
| cp.html, 24,603 B | text | 4.0 s, 6.0 KB/s | 18.7 s, 1.3 KB/s | 47.4 s, 0.51 KB/s | 202.8 s, 0.12 KB/s | 89.4 s, 0.27 KB/s | 434.0 s, 0.055 KB/s | `1e7fd9d7…` |
| apache_access.log, first 128 KiB | logs | 163.0 s, 0.79 KB/s (read back; now the CPU path) | 136.5 s, 0.94 KB/s | 213.3 s, 0.60 KB/s (read back; now the CPU path) | 188.3 s, 0.68 KB/s | 141.4 s, 0.91 KB/s | 606.8 s, 0.21 KB/s | `93cd10ff…` |
| apache_access.log, first 128 KiB | text | 5.9 s, 21.8 KB/s | 30.5 s, 4.2 KB/s | 60.7 s, 2.1 KB/s | 125.6 s, 1.0 KB/s | 121.6 s, 1.1 KB/s | 816.6 s, 0.16 KB/s | `b660d07e…` |

The first 1 MiB of `apache_access.log` (1,048,576 B, 64 blocks), same conditions, CUDA engine
only (the CPU times of this input are not measured; est. 8 times the 128 KiB times):

| Profile | Fold | Unfold | `.taco` |
|---|---|---|---|
| text | 24.5 s, **41.8 KB/s** (CUDA) | 58.9 s, **17.4 KB/s** (CUDA) | 124,564 B, `dbecfc53…` |
| logs | 1,206.2 s, 0.85 KB/s (on the CPU, this decision) | 1,019.7 s, 1.00 KB/s (CUDA, weights read back) | 90,780 B, `8eacb814…` |

Notes:

- Unfold has one slot for each coded block. cp.html has 2 blocks, so the CUDA unfold runs 2
  slots and is far from the decode rate of the backend (research note: 2,707 tokens/s at 8
  slots, 18,565 at 128 slots).
- The CPU unfold of the profile `text` (no prefill, 8 lanes for each thread) is 4 to 7 times
  slower than the CPU fold.

### The read back against the coder op (CUDA)

`CudaBatch` with 256 rows of `gpt2.hsm` in one call, 3 rounds, devbox (measured, a temporary
probe, removed):

| Call | Time |
|---|---|
| `run_coded` (24 bytes for each row back) | 5.4 to 8.4 ms |
| `run_without_readback` | 6.1 to 6.7 ms |
| `run_weights` (102 MB back) | 449 to 601 ms (about 0.2 GB/s) |

The read back uses `cuMemcpyDtoHAsync` into pageable memory (`driver.rs`). Pinned host memory
(`cuMemAllocHost`) is a possible improvement for the owner of `hardshell-cuda` (not measured).

### The Apple M2 (wgpu, Metal)

The Mac had memory pressure (6.5 GB of swap used by other programs), so these times are an
upper bound. 4,096 bytes of `lcet10.txt`: fold 25.0 s with Metal, 7.1 s on the CPU (6
threads). 32,768 bytes of `alice29.txt`: fold 591 s and unfold 119 s with Metal. The portable
path has no prefill, so `auto` does not use an integrated GPU (section 1).

## Consequences

- A change to a kernel or to the coder op must keep the self-test and `tests/backends.rs`
  green. A wrong GPU makes a wrong container only if it passes the self-test and still differs
  later; the checksum of the container then refuses the result at unfold.
- The self-test cannot catch a driver that crashes the process. Measured: the Mesa D3D12 driver
  on the AMD Radeon iGPU of devbox (wgpu, OpenGL, with `GALLIUM_DRIVER=d3d12`) stops the
  process at the first compute dispatch (ADR 0008). Without `GALLIUM_DRIVER`, wgpu on devbox
  finds only llvmpipe, which `auto` never selects.
- **The default profile `auto` codes as `logs`, so `taco fold` with no `--profile` folds on
  the CPU on every engine** until the GPU mixer exists (section 5). Only `--profile text`
  folds on the GPU. Unfold uses the GPU for every coded and mixed block.
- The logs profile on a GPU is limited by the read back of the weights and by the CPU work of
  the mixer and two tables (section 5). The exact GPU path removes both.
- Unfold has one slot for each coded block. A file of few blocks has few slots, so a small file
  does not fill the GPU.
