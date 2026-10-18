# ADR 0008: GPU backends

Date: 2026-10-18. Status: proposed (by the cuda agent). The owner requirements of 2026-10-18
are in the context.

## Context

Plan §12 phase 4 asks for a GPU path. CLAUDE.md §1 rule 10 permits a GPU path only if it gives
the same bits as the CPU reference path. A file that a Mac folds must unfold on a PC with a GPU,
and the reverse.

The owner set three requirements on 2026-10-18:

1. A user installs nothing by hand. Only the GPU driver that comes with the GPU is permitted.
   No CUDA toolkit, no NVRTC, no ROCm.
2. The GPU path must work on AMD GPUs too, and if possible on Intel GPUs and the Apple GPU.
3. The GPU path must be fast. The speed gate for NVIDIA is 10,000 tokens/s of GPT-2 small on
   the RTX 3060 of devbox.

Facts that we measured on 2026-10-18:

- devbox (WSL2) has no hardware Vulkan driver. Vulkan finds only llvmpipe (a CPU driver).
  OpenGL through the Mesa D3D12 driver finds the NVIDIA RTX 3060 and the AMD Radeon iGPU, if
  `GALLIUM_DRIVER=d3d12` is set. The NVIDIA driver puts `libcuda.so.1` in `/usr/lib/wsl/lib`.
- The Mac (M2) has Metal.
- OpenGL on D3D12 gives no access to the int8 tensor cores. The portable path reached 1,166
  tokens/s at batch 128 (measured, `docs/research/2026-10-18-gpu-backend.md`).

## Decision

### 1. Two backends with one op set

| Crate | GPUs | API | Status |
|---|---|---|---|
| `crates/hardshell-gpu` | NVIDIA, AMD, Intel, Apple | wgpu 30 (Vulkan, Metal, DX12, OpenGL), WGSL shaders | portable path |
| `crates/hardshell-cuda` | NVIDIA, compute capability 8.0 or more (Ampere and later) | CUDA driver API, PTX | fast path |

Both backends implement the same ops as kernels: `embed`, `norm`, `linear` (matrix product and
output rescale), `add`, `table`, `attention`, `logits`, and the coder interval. The host
composes the ops into the GPT-2 step. This is the op set of `docs/design/anymodel.md` §4. A
fused kernel is permitted later, if a test proves it bit-exact against the op-level path.

### 2. What the user installs

| Platform | Backend | The user needs |
|---|---|---|
| Linux, Windows with NVIDIA | `hardshell-cuda` | the NVIDIA driver (it contains `libcuda`) |
| Linux with AMD or Intel | `hardshell-gpu` (Vulkan) | the Mesa or vendor Vulkan driver, part of the GPU driver |
| Windows with AMD or Intel | `hardshell-gpu` (DX12 or Vulkan) | the GPU driver |
| macOS | `hardshell-gpu` (Metal) | nothing (Metal is part of macOS) |
| WSL2 | `hardshell-cuda`, or `hardshell-gpu` (OpenGL on D3D12) | the Windows GPU driver; OpenGL needs `GALLIUM_DRIVER=d3d12` |

No user installs a toolkit, a compiler or a runtime library.

### 3. Toolchain

- **wgpu.** `wgpu` 30.0.1 (MIT or Apache-2.0). wgpu translates WGSL to the shader language of
  the backend when the program runs. The crate is pure Rust and has no `unsafe` code of ours.
- **CUDA kernels.** The kernels are CUDA C in `crates/hardshell-cuda/kernels/hardshell.cu`.
  `tools/cuda/build_ptx.py` compiles them to PTX for `compute_80` with NVRTC. The PTX is
  committed in `crates/hardshell-cuda/kernels/hardshell.ptx` and the crate embeds it. At run
  time the NVIDIA driver compiles the PTX for the GPU. A test refuses a PTX whose source hash is
  not the hash of the committed source.
- **NVRTC.** Only the developer who changes a kernel runs NVRTC. It comes from the official PyPI
  wheel `nvidia-cuda-nvrtc-cu12==12.9.86`, installed in `~/.cache/taco/cuda/` on the build
  machine. Its licence is the NVIDIA proprietary licence (LicenseRef-NVIDIA-Proprietary). Taco
  does not distribute NVRTC; it distributes only the PTX that NVRTC made from our source.
- **Driver access.** `libloading` 0.8 (ISC) loads `libcuda` when the program runs. Thus the
  crate builds on every machine, also with no NVIDIA driver; there it gives `Error::NoDevice`.
  All `unsafe` code is in `crates/hardshell-cuda/src/driver.rs`, with a `SAFETY:` comment on
  every block.

### 4. Integer rules on the GPU

- **No floats.** No shader and no kernel uses a float type. Tests scan the WGSL sources and the
  PTX for float types.
- **WGSL.** WGSL has only 32-bit integers. The shaders emulate the `i64` and `i128` rules of
  `hardshell::int` with `u32` words. WGSL defines wrapping `u32` arithmetic on every device. The
  shaders use `i32` arithmetic only where a bound excludes overflow.
- **CUDA.** The kernels use native 64-bit integers and `__int128`. Signed overflow is undefined
  in C++, so each signed operation has a bound (`INVARIANT:`) that excludes it. A shift of a
  negative value uses unsigned operations.
- **Matrix products.** `dp4a` and `mma.sync.m16n8k32.s32.s8.s8.s32` give exact sums of `i8`
  products. Every sum is below `2^30` in magnitude (`hardshell::kernels::MAX_DOT_LEN_I32`), so
  no accumulator overflows. Integer addition is associative, so the tile order and the split-K
  order do not change a result. Split-K partial sums go to separate buffers, and the output op
  adds them. No kernel uses atomics.
- **Divisions.** A division is either an exact integer division, or a reciprocal estimate with
  one exact correction step. The kernel comment proves that the estimate is the quotient or the
  quotient minus 1.

### 5. Proof of the same bits

The CPU reference hashes are the contract (plan §11). A GPU backend is correct on a GPU only if
these tests pass on that GPU:

1. Each integer helper and the matrix product against `hardshell::int` and `hardshell::kernels`,
   on random shapes and extreme values (the accumulator bound, all-`-128` rows).
2. The golden HardShell vector: model `e3055785…7bb9`, output
   `d4063522517854fdcab605c3d3159bd3ac9a15eb92fa6721ff4a276658f90078`.
3. `conformance/tiny.hsm` at every step, with three context restarts.
4. Batches and prefill against independent CPU sessions, with restarts.
5. The coder interval against `taco_coder::FreqTable::from_weights`.
6. For a release: `gpt2.hsm` (`aaa03eb9…a5c2`) on real text windows, with the SHA-256 of all
   weights of each window equal to the CPU hash (`examples/gpt2.rs equiv`).

Tested on 2026-10-18 (measured):

| GPU | Backend | Tests 1 to 5 | Test 6 |
|---|---|---|---|
| Apple M2 | wgpu, Metal | pass | not run |
| NVIDIA RTX 3060 Laptop | wgpu, OpenGL on D3D12 | pass | 7 of 7 windows equal |
| llvmpipe (x86 CPU) | wgpu, Vulkan | pass | not run |
| AMD Radeon iGPU (Ryzen 7 5800H) | wgpu, OpenGL 4.2 on D3D12 | fail: the process crashes inside the Mesa driver at the first compute dispatch | not run |
| NVIDIA RTX 3060 Laptop | CUDA | pass | 7 of 7 windows equal, decode and prefill |

### 6. The read back

The coder needs one interval (start, frequency) of the frequency table for each token. The
coder op makes the table of `taco_coder::FreqTable::from_weights` on the GPU and gives only the
interval: 16 bytes (wgpu) or 24 bytes (CUDA) for each token, not 402 KB of weights. For the
decoder, the op takes the target value of the range decoder and gives the symbol and its
interval.

## Consequences

- A GPU path that is not on the list in section 5 is not trusted. A new GPU type runs tests 1
  to 5 before Taco uses it.
- A change to a kernel changes no compressed byte, because the kernels must give the CPU bits.
  A change to the CPU rules (ADR 0004) needs the same change in both backends.
- The CUDA path needs compute capability 8.0 or more (`mma.sync` m16n8k32). Older NVIDIA GPUs
  use `hardshell-gpu`.
- AMD is covered only by the portable path, and no test passed on AMD hardware yet: the only
  AMD GPU (the iGPU of devbox, through Mesa OpenGL on D3D12 in WSL2) crashes at the first
  compute dispatch. AMD needs a test on a native Vulkan driver (RADV or the AMD driver) before
  Taco can call AMD supported.
- `taco-coder` needs an encode and a decode with an interval, not a table, to use the coder
  op. That is a change to `taco-coder` and `taco` (not part of this ADR).
