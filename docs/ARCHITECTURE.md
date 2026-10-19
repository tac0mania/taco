# Taco: architecture

This document lists the parts of Taco and the permitted dependencies. The design is in
`docs/design/plan.md`. The rules are in `CLAUDE.md`.

## Crates

A crate may depend only on crates above it in this table.

| Crate | Job | Kernel crate | Plan |
|---|---|---|---|
| `taco-coder` | Range coder, binary coder, frequency tables. Integers in, bytes out. | Yes | §5 |
| `hardshell` | Integer inference engine. Model-file reader. Integer helpers (`int`). The reference path (the oracle, ADR 0013) and the CPU fast paths (NEON, AVX2). | Yes | §7, §8 |
| `taco-match` | The match model and the logistic mixer of the profile `logs` (ADR 0010). | Yes | §6 |
| `hardshell-gpu` | The portable GPU backend: wgpu and WGSL (Vulkan, Metal, DX12, OpenGL). Gives the bits of `hardshell` (ADR 0008). | No | §7 |
| `hardshell-cuda` | The NVIDIA backend: CUDA kernels as precompiled PTX, the driver loaded at run time. Gives the bits of `hardshell` (ADR 0008). | No | §7 |
| `taco-format` | The `.taco` container: header, block table, checksum, raw fallback, zstd blocks. | No | §9 |
| `taco` | The library. Tokenizer, model, coder and container together. | No | §4, §10 |
| `taco-cli` | The `taco` command (`fold`, `unfold`). No logic. | No | — |

`taco-coder` and `hardshell` do not depend on each other. The `taco` crate connects them.
`taco-match` depends on no crate of the workspace. `hardshell-gpu` and `hardshell-cuda`
depend on `hardshell` only. `taco` selects a backend at run time (ADR 0016).

A kernel crate obeys these rules:

- No floats in `src/` (`#![deny(clippy::float_arithmetic)]` and a check in `scripts/check.sh`).
- No I/O except reads from a byte slice or a reader that the caller gives.
- No clock, no randomness, no global state, no threads that change results.

## Permitted dependencies

The root `Cargo.toml` holds the versions (`[workspace.dependencies]`). This table lists the
external crates that a crate may use. It does not list the crates of this workspace.

| Crate | Dependency | Licence | Why | Notes |
|---|---|---|---|---|
| `taco-coder`, `hardshell`, `taco-match`, `hardshell-gpu`, `hardshell-cuda`, `taco`, `taco-format` | `thiserror` | MIT or Apache-2.0 | Error enum | |
| `hardshell` | `sha2` | MIT or Apache-2.0 | The model hash | |
| `hardshell` | `serde`, `serde_json` | MIT or Apache-2.0 | The JSON header of the `.hsm` file | Not in the prediction path: header parse only. |
| `hardshell-gpu` | `wgpu` (30.0.1; features `vulkan`, `metal`, `dx12`, `gles`, `wgsl`) | MIT or Apache-2.0 | The portable GPU backend | The output must equal the CPU bits (ADR 0013). |
| `hardshell-cuda` | `libloading` (0.8) | ISC | Loads the CUDA driver (`libcuda`) at run time | No CUDA toolkit at build or run time. The PTX is committed. |
| `taco-format` | `sha2` | MIT or Apache-2.0 | Model hash and checksum | |
| `taco-format` | `zstd-safe` (`=8.2.0`) | BSD-3-Clause | The zstd block kind (ADR 0011) | Exact version: the container hashes depend on it. |
| `taco-format` | `zstd-sys` (`=2.1.2`) | BSD-3-Clause (the crate and the bundled zstd 1.5.7 C source) | Pins the zstd C library to 1.5.7 (ADR 0011 §6) | The C source is bundled and built with `cc`. No system package. A build needs a C compiler. No default features. |

## Tools

`tools/` is a Python project (`uv`). The tools run before publication or for measurements. They
may use floats. Nothing in `crates/` calls a tool at run time.

| Tool | Job | Plan |
|---|---|---|
| `tools/baseline` | Float GPT-2 in NumPy. Compression sizes. Float drift between machines. | §12 phases 0, 1 |
| `tools/convert` | Float weights to a `.hsm` model file. Calibration (recipe `clip`). Tables. Merges. | §8 |
| `tools/convert/tiny.py` | The tiny conformance model from a fixed seed. | §11.1 |
| `tools/convert/exact.py` | The exact conversion with rational arithmetic, and the self-check (ADR 0014). | §8 |
| `tools/finetune` | Route A: fine-tune distilgpt2 on logs (PyTorch, tools only). | §12 |
| `tools/logs` | Log data, the logs measurements, the CUDA dump tool. | §12 |
| `tools/bench` | The benchmark against other compressors. | §12 |
| `tools/longctx` | Long-context and context-overlap measurements. | §12 |
| `tools/cuda` | `build_ptx.py`: builds the committed PTX with NVRTC (build time only). | §7 |

## Other directories

| Directory | Contents |
|---|---|
| `conformance/` | Conformance vectors and the tiny model (§11). |
| `scripts/` | `dx` (devbox), `mx` (Mac), `check.sh`, `check-tags.sh`, `check-no-floats.sh`, `xcheck` (Mac and devbox), `wcheck` (native Windows). |
| `docs/adr/` | Decisions. |
| `docs/research/` | Research notes and measurements. |
| `.agent-status/` | Agent status files. Not committed. |
