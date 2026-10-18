# Taco version 1: the production plan

Status: draft 1, 2026-10-19. Owner: Rohit. This plan obeys `docs/STYLE.md`. ADR 0012 defines
the two profiles (`text` and `logs`).

## §1 What "ready for production" means

| Area | Requirement |
|---|---|
| Correctness | Every file restores byte for byte. The checksum catches every error. Fuzz tests run on the container, the coder and the decoder. |
| Bit-exact | The same `.taco` file on every supported machine and backend (CPU, Apple GPU, NVIDIA, AMD, Intel). A conformance suite proves it before each release. |
| Speed | A measured gate for each backend (§3). |
| Machines | macOS (ARM, x86), Linux (x86, ARM), Windows (x86). The user installs nothing except a normal GPU driver. |
| Large files | Streaming: memory does not grow with the file size. Many blocks in parallel. |
| Format | Container version 2 frozen. Every later version reads version 2. |
| Models | `text` and `logs` model files, with hashes. A model file is never changed after release. The converter is exact, so anyone can make the identical file again from the published inputs (ADR 0014). |
| Usability | Clear errors, progress output, `taco --version`, man page, README with the measured benchmark. |
| Release | Signed binaries for each platform, a changelog, a licence. |

## §2 Work items

| # | Item | Depends on | Estimate (agent time, est.) |
|---|---|---|---|
| 1 | CPU speed: SIMD, prefill, batch (running) | — | hours |
| 2 | Speed gate: CUDA on the RTX 3060 (running) | — | 1 to 2 days |
| 3 | Portable GPU (wgpu): AMD, Intel, Apple; driver bug on the AMD iGPU (running) | — | 2 to 4 days |
| 4 | Logs: match model and mixer (running) | — | 1 day |
| 5 | Route A: a log model (running) | — | 1 to 2 days |
| 6 | Skip rule and speculative decoding | 4 | 2 to 3 days |
| 7 | zstd block kind (running) | — | 1 day |
| 8 | Container version 2: profile field, kind 2, match model, frozen | 4, 6, 7 | 1 day |
| 9 | Backend selection at run time: CUDA, then wgpu, then CPU SIMD; all conformance-tested | 1, 2, 3 | 1 to 2 days |
| 10 | Streaming of large files, progress, limits | 8 | 1 day |
| 11 | Profiles in the CLI (`--profile text|logs|auto`), model files found by hash | 5, 8 | 1 day |
| 12 | Builds and tests on all platforms (CI), cross-machine check on 3 or more machines | 9 | 2 days |
| 13 | Packaging: binaries, model distribution, docs, benchmark page | 12 | 2 days |
| 14 | **Exact model conversion (ADR 0014, owner requirement)**: weights, scales and tables in exact arithmetic; the same model file on every machine; self-check of the hash | — | 3 to 6 hours |

Total: about 3 to 4 weeks of agent time, with several items in parallel (est.).

## §3 Speed gates (targets, est.; each is measured before release)

| Backend | `text` fold | `text` unfold | `logs` fold | `logs` unfold |
|---|---|---|---|---|
| CUDA (RTX 3060) | ≥ 0.5 MB/s | ≥ 0.2 MB/s | ≥ 5 MB/s | ≥ 2 MB/s |
| wgpu (Apple, AMD, Intel GPU) | measured | measured | measured | measured |
| CPU SIMD (8 cores) | ≥ 20 KB/s | ≥ 8 KB/s | ≥ 100 KB/s | ≥ 50 KB/s |

The `logs` targets depend on items 5 and 6. If a gate fails, the release notes say so, and the
team decides before the release.

## §4 Owner decisions (2026-10-19)

1. **Repository and licence.** A private GitHub repository. Licence: Apache-2.0 (`LICENSE`).
2. **Model distribution.** Rohit left it to the lead. Decision: `taco` finds a model file by its
   hash in a local model directory. If the file is not there, it downloads it once from a fixed
   URL and checks the SHA-256 before use. The release also offers a full archive with the model
   files inside, for machines without network access. The download URL is a release decision:
   a private repository cannot serve public downloads.
3. **Platforms for version 1.** All of §1 where possible: macOS (ARM, x86), Linux (x86, ARM),
   Windows (x86).
4. **Windows.** Native Windows builds and tests on the Windows side of devbox are permitted.
