# Taco: session handoff

Written 2026-10-18 at about 15:00 +04, updated at about 16:30 +04 by the lead session (Claude). A new session or an agent
starts with no conversation context. This file is the context. Read it once from top to bottom.

- **Lead (a new session):** read sections 1 to 6. Then do section 5 in order.
- **Agent:** read sections 1, 2 and 6, then your brief. Your brief overrides this file.

## 1. What Taco is

Taco is a lossless compressor. GPT-2 small predicts the next token. A range coder stores only
the part that the model did not predict. HardShell, the engine, uses only integer arithmetic, so
every machine computes the same predictions bit for bit. Read `CLAUDE.md` (the rules) and
`docs/design/plan.md` (the design, with § numbers).

Owner decisions (ADR 0001, ADR 0005): the names are Taco, HardShell and `.taco`. GPT-2 small is
the only model of version 1. Nanbeige is not in the plan. Documents use ASD-STE100
(`docs/STYLE.md`). Commit after each small step that has a meaning.

## 2. Where things are

| Part | State | Tests (measured, Mac) |
|---|---|---|
| `crates/taco-coder` | Done. Range coder, binary coder, frequency tables (ADR 0003). | 47 pass |
| `crates/hardshell` | Done. Integer kernels, LayerNorm, GELU, softmax, `.hsm` reader and writer, integer GPT-2 with KV cache (ADR 0004). | 61 pass |
| `crates/taco-format` | Done. `.taco` container version 1 (ADR 0006). | 54 pass |
| `tools/baseline` | Done. NumPy GPT-2, Python BPE, Python coder, phase 0 and 1A scripts. | BPE 6,284 and coder 600 round trips |
| `tools/convert` | Done. GPT-2 to `gpt2.hsm` (recipe `clip`, beta correction), merges section, `conformance/tiny.hsm`. | converter hash repeats |
| `crates/taco` | Done. Tokenizer (equal to Python on 4,926,962 tokens), fold and unfold, ADR 0007. | 51 pass |
| `crates/taco-cli` | Done. `taco fold` and `taco unfold`. | 7 pass |

Key measurements (all measured on the Mac):

- Float GPT-2 with the integer coder: 1.992 bits per byte on 16 text files (32 KiB each). xz
  gives 2.508, bzip2 2.356, zstd 2.524. GPT-2 is about 21% smaller than xz.
- Binary files grow to 22.3 bits per byte with GPT-2. The raw-block fallback is necessary.
- Coder overhead against the table ideal: 0.021%.
- Float drift on one machine: full-sequence encode with KV-cache decode decoded 1 of 270
  windows. Batch 4 against batch 1, and float32 against float64, decoded 0 of 38.
- HardShell on a GPT-2-sized random model: 41.2 tokens/s on one thread. Real `gpt2.hsm`: about
  55 tokens/s on one thread (measured by `eval_bits`).
- Integer GPT-2 (`gpt2.hsm`, SHA-256 `aaa03eb9…a5c2`) against float on 16 held-out text files:
  1.871 against 1.867 bits per byte (+0.2%).
- `taco fold` with `gpt2.hsm`, all restored byte for byte: alice29 (32 KiB) 8,432 bytes (xz
  12,400); cp.html 4,759 (xz 7,652); lcet10 (4 KiB) 1,237 (xz 1,684); HDFS log (16 KiB) 3,129
  (xz 3,164); kennedy.xls (8 KiB) stored raw, 8,285 (xz 1,672). Speed: about 0.5 KB/s with 6
  threads.

Golden hashes (conformance vectors, plan §11). The Mac computed them. devbox has not checked
them yet.

| Vector | SHA-256 |
|---|---|
| Coder stream | `91676cdc669fca0eb22ed18f5f629882212523261f41d0d5c568b647e78d798f` |
| HardShell tiny model output | `d4063522517854fdcab605c3d3159bd3ac9a15eb92fa6721ff4a276658f90078` |
| Container vector A | `c1c07402a5dae2993c596345a696d35b7c1c0502fe304d31fa33d4f5ecefeeec` |
| taco golden container (tiny model) | `c3ff718b1c9682589836078b36d936836077744b03b1cc8af3a6ced0d5821b8a` |
| `gpt2.hsm` | `aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2` |
| `conformance/tiny.hsm` | `eb78b89a4f0f17c7245865aed22f86512a49e0272b7583a831c47e1a401418f8` |

## 3. Machines

- **Mac** (Apple Silicon, 8 GB, about 16 GB free disk). All work of 2026-10-18 ran here with
  `scripts/mx`. The GPT-2 weights and the corpus are in `tools/data/` (537 MB, ignored by git).
- **devbox** (Linux x86, RTX 3060 6 GB). Offline at 15:00. Rohit compacted the disk and frees
  more disk now. `scripts/dx` refuses to run while Windows C: has less than 15 GB free. Do not
  start Ubuntu or the keep-alive task without Rohit's approval.

## 4. Running agents

None at 16:30. Old status files are in `.agent-status/` (not committed).

## 5. Next steps, in order

1. When devbox is back: run the cross-machine check. Run the golden tests on devbox
   (`scripts/dx main cargo test --workspace`) and compare the hashes in section 2. Make
   `gpt2.hsm` on devbox (`python -m convert.gpt2 --recipe clip`) and compare its hash. Fold
   on the Mac, unfold on devbox, and the reverse. Then write `scripts/xcheck` (plan §11.3).
   That is the phase 2 gate.
2. Phase 1 part B (float drift, Mac against devbox): the commands are in §10 of
   `docs/research/2026-10-18-phase0-baseline.md`.
3. Phase 3 (quality): longer context (sliding window), the block size, an integer match model
   and a mixer for repeated text (logs, JSON), and a block kind that stores a block with zstd
   when GPT-2 is worse (binary data). The last two change the container format: each needs an
   ADR.
4. Phase 4 (speed): SIMD on the CPU, then integer CUDA kernels. Every fast path must give the
   same bits as the reference path.
5. A head-to-head benchmark against zstd --long, brotli, PPMd, paq8px and cmix (where memory
   allows) on the same files, for the README.

## 6. Notes

- `experiments/` holds the pure-Python prototype of 2026-10-18 (copied from the first scratch
  session). It is ignored by git. Its results are in `docs/research/2026-10-18-verification.md`.
- The Canterbury site refuses scripted downloads (HTTP 403). `tools/baseline` uses a GitHub
  mirror with the published sizes.
- Time estimates for the owner (est.): prototype with fold and unfold bit-exact on both machines
  in 3 to 5 days (fold and unfold already work on the Mac on 2026-10-18). Version 1 with a GPU
  path in 5 to 7 weeks.
