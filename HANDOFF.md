# Taco: session handoff

Written 2026-10-18 at about 15:00 +04, updated at about 18:10 +04 by the lead session (Claude). A new session or an agent
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

## 3. Repository and machines

- **Remote:** `origin` = https://github.com/tac0mania/taco (private, Rohit's tac0mania account,
  Apache-2.0). Push after each verified wave of commits. Never push weights, corpora or data.


- **Mac** (Apple Silicon, 8 GB, about 16 GB free disk). All work of 2026-10-18 ran here with
  `scripts/mx`. The GPT-2 weights and the corpus are in `tools/data/` (537 MB, ignored by git).
- **devbox** (Linux x86, RTX 3060 6 GB). Online again at 17:00. On 2026-10-18 the lead deleted
  the Hippo build caches (about 60 GB) and the local AI images except Ollama (about 17 GB), on
  Rohit's request. Linux uses 63 GB. Windows C: had 26 GB free, because the disk file shrinks
  only after a compaction or with sparse mode (Rohit's decision; it stops devbox). The local AI
  data volumes are kept. CUDA: no nvcc; NVRTC and the CUDA runtime exist as pip wheels in
  `~/.cache/hippo/cuda` (Hippo's, read only). `scripts/dx` refuses to run while C: has less
  than 15 GB free.

- **Colab** (Google, owner's account). The Colab CLI (`~/.local/bin/colab`) is signed in on
  the Mac. Rule from Rohit (2026-10-18): use the RTX 3060 on devbox first. Use a Colab GPU only
  for a heavy job that takes too long on the 3060, for example the conversion of a 3B to 4B
  model. Colab time is limited (the balance was 0 compute units on 2026-10-18). Rohit permits Colab
  for such jobs. Tell him when you start a session, and stop it when the job ends. A TPU runtime cannot run HardShell (no CUDA, no Vulkan).

## 4. Running agents

| Agent | Owns | Status file |
|---|---|---|
| simd | CPU speed: kernels, prefill (fold), batch (unfold) in `crates/hardshell`, calls in `crates/taco` | `.agent-status/simd.md` |
| cuda (now the portable GPU agent) | `crates/hardshell-gpu` (wgpu and WGSL first; CUDA later with precompiled PTX), ADR 0008 | `.agent-status/cuda.md` |

Owner requirements for speed work: users install nothing except a normal GPU driver; AMD,
NVIDIA, Intel and Apple GPUs must work; kernels are one op each (design in
`docs/design/anymodel.md`). Every fast path gives the same bits as the reference path.

Owner decisions of 2026-10-18 evening (Rohit):

- **Speed gate first.** Build a CUDA backend for NVIDIA (`crates/hardshell-cuda`, int8 tensor
  cores, kernels precompiled, the driver only at run time). Gate: GPT-2 small at 10,000 tokens/s
  or more on the RTX 3060 (batched decode), bit-identical to the CPU. If it fails, stop and
  rethink before more model work. First portable GPU result (GL on D3D12 in WSL, measured): 106,
  751, 1,311 and 1,166 tokens/s at batch 1, 8, 32 and 128.
- **Then Route A on the RTX 3060:** fine-tune distilgpt2 (GPT-2 architecture, 82M) on Loghub
  logs, convert it with `tools/convert`, and measure on log systems that the model did not see
  in training, and on English-heavy lines. PyTorch in `tools/` is permitted for this.
- Later ideas for speed (not started): speculative decoding with the match model, a skip of the
  model for very confident tokens, small specialist models (5M to 20M).

## 5. Next steps, in order

1. **Phase 2 gate: passed** on 2026-10-18 (`docs/research/2026-10-18-cross-machine.md`). Run
   `scripts/xcheck` again after every change to HardShell, the coder, the tokenizer or the
   container. It takes about 40 minutes with the reference engine.
2. Commit the simd and GPU work after each step when the agents report it ready. Run
   `scripts/xcheck` before each HardShell commit.
3. Logs (the first market): an integer match model and a mixer, so that Taco beats xz and zstd
   by a wide margin on logs. Needs an ADR (it changes the prediction path).
4. A zstd block kind for binary data (container change, ADR).
5. Float survey of other models (anymodel M0): SmolLM, Qwen2.5 0.5B/1.5B, Llama 3.2 1B,
   Nanbeige4.2-3B-Base. Gate: at least 10% better than GPT-2. Owner decisions are open:
   PyTorch in `tools/`, the models, and the loss that starts a fine-tune.
6. Phase 1 part B (float drift, Mac against devbox): commands in §10 of
   `docs/research/2026-10-18-phase0-baseline.md`.
7. A head-to-head benchmark (zstd --long, brotli, PPMd, paq8px, cmix) for the README.

Facts about Nanbeige4.2-3B (read from its GGUF header on devbox, measured): Apache-2.0,
4,169,800,704 parameters, 22 blocks run 2 times (`num_loops = 2`), width 3,072, 48 heads and 8
KV heads, context 262,144, SentencePiece-style vocabulary of 166,144 tokens. The file on devbox
is the chat model in Q4_K_M. The base model is `Nanbeige/Nanbeige4.2-3B-Base`.

## 6. Notes

- `experiments/` holds the pure-Python prototype of 2026-10-18 (copied from the first scratch
  session). It is ignored by git. Its results are in `docs/research/2026-10-18-verification.md`.
- The Canterbury site refuses scripted downloads (HTTP 403). `tools/baseline` uses a GitHub
  mirror with the published sizes.
- Time estimates for the owner (est.): prototype with fold and unfold bit-exact on both machines
  in 3 to 5 days (fold and unfold already work on the Mac on 2026-10-18). Version 1 with a GPU
  path in 5 to 7 weeks.
