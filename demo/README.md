# Taco demo

This folder has everything to try Taco. You do not compile anything, and nothing is
downloaded. The folder holds:

| File | What it is |
|---|---|
| `try` (macOS, Linux), `try.cmd` (Windows) | **The one command.** It folds and unfolds the sample file, checks the result and prints a hash. |
| `taco` (macOS, Linux), `taco.cmd` (Windows) | The Taco command for your own files. It selects the program for your machine and the model file of this folder. |
| `gpt2.hsm` | The model file (GPT-2 small, integers only, 127 MB). Every machine uses this same file. |
| `bin/` | The programs: `macos-arm64`, `linux-x86_64`, `windows-x86_64`. |
| `samples/sample.txt` | A short English text (3,672 bytes). |

> [!WARNING]
> **Taco is slow, and this is an alpha (a proof of concept).** It is an experiment, not a replacement for
> zip, zstd or xz. Keep a second copy of each file that you fold. Do not use it for important
> data.

> [!NOTE]
> Most of the code of Taco was written by Claude (an AI model by Anthropic) with Claude Code,
> under the direction of the author, to get to results faster. The tests, not the authorship,
> are the evidence that it works.

## 1. Run the demo (one command)

**Open a terminal in this folder** (the folder that has this README, `try` and `gpt2.hsm`):

| System | Do this |
|---|---|
| macOS | Open the Terminal app. Type `cd ` (with a space), drag this folder from Finder into the Terminal window, and press Return. |
| Linux | Open a terminal and `cd` to this folder, for example `cd ~/Downloads/taco-demo-0.1.0`. |
| Windows | Open this folder in Explorer. Click the address bar, type `cmd`, and press Enter. |

**Then type the command and press Return (Enter):**

| System | Command |
|---|---|
| macOS (Apple Silicon), Linux (x86-64) | `./try` |
| Windows (x86-64) | `try.cmd` |

The demo prints `PASS: the file came back byte for byte`, the sizes, the times, and the
SHA-256 of the `.taco` file. On an Apple M2 with the CPU only, the sample (3,672 bytes)
folds to 648 bytes in 6 s and unfolds in 9 s (measured).

## 2. Check the hash: the main point of Taco

The SHA-256 of the `.taco` file of the sample is the same on every machine and every
backend (CPU, NVIDIA, AMD, Apple, Intel GPU):

```
EXPECTED_SAMPLE_HASH
```

Run the demo on two different computers. The hash is the same. Thus a file that one machine
folds unfolds on the other machine. To use only the CPU, run `./try --backend cpu`. The hash
does not change.

## 3. Use it on your own files

```bash
./taco fold notes.txt              # makes notes.txt.taco
./taco unfold notes.txt.taco       # restores notes.txt; the checksum is verified
./taco --help                      # all options
```

On Windows, use `.\taco.cmd` in place of `./taco`.

Good files to try: English text, HTML, source code, logs. Photos, video, music and zip files
do not get smaller: Taco stores them with zstd or raw.

## 4. How long it takes

> [!CAUTION]
> Start with a file of a few KB. With the CPU only, a 100 KB file can take 10 minutes or more
> to unfold (est.).

| Machine | Fold (compress) | Unfold (restore) |
|---|---:|---:|
| NVIDIA GPU (RTX 3060 Laptop, CUDA), English text | 85 to 129 KB/s | 26 to 31 KB/s |
| CPU only (Apple M2, AMD Ryzen 7) | 0.3 to 2 KB/s | 0.1 to 0.5 KB/s |

These numbers are measured on our machines. Your speed can be different.

- Each command needs about 2 to 3 s to start (it loads the model and runs a GPU self-test).
- Unfold is slower than fold, because the decoder must find one token after the other.
- A small file is not much faster than one block: about 10 to 13 s to unfold a 16 KiB block
  on the RTX 3060 (est.).
- The Apple M2 GPU (Metal) is not faster than its CPU for Taco (measured).

## 5. If something does not work

| Problem | What to do |
|---|---|
| macOS: "cannot be opened because the developer cannot be verified" | Run `./try` again. The `taco` script removes the quarantine mark from the demo program. Or run `xattr -dr com.apple.quarantine .` in this folder. |
| Windows: "Windows protected your PC", or the program is blocked | Right-click the zip, select Properties, select Unblock, and unpack it again. |
| Linux: an error about `GLIBC` | The program needs glibc 2.34 or later (Ubuntu 22.04, Debian 12, Fedora 35 or later). |
| "this folder has no program for ..." | This demo has programs for macOS on Apple Silicon, Linux x86-64 and Windows x86-64 only. |
| `unfold` stops with a checksum error | Run it again with `--backend cpu`. If it still fails, please report it with the output of `./taco --version`. |
| It is very slow | That is normal on a CPU. See section 4. |

A GPU is optional. Taco selects the backend itself: CUDA on NVIDIA (RTX 30 series or later),
wgpu on other GPUs, else the CPU. It runs a self-test first and uses the next backend if the
test fails. The output is the same on each backend.

## 6. More

- The project README explains how Taco works, the results, the guarantees and the limits.
- `bin/<platform>/BUILDINFO` records the commit and the toolchain of each program.
- `SHA256SUMS` lists the SHA-256 of each file in this folder.
- Licence: Apache-2.0 (`LICENSE`). The libraries in the programs: `THIRD-PARTY-LICENSES.txt`.
  The model is converted from GPT-2 by OpenAI; its licence is in `MODEL-LICENSE.txt`
  ("Modified MIT License").
