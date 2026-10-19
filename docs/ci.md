# Continuous integration

This document tells what GitHub Actions runs for Taco, on which machines, and when. The
workflow is `.github/workflows/ci.yml`. The repository is `github.com/tac0mania/taco` (private).

## 1. What runs where

| Job | Machine | When | What |
|---|---|---|---|
| `check` | `ubuntu-latest` (Linux x86-64) | Each push to `main`, each pull request to `main`, by hand, weekly | `scripts/check-tags.sh`, `scripts/check-no-floats.sh`, `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace` |
| `conformance` | `ubuntu-latest` (Linux x86-64) | Each push, each pull request, by hand, weekly | The golden tests and the conformance program (section 3) |
| `conformance` | `ubuntu-24.04-arm` (Linux ARM64) | Each push, each pull request, by hand, weekly | The same |
| `conformance` | `macos-latest` (macOS ARM64) | By hand and weekly only | The same |
| `conformance` | `windows-latest` (Windows x86-64, MSVC) | By hand and weekly only | The same |
| `compare` | `ubuntu-latest` | After all `conformance` jobs | Compares the hash lists of all machines. A difference fails the run. |

"By hand" is `workflow_dispatch`: open the Actions tab, select `ci`, and click "Run workflow".
"Weekly" is the schedule: each Monday at 03:17 UTC.

`scripts/check.sh` runs the same checks as the job `check`. Run it on devbox before you push:

```
scripts/dx main scripts/check.sh
```

## 2. The cost rule

GitHub gives a private repository a number of free minutes each month. A macOS minute uses 10
Linux minutes, and a Windows minute uses 2 Linux minutes (published, GitHub billing). Thus:

- Linux x86-64 and Linux ARM64 run on each push and each pull request.
- macOS and Windows run only by hand and on the weekly schedule.

Do not add macOS or Windows to the push trigger. If a change can be different on macOS or on
Windows (for example a change to a SIMD kernel, to the zstd build, or to file paths), run the
workflow by hand on its branch before you merge it.

Estimated minutes (est.; the first runs on GitHub give the real numbers):

| Run | Jobs | Linux minutes |
|---|---|---|
| A push with a warm cache | `check` 8, x86-64 `conformance` 4, ARM64 `conformance` 4, `compare` 1 | about 17 |
| A push with no cache (a change to `Cargo.lock` or to the toolchain) | `check` 20, two `conformance` 10 each, `compare` 1 | about 41 |
| The weekly run | the push run, and macOS 6 x 10, Windows 10 x 2 | about 100 |

The basis of the estimate (measured on devbox, 2026-10-19, with the GPU hidden):

- A cold `conformance` job with 2 build jobs: 143 s to build the golden tests, 58 s to run
  them, 68 s to build the conformance program, 33 s to run it. That is 5 minutes.
- The `check` job with 4 build jobs and a partly warm target directory: 27 s for the tags,
  floats, fmt and clippy, and 136 s to build and run all tests (452 passed, 0 failed, 5
  ignored).

A GitHub runner of a private repository has 2 processors (published). It is slower than devbox
(est.).

The cache keeps the build output of `main`. A pull request reads the cache of `main` and does
not write a cache.

## 3. The cross-machine conformance check

Each `conformance` job does two things.

1. It runs the golden tests: `golden`, `gpt2`, `context`, `wide`, `hsm_merges`, `version2`,
   `zstd`, `container` and `logs_profile`. These tests compare the outputs of HardShell, the
   coder, the match model, the zstd blocks and the container with the hashes that the Mac
   recorded (`conformance/README.md`, `HANDOFF.md` §2). Thus each machine checks the
   conformance vectors.
2. It runs the conformance program `scripts/conformance`. The program makes 9 fixed inputs
   (English, UTF-8 text, a server log, JSON lines, binary records, random bytes, an empty
   file). It runs them through `conformance/tiny.hsm` on the CPU and writes the SHA-256 of
   each output to `SHA256SUMS`. The job uploads `SHA256SUMS`, `summary.txt` and the `.taco`
   files as an artifact.

The list has 50 hashes (measured):

| Name | What it is |
|---|---|
| `vector/…` | The committed conformance files. A damaged checkout stops the program. |
| `input/…` | The inputs. |
| `hardshell/<input>.weights` | All HardShell weights for the tokens of the input, in the order of the coder. |
| `coder/<input>.coded` | The range coder stream of the tokens, with tables from these weights. |
| `coder/<input>.mixed` | The same, with the match model and the mixer of the profile `logs`. |
| `taco/<case>.taco` | The container of `taco::fold` for 12 cases (profiles `text`, `logs`, `auto`; several block sizes). |

NOTE: the tiny model has random weights. A coded block is almost never smaller than zstd or
raw, so the containers alone do not test HardShell. The `hardshell/` and `coder/` lines test it.

The program also checks these items on each machine, and fails if one is not correct:

- The step path with the fastest kernels of the CPU (NEON or AVX2), the portable kernels, the
  prefill path and the reference session give the same weights (CLAUDE.md §1 rule 10).
- The range decoder gives the tokens back.
- A fold with 1 thread and a fold with 3 threads give the same container.
- `taco::unfold` restores each input byte for byte.

The job `compare` downloads the lists and compares each list with the list of Linux x86-64.
Any difference fails the run. The summary of the run shows a table of the machines.

### Run the program by hand

The program is its own Cargo workspace, so it does not change the root `Cargo.toml` or
`Cargo.lock`. Copy the root `Cargo.lock` first. Then the program uses the same dependency
versions as the workspace.

On devbox:

```
cp Cargo.lock scripts/conformance/Cargo.lock
scripts/dx main 'cargo run --manifest-path scripts/conformance/Cargo.toml -- --out /tmp/conformance'
```

On the Mac:

```
cp Cargo.lock scripts/conformance/Cargo.lock
scripts/mx cargo run --manifest-path scripts/conformance/Cargo.toml -- --out /tmp/conformance
```

To compare two machines, compare the two `SHA256SUMS` files with `diff`. They must be the same.

## 4. The GPU tests in CI

The GitHub runners have no GPU. The tests of `hardshell-cuda` and `hardshell-gpu`, and the GPU
tests of `crates/taco/tests/backends.rs`, then skip:

- `hardshell-cuda`: no CUDA driver library, so the device does not open (`Error::NoDevice`).
- `hardshell-gpu`: no GPU adapter. wgpu does not select a software adapter (llvmpipe,
  lavapipe), because `AdapterChoice::BestHardware` leaves out CPU adapters.

Any other device error fails the test, so a skip does not hide a real failure on these
runners. The job `check` shows the number of skips in its summary.

devbox showed this behaviour with the GPU hidden (measured, 2026-10-19). A user namespace hid
`/usr/lib/wsl/lib` (the CUDA driver) and `/dev/dxg`. All tests of `hardshell-cuda`,
`hardshell-gpu` and `backends.rs` passed, and the backends test reported "no CUDA driver
library" and "no GPU adapter found".

A real GPU run is manual:

- On devbox (NVIDIA RTX 3060): `scripts/dx main cargo test -p hardshell-cuda -p hardshell-gpu`.
  CAUTION: the GPU of devbox is shared. Ask the lead before you run GPU tests.
- On Windows (Vulkan, DX12): `scripts/wcheck` (the adapter list is in
  `docs/research/2026-10-19-windows.md` §7).
- On the Mac (Metal): `scripts/mx cargo test -p hardshell-gpu`.

## 5. The full cross-machine check with GPT-2 (manual)

CI does not have the GPT-2 model file `gpt2.hsm` (127 MB). The file is not in the repository
(plan §11.1). Thus the full check stays manual. Run it after each change to HardShell, the
coder, the tokenizer or the container (CLAUDE.md §1 rule 7).

1. Make sure that `tools/data/out/gpt2.hsm` exists on the Mac and on devbox, with the SHA-256
   `aaa03eb9…a5c2`. If it does not exist, run `cd tools && uv run python -m convert.gpt2`.
2. Make sure that the corpus is in `tools/data/corpus` and the logs are in `tools/data/logs`.
3. On the Mac, run `scripts/xcheck`. It folds on the Mac and on devbox, unfolds on the other
   machine, and compares the `.taco` files. It takes about 40 minutes (measured, 2026-10-18).
4. Commit the change. Then run `scripts/wcheck` for the Windows half. It builds the committed
   `HEAD` only.
5. Write the results in a research note, as in `docs/research/2026-10-18-cross-machine.md`.

## 6. Settings for the owner

These items are outside the repository. The lead or Rohit must do them.

1. Make sure that GitHub Actions is on for the repository (Settings, Actions, General).
2. ARM64 Linux runners (`ubuntu-24.04-arm`): UNVERIFIED that the plan of the account gives
   them to a private repository. If the job waits in the queue and does not start, set the
   repository variable `TACO_CI_ARM_LINUX` to `false` (Settings, Secrets and variables,
   Actions, Variables). The ARM64 half of the check then is the macOS job and the Mac.
3. Optional: protect `main` (Settings, Branches). Require the checks `check (Linux x86-64)`,
   `conformance (ubuntu-latest)` and `compare the machines`. NOTE: GitHub Free does not
   enforce branch protection on a private repository (published); GitHub Pro does.
4. Watch the minutes (Settings, Billing). Section 2 gives the estimate.

The workflow uses these actions: `actions/checkout@v5`, `Swatinem/rust-cache@v2`,
`actions/upload-artifact@v4` and `actions/download-artifact@v4`. If GitHub shows a warning
about an old Node.js version, change the action to its newest major version.
