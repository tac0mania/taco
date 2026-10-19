# Taco release: archives, signatures and the release checklist

Status: proposed, 2026-10-19 (the release agent). Owner: Rohit. This document obeys
`docs/STYLE.md`. It implements product-v1 §1 (row "Release") and work item 13. The model
files are in `docs/design/model-distribution.md`.

Decisions that Rohit must make are in §8.

## §1 What a release contains

| File | Contents |
|---|---|
| `taco-<version>-aarch64-apple-darwin.tar.gz` | macOS, Apple Silicon |
| `taco-<version>-x86_64-apple-darwin.tar.gz` | macOS, Intel |
| `taco-<version>-x86_64-unknown-linux-gnu.tar.gz` | Linux x86-64, glibc 2.34 or later (measured, §3.3) |
| `taco-<version>-aarch64-unknown-linux-gnu.tar.gz` | Linux arm64 (not built yet, §3.4) |
| `taco-<version>-x86_64-pc-windows-gnu.zip` | Windows 10 and 11, x86-64 |
| `taco-model-gpt2-aaa03eb9.tar` | `gpt2.hsm`, its licence and its record (model-distribution §8.2) |
| `SHA256SUMS` | The SHA-256 of each file above |

Each program archive has one directory `taco-<version>-<target>/` with these files:

| File | Source |
|---|---|
| `taco` or `taco.exe` | `cargo build --release --locked -p taco-cli --target <target>` |
| `LICENSE` | `LICENSE` (Apache-2.0) |
| `README.md` | `README.md` |
| `taco.1` | `docs/man/taco.1` |
| `INSTALL.txt` | `scripts/release/INSTALL.txt.in` |
| `THIRD-PARTY-LICENSES.txt` | `scripts/release/third_party.py`: each crate in the binary, and its licence texts |
| `BUILDINFO` | The version, the target, the commit, the tree state, `SOURCE_DATE_EPOCH`, the toolchain, the flags, the SHA-256 of the binary |
| `SHA256SUMS` | The SHA-256 of the other files of the archive |

The program archives do not contain a model file. The model is 127 MB (measured), and one
model archive serves every platform.

## §2 The scripts (`scripts/release/`)

| Script | Job | Runs on |
|---|---|---|
| `build.sh <target>` | Builds the binary and makes the archive in `target/dist/`. With `--prebuilt <binary>`, packages a binary that another machine built. | Any machine with the target |
| `package.py` | Makes a tar, tar.gz or zip with fixed metadata (sorted members, the epoch as the time, owner 0, fixed modes). | Any (Python standard library) |
| `third_party.py` | Lists the crates of the binary for a target (`cargo metadata --filter-platform`) and writes their licence texts. | Any |
| `smoke.sh <archive> <model> [backends]` | Unpacks an archive, checks `SHA256SUMS`, prints the libraries and the glibc version, and folds and unfolds 5 fixed inputs on each backend. | The target machine |
| `mac.sh [arm64\|x86_64]` | `build.sh` and `smoke.sh` on the Mac, with the rules of `scripts/mx`. | Mac |
| `linux.sh` | `build.sh` and `smoke.sh` on devbox through `scripts/dx release`, then copies the archive back. | Mac, to devbox |
| `windows.sh` | Builds `taco.exe` on the Windows side of devbox, packages the zip on the Mac, and runs a smoke test on Windows. | Mac, to Windows |
| `model-archive.sh <gpt2.hsm> <licence>` | Makes the model archive. Refuses a file with a wrong SHA-256 or size. | Any |

The smoke inputs are fixed data, so the `.taco` hashes of two machines must be equal. This is a
small cross-machine check of the release binaries themselves.

## §3 Platforms

### §3.1 Reproducible builds

`build.sh` makes the binary depend only on the source, the toolchain and the target:

- The toolchain is the one of `rust-toolchain.toml` (1.99.0). `BUILDINFO` records `rustc -vV`
  and the C compiler (for `zstd-sys`).
- `--locked` uses `Cargo.lock` exactly.
- `SOURCE_DATE_EPOCH` is the commit time. The archive uses it as the time of each member.
- `--remap-path-prefix` (Rust) and `-ffile-prefix-map` (C) replace the source directory and
  `CARGO_HOME` with `/taco` and `/cargo`. A third `--remap-path-prefix` maps the source of the
  standard library in the toolchain back to `/rustc/<commit>`. Without it, a toolchain with the
  `rust-src` component writes the local path (with the user name) into the binary: 33 paths in
  the arm64 binary of 2026-10-19 (measured; that binary was built before this fix).
- `MACOSX_DEPLOYMENT_TARGET` is 11.0 (arm64) or 10.15 (x86-64), so that the C code and the Rust
  code have the same minimum macOS. The arm64 binary of 2026-10-19 has `minos 11.0` (measured).
- The release binary has no debug information (`CARGO_PROFILE_RELEASE_DEBUG=0`,
  `CARGO_PROFILE_RELEASE_STRIP=debuginfo`). The symbol names stay. The root `Cargo.toml` is not
  changed.
- No `target-cpu` flag. The CPU fast paths (AVX2, NEON dot product) select themselves at run
  time (`hardshell/src/kernels`), so one binary runs on every CPU of the target.

Limits (not measured yet):

- The archive bytes depend on the zlib of the Python that packages them. The SHA-256 of the
  binary (in `BUILDINFO`) is the value to compare between two builds.
- A signature (§4) changes the binary. Compare the binary without its signature
  (`codesign --remove-signature` on macOS; on Windows, `osslsigncode remove-signature`).
- Measured on 2026-10-19: devbox built the Linux binary two times, each time in an empty build
  directory, and got the same SHA-256 (`21c5c711…cf1e`, 9,364,168 bytes). Both builds used the
  same source directory. A build in another source directory is the full proof. It is in the
  checklist (§5 step 8).

### §3.2 macOS (arm64 built and tested; x86-64 not built)

- `scripts/release/mac.sh arm64` built and tested the arm64 archive on 2026-10-19 (results in the
  report of the release agent and in `target/dist/*.smoke.txt`).
- The binary links only system libraries and frameworks: `libSystem`, `libiconv`, `libobjc`,
  Metal, QuartzCore, CoreGraphics, Foundation and CoreFoundation (measured with `otool -L`). The
  linker gives it an ad-hoc signature, which Apple Silicon requires.
- The smoke test passed on the backends `auto`, `cpu` and `gpu` (Metal). For each of the 5
  inputs, the 3 backends gave the same `.taco` file (measured).
- x86-64: `rustup target add x86_64-apple-darwin`, then `scripts/release/mac.sh x86_64`. Apple
  clang builds the C code of `zstd-sys` for x86-64 with no other tool (est.). Rosetta 2 is on
  the Mac (measured), so the smoke test can run the x86-64 binary. This build was not done:
  the Mac has 8 GB of memory and the rule is one build.
- Optional: one universal binary (`lipo -create`) instead of two archives. Not proposed for
  version 1.

### §3.3 Linux x86-64 (built and tested on devbox)

- `scripts/release/linux.sh` built the archive on devbox (Ubuntu 24.04, glibc 2.39) and tested
  it there. The build took 80 s (measured). The smoke test passed on the backends `auto` (CUDA),
  `cpu` and `cuda`. The 5 `.taco` files are equal on the three backends, and equal to the files
  of the Mac (CPU and Metal), byte for byte (measured).
- The binary needs `libc.so.6`, `libm.so.6`, `libgcc_s.so.1` and the dynamic loader only, and
  glibc 2.34 or later (measured with `objdump -T`). Thus it runs on Ubuntu 22.04 and later,
  Debian 12 and later, and RHEL 9 and later (published glibc versions of those systems; not
  tested). It does not run on RHEL 8 or Ubuntu 20.04.
- CUDA: `hardshell-cuda` loads `libcuda.so.1` from the NVIDIA driver at run time and gives it
  the embedded PTX. The machine needs no CUDA toolkit. wgpu loads `libvulkan.so.1` or the GL
  libraries at run time. A machine without a GPU uses the CPU.
- For an older glibc: build in a container with an older system (for example Debian 11, glibc
  2.31) with the same Rust toolchain. This needs a container image of about 1.5 GB (est.) on
  devbox, so it waits for a decision (§8 item 6).
- A static musl build is possible for the CPU path, but a static binary cannot load
  `libcuda.so.1` or the Vulkan loader. Thus musl removes the GPU. Not proposed.

### §3.4 Linux arm64 (not built)

devbox has no arm64 Rust target and no arm64 C compiler (measured). Two ways:

1. A cross build on devbox: `rustup target add aarch64-unknown-linux-gnu`, `sudo apt install
   gcc-aarch64-linux-gnu` (est. 100 to 200 MB), then
   `TACO_CROSS_CC=aarch64-linux-gnu-gcc scripts/release/build.sh aarch64-unknown-linux-gnu`.
   The test needs an arm64 machine or `qemu-user` (another install).
2. A native build on an arm64 Linux machine (for example a GitHub-hosted arm64 runner, or a
   Raspberry Pi 5) with `build.sh aarch64-unknown-linux-gnu` and `smoke.sh`.

Both need software that is not installed now. Rohit decides (§8 item 6).

### §3.5 Windows x86-64 (script written, not built on 2026-10-19)

- `scripts/release/windows.sh` uses the per-user toolchain of
  `docs/research/2026-10-19-windows.md` (Rust 1.99.0 `x86_64-pc-windows-gnu`, w64devkit GCC 16.2.0,
  in `C:\Users\rohit\taco-win`). It installs nothing. It builds the committed `HEAD`.
- **It did not run on 2026-10-19.** C: had 18 to 19 GB free (measured), less than the limit of 20 GB
  for work on Windows. The Linux disk of devbox is a file on C:. The script refuses to run below
  the limit and checks again after the build. A release build adds about 1 GB of build output
  in `C:\Users\rohit\taco-win\target` (est.).
- The `-gnu` binary links `msvcrt.dll` and Windows system DLLs. The w64devkit GCC is static
  only, so the binary needs no MinGW DLL (est.). The smoke test runs `taco.exe` with a PATH of
  the Windows directories only, to prove it.
- The DX12 shader compiler: §6.

## §4 Code signing and notarization

Nothing in this section is bought or started. Each item is Rohit's decision (§8).

### §4.1 macOS

| Item | What | Cost |
|---|---|---|
| Apple Developer Program | The membership that gives Developer ID certificates. As a person, or as an organization (an organization needs a D-U-N-S number). | 99 USD a year (published, Apple) |
| Developer ID Application certificate | Signs `taco`. Made in the developer account. | Included |
| Notary service (`xcrun notarytool`) | Apple scans the signed binary and records a ticket. | Included |

Steps for each macOS archive (after the decision):

1. `codesign --sign "Developer ID Application: <name> (<team id>)" --options runtime --timestamp taco`.
   The hardened runtime (`--options runtime`) is a condition of notarization. `taco` needs no
   entitlement: it does not compile code at run time, and Metal needs none (est.).
2. Make a zip of the signed binary and submit it:
   `xcrun notarytool submit taco.zip --keychain-profile taco-notary --wait`. An App Store Connect
   API key in the CI secrets replaces the keychain profile.
3. A bare binary cannot hold a stapled ticket (`stapler` works on `.app`, `.pkg` and `.dmg`).
   Gatekeeper then checks the ticket on the network at the first start. A signed and notarized
   `.pkg` (with a Developer ID Installer certificate) can hold the ticket, for machines
   without a network.
4. Build the archive again from the signed binary. Record the hash of the unsigned binary
   in `BUILDINFO` and of the signed binary in `SHA256SUMS`.

Without a signature: a browser download has the quarantine attribute, and Gatekeeper refuses to
start `taco` ("cannot be opened because the developer cannot be verified"). The user can run
`xattr -d com.apple.quarantine taco`. A download with `curl` has no quarantine attribute, so no
refusal occurs (est.).

### §4.2 Windows (Authenticode)

| Option | What | Cost and conditions |
|---|---|---|
| OV code signing certificate | A certificate from a public CA for a person or an organization. Since June 2023, the private key must be on a hardware token or in a cloud HSM (CA/Browser Forum rule, published). | About 200 to 500 USD a year (est., the price depends on the CA) |
| EV code signing certificate | As OV, with more identity checks. Microsoft says that EV no longer gives an immediate SmartScreen reputation (UNVERIFIED). | About 300 to 700 USD a year (est.) |
| Azure Artifact Signing (before: Trusted Signing) | Microsoft signs with a certificate that it manages. No token. | Basic tier about 9.99 USD a month (published in a news report, January 2026; check the Azure page). Open to organizations in the USA, Canada, the EU and the UK, and to persons in the USA and Canada only (UNVERIFIED). |

The tool: `signtool sign /fd sha256 /tr <timestamp URL> /td sha256 taco.exe` (Windows SDK), or
`osslsigncode`, which runs on Linux and macOS. Thus the signature step can run on devbox
Linux and needs no Windows SDK.

Without a signature: SmartScreen can show "Windows protected your PC" when a user starts
`taco.exe` from Explorer. A start from a terminal usually shows no warning (est.). Smart App
Control on Windows 11 (on new installations) can block an unsigned program (published,
Microsoft).

### §4.3 Linux and the checksums

Linux needs no signature. Every platform gets a signed `SHA256SUMS`:

- `minisign` or GPG with a key of the project (no cost), or
- Sigstore `cosign` with the GitHub OIDC identity (no key to keep, no cost), or GitHub
  artifact attestations. Whether attestations work for a private repository on the current
  GitHub plan is UNVERIFIED.

Recommendation: sign `SHA256SUMS` with minisign for version 1. Publish the public key in the
README and on the release page.

## §5 The release checklist

### §5.1 Versions

- The version is SemVer. The crates have `0.1.0` now. The first public release with container
  version 2 frozen (product-v1 §1) is `1.0.0`. Before it: `1.0.0-rc.1`, `1.0.0-rc.2`.
- A change that makes a container of an earlier release unreadable is not permitted
  (`CLAUDE.md` §1 rule 8). A new container version or a new model file is a minor version
  (`1.1.0`), because old files still unfold.
- The version is in `crates/taco-cli/Cargo.toml` (`build.sh` reads it there). All crates get the
  same version (request §9).
- The tag is `v<version>`, an annotated tag on the release commit. Its message lists the
  container versions, the model hashes and the conformance hashes.

### §5.2 The changelog

`CHANGELOG.md` at the root (request §9), in ASD-STE100, newest first. One section for each
version, with these parts:

1. **Formats**: the container versions that the release reads and writes; the model-file
   versions.
2. **Models**: name, SHA-256 and size of each model in the registry.
3. **Added**, **Changed**, **Fixed**: one line for each change, with the ADR or the commit.
4. **Measured**: bits per byte on the quality corpus, and the speed of each backend, with the
   machine (`CLAUDE.md` §8).
5. **Known problems**: for example a speed gate that fails (product-v1 §3).

### §5.3 The steps

Do the steps in this order. Stop at a failure and tell the lead.

1. **Freeze.** No open change to a format, a model file or the coder. Write the version in the
   crates. Write the changelog section.
2. **Static checks.** `cargo fmt --check`, `cargo clippy -- -D warnings` and `scripts/check.sh`
   on devbox (`scripts/dx main ...`).
3. **Tests.** `cargo test --workspace` on devbox and on the Mac (`scripts/mx`). The GPU tests
   (`crates/taco/tests/backends.rs`) on CUDA (devbox), wgpu Vulkan (devbox Windows), and wgpu
   Metal (Mac).
4. **Conformance vectors.** Every golden hash of `HANDOFF.md` §2 passes on the Mac, on devbox
   Linux and on Windows. The release notes list them.
5. **Cross-machine check.** `scripts/xcheck` (Mac and devbox, about 40 minutes) and
   `scripts/wcheck` (Windows). All `.taco` files equal; all inputs restored.
6. **Quality and speed.** The bits per byte on the quality corpus are not worse than the
   committed baseline. Measure the speed gates of product-v1 §3 on each backend.
7. **The benchmark page.** Update the results table of `README.md` §1 and the research note
   from `tools/bench`. Each number has the machine, the power setting of the GPU, the versions
   of the other compressors and the command. Numbers are `measured`.
8. **Archives.** `scripts/release/mac.sh arm64`, `mac.sh x86_64`, `linux.sh`, `windows.sh`, the
   Linux arm64 build (§3.4), `model-archive.sh`. Each `smoke.sh` passes. The smoke `.taco`
   hashes are equal on all machines. Build one target two times in two directories and compare
   the binary hash.
9. **Third-party licences.** Read the warnings of `third_party.py`. Each crate without a
   licence file has its licence named in `THIRD-PARTY-LICENSES.txt` (the script does this);
   check each new crate since the last release.
10. **Signatures** (if decided, §4). Sign, notarize, package again, and run `smoke.sh` on the
    signed archives.
11. **Checksums.** Make `SHA256SUMS` of all release files. Sign it (§4.3).
12. **Tag and release.** Make the tag. Make a draft GitHub release with the files and the
    changelog section. Publish the model archive and the `<hash>.hsm` assets on the model host
    (model-distribution §7). Then publish the release.
13. **After the release.** On a clean machine, download from the public URLs, verify
    `SHA256SUMS`, and run `smoke.sh`.

## §6 DXC for DX12 on Windows

### §6.1 The facts

- The Windows note (`docs/research/2026-10-19-windows.md` §8.2, measured): with the old shader
  compiler FXC, 7 of the 14 kernels of `hardshell-gpu` do not compile. With DXC
  (`dxcompiler.dll`, shader model 6), all 12 GPU tests pass on DX12, on the NVIDIA GPU and on
  the AMD GPU. Vulkan passes on both GPUs.
- wgpu 30.0.1 with `Dx12Compiler::Auto` (the default) loads `dxcompiler.dll` from the search
  path. If the DLL is not there, it uses FXC (read in `wgpu-hal-30.0.1/src/dx12/instance.rs`).
  It loads only `dxcompiler.dll`; it does not load `dxil.dll`. The test of the note used only
  `dxcompiler.dll`. DXC releases since 2025 have an internal validator, so `dxil.dll` is
  optional (published in the DXC release notes; UNVERIFIED for each version).
- A failed FXC compile takes long: each failed test took about 97 s (measured, Windows note
  §8.2). If `--backend auto` tries DX12 with FXC on a machine with a discrete AMD or Intel GPU,
  the user can wait minutes before the fallback to the CPU (est.).
- The DXC source has the University of Illinois/NCSA licence of LLVM (published, the
  `LICENSE.TXT` of `microsoft/DirectXShaderCompiler`). The release files of DXC also have
  Microsoft licence files; the terms for the redistribution of the binaries are UNVERIFIED.
  The latest release is v1.9.2609 of 2026-09-29; its Windows zip is 32,287,592 bytes
  (published, GitHub API, read 2026-10-19). The x64 `dxcompiler.dll` is about 20 MB (measured
  for the copy of Microsoft Edge in the note).
- On Windows with an NVIDIA GPU, `auto` selects CUDA first (ADR 0016). wgpu matters for AMD
  and Intel GPUs. `auto` uses wgpu only for a discrete GPU.

### §6.2 The options

| # | Option | For | Against |
|---|---|---|---|
| 1 | Ship `dxcompiler.dll` (x64) next to `taco.exe`, with its licence files | DX12 works on every Windows 10 and 11 machine with a DX12 GPU. No code change. | About 20 MB more (est. 7 to 9 MB more in the zip). The redistribution terms must be read. A third-party binary in the release. |
| 2 | Vulkan only on Windows; DX12 never | Small. Vulkan passes on NVIDIA and AMD (measured). AMD, NVIDIA and Intel drivers for Windows include Vulkan (est.). | A code change in `hardshell-gpu` (the backends of the instance on Windows). A machine with no Vulkan driver (old Intel GPUs, some virtual machines) uses the CPU. |
| 3 | The wgpu feature `static-dxc` | DXC inside `taco.exe`. | Only for the MSVC target, so it needs the Visual Studio Build Tools and the Windows SDK (est. 6 to 8 GB). A change to the root `Cargo.toml`. |
| 4 | Change the WGSL for FXC | No DLL. | 7 kernels change; every bit-exact test again; FXC is not maintained. |

### §6.3 Recommendation

Option 2 for version 1, with one addition: **DX12 only with DXC, never FXC.**

1. On Windows, `hardshell-gpu` asks wgpu for Vulkan first.
2. It sets `Dx12Compiler::DynamicDxc` (not `Auto`). Thus DX12 is used only when a
   `dxcompiler.dll` is on the search path, and FXC is never used. A user or an administrator
   who has no Vulkan driver can put `dxcompiler.dll` next to `taco.exe`.
3. The release does not ship DXC. The README and the man page say how to add it.

Reasons: no third-party binary and no unread licence in version 1; no 97 s FXC failure in
`auto`; NVIDIA uses CUDA; AMD and Intel discrete GPUs have Vulkan drivers (est.). If tests on
real AMD and Intel machines show machines without Vulkan, option 1 can follow in a minor
release.

This is a change in `crates/hardshell-gpu` (owned by the GPU agent). It needs the bit-exact
tests on Windows Vulkan again, and an ADR or an amendment to ADR 0008.

### §6.4 What the owner must decide

1. Option 2 (recommended) or option 1.
2. For option 1: a check of the DXC redistribution terms, and the acceptance of about 20 MB more
   in the Windows release.

## §7 What was built and tested on 2026-10-19

The report of the release agent has the sizes, the SHA-256 values and the smoke results. The
archives are in `target/dist/` (not committed). In short:

- macOS arm64: built and tested on the Mac.
- Linux x86-64: built and tested on devbox.
- macOS x86-64, Linux arm64, Windows x86-64: scripts only (§3.2, §3.4, §3.5).

These are test archives, not release archives. They come from commit `4925959` with the new
files of the release agent (the scripts and `docs/man/taco.1`), which were not committed. Their
`BUILDINFO` says `tree: clean`, because the first version of `build.sh` ignored untracked
files. `build.sh` now counts an untracked file as `dirty`.

## §8 Decisions for Rohit

1. **Apple Developer Program** (99 USD a year, published), to sign and notarize the macOS
   binaries (§4.1). Without it, users remove the quarantine attribute by hand.
2. **A Windows signature** (§4.2): an OV certificate on a token, Azure Artifact Signing (if
   eligible), or no signature in version 1.
3. **The signature of `SHA256SUMS`** (§4.3): minisign (recommended), GPG or cosign.
4. **DXC** (§6.4): option 2 (recommended) or option 1.
5. **Windows builds when C: has less than 20 GB free.** C: had 18 to 19 GB on 2026-10-19 (measured); `windows.sh` stopped at its first check.
   Compact the Linux disk of devbox, or permit a lower limit for the release build.
6. **More platforms**: an old-glibc container build (§3.3), a Linux arm64 build (§3.4), and the
   install of the x86-64 Rust target on the Mac (§3.2). Or a CI service with hosted runners
   for all platforms (the CI agent owns CI).
7. **The version of the first public release** (`1.0.0` when container version 2 is frozen,
   §5.1).

## §9 Requests to the lead

- Add `CHANGELOG.md` at the root (§5.2).
- Move the version to `[workspace.package]` in the root `Cargo.toml`, so that all crates have
  one version (§5.1).
- Add `/target/dist` handling to `.gitignore` if the lead wants it explicit (it is already under
  `/target`).
- Ask the GPU agent for §6.3 if Rohit selects option 2.
- Ask the CLI agent for `taco --version` (model-distribution §6.6). `build.sh` gives the commit
  in `TACO_COMMIT`; the CLI reads it with `option_env!("TACO_COMMIT")`.
