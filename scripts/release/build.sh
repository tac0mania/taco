#!/bin/bash
# Build the release archive of `taco` for one target (docs/design/release.md §3).
#
#   scripts/release/build.sh <target> [<out dir>]
#   scripts/release/build.sh --prebuilt <path to taco or taco.exe> <target> [<out dir>]
#
# <target> is one of:
#   aarch64-apple-darwin       on a Mac (Apple Silicon)
#   x86_64-apple-darwin        on a Mac, after `rustup target add x86_64-apple-darwin`
#   x86_64-unknown-linux-gnu   on Linux x86-64 (devbox: scripts/release/linux.sh)
#   aarch64-unknown-linux-gnu  on Linux arm64, or a cross build with TACO_CROSS_CC set
#   x86_64-pc-windows-gnu      only with --prebuilt (scripts/release/windows.sh builds it)
#
# The default <out dir> is target/dist (ignored by git and by devbox sync). The archive holds
# the binary, LICENSE, README.md, taco.1, INSTALL.txt, THIRD-PARTY-LICENSES.txt, BUILDINFO and
# SHA256SUMS. <out dir>/SHA256SUMS lists every archive in <out dir>.
#
# Reproducible build: the commit time is SOURCE_DATE_EPOCH, the toolchain is the one of
# rust-toolchain.toml, `--locked` uses Cargo.lock, and the build paths are replaced by fixed
# names (/taco, /cargo) in the binary. Without git (a devbox mirror), give TACO_COMMIT,
# TACO_TREE (clean or dirty) and SOURCE_DATE_EPOCH.
#
# Environment: CARGO_TARGET_DIR (default target/release-build), TACO_JOBS (build jobs),
# TACO_CROSS_CC (a C compiler for a cross target, for zstd-sys), TACO_BUILDINFO_EXTRA (a file
# whose lines are added to BUILDINFO).
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"

prebuilt=""
if [ "${1:-}" = --prebuilt ]; then
  prebuilt=${2:?usage: build.sh --prebuilt <binary> <target> [<out dir>]}
  shift 2
fi
target=${1:?usage: scripts/release/build.sh <target> [<out dir>]}
out=${2:-$root/target/dist}
case $target in
  aarch64-apple-darwin | x86_64-apple-darwin | x86_64-unknown-linux-gnu | aarch64-unknown-linux-gnu) fmt=tar.gz exe=taco ;;
  x86_64-pc-windows-gnu | x86_64-pc-windows-msvc) fmt=zip exe=taco.exe ;;
  *) echo "build.sh: unknown target $target" >&2; exit 64 ;;
esac
if [ "$fmt" = zip ] && [ -z "$prebuilt" ]; then
  echo "build.sh: build the Windows binary with scripts/release/windows.sh" >&2; exit 64
fi

# The source state.
if git -C "$root" rev-parse --git-dir > /dev/null 2>&1; then
  commit=${TACO_COMMIT:-$(git -C "$root" rev-parse HEAD)}
  epoch=${SOURCE_DATE_EPOCH:-$(git -C "$root" log -1 --format=%ct HEAD)}
  if [ -z "${TACO_TREE:-}" ]; then
    if [ -n "$(git -C "$root" status --porcelain)" ]; then tree=dirty; else tree=clean; fi
  else
    tree=$TACO_TREE
  fi
else
  commit=${TACO_COMMIT:?no git here: set TACO_COMMIT, TACO_TREE and SOURCE_DATE_EPOCH}
  epoch=${SOURCE_DATE_EPOCH:?no git here: set SOURCE_DATE_EPOCH}
  tree=${TACO_TREE:-unknown}
fi
export SOURCE_DATE_EPOCH=$epoch
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' crates/taco-cli/Cargo.toml | head -1)
[ -n "$version" ] || { echo "build.sh: no version in crates/taco-cli/Cargo.toml" >&2; exit 65; }
name="taco-$version-$target"
[ -f docs/man/taco.1 ] || { echo "build.sh: docs/man/taco.1 is missing" >&2; exit 66; }
if [ "$tree" != clean ]; then
  echo "build.sh: WARNING: the tree is $tree; do not publish this archive" >&2
fi

cargo_home=${CARGO_HOME:-$HOME/.cargo}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
stage="$work/$name"
mkdir -p "$stage" "$out"

# The build.
rustflags="--remap-path-prefix=$root=/taco --remap-path-prefix=$cargo_home=/cargo"
# NOTE: with the rust-src component installed, rustc writes the local path of the standard
# library source (in RUSTUP_HOME, with the user name) into panic locations. Map it back to the
# /rustc/<commit> path of the official build (measured on the Mac, 2026-10-19: 33 paths).
if [ -z "$prebuilt" ]; then
  sysroot=$(rustc --print sysroot)
  rustc_commit=$(rustc -vV | sed -n 's/^commit-hash: //p')
  rustflags="$rustflags --remap-path-prefix=$sysroot/lib/rustlib/src/rust=/rustc/$rustc_commit"
fi
cflags="-ffile-prefix-map=$root=/taco -ffile-prefix-map=$cargo_home=/cargo"
if [ -z "$prebuilt" ]; then
  if command -v rustup > /dev/null && ! rustup target list --installed | grep -qx "$target"; then
    echo "build.sh: the Rust target $target is not installed: rustup target add $target" >&2; exit 69
  fi
  export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$root/target/release-build}
  [ -n "${TACO_JOBS:-}" ] && export CARGO_BUILD_JOBS=$TACO_JOBS
  # NOTE: no debug information in the release binary: it holds build paths, and a panic in
  # library code is a bug that the lints already deny. Symbol names stay for backtraces.
  export CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_PROFILE_RELEASE_STRIP=debuginfo
  export RUSTFLAGS="$rustflags" CFLAGS="$cflags" CARGO_INCREMENTAL=0
  # The CLI can read the commit with option_env!("TACO_COMMIT") for `taco --version`.
  export TACO_COMMIT=$commit
  # The oldest macOS that the binary supports. 11.0 is the default of rustc for arm64; the
  # explicit value makes the C code of zstd-sys use the same value.
  case $target in
    aarch64-apple-darwin) export MACOSX_DEPLOYMENT_TARGET=11.0 ;;
    x86_64-apple-darwin) export MACOSX_DEPLOYMENT_TARGET=10.15 ;;
  esac
  if [ -n "${TACO_CROSS_CC:-}" ]; then
    tv=$(echo "$target" | tr '[:lower:]-' '[:upper:]_')
    export "CC_${target//-/_}=$TACO_CROSS_CC" "CARGO_TARGET_${tv}_LINKER=$TACO_CROSS_CC"
  fi
  echo "== cargo build --release --locked -p taco-cli --target $target"
  started=$(date +%s)
  cargo build --release --locked -p taco-cli --target "$target"
  echo "build time: $(( $(date +%s) - started )) s"
  bin="$CARGO_TARGET_DIR/$target/release/$exe"
else
  bin=$prebuilt
fi
[ -f "$bin" ] || { echo "build.sh: no binary at $bin" >&2; exit 66; }

# The files of the archive.
cp "$bin" "$stage/$exe"
chmod 755 "$stage/$exe"
cp LICENSE "$stage/LICENSE"
cp README.md "$stage/README.md"
cp docs/man/taco.1 "$stage/taco.1"
python3 -I "$root/scripts/release/third_party.py" --manifest "$root/Cargo.toml" --target "$target" \
  > "$stage/THIRD-PARTY-LICENSES.txt"
sed "s/@VERSION@/$version/g; s/@EXE@/$exe/g" "$root/scripts/release/INSTALL.txt.in" > "$stage/INSTALL.txt"

sha() { if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
{
  echo "name: taco"
  echo "version: $version"
  echo "target: $target"
  echo "commit: $commit"
  echo "tree: $tree"
  echo "source_date_epoch: $epoch"
  echo "binary_sha256: $(sha "$stage/$exe")"
  echo "cargo_command: cargo build --release --locked -p taco-cli --target $target"
  echo "rustflags: --remap-path-prefix=<source>=/taco --remap-path-prefix=<CARGO_HOME>=/cargo --remap-path-prefix=<sysroot>/lib/rustlib/src/rust=/rustc/<commit>"
  [ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ] && echo "macosx_deployment_target: $MACOSX_DEPLOYMENT_TARGET"
  echo "cflags: -ffile-prefix-map=<source>=/taco -ffile-prefix-map=<CARGO_HOME>=/cargo"
  echo "profile: release (Cargo.toml), debug=0, strip=debuginfo"
  if [ -z "$prebuilt" ]; then
    echo "build_host: $(uname -sm)"
    rustc -vV | sed 's/^/rustc: /'
    echo "cargo: $(cargo -V)"
    cc_bin=${TACO_CROSS_CC:-cc}
    echo "cc: $("$cc_bin" --version 2>&1 | head -1)"
  fi
  [ -n "${TACO_BUILDINFO_EXTRA:-}" ] && cat "$TACO_BUILDINFO_EXTRA"
  echo "model_text: gpt2.hsm aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2 (not in this archive)"
} > "$stage/BUILDINFO"

python3 -I "$root/scripts/release/package.py" --dir "$stage" --format "$fmt" --epoch "$epoch" \
  --out "$out"
echo "binary: $(sha "$stage/$exe")  $exe ($(wc -c < "$stage/$exe" | tr -d ' ') bytes)"
