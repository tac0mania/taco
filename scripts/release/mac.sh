#!/bin/bash
# Build and test the macOS release archive on the Mac (docs/design/release.md §3).
#
#   scripts/release/mac.sh [arm64|x86_64] [--no-smoke]
#
# The Mac rules of scripts/mx apply: one command at a time (the same lock), 2 build jobs at low
# priority, a refusal while the disk has less than 8 GB free. The build directory is
# ~/.cache/taco-mac/release-target; the script deletes it after the build, because the Mac has
# little disk. The archive goes to target/dist.
#
# x86_64 needs `rustup target add x86_64-apple-darwin` (about 30 MB, est.), and its smoke test
# runs under Rosetta 2. The model for the smoke test is TACO_MODEL, else
# tools/data/out/gpt2.hsm of the main checkout.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
arch=${1:-arm64}
smoke=1
[ "${2:-}" = --no-smoke ] && smoke=0
case $arch in
  arm64) target=aarch64-apple-darwin ;;
  x86_64) target=x86_64-apple-darwin ;;
  *) echo "usage: scripts/release/mac.sh [arm64|x86_64] [--no-smoke]" >&2; exit 64 ;;
esac
[ "$(uname -s)" = Darwin ] || { echo "mac.sh: run this on the Mac" >&2; exit 64; }
free_gb=$(df -g "$HOME" | awk 'NR==2 {print $4}')
if [ "${free_gb:-0}" -lt 8 ]; then echo "mac.sh: the Mac has only ${free_gb} GB free; refusing to run" >&2; exit 75; fi
lock_dir=$HOME/.cache/taco-mac
mkdir -p "$lock_dir"
export CARGO_TARGET_DIR=$lock_dir/release-target TACO_JOBS=${TACO_JOBS:-2}
/usr/bin/lockf -k "$lock_dir/lock" nice -n 10 "$root/scripts/release/build.sh" "$target"
rm -rf "$CARGO_TARGET_DIR"

if [ $smoke -eq 1 ]; then
  model=${TACO_MODEL:-$(git -C "$root" worktree list --porcelain | awk '/^worktree /{print $2; exit}')/tools/data/out/gpt2.hsm}
  version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$root/crates/taco-cli/Cargo.toml" | head -1)
  # NOTE: --backend auto does not use an integrated GPU (ADR 0016), so the Mac also tests gpu
  # (Metal) explicitly.
  /usr/bin/lockf -k "$lock_dir/lock" nice -n 10 "$root/scripts/release/smoke.sh" \
    "$root/target/dist/taco-$version-$target.tar.gz" "$model" auto cpu gpu
fi
