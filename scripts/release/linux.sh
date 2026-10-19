#!/bin/bash
# Build and test the Linux x86-64 release archive on devbox, from the Mac (release.md §3).
#
#   scripts/release/linux.sh [--no-smoke]
#
# scripts/dx copies the working tree (without .git) to devbox, so this script reads the commit,
# its time (SOURCE_DATE_EPOCH) and the tree state on the Mac and gives them to build.sh. The
# lane is "release" (build directory ~/targets/taco-release on devbox). The archive comes back
# to target/dist on the Mac.
#
# The binary loads libcuda (CUDA) and the Vulkan or GL loader (wgpu) at run time only, so it
# needs no CUDA toolkit and no GPU library at build time. It needs the glibc version that the
# smoke test prints, or later (built on Ubuntu 24.04).
#
# The smoke test uses the model of the main devbox mirror (~/mirror/taco/tools/data/out). It
# tests the backends auto, cpu and cuda. While a timing measurement holds ~/scratch/timing.lock,
# it tests cpu only, so that it does not use the GPU of the measurement.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
commit=$(git rev-parse HEAD)
epoch=$(git log -1 --format=%ct HEAD)
if [ -n "$(git status --porcelain)" ]; then tree=dirty; else tree=clean; fi
target=x86_64-unknown-linux-gnu
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' crates/taco-cli/Cargo.toml | head -1)
name=taco-$version-$target
mirror=$(basename "$root")

scripts/dx release env TACO_COMMIT=$commit SOURCE_DATE_EPOCH=$epoch TACO_TREE=$tree \
  scripts/release/build.sh $target

if [ "${1:-}" != --no-smoke ]; then
  # shellcheck disable=SC2016
  scripts/dx release 'if flock -n $HOME/scratch/timing.lock true 2>/dev/null; then b="auto cpu cuda"; else b=cpu; echo "smoke: a timing measurement holds the lock; cpu only" >&2; fi;' \
    scripts/release/smoke.sh target/dist/$name.tar.gz '$HOME/mirror/taco/tools/data/out/gpt2.hsm' '$b'
fi
mkdir -p target/dist
rsync -az "devbox:mirror/$mirror/target/dist/$name.tar.gz" "devbox:mirror/$mirror/target/dist/$name.tar.gz.smoke.txt" target/dist/ 2> /dev/null ||
  rsync -az "devbox:mirror/$mirror/target/dist/$name.tar.gz" target/dist/
( cd target/dist && shasum -a 256 taco-*.tar.gz taco-*.zip 2> /dev/null > SHA256SUMS || true )
echo "copied: target/dist/$name.tar.gz"
