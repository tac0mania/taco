#!/bin/bash
# Build and test the Windows x86-64 release archive on the Windows side of devbox, from the Mac
# (docs/design/release.md §3).
#
#   scripts/release/windows.sh [--no-smoke]
#
# It uses the per-user toolchain of docs/research/2026-10-19-windows.md in
# C:\Users\rohit\taco-win (Rust 1.99.0 x86_64-pc-windows-gnu and w64devkit GCC). It installs
# nothing. It copies the committed HEAD (git archive) to C:\Users\rohit\taco-win\release-src,
# builds taco.exe there with the flags of build.sh, copies taco.exe back, and packages the zip on
# the Mac with `build.sh --prebuilt`.
#
# WARNING: the Linux disk of devbox is a file on C:. The script refuses to run while C: has less
# than TACO_MIN_WINDOWS_GB (default 20) GB free, and again after the build.
#
# The smoke test runs taco.exe with a clean PATH (Windows directories only), so that a missing
# DLL of w64devkit shows as an error. It folds and unfolds one file on the CPU.
#
# DXC (the DX12 shader compiler) is not in the archive: it waits for the owner decision in
# docs/design/release.md §6.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
WIN=${TACO_WIN_HOST:-umes-win}
W='C:\Users\rohit\taco-win'
target=x86_64-pc-windows-gnu
min=${TACO_MIN_WINDOWS_GB:-20}
win() { ssh "$WIN" "$@" | tr -d '\r'; }
free_gb() { win 'powershell -NoProfile -Command "[int][math]::Floor((Get-PSDrive C).Free/1GB)"' | tr -dc 0-9; }
check_disk() {
  local gb; gb=$(free_gb)
  echo "C: has $gb GB free (limit $min GB)"
  if [ "$gb" -lt "$min" ]; then
    echo "windows.sh: C: has less than $min GB free; the Linux disk of devbox is there. Stop and tell the lead." >&2
    exit 75
  fi
}

check_disk
win "if exist $W\\cargo\\bin\\cargo.exe (if exist $W\\w64devkit\\bin\\gcc.exe echo toolchain ok)" | grep -q "toolchain ok" ||
  { echo "windows.sh: no toolchain in $W; see docs/research/2026-10-19-windows.md" >&2; exit 69; }

commit=$(git rev-parse HEAD)
epoch=$(git log -1 --format=%ct HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "windows.sh: NOTE: the working tree has changes; Windows builds the committed HEAD $commit" >&2
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# The environment of the build. The paths of the source and of CARGO_HOME become /taco and
# /cargo in the binary, as in build.sh. The gcc of w64devkit is static only, so an empty
# libgcc_eh.a satisfies rustc (windows note §8.1).
cat > "$tmp/env-release.cmd" <<EOF
@echo off
set W=$W
set RUSTUP_HOME=%W%\\rustup
set CARGO_HOME=%W%\\cargo
set CARGO_TARGET_DIR=%W%\\target
set CARGO_TERM_COLOR=never
set CARGO_INCREMENTAL=0
if "%TACO_JOBS%"=="" set TACO_JOBS=3
set CARGO_BUILD_JOBS=%TACO_JOBS%
set LIBRARY_PATH=%W%\\lib-extra
set PATH=%W%\\cargo\\bin;%W%\\w64devkit\\bin;%PATH%
set SOURCE_DATE_EPOCH=$epoch
set CARGO_PROFILE_RELEASE_DEBUG=0
set CARGO_PROFILE_RELEASE_STRIP=debuginfo
set RUSTFLAGS=--remap-path-prefix=%W%\\release-src=/taco --remap-path-prefix=%W%\\cargo=/cargo
set CFLAGS=-ffile-prefix-map=%W%\\release-src=/taco -ffile-prefix-map=%W%\\cargo=/cargo
cd /d %W%\\release-src
EOF
sed -i.bak 's/$/\r/' "$tmp/env-release.cmd"
git archive --format=tar --prefix=release-src/ -o "$tmp/release-src.tar" HEAD
win "if exist $W\\release-src rmdir /s /q $W\\release-src"
scp -q "$tmp/release-src.tar" "$tmp/env-release.cmd" "$WIN:C:/Users/rohit/taco-win/"
win "cd /d $W && tar -xf release-src.tar && del release-src.tar && (if not exist lib-extra\\libgcc_eh.a (mkdir lib-extra 2>nul & w64devkit\\bin\\ar.exe rcs lib-extra\\libgcc_eh.a)) && echo copied"
RUN="set \"TACO_JOBS=${TACO_JOBS:-3}\" && $W\\env-release.cmd && start /b /wait /belownormal"

echo "== cargo build --release --locked -p taco-cli --target $target (Windows)"
win "cmd /c \"$RUN cargo build --release --locked -p taco-cli --target $target 2>&1\""
check_disk
exe="$W\\target\\$target\\release\\taco.exe"
scp -q "$WIN:C:/Users/rohit/taco-win/target/$target/release/taco.exe" "$tmp/taco.exe"
{
  echo "build_host: Windows ($WIN), cmd.exe"
  win "cmd /c \"$W\\env-release.cmd && rustc -vV\"" | sed 's/^/rustc: /'
  echo "cargo: $(win "cmd /c \"$W\\env-release.cmd && cargo -V\"")"
  echo "cc: $(win "cmd /c \"$W\\env-release.cmd && gcc --version\"" | head -1)"
} > "$tmp/buildinfo-extra"
TACO_COMMIT=$commit SOURCE_DATE_EPOCH=$epoch TACO_TREE=clean TACO_BUILDINFO_EXTRA="$tmp/buildinfo-extra" \
  scripts/release/build.sh --prebuilt "$tmp/taco.exe" "$target"

if [ "${1:-}" != --no-smoke ]; then
  echo "== smoke test on Windows (clean PATH, CPU)"
  win "if exist $W\\data\\gpt2.hsm echo model ok" | grep -q "model ok" ||
    { echo "windows.sh: no $W\\data\\gpt2.hsm (scripts/wcheck copies it)" >&2; exit 66; }
  cat > "$tmp/smoke.ps1" <<'EOF'
param([string]$W, [string]$Exe)
$ErrorActionPreference = 'Stop'
$env:PATH = "C:\Windows\system32;C:\Windows"
$S = "$W\release-smoke"; New-Item -ItemType Directory -Force $S | Out-Null
Copy-Item -Force "$W\release-src\LICENSE" "$S\license.txt"
& $Exe --help 2>$null; if ($LASTEXITCODE -ne 0) { "help FAIL"; exit 1 }
& $Exe fold "$S\license.txt" -o "$S\license.txt.taco" --model "$W\data\gpt2.hsm" --backend cpu -f
if ($LASTEXITCODE -ne 0) { "fold FAIL"; exit 1 }
& $Exe unfold "$S\license.txt.taco" -o "$S\license.back" --model "$W\data\gpt2.hsm" --backend cpu -f
if ($LASTEXITCODE -ne 0) { "unfold FAIL"; exit 1 }
$a = (Get-FileHash -Algorithm SHA256 "$S\license.txt").Hash; $b = (Get-FileHash -Algorithm SHA256 "$S\license.back").Hash
"taco $((Get-FileHash -Algorithm SHA256 "$S\license.txt.taco").Hash.ToLower()) $((Get-Item "$S\license.txt.taco").Length) bytes"
if ($a -eq $b) { "SMOKE: PASS" } else { "SMOKE: FAIL"; exit 1 }
EOF
  scp -q "$tmp/smoke.ps1" "$WIN:C:/Users/rohit/taco-win/release-smoke.ps1"
  win "powershell -NoProfile -ExecutionPolicy Bypass -File $W\\release-smoke.ps1 -W $W -Exe $exe"
fi
