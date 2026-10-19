#!/bin/bash
# Make the demo folder and its zip: the programs for macOS arm64, Linux x86-64 and Windows
# x86-64, the model file, the launchers and the sample. The user unpacks it and runs one
# command (demo/README.md). Nothing is compiled or downloaded on the user's machine.
#
#   scripts/release/demo.sh <gpt2.hsm> <gpt-2 licence> [<dist dir>]
#
# <gpt-2 licence> is a reviewed copy of the licence of the GPT-2 weights (the "Modified MIT
# License" of OpenAI, https://github.com/openai/gpt-2/blob/master/LICENSE). The demo carries it
# as MODEL-LICENSE.txt, because the demo carries the converted weights.
#
# <dist dir> (default target/dist) holds the release archives that mac.sh, linux.sh and
# windows.sh made. The output is target/demo/taco-demo-<version>/ and
# target/demo/taco-demo-<version>.zip. Both are ignored by git: the model file is 127 MB.
set -euo pipefail
cd "$(dirname "$0")/../.."
model=${1:?usage: scripts/release/demo.sh <gpt2.hsm> <gpt-2 licence> [<dist dir>]}
licence=${2:?usage: scripts/release/demo.sh <gpt2.hsm> <gpt-2 licence> [<dist dir>]}
dist=${3:-target/dist}
[ -s "$licence" ] || { echo "demo.sh: the licence file $licence is empty or missing" >&2; exit 66; }
expected=aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2
actual=$(shasum -a 256 "$model" | cut -d' ' -f1)
if [ "$actual" != "$expected" ]; then
  echo "demo.sh: $model has SHA-256 $actual, not $expected; refusing" >&2; exit 65
fi
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' crates/taco-cli/Cargo.toml | head -1)
name="taco-demo-$version"
out=target/demo
stage="$out/$name"
rm -rf "$stage" "$out/$name.zip"
mkdir -p "$stage/bin" "$stage/samples"

# Each platform: the folder name in the demo, the release target, the program name.
missing=0
for row in "macos-arm64 aarch64-apple-darwin taco tar.gz" \
           "linux-x86_64 x86_64-unknown-linux-gnu taco tar.gz" \
           "windows-x86_64 x86_64-pc-windows-gnu taco.exe zip"; do
  set -- $row
  archive="$dist/taco-$version-$2.$4"
  if [ ! -f "$archive" ]; then
    echo "demo.sh: $archive is missing (build it with scripts/release/)" >&2; missing=1; continue
  fi
  tmp=$(mktemp -d)
  if [ "$4" = zip ]; then unzip -q "$archive" -d "$tmp"; else tar -xzf "$archive" -C "$tmp"; fi
  mkdir -p "$stage/bin/$1"
  cp "$tmp/taco-$version-$2/$3" "$stage/bin/$1/$3"
  cp "$tmp/taco-$version-$2/BUILDINFO" "$stage/bin/$1/BUILDINFO"
  [ -f "$stage/THIRD-PARTY-LICENSES.txt" ] || cp "$tmp/taco-$version-$2/THIRD-PARTY-LICENSES.txt" "$stage/"
  rm -rf "$tmp"
done
[ "$missing" = 0 ] || { echo "demo.sh: not all programs are present; refusing" >&2; exit 66; }

cp demo/README.md demo/taco demo/try "$stage/"
# NOTE: cmd.exe needs CRLF line ends for labels and goto.
for f in taco.cmd try.cmd; do perl -pe 's/\r?\n/\r\n/' "demo/$f" > "$stage/$f"; done
cp demo/samples/sample.txt "$stage/samples/"
cp LICENSE "$stage/LICENSE"
cp "$model" "$stage/gpt2.hsm"
cp "$licence" "$stage/MODEL-LICENSE.txt"
chmod 0755 "$stage/taco" "$stage/try" "$stage/bin/macos-arm64/taco" "$stage/bin/linux-x86_64/taco"
(cd "$stage" && find . -type f ! -name SHA256SUMS | sort | sed 's|^\./||' | xargs shasum -a 256 > SHA256SUMS)
(cd "$out" && zip -qrX "$name.zip" "$name")
echo "demo.sh: $stage"
echo "demo.sh: $out/$name.zip ($(du -h "$out/$name.zip" | cut -f1))"
