#!/bin/bash
# Make the model archive of a release: one published model file, its licence and its record
# (docs/design/model-distribution.md §7).
#
#   scripts/release/model-archive.sh <gpt2.hsm> <licence file> [<out dir>]
#
# <licence file> is the licence text of the source weights. For GPT-2 it is the "Modified MIT
# License" of OpenAI, https://github.com/openai/gpt-2/blob/master/LICENSE. The script does not
# download it: give a reviewed copy.
#
# The script refuses a model file whose SHA-256 or size is not the published value. The archive
# is a plain tar (taco-model-gpt2-aaa03eb9.tar), made with the fixed metadata of package.py. Its
# epoch is the commit time of the model release (TACO_MODEL_EPOCH, default 2026-10-18 00:00 UTC).
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
model=${1:?usage: model-archive.sh <gpt2.hsm> <licence file> [<out dir>]}
licence=${2:?usage: model-archive.sh <gpt2.hsm> <licence file> [<out dir>]}
out=${3:-$root/target/dist}
# The published models (model-distribution.md §2). One line for each model: name, SHA-256, size.
known_hash=aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2
known_size=127237568
sha() { if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
h=$(sha "$model")
s=$(wc -c < "$model" | tr -d ' ')
[ "$h" = "$known_hash" ] || { echo "model-archive.sh: SHA-256 $h is not the published gpt2.hsm" >&2; exit 65; }
[ "$s" = "$known_size" ] || { echo "model-archive.sh: size $s is not $known_size" >&2; exit 65; }
[ -s "$licence" ] || { echo "model-archive.sh: the licence file is empty or missing" >&2; exit 66; }

name=taco-model-gpt2-${known_hash:0:8}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/$name"
cp "$model" "$work/$name/gpt2.hsm"
cp "$licence" "$work/$name/LICENSE-GPT2.txt"
cat > "$work/$name/MODEL.txt" <<EOF
name: gpt2
file: gpt2.hsm
sha256: $known_hash
size: $known_size bytes
format: HardShell model file (.hsm), version 1
profiles: text, logs (version 1)
source weights: GPT-2 small (124M) of OpenAI, https://huggingface.co/openai-community/gpt2
source licence: Modified MIT License of OpenAI (LICENSE-GPT2.txt)
conversion: tools/convert (exact conversion, ADR 0014); calibration tools/convert/calib_stats.toml
note: taco checks the SHA-256 of the model file before it uses the file. Do not change this file.
EOF
python3 -I "$root/scripts/release/package.py" --dir "$work/$name" --format tar \
  --epoch "${TACO_MODEL_EPOCH:-1792281600}" --out "$out"
