#!/bin/bash
# Test one release archive: unpack it, check its SHA256SUMS, and fold and unfold fixed inputs.
#
#   scripts/release/smoke.sh <archive.tar.gz> <gpt2.hsm> [<backend>...]
#
# The default backends are "auto cpu". The inputs are made from fixed data (the first 4,096
# bytes of README.md and LICENSE of the archive, 3,000 bytes from SHA-256, 120 log lines, an
# empty file), so the .taco hashes of two machines can be compared. The result table goes to
# stdout and to <archive>.smoke.txt. The exit status is 1 if a check fails.
#
# The test does not need the source tree: it uses only the archive, the model and python3.
set -euo pipefail
archive=$(cd "$(dirname "${1:?usage: smoke.sh <archive> <model> [backend...]}")" && pwd)/$(basename "$1")
model=${2:?usage: smoke.sh <archive> <model> [backend...]}
shift 2
backends=("$@")
[ ${#backends[@]} -gt 0 ] || backends=(auto cpu)
sha() { if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
report="$archive.smoke.txt"
fail=0
say() { echo "$*" | tee -a "$report"; }
: > "$report"

say "archive: $(basename "$archive") $(sha "$archive") ($(wc -c < "$archive" | tr -d ' ') bytes)"
tar -xzf "$archive" -C "$work"
dir=$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1)
( cd "$dir" && if command -v sha256sum > /dev/null; then sha256sum -c --quiet SHA256SUMS; else shasum -a 256 -c --quiet SHA256SUMS; fi ) \
  && say "SHA256SUMS: ok" || { say "SHA256SUMS: FAIL"; fail=1; }
taco="$dir/taco"
say "host: $(uname -sm)"
grep -E '^(commit|tree|target|binary_sha256|rustc: release):' "$dir/BUILDINFO" | sed 's/^/buildinfo: /' | tee -a "$report"
case $(uname -s) in
  Darwin) say "libraries: $(otool -L "$taco" | tail -n +2 | awk '{print $1}' | xargs -n1 basename | tr '\n' ' ')"
          say "signature: $(codesign -dv "$taco" 2>&1 | grep -E '^Signature|^CodeDirectory' | tr '\n' ' ')" ;;
  Linux)  say "libraries: $(ldd "$taco" | awk '{print $1}' | tr '\n' ' ')"
          say "glibc needed: $(objdump -T "$taco" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)" ;;
esac
"$taco" --help > /dev/null 2>&1 && say "help: ok" || { say "help: FAIL"; fail=1; }
if command -v mandoc > /dev/null; then
  mandoc -T lint -W warning "$dir/taco.1" > /dev/null 2>&1 && say "man page: ok (mandoc)" || say "man page: mandoc warnings"
fi

m=$(sha "$model")
say "model: $m"
[ "$m" = aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2 ] || { say "the model is not gpt2.hsm aaa03eb9…a5c2"; fail=1; }

in="$work/in"; mkdir -p "$in"
head -c 4096 "$dir/README.md" > "$in/readme_4k.txt"
head -c 4096 "$dir/LICENSE" > "$in/license_4k.txt"
python3 -I -c 'import hashlib,sys; sys.stdout.buffer.write(b"".join(hashlib.sha256(str(i).encode()).digest() for i in range(94))[:3000])' > "$in/random_3000.bin"
python3 -I -c '
for i in range(120):
    print(f"2026-10-19 12:{i // 60:02d}:{i % 60:02d} INFO [worker-{i % 4}] GET /api/v1/items/{1000 + 7 * i} status={200 if i % 9 else 404} bytes={512 + 13 * i} ms={i % 17}")
' > "$in/server.logs.log"
: > "$in/empty.txt"

say ""
say "backend profile input bytes taco_bytes fold_s unfold_s restored taco_sha256"
for b in "${backends[@]}"; do
  for f in "$in"/*; do
    n=$(basename "$f")
    # NOTE: the profiles logs and auto fold on the CPU on every engine (ADR 0016), so the text
    # inputs use the profile text, which folds on the GPU. The ".logs." input uses auto.
    case $n in *.logs.*) p=auto ;; *) p=text ;; esac
    t0=$(date +%s)
    if ! "$taco" fold "$f" -o "$work/$n.$b.taco" --model "$model" --profile $p --backend "$b" -f 2> "$work/fold.err"; then
      say "$b $n FOLD-FAIL $(tail -1 "$work/fold.err")"; fail=1; continue
    fi
    t1=$(date +%s)
    if ! "$taco" unfold "$work/$n.$b.taco" -o "$work/$n.$b.back" --model "$model" --backend "$b" -f 2> "$work/unfold.err"; then
      say "$b $n UNFOLD-FAIL $(tail -1 "$work/unfold.err")"; fail=1; continue
    fi
    t2=$(date +%s)
    if [ "$(sha "$f")" = "$(sha "$work/$n.$b.back")" ]; then ok=ok; else ok=FAIL; fail=1; fi
    say "$b $p $n $(wc -c < "$f" | tr -d ' ') $(wc -c < "$work/$n.$b.taco" | tr -d ' ') $((t1 - t0)) $((t2 - t1)) $ok $(sha "$work/$n.$b.taco")"
    say "  fold: $(grep -o 'backend [a-z]*' "$work/fold.err" | head -1); unfold: $(grep -o 'backend [a-z]*' "$work/unfold.err" | head -1)"
  done
done
# Each backend must give the same .taco bytes (ADR 0016).
for f in "$in"/*; do
  n=$(basename "$f")
  hs=$(for b in "${backends[@]}"; do [ -f "$work/$n.$b.taco" ] && sha "$work/$n.$b.taco"; done | sort -u | wc -l | tr -d ' ')
  [ "$hs" -le 1 ] || { say "DIFFERENT .taco bytes between backends for $n"; fail=1; }
done
if [ $fail -eq 0 ]; then say "SMOKE: PASS"; else say "SMOKE: FAIL"; fi
exit $fail
