#!/bin/bash
# Enforces CLAUDE.md §1 rule 1: no floating-point types in the source of the kernel crates.
# Tests (tests/ and #[cfg(test)] modules in files under tests/) may use floats for reference
# values; src/ may not.
set -euo pipefail
cd "$(dirname "$0")/.."
KERNEL="taco-coder hardshell taco-match"
status=0
for c in $KERNEL; do
  [ -d "crates/$c/src" ] || continue
  if grep -RIn --include='*.rs' -E '\b(f32|f64|f16|bf16)\b' "crates/$c/src"; then
    echo "error: crates/$c/src uses a float type (CLAUDE.md §1 rule 1)" >&2; status=1
  fi
done
# The GPU kernels are in the prediction path too: CUDA C++ and WGSL. Comment lines are skipped,
# because a comment may say "no float".
for f in crates/hardshell-cuda/kernels/*.cu crates/hardshell-gpu/src/shaders/*.wgsl; do
  [ -f "$f" ] || continue
  if grep -nE '\b(float|double|half|__half|__nv_bfloat16|f32|f16)\b' "$f" | grep -vE '^[0-9]+:[[:space:]]*(//|\*|/\*)'; then
    echo "error: $f uses a float type (CLAUDE.md §1 rule 1)" >&2; status=1
  fi
done
exit $status
