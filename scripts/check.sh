#!/bin/bash
# Everything CI runs, in the order that fails fastest. Run it on devbox:
#   scripts/dx main scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/check-tags.sh
scripts/check-no-floats.sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --no-fail-fast
