#!/bin/bash
# Enforces the comment-tag rules in CLAUDE.md §4.
#   FIXME and XXX may not reach main.
#   A TODO must say who or what it waits for: TODO(phase-N), TODO(#issue) or TODO(name).
set -euo pipefail
cd "$(dirname "$0")/.."
status=0
if grep -RIn --exclude-dir=.venv --exclude-dir=target --exclude-dir=data --include='*.rs' --include='*.toml' --include='*.py' -E '\b(FIXME|XXX)\b' crates tools 2>/dev/null; then
  echo "error: FIXME/XXX must be fixed or turned into a tagged TODO" >&2; status=1
fi
if grep -RIn --exclude-dir=.venv --exclude-dir=target --exclude-dir=data --include='*.rs' --include='*.py' -E '\bTODO\b' crates tools 2>/dev/null | grep -vE 'TODO\((phase-[0-9]+|#[0-9]+|[a-z][a-z0-9-]*)\): '; then
  echo "error: write TODO(phase-N): …, TODO(#issue): … or TODO(owner): …" >&2; status=1
fi
exit $status
