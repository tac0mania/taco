# ADR 0001: owner decisions at the start

Date: 2026-10-18. Status: accepted.

Rohit made these decisions before the first build:

1. **Names.** The project is Taco. The command is `taco`, with `taco fold` and `taco unfold`.
   The integer engine is HardShell. The container extension is `.taco`.
2. **Location.** The repository is `~/Documents/apps/taco` on the Mac.
3. **Compute.** devbox (Linux x86, RTX 3060 with 6 GB) does the heavy work and the GPU work.
4. **Documents.** All documents use ASD-STE100 (`docs/STYLE.md`).
5. **Commits.** Commit after each small step that has a meaning.
6. **Working rules.** The rules follow the Hippo working rules, changed for Taco (`CLAUDE.md`).
   A handoff file (`HANDOFF.md`) keeps the state for the next session.

The lead made these decisions, from the research of 2026-10-18:

1. **Core language: Rust.** Integer operations in Rust have defined results (`wrapping_*`,
   `checked_*`, `div_euclid`). There is no undefined overflow.
2. **First model: GPT-2 small.** See plan §6.
3. **Python tools for the work before publication.** Model conversion and float baselines use
   floats, so they stay out of the crates.
