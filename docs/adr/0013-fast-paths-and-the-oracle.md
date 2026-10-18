# ADR 0013: fast paths, op kernels and the reference oracle

Date: 2026-10-19. Status: accepted. Proposed by the simd agent; accepted by the lead.

## Context

HardShell has three fast paths on the CPU (step, prefill, batch) and two GPU backends
(`hardshell-gpu`, `hardshell-cuda`). Each path must give the same bits as the reference path
(`CLAUDE.md` §1 rule 10). Without one fixed oracle and one structure, each backend can drift.

## Decision

1. **The oracle.** `hardshell::gpt2::ReferenceSession` keeps the original, simple code. It is
   the oracle for every other path: CPU fast paths, wgpu, CUDA, and every later backend. A new
   path is correct only if it gives the oracle's weights for every step of the conformance
   windows (per-window SHA-256), and the golden hashes stay unchanged.
2. **One kernel per op.** Every backend implements the same ops: `embed`, `norm`, `linear`,
   `attention`, `logits` (and the table and add steps inside them). Step, prefill and batch are
   built from these ops. A fused kernel is permitted only as an optimisation that a test proves
   equal to the op-level path.
3. **Attention with digits.** The attention output is `sum(p[j] * v[j])` with `p` in fixed point
   (plan §7.5). The fast paths split each `p[j]` into four signed base-256 digits, so the sum
   becomes four exact `i8` dot products, then combine them with shifts. This is exact: the
   digits and the combination give the same integer as the direct product.
4. **No saturating instructions** in any kernel. On x86, use sign extension to `i16` and
   `madd_epi16`. Do not use `maddubs_epi16`, because it saturates.
5. **Run-time selection** of the CPU kernels (NEON, AVX2, portable). Each selection passes the
   same tests.

## Consequences

- The oracle is slow. It is used in tests and in the conformance check, not in `fold` or
  `unfold`.
- The constants `FOLD_CHUNK = 32` and `MAX_LANES = 8` change speed, not bits. They need
  measurements on a quiet machine (`UNVERIFIED` until then).
- `FreqTable::from_weights` (one `u128` division for each symbol) is about 5% of unfold time now
  (measured). An exact faster form is future work in `taco-coder`.
