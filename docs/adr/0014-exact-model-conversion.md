# ADR 0014: model conversion is exact and reproducible on every machine

Date: 2026-10-19. Status: accepted (requirement). Decided by: Rohit. Implementation: not started.

## Context

The prediction path is integer-only (ADR 0002), so every machine computes the same predictions
from the same model file. But the converter (`tools/convert`) uses floating-point arithmetic to
make the model file: it divides float weights by scales and rounds them, and it calculates the
tables (GELU, `exp`) with floats.

A float result can depend on the machine, the library version, the order of operations, or
fused multiply-add. A weight on a rounding border (for example 89.004 on one machine and 90.001
on another) can then round to 89 on one machine and to 90 on another. The two model files are
then different models with different hashes.

Today this is safe but not guaranteed:

- The decoder never converts. It loads the published model file and checks its SHA-256 against
  the hash in the container. A different file is refused, never used (`CLAUDE.md` §1 rule 3).
- The Mac and devbox each made `gpt2.hsm` on 2026-10-18 and got the same SHA-256
  `aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2` (measured).

Rohit requires more: data in `.taco` files must stay recoverable for a long time. If the
published model file is lost, a person must be able to make the identical file again, on any
machine, from the published float weights. A float converter cannot guarantee that.

## Decision

The converter is exact. Its output depends only on its input files, never on the machine.

1. **Weights.** Each float32 weight is an exact number `m × 2^e`. The converter reads the bits
   and calculates `round(m × 2^e / scale)` with exact integer arithmetic and one written tie
   rule. Scales are exact fractions or integer pairs that the converter stores.
2. **Rescale constants.** The integer multipliers and shifts come from exact arithmetic on the
   stored scales.
3. **Tables.** GELU, `exp` and other tables are calculated with high precision (for example 100
   decimal digits) and then rounded with the same written rule. No `libm` function decides a
   table entry.
4. **Calibration.** The activation statistics come from a float run of the model. They are data:
   a committed file (`tools/convert/calib_stats.toml`) with its hash. The converter only reads
   them. A new calibration makes a new statistics file and so a new model file, by decision.
5. **Self-check.** The converter calculates the SHA-256 of its output and compares it with the
   published hash of that model. A mismatch stops the converter with an error.
6. **No hidden variation.** One thread, no BLAS in the conversion path, pinned Python and NumPy
   versions in `tools/uv.lock`.
7. **Proof.** The exact converter must give byte-identical files on the Mac (ARM) and on devbox
   (x86). It must also give `aaa03eb9…a5c2` for GPT-2 small, or report each weight that
   differs, with the reason and the quality effect. If the file changes, the old file stays
   available, so that old `.taco` files still unfold.

## Consequences

- A model file can be made again from its published inputs on any machine, at any time. This
  removes the last machine-dependent step between the published weights and the decoded data.
- For archives, keep three things: the `.taco` files, the model file (checked by its hash), and
  the Taco version or the specification. With exact conversion, the model file can also be
  rebuilt from the float weights and the committed calibration file.
- Estimated work: 3 to 6 hours of agent time (est.). Conversion stays fast (vectorised 64-bit
  integer arithmetic).
- This ADR applies to every converter, including the general converter of
  `docs/design/anymodel.md` and the converter of fine-tuned models (Route A).
