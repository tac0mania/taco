# ADR 0014: model conversion is exact and reproducible on every machine

Date: 2026-10-19. Status: accepted, implemented (2026-10-19). Decided by: Rohit. Implementation:
the exact agent, research note `docs/research/2026-10-19-exact-conversion.md`.

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

## Implementation (2026-10-19)

The converter in `tools/convert` obeys points 1 to 7. The research note
`docs/research/2026-10-19-exact-conversion.md` has the list of the float operations of
version 1 and the measurements.

1. **The tie rule.** Every rounding is `R`: the nearest integer, and the even integer for an
   exact tie (`exact.rhe`). This is the rule of `np.rint` in version 1.
2. **Weights.** `exact.quantise_rows` reads the float32 bits (`uint32` view) and calculates
   `R(127 * m / (m_max * 2^sh))` with `int64` arrays. The scale of a row is the exact fraction
   `max|row| / 127`.
3. **Rescale constants, biases, eps and the beta correction.** Exact `Fraction` calculations,
   then `R` (`convert.build`). The beta correction uses an exact integer matrix product
   (`exact.matvec`), not a float matrix product. The eps of GPT-2 is 1/100,000 exactly. An
   irrational ratio (`sqrt(2)`, `sqrt(hd)`) is never rounded directly: the converter compares
   squares.
4. **The range grid.** The 18 grid points of version 1 are fixed integer constants
   (`exact.GRID`). Thus the scales of version 1 did not change.
5. **Tables.** `decimal` at 50 digits in one fixed context, then `R`. The converter stops if an
   entry is nearer than 1e-30 to a tie (`exact.TIE_MARGIN`).
6. **Statistics.** The converter reads `calib_stats.toml` (GPT-2) and `tiny_stats.toml` (the
   tiny model) as exact binary64 values. `exact.no_float_model` makes a call of the float model
   an error during the conversion. `gpt2.ranges.json` records the SHA-256 of each input.
7. **Self-check.** `tools/convert/expected.toml` holds the expected SHA-256 and size for each
   set of inputs. The converter stops with an error and does not write the file on a mismatch
   or when no entry exists. `--new-hash` accepts a deliberate change and prints the new entry.
8. **No hidden variation.** One thread. The conversion has no float matrix product, so no BLAS.
   NumPy is pinned in `tools/uv.lock`.

Proof (measured):

- The Mac (arm64, Python 3.12.11) and devbox (x86_64, Python 3.12.15) make `gpt2.hsm` with
  the SHA-256 `aaa03eb9…a5c2`. This is the published file of version 1, so no weight and no
  table entry changed, and old `.taco` files still unfold.
- Both machines make `conformance/tiny.hsm` with the SHA-256 `eb78b89a…18f8`, the committed
  file.
- 4 of 124,337,664 GPT-2 weights are exact ties. Version 1 rounded them as `R` does, one of
  them by chance (research note, section 4).
- The conversion of GPT-2 small takes 70.7 s on the Mac and 48.1 s on devbox (version 1:
  16.9 s on the Mac). `docs/research/2026-10-18-integer-gpt2.md` gives 2.6 s for the
  converter of 2026-10-18. This ADR did not measure that converter again, so the two numbers
  are not comparable.
