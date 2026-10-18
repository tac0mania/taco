# Research: the exact model conversion (2026-10-19)

This note records the implementation of ADR 0014. The converter (`tools/convert`) now uses
exact arithmetic only. It makes the same `gpt2.hsm` and the same `conformance/tiny.hsm` as the
float converter of version 1, byte for byte.

## 1. Result

| Check | Result |
|---|---|
| `gpt2.hsm` on the Mac (ARM), exact converter | SHA-256 `aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2`, 127,237,568 bytes (measured) |
| `gpt2.hsm` on devbox (x86), exact converter | the same SHA-256 (measured) |
| `gpt2.hsm` of version 1 (published) | the same SHA-256. No weight and no table entry changed. |
| `conformance/tiny.hsm`, exact converter, Mac and devbox | SHA-256 `eb78b89a4f0f17c7245865aed22f86512a49e0272b7583a831c47e1a401418f8`, 306,368 bytes, the committed file (measured) |
| `python -m convert.test_exact`, Mac and devbox | 12 of 12 pass (measured) |

The file did not change, so the quality did not change. All old `.taco` files name the same
model hash. A frozen reproducer of version 1 is not necessary.

Machines (measured):

| | Mac | devbox |
|---|---|---|
| CPU | Apple M2 (arm64) | AMD Ryzen 7 5800H (x86_64) |
| Python | 3.12.11 | 3.12.15 |
| NumPy | 2.5.4 | 2.5.4 |
| libmpdec (`decimal`) | 4.0.0 | 4.0.0 |
| Time for `gpt2.hsm`, one thread | 70.7 s (version 1: 16.9 s) | 48.1 s |

The Python versions are not the same. The files are the same, because no step depends on a
float result of the platform.

## 2. The float operations of version 1

The table lists every float operation from the float32 weights and the statistics file to the
bytes of the model file. "v1" is the float converter of 2026-10-18.

| # | Step (module) | Float operation in v1 | Risk | Exact replacement |
|---|---|---|---|---|
| 1 | Read the weights (`model.load_gpt2`) | float32 copy and transpose | none (no rounding) | The converter reads the bits (`exact.float_parts`, `uint32` view). It refuses a float64 source. |
| 2 | Read the statistics (`recipe.load_stats`) | decimal text to binary64 (`tomllib`) | none (correctly rounded) | `exact.binary64_of_text`: the nearest binary64 value, calculated with `Fraction` |
| 3 | Range grid (`numerics.round_range`) | 4 square roots of 2, then repeated multiplication | the grid points drift from 2^(j/16) | `exact.GRID`: the 18 v1 grid points as fixed integers. Exact comparisons. |
| 4 | Residual scale (`recipe.residual_scale`) | `row_std_min / 2^12 / 2.0**0.5` | `pow` from the C library | Compare squares: `row_std_min^2 / 2^25` against the square of each grid point |
| 5 | Residual rule | `max / s > 2^28` | division | exact fraction |
| 6 | Activation scales (`Scales.from_ranges`) | `range / 127`, `range / 32767` | division | `Fraction` |
| 7 | Weight scales and `i8` weights (`quantise_rows`) | `max|w| / 127`, `rint(w / scale)` | two roundings before `rint`; a tie can go either way | `R(127 * m_w / (m_max * 2^sh))` in `int64` (`exact.quantise_rows`) |
| 8 | LayerNorm gamma | `max|g| / 32767`, `rint(g / s)` | the same as 7 | the same as 7, with 32,767 |
| 9 | LayerNorm beta | `rint(b / s_out)` | division | `R` of a `Fraction` |
| 10 | Beta correction (`_linear`, `in_err`) | `bq * s_out - b`; `q @ in_err` (float64 matrix product, BLAS); `* w_r`; subtraction | BLAS sum order can depend on the machine | Exact fractions. The product is an `int64` matrix product on 24-bit limbs (`exact.matvec`). |
| 11 | Linear bias | `rint(b' / (s_in * w_r))` | multiplication, division | `R(b / (s_in * w_r) - (W_q . e) / s_in)`, the same number |
| 12 | Rescale multipliers (`quant_mult_rows`) | ratios, the shift test, `rint(r * 2^s)` | a multiplier near 2^31 is sensitive to 1 ulp | `Fraction` |
| 13 | Score rescale | `q * k / (np.sqrt(hd) * delta)` | square root | `exact.quant_mult_over_sqrt`: rational if `hd` is a square, else a comparison of squares |
| 14 | Attention output and LayerNorm rescale | divisions by 2^30 and 2^40 | division | `Fraction` |
| 15 | LayerNorm eps (`ln_eps_units`) | `rint(1e-5 / s^2 * 2^16)` | 1e-5 is not exact in binary64 | `R(2^16 / (100,000 * s^2))`, eps = 1/100,000 exactly |
| 16 | Exponent tables | `det_exp` (21 Taylor terms in float64), `* 2^E`, `rint` | about 1e-16 relative error | `decimal` `exp` at 50 digits, then `R` (`exact.exp_table`) |
| 17 | GELU tables | `det_tanh`, the binary64 constants 0.044715 and sqrt(2/pi), `rint`, clamp | about 1e-15 relative error | `decimal` at 50 digits, then `R` (`exact.gelu_table`) |
| 18 | Tiny calibration (`tiny.calibrate`) | a float forward pass (`np.exp`, BLAS) in each conversion | the C library and BLAS | `--calibrate` writes `tiny_stats.toml` (committed). The conversion only reads it. |
| 19 | Tiny scale of the residual | `min(max / 2^24, row_std_min / 2^12 / np.sqrt(2))` | square root | Compare squares, exact minimum |
| 20 | Tiny weights (`tiny.random_model`) | binary64 additions and multiplications (Irwin-Hall) | none (each IEEE 754 operation is correctly rounded) | Kept. `test_tiny_generator_is_ieee` calculates the same values with `Fraction`. |
| 21 | The file (`hsmfile`) | none (integers) | none | unchanged |

## 3. The exact method

**The tie rule.** `R(x)` is the integer nearest to `x`. If `x` is exactly halfway between two
integers, `R(x)` is the even integer ("round half to even"). Every rounding of the converter
uses `R` (`exact.rhe`, `exact.rhe_vec`). This is also the rule of `np.rint`, which v1 used.

**Weights.** A float32 value is `(-1)^s * m * 2^e` with an integer `m` below 2^24. The bits
give `s`, `m` and `e` exactly. For a row with the largest magnitude `m_max * 2^e_max`, the
`i8` value of an element is `R(127 * m / (m_max * 2^sh))` with `sh = e_max - e`. If `sh` is 9
or more, the value is below 1/2 and `R` gives 0. Otherwise every number is below 2^62, so the
converter uses `int64` arrays. The row maximum comes from an integer comparison of the bit
patterns. A row of zeros has the scale 1. The scale of a row is the fraction `m_max * 2^e_max /
127`, and the later steps use it exactly.

**Scales and rescale constants.** Every activation scale is a fraction `GRID[j] * 2^k / 127`
(or `/ 32767`). Biases, multipliers, shifts, LayerNorm values and eps are exact `Fraction`
calculations, then `R`. These are 287,461 values for GPT-2 small (measured). An irrational
value (`sqrt(2)` in the residual rule, `sqrt(hd)` for a head width that is not a square) is
never rounded directly. The converter compares squares.

**The grid.** v1 calculated the grid points with float operations, and they drift from
2^(j/16) by up to 11 units of 2^-52 (measured). The exact converter keeps the 18 v1 points as
integer constants (`exact.GRID`). Thus the scales did not change. A grid of correctly rounded
2^(j/16) values is also exact, but it would change some multipliers.

**Tables.** The converter calculates `exp` and GELU with `decimal` at 50 digits, in one fixed
context (`exact.context`). `decimal` operations are correctly rounded (General Decimal
Arithmetic), so each machine gets the same digits. pi comes from a series and is compared with
70 published digits. No table entry can be an exact tie (except the exact entries `exp(0)` and
`gelu(0)`), but the converter stops if an entry is nearer than 1e-30 to a tie. Then `PREC`
must increase.

**Statistics.** The converter reads `calib_stats.toml` as data. `exact.no_float_model` makes a
call of the float model an error during the conversion. `gpt2.ranges.json` records the SHA-256
of the weights and of the statistics file.

**Self-check.** `tools/convert/expected.toml` holds the expected SHA-256 and size for each set
of inputs (model, weights, recipe, statistics SHA-256). The converter calculates the file in
memory and compares the hash. On a mismatch, or when no entry exists, it stops with an error
and does not write the file. After a deliberate change, `--new-hash` writes the file and
prints the new entry.

## 4. Why the file did not change

The float path of v1 was near a rounding border in a few places only (all measured, GPT-2
small):

- **Weights.** 4 of 124,337,664 weights are exact ties. No other weight is within 1e-9 of a
  tie.

  | Tensor | Row | Column | Exact value | `R` | v1 |
  |---|---:|---:|---:|---:|---:|
  | `wte` | 3,759 | 54 | -17.5 | -18 | -18 |
  | `h.2.attn_w` | 1,773 | 380 | -49.5 | -50 | -50 |
  | `h.8.attn_w` | 1,592 | 675 | -63.5 | -64 | -64 |
  | `h.11.fc_w` | 1,013 | 373 | -18.5 | -18 | -18 |

  In three rows, `max / 127` is exact in binary64, so the float quotient was the exact tie, and
  `np.rint` used the same rule. In row 1,592 of `h.8.attn_w`, the weight is `-max / 2` and the
  scale was rounded. The float quotient was -63.5 by chance. A different rounding of the scale
  gives -63.

- **Scalar values.** No bias, multiplier or other scalar value is a tie. The nearest is 1.2e-6
  units from a tie (a multiplier near 4.5e8). The float error of v1 at that value is about
  2e-7 units (est.: 4 roundings of 1.1e-16 relative).
- **Tables.** The nearest table entry is 1.5e-5 units from a tie. The float error of v1 is
  about 1e-13 units (est.).

Thus the agreement with v1 is a measured fact for these inputs, not a property of the float
method. A different model (for example Route A) can have a weight like row 1,592 that v1 rounds
in the other direction. The exact converter gives one result in each case.

## 5. Tests

`tools/convert/test_exact.py` (run `uv run --project tools python -m convert.test_exact`, about
5 s):

- The tie rule: halves, negative halves, zero, `rhe_vec`.
- The exact value of float bits: zero, -0.0, subnormals, the largest float32, random values of
  float32 and float64, and the refusal of inf and NaN.
- `quantise_rows`: the ties 1.5, 2.5, 3.5 and their negative values, a row of zeros, the
  largest float32 and its half, rows of subnormals, the cut at `sh = 9`, and a reference with
  `Fraction` on random rows (float32 and float64, `qmax` 127 and 32,767).
- Agreement with the v1 float rules on random float32 weights away from the borders (weights,
  scales, multipliers, grid, residual rule) and on full exponent and GELU tables.
- `quant_mult_over_sqrt` against `decimal` for head widths that are not squares.
- `matvec`, `binary64_of_text`, the tiny generator, the guard and the self-check.

## 6. Make the files again

1. Go to the `tools/` directory.
2. Run `nice uv run python -m convert.gpt2`. The tool prints the self-check result and writes
   `data/out/gpt2.hsm`.
3. Run `nice uv run python -m convert.tiny`. The tool writes `conformance/tiny.hsm`.
4. On devbox, run `scripts/dx exact "cd tools && uv run python -m convert.gpt2"`.

If the self-check fails, do not use the file. Tell the lead.

## 7. Open items

- Route A models and other new inputs have no entry in `expected.toml`. The converter stops
  until a person records the entry with `--new-hash`. The Route A command in
  `2026-10-19-route-a.md` needs that flag the first time.
- `conformance/README.md` says that the tiny calibration rounds a float forward pass. The
  conversion now reads `tools/convert/tiny_stats.toml`. The tiny model did not change, so this
  agent did not edit the README.
