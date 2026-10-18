# Research: the integer GPT-2 model and its quality (2026-10-18)

This note records the conversion of GPT-2 small to the HardShell model file `gpt2.hsm`, and the
first measurement of the integer model against the float model. All numbers are measured on
2026-10-18 on the Mac, unless a number has a different label.

## 1. Results in short

- `tools/convert` makes `gpt2.hsm` from the published weights in 2.6 s (measured). The file has
  127,237,568 bytes. Its SHA-256 is
  `aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2`.
- On the 19 evaluation windows (19,456 steps), the integer model gives **3.067 bits per byte**
  with the plan §5.1 tables. The float GPT-2 gives 3.380. On the 16 files that are not binary,
  the integer model gives **1.871** and the float model 1.867: a loss of **+0.2 %**.
- The loss for each file is in the range −30.4 % to +6.6 %. The integer model is worse on 10 of
  19 files, by 6.6 % or less. It is better on the binary files, on logs, and on files with tab
  characters.
- The rule of ADR 0004 section 7 ("the largest value maps to 127") does not work for GPT-2. It
  gives 5.99 bits per byte (+67 %, ideal size) on the evaluation windows. Two fixes, both
  selected on the calibration windows only, remove this loss (section 7).
- The NumPy integer simulation `convert.intsim` gives the same sizes as HardShell on the 11
  calibration windows and on the 19 evaluation windows (section 8). Thus the converter and
  HardShell agree.
- `eval_bits` gives the same bits and the same weight digests with 1 thread and with 6
  threads.

## 2. Setup

| Item | Value |
|---|---|
| Machine | Apple M2, 8 GB. Other agents used the CPU at the same time. |
| Weights | `openai-community/gpt2` `model.safetensors`, SHA-256 in `tools/baseline/manifest.toml` |
| Converter | `tools/convert` (Python 3.12, NumPy). Commands in section 10. |
| Engine | HardShell, `crates/hardshell`, release build, example `eval_bits` |
| Constants | ADR 0004: `K` = 48, `J` = 8, `P` = 30, exponent tables of 8,192 entries with a unit of 2^-8, `i16` GELU tables |
| Windows | The first 1,024 tokens of each of the 19 phase 0 corpus files (the phase 1 part A set). Model input: `<|endoftext|>`, then 1,023 tokens. |
| Float reference | `tools/baseline` GPT-2, float32, full sequence, as phase 0 |

## 3. Conversion method

The converter obeys ADR 0004 and the tensor table of `crates/hardshell/src/gpt2/mod.rs`. All
float work happens in the converter, one time.

1. **Weights.** Each matrix is `i8`, symmetric, with one scale for each output row:
   `scale = max|row| / 127`, `q = round(w / scale)`. The Conv1D weights of GPT-2 change to the
   `[out, in]` form. `wte` is the token embedding and also the output projection (tied).
2. **Biases.** A bias is `i32` in accumulator units: `round(b / (s_in * scale_row))`.
3. **Rescales.** Each rescale has one multiplier `m` for each row and one shift `s` for the
   tensor. The shift puts the largest multiplier in `[2^30, 2^31)`. The shifts are 19 to 83.
4. **LayerNorm.** `gamma` is `i16` with one scale for each LayerNorm. `beta` is `i32` at the
   output scale. `eps` is 2,366,912,989 in the units of `var * 2^16`.
5. **Beta correction (fix 2, section 7).** HardShell adds `beta` after the rescale (plan §7.3
   step 4). Thus the rounding of `beta` is a constant error in each channel, up to half a unit.
   The converter removes this error in the bias of the next linear layer (`c_attn` after
   `ln_1`, `c_fc` after `ln_2`): `b' = b - W_q * err`. `ln_f` has no next bias, so its error
   stays.
6. **Tables.** The GELU tables, `attn_exp` and `logits_exp` use only IEEE 754 basic operations
   (`convert.numerics`). The tool does not call `exp` or `tanh` of the platform library.
7. **Vocabulary and merges.** The vocabulary is the GPT-2 byte-level BPE vocabulary. Token
   50256 holds the text `<|endoftext|>`; the encoder never makes it. The 50,000 merges go into
   the new merges section (ADR 0004 addendum A). Merge rank `i` makes token `256 + i`
   (measured, all 50,000 merges).

The converter does not run the model. It reads the calibration statistics from the committed
file `tools/convert/calib_stats.toml`. Two runs gave the same SHA-256 (measured, same machine).
A run on devbox is not done (section 9).

## 4. Calibration set

`tools/convert/calibration.toml` records the URL, the size and the SHA-256 of each file. No file
is in the phase 0 corpus.

| Source | Class | Windows |
|---|---|---:|
| Gutenberg 1342 (Austen), byte 993 and byte 300,000 | English prose, CRLF line ends | 4 |
| Gutenberg 2000 (Cervantes), byte 904 and byte 1,000,000 | Spanish prose, CRLF line ends | 3 |
| CPython v3.12.0 `Objects/listobject.c` | C source | 2 |
| Go 1.22.0 `src/strings/strings.go` | Go source (tabs) | 1 |
| splitmix64, seed 20261018, 1,200 bytes | random bytes | 1 |
| **All** | | **11 windows, 11,264 tokens** |

`python -m convert.calibrate` runs the float GPT-2 (float32) over these windows in 46 s
(measured). For each activation tensor (a "site") it records the largest absolute value, four
high quantiles of the absolute value, and quantiles over the channels of the largest value of
each channel. For the residual stream it records the standard deviation of each row.

Main findings (measured, calibration windows):

- The largest residual value is 3,055, at position 0 (the `<|endoftext|>` token). At the other
  positions the largest value is 463. The smallest row standard deviation is 0.096.
- `ln_f` has a few outlier channels (496, 430, 36) with values up to 295. The median channel
  has a largest value of 2.25. The `wte` columns of these channels are small (standard
  deviation 0.001, against 0.144 for all of `wte`).
- The GELU outputs have rare large values: 61.9 in layer 2 (position 0, channel 666), 43.8 in
  layer 11. The 99.99 % quantile is 1.4 to 6.6.

## 5. Scales

**Residual.** The scale is 1.664e-5. The smallest row standard deviation is then 2^12.50 units
and the largest value is 2^27.45 units (the clamp is at 2^30).

NOTE: ADR 0004 section 1 asks for the largest value "near 2^24" and a row standard deviation
of 2^12 units or more. For GPT-2 both are not possible: with the largest value at 2^24, the
smallest row standard deviation is 2^9.0 units. The converter obeys the standard deviation
rule, because ADR 0004 section 2 measured the LayerNorm error against it. The largest value
is 1.5 times below 2^28 and about 6 times below the clamp. Position 0 is the same token in every
block, so its largest value is the same in every block.

**Activations.** A recipe selects the range of each site (`tools/convert/recipe.py`). Each range
is rounded up to the grid 2^(k/16), so a calibration with slightly different float bits gives
the same file.

| Recipe | Rule |
|---|---|
| `max` | ADR 0004 section 7: the largest calibration value maps to 127 (or 32,767). |
| `clip` (published) | As `max`, but `ln_f` uses the median over the channels of the channel maximum (range 2.28), and each GELU output uses the 90th percentile over the channels (ranges 2.6 to 3.5). Larger values saturate. |

`gpt2.hsm` uses `clip` with the beta correction. `tools/data/out/gpt2.ranges.json` records every
scale and shift.

## 6. Quality

Command: `eval_bits` on the 19 evaluation windows, 6 threads (section 10). These windows were
not used to select the recipe. "Ideal" is `-log2 p` of the actual tokens. "Table" is
`-log2(f / 2^32)` with the plan §5.1 tables. The range coder gives the table size to about
0.02 % (phase 0). The table size is the main number. All values are bits per byte (measured).
"Loss" is the change of the integer size against the float size.

| Window | Bytes | Float ideal | Integer ideal | Loss | Float table | Integer table | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| alice29.txt | 3,365 | 2.114 | 2.042 | −3.4 % | 2.025 | 2.042 | +0.8 % |
| asyoulik.txt | 3,058 | 3.656 | 2.679 | −26.7 % | 2.622 | 2.666 | +1.6 % |
| cp.html | 2,494 | 1.468 | 1.362 | −7.2 % | 1.464 | 1.362 | −7.0 % |
| fields.c | 2,823 | 2.447 | 1.999 | −18.3 % | 2.195 | 1.998 | −9.0 % |
| grammar.lsp | 2,130 | 1.394 | 1.401 | +0.5 % | 1.394 | 1.401 | +0.5 % |
| kennedy.xls | 1,121 | 17.296 | 15.146 | −12.4 % | 17.041 | 15.146 | −11.1 % |
| lcet10.txt | 2,170 | 2.397 | 2.328 | −2.9 % | 2.333 | 2.327 | −0.3 % |
| plrabn12.txt | 3,681 | 2.056 | 2.006 | −2.5 % | 2.043 | 2.006 | −1.8 % |
| ptt5 | 1,024 | 35.523 | 22.043 | −37.9 % | 31.671 | 22.042 | −30.4 % |
| sum | 1,032 | 22.428 | 20.073 | −10.5 % | 22.389 | 20.071 | −10.4 % |
| xargs.1 | 3,017 | 1.329 | 1.417 | +6.6 % | 1.329 | 1.417 | +6.6 % |
| pg22367.txt (German) | 2,081 | 3.393 | 3.494 | +3.0 % | 3.345 | 3.494 | +4.5 % |
| pg4650.txt (French) | 2,485 | 2.671 | 2.688 | +0.6 % | 2.641 | 2.688 | +1.8 % |
| pg24264.txt (Chinese) | 1,443 | 3.759 | 3.798 | +1.0 % | 3.706 | 3.798 | +2.5 % |
| textwrap.py | 3,068 | 1.299 | 1.231 | −5.2 % | 1.299 | 1.231 | −5.2 % |
| option.rs | 2,929 | 1.133 | 1.206 | +6.4 % | 1.133 | 1.206 | +6.4 % |
| twitter.json | 1,979 | 1.514 | 1.598 | +5.5 % | 1.514 | 1.598 | +5.5 % |
| HDFS_2k.log | 2,709 | 1.384 | 1.417 | +2.4 % | 1.377 | 1.417 | +2.9 % |
| Apache_2k.log | 2,824 | 0.731 | 0.556 | −24.0 % | 0.625 | 0.556 | −11.2 % |
| **16 files, not binary** | 42,256 | 1.984 | 1.872 | −5.7 % | 1.867 | **1.871** | **+0.2 %** |
| **3 binary files** | 3,177 | 24.838 | 18.969 | −23.6 % | 23.494 | 18.968 | −19.3 % |
| **All 19 files** | 45,433 | 3.582 | 3.068 | −14.4 % | 3.380 | **3.067** | **−9.3 %** |

No token of the integer model had a weight of 0 on these windows.

Findings:

- **The integer model is not only a copy of the float model with noise.** The saturation of
  the `clip` recipe changes the model. The float GPT-2 with the same saturation and no other
  quantisation gives 3.017 bits per byte ideal on all 19 files, and 1.831 on the 16 files that
  are not binary (measured, `convert.model` with a clamp). Thus the pure quantisation loss is
  +1.7 % (all) and +2.3 % (not binary), ideal size, against that model.
- The saturation removes most of the very small probabilities of GPT-2 (tab and carriage
  return after text, phase 0 §5). Thus the gain is largest on asyoulik.txt (tabs), fields.c,
  Apache_2k.log (CR) and the binary files. With the §5.1 tables, the float model already loses
  less on these tokens, so the table gain is smaller than the ideal gain.
- The loss is largest on xargs.1, option.rs and twitter.json (+5.5 % to +6.6 %), and on German
  (+4.5 %). The cause is not measured for each file.

## 7. Attempts

All recipe choices used the calibration windows only. The evaluation windows were measured
after the choice. The table gives ideal bits per byte, from `convert.intsim` (equal to
HardShell, section 8). "w = 0" is the count of tokens with weight 0; the ideal size does not
include them, so the `max` rows are too small.

| Recipe | Calibration (tuned) | Change | Evaluation (held out) | Change | w = 0 (eval) |
|---|---:|---:|---:|---:|---:|
| Float GPT-2 | 2.768 | | 3.582 | | |
| `max`, no beta correction (ADR 0004 rule) | 5.210 | +88.2 % | 5.991 | +67.2 % | 257 |
| `max` | 5.237 | +89.2 % | 5.927 | +65.4 % | 349 |
| `clip`, no beta correction (fix 1) | 2.795 | +1.0 % | 3.313 | −7.5 % | 0 |
| **`clip` (fix 1 and fix 2, published)** | **2.594** | **−6.3 %** | **3.068** | **−14.4 %** | 0 |

**Diagnosis.** `convert.fq` quantises one part at a time in the float model (fake
quantisation, 3 calibration windows, `max` recipe). Against the float model: the weights alone
+1.1 %, with `ln_1` +1.8 %, `ln_2` +2.4 %, the GELU outputs +8.5 %, `ln_f` +114 %. Each other
site added less than 0.2 percentage points to the 1.1 % of the weights (measured). The cause is the outlier channels of section 4: with
a range of 295, one `ln_f` unit is 2.3, and the typical channel (largest value 2.25) has only
one or two levels.

**Fix 1: a range for each site kind from the channel statistics.** For `ln_f` and the GELU
outputs, the range comes from the typical channel, and the outlier channels saturate. The
`wte` columns of the `ln_f` outlier channels are almost zero after the `i8` quantisation, so
their saturation costs little. Variants on the calibration windows (ideal, with fix 2): GELU
at the 99.99 % quantile 2.608, at the channel 50th percentile 2.700, at the channel 99th
percentile 2.600; `ln_f` at the channel 90th percentile 2.607. The selected `clip` gives 2.594.

**Fix 2: beta correction.** `convert.intsim` gave 2.795 on the calibration windows, but the
fake quantisation of the same recipe gave 2.573. A comparison of each site against the float
model showed an extra error after each LayerNorm (for example `h.0.ln_1`: 0.041 against 0.033
relative error, positions 1 to 1,023). The cause is the second rounding of `beta` (section 3,
item 5). The correction changed 2.795 to 2.594 on the calibration windows.

## 8. Checks

| Check | Result (measured) |
|---|---|
| `convert.intsim` against HardShell, recipe `clip` without fix 2, 11 calibration windows | 2.7947 bits per byte on both (the 4 digits that `convert.tune` prints); the first window 2,912.711705 bits on both (6 decimals) |
| `convert.intsim` against HardShell, recipe `clip`, 19 evaluation windows | 3.0677 (`convert.intsim`, 4 digits) and 3.067702 (HardShell) bits per byte |
| `eval_bits` with 1 thread against 6 threads | The same bits for each window, the same f64 bits of the totals (`410103773bef93a3`), and the same SHA-256 of all weights (`c9947765…50e6`) |
| Python writer against `HsmWriter` | The same bytes for `conformance/tiny.hsm` (test `hsm_writer_makes_the_same_bytes_as_the_python_writer`) |
| Converter run two times | The same SHA-256 |
| `cargo test -p hardshell` | 75 tests pass, including 14 new tests of the merges section |

## 9. Timing

| Part | Time (measured) |
|---|---|
| Calibration, 11 windows, float32 | 46 s |
| Conversion to `gpt2.hsm` | 2.6 s |
| Tiny model | 1.2 s |
| `convert.intsim`, 19 evaluation windows | 101 s (5.3 s for each window) |
| HardShell `eval_bits`, 19,456 steps, 1 thread | 353.8 s: 55.0 steps per second |
| HardShell `eval_bits`, 19,456 steps, 6 threads | 124.4 s: 156.4 steps per second |

The mean context length of a window is about 512 tokens, so a step is faster than the 41.2
steps per second of the HardShell benchmark at a full context of 1,024 tokens.

## 10. Commands

Run these commands from `tools/`, unless a command says otherwise.

1. `nice uv run python -m convert.calibrate --fetch`: downloads the calibration set and
   writes `convert/calib_stats.toml`.
2. `nice uv run python -m convert.gpt2 --recipe clip`: writes `data/out/gpt2.hsm`.
3. `nice uv run python -m convert.evalset`: writes `data/out/eval_windows.txt` and the float
   sizes `data/out/eval_float.json`.
4. From the repository root: `scripts/mx cargo run --release -p hardshell --example eval_bits
   -- tools/data/out/gpt2.hsm tools/data/out/eval_windows.txt --threads 6 --digest`.
5. `nice uv run python -m convert.report`: prints the table of section 6.
6. `nice uv run python -m convert.tune --set calib RECIPE ...`: measures recipes with the
   integer simulation. `nice uv run python -m convert.fq`: the fake quantisation.
7. `nice uv run python -m convert.tiny`: writes `conformance/tiny.hsm`.

## 11. Limits

- The evaluation covers the first window of each file only (19,456 steps). A larger held-out
  set can change the per-file numbers.
- The calibration set has 11 windows. The recipe has few free choices (two statistics), but
  the selection used these 11 windows.
- The gain over the float model comes from the saturation, not from the quantisation. A float
  model with the same saturation is better than the integer model by 1.7 % (ideal).
- `ln_f` keeps the beta rounding error, because the logits have no bias.
- The converter is not run on devbox. The cross-machine hash of `gpt2.hsm` is not checked.
- The causes of the larger losses on xargs.1, option.rs, twitter.json and German are not
  measured.

## 12. Requests to the lead

- **ADR 0004 section 7** says that the largest calibration value maps to 127. For GPT-2 this
  gives +67 % (section 7). Please change the rule in an ADR: the converter selects each range
  by a recipe that a measurement supports (here `clip`).
- **ADR 0004 section 1**: "near 2^24" and "2^12 units or more" cannot both hold for GPT-2.
  Please record the rule of section 5: the standard deviation rule first, the largest value at
  or below 2^28.
- **ADR 0004 section 2**: please record the beta correction (section 3, item 5) as a converter
  rule.
- `docs/ARCHITECTURE.md` names `tools/tinymodel` for the tiny model. It is now
  `tools/convert/tiny.py`.
- Run `python -m convert.gpt2` on devbox and compare the SHA-256 with section 1.
