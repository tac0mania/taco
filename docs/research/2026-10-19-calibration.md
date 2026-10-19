# Research: `i16` activations for confident models (2026-10-19)

This note records the calibration work for the log fine-tune of Route A (`distil_logs`,
distilgpt2 with 6 blocks, `docs/research/2026-10-19-route-a.md`). The question: why does the
integer model lose much more against float than GPT-2 does, and which change removes the loss?
All numbers are measured on 2026-10-19, unless a number has a different label. "bpb" is bits
per byte. "Loss" is the integer size against the float size of the same windows (ideal sizes,
`-log2 p`, as in the Route A note). The decision is ADR 0004 addendum B.

## 1. Results in short

- **The cause.** Fake quantisation on the val windows (24 windows of the 12 training systems,
  no evaluation data) put most of the loss at two sites: the GELU outputs (+9.3 %) and the
  output of `ln_f` (+8.3 %). With 127 levels, an `i8` range cannot hold the outlier channels
  and the typical channels of this model at the same time. A second, smaller cause is the two
  roundings of the integer LayerNorm (about 3 %, section 6).
- **The fix: hsm format version 2** (ADR 0004 addendum B). `i16` outputs at the GELU tables
  and at `ln_f`, and a LayerNorm beta with 8 more bits that HardShell adds before the one
  rounding (`ln_beta_frac = 8`). The recipe `wide` uses the largest calibration value as the
  range of every site. Version 1 files do not change: `gpt2.hsm` is still `aaa03eb9…a5c2`.
- **The fine-tuned model, before and after** (HardShell `eval_bits`, table size in bpb, 373
  Route A evaluation windows; loss of the ideal size against float):

  | Set | Float | `clip` (v1) | Loss | `wide` (v2) | Loss |
  |---|---:|---:|---:|---:|---:|
  | tail (12 training systems) | 0.2409 | 0.2810 | +16.7 % | **0.2425** | **+0.7 %** |
  | heldout (4 unseen systems) | 0.7009 | **0.6844** | −2.3 % | 0.6982 | −0.3 % |
  | heldout and tail | 0.4216 | 0.4395 | +4.3 % | **0.4215** | **+0.0 %** |
  | all 373 windows | 0.4046 | 0.4313 | +6.6 % | **0.4045** | **−0.0 %** |

- **The gate (integer loss of 2 % or less) is reached** on the tails (+0.7 %) and on the
  held-out systems (−0.3 %). The worst system is the Android tail (+3.0 %).
- **Caution: on the held-out systems, `wide` is 2.0 % larger than `clip`** (0.6982 against
  0.6844 bpb). The `clip` model was smaller than the float model there: its saturation of
  `ln_f` makes the model less confident, and that helps on unseen systems (Route A note,
  section 6). `wide` follows the float model.
- **GPT-2 does not get worse.** `gpt2.hsm` keeps `clip` (version 1). The rule `wide` is worse
  for GPT-2 (section 7), so it is not for GPT-2.

## 2. Diagnosis (val windows, fake quantisation)

Tool: `convert.fq_gpu` on the RTX 3060 (float32 with the quantisation steps of HardShell, site
by site). Windows: `val_windows.txt`, 2 windows from the val split of each training system (24
windows, 61,047 bytes). The val split selects the checkpoint in training. It is not evaluation
data. Float: 0.2238 bpb.

| Quantised parts | Integer bpb | Loss |
|---|---:|---:|
| `clip`, all sites | 0.2695 | +20.4 % |
| `clip`, all sites, `i16` GELU outputs and `i16` `ln_f`, `clip` ranges | 0.2667 | +19.2 % |
| `i16` GELU outputs and `ln_f`, range `max` (the recipe `wide`) | **0.2273** | **+1.55 %** |
| as `wide`, `ln_f` range `chan_q99` | 0.2274 | +1.6 % |
| as `wide`, `ln_f` range `chan_q90` | 0.2362 | +5.5 % |
| as `wide`, `ln_f` `i8` (`chan_q50`) | 0.2450 | +9.5 % |
| as `wide`, GELU outputs `i8` (`chan_q90`) | 0.2485 | +11.0 % |

The two `i16` sites are both necessary. The range must be `max`: the saturation of the outlier
channels causes the loss, not the number of levels alone. Leave-one-out on `wide`: no other
site kind costs more than 1 % (the largest: `ln_2`, 1.0 %).

The same rule for GPT-2 on its calibration windows (fake quantisation, 11 windows): `clip`
2.5723 bpb, `wide` 2.6544 bpb (+3.2 % against `clip`). Thus GPT-2 keeps `clip`.

## 3. The engine change (hsm format version 2)

ADR 0004 addendum B has the rules. In short:

- The header constant `ln_f_bits = 16` gives `i16` outputs of `ln_f`. A GELU table with the
  dtype `i16` gives `i16` GELU outputs.
- An `i16` activation enters the next matrix product with `i8` weights and `i64` accumulators
  (`kernels::matvec_i8_i16`, `Kernels::matmul_i8_i16`; bound `|acc| <= 2^38`).
- The header constant `ln_beta_frac = F` stores every LayerNorm beta with `F` more bits, and
  HardShell adds it before the one rounding of the LayerNorm (section 6).
- A file is version 2 if and only if it uses a rule of version 2. Every version 1 file keeps its
  bytes and its meaning.

Code: `hsm` (reader, writer, `required_version`), `gelu::GeluTable<i16>`,
`LayerNorm::apply_i16`, `LayerNorm::with_beta_frac`, `int::rescale_wide_offset`,
`gpt2::model::Gelu`, the reference session, and the fast ops `linear_wide_residual` and
`logits_wide`. The context agent applied the engine part
(`gpt2/engine.rs`) on 2026-10-19 at 03:15 +04. Tests: `crates/hardshell/tests/wide.rs`.

## 4. The range rule and the first measurement

The recipe `wide` of the first engine change had `i16` outputs at the two sites and the range
`max` at every site, with the LayerNorm of version 1. HardShell `eval_bits` on the 373 Route A
evaluation windows (table size):

| Set | Float | `clip` | Loss | `wide`, beta after the rounding | Loss |
|---|---:|---:|---:|---:|---:|
| tail | 0.2409 | 0.2810 | +16.7 % | 0.2492 | +3.3 % |
| heldout | 0.7009 | 0.6844 | −2.3 % | 0.7473 | +6.5 % |
| all | 0.4046 | 0.4313 | +6.6 % | 0.4228 | +4.4 % |

The fake quantisation predicted about +1.5 %. The integer simulation (`convert.intsim`, exact
integer arithmetic; it agrees with HardShell, for example 0.50 bpb on `heldout/Apache#0` in
both) gave much more. Thus an integer step that the fake quantisation does not model caused the
difference. Section 6 finds it.

## 5. The final measurement

The recipe `wide` with `ln_beta_frac = 8`: `distil_logs_wide.hsm`, 84,369,792 bytes, SHA-256
`d12e5f34f8af34581a859c512e186e30e9b304c1af7efac0fb0e9d9559aacf07`. HardShell `eval_bits`
(devbox, AVX2, 8 threads, prefill chunk 32) on the 373 Route A evaluation windows. "Table" is the
size with the 32-bit frequency tables (the range coder gives it to about 0.02 %). "Loss" is the
ideal size against the float size (`finetune.floateval`, the float file of the Route A note).
"v2 first" is the recipe of section 4 (two roundings in the LayerNorm).

| Set | Bytes | Float | `clip` table | Loss | v2 first | Loss | **`wide` table** | **Loss** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 956,593 | 0.4046 | 0.4313 | +6.6 % | 0.4228 | +4.4 % | **0.4045** | **−0.0 %** |
| heldout | 161,832 | 0.7009 | 0.6844 | −2.3 % | 0.7473 | +6.5 % | **0.6982** | **−0.3 %** |
| tail | 250,138 | 0.2409 | 0.2810 | +16.7 % | 0.2492 | +3.3 % | **0.2425** | **+0.7 %** |
| train2k (not fair) | 243,564 | 0.3983 | 0.4520 | +13.5 % | 0.4102 | +2.9 % | 0.3984 | +0.1 % |
| english | 153,655 | 0.3293 | 0.3435 | +4.3 % | 0.3463 | +5.2 % | 0.3281 | −0.4 % |
| structured | 147,404 | 0.4463 | 0.4661 | +4.5 % | 0.4618 | +3.4 % | 0.4467 | −0.0 % |

Held-out systems and tails (table size, bpb):

| System | Float | `clip` | `wide` | `wide` loss |
|---|---:|---:|---:|---:|
| heldout Apache | 0.4524 | 0.4686 | 0.4509 | −0.3 % |
| heldout HealthApp | 0.8886 | 0.9153 | 0.8937 | +0.8 % |
| heldout Proxifier | 0.6609 | 0.6411 | 0.6505 | −1.6 % |
| heldout Zookeeper | 0.8393 | 0.7503 | 0.8351 | −0.5 % |
| tail Android | 0.1777 | 0.2707 | 0.1830 | +3.0 % |
| tail BGL | 0.2493 | 0.2652 | 0.2501 | +0.3 % |
| tail HDFS | 0.4383 | 0.4857 | 0.4430 | +1.1 % |
| tail HPC | 0.0655 | 0.0764 | 0.0662 | +1.1 % |
| tail Hadoop | 0.2323 | 0.2854 | 0.2358 | +1.5 % |
| tail Linux | 0.2482 | 0.2540 | 0.2447 | −1.3 % |
| tail Mac | 0.5513 | 0.6399 | 0.5575 | +1.1 % |
| tail OpenSSH | 0.1983 | 0.2608 | 0.1982 | −0.0 % |
| tail OpenStack | 0.3567 | 0.3919 | 0.3611 | +1.2 % |
| tail Spark | 0.2409 | 0.2622 | 0.2375 | −1.4 % |
| tail Thunderbird | 0.1268 | 0.1327 | 0.1275 | +0.5 % |
| tail Windows | 0.0803 | 0.1303 | 0.0821 | +2.2 % |

The tails are 13.7 % smaller than with `clip` (0.2425 against 0.2810 bpb). The held-out
systems are 2.0 % larger than with `clip` (0.6982 against 0.6844 bpb). Six tokens of the 373
windows got the weight 0 (the logit is more than 32 nats below the largest logit); the coder
gives them a frequency of 1, so they cost about 32 bits each.


## 6. The LayerNorm: one rounding instead of two

HardShell of version 1 calculates step 4 of the LayerNorm in two roundings:
`y = rescale(t) + beta`, with `beta` rounded to whole output units (ADR 0004 section 7). The
beta rounding gives each channel a constant error of up to half a unit. The beta correction
subtracts the effect of that error in the bias of the next linear layer.

To find the cost, a test script replaced the integer LayerNorm of `intsim` with variants. Two
sets of windows, both outside the evaluation set:

- **proxy:** the 10 text and code windows of the GPT-2 calibration set (`calibration.toml`,
  without the random window). The fine-tune did not see this domain, so the windows show the
  behaviour on unseen data.
- **val:** the 24 val windows of section 2.

Loss of the integer simulation against float (the recipe `wide` in each row):

| LayerNorm | proxy | val |
|---|---:|---:|
| integer, two roundings, beta correction (version 1 rule) | +5.72 % | +4.62 % |
| integer, two roundings, no beta correction | +3.18 % | +9.89 % |
| float LayerNorm, one rounding, no beta correction | +0.18 % | |
| float LayerNorm, `i16` gamma, one rounding, no beta correction | +0.49 % | |
| float LayerNorm, two roundings, no beta correction | +3.01 % | |
| float LayerNorm, one rounding, with the beta correction of version 1 | +13.09 % | |
| **integer, `ln_beta_frac = 8` (one rounding), beta correction** | **+0.24 %** | **+1.63 %** |
| integer, `ln_beta_frac = 8`, no beta correction | +0.16 % | |
| integer, `ln_beta_frac = 4` | | +2.10 % |
| integer, `ln_beta_frac = 12` | | +1.67 % |
| fake quantisation (float LayerNorm) | +0.24 % | +1.55 % |

The two roundings cost about 3 % on both sets. The beta correction does not remove that cost
on the proxy windows, and it costs 13 % when the LayerNorm has no beta error. UNVERIFIED: the
correction is correct in the mean at layer 0 (it reduces the mean error of `q` from 0.009 to
0.004, measured), so the loss probably comes from the interaction with later layers. With
`ln_beta_frac = 8`, beta has 8 more bits and HardShell adds it before the one rounding. Then
the integer model is at the fake quantisation result on both sets. 8 bits is the best of 4, 8
and 12 on the val windows. The change needs no change in the engine code outside
`layernorm.rs`, so the fast paths and the reference path use it at the same time.

## 7. GPT-2

`gpt2.hsm` does not change: it keeps `clip` (version 1, `aaa03eb9…a5c2`). To check that the
new rule is not better for GPT-2, the converter made `gpt2_wide.hsm` (the recipe `wide`,
SHA-256 `9aa394b1…7f56`). HardShell `eval_bits`, table size in bpb:

| Set | Windows | `clip` (v1) | `wide` (v2) | `wide` against `clip` |
|---|---:|---:|---:|---:|
| 16 held-out text files (the integer GPT-2 note) | 16 | **1.8710** | 1.9025 | +1.7 % |
| logs, heldout | 64 | **0.6894** | 0.7125 | +3.4 % |
| logs, tail | 96 | **0.8480** | 0.8513 | +0.4 % |
| logs, heldout and tail | 160 | **0.7857** | 0.7967 | +1.4 % |

The float GPT-2 is 1.867 bpb on the text files (integer GPT-2 note). On the log windows, the
float ideal size (0.7527 bpb heldout, 0.8601 bpb tail; `convert.fq_gpu`, float32) is larger
than both integer sizes, because the float model gives some tokens a very small probability
and the ideal size has no floor. GPT-2 with `clip` is 0.2 % above float on text and below
float on logs. Thus the saturation of `clip` helps GPT-2, and `wide` removes that help. The
fake quantisation on the GPT-2 calibration windows showed the same (section 2).


## 8. Bit-exact checks and reproducibility

All measured on 2026-10-19.

| Check | Result |
|---|---|
| `cargo test -p hardshell --release`, Mac (NEON) and devbox (AVX2) | all pass, with `tests/wide.rs` (14 tests) |
| Fast paths against the reference on a version 2 model (`tests/wide.rs`: step, prefill with chunks 1, 7 and 64, batch of 3 lanes, every kernel selection, 135 tokens with 2 context restarts) | the same weights |
| Conformance vector of version 2: the tiny wide model (seed `0x7ac0_0001`, 150 tokens, reference session) | model `201f9fd6…af04`, weights `64108429…2eb8`, the same on the Mac and devbox |
| Version 1 vectors (`gpt2.rs` golden, `conformance/tiny.hsm`) | unchanged |
| `distil_logs_wide.hsm`, 2 windows (`heldout/Apache#0`, `tail/Windows#0`): Mac fast path, Mac reference path, devbox prefill path | the same weight digests (`4c7eb898…e3d0`, `f0be9551…99d0`) |
| `gpt2_wide.hsm` made on the Mac and on devbox | the same SHA-256 `9aa394b1…7f56` |
| `gpt2.hsm` and `conformance/tiny.hsm` made again with the changed converter (Mac) | `aaa03eb9…a5c2` and `eb78b89a…18f8`, the self-check passes |
| `distil_logs.hsm` (`clip`) made again on devbox | `c0d382bd…fa98`, the Route A file |
| `convert.intsim` against HardShell (`heldout/Apache#0`, first v2 rule) | 0.50 bpb in both |
| `convert.test_exact` | 14 of 14 pass (Mac) |

Speed (devbox, loaded machine, not a benchmark): `eval_bits` with `wide` ran 248 steps per
second on 8 threads, `clip` 282 steps per second on 6 threads. The `i16` matrix products of
`mlp.c_proj` and of the logits have only the portable kernel (`PERF` in `kernels/fast.rs`).

## 9. Recommendation and limits

1. **Use `wide` for a model that is fine-tuned on the logs of known systems.** It removes the
   integer loss on those systems (+16.7 % to +0.7 %) and keeps the float result on unseen
   systems.
2. **Keep `clip` (version 1) for GPT-2.** `gpt2.hsm` does not change.
3. **The held-out result of `clip` was better than float.** If a later model must be good on
   unseen systems, measure a temperature or a range rule for that case on data that is not in
   the evaluation set. This note did not test it.
4. **GPU.** `hardshell-cuda` and `hardshell-gpu` do not run version 2 yet. They need: an `i16`
   GELU table, `i8 x i16` matrix products for `mlp.c_proj` and the logits (for example two
   `i8` products of the high and low bytes), `i16` outputs of `ln_f`, and the LayerNorm rule of
   `ln_beta_frac`. Until then they must refuse a model with `Model::is_wide()`.
5. **SIMD.** An AVX2 and NEON kernel for `matmul_i8_i16` is the next speed step for `wide`.

Limits: the val set has 24 windows and the proxy set 10 windows. The choice of 8 bits for
`ln_beta_frac` comes from 3 values on the val windows. The fine-tuned model is one model.

## 10. Commands

Run from `tools/` on devbox, unless a command says otherwise. `M` is
`~/taco-train/models/distil_logs/model.safetensors`, `S` is
`~/taco-train/out/distil_logs.stats.toml`, `W` is `~/taco-train/work/eval/windows.txt`.

```bash
# Fake quantisation on the val windows (GPU; check nvidia-smi first)
PYTHONPATH=. ~/taco-train/venv/bin/python -m convert.fq_gpu --model $M --stats $S \
    --windows ~/taco-train/work/val_windows.txt --modes float all --spec clip --spec wide
# Integer simulation on the val windows
nice uv run python -m convert.tune --model $M --stats $S --windows ~/taco-train/work/val_windows.txt \
    --float F.json --workers 4 wide wide:frac=0 wide:frac=4
# Conversion (the self-check knows the hash; a new recipe needs --new-hash)
nice uv run python -m convert.gpt2 --model $M --stats $S --name distil_logs --recipe wide \
    --out distil_logs_wide.hsm
nice uv run python -m convert.gpt2 --recipe wide --out gpt2_wide.hsm
# Measurement (repository root)
cargo run --release -p hardshell --example eval_bits -- distil_logs_wide.hsm $W --threads 8 \
    --chunk 32 --digest > eval.tsv
```

NOTE: `convert.tune` with more than 4 workers can use all memory of devbox (about 1.5 GB for
each worker); the Linux kernel then stops a worker and the pool waits for ever.
