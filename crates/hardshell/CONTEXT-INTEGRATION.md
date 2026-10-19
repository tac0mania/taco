# Context overlap: what `taco` and the container need (ADR 0015)

This note is for the lead. HardShell has the context rule of ADR 0015 on every path. The crates
`taco`, `taco-format` and `taco-cli` do not use it yet. This note lists the changes that they
need. The context agent did not edit these crates.

## 1. The HardShell API

| Path | Call | Default |
|---|---|---|
| Oracle | `ReferenceSession::with_overlap(model, overlap)` | `ReferenceSession::new` = `Overlap::NONE` |
| Step and prefill (fold) | `SessionConfig::new().with_chunk(32).with_overlap(overlap)` | `Overlap::NONE` |
| Batch (unfold) | `Batch::new(model, lanes).with_overlap(overlap)` | `Overlap::NONE` |
| CUDA session | `CudaSession::with_overlap(&model, overlap)` | `CudaSession::new` = `Overlap::NONE` |
| CUDA batch | `CudaBatch::set_overlap(overlap)` before the first pass | `Overlap::NONE` |

- `Overlap` is `hardshell::gpt2::Overlap`, a `u16` newtype: `Overlap::new(512)`, `.get()`.
- `Model::check_overlap(overlap)` refuses `K > n_ctx - 2` with `Error::InvalidOverlap`. Every
  session and batch also refuses each step with this error. Call `check_overlap` when the
  decoder reads the header, so a bad value gives an error before the first block.
- With `Overlap::NONE` every path gives the same bits as before. The golden hashes do not
  change (measured: all `hardshell` tests on the Mac and on devbox).

## 2. Changes in `crates/taco`

1. `FoldOptions`: add `overlap: Overlap` and `with_overlap`. Default: see section 5.
2. `fold.rs`, line with `SessionConfig::new().with_chunk(FOLD_CHUNK)`: add
   `.with_overlap(options.overlap)`.
3. `unfold.rs`, `Batch::new(m, lanes)`: add `.with_overlap(overlap)` with the overlap of the
   header. Call `m.check_overlap(overlap)` first and map the error to a container error.
4. `codec.rs`: no change. The order of the calls (ADR 0007 §3) does not change; HardShell
   does the restart inside `step` and `step_many`.
5. `mix.rs` (logs profile): no change. The match model sees all tokens of the block.
6. Do not change `FOLD_CHUNK`. A session with an overlap runs the restart rows in groups of at
   least 32 rows.
7. Tests to add: a round trip with `K = 512` and blocks of more than 1,024 tokens (fold, then
   unfold, byte for byte); a container whose `K` is too large is refused; a new golden
   container with `K > 0` (the tiny model has `n_ctx = 64`, so use `K = 32`).

## 3. Changes in `crates/taco-format` (proposal)

The container stores one `K` for all blocks. The proposal uses the version 2 header and needs
no new version:

1. Bit 1 of the coding parts field (offset 90): the context overlap.
2. Bytes 2 and 3 of the coding parameters (offsets 94 and 95): `K` as a little-endian `u16`.
   Byte 0 stays the mixer rule; byte 1 stays reserved.
3. Bit 1 clear: `K = 0`, and bytes 2 and 3 must be 0. Bit 1 set: `K` must be 1 or more. Thus
   each `K` has one encoding.
4. A reader of today refuses bit 1 (`UnknownCodingParts`). Thus an old reader never decodes
   such a container with the wrong rule.
5. The `text` profile must permit this coding part. Today `check_profile` refuses every coding
   part for `text` (ADR 0012). This needs an ADR 0012 addendum.

An alternative is container version 3 with a field for the context rule. The lead decides
(plan §3 rule 8). Both store the same value.

## 4. Changes in `crates/taco-cli`

- `taco fold --overlap K` (default: section 5). `taco unfold` reads `K` from the header and
  has no option for it.

## 5. Recommended defaults

Measured in `docs/research/2026-10-19-long-context.md` (section 7). The lead decides.

| Option | Recommended | Today | Why |
|---|---|---|---|
| `FoldOptions::overlap` | `Overlap::new(512)` | `Overlap::NONE` | −1.8 % to −2.0 % on English, −6.0 % on HTML, −4.0 % on Chinese, −20.6 % on a log (64 KiB blocks). The CPU decoder takes 24 % more time. |
| `FoldOptions::block_size` | 65,536 | 16,384 | With `K = 512`: −0.4 % (text) to −2.9 % (log) against 16 KiB. Larger blocks add at most 1.3 %. |
| Speed option (if wanted) | `K = 384` | | 58 % more rows instead of 100 %; the same size on English, 0.4 % to 1.2 % larger on Chinese and logs. |

Keep `K = 0` for the conformance vectors that exist (`conformance/taco-v1-tiny.taco` and the
version 2 golden container). Add new vectors for the new defaults.

## 6. Conformance

- HardShell vector (ADR 0015 §6): the tiny model, 300 tokens, `K = 32`:
  `5a79928200ad35bb7e7a1be2a0a35e05dcecc001a89cf3cf0765dfc7e5304a83`.
- Real GPT-2 (`gpt2.hsm` `aaa03eb9…a5c2`), the first 2,100 tokens of
  `tools/data/out/eval_windows.txt`, `K = 512`: SHA-256 of all weights
  `09b99a923377b260b83d676c8799d4cc07f96f2996bced7471128d4d561bd452` (oracle, step, prefill and
  batch on the Mac NEON and devbox AVX2; CUDA decode and prefill on the RTX 3060).
- Run `scripts/xcheck` after the integration. With `K = 0` the container hashes do not
  change.
