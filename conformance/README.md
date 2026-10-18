# Conformance vectors

This directory holds the conformance vectors of Taco (plan §11). A HardShell implementation is
correct only when it gives the reference outputs bit for bit.

## The tiny model: `tiny.hsm`

| Item | Value |
|---|---|
| File | `conformance/tiny.hsm` |
| Size | 306,368 bytes (measured) |
| SHA-256 | `eb78b89a4f0f17c7245865aed22f86512a49e0272b7583a831c47e1a401418f8` |
| Tool | `tools/convert/tiny.py` |
| Seed | `0x7AC020261018` (splitmix64) |
| Format | `.hsm` version 1, with the merges section (ADR 0004, addendum A) |

The model has these parts:

- 2 blocks, width 64, 4 heads, an MLP width of 256, and a context of 64 tokens.
- A vocabulary of 300 tokens. Token `b` (0 to 255) is the single byte `b`. Tokens 256 to 298
  come from 43 BPE merges. Token 299 is `<|endoftext|>`. Thus any byte string can be
  tokenized (plan §10).
- Block 0 has an `i16` GELU table (65,536 entries). Block 1 has an `i8` GELU table (256
  entries). Thus the model uses both table types.
- The constants of GPT-2 small (ADR 0004): `K` = 48, `J` = 8, `P` = 30, and exponent tables of
  8,192 entries with a unit of 2^-8.

The weights are random. They are not a trained model, so the predictions have no meaning. The
model tests the arithmetic, not the quality.

NOTE: plan §11.1 says "about 100 KB". The file is 306,368 bytes, because the two exponent
tables (96 KiB) and the `i16` GELU table (64 KiB) have the sizes of GPT-2 small.

## Make the tiny model again

1. Go to the `tools/` directory.
2. Run `nice uv run python -m convert.tiny`.
3. Compare the SHA-256 that the tool prints with the SHA-256 above.

The tool uses splitmix64 and IEEE 754 basic operations for the weights. The calibration
statistics are committed data (`tools/convert/tiny_stats.toml`), and the conversion is exact
(ADR 0014), so a different machine gives the same file. If the hash
is different, do not commit the new file. Tell the lead.

## Tests that use the tiny model

- `crates/hardshell/tests/hsm_merges.rs`: the model loads, has its merges and runs. A second
  test writes the model again with `hardshell::hsm::HsmWriter` and compares the bytes. Thus the
  Python writer and the Rust writer make the same file.

## The version 1 container: `taco-v1-tiny.taco`

| Item | Value |
|---|---|
| File | `conformance/taco-v1-tiny.taco` |
| Size | 1,837 bytes (measured) |
| SHA-256 | `c3ff718b1c9682589836078b36d936836077744b03b1cc8af3a6ced0d5821b8a` |
| Made by | `golden_container_hash` in `crates/taco/tests/container.rs`, at commit `dc621ce` (container version 1) |
| Model | the tiny model of `crates/taco/tests/common` (SHA-256 `5ba63e7c…4ff`), not `tiny.hsm` |
| Blocks | 5 blocks of 512 bytes: coded, coded, raw, raw, coded (ADR 0007) |

The writer of version 2 cannot make this file again. It is the proof that the reader still
reads version 1 (plan §3 rule 8). `golden_v1_container_still_unfolds` in
`crates/taco/tests/container.rs` checks its hash and unfolds it. Do not change the file.

## The GPT-2 model

The GPT-2 model file is not committed (plan §11.1). `tools/convert/gpt2.py` makes it from the
published weights. The research note `docs/research/2026-10-18-integer-gpt2.md` records its
size and its SHA-256.
