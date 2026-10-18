# Integration of container version 2 into `crates/taco` (for the lead)

Written by the zstdblock agent, 2026-10-18. ADR 0011 has the rules. The zstdblock agent did not
edit `crates/taco`. Apply these steps in order, then run the checks at the end.

## 0. Before you change anything: keep a version 1 container

The `taco` golden container (tiny model, SHA-256 `c3ff718b…1b8a`, 1,837 bytes) is a version 1
file. Save its bytes now, with the current code, so that a test can prove that version 1 files
still unfold (CLAUDE.md §1 rule 8):

1. Write `f.container` of `golden_container_hash` in `crates/taco/tests/container.rs` to
   `conformance/taco-v1-tiny.taco` (one time, for example with a temporary `std::fs::write`).
2. Add a test that unfolds this file with the tiny model, checks the version (1) and compares
   the output with `golden_input()`.

## 1. Workspace dependencies

Move the two lines from `crates/taco-format/Cargo.toml` to the root `Cargo.toml`, keep the
exact versions, and use `zstd-safe.workspace = true` and `zstd-sys.workspace = true` in the
crate:

```toml
[workspace.dependencies]
# ADR 0011: exact versions; the encoder bytes (container hashes) depend on zstd 1.5.7.
zstd-safe = { version = "=8.2.0", default-features = false }
zstd-sys = { version = "=2.1.2", default-features = false }
```

Cargo.lock already has `zstd-safe 8.2.0`, `zstd-sys 2.1.2+zstd.1.5.7` and their build
dependencies (`cc`, `jobserver`, `shlex`, `find-msvc-tools`). Add both crates to the list of
permitted dependencies of `taco-format` in `docs/ARCHITECTURE.md` (licence BSD-3-Clause; C
source bundled, built with `cc`; no system package).

## 2. `fold.rs`: add the zstd candidate

In the closure of `map_blocks` in `fold`, add zstd to the block that `encode_block` selected:

```rust
            let coded = encode_block(session, &tokens, data.len())?;
            let block = coded.block.with_zstd(data)?;
            Ok(CodedBlock { block, ..coded })
```

`Block::with_zstd` returns `taco_format::Result`. `TacoError` has `From<FormatError>`, so `?`
works. Optional: add `zstd_blocks: u64` to `FoldStats` and count `Block::Zstd(_)` next to
`coded_blocks`. Correct the doc comment "The others are raw".

## 3. `unfold.rs`: decode kind 2

In `decode_group`, add this arm before the `_` arm:

```rust
            (BlockKind::Zstd, _) => {
                let mut bytes = Vec::new();
                match b.decode_zstd_into(&mut bytes) {
                    Ok(()) => parts[slot] = Some(bytes),
                    Err(e) => errors.push((b.index(), TacoError::Format(e))),
                }
            }
```

A zstd block needs no model, so the arm does not look at `model`. `decode_zstd_into` refuses a
frame that gives more or fewer bytes than the block and never allocates from a size in the
frame (ADR 0011 §5). Correct the NOTE of the `_` arm: it is now for kinds of later versions
only.

## 4. Optional speed step: stop the model at the zstd length

ADR 0011 §4.5: the encoder can stop the model when the coded payload reaches the length of the
zstd frame. The selected block and the bytes do not change. This saves model time on binary
blocks (est.: most of the time on `kennedy.xls`).

1. In the closure, call `taco_format::compress_zstd(data)?` before `encode_block`.
2. Give `encode_block` a bound `stop = min(original_len, zstd.len())` and use it in place of
   `original_len` in the three early-stop compares. Return "no coded candidate" when it stops.
3. Select with `Block::choose_smallest(data.len(), coded, Some(zstd))`.

This changes `encode_block` (ADR 0007 §5.2 says "the original length"). Update ADR 0007 §5.2 if
you take this step.

## 5. Tests in `crates/taco` that change

| Test | Change | Why |
|---|---|---|
| `container.rs::golden_container_hash` | New length and SHA-256; measure them. The kinds can now include `Zstd`. Keep a vector with coded and zstd blocks. | The header is 104 bytes, and zstd can win the raw blocks. |
| `container.rs` (lines 102, 130) | Check the offsets. `HEADER_LEN` is now 104, so `HEADER_LEN + 5 * n` stays right. | Version 2 header. |
| `round_trip.rs::random_bytes_are_stored_raw` | Should pass: zstd does not make random bytes smaller. Run it. | Growth bound. |
| `round_trip.rs` (line 45, empty input) | `HEADER_LEN` is now 104. No change. | — |
| New test | `kennedy.xls`-like binary data (or a fixture) gives zstd blocks and round-trips. | ADR 0011. |
| New test | `conformance/taco-v1-tiny.taco` unfolds (step 0). | Plan §3 rule 8. |

Also update the golden hash table in `HANDOFF.md` (container vector A is now a version 1
vector; add vectors B, C and the version 2 empty vector of ADR 0011).

## 6. Checks

1. `scripts/mx cargo test -p taco-format -p taco -p taco-cli`
2. `scripts/mx cargo clippy -p taco -p taco-format --all-targets -- -D warnings` and `cargo fmt`.
3. `scripts/dx main cargo test -p taco-format -p taco` (x86).
4. `scripts/xcheck`: the container changed, so run the cross-machine check (HANDOFF §5 item 1).
5. Measure `taco fold` on `kennedy.xls` (8 KiB): expected about 2,206 bytes (est. from
   `taco-format`: 104 + 5 + 2,097).

## API summary (`taco-format`)

- `BlockKind::Zstd` (kind byte 2), `BlockKind::min_version()`, `BlockKind::needs_model()`.
- `Block::Zstd(Vec<u8>)`, `Block::choose_smallest(original_len, coded, zstd)`,
  `Block::with_zstd(original)`, `Block::stored_len(original_len)`. `Block::choose` is kept.
- `compress_zstd(original)`, `BlockRef::decode_zstd_into(&mut out)`, `ZSTD_LEVEL` (19),
  `ZSTD_WINDOW_LOG_MAX` (23), `ZSTD_LIBRARY_VERSION`, `zstd_library_version()`.
- `Profile` (`Text`, `Logs`), `ContainerParams::with_profile`, `ContainerParams.profile`,
  `Header::profile()`, `Header::version()` (now 1 or 2), `Header::byte_len()`.
- `VERSION` is 2, `VERSION_1` is 1, `HEADER_LEN` is 104, `HEADER_LEN_V1` is 88.
- New errors: `ZstdNotSmaller`, `CorruptZstdBlock`, `Zstd`, `NotZstd`, `UnknownProfile`,
  `UnknownCodingParts`, `ReservedNotZero`.
