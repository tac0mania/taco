# ADR 0011: container format version 2 and the zstd block kind

Date: 2026-10-18. Status: accepted (proposed by the zstdblock agent). Plan: §3, §9, §11. Shared with
ADR 0010 (match model and mixer, accepted 2026-10-19) and ADR 0012 (two profiles). Updated on
2026-10-19 by the profile agent: ADR 0010 §5 gives the coding parts, the coding parameters and
block kind 3 their meaning (§2 and §3 below).

## Context

GPT-2 does not compress binary data. Taco stores the first 8,192 bytes of `kennedy.xls` raw, in
8,285 bytes (measured). xz gives 1,672 bytes and zstd 2,097 bytes (measured). A block kind that
stores a zstd frame closes most of this gap. A new block kind changes which files are valid, so
it needs a new container version (plan §3 rule 8).

ADR 0012 adds a profile (`text` or `logs`) to the container. ADR 0010 adds coding parts for the
`logs` profile. ADR 0012 says that all three changes go into one version. Thus version 2 holds
the zstd block kind, the profile field, and reserved fields for ADR 0010.

## Decision

### 1. Versions

1. The writer writes version 2 only. It writes version 2 also when no block is zstd.
2. The reader reads version 1 (ADR 0006, unchanged) and version 2.
3. A version 1 reader refuses version 2 with `UnsupportedVersion { version: 2 }`. ADR 0006
   §2 makes it read the version before any other field, so the longer header does no harm.
4. A version 1 container has the profile `text` and no coding parts.

### 2. The header of version 2

The first 88 bytes have the layout of version 1. Version 2 adds 16 bytes, so the header is
104 bytes. The block table starts at byte 104. All integers are little-endian.

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 to 87 | 88 | as version 1 | The version field is 2. The flags field is 0. |
| 88 | 1 | profile | 0 = `text`, 1 = `logs` (ADR 0012). A reader refuses another value. |
| 89 | 1 | reserved | 0. A reader refuses another value. |
| 90 | 2 | coding parts | `u16` bit set. Bit 0: the match model and the mixer (ADR 0010 §5). Bits 1 to 15 are reserved and must be 0 (`UnknownCodingParts`). |
| 92 | 12 | coding parameters | Byte 0: the mixer rule version if bit 0 is set (only version 1 is known; `UnknownMixerRule`), else 0. Bytes 1 to 11: 0 (`ReservedNotZero`). |

The profile `text` has no coding part. A reader refuses bit 0 with the profile `text`
(`CodingPartsNotInProfile`). ADR 0010 §5 has the full rules.

The proposed mapping of the first version of this ADR is now the rule (ADR 0010 §5):

| Need of ADR 0010 §5 | Place |
|---|---|
| The coding-part flag "match model and mixer" | Bit 0 of the coding parts (offset 90) |
| The mixer rule version, one `u8` | Byte 0 of the coding parameters (offset 92) |
| A block kind "coded with the match part" | Kind 3, "mixed" (§3) |

### 3. Block kinds

| Kind | Name | Versions | Stored bytes |
|---|---|---|---|
| 0 | raw | 1, 2 | The original bytes. The stored length equals the original length. |
| 1 | coded | 1, 2 | The payload of ADR 0007. The stored length is less than the original length. |
| 2 | zstd | 2 | One zstd frame (RFC 8878). The stored length is less than the original length. |
| 3 | mixed | 2 | The payload of ADR 0007, coded with the match model and the mixer (ADR 0010). The stored length is less than the original length. Only with bit 0 of the coding parts. |

1. A reader refuses kinds 2 and 3 in a version 1 container, and kinds 4 to 255 in both
   versions (`UnknownBlockKind`). It refuses kind 3 without bit 0 of the coding parts
   (`MixedWithoutCodingPart`) and in the raw-only mode (`CodedBlockInRawOnly`).
2. A zstd block needs no model. The raw-only mode (model hash all zero) permits zstd blocks. It
   still refuses coded and mixed blocks.
3. A container with a model hash needs that model, even if all its blocks are raw or zstd
   (plan §3 rule 3, unchanged).

### 4. The rule that selects the kind (encoder)

1. The candidates are raw (the original length), coded (if the encoder made a payload) and
   zstd. With the profile `logs`, mixed (ADR 0010 §5.7) is a fourth candidate.
2. The smallest stored length wins.
3. On a tie, the order is raw, then zstd, then coded, then mixed. A block is coded, mixed or
   zstd only if it is smaller than raw (plan §3 rule 6). zstd wins a tie with coded because it
   decodes without the model and thousands of times faster. Coded wins a tie with mixed
   because it decodes without the match model and the mixer.
4. The rule uses only lengths, so every machine selects the same kind.
5. Consequence: the encoder can stop the model when the coded payload reaches the length of
   the zstd frame. The selected kind and the bytes are the same.

`Block::choose_smallest(original_len, coded, zstd)` in `taco-format` is this rule for the first
three candidates, and `Block::or_mixed(original_len, mixed)` adds the fourth.
`Block::with_zstd(original)` adds the zstd candidate to a raw, coded or mixed block.

### 5. The rules of a zstd block (decoder)

The decoder refuses a zstd block with `CorruptZstdBlock` in these cases:

1. The stored bytes do not start with the zstd frame magic `28 B5 2F FD`. Thus skippable
   frames and legacy formats are refused.
2. The stored bytes are not exactly one frame: bytes follow the frame, or the frame is
   truncated.
3. The frame header has no content size, or the content size is not the original length of
   the block.
4. The frame needs a window larger than `2^23` bytes (8 MiB).
5. The zstd decoder refuses the frame.
6. The frame gives more bytes or fewer bytes than the original length of the block.

The decoder does not allocate memory from a size in the frame or in the header. It decodes in
pieces of at most 128 KiB, and it stops when the output is longer than the original length.
Thus the output buffer grows only with bytes that the frame really gives. The zstd window is at
most 8 MiB.

The parser does not decode zstd frames. It checks the stored length only (§3). The writer
checks rules 1 to 3 before it writes a zstd block. Other damage gives wrong bytes, and the
checksum of the container finds them (plan §3 rule 4).

### 6. The encoder and the conformance vectors

1. The encoder uses zstd level 19, the content size flag, no frame checksum, no dictionary id
   and one thread, with the one-shot `ZSTD_compress2` function.
2. The decoder output is the same for every conforming zstd decoder. The encoder output is
   not: it depends on the zstd library version, the level, the parameters and the API path.
   The zstd 1.5.7 command line tool made a different frame (same length, other bytes) for the
   block of vector B than the library function (measured).
3. Thus the crate pins the library: `zstd-safe = "=8.2.0"` and `zstd-sys = "=2.1.2"` (zstd
   1.5.7). Cargo.lock records it. A test checks that the linked library is 1.5.7.
4. The container golden hashes are valid for zstd 1.5.7 and the parameters of item 1. Vector
   B also has the hash of its zstd frame alone. If only that hash fails, the zstd library
   changed, not the layout.
5. A change of the zstd version, the level or the parameters changes the compressed bytes but
   not the format. It needs new conformance vectors (as ADR 0007 §5), not a new version.
6. Decoding is always exact, with any zstd version.

The golden vectors on the Mac (ARM) and devbox (x86) show that the pinned encoder gives the
same bytes on both machines (§Measurements). `scripts/xcheck` does not include them yet.

### 7. What the checksum does not cover

The checksum covers the original data only (ADR 0006 §10). A change of the profile byte from 0
to 1 gives a valid container. The decoder does not use the profile, so the output is still the
original data. A flip test confirms this (measured). A change from 1 to 0 in a container with the
coding part of ADR 0010 is refused (`CodingPartsNotInProfile`).

### 8. Growth bound

A zstd block and a mixed block are smaller than their original bytes. Thus a version 2
container is at most
`104 + 5 × block count + original length` bytes (`worst_case_len`). This is 16 bytes more than
version 1.

## Measurements

All on the Mac (Apple Silicon), zstd 1.5.7, unless stated.

| Case | Result |
|---|---|
| `kennedy.xls`, first 8,192 bytes, zstd levels 1, 3, 9, 15, 19, 22 (CLI) | 2,246, 2,231, 2,164, 2,096, 2,097, 2,097 bytes (measured) |
| The same, first 16,384 bytes | 3,158, 3,187, 3,062, 2,977, 2,960, 2,960 bytes (measured) |
| Blocks of 16 KiB, level 3, 9, 19: `kennedy.xls` (1,029,744 bytes) | 118,352, 113,796, 110,289 bytes; 186, 40, 3.1 MB/s (measured, CLI) |
| The same, `HDFS_2k.log` (287,848 bytes) | 61,186, 56,983, 52,852 bytes; 207, 21, 2.0 MB/s (measured, CLI) |
| The same, `alice29.txt` (152,089 bytes) | 65,005, 61,723, 60,393 bytes; 95, 12, 2.1 MB/s (measured, CLI) |
| `taco-format`, `kennedy.xls` first 8,192 bytes, one zstd block | container 2,206 bytes (raw today: 8,285) (measured) |
| `taco-format`, level 19, blocks of 8 and 16 KiB, release build, one thread | encode 1.5 to 5.4 MB/s, decode 58 to 766 MB/s (measured) |
| Golden vector B (version 2): 450 bytes, blocks of 100: raw, coded (7), zstd (76), coded (0), zstd (17) | 329 bytes, SHA-256 `18d7b8a97cf26ffc0a28d7e4575f870ed7463cabefded8450ee7f912d723e8d9` (measured) |
| Zstd frame of block 2 of vector B | 76 bytes, SHA-256 `4a62d79659c63a4dbc2ea9236d5001543baea7f608f045b4563b8262b11fe83d` (measured) |
| Golden vector C (version 2): raw-only mode, profile `logs`, 1,000 bytes of phrases, blocks of 256, all zstd | 570 bytes, SHA-256 `72dcd44f9b87a9e0fd7f4be6b6e8de252597ec98eac4b5e0bc5215b06ccb7a59` (measured) |
| Golden vector: empty data, version 2, raw-only mode, block size 65,536 | 104 bytes, SHA-256 `03de05c54a2e192c7dd046526ef8a3c9882b16465b9ab577b42c3477e130c390` (measured) |
| Version 1 vectors A and empty (ADR 0006), hand-made | the hashes of ADR 0006; both parse as version 1 (measured) |
| A separate Python copy of the version 2 layout, with the zstd frames of the Rust encoder | the same bytes for vectors B, C and empty (measured) |
| The zstd 1.5.7 CLI decodes the zstd frames of vectors B and C | the original bytes (measured) |
| The version 1 reader of commit `3cc17a6` reads vectors B and C | `UnsupportedVersion { version: 2 }` for both (measured) |
| Every bit flipped in each byte of a raw-only container with 2 zstd blocks (345 bytes, 2,760 cases) | 2,731 errors, 29 outputs equal to the original, 0 wrong outputs (measured) |

The 29 outputs equal to the original are 1 profile flip (§7) and 28 flips in zstd frames that
gave the same decoded bytes. The output is correct in each case.

devbox (Linux x86) built zstd 1.5.7 from the bundled source with no extra system package. It
passed the same 95 tests of `taco-format` and gave the same hashes for vectors A, B, C, the
frame of vector B and both empty vectors (measured, 2026-10-18). The 9 MiB block gave a frame
of 20,230 bytes on both machines, and the flip test gave the same counts (measured).

## Consequences

- The header is 104 bytes. Every container of the `taco` crate changes, so its golden vector
  changes. Old version 1 files still decode.
- `taco-format` depends on a C library (zstd, BSD-3-Clause), built from the bundled source by
  the `cc` crate. A build needs a C compiler. The Mac and devbox have one. No system zstd
  package is used.
- `docs/ARCHITECTURE.md` must list `zstd-safe` and `zstd-sys` as permitted dependencies of
  `taco-format`.
- An update of the zstd crates needs new conformance vectors and a new cross-machine check.
- ADR 0010 §5 fills the coding parts, the coding parameters and kind 3 (2026-10-19). Vectors
  A, B, C and both empty vectors do not change. Vector D (ADR 0010 §5.10) is new.
