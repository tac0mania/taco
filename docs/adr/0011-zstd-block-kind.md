# ADR 0011: container format version 2 and the zstd block kind

Date: 2026-10-18. Status: accepted (proposed by the zstdblock agent). Plan: §3, §9, §11. Shared with
ADR 0010 (match model and mixer, accepted 2026-10-19) and ADR 0012 (two profiles). Updated on
2026-10-19 by the profile agent: ADR 0010 §5 gives the coding parts, the coding parameters and
block kind 3 their meaning (§2 and §3 below). Updated on 2026-10-19 by the overlap agent (the
lead's decision): ADR 0015 §4 adds bit 1 of the coding parts, the context overlap, with `K` in
the coding parameter bytes 2 and 3 (§2 below). The version stays 2. Updated on 2026-10-19 by
the streamfreeze agent: the addendum "Version 2 is frozen" (at the end) is the full
specification, with the largest block size (64 MiB) as a new rule.

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
| 90 | 2 | coding parts | `u16` bit set. Bit 0: the match model and the mixer (ADR 0010 §5). Bit 1: the context overlap (ADR 0015 §4). Bits 2 to 15 are reserved and must be 0 (`UnknownCodingParts`). |
| 92 | 12 | coding parameters | Byte 0: the mixer rule version if bit 0 is set (only version 1 is known; `UnknownMixerRule`), else 0. Byte 1: 0. Bytes 2 and 3 (offsets 94 and 95): the overlap `K`, a little-endian `u16`, if bit 1 is set; `K` must be 1 or more (`ZeroContextOverlap`). Else both bytes are 0. Bytes 4 to 11: 0 (`ReservedNotZero`). |

The profile `text` has no match model and mixer. A reader refuses bit 0 with the profile
`text` (`CodingPartsNotInProfile`). Both profiles permit bit 1 (ADR 0012, addendum). ADR 0010
§5 has the full rules of bit 0, ADR 0015 §4 of bit 1.

Each set of coding parts has exactly one encoding: a part that is not set has 0 in its
parameter bytes, and the overlap part is set only for `K >= 1`. A reader that does not know a
bit refuses the container (`UnknownCodingParts`). Thus the reader of commit `2bdeac5` and older
refuses a container with an overlap, and never decodes it with the restart rule of `K = 0`.
The container does not know the context of the model, so it does not check `K <= n_ctx - 2`.
The decoder checks it with the model before the first block (`InvalidOverlap`).

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
| Golden vector E (version 2, ADR 0015 §4): vector B with the overlap `K = 512` (profile text) | 329 bytes, SHA-256 `d29fb5c2f32daf836657bb73a9f48d4b9eca966f08aec8c9571eacafdcf80408` (measured); equal to vector B except bytes 90, 94 and 95 |
| Golden vector F (version 2, ADR 0015 §4): vector D with the overlap `K = 512` (profile logs, the mixer) | 312 bytes, SHA-256 `9555a3e603c93f6f27943855b3faa47d1679200060fd07756ce29441ee75c6b6` (measured); equal to vector D except bytes 90, 94 and 95 |
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
- ADR 0015 §4 fills bit 1 and the coding parameter bytes 2 and 3 (2026-10-19). Vectors A to D
  and both empty vectors do not change. Vectors E and F are new.

## Addendum (2026-10-19): version 2 is frozen

Status: proposed by the streamfreeze agent; the lead accepts. Product-v1 §1 needs container
version 2 frozen. This addendum is the complete specification of version 2. A reader that obeys
it accepts and refuses the same files as `taco-format`. From this date, a change to any rule
below needs a new version, a reader for version 2, and a test (plan §3 rule 8).

The freeze adds one rule: the largest block size (§4). The lead decided it on 2026-10-19 after
fuzz findings F1 and F5. No file that `taco` wrote before is affected: the largest block size
of every test, vector and measurement is 1 MiB (measured: the logs research notes).

### 1. Layout

All integers are little-endian. The header is 104 bytes. The block table starts at byte 104.
Each table entry is 5 bytes. The stored bytes of the blocks follow the table, in table order,
with no gap and no padding.

| Offset | Size | Field | Valid values |
|---|---|---|---|
| 0 | 4 | magic | `54 41 43 4F` ("TACO") |
| 4 | 2 | version | 2 (1: ADR 0006) |
| 6 | 2 | flags | 0 |
| 8 | 32 | model hash | SHA-256 of the model file; 32 zero bytes: the raw-only mode |
| 40 | 8 | original length | any `u64` |
| 48 | 32 | checksum | SHA-256 of the original data |
| 80 | 4 | block size | 1 to 67,108,864 (64 MiB) |
| 84 | 4 | block count | exactly `ceil(original length / block size)` |
| 88 | 1 | profile | 0 (`text`) or 1 (`logs`) |
| 89 | 1 | reserved | 0 |
| 90 | 2 | coding parts | bit 0: match model and mixer; bit 1: context overlap; bits 2 to 15: 0 |
| 92 | 1 | coding parameter 0 | the mixer rule version (1 or 2, ADR 0017) if bit 0 is set, else 0 |
| 93 | 1 | coding parameter 1 | 0 |
| 94 | 2 | coding parameters 2 and 3 | the overlap `K` (`u16`, 1 or more) if bit 1 is set, else 0 |
| 96 | 8 | coding parameters 4 to 11 | 0 |
| 104 | 5 × count | block table | for each block: kind (`u8`), stored length (`u32`) |
| 104 + 5 × count | sum of stored lengths | stored bytes | the blocks, in table order |

Block `i` holds the original bytes from `i × block size` to `min((i + 1) × block size,
original length)`. Each block has the block size, and the last block can have fewer bytes.

| Kind | Name | Stored bytes | Stored length |
|---|---|---|---|
| 0 | raw | the original bytes | equal to the original length of the block |
| 1 | coded | the payload of ADR 0007 | less than the original length of the block |
| 2 | zstd | one zstd frame (RFC 8878, §5 above) | less than the original length of the block |
| 3 | mixed | the payload of ADR 0007, coded with the match model and the mixer of the rule version (ADR 0010; rule 2 adds the skip rule of ADR 0017 §4) | less than the original length of the block |

### 2. The refusals of the reader

The reader checks the fields in this order and refuses the input with the error in the
second column. The names are the variants of `taco_format::FormatError`.

| # | Rule | Error |
|---|---|---|
| 1 | The input starts with the magic. | `BadMagic`, or `Truncated` for a prefix of the magic |
| 2 | The version is 1 or 2. The reader reads it before any other field. | `UnsupportedVersion` |
| 3 | The input holds the full header (88 or 104 bytes). | `Truncated` |
| 4 | The flags are 0. | `UnknownFlags` |
| 5 | The block size is not 0. | `ZeroBlockSize` |
| 6 | The block size is at most 64 MiB (§4). | `BlockSizeTooLarge` |
| 7 | The block count is `ceil(original length / block size)`. | `WrongBlockCount` |
| 8 | The profile is 0 or 1. | `UnknownProfile` |
| 9 | The reserved byte 89 is 0. | `ReservedNotZero` |
| 10 | The coding parts have only bits 0 and 1. | `UnknownCodingParts` |
| 11 | With bit 0, parameter 0 is a known mixer rule version: 1 or 2. | `UnknownMixerRule` |
| 12 | With bit 1, `K` is 1 or more. | `ZeroContextOverlap` |
| 13 | A parameter byte of a part that is not set, parameter 1, and parameters 4 to 11 are 0. | `ReservedNotZero` |
| 14 | The profile `text` does not have bit 0. | `CodingPartsNotInProfile` |
| 15 | The input holds the full block table. The reader checks this before it allocates one entry for each block. | `Truncated` |
| 16 | Each kind is 0 to 3. | `UnknownBlockKind` |
| 17 | A raw block stores exactly its original length. | `RawLengthMismatch` |
| 18 | A coded or mixed block is not in the raw-only mode. | `CodedBlockInRawOnly` |
| 19 | A mixed block needs bit 0. | `MixedWithoutCodingPart` |
| 20 | A coded or mixed block is smaller than its original length. | `CodedNotSmaller` |
| 21 | A zstd block is smaller than its original length. | `ZstdNotSmaller` |
| 22 | The input ends exactly after the last stored byte. | `WrongTotalLength` |
| 23 | Every sum of lengths fits in `u64`. | `LengthOverflow` |

The reader does not check the model hash, the checksum, the payload of a coded or mixed block,
or the zstd frame. The decoder does:

1. Before the first block: the model hash equals the hash of the loaded model, except in the
   raw-only mode (`ModelHashMismatch`, `ModelRequired`). `K` is at most `n_ctx - 2` of the
   model (`InvalidOverlap`).
2. For each coded or mixed block (ADR 0007 §4): the varint is minimal and complete; the token
   count is 1 or more and at most the original length of the block; no token is
   `<|endoftext|>` or empty; the tokens give exactly the original length (`CorruptBlock`).
3. For each zstd block (§5 above): one frame with the magic, the content size equal to the
   original length, a window of at most 8 MiB, and exactly the original length of output
   (`CorruptZstdBlock`).
4. After the last block: the length and the SHA-256 of the restored data equal the header
   (`RestoredLengthMismatch`, `ChecksumMismatch`). A failure is an error, never a silent wrong
   output (plan §3 rule 4).

### 3. The rules of the writer

1. The writer writes version 2 only, with the flags 0 and every reserved byte 0.
2. The kind of each block is the smallest stored length of raw, zstd, coded and mixed, with the
   tie order raw, zstd, coded, mixed (§4 above). Thus a block is never larger than raw.
3. The writer refuses every input that the reader refuses: a block size above 64 MiB, a coded
   or mixed block that is not smaller, a zstd block that is not one frame of the original
   length, and an input that needs more than `2^32 - 1` blocks (`TooManyBlocks`).
4. The writer can write in one pass with a seek, or with temporary files (ADR 0018). Both give
   the bytes of `Container::build`.

### 4. The largest block size: 64 MiB (new rule of the freeze)

1. A block size above 67,108,864 bytes (64 MiB, `taco_format::MAX_BLOCK_SIZE`) is not valid,
   in version 1 and in version 2. The reader refuses it (`BlockSizeTooLarge`) before it reads
   the table. The writer refuses to write it.
2. Reason: the block size bounds the work and the memory of one block. A coded block has at
   most one token for each original byte (ADR 0007 §4.2), so a block costs at most 64 Mi model
   steps (before, a block of 112 bytes could state `2^32 - 1` steps: fuzz finding F5). The
   decoder holds at most the stored bytes and the restored bytes of a block (finding F1).
3. Choice of the value: the largest measured block size is 1 MiB (the logs profile, ADR 0012).
   64 MiB keeps a factor of 64 above it. The stream of ADR 0018 holds at most 64 MiB of data in
   one segment.
4. Version 1 gets the same rule. Version 1 was not released, and no version 1 file with a
   block size above 64 KiB exists (`conformance/taco-v1-tiny.taco` has 512).
5. Tests: `crates/taco-format/tests/frozen.rs`, and `crates/taco-cli/tests/security_cli.rs`
   (finding F1).

### 4b. Mixer rule version 2 (ADR 0017, merged on main at `c25c50c`)

1. Byte 0 of the coding parameters (offset 92) is the mixer rule version. The frozen version 2
   knows the rule versions 1 and 2. A reader refuses every other value (`UnknownMixerRule`).
   A reader of commit `c25c50c` or older refuses rule 2 with the same error, so no reader
   decodes a rule 2 block with the rules of version 1.
2. Block kind 3 does not change with rule 2. Its range coder bytes hold, for each position
   that the skip rule of ADR 0017 §4 selects, one flag symbol with the two-symbol table
   `[q_s, 2^16 - q_s]`. After the flag symbol 1 (a miss), the token follows, coded with the
   table of `taco_match::skip_miss_weights`. The token count of the payload counts tokens,
   not symbols (ADR 0017 §5.2).
3. The writer of `taco` writes rule version 1. A change of this default is an owner decision
   (it changes the bytes of the profile `logs`).
4. Gap: the `taco` library does not code or decode rule 2 yet (`crates/taco/src/mix.rs` and
   `codec.rs` know rule 1 only). Thus `taco::unfold` and the stream unfold refuse a rule 2
   container with `UnknownMixerRule { rule: 2 }` before the first block, and the vectors of §6
   have no rule 2 container. A rule 2 vector is necessary when the library writes rule 2.

### 5. The growth bound

Every block is at most its original length, so a version 2 container is at most `104 + 5 ×
block count + original length` bytes (`worst_case_len`). With the default block sizes (16 KiB
and 64 KiB), the growth is at most 0.031 % and 0.008 % of the original length, plus 104 bytes
(exact, from the formula).

### 6. Conformance vectors (`conformance/v2/`)

The vectors are small inputs folded with `conformance/tiny.hsm` (SHA-256 `eb78b89a…18f8`).
`conformance/v2/vectors.tsv` lists the options, the lengths and the SHA-256 of each input and
container. `crates/taco-cli/tests/conformance.rs` folds each input again (with `taco::fold` and
with the stream fold), compares the bytes, and unfolds each container (in memory and with both
stream readers). The test also unfolds the version 1 vector.

The weights of `tiny.hsm` are random, so English text codes larger than zstd. The inputs
`predicted_*` and `all_kinds` are bytes that the tiny model itself predicts, so they give coded
and mixed blocks. Thus the vectors hold all four block kinds, both profiles, `auto`, and `K = 0`
and `K > 0` with coded and mixed blocks. No GPT-2 vector is necessary.

| Vector | Profile | Block size | K | Kinds | SHA-256 of the container |
|---|---|---|---|---|---|
| empty | text | 16,384 | 0 | none | `bab7bf56209a25d0f5eb3e2cebec1f880f6223198e1b4b9697d394ed83252c44` |
| one_byte | text | 16,384 | 0 | raw | `1a101e98b93bd45b8c83a74ff66a101b6da8a32af76d63e057dcf464aa4f5d0f` |
| random_raw | text | 256 | 0 | raw | `8f9fb994f3017a3841707e2609244056bd58beab1f91f8af39229c763fa2d965` |
| text_k0 | text | 96 | 0 | zstd | `0ea7485329ff827accecba02add835e0b3c43ffeda3b986682029147c1aee246` |
| text_k20 | text | 96 | 20 | zstd | `56a82c36fe018daf0d54627787bf9adc47d825e52d912e3d3ce0b8d5f98d9fc4` |
| text_kinds | text | 96 | 0 | zstd, raw | `966bee33d8ecc33d93f88b8caa66179f5fe66fa488fdeebdffcda853ad580552` |
| logs_k0 | logs | 256 | 0 | zstd, raw | `5900c1f276ea410fc78dc0034879e03c9944e4abb427ca5ab6922cc9d11df430` |
| logs_default_k | logs | 256 | 32 | zstd, raw | `daca948e181af3b7fd53f0277fdfcfac3a9e97f2bcf3ba68e206bd0efaf4d9e3` |
| logs_kinds | logs | 96 | 32 | zstd, raw | `e38035c1cca8b7379fe7247bd2d8eb8fa3e60d7d23eee4060aa9d82b8c08ee85` |
| auto (gives `text`) | auto | 200 | 32 | zstd, raw | `803da88b6a26f89edfe5cace1f6de7420a47d614eb5bb23912d23a8628e714aa` |
| predicted_text_k0 | text | 300 | 0 | coded | `613f3dcb700582518891f542e0f47ea8497a1d5282d3955bade092c9690f9e6b` |
| predicted_text_k32 | text | 300 | 32 | coded | `2f12fecc06b07ffe0c2c571151299da738935eea2520efaa797de39c2f559ff0` |
| predicted_logs_k0 | logs | 400 | 0 | zstd, mixed | `b414551166a050bfad48b96562ff630bda6be0a5d9b5e0af46ca056a5e8b28ea` |
| predicted_logs_default_k | logs | 400 | 32 | zstd, mixed | `5871bc7e09d5262832ab0450c5b3cbd01e2e70edacad9f55582a4721a440b861` |
| predicted_auto (gives `logs`) | auto | 400 | 32 | zstd, mixed | `5871bc7e09d5262832ab0450c5b3cbd01e2e70edacad9f55582a4721a440b861` |
| all_kinds_text | text | 300 | 16 | coded, raw, zstd | `9e8f1258db04b1e2e7b846a9c57897e9ce05b5bd3acddc822ac5ba18d81aaf98` |
| all_kinds_logs | logs | 300 | 32 | mixed, raw, zstd | `24ecb02e09732f0225950b7dfc1de7f6351570a540d7f26f8da5b1d9c0bc4668` |

All hashes are measured on devbox (Linux x86) and on the Mac (Apple Silicon): the same test
passes on both. The zstd frames depend on zstd 1.5.7 and level 19 (§6 above). The container
vectors A to F of `taco-format` and the `taco` golden vectors do not change.
