# ADR 0006: the rules of the container (format version 1)

Date: 2026-10-18. Status: accepted (proposed by the format agent). Plan: §3, §9, §11.

## Context

Plan §9 gives the fields of the `.taco` container. It does not give all the rules that a
reader must check. A reader gets untrusted input. If two readers apply different rules, one
reader can accept a file that the other refuses. Thus the rules are part of the format, and
this ADR records them. The crate `taco-format` has the same text in its `lib.rs`
documentation.

## Decision

### 1. Layout

The layout is the layout of plan §9. All integers are little-endian. The header is 88 bytes.
Each entry of the block table is 5 bytes. The stored bytes of the blocks follow the table in
table order, with no gap and no padding.

### 2. The version field

The magic is at bytes 0 to 3, and the version is at bytes 4 and 5, in every version. A reader
reads the version before any other field. It refuses an unknown version with a typed error,
even if the input is shorter than the header of version 1.

### 3. Flags

Version 1 defines no flag. A reader refuses a container with a flag bit that is not 0. A new
flag needs a new version (plan §3 rule 8).

### 4. Blocks

1. The block size is at least 1 and at most `2^32 - 1` bytes.
2. The block count is exactly `ceil(original length / block size)`. A reader refuses another
   value.
3. Block `i` holds the original bytes from `i × block size` to the end of the block. Each
   block has `block size` bytes, and the last block can have fewer. Thus the original length
   of each block comes from the header, and the table does not store it.
4. The writer refuses data that needs more than `2^32 - 1` blocks.

### 5. Empty data

Empty data has 0 blocks and no block table. The container is the header only, 88 bytes. The
checksum is the SHA-256 of no bytes. The block size is the value that the writer received. It
is at least 1.

### 6. Block kinds

1. The kind is 0 (raw) or 1 (coded). A reader refuses another value.
2. A raw block stores exactly its original bytes. Its stored length equals its original
   length.
3. A coded block is smaller than its original bytes: its stored length is less than its
   original length. A reader refuses a coded block that is not smaller. Plan §3 rule 6 is
   thus a rule of the format, not only of the encoder. Consequently, every valid container
   is at most `88 + 5 × block count + original length` bytes.
4. A coded block of 0 bytes is valid. The range coder removes zero bytes from the end of its
   output (ADR 0003), so a block of very likely tokens can code to 0 bytes.

### 7. The raw-only mode

1. A model hash of 32 zero bytes is the raw-only mode. A container in the raw-only mode has
   no coded block. A reader refuses a coded block in it.
2. A container in the raw-only mode needs no model. The model-hash check passes with any
   loaded model.
3. A container with a model hash that is not zero needs that model, even if all its blocks
   are raw (plan §3 rule 3).

### 8. Length

The input ends exactly at the end of the last block. A reader refuses a missing byte and an
extra byte. Thus a container has exactly one length, and the reader finds a truncated file.

### 9. Safe parsing

1. All sums of lengths use checked arithmetic. An overflow is a typed error.
2. The reader checks that the block table is inside the input before it allocates one entry
   for each block. Thus the allocation is at most one entry for each 5 bytes of input.
3. The reader does not limit the original length. A coded block of a few bytes can hold many
   original bytes. The decoder must not allocate the original length before the blocks
   decode.

### 10. Checks that the parser does not do

The parser does not have the model or the original data. Thus the caller does two checks:

1. Before decoding: compare the model hash with the hash of the loaded model. A mismatch is a
   typed error that carries both hashes.
2. After decoding: compare the length and the SHA-256 of the restored data with the header. A
   mismatch is a typed error (plan §3 rule 4).

The checksum covers the original data only. It does not cover the header. Damage to the header
gives a parser error, a model-hash error, a decoder error or a checksum error. Damage to the
checksum field gives a checksum error, not wrong output.

## Measurements

| Case | Result |
|---|---|
| Golden vector A: 350 bytes, blocks of 100, raw, coded (7 bytes), coded (0 bytes), raw | 265 bytes, SHA-256 `c1c07402a5dae2993c596345a696d35b7c1c0502fe304d31fa33d4f5ecefeeec` (measured) |
| Golden vector: empty data, raw-only mode, block size 65,536 | 88 bytes, SHA-256 `2f8a2052a618a4ca1adfc67aa07836c7d4fda0b10738142e9327228ddd0bcdf7` (measured) |

A separate Python copy of the layout gave the same two hashes (measured).

## Consequences

- A reader that obeys this ADR accepts and refuses the same files as `taco-format`.
- An encoder must store a block raw if its coded form is not smaller. `Block::choose` in
  `taco-format` does this.
- A change to any rule above changes which files are valid. It needs a new container version
  and new conformance vectors (plan §3 rule 8, §11).
- The block size is free in version 1. Plan §13 question 3 selects the default block size.

## Addendum (2026-10-19): the freeze and the largest block size

Status: proposed by the streamfreeze agent; the lead decided the rule (fuzz findings F1 and F5).

1. Version 1 stays readable in every later reader (plan §3 rule 8). The writer does not write
   it (ADR 0011 §1). The rules above stay, with one change.
2. Change to §4.1: the block size is at least 1 and at most 67,108,864 bytes (64 MiB). A
   reader refuses a larger block size with `BlockSizeTooLarge` before it reads the table. ADR
   0011, addendum "Version 2 is frozen", §4 gives the reason. No version 1 file with a block
   size above 64 KiB exists: version 1 was not released.
3. The test that a version 1 file still unfolds: `golden_v1_container_still_unfolds`
   (`crates/taco/tests/container.rs`), and `the_version_1_vector_still_unfolds`
   (`crates/taco-cli/tests/conformance.rs`, also with the stream reader of ADR 0018).
