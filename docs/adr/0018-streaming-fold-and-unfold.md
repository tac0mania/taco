# ADR 0018: streaming fold and unfold with container version 2

Date: 2026-10-19. Status: proposed (by the streamfreeze agent; the lead decides). Plan: §3, §9.
Product: `docs/design/product-v1.md` §1 ("Large files", "Format"). Related: ADR 0006, ADR 0011
(and its addendum "Version 2 is frozen"), ADR 0015.

## Context

Before this ADR, `taco fold` and `taco unfold` read the whole input into memory, and they wrote
the whole output from memory. Thus the memory grew with the file size. Product-v1 §1 needs
streaming: the memory must not grow with the file size. It also needs container version 2
frozen. Thus the solution must not change the format, or it must add a new version.

The question: can a writer make a version 2 container in one pass, without a seek? The answer
is no. These fields come before the stored bytes of the first block:

| Offset | Field | Known when |
|---|---|---|
| 40 | original length | at the end of the input (a pipe), or before the start (a file) |
| 48 | checksum (SHA-256 of the input) | at the end of the input only |
| 84 | block count | as the original length |
| 88 | profile | at the end only with `--profile auto` (it depends on all blocks) |
| 104 | block table (kind and stored length of each block) | each entry after its block is coded |

The reader is different. A reader can read version 2 in one pass: the header, then the table,
then the blocks in table order. It needs the full table before block 0.

## Decision

There is no format change. Version 2 stays as ADR 0011 and its addendum state.

### 1. fold: the two output methods

1. **Seek back.** The output can seek and the input length is known before the start (the
   input is a regular file). The writer writes a header of zeros and a table of zeros. Then it
   writes the stored bytes of each block in order. It writes the table entries in groups of
   4,096 entries with a seek. At the end it writes the header. The extra disk space is zero.
2. **Spool.** The output cannot seek (a pipe), or the input length is not known (a pipe). The
   writer writes the stored bytes of each block to a temporary file, and the table entries to a
   second temporary file. At the end it writes the header, the table and the stored bytes to
   the output. The temporary files are at most `5 × block count + original length` bytes
   (`worst_case_len` minus the header). The output starts only after the input ends.

Both methods give the same bytes as `Container::build` for the same blocks. A test proves it
for every block kind, for empty input, and for each method.

The temporary files go to the directory of the output file, or to the temporary directory of
the system for stdout. They are deleted at the end, also after an error.

### 2. fold: files

The command writes a file output to a temporary name in the same directory. It renames the
file to the output name only after the container is complete. Thus an error never leaves a
partial `.taco` file with the output name.

### 3. unfold: one pass

1. The reader reads the header, then the table, then the blocks. It never seeks the stored
   bytes.
2. **A file input.** The reader knows the input length. Before it decodes a block, it reads
   the table in groups of 4,096 entries and checks every entry and the total length (the
   checks of `Container::parse`). Thus a header that claims more than the file holds is
   refused before any output, and the table needs no memory that grows with the file.
3. **A pipe input.** The reader keeps the table in memory: 5 bytes for each block. It
   allocates the table only as its bytes arrive, so a false block count costs no memory before
   the bytes exist. A missing byte or an extra byte is an error at the point where the reader
   finds it.
4. **A file output.** The command writes to a temporary name. It renames the file only after
   the length and the checksum are correct. Thus a failed check never leaves an output file
   with the output name (plan §3 rule 4).
5. **stdout.** The command writes each block when it is decoded. The checksum is known only at
   the end. A failed check is an error message and the exit status 1. The bytes that are
   already written stay in the pipe. This is the same as `gzip -dc` and `zstd -dc`. A script
   must check the exit status.

### 4. Memory

The work moves in segments of blocks. A segment has at most `clamp(64 MiB / block size, 1,
256)` blocks. Thus a segment holds at most 64 MiB of input, or one block when the block is
larger than 64 MiB.

| Part | fold | unfold |
|---|---|---|
| Data of one segment | input, tokens (4 bytes for each token), zstd frames, payloads | stored bytes, restored bytes |
| Fixed | the model, one session for each thread, a table buffer of 4,096 entries | the model, one batch for each thread, a table buffer of 4,096 entries |
| Grows with the file | nothing | a pipe input only: 5 bytes for each block |

Thus with a file input, the memory does not depend on the file size. The 200 MB measurement is
in "Measurements".

### 5. Limits

1. The block size is at most 64 MiB. This is a rule of the format (ADR 0011, addendum §4, the
   lead's decision after the fuzz findings F1 and F5): the writer refuses a larger block size
   and the reader refuses a header with one before it reads the table. A library caller can
   give a smaller limit for a stream (`StreamOptions::with_max_block_size`).
2. fold refuses an input that needs more than `2^32 - 1` blocks (`TooManyBlocks`), before the
   start for a file, and at that block for a pipe.
3. unfold refuses a file whose header and table claim more bytes than the file holds, before
   any output (§3.2).
4. A library caller can give the largest original length (`max_output_len`). The reader
   refuses a larger header before it reads the table.

### 6. Progress

The library calls a progress function after each block (input bytes and output bytes so far).
The command prints one line on stderr, at most 4 times in each second, only when stderr is a
terminal. `--quiet` stops the progress line and the summary.

## Alternatives that this ADR does not select

1. **Version 3 with a trailer.** The header would have no length, no checksum and no count.
   Each block would have its own entry before its bytes, and a trailer would hold the totals.
   A writer could then stream to a pipe with no temporary file and no delay. It needs a new
   version, a new reader, new conformance vectors and a new freeze. No requirement of
   product-v1 needs it. The lead can select it later; version 2 stays readable (rule 8).
2. **Keep the stored blocks in memory for a pipe.** Simple, but the memory grows with the
   output. Product-v1 §1 refuses it.
3. **A limit on the input size of a pipe.** Simple, but a pipe of logs can be long. The spool
   has no such limit.

## Consequences

- `taco fold - -o x.taco`, `taco fold x -o -`, `taco unfold - -o -` and `-c` work.
- The `.taco` bytes do not change. The conformance vectors do not change. The stream fold and
  `taco::fold` give the same bytes (test).
- A pipe fold needs free disk space for the spool, up to the output length.
- unfold to stdout can write bytes before it finds a damaged block. The exit status tells.

## Measurements

devbox (Linux x86, Ryzen 7 5800H), release build, `conformance/tiny.hsm`, `--threads 6`,
2026-10-19. The input: 200,000,000 bytes of random words with 1 % random 4 KiB regions (a
seeded Python generator). The default profile `auto` gave the profile `text` (no mixed block),
blocks of 64 KiB, `K = 32`, 3,052 blocks, a container of 146,213,809 bytes. Maximum resident
set size from `/usr/bin/time -v` (measured):

| Run | Method | Time | Maximum RSS |
|---|---|---|---|
| fold, file to file | seek back | 18 min 0 s | 254 MiB |
| fold, stdin to stdout (`cat big \| taco fold - > x`) | spool | 17 min 48 s | 276 MiB |
| unfold, file to file, CPU | seek source | 1.6 s | 38 MiB |
| unfold, stdin to stdout, CPU | pipe source | 0.8 s | 35 MiB |

- The pipe container is equal to the file container, byte for byte. Both unfolds restore the
  input (SHA-256 `ac9b69c3…0eecf`).
- The fold memory is mostly the match tables of the mixer (`auto` codes as `logs`: up to 24 MiB
  for each thread, ADR 0010 §7) and one segment: 256 blocks of 64 KiB, with their tokens. It
  does not depend on the input length. The old fold held the input, the tokens of all blocks
  and the container (est. more than 1 GB for this input).
- The unfold of these blocks (zstd and raw) holds one segment.
- Speed against `taco::fold` (the test `stream_speed_against_memory`, 32 MiB of the same input,
  blocks of 16 KiB, profile `text`, 2,048 blocks, 8 segments): fold 108.0 s (memory) and
  123.2 s (stream), then 119.5 s and 122.6 s in a second run. The load average of the shared
  machine was 12 to 14 (measured). Thus the barrier at the end of each segment costs 3 % to
  14 % in these runs. A queue of blocks with no barrier can remove it (not done).
- The CPU unfold of the same container: 80 ms and 83 ms (memory), 100 ms and 109 ms (stream).
