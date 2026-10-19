# ADR 0012: two profiles, text and logs, in one product

Date: 2026-10-19. Status: accepted. Decided by: Rohit.

## Context

Taco beats xz by 27% to 38% on English and HTML (measured). On logs, GPT-2 alone only ties xz
(measured). Logs need other parts: a match model, a mixer, a skip rule, speculative decoding,
and possibly a model that is fine-tuned on logs (Route A). Rohit asked for two versions: one
for logs and one for normal text. Both must be ready for production, with high throughput and
support for different machines.

## Decision

Taco has one engine, one container format and one command. It has two **profiles**. A profile
is a model file and a set of coding parts.

| Profile | Model file | Coding parts | Target data |
|---|---|---|---|
| `text` | `gpt2.hsm` (GPT-2 small, 124M) | HardShell only | prose, documents, HTML, code |
| `logs` | the best log model of Route A (or `gpt2.hsm` if Route A loses) | HardShell, match model, mixer, skip rule (planned, not built) | logs, JSON lines, CSV, telemetry |

- `taco fold --profile text|logs|auto`. `auto` is the default. The encoder selects the profile
  from a sample of the input. Only the encoder selects, so the selection rule does not need to
  be the same on every machine.
  NOTE (2026-10-19): the built rule is different. `auto` codes every block as `logs` and
  writes a `text` container when no block is mixed (ADR 0010 §5.8,
  `docs/research/2026-10-19-logs-profile.md`). It does not use a sample.
- The container records the model hash and the profile. `taco unfold` reads both, so the user
  never names a profile to decode.
- Both profiles obey every rule of `CLAUDE.md` §1: integer-only prediction, conformance vectors,
  the cross-machine check, and a speed gate.
- A profile changes only data (the model file) and the coding parts that the container names.
  There is no code path that only one machine uses.

## Consequences

- The container needs a profile field and a version change. The zstd block kind (ADR 0011)
  and the match model (ADR 0010) already need version 2. Do all three changes in one format
  version (version 2), not three.
- Each profile has its own conformance vectors and its own quality baseline.
- The release ships two model files, or downloads them on first use (an open decision, see
  `docs/design/product-v1.md`).

## Addendum (2026-10-19): the context overlap in both profiles

Decided by the lead, written by the overlap agent. ADR 0015 adds the context overlap `K` (a
restart of the GPT-2 context keeps the last `K` tokens). It is a coding part (bit 1 of the
coding parts, ADR 0011 §2), and it is not specific to logs. Thus:

- Both profiles permit the coding part "context overlap". The profile `text` still has no
  match model and mixer (bit 0).
- The default overlap of the profile `text` is `K = 0` (the decoder is faster). The default
  of the profiles `logs` and `auto` is half of the model context (`K = 512` for GPT-2), and
  `taco fold --overlap K` selects another value. `auto` codes as `logs`, so it has the
  defaults of `logs`; when it writes the `text` container, that container keeps the `K` of
  the coded blocks.
- The default block size of the profile `text` stays 16 KiB. The default of the profiles
  `logs` and `auto` is 64 KiB (`taco::LOGS_BLOCK_SIZE`).
- The decoder reads `K` and the block size from the header. The user never names them to
  decode.

Why 64 KiB for `logs` (measured, the 18 logs of `tools/data/logs`,
`docs/research/2026-10-19-overlap-cli.md`):

| Blocks, `K` | Bytes | Bits per byte | Against the old defaults |
|---|---:|---:|---:|
| 16 KiB, 0 (the old defaults) | 489,113 | 0.597 | |
| 16 KiB, 512 | 444,436 | 0.542 | −9.1 % |
| 64 KiB, 512 (the new defaults) | 363,234 | 0.443 | −25.7 % |

The speed cost (measured on devbox, `Apache_2k.log`, 171,239 bytes, 6 threads, other jobs on
other cores; not a benchmark):

| Step | 16 KiB, `K = 0` | 16 KiB, `K = 512` | 64 KiB, `K = 512` |
|---|---:|---:|---:|
| fold (the CPU on every engine) | 119.0 s | 185.3 s | 306.8 s (2.6 times) |
| unfold, the CPU | 274.7 s | 382.3 s | 985.7 s (3.6 times) |
| unfold, CUDA (RTX 3060 at its old 45 W limit) | 43.4 s | 85.5 s | 122.5 s (2.8 times) |

Most of the cost of 64 KiB on a small file is less parallel work: 171,239 bytes are 3 blocks,
and fold runs one thread for each block and unfold one lane or slot for each block. For each
byte, a fold of one thread takes 23 % more CPU time with 64 KiB than with 16 KiB (measured:
35,590 against 28,880 job-seconds for the 18 logs). A file of 384 KiB or more gives each of 6
threads a block. The size gain is larger than the cost, so 64 KiB is the default. A user who
needs speed can give `--block-size 16384` or `--overlap 0`; the decoder reads both from the
header.
