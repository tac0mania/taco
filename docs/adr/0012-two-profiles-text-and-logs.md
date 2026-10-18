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
| `logs` | the best log model of Route A (or `gpt2.hsm` if Route A loses) | HardShell, match model, mixer, skip rule | logs, JSON lines, CSV, telemetry |

- `taco fold --profile text|logs|auto`. `auto` is the default. The encoder selects the profile
  from a sample of the input. Only the encoder selects, so the selection rule does not need to
  be the same on every machine.
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
