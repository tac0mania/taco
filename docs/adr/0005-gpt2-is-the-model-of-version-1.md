# ADR 0005: GPT-2 small is the model of version 1

Date: 2026-10-18. Status: accepted. Decided by: Rohit.

## Context

devbox has Nanbeige4.2-3B (Q4_K_M GGUF, about 4.4B parameters, est.). It predicts text better
than GPT-2, so it can make smaller files. But it is about 30 times larger. It needs RoPE,
RMSNorm, SiLU and a solution for activation outliers. Its licence was not known when this ADR was written (later read from its GGUF header:
Apache-2.0, HANDOFF.md). Its speed on
the CPU (about 1 token per second, est.) makes the cross-machine check on the 8 GB Mac slow.

## Decision

Version 1 of Taco uses GPT-2 small only. HardShell implements the GPT-2 architecture. Quality
work (phase 3) and speed work (phase 4) use GPT-2. Nanbeige is not in the plan.

## Consequences

- All phases use one model, so each test run is short and the Mac can run every check.
- The compression ratio of version 1 is limited by GPT-2. Phase 0 measures that limit.
- HardShell stays general: the model file defines the dimensions, so a later model of the same
  family needs only a conversion. A model of a new family (for example Llama style) needs new
  blocks. That is phase 6 work, with its own ADR.
