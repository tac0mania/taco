# ADR 0002: the prediction path uses integers only

Date: 2026-10-18. Status: accepted.

## Context

The decoder must calculate the same frequency table as the encoder at every step. If one
frequency is different, the decoder can lose synchronisation. After that point, all output is
wrong, and no error occurs. Floating-point results change with the order of operations, the
library (cuBLAS, Accelerate, MKL), fused multiply-add, and the math library. A measurement on
2026-10-18 found that the Mac and devbox give different last bits for 347 of 200,000 `exp`
results and 8,344 of 200,000 `sin` results (measured). A published test (PMATIC, 2026) decoded
0 files when the encoder was an M2 Pro and the decoder was an M4 Max (published).

## Decision

The prediction path uses integer arithmetic only. Integer addition and multiplication have one
result in any order. The model file holds integer weights, integer rescale constants and
precalculated tables. A tool makes the model file one time. Every decoder loads that file.

## Consequences

- The ratio becomes a little worse, because integer weights are less precise. Phase 3 measures
  the loss. The loss never causes wrong output.
- We must write integer versions of LayerNorm, GELU and softmax (plan §7).
- GPU kernels must use integer instructions (for example `dp4a` or int8 tensor cores), and must
  give the same bits as the CPU path.
- Prior work uses the same principle for images (Ballé et al., 2019). Taco applies it to a
  language model and tests it across CPU types and GPUs.
