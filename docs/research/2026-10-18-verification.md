# Research: is the idea correct? (2026-10-18)

Three agents examined the idea before the project started. This note gives their results.

## 1. Theory

- **Lossless decoding is a mathematical guarantee** when the encoder and the decoder have the
  same frequency table at every step, and every symbol has a frequency of 1 or more. The
  quality of the model changes only the size. Expected size = H(p) + KL(p || q) bits.
- **Conditions:**
  - the tokenizer must restore every byte string exactly;
  - the model must see only tokens that the decoder already has;
  - the end of the data must be explicit (a length);
  - the versions of the model, the tokenizer and the coder must agree (a hash).
- **Coder width.** A 32-bit coder with a total of 2^16 cannot give 50,257 tokens a useful
  distribution. Taco uses a wider coder (plan §5).
- **Integer engines** are bit-exact if they have no undefined behaviour, no saturating
  instructions, one rounding rule, tables as data, and no floats.
- **Pigeonhole.** No compressor makes all inputs smaller. A raw-block fallback limits the
  growth.

## 2. Experiment (pure Python, integer context model, not a neural network)

| Result | Value |
|---|---|
| Round trips (8 classes and 500 fuzz inputs) | 508 of 508 passed (measured) |
| Worst-case growth | 1 byte (measured) |
| Compressed outputs, Mac (ARM) and devbox (x86) | 8 of 8 identical SHA-256 (measured) |
| Float decoder with float32 arithmetic | 50 of 50 decodes failed (measured) |
| Float decoder with a 1e-9 change | 13 of 50 decodes failed (measured) |
| Speed | about 30 s for each MB (measured, pure Python) |

When a float decode failed, the output after the first different frequency was wrong. In one
case, the first difference was at byte 5,976 of 8,192, so short tests can miss the failure.

## 3. Facts and prior work

| Claim | Result |
|---|---|
| Chinchilla 70B: enwik9 to 8.3%, gzip 32.3% (Delétang et al., 2023) | Confirmed. With the model size included, 14,008%. |
| H.264 uses an exact integer transform to stop decoder drift | Confirmed (Malvar et al.). |
| The Ethereum VM has no floating-point arithmetic | Confirmed. |
| I-BERT: integer-only GELU, softmax, LayerNorm (2021) | Confirmed. |
| Thinking Machines: batch invariance gives determinism on one setup | Confirmed. Not across hardware. |

Prior work that is near to Taco:

- **Ballé, Johnston, Minnen (ICLR 2019):** integer networks for reliable decoding across
  platforms, for image models.
- **ts_zip (Bellard):** a 169M language model. It claims results that do not depend on the CPU
  or the GPU. It uses controlled floats and is a closed binary.
- **PMATIC (2026):** accepts differences and sends extra bits. Plain arithmetic coding decoded 0
  files across two Apple chips (published).
- **Integer-only codecs for LiDAR (2026) and flows (2022).**

**Conclusion.** The principle is known. An open, integer-only language-model compressor that
decodes bit for bit across CPU and GPU, and across x86 and ARM, was not found. That is the work
of Taco.
