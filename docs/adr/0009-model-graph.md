# ADR 0009: the model file holds a graph of integer ops (format version 3)

Date: 2026-10-18. Status: proposed (by the anymodel design agent). Plan: §3, §6, §7, §8, §11,
§12 phase 6. Design: `docs/design/anymodel.md`.

NOTE (2026-10-19): this ADR first named the graph format "version 2". Model-file version 2
is now the `wide` rules (ADR 0004 addendum B, built). Thus the graph format is version 3.

## Context

HardShell runs one fixed forward pass: GPT-2 (`crates/hardshell/src/gpt2`). The converter
`tools/convert` converts only GPT-2. The GPT-2 recipe (`clip` and the beta correction) needed a
manual diagnosis (`docs/research/2026-10-18-integer-gpt2.md` §7).

The owner asks for one converter for many LLMs, with no new code for each model. The main open
decoder families differ in a small number of points (design §3.1, published configs). The
families are GPT-2, GPT-J, NeoX, Llama 2 and 3, Mistral, Qwen 2 and 3, Gemma, Phi, SmolLM and
OLMo 2.
These are the norm kind, the activation, the GLU form, RoPE, GQA, windows, soft-caps, QK-norm,
biases and the wiring of the residual.

ADR 0005 keeps GPT-2 as the only model of version 1. This ADR does not change that. It
proposes the model file format for the work after version 1.

## Decision

1. **Format version 3 of the `.hsm` file holds a graph.** The JSON header gets the fields
   `values`, `caches` and `graph`. The graph is an ordered list of ops. HardShell runs the ops
   in file order for each step. HardShell has no code for a model family.
2. **The op set is fixed and small:** `embed`, `norm` (modes `layer` and `rms`, with a
   `group` length), `linear`, `add`, `table`, `mul`, `rope`, `attention` (GQA, causal,
   `window`, optional score soft-cap), `logits` (optional bias and soft-cap) and, later,
   `hadamard`. Design §4 gives the semantics. Each op uses only the rules of plan §7.2.
3. **Every element-wise activation is a table.** A `table` op reads a tensor at the index
   `clamp(x + offset, 0, len - 1)`. A new activation needs a new table, not engine code.
4. **One graph has one encoding.** The graph lists every op of every layer. It has no loops
   and no templates. The tensors follow in the order of first use. Thus one model has one file
   and one model hash.
5. **The header describes the memory.** Each `attention` op names a cache with `n_kv_head`,
   `head_dim`, type and `window`. The session allocates all caches from the header before the
   first step.
6. **Numbers stay integers.** Every number in the header is a non-negative integer. A negative
   constant uses an `offset` field. Unknown ops, fields, types and names are errors, as in
   version 1.
7. **Version 1 stays readable.** The reader changes a version 1 GPT-2 file into the
   equivalent graph in memory. One interpreter runs both versions. A test shows that
   `conformance/tiny.hsm` and `gpt2.hsm` (version 1) give the golden hashes of `HANDOFF.md`
   §2 through the interpreter. Then the `gpt2` forward-pass module goes away.
8. **The model header changes two fields.** `start_token` replaces `eot_token`. `tokenizer`
   gives the kind (`byte_bpe` or `bpe_byte_fallback`) and the pre-tokenizer regex (`gpt2`,
   `llama3` or `none`). The decoder still uses only the vocabulary bytes (plan §10).
9. **The converter selects the recipe by measurement.** It searches each site on the
   calibration windows with a fixed budget. It records the choices in a committed
   `recipe.json`. The build of the model file runs on the CPU with deterministic NumPy, so two
   machines make the same file. The bias correction (the general form of the beta correction
   of ADR 0004 section 7) is a rule of every converter.

## Consequences

- A new checkpoint of a known family needs no code. A new family needs an architecture map in
  `tools/convert` (about 100 lines of Python, est.). A new kind of operation (for example
  mixture of experts, Mamba, RWKV) needs a new op, its own ADR and new conformance vectors.
- New conformance vectors: one tiny graph model for each feature set (design §5.5). The
  cross-machine check runs all of them. The old vectors stay valid.
- The `simd` and `cuda` work must implement ops (or fusions of ops that give the same bits),
  not one fused GPT-2 pass. Otherwise that work must change again.
- The interpreter adds some cost for each op. The gate of design §11 phase M2 permits at most
  a 3 % speed loss on GPT-2 (est. limit; measure it).
- Version 3 model files of 1B to 7B parameters are 1.2 GB to 7.2 GB (est.). The Mac (8 GB)
  cannot run a 7B model, so the cross-machine check limits the size to about 3B (est.).
- This ADR does not select a model. Design §11 phase M0 measures the float gain of candidate
  models first. If no candidate is at least 10 % smaller than GPT-2, the graph work stops.

## Not decided here

- The exact JSON schema of each op and the bounds that the loader checks. The hardshell agent
  writes them in the module documentation, as for version 1 (ADR 0004 section 8).
- The objective of the search (KL against float, or bits per byte) and the loss target that
  starts a quantisation-aware fine-tune. The owner decides (design §13).
- The `hadamard` op and the constant start prefix (design §5.4, §9.2). Each needs a
  measurement first.
