# ADR 0007: the stored bytes of a coded block (format version 1)

Date: 2026-10-18. Status: accepted (proposed by the taco agent). Plan: §4, §5, §7.7, §9, §10.

## Context

ADR 0006 defines the container. A coded block stores bytes that the container does not
explain. ADR 0003 says that the decoder must know the number of symbols, because the range
coder writes no end mark. Thus the stored bytes of a coded block must hold the token count, and
the order of the calls to HardShell and to the coder must be fixed. Each detail changes the
compressed bytes. Thus these rules are part of the format. The crate `taco` has the same text
in the documentation of `payload.rs` and `codec.rs`.

## Decision

### 1. Layout of the stored bytes

```
payload = varint(token count) || range coder bytes
```

1. The token count is an unsigned LEB128 varint. Each byte holds 7 bits, the low group first.
   Bit 7 is set on each byte except the last byte.
2. The varint is minimal. The last byte is not 0, except for the value 0 (one byte, 0x00). A
   value has at most 10 bytes. Thus each count has exactly one encoding.
3. The range coder bytes follow the varint directly. They are the output of
   `Encoder::finish` (ADR 0003). Their length is the rest of the payload.

### 2. The token stream of a block

1. The encoder tokenizes the original bytes of the block alone (plan §10). A token never
   crosses a block boundary.
2. The encoder never writes `<|endoftext|>` or a token with no bytes. Thus each token has at
   least 1 byte, and a block of `n` bytes has at most `n` tokens.

### 3. The order of the calls

Each block uses a new context. For a block with the tokens `t[0]` to `t[n-1]`:

1. `Session::start_block()` gives the weights for `t[0]`. The context starts with
   `<|endoftext|>` (plan §7.7).
2. For each `i` from 0 to `n - 1`:
   1. Make the table with `FreqTable::from_weights(weights, 32)` (plan §5.1, `T = 2^32`).
   2. Code `t[i]` with the table. The symbol is the token id.
   3. If `i + 1 < n`, `Session::step(t[i])` gives the weights for `t[i + 1]`.
3. HardShell starts the context again when it is full (ADR 0004 section 9).

The decoder does the same calls in the same order. It changes each token id to bytes with the
vocabulary of the model file only.

### 4. Rules of the decoder

The decoder refuses a coded block with a typed error (`CorruptBlock`) in these cases:

1. The varint is truncated, not minimal, or larger than a `u64`.
2. The token count is 0, or more than the original length of the block.
3. A decoded token is `<|endoftext|>` or has no bytes.
4. The decoded bytes are longer or shorter than the original length of the block.

The decoder does not allocate memory from the token count. Other damage gives wrong bytes. The
checksum of the container finds them (plan §3 rule 4).

### 5. Rules of the encoder

These rules select raw or coded. A decoder does not depend on them, but they change the bytes
of the container. Thus the conformance vectors depend on them.

1. A block is raw if the payload is not smaller than the original bytes (`Block::choose`, plan
   §3 rule 6).
2. The encoder stops with a raw block when the varint and the bytes that the coder has already
   written have at least the original length. This saves the model time on bytes that do not
   compress. The rule depends only on the input, so every machine selects the same form.
3. The encoder checks that the tokens give the original bytes with the vocabulary of the model
   before it codes the block. A failure is an error, not a damaged container.

### 6. Block size

The default block size is 16,384 bytes. The block size is a parameter, and the container
stores it (plan §9). The reasons are in the documentation of the `taco` crate (est.). Plan §13
question 3 asks for the measurement with GPT-2.

## Measurements

| Case | Result |
|---|---|
| Golden vector: tiny test model (`crates/taco/tests/common`, SHA-256 `5ba63e7c69063caa1430de0df5a9b3ed76db63de1a779d5ecce25484c88de4ff`), 2,202 bytes, blocks of 512 (coded, coded, raw, raw, coded) | 1,837 bytes, SHA-256 `c3ff718b1c9682589836078b36d936836077744b03b1cc8af3a6ced0d5821b8a` (measured, Mac) |
| GPT-2 (`tools/data/out/gpt2.hsm`, SHA-256 `49ea618d...1183`), 4,096 bytes of `lcet10.txt`, one block | 1,975 bytes, 3.857 bits per byte, 1,062 tokens, fold 16.2 s, unfold 15.2 s (measured, Mac, release build, one thread) |
| The same input, blocks of 512 bytes | 2,074 bytes, 4.050 bits per byte, 6.1 s with 8 threads (measured) |
| One bit flipped in each byte of a container of 315 bytes (315 cases) | 315 errors, 0 wrong outputs (measured) |
| Bits 0, 3 and 7 flipped in each byte of the same container (945 cases, an earlier run of the test) | 943 errors, 2 outputs equal to the original, 0 wrong outputs (measured) |

The two flips that gave the original bytes were in the last bits of a coder stream. The
decoder then decoded the same tokens. This is not a fault: the output is correct.

## Consequences

- A change to a rule in sections 1 to 3 changes the format. It needs a new container version
  and new conformance vectors (plan §3 rule 8, §11).
- A change to a rule in section 5 changes the compressed bytes but not the decoder. It needs
  new conformance vectors.
- The decoder does one HardShell step for each token. A damaged token count can cost time, but
  at most one step for each byte of the original length.
- The cross-machine check (plan §11.3) of the golden vector on devbox is not done.
