# ADR 0003: the range coder algorithm (format version 1)

Date: 2026-10-18. Status: accepted. Owner: the coder agent. Plan: §5, §5.1.

## Context

The encoder and the decoder must calculate the same integers at every step (plan §3). The plan
gives the requirements of the coder in §5, but not the exact algorithm. Each detail of the
algorithm changes the compressed bytes. Thus the algorithm is part of the format, and this ADR
records it. The crate `taco-coder` has the same text in its `lib.rs` documentation.

## Decision

The coder is an LZMA-style range coder with byte output and a carry. The window is 64 bits.

### State

| Name | Type | Start value | Meaning |
|---|---|---|---|
| `low` | `u128`, less than 2^65 | 0 | Bits 0 to 63 are the low end of the interval. Bit 64 is the carry. |
| `range` | `u64` | 2^64 - 1 | The width of the interval. |
| `cache` | `u8` | 0 | The first held byte. It can still receive a carry. |
| `pending` | count | 0 | The number of held bytes: `cache`, then `pending - 1` bytes of 0xFF. |
| `code` | `u64` | first 8 input bytes, big-endian | Decoder only. The coded value minus `low`. |

The decoder has the same `range` as the encoder at every step. The decoder reads a zero byte
for each position after the end of the input.

### Symbol with a frequency table

A table has `V` symbols and the total `T = 2^b`, with `b` from 0 to 32. Symbol `s` has the
start `c[s]` (the sum of the lower frequencies) and the frequency `f[s]`, with `f[s] >= 1`.

The encoder does these steps:

1. Set `r = range >> b`.
2. Add `r * c[s]` to `low`.
3. If `s` is the last symbol (`s = V - 1`), set `range = range - r * c[s]`. Else set
   `range = r * f[s]`.
4. Renormalise.

The decoder does these steps:

1. Set `r = range >> b`.
2. Set `t = min(code / r, T - 1)`.
3. Find the largest `s` with `c[s] <= t`. Use a binary search over the starts.
4. Subtract `r * c[s]` from `code`.
5. Set `range` with step 3 of the encoder.
6. Renormalise.

**Division rule.** The total is a power of two, so `r` is a shift in the encoder and in the
decoder. The decoder does one `u64` division, `code / r`. The last symbol receives the
remainder `range - r * T`. Thus no part of the range is lost. Each other symbol loses less than
2^-24 of its probability, because `r >= 2^56 / 2^32 = 2^24`.

### Binary decision

`p1` is the probability of 1 in units of 1/4096, with `p1` from 1 to 4095. Let
`bound = (range >> 12) * p1`. The value 1 owns `[0, bound)`. The value 0 owns `[bound, range)`.

- Encoder: for 1, set `range = bound`. For 0, add `bound` to `low` and subtract `bound` from
  `range`.
- Decoder: the bit is 1 if `code < bound`. For 1, set `range = bound`. For 0, subtract `bound`
  from `code` and from `range`.

Then both renormalise. Symbols and binary decisions can alternate in one stream.

### Renormalisation

While `range < 2^56`, shift `range` left by 8 bits. Then the encoder does `shift_low`, and the
decoder sets `code = (code << 8) | next byte`. After a symbol, `range >= 2^24`, so at most 4
bytes move. After a binary decision, `range >= 2^44`, so at most 2 bytes move.

### Carry rule (`shift_low`)

Let `carry` be bit 64 of `low`. Let `w` be bits 0 to 63 of `low`.

1. If `pending = 0`, or `w < 0xFF << 56`, or `carry = 1`, do these steps:
   1. If `pending > 0`, write `cache + carry`. Then write `pending - 1` bytes of
      `(0xFF + carry) mod 256`.
   2. Set `cache = w >> 56`. Set `pending = 0`.
2. Add 1 to `pending`. Set `low = (w << 8) mod 2^64`.

This is the LZMA rule with a 64-bit window. LZMA also writes a first byte that is always 0.
This coder does not write that byte. The interval stays inside `[0, 2^64 - 1)`, so a carry
cannot reach it. The case `pending = 0` puts the first top byte into `cache`.

### Flush (`finish`)

1. For `k` = 64, then `k` = 56: let `v` be `low` rounded up to a multiple of `2^k`. If
   `v < low + range`, set `low = v` and stop. Because `range >= 2^56`, `k = 56` always gives a
   value.
2. Do `shift_low` two times. This writes all held bytes and the top byte of `v`.
3. Remove all zero bytes from the end of the output.

The decoder reads zeros after the end, so it sees the value `v`. The flush writes at most one
byte more than the held bytes. An empty stream gives an empty output.

### Frequency tables (`from_weights`)

The conversion obeys plan §5.1. These details are not in the plan:

- The weight sum `W` is a `u128`. A `u64` sum can overflow with 50,257 weights.
- If `V > T`, the conversion returns an error.
- The table stores the starts `c[s]` as `u32`. Each start is less than `T`, so it fits.

## Measurements

These numbers come from the tests and the benchmark of the crate on the Mac (Apple Silicon).

| Case | Result |
|---|---|
| 50,257 symbols, `T = 2^32`, 1,000,000 symbols from the table distribution | 0.68 bytes more than the ideal size, 0.000047% (measured) |
| 256 symbols, `T = 2^16`, 1,000,000 symbols | 0.98 bytes more, 0.000188% (measured) |
| 2,000,000 binary decisions with random `p1` | 0.41 bytes more, 0.000228% (measured) |
| Encode speed, 50,257 symbols, `T = 2^32` | 75 million symbols each second (measured, criterion, the Mac shared with one other build) |
| Decode speed, same table | 8.9 million symbols each second (measured, same conditions) |
| `from_weights`, 50,257 weights | 0.53 ms for each table (measured, same conditions) |
| Golden vector: SHA-256 of the coded bytes (15,689 bytes) | `91676cdc669fca0eb22ed18f5f629882212523261f41d0d5c568b647e78d798f` (measured) |

The ideal size is the sum of `-log2(f / T)` over the symbols. The limit in the tests is 0.01%
plus 8 bytes. A separate Python copy of this algorithm gave the same golden bytes (measured).

## Consequences

- A change to any rule above changes the compressed bytes. It needs a new container version
  (plan §9) and new conformance vectors (plan §11).
- The decoder must get the exact coded bytes of a block, or those bytes and then zero bytes
  only. The container stores the length of each block (plan §9).
- The decoder must know the number of symbols. The coder does not write an end mark.
- On damaged input, the decoder returns wrong symbols but does not panic. The container
  checksum finds the error (plan §3 rule 4).
- The decoder does one 64-bit division and a binary search for each symbol. The encoder does
  no division.
