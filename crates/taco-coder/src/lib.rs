//! Range coder, binary coder and frequency tables for Taco: integers in, bytes out.
//!
//! PLAN(§5): this crate is the coder. PLAN(§5.1): [`freq`] converts model weights to
//! frequency tables. The crate is a kernel crate: it has no floats, no I/O, no clock and no
//! randomness. Every operation is an integer operation with one defined result, so every
//! machine writes the same bytes. ADR 0003 records the same algorithm.
//!
//! # The algorithm (format version 1)
//!
//! A change to any rule below changes the compressed bytes. It needs a new container version
//! (plan §9) and new conformance vectors (plan §11).
//!
//! ## State
//!
//! | Encoder | Width | Start value | Meaning |
//! |---|---|---|---|
//! | `low` | `u128`, < 2^65 | 0 | Bits 0..=63: the low end of the interval. Bit 64: the carry. |
//! | `range` | `u64` | 2^64 - 1 | The width of the interval. |
//! | `cache` | `u8` | 0 | The first held byte. It can still receive a carry. |
//! | `pending` | count | 0 | The held bytes: `cache`, then `pending - 1` bytes of 0xFF. |
//!
//! The decoder has `range` (the same values as the encoder) and `code: u64`. At the start,
//! `code` is the first 8 input bytes, big-endian. The decoder reads a zero byte for each
//! position after the end of the input.
//!
//! ## Symbol with a frequency table
//!
//! The table has `V` symbols, the total `T = 2^b` with `b <= 32`, and for symbol `s` the start
//! `c[s]` (the sum of the lower frequencies) and the frequency `f[s] >= 1`.
//!
//! - Encoder: `r = range >> b`. `low += r * c[s]`. If `s = V - 1` (the last symbol),
//!   `range = range - r * c[s]`; else `range = r * f[s]`. Then renormalise.
//! - Decoder: `r = range >> b`. `t = min(code / r, T - 1)`. Find the largest `s` with
//!   `c[s] <= t` (a binary search over the starts). `code -= r * c[s]`. Set `range` with the
//!   encoder rule. Then renormalise.
//!
//! The division rule: the total is a power of two, so the encoder divides with a shift. The
//! decoder does one `u64` division, `code / r`. The last symbol receives the remainder
//! `range - r * T`, so no part of the range is lost. Each other symbol loses less than 2^-24
//! of its probability, because `r >= 2^56 / 2^32 = 2^24`.
//!
//! ## Binary decision
//!
//! `p1` is the probability of 1, in units of 1/4096, with `1 <= p1 <= 4095`.
//! `bound = (range >> 12) * p1`. The value 1 owns `[0, bound)` and the value 0 owns
//! `[bound, range)`.
//!
//! - Encoder: for 1, `range = bound`. For 0, `low += bound` and `range -= bound`.
//! - Decoder: the bit is `code < bound`. For 1, `range = bound`. For 0, `code -= bound` and
//!   `range -= bound`.
//!
//! ## Renormalisation
//!
//! While `range < 2^56`: `range <<= 8`, then the encoder does `shift_low` and the decoder does
//! `code = (code << 8) | next_byte`. A symbol leaves `range >= 2^24`, so at most 4 bytes
//! move. A binary decision leaves `range >= 2^44`, so at most 2 bytes move.
//!
//! ## Carry rule (`shift_low`)
//!
//! Let `carry` be bit 64 of `low` and `w` the lower 64 bits.
//!
//! 1. If `pending = 0`, or `w < 0xFF << 56`, or `carry = 1`:
//!    - if `pending > 0`, write `cache + carry`, then `pending - 1` bytes of
//!      `(0xFF + carry) mod 256`;
//!    - set `cache = w >> 56` and `pending = 0`.
//! 2. `pending += 1`. `low = (w << 8) mod 2^64`.
//!
//! This is the LZMA rule with a 64-bit window. LZMA also writes a first byte that is always 0.
//! This coder does not write it: the interval stays inside `[0, 2^64 - 1)`, so no carry can
//! reach that byte. The `pending = 0` case puts the first top byte into `cache`.
//!
//! ## Flush (`finish`)
//!
//! 1. For `k` in (64, 56): let `v` be `low` rounded up to a multiple of `2^k`. If
//!    `v < low + range`, set `low = v` and stop. Because `range >= 2^56`, `k = 56` always
//!    succeeds.
//! 2. Do `shift_low` two times. This writes all held bytes and the top byte of `v`.
//! 3. Remove all trailing zero bytes from the output.
//!
//! The decoder reads zeros after the end, so it sees the value `v`. The flush writes at most
//! one byte more than the held bytes. An empty stream gives an empty output.
//!
//! # Cost
//!
//! - [`Encoder::encode`]: one shift, two multiplications, no division.
//! - [`Decoder::decode`]: one `u64` division and one binary search over `V` starts.
//! - [`Encoder::encode_bit`] and [`Decoder::decode_bit`]: one shift and one multiplication.
//! - [`FreqTable::from_weights`]: one `u128` division for each symbol.

#![deny(clippy::float_arithmetic)]

mod cast;
mod decoder;
mod encoder;
mod error;
pub mod freq;
mod types;

pub use decoder::Decoder;
pub use encoder::Encoder;
pub use error::CoderError;
pub use freq::{Freq, FreqBits, FreqTable};
pub use types::{Prob12, Symbol};

/// The renormalisation bound: after each operation, `range >= 2^56`.
pub(crate) const TOP: u64 = 1 << 56;

/// The start value of `range` in the encoder and in the decoder.
pub(crate) const INITIAL_RANGE: u64 = u64::MAX;
