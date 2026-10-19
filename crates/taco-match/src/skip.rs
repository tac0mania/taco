//! The skip rule of mixer rule version 2 (ADR 0017): a position that the mixer codes without
//! the GPT-2 weights.
//!
//! `tools/logs/skip.py` is a Python copy of these rules (the golden vectors of
//! `tests/golden.rs` are equal).
//!
//! # Rules
//!
//! The mixer has a context table of `2^table_bits` adaptive hit probabilities (22 bits, start
//! `2^21`, count 0). At a position with the match prediction `(p, L)` and the bucket
//! `b = bucket(L)`:
//!
//! 1. The slot is `((key + 1) * B mod 2^64) >> (64 - table_bits)` with
//!    `key = p * 21 + b` and `B = 0x9E3779B97F4A7C15` ([`SkipParams::slot`]). Its entry has
//!    the hit probability `h` and the count `n`.
//! 2. The mixer skips GPT-2 if `b >= min_bucket`, `n >= min_count` and `h >= min_hit`
//!    ([`crate::Mixer::skip`]). These values depend only on the tokens before the position,
//!    so the encoder and the decoder make the same decision.
//! 3. A skipped position codes a flag first: symbol 0 ("the token is `p`") with the weight
//!    `q_s` and symbol 1 ("another token") with the weight `2^16 - q_s`
//!    ([`SkipCode::flag_weights`]), `q_s = clamp(h >> 6, 1, 65535)`.
//! 4. On symbol 1, the position then codes the token in the table of the GPT-2 weights with
//!    the weight of `p` set to 0 ([`skip_miss_weights`]). Only this case needs the GPT-2
//!    weights of the position.
//! 5. A position that the rule does not skip uses the rules of mixer rule version 1.
//! 6. After the token ([`crate::Mixer::update`]), with `y = 1` if it is `p`: the entry of the
//!    slot and `hit[b]` move to `y * 2^22` with the rule of `hit[b]` of rule version 1 (ADR
//!    0010 §3 rule 6), at every position with a prediction. A mixed position also updates its
//!    weight set; a skipped position does not (it has no GPT-2 input).
//!
//! DETERMINISM: wrapping `u64` multiplication, shifts and comparisons.

use crate::error::MatchError;
use crate::logistic::P_ONE;
use crate::mixer::{BUCKETS, HIT_BITS};
use crate::types::TokenId;

/// The hash multiplier of the slot (the multiplier `B` of the match model).
const SLOT_B: u64 = 0x9E37_79B9_7F4A_7C15;

/// An adaptive hit probability: `hit` in 22 bits and the count of updates `count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Adaptive {
    pub(crate) hit: u32,
    pub(crate) count: u16,
}

impl Adaptive {
    /// The start: probability 0.5, no update.
    pub(crate) const START: Self = Self { hit: 1 << (HIT_BITS - 1), count: 0 };

    /// The rule of ADR 0010 §3 rule 6: `hit` moves to `y * 2^22` by `floor(2 * distance /
    /// (2n + 3))`, and the count grows by 1 to at most `limit`.
    pub(crate) fn update(&mut self, y: bool, limit: u16) {
        let n = u32::from(self.count);
        let h = self.hit;
        let den = 2 * n + 3;
        // INVARIANT: h <= 2^22, so the products are below 2^23 and h stays in 0..=2^22.
        self.hit = if y { h + (((1 << HIT_BITS) - h) * 2) / den } else { h - (h * 2) / den };
        let next = (n + 1).min(u32::from(limit));
        // INVARIANT: next <= limit <= u16::MAX.
        self.count = u16::try_from(next).unwrap_or(u16::MAX);
    }
}

/// The constants of the skip rule. They are part of the mixer rule version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SkipParams {
    min_hit: u32,
    min_count: u16,
    min_bucket: u8,
    table_bits: u8,
}

impl SkipParams {
    /// The constants of mixer rule version 2 (ADR 0017): a context table of 2^16 entries;
    /// skip when the entry has 2 or more earlier predictions and a hit probability of at
    /// least 0.9 (`3,774,873 / 2^22`), in every bucket.
    ///
    /// NOTE: chosen from the grid of `tools/logs/skip.py` (ADR 0017, measured on 18 logs, 64
    /// KiB and 1 MiB blocks).
    pub const V2: Self = Self { min_hit: 3_774_873, min_count: 2, min_bucket: 0, table_bits: 16 };

    /// Makes and checks the constants: `min_hit` is a 22-bit probability, `min_count` an update
    /// count, `min_bucket` a bucket, and `table_bits` the size of the context table.
    ///
    /// # Errors
    ///
    /// [`MatchError::InvalidParams`] if `min_hit` is more than `2^22`, `min_bucket` is not a
    /// bucket (`>= BUCKETS`), or `table_bits` is not in `4..=24`.
    pub fn new(
        min_hit: u32,
        min_count: u16,
        min_bucket: u8,
        table_bits: u8,
    ) -> Result<Self, MatchError> {
        if min_hit > 1 << HIT_BITS {
            return Err(MatchError::InvalidParams { reason: "min_hit must be at most 2^22" });
        }
        if usize::from(min_bucket) >= BUCKETS {
            return Err(MatchError::InvalidParams { reason: "min_bucket must be a bucket" });
        }
        if !(4..=24).contains(&table_bits) {
            return Err(MatchError::InvalidParams { reason: "table_bits must be 4 to 24" });
        }
        Ok(Self { min_hit, min_count, min_bucket, table_bits })
    }

    /// The smallest hit probability (22 bits) of a skip.
    #[must_use]
    pub const fn min_hit(&self) -> u32 {
        self.min_hit
    }

    /// The smallest update count of the context entry of a skip.
    #[must_use]
    pub const fn min_count(&self) -> u16 {
        self.min_count
    }

    /// The smallest bucket of a skip.
    #[must_use]
    pub const fn min_bucket(&self) -> u8 {
        self.min_bucket
    }

    /// The number of bits of a slot of the context table.
    #[must_use]
    pub const fn table_bits(&self) -> u8 {
        self.table_bits
    }

    /// The number of entries of the context table.
    pub(crate) const fn table_len(&self) -> usize {
        1 << self.table_bits
    }

    /// Rule 1: the slot of the predicted token `token` in the bucket `b`.
    pub(crate) fn slot(&self, token: TokenId, b: usize) -> usize {
        // INVARIANT: b < BUCKETS = 21, so the key fits in a u64 for every u32 token.
        let key = u64::from(token.get()) * (BUCKETS as u64) + b as u64;
        let x = key.wrapping_add(1).wrapping_mul(SLOT_B);
        // INVARIANT: the slot is below 2^table_bits <= 2^24, so it fits in a usize.
        usize::try_from(x >> (64 - u32::from(self.table_bits))).unwrap_or(0)
    }

    /// Rule 2: `true` if a position of bucket `b` whose entry has the count `count` and the
    /// hit probability `hit` is skipped.
    pub(crate) fn skips(&self, b: usize, count: u16, hit: u32) -> bool {
        b >= usize::from(self.min_bucket) && count >= self.min_count && hit >= self.min_hit
    }
}

/// The code of a skipped position (rules 3 and 4 of the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SkipCode {
    /// The predicted token.
    pub token: TokenId,
    /// `q_s`: the probability that the token is the predicted token, in units of 2^-16, in
    /// `1..=65535`.
    pub q: u32,
}

impl SkipCode {
    /// The weights of the flag table (rule 3): `[q_s, 2^16 - q_s]`. Symbol 0 is "the token is
    /// the predicted token", symbol 1 is "another token". Both are at least 1.
    #[must_use]
    pub fn flag_weights(&self) -> [u64; 2] {
        let q = self.q.clamp(1, P_ONE - 1);
        [u64::from(q), u64::from(P_ONE - q)]
    }

    /// The flag symbol of the token `actual`: 0 if it is the predicted token, else 1.
    #[must_use]
    pub fn flag_of(&self, actual: TokenId) -> usize {
        usize::from(actual != self.token)
    }
}

/// Rule 4: the weights of the token table after the flag "another token": `max(1, w[i])` for
/// each `i` that is not the predicted token, and 0 for the predicted token (the frequency table
/// gives it the smallest frequency). Cost: one pass over the weights.
///
/// # Errors
///
/// [`MatchError::EmptyWeights`] if `weights` is empty; [`MatchError::TokenOutOfRange`] if the
/// predicted token is not inside `weights`. The weights do not change on an error.
pub fn skip_miss_weights(weights: &mut [u64], predicted: TokenId) -> Result<(), MatchError> {
    if weights.is_empty() {
        return Err(MatchError::EmptyWeights);
    }
    let p = usize::try_from(predicted.get()).unwrap_or(usize::MAX);
    if p >= weights.len() {
        return Err(MatchError::TokenOutOfRange { token: predicted.get(), vocab: weights.len() });
    }
    for w in weights.iter_mut() {
        *w = (*w).max(1);
    }
    if let Some(w) = weights.get_mut(p) {
        *w = 0;
    }
    Ok(())
}
