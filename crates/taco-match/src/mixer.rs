//! The integer mixer: GPT-2 weights and a match prediction to the final weights.
//!
//! ADR 0010 section 3 records the rules. `tools/logs/mixer.py` is a Python copy of the same
//! rules. The output goes to `taco_coder::FreqTable::from_weights` (plan §5.1).
//!
//! # Rules
//!
//! For the weights `w` (one for each token) and the prediction `(p, L)`:
//!
//! 1. `W = sum(w)` in `u128`. With no prediction, each weight becomes `max(w, 1)`. Stop.
//! 2. `pg = clamp(floor(w[p] * 2^16 / W), 1, 65535)` (`pg = 1` if `W = 0`): the GPT-2
//!    probability of `p` in units of 2^-16.
//! 3. `b = bucket(L)`: `L - 1` for `L < 16`, then one bucket for each octave, at most 20.
//! 4. The inputs, in units of 1/256 nat: `x_g = stretch(pg)`, `x_m = stretch(hit[b] >> 6)`,
//!    and the bias 256. `hit[b]` is the adaptive probability (22 bits) that a prediction of
//!    bucket `b` is correct.
//! 5. The weight set is `c = 2 * b + a`, with `a = 1` if `p` is the GPT-2 token with the
//!    largest weight (the lowest id on a tie), else 0.
//!    `dot = wg[c] * x_g + wm[c] * x_m + wb[c] * 256`, with the weights in units of 2^-16.
//!    `x = clamp((dot + 2^15) >> 16, -4095, 4095)`. `q = squash(x)` is the final probability
//!    of `p`, in `1..=65535`.
//! 6. `R = max(W - w[p], 1)`, `m = 2^16 - q`, `k = max(0, bitlen(R) + 16 - 62)`. For each
//!    `i != p`: `w'[i] = max(1, (w[i] * m) >> k)`. Then `w'[p] = max(1, (R * q) >> k)`.
//!    Thus `p` has the probability `q / 2^16` (before the 1-floor and the shift), and the
//!    other tokens share `m / 2^16` in the proportions of GPT-2.
//! 7. After the actual token `a` ([`Mixer::update`]), with `y = 1` if `a = p`, else 0:
//!    `err = y * 2^16 - q`. Each weight of set `c` changes by
//!    `(x * err * lr + 2^19) >> 20` for its input `x`, then is clamped to `±2^22`. The rate
//!    is `lr = max(lr_min, floor(lr0 * D / (n + D)))`, with `n` the earlier updates of set `c`
//!    (at most 2^20).
//!    `hit[b]` moves to `y * 2^22` by `2 / (2n + 3)` of the distance (floor), and the count
//!    `n` grows by 1 up to `hit_limit`.
//!
//! Bounds: `w[i] <= R` for `i != p`, so `(w[i] * m) >> k < 2^62`, and `(R * q) >> k < 2^62`.
//! Every output weight is in `1..2^62`. `|x| <= 4095`, `|err| < 2^16` and `lr <= 255`, so a
//! weight change is less than `2^36` before the shift. `|dot| < 3 * 2^22 * 2^12 < 2^36`.
//!
//! DETERMINISM: integer operations only. The shifts of signed values are arithmetic shifts
//! (floor), as the rescale rule of plan §7.2. No `/` or `%` on a signed value.

use crate::error::MatchError;
use crate::logistic::{Logistic, P_ONE, X_MAX};
use crate::types::{MatchLen, Prediction, TokenId};

/// The number of confidence buckets.
pub const BUCKETS: usize = 21;
/// The number of weight sets: one for each bucket and each value of the agreement bit.
const SETS: usize = 2 * BUCKETS;
/// The limit of the update count of a weight set.
const COUNT_LIMIT: u32 = 1 << 20;
/// The number of bits of the adaptive hit probability.
const HIT_BITS: u32 = 22;
/// The limit of each mixer weight: `±2^22` (±64.0).
const W_LIMIT: i64 = 1 << 22;
/// The bias input.
const BIAS: i32 = 256;

/// The constants of the mixer (ADR 0010 section 3). They are part of the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MixerParams {
    lr: u8,
    lr_min: u8,
    lr_decay: u16,
    hit_limit: u16,
    wg_init: i32,
    wm_init: i32,
}

impl MixerParams {
    /// The constants of mixer rule version 1 (ADR 0010): the rate starts at 160 and
    /// falls to 16 (`lr_decay` 64), the hit count limit is 255, and both start weights are 1.0.
    pub const V1: Self = Self {
        lr: 160,
        lr_min: 16,
        lr_decay: 64,
        hit_limit: 255,
        wg_init: 1 << 16,
        wm_init: 1 << 16,
    };

    /// Makes and checks the constants. The learning rate of a weight set with `n` earlier
    /// updates is `max(lr_min, lr * lr_decay / (n + lr_decay))`, or `lr` if `lr_decay` is 0.
    /// `wg_init` and `wm_init` are the start weights of the GPT-2 input and the match input,
    /// in units of 2^-16.
    ///
    /// # Errors
    ///
    /// [`MatchError::InvalidParams`] if `lr` or `hit_limit` is 0, `lr_min` is more than `lr`,
    /// or a start weight is outside `±2^22`.
    pub fn new(
        lr: u8,
        lr_min: u8,
        lr_decay: u16,
        hit_limit: u16,
        wg_init: i32,
        wm_init: i32,
    ) -> Result<Self, MatchError> {
        if lr == 0 || hit_limit == 0 || lr_min > lr {
            return Err(MatchError::InvalidParams {
                reason: "lr and hit_limit must be >= 1, and lr_min <= lr",
            });
        }
        let lim = W_LIMIT;
        if i64::from(wg_init).abs() > lim || i64::from(wm_init).abs() > lim {
            return Err(MatchError::InvalidParams { reason: "a start weight is outside 2^22" });
        }
        Ok(Self { lr, lr_min, lr_decay, hit_limit, wg_init, wm_init })
    }

    /// The learning rate of a weight set after `n` updates.
    fn rate(&self, n: u32) -> i64 {
        if self.lr_decay == 0 {
            return i64::from(self.lr);
        }
        let d = u32::from(self.lr_decay);
        // INVARIANT: lr < 2^8 and d < 2^16, so the product is < 2^24.
        let r = u32::from(self.lr) * d / (n + d);
        i64::from(r.max(u32::from(self.lr_min)))
    }
}

/// The inputs of the last [`Mixer::mix`] with a prediction, for [`Mixer::update`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pending {
    token: TokenId,
    bucket: usize,
    set: usize,
    q: u32,
    xs: [i32; 3],
}

/// The mixer state of one block.
#[derive(Clone, Debug)]
pub struct Mixer {
    params: MixerParams,
    tables: Logistic,
    hit: [u32; BUCKETS],
    count: [u16; BUCKETS],
    weights: [[i32; 3]; SETS],
    updates: [u32; SETS],
    pending: Option<Pending>,
}

impl Mixer {
    /// Makes a mixer in its start state. Cost: it calculates the logistic tables (about
    /// 4,096 `u128` divisions and 144 KiB).
    #[must_use]
    pub fn new(params: MixerParams) -> Self {
        let mut m = Self {
            params,
            tables: Logistic::new(),
            hit: [0; BUCKETS],
            count: [0; BUCKETS],
            weights: [[0; 3]; SETS],
            updates: [0; SETS],
            pending: None,
        };
        m.reset();
        m
    }

    /// Puts the mixer into its start state for a new block. After `reset`, the mixer gives
    /// the same outputs as a new mixer.
    pub fn reset(&mut self) {
        self.hit = [1 << (HIT_BITS - 1); BUCKETS];
        self.count = [0; BUCKETS];
        self.weights = [[self.params.wg_init, self.params.wm_init, 0]; SETS];
        self.updates = [0; SETS];
        self.pending = None;
    }

    /// Changes the GPT-2 weights into the final weights (rules 1 to 6).
    ///
    /// Cost: two passes over the weights, with one `u128` multiplication for each weight when
    /// there is a prediction.
    ///
    /// # Errors
    ///
    /// - [`MatchError::EmptyWeights`] if `weights` is empty.
    /// - [`MatchError::TokenOutOfRange`] if the predicted token is not inside `weights`.
    ///
    /// The weights do not change when the call returns an error.
    pub fn mix(
        &mut self,
        weights: &mut [u64],
        prediction: Option<Prediction>,
    ) -> Result<(), MatchError> {
        self.pending = None;
        if weights.is_empty() {
            return Err(MatchError::EmptyWeights);
        }
        let Some(pred) = prediction else {
            for w in weights.iter_mut() {
                *w = (*w).max(1);
            }
            return Ok(());
        };
        let p = usize::try_from(pred.token.get()).unwrap_or(usize::MAX);
        let Some(&wp) = weights.get(p) else {
            return Err(MatchError::TokenOutOfRange {
                token: pred.token.get(),
                vocab: weights.len(),
            });
        };
        // INVARIANT: at most 2^64 weights below 2^64, so the sum is below 2^128.
        let mut total: u128 = 0;
        let mut top = 0usize;
        let mut top_w = 0u64;
        for (i, &w) in weights.iter().enumerate() {
            total += u128::from(w);
            if w > top_w {
                top = i;
                top_w = w;
            }
        }
        let q = self.mix_stats(total, wp, top == p, Some(pred)).unwrap_or(1);
        apply(weights, p, total, top_w, q);
        Ok(())
    }

    /// Rules 1 to 4 and the pending state of [`Mixer::mix`] from the statistics of the weights:
    /// `total = sum(w)`, `predicted = w[p]`, and `agree` (`p` is the token with the largest
    /// weight, the lowest id on a tie). Gives `q`, or `None` without a prediction. A later
    /// [`Mixer::update`] gives the state that `mix` and `update` give.
    ///
    /// This is the host step of the mixed coder op of a GPU (`hardshell-cuda`,
    /// `MIXED-INTEGRATION.md` section 3): the GPU gives the statistics, and the mixer state
    /// stays on the host. [`Mixer::mix`] uses it, so one code path sets the state. Cost: O(1).
    pub fn mix_stats(
        &mut self,
        total: u128,
        predicted: u64,
        agree: bool,
        prediction: Option<Prediction>,
    ) -> Option<u32> {
        self.pending = None;
        let pred = prediction?;
        let q = self.probability(u128::from(predicted), total, pred.length, agree);
        self.pending = Some(Pending { token: pred.token, ..q });
        Some(q.q)
    }

    /// Updates the state with the actual token (rule 7). It does nothing if the last call
    /// of [`Mixer::mix`] had no prediction.
    pub fn update(&mut self, actual: TokenId) {
        let Some(pend) = self.pending.take() else { return };
        let y = actual == pend.token;
        let err = if y { i64::from(P_ONE) } else { 0 } - i64::from(pend.q);
        let n = self.updates.get(pend.set).copied().unwrap_or(0);
        let lr = self.params.rate(n);
        if let Some(u) = self.updates.get_mut(pend.set) {
            *u = (n + 1).min(COUNT_LIMIT);
        }
        if let Some(wv) = self.weights.get_mut(pend.set) {
            for (w, &x) in wv.iter_mut().zip(pend.xs.iter()) {
                // INVARIANT: |x * err * lr| < 2^12 * 2^16 * 2^8 = 2^36 (module bounds).
                let d = (i64::from(x) * err * lr + (1 << 19)) >> 20;
                let v = (i64::from(*w) + d).clamp(-W_LIMIT, W_LIMIT);
                // INVARIANT: |v| <= 2^22, so it fits in an i32.
                *w = i32::try_from(v).unwrap_or(0);
            }
        }
        let b = pend.bucket.min(BUCKETS - 1);
        let n = u32::from(self.count[b]);
        let h = self.hit[b];
        let den = 2 * n + 3;
        // INVARIANT: h <= 2^22, so the products are below 2^23 and h stays in 0..=2^22.
        self.hit[b] = if y { h + (((1 << HIT_BITS) - h) * 2) / den } else { h - (h * 2) / den };
        let next = (n + 1).min(u32::from(self.params.hit_limit));
        // INVARIANT: next <= hit_limit <= u16::MAX.
        self.count[b] = u16::try_from(next).unwrap_or(u16::MAX);
    }

    /// Rules 2 to 5: the final probability of the predicted token.
    fn probability(&self, wp: u128, total: u128, length: MatchLen, agree: bool) -> Pending {
        let pg = (wp << 16).checked_div(total).unwrap_or(1);
        // INVARIANT: pg <= 2^16 because wp <= total; the clamp gives 1..=65535.
        let pg = u32::try_from(pg).unwrap_or(P_ONE).clamp(1, P_ONE - 1);
        let b = bucket(length);
        let xs =
            [self.tables.stretch(pg), self.tables.stretch(self.hit[b] >> (HIT_BITS - 16)), BIAS];
        let set = 2 * b + usize::from(agree);
        let wv = self.weights[set];
        let dot: i64 = wv.iter().zip(xs.iter()).map(|(&w, &x)| i64::from(w) * i64::from(x)).sum();
        let x = ((dot + (1 << 15)) >> 16).clamp(-i64::from(X_MAX), i64::from(X_MAX));
        let q = self.tables.squash(x);
        Pending { token: TokenId::new(0), bucket: b, set, q, xs }
    }
}

/// The confidence bucket of a match length: `L - 1` for `L < 16`, then one for each octave.
#[must_use]
pub fn bucket(length: MatchLen) -> usize {
    let l = length.get();
    if l < 16 {
        // INVARIANT: 1 <= l < 16.
        return usize::try_from(l - 1).unwrap_or(0);
    }
    let bits = 32 - l.leading_zeros(); // >= 5
    usize::try_from(15 + bits - 5).unwrap_or(BUCKETS - 1).min(BUCKETS - 1)
}

/// Rule 6: the predicted token `p` gets the probability `q / 2^16`.
fn apply(weights: &mut [u64], p: usize, total: u128, top_w: u64, q: u32) {
    let wp = weights.get(p).copied().map_or(0, u128::from);
    // INVARIANT: wp <= total, so the subtraction cannot overflow.
    let r = (total - wp).max(1);
    let m = u128::from(P_ONE - q);
    let bits = 128 - r.leading_zeros();
    // NOTE: ADR 0010 section 3 names this saturating subtraction: k = max(0, bitlen(R) - 46).
    let k = (bits + 16).saturating_sub(62);
    if top_w < 1 << 48 {
        // PERF: HardShell weights are below 2^41 (ADR 0004 section 5), so w * m < 2^64 and
        // the product fits in a u64. The same values as the u128 loop below. Cost for each
        // token: 216 us with the u128 loop only, 112 us with this loop (measured,
        // `examples/cost.rs`, Mac M2 shared with other work, 50,257 weights).
        let m64 = u64::from(P_ONE - q);
        for w in weights.iter_mut() {
            *w = ((*w * m64) >> k).max(1);
        }
    } else {
        for w in weights.iter_mut() {
            *w = narrow((u128::from(*w) * m) >> k).max(1);
        }
    }
    if let Some(w) = weights.get_mut(p) {
        *w = narrow((r * u128::from(q)) >> k).max(1);
    }
}

/// A value that the bounds of rule 6 keep below 2^62.
fn narrow(x: u128) -> u64 {
    // INVARIANT: x < 2^62 by the bounds in the module documentation.
    u64::try_from(x).unwrap_or(u64::MAX)
}

/// The final probability of the predicted token and the bucket, for tests and tools.
impl Mixer {
    /// The probability `q` (in units of 2^-16) that [`Mixer::mix`] would give the predicted
    /// token now, without a change of state.
    #[must_use]
    /// `agree` says that the predicted token is the GPT-2 token with the largest weight.
    pub fn peek(&self, wp: u64, total: u128, length: MatchLen, agree: bool) -> u32 {
        self.probability(u128::from(wp), total, length, agree).q
    }
}
