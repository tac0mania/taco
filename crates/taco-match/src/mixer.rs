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
//! Mixer rule version 2 (ADR 0017) adds the skip rule of module `skip`: with
//! [`MixerParams::V2`], the caller calls [`Mixer::skip`] first at each position, and calls
//! [`Mixer::mix`] (or [`Mixer::mix_stats`]) only if it gives `None`.
//!
//! DETERMINISM: integer operations only. The shifts of signed values are arithmetic shifts
//! (floor), as the rescale rule of plan §7.2. No `/` or `%` on a signed value.

use crate::error::MatchError;
use crate::logistic::{Logistic, P_ONE, X_MAX};
use crate::skip::{Adaptive, SkipCode, SkipParams};
use crate::types::{MatchLen, Prediction, TokenId};

/// The number of confidence buckets.
pub const BUCKETS: usize = 21;
/// The number of weight sets: one for each bucket and each value of the agreement bit.
const SETS: usize = 2 * BUCKETS;
/// The limit of the update count of a weight set.
const COUNT_LIMIT: u32 = 1 << 20;
/// The number of bits of the adaptive hit probability.
pub(crate) const HIT_BITS: u32 = 22;
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
    /// The skip rule (mixer rule version 2), or `None` (rule version 1).
    skip: Option<SkipParams>,
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
        skip: None,
    };

    /// The constants of mixer rule version 2 (ADR 0017): the constants of rule version 1 and
    /// the skip rule with [`SkipParams::V2`].
    pub const V2: Self = Self::V1.with_skip(SkipParams::V2);

    /// The same constants with the skip rule `skip` (module `skip`).
    #[must_use]
    pub const fn with_skip(mut self, skip: SkipParams) -> Self {
        self.skip = Some(skip);
        self
    }

    /// The skip rule, or `None` for a rule without it (rule version 1).
    #[must_use]
    pub const fn skip(&self) -> Option<SkipParams> {
        self.skip
    }

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
        Ok(Self { lr, lr_min, lr_decay, hit_limit, wg_init, wm_init, skip: None })
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

/// What [`Mixer::update`] changes after the last call. `ctx` is the entry of the context
/// table of the skip rule (`None` without a skip rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Next {
    /// A mixed position: the weight set, the hit probability of the bucket, the context entry.
    Mixed { pend: Pending, ctx: Option<usize> },
    /// A skipped position: the hit probability of the bucket and the context entry.
    Skipped { token: TokenId, bucket: usize, ctx: usize },
}

/// One change of the state, kept while a speculative run is open ([`Mixer::checkpoint`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Undo {
    Bucket { b: usize, old: Adaptive },
    Set { set: usize, weights: [i32; 3], updates: u32 },
    Ctx { idx: usize, old: Adaptive },
}

/// A point of a speculative run: [`Mixer::rollback`] gives the state of this point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    mark: usize,
    pending: Option<Next>,
    epoch: u64,
}

/// The mixer state of one block.
#[derive(Clone, Debug)]
pub struct Mixer {
    params: MixerParams,
    tables: Logistic,
    buckets: [Adaptive; BUCKETS],
    weights: [[i32; 3]; SETS],
    updates: [u32; SETS],
    /// The context table of the skip rule: one hit probability for each slot of
    /// (predicted token, bucket). Empty without a skip rule.
    ctx: Vec<Adaptive>,
    pending: Option<Next>,
    /// The changes since the first [`Mixer::checkpoint`] of an open speculative run.
    journal: Option<Vec<Undo>>,
    /// Counts the ends of speculative runs, so an old checkpoint is refused.
    epoch: u64,
}

impl Mixer {
    /// Makes a mixer in its start state. Cost: it calculates the logistic tables (about
    /// 4,096 `u128` divisions and 144 KiB); with a skip rule, also the context table (8 bytes
    /// for each entry: 512 KiB for [`SkipParams::V2`]).
    #[must_use]
    pub fn new(params: MixerParams) -> Self {
        let ctx = params.skip.map_or(0, |s| s.table_len());
        let mut m = Self {
            params,
            tables: Logistic::new(),
            buckets: [Adaptive::START; BUCKETS],
            weights: [[0; 3]; SETS],
            updates: [0; SETS],
            ctx: vec![Adaptive::START; ctx],
            pending: None,
            journal: None,
            epoch: 0,
        };
        m.reset();
        m
    }

    /// Puts the mixer into its start state for a new block. After `reset`, the mixer gives
    /// the same outputs as a new mixer. It ends an open speculative run. Cost: one pass over
    /// the context table.
    pub fn reset(&mut self) {
        self.buckets = [Adaptive::START; BUCKETS];
        self.weights = [[self.params.wg_init, self.params.wm_init, 0]; SETS];
        self.updates = [0; SETS];
        self.ctx.fill(Adaptive::START);
        self.pending = None;
        self.journal = None;
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// The constants of this mixer.
    #[must_use]
    pub const fn params(&self) -> &MixerParams {
        &self.params
    }

    /// Changes the GPT-2 weights into the final weights (rules 1 to 6).
    ///
    /// With a skip rule (rule version 2), call [`Mixer::skip`] first, and `mix` only if it
    /// gives `None`.
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
    /// stays on the host. Cost: O(1).
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
        let ctx = self.params.skip.map(|s| s.slot(pred.token, q.bucket));
        self.pending = Some(Next::Mixed { pend: Pending { token: pred.token, ..q }, ctx });
        Some(q.q)
    }

    /// The skip rule (mixer rule version 2, module `skip`): gives the [`SkipCode`] of this
    /// position if the rule codes it without the GPT-2 weights, and then remembers the
    /// position for [`Mixer::update`]. Gives `None` (and remembers nothing) without a
    /// prediction, for a mixer without a skip rule, or if the rule does not skip; then the
    /// caller calls [`Mixer::mix`] or [`Mixer::mix_stats`]. The decision depends only on the
    /// state, which depends only on earlier tokens. Cost: O(1).
    pub fn skip(&mut self, prediction: Option<Prediction>) -> Option<SkipCode> {
        self.pending = None;
        let pred = prediction?;
        let rule = self.params.skip?;
        let b = bucket(pred.length);
        let idx = rule.slot(pred.token, b);
        let e = self.ctx.get(idx).copied().unwrap_or(Adaptive::START);
        if !rule.skips(b, e.count, e.hit) {
            return None;
        }
        let q = (e.hit >> (HIT_BITS - 16)).clamp(1, P_ONE - 1);
        self.pending = Some(Next::Skipped { token: pred.token, bucket: b, ctx: idx });
        Some(SkipCode { token: pred.token, q })
    }

    /// Opens a speculative run (or continues the open one) and gives the current point. While
    /// a run is open, the mixer keeps a list of its changes, so [`Mixer::rollback`] can give
    /// the state of any point of the run. [`Mixer::commit`] ends the run.
    ///
    /// The decoder of a GPU uses it: the mixer must give `q` for the draft rows of a pass
    /// before the tokens are known, so it updates with the draft tokens and goes back to the
    /// first wrong row (ADR 0017). Cost: O(1); about 24 bytes for each later change.
    pub fn checkpoint(&mut self) -> Checkpoint {
        let journal = self.journal.get_or_insert_with(Vec::new);
        Checkpoint { mark: journal.len(), pending: self.pending, epoch: self.epoch }
    }

    /// Gives the state of the point `cp` of the open speculative run: the run stays open, so
    /// a later rollback to an earlier point of the run is possible. Cost: one step for each
    /// change since the point.
    ///
    /// # Errors
    ///
    /// [`MatchError::StaleCheckpoint`] if `cp` is not a point of the open run (the run ended,
    /// or the mixer was reset). The state does not change.
    pub fn rollback(&mut self, cp: &Checkpoint) -> Result<(), MatchError> {
        let open = self.journal.as_ref().is_some_and(|j| j.len() >= cp.mark);
        if cp.epoch != self.epoch || !open {
            return Err(MatchError::StaleCheckpoint);
        }
        let undo: Vec<Undo> = match self.journal.as_mut() {
            Some(j) => j.drain(cp.mark..).rev().collect(),
            None => Vec::new(),
        };
        for u in undo {
            match u {
                Undo::Bucket { b, old } => {
                    if let Some(e) = self.buckets.get_mut(b) {
                        *e = old;
                    }
                }
                Undo::Set { set, weights, updates } => {
                    if let Some(w) = self.weights.get_mut(set) {
                        *w = weights;
                    }
                    if let Some(n) = self.updates.get_mut(set) {
                        *n = updates;
                    }
                }
                Undo::Ctx { idx, old } => {
                    if let Some(e) = self.ctx.get_mut(idx) {
                        *e = old;
                    }
                }
            }
        }
        self.pending = cp.pending;
        Ok(())
    }

    /// Ends the open speculative run and keeps the current state. Every checkpoint of the run
    /// becomes stale.
    pub fn commit(&mut self) {
        self.journal = None;
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Updates the state with the actual token (rule 7, or rule 4 of the skip rule). It does
    /// nothing if the last call of [`Mixer::mix`], [`Mixer::mix_stats`] or [`Mixer::skip`]
    /// had no prediction (or `skip` gave `None` and no `mix` followed).
    pub fn update(&mut self, actual: TokenId) {
        let (pend, ctx) = match self.pending.take() {
            None => return,
            Some(Next::Skipped { token, bucket, ctx }) => {
                let y = actual == token;
                self.update_bucket(bucket, y);
                self.update_ctx(ctx, y);
                return;
            }
            Some(Next::Mixed { pend, ctx }) => (pend, ctx),
        };
        let y = actual == pend.token;
        let err = if y { i64::from(P_ONE) } else { 0 } - i64::from(pend.q);
        let n = self.updates.get(pend.set).copied().unwrap_or(0);
        let lr = self.params.rate(n);
        if let (Some(j), Some(&w)) = (self.journal.as_mut(), self.weights.get(pend.set)) {
            j.push(Undo::Set { set: pend.set, weights: w, updates: n });
        }
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
        self.update_bucket(pend.bucket, y);
        if let Some(idx) = ctx {
            self.update_ctx(idx, y);
        }
    }

    /// The hit probability and the count of bucket `b` after a prediction with the result `y`.
    fn update_bucket(&mut self, b: usize, y: bool) {
        let limit = self.params.hit_limit;
        if let Some(e) = self.buckets.get_mut(b) {
            if let Some(j) = self.journal.as_mut() {
                j.push(Undo::Bucket { b, old: *e });
            }
            e.update(y, limit);
        }
    }

    /// The entry `idx` of the context table after a prediction with the result `y`.
    fn update_ctx(&mut self, idx: usize, y: bool) {
        let limit = self.params.hit_limit;
        if let Some(e) = self.ctx.get_mut(idx) {
            if let Some(j) = self.journal.as_mut() {
                j.push(Undo::Ctx { idx, old: *e });
            }
            e.update(y, limit);
        }
    }

    /// Rules 2 to 5: the final probability of the predicted token.
    fn probability(&self, wp: u128, total: u128, length: MatchLen, agree: bool) -> Pending {
        let pg = (wp << 16).checked_div(total).unwrap_or(1);
        // INVARIANT: pg <= 2^16 because wp <= total; the clamp gives 1..=65535.
        let pg = u32::try_from(pg).unwrap_or(P_ONE).clamp(1, P_ONE - 1);
        let b = bucket(length);
        let hit = self.buckets[b].hit;
        let xs = [self.tables.stretch(pg), self.tables.stretch(hit >> (HIT_BITS - 16)), BIAS];
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

/// The constants of a mixer rule version of the container (byte 0 of the coding parameters,
/// ADR 0010 section 5): version 1 (ADR 0010) and version 2 (ADR 0017, the skip rule). `None`
/// for another version: the reader refuses it.
#[must_use]
pub fn rule_constants(version: u8) -> Option<(crate::MatchParams, MixerParams)> {
    match version {
        1 => Some((crate::MatchParams::V1, MixerParams::V1)),
        2 => Some((crate::MatchParams::V1, MixerParams::V2)),
        _ => None,
    }
}
