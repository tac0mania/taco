//! The mixed coder op of the profile `logs`: the coder interval of the weights of the match
//! model and the mixer (ADR 0010 section 3), with no read back of the weights (ADR 0016 §5).
//!
//! PLAN(§5.1): the op gives the interval of `FreqTable::from_weights(Mixer::mix(w, p), bits)`
//! in two GPU parts with one host step between them. The match model and the adaptive state of
//! the mixer stay on the host:
//!
//! 1. The GPU gives the statistics [`MixStats`] of the GPT-2 weights `w` of each row: `W =
//!    sum(w)`, `w[p]` for the prediction `p` of the row, and the top token (24 bytes).
//! 2. The host mixer computes `q` from `W`, `w[p]`, `top == p` and its state (rules 2 to 4),
//!    and this module computes the rest of rule 5 exactly as `taco_match::Mixer::mix`:
//!    `R = max(W - w[p], 1)`, `m = 2^16 - q`, `k = max(0, bitlen(R) + 16 - 62)` and
//!    `w'[p] = max(1, (R * q) >> k)` ([`mix_record`]).
//! 3. The GPU makes the table of `w'[i] = max(1, (w[i] * m) >> k)` for `i != p` and `w'[p]`
//!    (or `max(1, w[i])` for a row without a prediction, rule 1), and answers the coder query
//!    (24 bytes).
//!
//! DETERMINISM: integer operations only. The values equal the values of `Mixer::mix`: the
//! statistics are its pass 1 (the u64 sum equals its u128 sum, because GPT-2 weights are at most
//! 2^40), `w[i] <= R` for `i != p` makes each product fit after the shift (`< 2^62`), and the
//! coder op is `from_weights` for every weight sum below 2^64 (the mixed sum is below 2^63).
//! `tests/mixed.rs` compares every interval with the CPU mixer and table.

use hardshell::TokenId;

use crate::batch::{CoderInterval, CoderQuery, coder_words, intervals_of};
use crate::device::{CudaDevice, bytes_u64, u32_bytes, u32_of, u64_bytes};
use crate::driver::Arg;
use crate::error::{Error, Result};

/// The prediction word of a row without a prediction (`MIX_NONE` in `kernels/hardshell.cu`).
pub(crate) const MIX_NONE: u32 = u32::MAX;

/// What the host mixer needs from the GPT-2 weights `w` of one row (ADR 0010 section 3, rules
/// 1 to 3): the values of pass 1 of `taco_match::Mixer::mix`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MixStats {
    /// `W = sum(w)`. GPT-2 weights are at most 2^40, so the sum is exact in a `u64`.
    pub total: u64,
    /// `w[p]` for the predicted token `p` of the row.
    pub predicted: u64,
    /// The token with the largest weight, the lowest id on a tie, and token 0 if every weight
    /// is 0. The agreement bit of rule 3 is `top == p`.
    pub top: TokenId,
}

impl MixStats {
    /// The statistics of the three u64 words of the stats kernel.
    pub(crate) fn from_words(w: [u64; 3]) -> Self {
        Self {
            total: w[0],
            predicted: w[1],
            top: TokenId::new(u32::try_from(w[2]).unwrap_or(u32::MAX)),
        }
    }
}

/// The mix record of a row for the mixed coder kernels (8 u32 words, `kernels/hardshell.cu`):
/// `p`, `k`, `m`, 0, `w'[p]` (low word, high word), 0, 0. The prediction is `p`, and `q` is the
/// probability of `p` from the host mixer, in units of 2^-16.
///
/// The values are rule 5 of ADR 0010 section 3, with the expressions of `taco_match::Mixer`
/// (`apply`): `R = max(W - w[p], 1)`, `m = 2^16 - q`, `k = max(0, bitlen(R) + 16 - 62)`, and
/// `w'[p] = max(1, (R * q) >> k)`.
///
/// # Errors
/// [`Error::Unsupported`] if `q` is not in `1..=65535` (the range of `squash`), or if
/// `w[p] > W` (statistics that no weights give).
pub(crate) fn mix_record(stats: MixStats, p: u32, q: u32) -> Result<[u32; 8]> {
    if q == 0 || q >= 1 << 16 {
        return Err(Error::Unsupported { reason: "the mixer probability q must be in 1..=65535" });
    }
    let total = u128::from(stats.total);
    let wp = u128::from(stats.predicted);
    let r = total
        .checked_sub(wp)
        .ok_or(Error::Unsupported { reason: "the weight of the prediction is above the sum" })?
        .max(1);
    let m = (1u32 << 16) - q;
    let bits = 128 - r.leading_zeros();
    // NOTE: ADR 0010 section 3 names this saturating subtraction: k = max(0, bitlen(R) - 46).
    let k = (bits + 16).saturating_sub(62);
    // INVARIANT: R < 2^64 and q < 2^16, so the product is below 2^80; after the shift it is
    // below 2^62 (k >= bitlen(R) + 16 - 62).
    let wpm = u64::try_from((r * u128::from(q)) >> k)
        .map_err(|_| Error::Unsupported { reason: "a mixed weight is above 2^64" })?
        .max(1);
    let lo = u32::try_from(wpm & 0xffff_ffff).unwrap_or(0);
    let hi = u32::try_from(wpm >> 32).unwrap_or(0);
    Ok([p, k, m, 0, lo, hi, 0, 0])
}

/// The record of a row without a prediction: the kernels use `max(1, w[i])`.
pub(crate) const NO_MIX: [u32; 8] = [MIX_NONE, 0, 0, 0, 0, 0, 0, 0];

/// The table word of a plain row (`MIX_PLAIN` in `kernels/hardshell.cu`): the weights stay.
pub(crate) const MIX_PLAIN: u32 = u32::MAX - 1;

/// The record of a plain row: the table of GPT-2 alone.
pub(crate) const PLAIN: [u32; 8] = [MIX_PLAIN, 0, 0, 0, 0, 0, 0, 0];

/// The table of one input of the mixed coder op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MixInput {
    /// The table of GPT-2 alone, `FreqTable::from_weights(w)`: a coded block (block kind 1).
    Plain,
    /// The table of the logs profile, `FreqTable::from_weights(Mixer::mix(w, prediction))`, with
    /// the prediction of the match model: a mixed block (block kind 3, ADR 0010 section 5).
    Mixed(Option<TokenId>),
}

/// The table words of `tables`, checked against the vocabulary.
///
/// # Errors
/// [`Error::Unsupported`] if a predicted token is not in the vocabulary.
pub(crate) fn table_words(tables: &[MixInput], vocab: u32) -> Result<Vec<u32>> {
    tables
        .iter()
        .map(|t| match *t {
            MixInput::Plain => Ok(MIX_PLAIN),
            MixInput::Mixed(None) => Ok(MIX_NONE),
            MixInput::Mixed(Some(p)) if p.get() < vocab => Ok(p.get()),
            MixInput::Mixed(Some(_)) => {
                Err(Error::Unsupported { reason: "a predicted token is out of range" })
            }
        })
        .collect()
}

/// The prediction words of `predictions`, checked against the vocabulary.
///
/// # Errors
/// [`Error::Unsupported`] if a predicted token is not in the vocabulary.
pub(crate) fn prediction_words(predictions: &[Option<TokenId>], vocab: u32) -> Result<Vec<u32>> {
    predictions
        .iter()
        .map(|p| match p {
            None => Ok(MIX_NONE),
            Some(t) if t.get() < vocab => Ok(t.get()),
            Some(_) => Err(Error::Unsupported { reason: "a predicted token is out of range" }),
        })
        .collect()
}

/// The mixed coder op on given weights, for tests and measurements: `weights` holds one row of
/// `vocab` weights for each query. For each row: the statistics of the GPT-2 weights, then the
/// interval of `queries[i]` in the table of the mixed weights. `mixer(i, stats)` gives `q` for
/// each row `i` that has a prediction, in row order. The sum of the weights of a row must be
/// below 2^64.
///
/// Cost: one upload of the weights, two kernels, and two read backs of 24 bytes for each row.
///
/// # Errors
/// [`Error::LengthMismatch`] if the lengths do not agree; [`Error::Unsupported`] for a bad
/// `freq_bits`, query, prediction or `q`, or a weight sum of 2^64 or more; [`Error::Cuda`] on a
/// driver error.
pub fn mixed_intervals_of_weights<F>(
    dev: &CudaDevice,
    weights: &[u64],
    queries: &[CoderQuery],
    predictions: &[Option<TokenId>],
    freq_bits: u32,
    mut mixer: F,
) -> Result<Vec<(MixStats, CoderInterval)>>
where
    F: FnMut(usize, MixStats) -> u32,
{
    let rows = queries.len();
    if predictions.len() != rows {
        return Err(Error::LengthMismatch { expected: rows, found: predictions.len() });
    }
    if rows == 0 {
        return Ok(Vec::new());
    }
    let vocab = weights.len() / rows;
    if vocab == 0 || vocab * rows != weights.len() {
        return Err(Error::LengthMismatch { expected: rows, found: weights.len() });
    }
    for row in weights.chunks(vocab) {
        if row.iter().try_fold(0u64, |a, &w| a.checked_add(w)).is_none() {
            return Err(Error::Unsupported { reason: "a weight sum is 2^64 or more" });
        }
    }
    let v32 = u32_of(vocab, "the vocabulary is too large")?;
    let q = coder_words(queries, v32, freq_bits)?;
    let pred = prediction_words(predictions, v32)?;
    let n = u32_of(rows, "too many rows")?;
    let ctx = &dev.ctx;
    let w = dev.upload(&u64_bytes(weights))?;
    let mut mix_words: Vec<u32> = Vec::with_capacity(8 * rows);
    for &p in &pred {
        mix_words.extend([p, 0, 0, 0, 0, 0, 0, 0]);
    }
    let mix = dev.upload(&u32_bytes(&mix_words))?;
    let stats_buf = dev.zeroed(24 * rows)?;
    dev.launch(
        &dev.k.mix_stats_weights,
        (n, 1),
        256,
        &[Arg::Ptr(w.ptr()), Arg::Ptr(mix.ptr()), Arg::Ptr(stats_buf.ptr()), Arg::U32(v32)],
    )?;
    let mut bytes = vec![0u8; 24 * rows];
    ctx.read(&stats_buf, &mut bytes)?;
    let stats: Vec<MixStats> =
        bytes_u64(&bytes).as_chunks::<3>().0.iter().map(|c| MixStats::from_words(*c)).collect();
    mix_words.clear();
    for (i, (&p, &st)) in pred.iter().zip(&stats).enumerate() {
        let rec = if p == MIX_NONE { NO_MIX } else { mix_record(st, p, mixer(i, st))? };
        mix_words.extend(rec);
    }
    ctx.write(&mix, 0, &u32_bytes(&mix_words))?;
    let qw: Vec<u32> = q.iter().flat_map(|&(kind, v)| [kind, v, 0, 0]).collect();
    let qbuf = dev.upload(&u32_bytes(&qw))?;
    let out = dev.zeroed(24 * rows)?;
    dev.launch(
        &dev.k.coder_mixed_weights,
        (n, 1),
        256,
        &[
            Arg::Ptr(w.ptr()),
            Arg::Ptr(mix.ptr()),
            Arg::Ptr(qbuf.ptr()),
            Arg::Ptr(out.ptr()),
            Arg::U32(v32),
            Arg::U64(1u64 << freq_bits),
        ],
    )?;
    ctx.read(&out, &mut bytes)?;
    let iv = intervals_of(&bytes_u64(&bytes), v32);
    Ok(stats.into_iter().zip(iv).collect())
}
