//! The mixed coder op in a batch: the profile `logs` with a read back of 48 bytes for each
//! input instead of the weights (ADR 0010, ADR 0016 §5, crate module `mixed`).
//!
//! A pass of [`Out::Mixed`] runs the forward pass and the stats kernel in its recorded graph,
//! reads back the statistics, runs the host mixer for each input of the pass in input order,
//! writes the mix records, and launches the mixed coder kernel on the logits that the pass left
//! on the GPU (and, for [`CudaBatch::run_coded_dual`], the coder kernel of GPT-2 alone). The
//! kernels read the weights from the same source as the coder op of the pass shape
//! (`mix_src`), so no pass writes the full weights.

use hardshell::TokenId;

use super::{
    Ask, CoderInterval, CoderQuery, CudaBatch, LOGIT_TILE, Out, Row, coder_words, intervals_of,
};
use crate::device::{bytes_u64, u32_bytes, u32_of};
use crate::driver::{Arg, DevBuf};
use crate::error::{Error, Result};
use crate::mixed::{
    MIX_NONE, MIX_PLAIN, MixInput, MixStats, NO_MIX, PLAIN, mix_record, prediction_words,
    table_words,
};
use crate::ops::MatKernel;

/// The tables of a mixed call and the host step of the mixer.
pub(crate) struct Mixing<'a> {
    /// The table word of each input: the predicted token, `MIX_NONE` (a mixed table without a
    /// prediction) or `MIX_PLAIN` (the table of GPT-2 alone).
    pub(crate) predicted: &'a [u32],
    /// Gives `q` for input `i` from its statistics.
    pub(crate) mixer: &'a mut dyn FnMut(usize, MixStats) -> u32,
    /// With `Some`, the interval of GPT-2 alone of each input too (3 values for each input).
    pub(crate) plain: Option<&'a mut [u64]>,
}

/// Where the stats kernel and the mixed coder kernel of a pass read the GPT-2 weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MixSrc {
    /// The weights of `logits_weights` (the op-level path).
    Weights,
    /// The i64 logits of `logits_scale` and the table (the fused path, dp4a products).
    Logits,
    /// The compact logits of `gemm_tc*_logits` (the fused path, tensor-core products).
    D16,
}

impl CudaBatch<'_> {
    /// Runs the inputs as [`CudaBatch::run_coded`] does, but the table of each input is the
    /// table that `tables[i]` names: for [`MixInput::Mixed`], the table of the logs profile,
    /// `FreqTable::from_weights(w', freq_bits)` with `w'` the weights of
    /// `taco_match::Mixer::mix(w, prediction)` (ADR 0010 section 3); for [`MixInput::Plain`],
    /// the table of [`CudaBatch::run_coded`] (a coded block in the same step as mixed blocks).
    /// `queries[i]` and `tables[i]` belong to `inputs[i]`.
    ///
    /// For each input with a prediction, the batch calls `mixer(i, stats)` with the
    /// [`MixStats`] of the GPT-2 weights of the input. The call gives `q`, the probability of
    /// the predicted token in units of 2^-16 (rules 2 to 4: `Mixer` with `W = stats.total`,
    /// `w[p] = stats.predicted` and the agreement bit `stats.top == p`). The calls come in
    /// input order, so the caller can update its mixer state between them (in fold, with the
    /// known token of input `i`). An input without a prediction has the weights `max(1, w)`
    /// (rule 1), and no call.
    ///
    /// Cost: as [`CudaBatch::run_coded`], plus for each pass one more read back (24 bytes for
    /// each row), the host step, one write of 32 bytes for each row and one more kernel.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::run_coded`]; [`Error::LengthMismatch`] if `tables` has
    /// another length; [`Error::Unsupported`] if a predicted token is out of range or `mixer`
    /// gives a `q` outside `1..=65535`. After an error no position changed, but `mixer` can
    /// have been called: the caller must drop its mixer state.
    pub fn run_coded_mixed<F>(
        &mut self,
        inputs: &[(usize, TokenId)],
        queries: &[CoderQuery],
        tables: &[MixInput],
        freq_bits: u32,
        mut mixer: F,
    ) -> Result<Vec<CoderInterval>>
    where
        F: FnMut(usize, MixStats) -> u32,
    {
        for len in [queries.len(), tables.len()] {
            if len != inputs.len() {
                return Err(Error::LengthMismatch { expected: inputs.len(), found: len });
            }
        }
        let vocab = self.model.dims.vocab;
        let q = coder_words(queries, vocab, freq_bits)?;
        let predicted = table_words(tables, vocab)?;
        let mut raw = vec![0u64; 3 * inputs.len()];
        let mut m = Mixing { predicted: &predicted, mixer: &mut mixer, plain: None };
        let ask = Ask { out: Out::Mixed(freq_bits), queries: &q, mixing: Some(&mut m) };
        self.run(inputs, ask, &mut raw, None)?;
        Ok(intervals_of(&raw, vocab))
    }

    /// The two intervals of each token for the encoder of the profile `logs`, which codes each
    /// block with GPT-2 alone and mixed from the same HardShell weights (ADR 0010 section 5,
    /// rule 7.1): for each input, the interval of `tokens[i]` in the table of
    /// [`CudaBatch::run_coded`] (first) and in the mixed table of
    /// [`CudaBatch::run_coded_mixed`] with the prediction `predictions[i]` (second). `mixer`
    /// is called as for [`CudaBatch::run_coded_mixed`].
    ///
    /// Cost: as [`CudaBatch::run_coded_mixed`], plus one more coder kernel and one more read
    /// back of 24 bytes for each row in each pass.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::run_coded_mixed`].
    pub fn run_coded_dual<F>(
        &mut self,
        inputs: &[(usize, TokenId)],
        tokens: &[TokenId],
        predictions: &[Option<TokenId>],
        freq_bits: u32,
        mut mixer: F,
    ) -> Result<Vec<(CoderInterval, CoderInterval)>>
    where
        F: FnMut(usize, MixStats) -> u32,
    {
        for len in [tokens.len(), predictions.len()] {
            if len != inputs.len() {
                return Err(Error::LengthMismatch { expected: inputs.len(), found: len });
            }
        }
        let vocab = self.model.dims.vocab;
        let queries: Vec<CoderQuery> = tokens.iter().map(|&t| CoderQuery::Encode(t)).collect();
        let q = coder_words(&queries, vocab, freq_bits)?;
        let predicted = prediction_words(predictions, vocab)?;
        let mut raw = vec![0u64; 3 * inputs.len()];
        let mut plain = vec![0u64; 3 * inputs.len()];
        let mut m = Mixing { predicted: &predicted, mixer: &mut mixer, plain: Some(&mut plain) };
        let ask = Ask { out: Out::Mixed(freq_bits), queries: &q, mixing: Some(&mut m) };
        self.run(inputs, ask, &mut raw, None)?;
        Ok(intervals_of(&plain, vocab).into_iter().zip(intervals_of(&raw, vocab)).collect())
    }

    /// Writes the prediction word of each row of `pass` (the stats kernel reads it).
    pub(super) fn write_predictions(&self, pass: &[Row], m: &Mixing<'_>) -> Result<()> {
        let mut words = Vec::with_capacity(8 * pass.len());
        for row in pass {
            let p = row.input.and_then(|i| m.predicted.get(i).copied()).unwrap_or(MIX_NONE);
            words.extend([p, 0, 0, 0, 0, 0, 0, 0]);
        }
        self.model.dev.ctx.write(&self.b.mix, 0, &u32_bytes(&words))
    }

    /// The host step and the mixed coder kernel of a pass whose graph ran the stats kernel.
    pub(super) fn mix_pass(&self, pass: &[Row], bits: u32, m: &mut Mixing<'_>) -> Result<()> {
        let ctx = &self.model.dev.ctx;
        let mut bytes = vec![0u8; 24 * pass.len()];
        ctx.read(&self.b.stats, &mut bytes)?;
        let vals = bytes_u64(&bytes);
        let mut words = Vec::with_capacity(8 * pass.len());
        for (row, st) in pass.iter().zip(vals.as_chunks::<3>().0) {
            let input = row.input.map(|i| (i, m.predicted.get(i).copied().unwrap_or(MIX_NONE)));
            let rec = match input {
                Some((_, MIX_PLAIN)) => PLAIN,
                Some((i, p)) if p != MIX_NONE => {
                    let stats = MixStats::from_words(*st);
                    let q = (m.mixer)(i, stats);
                    mix_record(stats, p, q)?
                }
                _ => NO_MIX,
            };
            words.extend(rec);
        }
        ctx.write(&self.b.mix, 0, &u32_bytes(&words))?;
        let n = u32_of(pass.len(), "too many rows")?;
        let src = self.mix_src(MatKernel::for_rows(pass.len()));
        if let Some(plain) = m.plain.as_deref_mut() {
            let out = &self.b.plain;
            match src {
                MixSrc::Weights => self.coder_interval(n, bits, out)?,
                MixSrc::Logits => self.coder_logits(n, bits, out)?,
                MixSrc::D16 => self.coder_d16(n, bits, out)?,
            }
            ctx.read(out, &mut bytes)?;
            let vals = bytes_u64(&bytes);
            for (r, row) in pass.iter().enumerate() {
                if let Some(i) = row.input
                    && let (Some(dst), Some(v)) =
                        (plain.get_mut(3 * i..3 * i + 3), vals.get(3 * r..3 * r + 3))
                {
                    dst.copy_from_slice(v);
                }
            }
        }
        self.coder_mixed(src, n, bits)
    }

    /// The weight source of a pass of `kernel` products: the source of its coder op.
    pub(super) fn mix_src(&self, kernel: MatKernel) -> MixSrc {
        if self.op_level {
            MixSrc::Weights
        } else if self.compact_kernel(kernel).is_some() {
            MixSrc::D16
        } else {
            MixSrc::Logits
        }
    }

    /// Launches the stats kernel of `n` rows.
    pub(super) fn mix_stats(&self, src: MixSrc, n: u32) -> Result<()> {
        let m = self.model;
        let b = &self.b;
        let ptr = |x: &DevBuf| Arg::Ptr(x.ptr());
        let vocab = m.dims.vocab;
        let (k, args) = match src {
            MixSrc::Weights => (
                &m.dev.k.mix_stats_weights,
                vec![ptr(&b.weights), ptr(&b.mix), ptr(&b.stats), Arg::U32(vocab)],
            ),
            MixSrc::Logits => (
                &m.dev.k.mix_stats_logits,
                vec![
                    ptr(&b.logit),
                    ptr(&m.logits_exp),
                    ptr(&b.mix),
                    ptr(&b.stats),
                    Arg::U32(vocab),
                    Arg::U32(m.logits_len),
                ],
            ),
            MixSrc::D16 => (
                &m.dev.k.mix_stats_d16,
                vec![
                    ptr(&b.d16),
                    ptr(&b.tmax),
                    ptr(&m.logits_exp),
                    ptr(&b.mix),
                    ptr(&b.stats),
                    Arg::U32(vocab),
                    Arg::U32(vocab.div_ceil(LOGIT_TILE)),
                    Arg::U32(m.logits_len),
                ],
            ),
        };
        m.dev.launch(k, (n, 1), 256, &args)
    }

    /// Launches the mixed coder kernel of `n` rows.
    fn coder_mixed(&self, src: MixSrc, n: u32, bits: u32) -> Result<()> {
        let m = self.model;
        let b = &self.b;
        let ptr = |x: &DevBuf| Arg::Ptr(x.ptr());
        let vocab = m.dims.vocab;
        let total = Arg::U64(1u64 << bits);
        let (k, args) = match src {
            MixSrc::Weights => (
                &m.dev.k.coder_mixed_weights,
                vec![
                    ptr(&b.weights),
                    ptr(&b.mix),
                    ptr(&b.queries),
                    ptr(&b.intervals),
                    Arg::U32(vocab),
                    total,
                ],
            ),
            MixSrc::Logits => (
                &m.dev.k.coder_mixed_logits,
                vec![
                    ptr(&b.logit),
                    ptr(&m.logits_exp),
                    ptr(&b.mix),
                    ptr(&b.queries),
                    ptr(&b.intervals),
                    Arg::U32(vocab),
                    Arg::U32(m.logits_len),
                    total,
                ],
            ),
            MixSrc::D16 => (
                &m.dev.k.coder_mixed_d16,
                vec![
                    ptr(&b.d16),
                    ptr(&b.tmax),
                    ptr(&m.logits_exp),
                    ptr(&b.mix),
                    ptr(&b.queries),
                    ptr(&b.intervals),
                    Arg::U32(vocab),
                    Arg::U32(vocab.div_ceil(LOGIT_TILE)),
                    Arg::U32(m.logits_len),
                    total,
                ],
            ),
        };
        m.dev.launch(k, (n, 1), 256, &args)
    }
}
