//! Many rows of one slot in one call, and the truncation of a slot: the engine side of
//! speculative decoding (ADR 0017).
//!
//! The decoder runs the known token and the draft tokens of a slot in one call
//! ([`GpuBatch::run_rows`] or [`GpuBatch::run_rows_coded`]; a slot can occur many times, and
//! its rows run at consecutive positions in one pass), then truncates the slot after the first
//! token that differs from the draft ([`GpuBatch::truncate`]). The rules of the truncation are
//! the rules of `hardshell::gpt2::StreamState`, so a slot gives the weights of a CPU session
//! with the same calls, bit for bit.
//!
//! DETERMINISM: a pass with many rows of one slot is the pass of a restart (ADR 0015): the
//! `kv_store` dispatch writes the keys and values of all rows before the `attend` dispatch
//! reads them, so each row sees the rows before it, as on the CPU.

use hardshell::TokenId;
use hardshell::gpt2::StreamState;

use super::context::Row;
use super::record::Mode;
use super::{CoderInterval, CoderQuery, GpuBatch};
use crate::error::{Error, Result};

impl GpuBatch<'_> {
    /// The length of the stream of `slot`: the number of inputs since its last reset (0 for a
    /// slot that is not valid). The `<|endoftext|>` input of a block start is one input.
    #[must_use]
    pub fn stream_len(&self, slot: usize) -> usize {
        self.streams.get(slot).map_or(0, StreamState::len)
    }

    /// The number of tokens that `slot` can take before its next context restart: a draft of
    /// at most this many rows never needs a rebuild at a [`GpuBatch::truncate`] (0 for a slot
    /// that is not valid).
    #[must_use]
    pub fn room(&self, slot: usize) -> usize {
        if slot >= self.slots {
            return 0;
        }
        StreamState::room(self.position(slot), self.model.n_ctx(), self.overlap.tokens())
    }

    /// Runs the inputs `(slot, token)` and gives the weights of each input, in input order
    /// (`inputs.len() * vocab` values). A slot can occur many times: its tokens run at
    /// consecutive positions, and a slot whose context is full starts again first (ADR 0015).
    /// Each row gives the weights of `hardshell::gpt2::Session::step` with the same inputs, bit
    /// for bit. [`GpuBatch::weights`] keeps the old values.
    ///
    /// Cost: one pass for each `max_rows` rows (and a pass before a restart of a slot that has
    /// rows in the pass); one read back of `vocab * 8` bytes for each row.
    ///
    /// # Errors
    /// [`Error::BadSlot`] if a slot is not valid; [`Error::Model`] with `TokenOutOfRange` if a
    /// token is not in the vocabulary (no slot changed); [`Error::Gpu`] on a GPU error (then
    /// reset the slots of the call).
    pub fn run_rows(&mut self, inputs: &[(usize, TokenId)]) -> Result<Vec<u64>> {
        let v = self.model.vocab_size();
        let mut words = vec![0u32; 2 * v * inputs.len()];
        self.run_planned(inputs, None, &mut words)?;
        Ok(words
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u64::from(p[0]) | (u64::from(p[1]) << 32))
            .collect())
    }

    /// Runs the inputs as [`GpuBatch::run_rows`] does, then makes the coder interval of each
    /// input on the GPU and reads back only the intervals, as [`GpuBatch::step_coded`] does.
    /// `queries[i]` belongs to `inputs[i]`. For the draft rows of a decoder, the first row of a
    /// slot asks [`CoderQuery::Decode`] and the later rows [`CoderQuery::Encode`] of the draft
    /// token (ADR 0017).
    ///
    /// # Errors
    /// The errors of [`GpuBatch::run_rows`] and of [`GpuBatch::step_coded`].
    pub fn run_rows_coded(
        &mut self,
        inputs: &[(usize, TokenId)],
        queries: &[CoderQuery],
        freq_bits: u32,
    ) -> Result<Vec<CoderInterval>> {
        if queries.len() != inputs.len() {
            return Err(Error::LengthMismatch { expected: inputs.len(), found: queries.len() });
        }
        let words = self.coder_words(queries, freq_bits)?;
        let mut out = vec![0u32; 4 * inputs.len()];
        self.run_planned(inputs, Some(&words), &mut out)?;
        let vocab = self.model.dims.vocab;
        Ok(out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| CoderInterval {
                symbol: TokenId::new(w[0]),
                start: u64::from(w[1]),
                freq: u64::from(w[2]) | (u64::from(w[3]) << 32),
                is_last: w[0] + 1 == vocab,
            })
            .collect())
    }

    /// Truncates the stream of `slot` to its first `len` inputs: the slot then gives the
    /// weights of a slot that ran only these inputs, as `hardshell::gpt2::Session::truncate`,
    /// bit for bit. [`GpuBatch::weights`] keeps the old values.
    ///
    /// Cost: none inside the current context. A length before the last context restart runs
    /// the rows of the old context that the new context wrote again (restart passes, no read
    /// back).
    ///
    /// # Errors
    /// [`Error::BadSlot`] if the slot is not valid; [`Error::Model`] with `BadTruncate` if
    /// `len` is more than [`GpuBatch::stream_len`] or before the context restart before the
    /// last one; [`Error::Gpu`] on a GPU error. The slot does not change on an error.
    pub fn truncate(&mut self, slot: usize, len: usize) -> Result<()> {
        let n_ctx = self.model.n_ctx();
        let Some(st) = self.streams.get(slot).copied() else {
            return Err(Error::BadSlot { slot, slots: self.slots });
        };
        let t = st.plan_truncate(len, self.position(slot), n_ctx)?;
        if t.restore {
            let prev = self.prev.get(slot).cloned().unwrap_or_default();
            let rows: Vec<Row> =
                t.rebuild.clone().filter_map(|p| prev.get(p).map(|&tok| (slot, tok, p))).collect();
            for pass in rows.chunks(self.max_rows) {
                self.forward(pass, Mode::Restart)?;
            }
            if let Some(h) = self.hist.get_mut(slot) {
                h.set_tokens(&prev);
            }
        }
        if let (Some(s), Some(p)) = (self.streams.get_mut(slot), self.pos.get_mut(slot)) {
            s.commit_truncate(&t, n_ctx);
            *p = t.pos;
        }
        Ok(())
    }

    /// The coder query words (kind, value, 0, 0) of `queries`, and the coder parameters.
    fn coder_words(&self, queries: &[CoderQuery], freq_bits: u32) -> Result<Vec<u32>> {
        let vocab = self.model.dims.vocab;
        if freq_bits == 0 || freq_bits > 32 || u64::from(vocab) > 1u64 << freq_bits {
            return Err(Error::Unsupported {
                reason: "freq_bits must be in 1..=32 and 2^freq_bits >= vocab",
            });
        }
        let total = 1u64 << freq_bits;
        let mut words = Vec::with_capacity(4 * queries.len());
        for q in queries {
            let (kind, v) = match *q {
                CoderQuery::Encode(t) if t.get() < vocab => (0, t.get()),
                CoderQuery::Decode(x) if x < total => (1, u32::try_from(x).unwrap_or(u32::MAX)),
                _ => {
                    return Err(Error::Unsupported {
                        reason: "a coder token or target is out of range",
                    });
                }
            };
            words.extend([kind, v, 0, 0]);
        }
        // INVARIANT: 1 <= vocab <= total <= 2^32, so T - V < 2^32.
        let spare = u32::try_from(total - u64::from(vocab)).unwrap_or(u32::MAX);
        let (t_lo, t_hi) = ((total & 0xffff_ffff) as u32, (total >> 32) as u32);
        self.model.dev.write_words(&self.plan.io.freq_params, &[vocab, t_lo, t_hi, spare]);
        Ok(words)
    }

    /// Plans and runs the passes of the inputs: rows in input order, a pass of at most
    /// `max_rows` rows, and the restart passes of a slot whose context is full (after a pass
    /// with the earlier rows of that slot). With `queries`, each pass runs the coder op and
    /// `out` gets 4 words for each input; else `out` gets the `2 * vocab` words of the weights.
    fn run_planned(
        &mut self,
        inputs: &[(usize, TokenId)],
        queries: Option<&[u32]>,
        out: &mut [u32],
    ) -> Result<()> {
        let vocab = self.model.dims.vocab;
        for &(slot, tok) in inputs {
            if slot >= self.slots {
                return Err(Error::BadSlot { slot, slots: self.slots });
            }
            if tok.get() >= vocab {
                return Err(Error::Model(hardshell::Error::TokenOutOfRange {
                    token: tok.get(),
                    vocab_size: vocab,
                }));
            }
        }
        let n_ctx = self.model.n_ctx();
        let eot = self.model.eot;
        let k = self.overlap.tokens();
        let mut cur: Vec<(Row, usize)> = Vec::new();
        let mut in_cur = vec![false; self.slots];
        for (i, &(s, t)) in inputs.iter().enumerate() {
            let mut pos = self.position(s);
            if pos >= n_ctx {
                if in_cur.get(s).copied().unwrap_or(false) {
                    self.flush(&mut cur, &mut in_cur, queries, out)?;
                }
                // PLAN(§7.7): the context of this slot starts again, with its last `k` tokens
                // (ADR 0015).
                let mut restart: Vec<Row> = Vec::new();
                if let Some(h) = self.hist.get(s) {
                    h.restart_rows(s, k, eot, &mut restart);
                }
                for group in restart.chunks(self.max_rows) {
                    self.forward(group, Mode::Restart)?;
                }
                if let (Some(h), Some(prev), Some(st)) =
                    (self.hist.get_mut(s), self.prev.get_mut(s), self.streams.get_mut(s))
                {
                    prev.clear();
                    prev.extend_from_slice(h.tokens());
                    h.restart(k, eot);
                    st.restart(k);
                }
                pos = k + 1;
            }
            if let (Some(h), Some(q), Some(st)) =
                (self.hist.get_mut(s), self.pos.get_mut(s), self.streams.get_mut(s))
            {
                h.set(pos, t);
                st.input(pos);
                *q = pos + 1;
            }
            cur.push(((s, t, pos), i));
            if let Some(x) = in_cur.get_mut(s) {
                *x = true;
            }
            if cur.len() >= self.max_rows {
                self.flush(&mut cur, &mut in_cur, queries, out)?;
            }
        }
        self.flush(&mut cur, &mut in_cur, queries, out)
    }

    /// Runs the rows of `cur` in one pass and copies the result of each row to its input.
    fn flush(
        &mut self,
        cur: &mut Vec<(Row, usize)>,
        in_cur: &mut [bool],
        queries: Option<&[u32]>,
        out: &mut [u32],
    ) -> Result<()> {
        if cur.is_empty() {
            return Ok(());
        }
        let rows: Vec<Row> = cur.iter().map(|&(r, _)| r).collect();
        let per = match queries {
            Some(q) => {
                let mut words = Vec::with_capacity(4 * cur.len());
                for &(_, i) in cur.iter() {
                    words.extend_from_slice(q.get(4 * i..4 * i + 4).unwrap_or(&[0, 0, 0, 0]));
                }
                self.model.dev.write_words(&self.plan.io.queries, &words);
                self.forward(&rows, Mode::Intervals)?;
                4
            }
            None => {
                self.forward(&rows, Mode::Rows)?;
                2 * self.model.vocab_size()
            }
        };
        let src = if queries.is_some() { &self.last_intervals } else { &self.last_rows };
        for (r, &(_, i)) in cur.iter().enumerate() {
            if let (Some(dst), Some(v)) =
                (out.get_mut(i * per..(i + 1) * per), src.get(r * per..(r + 1) * per))
            {
                dst.copy_from_slice(v);
            }
        }
        cur.clear();
        in_cur.iter_mut().for_each(|x| *x = false);
        Ok(())
    }
}
