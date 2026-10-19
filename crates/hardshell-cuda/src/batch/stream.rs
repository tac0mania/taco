//! The truncation of a slot: the engine side of speculative decoding (ADR 0017).
//!
//! The decoder runs the known token and the draft tokens of a slot in one call (a slot can
//! occur many times in the inputs of [`CudaBatch::run_coded`], [`CudaBatch::run_coded_mixed`]
//! or [`CudaBatch::run_weights`]; its rows run at consecutive positions, as prefill), then
//! truncates the slot after the first token that differs from the draft. The rules are the
//! rules of `hardshell::gpt2::StreamState`, so a slot gives the weights of a CPU session with
//! the same truncations, bit for bit.

use hardshell::gpt2::StreamState;

use super::{CudaBatch, Out, Row};
use crate::error::{Error, Result};

impl CudaBatch<'_> {
    /// The length of the stream of `slot`: the number of inputs since its last reset (0 for a
    /// slot that is not valid). The `<|endoftext|>` input of a block start is one input.
    #[must_use]
    pub fn stream_len(&self, slot: usize) -> usize {
        self.streams.get(slot).map_or(0, StreamState::len)
    }

    /// The number of tokens that `slot` can take before its next context restart: a draft of
    /// at most this many rows never needs a rebuild at a [`CudaBatch::truncate`] (0 for a slot
    /// that is not valid).
    #[must_use]
    pub fn room(&self, slot: usize) -> usize {
        if slot >= self.slots {
            return 0;
        }
        StreamState::room(self.position(slot), self.model.n_ctx(), self.overlap.tokens())
    }

    /// Truncates the stream of `slot` to its first `len` inputs: the slot then gives the
    /// weights of a slot that ran only these inputs, as `hardshell::gpt2::Session::truncate`,
    /// bit for bit.
    ///
    /// Cost: none inside the current context. A length before the last context restart runs
    /// the rows of the old context that the new context wrote again (passes of at most
    /// [`CudaBatch::max_rows`] rows, no read back).
    ///
    /// # Errors
    /// [`Error::BadSlot`] if the slot is not valid; [`Error::Model`] with `BadTruncate` if
    /// `len` is more than [`CudaBatch::stream_len`] or before the context restart before the
    /// last one; [`Error::Cuda`] on a driver error. The slot does not change on an error.
    pub fn truncate(&mut self, slot: usize, len: usize) -> Result<()> {
        let n_ctx = self.model.n_ctx();
        let Some(st) = self.streams.get(slot).copied() else {
            return Err(Error::BadSlot { slot, slots: self.slots });
        };
        let t = st.plan_truncate(len, self.position(slot), n_ctx)?;
        if t.restore {
            let prev = self.prev.get(slot).map_or(&[][..], Vec::as_slice);
            let rows: Vec<Row> = t
                .rebuild
                .clone()
                .filter_map(|pos| prev.get(pos).map(|&token| Row { slot, token, pos, input: None }))
                .collect();
            for pass in rows.chunks(self.max_rows) {
                self.forward(pass, Out::Nothing, None)?;
            }
            self.model.dev.ctx.synchronize()?;
            let prev = self.prev.get(slot).cloned().unwrap_or_default();
            if let Some(dst) = self.hist.get_mut(slot * n_ctx..(slot + 1) * n_ctx)
                && dst.len() == prev.len()
            {
                dst.copy_from_slice(&prev);
            }
        }
        if let (Some(s), Some(p)) = (self.streams.get_mut(slot), self.pos.get_mut(slot)) {
            s.commit_truncate(&t, n_ctx);
            *p = t.pos;
        }
        Ok(())
    }
}
