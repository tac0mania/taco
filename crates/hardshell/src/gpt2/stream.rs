//! The inputs of one stream (a session, a lane of a batch, or a slot of a GPU batch) and its
//! truncation: the engine side of speculative decoding (ADR 0017).
//!
//! An input is a row that the caller gives: the `<|endoftext|>` row of a block start, or a
//! token. The length of a stream is its number of inputs since the last reset. The rows of a
//! context restart (ADR 0015) are not inputs.
//!
//! A truncation to the length `len` gives the state of a stream that ran only its first `len`
//! inputs: the decoder runs `k` draft tokens in one pass, then drops the inputs after the first
//! token that differs from the draft.
//!
//! 1. Inside the current context, the truncation only moves the position back. The rows at
//!    the lower positions do not depend on the later rows (causal attention), so their cache
//!    entries are the entries of a stream that never ran the later rows. The next rows write
//!    over the dropped positions.
//! 2. Before the last restart, the stream takes the history of the context before the restart
//!    (kept at the restart, `n_ctx` tokens) and runs its rows again at the positions that the
//!    new context wrote (the rebuild rows, without weights). A forward row depends only on its
//!    token, its position and the rows before it, so the rebuilt cache entries are the old
//!    entries, bit for bit. The positions that the new context did not write keep their old
//!    entries.
//! 3. A length before the restart before the last one is an error (`Error::BadTruncate`).
//!    A pass with at most `n_ctx - K - 1` rows of one stream never needs it.
//!
//! [`StreamState`] is the bookkeeping of these rules. Every backend uses it (the CPU paths of
//! this crate, `hardshell-cuda` and `hardshell-gpu`), so the rules are in one place; each
//! backend keeps the token history and the KV cache.
//!
//! PERF: a rebuild runs up to `K + 1 + d` rows for a pass of `d` rows that crossed a restart
//! (est. for GPT-2 with `K = 512`: about 0.5 of the rows of a full window). A decoder that
//! ends each draft at the restart ([`StreamState::room`]) never rebuilds.
//!
//! DETERMINISM: the rebuild rows are normal forward rows of the kept tokens.

use core::ops::Range;

use super::context::History;
use super::engine::Row;
use crate::error::{Error, Result};

/// The bookkeeping of the inputs of one stream for a truncation (module documentation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamState {
    /// The number of inputs since the last reset.
    len: usize,
    /// The first position of the current context that holds an input: 0 for the first
    /// context, `K + 1` after a restart.
    first: usize,
    /// One more than the highest position that the current context wrote.
    written: usize,
    /// `first` of the context before the last restart, if there was a restart.
    prev_first: Option<usize>,
}

/// A planned truncation ([`StreamState::plan_truncate`]).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Truncation {
    /// The length after the truncation.
    pub len: usize,
    /// The position after the truncation.
    pub pos: usize,
    /// `true` if the stream goes back to the context before the last restart: the backend
    /// takes the kept history of that context.
    pub restore: bool,
    /// The positions that the backend runs again with the kept history (rows without
    /// weights), before the truncation is kept. Empty if `restore` is `false`.
    pub rebuild: Range<usize>,
}

impl StreamState {
    /// An empty stream.
    #[must_use]
    pub const fn new() -> Self {
        Self { len: 0, first: 0, written: 0, prev_first: None }
    }

    /// Empties the stream (a reset of the context).
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// The number of inputs since the last reset.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` if the stream has no input.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of inputs that the stream can take at the position `pos` (context of
    /// `n_ctx`, overlap `k`) before the next restart: a pass of at most this many rows of the
    /// stream never needs a rebuild at a truncation. At a full context (`pos >= n_ctx`), the
    /// next input restarts first, and the value is the room of the new context,
    /// `n_ctx - k - 1`.
    #[must_use]
    pub const fn room(pos: usize, n_ctx: usize, k: usize) -> usize {
        if pos >= n_ctx { n_ctx.saturating_sub(k + 1) } else { n_ctx - pos }
    }

    /// Records an input at the position `pos`.
    pub fn input(&mut self, pos: usize) {
        self.len += 1;
        self.written = self.written.max(pos + 1);
    }

    /// Records a context restart with the overlap `k`: the next input goes to `k + 1`. The
    /// backend keeps the history of the full context before it (for [`Truncation::restore`]).
    pub fn restart(&mut self, k: usize) {
        self.prev_first = Some(self.first);
        self.first = k + 1;
        self.written = k + 1;
    }

    /// Plans the truncation to `len` inputs at the position `pos` with a context of `n_ctx`
    /// positions. It changes nothing.
    ///
    /// # Errors
    /// [`Error::BadTruncate`] if `len` is more than the length, or before the restart before
    /// the last one.
    pub fn plan_truncate(&self, len: usize, pos: usize, n_ctx: usize) -> Result<Truncation> {
        let bad = || Error::BadTruncate { len, stream_len: self.len };
        let drop = self.len.checked_sub(len).ok_or_else(bad)?;
        let here = pos.saturating_sub(self.first);
        if drop <= here {
            return Ok(Truncation { len, pos: pos - drop, restore: false, rebuild: 0..0 });
        }
        let prev_first = self.prev_first.ok_or_else(bad)?;
        let before = drop - here;
        if before > n_ctx.saturating_sub(prev_first) {
            return Err(bad());
        }
        // The old context was full: its position was n_ctx.
        let target = n_ctx - before;
        Ok(Truncation { len, pos: target, restore: true, rebuild: 0..target.min(self.written) })
    }

    /// Keeps a planned truncation (after the backend ran its rebuild rows and took the kept
    /// history), for a context of `n_ctx` positions.
    pub fn commit_truncate(&mut self, t: &Truncation, n_ctx: usize) {
        self.len = t.len;
        if t.restore
            && let Some(first) = self.prev_first.take()
        {
            self.first = first;
            // INVARIANT: every position of the old context is written; only the positions
            // below `t.pos` have a meaning.
            self.written = n_ctx;
        }
    }
}

/// The inputs of one CPU stream: the history of the context, the bookkeeping, and the history
/// of the context before the last restart.
#[derive(Clone, Debug)]
pub(crate) struct Stream {
    hist: History,
    state: StreamState,
    n_ctx: usize,
    /// The tokens of the full context before the last restart (empty before the first).
    prev: Vec<usize>,
}

impl Stream {
    /// An empty stream for a context of `n_ctx` positions. Cost: `8 * n_ctx` bytes, and
    /// `8 * n_ctx` more at the first restart.
    pub(crate) fn new(n_ctx: usize) -> Self {
        Self { hist: History::new(n_ctx), state: StreamState::new(), n_ctx, prev: Vec::new() }
    }

    /// Empties the stream.
    pub(crate) fn clear(&mut self) {
        self.state.clear();
    }

    /// The number of inputs since the last reset.
    pub(crate) fn len(&self) -> usize {
        self.state.len()
    }

    /// [`StreamState::room`] at the position `pos` with the overlap `k`.
    pub(crate) fn room(&self, pos: usize, k: usize) -> usize {
        StreamState::room(pos, self.n_ctx, k)
    }

    /// Records the input `tok` at position `pos`.
    pub(crate) fn input(&mut self, pos: usize, tok: usize) {
        self.hist.set(pos, tok);
        self.state.input(pos);
    }

    /// The context rule of a full context (`History::restart`). It keeps the old context for a
    /// later truncation. Gives the position of the next input, `k + 1`.
    pub(crate) fn restart(
        &mut self,
        k: usize,
        eot: usize,
        lane: usize,
        rows: &mut Vec<Row>,
    ) -> usize {
        self.prev.clear();
        self.prev.extend_from_slice(self.hist.tokens());
        let next = self.hist.restart(k, eot, lane, rows);
        self.state.restart(next - 1);
        next
    }

    /// Plans the truncation to `len` inputs at the position `pos`, with no change: pushes the
    /// rebuild rows of `lane` (rule 2 of the module documentation) to `rows`.
    ///
    /// # Errors
    /// The errors of [`StreamState::plan_truncate`].
    pub(crate) fn plan_truncate(
        &self,
        len: usize,
        pos: usize,
        lane: usize,
        rows: &mut Vec<Row>,
    ) -> Result<Truncation> {
        let t = self.state.plan_truncate(len, pos, self.n_ctx)?;
        for p in t.rebuild.clone() {
            let tok = self.prev.get(p).copied().unwrap_or(0);
            rows.push(Row { lane, pos: p, tok, out: None });
        }
        Ok(t)
    }

    /// Keeps a planned truncation after its rebuild rows ran.
    pub(crate) fn commit_truncate(&mut self, t: &Truncation) {
        if t.restore {
            self.hist.set_tokens(&self.prev);
        }
        self.state.commit_truncate(t, self.n_ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stream of `n` inputs with the context rule (`n_ctx`, `k`), as a session runs them.
    fn run(n_ctx: usize, k: usize, n: usize) -> (Stream, usize) {
        let mut s = Stream::new(n_ctx);
        let mut pos = 0;
        let mut rows = Vec::new();
        for i in 0..n {
            if pos >= n_ctx {
                pos = s.restart(k, 7, 0, &mut rows);
            }
            s.input(pos, 100 + i);
            pos += 1;
        }
        (s, pos)
    }

    #[test]
    fn truncation_inside_the_context_moves_the_position_only() {
        let (s, pos) = run(10, 3, 8);
        let mut rows = Vec::new();
        let cut = s.plan_truncate(5, pos, 0, &mut rows).unwrap();
        assert_eq!((cut.pos, cut.restore, rows.len()), (5, false, 0));
    }

    #[test]
    fn truncation_before_a_restart_rebuilds_the_written_positions() {
        // n_ctx 10, K 3: inputs 0..10 fill the context; input 10 restarts (rows 0..=3) and
        // goes to position 4; inputs 11 and 12 go to 5 and 6.
        let (s, pos) = run(10, 3, 13);
        assert_eq!(pos, 7);
        let mut rows = Vec::new();
        // Keep 8 inputs: drop the 3 inputs of the new context and 2 of the old one.
        let cut = s.plan_truncate(8, pos, 0, &mut rows).unwrap();
        assert_eq!((cut.pos, cut.restore, cut.rebuild.clone()), (8, true, 0..7));
        let got: Vec<(usize, usize)> = rows.iter().map(|r| (r.pos, r.tok)).collect();
        let want: Vec<(usize, usize)> = (0..7).map(|p| (p, 100 + p)).collect();
        assert_eq!(got, want);
        let mut s = s;
        s.commit_truncate(&cut);
        assert_eq!(s.len(), 8);
        assert!(s.plan_truncate(0, cut.pos, 0, &mut Vec::new()).is_ok());
    }

    #[test]
    fn truncation_too_far_is_an_error() {
        let (s, pos) = run(10, 3, 30);
        assert!(s.plan_truncate(31, pos, 0, &mut Vec::new()).is_err());
        // The contexts hold the inputs 0..10, 10..16, 16..22, 22..28 and 28..30: the length 22
        // is the start of the context before the last restart.
        assert!(s.plan_truncate(21, pos, 0, &mut Vec::new()).is_err());
        assert!(s.plan_truncate(22, pos, 0, &mut Vec::new()).is_ok());
    }

    #[test]
    fn room_is_the_rest_of_the_context() {
        assert_eq!(StreamState::room(3, 10, 4), 7);
        assert_eq!(StreamState::room(10, 10, 4), 5);
        assert_eq!(StreamState::room(10, 10, 0), 9);
    }
}
