//! The context rule of a slot (plan §7.7, ADR 0015): the tokens of its context, and the rows of
//! a restart.
//!
//! A block starts with `<|endoftext|>` at position 0. When the context of a slot holds `n_ctx`
//! tokens and the caller gives one more token `t`, the slot restarts its context with the
//! overlap `K`: `<|endoftext|>` at position 0, the last `K` tokens of the block at the
//! positions `1..=K` (rows whose weights are not used), then `t` at position `K + 1`. This is
//! the rule of `hardshell::gpt2::context`, and `K = 0` is the rule of ADR 0004 section 9.
//!
//! DETERMINISM: the kept tokens are the last `K` tokens that the slot ran, so the rule depends
//! only on the tokens of the block and on `K`. The restart rows are normal forward rows, so the
//! GPU computes them with the integer rules of every other row.

use hardshell::TokenId;

/// The fewest rows of one forward pass of a batch with a context overlap, so the `K + 1`
/// restart rows of a slot run in few passes.
///
/// NOTE: the value of `RESTART_ROWS` in `hardshell::gpt2::context` (the prefill chunk of the
/// encoder). The size changes the speed and the GPU memory, not the weights.
pub(super) const RESTART_ROWS: usize = 32;

/// One row of a forward pass: slot, token, position.
pub(super) type Row = (usize, TokenId, usize);

/// The tokens of the context of one slot, by position: `toks[p]` is the token at position `p`.
/// Only the positions below the position of the slot have a meaning.
pub(super) struct History {
    toks: Vec<TokenId>,
}

impl History {
    /// An empty history for a context of `n_ctx` positions. Cost: `4 * n_ctx` bytes.
    pub(super) fn new(n_ctx: usize) -> Self {
        Self { toks: vec![TokenId::new(0); n_ctx] }
    }

    /// Records `tok` at position `pos`. A position outside the context is ignored; the slot
    /// never runs such a row.
    pub(super) fn set(&mut self, pos: usize, tok: TokenId) {
        if let Some(t) = self.toks.get_mut(pos) {
            *t = tok;
        }
    }

    /// Pushes the `k + 1` rows of a restart of the full context of `slot` with the overlap
    /// `k`: `eot` at position 0, then the tokens at the positions `n_ctx - k .. n_ctx` at the
    /// positions `1..=k`. The history does not change; [`History::restart`] changes it after
    /// the passes.
    ///
    /// The caller checked `k <= n_ctx - 2`. Cost: `k + 1` rows.
    pub(super) fn restart_rows(&self, slot: usize, k: usize, eot: TokenId, rows: &mut Vec<Row>) {
        let k = self.kept(k);
        rows.push((slot, eot, 0));
        let kept = self.toks.get(self.toks.len() - k..).unwrap_or(&[]);
        rows.extend(kept.iter().enumerate().map(|(i, &t)| (slot, t, i + 1)));
    }

    /// Restarts the full context with the overlap `k`: moves the tokens at the positions
    /// `n_ctx - k .. n_ctx` to the positions `1..=k`, and puts `eot` at position 0. These are
    /// the tokens of [`History::restart_rows`]. Cost: one copy of `k` tokens.
    pub(super) fn restart(&mut self, k: usize, eot: TokenId) {
        let k = self.kept(k);
        let n_ctx = self.toks.len();
        // NOTE: the ranges overlap when 2k >= n_ctx; `copy_within` is a memmove.
        if k > 0 {
            self.toks.copy_within(n_ctx - k..n_ctx, 1);
        }
        self.set(0, eot);
    }

    /// `k`, at most `n_ctx - 2` (the batch checked it; the `min` only keeps the ranges valid).
    fn kept(&self, k: usize) -> usize {
        k.min(self.toks.len().saturating_sub(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(n_ctx: usize) -> History {
        let mut h = History::new(n_ctx);
        for p in 0..n_ctx {
            h.set(p, TokenId::new(100 + u32::try_from(p).unwrap()));
        }
        h
    }

    fn ids(rows: &[Row]) -> Vec<(usize, u32, usize)> {
        rows.iter().map(|&(s, t, p)| (s, t.get(), p)).collect()
    }

    #[test]
    fn restart_rows_keep_the_last_k_tokens_in_order() {
        // n_ctx 10, K 3: the kept tokens are the positions 7, 8 and 9.
        let mut h = full(10);
        let mut rows = Vec::new();
        h.restart_rows(2, 3, TokenId::new(7), &mut rows);
        assert_eq!(ids(&rows), [(2, 7, 0), (2, 107, 1), (2, 108, 2), (2, 109, 3)]);
        h.restart(3, TokenId::new(7));
        assert_eq!(h.toks[..4].iter().map(|t| t.get()).collect::<Vec<_>>(), [7, 107, 108, 109]);
    }

    #[test]
    fn restart_with_more_than_half_the_context_is_a_memmove() {
        // n_ctx 10, K 8 (the largest): the positions 2..10 move to 1..9.
        let mut h = full(10);
        let mut rows = Vec::new();
        h.restart_rows(0, 8, TokenId::new(7), &mut rows);
        let want: Vec<u32> = std::iter::once(7).chain(102..110).collect();
        assert_eq!(rows.iter().map(|r| r.1.get()).collect::<Vec<_>>(), want);
        h.restart(8, TokenId::new(7));
        assert_eq!(h.toks[..9].iter().map(|t| t.get()).collect::<Vec<_>>(), want);
    }

    #[test]
    fn restart_without_overlap_is_the_old_rule() {
        let mut h = full(10);
        let mut rows = Vec::new();
        h.restart_rows(1, 0, TokenId::new(7), &mut rows);
        assert_eq!(ids(&rows), [(1, 7, 0)]);
        h.restart(0, TokenId::new(7));
        assert_eq!(h.toks[0].get(), 7);
    }
}
