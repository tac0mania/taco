//! The context rule (plan §7.7, ADR 0015): what a session does when its context is full.
//!
//! A block starts with `<|endoftext|>` at position 0. When the context holds `n_ctx` tokens and
//! the caller gives one more token `t`, the session restarts the context with an overlap of
//! `K` tokens:
//!
//! 1. It takes the last `K` tokens of the block before `t`. These are the tokens at the
//!    positions `n_ctx - K .. n_ctx` of the full context.
//! 2. It empties the context and runs `<|endoftext|>` at position 0, then the `K` tokens at the
//!    positions `1..=K`. These rows give no weights.
//! 3. It runs `t` at position `K + 1`. Its weights are the result of the step.
//!
//! `K = 0` is the rule of ADR 0004 section 9 (and of every container before ADR 0015):
//! `<|endoftext|>` at position 0, then `t` at position 1. `K` is at most `n_ctx - 2`, so `t`
//! has a position in the context.
//!
//! DETERMINISM: the kept tokens are the last `K` tokens of the input, so the rule depends only
//! on the tokens of the block and on `K`. The restart rows are normal forward rows, so every
//! path (reference, step, prefill, batch, GPU) computes them with the same integer rules.

use super::Model;
use super::engine::Row;
use crate::error::{Error, Result};

/// The smallest number of rows of one forward pass of a session or a batch with a context
/// overlap: the `K + 1` restart rows then run in groups of this size, not of `chunk` or
/// `lanes` rows.
///
/// NOTE: the value is `FOLD_CHUNK` of `crates/taco` (the prefill chunk of the encoder,
/// measured by the simd agent: chunk 32 gave 4.8 times the steps per second of the step path
/// on one M2 thread). The size changes the speed, not the weights.
pub(crate) const RESTART_ROWS: usize = 32;

/// The overlap `K` of the context rule: the number of tokens that a context restart keeps
/// (module documentation). `Overlap::NONE` (`K = 0`) is the default and the rule of
/// containers before ADR 0015.
///
/// A value is valid for a model with a context of `n_ctx` tokens if `K <= n_ctx - 2`
/// ([`Model::check_overlap`]). A session with an overlap that is not valid for its model
/// refuses every step with [`Error::InvalidOverlap`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Overlap(u16);

impl Overlap {
    /// No overlap: a restart keeps no token (ADR 0004 section 9).
    pub const NONE: Self = Self(0);

    /// The overlap of `k` tokens.
    #[must_use]
    pub const fn new(k: u16) -> Self {
        Self(k)
    }

    /// The number of tokens that a restart keeps.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }

    /// The number of tokens that a restart keeps, as a `usize`.
    #[must_use]
    pub const fn tokens(self) -> usize {
        self.0 as usize
    }

    /// `true` for [`Overlap::NONE`].
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

impl Model {
    /// Checks that `overlap` is valid for this model: `K <= n_ctx - 2`. The decoder calls this
    /// before it decodes a block, so a damaged or wrong overlap is an error at the start.
    ///
    /// # Errors
    /// [`Error::InvalidOverlap`] if `K > n_ctx - 2`.
    pub fn check_overlap(&self, overlap: Overlap) -> Result<()> {
        let n_ctx = self.n_ctx();
        if overlap.tokens().saturating_add(2) > n_ctx {
            return Err(Error::InvalidOverlap { overlap: overlap.get(), n_ctx });
        }
        Ok(())
    }
}

/// The tokens of the context of one block, by position: `toks[p]` is the token at position
/// `p`. Only positions below the position of the session have a meaning.
#[derive(Clone, Debug)]
pub(crate) struct History {
    toks: Vec<usize>,
}

impl History {
    /// An empty history for a context of `n_ctx` positions. Cost: `8 * n_ctx` bytes.
    pub(crate) fn new(n_ctx: usize) -> Self {
        Self { toks: vec![0; n_ctx] }
    }

    /// Records `tok` at position `pos`. A position outside the context is ignored; the engine
    /// refuses such a row first.
    pub(crate) fn set(&mut self, pos: usize, tok: usize) {
        if let Some(t) = self.toks.get_mut(pos) {
            *t = tok;
        }
    }

    /// The token at position `pos`, or 0 outside the context.
    pub(crate) fn get(&self, pos: usize) -> usize {
        self.toks.get(pos).copied().unwrap_or(0)
    }

    /// Restarts a full context with an overlap of `k` tokens: moves the tokens at the
    /// positions `n_ctx - k .. n_ctx` to the positions `1..=k`, puts `eot` at position 0, and
    /// pushes one row without weights for each of the positions `0..=k` of `lane`. Gives the
    /// position of the next token, `k + 1`.
    ///
    /// The caller checked `k <= n_ctx - 2` (`Model::check_overlap`). Cost: one copy of `k`
    /// tokens and `k + 1` rows.
    pub(crate) fn restart(
        &mut self,
        k: usize,
        eot: usize,
        lane: usize,
        rows: &mut Vec<Row>,
    ) -> usize {
        let n_ctx = self.toks.len();
        let k = k.min(n_ctx.saturating_sub(2));
        // NOTE: the ranges overlap when 2k >= n_ctx; `copy_within` is a memmove.
        if k > 0 {
            self.toks.copy_within(n_ctx - k..n_ctx, 1);
        }
        self.set(0, eot);
        for pos in 0..=k {
            rows.push(Row { lane, pos, tok: self.get(pos), out: None });
        }
        k + 1
    }
}

/// The overlap of a session, as a `usize`, if it is valid for `model`.
///
/// # Errors
/// [`Error::InvalidOverlap`] if it is not valid.
pub(crate) fn checked(model: &Model, overlap: Overlap) -> Result<usize> {
    model.check_overlap(overlap)?;
    Ok(overlap.tokens())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn restart(n_ctx: usize, k: usize) -> (Vec<usize>, Vec<Row>, usize) {
        let mut h = History::new(n_ctx);
        for p in 0..n_ctx {
            h.set(p, 100 + p);
        }
        let mut rows = Vec::new();
        let next = h.restart(k, 7, 3, &mut rows);
        ((0..=k).map(|p| h.get(p)).collect(), rows, next)
    }

    #[test]
    fn restart_keeps_the_last_k_tokens_in_order() {
        // n_ctx 10, K 3: the kept tokens are the positions 7, 8 and 9.
        let (toks, rows, next) = restart(10, 3);
        assert_eq!(toks, [7, 107, 108, 109]);
        assert_eq!(next, 4);
        assert_eq!(rows.len(), 4);
        for (p, r) in rows.iter().enumerate() {
            assert_eq!((r.lane, r.pos, r.tok, r.out), (3, p, toks[p], None));
        }
    }

    #[test]
    fn restart_with_more_than_half_the_context_is_a_memmove() {
        // n_ctx 10, K 8 (the largest): positions 2..10 move to 1..9.
        let (toks, _, next) = restart(10, 8);
        assert_eq!(toks, [7, 102, 103, 104, 105, 106, 107, 108, 109]);
        assert_eq!(next, 9);
    }

    #[test]
    fn restart_without_overlap_is_the_old_rule() {
        let (toks, rows, next) = restart(10, 0);
        assert_eq!(toks, [7]);
        assert_eq!(rows.len(), 1);
        assert_eq!(next, 1);
    }
}
