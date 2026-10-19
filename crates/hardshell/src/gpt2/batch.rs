//! A batch of independent GPT-2 sessions in lockstep: one call runs one step of each lane
//! (block), so each weight of the model is read once for all the lanes. This is the batch path
//! of the decoder, where the next token of a block is known only after its weights.
//!
//! DETERMINISM: each lane has its own KV cache and position, and the ops never mix the values
//! of two rows. Thus each lane gives the same weights as a [`super::Session`] (and as
//! [`super::ReferenceSession`]) with the same inputs, bit for bit, for every mix of lanes in
//! each call (`tests/fast_paths.rs`).

use super::Model;
use super::attention::KvCache;
use super::context::{self, History, Overlap, RESTART_ROWS};
use super::engine::{self, Row, Work};
use crate::error::{Error, Result};
use crate::kernels::Kernels;
use crate::types::TokenId;

/// The input of one lane for one [`Batch::run`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchInput {
    /// The same as [`super::Session::start_block`]: empty the context, run `<|endoftext|>`.
    Start,
    /// The same as [`super::Session::step`] with this token.
    Token(TokenId),
}

/// `lanes` independent sessions of one model that run in lockstep.
#[derive(Debug)]
pub struct Batch<'m> {
    model: &'m Model,
    kernels: Kernels,
    overlap: Overlap,
    caches: Vec<KvCache>,
    /// The tokens of the context of each lane, for the overlap of a restart.
    hist: Vec<History>,
    pos: Vec<usize>,
    /// `true` for a lane whose output slot holds the weights of its last input.
    ready: Vec<bool>,
    work: Work,
    rows: Vec<Row>,
}

impl<'m> Batch<'m> {
    /// Makes `lanes` sessions (at least 1) with empty contexts and the fastest kernels of this
    /// CPU. Cost: one KV cache for each lane (for GPT-2 small, 18.9 MB each, est.) and about
    /// `20 * lanes * vocab` bytes of work buffers.
    #[must_use]
    pub fn new(model: &'m Model, lanes: usize) -> Self {
        Self::with_kernels(model, lanes, Kernels::detect())
    }

    /// The same as [`Batch::new`], with the kernels `kernels`.
    #[must_use]
    pub fn with_kernels(model: &'m Model, lanes: usize, kernels: Kernels) -> Self {
        let lanes = lanes.max(1);
        Self {
            model,
            kernels,
            overlap: Overlap::NONE,
            caches: (0..lanes).map(|_| KvCache::new(model)).collect(),
            hist: (0..lanes).map(|_| History::new(model.n_ctx())).collect(),
            pos: vec![0; lanes],
            ready: vec![false; lanes],
            work: Work::new(model, lanes, lanes),
            rows: Vec::with_capacity(2 * lanes),
        }
    }

    /// The same batch with the context overlap `overlap` (ADR 0015) for every lane. Call it
    /// before the first [`Batch::run`]. With an overlap, a forward pass has room for at least
    /// 32 rows, so the `K + 1` restart rows of a lane run in few passes (about
    /// `640 * vocab` more bytes of work buffers for a batch of fewer than 32 lanes, est.).
    ///
    /// The overlap is checked at each call of [`Batch::run`] ([`Model::check_overlap`]).
    #[must_use]
    pub fn with_overlap(mut self, overlap: Overlap) -> Self {
        self.overlap = overlap;
        let lanes = self.lanes();
        if !overlap.is_none() && self.work.cap() < RESTART_ROWS {
            self.work = Work::new(self.model, RESTART_ROWS, lanes);
        }
        self
    }

    /// The context overlap of the lanes (ADR 0015).
    #[must_use]
    pub fn overlap(&self) -> Overlap {
        self.overlap
    }

    /// The number of lanes.
    #[must_use]
    pub fn lanes(&self) -> usize {
        self.pos.len()
    }

    /// The number of tokens in the context of `lane`, or `None` if there is no such lane.
    #[must_use]
    pub fn position(&self, lane: usize) -> Option<usize> {
        self.pos.get(lane).copied()
    }

    /// Runs one input for each lane with `Some` input, all at once. `inputs` has one entry for
    /// each lane; a lane with `None` does not change. After the call, [`Batch::weights`] gives
    /// the weights of each lane that had an input.
    ///
    /// Cost: one forward pass for all the lanes with an input; each weight of the model is
    /// read once for all of them.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `inputs` does not have one entry for each lane;
    /// [`Error::TokenOutOfRange`] if a token is not in the vocabulary;
    /// [`Error::InvalidOverlap`] if the overlap is not valid for the model. On an error no
    /// lane changes.
    pub fn run(&mut self, inputs: &[Option<BatchInput>]) -> Result<()> {
        let lanes = self.lanes();
        if inputs.len() != lanes {
            return Err(Error::LengthMismatch { expected: lanes, found: inputs.len() });
        }
        let model = self.model;
        let n_ctx = model.n_ctx();
        let k = context::checked(model, self.overlap)?;
        let eot = model.check_token(model.eot_token())?;
        // Check every token first, so that an error does not change a history.
        for input in inputs.iter().flatten() {
            if let BatchInput::Token(t) = *input {
                model.check_token(t)?;
            }
        }
        self.rows.clear();
        for (lane, input) in inputs.iter().enumerate() {
            let Some(input) = input else { continue };
            let (Some(&pos), Some(hist)) = (self.pos.get(lane), self.hist.get_mut(lane)) else {
                continue;
            };
            match *input {
                BatchInput::Start => {
                    hist.set(0, eot);
                    self.rows.push(Row { lane, pos: 0, tok: eot, out: Some(lane) });
                }
                BatchInput::Token(t) => {
                    let tok = model.check_token(t)?;
                    let mut pos = pos;
                    if pos >= n_ctx {
                        // PLAN(§7.7): the context of this lane starts again, with the last `k`
                        // tokens (ADR 0015).
                        pos = hist.restart(k, eot, lane, &mut self.rows);
                    }
                    hist.set(pos, tok);
                    self.rows.push(Row { lane, pos, tok, out: Some(lane) });
                }
            }
        }
        // NOTE: a group boundary can fall between the rows of a restart; the earlier group
        // writes the cache at the lower positions before the later group reads them.
        for group in self.rows.chunks(self.work.cap()) {
            engine::forward(self.kernels, model, &mut self.caches, group, &mut self.work)?;
        }
        for row in &self.rows {
            if let (Some(p), Some(r)) = (self.pos.get_mut(row.lane), self.ready.get_mut(row.lane)) {
                *p = row.pos + 1;
                *r = true;
            }
        }
        Ok(())
    }

    /// The weights of the last input of `lane`, or `None` if the lane has had no input or
    /// does not exist.
    #[must_use]
    pub fn weights(&self, lane: usize) -> Option<&[u64]> {
        if !self.ready.get(lane).copied().unwrap_or(false) {
            return None;
        }
        self.work.weights(self.model.vocab_size(), lane)
    }
}
