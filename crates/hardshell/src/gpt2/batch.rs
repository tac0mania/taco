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
    caches: Vec<KvCache>,
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
            caches: (0..lanes).map(|_| KvCache::new(model)).collect(),
            pos: vec![0; lanes],
            ready: vec![false; lanes],
            work: Work::new(model, lanes, lanes),
            rows: Vec::with_capacity(2 * lanes),
        }
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
    /// [`Error::TokenOutOfRange`] if a token is not in the vocabulary. On an error no lane
    /// changes.
    pub fn run(&mut self, inputs: &[Option<BatchInput>]) -> Result<()> {
        let lanes = self.lanes();
        if inputs.len() != lanes {
            return Err(Error::LengthMismatch { expected: lanes, found: inputs.len() });
        }
        let model = self.model;
        let n_ctx = model.n_ctx();
        let eot = model.check_token(model.eot_token())?;
        self.rows.clear();
        for (lane, input) in inputs.iter().enumerate() {
            let Some(input) = input else { continue };
            let pos = self.pos.get(lane).copied().unwrap_or(0);
            match *input {
                BatchInput::Start => {
                    self.rows.push(Row { lane, pos: 0, tok: eot, out: Some(lane) });
                }
                BatchInput::Token(t) => {
                    let tok = model.check_token(t)?;
                    if pos >= n_ctx {
                        // PLAN(§7.7): the context of this lane starts again.
                        self.rows.push(Row { lane, pos: 0, tok: eot, out: None });
                        self.rows.push(Row { lane, pos: 1, tok, out: Some(lane) });
                    } else {
                        self.rows.push(Row { lane, pos, tok, out: Some(lane) });
                    }
                }
            }
        }
        // NOTE: a group boundary can fall between the two rows of a restart; the first group
        // writes the cache at position 0 before the second group reads it.
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
