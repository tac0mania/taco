//! A GPT-2 session: the context (KV cache) of one block. It runs one token at a time
//! ([`Session::step`]) or many known tokens at a time ([`Session::step_many`], the prefill
//! path of the encoder).
//!
//! Both paths use the op kernels of `ops` through `engine::forward`, and both give the same
//! weights as [`super::ReferenceSession`], bit for bit (`tests/fast_paths.rs`).

use super::Model;
use super::attention::KvCache;
use super::context::{self, Overlap, RESTART_ROWS};
use super::engine::{self, Row, Work};
use super::stream::Stream;
use crate::error::{Error, Result};
use crate::kernels::Kernels;
use crate::types::TokenId;

/// The options of a [`Session`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SessionConfig {
    /// The largest number of tokens of one [`Session::step_many`] call: the number of rows
    /// that share one read of the weights. At least 1.
    pub chunk: usize,
    /// The kernels. Every selection gives the same weights; only the speed differs.
    pub kernels: Kernels,
    /// The overlap of the context rule (ADR 0015). It changes the weights after a context
    /// restart, so the encoder and the decoder must use the same value. Default:
    /// [`Overlap::NONE`].
    pub overlap: Overlap,
}

impl SessionConfig {
    /// The default options: a chunk of 1 token (the step path) and the fastest kernels of
    /// this CPU.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The same options with another chunk size (at least 1).
    #[must_use]
    pub const fn with_chunk(mut self, chunk: usize) -> Self {
        self.chunk = if chunk == 0 { 1 } else { chunk };
        self
    }

    /// The same options with other kernels.
    #[must_use]
    pub const fn with_kernels(mut self, kernels: Kernels) -> Self {
        self.kernels = kernels;
        self
    }

    /// The same options with the context overlap `overlap` (ADR 0015).
    #[must_use]
    pub const fn with_overlap(mut self, overlap: Overlap) -> Self {
        self.overlap = overlap;
        self
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self { chunk: 1, kernels: Kernels::detect(), overlap: Overlap::NONE }
    }
}

/// The state of one block of input: the KV cache, the position and the work buffers.
///
/// A session borrows its model. Two sessions of one model give the same weights for the same
/// tokens, for every chunk size and every kernel selection (DETERMINISM: every step uses only
/// the integer rules of `int`, and every kernel is exact).
#[derive(Debug)]
pub struct Session<'m> {
    model: &'m Model,
    kernels: Kernels,
    overlap: Overlap,
    /// The largest number of tokens of one call.
    chunk: usize,
    /// The number of tokens in the context.
    pos: usize,
    /// The inputs: the tokens of the context (for the overlap of a restart), the length, and the
    /// context before the last restart (for a truncation).
    stream: Stream,
    /// One KV cache: the engine indexes caches by lane, and a session has lane 0 only.
    cache: [KvCache; 1],
    work: Work,
    rows: Vec<Row>,
}

impl<'m> Session<'m> {
    /// Makes a session with an empty context and the default options (the step path). Cost:
    /// allocates the KV cache, `2 * n_layer * n_ctx * d_model` bytes.
    ///
    /// PERF: GPT-2 small needs 18.9 MB of cache (est.: 2 * 12 * 1,024 * 768 bytes). A model
    /// file with many layers and a long context can ask for much more memory than its own size;
    /// the decoder loads only model files with a known hash.
    #[must_use]
    pub fn new(model: &'m Model) -> Self {
        Self::with_config(model, SessionConfig::new())
    }

    /// Makes a session with the options `config`. Cost: the KV cache, and work buffers of
    /// about `12 * chunk * vocab` bytes (for GPT-2 small and a chunk of 32: 19 MB, est.).
    ///
    /// The overlap of `config` is checked at each step: a session with an overlap that is not
    /// valid for `model` refuses every step ([`Model::check_overlap`]). With an overlap, a
    /// forward pass has room for at least 32 rows, so the `K + 1` restart rows run in few
    /// passes also for a small chunk (about `4 * 32 * vocab` more bytes for a chunk below 32,
    /// est.).
    #[must_use]
    pub fn with_config(model: &'m Model, config: SessionConfig) -> Self {
        let chunk = config.chunk.max(1);
        let rows = if config.overlap.is_none() { chunk } else { chunk.max(RESTART_ROWS) };
        Self {
            model,
            kernels: config.kernels,
            overlap: config.overlap,
            chunk,
            pos: 0,
            stream: Stream::new(model.n_ctx()),
            cache: [KvCache::new(model)],
            work: Work::new(model, rows, chunk),
            rows: Vec::with_capacity(2 * chunk),
        }
    }

    /// Empties the context. Cost: no memory is written; old cache entries are not read again.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.stream.clear();
    }

    /// The length of the stream: the number of inputs since the last reset. The
    /// `<|endoftext|>` row of [`Session::start_block`] is one input, and each token is one
    /// input. The rows of a context restart are not inputs.
    #[must_use]
    pub fn stream_len(&self) -> usize {
        self.stream.len()
    }

    /// The number of tokens that the session can take before its next context restart. A
    /// draft of at most this many tokens never needs a rebuild at a [`Session::truncate`].
    #[must_use]
    pub fn room(&self) -> usize {
        self.stream.room(self.pos, self.overlap.tokens())
    }

    /// Truncates the stream to its first `len` inputs (speculative decoding, ADR 0017): after
    /// the call, the session gives the weights of a session that ran only these inputs, bit for
    /// bit. The decoder runs a draft with [`Session::step_many`], then truncates the inputs
    /// after the first token that differs from the draft.
    ///
    /// Cost: none inside the current context. A length before the last context restart runs
    /// the rows of the old context that the new context wrote again, without the logits (module
    /// `stream`; at most `K + 1 + d` rows for a draft of `d` tokens).
    ///
    /// # Errors
    /// [`Error::BadTruncate`] if `len` is more than [`Session::stream_len`] or before the
    /// context restart before the last one. The session does not change.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        self.rows.clear();
        let cut = self.stream.plan_truncate(len, self.pos, 0, &mut self.rows)?;
        for group in self.rows.chunks(self.work.cap()) {
            engine::forward(self.kernels, self.model, &mut self.cache, group, &mut self.work)?;
        }
        self.stream.commit_truncate(&cut);
        self.pos = cut.pos;
        Ok(())
    }

    /// The number of tokens in the context.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// The largest number of tokens of one [`Session::step_many`] call.
    #[must_use]
    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// The kernels of this session.
    #[must_use]
    pub fn kernels(&self) -> Kernels {
        self.kernels
    }

    /// The context overlap of this session (ADR 0015).
    #[must_use]
    pub fn overlap(&self) -> Overlap {
        self.overlap
    }

    /// Starts a block (plan §7.7): empties the context and runs `<|endoftext|>`. The result is
    /// the weights for the first token of the block.
    ///
    /// # Errors
    /// The errors of [`Session::step`].
    pub fn start_block(&mut self) -> Result<&[u64]> {
        self.reset();
        context::checked(self.model, self.overlap)?;
        let eot = self.model.check_token(self.model.eot_token())?;
        self.stream.input(0, eot);
        self.rows.clear();
        self.rows.push(Row { lane: 0, pos: 0, tok: eot, out: Some(0) });
        engine::forward(self.kernels, self.model, &mut self.cache, &self.rows, &mut self.work)?;
        self.pos = 1;
        self.weights(0)
    }

    /// Adds `token` to the context and runs the model. The result has one weight for each
    /// token of the vocabulary: the prediction for the next token (plan §7.6). The weight of
    /// the most likely token is at least 1, and the sum is at most `2^60`.
    ///
    /// If the context is full, the session starts again first, with the overlap of its
    /// options (module `context`, ADR 0015). Cost: one forward pass, about
    /// `n_layer * 12 * d_model^2 + vocab * d_model` multiply-adds for GPT-2 shapes, plus the
    /// attention over the context; at a restart, also `K + 1` rows without the logits.
    ///
    /// # Errors
    /// [`Error::TokenOutOfRange`] if `token` is not in the vocabulary;
    /// [`Error::InvalidOverlap`] if the overlap is not valid for the model.
    pub fn step(&mut self, token: TokenId) -> Result<&[u64]> {
        self.run(core::slice::from_ref(&token))?;
        self.weights(0)
    }

    /// Adds all `tokens` to the context, in order, and runs the model for all of them at once
    /// (the prefill path). The result is the same as one [`Session::step`] for each token:
    /// item `i` of the result is the weights after `tokens[i]`, bit for bit. Context restarts
    /// (plan §7.7) happen at the same tokens as with `step`.
    ///
    /// Cost: the multiply-adds of one `step` for each token, but each weight of the model is
    /// read from memory once for all the tokens of the call.
    ///
    /// This is also the draft pass of speculative decoding (ADR 0017): the known token and the
    /// draft tokens in one call, then [`Session::truncate`] after the first wrong token.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `tokens` has more than [`Session::chunk`] tokens;
    /// [`Error::TokenOutOfRange`] if a token is not in the vocabulary;
    /// [`Error::InvalidOverlap`] if the overlap is not valid for the model. On an error the
    /// context does not change.
    pub fn step_many(&mut self, tokens: &[TokenId]) -> Result<StepWeights<'_>> {
        self.run(tokens)?;
        let vocab = self.model.vocab_size();
        let data = self.work.weights_prefix(vocab, tokens.len()).unwrap_or(&[]);
        Ok(StepWeights { data, vocab })
    }

    /// The rows for `tokens`, then the forward pass in groups of at most `work.cap()` rows.
    fn run(&mut self, tokens: &[TokenId]) -> Result<()> {
        if tokens.len() > self.chunk {
            return Err(Error::LengthMismatch { expected: self.chunk, found: tokens.len() });
        }
        let cap = self.work.cap();
        let model = self.model;
        let n_ctx = model.n_ctx();
        let k = context::checked(model, self.overlap)?;
        let eot = model.check_token(model.eot_token())?;
        // Check every token first, so that an error does not change the history.
        for &t in tokens {
            model.check_token(t)?;
        }
        self.rows.clear();
        let mut pos = self.pos;
        for (i, &t) in tokens.iter().enumerate() {
            let tok = model.check_token(t)?;
            if pos >= n_ctx {
                // PLAN(§7.7): the context starts again with <|endoftext|> and the last `k`
                // tokens (ADR 0015). Their weights are not used, so the rows have no output
                // slot.
                pos = self.stream.restart(k, eot, 0, &mut self.rows);
            }
            self.stream.input(pos, tok);
            self.rows.push(Row { lane: 0, pos, tok, out: Some(i) });
            pos += 1;
        }
        for group in self.rows.chunks(cap) {
            engine::forward(self.kernels, model, &mut self.cache, group, &mut self.work)?;
        }
        self.pos = pos;
        Ok(())
    }

    fn weights(&self, slot: usize) -> Result<&[u64]> {
        let vocab = self.model.vocab_size();
        self.work.weights(vocab, slot).ok_or(Error::LengthMismatch { expected: vocab, found: 0 })
    }
}

/// The weights of a [`Session::step_many`] call: one row of `vocab_size` weights for each
/// token.
#[derive(Clone, Copy, Debug)]
pub struct StepWeights<'a> {
    data: &'a [u64],
    vocab: usize,
}

impl<'a> StepWeights<'a> {
    /// The rows of `data`, `vocab` weights each.
    pub(crate) fn new(data: &'a [u64], vocab: usize) -> Self {
        Self { data, vocab }
    }

    /// The number of rows: the number of tokens of the call.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len() / self.vocab.max(1)
    }

    /// `true` if the call had no tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The weights after token `i` of the call, or `None` if `i >= len()`.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&'a [u64]> {
        self.data.get(i * self.vocab..(i + 1) * self.vocab)
    }

    /// The rows in token order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'a [u64]> + use<'a> {
        self.data.chunks_exact(self.vocab.max(1))
    }
}
