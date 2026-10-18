//! A GPT-2 session: the context (KV cache) of one block. It runs one token at a time
//! ([`Session::step`]) or many known tokens at a time ([`Session::step_many`], the prefill
//! path of the encoder).
//!
//! Both paths use the op kernels of `ops` through `engine::forward`, and both give the same
//! weights as [`super::ReferenceSession`], bit for bit (`tests/fast_paths.rs`).

use super::Model;
use super::attention::KvCache;
use super::engine::{self, Row, Work};
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
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self { chunk: 1, kernels: Kernels::detect() }
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
    /// The number of tokens in the context.
    pos: usize,
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
    #[must_use]
    pub fn with_config(model: &'m Model, config: SessionConfig) -> Self {
        let chunk = config.chunk.max(1);
        Self {
            model,
            kernels: config.kernels,
            pos: 0,
            cache: [KvCache::new(model)],
            work: Work::new(model, chunk, chunk),
            rows: Vec::with_capacity(2 * chunk),
        }
    }

    /// Empties the context. Cost: no memory is written; old cache entries are not read again.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// The number of tokens in the context.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// The largest number of tokens of one [`Session::step_many`] call.
    #[must_use]
    pub fn chunk(&self) -> usize {
        self.work.cap()
    }

    /// The kernels of this session.
    #[must_use]
    pub fn kernels(&self) -> Kernels {
        self.kernels
    }

    /// Starts a block (plan §7.7): empties the context and runs `<|endoftext|>`. The result is
    /// the weights for the first token of the block.
    ///
    /// # Errors
    /// The errors of [`Session::step`].
    pub fn start_block(&mut self) -> Result<&[u64]> {
        self.reset();
        let eot = self.model.check_token(self.model.eot_token())?;
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
    /// If the context is full, the session starts again first (module documentation of
    /// `gpt2`). Cost: one forward pass, about `n_layer * 12 * d_model^2 + vocab * d_model`
    /// multiply-adds for GPT-2 shapes, plus the attention over the context.
    ///
    /// # Errors
    /// [`Error::TokenOutOfRange`] if `token` is not in the vocabulary.
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
    /// # Errors
    /// [`Error::LengthMismatch`] if `tokens` has more than [`Session::chunk`] tokens;
    /// [`Error::TokenOutOfRange`] if a token is not in the vocabulary. On an error the context
    /// does not change.
    pub fn step_many(&mut self, tokens: &[TokenId]) -> Result<StepWeights<'_>> {
        self.run(tokens)?;
        let vocab = self.model.vocab_size();
        let data = self.work.weights_prefix(vocab, tokens.len()).unwrap_or(&[]);
        Ok(StepWeights { data, vocab })
    }

    /// The rows for `tokens`, then the forward pass in groups of at most `chunk` rows.
    fn run(&mut self, tokens: &[TokenId]) -> Result<()> {
        let cap = self.work.cap();
        if tokens.len() > cap {
            return Err(Error::LengthMismatch { expected: cap, found: tokens.len() });
        }
        let model = self.model;
        let n_ctx = model.n_ctx();
        let eot = model.check_token(model.eot_token())?;
        self.rows.clear();
        let mut pos = self.pos;
        for (i, &t) in tokens.iter().enumerate() {
            let tok = model.check_token(t)?;
            if pos >= n_ctx {
                // PLAN(§7.7): the context starts again with <|endoftext|>. Its weights are not
                // used, so the row has no output slot.
                self.rows.push(Row { lane: 0, pos: 0, tok: eot, out: None });
                pos = 1;
            }
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
