//! One GPT-2 session on a CUDA device, with the API shape of `hardshell::gpt2::Session`.

use hardshell::TokenId;

use crate::batch::CudaBatch;
use crate::error::Result;
use crate::model::CudaModel;

/// One session: a batch with one slot. It gives the same weights as
/// `hardshell::gpt2::Session` for the same tokens, bit for bit.
#[derive(Debug)]
pub struct CudaSession<'m> {
    batch: CudaBatch<'m>,
    eot: TokenId,
}

impl<'m> CudaSession<'m> {
    /// Makes a session with an empty context.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::new`].
    pub fn new(model: &'m CudaModel<'m>) -> Result<Self> {
        Ok(Self { batch: CudaBatch::new(model, 1, 2)?, eot: model.eot_token() })
    }

    /// Empties the context.
    pub fn reset(&mut self) {
        self.batch.reset(0);
    }

    /// The number of tokens in the context.
    #[must_use]
    pub fn position(&self) -> usize {
        self.batch.position(0)
    }

    /// Starts a block (plan §7.7): empties the context and runs `<|endoftext|>`.
    ///
    /// # Errors
    /// The errors of [`CudaSession::step`].
    pub fn start_block(&mut self) -> Result<&[u64]> {
        self.reset();
        self.step(self.eot)
    }

    /// Adds `token` to the context and gives the weights for the next token, as
    /// `hardshell::gpt2::Session::step` does.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::step`].
    pub fn step(&mut self, token: TokenId) -> Result<&[u64]> {
        self.batch.step(&[(0, token)])?;
        Ok(self.batch.weights(0).unwrap_or(&[]))
    }
}
