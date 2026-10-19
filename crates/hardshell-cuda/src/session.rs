//! One GPT-2 session on a CUDA device, with the API shape of `hardshell::gpt2::Session`.

use hardshell::TokenId;
use hardshell::gpt2::Overlap;

use crate::batch::CudaBatch;
use crate::error::Result;
use crate::model::CudaModel;

/// The most rows of one pass of a [`CudaSession`] with a context overlap.
///
/// NOTE: the prefill chunk of the CPU encoder (`FOLD_CHUNK = 32` in `crates/taco`) times 2.
/// It changes the speed of a restart, not the weights. Each row costs about `20 * vocab` bytes
/// of GPU memory (1 MB for GPT-2 small, est.).
pub const RESTART_ROWS: usize = 64;

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

    /// Makes a session with an empty context and the context overlap `overlap` (ADR 0015),
    /// as `hardshell::gpt2::SessionConfig::with_overlap`. The `K + 1` rows of a restart run
    /// in passes of up to [`RESTART_ROWS`] rows.
    ///
    /// # Errors
    /// The errors of [`CudaBatch::new`] and [`CudaBatch::set_overlap`].
    pub fn with_overlap(model: &'m CudaModel<'m>, overlap: Overlap) -> Result<Self> {
        let rows = if overlap.is_none() { 2 } else { RESTART_ROWS };
        let mut batch = CudaBatch::new(model, 1, rows)?;
        batch.set_overlap(overlap)?;
        Ok(Self { batch, eot: model.eot_token() })
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
