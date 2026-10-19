//! One interface over the two GPU backends: many independent HardShell sessions ("slots") that
//! run together, and give either the coder interval of each input (the coder op on the GPU)
//! or the full weights of each input.
//!
//! DETERMINISM: both backends give the weights of `hardshell::gpt2::Session` bit for bit
//! (ADR 0008, tests in their crates), and their coder op gives the interval of
//! `taco_coder::FreqTable::from_weights(weights, 32)`. [`crate::backend`] checks both on the
//! loaded model before it uses a device.

use hardshell::TokenId;
use hardshell::gpt2::Overlap;
use hardshell_cuda::{CudaBatch, CudaModel};
use hardshell_gpu::{GpuBatch, GpuModel};
use taco_coder::{CodeInterval, FreqBits};

use crate::backend::Backend;
use crate::codec::FREQ_BITS;
use crate::error::Result;

/// A question to the coder op for one input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Query {
    /// The interval of this token (the encoder).
    Encode(TokenId),
    /// The token whose interval holds this target (the decoder).
    Decode(u64),
}

/// The answer of the coder op: a token and its interval in the table of `T = 2^32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Answer {
    pub(crate) token: TokenId,
    pub(crate) interval: CodeInterval,
}

/// Many sessions of one model on one GPU.
pub(crate) trait Lanes {
    /// The number of slots.
    fn slots(&self) -> usize;

    /// The most inputs of one slot in one call: 1 for a backend without prefill.
    fn per_slot(&self) -> usize;

    /// The backend.
    fn backend(&self) -> Backend;

    /// Empties the context of `slot`.
    fn reset(&mut self, slot: usize);

    /// Runs the inputs (slot, token) and gives the answer of the coder op to `queries[i]`
    /// for each input `i`. The inputs of one slot run at consecutive positions; a full context
    /// starts again as on the CPU (plan §7.7), with the overlap of the slots (ADR 0015).
    fn coded(&mut self, inputs: &[(usize, TokenId)], queries: &[Query]) -> Result<Vec<Answer>>;

    /// Runs the inputs and gives the weights of each input, `vocab` values each, in input
    /// order.
    fn weights(&mut self, inputs: &[(usize, TokenId)]) -> Result<Vec<u64>>;
}

fn freq_bits() -> Result<FreqBits> {
    Ok(FreqBits::new(FREQ_BITS)?)
}

/// Checks one interval of a coder op and makes it a [`CodeInterval`]. A GPU that gives an
/// interval outside the total, or a wrong "last symbol" flag, is a fault of the device.
fn answer(backend: Backend, token: TokenId, start: u64, freq: u64, last: bool) -> Result<Answer> {
    let interval = CodeInterval::new(start, freq, freq_bits()?)
        .map_err(|e| backend.fault(format!("the coder op gave a bad interval: {e}")))?;
    if interval.is_last() != last {
        return Err(backend.fault("the coder op gave a wrong last-symbol flag".to_owned()));
    }
    Ok(Answer { token, interval })
}

/// The CUDA backend: [`CudaBatch`], with prefill.
pub(crate) struct CudaLanes<'m> {
    batch: CudaBatch<'m>,
}

impl<'m> CudaLanes<'m> {
    /// A batch of `slots` slots, passes of up to `rows` rows, and the context overlap
    /// `overlap` (ADR 0015).
    pub(crate) fn new(
        model: &'m CudaModel<'m>,
        slots: usize,
        rows: usize,
        overlap: Overlap,
    ) -> Result<Self> {
        let mut batch = CudaBatch::new(model, slots, rows).map_err(|e| Backend::Cuda.fault(e))?;
        batch.set_overlap(overlap).map_err(|e| Backend::Cuda.fault(e))?;
        Ok(Self { batch })
    }
}

impl Lanes for CudaLanes<'_> {
    fn slots(&self) -> usize {
        self.batch.slots()
    }

    fn per_slot(&self) -> usize {
        usize::MAX
    }

    fn backend(&self) -> Backend {
        Backend::Cuda
    }

    fn reset(&mut self, slot: usize) {
        self.batch.reset(slot);
    }

    fn coded(&mut self, inputs: &[(usize, TokenId)], queries: &[Query]) -> Result<Vec<Answer>> {
        use hardshell_cuda::CoderQuery;
        let q: Vec<CoderQuery> = queries
            .iter()
            .map(|q| match *q {
                Query::Encode(t) => CoderQuery::Encode(t),
                Query::Decode(x) => CoderQuery::Decode(x),
            })
            .collect();
        let out =
            self.batch.run_coded(inputs, &q, FREQ_BITS).map_err(|e| Backend::Cuda.fault(e))?;
        out.iter().map(|a| answer(Backend::Cuda, a.symbol, a.start, a.freq, a.is_last)).collect()
    }

    fn weights(&mut self, inputs: &[(usize, TokenId)]) -> Result<Vec<u64>> {
        self.batch.run_weights(inputs).map_err(|e| Backend::Cuda.fault(e))
    }
}

/// The portable backend: [`GpuBatch`], one input for each slot in a call. A slot whose context
/// is full starts again with the overlap of the batch (ADR 0015), inside the backend.
pub(crate) struct GpuLanes<'m> {
    batch: GpuBatch<'m>,
    vocab: usize,
}

impl<'m> GpuLanes<'m> {
    /// A batch of `slots` slots with the context overlap `overlap` (ADR 0015).
    pub(crate) fn new(model: &'m GpuModel, slots: usize, overlap: Overlap) -> Result<Self> {
        let batch =
            GpuBatch::with_overlap(model, slots, overlap).map_err(|e| Backend::Gpu.fault(e))?;
        Ok(Self { batch, vocab: model.vocab_size() })
    }
}

impl Lanes for GpuLanes<'_> {
    fn slots(&self) -> usize {
        self.batch.slots()
    }

    fn per_slot(&self) -> usize {
        1
    }

    fn backend(&self) -> Backend {
        Backend::Gpu
    }

    fn reset(&mut self, slot: usize) {
        self.batch.reset(slot);
    }

    fn coded(&mut self, inputs: &[(usize, TokenId)], queries: &[Query]) -> Result<Vec<Answer>> {
        use hardshell_gpu::CoderQuery;
        let q: Vec<CoderQuery> = queries
            .iter()
            .map(|q| match *q {
                Query::Encode(t) => CoderQuery::Encode(t),
                Query::Decode(x) => CoderQuery::Decode(x),
            })
            .collect();
        let out =
            self.batch.step_coded(inputs, &q, FREQ_BITS).map_err(|e| Backend::Gpu.fault(e))?;
        if out.len() != inputs.len() {
            return Err(Backend::Gpu.fault("the coder op gave too few intervals".to_owned()));
        }
        out.iter().map(|a| answer(Backend::Gpu, a.symbol, a.start, a.freq, a.is_last)).collect()
    }

    fn weights(&mut self, inputs: &[(usize, TokenId)]) -> Result<Vec<u64>> {
        self.batch.step(inputs).map_err(|e| Backend::Gpu.fault(e))?;
        let mut out = Vec::with_capacity(inputs.len() * self.vocab);
        for &(slot, _) in inputs {
            let w = self
                .batch
                .weights(slot)
                .ok_or_else(|| Backend::Gpu.fault("a slot has no weights".to_owned()))?;
            out.extend_from_slice(w);
        }
        Ok(out)
    }
}
