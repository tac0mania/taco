//! `unfold` on a GPU backend: each coded or mixed block decodes in one slot, and all slots step
//! together, one token for each slot in each step (ADR 0016).
//!
//! PLAN(§4): the calls are the calls of ADR 0007 section 3 (ADR 0010 section 4 for a mixed
//! block). For each step:
//!
//! - **Coded block.** The range decoder gives its target value
//!   ([`BlockDecoder::target`]). The GPU runs the last token and its coder op finds the token
//!   whose interval in the table of `FreqTable::from_weights(weights, 32)` holds the target.
//!   The decoder takes that interval ([`BlockDecoder::next_interval`]). Only 24 bytes for each
//!   token come back.
//! - **Mixed block.** The mixer changes the weights before the table, so the GPU gives the full
//!   weights back, and CPU threads run the mixer, the table and the decoder
//!   ([`BlockDecoder::next`]).
//!
//! DETERMINISM: the weights and the intervals are the CPU values, bit for bit (ADR 0008), so
//! each decoder makes the calls of the CPU path in the same order.

use std::collections::VecDeque;

use hardshell::TokenId;
use hardshell::gpt2::Model;

use crate::backend::Engine;
use crate::codec::BlockDecoder;
use crate::error::{Result, TacoError};
use crate::lanes::Query;
use crate::parallel::Threads;

/// A coded or mixed block that waits for a slot: its index in the group and its decoder.
pub(crate) struct Job<'a> {
    pub(crate) part: usize,
    pub(crate) index: u32,
    pub(crate) decoder: BlockDecoder<'a>,
}

/// A block in a slot, with the input of its next step.
struct Lane<'a> {
    job: Job<'a>,
    input: TokenId,
}

/// The decoded bytes and the faults of the jobs: `(part, bytes)` and `(block index, error)`.
pub(crate) type Decoded = (Vec<(usize, Vec<u8>)>, Vec<(u32, TacoError)>);

/// Decodes `jobs` on the GPU of `engine`.
///
/// # Errors
/// [`TacoError::Backend`] on a fault of the GPU. A damaged block is not an error of this
/// function: its error is in the result, with its block index.
pub(crate) fn decode_jobs<'a>(
    engine: &Engine<'_>,
    model: &Model,
    jobs: Vec<Job<'a>>,
    threads: Threads,
) -> Result<Decoded> {
    let mut done = Vec::new();
    let mut errors = Vec::new();
    // NOTE: one row for each slot in a step; a restart adds one row for its slot.
    let rows = jobs.len().clamp(2, MAX_ROWS);
    let Some(mut lanes) = engine.lanes(jobs.len(), rows)? else {
        return Err(TacoError::WorkerFailed);
    };
    let lanes = lanes.as_mut();
    let eot = model.eot_token();
    let vocab = model.vocab_size();
    let mut queue: VecDeque<Job<'a>> = jobs.into();
    let mut slots: Vec<Option<Lane<'a>>> = (0..lanes.slots()).map(|_| None).collect();
    loop {
        for (slot, s) in slots.iter_mut().enumerate() {
            if s.is_none()
                && let Some(job) = queue.pop_front()
            {
                lanes.reset(slot);
                *s = Some(Lane { job, input: eot });
            }
        }
        if slots.iter().all(Option::is_none) {
            return Ok((done, errors));
        }
        let mut outcome: Vec<(usize, Result<Option<TokenId>>)> = Vec::new();
        // Coded blocks: the coder op on the GPU.
        let mut inputs = Vec::new();
        let mut queries = Vec::new();
        for (slot, s) in slots.iter().enumerate() {
            if let Some(l) = s
                && !l.job.decoder.is_mixed()
            {
                inputs.push((slot, l.input));
                queries.push(Query::Decode(l.job.decoder.target()?));
            }
        }
        if !inputs.is_empty() {
            let answers = lanes.coded(&inputs, &queries)?;
            if answers.len() != inputs.len() {
                return Err(lanes.backend().fault("the coder op gave too few answers"));
            }
            for (&(slot, _), a) in inputs.iter().zip(answers) {
                if let Some(Some(l)) = slots.get_mut(slot) {
                    outcome.push((slot, l.job.decoder.next_interval(a.token, a.interval)));
                }
            }
        }
        // Mixed blocks: the weights back, the mixer and the decoder on CPU threads.
        let mixed: Vec<(usize, TokenId)> = slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| {
                s.as_ref().filter(|l| l.job.decoder.is_mixed()).map(|l| (slot, l.input))
            })
            .collect();
        if !mixed.is_empty() {
            let weights = lanes.weights(&mixed)?;
            if weights.len() != mixed.len() * vocab {
                return Err(lanes.backend().fault("the weights have a wrong length"));
            }
            let mut work: Vec<(usize, &mut BlockDecoder<'a>, &[u64])> = Vec::new();
            let mut rows = weights.chunks(vocab.max(1));
            for (slot, s) in slots.iter_mut().enumerate() {
                if let Some(l) = s
                    && l.job.decoder.is_mixed()
                    && let Some(w) = rows.next()
                {
                    work.push((slot, &mut l.job.decoder, w));
                }
            }
            outcome.extend(run_decoders(&mut work, threads)?);
        }
        for (slot, r) in outcome {
            let Some(s) = slots.get_mut(slot) else { continue };
            match r {
                Ok(Some(t)) => {
                    if let Some(l) = s.as_mut() {
                        l.input = t;
                    }
                }
                Ok(None) => {
                    if let Some(l) = s.take() {
                        let (part, index) = (l.job.part, l.job.index);
                        match l.job.decoder.finish() {
                            Ok(bytes) => done.push((part, bytes)),
                            Err(e) => errors.push((index, e)),
                        }
                    }
                }
                Err(e) => {
                    if let Some(l) = s.take() {
                        errors.push((l.job.index, e));
                    }
                }
            }
        }
    }
}

/// The most rows of one pass in unfold: one for each slot (224 slots fill the RTX 3060).
const MAX_ROWS: usize = 224;

/// Runs one step of the mixed decoders of `work` on up to `threads` threads.
fn run_decoders(
    work: &mut [(usize, &mut BlockDecoder<'_>, &[u64])],
    threads: Threads,
) -> Result<Vec<(usize, Result<Option<TokenId>>)>> {
    let per = work.len().div_ceil(threads.get().max(1)).max(1);
    let parts: Vec<Vec<(usize, Result<Option<TokenId>>)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = work
            .chunks_mut(per)
            .map(|part| {
                scope.spawn(move || {
                    part.iter_mut().map(|(slot, d, w)| (*slot, d.next(w))).collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().map_err(|_| TacoError::WorkerFailed))
            .collect::<Result<Vec<_>>>()
    })?;
    Ok(parts.into_iter().flatten().collect())
}
