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
//! - **Mixed block, CUDA.** The mixed coder op (ADR 0016 section 5) finds the token in the
//!   table of the mixer weights. The match model gives the prediction before the step; the GPU
//!   gives the statistics of the weights, the host mixer gives `q` (`Mixer::mix_stats`), and
//!   the GPU makes the mixer table and answers the query. Coded and mixed blocks run in the
//!   same call. Then the decoder takes the interval, and the mixer and the match model take the
//!   token ([`BlockDecoder::next_mixed_interval`]).
//! - **Mixed block, wgpu.** The backend has no mixed op, so the GPU gives the full weights
//!   back, and CPU threads run the mixer, the table and the decoder ([`BlockDecoder::next`]).
//!
//! DETERMINISM: the weights and the intervals are the CPU values, bit for bit (ADR 0008), so
//! each decoder makes the calls of the CPU path in the same order.

use std::collections::VecDeque;

use hardshell::TokenId;
use hardshell::gpt2::{Model, Overlap};
use taco_format::MixerRule;

use crate::backend::Engine;
use crate::codec::BlockDecoder;
use crate::error::{Result, TacoError};
use crate::lanes::{Lanes, Query, Stats, Table};
use crate::mix::MixState;
use crate::parallel::Threads;

/// A coded or mixed block that waits for a slot: its part (the position in the output), its
/// block index, its stored payload and original length, and the mixer rule of a mixed block.
///
/// NOTE: the decoder, and for a mixed block its match model and mixer (24 MiB of zeroed match
/// tables and 144 KiB of logistic tables), are made when the block enters a slot and dropped
/// when it leaves. Thus the memory is bounded by the slot count, not by the block count of a
/// container (security finding F2: 20,000 mixed blocks of 2 bytes used 11 GB before).
pub(crate) struct Job<'a> {
    pub(crate) part: usize,
    pub(crate) index: u32,
    pub(crate) payload: &'a [u8],
    pub(crate) original_len: u32,
    pub(crate) mix: Option<MixerRule>,
}

/// A block in a slot: its part, its block index and its decoder.
struct Running<'a> {
    part: usize,
    index: u32,
    decoder: BlockDecoder<'a>,
}

/// A block in a slot, with the input of its next step.
struct Lane<'a> {
    job: Running<'a>,
    input: TokenId,
}

/// The decoded bytes and the faults of the jobs: `(part, bytes)` and `(block index, error)`.
pub(crate) type Decoded = (Vec<(usize, Vec<u8>)>, Vec<(u32, TacoError)>);

/// Decodes `jobs` on the GPU of `engine`. Each slot starts its context again with the last
/// `overlap` tokens when it is full (ADR 0015), inside the backend.
///
/// # Errors
/// [`TacoError::Backend`] on a fault of the GPU. A damaged block is not an error of this
/// function: its error is in the result, with its block index.
pub(crate) fn decode_jobs<'a>(
    engine: &Engine<'_>,
    model: &'a Model,
    jobs: Vec<Job<'a>>,
    threads: Threads,
    overlap: Overlap,
) -> Result<Decoded> {
    let mut done = Vec::new();
    let mut errors = Vec::new();
    // NOTE: one row for each slot in a step; a restart adds `K + 1` rows for its slot. With an
    // overlap the passes have room for MAX_ROWS rows, so a restart of K = 512 takes 3 passes
    // and not 257 passes of 2 rows.
    let rows = if overlap.is_none() { jobs.len().clamp(2, MAX_ROWS) } else { MAX_ROWS };
    let Some(mut lanes) = engine.lanes(jobs.len(), rows, overlap)? else {
        return Err(TacoError::WorkerFailed);
    };
    let lanes = lanes.as_mut();
    let eot = model.eot_token();
    let vocab = model.vocab_size();
    let (mixed_jobs, coded_jobs): (Vec<Job<'a>>, Vec<Job<'a>>) =
        jobs.into_iter().partition(|j| j.mix.is_some());
    let mut queue: VecDeque<Job<'a>> = coded_jobs.into();
    let mut mixed_queue: VecDeque<Job<'a>> = mixed_jobs.into();
    let mut slots: Vec<Option<Lane<'a>>> = (0..lanes.slots()).map(|_| None).collect();
    loop {
        let mut live_mixed = slots.iter().flatten().filter(|l| l.job.decoder.is_mixed()).count();
        for (slot, s) in slots.iter_mut().enumerate() {
            while s.is_none() {
                // NOTE: at most MAX_MIXED_SLOTS mixed blocks at the same time; the other slots
                // take coded blocks.
                let next = if live_mixed < MAX_MIXED_SLOTS && !mixed_queue.is_empty() {
                    live_mixed += 1;
                    mixed_queue.pop_front()
                } else {
                    queue.pop_front()
                };
                let Some(job) = next else { break };
                let mix = job.mix.map(MixState::new);
                match BlockDecoder::new(model, job.payload, job.index, job.original_len, mix) {
                    Ok(decoder) => {
                        lanes.reset(slot);
                        let job = Running { part: job.part, index: job.index, decoder };
                        *s = Some(Lane { job, input: eot });
                    }
                    Err(e) => {
                        if job.mix.is_some() {
                            live_mixed = live_mixed.saturating_sub(1);
                        }
                        errors.push((job.index, e));
                    }
                }
            }
        }
        if slots.iter().all(Option::is_none) {
            return Ok((done, errors));
        }
        // NOTE: a step without a mixed block uses the coder op of the profile `text`: the
        // mixed op adds the stats kernel and one more read back for each pass (it gave 74 % to
        // 94 % of the tokens/s of the text op, measured, `MIXED-INTEGRATION.md` section 7).
        let any_mixed = slots.iter().flatten().any(|l| l.job.decoder.is_mixed());
        if any_mixed && lanes.has_mixer() {
            let outcome = step_mixed(lanes, &mut slots)?;
            settle(&mut slots, outcome, &mut done, &mut errors);
            continue;
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
        settle(&mut slots, outcome, &mut done, &mut errors);
    }
}

/// The result of one step of each slot: the next token, `None` at the end of the block, or
/// the fault of the block.
type Outcome = Vec<(usize, Result<Option<TokenId>>)>;

/// Applies the outcome of a step: the next input of each slot, and the blocks that ended or
/// failed leave their slots.
fn settle(
    slots: &mut [Option<Lane<'_>>],
    outcome: Outcome,
    done: &mut Vec<(usize, Vec<u8>)>,
    errors: &mut Vec<(u32, TacoError)>,
) {
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

/// One step of every slot with the mixed coder op (CUDA, ADR 0016 section 5): coded and mixed
/// blocks in one call, 24 bytes back for each token twice, no CPU threads. For a mixed block
/// the table is the mixer table with the prediction of its match model; the host step gives
/// `q` from its mixer (`Mixer::mix_stats`). The decoder then updates the mixer and the match
/// model with the token (`gpu_decode` of `hardshell-cuda/examples/mixed.rs`).
fn step_mixed(lanes: &mut dyn Lanes, slots: &mut [Option<Lane<'_>>]) -> Result<Outcome> {
    let mut inputs = Vec::new();
    let mut queries = Vec::new();
    let mut tables = Vec::new();
    let mut preds = Vec::new();
    for (slot, s) in slots.iter_mut().enumerate() {
        let Some(l) = s.as_mut() else { continue };
        inputs.push((slot, l.input));
        queries.push(Query::Decode(l.job.decoder.target()?));
        let pred = l.job.decoder.mix_state().map(|m| m.predict());
        tables.push(match pred {
            None => Table::Plain,
            Some(p) => Table::Mixed(p.map(|p| TokenId::new(p.token.get()))),
        });
        preds.push(pred.flatten());
    }
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    // The host step of input `i`: `q` from the mixer of its block. The token is not known yet,
    // so the update comes after the decoder step. A `q` of 0 makes the op fail.
    let owners: Vec<usize> = inputs.iter().map(|&(slot, _)| slot).collect();
    let mut host = |i: usize, st: Stats| -> u32 {
        let (Some(&slot), Some(&p)) = (owners.get(i), preds.get(i)) else { return 0 };
        let Some(state) =
            slots.get_mut(slot).and_then(Option::as_mut).and_then(|l| l.job.decoder.mix_state())
        else {
            return 0;
        };
        state.mix_stats(st.total, st.predicted, st.top, p).unwrap_or(0)
    };
    let answers = lanes.mixed(&inputs, &queries, &tables, &mut host)?;
    if answers.len() != inputs.len() {
        return Err(lanes.backend().fault("the mixed coder op gave too few answers"));
    }
    let mut outcome = Vec::with_capacity(answers.len());
    for (&(slot, _), a) in inputs.iter().zip(answers) {
        if let Some(Some(l)) = slots.get_mut(slot) {
            let d = &mut l.job.decoder;
            let r = if d.is_mixed() {
                d.next_mixed_interval(a.token, a.interval)
            } else {
                d.next_interval(a.token, a.interval)
            };
            outcome.push((slot, r));
        }
    }
    Ok(outcome)
}

/// The most rows of one pass in unfold: one for each slot (224 slots fill the RTX 3060).
const MAX_ROWS: usize = 224;

/// The most mixed blocks that decode at the same time. Each holds a match model and a mixer:
/// with glibc, the 24 MiB of match tables are zeroed with a write after the first free (the
/// heap reuses the memory), so each live state costs 24 MiB of resident memory.
// NOTE: measured on devbox (tiny model, CUDA), the growth of the peak resident memory: GPU fold
// of 20,000 logs blocks of 2 bytes with 64 states 1,176,216 KiB; with 16 states 513,388 KiB
// (5,000 blocks; the engine alone 111,040 KiB); GPU unfold of 20,000 mixed blocks
// (`tests/security_unfold.rs`) with 16 states 525,728 KiB, above its limit of 512 MiB. 12
// states keep both below the limit (the values with 12 are in the gpucli2 report).
pub(crate) const MAX_MIXED_SLOTS: usize = 12;

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
