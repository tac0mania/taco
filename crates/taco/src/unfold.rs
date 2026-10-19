//! `unfold`: a `.taco` container back to the original bytes (plan §4).

use hardshell::gpt2::{Batch, BatchInput, Model, Overlap};
use taco_format::{BlockKind, BlockRef, Container, FormatError, MixerRule};

use crate::backend::{Backend, Engine};
use crate::codec::BlockDecoder;
use crate::container_hash;
use crate::error::{Result, TacoError};
use crate::gpu_unfold::{Job, decode_jobs};
use crate::mix::MixState;
use crate::parallel::{Threads, map_blocks};

/// The options of [`unfold`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnfoldOptions {
    /// The number of worker threads. The output does not depend on it.
    pub threads: Threads,
    /// The largest original length that the caller accepts, or `None` for no limit. A
    /// container of a few bytes can state a very large original length.
    pub max_output_len: Option<u64>,
}

impl UnfoldOptions {
    /// The default options: one thread, no limit.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The same options with another number of threads.
    #[must_use]
    pub const fn with_threads(mut self, threads: Threads) -> Self {
        self.threads = threads;
        self
    }

    /// The same options with a limit on the original length.
    #[must_use]
    pub const fn with_max_output_len(mut self, limit: Option<u64>) -> Self {
        self.max_output_len = limit;
        self
    }
}

/// The context overlap `K` of a container (ADR 0015 §4): [`Overlap::NONE`] for a container
/// without the coding part. Cost: one parse of the header and the block table, no copy.
///
/// # Errors
/// [`TacoError::Format`] if the container is not valid.
pub fn container_overlap(container: &[u8]) -> Result<Overlap> {
    let parsed = Container::parse(container)?;
    Ok(Overlap::new(parsed.header().coding_parts().overlap().get()))
}

/// Restores the original bytes from a `.taco` container (plan §4).
///
/// The steps:
///
/// 1. Parse and check the container (ADR 0006). The parser does not copy block data.
/// 2. Refuse the container if it needs a model and `model` is `None`, or if its model hash is
///    not the hash of `model` (plan §3 rule 3). A container in the raw-only mode needs no
///    model. Refuse a context overlap that is not valid for the model (`K > n_ctx - 2`,
///    ADR 0015).
/// 3. Decode each coded block with a new HardShell context: the range decoder gives token
///    ids, and the vocabulary gives their bytes. When the context is full, it starts again
///    with the last `K` tokens of the block (the overlap of the header, ADR 0015). The
///    decoder never tokenizes. Decode each mixed
///    block in the same way, with a new match model and mixer of the rule version of the
///    header between HardShell and the table (ADR 0010 section 4). Copy each raw block.
///    Decode each zstd block (ADR 0011 §5) without the model. Each thread decodes groups of up
///    to 8 blocks in lockstep in one `hardshell::gpt2::Batch`, so each weight of the model is
///    read once for the group. The weights of each block are the same as with its own session
///    (a HardShell test proves it). The mixer state belongs to the block, so a mixed block
///    works in any lane.
///
/// The decoder selects the coding parts from the header (the coding parts and the block
/// kinds), not from the profile. The caller does not name a profile.
/// 4. Verify the length and the SHA-256 of the result (plan §3 rule 4).
///
/// No buffer is allocated from a length in the container before the blocks decode (ADR 0006
/// section 9.3). Thus a damaged length can cost time, but not a large allocation.
///
/// Cost: for each token, one HardShell step and one frequency table, and a binary search in
/// the decoder; for a mixed block also the mixer (one or two passes over the weights).
/// Memory: the decoded blocks, then one copy into the output; for each mixed block that
/// decodes at the same time, 24 MiB of zeroed match tables (ADR 0010 section 7).
///
/// # Errors
///
/// - [`TacoError::Format`] if the container is not valid, or a zstd block does not decode to
///   its length (`FormatError::CorruptZstdBlock`).
/// - [`TacoError::ModelRequired`] or [`TacoError::ModelHashMismatch`] for a wrong model.
/// - [`TacoError::Model`] with `InvalidOverlap` if the overlap is not valid for the model.
/// - [`TacoError::OutputTooLarge`] if the original length is more than the limit.
/// - [`TacoError::CorruptBlock`] if a coded block does not decode to its length.
/// - [`TacoError::ChecksumMismatch`] if the restored bytes have a different SHA-256.
/// - [`TacoError::WorkerFailed`] if a worker thread stopped without a result.
pub fn unfold(model: Option<&Model>, container: &[u8], options: &UnfoldOptions) -> Result<Vec<u8>> {
    unfold_with(&Engine::cpu(), model, container, options)
}

/// [`unfold`] on the backend of `engine` (ADR 0016). The output is the output of [`unfold`],
/// on every backend. A container of any backend unfolds on every backend.
///
/// On a GPU, every coded and mixed block decodes in its own slot, and all slots step together
/// (one token for each slot in each step). For a coded block the GPU coder op finds the token
/// for the target value of the range decoder (24 bytes for each token come back). For a mixed
/// block the GPU gives the full weights back (about 402 KB for each token of GPT-2 small), and
/// the mixer and the decoder run on `options.threads` CPU threads. Raw and zstd blocks do not
/// use the GPU.
///
/// # Errors
///
/// The errors of [`unfold`], and [`TacoError::Backend`] on a fault of the GPU.
pub fn unfold_with(
    engine: &Engine<'_>,
    model: Option<&Model>,
    container: &[u8],
    options: &UnfoldOptions,
) -> Result<Vec<u8>> {
    let parsed = Container::parse(container)?;
    let header = parsed.header();
    let model = match (header.needs_model(), model) {
        (false, m) => m,
        (true, None) => return Err(TacoError::ModelRequired { container: header.model_hash() }),
        (true, Some(m)) => {
            let loaded = container_hash(m);
            if header.model_hash() != loaded {
                return Err(TacoError::ModelHashMismatch {
                    container: header.model_hash(),
                    loaded,
                });
            }
            Some(m)
        }
    };
    if let Some(limit) = options.max_output_len
        && header.original_len() > limit
    {
        return Err(TacoError::OutputTooLarge { original_len: header.original_len(), limit });
    }
    let blocks = parsed.blocks();
    let rule = header.coding_parts().mixer();
    let overlap = Overlap::new(header.coding_parts().overlap().get());
    if let Some(m) = model {
        m.check_overlap(overlap)?;
    }
    let parts: Vec<Vec<u8>> = match model {
        Some(m) if engine.backend() != Backend::Cpu => {
            decode_on_gpu(engine, m, (rule, overlap), blocks, options.threads)?
        }
        _ => {
            // Groups of up to MAX_LANES blocks decode in lockstep on one thread (the batch path of
            // HardShell). The group size gives each thread at least one group when there are enough
            // blocks. The output does not depend on it (each block is independent).
            // NOTE: lanes come before threads. On the M2, 2 blocks of alice29 (32 KiB) unfold in 48.9 s
            // on 1 thread with 2 lanes, and in 86.5 s on 2 threads with 1 lane each (measured).
            let fill = blocks.len().min(MIN_LANES);
            let group_size =
                blocks.len().div_ceil(options.threads.get()).max(fill).clamp(1, MAX_LANES);
            let group_parts = map_blocks(
                blocks.len().div_ceil(group_size),
                options.threads,
                || None,
                |batch: &mut Option<Batch<'_>>, g| {
                    let end = ((g + 1) * group_size).min(blocks.len());
                    let group = blocks.get(g * group_size..end).unwrap_or(&[]);
                    decode_group(model, (rule, overlap), batch, group_size, group)
                },
            )?;
            group_parts.into_iter().flatten().collect()
        }
    };
    let total: usize = parts.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total);
    for p in &parts {
        out.extend_from_slice(p);
    }
    drop(parts);
    header.verify_checksum(&out).map_err(|e| match e {
        FormatError::ChecksumMismatch { expected, found } => {
            TacoError::ChecksumMismatch { expected, found }
        }
        other => TacoError::Format(other),
    })?;
    Ok(out)
}

/// The smallest number of lanes of a group when the container has that many blocks.
const MIN_LANES: usize = 4;

/// The largest number of blocks that one thread decodes in lockstep. Each lane holds one KV
/// cache (for GPT-2 small, 18.9 MB, est.).
// UNVERIFIED: 8 lanes is an estimate of the point where the batch stops to be limited by the
// memory bandwidth on the M2; the step and batch measurements are in the simd agent report.
const MAX_LANES: usize = 8;

/// Decodes one group of blocks: raw blocks are copied, zstd blocks are decoded, and the coded
/// and mixed blocks run in lockstep in the lanes of `batch` (made on the first use, with
/// `lanes` lanes). `rule` is the mixer rule version of the header, if the container has the
/// match model and the mixer, and `overlap` the context overlap of the header.
///
/// If several blocks of the group fail, the result is the error of the block with the lowest
/// index, so the error does not depend on the grouping.
fn decode_group<'m>(
    model: Option<&'m Model>,
    (rule, overlap): (Option<MixerRule>, Overlap),
    batch: &mut Option<Batch<'m>>,
    lanes: usize,
    group: &[BlockRef<'_>],
) -> Result<Vec<Vec<u8>>> {
    let mut parts: Vec<Option<Vec<u8>>> = vec![None; group.len()];
    let mut errors: Vec<(u32, TacoError)> = Vec::new();
    let mut lane_of: Vec<Option<(usize, BlockDecoder<'_>)>> = Vec::new();
    for (slot, b) in group.iter().enumerate() {
        match (b.kind(), model) {
            (BlockKind::Raw, _) => parts[slot] = Some(b.stored().to_vec()),
            (BlockKind::Coded, Some(m)) => {
                match BlockDecoder::new(m, b.stored(), b.index(), b.original_len(), None) {
                    Ok(d) => lane_of.push(Some((slot, d))),
                    Err(e) => errors.push((b.index(), e)),
                }
            }
            // NOTE: the parser refuses a mixed block without a rule version in the header.
            (BlockKind::Mixed, Some(m)) if rule.is_some() => {
                let mix = rule.map(MixState::new);
                match BlockDecoder::new(m, b.stored(), b.index(), b.original_len(), mix) {
                    Ok(d) => lane_of.push(Some((slot, d))),
                    Err(e) => errors.push((b.index(), e)),
                }
            }
            // A zstd block needs no model (ADR 0011 §3.2).
            (BlockKind::Zstd, _) => {
                let mut bytes = Vec::new();
                match b.decode_zstd_into(&mut bytes) {
                    Ok(()) => parts[slot] = Some(bytes),
                    Err(e) => errors.push((b.index(), TacoError::Format(e))),
                }
            }
            // NOTE: the parser refuses a coded block in the raw-only mode, and `unfold`
            // refuses a missing model. Thus this arm is for block kinds of later versions only.
            _ => errors.push((
                b.index(),
                TacoError::CorruptBlock {
                    block: b.index(),
                    reason: "the block kind needs a model or a newer reader",
                },
            )),
        }
    }
    if let (Some(m), false) = (model, lane_of.is_empty()) {
        let batch = batch.get_or_insert_with(|| Batch::new(m, lanes).with_overlap(overlap));
        let mut inputs: Vec<Option<BatchInput>> = vec![None; batch.lanes()];
        for (input, lane) in inputs.iter_mut().zip(&lane_of) {
            *input = lane.as_ref().map(|_| BatchInput::Start);
        }
        while inputs.iter().any(Option::is_some) {
            batch.run(&inputs)?;
            for (lane, (input, state)) in inputs.iter_mut().zip(lane_of.iter_mut()).enumerate() {
                if input.is_none() {
                    continue;
                }
                let Some((slot, dec)) = state.as_mut() else { continue };
                let weights = batch.weights(lane).ok_or(TacoError::WorkerFailed)?;
                match dec.next(weights) {
                    Ok(Some(t)) => *input = Some(BatchInput::Token(t)),
                    Ok(None) => {
                        *input = None;
                        let slot = *slot;
                        if let Some((_, dec)) = state.take() {
                            let index = group.get(slot).map_or(0, BlockRef::index);
                            match dec.finish() {
                                Ok(bytes) => parts[slot] = Some(bytes),
                                Err(e) => errors.push((index, e)),
                            }
                        }
                    }
                    Err(e) => {
                        *input = None;
                        errors.push((group.get(*slot).map_or(0, BlockRef::index), e));
                        *state = None;
                    }
                }
            }
        }
    }
    if let Some((_, e)) = errors.into_iter().min_by_key(|(i, _)| *i) {
        return Err(e);
    }
    parts.into_iter().map(|p| p.ok_or(TacoError::WorkerFailed)).collect()
}

/// Decodes all blocks with the GPU of `engine`: raw and zstd blocks on the CPU threads, coded
/// and mixed blocks in the slots of the GPU. The error is the error of the block with the
/// lowest index, as on the CPU.
fn decode_on_gpu(
    engine: &Engine<'_>,
    model: &Model,
    (rule, overlap): (Option<MixerRule>, Overlap),
    blocks: &[BlockRef<'_>],
    threads: Threads,
) -> Result<Vec<Vec<u8>>> {
    // The blocks without the model, on the threads; the others become jobs.
    let direct = map_blocks(
        blocks.len(),
        threads,
        || (),
        |(), i| {
            let Some(b) = blocks.get(i) else { return Ok(None) };
            Ok(match b.kind() {
                BlockKind::Raw => Some(Ok(b.stored().to_vec())),
                BlockKind::Zstd => {
                    let mut bytes = Vec::new();
                    Some(b.decode_zstd_into(&mut bytes).map(|()| bytes).map_err(TacoError::Format))
                }
                _ => None,
            })
        },
    )?;
    let mut parts: Vec<Option<Vec<u8>>> = vec![None; blocks.len()];
    let mut errors: Vec<(u32, TacoError)> = Vec::new();
    let mut jobs = Vec::new();
    for (part, (b, d)) in blocks.iter().zip(direct).enumerate() {
        match (d, b.kind()) {
            (Some(Ok(bytes)), _) => parts[part] = Some(bytes),
            (Some(Err(e)), _) => errors.push((b.index(), e)),
            (None, kind) => {
                // NOTE: the parser refuses a mixed block without a rule version in the header.
                let mix = match (kind, rule) {
                    (BlockKind::Coded, _) => None,
                    (BlockKind::Mixed, Some(r)) => Some(MixState::new(r)),
                    _ => {
                        errors.push((
                            b.index(),
                            TacoError::CorruptBlock {
                                block: b.index(),
                                reason: "the block kind needs a model or a newer reader",
                            },
                        ));
                        continue;
                    }
                };
                match BlockDecoder::new(model, b.stored(), b.index(), b.original_len(), mix) {
                    Ok(decoder) => jobs.push(Job { part, index: b.index(), decoder }),
                    Err(e) => errors.push((b.index(), e)),
                }
            }
        }
    }
    if !jobs.is_empty() {
        let (done, failed) = decode_jobs(engine, model, jobs, threads, overlap)?;
        for (part, bytes) in done {
            if let Some(p) = parts.get_mut(part) {
                *p = Some(bytes);
            }
        }
        errors.extend(failed);
    }
    if let Some((_, e)) = errors.into_iter().min_by_key(|(i, _)| *i) {
        return Err(e);
    }
    parts.into_iter().map(|p| p.ok_or(TacoError::WorkerFailed)).collect()
}
