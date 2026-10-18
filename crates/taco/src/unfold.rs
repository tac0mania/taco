//! `unfold`: a `.taco` container back to the original bytes (plan §4).

use hardshell::gpt2::{Model, Session};
use taco_format::{BlockKind, Container, FormatError};

use crate::codec::decode_block;
use crate::container_hash;
use crate::error::{Result, TacoError};
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

/// Restores the original bytes from a `.taco` container (plan §4).
///
/// The steps:
///
/// 1. Parse and check the container (ADR 0006). The parser does not copy block data.
/// 2. Refuse the container if it needs a model and `model` is `None`, or if its model hash is
///    not the hash of `model` (plan §3 rule 3). A container in the raw-only mode needs no
///    model.
/// 3. Decode each coded block with a new HardShell session: the range decoder gives token
///    ids, and the vocabulary gives their bytes. The decoder never tokenizes. Copy each raw
///    block.
/// 4. Verify the length and the SHA-256 of the result (plan §3 rule 4).
///
/// No buffer is allocated from a length in the container before the blocks decode (ADR 0006
/// section 9.3). Thus a damaged length can cost time, but not a large allocation.
///
/// Cost: for each token, one HardShell step and one frequency table, and a binary search in
/// the decoder. Memory: the decoded blocks, then one copy into the output.
///
/// # Errors
///
/// - [`TacoError::Format`] if the container is not valid.
/// - [`TacoError::ModelRequired`] or [`TacoError::ModelHashMismatch`] for a wrong model.
/// - [`TacoError::OutputTooLarge`] if the original length is more than the limit.
/// - [`TacoError::CorruptBlock`] if a coded block does not decode to its length.
/// - [`TacoError::ChecksumMismatch`] if the restored bytes have a different SHA-256.
/// - [`TacoError::WorkerFailed`] if a worker thread stopped without a result.
pub fn unfold(model: Option<&Model>, container: &[u8], options: &UnfoldOptions) -> Result<Vec<u8>> {
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
    let parts = map_blocks(
        blocks.len(),
        options.threads,
        || model.map(Session::new),
        |session, i| {
            let Some(b) = blocks.get(i) else { return Err(TacoError::WorkerFailed) };
            match (b.kind(), session.as_mut(), model) {
                (BlockKind::Raw, _, _) => Ok(b.stored().to_vec()),
                (BlockKind::Coded, Some(s), Some(m)) => {
                    decode_block(s, m, b.stored(), b.index(), b.original_len())
                }
                // NOTE: the parser refuses a coded block in the raw-only mode, and step 2
                // refuses a missing model. Thus this arm is for block kinds of later versions.
                _ => Err(TacoError::CorruptBlock {
                    block: b.index(),
                    reason: "the block kind needs a model or a newer reader",
                }),
            }
        },
    )?;
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
