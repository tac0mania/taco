//! `fold`: input bytes to a `.taco` container (plan §4).

use hardshell::gpt2::{Model, Session, SessionConfig};
use taco_format::{Block, BlockSize, Container, ContainerParams};

use crate::codec::{encode_block, tokens_restore};
use crate::error::{Result, TacoError};
use crate::parallel::{Threads, map_blocks};
use crate::tokenizer::Tokenizer;
use crate::{DEFAULT_BLOCK_SIZE, container_hash};

/// The number of tokens that the encoder gives to HardShell in one call (the prefill path).
// NOTE: measured on the M2 with gpt2.hsm (1,024 tokens, one thread): 8 tokens 329, 32 tokens
// 345 and 64 tokens 343 steps per second (simd agent). The work buffers grow with it: about
// 12 * 32 * 50,257 bytes = 19 MB for each thread (est.).
const FOLD_CHUNK: usize = 32;

/// The options of [`fold`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FoldOptions {
    /// The number of input bytes in each block (the last block can be shorter).
    pub block_size: BlockSize,
    /// The number of worker threads. The container does not depend on it.
    pub threads: Threads,
}

impl FoldOptions {
    /// The default options: blocks of [`DEFAULT_BLOCK_SIZE`] bytes, one thread.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The same options with another block size.
    #[must_use]
    pub const fn with_block_size(mut self, block_size: BlockSize) -> Self {
        self.block_size = block_size;
        self
    }

    /// The same options with another number of threads.
    #[must_use]
    pub const fn with_threads(mut self, threads: Threads) -> Self {
        self.threads = threads;
        self
    }
}

impl Default for FoldOptions {
    fn default() -> Self {
        Self { block_size: DEFAULT_BLOCK_SIZE, threads: Threads::ONE }
    }
}

/// Numbers about one [`fold`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FoldStats {
    /// The length of the input, in bytes.
    pub original_len: u64,
    /// The length of the container, in bytes.
    pub container_len: u64,
    /// The number of blocks.
    pub blocks: u64,
    /// The number of blocks that are stored coded. The others are raw.
    pub coded_blocks: u64,
    /// The number of tokens of all blocks, coded and raw.
    pub tokens: u64,
}

/// The result of [`fold`]: the container and numbers about it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Folded {
    /// The bytes of the `.taco` container.
    pub container: Vec<u8>,
    /// Numbers about the container.
    pub stats: FoldStats,
}

/// Compresses `input` into a `.taco` container (plan §4, §9).
///
/// The input is divided into blocks of `options.block_size` bytes. Each block is tokenized,
/// and coded with a new HardShell session and the range coder. A block whose coded form is not
/// smaller is stored raw (plan §3 rule 6). The container stores the model hash and the
/// SHA-256 of the input. The output is the same for every number of threads and on every
/// machine.
///
/// Before a block is coded, its tokens are checked against the vocabulary of the model: they
/// must give the block bytes exactly. Thus a tokenizer that does not agree with the model is an
/// error here, not a damaged container.
///
/// Cost: for each token, one HardShell step and one frequency table of the full vocabulary.
/// The steps of a block run 32 tokens at a time (`Session::step_many`). Each thread holds one
/// session (for GPT-2 small, about 19 MB of KV cache and 19 MB of work buffers, est.). Memory:
/// the input, the container, and the tokens and payload of each block.
///
/// # Errors
///
/// - [`TacoError::TokenizerMismatch`] if the tokenizer has a different vocabulary size.
/// - [`TacoError::TokenizerRoundTrip`] if the tokens of a block do not give its bytes.
/// - [`TacoError::Model`], [`TacoError::Coder`] or [`TacoError::Format`] on an internal fault.
/// - [`TacoError::WorkerFailed`] if a worker thread stopped without a result.
pub fn fold(
    model: &Model,
    tokenizer: &Tokenizer,
    input: &[u8],
    options: &FoldOptions,
) -> Result<Folded> {
    if tokenizer.vocab_size() != model.vocab_size() {
        return Err(TacoError::TokenizerMismatch {
            tokenizer: tokenizer.vocab_size(),
            model: model.vocab_size(),
        });
    }
    let chunks: Vec<&[u8]> = options.block_size.split(input).collect();
    let coded = map_blocks(
        chunks.len(),
        options.threads,
        || Session::with_config(model, SessionConfig::new().with_chunk(FOLD_CHUNK)),
        |session, i| {
            let index = u32::try_from(i)
                .map_err(|_| TacoError::InputTooLarge { reason: "more than u32::MAX blocks" })?;
            let data = chunks.get(i).copied().unwrap_or(&[]);
            let tokens = tokenizer.encode(data);
            if !tokens_restore(model, &tokens, data) {
                return Err(TacoError::TokenizerRoundTrip { block: index });
            }
            encode_block(session, &tokens, data.len())
        },
    )?;
    let mut stats = FoldStats::default();
    let mut blocks = Vec::with_capacity(coded.len());
    for c in coded {
        stats.blocks += 1;
        stats.tokens += c.tokens;
        if matches!(c.block, Block::Coded(_)) {
            stats.coded_blocks += 1;
        }
        blocks.push(c.block);
    }
    let params = ContainerParams::new(container_hash(model), options.block_size);
    let container = Container::build(&params, input, &blocks)?;
    let to_u64 =
        |n: usize| u64::try_from(n).map_err(|_| TacoError::InputTooLarge { reason: "length" });
    stats.original_len = to_u64(input.len())?;
    stats.container_len = to_u64(container.len())?;
    Ok(Folded { container, stats })
}
