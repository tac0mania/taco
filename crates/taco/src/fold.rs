//! `fold`: input bytes to a `.taco` container (plan §4).

use hardshell::gpt2::{Model, Session, SessionConfig};
use taco_format::{
    Block, BlockSize, CodingParts, Container, ContainerParams, HEADER_LEN, MixerRule, Profile,
    TABLE_ENTRY_LEN, compress_zstd,
};

use hardshell::TokenId;

use crate::backend::{Backend, Engine};
use crate::codec::{CodedBlock, encode_block, tokens_restore};
use crate::error::{Result, TacoError};
use crate::gpu_fold::{BlockWork, code_blocks};
use crate::mix::MixState;
use crate::parallel::{Threads, map_blocks};
use crate::tokenizer::Tokenizer;
use crate::{DEFAULT_BLOCK_SIZE, container_hash};

/// The number of tokens that the encoder gives to HardShell in one call (the prefill path).
// NOTE: measured on the M2 with gpt2.hsm (1,024 tokens, one thread): 8 tokens 329, 32 tokens
// 345 and 64 tokens 343 steps per second (simd agent). The work buffers grow with it: about
// 12 * 32 * 50,257 bytes = 19 MB for each thread (est.).
const FOLD_CHUNK: usize = 32;

/// The profile that [`fold`] uses (ADR 0012).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FoldProfile {
    /// The profile `text`: HardShell alone. Each block is the smallest of raw, zstd and coded.
    #[default]
    Text,
    /// The profile `logs`: HardShell, the match model and the mixer (ADR 0010, mixer rule
    /// version 1). Each block is the smallest of raw, zstd, coded and mixed, with the tie
    /// order raw, zstd, coded, mixed. Thus each block is never larger than with the profile
    /// `text`.
    Logs,
    /// The encoder selects. It codes as [`FoldProfile::Logs`]. If no block is stored mixed,
    /// it writes the container of [`FoldProfile::Text`] (the same blocks, the profile `text`,
    /// no coding part), else the container of [`FoldProfile::Logs`]. Thus the result is the
    /// smaller of the two containers, and the text container on a tie (it decodes without the
    /// match model and the mixer).
    ///
    /// The rule is for the encoder only: the decoder reads the profile from the header. It
    /// depends only on the input, the model and the block size, so it gives the same bytes on
    /// every machine and for every number of threads.
    Auto,
}

/// The options of [`fold`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FoldOptions {
    /// The number of input bytes in each block (the last block can be shorter).
    pub block_size: BlockSize,
    /// The number of worker threads. The container does not depend on it.
    pub threads: Threads,
    /// The profile. The default is [`FoldProfile::Text`].
    pub profile: FoldProfile,
}

impl FoldOptions {
    /// The default options: blocks of [`DEFAULT_BLOCK_SIZE`] bytes, one thread, the profile
    /// `text`.
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

    /// `true` if [`fold_with`] runs HardShell on a GPU engine with these options: the profile
    /// `text`. The profiles `logs` and `auto` fold on the CPU on every engine, because the
    /// mixer needs all weights of each token (ADR 0016 section 5).
    #[must_use]
    pub const fn uses_gpu(&self) -> bool {
        matches!(self.profile, FoldProfile::Text)
    }

    /// The same options with another profile.
    #[must_use]
    pub const fn with_profile(mut self, profile: FoldProfile) -> Self {
        self.profile = profile;
        self
    }
}

impl Default for FoldOptions {
    fn default() -> Self {
        Self { block_size: DEFAULT_BLOCK_SIZE, threads: Threads::ONE, profile: FoldProfile::Text }
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
    /// The number of blocks that are stored coded (HardShell alone).
    pub coded_blocks: u64,
    /// The number of blocks that are stored mixed (the match model and the mixer, ADR 0010).
    pub mixed_blocks: u64,
    /// The number of blocks that are stored as a zstd frame (ADR 0011). The blocks that are
    /// not coded, not mixed and not zstd are raw.
    pub zstd_blocks: u64,
    /// The number of tokens of all blocks, of every kind.
    pub tokens: u64,
    /// The profile of the container.
    pub profile: Profile,
    /// The length of the container that the profile `text` gives for the same input and block
    /// size, in bytes: each mixed block is replaced by the smallest of raw, zstd and coded.
    /// For the profile `text` this is `container_len`. The encoder knows it without a second
    /// run, because the coded candidate of each block is the same in both profiles.
    pub text_container_len: u64,
    /// The backend that ran HardShell (ADR 0016). The container does not depend on it.
    pub backend: Backend,
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
/// and coded with a new HardShell session and the range coder. Each block is also compressed
/// into one zstd frame (ADR 0011). The smallest of raw, zstd and coded is stored, with the tie
/// order of `Block::choose_smallest` (plan §3 rule 6, ADR 0011 §4). The model stops when the
/// payload reaches the smaller of the block length and the zstd length; this does not change
/// the selected block. The container is version 2. It stores the model hash, the profile
/// (ADR 0012) and the SHA-256 of the input. The output is the same for every number of threads
/// and on every machine.
///
/// With the profile `logs` (or `auto`), a second range encoder codes each block with the
/// match model and the mixer (ADR 0010, a new state at each block start), from the same
/// HardShell weights. The mixed payload joins the candidates last (`Block::or_mixed`): it is
/// stored only if it is smaller than raw, zstd and coded. The container then has the coding
/// part "match model and mixer" with mixer rule version 1. [`FoldProfile::Auto`] gives the
/// rule for `auto`.
///
/// Before a block is coded, its tokens are checked against the vocabulary of the model: they
/// must give the block bytes exactly. Thus a tokenizer that does not agree with the model is an
/// error here, not a damaged container.
///
/// Cost: one zstd compression of each block at level 19 (1.5 to 5.4 MB/s on one thread,
/// measured by the zstdblock agent), then, for each token up to the stop, one HardShell step
/// and one frequency table of the full vocabulary. The profile `logs` adds, for each token, the
/// mixer (one or two passes over the weights) and a second frequency table; and for each
/// block, the match tables (24 MiB of zeroed memory for each thread, ADR 0010 section 7).
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
    fold_with(&Engine::cpu(), model, tokenizer, input, options)
}

/// [`fold`] on the backend of `engine` (ADR 0016). The container is the container of
/// [`fold`], byte for byte, on every backend.
///
/// On a GPU with the profile `text`, the blocks are tokenized and compressed with zstd on
/// `options.threads` threads first. Then HardShell runs the blocks on the GPU: the GPU coder op
/// gives the coder interval of each token (24 bytes for each token come back), and the windows
/// of the context (1,024 tokens for GPT-2) run in parallel slots. The model stops early for a
/// block whose candidate reaches the stop length, as on the CPU, but the rows of a call that is
/// already on the GPU still run. Memory: the tokens and the zstd frame of every block, and the
/// payloads.
///
/// The profiles `logs` and `auto` run on the CPU on every engine (`FoldStats::backend` says
/// so): their mixer needs all weights of each token, and on devbox the read back of the
/// weights made the GPU slower than the CPU (ADR 0016 section 5).
///
/// # Errors
///
/// The errors of [`fold`], and [`TacoError::Backend`] on a fault of the GPU.
pub fn fold_with(
    engine: &Engine<'_>,
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
    let rule = match options.profile {
        FoldProfile::Text => None,
        FoldProfile::Logs | FoldProfile::Auto => Some(MixerRule::V1),
    };
    let prepare = |i: usize| -> Result<(Vec<TokenId>, Vec<u8>)> {
        let index = u32::try_from(i)
            .map_err(|_| TacoError::InputTooLarge { reason: "more than u32::MAX blocks" })?;
        let data = chunks.get(i).copied().unwrap_or(&[]);
        let tokens = tokenizer.encode(data);
        if !tokens_restore(model, &tokens, data) {
            return Err(TacoError::TokenizerRoundTrip { block: index });
        }
        // ADR 0011 §4.5: the zstd frame comes first, so the model can stop at its length.
        Ok((tokens, compress_zstd(data)?))
    };
    let choose = |i: usize, coded: CodedBlock, zstd: Vec<u8>| {
        let len = chunks.get(i).map_or(0, |d| d.len());
        let text = Block::choose_smallest(len, coded.payload, Some(zstd));
        let text_len = text.stored_len(len);
        (text.or_mixed(len, coded.mixed), text_len, coded.tokens)
    };
    // NOTE: the profiles `logs` and `auto` run on the CPU on every engine. Their mixer needs all
    // weights of each token, and the read back made the CUDA fold slower than the CPU fold:
    // 128 KiB of `apache_access.log` took 162.8 s with CUDA and 141.4 s on 6 CPU threads
    // (measured, devbox, ADR 0016 section 5).
    let on_gpu = engine.backend() != Backend::Cpu && options.uses_gpu();
    let coded = if !on_gpu {
        map_blocks(
            chunks.len(),
            options.threads,
            || Session::with_config(model, SessionConfig::new().with_chunk(FOLD_CHUNK)),
            |session, i| {
                let (tokens, zstd) = prepare(i)?;
                let stop = chunks.get(i).map_or(0, |d| d.len()).min(zstd.len());
                let coded = encode_block(session, &tokens, stop, rule.map(MixState::new))?;
                Ok(choose(i, coded, zstd))
            },
        )?
    } else {
        let prepared = map_blocks(chunks.len(), options.threads, || (), |(), i| prepare(i))?;
        let mut work = Vec::with_capacity(prepared.len());
        let mut frames = Vec::with_capacity(prepared.len());
        for (i, (tokens, zstd)) in prepared.into_iter().enumerate() {
            let stop = chunks.get(i).map_or(0, |d| d.len()).min(zstd.len());
            work.push(BlockWork::new(tokens, stop)?);
            frames.push(zstd);
        }
        let coded = code_blocks(engine, model, work)?;
        coded.into_iter().zip(frames).enumerate().map(|(i, (c, z))| choose(i, c, z)).collect()
    };
    let to_u64 =
        |n: usize| u64::try_from(n).map_err(|_| TacoError::InputTooLarge { reason: "length" });
    let mut stats = FoldStats::default();
    let mut blocks = Vec::with_capacity(coded.len());
    let mut text_len = to_u64(HEADER_LEN)?;
    for (block, text_stored, tokens) in coded {
        stats.blocks += 1;
        stats.tokens += tokens;
        text_len += to_u64(TABLE_ENTRY_LEN + text_stored)?;
        match block {
            Block::Coded(_) => stats.coded_blocks += 1,
            Block::Mixed(_) => stats.mixed_blocks += 1,
            Block::Zstd(_) => stats.zstd_blocks += 1,
            _ => {}
        }
        blocks.push(block);
    }
    // FoldProfile::Auto: the text container if no block is mixed. Then the blocks are the
    // blocks of the profile `text`, because a block is mixed only if the mixed payload wins.
    let (profile, coding) = match (options.profile, stats.mixed_blocks) {
        (FoldProfile::Text, _) | (FoldProfile::Auto, 0) => (Profile::Text, CodingParts::NONE),
        _ => (Profile::Logs, CodingParts::with_mixer(MixerRule::V1)),
    };
    let params = ContainerParams::new(container_hash(model), options.block_size)
        .with_profile(profile)
        .with_coding(coding);
    let container = Container::build(&params, input, &blocks)?;
    stats.original_len = to_u64(input.len())?;
    stats.container_len = to_u64(container.len())?;
    stats.profile = profile;
    stats.text_container_len = text_len;
    stats.backend = if on_gpu { engine.backend() } else { Backend::Cpu };
    Ok(Folded { container, stats })
}
