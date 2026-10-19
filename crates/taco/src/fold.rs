//! `fold`: input bytes to a `.taco` container (plan §4).

use hardshell::gpt2::{Model, Overlap, Session, SessionConfig};
use taco_format::{
    Block, BlockSize, CodingParts, Container, ContainerParams, ContextOverlap, HEADER_LEN,
    MixerRule, Profile, TABLE_ENTRY_LEN, compress_zstd,
};

use hardshell::TokenId;

use crate::backend::{Backend, Engine};
use crate::codec::{CodedBlock, encode_block, tokens_restore};
use crate::error::{Result, TacoError};
use crate::gpu_fold::{BlockWork, code_blocks};
use crate::mix::MixState;
use crate::parallel::{Threads, map_blocks};
use crate::tokenizer::Tokenizer;
use crate::{DEFAULT_BLOCK_SIZE, LOGS_BLOCK_SIZE, SPREAD_BLOCKS, SPREAD_MIN_BLOCK, container_hash};

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

impl FoldProfile {
    /// The block size of this profile when the caller does not give one:
    /// [`DEFAULT_BLOCK_SIZE`] for `text`, [`LOGS_BLOCK_SIZE`] for `logs` and `auto` (`auto`
    /// codes as `logs`).
    #[must_use]
    pub const fn default_block_size(self) -> BlockSize {
        match self {
            Self::Text => DEFAULT_BLOCK_SIZE,
            Self::Logs | Self::Auto => LOGS_BLOCK_SIZE,
        }
    }

    /// The context overlap of this profile for `model` when the caller does not give one
    /// (ADR 0015 §5): [`Overlap::NONE`] for `text` (the decoder is faster), and half of the
    /// context of the model for `logs` and `auto` (`auto` codes as `logs`): `K = 512` for
    /// GPT-2.
    ///
    /// NOTE: `K = 512` gave −20.6 % table bits on 1 MiB of an Apache log against `K = 0`, and
    /// 0.4 % to 1.2 % less than `K = 384` (measured, RTX 3060,
    /// `docs/research/2026-10-19-long-context.md` section 7). The logs measurement of this
    /// default is `docs/research/2026-10-19-overlap-cli.md`. The value is half of the context,
    /// so a model with a short context (the test models) gets a valid overlap too.
    #[must_use]
    pub fn default_overlap(self, model: &Model) -> Overlap {
        match self {
            Self::Text => Overlap::NONE,
            Self::Logs | Self::Auto => {
                let n_ctx = model.n_ctx();
                let half = (n_ctx / 2).min(n_ctx.saturating_sub(2));
                Overlap::new(u16::try_from(half).unwrap_or(u16::MAX))
            }
        }
    }
}

/// The options of [`fold`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FoldOptions {
    /// The number of input bytes in each block (the last block can be shorter), or `None`
    /// for the default of the profile ([`FoldProfile::default_block_size`]).
    pub block_size: Option<BlockSize>,
    /// The number of worker threads. The container does not depend on it.
    pub threads: Threads,
    /// The profile. The default is [`FoldProfile::Text`].
    pub profile: FoldProfile,
    /// The context overlap `K` of every block (ADR 0015), or `None` for the default of the
    /// profile ([`FoldProfile::default_overlap`]). The container stores it, so the decoder
    /// needs no option.
    pub overlap: Option<Overlap>,
    /// With the profile `text` and no `block_size`: smaller blocks for a small input, so that
    /// more blocks decode in parallel on a GPU ([`FoldOptions::block_size_for`]). The default
    /// is `false`. The container stores the block size, so the decoder needs no option. Only
    /// [`fold`] and [`fold_with`] apply it: the stream fold (`crate::stream`) does not know the
    /// input length before it writes the header, so it keeps the block size of the options.
    pub spread: bool,
}

impl FoldOptions {
    /// The default options: one thread, the profile `text`, and the block size and the
    /// context overlap of the profile (16 KiB blocks and no overlap for `text`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The same options with another block size.
    #[must_use]
    pub const fn with_block_size(mut self, block_size: BlockSize) -> Self {
        self.block_size = Some(block_size);
        self
    }

    /// The same options with the context overlap `overlap` (ADR 0015). [`fold`] refuses an
    /// overlap that is not valid for the model (`K > n_ctx - 2`).
    #[must_use]
    pub const fn with_overlap(mut self, overlap: Overlap) -> Self {
        self.overlap = Some(overlap);
        self
    }

    /// The block size that [`fold`] uses: the block size of the options, else the default of
    /// the profile.
    #[must_use]
    pub const fn effective_block_size(&self) -> BlockSize {
        match self.block_size {
            Some(b) => b,
            None => self.profile.default_block_size(),
        }
    }

    /// The same options with the spread rule on or off ([`FoldOptions::spread`]).
    #[must_use]
    pub const fn with_spread(mut self, spread: bool) -> Self {
        self.spread = spread;
        self
    }

    /// The block size that [`fold`] uses for an input of `input_len` bytes. It is
    /// [`FoldOptions::effective_block_size`], except with [`FoldOptions::spread`], the profile
    /// `text` and no `block_size`: then it is the smallest power of two that gives the input at
    /// most [`SPREAD_BLOCKS`] blocks, but at least [`SPREAD_MIN_BLOCK`] and at most the default
    /// of the profile. The rule depends only on the input length, so every machine and every
    /// backend selects the same block size.
    ///
    /// The unfold of one block is sequential (one model step for each token), so a small file
    /// with few large blocks unfolds no faster than its largest block. Smaller blocks run in
    /// more GPU slots at the same time.
    #[must_use]
    pub fn block_size_for(&self, input_len: usize) -> BlockSize {
        let default = self.effective_block_size();
        if !self.spread || self.block_size.is_some() || self.profile != FoldProfile::Text {
            return default;
        }
        let per_block = input_len.div_ceil(SPREAD_BLOCKS);
        let size = per_block
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
            .clamp(SPREAD_MIN_BLOCK.as_usize(), default.as_usize());
        u32::try_from(size).ok().and_then(|s| BlockSize::new(s).ok()).unwrap_or(default)
    }

    /// The context overlap that [`fold`] uses with `model`: the overlap of the options, else
    /// the default of the profile.
    ///
    /// # Errors
    /// [`TacoError::Model`] with `InvalidOverlap` if it is more than `n_ctx - 2`.
    pub fn effective_overlap(&self, model: &Model) -> Result<Overlap> {
        let overlap = self.overlap.unwrap_or_else(|| self.profile.default_overlap(model));
        model.check_overlap(overlap)?;
        Ok(overlap)
    }

    /// The same options with another number of threads.
    #[must_use]
    pub const fn with_threads(mut self, threads: Threads) -> Self {
        self.threads = threads;
        self
    }

    /// `true` if [`fold_with`] can run HardShell on a GPU engine with these options: every
    /// profile. The profile `text` folds on every GPU engine. The profiles `logs` and `auto`
    /// need the mixed coder op (ADR 0016 section 5), which only the CUDA engine has; on the
    /// wgpu engine they fold on the CPU (`FoldStats::backend` says so).
    #[must_use]
    pub const fn uses_gpu(&self) -> bool {
        matches!(self.profile, FoldProfile::Text | FoldProfile::Logs | FoldProfile::Auto)
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
        Self {
            block_size: None,
            threads: Threads::ONE,
            profile: FoldProfile::Text,
            overlap: None,
            spread: false,
        }
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
    /// The context overlap of the container (ADR 0015).
    pub overlap: Overlap,
    /// The block size of the container ([`FoldOptions::block_size_for`]). `None` only in
    /// `FoldStats::default()`.
    pub block_size: Option<BlockSize>,
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
/// The input is divided into blocks of `options.effective_block_size()` bytes. Each block is
/// tokenized, and coded with a new HardShell session and the range coder. When the context is
/// full, the session starts again with the last `K` tokens (the context overlap of ADR 0015,
/// `options.effective_overlap(model)`); the container stores `K`. Each block is also compressed
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
/// - [`TacoError::Model`] with `InvalidOverlap` if the overlap is more than `n_ctx - 2`.
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
/// On CUDA, the profiles `logs` and `auto` run each block in one slot (the mixer state is
/// sequential over the block), and the dual coder op gives the intervals of GPT-2 alone and of
/// the mixer table for each token (48 bytes come back); the match model and the mixer run on
/// the host. On wgpu they run on the CPU (`FoldStats::backend` says so): there the mixer
/// needs all weights of each token, and the read back made the GPU slower than the CPU (ADR
/// 0016 section 5).
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
    let mut plan = FoldPlan::new(model, tokenizer, options)?;
    plan.block_size = options.block_size_for(input.len());
    let chunks: Vec<&[u8]> = plan.block_size.split(input).collect();
    let coded = code_segment(engine, model, tokenizer, &plan, (0, &chunks), &|_| {})?;
    let mut stats = FoldStats::default();
    let mut blocks = Vec::with_capacity(coded.len());
    for (block, text_stored, tokens) in coded {
        stats.count_block(&block, text_stored, tokens)?;
        blocks.push(block);
    }
    let (profile, coding) = plan.container_parts(stats.mixed_blocks);
    let params = ContainerParams::new(container_hash(model), plan.block_size)
        .with_profile(profile)
        .with_coding(coding);
    let container = Container::build(&params, input, &blocks)?;
    stats.finish(&plan, engine, profile, to_u64(input.len())?, to_u64(container.len())?)?;
    Ok(Folded { container, stats })
}

/// `n` as a `u64`, or [`TacoError::InputTooLarge`].
fn to_u64(n: usize) -> Result<u64> {
    u64::try_from(n).map_err(|_| TacoError::InputTooLarge { reason: "length" })
}

/// One coded block: the selected form, its stored length with the profile `text`, and the
/// number of tokens.
pub(crate) type CodedResult = (Block, usize, u64);

/// The checked settings of one fold. [`fold_with`] and the stream fold (`crate::stream`) use
/// the same plan, so they code each block in the same way.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FoldPlan {
    /// The block size of the container.
    pub(crate) block_size: BlockSize,
    /// The context overlap `K`, checked against the model.
    pub(crate) overlap: Overlap,
    rule: Option<MixerRule>,
    profile: FoldProfile,
    threads: Threads,
    /// [`FoldOptions::uses_gpu`].
    uses_gpu: bool,
}

impl FoldPlan {
    /// Checks `options` against `model` and `tokenizer`.
    ///
    /// # Errors
    /// [`TacoError::TokenizerMismatch`], or [`TacoError::Model`] with `InvalidOverlap`.
    pub(crate) fn new(model: &Model, tokenizer: &Tokenizer, options: &FoldOptions) -> Result<Self> {
        if tokenizer.vocab_size() != model.vocab_size() {
            return Err(TacoError::TokenizerMismatch {
                tokenizer: tokenizer.vocab_size(),
                model: model.vocab_size(),
            });
        }
        let rule = match options.profile {
            FoldProfile::Text => None,
            FoldProfile::Logs | FoldProfile::Auto => Some(MixerRule::V1),
        };
        Ok(Self {
            block_size: options.effective_block_size(),
            overlap: options.effective_overlap(model)?,
            rule,
            profile: options.profile,
            threads: options.threads,
            uses_gpu: options.uses_gpu(),
        })
    }

    /// `true` if HardShell runs on the GPU of `engine`: the profile `text` on a GPU engine,
    /// and the profiles `logs` and `auto` on the CUDA engine (the mixed coder op).
    pub(crate) fn on_gpu(&self, engine: &Engine<'_>) -> bool {
        // NOTE: on wgpu the profiles `logs` and `auto` run on the CPU: there the mixer needs all
        // weights of each token, and the read back made the GPU fold slower than the CPU fold
        // (ADR 0016 section 5).
        engine.backend() != Backend::Cpu
            && self.uses_gpu
            && (self.rule.is_none() || engine.mixes_on_gpu())
    }

    /// The profile and the coding parts while the blocks are coded: the parts of the profile
    /// `logs` for `logs` and `auto`, so that a mixed block is permitted.
    pub(crate) fn block_parts(&self) -> (Profile, CodingParts) {
        let overlap = ContextOverlap::new(self.overlap.get());
        match self.rule {
            None => (Profile::Text, CodingParts::NONE.with_overlap(overlap)),
            Some(rule) => (Profile::Logs, CodingParts::with_mixer(rule).with_overlap(overlap)),
        }
    }

    /// The profile and the coding parts of the container after all blocks, with
    /// `mixed_blocks` mixed blocks. [`FoldProfile::Auto`] gives the text container if no block
    /// is mixed. Then the blocks are the blocks of the profile `text`, because a block is mixed
    /// only if the mixed payload wins.
    pub(crate) fn container_parts(&self, mixed_blocks: u64) -> (Profile, CodingParts) {
        let overlap = ContextOverlap::new(self.overlap.get());
        // ADR 0015 §4: the container stores K for all blocks, in both profiles.
        match (self.profile, mixed_blocks, self.rule) {
            (FoldProfile::Text, _, _) | (FoldProfile::Auto, 0, _) | (_, _, None) => {
                (Profile::Text, CodingParts::NONE.with_overlap(overlap))
            }
            (_, _, Some(rule)) => {
                (Profile::Logs, CodingParts::with_mixer(rule).with_overlap(overlap))
            }
        }
    }
}

impl FoldStats {
    /// Adds one coded block to the numbers.
    pub(crate) fn count_block(
        &mut self,
        block: &Block,
        text_stored: usize,
        tokens: u64,
    ) -> Result<()> {
        if self.blocks == 0 {
            self.text_container_len = to_u64(HEADER_LEN)?;
        }
        self.blocks += 1;
        self.tokens += tokens;
        self.text_container_len += to_u64(TABLE_ENTRY_LEN + text_stored)?;
        match block {
            Block::Coded(_) => self.coded_blocks += 1,
            Block::Mixed(_) => self.mixed_blocks += 1,
            Block::Zstd(_) => self.zstd_blocks += 1,
            _ => {}
        }
        Ok(())
    }

    /// Sets the numbers that are known at the end.
    pub(crate) fn finish(
        &mut self,
        plan: &FoldPlan,
        engine: &Engine<'_>,
        profile: Profile,
        original_len: u64,
        container_len: u64,
    ) -> Result<()> {
        if self.blocks == 0 {
            self.text_container_len = to_u64(HEADER_LEN)?;
        }
        self.original_len = original_len;
        self.container_len = container_len;
        self.profile = profile;
        self.backend = if plan.on_gpu(engine) { engine.backend() } else { Backend::Cpu };
        self.overlap = plan.overlap;
        self.block_size = Some(plan.block_size);
        Ok(())
    }
}

/// Codes the blocks `chunks` (the blocks `first_block..` of the input) with `plan`, and gives
/// each block in order. `on_block(len)` runs after each block with its input length, on the
/// thread that coded it. The result of each block does not depend on the other blocks of the
/// call, so a caller can code a large input in parts (`crate::stream`).
///
/// # Errors
///
/// The block errors of [`fold`], and [`TacoError::Backend`] on a fault of the GPU.
pub(crate) fn code_segment(
    engine: &Engine<'_>,
    model: &Model,
    tokenizer: &Tokenizer,
    plan: &FoldPlan,
    (first_block, chunks): (u64, &[&[u8]]),
    on_block: &(dyn Fn(usize) + Sync),
) -> Result<Vec<CodedResult>> {
    let rule = plan.rule;
    let prepare = |i: usize| -> Result<(Vec<TokenId>, Vec<u8>)> {
        let index = u64::try_from(i)
            .ok()
            .and_then(|i| i.checked_add(first_block))
            .and_then(|i| u32::try_from(i).ok())
            .ok_or(TacoError::InputTooLarge { reason: "more than u32::MAX blocks" })?;
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
    let config = SessionConfig::new().with_chunk(FOLD_CHUNK).with_overlap(plan.overlap);
    if !plan.on_gpu(engine) {
        return map_blocks(
            chunks.len(),
            plan.threads,
            || Session::with_config(model, config),
            |session, i| {
                let (tokens, zstd) = prepare(i)?;
                let len = chunks.get(i).map_or(0, |d| d.len());
                let coded =
                    encode_block(session, &tokens, len.min(zstd.len()), rule.map(MixState::new))?;
                on_block(len);
                Ok(choose(i, coded, zstd))
            },
        );
    }
    let prepared = map_blocks(chunks.len(), plan.threads, || (), |(), i| prepare(i))?;
    let mut work = Vec::with_capacity(prepared.len());
    let mut frames = Vec::with_capacity(prepared.len());
    for (i, (tokens, zstd)) in prepared.into_iter().enumerate() {
        let stop = chunks.get(i).map_or(0, |d| d.len()).min(zstd.len());
        work.push(BlockWork::new(tokens, stop, rule)?);
        frames.push(zstd);
    }
    let coded = code_blocks(engine, model, work, plan.overlap)?;
    Ok(coded
        .into_iter()
        .zip(frames)
        .enumerate()
        .map(|(i, (c, z))| {
            on_block(chunks.get(i).map_or(0, |d| d.len()));
            choose(i, c, z)
        })
        .collect())
}
