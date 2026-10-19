//! One block through the model and the range coder: tokens to a payload, and back.
//!
//! PLAN(§4): for each token, HardShell gives one weight for each token of the vocabulary,
//! `taco_coder::FreqTable::from_weights` makes a table with the total `2^32` (§5.1), and the
//! range coder codes the token with that table. The encoder and the decoder make the same
//! calls in the same order, so they compute the same tables.
//!
//! The order of the calls for a block of `n` tokens `t[0..n]`:
//!
//! 1. `Session::start_block()` gives the weights for `t[0]` (§7.7: the context starts with
//!    `<|endoftext|>`).
//! 2. For `i` in `0..n`: code `t[i]` with the table of the current weights. Then, if
//!    `i + 1 < n`, `Session::step(t[i])` gives the weights for `t[i + 1]`.
//!
//! The encoder knows all tokens, so it uses `Session::step_many` (many steps at once); the
//! decoder runs one step for each token, in a lane of a `Batch`. Both give the weights of
//! these calls, bit for bit.
//!
//! A mixed block (the profile `logs`, ADR 0010) makes the same calls. Between the HardShell
//! weights and the table, the match model and the mixer change the weights
//! ([`crate::mix::MixState::mix`]), and after the token they observe it. The payload has the
//! same layout (ADR 0007 §1).
//!
//! The session starts the context again when it is full (§7.7, inside HardShell).

use hardshell::TokenId;
use hardshell::gpt2::{Model, Session};
use taco_coder::{CodeInterval, Decoder, Encoder, FreqBits, FreqTable, Symbol};

use crate::error::{Result, TacoError};
use crate::mix::MixState;
use crate::payload::{read_varint, varint_len, write_varint};

/// The number of bits of the total of each frequency table: `T = 2^32` (plan §5, §5.1).
pub const FREQ_BITS: u32 = 32;

fn freq_bits() -> Result<FreqBits> {
    Ok(FreqBits::new(FREQ_BITS)?)
}

/// The result of the encoder for one block.
#[derive(Debug)]
pub(crate) struct CodedBlock {
    /// The payload with the HardShell weights (ADR 0007 §1), or `None` if the encoder stopped
    /// early because the payload was not smaller than the bound.
    pub(crate) payload: Option<Vec<u8>>,
    /// The payload with the weights of the match model and the mixer (ADR 0010), with the same
    /// layout and the same stop rule. `None` if the encoder had no mixer or stopped early.
    pub(crate) mixed: Option<Vec<u8>>,
    /// The number of tokens of the block.
    pub(crate) tokens: u64,
}

/// Checks that `tokens` give exactly `data` with the vocabulary of `model`. Cost: one pass.
pub(crate) fn tokens_restore(model: &Model, tokens: &[TokenId], data: &[u8]) -> bool {
    let vocab = model.vocabulary();
    let mut rest = data;
    for &t in tokens {
        let Some(bytes) = vocab.token(t) else { return false };
        if t == model.eot_token() || bytes.is_empty() {
            return false;
        }
        match rest.strip_prefix(bytes) {
            Some(r) => rest = r,
            None => return false,
        }
    }
    rest.is_empty()
}

/// One range encoder of a block, with its stop.
struct Candidate {
    encoder: Encoder,
    /// False after the encoder reached the stop length: its payload loses (ADR 0007 §5.2).
    live: bool,
}

impl Candidate {
    fn new(stop: usize) -> Self {
        Self { encoder: Encoder::with_capacity(stop), live: true }
    }

    /// Codes `token` with the table of `weights`, and stops the candidate at `stop` bytes.
    fn encode(
        &mut self,
        weights: &[u64],
        token: TokenId,
        prefix: usize,
        stop: usize,
    ) -> Result<()> {
        let table = FreqTable::from_weights(weights, freq_bits()?)?;
        self.encoder.encode(&table, Symbol::new(token.get()))?;
        self.check_stop(prefix, stop);
        Ok(())
    }

    /// Codes the interval of a token in the table of its weights (from a GPU coder op), and
    /// stops the candidate at `stop` bytes. The bytes are the bytes of [`Candidate::encode`]
    /// (a `taco-coder` test proves it).
    fn encode_interval(&mut self, interval: CodeInterval, prefix: usize, stop: usize) {
        self.encoder.encode_interval(interval);
        self.check_stop(prefix, stop);
    }

    /// Stops the candidate when its payload cannot be shorter than `stop`. This uses the lower
    /// bound of the final coder length, not the bytes written: `finish` removes trailing zero
    /// bytes (4,096 zero bytes code to 2 bytes, measured with the tiny model).
    fn check_stop(&mut self, prefix: usize, stop: usize) {
        if prefix + self.encoder.min_final_len() >= stop {
            self.live = false;
        }
    }

    /// The payload: the varint of `count`, then the coder bytes. `None` if the candidate
    /// stopped.
    fn finish(self, count: u64) -> Option<Vec<u8>> {
        if !self.live {
            return None;
        }
        let coded = self.encoder.finish();
        let mut payload = Vec::with_capacity(varint_len(count) + coded.len());
        write_varint(count, &mut payload);
        payload.extend_from_slice(&coded);
        Some(payload)
    }
}

/// The encoders of one block: GPT-2 alone, and the mixed form if the profile has the match
/// model and the mixer. The caller gives the tokens in order, each with the HardShell weights
/// of its position ([`BlockEncoder::push`]) or, for GPT-2 alone, with its interval from a GPU
/// coder op ([`BlockEncoder::push_interval`]).
pub(crate) struct BlockEncoder {
    plain: Candidate,
    /// The mixed candidate and its state. The state is `None` while the block waits for a GPU
    /// slot ([`BlockEncoder::with_mixer_later`]), so the memory of many waiting blocks does
    /// not grow with the block count.
    mixed: Option<(Candidate, Option<MixState>)>,
    count: u64,
    prefix: usize,
    stop: usize,
}

impl BlockEncoder {
    /// The encoders of a block of `count` tokens with the stop length `stop` (ADR 0007 §5.2).
    /// `mix` is a new state (the block start) for the profile `logs`, else `None`.
    pub(crate) fn new(count: u64, stop: usize, mix: Option<MixState>) -> Self {
        Self {
            plain: Candidate::new(stop),
            mixed: mix.map(|m| (Candidate::new(stop), Some(m))),
            count,
            prefix: varint_len(count),
            stop,
        }
    }

    /// As [`BlockEncoder::new`] with a mixer, but the mixer state comes later
    /// ([`BlockEncoder::set_mix_state`], when the block enters a GPU slot).
    pub(crate) fn with_mixer_later(count: u64, stop: usize) -> Self {
        let mut enc = Self::new(count, stop, None);
        enc.mixed = Some((Candidate::new(stop), None));
        enc
    }

    /// Gives the mixed candidate its state (a new state: the block start).
    pub(crate) fn set_mix_state(&mut self, state: MixState) {
        if let Some((_, m)) = self.mixed.as_mut() {
            *m = Some(state);
        }
    }

    /// Drops the mixer state (the block left its GPU slot). The payloads stay.
    pub(crate) fn drop_mix_state(&mut self) {
        if let Some((_, m)) = self.mixed.as_mut() {
            *m = None;
        }
    }

    /// `true` when every candidate has stopped, so the model can stop too. A block whose
    /// varint alone reaches the stop length is stopped before the first token.
    pub(crate) fn stopped(&self) -> bool {
        let mixed_live = self.mixed.as_ref().is_some_and(|(c, _)| c.live);
        self.prefix >= self.stop || (!self.plain.live && !mixed_live)
    }

    /// Codes one token with the HardShell `weights` in each live candidate. Returns `true`
    /// when every candidate has stopped, so the model can stop too.
    pub(crate) fn push(&mut self, weights: &[u64], token: TokenId) -> Result<bool> {
        let (prefix, stop) = (self.prefix, self.stop);
        if self.plain.live {
            self.plain.encode(weights, token, prefix, stop)?;
        }
        if let Some((cand, state)) = self.mixed.as_mut()
            && cand.live
        {
            // ADR 0010 section 4: mix with the prediction before the token, code, then
            // update the mixer and the match model with the token.
            let state = state.as_mut().ok_or(TacoError::WorkerFailed)?;
            cand.encode(state.mix(weights)?, token, prefix, stop)?;
            state.observe(token);
        }
        Ok(self.stopped())
    }

    /// Codes one token of GPT-2 alone with its `interval` in the table of its HardShell
    /// weights. Returns `true` when the candidate has stopped. An encoder with the mixed
    /// candidate needs the weights: use [`BlockEncoder::push`].
    pub(crate) fn push_interval(&mut self, interval: CodeInterval) -> bool {
        let (prefix, stop) = (self.prefix, self.stop);
        if self.plain.live {
            self.plain.encode_interval(interval, prefix, stop);
        }
        self.stopped()
    }

    /// Codes one token with its two intervals from the GPU dual op (`run_coded_dual`): `plain`
    /// in the table of its HardShell weights, `mixed` in the table of the mixer weights. The
    /// caller ran the mixer state of [`BlockEncoder::mix_state`] for this token. Each live
    /// candidate codes its interval; the bytes are the bytes of [`BlockEncoder::push`].
    /// Returns `true` when every candidate has stopped.
    pub(crate) fn push_dual(&mut self, plain: CodeInterval, mixed: CodeInterval) -> bool {
        let (prefix, stop) = (self.prefix, self.stop);
        if self.plain.live {
            self.plain.encode_interval(plain, prefix, stop);
        }
        if let Some((cand, _)) = self.mixed.as_mut()
            && cand.live
        {
            cand.encode_interval(mixed, prefix, stop);
        }
        self.stopped()
    }

    /// The match model and the mixer of the mixed candidate, or `None` without a mixer or
    /// before [`BlockEncoder::set_mix_state`].
    pub(crate) fn mix_state(&mut self) -> Option<&mut MixState> {
        self.mixed.as_mut().and_then(|(_, m)| m.as_mut())
    }

    /// `true` if the encoder has the mixed candidate (the profile `logs`).
    pub(crate) fn has_mixer(&self) -> bool {
        self.mixed.is_some()
    }

    /// The payloads of GPT-2 alone and of the mixed form, as [`encode_block`] gives them.
    pub(crate) fn finish(self) -> CodedBlock {
        if self.prefix >= self.stop {
            return CodedBlock { payload: None, mixed: None, tokens: self.count };
        }
        let mixed = self.mixed.and_then(|(cand, _)| cand.finish(self.count));
        CodedBlock { payload: self.plain.finish(self.count), mixed, tokens: self.count }
    }
}

/// Codes the tokens of one block, and stops early when the payload cannot be smaller than
/// `stop` bytes.
///
/// `stop` is the smallest stored length of the other candidates: the original length of the
/// block, or the length of its zstd frame if that is smaller (ADR 0007 §5, ADR 0011 §4.5). The
/// encoder stops, with no payload, when the varint and the bytes that the range coder has
/// written are already at least `stop` bytes. The bytes of the coder only grow, so the full
/// payload then has at least `stop` bytes and loses to raw or zstd under
/// `Block::choose_smallest` (zstd wins a tie with coded and mixed). Thus the stop does not
/// change the selected block. The rule depends only on the input, so every machine selects the
/// same form.
///
/// With `mix` (the profile `logs`), a second range encoder codes the same tokens with the
/// weights of the match model and the mixer (ADR 0010 section 4), from the same HardShell
/// weights. Each encoder has its own stop; the model stops when both stopped. `mix` must be a
/// new state (the block start). Without `mix`, the result is the result of the text profile,
/// byte for byte.
///
/// All tokens are known, so the session runs them `session.chunk()` at a time
/// (`Session::step_many`, the prefill path). The weights are the same as with one step for
/// each token (a HardShell test proves it), so the coded bytes are the same too.
///
/// Cost: one HardShell step and one frequency table for each token except the last, up to the
/// stop. With `mix`: also a copy of the weights, the mixer (one or two passes over the
/// weights) and a second frequency table for each token.
pub(crate) fn encode_block(
    session: &mut Session<'_>,
    tokens: &[TokenId],
    stop: usize,
    mix: Option<MixState>,
) -> Result<CodedBlock> {
    let count = u64::try_from(tokens.len())
        .map_err(|_| TacoError::InputTooLarge { reason: "token count" })?;
    let mut enc = BlockEncoder::new(count, stop, mix);
    let stopped = Ok(CodedBlock { payload: None, mixed: None, tokens: count });
    if enc.stopped() {
        return stopped;
    }
    let first = session.start_block()?;
    if let Some((&t0, rest)) = tokens.split_first() {
        if enc.push(first, t0)? {
            return stopped;
        }
        // The weights after token `i` code token `i + 1`. The last token needs no step.
        let inputs = tokens.get(..tokens.len() - 1).unwrap_or(&[]);
        let mut next = rest.iter();
        for part in inputs.chunks(session.chunk().max(1)) {
            for weights in session.step_many(part)?.iter() {
                let Some(&t) = next.next() else { break };
                if enc.push(weights, t)? {
                    return stopped;
                }
            }
        }
    }
    Ok(enc.finish())
}

/// The decoder state of one coded or mixed block: the range decoder, the bytes so far, the
/// count, and for a mixed block the match model and the mixer.
///
/// The caller gives the HardShell weights of each step: first the weights of `start_block`,
/// then the weights after each token that [`BlockDecoder::next`] returned. Thus one state
/// machine serves a single session and a lane of a `hardshell::gpt2::Batch`. The mixer state
/// belongs to the block, so it works in any lane.
pub(crate) struct BlockDecoder<'a> {
    decoder: Decoder<'a>,
    model: &'a Model,
    index: u32,
    count: u64,
    done: u64,
    limit: usize,
    out: Vec<u8>,
    mix: Option<MixState>,
}

impl<'a> BlockDecoder<'a> {
    /// Reads the token count of the payload of block `index` of `original_len` bytes. `mix` is
    /// a new state for a mixed block (ADR 0010), `None` for a coded block.
    ///
    /// # Errors
    /// [`TacoError::CorruptBlock`] if the count is not valid or does not agree with the length.
    pub(crate) fn new(
        model: &'a Model,
        payload: &'a [u8],
        index: u32,
        original_len: u32,
        mix: Option<MixState>,
    ) -> Result<Self> {
        let corrupt = |reason| TacoError::CorruptBlock { block: index, reason };
        let (count, coded) = read_varint(payload).ok_or(corrupt("the token count is not valid"))?;
        // INVARIANT: every token has at least one byte, so a valid block has at most
        // `original_len` tokens.
        if count > u64::from(original_len) || count == 0 {
            return Err(corrupt("the token count does not agree with the block length"));
        }
        let limit = usize::try_from(original_len)
            .map_err(|_| TacoError::InputTooLarge { reason: "block length" })?;
        let decoder = Decoder::new(coded);
        Ok(Self { decoder, model, index, count, done: 0, limit, out: Vec::new(), mix })
    }

    /// Decodes one token with the table of `weights` (after the mixer, for a mixed block).
    /// Returns the token if the block has more tokens (the caller steps the model with it), or
    /// `None` if the block is complete.
    ///
    /// # Errors
    /// [`TacoError::CorruptBlock`] if the token has no bytes or the bytes are too long.
    pub(crate) fn next(&mut self, weights: &[u64]) -> Result<Option<TokenId>> {
        let table = match self.mix.as_mut() {
            Some(state) => FreqTable::from_weights(state.mix(weights)?, freq_bits()?)?,
            None => FreqTable::from_weights(weights, freq_bits()?)?,
        };
        let token = TokenId::new(self.decoder.decode(&table).get());
        if let Some(state) = self.mix.as_mut() {
            state.observe(token);
        }
        self.accept(token)
    }

    /// `true` if the block is mixed: [`BlockDecoder::next`] needs the HardShell weights. A
    /// coded block can also decode with [`BlockDecoder::target`] and
    /// [`BlockDecoder::next_interval`].
    pub(crate) fn is_mixed(&self) -> bool {
        self.mix.is_some()
    }

    /// The target value of the range decoder for the next token (ADR 0003 decoder step 2,
    /// with `T = 2^32`). A GPU coder op finds the token whose interval holds it.
    ///
    /// # Errors
    /// [`TacoError::Coder`] only on an internal fault (the constant `FREQ_BITS`).
    pub(crate) fn target(&self) -> Result<u64> {
        Ok(self.decoder.target(freq_bits()?))
    }

    /// Decodes the next token of a coded block from its symbol and interval, which a GPU
    /// coder op found for [`BlockDecoder::target`]. The result is the result of
    /// [`BlockDecoder::next`] with the HardShell weights.
    ///
    /// # Errors
    /// [`TacoError::Coder`] if the interval does not hold the target (a fault of the coder
    /// op), or if the block is mixed; [`TacoError::CorruptBlock`] as [`BlockDecoder::next`].
    pub(crate) fn next_interval(
        &mut self,
        token: TokenId,
        interval: CodeInterval,
    ) -> Result<Option<TokenId>> {
        if self.mix.is_some() {
            return Err(TacoError::CorruptBlock {
                block: self.index,
                reason: "a mixed block needs the weights of each token",
            });
        }
        self.decoder.decode_interval(interval)?;
        self.accept(token)
    }

    /// Decodes the next token of a mixed block from its symbol and interval, which the GPU
    /// mixed coder op found for [`BlockDecoder::target`] in the table of the mixer weights.
    /// Before the op, the caller took the prediction ([`MixState::predict`]) and, in the host
    /// step of the op, called [`MixState::mix_stats`] for this position. Then this call
    /// updates the mixer and the match model with the token (ADR 0010 section 4). The result
    /// is the result of [`BlockDecoder::next`] with the HardShell weights.
    ///
    /// # Errors
    /// [`TacoError::Coder`] if the interval does not hold the target (a fault of the coder
    /// op), [`TacoError::CorruptBlock`] if the block is not mixed, or as
    /// [`BlockDecoder::next`].
    pub(crate) fn next_mixed_interval(
        &mut self,
        token: TokenId,
        interval: CodeInterval,
    ) -> Result<Option<TokenId>> {
        let Some(state) = self.mix.as_mut() else {
            return Err(TacoError::CorruptBlock {
                block: self.index,
                reason: "a coded block has no mixer",
            });
        };
        self.decoder.decode_interval(interval)?;
        state.observe(token);
        self.accept(token)
    }

    /// The match model and the mixer of a mixed block, or `None` for a coded block.
    pub(crate) fn mix_state(&mut self) -> Option<&mut MixState> {
        self.mix.as_mut()
    }

    /// Adds the bytes of a decoded token. Returns the token if the block has more tokens.
    fn accept(&mut self, token: TokenId) -> Result<Option<TokenId>> {
        let corrupt = |reason| TacoError::CorruptBlock { block: self.index, reason };
        let vocab = self.model.vocabulary();
        let bytes = vocab.token(token).ok_or(corrupt("a token is not in the vocabulary"))?;
        if token == self.model.eot_token() || bytes.is_empty() {
            return Err(corrupt("the block has a token with no bytes"));
        }
        // INVARIANT: out.len() <= limit, by this check in the steps before.
        if bytes.len() > self.limit - self.out.len() {
            return Err(corrupt("the tokens are longer than the block"));
        }
        self.out.extend_from_slice(bytes);
        self.done += 1;
        Ok((self.done < self.count).then_some(token))
    }

    /// The bytes of the complete block.
    ///
    /// # Errors
    /// [`TacoError::CorruptBlock`] if the tokens are shorter than the block.
    pub(crate) fn finish(self) -> Result<Vec<u8>> {
        if self.out.len() != self.limit {
            return Err(TacoError::CorruptBlock {
                block: self.index,
                reason: "the tokens are shorter than the block",
            });
        }
        Ok(self.out)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)] // Test data from integer generators.
mod tests {
    use hardshell::gpt2::{Model, Session, SessionConfig};
    use taco_format::{Block, BlockKind, MixerRule, compress_zstd};

    use super::encode_block;
    use crate::mix::MixState;
    use crate::tokenizer::Tokenizer;

    /// The committed tiny model of the conformance vectors (random weights, with merges).
    const TINY: &[u8] =
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../conformance/tiny.hsm"));

    #[test]
    fn the_stop_at_the_zstd_length_does_not_change_the_selected_block() {
        // ADR 0007 §5.2, ADR 0011 §4.5.
        let model = Model::load(TINY).unwrap();
        let merges: Vec<_> = model
            .vocabulary()
            .merges()
            .unwrap_or(&[])
            .iter()
            .map(|m| (m.left(), m.right()))
            .collect();
        let tokenizer =
            Tokenizer::for_vocabulary(model.vocabulary(), model.eot_token(), &merges).unwrap();
        let mut session = Session::with_config(&model, SessionConfig::new().with_chunk(8));
        let text = "The quick brown fox jumps over the lazy dog. ";
        let mut inputs: Vec<Vec<u8>> = vec![
            b"a".to_vec(),
            text.as_bytes().to_vec(),
            text.repeat(20).into_bytes(),
            "dasdasdasdada".repeat(30).into_bytes(),
            (0..=255u8).collect(),
            (0..600u32).flat_map(|i| [(i % 7) as u8, 0, 0, 1]).collect(),
            // The coder writes zero bytes that `finish` removes again: the bytes written are
            // not a lower bound of the payload (found in a review, 2026-10-19).
            vec![0u8; 4096],
        ];
        let mut x = 0x7AC0_u64;
        inputs.push(
            (0..900)
                .map(|_| {
                    x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (x >> 56) as u8
                })
                .collect(),
        );
        // The random tiny model codes single bytes at about 8.2 bits each (est.), so only text of
        // merged (multi-byte) tokens codes smaller than raw. Merged tokens in a random order
        // (coded wins or zstd wins) and repeated (zstd wins).
        let vocab = model.vocabulary();
        let merged: Vec<&[u8]> = (256..u32::try_from(model.vocab_size()).unwrap())
            .filter_map(|i| vocab.token(hardshell::TokenId::new(i)))
            .filter(|b| b.len() > 1)
            .collect();
        let mut picked = Vec::new();
        for _ in 0..400 {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            picked.extend_from_slice(merged[(x >> 33) as usize % merged.len()]);
        }
        inputs.push(picked[..picked.len() / 40].to_vec());
        inputs.push(picked[..picked.len() / 8].to_vec());
        inputs.push(picked.clone());
        inputs.push(picked[..picked.len() / 4].repeat(3));
        let mut early = 0;
        let mut kinds = Vec::new();
        for data in &inputs {
            let tokens = tokenizer.encode(data);
            let zstd = compress_zstd(data).unwrap();
            let full = encode_block(&mut session, &tokens, data.len(), None).unwrap();
            let stop = data.len().min(zstd.len());
            let short = encode_block(&mut session, &tokens, stop, None).unwrap();
            assert_eq!(full.tokens, short.tokens);
            if full.payload.is_some() && short.payload.is_none() {
                early += 1;
            }
            let selected = Block::choose_smallest(data.len(), short.payload, Some(zstd.clone()));
            assert_eq!(
                Block::choose_smallest(data.len(), full.payload, Some(zstd)),
                selected,
                "{} bytes",
                data.len()
            );
            kinds.push(selected.kind());
            // The same rule for the mixed candidate of the profile logs (ADR 0010 §5.7).
            let mix = || MixState::new(MixerRule::V1);
            let full = encode_block(&mut session, &tokens, data.len(), Some(mix())).unwrap();
            let short = encode_block(&mut session, &tokens, stop, Some(mix())).unwrap();
            let zstd_again = compress_zstd(data).unwrap();
            let pick = |c: super::CodedBlock| {
                Block::choose_smallest(data.len(), c.payload, Some(zstd_again.clone()))
                    .or_mixed(data.len(), c.mixed)
            };
            assert_eq!(pick(full), pick(short), "{} bytes, mixed", data.len());
        }
        // The stop happened before the original length for some inputs, and the inputs give
        // each kind.
        assert!(early > 0);
        for kind in [BlockKind::Raw, BlockKind::Coded, BlockKind::Zstd] {
            assert!(kinds.contains(&kind), "{kinds:?}");
        }
    }
}
