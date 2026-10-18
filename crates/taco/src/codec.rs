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
use taco_coder::{Decoder, Encoder, FreqBits, FreqTable, Symbol};

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
        if prefix + self.encoder.bytes_written() >= stop {
            self.live = false;
        }
        Ok(())
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
/// model and the mixer.
struct BlockEncoder<'a> {
    plain: Candidate,
    mixed: Option<(Candidate, &'a mut MixState)>,
    prefix: usize,
    stop: usize,
}

impl BlockEncoder<'_> {
    /// Codes one token with the HardShell `weights` in each live candidate. Returns `true`
    /// when every candidate has stopped, so the model can stop too.
    fn push(&mut self, weights: &[u64], token: TokenId) -> Result<bool> {
        let (prefix, stop) = (self.prefix, self.stop);
        if self.plain.live {
            self.plain.encode(weights, token, prefix, stop)?;
        }
        let mut mixed_live = false;
        if let Some((cand, state)) = self.mixed.as_mut()
            && cand.live
        {
            // ADR 0010 section 4: mix with the prediction before the token, code, then
            // update the mixer and the match model with the token.
            cand.encode(state.mix(weights)?, token, prefix, stop)?;
            state.observe(token);
            mixed_live = cand.live;
        }
        Ok(!self.plain.live && !mixed_live)
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
    mix: Option<&mut MixState>,
) -> Result<CodedBlock> {
    let count = u64::try_from(tokens.len())
        .map_err(|_| TacoError::InputTooLarge { reason: "token count" })?;
    let stopped = Ok(CodedBlock { payload: None, mixed: None, tokens: count });
    let prefix = varint_len(count);
    if prefix >= stop {
        return stopped;
    }
    let mut enc = BlockEncoder {
        plain: Candidate::new(stop),
        mixed: mix.map(|m| (Candidate::new(stop), m)),
        prefix,
        stop,
    };
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
    let mixed = enc.mixed.and_then(|(cand, _)| cand.finish(count));
    Ok(CodedBlock { payload: enc.plain.finish(count), mixed, tokens: count })
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
        let corrupt = |reason| TacoError::CorruptBlock { block: self.index, reason };
        let table = match self.mix.as_mut() {
            Some(state) => FreqTable::from_weights(state.mix(weights)?, freq_bits()?)?,
            None => FreqTable::from_weights(weights, freq_bits()?)?,
        };
        let token = TokenId::new(self.decoder.decode(&table).get());
        if let Some(state) = self.mix.as_mut() {
            state.observe(token);
        }
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
            let full = encode_block(&mut session, &tokens, data.len(), Some(&mut mix())).unwrap();
            let short = encode_block(&mut session, &tokens, stop, Some(&mut mix())).unwrap();
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
