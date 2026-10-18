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
//! The session starts the context again when it is full (§7.7, inside HardShell).

use hardshell::TokenId;
use hardshell::gpt2::{Model, Session};
use taco_coder::{Decoder, Encoder, FreqBits, FreqTable, Symbol};

use crate::error::{Result, TacoError};
use crate::payload::{read_varint, varint_len, write_varint};

/// The number of bits of the total of each frequency table: `T = 2^32` (plan §5, §5.1).
pub const FREQ_BITS: u32 = 32;

fn freq_bits() -> Result<FreqBits> {
    Ok(FreqBits::new(FREQ_BITS)?)
}

/// The result of the encoder for one block.
#[derive(Debug)]
pub(crate) struct CodedBlock {
    /// The payload (ADR 0007 §1), or `None` if the encoder stopped early because the payload
    /// was not smaller than the bound.
    pub(crate) payload: Option<Vec<u8>>,
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

/// Codes the tokens of one block, and stops early when the payload cannot be smaller than
/// `stop` bytes.
///
/// `stop` is the smallest stored length of the other candidates: the original length of the
/// block, or the length of its zstd frame if that is smaller (ADR 0007 §5, ADR 0011 §4.5). The
/// encoder stops, with no payload, when the varint and the bytes that the range coder has
/// written are already at least `stop` bytes. The bytes of the coder only grow, so the full
/// payload then has at least `stop` bytes and loses to raw or zstd under
/// `Block::choose_smallest` (zstd wins a tie with coded). Thus the stop does not change the
/// selected block. The rule depends only on the input, so every machine selects the same form.
///
/// All tokens are known, so the session runs them `session.chunk()` at a time
/// (`Session::step_many`, the prefill path). The weights are the same as with one step for
/// each token (a HardShell test proves it), so the coded bytes are the same too.
///
/// Cost: one HardShell step and one frequency table for each token except the last, up to the
/// stop.
pub(crate) fn encode_block(
    session: &mut Session<'_>,
    tokens: &[TokenId],
    stop: usize,
) -> Result<CodedBlock> {
    let count = u64::try_from(tokens.len())
        .map_err(|_| TacoError::InputTooLarge { reason: "token count" })?;
    let stopped = Ok(CodedBlock { payload: None, tokens: count });
    let prefix = varint_len(count);
    if prefix >= stop {
        return stopped;
    }
    let bits = freq_bits()?;
    let mut encoder = Encoder::with_capacity(stop);
    let first = FreqTable::from_weights(session.start_block()?, bits)?;
    if let Some((&t0, rest)) = tokens.split_first() {
        encoder.encode(&first, Symbol::new(t0.get()))?;
        if prefix + encoder.bytes_written() >= stop {
            return stopped;
        }
        // The weights after token `i` code token `i + 1`. The last token needs no step.
        let inputs = tokens.get(..tokens.len() - 1).unwrap_or(&[]);
        let mut next = rest.iter();
        for part in inputs.chunks(session.chunk().max(1)) {
            for weights in session.step_many(part)?.iter() {
                let Some(&t) = next.next() else { break };
                let table = FreqTable::from_weights(weights, bits)?;
                encoder.encode(&table, Symbol::new(t.get()))?;
                if prefix + encoder.bytes_written() >= stop {
                    return stopped;
                }
            }
        }
    }
    let coded = encoder.finish();
    let mut payload = Vec::with_capacity(prefix + coded.len());
    write_varint(count, &mut payload);
    payload.extend_from_slice(&coded);
    Ok(CodedBlock { payload: Some(payload), tokens: count })
}

/// The decoder state of one coded block: the range decoder, the bytes so far and the count.
///
/// The caller gives the weights of each step: first the weights of `start_block`, then the
/// weights after each token that [`BlockDecoder::next`] returned. Thus one state machine
/// serves a single session and a lane of a `hardshell::gpt2::Batch`.
pub(crate) struct BlockDecoder<'a> {
    decoder: Decoder<'a>,
    model: &'a Model,
    index: u32,
    count: u64,
    done: u64,
    limit: usize,
    out: Vec<u8>,
}

impl<'a> BlockDecoder<'a> {
    /// Reads the token count of the payload of coded block `index` of `original_len` bytes.
    ///
    /// # Errors
    /// [`TacoError::CorruptBlock`] if the count is not valid or does not agree with the length.
    pub(crate) fn new(
        model: &'a Model,
        payload: &'a [u8],
        index: u32,
        original_len: u32,
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
        Ok(Self { decoder, model, index, count, done: 0, limit, out: Vec::new() })
    }

    /// Decodes one token with the table of `weights`. Returns the token if the block has more
    /// tokens (the caller steps the model with it), or `None` if the block is complete.
    ///
    /// # Errors
    /// [`TacoError::CorruptBlock`] if the token has no bytes or the bytes are too long.
    pub(crate) fn next(&mut self, weights: &[u64]) -> Result<Option<TokenId>> {
        let corrupt = |reason| TacoError::CorruptBlock { block: self.index, reason };
        let table = FreqTable::from_weights(weights, freq_bits()?)?;
        let token = TokenId::new(self.decoder.decode(&table).get());
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
    use taco_format::{Block, BlockKind, compress_zstd};

    use super::encode_block;
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
            let full = encode_block(&mut session, &tokens, data.len()).unwrap();
            let stop = data.len().min(zstd.len());
            let short = encode_block(&mut session, &tokens, stop).unwrap();
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
        }
        // The stop happened before the original length for some inputs, and the inputs give
        // each kind.
        assert!(early > 0);
        for kind in [BlockKind::Raw, BlockKind::Coded, BlockKind::Zstd] {
            assert!(kinds.contains(&kind), "{kinds:?}");
        }
    }
}
