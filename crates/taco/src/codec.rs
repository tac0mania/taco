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
use taco_format::Block;

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
    /// The stored form: coded if the payload is smaller than the block, else raw.
    pub(crate) block: Block,
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

/// Codes the tokens of one block of `original_len` bytes, and selects the stored form.
///
/// The encoder rule (ADR 0007): the block is raw if the payload is not smaller than the
/// block. The encoder stops early, with a raw block, when the varint and the bytes that the
/// range coder has written are already at least `original_len` bytes. This rule depends only
/// on the input, so every machine selects the same form.
///
/// All tokens are known, so the session runs them `session.chunk()` at a time
/// (`Session::step_many`, the prefill path). The weights are the same as with one step for
/// each token (a HardShell test proves it), so the coded bytes are the same too.
///
/// Cost: one HardShell step and one frequency table for each token except the last.
pub(crate) fn encode_block(
    session: &mut Session<'_>,
    tokens: &[TokenId],
    original_len: usize,
) -> Result<CodedBlock> {
    let count = u64::try_from(tokens.len())
        .map_err(|_| TacoError::InputTooLarge { reason: "token count" })?;
    let raw = Ok(CodedBlock { block: Block::Raw, tokens: count });
    let prefix = varint_len(count);
    if prefix >= original_len {
        return raw;
    }
    let bits = freq_bits()?;
    let mut encoder = Encoder::with_capacity(original_len);
    let first = FreqTable::from_weights(session.start_block()?, bits)?;
    if let Some((&t0, rest)) = tokens.split_first() {
        encoder.encode(&first, Symbol::new(t0.get()))?;
        if prefix + encoder.bytes_written() >= original_len {
            return raw;
        }
        // The weights after token `i` code token `i + 1`. The last token needs no step.
        let inputs = tokens.get(..tokens.len() - 1).unwrap_or(&[]);
        let mut next = rest.iter();
        for part in inputs.chunks(session.chunk().max(1)) {
            for weights in session.step_many(part)?.iter() {
                let Some(&t) = next.next() else { break };
                let table = FreqTable::from_weights(weights, bits)?;
                encoder.encode(&table, Symbol::new(t.get()))?;
                if prefix + encoder.bytes_written() >= original_len {
                    return raw;
                }
            }
        }
    }
    let coded = encoder.finish();
    let mut payload = Vec::with_capacity(prefix + coded.len());
    write_varint(count, &mut payload);
    payload.extend_from_slice(&coded);
    Ok(CodedBlock { block: Block::choose(payload, original_len), tokens: count })
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
