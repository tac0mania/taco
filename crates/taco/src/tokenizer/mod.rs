//! The encoder side of the tokenizer: GPT-2 byte-level BPE, a port of
//! `tools/baseline/bpe.py`.
//!
//! PLAN(§10): the tokenizer takes any byte string. It divides the input into maximal runs of
//! well-formed UTF-8 and single bytes that are not part of well-formed UTF-8. A UTF-8 run goes
//! through the GPT-2 pre-tokenizer (`pretokenize`) and the BPE merges (`bpe`). A byte outside
//! a UTF-8 run becomes its single-byte token. Each token has fixed bytes, so the
//! concatenation of the bytes of the tokens is the input, for every input.
//!
//! For valid UTF-8 the output is the canonical GPT-2 tokenization, if the vocabulary and the
//! merges are those of GPT-2. For other input it is not canonical, because a piece never
//! crosses an invalid byte. Plan §10 permits this.
//!
//! The decoder never tokenizes (plan §4). It uses only the vocabulary of the model file.
//!
//! DETERMINISM: the output depends only on the input bytes, the vocabulary and the merges.
//! The Unicode classes are fixed tables in the source (`unicode_tables`).

mod bpe;
mod pretokenize;
mod unicode_tables;

use std::collections::HashMap;

use hardshell::TokenId;
use hardshell::hsm::Vocabulary;

use crate::error::{Result, TacoError};
use bpe::{Merge, MergeMap};
use pretokenize::Pieces;

/// What is wrong with a merge that [`Tokenizer::new`] refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MergeFault {
    /// A part of the pair is not in the vocabulary.
    TokenOutOfRange,
    /// A part of the pair is the `<|endoftext|>` token or a token with no bytes.
    NotAByteToken,
    /// No token has the joined bytes of the pair.
    NoJoinedToken,
    /// Two or more tokens have the joined bytes of the pair, so the result is not defined.
    AmbiguousJoinedToken,
    /// The same pair occurs two times in the list.
    Duplicate,
}

impl core::fmt::Display for MergeFault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::TokenOutOfRange => "a part is not in the vocabulary",
            Self::NotAByteToken => "a part is <|endoftext|> or has no bytes",
            Self::NoJoinedToken => "no token has the joined bytes",
            Self::AmbiguousJoinedToken => "two or more tokens have the joined bytes",
            Self::Duplicate => "the pair occurs two times",
        })
    }
}

/// A byte-level BPE encoder for one vocabulary and one list of merges.
///
/// Make it one time for a model. It is `Send` and `Sync`, so the blocks of one input can use
/// it from several threads.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    /// The single-byte token of each byte value.
    byte_tokens: [u32; 256],
    merges: MergeMap,
    vocab_size: usize,
}

/// The lookup state of one byte string of the vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    One(u32),
    Many,
}

impl Tokenizer {
    /// Makes a tokenizer from the bytes of each token (in token-id order), the
    /// `<|endoftext|>` token (if the vocabulary has one), and the BPE merges in rank order
    /// (rank 0 first).
    ///
    /// The `<|endoftext|>` token and tokens with no bytes are never in the output. Each of the
    /// other byte strings maps to its token. For each merge, the joined token is the one token
    /// whose bytes are the bytes of the left part, then the bytes of the right part.
    ///
    /// Cost: one pass over the vocabulary into a hash map from bytes to ids, then one look-up
    /// for each merge. The map is freed at the end; the tokenizer keeps about 24 bytes for
    /// each merge.
    ///
    /// # Errors
    ///
    /// - [`TacoError::MissingByteToken`] if a byte value has no single-byte token, or two.
    /// - [`TacoError::InvalidMerge`] if a merge is not valid ([`MergeFault`]).
    /// - [`TacoError::VocabularyTooLarge`] if there are more than `u32::MAX` tokens.
    pub fn new<'v, I>(
        tokens: I,
        eot: Option<TokenId>,
        merges: &[(TokenId, TokenId)],
    ) -> Result<Self>
    where
        I: IntoIterator<Item = &'v [u8]>,
    {
        let mut owners: HashMap<&'v [u8], Owner> = HashMap::new();
        let mut parts: Vec<Option<&'v [u8]>> = Vec::new();
        for (index, bytes) in tokens.into_iter().enumerate() {
            let id = u32::try_from(index).map_err(|_| TacoError::VocabularyTooLarge)?;
            let usable = !bytes.is_empty() && eot.is_none_or(|e| e.get() != id);
            parts.push(usable.then_some(bytes));
            if usable {
                owners.entry(bytes).and_modify(|o| *o = Owner::Many).or_insert(Owner::One(id));
            }
        }
        let mut byte_tokens = [0u32; 256];
        for (b, slot) in (0u8..=255).zip(byte_tokens.iter_mut()) {
            match owners.get(&[b][..]) {
                Some(Owner::One(id)) => *slot = *id,
                _ => return Err(TacoError::MissingByteToken { byte: b }),
            }
        }
        let mut map = MergeMap::with_capacity(merges.len());
        let mut joined = Vec::new();
        for (index, &(left, right)) in merges.iter().enumerate() {
            let rank = u32::try_from(index).map_err(|_| TacoError::VocabularyTooLarge)?;
            let fault = |fault| TacoError::InvalidMerge { rank, fault };
            let part = |t: TokenId| -> Result<&'v [u8]> {
                let slot = usize::try_from(t.get()).ok().and_then(|i| parts.get(i));
                match slot {
                    None => Err(fault(MergeFault::TokenOutOfRange)),
                    Some(None) => Err(fault(MergeFault::NotAByteToken)),
                    Some(Some(bytes)) => Ok(bytes),
                }
            };
            joined.clear();
            joined.extend_from_slice(part(left)?);
            joined.extend_from_slice(part(right)?);
            let merged = match owners.get(joined.as_slice()) {
                Some(Owner::One(id)) => *id,
                Some(Owner::Many) => return Err(fault(MergeFault::AmbiguousJoinedToken)),
                None => return Err(fault(MergeFault::NoJoinedToken)),
            };
            if map.insert((left.get(), right.get()), Merge { rank, merged }).is_some() {
                return Err(fault(MergeFault::Duplicate));
            }
        }
        Ok(Self { byte_tokens, merges: map, vocab_size: parts.len() })
    }

    /// Makes a tokenizer for the vocabulary of a model file. See [`Tokenizer::new`].
    ///
    /// # Errors
    ///
    /// The errors of [`Tokenizer::new`].
    pub fn for_vocabulary(
        vocabulary: &Vocabulary,
        eot: TokenId,
        merges: &[(TokenId, TokenId)],
    ) -> Result<Self> {
        let count = u32::try_from(vocabulary.len()).map_err(|_| TacoError::VocabularyTooLarge)?;
        let tokens = (0..count).map(|i| vocabulary.token(TokenId::new(i)).unwrap_or(&[]));
        Self::new(tokens, Some(eot), merges)
    }

    /// The number of tokens of the vocabulary that the tokenizer was made for.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The number of merges.
    #[must_use]
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// The token ids of `data`. The concatenation of the bytes of the tokens is `data`.
    ///
    /// Cost: one pass over `data`, and `O(n log n)` for a piece of `n` bytes. It allocates
    /// the output and a few words for each byte of the longest piece.
    #[must_use]
    pub fn encode(&self, data: &[u8]) -> Vec<TokenId> {
        let mut out = Vec::new();
        self.encode_into(data, &mut out);
        out
    }

    /// Appends the token ids of `data` to `out`. See [`Tokenizer::encode`].
    pub fn encode_into(&self, data: &[u8], out: &mut Vec<TokenId>) {
        let mut ids: Vec<u32> = Vec::new();
        let mut symbols: Vec<u32> = Vec::new();
        for chunk in data.utf8_chunks() {
            for piece in Pieces::new(chunk.valid()) {
                symbols.clear();
                symbols.extend(piece.bytes().map(|b| self.byte_token(b)));
                bpe::merge_into(&symbols, &self.merges, &mut ids);
            }
            // NOTE: `utf8_chunks` and the UTF8_RUN regex of the reference give the same runs:
            // an invalid part starts with a byte that cannot start a valid sequence there, and
            // the reference gives each byte outside a run its single-byte token.
            ids.extend(chunk.invalid().iter().map(|&b| self.byte_token(b)));
        }
        out.extend(ids.into_iter().map(TokenId::new));
    }

    fn byte_token(&self, b: u8) -> u32 {
        self.byte_tokens.get(usize::from(b)).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_vocab(extra: &[&[u8]]) -> Vec<Vec<u8>> {
        let mut v: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        v.extend(extra.iter().map(|e| e.to_vec()));
        v
    }

    fn tok(vocab: &[Vec<u8>], eot: Option<u32>, merges: &[(u32, u32)]) -> Result<Tokenizer> {
        let merges: Vec<_> =
            merges.iter().map(|&(a, b)| (TokenId::new(a), TokenId::new(b))).collect();
        Tokenizer::new(vocab.iter().map(Vec::as_slice), eot.map(TokenId::new), &merges)
    }

    #[test]
    fn merges_join_pieces() {
        let v = byte_vocab(&[b"th", b"the", b" the", b"<|endoftext|>"]);
        let t = tok(&v, Some(259), &[(116, 104), (256, 101), (32, 257)]).unwrap();
        let ids: Vec<u32> = t.encode(b"the  the").iter().map(|t| t.get()).collect();
        assert_eq!(ids, [257, 32, 258]);
    }

    #[test]
    fn refuses_bad_merges() {
        let v = byte_vocab(&[b"ab", b"<|endoftext|>"]);
        let err = |m: &[(u32, u32)]| match tok(&v, Some(257), m) {
            Err(TacoError::InvalidMerge { fault, .. }) => fault,
            other => panic!("{other:?}"),
        };
        assert_eq!(err(&[(97, 99)]), MergeFault::NoJoinedToken);
        assert_eq!(err(&[(97, 1000)]), MergeFault::TokenOutOfRange);
        assert_eq!(err(&[(97, 257)]), MergeFault::NotAByteToken);
        assert_eq!(err(&[(97, 98), (97, 98)]), MergeFault::Duplicate);
        let dup = byte_vocab(&[b"ab", b"ab"]);
        assert!(matches!(
            tok(&dup, None, &[(97, 98)]),
            Err(TacoError::InvalidMerge { fault: MergeFault::AmbiguousJoinedToken, .. })
        ));
    }

    #[test]
    fn refuses_a_vocabulary_without_all_bytes() {
        let mut v = byte_vocab(&[]);
        v[7] = b"xx".to_vec();
        assert!(matches!(tok(&v, None, &[]), Err(TacoError::MissingByteToken { byte: 7 })));
    }

    #[test]
    fn invalid_utf8_uses_single_byte_tokens() {
        let v = byte_vocab(&[]);
        let t = tok(&v, None, &[]).unwrap();
        let data = [0xE0, 0x41, 0xFF, 0xC3, 0xA9];
        let ids: Vec<u32> = t.encode(&data).iter().map(|t| t.get()).collect();
        assert_eq!(ids, [0xE0, 0x41, 0xFF, 0xC3, 0xA9]);
    }
}
