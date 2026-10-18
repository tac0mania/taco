//! The BPE merges of the encoder: an optional section of the `.hsm` file.
//!
//! The section holds one pair of token ids for each merge, in rank order (rank 0 first). Each
//! pair is two `u32` values, little-endian: `left`, then `right`. Only the encoder (the
//! tokenizer) uses the merges. The decoder changes token ids to bytes with the vocabulary and
//! does not read this section. The model hash covers the full file, so it covers the merges too.

use std::collections::{BTreeMap, BTreeSet};

use super::reader::Vocabulary;
use crate::error::{Error, Result};
use crate::types::TokenId;

/// The length of one merge in the file, in bytes: two `u32` values.
pub const MERGE_PAIR_LEN: u64 = 8;

/// One BPE merge: the encoder can join the token `left` and the token `right` into the token
/// `merged`. The bytes of `merged` are the bytes of `left`, then the bytes of `right`.
///
/// The reader calculates `merged` from the vocabulary; the file stores only `left` and `right`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MergePair {
    left: TokenId,
    right: TokenId,
    merged: TokenId,
}

impl MergePair {
    /// The first token of the pair.
    #[must_use]
    pub const fn left(self) -> TokenId {
        self.left
    }

    /// The second token of the pair.
    #[must_use]
    pub const fn right(self) -> TokenId {
        self.right
    }

    /// The token whose bytes are the bytes of `left`, then the bytes of `right`.
    #[must_use]
    pub const fn merged(self) -> TokenId {
        self.merged
    }
}

fn bad(rank: usize, reason: &'static str) -> Error {
    Error::InvalidHeader { field: format!("merges[{rank}]"), reason }
}

/// The first 4 bytes of `b` as a little-endian `u32` (the caller gives 4 bytes).
fn le_u32(b: &[u8]) -> u32 {
    let mut a = [0u8; 4];
    for (d, s) in a.iter_mut().zip(b) {
        *d = *s;
    }
    u32::from_le_bytes(a)
}

/// Parses and validates the merges section against the vocabulary.
///
/// The rules: the section has at least one pair and a length that is a multiple of 8; each id
/// is less than the vocabulary size; each part has at least one byte; no pair occurs two
/// times; and the joined bytes of each pair are the bytes of exactly one token.
///
/// Cost: one pass over the vocabulary to make a map from bytes to ids, then one look-up for
/// each merge (`O(n log n)`).
pub(crate) fn parse(section: &[u8], vocab: &Vocabulary) -> Result<Box<[MergePair]>> {
    let len = u64::try_from(section.len()).unwrap_or(u64::MAX);
    if len == 0 || len % MERGE_PAIR_LEN != 0 {
        return Err(Error::InvalidHeader {
            field: "merges.length".into(),
            reason: "must be a multiple of 8 and at least 8 (omit the section for no merges)",
        });
    }
    // NOTE: a BTreeMap, not a HashMap, so that the kernel crate uses no random state. A byte
    // string that two tokens share maps to None: a merge cannot select one of them.
    let mut by_bytes: BTreeMap<&[u8], Option<TokenId>> = BTreeMap::new();
    for i in 0..vocab.len() {
        let id = TokenId::new(u32::try_from(i).unwrap_or(u32::MAX));
        if let Some(b) = vocab.token(id) {
            by_bytes.entry(b).and_modify(|e| *e = None).or_insert(Some(id));
        }
    }
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(section.len() / 8);
    let mut joined = Vec::new();
    let (pairs, _) = section.as_chunks::<8>();
    for (rank, chunk) in pairs.iter().enumerate() {
        let (l, r) = chunk.split_at(4);
        let left = TokenId::new(le_u32(l));
        let right = TokenId::new(le_u32(r));
        let (Some(lb), Some(rb)) = (vocab.token(left), vocab.token(right)) else {
            return Err(bad(rank, "a token id is not less than the vocabulary size"));
        };
        if lb.is_empty() || rb.is_empty() {
            return Err(bad(rank, "a part of the merge has no bytes"));
        }
        if !seen.insert((left, right)) {
            return Err(bad(rank, "the pair occurs two times"));
        }
        joined.clear();
        joined.extend_from_slice(lb);
        joined.extend_from_slice(rb);
        let merged = match by_bytes.get(joined.as_slice()) {
            Some(Some(id)) => *id,
            Some(None) => return Err(bad(rank, "two tokens have the joined bytes")),
            None => return Err(bad(rank, "the joined bytes are not a token")),
        };
        out.push(MergePair { left, right, merged });
    }
    Ok(out.into_boxed_slice())
}

/// Encodes merge pairs in the format of the section. Cost: one pass.
pub(crate) fn encode(pairs: &[(TokenId, TokenId)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pairs.len() * 8);
    for (l, r) in pairs {
        out.extend_from_slice(&l.get().to_le_bytes());
        out.extend_from_slice(&r.get().to_le_bytes());
    }
    out
}
