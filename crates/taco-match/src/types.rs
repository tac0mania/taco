//! Newtypes with a meaning: token ids, match lengths and predictions.

/// The id of one token of the model vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(u32);

impl TokenId {
    /// Makes a token id.
    #[must_use]
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The id as an integer.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The length of a match in tokens: the number of tokens before the predicted token that are
/// equal to the tokens before the current position. It is at least 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MatchLen(u32);

impl MatchLen {
    /// Makes a length. Returns `None` for 0.
    #[must_use]
    #[inline]
    pub const fn new(len: u32) -> Option<Self> {
        if len == 0 { None } else { Some(Self(len)) }
    }

    /// The length as an integer (at least 1).
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The prediction of the match model for the next token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Prediction {
    /// The token that followed the earlier occurrence of the context.
    pub token: TokenId,
    /// The match length: the confidence input of the mixer.
    pub length: MatchLen,
}
