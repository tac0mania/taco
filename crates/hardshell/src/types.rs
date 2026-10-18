//! Newtypes for values with a meaning: token ids and model hashes.

use core::fmt;

/// The id of one token of the vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(u32);

impl TokenId {
    /// Makes a token id. A model checks it against its vocabulary size when it uses it.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The id as a number.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The SHA-256 of a full model file (plan §2, §8). It identifies the model and its tokenizer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelHash([u8; 32]);

impl ModelHash {
    /// Makes a model hash from its 32 bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The 32 bytes of the hash.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ModelHash {
    /// Lower-case hexadecimal, 64 characters.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for ModelHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelHash({self})")
    }
}
