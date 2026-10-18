//! Newtypes for the symbols and the binary probabilities that the coder takes and returns.

use crate::CoderError;

/// The id of a symbol in a [`FreqTable`](crate::FreqTable): `0..table.symbol_count()`.
///
/// The `taco` crate maps a token id to a symbol. The coder does not know about tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(u32);

impl Symbol {
    /// Makes a symbol from its id. A table checks the id when the coder uses it.
    #[must_use]
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The id of the symbol.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for Symbol {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

/// The probability that a binary decision is 1, in units of 1/4096 (12 bits).
///
/// The value is in `1..=4095`, so both outcomes can be coded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Prob12(u16);

impl Prob12 {
    /// The number of bits in the probability scale: the scale is `2^12 = 4096`.
    pub const BITS: u32 = 12;
    /// The probability 1/2.
    pub const HALF: Self = Self(2048);
    /// The smallest valid value, 1/4096.
    pub const MIN: Self = Self(1);
    /// The largest valid value, 4095/4096.
    pub const MAX: Self = Self(4095);

    /// Makes a probability of 1 from `p1 / 4096`.
    ///
    /// # Errors
    ///
    /// [`CoderError::ProbabilityOutOfRange`] if `p1` is 0 or more than 4095.
    #[inline]
    pub const fn new(p1: u16) -> Result<Self, CoderError> {
        if p1 >= 1 && p1 <= 4095 {
            Ok(Self(p1))
        } else {
            Err(CoderError::ProbabilityOutOfRange { p1 })
        }
    }

    /// The probability of 1, in units of 1/4096.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u16 {
        self.0
    }
}
