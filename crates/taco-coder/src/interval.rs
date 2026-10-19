//! A coder interval that comes from outside a [`FreqTable`]: for example from a GPU coder op
//! that makes the table of [`FreqTable::from_weights`] on the device and gives back only the
//! interval of one symbol.
//!
//! PLAN(§5): the range coder needs three values for each symbol: the start `c[s]`, the
//! frequency `f[s]` and the total `T = 2^b`. The rule "the last symbol receives the remainder"
//! needs a fourth fact, "is `s` the last symbol". In a valid table every start is less than
//! `T`, so a symbol is the last one exactly when `c[s] + f[s] = T`. Thus the three values are
//! sufficient, and [`Encoder::encode_interval`](crate::Encoder::encode_interval) gives the same
//! bytes as [`Encoder::encode`](crate::Encoder::encode) with the table of the interval (tests
//! in `tests/interval.rs` prove it).

use crate::{CoderError, FreqBits, FreqTable, Symbol};

/// The interval `[start, start + freq)` of one symbol in a table with the total `2^bits`.
///
/// INVARIANT: `freq >= 1` and `start + freq <= 2^bits`. The constructor checks both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CodeInterval {
    start: u64,
    freq: u64,
    bits: FreqBits,
}

impl CodeInterval {
    /// Makes the interval `[start, start + freq)` of a table with the total `2^bits`.
    ///
    /// # Errors
    ///
    /// [`CoderError::InvalidInterval`] if `freq` is 0 or `start + freq` is more than the total.
    pub const fn new(start: u64, freq: u64, bits: FreqBits) -> Result<Self, CoderError> {
        let total = bits.total();
        // INVARIANT: the first test gives start <= total, so total - start does not wrap.
        if freq == 0 || start > total || freq > total - start {
            return Err(CoderError::InvalidInterval { start, freq, total });
        }
        Ok(Self { start, freq, bits })
    }

    /// The sum of the frequencies of the lower symbols.
    #[must_use]
    #[inline]
    pub const fn start(self) -> u64 {
        self.start
    }

    /// The frequency of the symbol. It is at least 1.
    #[must_use]
    #[inline]
    pub const fn freq(self) -> u64 {
        self.freq
    }

    /// The number of bits of the table total.
    #[must_use]
    #[inline]
    pub const fn freq_bits(self) -> FreqBits {
        self.bits
    }

    /// `true` if the symbol is the last symbol of its table (`start + freq = 2^bits`). The
    /// last symbol receives the remainder of the range (ADR 0003).
    #[must_use]
    #[inline]
    pub const fn is_last(self) -> bool {
        // INVARIANT: start + freq <= total <= 2^32 (constructor), so the sum does not overflow.
        self.start + self.freq == self.bits.total()
    }

    /// `true` if `target` is in `[start, start + freq)`.
    #[must_use]
    #[inline]
    pub const fn contains(self, target: u64) -> bool {
        target >= self.start && target - self.start < self.freq
    }
}

impl FreqTable {
    /// The interval of `symbol`, or `None` if the symbol is not in the table. Encoding this
    /// interval gives the same bytes as encoding the symbol with the table.
    #[must_use]
    pub fn code_interval(&self, symbol: Symbol) -> Option<CodeInterval> {
        let iv = self.interval_of(symbol)?;
        Some(CodeInterval { start: iv.start, freq: iv.freq, bits: self.freq_bits() })
    }

    /// The symbol whose interval holds `target`, and that interval, as the decoder finds it.
    /// `None` if `target` is not less than the total.
    #[must_use]
    pub fn symbol_at(&self, target: u64) -> Option<(Symbol, CodeInterval)> {
        if target >= self.total() {
            return None;
        }
        let (i, iv) = self.find(target);
        let symbol = Symbol::new(u32::try_from(i).ok()?);
        Some((symbol, CodeInterval { start: iv.start, freq: iv.freq, bits: self.freq_bits() }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(b: u32) -> FreqBits {
        FreqBits::new(b).unwrap()
    }

    #[test]
    fn new_refuses_zero_frequency_and_intervals_past_the_total() {
        assert_eq!(
            CodeInterval::new(3, 0, bits(3)),
            Err(CoderError::InvalidInterval { start: 3, freq: 0, total: 8 })
        );
        assert_eq!(
            CodeInterval::new(7, 2, bits(3)),
            Err(CoderError::InvalidInterval { start: 7, freq: 2, total: 8 })
        );
        assert_eq!(
            CodeInterval::new(9, 1, bits(3)),
            Err(CoderError::InvalidInterval { start: 9, freq: 1, total: 8 })
        );
        assert_eq!(
            CodeInterval::new(u64::MAX, u64::MAX, bits(32)),
            Err(CoderError::InvalidInterval { start: u64::MAX, freq: u64::MAX, total: 1 << 32 })
        );
        assert!(CodeInterval::new(0, 1 << 32, bits(32)).unwrap().is_last());
        assert!(!CodeInterval::new(0, 7, bits(3)).unwrap().is_last());
    }

    #[test]
    fn table_intervals_agree_with_the_table() {
        let t = FreqTable::from_freqs(&[2, 1, 5], bits(3)).unwrap();
        let iv = t.code_interval(Symbol::new(2)).unwrap();
        assert_eq!((iv.start(), iv.freq(), iv.is_last()), (3, 5, true));
        assert!(!t.code_interval(Symbol::new(1)).unwrap().is_last());
        assert_eq!(t.code_interval(Symbol::new(3)), None);
        for target in 0..8 {
            let (s, iv) = t.symbol_at(target).unwrap();
            assert!(iv.contains(target));
            assert_eq!(t.code_interval(s), Some(iv));
        }
        assert_eq!(t.symbol_at(8), None);
    }
}
