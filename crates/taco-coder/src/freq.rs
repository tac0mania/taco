//! Frequency tables: validated integer frequencies with a total of exactly `2^freq_bits`.
//!
//! PLAN(§5.1): [`FreqTable::from_weights`] is the only conversion from model weights to
//! frequencies. It uses integer operations only, so every machine makes the same table.

use crate::CoderError;
use crate::Symbol;
use crate::cast::{low_u64, u32_bounded};

/// The number of bits of a table total: the total is `2^bits`, with `bits` in `0..=32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FreqBits(u32);

impl FreqBits {
    /// The largest number of bits. With range >= 2^56, `range / total` is then >= 2^24 (§5).
    pub const MAX: u32 = 32;

    /// Makes a bit count for a table total of `2^bits`.
    ///
    /// # Errors
    ///
    /// [`CoderError::FreqBitsTooLarge`] if `bits` is more than [`FreqBits::MAX`].
    #[inline]
    pub const fn new(bits: u32) -> Result<Self, CoderError> {
        if bits <= Self::MAX {
            Ok(Self(bits))
        } else {
            Err(CoderError::FreqBitsTooLarge { bits, max: Self::MAX })
        }
    }

    /// The number of bits.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The table total, `2^bits`. It is at most 2^32.
    #[must_use]
    #[inline]
    pub const fn total(self) -> u64 {
        // INVARIANT: self.0 <= 32, so the shift is in range and the result fits.
        1u64 << self.0
    }
}

/// The frequency of one symbol: at least 1, at most the table total (2^32 at most).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Freq(u64);

impl Freq {
    /// The frequency as an integer.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The part of the total that one symbol owns: `[start, start + freq)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Interval {
    /// The cumulative frequency of all lower symbols.
    pub(crate) start: u64,
    /// The frequency of the symbol.
    pub(crate) freq: u64,
    /// The symbol is the last symbol of the table. It receives the remainder of the range.
    pub(crate) is_last: bool,
}

/// A frequency table: one frequency of 1 or more for each symbol, with a total of exactly
/// `2^freq_bits`.
///
/// The table stores the cumulative start of each symbol as a `u32` (4 bytes for each symbol).
/// Each start is less than the total, so it fits. The decoder finds a symbol with a binary
/// search over the starts.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FreqTable {
    /// `starts[i]` is the sum of the frequencies of symbols `0..i`. `starts[0]` is 0.
    /// INVARIANT: not empty, strictly increasing, every value < total.
    starts: Vec<u32>,
    bits: FreqBits,
}

impl FreqTable {
    /// Converts non-negative model weights to a table with the total `T = 2^freq_bits` (§5.1).
    ///
    /// With `V` symbols and the weight sum `W`:
    ///
    /// 1. If `W` is 0, every weight is 1.
    /// 2. `f[i] = 1 + floor(w[i] * (T - V) / W)`, with the product and `W` in `u128`.
    /// 3. The remainder `T - sum(f)` goes to the symbol with the largest weight. If two symbols
    ///    have the same weight, the lower id receives it.
    ///
    /// Cost: three passes over the weights, with one `u128` multiplication and one `u128`
    /// division for each symbol. It allocates 4 bytes for each symbol.
    ///
    /// # Errors
    ///
    /// - [`CoderError::EmptyTable`] if `weights` is empty.
    /// - [`CoderError::TooManySymbols`] if `V > T`.
    pub fn from_weights(weights: &[u64], freq_bits: FreqBits) -> Result<Self, CoderError> {
        let total = freq_bits.total();
        let symbols = check_symbol_count(weights.len(), total)?;

        // Pass 1: the weight sum and the symbol with the largest weight (lowest id on a tie).
        // INVARIANT: V <= 2^32 and w < 2^64, so the sum is < 2^96 and fits in a u128.
        let mut weight_sum: u128 = 0;
        let mut largest = 0usize;
        let mut largest_weight = 0u64;
        for (i, &w) in weights.iter().enumerate() {
            weight_sum += u128::from(w);
            if w > largest_weight {
                largest = i;
                largest_weight = w;
            }
        }
        let all_zero = weight_sum == 0;
        if all_zero {
            // Rule 1: every weight is 1. Symbol 0 is then the largest (lowest id on a tie).
            weight_sum = u128::from(symbols);
        }

        // Pass 2: the frequencies before the remainder, as cumulative starts.
        // INVARIANT: symbols <= total (checked above).
        let spare = u128::from(total - symbols);
        let mut starts = Vec::with_capacity(weights.len());
        let mut acc: u64 = 0;
        for &w in weights {
            let w = if all_zero { 1 } else { u128::from(w) };
            // INVARIANT: w <= W, so q <= spare < 2^32 and the conversion is exact.
            let q = low_u64(w * spare / weight_sum);
            // INVARIANT: the sum of 1 + q over the symbols before this one is at most
            // (symbols before) + spare < total <= 2^32, so acc fits in a u32.
            starts.push(u32_bounded(acc));
            acc += 1 + q;
        }

        // Pass 3: the remainder goes to `largest`, so every later start moves up by it.
        // INVARIANT: acc <= V + spare = T, and the final starts stay < T.
        let remainder = u32_bounded(total - acc);
        if remainder > 0 {
            for start in starts.iter_mut().skip(largest + 1) {
                *start += remainder;
            }
        }
        Ok(Self { starts, bits: freq_bits })
    }

    /// Makes a table from explicit frequencies and checks them.
    ///
    /// Cost: two passes over the frequencies. It allocates 4 bytes for each symbol.
    ///
    /// # Errors
    ///
    /// - [`CoderError::EmptyTable`] if `freqs` is empty.
    /// - [`CoderError::TooManySymbols`] if there are more symbols than `2^freq_bits`.
    /// - [`CoderError::ZeroFrequency`] if a frequency is 0.
    /// - [`CoderError::WrongTotal`] if the sum is not exactly `2^freq_bits`.
    pub fn from_freqs(freqs: &[u64], freq_bits: FreqBits) -> Result<Self, CoderError> {
        let total = freq_bits.total();
        check_symbol_count(freqs.len(), total)?;
        // INVARIANT: at most 2^32 values below 2^64, so the sum fits in a u128.
        let mut sum: u128 = 0;
        for (symbol, &f) in freqs.iter().enumerate() {
            if f == 0 {
                return Err(CoderError::ZeroFrequency { symbol });
            }
            sum += u128::from(f);
        }
        if sum != u128::from(total) {
            return Err(CoderError::WrongTotal { expected: total, found: sum });
        }
        let mut starts = Vec::with_capacity(freqs.len());
        let mut acc: u64 = 0;
        for &f in freqs {
            // INVARIANT: every f >= 1 and the sum is T, so each start is < T <= 2^32.
            starts.push(u32_bounded(acc));
            acc += f;
        }
        Ok(Self { starts, bits: freq_bits })
    }

    /// The number of symbols. It is at least 1.
    #[must_use]
    #[inline]
    pub fn symbol_count(&self) -> usize {
        self.starts.len()
    }

    /// The number of bits of the total.
    #[must_use]
    #[inline]
    pub const fn freq_bits(&self) -> FreqBits {
        self.bits
    }

    /// The total of the frequencies, `2^freq_bits`.
    #[must_use]
    #[inline]
    pub const fn total(&self) -> u64 {
        self.bits.total()
    }

    /// The frequency of `symbol`, or `None` if the symbol is not in the table.
    #[must_use]
    pub fn freq(&self, symbol: Symbol) -> Option<Freq> {
        let i = usize::try_from(symbol.get()).ok()?;
        (i < self.starts.len()).then(|| Freq(self.interval(i).freq))
    }

    /// The frequencies of all symbols in id order. Cost: O(1) for each item.
    pub fn freqs(&self) -> impl ExactSizeIterator<Item = Freq> + '_ {
        (0..self.starts.len()).map(|i| Freq(self.interval(i).freq))
    }

    /// The interval of the symbol with index `i`, or `None` if the index is out of range.
    #[inline]
    pub(crate) fn interval_of(&self, symbol: Symbol) -> Option<Interval> {
        let i = usize::try_from(symbol.get()).ok()?;
        (i < self.starts.len()).then(|| self.interval(i))
    }

    /// The symbol whose interval holds `target`, and that interval. Cost: one binary search.
    ///
    /// The caller guarantees `target < total`.
    #[inline]
    pub(crate) fn find(&self, target: u64) -> (usize, Interval) {
        // INVARIANT: starts[0] == 0 <= target, so the partition point is >= 1 and the index
        // is in range.
        let i = self.starts.partition_point(|&s| u64::from(s) <= target).wrapping_sub(1);
        (i, self.interval(i))
    }

    /// The interval of index `i`. The caller guarantees `i < symbol_count()`.
    #[inline]
    fn interval(&self, i: usize) -> Interval {
        let start = u64::from(self.starts[i]);
        let (end, is_last) = match self.starts.get(i + 1) {
            Some(&next) => (u64::from(next), false),
            None => (self.total(), true),
        };
        Interval { start, freq: end - start, is_last }
    }
}

/// Checks `1 <= len <= total` and returns `len` as a `u64`.
fn check_symbol_count(len: usize, total: u64) -> Result<u64, CoderError> {
    if len == 0 {
        return Err(CoderError::EmptyTable);
    }
    match u64::try_from(len) {
        Ok(v) if v <= total => Ok(v),
        _ => Err(CoderError::TooManySymbols { symbols: len, total }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(b: u32) -> FreqBits {
        FreqBits::new(b).unwrap()
    }

    fn freqs_of(t: &FreqTable) -> Vec<u64> {
        t.freqs().map(Freq::get).collect()
    }

    #[test]
    fn from_weights_gives_remainder_to_largest_weight() {
        let t = FreqTable::from_weights(&[1, 2, 3, 4], bits(4)).unwrap();
        assert_eq!(freqs_of(&t), [2, 3, 4, 7]);
    }

    #[test]
    fn from_weights_gives_remainder_to_lowest_id_on_a_tie() {
        let t = FreqTable::from_weights(&[3, 3, 1], bits(3)).unwrap();
        assert_eq!(freqs_of(&t), [4, 3, 1]);
    }

    #[test]
    fn from_weights_treats_all_zero_weights_as_all_ones() {
        let t = FreqTable::from_weights(&[0, 0, 0], bits(2)).unwrap();
        assert_eq!(freqs_of(&t), [2, 1, 1]);
    }

    #[test]
    fn from_weights_gives_zero_weights_frequency_one() {
        let t = FreqTable::from_weights(&[0, u64::MAX, 0], bits(32)).unwrap();
        assert_eq!(freqs_of(&t), [1, (1 << 32) - 2, 1]);
    }

    #[test]
    fn from_weights_accepts_exactly_total_symbols() {
        let t = FreqTable::from_weights(&[9; 16], bits(4)).unwrap();
        assert_eq!(freqs_of(&t), [1; 16]);
    }

    #[test]
    fn from_weights_refuses_more_symbols_than_total() {
        assert_eq!(
            FreqTable::from_weights(&[1; 17], bits(4)),
            Err(CoderError::TooManySymbols { symbols: 17, total: 16 })
        );
    }

    #[test]
    fn from_weights_refuses_empty_weights() {
        assert_eq!(FreqTable::from_weights(&[], bits(8)), Err(CoderError::EmptyTable));
    }

    #[test]
    fn one_symbol_owns_the_full_total() {
        let t = FreqTable::from_weights(&[5], bits(32)).unwrap();
        assert_eq!(freqs_of(&t), [1 << 32]);
        let t = FreqTable::from_freqs(&[1], bits(0)).unwrap();
        assert_eq!(freqs_of(&t), [1]);
    }

    #[test]
    fn freq_bits_refuses_more_than_32() {
        assert_eq!(FreqBits::new(33), Err(CoderError::FreqBitsTooLarge { bits: 33, max: 32 }));
        assert_eq!(bits(32).total(), 1 << 32);
    }

    #[test]
    fn from_freqs_refuses_zero_and_wrong_totals() {
        assert_eq!(
            FreqTable::from_freqs(&[4, 0, 4], bits(3)),
            Err(CoderError::ZeroFrequency { symbol: 1 })
        );
        assert_eq!(
            FreqTable::from_freqs(&[4, 3], bits(3)),
            Err(CoderError::WrongTotal { expected: 8, found: 7 })
        );
        assert_eq!(
            FreqTable::from_freqs(&[u64::MAX, u64::MAX], bits(32)),
            Err(CoderError::WrongTotal { expected: 1 << 32, found: 2 * u128::from(u64::MAX) })
        );
        assert_eq!(FreqTable::from_freqs(&[], bits(3)), Err(CoderError::EmptyTable));
    }

    #[test]
    fn find_returns_the_symbol_that_owns_the_target() {
        let t = FreqTable::from_freqs(&[2, 1, 5], bits(3)).unwrap();
        let ids: Vec<usize> = (0..8).map(|x| t.find(x).0).collect();
        assert_eq!(ids, [0, 0, 1, 2, 2, 2, 2, 2]);
        assert!(t.find(7).1.is_last);
        assert_eq!(t.find(2).1, Interval { start: 2, freq: 1, is_last: false });
    }

    #[test]
    fn freq_of_unknown_symbol_is_none() {
        let t = FreqTable::from_freqs(&[2, 6], bits(3)).unwrap();
        assert_eq!(t.freq(Symbol::new(1)).map(Freq::get), Some(6));
        assert_eq!(t.freq(Symbol::new(2)), None);
    }
}
