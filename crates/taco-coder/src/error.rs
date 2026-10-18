//! The error type of the crate: one variant for each fault that a caller can act on.

/// A fault in a frequency table, a probability, or a symbol that a caller gave to the coder.
///
/// Errors carry ids, counts and totals. They never carry the data that the caller codes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CoderError {
    /// The requested number of frequency bits is more than [`FreqBits::MAX`](crate::FreqBits::MAX).
    #[error("frequency bits {bits} is more than the maximum {max}")]
    FreqBitsTooLarge {
        /// The number of bits that the caller requested.
        bits: u32,
        /// The maximum number of bits.
        max: u32,
    },
    /// The weights or the frequencies have no symbols.
    #[error("the frequency table has no symbols")]
    EmptyTable,
    /// The table has more symbols than its total, so a symbol cannot have a frequency of 1.
    #[error(
        "{symbols} symbols do not fit in a total of {total}: each symbol needs a frequency of 1"
    )]
    TooManySymbols {
        /// The number of symbols that the caller gave.
        symbols: usize,
        /// The total of the table, `2^freq_bits`.
        total: u64,
    },
    /// A frequency is 0, so its symbol cannot be coded (plan §3 rule 5).
    #[error("symbol {symbol} has the frequency 0")]
    ZeroFrequency {
        /// The id of the symbol.
        symbol: usize,
    },
    /// The frequencies do not add up to exactly `2^freq_bits`.
    #[error("the frequencies add up to {found}, not to the total {expected}")]
    WrongTotal {
        /// The total of the table, `2^freq_bits`.
        expected: u64,
        /// The sum of the frequencies that the caller gave.
        found: u128,
    },
    /// The symbol is not in the table.
    #[error("symbol {symbol} is not in a table of {symbols} symbols")]
    SymbolOutOfRange {
        /// The id of the symbol.
        symbol: u32,
        /// The number of symbols in the table.
        symbols: usize,
    },
    /// The probability is not in `1..=4095`.
    #[error("the probability {p1} is not in 1..=4095")]
    ProbabilityOutOfRange {
        /// The probability that the caller gave, in units of 1/4096.
        p1: u16,
    },
}
