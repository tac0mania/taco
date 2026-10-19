//! The range encoder: symbols and binary decisions in, bytes out.

use core::iter::repeat_n;

use crate::cast::{carry_bit, low_u64, top_byte};
use crate::{CodeInterval, CoderError, FreqTable, INITIAL_RANGE, Prob12, Symbol, TOP};

/// The range encoder. It writes bytes to a `Vec<u8>` and returns them from [`Encoder::finish`].
///
/// The exact algorithm is in the crate documentation. A [`Decoder`](crate::Decoder) that gets
/// the same calls with the same tables and probabilities returns the same symbols and bits.
#[derive(Debug, Clone)]
pub struct Encoder {
    /// Bits 0 to 63: the low end of the interval. Bit 64: a carry into the held bytes.
    /// INVARIANT: `low + range < 2^65`. After `shift_low`, `low < 2^64`.
    low: u128,
    /// The width of the interval. INVARIANT: `range >= 2^56` between calls.
    range: u64,
    /// The first held byte. It can still receive a carry.
    cache: u8,
    /// The number of held bytes: `cache` and then `pending - 1` bytes of 0xFF.
    /// It is 0 only before the first byte.
    pending: usize,
    out: Vec<u8>,
    /// The number of carries into the held bytes, for the carry tests.
    #[cfg(test)]
    carries: usize,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// Makes an encoder with an empty output.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Makes an encoder whose output can hold `bytes` bytes before it allocates again.
    #[must_use]
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            low: 0,
            range: INITIAL_RANGE,
            cache: 0,
            pending: 0,
            out: Vec::with_capacity(bytes),
            #[cfg(test)]
            carries: 0,
        }
    }

    /// Encodes `symbol` with `table`.
    ///
    /// Cost: one shift, two multiplications, and at most 4 output bytes (amortised: the
    /// information content of the symbol). It does no division.
    ///
    /// # Errors
    ///
    /// [`CoderError::SymbolOutOfRange`] if the symbol is not in the table. The encoder state
    /// does not change in that case.
    pub fn encode(&mut self, table: &FreqTable, symbol: Symbol) -> Result<(), CoderError> {
        let Some(iv) = table.interval_of(symbol) else {
            return Err(CoderError::SymbolOutOfRange {
                symbol: symbol.get(),
                symbols: table.symbol_count(),
            });
        };
        self.encode_parts(iv.start, iv.freq, iv.is_last, table.freq_bits().get());
        Ok(())
    }

    /// Encodes the symbol of `interval`. This gives the same bytes as [`Encoder::encode`] with
    /// the table whose interval it is: a symbol is the last one of its table exactly when
    /// `start + freq` is the total (see [`CodeInterval::is_last`]).
    ///
    /// Cost: as [`Encoder::encode`].
    pub fn encode_interval(&mut self, interval: CodeInterval) {
        self.encode_parts(
            interval.start(),
            interval.freq(),
            interval.is_last(),
            interval.freq_bits().get(),
        );
    }

    /// The symbol step of ADR 0003 for the interval `[start, start + freq)` of a total of
    /// `2^bits`.
    #[inline]
    fn encode_parts(&mut self, start: u64, freq: u64, is_last: bool, bits: u32) {
        // DETERMINISM: the total is 2^bits, so `range >> bits` is floor(range / total).
        let r = self.range >> bits;
        // INVARIANT: start < T and r * T <= range, so r * start < 2^64.
        let offset = r * start;
        self.low += u128::from(offset);
        // The last symbol takes the rest of the range. The decoder uses the same rule.
        // INVARIANT: for other symbols, start + freq < T, so r * freq < range.
        self.range = if is_last { self.range - offset } else { r * freq };
        self.renormalise();
    }

    /// Encodes one binary decision. `p1` is the probability that `bit` is `true`.
    ///
    /// Cost: one shift, one multiplication, and at most 2 output bytes. It does no division.
    pub fn encode_bit(&mut self, p1: Prob12, bit: bool) {
        // INVARIANT: p1 <= 4095, so bound < range; p1 >= 1 and range >= 2^56, so bound >= 2^44.
        let bound = (self.range >> Prob12::BITS) * u64::from(p1.get());
        if bit {
            self.range = bound;
        } else {
            self.low += u128::from(bound);
            self.range -= bound;
        }
        self.renormalise();
    }

    /// Ends the stream and returns the bytes.
    ///
    /// The flush selects the value in `[low, low + range)` with the most trailing zero bits
    /// (a multiple of 2^64 or of 2^56), writes the held bytes and the top byte of that value,
    /// and then removes all trailing zero bytes. The decoder reads zero bytes after the end, so
    /// the removed bytes cost nothing. An empty stream gives zero bytes.
    ///
    /// Cost: at most `pending + 1` bytes are written.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        let end = self.low + u128::from(self.range);
        for k in [64u32, 56] {
            let mask = (1u128 << k) - 1;
            let v = (self.low + mask) & !mask;
            // INVARIANT: range >= 2^56, so k = 56 always gives v < end.
            if v < end {
                self.low = v;
                break;
            }
        }
        self.shift_low();
        self.shift_low();
        while self.out.last() == Some(&0) {
            self.out.pop();
        }
        self.out
    }

    /// The number of bytes written so far. The held bytes are not counted.
    #[must_use]
    pub fn bytes_written(&self) -> usize {
        self.out.len()
    }

    /// Shifts bytes out until `range >= 2^56`. Each step moves one byte.
    #[inline]
    fn renormalise(&mut self) {
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// Moves the top byte of `low` into the held bytes, and writes held bytes that can no
    /// longer receive a carry.
    #[inline]
    fn shift_low(&mut self) {
        let carry = carry_bit(self.low);
        let window = low_u64(self.low);
        if self.pending == 0 || window < 0xFF00_0000_0000_0000 || carry != 0 {
            // NOTE: when pending == 0 there is no held byte. The interval is then still inside
            // [0, 2^64 - 1), so carry == 0 and no bit is lost.
            if self.pending > 0 {
                #[cfg(test)]
                {
                    self.carries += usize::from(carry);
                }
                // INVARIANT: a held cache byte is followed by values below (cache + 2) * 2^56
                // at its position, so cache + carry <= 0xFF (the LZMA carry argument).
                self.out.push(self.cache.wrapping_add(carry));
                // 0xFF + 1 is 0x00 with the carry going into the cache byte above.
                self.out.extend(repeat_n(0xFFu8.wrapping_add(carry), self.pending - 1));
            }
            self.cache = top_byte(window);
            self.pending = 0;
        }
        // INVARIANT: pending counts bytes in memory, so it cannot reach usize::MAX.
        self.pending += 1;
        self.low = u128::from(window << 8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Decoder, FreqBits};

    fn table(freqs: &[u64], bits: u32) -> FreqTable {
        FreqTable::from_freqs(freqs, FreqBits::new(bits).unwrap()).unwrap()
    }

    fn encode_all(t: &FreqTable, symbols: &[u32]) -> (Vec<u8>, usize) {
        let mut e = Encoder::new();
        for &s in symbols {
            e.encode(t, Symbol::new(s)).unwrap();
        }
        let carries = e.carries;
        (e.finish(), carries)
    }

    fn decode_all(t: &FreqTable, bytes: &[u8], n: usize) -> Vec<u32> {
        let mut d = Decoder::new(bytes);
        (0..n).map(|_| d.decode(t).get()).collect()
    }

    #[test]
    fn empty_stream_gives_no_bytes() {
        assert_eq!(Encoder::new().finish(), Vec::<u8>::new());
    }

    #[test]
    fn encode_refuses_symbol_outside_table() {
        let t = table(&[1, 1], 1);
        let mut e = Encoder::new();
        assert_eq!(
            e.encode(&t, Symbol::new(2)),
            Err(CoderError::SymbolOutOfRange { symbol: 2, symbols: 2 })
        );
        assert_eq!(e.finish(), Vec::<u8>::new());
    }

    #[test]
    fn carries_propagate_through_runs_of_ff() {
        // The last symbol sits at the top of the range, so low moves up to 0xFF.. bytes. The
        // first symbol then follows; together they make carries.
        let t = table(&[1, (1 << 32) - 2, 1], 32);
        let mut symbols = Vec::new();
        let mut x: u32 = 12345;
        for _ in 0..20_000 {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            symbols.push(if x >> 31 == 0 { 2 } else { 1 });
            if (x >> 20) & 0x3F == 0 {
                symbols.push(0);
            }
        }
        let (bytes, carries) = encode_all(&t, &symbols);
        assert!(carries > 0, "the sequence did not make a carry");
        assert_eq!(decode_all(&t, &bytes, symbols.len()), symbols);
    }

    #[test]
    fn long_run_of_most_probable_symbol_round_trips() {
        let t = table(&[(1 << 32) - 1, 1], 32);
        let symbols = vec![0u32; 200_000];
        let (bytes, _) = encode_all(&t, &symbols);
        // Each symbol costs about 3.4e-10 bits, so the output is nearly empty.
        assert!(bytes.len() <= 2, "{} bytes", bytes.len());
        assert_eq!(decode_all(&t, &bytes, symbols.len()), symbols);
    }

    #[test]
    fn long_run_of_least_probable_symbol_round_trips() {
        // The least probable symbol at the bottom and at the top (the last symbol) of the range.
        for (freqs, lps) in [([1, (1 << 32) - 1], 0u32), ([(1 << 32) - 1, 1], 1)] {
            let t = table(&freqs, 32);
            let symbols = vec![lps; 10_000];
            let (bytes, _) = encode_all(&t, &symbols);
            // Each symbol costs at most 32 bits plus 2^-24 of a symbol. The last symbol also
            // receives the rounding remainder, so it can cost less.
            assert!(bytes.len() <= 40_002, "{} bytes", bytes.len());
            assert_eq!(decode_all(&t, &bytes, symbols.len()), symbols);
        }
    }

    #[test]
    fn bits_at_extreme_probabilities_round_trip() {
        let mut e = Encoder::new();
        let mut want = Vec::new();
        for i in 0..50_000u32 {
            let (p, b) = match i % 4 {
                0 => (Prob12::MIN, true),
                1 => (Prob12::MAX, false),
                2 => (Prob12::MAX, true),
                _ => (Prob12::MIN, false),
            };
            e.encode_bit(p, b);
            want.push((p, b));
        }
        let bytes = e.finish();
        let mut d = Decoder::new(&bytes);
        for (p, b) in want {
            assert_eq!(d.decode_bit(p), b);
        }
    }

    #[test]
    fn single_symbol_table_costs_nothing() {
        let t = table(&[1 << 32], 32);
        let (bytes, _) = encode_all(&t, &[0; 1000]);
        assert!(bytes.is_empty());
        assert_eq!(decode_all(&t, &bytes, 1000), vec![0; 1000]);
        let t = table(&[1], 0);
        let (bytes, _) = encode_all(&t, &[0; 10]);
        assert!(bytes.is_empty());
    }

    #[test]
    fn output_never_ends_in_a_zero_byte() {
        let t = table(&[3, 1, 4], 3);
        for n in 0..200u32 {
            let symbols: Vec<u32> = (0..n).map(|i| (i * 7 + n) % 3).collect();
            let (bytes, _) = encode_all(&t, &symbols);
            assert_ne!(bytes.last(), Some(&0));
            assert_eq!(decode_all(&t, &bytes, symbols.len()), symbols);
        }
    }
}
