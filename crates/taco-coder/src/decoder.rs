//! The range decoder: bytes in, symbols and binary decisions out.

use crate::cast::u32_from_usize_bounded;
use crate::{CodeInterval, CoderError, FreqBits, FreqTable, INITIAL_RANGE, Prob12, Symbol, TOP};

/// The range decoder. It reads from a byte slice, and reads zero bytes after its end.
///
/// The caller must make the same sequence of calls as the encoder, with the same tables and
/// probabilities, and must know the number of symbols (the container stores it). The input
/// must be the exact output of [`Encoder::finish`](crate::Encoder::finish), or that output
/// followed by zero bytes.
///
/// On other input the decoder returns arbitrary symbols. It does not panic and it does not
/// read out of bounds. The container checksum detects the wrong output (plan §3 rule 4).
#[derive(Debug, Clone)]
pub struct Decoder<'a> {
    input: &'a [u8],
    /// The index of the next byte in `input`.
    pos: usize,
    /// The coded value minus the encoder's `low`, in the same 64-bit window.
    /// INVARIANT: `code < range` for input from the encoder.
    code: u64,
    /// The width of the interval, equal to the encoder's `range` at every step.
    range: u64,
}

impl<'a> Decoder<'a> {
    /// Makes a decoder and reads the first 8 bytes (zero bytes after the end of `input`).
    #[must_use]
    pub fn new(input: &'a [u8]) -> Self {
        let mut d = Self { input, pos: 0, code: 0, range: INITIAL_RANGE };
        for _ in 0..8 {
            d.code = (d.code << 8) | u64::from(d.next_byte());
        }
        d
    }

    /// Decodes one symbol with `table`.
    ///
    /// Cost: one 64-bit division, one binary search over the table (about 16 steps for
    /// 50,257 symbols), and at most 4 input bytes.
    #[must_use]
    pub fn decode(&mut self, table: &FreqTable) -> Symbol {
        // DETERMINISM: the same division as the encoder: floor(range / 2^bits).
        let r = self.range >> table.freq_bits().get();
        // INVARIANT: range >= 2^56 and bits <= 32, so r >= 2^24 and the division is defined.
        // The clamp sends values above r * T to the last symbol, which owns the remainder.
        let target = (self.code / r).min(table.total() - 1);
        let (index, iv) = table.find(target);
        // INVARIANT: start <= target, so r * start <= r * target <= code.
        let offset = r * iv.start;
        self.code -= offset;
        self.range = if iv.is_last { self.range - offset } else { r * iv.freq };
        self.renormalise();
        // INVARIANT: index < symbol_count <= 2^32.
        Symbol::new(u32_from_usize_bounded(index))
    }

    /// The first half of a decode with a table of the total `2^bits` that the decoder does not
    /// hold (for example on a GPU): the target value `t = min(code / r, T - 1)` with
    /// `r = range >> bits`. The symbol is the one whose interval holds `t`. The state does not
    /// change. Then [`Decoder::decode_interval`] with that interval finishes the decode.
    ///
    /// Cost: one 64-bit division.
    #[must_use]
    pub fn target(&self, bits: FreqBits) -> u64 {
        // INVARIANT: range >= 2^56 and bits <= 32, so r >= 2^24 and the division is defined.
        let r = self.range >> bits.get();
        (self.code / r).min(bits.total() - 1)
    }

    /// The second half of a decode: takes the interval of the decoded symbol. Together with
    /// [`Decoder::target`], this gives the same state as [`Decoder::decode`] with the table of
    /// the interval.
    ///
    /// # Errors
    ///
    /// [`CoderError::TargetNotInInterval`] if the interval does not hold the target of
    /// [`Decoder::target`] for its total. The state does not change in that case.
    pub fn decode_interval(&mut self, interval: CodeInterval) -> Result<(), CoderError> {
        let bits = interval.freq_bits();
        let target = self.target(bits);
        if !interval.contains(target) {
            return Err(CoderError::TargetNotInInterval {
                target,
                start: interval.start(),
                freq: interval.freq(),
            });
        }
        let r = self.range >> bits.get();
        // INVARIANT: start <= target, so r * start <= r * target <= code.
        let offset = r * interval.start();
        self.code -= offset;
        self.range = if interval.is_last() { self.range - offset } else { r * interval.freq() };
        self.renormalise();
        Ok(())
    }

    /// Decodes one binary decision. `p1` is the probability of `true`.
    ///
    /// Cost: one shift, one multiplication, one comparison, and at most 2 input bytes.
    #[must_use]
    pub fn decode_bit(&mut self, p1: Prob12) -> bool {
        let bound = (self.range >> Prob12::BITS) * u64::from(p1.get());
        let bit = self.code < bound;
        if bit {
            self.range = bound;
        } else {
            self.code -= bound;
            self.range -= bound;
        }
        self.renormalise();
        bit
    }

    /// Reads bytes until `range >= 2^56`, as the encoder writes them.
    #[inline]
    fn renormalise(&mut self) {
        while self.range < TOP {
            self.range <<= 8;
            // NOTE: on input from the encoder, code < range < 2^56 here, so no bit is lost.
            // On other input, the shift drops top bits. That cannot panic.
            self.code = (self.code << 8) | u64::from(self.next_byte());
        }
    }

    /// The next input byte, or 0 after the end of the input.
    #[inline]
    fn next_byte(&mut self) -> u8 {
        match self.input.get(self.pos) {
            Some(&b) => {
                self.pos += 1;
                b
            }
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FreqBits;

    #[test]
    fn garbage_input_does_not_panic() {
        let t = FreqTable::from_freqs(&[1, 2, 5], FreqBits::new(3).unwrap()).unwrap();
        for input in [&[0xFFu8; 64][..], &[0u8; 3][..], &[][..], &[0x80, 0xFF, 0x00, 0x7F][..]] {
            let mut d = Decoder::new(input);
            for i in 0..1000u32 {
                let s = d.decode(&t);
                assert!(s.get() < 3);
                let _bit: bool =
                    d.decode_bit(Prob12::new(u16::try_from(i % 4095 + 1).unwrap()).unwrap());
            }
        }
    }

    #[test]
    fn empty_input_decodes_symbol_zero() {
        let t = FreqTable::from_freqs(&[5, 3], FreqBits::new(3).unwrap()).unwrap();
        let mut d = Decoder::new(&[]);
        assert_eq!(d.decode(&t), Symbol::new(0));
    }
}
