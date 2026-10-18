//! The stored bytes of a coded block (ADR 0007): the token count, then the range coder bytes.
//!
//! ```text
//! payload = varint(token count) || range coder output
//! ```
//!
//! The varint is unsigned LEB128: 7 bits in each byte, the low group first. Bit 7 is set on
//! every byte except the last. The encoding is minimal: the last byte is not 0, except for the
//! value 0 itself (one byte, 0x00). A value has at most 10 bytes. The reader refuses a
//! varint that is not minimal, that ends in the middle, or that does not fit in a `u64`.
//! Thus every count has exactly one encoding.
//!
//! The range coder output follows the varint directly. Its length is the rest of the payload.
//! The decoder of `taco-coder` reads zero bytes after its end (ADR 0003).

/// The largest number of bytes in a varint of a `u64`.
const MAX_VARINT_LEN: usize = 10;

/// Appends the minimal LEB128 encoding of `value` to `out`. Cost: at most 10 bytes.
pub(crate) fn write_varint(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let low = value.to_le_bytes()[0] & 0x7F;
        value >>= 7;
        if value == 0 {
            out.push(low);
            return;
        }
        out.push(low | 0x80);
    }
}

/// The number of bytes of the minimal LEB128 encoding of `value`: 1 to 10.
pub(crate) fn varint_len(mut value: u64) -> usize {
    let mut len = 1;
    while value > 0x7F {
        value >>= 7;
        len += 1;
    }
    len
}

/// Reads a minimal LEB128 varint from the start of `bytes`. Returns the value and the rest of
/// the bytes, or `None` if the varint is truncated, not minimal, or larger than `u64::MAX`.
pub(crate) fn read_varint(bytes: &[u8]) -> Option<(u64, &[u8])> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    for (i, &b) in bytes.iter().enumerate().take(MAX_VARINT_LEN) {
        let group = u64::from(b & 0x7F);
        // The tenth byte holds bit 63 only: a larger group does not fit in a u64.
        if i == MAX_VARINT_LEN - 1 && group > 1 {
            return None;
        }
        // INVARIANT: shift <= 63, because i < 10.
        value |= group << shift;
        shift += 7;
        if b & 0x80 == 0 {
            // Minimal: a last byte of 0 is permitted only as the first byte.
            if b == 0 && i > 0 {
                return None;
            }
            return Some((value, bytes.get(i + 1..).unwrap_or(&[])));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn known_encodings() {
        let enc = |v| {
            let mut o = Vec::new();
            write_varint(v, &mut o);
            o
        };
        assert_eq!(enc(0), [0x00]);
        assert_eq!(enc(1), [0x01]);
        assert_eq!(enc(127), [0x7F]);
        assert_eq!(enc(128), [0x80, 0x01]);
        assert_eq!(enc(16_384), [0x80, 0x80, 0x01]);
        assert_eq!(enc(u64::MAX).len(), 10);
        assert_eq!(*enc(u64::MAX).last().unwrap(), 0x01);
    }

    #[test]
    fn refuses_bad_varints() {
        assert_eq!(read_varint(&[]), None);
        assert_eq!(read_varint(&[0x80]), None);
        assert_eq!(read_varint(&[0x80, 0x00]), None);
        assert_eq!(read_varint(&[0xFF; 10]), None);
        let mut too_big = vec![0xFF; 9];
        too_big.push(0x02);
        assert_eq!(read_varint(&too_big), None);
        assert_eq!(read_varint(&[0xFF; 11]), None);
    }

    proptest! {
        #[test]
        fn varint_round_trip(v in any::<u64>(), tail in proptest::collection::vec(any::<u8>(), 0..8)) {
            let mut o = Vec::new();
            write_varint(v, &mut o);
            prop_assert_eq!(o.len(), varint_len(v));
            o.extend_from_slice(&tail);
            let (back, rest) = read_varint(&o).unwrap();
            prop_assert_eq!(back, v);
            prop_assert_eq!(rest, &tail[..]);
        }
    }
}
