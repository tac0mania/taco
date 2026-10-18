//! Narrowing integer conversions whose bound the caller proves.
//!
//! Each function here truncates on purpose. The caller writes an `INVARIANT:` comment that
//! shows the value fits, or the function masks the value first.

/// The lower 64 bits of `x`. This drops the carry bit of the encoder's `low`.
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "the truncation to 64 bits is the purpose")]
pub(crate) const fn low_u64(x: u128) -> u64 {
    x as u64
}

/// Bit 64 of `x`, as 0 or 1. The mask makes the cast exact.
#[inline]
pub(crate) const fn carry_bit(x: u128) -> u8 {
    ((x >> 64) & 1) as u8
}

/// The top byte (bits 56 to 63) of `x`. The shift leaves 8 bits, so the cast is exact.
#[inline]
pub(crate) const fn top_byte(x: u64) -> u8 {
    (x >> 56) as u8
}

/// `x` as a `u32`. The caller guarantees that `x < 2^32`.
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "the caller proves x < 2^32")]
pub(crate) const fn u32_bounded(x: u64) -> u32 {
    x as u32
}

/// `x` as a `u32`. The caller guarantees that `x < 2^32`.
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "the caller proves x < 2^32")]
pub(crate) const fn u32_from_usize_bounded(x: usize) -> u32 {
    x as u32
}
