//! The coding parts of a version 2 header (offsets 90 to 103): the parts of the prediction path
//! in addition to HardShell, and their parameters.
//!
//! | Bit of the coding parts (offset 90) | Part | Parameter bytes (offset 92 + i) |
//! |---|---|---|
//! | 0 | The match model and the mixer (ADR 0010 §5) | Byte 0: the mixer rule version (1 or 2, ADR 0017). |
//! | 1 | The context overlap (ADR 0015) | Bytes 2 and 3: `K`, a little-endian `u16`, 1 or more. |
//! | 2 to 15 | Reserved: a reader refuses them | |
//!
//! Byte 1 and bytes 4 to 11 of the parameters are reserved and 0. The parameter bytes of a part
//! that is not set are 0. Thus each set of parts has exactly one encoding.

use crate::error::{FormatError, Result};

/// The length of the coding parameters field of the version 2 header, in bytes.
pub(crate) const CODING_PARAMS_LEN: usize = 12;

/// The version of the rules and the constants of the match model and the mixer (ADR 0010
/// sections 2 and 3). Byte 0 of the coding parameters (offset 92) of a version 2 header holds
/// it when the coding parts have the match model and the mixer.
///
/// The rule version fixes every constant, so the header has no free parameter. A reader
/// refuses a rule version that it does not know.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MixerRule(u8);

impl MixerRule {
    /// Mixer rule version 1: ADR 0010 sections 2 and 3.
    pub const V1: Self = Self(1);

    /// Mixer rule version 2: rule version 1 and the skip rule (ADR 0017 §4 and §5). A skipped
    /// position codes the flag symbol with the table `[q_s, 2^16 - q_s]` and, after a miss, the
    /// token with `taco_match::skip_miss_weights`. The format reads it (frozen version 2); the
    /// writer of `taco` writes rule version 1 by default (an owner decision).
    pub const V2: Self = Self(2);

    /// The rule version of a byte, or `None` if this reader does not know the version.
    #[must_use]
    #[inline]
    pub const fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            _ => None,
        }
    }

    /// The byte of the rule version.
    #[must_use]
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self.0
    }
}

/// The context overlap `K` of a container (ADR 0015): the number of tokens that a context
/// restart of HardShell keeps. All blocks of a container use the same `K`.
///
/// [`ContextOverlap::NONE`] (`K = 0`) is the rule of ADR 0004 section 9 and of every container
/// before ADR 0015. The container does not know the context of the model, so it does not
/// check `K <= n_ctx - 2`. The decoder checks it with the model before the first block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContextOverlap(u16);

impl ContextOverlap {
    /// No overlap: a restart keeps no token.
    pub const NONE: Self = Self(0);

    /// The overlap of `k` tokens. `0` is [`ContextOverlap::NONE`].
    #[must_use]
    #[inline]
    pub const fn new(k: u16) -> Self {
        Self(k)
    }

    /// The number of tokens that a restart keeps.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u16 {
        self.0
    }

    /// `true` for [`ContextOverlap::NONE`].
    #[must_use]
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

/// The coding parts of a version 2 container (ADR 0010 §5, ADR 0011 §2, ADR 0015 §4): the
/// parts of the prediction path in addition to HardShell. The module documentation gives the
/// layout. A version 1 container has no coding parts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CodingParts {
    mixer: Option<MixerRule>,
    overlap: ContextOverlap,
}

impl CodingParts {
    /// No coding part: HardShell alone, with the restart rule of ADR 0004 section 9.
    pub const NONE: Self = Self { mixer: None, overlap: ContextOverlap::NONE };

    /// Bit 0 of the coding parts field: the match model and the mixer.
    pub const MIXER_BIT: u16 = 1;

    /// Bit 1 of the coding parts field: the context overlap.
    pub const OVERLAP_BIT: u16 = 2;

    /// The match model and the mixer with the rule version `rule`, and no context overlap.
    #[must_use]
    #[inline]
    pub const fn with_mixer(rule: MixerRule) -> Self {
        Self { mixer: Some(rule), overlap: ContextOverlap::NONE }
    }

    /// The same parts with the context overlap `overlap`. [`ContextOverlap::NONE`] clears the
    /// part.
    #[must_use]
    #[inline]
    pub const fn with_overlap(self, overlap: ContextOverlap) -> Self {
        Self { overlap, ..self }
    }

    /// The rule version of the match model and the mixer, or `None` if the container does not
    /// use them. A mixed block ([`crate::BlockKind::Mixed`]) is valid only if this is `Some`.
    #[must_use]
    #[inline]
    pub const fn mixer(&self) -> Option<MixerRule> {
        self.mixer
    }

    /// The context overlap of every block (ADR 0015). [`ContextOverlap::NONE`] if bit 1 is
    /// clear.
    #[must_use]
    #[inline]
    pub const fn overlap(&self) -> ContextOverlap {
        self.overlap
    }

    /// True if the container has no coding part.
    #[must_use]
    #[inline]
    pub const fn is_none(&self) -> bool {
        self.mixer.is_none() && self.overlap.is_none()
    }

    /// The value of the coding parts field (offset 90).
    #[must_use]
    #[inline]
    pub(crate) const fn bits(&self) -> u16 {
        let mixer = if self.mixer.is_some() { Self::MIXER_BIT } else { 0 };
        let overlap = if self.overlap.is_none() { 0 } else { Self::OVERLAP_BIT };
        mixer | overlap
    }

    /// The 12 bytes of the coding parameters field (offset 92).
    #[must_use]
    pub(crate) const fn params(&self) -> [u8; CODING_PARAMS_LEN] {
        let mut p = [0; CODING_PARAMS_LEN];
        if let Some(rule) = self.mixer {
            p[0] = rule.as_u8();
        }
        let [lo, hi] = self.overlap.get().to_le_bytes();
        p[2] = lo;
        p[3] = hi;
        p
    }

    /// Reads and checks the coding parts field and the coding parameters field.
    ///
    /// # Errors
    ///
    /// - [`FormatError::UnknownCodingParts`] if a bit other than bits 0 and 1 is set.
    /// - [`FormatError::UnknownMixerRule`] if bit 0 is set and byte 0 of the parameters is not
    ///   a known rule version.
    /// - [`FormatError::ZeroContextOverlap`] if bit 1 is set and bytes 2 and 3 are 0.
    /// - [`FormatError::ReservedNotZero`] if a parameter byte of a part that is not set is not
    ///   0, or if byte 1 or bytes 4 to 11 of the parameters are not 0.
    pub(crate) fn read(bits: u16, params: &[u8; CODING_PARAMS_LEN]) -> Result<Self> {
        if bits & !(Self::MIXER_BIT | Self::OVERLAP_BIT) != 0 {
            return Err(FormatError::UnknownCodingParts { parts: bits });
        }
        let reserved = FormatError::ReservedNotZero { field: "coding parameters" };
        let [rule, byte_1, lo, hi, rest @ ..] = *params;
        if byte_1 != 0 || rest != [0; CODING_PARAMS_LEN - 4] {
            return Err(reserved);
        }
        let mixer = if bits & Self::MIXER_BIT == 0 {
            if rule != 0 {
                return Err(reserved);
            }
            None
        } else {
            Some(MixerRule::from_u8(rule).ok_or(FormatError::UnknownMixerRule { rule })?)
        };
        let k = u16::from_le_bytes([lo, hi]);
        let overlap = match (bits & Self::OVERLAP_BIT != 0, k) {
            (false, 0) => ContextOverlap::NONE,
            (false, _) => return Err(reserved),
            (true, 0) => return Err(FormatError::ZeroContextOverlap),
            (true, k) => ContextOverlap::new(k),
        };
        Ok(Self { mixer, overlap })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_rules_1_and_2_are_the_only_rules() {
        for b in 0..=255u8 {
            let expected = match b {
                1 => Some(MixerRule::V1),
                2 => Some(MixerRule::V2),
                _ => None,
            };
            assert_eq!(MixerRule::from_u8(b), expected);
        }
        assert_eq!((MixerRule::V1.as_u8(), MixerRule::V2.as_u8()), (1, 2));
    }

    #[test]
    fn coding_parts_read_and_write_the_same_fields() {
        let mixer = CodingParts::with_mixer(MixerRule::V1);
        assert_eq!(mixer.bits(), 1);
        assert_eq!(mixer.params()[0], 1);
        assert_eq!(&mixer.params()[1..], &[0; 11]);
        assert_eq!(CodingParts::read(mixer.bits(), &mixer.params()), Ok(mixer));
        assert_eq!(CodingParts::NONE.bits(), 0);
        assert_eq!(CodingParts::NONE.params(), [0; 12]);
        assert_eq!(CodingParts::read(0, &[0; 12]), Ok(CodingParts::NONE));
        assert!(CodingParts::NONE.is_none() && !mixer.is_none());
    }

    #[test]
    fn overlap_is_bit_1_with_k_in_bytes_2_and_3() {
        for k in [1u16, 2, 12, 255, 256, 512, 1022, u16::MAX] {
            let o = CodingParts::NONE.with_overlap(ContextOverlap::new(k));
            assert_eq!(o.bits(), 2);
            let mut want = [0; 12];
            want[2..4].copy_from_slice(&k.to_le_bytes());
            assert_eq!(o.params(), want);
            assert_eq!(CodingParts::read(2, &want), Ok(o));
            assert_eq!(o.overlap().get(), k);
            assert!(!o.is_none() && o.mixer().is_none());
            // Both parts together.
            let both = CodingParts::with_mixer(MixerRule::V1).with_overlap(ContextOverlap::new(k));
            assert_eq!(both.bits(), 3);
            assert_eq!(both.params()[0], 1);
            assert_eq!(CodingParts::read(3, &both.params()), Ok(both));
        }
        // K = 512 is the bytes 00 02.
        let p = CodingParts::NONE.with_overlap(ContextOverlap::new(512)).params();
        assert_eq!(&p[2..4], &[0x00, 0x02]);
        // NONE clears the part.
        let cleared = CodingParts::NONE
            .with_overlap(ContextOverlap::new(5))
            .with_overlap(ContextOverlap::NONE);
        assert_eq!(cleared, CodingParts::NONE);
    }

    #[test]
    fn coding_parts_refuse_unknown_bits_rules_and_bytes() {
        let mut p = [0; 12];
        p[0] = 1;
        for bits in [4, 5, 7, 0x8000, u16::MAX] {
            assert_eq!(
                CodingParts::read(bits, &p),
                Err(FormatError::UnknownCodingParts { parts: bits })
            );
        }
        p[0] = 2;
        assert_eq!(CodingParts::read(1, &p).map(|c| c.mixer()), Ok(Some(MixerRule::V2)));
        for rule in [0, 3, 0xFF] {
            p[0] = rule;
            assert_eq!(CodingParts::read(1, &p), Err(FormatError::UnknownMixerRule { rule }));
        }
        // A rule version without the part, and a byte after the rule version.
        p[0] = 1;
        let reserved = Err(FormatError::ReservedNotZero { field: "coding parameters" });
        assert_eq!(CodingParts::read(0, &p), reserved);
        for i in [1, 4, 5, 6, 7, 8, 9, 10, 11] {
            let mut q = [0; 12];
            q[0] = 1;
            q[i] = 0x10;
            assert_eq!(CodingParts::read(1, &q), reserved);
            assert_eq!(
                CodingParts::read(3, &{
                    let mut r = q;
                    r[2] = 1;
                    r
                }),
                reserved
            );
        }
    }

    #[test]
    fn overlap_bytes_need_the_bit_and_the_bit_needs_k() {
        let reserved = Err(FormatError::ReservedNotZero { field: "coding parameters" });
        for (lo, hi) in [(1, 0), (0, 1), (0xFF, 0xFF)] {
            let mut p = [0; 12];
            p[2] = lo;
            p[3] = hi;
            assert_eq!(CodingParts::read(0, &p), reserved);
            p[0] = 1;
            assert_eq!(CodingParts::read(1, &p), reserved);
        }
        assert_eq!(CodingParts::read(2, &[0; 12]), Err(FormatError::ZeroContextOverlap));
        let mut p = [0; 12];
        p[0] = 1;
        assert_eq!(CodingParts::read(3, &p), Err(FormatError::ZeroContextOverlap));
    }
}
