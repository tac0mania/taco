//! Integer `squash` and `stretch` tables: the logistic function and its inverse.
//!
//! Probabilities have 16 bits: `p` in `1..=65535` means `p / 2^16`. The logistic domain `x` is
//! in units of 1/256 nat, in `-4095..=4095` (±16 nats).
//!
//! The tables are calculated with integers only, so every machine makes the same tables:
//!
//! 1. `E[0] = 2^62`, `E[j + 1] = (E[j] * C + 2^61) >> 62`, with `C = round(2^62 * exp(-1/256))`.
//!    Thus `E[j]` is `2^62 * exp(-j/256)`, rounded at each step.
//! 2. For `x >= 0`: `s(x) = (2^78 + d / 2) / d` with `d = 2^62 + E[x]` (round half up), and
//!    `s(-x) = 2^16 - s(x)`. `squash(x)` is `s(x)` clamped to `[1, 65535]`.
//! 3. `stretch(p)` is the smallest `x` with `squash(x) >= p`, or 4095 if no `x` has it.
//!
//! DETERMINISM: `u128` multiplication, shifts and one unsigned division for each entry.

/// The largest magnitude of the logistic domain, in units of 1/256 nat.
pub(crate) const X_MAX: i32 = 4095;
/// The probability 1 in units of 2^-16.
pub(crate) const P_ONE: u32 = 1 << 16;

/// `round(2^62 * exp(-1/256))`, calculated with 60-digit decimal arithmetic (Python
/// `decimal`, 2026-10-18): 4593706758521714574.2095...
const C_EXP: u128 = 4_593_706_758_521_714_574;

/// The squash and stretch tables (16 KiB and 128 KiB).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Logistic {
    /// `squash[x + X_MAX]` for `x` in `-X_MAX..=X_MAX`.
    squash: Vec<u16>,
    /// `stretch[p]` for `p` in `0..2^16`.
    stretch: Vec<i16>,
}

impl Logistic {
    /// Calculates both tables. Cost: about 4,096 `u128` divisions and one pass over 2^16
    /// entries.
    pub(crate) fn new() -> Self {
        let n = usize::try_from(X_MAX).unwrap_or(0);
        let mut e: Vec<u128> = Vec::with_capacity(n + 1);
        e.push(1u128 << 62);
        for j in 0..n {
            // INVARIANT: e[j] <= 2^62 and C < 2^62, so the product is < 2^124.
            let next = (e[j] * C_EXP + (1u128 << 61)) >> 62;
            e.push(next);
        }
        let mut squash = vec![0u16; 2 * n + 1];
        for (x, &ex) in e.iter().enumerate() {
            let d = (1u128 << 62) + ex;
            let s = ((1u128 << 78) + d / 2) / d;
            // INVARIANT: 2^15 <= s <= 2^16, so both values are in 0..=2^16 before the clamp.
            let s = u32::try_from(s).unwrap_or(P_ONE);
            squash[n + x] = clamp_p(s);
            squash[n - x] = clamp_p(P_ONE - s);
        }
        let mut stretch = vec![0i16; 1 << 16];
        let mut x = 0usize; // index into squash, that is x + X_MAX
        for (p, out) in stretch.iter_mut().enumerate() {
            while x < squash.len() && usize::from(squash[x]) < p {
                x += 1;
            }
            let v = i32::try_from(x.min(2 * n)).unwrap_or(0) - X_MAX;
            // INVARIANT: -4095 <= v <= 4095, so it fits in an i16.
            *out = i16::try_from(v).unwrap_or(0);
        }
        Self { squash, stretch }
    }

    /// `squash(x)` with `x` clamped to `-X_MAX..=X_MAX`. The result is in `1..=65535`.
    pub(crate) fn squash(&self, x: i64) -> u32 {
        let c = x.clamp(-i64::from(X_MAX), i64::from(X_MAX)) + i64::from(X_MAX);
        let i = usize::try_from(c).unwrap_or(0);
        self.squash.get(i).map_or(1, |&v| u32::from(v))
    }

    /// `stretch(p)` for `p` in `0..2^16` (larger values use 65535).
    pub(crate) fn stretch(&self, p: u32) -> i32 {
        let i = usize::try_from(p.min(P_ONE - 1)).unwrap_or(0);
        self.stretch.get(i).map_or(0, |&v| i32::from(v))
    }

    /// The squash table, for the golden test.
    #[cfg(test)]
    pub(crate) fn squash_table(&self) -> &[u16] {
        &self.squash
    }

    /// The stretch table, for the golden test.
    #[cfg(test)]
    pub(crate) fn stretch_table(&self) -> &[i16] {
        &self.stretch
    }
}

fn clamp_p(p: u32) -> u16 {
    // INVARIANT: the result is in 1..=65535, so it fits in a u16.
    u16::try_from(p.clamp(1, P_ONE - 1)).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]

    use super::*;

    #[test]
    fn squash_is_symmetric_and_monotonic() {
        let t = Logistic::new();
        assert_eq!(t.squash(0), 32_768);
        for x in 1..=i64::from(X_MAX) {
            assert!(t.squash(x) >= t.squash(x - 1));
            if t.squash(x) < P_ONE - 1 {
                assert_eq!(t.squash(x) + t.squash(-x), P_ONE, "x {x}");
            }
        }
        assert_eq!(t.squash(i64::from(X_MAX)), P_ONE - 1);
        assert_eq!(t.squash(-i64::from(X_MAX)), 1);
        assert_eq!(t.squash(i64::MAX), P_ONE - 1);
    }

    #[test]
    fn stretch_inverts_squash() {
        let t = Logistic::new();
        for p in 1..P_ONE {
            let x = t.stretch(p);
            assert!(t.squash(i64::from(x)) >= p, "p {p}");
            if x > -X_MAX {
                assert!(t.squash(i64::from(x) - 1) < p, "p {p}");
            }
        }
        assert_eq!(t.stretch(32_768), 0);
    }

    #[test]
    fn squash_is_near_the_logistic_function() {
        // Reference points from the published values of 1 / (1 + exp(-x)):
        // x = 1 nat: 0.7310585786, x = 4 nats: 0.9820137900.
        let t = Logistic::new();
        let one = t.squash(256);
        assert!((47_910..=47_912).contains(&one), "{one}");
        let four = t.squash(1024);
        assert!((64_357..=64_359).contains(&four), "{four}");
        assert_eq!(t.squash_table().len(), 8191);
    }

    /// Golden vector: the SHA-256 of both tables (little-endian). `tools/logs/vectors.py`
    /// (the Python mirror) gave the same two values on 2026-10-18 (measured).
    #[test]
    fn tables_match_the_python_mirror() {
        use sha2::{Digest, Sha256};
        let t = Logistic::new();
        let sq: Vec<u8> = t.squash_table().iter().flat_map(|v| v.to_le_bytes()).collect();
        let st: Vec<u8> = t.stretch_table().iter().flat_map(|v| v.to_le_bytes()).collect();
        let hex =
            |b: &[u8]| -> String { Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect() };
        assert_eq!(hex(&sq), "345a564594a6032f8b2f13850e4b38ce420a8c50c4953f2cffd244e9d29781bc");
        assert_eq!(hex(&st), "c1ed057a2aed7ea9f38e6cce27eaee39ccee69719aba90c0bb39ba5d02aad142");
    }
}
