//! Property tests for `hardshell::int` against `i128` and float references.
//!
//! Tests may use floats (CLAUDE.md §1 rule 1 applies to `src/` only).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// Test helpers fail loudly.

// Reference values in tests convert between floats and integers on purpose.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]

use hardshell::int::{
    Divisor, Table, clamp, clamp_wide, floor_div, isqrt_u64, isqrt_u128, rescale, rescale_wide,
};
use proptest::prelude::*;

/// Exact reference: floor((x * m + 2^(s-1)) / 2^s), with big-integer style arithmetic in i128
/// and a floor division that does not use a shift.
fn rescale_reference(p: i128, s: u8) -> i128 {
    if s == 0 {
        return p;
    }
    if s >= 127 {
        // |p| < 2^126 here, so the exact result is 0 (see the src docs).
        return 0;
    }
    let d = 1i128 << s;
    let half = 1i128 << (s - 1);
    (p + half).div_euclid(d)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn rescale_matches_exact_reference(x in any::<i64>(), m in any::<i32>(), s in any::<u8>()) {
        let p = i128::from(x) * i128::from(m);
        prop_assert_eq!(rescale(x, m, s), rescale_reference(p, s));
    }

    #[test]
    fn rescale_wide_matches_exact_reference(
        hi in -(1i64 << 30)..(1i64 << 30), lo in any::<u64>(), m in any::<i32>(), s in any::<u8>()
    ) {
        // |x| < 2^94 < 2^95.
        let x = (i128::from(hi) << 64) | i128::from(lo);
        let p = x * i128::from(m);
        prop_assert_eq!(rescale_wide(x, m, s), rescale_reference(p, s));
    }

    #[test]
    fn rescale_is_within_half_of_real_quotient(
        x in -(1i64 << 40)..(1i64 << 40), m in -(1i32 << 20)..(1i32 << 20), s in 1u8..40
    ) {
        // Float reference: the exact quotient is representable to within 2^-12 here
        // (|x * m| < 2^60, f64 has 53 bits): the rounded result is within 0.5 (+ slack) of it.
        let q = (x as f64) * (m as f64) / 2f64.powi(i32::from(s));
        let r = rescale(x, m, s) as f64;
        prop_assert!((r - q) <= 0.5 + 1e-3 * q.abs().max(1.0) && (q - r) < 0.5 + 1e-3 * q.abs().max(1.0));
    }

    #[test]
    fn clamp_matches_min_max(x in any::<i64>(), a in any::<i64>(), b in any::<i64>()) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        prop_assert_eq!(clamp(x, lo, hi), x.max(lo).min(hi));
        prop_assert_eq!(clamp_wide(i128::from(x) << 3, lo, hi), (i128::from(x) << 3).clamp(i128::from(lo), i128::from(hi)) as i64);
    }

    #[test]
    fn floor_div_matches_float_floor(a in any::<i64>(), b in 1i64..i64::MAX) {
        let q = floor_div(a, Divisor::new(b).unwrap());
        // Exact check with i128: q * b <= a < (q + 1) * b.
        let (a2, b2, q2) = (i128::from(a), i128::from(b), i128::from(q));
        prop_assert!(q2 * b2 <= a2 && a2 < (q2 + 1) * b2);
    }

    #[test]
    fn isqrt_u64_is_floor_sqrt(x in any::<u64>()) {
        let r = isqrt_u64(x);
        prop_assert_eq!(r, x.isqrt());
        let (r2, x2) = (u128::from(r), u128::from(x));
        prop_assert!(r2 * r2 <= x2 && x2 < (r2 + 1) * (r2 + 1));
    }

    #[test]
    fn isqrt_u128_is_floor_sqrt(x in any::<u128>()) {
        let r = isqrt_u128(x);
        prop_assert_eq!(r, x.isqrt());
        prop_assert!(r * r <= x);
        // (r + 1)^2 can overflow u128 only for r = 2^64 - 1.
        if let Some(n) = (r + 1).checked_mul(r + 1) { prop_assert!(x < n); }
    }

    #[test]
    fn isqrt_matches_float_for_small_inputs(x in 0u64..(1u64 << 48)) {
        // For x < 2^48 the float square root, correctly rounded, has the same floor.
        prop_assert_eq!(isqrt_u64(x), (x as f64).sqrt().floor() as u64);
    }

    #[test]
    fn table_lookup_clamps(entries in proptest::collection::vec(any::<i32>(), 1..64), i in any::<i64>()) {
        let t = Table::new(entries.clone()).unwrap();
        let idx = i.clamp(0, (entries.len() - 1) as i64) as usize;
        prop_assert_eq!(t.lookup(i), entries[idx]);
    }
}
