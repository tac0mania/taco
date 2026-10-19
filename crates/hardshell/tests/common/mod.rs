//! Shared test code: a seeded integer PRNG, float helpers that give the same bits on every
//! machine, and the converter rules (rescale constants and tables) that a tool uses.
//!
//! NOTE: the golden test builds its model with this code, and the model must be the same on
//! every machine. So the float helpers here use only `+`, `-`, `*`, `/`, `sqrt` and `round`,
//! which IEEE 754 defines exactly. They never call `exp`, `tanh`, `ln` or `powi` from the
//! platform math library, because those results can differ in the last bit (ADR 0002).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(dead_code, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

pub(crate) mod builder;
pub(crate) mod reference;

/// splitmix64: a seeded integer PRNG with one defined output sequence.
#[derive(Clone, Debug)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform value in [0, 1) with 53 random bits (exact conversion).
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// An approximately normal value with mean 0 and variance 1 (Irwin-Hall, 12 uniforms).
    pub(crate) fn normal(&mut self) -> f64 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.uniform();
        }
        s - 6.0
    }

    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// 2^k, exact, for -1022 <= k <= 1023.
pub(crate) fn pow2(k: i32) -> f64 {
    f64::from_bits(((1023 + k) as u64) << 52)
}

const LN2: f64 = std::f64::consts::LN_2;

/// exp(x) with basic operations only: range reduction by ln 2, then a Taylor series.
/// The relative error is about 1e-16 (measured against `f64::exp` in `det_exp_is_accurate`).
pub(crate) fn det_exp(x: f64) -> f64 {
    if x < -700.0 {
        return 0.0;
    }
    if x > 700.0 {
        return f64::INFINITY;
    }
    let k = (x / LN2).round();
    let r = x - k * LN2;
    // |r| <= 0.35: 22 terms are more than enough for f64.
    let mut term = 1.0;
    let mut sum = 1.0;
    for i in 1..22 {
        term = term * r / f64::from(i);
        sum += term;
    }
    sum * pow2(k as i32)
}

/// tanh with basic operations only.
pub(crate) fn det_tanh(x: f64) -> f64 {
    if x > 20.0 {
        return 1.0;
    }
    if x < -20.0 {
        return -1.0;
    }
    let e = det_exp(2.0 * x);
    (e - 1.0) / (e + 1.0)
}

/// GPT-2 GELU (the tanh form, "gelu_new").
pub(crate) fn gelu(x: f64) -> f64 {
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2 / pi)
    0.5 * x * (1.0 + det_tanh(C * (x + 0.044_715 * x * x * x)))
}

/// The rescale constants (m, s) for one positive ratio r: m = round(r * 2^s) with
/// 2^30 <= m < 2^31, and the largest such s up to 126.
pub(crate) fn quant_mult(r: f64) -> (i32, u8) {
    let (m, s) = quant_mult_rows(&[r]);
    (m[0], s)
}

/// Rescale constants for a list of positive ratios with one shared shift: s is chosen for the
/// largest ratio, and each m = round(r * 2^s).
pub(crate) fn quant_mult_rows(rs: &[f64]) -> (Vec<i32>, u8) {
    let max = rs.iter().copied().fold(0.0, f64::max);
    assert!(max > 0.0 && max < 2_147_483_647.0, "ratio out of range: {max}");
    let mut s: i32 = 0;
    while s < 126 && max * pow2(s + 1) < 2_147_483_647.0 {
        s += 1;
    }
    // Here max * 2^s is in [2^30, 2^31) unless s hit 126.
    let m = rs.iter().map(|&r| (r * pow2(s)).round() as i32).collect();
    (m, s as u8)
}

/// table[i] = round(2^e * exp(-i * delta)), as i32.
pub(crate) fn exp_table_i32(e: i32, delta: f64, len: usize) -> Vec<i32> {
    (0..len).map(|i| (pow2(e) * det_exp(-(i as f64) * delta)).round() as i32).collect()
}

/// table[i] = round(2^e * exp(-i * delta)), as u64.
pub(crate) fn exp_table_u64(e: i32, delta: f64, len: usize) -> Vec<u64> {
    (0..len).map(|i| (pow2(e) * det_exp(-(i as f64) * delta)).round() as u64).collect()
}

/// A GELU table for inputs of `bits` bits at scale s_in, with i8 outputs at scale s_out.
/// A GELU table with `i16` outputs (hsm format version 2): the rule of [`gelu_table`] with the
/// clamp at 32,767.
pub(crate) fn gelu_table_i16(bits: u32, s_in: f64, s_out: f64) -> Vec<i16> {
    let half = 1i64 << (bits - 1);
    (-half..half)
        .map(|q| (gelu(q as f64 * s_in) / s_out).round().clamp(-32767.0, 32767.0) as i16)
        .collect()
}

pub(crate) fn gelu_table(bits: u32, s_in: f64, s_out: f64) -> Vec<i8> {
    let half = 1i64 << (bits - 1);
    (-half..half)
        .map(|q| (gelu(q as f64 * s_in) / s_out).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Float softmax.
pub(crate) fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = x.iter().map(|&v| det_exp(v - max)).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}

/// KL(p || q) in bits. Entries with p = 0 add nothing.
pub(crate) fn kl_bits(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).filter(|(pi, _)| **pi > 0.0).map(|(&pi, &qi)| pi * (pi / qi).ln()).sum::<f64>()
        / LN2
}

#[test]
fn det_exp_is_accurate() {
    let mut worst: f64 = 0.0;
    for i in -4000..4000 {
        let x = f64::from(i) * 0.01;
        let rel = (det_exp(x) - x.exp()).abs() / x.exp();
        worst = worst.max(rel);
    }
    assert!(worst < 1e-14, "det_exp relative error {worst}");
}
