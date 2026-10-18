//! Measurements of the integer LayerNorm, GELU, softmax and logit weights against float
//! references. The numbers that these tests print are the source of the constants in ADR 0004.
//!
//! Run with `cargo test -p hardshell --test numerics -- --nocapture` to see the numbers.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss, clippy::print_stdout)]

mod common;

use common::{Rng, exp_table_i32, exp_table_u64, gelu, gelu_table, kl_bits, pow2, quant_mult};
use hardshell::gelu::GeluTable;
use hardshell::layernorm::{LayerNorm, LnConstants};
use hardshell::logits::{LogitTable, logits_to_weights};
use hardshell::softmax::{ExpTable, softmax};

/// The residual scale of these measurements: one unit is 2^-12 in real numbers.
const S_RES: f64 = 1.0 / 4096.0;
const EPS: f64 = 1e-5;

struct LnStats {
    max_err_lsb: f64,
    mismatch: f64,
}

/// Runs integer LayerNorm on rows with a standard deviation of `sigma` units, and compares the
/// result with a float LayerNorm that uses the same (quantised) gamma and beta.
fn measure_layernorm(k: u8, j: u8, sigma: f64, rows: usize, rng: &mut Rng) -> LnStats {
    let n = 768;
    let eps_q = ((EPS / (S_RES * S_RES)) * pow2(2 * i32::from(j))).round().max(1.0) as u64;
    let c = LnConstants::new(k, j, eps_q).unwrap();
    let mut max_err: f64 = 0.0;
    let mut mismatches = 0usize;
    for _ in 0..rows {
        let gamma: Vec<f64> = (0..n).map(|_| 1.0 + 0.3 * rng.normal()).collect();
        let s_gamma = gamma.iter().fold(0.0f64, |a, &g| a.max(g.abs())) / 32767.0;
        let gamma_q: Vec<i16> = gamma.iter().map(|&g| (g / s_gamma).round() as i16).collect();
        let offset = rng.normal() * sigma * 3.0;
        let mut x: Vec<i32> =
            (0..n).map(|_| (offset + sigma * rng.normal()).round() as i32).collect();
        // Two outlier channels, as in GPT-2 (published: a few residual channels are large).
        x[3] = (offset + 30.0 * sigma).round() as i32;
        x[400] = (offset - 25.0 * sigma).round() as i32;

        // Float reference with the quantised gamma (real units: x * S_RES).
        let xr: Vec<f64> = x.iter().map(|&v| f64::from(v) * S_RES).collect();
        let mean = xr.iter().sum::<f64>() / n as f64;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let inv = 1.0 / (var + EPS).sqrt();
        let normed: Vec<f64> = xr
            .iter()
            .zip(&gamma_q)
            .map(|(v, &g)| (v - mean) * inv * f64::from(g) * s_gamma)
            .collect();
        let s_out = normed.iter().fold(0.0f64, |a, v| a.max(v.abs())) / 127.0;
        let beta_q: Vec<i32> =
            (0..n).map(|_| (0.05 * rng.normal() / s_out).round() as i32).collect();
        let (m, s) = quant_mult(s_gamma / (pow2(i32::from(k) - i32::from(j)) * s_out));
        let ln = LayerNorm::new(gamma_q.clone(), beta_q.clone(), m, s).unwrap();
        let mut out = vec![0i8; n];
        ln.apply(c, &x, &mut out).unwrap();
        for i in 0..n {
            let want = (normed[i] / s_out + f64::from(beta_q[i])).clamp(-127.0, 127.0);
            let err = (f64::from(out[i]) - want).abs();
            max_err = max_err.max(err);
            if f64::from(out[i]) != want.round() {
                mismatches += 1;
            }
        }
    }
    LnStats { max_err_lsb: max_err, mismatch: mismatches as f64 / (rows * n) as f64 }
}

#[test]
fn layernorm_error_against_float() {
    let mut rng = Rng::new(1);
    println!("LayerNorm, width 768, error in output LSB (i8), 20 rows for each sigma:");
    for (k, j) in [(48u8, 8u8), (40, 8), (32, 0), (48, 0), (48, 16)] {
        let mut worst: f64 = 0.0;
        for log_sigma in [4, 8, 12, 16, 20, 24] {
            let st = measure_layernorm(k, j, pow2(log_sigma), 20, &mut rng);
            println!(
                "  K={k:2} J={j:2} sigma=2^{log_sigma:2}: max error {:.4} LSB, rounding differs for {:.5}",
                st.max_err_lsb, st.mismatch
            );
            worst = worst.max(st.max_err_lsb);
            if (k, j) == (48, 8) && (12..=24).contains(&log_sigma) {
                // The chosen constants: no more than half an LSB plus a small margin.
                assert!(st.max_err_lsb < 0.51, "K=48 J=8 sigma=2^{log_sigma}: {}", st.max_err_lsb);
            }
        }
        println!("  K={k:2} J={j:2}: worst {worst:.4} LSB");
    }
}

#[test]
fn gelu_table_error_against_float() {
    // GPT-2 MLP inputs are mostly in [-10, 10] (est.). Range R = 10 for both table sizes.
    let r = 10.0;
    let s_out = 10.0 / 127.0;
    println!("GELU table, input range +-{r}, output scale {s_out:.4}:");
    for bits in [8u32, 16] {
        let s_in = r / pow2(bits as i32 - 1);
        let t = GeluTable::new(gelu_table(bits, s_in, s_out)).unwrap();
        let mut rng = Rng::new(2);
        let (mut max_err, mut sq) = (0.0f64, 0.0f64);
        let samples = 200_000;
        for _ in 0..samples {
            let x = (rng.uniform() * 2.0 - 1.0) * r;
            let q = (x / s_in).round() as i64;
            let y = f64::from(t.apply(q)) * s_out;
            let err = (y - gelu(x)).abs() / s_out;
            max_err = max_err.max(err);
            sq += err * err;
        }
        let rms = (sq / f64::from(samples)).sqrt();
        println!("  input i{bits}: max error {max_err:.4} LSB, rms {rms:.4} LSB");
        if bits == 16 {
            assert!(max_err < 0.52, "i16 GELU max error {max_err}");
        } else {
            assert!(max_err < 1.1, "i8 GELU max error {max_err}");
        }
    }
}

/// Integer softmax on real scores quantised to `delta`, against the float softmax.
fn measure_softmax(e: i32, delta: f64, len: usize, p: u8, n: usize, sigma: f64) -> (f64, f64) {
    let table = ExpTable::new(exp_table_i32(e, delta, len)).unwrap();
    let mut rng = Rng::new(3 + n as u64);
    let (mut max_abs, mut kl_sum) = (0.0f64, 0.0f64);
    let rows = 200;
    for _ in 0..rows {
        let x: Vec<f64> = (0..n).map(|_| sigma * rng.normal()).collect();
        let s: Vec<i64> = x.iter().map(|v| (v / delta).round() as i64).collect();
        let mut out = vec![0i64; n];
        softmax(&s, &table, p, &mut out).unwrap();
        let want = common::softmax(&x);
        let got: Vec<f64> = out.iter().map(|&v| v as f64 / pow2(i32::from(p))).collect();
        for (g, w) in got.iter().zip(&want) {
            max_abs = max_abs.max((g - w).abs());
        }
        // q needs no zero entries for KL; add the smallest step.
        let q: Vec<f64> = got.iter().map(|&g| g.max(pow2(-i32::from(p)))).collect();
        kl_sum += kl_bits(&want, &q);
    }
    (max_abs, kl_sum / rows as f64)
}

#[test]
fn softmax_error_against_float() {
    println!("Attention softmax, scores N(0, 3^2), 200 rows:");
    for (e, p, delta_log, len) in [
        (30, 24, 4, 512),
        (30, 24, 6, 2048),
        (30, 24, 8, 8192),
        (30, 30, 8, 8192),
        (24, 30, 8, 8192),
        (30, 30, 10, 32768),
    ] {
        let delta = pow2(-delta_log);
        for n in [16usize, 128, 1024] {
            let (max_abs, kl) = measure_softmax(e, delta, len, p, n, 3.0);
            println!(
                "  E={e} P={p} delta=2^-{delta_log} len={len:5} n={n:4}: max |p - p_float| {max_abs:.2e}, KL {kl:.2e} bits"
            );
            if (e, p, delta_log) == (30, 30, 8) {
                assert!(max_abs < 2e-3, "softmax max error {max_abs}");
                assert!(kl < 1e-5, "softmax KL {kl}");
            }
        }
    }
}

#[test]
fn logit_weights_error_against_float() {
    let v = 50_257;
    println!("Logit weights, E=40, vocabulary {v}, logits N(0, 3^2) plus 5 peaks, 20 rows:");
    for (delta_log, len) in [(4, 512), (6, 2048), (8, 8192)] {
        let delta = pow2(-delta_log);
        let table = LogitTable::new(exp_table_u64(40, delta, len)).unwrap();
        let mut rng = Rng::new(4);
        let mut kl_sum = 0.0;
        let rows = 20;
        for _ in 0..rows {
            let mut x: Vec<f64> = (0..v).map(|_| 3.0 * rng.normal()).collect();
            for _ in 0..5 {
                let i = rng.below(v as u64) as usize;
                x[i] += 12.0;
            }
            let l: Vec<i64> = x.iter().map(|t| (t / delta).round() as i64).collect();
            let mut w = vec![0u64; v];
            logits_to_weights(&l, &table, &mut w).unwrap();
            let total: f64 = w.iter().map(|&a| a as f64).sum();
            // The coder gives every token at least 1 of 2^32 (plan §5.1); do the same here.
            let q: Vec<f64> = w.iter().map(|&a| (a as f64 / total).max(pow2(-32))).collect();
            kl_sum += kl_bits(&common::softmax(&x), &q);
        }
        let kl = kl_sum / f64::from(rows);
        println!("  delta=2^-{delta_log} len={len:5}: mean KL {kl:.2e} bits per token");
        if delta_log == 8 {
            assert!(kl < 1e-5, "logit KL {kl}");
        }
    }
}
