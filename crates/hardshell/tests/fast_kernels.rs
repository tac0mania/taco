//! Property tests: every fast kernel ([`Kernels`]) against the reference kernels and an `i128`
//! reference, on random shapes, on remainder lengths and on the extreme values -128 and 127.
//!
//! NOTE: Miri cannot run the SIMD intrinsics. These tests run the SIMD path of the machine that
//! runs them (NEON `sdot` on aarch64, AVX2 on x86_64) and the portable path.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]

use hardshell::int::{clamp_wide, rescale};
use hardshell::kernels::{Kernels, MAX_DOT_LEN_I32, MatRef, dot_i8, matvec_i8, rescale_clamp};
use hardshell::tensor::Matrix;
use proptest::prelude::*;

/// An `i8` value with a high chance of the extremes -128, -127 and 127.
fn edgy_i8() -> impl Strategy<Value = i8> {
    prop_oneof![3 => any::<i8>(), 1 => Just(-128i8), 1 => Just(127i8), 1 => Just(-127i8)]
}

fn reference(w: &[i8], x: &[i8], cols: usize) -> Vec<i32> {
    let rows = w.len() / cols;
    let mut out = Vec::new();
    for xr in x.chunks_exact(cols) {
        for wr in w.chunks_exact(cols) {
            let want: i128 = wr.iter().zip(xr).map(|(&a, &b)| i128::from(a) * i128::from(b)).sum();
            out.push(i32::try_from(want).unwrap());
        }
    }
    assert_eq!(out.len(), rows * (x.len() / cols));
    out
}

fn shapes() -> impl Strategy<Value = (usize, usize, usize, Vec<i8>, Vec<i8>)> {
    (1usize..19, 1usize..150, 1usize..11).prop_flat_map(|(rows, cols, n)| {
        (
            Just(rows),
            Just(cols),
            Just(n),
            proptest::collection::vec(edgy_i8(), rows * cols),
            proptest::collection::vec(edgy_i8(), n * cols),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn every_kernel_matmul_equals_the_reference((rows, cols, n, w, x) in shapes()) {
        let want = reference(&w, &x, cols);
        let m = Matrix::new(rows, cols, w.clone()).unwrap();
        for k in Kernels::available() {
            let mut out = vec![0i32; rows * n];
            k.matmul_i8(&w, &x, cols, &mut out).unwrap();
            prop_assert_eq!(&out, &want, "kernels {}", k.name());
            // And the single-row form against `matvec_i8`.
            let xr = &x[..cols];
            let mut one = vec![0i32; rows];
            k.matmul_i8(&w, xr, cols, &mut one).unwrap();
            let mut refv = vec![0i32; rows];
            matvec_i8(&m, xr, &mut refv).unwrap();
            prop_assert_eq!(&one, &refv, "kernels {}", k.name());
        }
    }

    #[test]
    fn every_kernel_strided_matmul_equals_the_reference(
        (rows, cols, n, w, x) in shapes(),
        extra in 0usize..40,
    ) {
        // The weights are the first `cols` columns of a matrix with `cols + extra` columns.
        let stride = cols + extra;
        let mut wide = vec![99i8; rows * stride];
        for r in 0..rows {
            wide[r * stride..r * stride + cols].copy_from_slice(&w[r * cols..(r + 1) * cols]);
        }
        let used = (rows - 1) * stride + cols;
        let want = reference(&w, &x, cols);
        for k in Kernels::available() {
            let mut out = vec![0i32; rows * n];
            let m = MatRef { data: &wide[..used], rows, stride };
            k.matmul_i8_strided(m, &x, cols, &mut out).unwrap();
            prop_assert_eq!(&out, &want, "kernels {}", k.name());
        }
    }

    #[test]
    fn every_kernel_dot_equals_the_reference(
        a in proptest::collection::vec(edgy_i8(), 0..300),
        b in proptest::collection::vec(edgy_i8(), 0..300),
    ) {
        for k in Kernels::available() {
            prop_assert_eq!(k.dot_i8(&a, &b), dot_i8(&a, &b), "kernels {}", k.name());
        }
    }

    #[test]
    fn rescale_clamp_equals_rescale_then_clamp(
        x in prop_oneof![any::<i64>(), -(1i64 << 33)..(1i64 << 33), Just(i64::MIN), Just(i64::MAX)],
        m in prop_oneof![any::<i32>(), Just(i32::MIN), Just(i32::MAX)],
        s in prop_oneof![0u8..=70, any::<u8>(), Just(62u8), Just(63u8)],
        lo in prop_oneof![Just(-127i64), Just(-(1i64 << 40)), Just(i64::MIN), any::<i64>()],
        hi in prop_oneof![Just(127i64), Just(1i64 << 40), Just(i64::MAX), any::<i64>()],
    ) {
        prop_assert_eq!(rescale_clamp(x, m, s, lo, hi), clamp_wide(rescale(x, m, s), lo, hi));
    }
}

#[test]
fn every_kernel_is_exact_at_the_accumulator_bound() {
    // All products are +2^14 or -2^14: the sum is exactly +-2^30 at the longest length.
    let n = MAX_DOT_LEN_I32;
    for k in Kernels::available() {
        let w = vec![-128i8; 2 * n];
        let mut out = vec![0i32; 2];
        k.matmul_i8(&w, &vec![-128i8; n], n, &mut out).unwrap();
        assert_eq!(out, [1 << 30, 1 << 30], "kernels {}", k.name());
        let x: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { -128 } else { 127 }).collect();
        let want = dot_i8(&w[..n], &x);
        assert_eq!(k.dot_i8(&w[..n], &x), want, "kernels {}", k.name());
        let mut out = vec![0i32; 2];
        k.matmul_i8(&w, &x, n, &mut out).unwrap();
        assert_eq!(out, [want, want], "kernels {}", k.name());
    }
}

#[test]
fn matmul_refuses_bad_shapes() {
    for k in Kernels::available() {
        let mut out = [0i32; 4];
        assert!(k.matmul_i8(&[0; 6], &[0; 3], 3, &mut out[..2]).is_ok());
        assert!(k.matmul_i8(&[0; 6], &[0; 3], 3, &mut out[..3]).is_err());
        assert!(k.matmul_i8(&[0; 7], &[0; 3], 3, &mut out[..2]).is_err());
        assert!(k.matmul_i8(&[0; 6], &[0; 4], 3, &mut out[..2]).is_err());
        assert!(k.matmul_i8(&[0; 6], &[0; 3], 0, &mut out[..0]).is_err());
        let wide = vec![0i8; MAX_DOT_LEN_I32 + 1];
        assert!(k.matmul_i8(&wide, &wide, MAX_DOT_LEN_I32 + 1, &mut out[..1]).is_err());
    }
}

#[test]
fn this_machine_has_a_simd_kernel() {
    // NOTE: the Mac (M2) has FEAT_DotProd and devbox (Ryzen 5800H) has AVX2. If this fails on
    // another machine, the portable kernels still give the same bits; only the speed differs.
    let names: Vec<&str> = Kernels::available().iter().map(|k| k.name()).collect();
    println!("kernels on this machine: {names:?}");
    assert!(names.len() == 2 || !cfg!(any(target_arch = "aarch64", target_arch = "x86_64")));
}
