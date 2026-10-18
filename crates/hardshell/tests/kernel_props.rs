//! Property tests for the matrix-vector kernels against an `i128` reference.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.

use hardshell::kernels::{matvec_i8, matvec_i8_i16};
use hardshell::tensor::Matrix;
use proptest::prelude::*;

fn mat_and_vec<T: Arbitrary + Clone + core::fmt::Debug>()
-> impl Strategy<Value = (usize, usize, Vec<i8>, Vec<T>)> {
    (1usize..12, 1usize..200).prop_flat_map(|(rows, cols)| {
        (
            Just(rows),
            Just(cols),
            proptest::collection::vec(any::<i8>(), rows * cols),
            proptest::collection::vec(any::<T>(), cols),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn matvec_i8_matches_i128_reference((rows, cols, w, x) in mat_and_vec::<i8>()) {
        let m = Matrix::new(rows, cols, w.clone()).unwrap();
        let mut out = vec![0i32; rows];
        matvec_i8(&m, &x, &mut out).unwrap();
        for r in 0..rows {
            let want: i128 = (0..cols).map(|c| i128::from(w[r * cols + c]) * i128::from(x[c])).sum();
            prop_assert_eq!(i128::from(out[r]), want);
        }
    }

    #[test]
    fn matvec_i8_i16_matches_i128_reference((rows, cols, w, x) in mat_and_vec::<i16>()) {
        let m = Matrix::new(rows, cols, w.clone()).unwrap();
        let mut out = vec![0i64; rows];
        matvec_i8_i16(&m, &x, &mut out).unwrap();
        for r in 0..rows {
            let want: i128 = (0..cols).map(|c| i128::from(w[r * cols + c]) * i128::from(x[c])).sum();
            prop_assert_eq!(i128::from(out[r]), want);
        }
    }
}
