//! Property tests for the integer LayerNorm, GELU table, softmax and logit weights. These use
//! integers only: they check exact properties, not the error against floats.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.

use hardshell::Error;
use hardshell::gelu::GeluTable;
use hardshell::layernorm::{LayerNorm, LnConstants};
use hardshell::logits::{LogitTable, logits_to_weights};
use hardshell::softmax::{ExpTable, softmax};
use proptest::prelude::*;

fn exp_table() -> Vec<i32> {
    // A decreasing table without floats: 2^30 * (255/256)^i, in integers.
    let mut t = Vec::with_capacity(4096);
    let mut v: i64 = 1 << 30;
    for _ in 0..4096 {
        t.push(i32::try_from(v).unwrap());
        v = v * 255 / 256;
    }
    t
}

fn ln_strategy() -> impl Strategy<Value = (Vec<i32>, Vec<i16>, Vec<i32>, i32, u8)> {
    (1usize..300).prop_flat_map(|n| {
        (
            proptest::collection::vec(-(1i32 << 24)..(1 << 24), n),
            proptest::collection::vec(any::<i16>(), n),
            proptest::collection::vec(-1000i32..1000, n),
            1i32..i32::MAX,
            40u8..90,
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn layernorm_ignores_a_constant_offset(
        (x, g, b, m, s) in ln_strategy(), c in -(1i32 << 24)..(1 << 24)
    ) {
        let consts = LnConstants::new(48, 8, 12345).unwrap();
        let ln = LayerNorm::new(g, b, m, s).unwrap();
        let mut y0 = vec![0i8; x.len()];
        let mut y1 = vec![0i8; x.len()];
        ln.apply(consts, &x, &mut y0).unwrap();
        // The mean moves by exactly c (floor_div(sum + n*c, n) = mean + c), so the output is
        // the same, bit for bit.
        let shifted: Vec<i32> = x.iter().map(|&v| v + c).collect();
        ln.apply(consts, &shifted, &mut y1).unwrap();
        prop_assert_eq!(&y0, &y1);
        prop_assert!(y0.iter().all(|&v| v != i8::MIN));
    }

    #[test]
    fn layernorm_accepts_every_i32_input(
        x in proptest::collection::vec(any::<i32>(), 1..64), k in 0u8..=48, j in 0u8..=16,
        eps in 1u64..(1 << 62), m in any::<i32>(), s in any::<u8>()
    ) {
        let n = x.len();
        let ln = LayerNorm::new(vec![i16::MIN; n], vec![0; n], m, s).unwrap();
        let mut y = vec![0i8; n];
        ln.apply(LnConstants::new(k, j, eps).unwrap(), &x, &mut y).unwrap();
        prop_assert!(y.iter().all(|&v| (-127..=127).contains(&v)));
    }

    #[test]
    fn softmax_mass_is_close_to_one_and_ordered(
        scores in proptest::collection::vec(-5000i64..5000, 1..200), p in 0u8..=30
    ) {
        let t = ExpTable::new(exp_table()).unwrap();
        let mut out = vec![0i64; scores.len()];
        softmax(&scores, &t, p, &mut out).unwrap();
        let sum: i64 = out.iter().sum();
        let n = i64::try_from(scores.len()).unwrap();
        // Each floor division loses less than 1.
        prop_assert!(sum <= 1 << p && sum > (1 << p) - n);
        for i in 0..scores.len() {
            for j in 0..scores.len() {
                if scores[i] >= scores[j] { prop_assert!(out[i] >= out[j]); }
            }
        }
    }

    #[test]
    fn logit_weights_ignore_a_constant_offset(
        l in proptest::collection::vec(-100_000i64..100_000, 1..300), c in -(1i64 << 40)..(1 << 40)
    ) {
        let table: Vec<u64> = exp_table().into_iter().map(|v| u64::try_from(v).unwrap() << 10).collect();
        let t = LogitTable::new(table).unwrap();
        let mut w0 = vec![0u64; l.len()];
        let mut w1 = vec![0u64; l.len()];
        logits_to_weights(&l, &t, &mut w0).unwrap();
        let shifted: Vec<i64> = l.iter().map(|&v| v + c).collect();
        logits_to_weights(&shifted, &t, &mut w1).unwrap();
        prop_assert_eq!(&w0, &w1);
        prop_assert_eq!(w0.iter().max().copied(), Some(t.entries()[0]));
    }

    #[test]
    fn gelu_clamps_inputs_to_the_table(x in any::<i64>()) {
        let entries: Vec<i8> = (0..256).map(|i| i8::try_from(i / 2 - 64).unwrap()).collect();
        let t = GeluTable::new(entries.clone()).unwrap();
        let idx = usize::try_from(x.clamp(-128, 127) + 128).unwrap();
        prop_assert_eq!(t.apply(x), entries[idx]);
    }
}

#[test]
fn tables_refuse_bad_entries() {
    assert!(ExpTable::new(vec![]).is_err());
    assert!(ExpTable::new(vec![0, 0]).is_err());
    assert!(ExpTable::new(vec![5, 6]).is_err());
    assert!(ExpTable::new(vec![5, -1]).is_err());
    assert!(LogitTable::new(vec![(1 << 40) + 1]).is_err());
    assert!(LogitTable::new(vec![1 << 40, 1, 0]).is_ok());
    assert!(matches!(GeluTable::new(vec![0; 100]), Err(Error::LengthMismatch { .. })));
    assert!(LnConstants::new(49, 0, 1).is_err());
    assert!(LnConstants::new(48, 17, 1).is_err());
    assert!(LnConstants::new(48, 8, 0).is_err());
    assert!(LayerNorm::new(vec![1; 3], vec![1; 2], 1, 1).is_err());
    let t = ExpTable::new(vec![4, 2, 1]).unwrap();
    assert!(softmax(&[1, 2], &t, 31, &mut [0, 0]).is_err());
    assert!(softmax(&[], &t, 3, &mut []).is_err());
}

#[test]
fn layernorm_of_a_constant_row_is_beta() {
    let ln = LayerNorm::new(vec![1000; 4], vec![-3, 0, 5, 200], 1 << 30, 40).unwrap();
    let mut y = [0i8; 4];
    ln.apply(LnConstants::new(48, 8, 1).unwrap(), &[77; 4], &mut y).unwrap();
    assert_eq!(y, [-3, 0, 5, 127]);
}
