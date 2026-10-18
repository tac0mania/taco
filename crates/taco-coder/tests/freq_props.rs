//! Property tests for `FreqTable::from_weights` (plan §5.1) and `FreqTable::from_freqs`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::bits;
use proptest::prelude::*;
use taco_coder::{Freq, FreqTable};

/// A direct copy of the §5.1 steps, written separately from the crate, with a `Vec` of
/// frequencies instead of cumulative starts.
fn reference(weights: &[u64], b: u32) -> Vec<u64> {
    let t = 1u128 << b;
    let v = weights.len() as u128;
    let mut w: Vec<u128> = weights.iter().map(|&x| u128::from(x)).collect();
    if w.iter().all(|&x| x == 0) {
        w = vec![1; weights.len()];
    }
    let sum: u128 = w.iter().sum();
    let mut f: Vec<u128> = w.iter().map(|&x| 1 + x * (t - v) / sum).collect();
    let mut best = 0;
    for i in 1..w.len() {
        if w[i] > w[best] {
            best = i;
        }
    }
    let rest = t - f.iter().sum::<u128>();
    f[best] += rest;
    f.into_iter().map(|x| u64::try_from(x).unwrap()).collect()
}

fn freqs(t: &FreqTable) -> Vec<u64> {
    t.freqs().map(Freq::get).collect()
}

fn weight() -> impl Strategy<Value = u64> {
    prop_oneof![Just(0u64), Just(u64::MAX), 0u64..8, any::<u64>(), 0u64..1_000_000]
}

/// Weights and a bit count that has room for all symbols.
fn weights_and_bits() -> impl Strategy<Value = (Vec<u64>, u32)> {
    proptest::collection::vec(weight(), 1..600).prop_flat_map(|w| {
        let min_bits = usize::BITS - (w.len() - 1).leading_zeros();
        (Just(w), min_bits..=32)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn total_is_exact_and_every_freq_is_at_least_one((w, b) in weights_and_bits()) {
        let t = FreqTable::from_weights(&w, bits(b)).unwrap();
        let f = freqs(&t);
        prop_assert_eq!(f.len(), w.len());
        prop_assert!(f.iter().all(|&x| x >= 1));
        prop_assert_eq!(f.iter().map(|&x| u128::from(x)).sum::<u128>(), 1u128 << b);
        prop_assert_eq!(t.total(), 1u64 << b);
    }

    #[test]
    fn from_weights_matches_the_reference_and_is_deterministic((w, b) in weights_and_bits()) {
        let a = FreqTable::from_weights(&w, bits(b)).unwrap();
        let again = FreqTable::from_weights(&w, bits(b)).unwrap();
        prop_assert_eq!(&a, &again);
        prop_assert_eq!(freqs(&a), reference(&w, b));
    }

    #[test]
    fn a_larger_weight_never_gets_a_smaller_frequency((w, b) in weights_and_bits()) {
        let f = freqs(&FreqTable::from_weights(&w, bits(b)).unwrap());
        // Group the frequencies by weight, in weight order. The largest frequency of a group
        // is not more than the smallest frequency of the next group. (Inside the last group,
        // the remainder can make one member larger than the others.)
        let mut groups: std::collections::BTreeMap<u64, (u64, u64)> = Default::default();
        for (i, &x) in w.iter().enumerate() {
            let g = groups.entry(x).or_insert((u64::MAX, 0));
            g.0 = g.0.min(f[i]);
            g.1 = g.1.max(f[i]);
        }
        let ordered: Vec<(u64, (u64, u64))> = groups.into_iter().collect();
        for pair in ordered.windows(2) {
            let ((wa, (_, max_a)), (wb, (min_b, _))) = (pair[0], pair[1]);
            prop_assert!(max_a <= min_b, "weight {} has f {} but weight {} has f {}", wa, max_a, wb, min_b);
        }
    }

    #[test]
    fn a_larger_weight_for_one_symbol_never_lowers_its_frequency(
        (w, b) in weights_and_bits(),
        pick in any::<usize>(),
        add in any::<u64>(),
    ) {
        let i = pick % w.len();
        let mut w2 = w.clone();
        w2[i] = w2[i].saturating_add(add);
        let f1 = freqs(&FreqTable::from_weights(&w, bits(b)).unwrap());
        let f2 = freqs(&FreqTable::from_weights(&w2, bits(b)).unwrap());
        prop_assert!(f2[i] >= f1[i], "f went from {} to {}", f1[i], f2[i]);
    }

    #[test]
    fn from_freqs_accepts_every_from_weights_table((w, b) in weights_and_bits()) {
        let a = FreqTable::from_weights(&w, bits(b)).unwrap();
        let c = FreqTable::from_freqs(&freqs(&a), bits(b)).unwrap();
        prop_assert_eq!(a, c);
    }
}

#[test]
fn from_weights_handles_the_gpt2_vocabulary_at_32_bits() {
    let w = common::skewed_weights(50_257, 7);
    let t = FreqTable::from_weights(&w, bits(32)).unwrap();
    let f = freqs(&t);
    assert_eq!(f.len(), 50_257);
    assert_eq!(f.iter().sum::<u64>(), 1 << 32);
    assert_eq!(f, reference(&w, 32));
}

#[test]
fn from_weights_handles_the_gpt2_vocabulary_at_the_smallest_total() {
    // 2^16 = 65,536 is the smallest power of two above 50,257.
    let w = common::skewed_weights(50_257, 8);
    let t = FreqTable::from_weights(&w, bits(16)).unwrap();
    assert_eq!(freqs(&t), reference(&w, 16));
    assert!(FreqTable::from_weights(&w, bits(15)).is_err());
}
