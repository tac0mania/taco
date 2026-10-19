//! `Mixer::mix_stats` (the host step of the mixed coder op, `hardshell-cuda`
//! MIXED-INTEGRATION.md section 3): with the statistics of the weights, then `update`, it gives
//! the state that `mix` and `update` give.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::SplitMix64;
use proptest::prelude::*;
use taco_match::{MatchLen, MatchModel, MatchParams, Mixer, MixerParams, Prediction, TokenId};

/// Tokens with repeats: a few short motifs and random tokens from a small vocabulary.
fn stream(seed: u64, n: usize, v: u64) -> Vec<u32> {
    let mut rng = SplitMix64(seed);
    let motifs: Vec<Vec<u32>> =
        (0..4).map(|_| (0..2 + rng.below(12)).map(|_| rng.below(v) as u32).collect()).collect();
    let mut out = Vec::new();
    while out.len() < n {
        if rng.below(4) == 0 {
            out.push(rng.below(v) as u32);
        } else {
            out.extend(&motifs[rng.below(4) as usize]);
        }
    }
    out.truncate(n);
    out
}

fn weights(rng: &mut SplitMix64, vocab: usize) -> Vec<u64> {
    (0..vocab)
        .map(|_| match rng.below(6) {
            0 => 0,
            _ => rng.next_u64() >> (24 + rng.below(40)),
        })
        .collect()
}

/// `W`, `w[p]` and the agreement bit of `w` (pass 1 of `Mixer::mix`: the lowest id on a tie).
fn stats(w: &[u64], p: TokenId) -> (u128, u64, bool) {
    let total = w.iter().map(|&x| u128::from(x)).sum();
    let mut top = 0;
    for (i, &x) in w.iter().enumerate() {
        if x > w[top] {
            top = i;
        }
    }
    (total, w[p.get() as usize], top == p.get() as usize)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn mix_stats_then_update_equals_mix_then_update(seed in any::<u64>(), n in 1usize..400) {
        let toks = stream(seed, n, 30);
        let mut model = MatchModel::new(MatchParams::new(&[8, 4, 2, 1], 10, 16, 1000).unwrap());
        let mut a = Mixer::new(MixerParams::V1);
        let mut b = Mixer::new(MixerParams::V1);
        let mut rng = SplitMix64(seed ^ 3);
        for &t in &toks {
            let pred = model.predict();
            let mut w = weights(&mut rng, 30);
            if let Some(p) = pred {
                let (total, wp, agree) = stats(&w, p.token);
                let want = a.peek(wp, total, p.length, agree);
                prop_assert_eq!(b.mix_stats(total, wp, agree, pred), Some(want));
            } else {
                prop_assert_eq!(b.mix_stats(0, 0, false, None), None);
            }
            a.mix(&mut w, pred).unwrap();
            a.update(TokenId::new(t));
            b.update(TokenId::new(t));
            model.observe(TokenId::new(t));
        }
        // The same state: the same mixed weights at a last position.
        let p = Prediction { token: TokenId::new(3), length: MatchLen::new(9).unwrap() };
        let mut wa = weights(&mut rng, 30);
        let mut wb = wa.clone();
        a.mix(&mut wa, Some(p)).unwrap();
        b.mix(&mut wb, Some(p)).unwrap();
        prop_assert_eq!(wa, wb);
    }
}

#[test]
fn mix_stats_without_a_prediction_clears_the_pending_update() {
    let mut m = Mixer::new(MixerParams::V1);
    let p = Prediction { token: TokenId::new(1), length: MatchLen::new(4).unwrap() };
    let before = m.peek(5, 10, p.length, true);
    m.mix_stats(10, 5, true, Some(p));
    assert_eq!(m.mix_stats(10, 5, true, None), None);
    m.update(TokenId::new(1)); // no pending state: no change
    assert_eq!(m.peek(5, 10, p.length, true), before);
}
