//! Property tests of the match model and the mixer: determinism, no zero weights, reset
//! equivalence, bounds, and a step-by-step reference of the match rules.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use std::collections::HashMap;

use common::{SplitMix64, run_block, synthetic_log};
use proptest::prelude::*;
use taco_match::{MatchLen, MatchModel, MatchParams, Mixer, MixerParams, Prediction, TokenId};

fn small_params() -> impl Strategy<Value = MatchParams> {
    (prop::sample::subsequence(vec![12u8, 8, 6, 4, 3, 2, 1], 1..5), 4u8..12, 0u16..20, 1u32..80)
        .prop_map(|(orders, bits, extra, max_len)| {
            let v = u16::from(orders[0]) + extra;
            MatchParams::new(&orders, bits, v, max_len).unwrap()
        })
}

/// Tokens with repeats: a few short motifs and random tokens from a small vocabulary.
fn token_stream() -> impl Strategy<Value = Vec<u32>> {
    (any::<u64>(), 0usize..600, 2u64..40).prop_map(|(seed, n, v)| {
        let mut rng = SplitMix64(seed);
        let motifs: Vec<Vec<u32>> =
            (0..4).map(|_| (0..2 + rng.below(10)).map(|_| rng.below(v) as u32).collect()).collect();
        let mut out = Vec::new();
        while out.len() < n {
            if rng.below(3) == 0 {
                out.push(rng.below(v) as u32);
            } else {
                out.extend(&motifs[rng.below(4) as usize]);
            }
        }
        out.truncate(n);
        out
    })
}

/// The rules of the module documentation of `matcher.rs`, written again in the simplest form.
fn reference(tokens: &[u32], p: &MatchParams) -> Vec<Option<(u32, u32)>> {
    const B: u64 = 0x9E37_79B9_7F4A_7C15;
    const MIX: u64 = 0xD6E8_FEB8_6659_FD93;
    let slot = |h: &[u32], i: usize, n: usize| -> u64 {
        let (mut acc, mut bk) = (0u64, 1u64);
        for k in 1..=n {
            acc = acc.wrapping_add((u64::from(h[i - k]) + 1).wrapping_mul(bk));
            bk = bk.wrapping_mul(B);
        }
        (acc ^ (acc >> 32)).wrapping_mul(MIX) >> (64 - u32::from(p.hash_bits()))
    };
    let mut tables: Vec<HashMap<u64, usize>> = vec![HashMap::new(); p.orders().len()];
    let mut h: Vec<u32> = Vec::new();
    let (mut ptr, mut len) = (0usize, 0u32);
    let mut out = Vec::new();
    for &t in tokens {
        let i = h.len();
        if len < u32::from(p.orders()[0]) {
            for (o, &n) in p.orders().iter().enumerate() {
                let n = usize::from(n);
                if n as u32 <= len || i < n {
                    continue;
                }
                let Some(&c) = tables[o].get(&slot(&h, i, n)) else { continue };
                let mut v = 0usize;
                while v < usize::from(p.verify_max()) && v < c && h[c - v - 1] == h[i - v - 1] {
                    v += 1;
                }
                if v >= n {
                    ptr = c;
                    len = v as u32;
                    break;
                }
            }
        }
        for (o, &n) in p.orders().iter().enumerate() {
            if i >= usize::from(n) {
                tables[o].insert(slot(&h, i, usize::from(n)), i);
            }
        }
        if len > 0 {
            out.push(Some((h[ptr], len)));
            if h[ptr] == t {
                ptr += 1;
                len = (len + 1).min(p.max_len());
            } else {
                len = 0;
            }
        } else {
            out.push(None);
        }
        h.push(t);
    }
    out
}

fn predictions(tokens: &[u32], p: MatchParams) -> Vec<Option<(u32, u32)>> {
    let mut m = MatchModel::new(p);
    tokens
        .iter()
        .map(|&t| {
            let r = m.predict().map(|x| (x.token.get(), x.length.get()));
            m.observe(TokenId::new(t));
            r
        })
        .collect()
}

fn random_weights(rng: &mut SplitMix64, vocab: usize) -> Vec<u64> {
    (0..vocab)
        .map(|_| match rng.below(5) {
            0 => 0,
            1 => u64::MAX >> rng.below(64),
            _ => rng.next_u64() >> (20 + rng.below(44)),
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn match_model_obeys_the_reference(tokens in token_stream(), p in small_params()) {
        prop_assert_eq!(predictions(&tokens, p), reference(&tokens, &p));
    }

    #[test]
    fn predictions_are_earlier_tokens_and_lengths_are_bounded(
        tokens in token_stream(), p in small_params()
    ) {
        for (i, pr) in predictions(&tokens, p).iter().enumerate() {
            if let Some((t, l)) = pr {
                prop_assert!(tokens[..i].contains(t));
                prop_assert!(*l >= 1 && *l <= p.max_len().max(u32::from(p.verify_max())));
            }
        }
    }

    #[test]
    fn reset_gives_the_state_of_a_new_model_and_mixer(
        a in token_stream(), b in token_stream(), p in small_params(), seed in any::<u64>()
    ) {
        let mut model = MatchModel::new(p);
        let mut mixer = Mixer::new(MixerParams::V1);
        let mut rng = SplitMix64(seed);
        run_block(&mut model, &mut mixer, &a, |_| random_weights(&mut rng, 64));
        model.reset();
        mixer.reset();
        let b: Vec<u32> = b.iter().map(|t| t % 64).collect();
        let mut r1 = SplitMix64(seed ^ 1);
        let used = run_block(&mut model, &mut mixer, &b, |_| random_weights(&mut r1, 64));
        let mut r2 = SplitMix64(seed ^ 1);
        let fresh = run_block(
            &mut MatchModel::new(p), &mut Mixer::new(MixerParams::V1), &b,
            |_| random_weights(&mut r2, 64),
        );
        prop_assert_eq!(used, fresh);
    }

    #[test]
    fn the_same_input_gives_the_same_output(tokens in token_stream(), seed in any::<u64>()) {
        let tokens: Vec<u32> = tokens.iter().map(|t| t % 50).collect();
        let p = MatchParams::new(&[8, 4, 2, 1], 10, 16, 1000).unwrap();
        let mut r1 = SplitMix64(seed);
        let one = run_block(&mut MatchModel::new(p), &mut Mixer::new(MixerParams::V1), &tokens,
            |_| random_weights(&mut r1, 50));
        let mut r2 = SplitMix64(seed);
        let two = run_block(&mut MatchModel::new(p), &mut Mixer::new(MixerParams::V1), &tokens,
            |_| random_weights(&mut r2, 50));
        prop_assert_eq!(one, two);
    }

    #[test]
    fn mixed_weights_are_never_zero_and_below_2_pow_62(
        seed in any::<u64>(), vocab in 1usize..300, len in 1u32..5000, rounds in 1usize..30
    ) {
        let mut rng = SplitMix64(seed);
        let mut mixer = Mixer::new(MixerParams::V1);
        for _ in 0..rounds {
            let mut w = random_weights(&mut rng, vocab);
            let token = TokenId::new(rng.below(vocab as u64) as u32);
            let pred = Prediction { token, length: MatchLen::new(len).unwrap() };
            let some = rng.below(4) != 0;
            mixer.mix(&mut w, some.then_some(pred)).unwrap();
            if some {
                prop_assert!(w.iter().all(|&x| (1..1u64 << 62).contains(&x)));
            } else {
                prop_assert!(w.iter().all(|&x| x >= 1));
            }
            mixer.update(TokenId::new(rng.below(vocab as u64) as u32));
        }
    }

    #[test]
    fn the_predicted_token_gets_the_mixer_probability(
        seed in any::<u64>(), vocab in 2usize..300, len in 1u32..300
    ) {
        // HardShell weights: at most 2^40 each (ADR 0004 section 5).
        let mut rng = SplitMix64(seed);
        let mut w: Vec<u64> = (0..vocab).map(|_| rng.next_u64() >> (24 + rng.below(40))).collect();
        w[0] = 1 << 40;
        let token = TokenId::new(rng.below(vocab as u64) as u32);
        let length = MatchLen::new(len).unwrap();
        let mut mixer = Mixer::new(MixerParams::V1);
        let total: u128 = w.iter().map(|&x| u128::from(x)).sum();
        let agree = w.iter().enumerate().max_by_key(|(i, x)| (**x, std::cmp::Reverse(*i))).map(|(i, _)| i) == Some(token.get() as usize);
        let q = mixer.peek(w[token.get() as usize], total, length, agree);
        mixer.mix(&mut w, Some(Prediction { token, length })).unwrap();
        let sum: u128 = w.iter().map(|&x| u128::from(x)).sum();
        let wp = u128::from(w[token.get() as usize]);
        // wp / sum = q / 2^16, up to the 1-floor of each weight and the shift.
        let lhs = wp << 16;
        let rhs = sum * u128::from(q);
        let tol = (sum >> 20).max(1) + (vocab as u128) * 65_536;
        prop_assert!(lhs.abs_diff(rhs) <= tol, "lhs {lhs} rhs {rhs} tol {tol}");
    }

    #[test]
    fn no_prediction_leaves_the_weights_except_zeros(seed in any::<u64>(), vocab in 1usize..300) {
        let mut rng = SplitMix64(seed);
        let w = random_weights(&mut rng, vocab);
        let mut out = w.clone();
        Mixer::new(MixerParams::V1).mix(&mut out, None).unwrap();
        prop_assert!(w.iter().zip(&out).all(|(&a, &b)| b == a.max(1)));
    }
}

#[test]
fn a_draft_of_a_periodic_sequence_is_its_continuation() {
    // Period 3, with a period-1 run inside: the draft reads its own earlier tokens.
    let unit = [5u32, 9, 9, 9, 2];
    let tokens: Vec<u32> = unit.iter().cycle().take(40).copied().collect();
    let mut m = MatchModel::new(MatchParams::V1);
    for &t in &tokens {
        m.observe(TokenId::new(t));
    }
    let draft: Vec<u32> = m.draft(8).iter().map(|t| t.get()).collect();
    let next: Vec<u32> = unit.iter().cycle().skip(40).take(8).copied().collect();
    assert_eq!(draft, next);
    assert_eq!(m.predict().map(|p| p.token.get()), Some(next[0]));
    assert!(MatchModel::new(MatchParams::V1).draft(8).is_empty());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn a_draft_starts_with_the_prediction_and_changes_nothing(
        tokens in token_stream(), p in small_params(), k in 0usize..12
    ) {
        let mut m = MatchModel::new(p);
        for &t in &tokens {
            m.observe(TokenId::new(t));
        }
        let before = m.predict();
        let d = m.draft(k);
        prop_assert!(d.len() <= k);
        prop_assert_eq!(m.predict(), before);
        if let (Some(first), Some(pred)) = (d.first(), before) {
            prop_assert_eq!(*first, pred.token);
        }
        if k > 0 {
            prop_assert_eq!(before.is_none(), d.is_empty());
        }
    }
}

#[test]
fn a_prediction_outside_the_weights_is_an_error_and_changes_nothing() {
    let mut w = vec![5u64, 7, 9];
    let pred = Prediction { token: TokenId::new(3), length: MatchLen::new(4).unwrap() };
    let err = Mixer::new(MixerParams::V1).mix(&mut w, Some(pred));
    assert!(err.is_err());
    assert_eq!(w, vec![5, 7, 9]);
    assert!(Mixer::new(MixerParams::V1).mix(&mut [], None).is_err());
}

#[test]
fn the_mixer_learns_to_trust_a_reliable_match() {
    // GPT-2 gives the predicted token 1 %, and the prediction is always correct.
    let mut mixer = Mixer::new(MixerParams::V1);
    let len = MatchLen::new(20).unwrap();
    let first = mixer.peek(1 << 33, 1 << 40, len, false);
    for _ in 0..200 {
        let mut w = vec![1u64 << 33; 128];
        let pred = Prediction { token: TokenId::new(7), length: len };
        mixer.mix(&mut w, Some(pred)).unwrap();
        mixer.update(TokenId::new(7));
    }
    let later = mixer.peek(1 << 33, 1 << 40, len, false);
    assert!(first < 1000, "{first}");
    assert!(later > 60_000, "{later}");
}

#[test]
fn invalid_parameters_are_refused() {
    assert!(MatchParams::new(&[], 10, 10, 10).is_err());
    assert!(MatchParams::new(&[4, 8], 10, 10, 10).is_err());
    assert!(MatchParams::new(&[4, 4], 10, 10, 10).is_err());
    assert!(MatchParams::new(&[65], 10, 70, 10).is_err());
    assert!(MatchParams::new(&[8], 3, 10, 10).is_err());
    assert!(MatchParams::new(&[8], 10, 7, 10).is_err());
    assert!(MatchParams::new(&[8], 10, 8, 0).is_err());
    assert!(MixerParams::new(0, 0, 0, 10, 0, 0).is_err());
    assert!(MixerParams::new(8, 9, 0, 10, 0, 0).is_err());
    assert!(MixerParams::new(1, 0, 0, 10, 1 << 23, 0).is_err());
    assert_eq!(MixerParams::new(160, 16, 64, 255, 1 << 16, 1 << 16).unwrap(), MixerParams::V1);
    assert_eq!(MatchParams::new(&[32, 16, 8, 4, 2, 1], 20, 64, 65_535).unwrap(), MatchParams::V1);
}

#[test]
fn the_synthetic_log_has_long_matches() {
    let tokens = synthetic_log(7, 3000);
    let p = predictions(&tokens, MatchParams::V1);
    let hits = p.iter().zip(&tokens).filter(|(x, t)| x.map(|x| x.0) == Some(**t)).count();
    // 912 of 3,000 tokens (measured): a variable field follows 1 in 4 template tokens.
    assert!(hits > 600, "{hits}");
}
