//! Tests of the parts for speculative decoding and the skip rule (ADR 0017): `mix_stats`,
//! `speculate`, checkpoints of the mixer, and mixer rule version 2.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{SplitMix64, synthetic_log};
use proptest::prelude::*;
use taco_match::{
    MatchError, MatchLen, MatchModel, MatchParams, Mixer, MixerParams, Prediction, SkipParams,
    TokenId, rule_constants, skip_miss_weights,
};

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
    (0..vocab).map(|_| rng.next_u64() >> (24 + rng.below(40))).collect()
}

/// `W`, `w[p]` and the agreement bit of the weights `w` (pass 1 of `Mixer::mix`).
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

/// A small skip rule: a table of 2^6 entries (many collisions), skip from 0.5 and one update.
fn small_skip() -> MixerParams {
    MixerParams::V1.with_skip(SkipParams::new(1 << 21, 1, 2, 6).unwrap())
}

/// One position of the calls of ADR 0017 for rule version 2 (also for rule 1: `skip` gives
/// `None`): gives a value that names the decision and `q`.
fn position(m: &mut Mixer, w: &[u64], pred: Option<Prediction>, actual: u32) -> (u8, u32) {
    let out = match pred {
        None => (0, 0),
        Some(p) => match m.skip(Some(p)) {
            Some(code) => (2, code.q),
            None => {
                let (total, wp, agree) = stats(w, p.token);
                (1, m.mix_stats(total, wp, agree, Some(p)).unwrap())
            }
        },
    };
    m.update(TokenId::new(actual));
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

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
        // The same state: the same mixed weights for a last position.
        let p = Prediction { token: TokenId::new(3), length: MatchLen::new(9).unwrap() };
        let (mut wa, mut wb) = (weights(&mut rng, 30), Vec::new());
        wb.clone_from(&wa);
        a.mix(&mut wa, Some(p)).unwrap();
        b.mix(&mut wb, Some(p)).unwrap();
        prop_assert_eq!(wa, wb);
    }

    #[test]
    fn speculate_gives_the_predictions_along_the_draft(
        seed in any::<u64>(), n in 0usize..500, v in 2u64..20, k in 0usize..40, small in any::<bool>()
    ) {
        let params = if small {
            MatchParams::new(&[8, 4, 2, 1], 5, 16, 30).unwrap()
        } else {
            MatchParams::V1
        };
        let mut m = MatchModel::new(params);
        for &t in &stream(seed, n, v) {
            m.observe(TokenId::new(t));
        }
        let spec = m.speculate(k);
        prop_assert!(spec.len() <= k);
        prop_assert_eq!(m.speculate(k), spec.clone(), "speculate changed the model");
        let mut c = m.clone();
        for (j, s) in spec.iter().enumerate() {
            prop_assert_eq!(c.predict(), Some(*s), "position {}", j);
            c.observe(s.token);
        }
        if spec.len() < k {
            prop_assert_eq!(c.predict(), None);
        }
        if let (Some(first), Some(d)) = (spec.first(), m.draft(1).first()) {
            prop_assert_eq!(first.token, *d);
        }
    }

    #[test]
    fn a_rollback_gives_the_state_of_its_checkpoint(seed in any::<u64>(), n in 1usize..300) {
        let toks = stream(seed, n, 25);
        let mut model = MatchModel::new(MatchParams::new(&[8, 4, 2, 1], 10, 16, 1000).unwrap());
        let mut m = Mixer::new(small_skip());
        let mut rng = SplitMix64(seed ^ 9);
        let half = n / 2;
        for &t in &toks[..half] {
            let w = weights(&mut rng, 25);
            position(&mut m, &w, model.predict(), t);
            model.observe(TokenId::new(t));
        }
        let at_cp = m.clone();
        let cp = m.checkpoint();
        let mut mid = None;
        let mut rng2 = SplitMix64(seed ^ 11);
        for (j, &t) in toks[half..].iter().enumerate() {
            if j == 3 {
                mid = Some((m.clone(), m.checkpoint()));
            }
            let w = weights(&mut rng2, 25);
            // A wrong token now and then, as a wrong draft gives.
            let a = if rng2.below(3) == 0 { (t + 1) % 25 } else { t };
            position(&mut m, &w, model.predict(), a);
            model.observe(TokenId::new(t));
        }
        // Back to the middle, then to the start: both give the saved states.
        let probe = |x: &mut Mixer, y: &mut Mixer, rng: &mut SplitMix64| -> Result<(), TestCaseError> {
            for i in 0..20u32 {
                let w = weights(rng, 25);
                let p = Prediction { token: TokenId::new(i % 25), length: MatchLen::new(1 + i).unwrap() };
                prop_assert_eq!(position(x, &w, Some(p), i % 7), position(y, &w, Some(p), i % 7));
            }
            Ok(())
        };
        if let Some((mut saved, cp_mid)) = mid {
            m.rollback(&cp_mid).unwrap();
            let mut copy = m.clone();
            m.rollback(&cp_mid).unwrap();
            probe(&mut copy, &mut saved, &mut SplitMix64(seed ^ 5))?;
        }
        m.rollback(&cp).unwrap();
        let mut copy = m.clone();
        probe(&mut copy, &mut at_cp.clone(), &mut SplitMix64(seed ^ 6))?;
        m.commit();
        prop_assert_eq!(m.rollback(&cp), Err(MatchError::StaleCheckpoint));
    }

    #[test]
    fn rule_two_decisions_depend_only_on_earlier_tokens(seed in any::<u64>(), n in 1usize..300) {
        // Two mixers with the same tokens and other GPT-2 weights skip at the same positions
        // until the first mixed position whose weights differ changes nothing of the skip state.
        let toks = stream(seed, n, 20);
        let mut model = MatchModel::new(MatchParams::new(&[8, 4, 2, 1], 10, 16, 1000).unwrap());
        let mut a = Mixer::new(small_skip());
        let mut b = Mixer::new(small_skip());
        let (mut ra, mut rb) = (SplitMix64(seed), SplitMix64(seed ^ 77));
        for &t in &toks {
            let pred = model.predict();
            let da = a.skip(pred);
            let db = b.skip(pred);
            prop_assert_eq!(da, db);
            if da.is_none() {
                let wa = weights(&mut ra, 20);
                let wb = weights(&mut rb, 20);
                if let Some(p) = pred {
                    let (t1, w1, g1) = stats(&wa, p.token);
                    let (t2, w2, g2) = stats(&wb, p.token);
                    a.mix_stats(t1, w1, g1, pred);
                    b.mix_stats(t2, w2, g2, pred);
                }
            }
            a.update(TokenId::new(t));
            b.update(TokenId::new(t));
            model.observe(TokenId::new(t));
        }
    }
}

#[test]
fn rule_one_never_skips() {
    let mut m = Mixer::new(MixerParams::V1);
    let p = Prediction { token: TokenId::new(5), length: MatchLen::new(40).unwrap() };
    for _ in 0..500 {
        assert_eq!(m.skip(Some(p)), None);
        let (total, wp, agree) = (1u128 << 40, 1u64 << 39, true);
        m.mix_stats(total, wp, agree, Some(p));
        m.update(TokenId::new(5));
    }
    assert_eq!(MixerParams::V1.skip(), None);
}

#[test]
fn rule_two_skips_a_reliable_prediction_and_learns_a_miss() {
    let mut m = Mixer::new(MixerParams::V2);
    let p = Prediction { token: TokenId::new(5), length: MatchLen::new(40).unwrap() };
    // The start probability is 0.5: two hits give 0.9, the limit of rule version 2.
    for _ in 0..2 {
        assert_eq!(m.skip(Some(p)), None);
        m.mix_stats(1 << 40, 1 << 39, true, Some(p));
        m.update(TokenId::new(5));
    }
    let code = m.skip(Some(p)).unwrap();
    assert_eq!(code.token, TokenId::new(5));
    assert!(code.q >= 58_982, "{}", code.q); // 0.9 * 2^16
    m.update(TokenId::new(5));
    let higher = m.skip(Some(p)).unwrap().q;
    assert!(higher > code.q);
    m.update(TokenId::new(6)); // a miss
    let lower = m.skip(Some(p)).map_or(0, |c| c.q);
    assert!(lower < higher);
    // Another bucket (length 3) of the same token has its own entry.
    let short = Prediction { token: TokenId::new(5), length: MatchLen::new(3).unwrap() };
    m.update(TokenId::new(5));
    assert_eq!(m.skip(Some(short)), None);
}

#[test]
fn the_flag_and_the_miss_table_obey_the_rules() {
    let mut m = Mixer::new(small_skip());
    let p = Prediction { token: TokenId::new(2), length: MatchLen::new(5).unwrap() };
    m.mix_stats(100, 50, true, Some(p));
    m.update(TokenId::new(2));
    let code = m.skip(Some(p)).unwrap();
    let [hit, miss] = code.flag_weights();
    assert_eq!(hit + miss, 1 << 16);
    assert!(hit >= 1 && miss >= 1);
    assert_eq!(code.flag_of(TokenId::new(2)), 0);
    assert_eq!(code.flag_of(TokenId::new(3)), 1);
    let mut w = vec![0u64, 7, 9, 0];
    skip_miss_weights(&mut w, TokenId::new(2)).unwrap();
    assert_eq!(w, vec![1, 7, 0, 1]);
    assert_eq!(
        skip_miss_weights(&mut w, TokenId::new(4)).unwrap_err(),
        MatchError::TokenOutOfRange { token: 4, vocab: 4 }
    );
    assert_eq!(skip_miss_weights(&mut [], TokenId::new(0)).unwrap_err(), MatchError::EmptyWeights);
}

#[test]
fn a_reset_ends_a_speculative_run() {
    let mut m = Mixer::new(MixerParams::V2);
    let cp = m.checkpoint();
    m.reset();
    assert_eq!(m.rollback(&cp), Err(MatchError::StaleCheckpoint));
    let fresh = Mixer::new(MixerParams::V2);
    let p = Prediction { token: TokenId::new(1), length: MatchLen::new(2).unwrap() };
    assert_eq!(
        m.clone().mix_stats(9, 3, false, Some(p)),
        fresh.clone().mix_stats(9, 3, false, Some(p))
    );
}

#[test]
fn the_rule_versions_have_their_constants() {
    assert_eq!(rule_constants(1), Some((MatchParams::V1, MixerParams::V1)));
    assert_eq!(rule_constants(2), Some((MatchParams::V1, MixerParams::V2)));
    assert_eq!(rule_constants(0), None);
    assert_eq!(rule_constants(3), None);
    assert_eq!(MixerParams::V2.skip(), Some(SkipParams::V2));
    assert_eq!(SkipParams::new(3_774_873, 2, 0, 16).unwrap(), SkipParams::V2);
    assert!(SkipParams::new((1 << 22) + 1, 0, 0, 16).is_err());
    assert!(SkipParams::new(0, 0, 21, 16).is_err());
    assert!(SkipParams::new(0, 0, 0, 3).is_err());
    assert!(SkipParams::new(0, 0, 0, 25).is_err());
}

#[test]
fn rule_two_skips_most_tokens_of_the_synthetic_log_templates() {
    let toks = synthetic_log(20_261_018, 4000);
    let mut model = MatchModel::new(MatchParams::V1);
    let mut m = Mixer::new(MixerParams::V2);
    let mut rng = SplitMix64(6);
    let (mut skipped, mut hits) = (0, 0);
    for &t in &toks {
        let pred = model.predict();
        let w = weights(&mut rng, 1000);
        if position(&mut m, &w, pred, t).0 == 2 {
            skipped += 1;
            hits += usize::from(pred.map(|p| p.token.get()) == Some(t));
        }
        model.observe(TokenId::new(t));
    }
    // The decisions do not depend on the GPT-2 weights, so this is the count of the golden
    // vector run of `tools/logs/skip.py` (239, measured): the template tokens, not the fields.
    assert_eq!(skipped, 239);
    // A field follows a template token 1 time in 4, so many skips miss here (157 hit, measured).
    assert!(hits * 2 > skipped, "{hits} of {skipped}");
}
