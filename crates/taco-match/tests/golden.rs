//! Golden vectors: the SHA-256 of the predictions and of the mixed weights for a synthetic
//! log (plan §11). `tools/logs/vectors.py` (the Python mirror, used for the measurements of
//! ADR 0010) gave the same values on 2026-10-18 (measured). A change to a rule of the match
//! model or the mixer changes a hash and needs a new mixer rule version (ADR 0010).

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{SplitMix64, run_block, synthetic_log, weights};
use sha2::{Digest, Sha256};
use taco_match::{MatchModel, MatchParams, Mixer, MixerParams, SkipParams, TokenId};

const SEED: u64 = 20_261_018;

fn hex(d: &[u8]) -> String {
    d.iter().map(|x| format!("{x:02x}")).collect()
}

fn hashes(params: MatchParams) -> (String, String) {
    let tokens = synthetic_log(SEED, 4000);
    let mut model = MatchModel::new(params);
    let mut mixer = Mixer::new(MixerParams::V1);
    let mut rng = SplitMix64(SEED ^ 0xA5A5);
    let (preds, outs) = run_block(&mut model, &mut mixer, &tokens, |t| weights(&mut rng, t));
    let mut hp = Sha256::new();
    for p in &preds {
        let (t, l) = p.map_or((u32::MAX, 0), |p| (p.token.get(), p.length.get()));
        hp.update(t.to_le_bytes());
        hp.update(l.to_le_bytes());
    }
    let mut hw = Sha256::new();
    for w in &outs {
        for x in w {
            hw.update(x.to_le_bytes());
        }
    }
    (hex(&hp.finalize()), hex(&hw.finalize()))
}

#[test]
fn v1_parameters_match_the_python_mirror() {
    let (p, w) = hashes(MatchParams::V1);
    assert_eq!(p, "a7ccc360407bc350f5444defc5dcb8b173ec6e7ee8c17b02998d70c20c73b23d");
    assert_eq!(w, "31d87e9efc45a331b7019e4c8f6058cea247637a5a29a3b0016981407821d614");
}

#[test]
fn small_tables_match_the_python_mirror() {
    // Tables of 2^6 entries: many hash collisions, so the verification rule matters.
    let (p, w) = hashes(MatchParams::new(&[8, 4, 2, 1], 6, 16, 40).unwrap());
    assert_eq!(p, "f2e7ff6ecc83ea8e47192c7c7e1f17118d23c93ba4c48da99be9401ddce31e51");
    assert_eq!(w, "f37ecdd0307952da684e90a350488d056b8d2b498fb1ac603168dd74baf5d51b");
}

/// The decisions of mixer rule version 2 (ADR 0017) on the synthetic log: for each position,
/// 0 (no prediction), 1 and `q` (mixed), or 2 and `q_s` (skipped), as `u8` and `u32` LE.
fn skip_hashes(params: MixerParams) -> (usize, String) {
    let tokens = synthetic_log(SEED, 4000);
    let mut model = MatchModel::new(MatchParams::V1);
    let mut mixer = Mixer::new(params);
    let mut rng = SplitMix64(SEED ^ 0xA5A5);
    let mut h = Sha256::new();
    let mut skipped = 0;
    for &t in &tokens {
        let pred = model.predict();
        let w = weights(&mut rng, t);
        let (kind, q) = match pred {
            None => (0u8, 0u32),
            Some(p) => match mixer.skip(Some(p)) {
                Some(code) => {
                    skipped += 1;
                    (2, code.q)
                }
                None => {
                    let total: u128 = w.iter().map(|&x| u128::from(x)).sum();
                    let mut top = 0;
                    for (i, &x) in w.iter().enumerate() {
                        if x > w[top] {
                            top = i;
                        }
                    }
                    let pi = p.token.get() as usize;
                    (1, mixer.mix_stats(total, w[pi], top == pi, Some(p)).unwrap())
                }
            },
        };
        h.update([kind]);
        h.update(q.to_le_bytes());
        mixer.update(TokenId::new(t));
        model.observe(TokenId::new(t));
    }
    (skipped, hex(&h.finalize()))
}

#[test]
fn rule_two_decisions_match_the_python_mirror() {
    // `python -m logs.skip vectors` (tools/logs/skip.py) gave the same values (measured,
    // devbox, 2026-10-19).
    let (n, d) = skip_hashes(MixerParams::V2);
    assert_eq!(n, 239);
    assert_eq!(d, "f6f7755bc8649d13bd2548b92408bfa41ab29ae42c0ec3dc22c74587f99f930f");
    let small = MixerParams::V1.with_skip(SkipParams::new(1 << 21, 1, 2, 6).unwrap());
    let (n, d) = skip_hashes(small);
    assert_eq!(n, 453);
    assert_eq!(d, "8781e1040993e54e93078988dbd392a867a00d2f9d2bab2294d764241f38f212");
}
