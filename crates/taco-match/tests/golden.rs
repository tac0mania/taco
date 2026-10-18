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
use taco_match::{MatchModel, MatchParams, Mixer, MixerParams};

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
