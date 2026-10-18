//! Helpers that the tests share: splitmix64 and the synthetic log of the golden vectors.

#![allow(dead_code, reason = "each test file uses a part of the helpers")]

use taco_match::{MatchModel, Mixer, Prediction, TokenId};

/// splitmix64, the same generator as `tools/convert/rng.py`.
pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

pub(crate) const VOCAB: usize = 1000;
pub(crate) const NEWLINE: u32 = 198;

/// The synthetic log of `tools/logs/vectors.py`: 8 templates with variable fields.
pub(crate) fn synthetic_log(seed: u64, count: usize) -> Vec<u32> {
    let mut rng = SplitMix64(seed);
    let v = VOCAB as u64;
    let mut templates = Vec::new();
    for _ in 0..8 {
        let n = 5 + rng.below(20);
        templates.push((0..n).map(|_| rng.below(v) as u32).collect::<Vec<_>>());
    }
    let mut out = Vec::new();
    while out.len() < count {
        let t = templates[rng.below(8) as usize].clone();
        for tok in t {
            out.push(tok);
            if rng.below(4) == 0 {
                for _ in 0..1 + rng.below(3) {
                    out.push(rng.below(v) as u32);
                }
            }
        }
        out.push(NEWLINE);
    }
    out.truncate(count);
    out
}

/// The weights of `tools/logs/vectors.py` for one position.
pub(crate) fn weights(rng: &mut SplitMix64, actual: u32) -> Vec<u64> {
    let mut w: Vec<u64> = (0..VOCAB).map(|_| rng.next_u64() >> (20 + rng.below(44))).collect();
    if rng.below(2) == 0 {
        w[actual as usize] = 1 << 40;
    }
    w
}

/// One block through the model and the mixer, in the order of the crate documentation.
/// Returns the predictions and the mixed weights of each position.
pub(crate) fn run_block(
    model: &mut MatchModel,
    mixer: &mut Mixer,
    tokens: &[u32],
    mut make_weights: impl FnMut(u32) -> Vec<u64>,
) -> (Vec<Option<Prediction>>, Vec<Vec<u64>>) {
    let mut preds = Vec::new();
    let mut outs = Vec::new();
    for &t in tokens {
        let p = model.predict();
        let mut w = make_weights(t);
        mixer.mix(&mut w, p).unwrap();
        mixer.update(TokenId::new(t));
        model.observe(TokenId::new(t));
        preds.push(p);
        outs.push(w);
    }
    (preds, outs)
}
