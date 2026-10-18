//! The cost of the match model and the mixer for each token, with the GPT-2 vocabulary.
//!
//! Run: `cargo run --release -p taco-match --example cost`. The input is a synthetic log of
//! 20,000 tokens and HardShell-like weights (at most 2^40 each) for 50,257 tokens.

#![allow(
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_arithmetic,
    reason = "a measurement program, not library code"
)]

use std::time::Instant;

use taco_match::{MatchModel, MatchParams, Mixer, MixerParams, TokenId};

const VOCAB: usize = 50_257;

fn main() {
    let mut s = 0x1234_5678_u64;
    let mut next = move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let template: Vec<u32> = (0..30).map(|_| (next() % 50_000) as u32).collect();
    let mut tokens = Vec::new();
    while tokens.len() < 20_000 {
        for &t in &template {
            tokens.push(if next() % 5 == 0 { (next() % 50_000) as u32 } else { t });
        }
    }
    let base: Vec<u64> = (0..VOCAB).map(|_| next() >> (24 + next() % 40)).collect();
    let mut model = MatchModel::new(MatchParams::V1);
    let mut mixer = Mixer::new(MixerParams::V1);
    let mut w = base.clone();
    let t0 = Instant::now();
    let mut predicted = 0usize;
    for &t in &tokens {
        w.copy_from_slice(&base);
        let p = model.predict();
        predicted += usize::from(p.is_some());
        mixer.mix(&mut w, p).unwrap();
        mixer.update(TokenId::new(t));
        model.observe(TokenId::new(t));
    }
    let all = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    for _ in &tokens {
        w.copy_from_slice(&base);
    }
    let copy = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    model.reset();
    mixer.reset();
    let reset = t2.elapsed().as_secs_f64();
    let n = tokens.len() as f64;
    println!(
        "{} tokens, {} with a prediction: {:.1} us for each token (the weight copy {:.1} us \
         not included), reset {:.2} ms",
        tokens.len(),
        predicted,
        (all - copy) / n * 1e6,
        copy / n * 1e6,
        reset * 1e3
    );
}
