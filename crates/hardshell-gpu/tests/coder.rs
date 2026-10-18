//! The coder op on the GPU against `taco_coder::FreqTable::from_weights` on the CPU weights:
//! the interval of a token (encode) and the symbol of a target (decode), bit for bit.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::print_stderr)]

#[path = "../../hardshell/tests/common/mod.rs"]
mod common;
mod support;

use common::builder::{build, tiny_dims};
use hardshell::TokenId;
use hardshell::gpt2::{Model, Session};
use hardshell_gpu::{CoderInterval, CoderQuery, GpuBatch, GpuModel};
use support::device;
use taco_coder::{FreqBits, FreqTable};

/// The starts and frequencies of the CPU table.
fn cpu_table(w: &[u64], bits: u32) -> (Vec<u64>, Vec<u64>) {
    let t = FreqTable::from_weights(w, FreqBits::new(bits).unwrap()).unwrap();
    let freqs: Vec<u64> = t.freqs().map(|f| f.get()).collect();
    let mut starts = Vec::with_capacity(freqs.len());
    let mut acc = 0;
    for f in &freqs {
        starts.push(acc);
        acc += f;
    }
    assert_eq!(acc, 1u64 << bits);
    (starts, freqs)
}

fn expect(starts: &[u64], freqs: &[u64], q: CoderQuery) -> CoderInterval {
    let i = match q {
        CoderQuery::Encode(t) => t.get() as usize,
        CoderQuery::Decode(x) => starts.partition_point(|&s| s <= x) - 1,
        _ => unreachable!(),
    };
    CoderInterval {
        symbol: TokenId::new(i as u32),
        start: starts[i],
        freq: freqs[i],
        is_last: i + 1 == starts.len(),
    }
}

#[test]
fn coder_intervals_equal_the_cpu_table() {
    let Some(dev) = device("coder_intervals_equal_the_cpu_table") else { return };
    let b = build(tiny_dims(), 31, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = GpuModel::load(&dev, &b.bytes).unwrap();
    let slots = 6;
    let mut sessions: Vec<Session<'_>> = (0..slots).map(|_| Session::new(&cpu)).collect();
    let mut batch = GpuBatch::new(&gpu, slots).unwrap();
    let mut r = common::Rng::new(0xc0de);
    let mut checked = 0;
    for step in 0..120 {
        // 32 bits as in taco; 9 bits leaves only 212 spare units for 300 symbols, so the
        // remainder rule matters.
        let bits = [32u32, 12, 9][step % 3];
        let total = 1u64 << bits;
        let mut inputs = Vec::new();
        let mut queries = Vec::new();
        let mut wants = Vec::new();
        for (slot, s) in sessions.iter_mut().enumerate() {
            let token = if step == 0 { cpu.eot_token() } else { TokenId::new(r.below(300) as u32) };
            let w = s.step(token).unwrap();
            let (starts, freqs) = cpu_table(w, bits);
            let q = match r.below(5) {
                0 => CoderQuery::Decode(0),
                1 => CoderQuery::Decode(total - 1),
                2 => {
                    // A target at the start of a symbol, or just below it.
                    let i = 1 + r.below(299) as usize;
                    CoderQuery::Decode(starts[i] - r.below(2))
                }
                3 => CoderQuery::Decode(r.below(total)),
                _ => CoderQuery::Encode(TokenId::new(r.below(300) as u32)),
            };
            wants.push(expect(&starts, &freqs, q));
            inputs.push((slot, token));
            queries.push(q);
        }
        let got = batch.step_coded(&inputs, &queries, bits).unwrap();
        assert_eq!(got, wants, "step {step}, bits {bits}");
        checked += got.len();
    }
    eprintln!("coder_intervals_equal_the_cpu_table: {checked} intervals equal");
}
