//! The mixed coder op (the profile `logs`, ADR 0010) against the CPU: `taco_match::Mixer::mix`
//! and `taco_coder::FreqTable::from_weights`. A one-bit difference fails.
//!
//! - The batch op on the GPT-2 step: the statistics and every interval of random calls (decode
//!   and prefill rows, restarts, a context overlap, plain rows among mixed rows, and the dual
//!   op of the encoder), on the op-level path and on the fused paths (dp4a and tensor-core
//!   products), with the real mixer state of each slot.
//! - The op on given weights: zero weights, all-zero rows, one-hot rows, weights above 2^48
//!   (the u128 loop of the mixer), every edge of `q`, and a vocabulary of 50,257.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::print_stderr)]

#[path = "../../hardshell/tests/common/mod.rs"]
mod common;
mod support;

use common::builder::{build, build_wide, tiny_dims};
use hardshell::TokenId;
use hardshell::gpt2::{Model, Overlap, Session, SessionConfig};
use hardshell_cuda::{
    CoderInterval, CoderQuery, CudaBatch, CudaModel, MixInput, MixStats, mixed_intervals_of_weights,
};
use support::{Rng, device};
use taco_coder::{FreqBits, FreqTable};
use taco_match::{MatchLen, Mixer, MixerParams, Prediction};

/// The interval of `q` in the table of `w` (the table path of the CPU).
fn cpu_interval(w: &[u64], bits: u32, q: CoderQuery) -> CoderInterval {
    let t = FreqTable::from_weights(w, FreqBits::new(bits).unwrap()).unwrap();
    let freqs: Vec<u64> = t.freqs().map(|f| f.get()).collect();
    let mut starts = Vec::with_capacity(freqs.len());
    let mut acc = 0;
    for f in &freqs {
        starts.push(acc);
        acc += f;
    }
    let i = match q {
        CoderQuery::Encode(t) => t.get() as usize,
        CoderQuery::Decode(x) => starts.partition_point(|&s| s <= x) - 1,
        _ => unreachable!(),
    };
    CoderInterval {
        symbol: TokenId::new(i as u32),
        start: starts[i],
        freq: freqs[i],
        is_last: i + 1 == freqs.len(),
    }
}

/// Pass 1 of `Mixer::mix`: (W, w[p], top) with the loop that keeps the first strictly larger
/// weight, from token 0 with weight 0.
fn cpu_stats(w: &[u64], p: Option<usize>) -> (u128, u64, usize) {
    let mut total = 0u128;
    let (mut top, mut top_w) = (0usize, 0u64);
    for (i, &x) in w.iter().enumerate() {
        total += u128::from(x);
        if x > top_w {
            top = i;
            top_w = x;
        }
    }
    (total, p.map_or(0, |p| w[p]), top)
}

/// Rule 5 of ADR 0010 section 3 for a given `q`, written out (rule 1 without a prediction).
/// The batch test proves it equal to `Mixer::mix` for the `q` of the mixer.
fn apply_ref(w: &[u64], p: Option<usize>, q: u32) -> Vec<u64> {
    let Some(p) = p else { return w.iter().map(|&x| x.max(1)).collect() };
    let (total, wp, _) = cpu_stats(w, Some(p));
    let r = (total - u128::from(wp)).max(1);
    let m = u128::from(65_536 - q);
    let k = (128 - r.leading_zeros() + 16).saturating_sub(62);
    let mut out: Vec<u64> =
        w.iter().map(|&x| u64::try_from((u128::from(x) * m) >> k).unwrap().max(1)).collect();
    out[p] = u64::try_from((r * u128::from(q)) >> k).unwrap().max(1);
    out
}

fn check_stats(got: MixStats, w: &[u64], p: usize, what: &str) {
    let (total, wp, top) = cpu_stats(w, Some(p));
    assert_eq!(u128::from(got.total), total, "{what}: W");
    assert_eq!(got.predicted, wp, "{what}: w[p]");
    assert_eq!(got.top, TokenId::new(top as u32), "{what}: top");
}

fn random_query(r: &mut Rng, bits: u32, vocab: u64) -> CoderQuery {
    let total = 1u64 << bits;
    match r.below(4) {
        0 => CoderQuery::Decode(0),
        1 => CoderQuery::Decode(total - 1),
        2 => CoderQuery::Decode(r.below(total)),
        _ => CoderQuery::Encode(TokenId::new(r.below(vocab) as u32)),
    }
}

/// One expected row of the batch test.
struct Want {
    /// `None`: a plain input (the table of GPT-2 alone).
    table: Option<Option<usize>>,
    q: u32,
    weights: Vec<u64>,
    plain: CoderInterval,
    interval: CoderInterval,
}

/// Random calls on a batch of `slots` slots: each slot has a CPU session and a real mixer.
/// For each input: a plain table (not with `dual`), or a mixed table whose prediction is none,
/// the GPT-2 top token, a token of low weight or a random token, with a random match length;
/// the actual token updates the mixer. With `dual`, every query encodes a token and the batch
/// gives both intervals (`run_coded_dual`).
fn batch_against_cpu(slots: usize, calls: usize, op_level: bool, overlap: u16, dual: bool) {
    batch_against_cpu_of(false, slots, calls, op_level, overlap, dual);
}

/// [`batch_against_cpu`] on the tiny model of model-file version 1, or with `wide` of version 2
/// (ADR 0004 addendum B: i16 GELU and ln_f outputs, ln_beta_frac).
fn batch_against_cpu_of(
    wide: bool,
    slots: usize,
    calls: usize,
    op_level: bool,
    overlap: u16,
    dual: bool,
) {
    let Some(dev) = device("batch_against_cpu") else { return };
    let b = if wide { build_wide(tiny_dims(), 51, 64) } else { build(tiny_dims(), 51, 64) };
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let cfg = SessionConfig::new().with_overlap(Overlap::new(overlap));
    let mut sessions: Vec<Session<'_>> =
        (0..slots).map(|_| Session::with_config(&cpu, cfg)).collect();
    let mut mixers: Vec<Mixer> = (0..slots).map(|_| Mixer::new(MixerParams::V1)).collect();
    let mut batch = CudaBatch::new(&gpu, slots, 2 * slots).unwrap();
    batch.set_op_level(op_level);
    batch.set_overlap(Overlap::new(overlap)).unwrap();
    let mut r = Rng::new(0x31c5 ^ slots as u64 ^ u64::from(overlap) ^ u64::from(dual));
    let vocab = gpu.vocab_size();
    let (mut agree, mut predicted, mut plain_rows) = (0usize, 0usize, 0usize);
    for call in 0..calls {
        let bits = [32u32, 12, 9][call % 3];
        let (mut inputs, mut queries, mut tables, mut wants) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::<Want>::new());
        for _ in 0..2 {
            for (slot, s) in sessions.iter_mut().enumerate() {
                if r.below(2) == 0 && !inputs.is_empty() {
                    continue;
                }
                let token = if call == 0 && !inputs.iter().any(|&(x, _)| x == slot) {
                    cpu.eot_token()
                } else {
                    TokenId::new(r.below(300) as u32)
                };
                let w = s.step(token).unwrap().to_vec();
                let (_, _, top) = cpu_stats(&w, None);
                let low = (0..vocab).min_by_key(|&i| w[i]).unwrap();
                let table = match r.below(6) {
                    0 if !dual => None,
                    0 | 1 => Some(None),
                    2 | 3 => Some(Some(top)),
                    4 => Some(Some(low)),
                    _ => Some(Some(r.below(vocab as u64) as usize)),
                };
                let query = if dual {
                    CoderQuery::Encode(TokenId::new(r.below(vocab as u64) as u32))
                } else {
                    random_query(&mut r, bits, vocab as u64)
                };
                let mixer = &mut mixers[slot];
                let (q, mixed) = match table {
                    None => {
                        plain_rows += 1;
                        (0, w.clone())
                    }
                    Some(None) => {
                        let mut m = w.clone();
                        mixer.mix(&mut m, None).unwrap();
                        (0, m)
                    }
                    Some(Some(p)) => {
                        predicted += 1;
                        agree += usize::from(p == top);
                        let len = MatchLen::new(1 + r.below(300) as u32).unwrap();
                        let (total, wp, _) = cpu_stats(&w, Some(p));
                        let q = mixer.peek(wp, total, len, p == top);
                        let mut m = w.clone();
                        let prediction =
                            Prediction { token: taco_match::TokenId::new(p as u32), length: len };
                        mixer.mix(&mut m, Some(prediction)).unwrap();
                        (q, m)
                    }
                };
                if let Some(pred) = table {
                    // The written-out rule 5 is the rule of the mixer.
                    assert_eq!(apply_ref(&w, pred, q), mixed, "call {call}: apply_ref");
                    let actual =
                        if r.below(2) == 0 { pred.unwrap_or(0) } else { r.below(300) as usize };
                    mixer.update(taco_match::TokenId::new(actual as u32));
                }
                wants.push(Want {
                    table,
                    q,
                    plain: cpu_interval(&w, bits, query),
                    interval: cpu_interval(&mixed, bits, query),
                    weights: w,
                });
                inputs.push((slot, token));
                queries.push(query);
                tables.push(match table {
                    None => MixInput::Plain,
                    Some(p) => MixInput::Mixed(p.map(|p| TokenId::new(p as u32))),
                });
            }
        }
        let mut calls_seen = Vec::new();
        let host = |i: usize, st: MixStats| {
            let want = &wants[i];
            let p = want.table.flatten().unwrap();
            check_stats(st, &want.weights, p, &format!("call {call} input {i}"));
            calls_seen.push(i);
            want.q
        };
        let what = format!("call {call} bits {bits} op_level {op_level} dual {dual} wide {wide}");
        if dual {
            let tokens: Vec<TokenId> = queries
                .iter()
                .map(|q| match *q {
                    CoderQuery::Encode(t) => t,
                    _ => unreachable!(),
                })
                .collect();
            let preds: Vec<Option<TokenId>> = tables
                .iter()
                .map(|t| match *t {
                    MixInput::Mixed(p) => p,
                    _ => unreachable!(),
                })
                .collect();
            let got = batch.run_coded_dual(&inputs, &tokens, &preds, bits, host).unwrap();
            let want_iv: Vec<(CoderInterval, CoderInterval)> =
                wants.iter().map(|w| (w.plain, w.interval)).collect();
            assert_eq!(got, want_iv, "{what}");
        } else {
            let got = batch.run_coded_mixed(&inputs, &queries, &tables, bits, host).unwrap();
            let want_iv: Vec<CoderInterval> = wants.iter().map(|w| w.interval).collect();
            assert_eq!(got, want_iv, "{what}");
        }
        let expected: Vec<usize> =
            (0..wants.len()).filter(|&i| wants[i].table.flatten().is_some()).collect();
        assert_eq!(calls_seen, expected, "call {call}: the mixer calls in input order");
        for (slot, s) in sessions.iter().enumerate() {
            assert_eq!(batch.position(slot), s.position(), "call {call} slot {slot}");
        }
    }
    eprintln!(
        "batch_against_cpu: {predicted} predictions, {agree} on the top token, {plain_rows} plain rows"
    );
}

#[test]
fn mixed_intervals_equal_the_cpu_mixer_on_the_op_level_path() {
    batch_against_cpu(4, 45, true, 0, false);
    batch_against_cpu(4, 30, true, 0, true);
}

#[test]
fn mixed_intervals_equal_the_cpu_mixer_on_the_fused_paths() {
    // 4 slots: passes of at most 8 rows (dp4a products, the i64 logits); 40 slots: passes of up
    // to 80 rows (tensor-core products, the compact logits).
    batch_against_cpu(4, 60, false, 0, false);
    batch_against_cpu(40, 20, false, 0, false);
}

#[test]
fn dual_intervals_equal_the_cpu_tables_on_the_fused_paths() {
    batch_against_cpu(4, 45, false, 0, true);
    batch_against_cpu(40, 15, false, 0, true);
}

#[test]
fn mixed_intervals_equal_the_cpu_mixer_with_a_context_overlap() {
    batch_against_cpu(3, 70, false, 9, false);
    batch_against_cpu(3, 40, false, 9, true);
}

#[test]
fn mixed_and_dual_intervals_on_a_wide_model_equal_the_cpu() {
    // Model-file version 2: the op-level path, the fused path with dp4a products (4 slots) and
    // with tensor-core products (40 slots; i64 logits, no compact logits), and an overlap.
    batch_against_cpu_of(true, 4, 30, true, 0, false);
    batch_against_cpu_of(true, 4, 45, false, 0, false);
    batch_against_cpu_of(true, 40, 15, false, 0, false);
    batch_against_cpu_of(true, 4, 30, false, 0, true);
    batch_against_cpu_of(true, 40, 10, false, 0, true);
    batch_against_cpu_of(true, 3, 50, false, 9, false);
}

#[test]
fn mixed_op_refuses_bad_arguments() {
    let Some(dev) = device("mixed_op_refuses_bad_arguments") else { return };
    let b = build(tiny_dims(), 52, 64);
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let mut batch = CudaBatch::new(&gpu, 1, 4).unwrap();
    let inputs = [(0, gpu.eot_token())];
    let enc = [CoderQuery::Encode(TokenId::new(1))];
    assert!(batch.run_coded_mixed(&inputs, &enc, &[], 16, |_, _| 1).is_err());
    let bad_token = [MixInput::Mixed(Some(TokenId::new(300)))];
    assert!(batch.run_coded_mixed(&inputs, &enc, &bad_token, 16, |_, _| 1).is_err());
    let p = [MixInput::Mixed(Some(TokenId::new(3)))];
    assert!(batch.run_coded_mixed(&inputs, &enc, &p, 16, |_, _| 0).is_err());
    assert!(batch.run_coded_mixed(&inputs, &enc, &p, 16, |_, _| 65_536).is_err());
    assert_eq!(batch.position(0), 0, "an error changes no position");
    assert!(batch.run_coded_mixed(&inputs, &enc, &p, 16, |_, _| 65_535).is_ok());
    assert_eq!(batch.position(0), 1);
}

/// A row of weights of one of several shapes.
fn weight_row(r: &mut Rng, vocab: usize, shape: u64) -> Vec<u64> {
    match shape {
        // GPT-2-like: at most 2^40, a few zeros.
        0 => (0..vocab)
            .map(|_| {
                let e = 1 + r.below(40);
                if r.below(20) == 0 { 0 } else { r.below(1 << e) }
            })
            .collect(),
        // All zero.
        1 => vec![0; vocab],
        // One-hot: W = w[p] for the hot token.
        2 => {
            let mut w = vec![0; vocab];
            w[r.below(vocab as u64) as usize] = 1 + r.below(1 << 40);
            w
        }
        // Large weights (above 2^48: the u128 loop of the mixer), the sum below 2^64.
        3 => {
            let cap = (u64::MAX / vocab as u64) >> 1;
            (0..vocab).map(|_| r.below(cap)).collect()
        }
        // Ties: few distinct values.
        _ => (0..vocab).map(|_| [0, 5, 7, 7][r.below(4) as usize]).collect(),
    }
}

fn given_weights(vocab: usize, rows: usize, bits: &[u32], seed: u64) {
    let Some(dev) = device("given_weights") else { return };
    let mut r = Rng::new(seed);
    for (round, &bits) in bits.iter().enumerate() {
        let mut weights = Vec::with_capacity(rows * vocab);
        let (mut queries, mut preds, mut qs, mut wants) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for row in 0..rows {
            let w = weight_row(&mut r, vocab, (row as u64 + round as u64) % 5);
            let (_, _, top) = cpu_stats(&w, None);
            let pred = match r.below(5) {
                0 => None,
                1 => Some(top),
                2 => Some((top + 1) % vocab),
                _ => Some(r.below(vocab as u64) as usize),
            };
            let q = match r.below(7) {
                0 => 1,
                1 => 2,
                2 => 32_768,
                3 => 65_534,
                4 => 65_535,
                _ => 1 + r.below(65_535) as u32,
            };
            let query = random_query(&mut r, bits, vocab as u64);
            wants.push(cpu_interval(&apply_ref(&w, pred, q), bits, query));
            weights.extend_from_slice(&w);
            queries.push(query);
            preds.push(pred.map(|p| TokenId::new(p as u32)));
            qs.push(q);
        }
        let got = mixed_intervals_of_weights(&dev, &weights, &queries, &preds, bits, |i, _| qs[i])
            .unwrap();
        for (i, ((st, iv), want)) in got.iter().zip(&wants).enumerate() {
            let w = &weights[i * vocab..(i + 1) * vocab];
            if let Some(p) = preds[i] {
                check_stats(*st, w, p.get() as usize, &format!("round {round} row {i}"));
            }
            assert_eq!(iv, want, "round {round} row {i} bits {bits} q {}", qs[i]);
        }
    }
}

#[test]
fn mixed_op_on_given_weights_equals_the_cpu_table() {
    given_weights(300, 120, &[9, 16, 32], 0x6d78);
}

#[test]
fn mixed_op_on_gpt2_sized_rows_equals_the_cpu_table() {
    given_weights(50_257, 25, &[16, 32], 0x6d79);
}
