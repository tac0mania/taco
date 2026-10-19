//! Speculative decoding on the CUDA GPU against the CPU (ADR 0017): draft rows of many slots in
//! one call, then a truncation of each slot after its first wrong draft token, with context
//! restarts (overlap `K`, ADR 0015) and truncations before a restart. Each used row must give
//! the weights (or the coder interval) of the CPU session that never ran a wrong token. A
//! one-bit difference fails.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::print_stdout)]

#[path = "../../hardshell/tests/common/mod.rs"]
mod common;
mod support;

use common::builder::{build, tiny_dims};
use common::reference::Dims;
use hardshell::TokenId;
use hardshell::gpt2::{Model, Overlap, Session, SessionConfig};
use hardshell_cuda::{CoderQuery, CudaBatch, CudaModel, CudaSession};
use sha2::{Digest, Sha256};
use support::device;
use taco_coder::{FreqBits, FreqTable};

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<TokenId> {
    let mut r = common::Rng::new(seed);
    (0..n).map(|_| TokenId::new(r.below(vocab as u64) as u32)).collect()
}

/// The SHA-256 of one row of weights.
fn digest(w: &[u64]) -> [u8; 32] {
    let mut h = Sha256::new();
    for x in w {
        h.update(x.to_le_bytes());
    }
    h.finalize().into()
}

/// The CPU weights of a stream: item `i` is the row after input `i` (input 0 is the start).
fn cpu_rows(m: &Model, toks: &[TokenId], o: Overlap) -> Vec<Vec<u64>> {
    let mut s = Session::with_config(m, SessionConfig::new().with_chunk(32).with_overlap(o));
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for part in toks.chunks(32) {
        out.extend(s.step_many(part).unwrap().iter().map(<[u64]>::to_vec));
    }
    out
}

/// A token that is not `t`.
fn wrong(t: TokenId, r: &mut common::Rng, vocab: usize) -> TokenId {
    let other = (u64::from(t.get()) + 1 + r.below(vocab as u64 - 1)) % vocab as u64;
    TokenId::new(other as u32)
}

/// One draft of a slot at the stream length `n_in`: the known token and `d - 1` draft tokens,
/// the first `good` of them correct.
struct Draft {
    slot: usize,
    at: usize,
    len: usize,
    good: usize,
}

/// Drafts for the slots, as one input list.
fn drafts(
    seqs: &[Vec<TokenId>],
    n_in: &[usize],
    max_d: usize,
    r: &mut common::Rng,
    vocab: usize,
) -> (Vec<(usize, TokenId)>, Vec<Draft>) {
    let (mut inputs, mut out) = (Vec::new(), Vec::new());
    for (slot, seq) in seqs.iter().enumerate() {
        let n = n_in[slot];
        if n > seq.len() || r.below(5) == 0 {
            continue;
        }
        let d = (1 + r.below(max_d as u64) as usize).min(seq.len() + 1 - n);
        let good = r.below(d as u64) as usize;
        out.push(Draft { slot, at: inputs.len(), len: d, good });
        for j in 0..d {
            let t = seq[n - 1 + j];
            inputs.push((slot, if j <= good { t } else { wrong(t, r, vocab) }));
        }
    }
    (inputs, out)
}

/// Speculative decoding of `slots` streams with overlap `k` in passes of `rows` rows, against
/// the CPU. Even calls read the weights; odd calls ask the coder op for the interval of the
/// true next token of each valid row.
fn case(dims: Dims, k: u16, slots: usize, rows: usize, seed: u64) {
    let Some(dev) = device("speculative case") else { return };
    let b = build(dims, seed, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let o = Overlap::new(k);
    let n = dims.n_ctx - 1 + 3 * (dims.n_ctx - 1 - usize::from(k)) + 4;
    let seqs: Vec<Vec<TokenId>> =
        (0..slots).map(|j| tokens(seed ^ (j as u64 + 1), n + 3 * j, dims.vocab)).collect();
    let want: Vec<Vec<Vec<u64>>> = seqs.iter().map(|s| cpu_rows(&cpu, s, o)).collect();
    let mut batch = CudaBatch::new(&gpu, slots, rows).unwrap();
    batch.set_overlap(o).unwrap();
    let starts: Vec<(usize, TokenId)> = (0..slots).map(|s| (s, gpu.eot_token())).collect();
    let w = batch.run_weights(&starts).unwrap();
    let v = gpu.vocab_size();
    for s in 0..slots {
        assert_eq!(&w[s * v..(s + 1) * v], &want[s][0][..]);
        assert_eq!(batch.stream_len(s), 1);
    }
    let bits = FreqBits::new(32).unwrap();
    let mut r = common::Rng::new(seed ^ 0xd4af);
    let max_d = (dims.n_ctx - 1 - usize::from(k)).clamp(1, 12);
    let mut n_in = vec![1usize; slots];
    let mut call = 0;
    let mut cut_back = 0;
    while (0..slots).any(|s| n_in[s] <= seqs[s].len()) {
        let (inputs, ds) = drafts(&seqs, &n_in, max_d, &mut r, dims.vocab);
        if call % 2 == 0 {
            let w = batch.run_weights(&inputs).unwrap();
            for d in &ds {
                for j in 0..=d.good {
                    let row = &w[(d.at + j) * v..(d.at + j + 1) * v];
                    let ctx = format!("K {k}, slot {}, len {}, row {j}", d.slot, n_in[d.slot]);
                    assert_eq!(row, &want[d.slot][n_in[d.slot] + j][..], "{ctx}");
                }
            }
        } else {
            let mut queries = vec![CoderQuery::Encode(TokenId::new(0)); inputs.len()];
            for d in &ds {
                for j in 0..=d.good {
                    if let Some(&t) = seqs[d.slot].get(n_in[d.slot] - 1 + j + 1) {
                        queries[d.at + j] = CoderQuery::Encode(t);
                    }
                }
            }
            let iv = batch.run_coded(&inputs, &queries, 32).unwrap();
            for d in &ds {
                for j in 0..=d.good {
                    let CoderQuery::Encode(t) = queries[d.at + j] else { unreachable!() };
                    let table = FreqTable::from_weights(&want[d.slot][n_in[d.slot] + j], bits);
                    let freq: Vec<u64> = table.unwrap().freqs().map(|f| f.get()).collect();
                    let sym = t.get() as usize;
                    let start: u64 = freq[..sym].iter().sum();
                    let got = iv[d.at + j];
                    let ctx = format!("coder, K {k}, slot {}", d.slot);
                    assert_eq!((got.symbol, got.start, got.freq), (t, start, freq[sym]), "{ctx}");
                }
            }
        }
        for d in &ds {
            assert_eq!(batch.stream_len(d.slot), n_in[d.slot] + d.len);
            // Usually all valid rows; sometimes fewer, so a truncation also goes back before a
            // restart.
            let kept =
                if r.below(4) == 0 { 1 + r.below(d.good as u64 + 1) as usize } else { d.good + 1 };
            if kept < d.good + 1 {
                cut_back += 1;
            }
            batch.truncate(d.slot, n_in[d.slot] + kept).unwrap();
            n_in[d.slot] += kept;
        }
        call += 1;
    }
    if max_d > 1 {
        assert!(cut_back > 0);
    }
}

#[test]
fn draft_rows_and_truncations_equal_the_cpu() {
    for k in [0u16, 1, 32, 62] {
        case(tiny_dims(), k, 5, 9, 0x5bec ^ u64::from(k));
    }
}

#[test]
fn draft_rows_in_passes_of_two_rows_equal_the_cpu() {
    // Passes of 2 rows: every draft and every restart crosses passes.
    case(tiny_dims(), 20, 3, 2, 0x5bed);
}

#[test]
fn odd_shapes_truncate_as_the_cpu() {
    let d = Dims { n_layer: 1, n_head: 3, d_model: 60, d_ff: 100, vocab: 300, n_ctx: 20 };
    for k in [0u16, 9, 18] {
        case(d, k, 4, 16, 0x5bee ^ u64::from(k));
    }
}

#[test]
fn a_session_truncates_across_a_restart() {
    // n_ctx 64, K 32: a pass of 8 wrong rows from input 60 crosses the restart; the truncation
    // to 60 inputs goes back before it.
    let Some(dev) = device("a_session_truncates_across_a_restart") else { return };
    let d = tiny_dims();
    let b = build(d, 41, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let o = Overlap::new(32);
    let toks = tokens(42, 160, d.vocab);
    let want = cpu_rows(&cpu, &toks, o);
    let mut batch = CudaBatch::new(&gpu, 1, 64).unwrap();
    batch.set_overlap(o).unwrap();
    let mut inputs = vec![(0, gpu.eot_token())];
    inputs.extend(toks[..59].iter().map(|&t| (0, t)));
    batch.run_weights(&inputs).unwrap();
    let mut r = common::Rng::new(43);
    let bad: Vec<(usize, TokenId)> =
        toks[59..67].iter().map(|&t| (0, wrong(t, &mut r, d.vocab))).collect();
    batch.run_weights(&bad).unwrap();
    assert!(batch.position(0) < 64);
    assert!(batch.truncate(0, 69).is_err());
    batch.truncate(0, 60).unwrap();
    assert_eq!((batch.position(0), batch.stream_len(0)), (60, 60));
    let more: Vec<(usize, TokenId)> = toks[59..90].iter().map(|&t| (0, t)).collect();
    let w = batch.run_weights(&more).unwrap();
    let v = gpu.vocab_size();
    for j in 0..more.len() {
        assert_eq!(digest(&w[j * v..(j + 1) * v]), digest(&want[60 + j]), "row {j}");
    }
    let mut s = CudaSession::with_overlap(&gpu, o).unwrap();
    s.start_block().unwrap();
    for &t in &toks[..70] {
        s.step(t).unwrap();
    }
    s.truncate(63).unwrap();
    assert_eq!(s.stream_len(), 63);
    assert_eq!(s.step(toks[62]).unwrap(), &want[63][..]);
}

/// Real GPT-2 (`TACO_MODEL_DIR`, default `tools/data/out`): 2 slots with the evaluation
/// tokens, drafts of up to 16 rows, overlap 0 and 512. Each valid row is compared with the CPU
/// by its SHA-256.
#[test]
#[ignore = "needs gpt2.hsm and eval_windows.txt and takes some minutes"]
fn real_model_drafts_equal_the_cpu() {
    let Some(dev) = device("real_model_drafts_equal_the_cpu") else { return };
    let root = std::env::var("TACO_MODEL_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/data/out").into());
    let bytes = std::fs::read(format!("{root}/gpt2.hsm")).expect("gpt2.hsm");
    let cpu = Model::load(&bytes).unwrap();
    let gpu = CudaModel::load(&dev, &bytes).unwrap();
    let text = std::fs::read_to_string(format!("{root}/eval_windows.txt")).unwrap();
    let all: Vec<TokenId> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| {
            let ids = l.split_once('\t').unwrap().1.to_owned();
            ids.split_whitespace().map(|t| TokenId::new(t.parse().unwrap())).collect::<Vec<_>>()
        })
        .take(4200)
        .collect();
    let seqs = [all[..2100].to_vec(), all[2100..3400].to_vec()];
    for k in [0u16, 512] {
        let o = Overlap::new(k);
        let want: Vec<Vec<[u8; 32]>> =
            seqs.iter().map(|s| cpu_rows(&cpu, s, o).iter().map(|w| digest(w)).collect()).collect();
        let mut batch = CudaBatch::new(&gpu, 2, 64).unwrap();
        batch.set_overlap(o).unwrap();
        let w = batch.run_weights(&[(0, gpu.eot_token()), (1, gpu.eot_token())]).unwrap();
        let v = gpu.vocab_size();
        assert_eq!(digest(&w[..v]), want[0][0]);
        let mut r = common::Rng::new(0x7ea1 ^ u64::from(k));
        let mut n_in = vec![1usize; 2];
        let (mut rows, mut kept_total, mut passes) = (0usize, 0usize, 0usize);
        while (0..2).any(|s| n_in[s] <= seqs[s].len()) {
            let (inputs, ds) = drafts(&seqs, &n_in, 16, &mut r, v);
            let w = batch.run_weights(&inputs).unwrap();
            rows += inputs.len();
            passes += 1;
            for d in &ds {
                for j in 0..=d.good {
                    let got = digest(&w[(d.at + j) * v..(d.at + j + 1) * v]);
                    assert_eq!(got, want[d.slot][n_in[d.slot] + j], "K {k}, slot {}", d.slot);
                }
                batch.truncate(d.slot, n_in[d.slot] + d.good + 1).unwrap();
                n_in[d.slot] += d.good + 1;
                kept_total += d.good + 1;
            }
        }
        println!("real model, K {k}: {passes} calls, {rows} rows, {kept_total} rows kept, equal");
    }
}
