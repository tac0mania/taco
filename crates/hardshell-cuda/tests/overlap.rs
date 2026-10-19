//! The context rule with overlap (ADR 0015) on the CUDA GPU against the CPU: the golden
//! overlap hash (step and prefill), many slots with restarts in passes of a few rows, the
//! shapes of the attention kernels, the coder op, and real GPT-2 (ignored by default). A
//! one-bit difference fails.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::print_stderr)]
#![allow(clippy::print_stdout)]

#[path = "../../hardshell/tests/common/mod.rs"]
mod common;
mod support;

use common::builder::{build, tiny_dims};
use common::reference::Dims;
use hardshell::TokenId;
use hardshell::gpt2::{Model, Overlap, ReferenceSession, Session, SessionConfig};
use hardshell_cuda::{CoderQuery, CudaBatch, CudaModel, CudaSession};
use sha2::{Digest, Sha256};
use support::device;
use taco_coder::{FreqBits, FreqTable};

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<TokenId> {
    let mut r = common::Rng::new(seed);
    (0..n).map(|_| TokenId::new(r.below(vocab as u64) as u32)).collect()
}

fn hex(h: &[u8]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

/// The hash of `golden_overlap_conformance_hash` in `crates/hardshell/tests/context.rs`.
const GOLDEN_OVERLAP_OUTPUT: &str =
    "5a79928200ad35bb7e7a1be2a0a35e05dcecc001a89cf3cf0765dfc7e5304a83";

#[test]
fn golden_overlap_hash_on_cuda() {
    let Some(dev) = device("golden_overlap_hash_on_cuda") else { return };
    let b = build(tiny_dims(), 0x7ac0_0001, 64);
    let m = CudaModel::load(&dev, &b.bytes).unwrap();
    let mut s = CudaSession::with_overlap(&m, Overlap::new(32)).unwrap();
    let mut h = Sha256::new();
    for x in s.start_block().unwrap() {
        h.update(x.to_le_bytes());
    }
    for t in tokens(0x7ac0_0002, 300, 300) {
        for x in s.step(t).unwrap() {
            h.update(x.to_le_bytes());
        }
    }
    assert_eq!(hex(&h.finalize()), GOLDEN_OVERLAP_OUTPUT);
}

#[test]
fn golden_overlap_hash_with_prefill() {
    // The same vector in one call: 301 rows of one slot and 8 restarts of 33 rows each, in
    // passes of 40 or 7 rows, so restarts cross passes.
    let Some(dev) = device("golden_overlap_hash_with_prefill") else { return };
    let b = build(tiny_dims(), 0x7ac0_0001, 64);
    let m = CudaModel::load(&dev, &b.bytes).unwrap();
    for rows in [40, 7, 2] {
        let mut batch = CudaBatch::new(&m, 1, rows).unwrap();
        batch.set_overlap(Overlap::new(32)).unwrap();
        let mut inputs = vec![(0, m.eot_token())];
        inputs.extend(tokens(0x7ac0_0002, 300, 300).into_iter().map(|t| (0, t)));
        let w = batch.run_weights(&inputs).unwrap();
        let mut h = Sha256::new();
        for x in w {
            h.update(x.to_le_bytes());
        }
        assert_eq!(hex(&h.finalize()), GOLDEN_OVERLAP_OUTPUT, "passes of {rows} rows");
    }
}

#[test]
fn an_overlap_that_does_not_fit_is_refused() {
    let Some(dev) = device("an_overlap_that_does_not_fit_is_refused") else { return };
    let b = build(tiny_dims(), 3, 64);
    let m = CudaModel::load(&dev, &b.bytes).unwrap();
    let mut batch = CudaBatch::new(&m, 1, 8).unwrap();
    assert!(batch.set_overlap(Overlap::new(62)).is_ok());
    assert!(batch.set_overlap(Overlap::new(63)).is_err());
    assert_eq!(batch.overlap(), Overlap::new(62));
    assert!(CudaSession::with_overlap(&m, Overlap::new(63)).is_err());
}

/// CPU sessions and one GPU batch with overlap `k` and the same token streams. Each call gives
/// each slot 0 to `run` tokens; some slots start a new block.
fn batch_against_sessions(dims: Dims, k: u16, slots: usize, calls: usize, run: u64, rows: usize) {
    let Some(dev) = device("batch_against_sessions") else { return };
    let seed = 0x0e1 ^ u64::from(k) ^ ((dims.n_ctx as u64) << 8);
    let b = build(dims, seed, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let o = Overlap::new(k);
    let cfg = SessionConfig::new().with_overlap(o);
    let mut sessions: Vec<Session<'_>> =
        (0..slots).map(|_| Session::with_config(&cpu, cfg)).collect();
    let mut batch = CudaBatch::new(&gpu, slots, rows).unwrap();
    batch.set_overlap(o).unwrap();
    let mut r = common::Rng::new(seed ^ 0x5eed);
    let v = gpu.vocab_size();
    let mut restarts = 0;
    for call in 0..calls {
        let (mut inputs, mut wants) = (Vec::new(), Vec::new());
        for (slot, s) in sessions.iter_mut().enumerate() {
            if r.below(4) == 0 {
                continue;
            }
            if r.below(53) == 0 {
                batch.reset(slot);
                s.reset();
                wants.push(s.step(gpu.eot_token()).unwrap().to_vec());
                inputs.push((slot, gpu.eot_token()));
            }
            for _ in 0..r.below(run + 1) {
                if s.position() >= dims.n_ctx {
                    restarts += 1;
                }
                let token = TokenId::new(r.below(dims.vocab as u64) as u32);
                wants.push(s.step(token).unwrap().to_vec());
                inputs.push((slot, token));
            }
        }
        let got = batch.run_weights(&inputs).unwrap();
        for (i, want) in wants.iter().enumerate() {
            assert_eq!(&got[i * v..(i + 1) * v], &want[..], "K {k}, call {call} input {i}");
        }
        for (slot, s) in sessions.iter().enumerate() {
            assert_eq!(batch.position(slot), s.position(), "K {k}, call {call} slot {slot}");
        }
    }
    eprintln!("K {k}, n_ctx {}: {restarts} restarts", dims.n_ctx);
    assert!(restarts >= 3 * slots, "the test needs several restarts in each slot");
}

#[test]
fn slots_with_overlap_equal_cpu_sessions() {
    // The tiny model (n_ctx 64): K from 0 to 62; decode (one token) and prefill (up to 9
    // tokens) together, in passes of 16 rows: every restart crosses passes.
    for k in [0, 1, 16, 32, 48, 62] {
        batch_against_sessions(tiny_dims(), k, 4, 120, 9, 16);
    }
}

#[test]
fn overlap_with_the_shapes_of_the_attention_kernels() {
    // head_dim 64 (attend_rows), n_ctx 100 (not a multiple of 32), passes of 37 rows.
    let d64 = Dims { n_layer: 1, n_head: 3, d_model: 192, d_ff: 256, vocab: 300, n_ctx: 100 };
    batch_against_sessions(d64, 50, 3, 90, 12, 37);
    batch_against_sessions(d64, 98, 3, 150, 4, 37);
    // head_dim 12 (one thread for each key), n_ctx 21, passes of 5 rows.
    let d12 = Dims { n_layer: 1, n_head: 4, d_model: 48, d_ff: 96, vocab: 300, n_ctx: 21 };
    batch_against_sessions(d12, 10, 3, 80, 5, 5);
}

#[test]
fn coder_intervals_with_overlap_equal_the_cpu_table() {
    let Some(dev) = device("coder_intervals_with_overlap_equal_the_cpu_table") else { return };
    let b = build(tiny_dims(), 31, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let o = Overlap::new(40);
    let mut s = Session::with_config(&cpu, SessionConfig::new().with_overlap(o));
    let mut batch = CudaBatch::new(&gpu, 1, 24).unwrap();
    batch.set_overlap(o).unwrap();
    let toks = tokens(0xc0de, 200, 300);
    let mut inputs = vec![(0, gpu.eot_token())];
    inputs.extend(toks.iter().map(|&t| (0, t)));
    let mut wants = vec![s.start_block().unwrap().to_vec()];
    wants.extend(toks.iter().map(|&t| s.step(t).unwrap().to_vec()));
    let queries: Vec<CoderQuery> =
        toks.iter().chain(std::iter::once(&toks[0])).map(|&t| CoderQuery::Encode(t)).collect();
    let got = batch.run_coded(&inputs, &queries, 32).unwrap();
    for (i, (g, w)) in got.iter().zip(&wants).enumerate() {
        let t = FreqTable::from_weights(w, FreqBits::new(32).unwrap()).unwrap();
        let sym = g.symbol.get() as usize;
        let freq: Vec<u64> = t.freqs().map(|f| f.get()).collect();
        let start: u64 = freq[..sym].iter().sum();
        assert_eq!((g.start, g.freq), (start, freq[sym]), "input {i}");
    }
}

/// Real GPT-2: 2,100 tokens of the evaluation windows with overlap 512 (two restarts). The GPU
/// decode (one token at a time) and the GPU prefill (passes of 64 rows) against the CPU prefill
/// path (chunk 32), and the oracle's digest from `crates/hardshell/tests/context.rs`.
#[test]
#[ignore = "needs tools/data/out/gpt2.hsm and a CUDA device"]
fn real_model_overlap_on_cuda_equals_the_cpu() {
    let Some(dev) = device("real_model_overlap_on_cuda_equals_the_cpu") else { return };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/data/out");
    let bytes = std::fs::read(format!("{root}/gpt2.hsm")).unwrap();
    let cpu = Model::load(&bytes).unwrap();
    let gpu = CudaModel::load(&dev, &bytes).unwrap();
    let text = std::fs::read_to_string(format!("{root}/eval_windows.txt")).unwrap();
    let toks: Vec<TokenId> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| {
            let ids = l.split_once('\t').unwrap().1.to_owned();
            ids.split_whitespace().map(|t| TokenId::new(t.parse().unwrap())).collect::<Vec<_>>()
        })
        .take(2100)
        .collect();
    let o = Overlap::new(512);
    let cfg = SessionConfig::new().with_chunk(32).with_overlap(o);
    let mut s = Session::with_config(&cpu, cfg);
    let mut want = vec![s.start_block().unwrap().to_vec()];
    for part in toks.chunks(32) {
        want.extend(s.step_many(part).unwrap().iter().map(<[u64]>::to_vec));
    }
    let mut h = Sha256::new();
    for w in &want {
        for x in w {
            h.update(x.to_le_bytes());
        }
    }
    println!("real model, overlap 512, {} tokens: CPU digest {}", toks.len(), hex(&h.finalize()));

    let mut g = CudaSession::with_overlap(&gpu, o).unwrap();
    assert_eq!(g.start_block().unwrap(), &want[0][..]);
    for (i, &t) in toks.iter().enumerate() {
        assert_eq!(g.step(t).unwrap(), &want[i + 1][..], "decode step {i}");
    }
    let mut batch = CudaBatch::new(&gpu, 1, 64).unwrap();
    batch.set_overlap(o).unwrap();
    let mut inputs = vec![(0, gpu.eot_token())];
    inputs.extend(toks.iter().map(|&t| (0, t)));
    let got = batch.run_weights(&inputs).unwrap();
    let v = gpu.vocab_size();
    for (i, w) in want.iter().enumerate() {
        assert_eq!(&got[i * v..(i + 1) * v], &w[..], "prefill row {i}");
    }
    // The oracle (slow): only the first 1,200 tokens, with one restart.
    let mut r = ReferenceSession::with_overlap(&cpu, o);
    assert_eq!(r.start_block().unwrap(), &want[0][..]);
    for (i, &t) in toks.iter().take(1200).enumerate() {
        assert_eq!(r.step(t).unwrap(), &want[i + 1][..], "oracle step {i}");
    }
    println!("real model: CUDA decode and prefill with overlap 512 equal the CPU");
}
