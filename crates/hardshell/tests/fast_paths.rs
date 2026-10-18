//! Equivalence tests: every fast path of HardShell gives the same weights as the reference
//! session (`ReferenceSession`), bit for bit (CLAUDE.md §1 rule 10).
//!
//! The paths: `Session::step` (the step path), `Session::step_many` with many chunk sizes
//! (the prefill path), and `Batch` with lanes that start at different times and sometimes
//! wait (the batch path). Each path runs with every kernel selection of this machine
//! (portable and SIMD). The models have the tiny shape and odd shapes (a head width of 20 or
//! 33, widths that are not multiples of 16), and the token sequences are longer than two
//! contexts, so they include context restarts.
//!
//! The test `real_model_paths_equal_the_reference` uses `tools/data/out/gpt2.hsm`. It is
//! ignored by default, because it needs the model file (not in git) and about a minute.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(clippy::cast_possible_truncation, clippy::print_stdout)]

mod common;

use common::Rng;
use common::builder::{build, tiny_dims};
use common::reference::Dims;
use hardshell::gpt2::{Batch, BatchInput, Model, ReferenceSession, Session, SessionConfig};
use hardshell::kernels::Kernels;
use hardshell::{Error, TokenId};

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<TokenId> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| TokenId::new(r.below(vocab as u64) as u32)).collect()
}

/// The weights of the start, then of each token, from the reference.
fn reference(m: &Model, toks: &[TokenId]) -> Vec<Vec<u64>> {
    let mut s = ReferenceSession::new(m);
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for &t in toks {
        out.push(s.step(t).unwrap().to_vec());
    }
    out
}

fn chunked(m: &Model, toks: &[TokenId], chunk: usize, k: Kernels) -> Vec<Vec<u64>> {
    let cfg = SessionConfig::new().with_chunk(chunk).with_kernels(k);
    let mut s = Session::with_config(m, cfg);
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for part in toks.chunks(chunk) {
        let w = s.step_many(part).unwrap();
        assert_eq!(w.len(), part.len());
        out.extend(w.iter().map(<[u64]>::to_vec));
    }
    out
}

fn stepped(m: &Model, toks: &[TokenId], k: Kernels) -> Vec<Vec<u64>> {
    let mut s = Session::with_config(m, SessionConfig::new().with_kernels(k));
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for &t in toks {
        out.push(s.step(t).unwrap().to_vec());
    }
    out
}

/// Runs `seqs` in a batch of `seqs.len()` lanes. Lane `j` starts at call `j`, and a lane
/// waits (no input) on the calls that `rng` selects.
fn batched(m: &Model, seqs: &[Vec<TokenId>], k: Kernels, seed: u64) -> Vec<Vec<Vec<u64>>> {
    let lanes = seqs.len();
    let mut b = Batch::with_kernels(m, lanes, k);
    let mut rng = Rng::new(seed);
    let mut next = vec![0usize; lanes]; // 0 = not started; i = tokens 0..i-1 given
    let mut out: Vec<Vec<Vec<u64>>> = vec![Vec::new(); lanes];
    let mut call = 0;
    while (0..lanes).any(|j| next[j] <= seqs[j].len()) {
        let inputs: Vec<Option<BatchInput>> = (0..lanes)
            .map(|j| {
                if call < j || next[j] > seqs[j].len() || rng.below(4) == 0 {
                    None
                } else if next[j] == 0 {
                    Some(BatchInput::Start)
                } else {
                    Some(BatchInput::Token(seqs[j][next[j] - 1]))
                }
            })
            .collect();
        b.run(&inputs).unwrap();
        for (j, inp) in inputs.iter().enumerate() {
            if inp.is_some() {
                out[j].push(b.weights(j).unwrap().to_vec());
                next[j] += 1;
            }
        }
        call += 1;
    }
    out
}

fn odd_dims() -> Vec<Dims> {
    vec![
        tiny_dims(),
        Dims { n_layer: 1, n_head: 3, d_model: 60, d_ff: 100, vocab: 300, n_ctx: 20 },
        Dims { n_layer: 2, n_head: 1, d_model: 33, d_ff: 70, vocab: 263, n_ctx: 9 },
    ]
}

#[test]
fn step_and_prefill_equal_the_reference() {
    for (i, d) in odd_dims().into_iter().enumerate() {
        let b = build(d, 100 + i as u64, 64);
        let m = Model::load(&b.bytes).unwrap();
        let toks = tokens(200 + i as u64, 2 * d.n_ctx + 7, d.vocab);
        let want = reference(&m, &toks);
        for k in Kernels::available() {
            assert_eq!(stepped(&m, &toks, k), want, "step, dims {i}, kernels {}", k.name());
            for chunk in [1, 2, 3, 5, 8, 16, d.n_ctx - 1, d.n_ctx, d.n_ctx + 3, 64] {
                let got = chunked(&m, &toks, chunk, k);
                assert_eq!(got, want, "chunk {chunk}, dims {i}, kernels {}", k.name());
            }
        }
    }
}

#[test]
fn batch_equals_the_reference() {
    for (i, d) in odd_dims().into_iter().enumerate() {
        let b = build(d, 300 + i as u64, 64);
        let m = Model::load(&b.bytes).unwrap();
        let seqs: Vec<Vec<TokenId>> = (0..4)
            .map(|j| tokens(400 + 10 * i as u64 + j, d.n_ctx + 5 + 3 * j as usize, d.vocab))
            .collect();
        let want: Vec<Vec<Vec<u64>>> = seqs.iter().map(|s| reference(&m, s)).collect();
        for k in Kernels::available() {
            for seed in [1u64, 2] {
                let got = batched(&m, &seqs, k, seed);
                assert_eq!(got, want, "batch, dims {i}, kernels {}, seed {seed}", k.name());
            }
        }
    }
}

#[test]
fn a_restarted_lane_equals_a_fresh_session() {
    // A lane that runs a block, then starts again, gives the same weights as a fresh session.
    let d = tiny_dims();
    let b = build(d, 9, 64);
    let m = Model::load(&b.bytes).unwrap();
    let (first, second) = (tokens(10, 30, d.vocab), tokens(11, 12, d.vocab));
    let mut batch = Batch::new(&m, 2);
    let mut got = Vec::new();
    let run_seq = |batch: &mut Batch<'_>, seq: &[TokenId], got: &mut Vec<Vec<u64>>| {
        batch.run(&[None, Some(BatchInput::Start)]).unwrap();
        got.push(batch.weights(1).unwrap().to_vec());
        for &t in seq {
            batch.run(&[None, Some(BatchInput::Token(t))]).unwrap();
            got.push(batch.weights(1).unwrap().to_vec());
        }
    };
    run_seq(&mut batch, &first, &mut Vec::new());
    run_seq(&mut batch, &second, &mut got);
    assert_eq!(got, reference(&m, &second));
    assert_eq!(batch.weights(0), None);
    assert_eq!(batch.position(1), Some(13));
}

#[test]
fn step_many_refuses_bad_input_and_keeps_the_context() {
    let d = tiny_dims();
    let b = build(d, 5, 64);
    let m = Model::load(&b.bytes).unwrap();
    let mut s = Session::with_config(&m, SessionConfig::new().with_chunk(4));
    s.start_block().unwrap();
    let toks = tokens(6, 5, d.vocab);
    assert!(matches!(s.step_many(&toks), Err(Error::LengthMismatch { .. })));
    let bad = [toks[0], TokenId::new(300)];
    assert_eq!(
        s.step_many(&bad).unwrap_err(),
        Error::TokenOutOfRange { token: 300, vocab_size: 300 }
    );
    assert_eq!(s.position(), 1);
    assert!(s.step_many(&[]).unwrap().is_empty());
    let mut batch = Batch::new(&m, 2);
    assert!(matches!(batch.run(&[None]), Err(Error::LengthMismatch { .. })));
    let r = batch.run(&[Some(BatchInput::Start), Some(BatchInput::Token(TokenId::new(300)))]);
    assert!(matches!(r, Err(Error::TokenOutOfRange { .. })));
    assert_eq!(batch.position(0), Some(0));
    assert_eq!(batch.weights(0), None);
}

/// Two windows of the evaluation set (the first 256 tokens of each), on the real model: the
/// reference, the step path, the prefill path (chunks of 32 and 50) and a batch of 2 lanes.
#[test]
#[ignore = "needs tools/data/out/gpt2.hsm and takes about a minute"]
fn real_model_paths_equal_the_reference() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/data/out");
    let bytes = std::fs::read(format!("{root}/gpt2.hsm")).expect("tools/data/out/gpt2.hsm");
    let m = Model::load(&bytes).unwrap();
    let text = std::fs::read_to_string(format!("{root}/eval_windows.txt")).unwrap();
    let seqs: Vec<Vec<TokenId>> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .take(2)
        .map(|l| {
            let ids = l.split_once('\t').unwrap().1;
            ids.split_whitespace().take(256).map(|t| TokenId::new(t.parse().unwrap())).collect()
        })
        .collect();
    let want: Vec<Vec<Vec<u64>>> = seqs.iter().map(|s| reference(&m, s)).collect();
    let k = Kernels::detect();
    for (s, w) in seqs.iter().zip(&want) {
        assert_eq!(&stepped(&m, s, k), w, "step");
        assert_eq!(&chunked(&m, s, 32, k), w, "chunk 32");
        assert_eq!(&chunked(&m, s, 50, k), w, "chunk 50");
    }
    assert_eq!(batched(&m, &seqs, k, 3), want, "batch");
    println!("real model: 2 windows of 256 tokens equal the reference ({})", k.name());
}
