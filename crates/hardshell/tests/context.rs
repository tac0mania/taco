//! Tests of the context rule with overlap (ADR 0015).
//!
//! 1. The definition: with overlap `K`, the weights after a restart are the weights of a fresh
//!    session that ran `<|endoftext|>`, the last `K` tokens, then the new token. The test
//!    builds the expected weights from fresh sessions without overlap, window by window.
//! 2. The oracle: `ReferenceSession::with_overlap` gives these weights.
//! 3. The fast paths: `Session::step`, `Session::step_many` (many chunk sizes) and `Batch`
//!    give the oracle's weights, bit for bit, for every kernel selection, for odd model shapes
//!    and for `K` from 0 to `n_ctx - 2`, over sequences with three or more restarts.
//! 4. `K = 0` is the old rule: the weights equal those of a session without the option.
//! 5. A conformance vector (golden hash) for the tiny model with `K = 32`.
//!
//! The test `real_model_overlap_paths_equal_the_reference` uses `tools/data/out/gpt2.hsm`. It
//! is ignored by default, because it needs the model file (not in git) and some minutes.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(clippy::cast_possible_truncation, clippy::print_stdout)]

mod common;

use common::Rng;
use common::builder::{build, build_wide, tiny_dims};
use common::reference::Dims;
use hardshell::gpt2::{
    Batch, BatchInput, Model, Overlap, ReferenceSession, Session, SessionConfig,
};
use hardshell::kernels::Kernels;
use hardshell::{Error, TokenId};
use sha2::{Digest, Sha256};

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<TokenId> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| TokenId::new(r.below(vocab as u64) as u32)).collect()
}

/// The number of tokens of `n` inputs that need `restarts` restarts or more:
/// `n_ctx - 1` tokens fill the first context, and each restart adds `n_ctx - 1 - k`.
fn len_with_restarts(n_ctx: usize, k: usize, restarts: usize) -> usize {
    n_ctx - 1 + restarts * (n_ctx - 1 - k) + 3
}

/// The weights of the start, then of each token, from the oracle with overlap `k`.
fn reference(m: &Model, toks: &[TokenId], k: Overlap) -> Vec<Vec<u64>> {
    let mut s = ReferenceSession::with_overlap(m, k);
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for &t in toks {
        out.push(s.step(t).unwrap().to_vec());
    }
    out
}

/// The definition of the rule, from fresh sessions without overlap: window 0 holds the first
/// `n_ctx - 1` tokens; each later window starts a fresh session, steps the last `k` tokens
/// before it (their weights are not used), then its own `n_ctx - 1 - k` tokens.
fn by_definition(m: &Model, toks: &[TokenId], k: usize) -> Vec<Vec<u64>> {
    let n_ctx = m.n_ctx();
    let mut s = ReferenceSession::new(m);
    let mut out = vec![s.start_block().unwrap().to_vec()];
    let first = toks.len().min(n_ctx - 1);
    for &t in &toks[..first] {
        out.push(s.step(t).unwrap().to_vec());
    }
    let mut start = first;
    while start < toks.len() {
        let end = (start + n_ctx - 1 - k).min(toks.len());
        s.start_block().unwrap();
        for &t in &toks[start - k..start] {
            s.step(t).unwrap();
        }
        for &t in &toks[start..end] {
            out.push(s.step(t).unwrap().to_vec());
        }
        assert!(s.position() <= n_ctx, "a window must fit in the context");
        start = end;
    }
    out
}

fn stepped(m: &Model, toks: &[TokenId], k: Kernels, o: Overlap) -> Vec<Vec<u64>> {
    let mut s = Session::with_config(m, SessionConfig::new().with_kernels(k).with_overlap(o));
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for &t in toks {
        out.push(s.step(t).unwrap().to_vec());
    }
    out
}

fn chunked(m: &Model, toks: &[TokenId], chunk: usize, k: Kernels, o: Overlap) -> Vec<Vec<u64>> {
    let cfg = SessionConfig::new().with_chunk(chunk).with_kernels(k).with_overlap(o);
    let mut s = Session::with_config(m, cfg);
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for part in toks.chunks(chunk) {
        let w = s.step_many(part).unwrap();
        assert_eq!(w.len(), part.len());
        out.extend(w.iter().map(<[u64]>::to_vec));
    }
    out
}

/// Runs `seqs` in a batch of `seqs.len()` lanes with overlap `o`. Lane `j` starts at call `j`,
/// and a lane waits (no input) on the calls that `rng` selects, so the restarts of the lanes
/// fall in different calls.
fn batched(
    m: &Model,
    seqs: &[Vec<TokenId>],
    k: Kernels,
    o: Overlap,
    seed: u64,
) -> Vec<Vec<Vec<u64>>> {
    let lanes = seqs.len();
    let mut b = Batch::with_kernels(m, lanes, k).with_overlap(o);
    let mut rng = Rng::new(seed);
    let mut next = vec![0usize; lanes];
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

/// The overlaps to test for a context of `n_ctx`: none, one token, about a quarter, a half,
/// three quarters, and the largest.
fn overlaps(n_ctx: usize) -> Vec<usize> {
    let mut v = vec![0, 1, n_ctx / 4, n_ctx / 2, 3 * n_ctx / 4, n_ctx - 2];
    v.dedup();
    v
}

#[test]
fn the_reference_obeys_the_definition() {
    for (i, d) in odd_dims().into_iter().enumerate() {
        let b = build(d, 500 + i as u64, 64);
        let m = Model::load(&b.bytes).unwrap();
        for k in overlaps(d.n_ctx) {
            let toks = tokens(600 + i as u64, len_with_restarts(d.n_ctx, k, 3), d.vocab);
            let o = Overlap::new(k as u16);
            assert_eq!(
                reference(&m, &toks, o),
                by_definition(&m, &toks, k),
                "dims {i}, overlap {k}"
            );
        }
    }
}

#[test]
fn overlap_zero_is_the_old_rule() {
    let d = tiny_dims();
    let b = build(d, 7, 64);
    let m = Model::load(&b.bytes).unwrap();
    let toks = tokens(8, 3 * d.n_ctx, d.vocab);
    let mut old = ReferenceSession::new(&m);
    let mut want = vec![old.start_block().unwrap().to_vec()];
    for &t in &toks {
        want.push(old.step(t).unwrap().to_vec());
    }
    assert_eq!(reference(&m, &toks, Overlap::NONE), want);
    assert_eq!(chunked(&m, &toks, 32, Kernels::detect(), Overlap::NONE), want);
    assert_eq!(SessionConfig::new().overlap, Overlap::NONE);
}

#[test]
fn a_large_overlap_changes_the_weights_after_the_first_restart_only() {
    let d = tiny_dims();
    let b = build(d, 7, 64);
    let m = Model::load(&b.bytes).unwrap();
    let toks = tokens(8, 2 * d.n_ctx, d.vocab);
    let old = reference(&m, &toks, Overlap::NONE);
    let new = reference(&m, &toks, Overlap::new(32));
    // Item i is the weights after i tokens; the first restart is at token n_ctx (index n_ctx).
    assert_eq!(old[..d.n_ctx], new[..d.n_ctx]);
    assert_ne!(old[d.n_ctx..], new[d.n_ctx..]);
}

#[test]
fn step_and_prefill_with_overlap_equal_the_reference() {
    for (i, d) in odd_dims().into_iter().enumerate() {
        let b = build(d, 100 + i as u64, 64);
        let m = Model::load(&b.bytes).unwrap();
        for k in overlaps(d.n_ctx) {
            let o = Overlap::new(k as u16);
            let toks = tokens(200 + i as u64, len_with_restarts(d.n_ctx, k, 3), d.vocab);
            let want = reference(&m, &toks, o);
            for kern in Kernels::available() {
                let name = kern.name();
                assert_eq!(stepped(&m, &toks, kern, o), want, "step, dims {i}, K {k}, {name}");
                for chunk in [1, 2, 3, 5, 8, 16, d.n_ctx - 1, d.n_ctx, d.n_ctx + 3, 64] {
                    let got = chunked(&m, &toks, chunk, kern, o);
                    assert_eq!(got, want, "chunk {chunk}, dims {i}, K {k}, {name}");
                }
            }
        }
    }
}

#[test]
fn batch_with_overlap_equals_the_reference() {
    for (i, d) in odd_dims().into_iter().enumerate() {
        let b = build(d, 300 + i as u64, 64);
        let m = Model::load(&b.bytes).unwrap();
        for k in overlaps(d.n_ctx) {
            let o = Overlap::new(k as u16);
            let n = len_with_restarts(d.n_ctx, k, 3);
            let seqs: Vec<Vec<TokenId>> = (0..4)
                .map(|j| tokens(400 + 10 * i as u64 + j, n + 3 * j as usize, d.vocab))
                .collect();
            let want: Vec<Vec<Vec<u64>>> = seqs.iter().map(|s| reference(&m, s, o)).collect();
            for kern in Kernels::available() {
                for seed in [1u64, 2] {
                    let got = batched(&m, &seqs, kern, o, seed);
                    assert_eq!(got, want, "batch, dims {i}, K {k}, {}, seed {seed}", kern.name());
                }
            }
        }
    }
}

#[test]
fn an_overlap_that_does_not_fit_is_refused() {
    let d = tiny_dims();
    let b = build(d, 5, 64);
    let m = Model::load(&b.bytes).unwrap();
    let n_ctx = d.n_ctx;
    let largest = Overlap::new((n_ctx - 2) as u16);
    let bad = Overlap::new((n_ctx - 1) as u16);
    assert_eq!(m.check_overlap(largest), Ok(()));
    let want = Error::InvalidOverlap { overlap: bad.get(), n_ctx };
    assert_eq!(m.check_overlap(bad), Err(want.clone()));
    assert!(m.check_overlap(Overlap::new(u16::MAX)).is_err());

    let mut r = ReferenceSession::with_overlap(&m, bad);
    assert_eq!(r.start_block().unwrap_err(), want);
    let mut s = Session::with_config(&m, SessionConfig::new().with_chunk(4).with_overlap(bad));
    assert_eq!(s.start_block().unwrap_err(), want);
    assert_eq!(s.step_many(&tokens(1, 3, d.vocab)).unwrap_err(), want);
    assert_eq!(s.position(), 0);
    let mut batch = Batch::new(&m, 2).with_overlap(bad);
    assert_eq!(batch.run(&[Some(BatchInput::Start), None]).unwrap_err(), want);
    assert_eq!(batch.position(0), Some(0));
    assert_eq!(batch.overlap(), bad);
}

#[test]
fn a_bad_token_after_a_full_context_keeps_the_history() {
    // A refused call must not change the kept tokens of the next restart.
    let d = tiny_dims();
    let b = build(d, 5, 64);
    let m = Model::load(&b.bytes).unwrap();
    let o = Overlap::new(40);
    let toks = tokens(6, len_with_restarts(d.n_ctx, 40, 2), d.vocab);
    let want = reference(&m, &toks, o);
    let mut s = Session::with_config(&m, SessionConfig::new().with_chunk(8).with_overlap(o));
    let mut got = vec![s.start_block().unwrap().to_vec()];
    let mut i = 0;
    while i < toks.len() {
        let end = (i + 8).min(toks.len());
        // The same tokens, but the last one is not in the vocabulary.
        let mut bad = toks[i..end].to_vec();
        bad.pop();
        bad.push(TokenId::new(9999));
        let pos = s.position();
        assert!(matches!(s.step_many(&bad), Err(Error::TokenOutOfRange { .. })));
        assert_eq!(s.position(), pos);
        got.extend(s.step_many(&toks[i..end]).unwrap().iter().map(<[u64]>::to_vec));
        i = end;
    }
    assert_eq!(got, want);
}

#[test]
fn overlap_on_a_wide_model_equals_the_reference() {
    // A model of hsm format version 2 (i16 GELU and ln_f outputs, ADR 0004 addendum B) with
    // overlap 32: the restart rule does not depend on the numerics of the rows.
    let d = tiny_dims();
    let b = build_wide(d, 11, 64);
    let m = Model::load(&b.bytes).unwrap();
    let o = Overlap::new(32);
    let toks = tokens(12, len_with_restarts(d.n_ctx, 32, 3), d.vocab);
    let want = reference(&m, &toks, o);
    assert_eq!(want, by_definition(&m, &toks, 32));
    for kern in Kernels::available() {
        assert_eq!(stepped(&m, &toks, kern, o), want, "step, {}", kern.name());
        assert_eq!(chunked(&m, &toks, 7, kern, o), want, "chunk 7, {}", kern.name());
        let seqs = vec![toks.clone(), toks[..100].to_vec()];
        let got = batched(&m, &seqs, kern, o, 5);
        assert_eq!(got[0], want, "batch, {}", kern.name());
    }
}

/// The conformance vector of ADR 0015: the golden tiny model of `tests/gpt2.rs`, the same 150
/// tokens and 150 more, overlap 32 (half the context of 64). SHA-256 over all weights as
/// little-endian `u64`, in step order. The sequence has 8 restarts.
#[test]
fn golden_overlap_conformance_hash() {
    let b = build(tiny_dims(), 0x7ac0_0001, 64);
    let m = Model::load(&b.bytes).unwrap();
    let toks = tokens(0x7ac0_0002, 300, 300);
    let o = Overlap::new(32);
    let mut s = Session::with_config(&m, SessionConfig::new().with_overlap(o));
    let mut h = Sha256::new();
    let mut hash = |w: &[u64]| {
        for x in w {
            h.update(x.to_le_bytes());
        }
    };
    hash(s.start_block().unwrap());
    for &t in &toks {
        hash(s.step(t).unwrap());
    }
    let out: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    println!("golden overlap output hash {out}");
    let mut r = Sha256::new();
    for w in reference(&m, &toks, o) {
        for x in w {
            r.update(x.to_le_bytes());
        }
    }
    let oracle: String = r.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(out, oracle);
    assert_eq!(out, GOLDEN_OVERLAP_OUTPUT);
}

/// The SHA-256 of all weights of `golden_overlap_conformance_hash` (measured on the Mac, ARM,
/// 2026-10-19).
const GOLDEN_OVERLAP_OUTPUT: &str =
    "5a79928200ad35bb7e7a1be2a0a35e05dcecc001a89cf3cf0765dfc7e5304a83";

/// Real GPT-2: one window of the evaluation set, 2,100 tokens with overlap 512 (two restarts).
/// The oracle, the step path, the prefill path (chunk 32) and a batch of 2 lanes. Prints the
/// SHA-256 of the oracle's weights, for the GPU test and the integration note.
#[test]
#[ignore = "needs tools/data/out/gpt2.hsm and takes some minutes"]
fn real_model_overlap_paths_equal_the_reference() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/data/out");
    let bytes = std::fs::read(format!("{root}/gpt2.hsm")).expect("tools/data/out/gpt2.hsm");
    let m = Model::load(&bytes).unwrap();
    let toks = real_tokens(root, 2100);
    let o = Overlap::new(512);
    let want = reference(&m, &toks, o);
    let mut h = Sha256::new();
    for w in &want {
        for x in w {
            h.update(x.to_le_bytes());
        }
    }
    let digest: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    println!("real model, overlap 512, {} tokens: oracle digest {digest}", toks.len());
    let k = Kernels::detect();
    assert_eq!(chunked(&m, &toks, 32, k, o), want, "chunk 32");
    assert_eq!(stepped(&m, &toks, k, o), want, "step");
    let short: Vec<TokenId> = toks[..1500].to_vec();
    let seqs = vec![toks.clone(), short.clone()];
    let got = batched(&m, &seqs, k, o, 3);
    assert_eq!(got[0], want, "batch lane 0");
    assert_eq!(got[1][..], want[..=1500], "batch lane 1");
    println!("real model: overlap 512 paths equal the reference ({})", k.name());
}

/// The first `n` tokens of the evaluation windows, joined in file order.
fn real_tokens(root: &str, n: usize) -> Vec<TokenId> {
    let text = std::fs::read_to_string(format!("{root}/eval_windows.txt")).unwrap();
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| {
            let ids = l.split_once('\t').unwrap().1.to_owned();
            ids.split_whitespace().map(|t| TokenId::new(t.parse().unwrap())).collect::<Vec<_>>()
        })
        .take(n)
        .collect()
}
