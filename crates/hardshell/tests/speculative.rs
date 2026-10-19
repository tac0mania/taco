//! Tests of the engine side of speculative decoding (ADR 0017): the draft pass and the
//! truncation of a stream.
//!
//! The decoder runs the known token and `d - 1` draft tokens in one pass, uses the rows up to
//! the first wrong draft token, and truncates the stream after them. The tests simulate this
//! with drafts of random length and random correct prefixes, also across context restarts with
//! an overlap `K` (ADR 0015) and truncations before the restart, and compare every used row
//! with the oracle that never ran a wrong token (`ReferenceSession`, ADR 0013).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(clippy::cast_possible_truncation)]

mod common;

use common::Rng;
use common::builder::{build, build_wide, tiny_dims};
use common::reference::Dims;
use hardshell::gpt2::{
    Batch, BatchInput, Model, Overlap, ReferenceSession, Session, SessionConfig,
};
use hardshell::kernels::Kernels;
use hardshell::{Error, TokenId};

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<TokenId> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| TokenId::new(r.below(vocab as u64) as u32)).collect()
}

/// The oracle: item `i` is the weights after input `i` (input 0 is the start, input `i` is
/// `toks[i - 1]`).
fn reference(m: &Model, toks: &[TokenId], k: Overlap) -> Vec<Vec<u64>> {
    let mut s = ReferenceSession::with_overlap(m, k);
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for &t in toks {
        out.push(s.step(t).unwrap().to_vec());
    }
    out
}

/// A token that is not `t`.
fn wrong(t: TokenId, rng: &mut Rng, vocab: usize) -> TokenId {
    let other = (u64::from(t.get()) + 1 + rng.below(vocab as u64 - 1)) % vocab as u64;
    TokenId::new(other as u32)
}

/// One draft pass at the stream length `n_in`: the known token and `d - 1` draft tokens, of
/// which the first `good` are correct. Gives the inputs and `good`.
fn draft(
    toks: &[TokenId],
    n_in: usize,
    max_d: usize,
    rng: &mut Rng,
    vocab: usize,
) -> (Vec<TokenId>, usize) {
    let left = toks.len() + 1 - n_in;
    let d = (1 + rng.below(max_d as u64) as usize).min(left);
    let good = rng.below(d as u64) as usize;
    let inputs = (0..d)
        .map(|j| {
            let t = toks[n_in - 1 + j];
            if j <= good { t } else { wrong(t, rng, vocab) }
        })
        .collect();
    (inputs, good)
}

/// How many inputs of a pass with `good` correct draft tokens the decoder keeps: usually all
/// valid rows (`good + 1`), sometimes fewer (any length from 1), so a truncation also drops
/// correct inputs and goes back before a restart.
fn keep(good: usize, rng: &mut Rng) -> usize {
    if rng.below(4) == 0 { 1 + rng.below(good as u64 + 1) as usize } else { good + 1 }
}

/// Speculative decoding of `toks` with a session: every used row equals the oracle.
fn session_case(m: &Model, toks: &[TokenId], o: Overlap, kern: Kernels, max_d: usize, seed: u64) {
    let want = reference(m, toks, o);
    let cfg = SessionConfig::new().with_chunk(max_d).with_kernels(kern).with_overlap(o);
    let mut s = Session::with_config(m, cfg);
    assert_eq!(s.start_block().unwrap(), &want[0][..]);
    assert_eq!(s.stream_len(), 1);
    let mut rng = Rng::new(seed);
    let vocab = m.vocab_size();
    let mut n_in = 1;
    while n_in <= toks.len() {
        let (inputs, good) = draft(toks, n_in, max_d, &mut rng, vocab);
        let w = s.step_many(&inputs).unwrap();
        for r in 0..=good {
            let ctx = format!("K {}, {}, len {n_in}, row {r}", o.get(), kern.name());
            assert_eq!(w.get(r).unwrap(), &want[n_in + r][..], "{ctx}");
        }
        assert_eq!(s.stream_len(), n_in + inputs.len());
        let kept = keep(good, &mut rng);
        s.truncate(n_in + kept - 1).unwrap();
        // The truncated stream ends after the known input of this pass and `kept - 1` drafts:
        // run the last kept input again, so the next pass starts with known weights.
        let last = n_in + kept - 1;
        assert_eq!(s.stream_len(), last);
        let again = s.step(toks[last - 1]).unwrap();
        assert_eq!(again, &want[last][..], "again, K {}, len {last}", o.get());
        n_in = last + 1;
    }
}

fn odd_dims() -> Vec<Dims> {
    vec![
        tiny_dims(),
        Dims { n_layer: 1, n_head: 3, d_model: 60, d_ff: 100, vocab: 300, n_ctx: 20 },
        Dims { n_layer: 2, n_head: 1, d_model: 33, d_ff: 70, vocab: 263, n_ctx: 9 },
    ]
}

/// Overlaps 0, 1, half the context and the largest.
fn overlaps(n_ctx: usize) -> Vec<usize> {
    let mut v = vec![0, 1, n_ctx / 2, n_ctx - 2];
    v.dedup();
    v
}

#[test]
fn draft_passes_and_truncations_equal_the_reference() {
    for (i, d) in odd_dims().into_iter().enumerate() {
        let b = build(d, 900 + i as u64, 64);
        let m = Model::load(&b.bytes).unwrap();
        for k in overlaps(d.n_ctx) {
            let o = Overlap::new(k as u16);
            // At least 4 restarts.
            let toks = tokens(910 + i as u64, d.n_ctx - 1 + 4 * (d.n_ctx - 1 - k) + 5, d.vocab);
            // A draft of at most n_ctx - K - 1 rows crosses at most one restart.
            let max_d = (d.n_ctx - 1 - k).clamp(1, 17);
            for kern in Kernels::available() {
                for seed in [1u64, 2, 3] {
                    session_case(&m, &toks, o, kern, max_d, seed);
                }
            }
        }
    }
}

#[test]
fn a_wide_model_truncates_as_the_reference() {
    let d = tiny_dims();
    let b = build_wide(d, 31, 64);
    let m = Model::load(&b.bytes).unwrap();
    let o = Overlap::new(32);
    let toks = tokens(32, 200, d.vocab);
    for kern in Kernels::available() {
        session_case(&m, &toks, o, kern, 16, 4);
    }
}

#[test]
fn truncation_across_a_restart_equals_never_running_the_dropped_tokens() {
    // n_ctx 64, K 32: input 64 restarts the context. A pass of 8 rows from input 60 crosses the
    // restart; a truncation to 62 inputs goes back before it.
    let d = tiny_dims();
    let b = build(d, 41, 64);
    let m = Model::load(&b.bytes).unwrap();
    let o = Overlap::new(32);
    let toks = tokens(42, 160, d.vocab);
    let want = reference(&m, &toks, o);
    let mut s = Session::with_config(&m, SessionConfig::new().with_chunk(64).with_overlap(o));
    s.start_block().unwrap();
    s.step_many(&toks[..59]).unwrap();
    assert_eq!(s.stream_len(), 60);
    let mut rng = Rng::new(43);
    let bad: Vec<TokenId> = toks[59..67].iter().map(|&t| wrong(t, &mut rng, d.vocab)).collect();
    s.step_many(&bad).unwrap();
    assert!(s.position() < 64, "the pass restarted the context");
    s.truncate(60).unwrap();
    assert_eq!(s.position(), 60);
    let w = s.step_many(&toks[59..80]).unwrap();
    for (r, row) in w.iter().enumerate() {
        assert_eq!(row, &want[60 + r][..], "row {r}");
    }
    // Truncate to the input before the restart, then step one token at a time.
    s.truncate(64).unwrap();
    for (j, &t) in toks[63..100].iter().enumerate() {
        assert_eq!(s.step(t).unwrap(), &want[64 + j][..], "step {j}");
    }
}

#[test]
fn a_truncation_too_far_back_is_refused_and_changes_nothing() {
    let d = tiny_dims();
    let b = build(d, 51, 64);
    let m = Model::load(&b.bytes).unwrap();
    let o = Overlap::new(16);
    let toks = tokens(52, 200, d.vocab);
    let want = reference(&m, &toks, o);
    let mut s = Session::with_config(&m, SessionConfig::new().with_chunk(64).with_overlap(o));
    s.start_block().unwrap();
    for part in toks[..150].chunks(64) {
        s.step_many(part).unwrap();
    }
    let len = s.stream_len();
    assert_eq!(len, 151);
    let more = Error::BadTruncate { len: 152, stream_len: 151 };
    assert_eq!(s.truncate(152), Err(more));
    // Contexts: inputs 0..64, 64..111, 111..158. The length 63 is before the restart before
    // the last one.
    assert!(matches!(s.truncate(63), Err(Error::BadTruncate { .. })));
    assert_eq!(s.stream_len(), len);
    assert_eq!(s.step(toks[150]).unwrap(), &want[151][..]);
    // The start of the context before the last restart is still in reach.
    s.truncate(64).unwrap();
    assert_eq!(s.step(toks[63]).unwrap(), &want[64][..]);
}

#[test]
fn room_ends_a_draft_at_the_restart() {
    let d = tiny_dims();
    let b = build(d, 61, 64);
    let m = Model::load(&b.bytes).unwrap();
    let o = Overlap::new(20);
    let mut s = Session::with_config(&m, SessionConfig::new().with_chunk(64).with_overlap(o));
    s.start_block().unwrap();
    assert_eq!(s.room(), 63);
    let toks = tokens(62, 63, d.vocab);
    s.step_many(&toks).unwrap();
    assert_eq!(s.position(), 64);
    assert_eq!(s.room(), 64 - 20 - 1);
}

/// Speculative decoding of many streams in a batch, with the passes of the lanes in one call.
fn batch_case(m: &Model, seqs: &[Vec<TokenId>], o: Overlap, kern: Kernels, seed: u64) {
    let lanes = seqs.len();
    let want: Vec<Vec<Vec<u64>>> = seqs.iter().map(|s| reference(m, s, o)).collect();
    let mut b = Batch::with_kernels(m, lanes, kern).with_overlap(o);
    let starts = vec![Some(BatchInput::Start); lanes];
    b.run(&starts).unwrap();
    for (j, w) in want.iter().enumerate() {
        assert_eq!(b.weights(j).unwrap(), &w[0][..]);
    }
    let mut rng = Rng::new(seed);
    let vocab = m.vocab_size();
    let max_d = (m.n_ctx() - 1 - o.tokens()).clamp(1, 9);
    let mut n_in = vec![1usize; lanes];
    while (0..lanes).any(|j| n_in[j] <= seqs[j].len()) {
        let mut inputs = Vec::new();
        let mut passes = Vec::new();
        for j in 0..lanes {
            if n_in[j] > seqs[j].len() || rng.below(5) == 0 {
                continue;
            }
            let (d, good) = draft(&seqs[j], n_in[j], max_d, &mut rng, vocab);
            passes.push((j, inputs.len(), d.len(), good));
            inputs.extend(d.into_iter().map(|t| (j, t)));
        }
        let w = b.run_rows(&inputs).unwrap();
        for &(j, at, _, good) in &passes {
            for r in 0..=good {
                assert_eq!(w.get(at + r).unwrap(), &want[j][n_in[j] + r][..], "lane {j}");
            }
        }
        for &(j, _, d, good) in &passes {
            assert_eq!(b.stream_len(j), Some(n_in[j] + d));
            let kept = keep(good, &mut rng);
            b.truncate(j, n_in[j] + kept).unwrap();
            n_in[j] += kept;
        }
    }
}

#[test]
fn batch_draft_rows_and_truncations_equal_the_reference() {
    for (i, d) in odd_dims().into_iter().enumerate() {
        let b = build(d, 700 + i as u64, 64);
        let m = Model::load(&b.bytes).unwrap();
        for k in overlaps(d.n_ctx) {
            let o = Overlap::new(k as u16);
            let n = d.n_ctx - 1 + 3 * (d.n_ctx - 1 - k) + 3;
            let seqs: Vec<Vec<TokenId>> = (0..3)
                .map(|j| tokens(710 + 10 * i as u64 + j, n + 5 * j as usize, d.vocab))
                .collect();
            for kern in Kernels::available() {
                batch_case(&m, &seqs, o, kern, 7 + i as u64);
            }
        }
    }
}

#[test]
fn a_batch_run_after_run_rows_continues_each_lane() {
    let d = tiny_dims();
    let b = build(d, 81, 64);
    let m = Model::load(&b.bytes).unwrap();
    let toks = tokens(82, 40, d.vocab);
    let want = reference(&m, &toks, Overlap::NONE);
    let mut batch = Batch::new(&m, 2);
    batch.run(&[Some(BatchInput::Start), None]).unwrap();
    let inputs: Vec<(usize, TokenId)> = toks[..10].iter().map(|&t| (0, t)).collect();
    batch.run_rows(&inputs).unwrap();
    assert_eq!(batch.weights(0), None);
    batch.truncate(0, 6).unwrap();
    batch.run(&[Some(BatchInput::Token(toks[5])), Some(BatchInput::Start)]).unwrap();
    assert_eq!(batch.weights(0).unwrap(), &want[6][..]);
    assert_eq!(batch.stream_len(0), Some(7));
    assert_eq!(batch.stream_len(1), Some(1));
    assert!(batch.truncate(2, 0).is_err());
    assert!(batch.run_rows(&[(2, toks[0])]).is_err());
}
