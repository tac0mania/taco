//! The context rule with overlap `K` (ADR 0015) on the wgpu backend against the CPU oracle
//! (`ReferenceSession::with_overlap`): the golden overlap hash, the committed tiny model, a
//! model with the GPT-2 context of 1,024 tokens, batches with restarts of many slots in one
//! pass, the coder op, and real GPT-2 (ignored by default). A one-bit difference fails.
//!
//! The values of `K` are 0, 1, 12, 22 and 512. The tiny models have a context of 64 tokens, so
//! they refuse `K = 512` (`InvalidOverlap`), as the CPU does.
//!
//! The test `real_gpt2_with_overlap_equals_the_cpu` uses `tools/data/out/gpt2.hsm` (SHA-256
//! `aaa03eb9…a5c2`) and `tools/data/out/eval_windows.txt`. Run it with `--ignored --release`.
//! It skips with a message if a file is missing.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::print_stderr)]

#[path = "../../hardshell/tests/common/mod.rs"]
mod common;
mod support;

use common::builder::{build, tiny_dims};
use common::reference::Dims;
use hardshell::TokenId;
use hardshell::gpt2::{Model, Overlap, ReferenceSession, Session, SessionConfig};
use hardshell_gpu::{CoderQuery, Error, GpuBatch, GpuModel, GpuSession};
use sha2::{Digest, Sha256};
use support::device;
use taco_coder::{FreqBits, FreqTable};

/// The overlaps of the brief.
const OVERLAPS: [u16; 5] = [0, 1, 12, 22, 512];

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<TokenId> {
    let mut r = common::Rng::new(seed);
    (0..n).map(|_| TokenId::new(r.below(vocab as u64) as u32)).collect()
}

fn hex(h: &[u8]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

/// The number of restarts of a block of `n` tokens after the start: the first when the context
/// is full, then one after each `n_ctx - 1 - k` tokens.
fn restarts(n: usize, n_ctx: usize, k: usize) -> usize {
    if n < n_ctx { 0 } else { 1 + (n - n_ctx) / (n_ctx - 1 - k) }
}

/// One GPU session with overlap `k` against the oracle, at every step. Gives `false` if the
/// model refuses `k` (on the CPU and on the GPU).
fn session_against_oracle(cpu: &Model, gpu: &GpuModel, toks: &[TokenId], k: u16) -> bool {
    let o = Overlap::new(k);
    if usize::from(k) + 2 > cpu.n_ctx() {
        assert!(cpu.check_overlap(o).is_err(), "K {k}: the CPU accepts it");
        let e = GpuSession::with_overlap(gpu, o).unwrap_err();
        assert!(
            matches!(e, Error::Model(hardshell::Error::InvalidOverlap { .. })),
            "K {k}: the GPU gives {e}"
        );
        return false;
    }
    let mut c = ReferenceSession::with_overlap(cpu, o);
    let mut g = GpuSession::with_overlap(gpu, o).unwrap();
    assert_eq!(g.overlap(), o);
    assert_eq!(g.start_block().unwrap(), c.start_block().unwrap(), "K {k}: start");
    for (i, &t) in toks.iter().enumerate() {
        assert_eq!(g.step(t).unwrap(), c.step(t).unwrap(), "K {k}: step {i}");
        assert_eq!(g.position(), c.position(), "K {k}: position after step {i}");
    }
    let n = restarts(toks.len(), cpu.n_ctx(), usize::from(k));
    eprintln!("K {k}, n_ctx {}: {} steps, {n} restarts, equal", cpu.n_ctx(), toks.len());
    assert!(n >= 2, "the test needs at least two restarts");
    true
}

/// The hash of `golden_overlap_conformance_hash` in `crates/hardshell/tests/context.rs` (ADR
/// 0015 section 6): the golden tiny model, 300 tokens, `K = 32`.
const GOLDEN_OVERLAP_OUTPUT: &str =
    "5a79928200ad35bb7e7a1be2a0a35e05dcecc001a89cf3cf0765dfc7e5304a83";

#[test]
fn golden_overlap_hash_on_the_gpu() {
    let Some(dev) = device("golden_overlap_hash_on_the_gpu") else { return };
    let b = build(tiny_dims(), 0x7ac0_0001, 64);
    let m = GpuModel::load(&dev, &b.bytes).unwrap();
    let mut s = GpuSession::with_overlap(&m, Overlap::new(32)).unwrap();
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
fn committed_tiny_model_with_overlap_equals_the_oracle() {
    let Some(dev) = device("committed_tiny_model_with_overlap_equals_the_oracle") else { return };
    let bytes =
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../conformance/tiny.hsm")).unwrap();
    let cpu = Model::load(&bytes).unwrap();
    let gpu = GpuModel::load(&dev, &bytes).unwrap();
    // 300 tokens: 4 restarts with K = 0, 6 with K = 22 (measured).
    let toks = tokens(0x0e1a_0001, 300, cpu.vocab_size());
    let ran: Vec<u16> =
        OVERLAPS.into_iter().filter(|&k| session_against_oracle(&cpu, &gpu, &toks, k)).collect();
    assert_eq!(ran, [0, 1, 12, 22], "n_ctx 64 refuses K = 512 only");
}

#[test]
fn the_largest_overlap_fits_and_the_next_is_refused() {
    let Some(dev) = device("the_largest_overlap_fits_and_the_next_is_refused") else { return };
    let b = build(tiny_dims(), 3, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = GpuModel::load(&dev, &b.bytes).unwrap();
    // K = n_ctx - 2 = 62: a restart keeps all tokens except the first, every step restarts.
    let toks = tokens(0x0e1a_0002, 140, 300);
    assert!(session_against_oracle(&cpu, &gpu, &toks, 62));
    assert!(GpuBatch::with_overlap(&gpu, 3, Overlap::new(63)).is_err());
    assert!(cpu.check_overlap(Overlap::new(63)).is_err());
    assert_eq!(GpuBatch::new(&gpu, 2).unwrap().overlap(), Overlap::NONE);
}

#[test]
fn gpt2_context_with_every_overlap_equals_the_oracle() {
    // The context of GPT-2 (1,024 tokens) in a narrow model, so K = 512 runs: 2,100 tokens
    // give 3 restarts with K = 512 and 2 with K = 0, 1, 12 and 22.
    let Some(dev) = device("gpt2_context_with_every_overlap_equals_the_oracle") else { return };
    let dims = Dims { n_layer: 1, n_head: 4, d_model: 64, d_ff: 128, vocab: 300, n_ctx: 1024 };
    let b = build(dims, 0x0e1a_0003, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = GpuModel::load(&dev, &b.bytes).unwrap();
    let toks = tokens(0x0e1a_0004, 2100, 300);
    for k in OVERLAPS {
        assert!(session_against_oracle(&cpu, &gpu, &toks, k), "K {k}");
    }
}

/// Oracle sessions and one GPU batch with overlap `k` and the same token streams. Each step
/// uses a changing subset of the slots, and some slots start a new block. With many slots,
/// the restart rows of several slots share a pass, and a restart crosses passes.
fn batch_against_oracles(dims: Dims, k: u16, slots: usize, steps: usize) {
    let Some(dev) = device("batch_against_oracles") else { return };
    let seed = 0x0e1b ^ u64::from(k) ^ ((slots as u64) << 16);
    let b = build(dims, seed, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = GpuModel::load(&dev, &b.bytes).unwrap();
    let o = Overlap::new(k);
    let mut sessions: Vec<ReferenceSession<'_>> =
        (0..slots).map(|_| ReferenceSession::with_overlap(&cpu, o)).collect();
    let mut batch = GpuBatch::with_overlap(&gpu, slots, o).unwrap();
    let mut r = common::Rng::new(seed ^ 0x5eed);
    let mut restarted = 0;
    for step in 0..steps {
        let (mut inputs, mut wants) = (Vec::new(), Vec::new());
        for (slot, s) in sessions.iter_mut().enumerate() {
            // About 7 of 8 slots take part in each step.
            if r.below(8) == 0 {
                continue;
            }
            let token = if step == 0 || r.below(997) == 0 {
                // A new block: the GPU path does reset + step(eot), as start_block does.
                batch.reset(slot);
                s.reset();
                gpu.eot_token()
            } else {
                TokenId::new(r.below(dims.vocab as u64) as u32)
            };
            if s.position() >= dims.n_ctx {
                restarted += 1;
            }
            wants.push(s.step(token).unwrap().to_vec());
            inputs.push((slot, token));
        }
        batch.step(&inputs).unwrap();
        for (&(slot, _), want) in inputs.iter().zip(&wants) {
            assert_eq!(batch.weights(slot).unwrap(), &want[..], "K {k}, step {step} slot {slot}");
            assert_eq!(batch.position(slot), sessions[slot].position(), "K {k}, step {step}");
        }
    }
    eprintln!("K {k}, {slots} slots, {steps} steps: {restarted} restarts, equal");
    assert!(restarted >= 2 * slots, "the test needs several restarts in each slot");
}

#[test]
fn small_batch_with_overlap_equals_oracle_sessions() {
    // 5 slots: passes of 32 rows, so the restarts of all slots fit in one or two passes.
    for k in [0, 1, 12, 22] {
        batch_against_oracles(tiny_dims(), k, 5, 220);
    }
}

#[test]
fn large_batch_with_overlap_equals_oracle_sessions() {
    // 40 slots: passes of 40 rows, so 23-row restarts of many slots cross passes, and the
    // tiled kernel runs.
    batch_against_oracles(tiny_dims(), 22, 40, 200);
    batch_against_oracles(tiny_dims(), 1, 40, 260);
}

#[test]
fn coder_intervals_with_overlap_equal_the_cpu_table() {
    let Some(dev) = device("coder_intervals_with_overlap_equal_the_cpu_table") else { return };
    let b = build(tiny_dims(), 31, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = GpuModel::load(&dev, &b.bytes).unwrap();
    for k in [1, 12, 22] {
        let o = Overlap::new(k);
        let mut s = ReferenceSession::with_overlap(&cpu, o);
        let mut batch = GpuBatch::with_overlap(&gpu, 1, o).unwrap();
        let toks = tokens(0xc0de ^ u64::from(k), 200, 300);
        let mut w = s.start_block().unwrap().to_vec();
        batch.reset(0);
        let mut prev = gpu.eot_token();
        for (i, &t) in toks.iter().enumerate() {
            // The interval of `t` in the table of the weights after `prev`.
            let got = batch.step_coded(&[(0, prev)], &[CoderQuery::Encode(t)], 32).unwrap();
            let table = FreqTable::from_weights(&w, FreqBits::new(32).unwrap()).unwrap();
            let freq: Vec<u64> = table.freqs().map(|f| f.get()).collect();
            let sym = t.get() as usize;
            let start: u64 = freq[..sym].iter().sum();
            let want = (t, start, freq[sym]);
            assert_eq!((got[0].symbol, got[0].start, got[0].freq), want, "K {k}, token {i}");
            w = s.step(t).unwrap().to_vec();
            prev = t;
        }
    }
}

/// Real GPT-2 (`gpt2.hsm`), the tokens of the evaluation windows. For `K` = 0, 1, 12 and 22:
/// 1,300 tokens (one restart), GPU decode against the CPU prefill path (chunk 32). For
/// `K = 512`: 2,100 tokens (two restarts), GPU decode and a GPU batch of 2 slots against the
/// CPU prefill path and the SHA-256 of the oracle's weights (ADR 0015, measurements).
///
/// NOTE: the CPU prefill path is the reference here, because the oracle takes minutes for
/// real GPT-2. `crates/hardshell/tests/context.rs` proves that it equals the oracle with
/// overlap; the `K = 512` digest is the oracle's.
#[test]
#[ignore = "needs tools/data/out/gpt2.hsm and takes some minutes"]
fn real_gpt2_with_overlap_equals_the_cpu() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/data/out");
    let (Ok(bytes), Ok(text)) = (
        std::fs::read(format!("{root}/gpt2.hsm")),
        std::fs::read_to_string(format!("{root}/eval_windows.txt")),
    ) else {
        eprintln!("real_gpt2_with_overlap_equals_the_cpu: SKIPPED, no {root}/gpt2.hsm");
        return;
    };
    let Some(dev) = device("real_gpt2_with_overlap_equals_the_cpu") else { return };
    let cpu = Model::load(&bytes).unwrap();
    assert_eq!(format!("{}", cpu.hash()), REAL_MODEL, "a different gpt2.hsm");
    let gpu = GpuModel::load(&dev, &bytes).unwrap();
    let toks: Vec<TokenId> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| {
            let ids = l.split_once('\t').unwrap().1.to_owned();
            ids.split_whitespace().map(|t| TokenId::new(t.parse().unwrap())).collect::<Vec<_>>()
        })
        .take(2100)
        .collect();
    assert_eq!(toks.len(), 2100);
    let v = gpu.vocab_size();
    for k in OVERLAPS {
        let o = Overlap::new(k);
        let n = if k == 512 { 2100 } else { 1300 };
        let want = cpu_prefill(&cpu, &toks[..n], o);
        let mut g = GpuSession::with_overlap(&gpu, o).unwrap();
        assert_eq!(g.start_block().unwrap(), &want[0][..], "K {k}: start");
        for (i, &t) in toks[..n].iter().enumerate() {
            assert_eq!(g.step(t).unwrap(), &want[i + 1][..], "K {k}: decode step {i}");
        }
        eprintln!("real GPT-2, K {k}: {n} GPU steps equal the CPU");
        if k != 512 {
            continue;
        }
        let mut h = Sha256::new();
        for x in want.iter().flatten() {
            h.update(x.to_le_bytes());
        }
        assert_eq!(hex(&h.finalize()), REAL_OVERLAP_512, "the CPU digest");
        // Two slots: slot 1 starts 600 tokens later, so the slots restart at other steps.
        let mut batch = GpuBatch::with_overlap(&gpu, 2, o).unwrap();
        let mut steps = [0usize; 2];
        while steps[0] <= n {
            let mut inputs = Vec::new();
            for (slot, lag) in [(0, 0), (1, 600)] {
                let i = steps[slot];
                if steps[0] >= lag && i <= n {
                    inputs.push((slot, if i == 0 { gpu.eot_token() } else { toks[i - 1] }));
                }
            }
            batch.step(&inputs).unwrap();
            for &(slot, _) in &inputs {
                let i = steps[slot];
                assert_eq!(batch.weights(slot).unwrap(), &want[i][..], "batch slot {slot} row {i}");
                assert_eq!(batch.weights(slot).unwrap().len(), v);
                steps[slot] += 1;
            }
        }
        eprintln!("real GPT-2, K 512: a batch of 2 slots equals the oracle digest");
    }
}

/// The SHA-256 of `tools/data/out/gpt2.hsm` (HANDOFF.md section 2).
const REAL_MODEL: &str = "aaa03eb96e9ad7957432168c8ac6b4a99ed55d57e637e4786a664b51001ba5c2";

/// The SHA-256 of the oracle's weights for real GPT-2, 2,100 tokens, `K = 512` (ADR 0015,
/// measured on the Mac, devbox and the RTX 3060).
const REAL_OVERLAP_512: &str = "09b99a923377b260b83d676c8799d4cc07f96f2996bced7471128d4d561bd452";

/// All weights of a block with the CPU prefill path (chunk 32) and overlap `o`.
fn cpu_prefill(m: &Model, toks: &[TokenId], o: Overlap) -> Vec<Vec<u64>> {
    let mut s = Session::with_config(m, SessionConfig::new().with_chunk(32).with_overlap(o));
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for part in toks.chunks(32) {
        out.extend(s.step_many(part).unwrap().iter().map(<[u64]>::to_vec));
    }
    out
}
