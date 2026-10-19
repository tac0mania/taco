//! The GPT-2 step on the CUDA GPU against the CPU reference: the golden conformance hash, the
//! committed tiny model, batches, prefill with restarts, and the coder op. A one-bit
//! difference fails.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::print_stderr)]

// The model builder of the hardshell tests. The golden model comes from it.
#[path = "../../hardshell/tests/common/mod.rs"]
mod common;
mod support;

use common::builder::{build, tiny_dims};
use common::reference::Dims;
use hardshell::TokenId;
use hardshell::gpt2::{Model, Session};
use hardshell_cuda::{CoderInterval, CoderQuery, CudaBatch, CudaModel, CudaSession};
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

const GOLDEN_MODEL: &str = "e3055785bc4509abf7b017e52a0f99030a508cfe0810293e7a5efdccdf2d7bb9";
const GOLDEN_OUTPUT: &str = "d4063522517854fdcab605c3d3159bd3ac9a15eb92fa6721ff4a276658f90078";

#[test]
fn golden_conformance_hash_on_cuda() {
    let Some(dev) = device("golden_conformance_hash_on_cuda") else { return };
    let b = build(tiny_dims(), 0x7ac0_0001, 64);
    let m = CudaModel::load(&dev, &b.bytes).unwrap();
    assert_eq!(format!("{}", m.hash()), GOLDEN_MODEL);
    let mut s = CudaSession::new(&m).unwrap();
    let mut h = Sha256::new();
    for x in s.start_block().unwrap() {
        h.update(x.to_le_bytes());
    }
    for t in tokens(0x7ac0_0002, 150, 300) {
        for x in s.step(t).unwrap() {
            h.update(x.to_le_bytes());
        }
    }
    assert_eq!(hex(&h.finalize()), GOLDEN_OUTPUT);
}

#[test]
fn load_with_the_cpu_model_gives_the_golden_hash_and_another_file_keeps_its_hash() {
    let Some(dev) = device("load_with_the_cpu_model") else { return };
    let b = build(tiny_dims(), 0x7ac0_0001, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let m = CudaModel::load_with(&dev, &b.bytes, &cpu).unwrap();
    assert_eq!(format!("{}", m.hash()), GOLDEN_MODEL);
    let mut s = CudaSession::new(&m).unwrap();
    let mut h = Sha256::new();
    for x in s.start_block().unwrap() {
        h.update(x.to_le_bytes());
    }
    for t in tokens(0x7ac0_0002, 150, 300) {
        for x in s.step(t).unwrap() {
            h.update(x.to_le_bytes());
        }
    }
    assert_eq!(hex(&h.finalize()), GOLDEN_OUTPUT);
    // A CPU model of another file: the result is the model of the bytes, with their hash.
    let other = build(tiny_dims(), 0x7ac0_0009, 64);
    let m = CudaModel::load_with(&dev, &other.bytes, &cpu).unwrap();
    assert_eq!(m.hash(), Model::load(&other.bytes).unwrap().hash());
    assert_ne!(format!("{}", m.hash()), GOLDEN_MODEL);
}

#[test]
fn golden_conformance_hash_with_prefill() {
    // The same vector in one call: 151 rows of one slot, with two restarts, in passes of 40 rows.
    let Some(dev) = device("golden_conformance_hash_with_prefill") else { return };
    let b = build(tiny_dims(), 0x7ac0_0001, 64);
    let m = CudaModel::load(&dev, &b.bytes).unwrap();
    let mut batch = CudaBatch::new(&m, 1, 40).unwrap();
    let mut inputs = vec![(0, m.eot_token())];
    inputs.extend(tokens(0x7ac0_0002, 150, 300).into_iter().map(|t| (0, t)));
    let w = batch.run_weights(&inputs).unwrap();
    let mut h = Sha256::new();
    for x in w {
        h.update(x.to_le_bytes());
    }
    assert_eq!(hex(&h.finalize()), GOLDEN_OUTPUT);
}

#[test]
fn committed_tiny_model_equals_the_cpu_at_every_step() {
    let Some(dev) = device("committed_tiny_model_equals_the_cpu_at_every_step") else { return };
    let bytes =
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../conformance/tiny.hsm")).unwrap();
    let cpu = Model::load(&bytes).unwrap();
    let gpu = CudaModel::load(&dev, &bytes).unwrap();
    let mut c = Session::new(&cpu);
    let mut g = CudaSession::new(&gpu).unwrap();
    assert_eq!(c.start_block().unwrap(), g.start_block().unwrap());
    for (i, t) in tokens(0x7ac0_0003, 200, 300).into_iter().enumerate() {
        let want = c.step(t).unwrap().to_vec();
        assert_eq!(g.step(t).unwrap(), &want[..], "step {i}");
        assert_eq!(c.position(), g.position(), "step {i}");
    }
}

/// CPU sessions and one GPU batch with the same token streams; each call gives every slot
/// 0 to 3 tokens (decode and prefill together), some slots start a new block.
fn batch_against_sessions(slots: usize, calls: usize, seed: u64, max_rows: usize) {
    let Some(dev) = device("batch_against_sessions") else { return };
    let b = build(tiny_dims(), seed, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let mut sessions: Vec<Session<'_>> = (0..slots).map(|_| Session::new(&cpu)).collect();
    let mut batch = CudaBatch::new(&gpu, slots, max_rows).unwrap();
    let mut r = common::Rng::new(seed ^ 0x5eed);
    let v = gpu.vocab_size();
    for call in 0..calls {
        let mut inputs = Vec::new();
        let mut wants = Vec::new();
        for _ in 0..3 {
            for (slot, s) in sessions.iter_mut().enumerate() {
                if r.below(3) == 0 {
                    continue;
                }
                let token = if r.below(97) == 0 && !inputs.iter().any(|&(x, _)| x == slot) {
                    batch.reset(slot);
                    s.reset();
                    gpu.eot_token()
                } else {
                    TokenId::new(r.below(300) as u32)
                };
                wants.push(s.step(token).unwrap().to_vec());
                inputs.push((slot, token));
            }
        }
        let got = batch.run_weights(&inputs).unwrap();
        for (i, want) in wants.iter().enumerate() {
            assert_eq!(&got[i * v..(i + 1) * v], &want[..], "call {call} input {i}");
        }
        for (slot, s) in sessions.iter().enumerate() {
            assert_eq!(batch.position(slot), s.position(), "call {call} slot {slot}");
        }
    }
}

#[test]
fn small_batch_with_prefill_equals_independent_sessions() {
    batch_against_sessions(3, 60, 21, 512);
}

#[test]
fn large_batch_with_prefill_equals_independent_sessions() {
    // 12 slots, up to 36 rows in a call, passes of at most 16 rows.
    batch_against_sessions(12, 40, 22, 16);
}

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

#[test]
fn coder_intervals_equal_the_cpu_table() {
    // 4 slots: passes of at most 8 rows (dp4a products, coder op on i64 logits); 40 slots: passes
    // of up to 80 rows (tensor-core products, coder op on the compact logits).
    coder_against_cpu(4, 60);
    coder_against_cpu(40, 20);
}

fn coder_against_cpu(slots: usize, calls: usize) {
    let Some(dev) = device("coder_intervals_equal_the_cpu_table") else { return };
    let b = build(tiny_dims(), 31, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let mut sessions: Vec<Session<'_>> = (0..slots).map(|_| Session::new(&cpu)).collect();
    let mut batch = CudaBatch::new(&gpu, slots, 2 * slots).unwrap();
    let mut r = common::Rng::new(0xc0de);
    for call in 0..calls {
        let bits = [32u32, 12, 9][call % 3];
        let total = 1u64 << bits;
        let (mut inputs, mut queries, mut wants) = (Vec::new(), Vec::new(), Vec::new());
        // Each slot gets 1 or 2 tokens: decode rows and prefill rows.
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
                let w = s.step(token).unwrap();
                let q = match r.below(4) {
                    0 => CoderQuery::Decode(0),
                    1 => CoderQuery::Decode(total - 1),
                    2 => CoderQuery::Decode(r.below(total)),
                    _ => CoderQuery::Encode(TokenId::new(r.below(300) as u32)),
                };
                wants.push(cpu_interval(w, bits, q));
                inputs.push((slot, token));
                queries.push(q);
            }
        }
        assert_eq!(
            batch.run_coded(&inputs, &queries, bits).unwrap(),
            wants,
            "call {call} bits {bits}"
        );
    }
}

/// One slot of a model with these dimensions: all steps in passes of `rows` rows (prefill), and
/// the same tokens one at a time in a second slot (decode), against the CPU at every step.
fn shape_against_cpu(dims: Dims, seed: u64, steps: usize, rows: usize) {
    let Some(dev) = device("shape_against_cpu") else { return };
    let b = build(dims, seed, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let v = gpu.vocab_size();
    let toks = tokens(seed ^ 0xa77e, steps, dims.vocab);
    let mut s = Session::new(&cpu);
    let mut want = Vec::with_capacity(steps * v);
    for &t in &toks {
        want.extend_from_slice(s.step(t).unwrap());
    }
    let mut batch = CudaBatch::new(&gpu, 2, rows).unwrap();
    let inputs: Vec<(usize, TokenId)> = toks.iter().map(|&t| (0, t)).collect();
    let got = batch.run_weights(&inputs).unwrap();
    for i in 0..steps {
        assert_eq!(&got[i * v..(i + 1) * v], &want[i * v..(i + 1) * v], "prefill step {i}");
    }
    for (i, &t) in toks.iter().enumerate() {
        let got = batch.run_weights(&[(1, t)]).unwrap();
        assert_eq!(&got[..], &want[i * v..(i + 1) * v], "decode step {i}");
    }
}

#[test]
fn attention_with_head_dim_32_and_a_context_of_30() {
    // head_dim 32: two lanes read one key; n_ctx is not a multiple of 4; two restarts.
    let dims = Dims { n_layer: 1, n_head: 2, d_model: 64, d_ff: 128, vocab: 300, n_ctx: 30 };
    shape_against_cpu(dims, 41, 75, 16);
}

#[test]
fn attention_with_head_dim_12() {
    // head_dim 12: one thread reads one key in words; three threads share a quad of values.
    let dims = Dims { n_layer: 1, n_head: 4, d_model: 48, d_ff: 96, vocab: 300, n_ctx: 21 };
    shape_against_cpu(dims, 42, 50, 7);
}

#[test]
fn attention_with_head_dim_256_and_a_context_of_1100() {
    // head_dim 256: sixteen lanes read one key; the scores of a context longer than 1,024
    // positions are in global memory; one restart.
    let dims = Dims { n_layer: 1, n_head: 1, d_model: 256, d_ff: 256, vocab: 300, n_ctx: 1100 };
    shape_against_cpu(dims, 43, 1150, 512);
}

#[test]
fn attention_of_row_groups_with_head_dim_64() {
    // head_dim 64 (GPT-2): prefill passes run attend_rows (groups of up to 16 rows, tensor
    // cores); passes of 37 rows cut the groups; n_ctx 100 is not a multiple of 32; two restarts.
    let dims = Dims { n_layer: 1, n_head: 3, d_model: 192, d_ff: 256, vocab: 300, n_ctx: 100 };
    shape_against_cpu(dims, 44, 230, 37);
}

#[test]
fn attention_of_row_groups_with_head_dim_128() {
    // head_dim 128: four n-tiles of values for each warp; n_ctx 70; one restart.
    let dims = Dims { n_layer: 2, n_head: 2, d_model: 256, d_ff: 256, vocab: 300, n_ctx: 70 };
    shape_against_cpu(dims, 45, 120, 64);
}

#[test]
fn prefill_runs_of_many_slots_equal_independent_sessions() {
    // Each call gives some slots a run of 1 to 40 consecutive tokens (row groups of
    // attend_rows, also groups of one row), in passes of at most 96 rows; some slots restart.
    let Some(dev) = device("prefill_runs_of_many_slots_equal_independent_sessions") else { return };
    let dims = Dims { n_layer: 2, n_head: 2, d_model: 128, d_ff: 256, vocab: 300, n_ctx: 90 };
    let b = build(dims, 46, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let slots = 6;
    let mut sessions: Vec<Session<'_>> = (0..slots).map(|_| Session::new(&cpu)).collect();
    let mut batch = CudaBatch::new(&gpu, slots, 96).unwrap();
    let mut r = common::Rng::new(0x9a0d);
    let v = gpu.vocab_size();
    for call in 0..30 {
        let (mut inputs, mut wants) = (Vec::new(), Vec::new());
        for (slot, s) in sessions.iter_mut().enumerate() {
            if r.below(4) == 0 {
                continue;
            }
            if r.below(9) == 0 {
                batch.reset(slot);
                s.reset();
            }
            for _ in 0..1 + r.below(40) {
                let token = TokenId::new(r.below(300) as u32);
                wants.push(s.step(token).unwrap().to_vec());
                inputs.push((slot, token));
            }
        }
        let got = batch.run_weights(&inputs).unwrap();
        for (i, want) in wants.iter().enumerate() {
            assert_eq!(&got[i * v..(i + 1) * v], &want[..], "call {call} input {i}");
        }
    }
}

#[test]
fn norm_of_rows_longer_than_1024_values() {
    // d_model 1,280: the norm kernels read the row from memory (more than NORM_T * NORM_V
    // values); head_dim 256.
    let dims = Dims { n_layer: 1, n_head: 5, d_model: 1280, d_ff: 256, vocab: 300, n_ctx: 24 };
    shape_against_cpu(dims, 47, 40, 16);
}
