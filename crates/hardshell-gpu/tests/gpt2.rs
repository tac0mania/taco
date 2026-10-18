//! The GPT-2 step on the GPU against the CPU reference: the golden conformance hash, the
//! committed tiny model, and batches of independent slots. A one-bit difference fails.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::print_stderr)]

// The model builder of the hardshell tests. The golden model comes from it, so this test uses
// the same code and does not copy it.
#[path = "../../hardshell/tests/common/mod.rs"]
mod common;
mod support;

use common::builder::{build, tiny_dims};
use hardshell::TokenId;
use hardshell::gpt2::{Model, Session};
use hardshell_gpu::{GpuBatch, GpuModel, GpuSession};
use sha2::{Digest, Sha256};
use support::device;

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<TokenId> {
    let mut r = common::Rng::new(seed);
    (0..n).map(|_| TokenId::new(r.below(vocab as u64) as u32)).collect()
}

fn hex(h: &[u8]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

/// The golden vector of `crates/hardshell/tests/gpt2.rs`: the same model, tokens and hash.
const GOLDEN_MODEL: &str = "e3055785bc4509abf7b017e52a0f99030a508cfe0810293e7a5efdccdf2d7bb9";
const GOLDEN_OUTPUT: &str = "d4063522517854fdcab605c3d3159bd3ac9a15eb92fa6721ff4a276658f90078";

#[test]
fn golden_conformance_hash_on_the_gpu() {
    let Some(dev) = device("golden_conformance_hash_on_the_gpu") else { return };
    let b = build(tiny_dims(), 0x7ac0_0001, 64);
    let m = GpuModel::load(&dev, &b.bytes).unwrap();
    assert_eq!(format!("{}", m.hash()), GOLDEN_MODEL);
    let mut s = GpuSession::new(&m).unwrap();
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
fn committed_tiny_model_equals_the_cpu_at_every_step() {
    let Some(dev) = device("committed_tiny_model_equals_the_cpu_at_every_step") else { return };
    // Block 0 has an i16 GELU table and block 1 an i8 table (conformance/README.md).
    let bytes =
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../conformance/tiny.hsm")).unwrap();
    let cpu = Model::load(&bytes).unwrap();
    let gpu = GpuModel::load(&dev, &bytes).unwrap();
    assert_eq!(cpu.hash(), gpu.hash());
    let mut c = Session::new(&cpu);
    let mut g = GpuSession::new(&gpu).unwrap();
    assert_eq!(c.start_block().unwrap(), g.start_block().unwrap());
    // 200 tokens: more than three contexts of 64, so the restart occurs three times.
    for (i, t) in tokens(0x7ac0_0003, 200, 300).into_iter().enumerate() {
        let want = c.step(t).unwrap().to_vec();
        assert_eq!(g.step(t).unwrap(), &want[..], "step {i}");
        assert_eq!(c.position(), g.position(), "step {i}");
    }
}

/// Runs `slots` CPU sessions and one GPU batch with the same token streams. Each step uses a
/// changing subset of the slots, some slots restart their block, and every weight vector of
/// every slot must agree.
fn batch_against_sessions(slots: usize, steps: usize, seed: u64) {
    let Some(dev) = device("batch_against_sessions") else { return };
    let b = build(tiny_dims(), seed, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = GpuModel::load(&dev, &b.bytes).unwrap();
    let mut sessions: Vec<Session<'_>> = (0..slots).map(|_| Session::new(&cpu)).collect();
    let mut batch = GpuBatch::new(&gpu, slots).unwrap();
    let mut r = common::Rng::new(seed ^ 0x5eed);
    for step in 0..steps {
        let mut inputs = Vec::new();
        let mut wants = Vec::new();
        for (slot, s) in sessions.iter_mut().enumerate() {
            // About 3 of 4 slots take part in each step.
            if r.below(4) == 0 {
                continue;
            }
            let token = if r.below(97) == 0 {
                // A new block: the GPU path does reset + step(eot), as start_block does.
                batch.reset(slot);
                s.reset();
                gpu.eot_token()
            } else {
                TokenId::new(r.below(300) as u32)
            };
            wants.push(s.step(token).unwrap().to_vec());
            inputs.push((slot, token));
        }
        batch.step(&inputs).unwrap();
        for (&(slot, _), want) in inputs.iter().zip(&wants) {
            assert_eq!(batch.weights(slot).unwrap(), &want[..], "step {step} slot {slot}");
            assert_eq!(batch.position(slot), sessions[slot].position());
        }
    }
}

#[test]
fn small_batch_equals_independent_sessions() {
    batch_against_sessions(5, 150, 21);
}

#[test]
fn large_batch_equals_independent_sessions() {
    // 12 slots: about 9 rows in each step, so the tiled kernel runs.
    batch_against_sessions(12, 90, 22);
}

#[test]
fn step_refuses_bad_slots_and_tokens() {
    let Some(dev) = device("step_refuses_bad_slots_and_tokens") else { return };
    let b = build(tiny_dims(), 1, 16);
    let gpu = GpuModel::load(&dev, &b.bytes).unwrap();
    let mut batch = GpuBatch::new(&gpu, 2).unwrap();
    let t = TokenId::new(5);
    assert!(batch.step(&[(2, t)]).is_err());
    assert!(batch.step(&[(0, t), (0, t)]).is_err());
    assert!(batch.step(&[(1, TokenId::new(300))]).is_err());
    assert_eq!(batch.position(0), 0);
    assert_eq!(batch.position(1), 0);
}
