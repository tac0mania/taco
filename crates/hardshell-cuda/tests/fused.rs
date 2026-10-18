//! The fused path against the op-level path (ADR 0013: a fused kernel needs a test that proves it
//! equal to the op-level path). Both paths run the same inputs, with decode and prefill rows,
//! restarts, and coder queries; every weight and every interval must be equal. A one-bit
//! difference fails.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::print_stderr)]

#[path = "../../hardshell/tests/common/mod.rs"]
mod common;
mod support;

use common::builder::{build, tiny_dims};
use common::reference::Dims;
use hardshell::TokenId;
use hardshell_cuda::{CoderQuery, CudaBatch, CudaModel};
use support::device;

/// Runs `calls` random calls on two batches (op-level and fused) of the same model.
fn paths_agree(dims: Dims, seed: u64, slots: usize, calls: usize, max_rows: usize) {
    let Some(dev) = device("paths_agree") else { return };
    let b = build(dims, seed, 64);
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let mut op = CudaBatch::new(&gpu, slots, max_rows).unwrap();
    op.set_op_level(true);
    let mut fused = CudaBatch::new(&gpu, slots, max_rows).unwrap();
    let mut r = common::Rng::new(seed ^ 0xf05e);
    let vocab = dims.vocab as u64;
    for call in 0..calls {
        let mut inputs = Vec::new();
        for _ in 0..1 + r.below(4) {
            for slot in 0..slots {
                if r.below(3) == 0 {
                    continue;
                }
                inputs.push((slot, TokenId::new(r.below(vocab) as u32)));
            }
        }
        if inputs.is_empty() {
            inputs.push((0, TokenId::new(1)));
        }
        if call % 2 == 0 {
            assert_eq!(
                op.run_weights(&inputs).unwrap(),
                fused.run_weights(&inputs).unwrap(),
                "weights, call {call}"
            );
        } else {
            let bits = [32u32, 16, 10][call % 3];
            let total = 1u64 << bits;
            let queries: Vec<CoderQuery> = inputs
                .iter()
                .map(|_| match r.below(3) {
                    0 => CoderQuery::Decode(r.below(total)),
                    1 => CoderQuery::Decode(total - 1),
                    _ => CoderQuery::Encode(TokenId::new(r.below(vocab) as u32)),
                })
                .collect();
            assert_eq!(
                op.run_coded(&inputs, &queries, bits).unwrap(),
                fused.run_coded(&inputs, &queries, bits).unwrap(),
                "intervals, call {call}"
            );
        }
        for slot in 0..slots {
            assert_eq!(op.position(slot), fused.position(slot), "call {call} slot {slot}");
        }
    }
}

#[test]
fn fused_path_equals_the_op_level_path_on_the_tiny_model() {
    // 6 slots, up to 24 rows in a call, passes of at most 10 rows, many restarts (n_ctx 64).
    paths_agree(tiny_dims(), 51, 6, 120, 10);
}

#[test]
fn fused_path_equals_the_op_level_path_in_large_passes() {
    // Passes of up to 200 rows: the tensor-core products with 128-row tiles.
    paths_agree(tiny_dims(), 52, 64, 30, 200);
}

#[test]
fn fused_path_equals_the_op_level_path_for_other_shapes() {
    let shapes = [
        Dims { n_layer: 1, n_head: 2, d_model: 64, d_ff: 128, vocab: 300, n_ctx: 30 },
        Dims { n_layer: 3, n_head: 4, d_model: 48, d_ff: 96, vocab: 257, n_ctx: 21 },
        Dims { n_layer: 1, n_head: 1, d_model: 256, d_ff: 512, vocab: 1000, n_ctx: 40 },
    ];
    for (i, dims) in shapes.into_iter().enumerate() {
        paths_agree(dims, 60 + i as u64, 5, 40, 12);
    }
}
