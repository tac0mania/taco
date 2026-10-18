//! Benchmarks of `Session::step` on one thread: the tiny model, and a random model with the
//! shapes of GPT-2 small (12 layers, width 768, vocabulary 50,257, context 1,024).
//!
//! `cargo bench -p hardshell --bench step -- tiny` runs only the tiny model.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::print_stdout)]
#![allow(clippy::cast_possible_truncation, missing_docs)]

#[path = "../tests/common/mod.rs"]
mod common;

use std::cell::OnceCell;
use std::hint::black_box;
use std::time::{Duration, Instant};

use common::Rng;
use common::builder::{build, gpt2_small_dims, tiny_dims};
use common::reference::Dims;
use criterion::{Criterion, criterion_group, criterion_main};
use hardshell::TokenId;
use hardshell::gpt2::{Model, Session};

fn bench_model(c: &mut Criterion, name: &str, dims: Dims, calib: usize, time: Duration) {
    // NOTE: criterion calls the closure once for each sample, so the model is built one time,
    // and only if the filter selects this benchmark.
    let model: OnceCell<Model> = OnceCell::new();
    let mut session: Option<Session<'_>> = None;
    let mut rng = Rng::new(7);
    let mut group = c.benchmark_group(name);
    group.sample_size(10).measurement_time(time);
    group.bench_function("step", |b| {
        let m = model.get_or_init(|| {
            let t0 = Instant::now();
            let built = build(dims, 42, calib);
            let m = Model::load(&built.bytes).unwrap();
            println!(
                "{name}: model file {} bytes, built and loaded in {:.1} s",
                built.bytes.len(),
                t0.elapsed().as_secs_f64()
            );
            m
        });
        let s = session.get_or_insert_with(|| {
            let mut s = Session::new(m);
            s.start_block().unwrap();
            s
        });
        // NOTE: the position moves with each step, and the session restarts when the context
        // is full. Thus the mean covers many context positions.
        b.iter(|| {
            let t = TokenId::new(rng.below(dims.vocab as u64) as u32);
            black_box(s.step(t).unwrap()[0])
        });
    });
    group.finish();
    let Some(m) = model.get() else { return };
    // One full context, from position 0 to n_ctx - 1: the mean speed over all positions.
    let mut s = Session::new(m);
    s.start_block().unwrap();
    let t0 = Instant::now();
    for _ in 1..dims.n_ctx {
        let t = TokenId::new(rng.below(dims.vocab as u64) as u32);
        black_box(s.step(t).unwrap()[0]);
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "{name}: full context of {} tokens in {secs:.2} s: {:.1} tokens/s (one thread)",
        dims.n_ctx,
        dims.n_ctx as f64 / secs
    );
}

fn tiny(c: &mut Criterion) {
    bench_model(c, "tiny", tiny_dims(), 64, Duration::from_secs(5));
}

fn gpt2_small(c: &mut Criterion) {
    bench_model(c, "gpt2_small_shape", gpt2_small_dims(), 4, Duration::from_secs(30));
}

criterion_group!(benches, tiny, gpt2_small);
criterion_main!(benches);
