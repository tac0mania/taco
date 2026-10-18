//! Benchmarks of the HardShell paths on one thread: the reference session, the step path, the
//! prefill path (`Session::step_many`, 32 tokens) and the batch path (`Batch`, 8 lanes), on
//! the tiny model and on a random model with the shapes of GPT-2 small.
//!
//! `cargo bench -p hardshell --bench paths -- tiny` runs only the tiny model. The throughput
//! is in tokens (steps). The context position moves with each call and restarts when the
//! context is full, so the mean covers many positions.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::print_stdout)]
#![allow(clippy::cast_possible_truncation, missing_docs)]

#[path = "../tests/common/mod.rs"]
mod common;

use std::cell::OnceCell;
use std::hint::black_box;
use std::time::Duration;

use common::Rng;
use common::builder::{build, gpt2_small_dims, tiny_dims};
use common::reference::Dims;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use hardshell::TokenId;
use hardshell::gpt2::{Batch, BatchInput, Model, ReferenceSession, Session, SessionConfig};

const CHUNK: usize = 32;
const LANES: usize = 8;

fn bench_model(c: &mut Criterion, name: &str, dims: Dims, calib: usize, time: Duration) {
    // NOTE: the model is built one time, and only if the filter selects a benchmark of it.
    let model: OnceCell<Model> = OnceCell::new();
    let get = || model.get_or_init(|| Model::load(&build(dims, 42, calib).bytes).unwrap());
    let mut rng = Rng::new(7);
    let mut tok = move || TokenId::new(rng.below(dims.vocab as u64) as u32);
    let mut group = c.benchmark_group(name);
    group.sample_size(10).measurement_time(time);

    group.throughput(Throughput::Elements(1));
    let mut reference: Option<ReferenceSession<'_>> = None;
    group.bench_function("reference_step", |b| {
        let s = reference.get_or_insert_with(|| ReferenceSession::new(get()));
        b.iter(|| black_box(s.step(tok()).unwrap()[0]));
    });
    let mut step: Option<Session<'_>> = None;
    group.bench_function("step", |b| {
        let s = step.get_or_insert_with(|| Session::new(get()));
        b.iter(|| black_box(s.step(tok()).unwrap()[0]));
    });

    group.throughput(Throughput::Elements(CHUNK as u64));
    let mut chunk: Option<Session<'_>> = None;
    group.bench_function("step_many_32", |b| {
        let cfg = SessionConfig::new().with_chunk(CHUNK);
        let s = chunk.get_or_insert_with(|| Session::with_config(get(), cfg));
        b.iter(|| {
            let toks: Vec<TokenId> = (0..CHUNK).map(|_| tok()).collect();
            black_box(s.step_many(&toks).unwrap().get(0).map(|w| w[0]))
        });
    });

    group.throughput(Throughput::Elements(LANES as u64));
    let mut batch: Option<Batch<'_>> = None;
    group.bench_function("batch_8", |b| {
        let s = batch.get_or_insert_with(|| Batch::new(get(), LANES));
        b.iter(|| {
            let inputs: Vec<Option<BatchInput>> =
                (0..LANES).map(|_| Some(BatchInput::Token(tok()))).collect();
            s.run(&inputs).unwrap();
            black_box(s.weights(0).map(|w| w[0]))
        });
    });
    group.finish();
}

fn tiny(c: &mut Criterion) {
    bench_model(c, "tiny", tiny_dims(), 64, Duration::from_secs(3));
}

fn gpt2_small(c: &mut Criterion) {
    bench_model(c, "gpt2_small_shape", gpt2_small_dims(), 4, Duration::from_secs(20));
}

criterion_group!(benches, tiny, gpt2_small);
criterion_main!(benches);
