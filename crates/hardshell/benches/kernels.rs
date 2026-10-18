//! Benchmarks of the matrix kernels with the shapes of GPT-2 small: the reference `matvec_i8`
//! and each fast [`Kernels`] selection, for one input row (the step path) and for 8 and 32 input
//! rows (the prefill and batch paths).
//!
//! `cargo bench -p hardshell --bench kernels`. The output gives the time of one call; the
//! multiply-adds of one call are `rows * cols * n`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::print_stdout)]
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, missing_docs)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use hardshell::kernels::{Kernels, matvec_i8};
use hardshell::tensor::Matrix;

fn values(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (s >> 56) as i8
        })
        .collect()
}

fn bench_shape(c: &mut Criterion, name: &str, rows: usize, cols: usize) {
    let w = values(rows * cols, 1);
    let m = Matrix::new(rows, cols, w.clone()).unwrap();
    let mut group = c.benchmark_group(name);
    group.sample_size(20).measurement_time(Duration::from_secs(3));
    for n in [1usize, 8, 32] {
        let x = values(n * cols, 2);
        let mut out = vec![0i32; rows * n];
        group.throughput(Throughput::Elements((rows * cols * n) as u64));
        if n == 1 {
            group.bench_function("reference/n1", |b| {
                b.iter(|| matvec_i8(&m, black_box(&x), &mut out).unwrap());
            });
        }
        for k in Kernels::available() {
            group.bench_function(format!("{}/n{n}", k.name()), |b| {
                b.iter(|| k.matmul_i8(&w, black_box(&x), cols, &mut out).unwrap());
            });
        }
    }
    group.finish();
}

fn kernels(c: &mut Criterion) {
    bench_shape(c, "c_fc_3072x768", 3072, 768);
    bench_shape(c, "mlp_proj_768x3072", 768, 3072);
    bench_shape(c, "wte_50257x768", 50_257, 768);
}

criterion_group!(benches, kernels);
criterion_main!(benches);
