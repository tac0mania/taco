//! Speed of the coder with a 50,257-symbol table (the GPT-2 vocabulary size).
//!
//! The table comes from `from_weights` with skewed weights, so a few symbols are frequent,
//! as with a language model. The symbols are drawn from the table's own distribution.

#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs, reason = "benchmark code")]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use taco_coder::{Decoder, Encoder, FreqBits, FreqTable, Symbol};

const VOCAB: usize = 50_257;
const SYMBOLS: usize = 100_000;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Weights like a model's output: the weight of a symbol falls fast with a random rank.
fn weights(seed: u64) -> Vec<u64> {
    let mut s = seed;
    (0..VOCAB).map(|_| u64::MAX >> (splitmix64(&mut s) % 64)).collect()
}

/// Symbols drawn from the distribution of `table`.
fn sample(table: &FreqTable, n: usize, seed: u64) -> Vec<Symbol> {
    let mut ends = Vec::with_capacity(table.symbol_count());
    let mut acc = 0u64;
    for f in table.freqs() {
        acc += f.get();
        ends.push(acc);
    }
    let mut s = seed;
    (0..n)
        .map(|_| {
            let x = splitmix64(&mut s) % table.total();
            let i = ends.partition_point(|&e| e <= x);
            Symbol::new(u32::try_from(i).unwrap())
        })
        .collect()
}

fn bench_coder(c: &mut Criterion) {
    let table = FreqTable::from_weights(&weights(1), FreqBits::new(32).unwrap()).unwrap();
    let symbols = sample(&table, SYMBOLS, 2);
    let mut e = Encoder::new();
    for &s in &symbols {
        e.encode(&table, s).unwrap();
    }
    let bytes = e.finish();

    let mut g = c.benchmark_group("coder_50257");
    g.throughput(Throughput::Elements(SYMBOLS as u64));
    g.bench_function("encode", |b| {
        b.iter(|| {
            let mut e = Encoder::with_capacity(bytes.len());
            for &s in &symbols {
                e.encode(&table, s).unwrap();
            }
            black_box(e.finish())
        });
    });
    g.bench_function("decode", |b| {
        b.iter(|| {
            let mut d = Decoder::new(black_box(&bytes));
            let mut x = 0u32;
            for _ in 0..SYMBOLS {
                x ^= d.decode(&table).get();
            }
            black_box(x)
        });
    });
    g.finish();

    let w = weights(3);
    let mut g = c.benchmark_group("freq_50257");
    g.throughput(Throughput::Elements(VOCAB as u64));
    g.bench_function("from_weights", |b| {
        b.iter(|| black_box(FreqTable::from_weights(black_box(&w), FreqBits::new(32).unwrap())))
    });
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = bench_coder
}
criterion_main!(benches);
