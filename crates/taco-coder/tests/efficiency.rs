//! The coded size against the ideal size, the sum of -log2(p) over the coded values.
//!
//! The limit is 0.01% of the ideal size plus 8 bytes. The tests print the measured overhead.
//! Floats are permitted here: this file is a test, not the prediction path.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{bits, sample, skewed_weights, splitmix64};
use taco_coder::{Decoder, Encoder, FreqTable, Prob12, Symbol};

fn check(name: &str, coded_bytes: usize, ideal_bits: f64) {
    let coded_bits = (coded_bytes * 8) as f64;
    let overhead = coded_bits - ideal_bits;
    println!(
        "{name}: coded {coded_bytes} bytes, ideal {:.1} bytes, overhead {:.2} bytes = {:.6}% (measured)",
        ideal_bits / 8.0,
        overhead / 8.0,
        100.0 * overhead / ideal_bits
    );
    assert!(coded_bits <= ideal_bits * 1.0001 + 64.0, "{name}: overhead too large");
}

fn ideal_symbol_bits(table: &FreqTable, symbols: &[Symbol]) -> f64 {
    let total = table.total() as f64;
    symbols.iter().map(|&s| -((table.freq(s).unwrap().get() as f64) / total).log2()).sum()
}

fn symbols_case(name: &str, table: &FreqTable, symbols: &[Symbol]) {
    let mut e = Encoder::new();
    for &s in symbols {
        e.encode(table, s).unwrap();
    }
    let bytes = e.finish();
    let mut d = Decoder::new(&bytes);
    for &s in symbols {
        assert_eq!(d.decode(table), s);
    }
    check(name, bytes.len(), ideal_symbol_bits(table, symbols));
}

#[test]
fn gpt2_size_table_is_within_limit_of_ideal() {
    let t = FreqTable::from_weights(&skewed_weights(50_257, 21), bits(32)).unwrap();
    let symbols = sample(&t, 1_000_000, 22);
    symbols_case("50,257 symbols, T = 2^32, 1e6 symbols", &t, &symbols);
}

#[test]
fn small_table_is_within_limit_of_ideal() {
    let t = FreqTable::from_weights(&skewed_weights(256, 31), bits(16)).unwrap();
    let symbols = sample(&t, 1_000_000, 32);
    symbols_case("256 symbols, T = 2^16, 1e6 symbols", &t, &symbols);
}

#[test]
fn uniform_symbols_against_rare_symbols_are_within_limit() {
    // Symbols drawn uniformly by id, so rare symbols (frequency 1) occur often.
    let t = FreqTable::from_weights(&skewed_weights(1000, 41), bits(32)).unwrap();
    let mut s = 42u64;
    let symbols: Vec<Symbol> = (0..200_000)
        .map(|_| Symbol::new(u32::try_from(splitmix64(&mut s) % 1000).unwrap()))
        .collect();
    symbols_case("1,000 symbols, T = 2^32, uniform ids, 2e5 symbols", &t, &symbols);
}

#[test]
fn bits_are_within_limit_of_ideal() {
    let mut s = 51u64;
    let mut ops = Vec::with_capacity(2_000_000);
    let mut ideal = 0.0f64;
    for _ in 0..2_000_000 {
        let x = splitmix64(&mut s);
        let p = u16::try_from(x % 4095 + 1).unwrap();
        let bit = ((x >> 32) % 4096) < u64::from(p);
        let pr = f64::from(p) / 4096.0;
        ideal -= if bit { pr.log2() } else { (1.0 - pr).log2() };
        ops.push((Prob12::new(p).unwrap(), bit));
    }
    let mut e = Encoder::new();
    for &(p, b) in &ops {
        e.encode_bit(p, b);
    }
    let bytes = e.finish();
    let mut d = Decoder::new(&bytes);
    for &(p, b) in &ops {
        assert_eq!(d.decode_bit(p), b);
    }
    check("binary, random p1, 2e6 bits", bytes.len(), ideal);
}
