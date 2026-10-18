//! Edge cases of the coder: empty input, one symbol, frequency-1 symbols, the 2^32 total and
//! the GPT-2 vocabulary size.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{bits, sample, skewed_weights};
use taco_coder::{Decoder, Encoder, FreqTable, Prob12, Symbol};

fn round_trip(table: &FreqTable, symbols: &[Symbol]) -> Vec<u8> {
    let mut e = Encoder::new();
    for &s in symbols {
        e.encode(table, s).unwrap();
    }
    let bytes = e.finish();
    let mut d = Decoder::new(&bytes);
    for (i, &s) in symbols.iter().enumerate() {
        assert_eq!(d.decode(table), s, "symbol {i}");
    }
    bytes
}

#[test]
fn empty_sequence_gives_empty_output() {
    let t = FreqTable::from_weights(&[1, 2, 3], bits(32)).unwrap();
    assert!(round_trip(&t, &[]).is_empty());
}

#[test]
fn one_symbol_round_trips_in_every_table_size() {
    for b in 1..=32 {
        let t = FreqTable::from_weights(&[5, 1], bits(b)).unwrap();
        for s in 0..2 {
            round_trip(&t, &[Symbol::new(s)]);
        }
    }
}

#[test]
fn frequency_one_symbols_round_trip_at_total_2_pow_32() {
    // 1,000 symbols of frequency 1 and one symbol with the rest, in the middle.
    let mut f = vec![1u64; 1001];
    f[500] = (1 << 32) - 1000;
    let t = FreqTable::from_freqs(&f, bits(32)).unwrap();
    let symbols: Vec<Symbol> = (0..5000u32).map(|i| Symbol::new((i * 37) % 1001)).collect();
    round_trip(&t, &symbols);
}

#[test]
fn gpt2_vocabulary_table_round_trips() {
    let t = FreqTable::from_weights(&skewed_weights(50_257, 11), bits(32)).unwrap();
    assert_eq!(t.symbol_count(), 50_257);
    let mut symbols = sample(&t, 20_000, 12);
    // Every symbol id at least once, including the rarest ones.
    symbols.extend((0..50_257u32).map(Symbol::new));
    round_trip(&t, &symbols);
}

#[test]
fn gpt2_vocabulary_with_uniform_weights_round_trips() {
    let t = FreqTable::from_weights(&vec![0; 50_257], bits(16)).unwrap();
    let symbols: Vec<Symbol> = (0..50_257u32).rev().map(Symbol::new).collect();
    let bytes = round_trip(&t, &symbols);
    // Uniform over 2^16 slots: each symbol costs at most 16 bits.
    assert!(bytes.len() <= 50_257 * 2 + 1, "{} bytes", bytes.len());
}

#[test]
fn the_first_and_the_last_symbol_alternate_at_2_pow_32() {
    let t = FreqTable::from_freqs(&[1, (1 << 32) - 2, 1], bits(32)).unwrap();
    let symbols: Vec<Symbol> =
        (0..10_000u32).map(|i| Symbol::new(if i % 2 == 0 { 0 } else { 2 })).collect();
    round_trip(&t, &symbols);
}

#[test]
fn bits_and_symbols_share_one_stream() {
    let t = FreqTable::from_weights(&[9, 0, 4, 1], bits(20)).unwrap();
    let mut e = Encoder::new();
    for i in 0..3000u32 {
        e.encode(&t, Symbol::new(i % 4)).unwrap();
        e.encode_bit(Prob12::new(u16::try_from(i % 4095 + 1).unwrap()).unwrap(), i % 3 == 0);
    }
    let bytes = e.finish();
    let mut d = Decoder::new(&bytes);
    for i in 0..3000u32 {
        assert_eq!(d.decode(&t), Symbol::new(i % 4));
        assert_eq!(
            d.decode_bit(Prob12::new(u16::try_from(i % 4095 + 1).unwrap()).unwrap()),
            i % 3 == 0
        );
    }
}

#[test]
fn prob12_refuses_zero_and_4096() {
    assert!(Prob12::new(0).is_err());
    assert!(Prob12::new(4096).is_err());
    assert_eq!(Prob12::new(4095).unwrap(), Prob12::MAX);
}
