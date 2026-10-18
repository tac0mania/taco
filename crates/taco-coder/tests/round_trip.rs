//! Property tests: every valid table and every symbol or bit sequence comes back exactly (§5.7).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use std::collections::BTreeSet;

use common::bits;
use proptest::prelude::*;
use taco_coder::{Decoder, Encoder, FreqTable, Prob12, Symbol};

/// A valid table from random cut points: `n - 1` distinct cuts in `1..T` give `n` frequencies.
fn cut_table() -> impl Strategy<Value = FreqTable> {
    (0u32..=32)
        .prop_flat_map(|b| {
            let total = 1u64 << b;
            let max_cuts = usize::try_from((total - 1).min(300)).unwrap();
            (Just(b), proptest::collection::btree_set(1..total.max(2), 0..=max_cuts))
        })
        .prop_map(|(b, cuts): (u32, BTreeSet<u64>)| {
            let total = 1u64 << b;
            let mut freqs = Vec::new();
            let mut last = 0;
            for c in cuts.into_iter().filter(|&c| c < total) {
                freqs.push(c - last);
                last = c;
            }
            freqs.push(total - last);
            FreqTable::from_freqs(&freqs, bits(b)).unwrap()
        })
}

/// A valid table from random weights, with many zero and many huge weights.
fn weight_table() -> impl Strategy<Value = FreqTable> {
    let weight = prop_oneof![Just(0u64), Just(u64::MAX), 0u64..16, any::<u64>()];
    (proptest::collection::vec(weight, 1..400), 9u32..=32)
        .prop_map(|(w, b)| FreqTable::from_weights(&w, bits(b)).unwrap())
}

fn any_table() -> impl Strategy<Value = FreqTable> {
    prop_oneof![cut_table(), weight_table()]
}

/// A table and a sequence of its symbols. A third of the symbols are the table's rarest.
fn table_and_symbols() -> impl Strategy<Value = (FreqTable, Vec<Symbol>)> {
    any_table().prop_flat_map(|t| {
        let n = u32::try_from(t.symbol_count()).unwrap();
        let rarest = (0..n).min_by_key(|&i| t.freq(Symbol::new(i)).unwrap()).unwrap();
        let sym = prop_oneof![2 => 0..n, 1 => Just(rarest)].prop_map(Symbol::new);
        (Just(t), proptest::collection::vec(sym, 0..2000))
    })
}

#[derive(Debug, Clone)]
enum Op {
    Symbol(usize, Symbol),
    Bit(Prob12, bool),
}

fn prob() -> impl Strategy<Value = Prob12> {
    prop_oneof![Just(1u16), Just(4095u16), 1u16..=4095].prop_map(|p| Prob12::new(p).unwrap())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn symbols_round_trip((table, symbols) in table_and_symbols()) {
        let mut e = Encoder::new();
        for &s in &symbols {
            e.encode(&table, s).unwrap();
        }
        let bytes = e.finish();
        prop_assert_ne!(bytes.last(), Some(&0));
        let mut d = Decoder::new(&bytes);
        for &s in &symbols {
            prop_assert_eq!(d.decode(&table), s);
        }
    }

    #[test]
    fn bits_round_trip(ops in proptest::collection::vec((prob(), any::<bool>()), 0..4000)) {
        let mut e = Encoder::new();
        for &(p, b) in &ops {
            e.encode_bit(p, b);
        }
        let bytes = e.finish();
        let mut d = Decoder::new(&bytes);
        for &(p, b) in &ops {
            prop_assert_eq!(d.decode_bit(p), b);
        }
    }

    #[test]
    fn mixed_bits_and_symbols_round_trip(
        tables in proptest::collection::vec(any_table(), 1..4),
        raw in proptest::collection::vec((any::<bool>(), any::<u32>(), prob(), any::<bool>()), 0..2000),
    ) {
        let ops: Vec<Op> = raw
            .into_iter()
            .map(|(is_bit, x, p, b)| {
                if is_bit {
                    Op::Bit(p, b)
                } else {
                    let t = usize::try_from(x).unwrap() % tables.len();
                    let n = u32::try_from(tables[t].symbol_count()).unwrap();
                    Op::Symbol(t, Symbol::new((x >> 2) % n))
                }
            })
            .collect();
        let mut e = Encoder::new();
        for op in &ops {
            match *op {
                Op::Symbol(t, s) => e.encode(&tables[t], s).unwrap(),
                Op::Bit(p, b) => e.encode_bit(p, b),
            }
        }
        let bytes = e.finish();
        let mut d = Decoder::new(&bytes);
        for op in &ops {
            match *op {
                Op::Symbol(t, s) => prop_assert_eq!(d.decode(&tables[t]), s),
                Op::Bit(p, b) => prop_assert_eq!(d.decode_bit(p), b),
            }
        }
    }

    #[test]
    fn trailing_zero_bytes_do_not_change_the_decode((table, symbols) in table_and_symbols(), pad in 0usize..20) {
        let mut e = Encoder::new();
        for &s in &symbols {
            e.encode(&table, s).unwrap();
        }
        let mut bytes = e.finish();
        bytes.resize(bytes.len() + pad, 0);
        let mut d = Decoder::new(&bytes);
        for &s in &symbols {
            prop_assert_eq!(d.decode(&table), s);
        }
    }

    #[test]
    fn random_input_decodes_without_panic(table in any_table(), input in proptest::collection::vec(any::<u8>(), 0..64)) {
        let mut d = Decoder::new(&input);
        for i in 0..500u16 {
            let s = d.decode(&table);
            prop_assert!(usize::try_from(s.get()).unwrap() < table.symbol_count());
            let _bit: bool = d.decode_bit(Prob12::new(i * 8 + 1).unwrap());
        }
    }
}
