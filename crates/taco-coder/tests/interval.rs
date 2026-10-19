//! The interval path (a [`CodeInterval`] from outside a table, for example from a GPU coder op)
//! gives the bytes and the symbols of the table path, bit for bit. ADR 0003 stays valid.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{bits, sample, skewed_weights, splitmix64};
use proptest::prelude::*;
use taco_coder::{CodeInterval, CoderError, Decoder, Encoder, FreqTable, Symbol};

/// Tables from random weights, with zero and huge weights, and a sequence of their symbols.
/// Each symbol has its own table, as in Taco (one table for each token).
fn tables_and_symbols() -> impl Strategy<Value = Vec<(FreqTable, Symbol)>> {
    let weight = prop_oneof![Just(0u64), Just(u64::MAX), 0u64..16, any::<u64>()];
    let one = (proptest::collection::vec(weight, 1..300), 9u32..=32, any::<u32>()).prop_map(
        |(w, b, pick)| {
            let t = FreqTable::from_weights(&w, bits(b)).unwrap();
            let n = u32::try_from(t.symbol_count()).unwrap();
            // A third of the picks are the last symbol, which takes the remainder.
            let s = if pick % 3 == 0 { n - 1 } else { pick % n };
            (t, Symbol::new(s))
        },
    );
    proptest::collection::vec(one, 0..400)
}

fn table_bytes(items: &[(FreqTable, Symbol)]) -> Vec<u8> {
    let mut e = Encoder::new();
    for (t, s) in items {
        e.encode(t, *s).unwrap();
    }
    e.finish()
}

fn interval_bytes(items: &[(FreqTable, Symbol)]) -> Vec<u8> {
    let mut e = Encoder::new();
    for (t, s) in items {
        let iv = t.code_interval(*s).unwrap();
        // The interval crosses the boundary as three integers, as from a GPU.
        e.encode_interval(CodeInterval::new(iv.start(), iv.freq(), iv.freq_bits()).unwrap());
    }
    e.finish()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn encode_interval_gives_the_bytes_of_the_table_path(items in tables_and_symbols()) {
        prop_assert_eq!(interval_bytes(&items), table_bytes(&items));
    }

    #[test]
    fn two_part_decode_gives_the_symbols_and_state_of_the_table_path(
        items in tables_and_symbols()
    ) {
        let bytes = table_bytes(&items);
        let mut by_table = Decoder::new(&bytes);
        let mut by_interval = Decoder::new(&bytes);
        for (t, s) in &items {
            let want = by_table.decode(t);
            prop_assert_eq!(want, *s);
            let target = by_interval.target(t.freq_bits());
            let (got, iv) = t.symbol_at(target).unwrap();
            prop_assert_eq!(got, want);
            by_interval.decode_interval(iv).unwrap();
            // The same state: the next target agrees for every total.
            for b in [0u32, 9, 16, 32] {
                prop_assert_eq!(by_interval.target(bits(b)), by_table.target(bits(b)));
            }
        }
    }
}

#[test]
fn a_long_stream_of_model_like_tables_is_equal_on_both_paths() {
    // 50,257 symbols and T = 2^32, as with GPT-2 (plan §5.1).
    let mut seed = 0x7AC0_0016;
    let mut items = Vec::new();
    for k in 0..40 {
        let t = FreqTable::from_weights(&skewed_weights(50_257, k), bits(32)).unwrap();
        for s in sample(&t, 50, splitmix64(&mut seed)) {
            items.push((t.clone(), s));
        }
        items.push((t.clone(), Symbol::new(50_256)));
    }
    let bytes = table_bytes(&items);
    assert_eq!(interval_bytes(&items), bytes);
    let mut d = Decoder::new(&bytes);
    for (t, s) in &items {
        let (got, iv) = t.symbol_at(d.target(t.freq_bits())).unwrap();
        assert_eq!(got, *s);
        d.decode_interval(iv).unwrap();
    }
}

#[test]
fn decode_interval_refuses_an_interval_without_the_target_and_keeps_the_state() {
    let t = FreqTable::from_freqs(&[2, 1, 5], bits(3)).unwrap();
    let mut e = Encoder::new();
    for s in [2u32, 0, 1, 2] {
        e.encode(&t, Symbol::new(s)).unwrap();
    }
    let bytes = e.finish();
    let mut d = Decoder::new(&bytes);
    let target = d.target(bits(3));
    let (right, iv) = t.symbol_at(target).unwrap();
    assert_eq!(right, Symbol::new(2));
    let wrong = t.code_interval(Symbol::new(0)).unwrap();
    assert_eq!(
        d.decode_interval(wrong),
        Err(CoderError::TargetNotInInterval { target, start: 0, freq: 2 })
    );
    assert_eq!(d.target(bits(3)), target);
    d.decode_interval(iv).unwrap();
    for s in [0u32, 1, 2] {
        assert_eq!(d.decode(&t), Symbol::new(s));
    }
}
