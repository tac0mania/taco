//! Golden vector: a fixed table and a fixed stream of symbols and bits, with the SHA-256 of
//! the table and of the coded bytes. This is a conformance vector (plan §11). A change to the
//! coder or to `from_weights` changes a hash and needs a new format version (plan §5, §9).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{bits, ends, splitmix64};
use sha2::{Digest, Sha256};
use taco_coder::{Decoder, Encoder, FreqTable, Prob12, Symbol};

// NOTE: an independent Python implementation of the algorithm in the crate documentation
// gave the same three values on 2026-10-18 (measured).

/// SHA-256 of the frequencies of table A, each as a little-endian u64.
const TABLE_A_SHA256: &str = "6dce098f222c6f262440d5def6ca5bbb7f1cb34c4cef5a26a19835b77c56ff4d";
/// SHA-256 of the coded bytes.
const STREAM_SHA256: &str = "91676cdc669fca0eb22ed18f5f629882212523261f41d0d5c568b647e78d798f";
/// The length of the coded bytes.
const STREAM_LEN: usize = 15_689;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    A(Symbol),
    B(Symbol),
    Bit(Prob12, bool),
}

/// Table A: 1,000 symbols from `from_weights` (T = 2^24). Table B: frequency-1 symbols at the
/// bottom and at the top of a 2^32 total, which makes carries.
fn tables() -> (FreqTable, FreqTable) {
    let mut s = 0x7AC0_C0DE;
    let w: Vec<u64> = (0..1000).map(|_| splitmix64(&mut s) >> (splitmix64(&mut s) % 64)).collect();
    let a = FreqTable::from_weights(&w, bits(24)).unwrap();
    let b = FreqTable::from_freqs(&[1, (1 << 32) - 2, 1], bits(32)).unwrap();
    (a, b)
}

fn ops(a: &FreqTable) -> Vec<Op> {
    let a_ends = ends(a);
    let mut s = 0x5EED;
    (0..20_000)
        .map(|_| {
            let x = splitmix64(&mut s);
            match x % 8 {
                0..=4 => {
                    let t = (x >> 8) % a.total();
                    Op::A(Symbol::new(u32::try_from(a_ends.partition_point(|&e| e <= t)).unwrap()))
                }
                5 => Op::B(Symbol::new(u32::try_from((x >> 8) % 3).unwrap())),
                _ => {
                    let p = u16::try_from((x >> 8) % 4095 + 1).unwrap();
                    Op::Bit(Prob12::new(p).unwrap(), (x >> 40) % 4096 < u64::from(p))
                }
            }
        })
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn golden_stream_has_the_published_hashes() {
    let (a, b) = tables();
    let table_bytes: Vec<u8> = a.freqs().flat_map(|f| f.get().to_le_bytes()).collect();

    let ops = ops(&a);
    let mut e = Encoder::new();
    for &op in &ops {
        match op {
            Op::A(s) => e.encode(&a, s).unwrap(),
            Op::B(s) => e.encode(&b, s).unwrap(),
            Op::Bit(p, bit) => e.encode_bit(p, bit),
        }
    }
    let bytes = e.finish();

    let mut d = Decoder::new(&bytes);
    for &op in &ops {
        let got = match op {
            Op::A(_) => Op::A(d.decode(&a)),
            Op::B(_) => Op::B(d.decode(&b)),
            Op::Bit(p, _) => Op::Bit(p, d.decode_bit(p)),
        };
        assert_eq!(got, op);
    }

    println!("table A sha256 {}", hex(&table_bytes));
    println!("stream sha256 {} length {}", hex(&bytes), bytes.len());
    assert_eq!(hex(&table_bytes), TABLE_A_SHA256);
    assert_eq!(bytes.len(), STREAM_LEN);
    assert_eq!(hex(&bytes), STREAM_SHA256);
}
