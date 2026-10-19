//! Fuzz tests of the range decoder (docs/security.md, target T4).
//!
//! The decoder reads untrusted bytes: the coded blocks of a container. The required behaviour
//! is: no panic, no read out of bounds, every symbol is in the table, a bounded number of input
//! bytes for each call, and the GPU path (`target` and `decode_interval`) gives the state of
//! the table path. The container checksum finds a wrong symbol (plan §3 rule 4).
//!
//! Each case comes from one `u64` seed, so a failure is reproducible: the message names the
//! seed. The proptest cases run in every `cargo test`. The long run is `#[ignore]`:
//!
//! ```text
//! TACO_FUZZ_CASES=5000000 TACO_FUZZ_SEED=1 cargo test --release -p taco-coder \
//!     --test fuzz_decoder -- --ignored --nocapture
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::cast_possible_truncation,
    reason = "test code: a failure must stop the test with a message"
)]

use proptest::prelude::*;
use taco_coder::{CodeInterval, Decoder, Encoder, FreqBits, FreqTable, Prob12, Symbol};

/// The splitmix64 generator: integer only, the same on every machine.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// The smallest `b` with `2^b >= n`.
fn ceil_log2(n: u64) -> u32 {
    if n <= 1 { 0 } else { 64 - (n - 1).leading_zeros() }
}

/// A random table: 1 to 50,257 symbols, any total that holds them, and weights with zeros,
/// huge values and skew.
fn table(rng: &mut Rng) -> FreqTable {
    let n = match rng.below(8) {
        0 => 1,
        1 => 2,
        2..=5 => 1 + rng.below(300),
        6 => 1 + rng.below(4096),
        _ => 50_257,
    };
    let lo = ceil_log2(n);
    let bits = FreqBits::new(lo + rng.below(u64::from(33 - lo)) as u32).unwrap();
    let style = rng.below(5);
    let weights: Vec<u64> = (0..n)
        .map(|_| match style {
            0 => 0,
            1 => rng.next(),
            2 => u64::MAX - rng.below(3),
            3 => rng.below(4),
            _ => 1u64 << rng.below(64),
        })
        .collect();
    FreqTable::from_weights(&weights, bits).unwrap()
}

/// Random input: empty, all zero, all 0xFF, random bytes or valid coder output with a tail.
fn input(rng: &mut Rng, t: &FreqTable) -> Vec<u8> {
    let len = rng.below(80) as usize;
    match rng.below(5) {
        0 => vec![0; len],
        1 => vec![0xFF; len],
        2 => {
            let mut e = Encoder::new();
            let count = t.symbol_count() as u64;
            for _ in 0..rng.below(64) {
                e.encode(t, Symbol::new(rng.below(count) as u32)).unwrap();
            }
            let mut out = e.finish();
            out.extend((0..rng.below(4)).map(|_| rng.next() as u8));
            out
        }
        _ => (0..len).map(|_| rng.next() as u8).collect(),
    }
}

/// One fuzz case. Panics (with the seed) if the decoder breaks a rule.
fn run_case(seed: u64) {
    let mut rng = Rng(seed);
    let t = table(&mut rng);
    let bytes = input(&mut rng, &t);
    let count = t.symbol_count();
    let bits = t.freq_bits();
    let mut d = Decoder::new(&bytes);
    for step in 0..rng.below(400) {
        match rng.below(4) {
            0 | 1 => {
                let s = d.decode(&t);
                assert!((s.get() as usize) < count, "seed {seed} step {step}: symbol {s:?}");
            }
            2 => {
                let p = Prob12::new(1 + rng.below(4095) as u16).unwrap();
                let _bit: bool = d.decode_bit(p);
            }
            _ => {
                // The GPU path: target, then the interval of the symbol at the target. It must
                // give the symbol and the state of the table path.
                let mut table_path = d.clone();
                let target = d.target(bits);
                let (s, iv) = t.symbol_at(target).unwrap_or_else(|| {
                    panic!("seed {seed} step {step}: target {target} not below the total")
                });
                // A wrong interval is refused and leaves the state as it was.
                if let Some(wrong) = wrong_interval(&t, iv) {
                    let before = format!("{d:?}");
                    assert!(d.decode_interval(wrong).is_err(), "seed {seed} step {step}");
                    assert_eq!(format!("{d:?}"), before, "seed {seed} step {step}");
                }
                d.decode_interval(iv).unwrap();
                assert_eq!(table_path.decode(&t), s, "seed {seed} step {step}");
                assert_eq!(format!("{d:?}"), format!("{table_path:?}"), "seed {seed} step {step}");
            }
        }
    }
}

/// An interval of the table that does not hold the target of `right`, if the table has one.
fn wrong_interval(t: &FreqTable, right: CodeInterval) -> Option<CodeInterval> {
    let end = right.start() + right.freq();
    let other = if end < t.total() { end } else { right.start().checked_sub(1)? };
    t.symbol_at(other).map(|(_, iv)| iv)
}

/// The valid round trip: symbols that the encoder wrote come back from the decoder, also
/// with zero bytes after the end (ADR 0003).
fn round_trip_case(seed: u64) {
    let mut rng = Rng(seed);
    let t = table(&mut rng);
    let count = t.symbol_count() as u64;
    let symbols: Vec<Symbol> =
        (0..rng.below(200)).map(|_| Symbol::new(rng.below(count) as u32)).collect();
    let mut e = Encoder::new();
    for &s in &symbols {
        e.encode(&t, s).unwrap();
    }
    let mut bytes = e.finish();
    bytes.extend(std::iter::repeat_n(0, rng.below(16) as usize));
    let mut d = Decoder::new(&bytes);
    for (i, &s) in symbols.iter().enumerate() {
        assert_eq!(d.decode(&t), s, "seed {seed} symbol {i}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_048))]

    #[test]
    fn decoder_obeys_its_rules_on_any_input(seed in any::<u64>()) {
        run_case(seed);
    }

    #[test]
    fn decoder_restores_what_the_encoder_wrote(seed in any::<u64>()) {
        round_trip_case(seed);
    }
}

/// The number of cases and the first seed of a long run, from `TACO_FUZZ_CASES` and
/// `TACO_FUZZ_SEED`.
fn long_run_params(default_cases: u64) -> (u64, u64) {
    let get = |name: &str, default: u64| {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    (get("TACO_FUZZ_CASES", default_cases), get("TACO_FUZZ_SEED", 1))
}

/// The long randomized run, on all processors. Each thread takes every `n`-th seed.
#[test]
#[ignore = "long run: start it with --ignored (docs/security.md)"]
fn long_run_decoder() {
    let (cases, first) = long_run_params(1_000_000);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    std::thread::scope(|s| {
        for k in 0..threads {
            s.spawn(move || {
                let mut i = k;
                while i < cases {
                    run_case(first.wrapping_add(i));
                    round_trip_case(first.wrapping_add(i) ^ 0x5EED);
                    i += threads;
                }
            });
        }
    });
    println!(
        "fuzz_decoder: {cases} cases from seed {first} passed (each: one garbage run and one round trip)"
    );
}
