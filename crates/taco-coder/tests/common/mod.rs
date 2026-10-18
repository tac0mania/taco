//! Helpers for the integration tests: a small integer PRNG and table builders.

#![allow(dead_code, reason = "each test file uses a different part")]
#![allow(clippy::unwrap_used, reason = "test code: a failure must stop the test")]

use taco_coder::{FreqBits, FreqTable, Symbol};

/// The splitmix64 generator (Steele, Lea, Flood, 2014). Integer only, so it is the same on
/// every machine.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A frequency bit count, for tests.
pub(crate) fn bits(b: u32) -> FreqBits {
    FreqBits::new(b).unwrap()
}

/// Skewed weights, like the output of a language model: `u64::MAX >> (x % 64)`.
pub(crate) fn skewed_weights(n: usize, seed: u64) -> Vec<u64> {
    let mut s = seed;
    (0..n).map(|_| u64::MAX >> (splitmix64(&mut s) % 64)).collect()
}

/// `n` symbols drawn from the distribution of `table`.
pub(crate) fn sample(table: &FreqTable, n: usize, seed: u64) -> Vec<Symbol> {
    let ends = ends(table);
    let mut s = seed;
    (0..n)
        .map(|_| {
            let x = splitmix64(&mut s) % table.total();
            let i = ends.partition_point(|&e| e <= x);
            Symbol::new(u32::try_from(i).unwrap())
        })
        .collect()
}

/// The cumulative end of each symbol: `ends[i] = f[0] + ... + f[i]`.
pub(crate) fn ends(table: &FreqTable) -> Vec<u64> {
    let mut acc = 0u64;
    table
        .freqs()
        .map(|f| {
            acc += f.get();
            acc
        })
        .collect()
}
