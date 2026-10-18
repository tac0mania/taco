//! Integer attention softmax with an exponent table (plan §7.5).
//!
//! For one row of scores `s` (already rescaled to the units of the table):
//!
//! 1. `d[j] = max(s) - s[j]`, so `d[j] >= 0`.
//! 2. `e[j] = table[clamp(d[j], 0, len - 1)]`. A large difference takes the last entry.
//! 3. `p[j] = floor_div(e[j] * 2^P, sum(e))`.
//!
//! The converter makes the table: `table[i] = round(2^E * exp(-i * delta))`, where `delta` is
//! the real value of one score unit. The table is data in the model file.
//!
//! DETERMINISM: a table look-up with a clamped index, an integer sum and `floor_div`. No step
//! depends on the machine.
//!
//! PLAN(§7.5). ADR 0004 records `E`, `delta`, the length and `P`, with the measured error.

use crate::error::{Error, Result};
use crate::int::{Divisor, Table, clamp_wide, floor_div};

/// The largest permitted `P`.
pub const MAX_P: u8 = 30;
/// The largest number of scores in one row: `2^31`. Thus `sum(e) < 2^31 * 2^31 = 2^62`.
pub const MAX_ROW: usize = 1 << 31;

/// An exponent table for the attention softmax. The entries are in `0..2^31` (an `i32` that is
/// not negative), do not increase, and the first entry is at least 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpTable {
    table: Table<i32>,
}

impl ExpTable {
    /// Validates and makes a table. Cost: one pass.
    ///
    /// # Errors
    /// [`Error::TensorMismatch`] (with the name `attn_exp`) if the table is empty, an entry is
    /// negative, an entry is larger than the entry before it, or the first entry is 0.
    pub fn new(entries: Vec<i32>) -> Result<Self> {
        let bad = |reason| Err(Error::TensorMismatch { name: "attn_exp".into(), reason });
        if !is_valid_exp_table(&entries, 0) {
            return bad("entries must be >= 0, not increasing, and the first entry >= 1");
        }
        match Table::new(entries) {
            Some(table) => Ok(Self { table }),
            None => bad("the table is empty"),
        }
    }

    /// The entries in index order.
    #[must_use]
    pub fn entries(&self) -> &[i32] {
        self.table.as_slice()
    }

    /// The entry for the difference `d`, with the index clamped to the table. Cost: one
    /// look-up.
    #[must_use]
    pub fn lookup(&self, d: i128) -> i32 {
        self.table.lookup(clamp_wide(d, 0, i64::MAX))
    }
}

/// `true` if `entries` is not empty, every entry is `>= min`, no entry is larger than the entry
/// before it, and the first entry is at least 1.
pub(crate) fn is_valid_exp_table<T: Copy + PartialOrd + From<u8>>(entries: &[T], min: u8) -> bool {
    let Some(&first) = entries.first() else { return false };
    first >= T::from(1)
        && entries.iter().all(|&e| e >= T::from(min))
        && entries.windows(2).all(|w| w.get(1) <= w.first())
}

/// Calculates the softmax probabilities of one row of scores. `out[j]` is `p[j]`, a value in
/// `0..=2^P`, and the sum of `out` is at most `2^P`. Cost: three passes over the row.
///
/// # Errors
/// [`Error::LengthMismatch`] if `out` is shorter than `scores`, or if `scores` is empty or has
/// more than [`MAX_ROW`] entries; [`Error::InvalidHeader`] if `p > 30`.
pub fn softmax(scores: &[i64], table: &ExpTable, p: u8, out: &mut [i64]) -> Result<()> {
    if p > MAX_P {
        return Err(Error::InvalidHeader {
            field: "constants.softmax_p".into(),
            reason: "must be at most 30",
        });
    }
    if scores.is_empty() || scores.len() > MAX_ROW || out.len() < scores.len() {
        return Err(Error::LengthMismatch { expected: scores.len(), found: out.len() });
    }
    let max = scores.iter().copied().max().unwrap_or(0);
    let mut sum: i64 = 0;
    for (o, &s) in out.iter_mut().zip(scores) {
        let e = i64::from(table.lookup(i128::from(max) - i128::from(s)));
        *o = e;
        // INVARIANT: at most 2^31 entries below 2^31, so sum < 2^62.
        sum += e;
    }
    // INVARIANT: the maximum score has d = 0 and takes the first entry, which is >= 1, so
    // sum >= 1.
    let div = Divisor::new(sum).unwrap_or(Divisor::MIN);
    for o in out.iter_mut().take(scores.len()) {
        // INVARIANT: e < 2^31 and p <= 30, so e << p < 2^61.
        *o = floor_div(*o << p, div);
    }
    Ok(())
}
