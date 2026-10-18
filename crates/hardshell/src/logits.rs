//! The output logits to one `u64` weight for each token, with an exponent table (plan §7.6).
//!
//! For the logits `l` (already rescaled to the units of the table):
//! `w[i] = table[clamp(max(l) - l[i], 0, len - 1)]`.
//!
//! The converter makes the table: `table[i] = round(2^E * exp(-i * delta))`. The weights go to
//! `taco-coder::freq`, which gives every token a frequency of 1 or more (plan §5.1).
//!
//! Bound: every entry is at most `2^40` and the vocabulary has at most `2^20` tokens, so the sum
//! of the weights is at most `2^60` and fits in a `u64`.
//!
//! DETERMINISM: one integer subtraction and one table look-up for each token.
//!
//! PLAN(§7.6). ADR 0004 records `E`, `delta` and the length, with the measured error.

use crate::error::{Error, Result};
use crate::int::{Table, clamp_wide};
use crate::softmax::is_valid_exp_table;

/// The largest permitted entry of the logit table: `2^40`.
pub const MAX_ENTRY: u64 = 1 << 40;
/// The largest permitted vocabulary: `2^20` tokens.
pub const MAX_VOCAB: usize = 1 << 20;

/// An exponent table for the output logits. The entries are at most `2^40`, do not increase,
/// and the first entry is at least 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogitTable {
    table: Table<u64>,
}

impl LogitTable {
    /// Validates and makes a table. Cost: one pass.
    ///
    /// # Errors
    /// [`Error::TensorMismatch`] (with the name `logits_exp`) if the table is empty, an entry is
    /// larger than `2^40` or than the entry before it, or the first entry is 0.
    pub fn new(entries: Vec<u64>) -> Result<Self> {
        let bad = |reason| Err(Error::TensorMismatch { name: "logits_exp".into(), reason });
        if !is_valid_exp_table(&entries, 0) || entries.iter().any(|&e| e > MAX_ENTRY) {
            return bad("entries must be <= 2^40, not increasing, and the first entry >= 1");
        }
        match Table::new(entries) {
            Some(table) => Ok(Self { table }),
            None => bad("the table is empty"),
        }
    }

    /// The entries in index order.
    #[must_use]
    pub fn entries(&self) -> &[u64] {
        self.table.as_slice()
    }
}

/// Changes the logits into weights. The token with the largest logit has the weight
/// `table[0] >= 1`. Cost: two passes over the logits.
///
/// # Errors
/// [`Error::LengthMismatch`] if `out` and `logits` have different lengths, or if the length is
/// 0 or more than [`MAX_VOCAB`].
pub fn logits_to_weights(logits: &[i64], table: &LogitTable, out: &mut [u64]) -> Result<()> {
    if logits.is_empty() || logits.len() > MAX_VOCAB || out.len() != logits.len() {
        return Err(Error::LengthMismatch { expected: logits.len(), found: out.len() });
    }
    let max = logits.iter().copied().max().unwrap_or(0);
    for (o, &l) in out.iter_mut().zip(logits) {
        let d = clamp_wide(i128::from(max) - i128::from(l), 0, i64::MAX);
        *o = table.table.lookup(d);
    }
    Ok(())
}
