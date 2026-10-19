//! GELU as a table with one entry for each input value (plan §7.4).
//!
//! The input is a signed integer of `b` bits, with `b = 8` (256 entries) or `b = 16` (65,536
//! entries). The entry at index `x + 2^(b-1)` is the output for the input `x`. The output is an
//! `i8` activation (`-127..=127`, format version 1), or an `i16` activation (`-32767..=32767`,
//! format version 2, ADR 0004 addendum B). A tool calculates the table one time, before
//! publication. The table is data in the model file. HardShell does only the look-up.
//!
//! DETERMINISM: HardShell does no GELU arithmetic, only a look-up with a clamped index. The
//! work happens one time in the tool, and its result is fixed in the model file.
//!
//! PLAN(§7.4).

use crate::error::{Error, Result};
use crate::int::{Table, clamp};

/// A GELU table for `i8` or `i16` inputs. The output type `T` is `i8` (the default) or `i16`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeluTable<T = i8> {
    table: Table<T>,
    /// `2^(b-1)`: the index of the input 0.
    offset: i64,
}

impl<T: Copy> GeluTable<T> {
    /// Makes a table from its entries. The length selects the input type: 256 entries for
    /// `i8` inputs, 65,536 entries for `i16` inputs.
    ///
    /// NOTE: the table does not check the output range. The converter clamps each entry to the
    /// symmetric activation range, and the next layer accepts every value of `T`.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if the length is not 256 or 65,536.
    pub fn new(entries: Vec<T>) -> Result<Self> {
        let offset = match entries.len() {
            256 => 128,
            65_536 => 32_768,
            n => return Err(Error::LengthMismatch { expected: 65_536, found: n }),
        };
        let table = Table::new(entries).ok_or(Error::LengthMismatch { expected: 256, found: 0 })?;
        Ok(Self { table, offset })
    }

    /// The smallest input of the table: `-2^(b-1)`.
    #[must_use]
    pub fn min_input(&self) -> i64 {
        -self.offset
    }

    /// The largest input of the table: `2^(b-1) - 1`.
    #[must_use]
    pub fn max_input(&self) -> i64 {
        self.offset - 1
    }

    /// GELU of `x`. An input outside the table range is clamped to it. Cost: one look-up.
    #[must_use]
    pub fn apply(&self, x: i64) -> T {
        let x = clamp(x, -self.offset, self.offset - 1);
        // INVARIANT: -2^15 <= x < 2^15, so x + offset is in 0..len and cannot overflow.
        self.table.lookup(x + self.offset)
    }

    /// The entries in index order.
    #[must_use]
    pub fn entries(&self) -> &[T] {
        self.table.as_slice()
    }
}
