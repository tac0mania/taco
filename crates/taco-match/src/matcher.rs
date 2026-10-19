//! The token match model: the longest earlier occurrence of the current context in the block.
//!
//! PLAN(§7.7): blocks are independent, so the model starts empty at each block start
//! ([`MatchModel::reset`]). ADR 0010 section 2 records the rules. `tools/logs/matchmodel.py` is
//! a Python copy of the same rules.
//!
//! # Rules
//!
//! The history `h` holds the tokens of the block that the model has seen. Position `i` is the
//! position of the next token, so the context of position `i` is `h[i-N..i)` for order `N`.
//!
//! 1. Hash of order `N` at position `i >= N`, all operations modulo 2^64:
//!    `H = sum over k in 1..=N of (h[i-k] + 1) * B^(k-1)`, then
//!    `slot = ((H ^ (H >> 32)) * MIX) >> (64 - hash_bits)`.
//! 2. Each order has one table of `2^hash_bits` entries. An entry holds the last position
//!    `j >= N` with that slot (stored as `j + 1`; 0 is empty).
//! 3. After the token at position `i - 1` is known ([`MatchModel::observe`]):
//!    1. If the match length `len > 0` and `h[ptr]` is that token, `ptr += 1` and
//!       `len = min(len + 1, max_len)`. Else `len = 0`.
//!    2. Append the token to `h`. Now the next position is `i`.
//!    3. If `len` is less than the largest order, look up the orders `N > len` with `N <= i`,
//!       longest first. A candidate `c` from the table of order `N` is accepted if the
//!       verified length `L` (the count of `k` in `1..=verify_max` with `h[c-k] = h[i-k]`,
//!       stopping at the first difference or at the block start) is at least `N`. Then
//!       `ptr = c` and `len = L`. The first accepted order stops the look-up.
//!    4. Put position `i` into the table of each order `N <= i`.
//! 4. The prediction ([`MatchModel::predict`]) is `h[ptr]` with the length `len`, if `len > 0`.
//!
//! DETERMINISM: integer operations only (wrapping multiplication and addition of `u64`, shifts,
//! comparisons). The table contents depend only on the tokens of the block.

use crate::error::MatchError;
use crate::types::{MatchLen, Prediction, TokenId};

/// The hash multiplier `B` (the golden-ratio constant of splitmix64).
const HASH_B: u64 = 0x9E37_79B9_7F4A_7C15;
/// The final mix multiplier.
const HASH_MIX: u64 = 0xD6E8_FEB8_6659_FD93;
/// The largest permitted order.
pub const MAX_ORDER: u8 = 64;
/// The largest number of orders.
pub const MAX_ORDERS: usize = 8;

/// The parameters of the match model (ADR 0010 section 2). They are part of the format: a
/// change changes the compressed bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatchParams {
    orders: [u8; MAX_ORDERS],
    count: u8,
    hash_bits: u8,
    verify_max: u16,
    max_len: u32,
}

impl MatchParams {
    /// The parameters of mixer rule version 1 (ADR 0010): orders 32, 16, 8, 4, 2 and 1,
    /// tables of 2^20 entries, verification of at most 64 tokens, lengths up to 65,535.
    pub const V1: Self = Self {
        orders: [32, 16, 8, 4, 2, 1, 0, 0],
        count: 6,
        hash_bits: 20,
        verify_max: 64,
        max_len: 65_535,
    };

    /// Makes and checks parameters.
    ///
    /// # Errors
    ///
    /// [`MatchError::InvalidParams`] if `orders` is empty, has more than [`MAX_ORDERS`]
    /// entries, is not strictly decreasing, or has an order of 0 or more than [`MAX_ORDER`];
    /// if `hash_bits` is not in `4..=26`; if `verify_max` is less than the largest order; or if
    /// `max_len` is 0.
    pub fn new(
        orders: &[u8],
        hash_bits: u8,
        verify_max: u16,
        max_len: u32,
    ) -> Result<Self, MatchError> {
        let bad = |reason| Err(MatchError::InvalidParams { reason });
        if orders.is_empty() || orders.len() > MAX_ORDERS {
            return bad("the number of orders must be 1 to 8");
        }
        if orders.windows(2).any(|w| w[0] <= w[1]) {
            return bad("the orders must be strictly decreasing");
        }
        if orders.iter().any(|&o| o == 0 || o > MAX_ORDER) {
            return bad("each order must be 1 to 64");
        }
        if !(4..=26).contains(&hash_bits) {
            return bad("hash_bits must be 4 to 26");
        }
        if verify_max < u16::from(orders[0]) {
            return bad("verify_max must be at least the largest order");
        }
        if max_len == 0 {
            return bad("max_len must be at least 1");
        }
        let mut arr = [0u8; MAX_ORDERS];
        arr[..orders.len()].copy_from_slice(orders);
        // INVARIANT: orders.len() <= 8, so it fits in a u8.
        let count = u8::try_from(orders.len()).unwrap_or(u8::MAX);
        Ok(Self { orders: arr, count, hash_bits, verify_max, max_len })
    }

    /// The orders, longest first.
    #[must_use]
    pub fn orders(&self) -> &[u8] {
        &self.orders[..usize::from(self.count)]
    }

    /// The number of bits of a table index.
    #[must_use]
    pub const fn hash_bits(&self) -> u8 {
        self.hash_bits
    }

    /// The largest verified length of a look-up.
    #[must_use]
    pub const fn verify_max(&self) -> u16 {
        self.verify_max
    }

    /// The largest match length.
    #[must_use]
    pub const fn max_len(&self) -> u32 {
        self.max_len
    }
}

/// The match model of one block.
///
/// Memory: one `u32` for each table entry (`orders × 2^hash_bits`, 24 MiB for
/// [`MatchParams::V1`]) and one `u32` for each token of the block.
#[derive(Clone, Debug)]
pub struct MatchModel {
    params: MatchParams,
    /// `tables[o * size + slot]` is `position + 1`, or 0 for an empty entry.
    tables: Vec<u32>,
    history: Vec<u32>,
    ptr: usize,
    len: u32,
}

impl MatchModel {
    /// Makes an empty model. Cost: it allocates and zeroes the tables.
    #[must_use]
    pub fn new(params: MatchParams) -> Self {
        let size = 1usize << params.hash_bits;
        let tables = vec![0u32; size * usize::from(params.count)];
        Self { params, tables, history: Vec::new(), ptr: 0, len: 0 }
    }

    /// The parameters.
    #[must_use]
    pub const fn params(&self) -> &MatchParams {
        &self.params
    }

    /// Empties the model for a new block. After `reset`, the model gives the same predictions
    /// as a new model. Cost: one pass over the tables.
    pub fn reset(&mut self) {
        self.tables.fill(0);
        self.history.clear();
        self.ptr = 0;
        self.len = 0;
    }

    /// The prediction for the next token, if the model has a match.
    #[must_use]
    pub fn predict(&self) -> Option<Prediction> {
        let length = MatchLen::new(self.len)?;
        let token = *self.history.get(self.ptr)?;
        Some(Prediction { token: TokenId::new(token), length })
    }

    /// A draft of the next `k` tokens: the tokens that follow the current match, as if each
    /// drafted token is correct. It does not change the model (speculative decoding, ADR 0010
    /// section 8). The draft is empty with no match. The first token is the token of
    /// [`MatchModel::predict`]. A later prediction can differ from the draft, because a correct
    /// token can make a longer match replace the current match (rule 3.3).
    ///
    /// A match can overlap the position (a repeat with a short period); then the draft reads
    /// its own earlier tokens. Cost: `k` reads.
    #[must_use]
    pub fn draft(&self, k: usize) -> Vec<TokenId> {
        let mut out: Vec<u32> = Vec::with_capacity(k);
        if self.len == 0 {
            return Vec::new();
        }
        let n = self.history.len();
        for j in 0..k {
            let src = self.ptr + j;
            let tok = if src < n { self.history.get(src) } else { out.get(src - n) };
            match tok {
                Some(&t) => out.push(t),
                None => break,
            }
        }
        out.into_iter().map(TokenId::new).collect()
    }

    /// The predictions of the next `k` positions if each prediction is correct: item 0 is
    /// [`MatchModel::predict`], and item `j` is the prediction that `predict` gives after
    /// [`MatchModel::observe`] of the tokens of items `0..j`. It stops at the first position
    /// without a prediction. It does not change the model (speculative decoding, ADR 0017).
    ///
    /// The tokens are the draft of the decoder, and the predictions are the exact inputs of the
    /// mixer at the draft positions: a correct token can make a longer match replace the
    /// current match (rule 3.3), so the tokens can differ from [`MatchModel::draft`].
    ///
    /// Cost: the cost of `k - 1` calls of `observe`, with the table writes in a small list
    /// (one look-up of the list for each table read, at most `orders * k` entries).
    #[must_use]
    pub fn speculate(&self, k: usize) -> Vec<Prediction> {
        let mut out = Vec::with_capacity(k);
        let mut store = Overlay {
            history: &self.history,
            tables: &self.tables,
            ext: Vec::new(),
            writes: Vec::new(),
        };
        let (mut ptr, mut len) = (self.ptr, self.len);
        while out.len() < k {
            let Some(length) = MatchLen::new(len) else { break };
            let Some(token) = store.hist(ptr) else { break };
            out.push(Prediction { token: TokenId::new(token), length });
            if out.len() < k {
                step(&self.params, &mut store, &mut ptr, &mut len, token);
            }
        }
        out
    }

    /// Adds the token that the coder coded (rule 3 of the module documentation).
    ///
    /// Cost: one hash pass over the last `orders[0]` tokens, and, while the match is shorter
    /// than the largest order, at most one verification of `verify_max` tokens for each order.
    pub fn observe(&mut self, token: TokenId) {
        let mut store = Owned { history: &mut self.history, tables: &mut self.tables };
        step(&self.params, &mut store, &mut self.ptr, &mut self.len, token.get());
    }
}

/// The history and the tables that rule 3 reads and writes.
trait Store {
    /// The number of tokens in the history.
    fn len(&self) -> usize;
    /// The token at position `i`, or `None` if `i >= len()`.
    fn hist(&self, i: usize) -> Option<u32>;
    /// Appends a token to the history.
    fn push(&mut self, t: u32);
    /// The table entry `idx` (0 for an empty entry).
    fn entry(&self, idx: usize) -> u32;
    /// Writes the table entry `idx`.
    fn set_entry(&mut self, idx: usize, v: u32);
}

/// The store of the model itself ([`MatchModel::observe`]).
struct Owned<'a> {
    history: &'a mut Vec<u32>,
    tables: &'a mut Vec<u32>,
}

impl Store for Owned<'_> {
    fn len(&self) -> usize {
        self.history.len()
    }
    fn hist(&self, i: usize) -> Option<u32> {
        self.history.get(i).copied()
    }
    fn push(&mut self, t: u32) {
        self.history.push(t);
    }
    fn entry(&self, idx: usize) -> u32 {
        self.tables.get(idx).copied().unwrap_or(0)
    }
    fn set_entry(&mut self, idx: usize, v: u32) {
        if let Some(e) = self.tables.get_mut(idx) {
            *e = v;
        }
    }
}

/// A store that reads the model and keeps its own writes ([`MatchModel::speculate`]).
struct Overlay<'a> {
    history: &'a [u32],
    tables: &'a [u32],
    /// The tokens after `history`.
    ext: Vec<u32>,
    /// The table writes in order; the last write of an entry is its value.
    writes: Vec<(usize, u32)>,
}

impl Store for Overlay<'_> {
    fn len(&self) -> usize {
        self.history.len() + self.ext.len()
    }
    fn hist(&self, i: usize) -> Option<u32> {
        match i.checked_sub(self.history.len()) {
            None => self.history.get(i).copied(),
            Some(j) => self.ext.get(j).copied(),
        }
    }
    fn push(&mut self, t: u32) {
        self.ext.push(t);
    }
    fn entry(&self, idx: usize) -> u32 {
        match self.writes.iter().rev().find(|w| w.0 == idx) {
            Some(&(_, v)) => v,
            None => self.tables.get(idx).copied().unwrap_or(0),
        }
    }
    fn set_entry(&mut self, idx: usize, v: u32) {
        self.writes.push((idx, v));
    }
}

/// Rule 3 for the token `t`: the match, the history and the tables of `s`.
fn step<S: Store>(params: &MatchParams, s: &mut S, ptr: &mut usize, len: &mut u32, t: u32) {
    if *len > 0 {
        if s.hist(*ptr) == Some(t) {
            *ptr += 1;
            *len = (*len + 1).min(params.max_len);
        } else {
            *len = 0;
        }
    }
    // NOTE: a position must fit in a table entry (`position + 1` in a u32). After 2^32 - 2
    // tokens (more than a block can hold, ADR 0006) the model stops: no new entries.
    if s.len() >= usize::try_from(u32::MAX - 1).unwrap_or(usize::MAX) {
        *len = 0;
        return;
    }
    s.push(t);
    let i = s.len();
    let slots = slots(params, s, i);
    let top = u32::from(params.orders[0]);
    if *len < top {
        look_up(params, s, i, &slots, ptr, len);
    }
    let size = 1usize << params.hash_bits;
    // INVARIANT: i < u32::MAX - 1 (checked above), so i + 1 fits in a u32.
    let entry = u32::try_from(i + 1).unwrap_or(0);
    for (o, slot) in slots.iter().enumerate() {
        if let Some(slot) = *slot {
            s.set_entry(o * size + slot, entry);
        }
    }
}

/// The slot of each order at position `i`, or `None` if `i` is less than the order.
fn slots<S: Store>(params: &MatchParams, s: &S, i: usize) -> [Option<usize>; MAX_ORDERS] {
    let mut out = [None; MAX_ORDERS];
    let orders = params.orders();
    let mut acc: u64 = 0;
    let mut bk: u64 = 1;
    let mut next = orders.len();
    // Orders are decreasing, so walk them from the shortest: index next - 1.
    for k in 1..=usize::from(orders[0]) {
        if k > i {
            break;
        }
        let tok = u64::from(s.hist(i - k).unwrap_or(0));
        acc = acc.wrapping_add((tok + 1).wrapping_mul(bk));
        bk = bk.wrapping_mul(HASH_B);
        while next > 0 && usize::from(orders[next - 1]) == k {
            let x = (acc ^ (acc >> 32)).wrapping_mul(HASH_MIX);
            let slot = x >> (64 - u32::from(params.hash_bits));
            // INVARIANT: slot < 2^hash_bits <= 2^26, so it fits in a usize.
            out[next - 1] = usize::try_from(slot).ok();
            next -= 1;
        }
    }
    out
}

/// Rule 3.3: the longest verified match at position `i`.
fn look_up<S: Store>(
    params: &MatchParams,
    s: &S,
    i: usize,
    slots: &[Option<usize>; MAX_ORDERS],
    ptr: &mut usize,
    len: &mut u32,
) {
    let size = 1usize << params.hash_bits;
    for (o, &order) in params.orders().iter().enumerate() {
        let order32 = u32::from(order);
        if order32 <= *len {
            break;
        }
        let Some(slot) = slots[o] else { continue };
        let entry = s.entry(o * size + slot);
        if entry == 0 {
            continue;
        }
        let c = usize::try_from(entry - 1).unwrap_or(usize::MAX);
        let v = common_suffix(params, s, c, i);
        if v >= order32 {
            *ptr = c;
            *len = v;
            return;
        }
    }
}

/// The count of `k` in `1..=verify_max` with `h[c-k] = h[i-k]`, stopping at the first
/// difference or at the block start. Requires `c < i <= h.len()`.
fn common_suffix<S: Store>(params: &MatchParams, s: &S, c: usize, i: usize) -> u32 {
    let limit = usize::from(params.verify_max).min(c);
    let same = (1..=limit).take_while(|&k| s.hist(c - k) == s.hist(i - k)).count();
    // INVARIANT: same <= verify_max <= u16::MAX.
    u32::try_from(same).unwrap_or(0)
}
