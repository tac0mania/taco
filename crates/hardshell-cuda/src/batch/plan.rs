//! The plan of a call: the inputs become rows, and the rows become passes of at most
//! `max_rows` rows. A slot whose context is full starts again with the context rule of
//! `hardshell::gpt2` (plan §7.7, ADR 0015): `<|endoftext|>` at position 0, the last `K` tokens
//! of the slot at the positions `1..=K`, then the token at position `K + 1`.

use std::collections::HashMap;

use hardshell::TokenId;
use hardshell::gpt2::StreamState;

use super::{CudaBatch, Row};

/// The passes of a call, and the positions and the histories after it. The batch commits them
/// only after all passes ran.
pub(super) struct Plan {
    pub(super) passes: Vec<Vec<Row>>,
    pos: Vec<usize>,
    /// The new history of each slot that the call changed.
    hist: HashMap<usize, Vec<TokenId>>,
    /// The new inputs of each slot that the call changed (ADR 0017).
    streams: HashMap<usize, StreamState>,
    /// The history before the last restart of each slot that restarted in the call.
    prev: HashMap<usize, Vec<TokenId>>,
}

/// The rows of the pass that the planner fills, and the passes before it.
struct Passes {
    max_rows: usize,
    done: Vec<Vec<Row>>,
    cur: Vec<Row>,
    in_pass: Vec<bool>,
}

impl Passes {
    /// Ends the current pass.
    fn flush(&mut self) {
        if !self.cur.is_empty() {
            self.done.push(std::mem::take(&mut self.cur));
        }
        self.in_pass.iter_mut().for_each(|x| *x = false);
    }

    /// Adds a row; a full pass ends first. The rows of one slot at consecutive positions can
    /// fall in two passes: the earlier pass writes the cache positions that the later reads.
    fn push(&mut self, row: Row) {
        if self.cur.len() >= self.max_rows {
            self.flush();
        }
        if let Some(x) = self.in_pass.get_mut(row.slot) {
            *x = true;
        }
        self.cur.push(row);
    }
}

impl CudaBatch<'_> {
    /// Splits the inputs into passes of at most `max_rows` rows. A restart of a slot starts a
    /// new pass if the slot already has rows in the current pass, because the restart writes
    /// cache positions that those rows read. The caller checked the slots and the tokens.
    pub(super) fn plan(&self, inputs: &[(usize, TokenId)]) -> Plan {
        let n_ctx = self.model.n_ctx();
        // INVARIANT: `set_overlap` checked `k <= n_ctx - 2`; the `min` only keeps the ranges
        // below in bounds.
        let k = self.overlap.tokens().min(n_ctx.saturating_sub(2));
        let eot = self.model.eot_token();
        let mut pos = self.pos.clone();
        let mut hist: HashMap<usize, Vec<TokenId>> = HashMap::new();
        let mut streams: HashMap<usize, StreamState> = HashMap::new();
        let mut prev: HashMap<usize, Vec<TokenId>> = HashMap::new();
        let mut p = Passes {
            max_rows: self.max_rows,
            done: Vec::new(),
            cur: Vec::new(),
            in_pass: vec![false; self.slots],
        };
        for (i, &(slot, token)) in inputs.iter().enumerate() {
            let h = hist.entry(slot).or_insert_with(|| {
                self.hist
                    .get(slot * n_ctx..(slot + 1) * n_ctx)
                    .map_or_else(|| vec![TokenId::new(0); n_ctx], <[TokenId]>::to_vec)
            });
            let st = streams
                .entry(slot)
                .or_insert_with(|| self.streams.get(slot).copied().unwrap_or_default());
            let mut at = pos.get(slot).copied().unwrap_or(0);
            if at >= n_ctx {
                if p.in_pass.get(slot).copied().unwrap_or(false) {
                    p.flush();
                }
                // The full context before the restart, for a truncation (ADR 0017).
                prev.insert(slot, h.clone());
                st.restart(k);
                // PLAN(§7.7): the context starts again with <|endoftext|> at position 0 and
                // the last `k` tokens at 1..=k (ADR 0015). These rows answer no input.
                // NOTE: the ranges overlap when 2k >= n_ctx; `copy_within` is a memmove.
                h.copy_within(n_ctx - k..n_ctx, 1);
                if let Some(t) = h.first_mut() {
                    *t = eot;
                }
                for q in 0..=k {
                    let t = h.get(q).copied().unwrap_or(eot);
                    p.push(Row { slot, token: t, pos: q, input: None });
                }
                at = k + 1;
            }
            if let Some(t) = h.get_mut(at) {
                *t = token;
            }
            st.input(at);
            p.push(Row { slot, token, pos: at, input: Some(i) });
            if let Some(x) = pos.get_mut(slot) {
                *x = at + 1;
            }
        }
        p.flush();
        Plan { passes: p.done, pos, hist, streams, prev }
    }

    /// Keeps the positions and the histories of a plan whose passes ran.
    pub(super) fn commit(&mut self, plan: Plan) {
        let n_ctx = self.model.n_ctx();
        self.pos = plan.pos;
        for (slot, h) in plan.hist {
            if let Some(dst) = self.hist.get_mut(slot * n_ctx..(slot + 1) * n_ctx) {
                dst.copy_from_slice(&h);
            }
        }
        for (slot, s) in plan.streams {
            if let Some(dst) = self.streams.get_mut(slot) {
                *dst = s;
            }
        }
        for (slot, h) in plan.prev {
            if let Some(dst) = self.prev.get_mut(slot) {
                *dst = h;
            }
        }
    }
}
