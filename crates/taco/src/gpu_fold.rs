//! `fold` on a GPU backend: the HardShell steps of many blocks run together on the GPU, and the
//! range encoders run on the CPU (ADR 0016).
//!
//! PLAN(§4): the calls are the calls of ADR 0007 section 3 (and ADR 0010 section 4 for the
//! profile `logs`); only the place where HardShell runs changes.
//!
//! The GPU coder op gives the interval of each token in the table of
//! `FreqTable::from_weights(weights, 32)`, so only 24 bytes for each token come back. The
//! encoder codes the interval ([`BlockEncoder::push_interval`]). A block is divided into its
//! context windows (ADR 0015 section 2 with `K = 0`): window 0 holds the first `n_ctx` tokens;
//! each later window holds `n_ctx - 1` tokens and starts with a new context that holds
//! `<|endoftext|>` only. The windows are independent, so they run in parallel slots; the
//! encoder of a block takes their intervals in token order.
//!
//! This path is for the profile `text` only. The mixer of the profile `logs` changes the
//! weights before the table, so it needs all weights of each token. On devbox the read back of
//! the weights made the CUDA fold slower than the CPU fold (ADR 0016 section 5), so
//! [`crate::fold_with`] codes the profiles `logs` and `auto` on the CPU.
//!
//! DETERMINISM: the weights and the intervals are the CPU values, bit for bit (ADR 0008), and
//! [`crate::codec`] codes them in the order of the CPU path. Thus the payloads are the CPU
//! payloads, byte for byte (tests in `tests/backends.rs`).

use std::collections::{BTreeMap, VecDeque};

use hardshell::TokenId;
use hardshell::gpt2::Model;

use crate::backend::{Backend, Engine};
use crate::codec::{BlockEncoder, CodedBlock};
use crate::error::{Result, TacoError};
use crate::lanes::{Answer, Lanes, Query};

/// One block before the model: its tokens and its encoder (GPT-2 alone, with the stop length
/// of ADR 0007 §5.2).
pub(crate) struct BlockWork {
    tokens: Vec<TokenId>,
    enc: BlockEncoder,
}

impl BlockWork {
    /// A block of `tokens` with the stop length `stop`.
    pub(crate) fn new(tokens: Vec<TokenId>, stop: usize) -> Result<Self> {
        let count = u64::try_from(tokens.len())
            .map_err(|_| TacoError::InputTooLarge { reason: "token count" })?;
        Ok(Self { tokens, enc: BlockEncoder::new(count, stop, None) })
    }

    /// `true` if the model has work for the block: it has tokens, and its encoder did not stop.
    fn needs_model(&self) -> bool {
        !self.tokens.is_empty() && !self.enc.stopped()
    }
}

/// The rows of one pass of the CUDA backend in the profile `text`.
// NOTE: prefill 8 slots x 512 tokens gave 27,781 to 30,469 tokens/s, and 8 x 1,024 gave
// 25,819 to 26,191 (measured, RTX 3060, research note 2026-10-18 section 7.4).
const TEXT_ROWS: usize = 512;

/// The most windows that run at the same time in the profile `text`. Each slot holds one KV
/// cache (18.9 MB for GPT-2 small).
const TEXT_SLOTS: usize = 64;

/// Codes `blocks` (profile `text`) on the GPU of `engine`. Each result is the result of
/// [`crate::codec::encode_block`] for the same tokens and stop length, with no mixer.
///
/// # Errors
/// [`TacoError::Backend`] on a GPU fault; the errors of the encoders.
pub(crate) fn code_blocks(
    engine: &Engine<'_>,
    model: &Model,
    mut blocks: Vec<BlockWork>,
) -> Result<Vec<CodedBlock>> {
    let windows = windows_of(&blocks, model.n_ctx());
    if let Some(mut lanes) = engine.lanes(windows.len().min(TEXT_SLOTS), TEXT_ROWS)? {
        code_with_intervals(lanes.as_mut(), model, &mut blocks, windows)?;
    }
    Ok(blocks.into_iter().map(|b| b.enc.finish()).collect())
}

/// A context window of a block: the tokens `first..first + len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Window {
    block: usize,
    first: usize,
    len: usize,
}

impl Window {
    /// The rows of the window: the input token of each row, and the token that its weights
    /// code. Window 0 starts with the block start (`<|endoftext|>` codes `t[0]`). A later
    /// window starts with a new context: `<|endoftext|>` (its weights are not used), then
    /// `t[first - 1]`, whose weights code `t[first]`, as the restart rule of ADR 0004 section 9.
    fn rows(self, tokens: &[TokenId], eot: TokenId) -> Vec<(TokenId, Option<TokenId>)> {
        let t = |i: usize| tokens.get(i).copied().unwrap_or(eot);
        let mut rows = Vec::with_capacity(self.len + 1);
        if self.first == 0 {
            rows.push((eot, Some(t(0))));
        } else {
            rows.push((eot, None));
            rows.push((t(self.first - 1), Some(t(self.first))));
        }
        for i in self.first + 1..self.first + self.len {
            rows.push((t(i - 1), Some(t(i))));
        }
        rows
    }
}

/// The windows of every block that the encoder does not skip, in block order. Window 0 has
/// `n_ctx` tokens (positions 0 to `n_ctx - 1`); each later window has `n_ctx - 1` tokens
/// (positions 1 to `n_ctx - 1`, after `<|endoftext|>`).
fn windows_of(blocks: &[BlockWork], n_ctx: usize) -> Vec<Window> {
    let mut out = Vec::new();
    for (block, b) in blocks.iter().enumerate() {
        if !b.needs_model() {
            continue;
        }
        let n = b.tokens.len();
        let mut first = 0;
        // INVARIANT: a model has n_ctx >= 2 (a restart needs two positions), so each window
        // has at least one token and the loop ends.
        let mut size = n_ctx.max(2);
        while first < n {
            let len = size.min(n - first);
            out.push(Window { block, first, len });
            first += len;
            size = n_ctx.max(2) - 1;
        }
    }
    out
}

/// A window in a slot: its rows, the rows that ran, the answers so far, and the answers that
/// the encoder of its block already took.
struct Running {
    window: Window,
    rows: Vec<(TokenId, Option<TokenId>)>,
    done: usize,
    answers: Vec<Answer>,
    coded: usize,
}

/// The profile `text`: windows in slots, intervals from the coder op.
///
/// The encoder of a block takes the answers of its first unfinished window as they arrive, and
/// the answers of a later window when the windows before it are complete. When the encoders of
/// a block stop (ADR 0007 §5.2), its windows leave their slots and its other windows do not
/// start.
fn code_with_intervals(
    lanes: &mut dyn Lanes,
    model: &Model,
    blocks: &mut [BlockWork],
    windows: Vec<Window>,
) -> Result<()> {
    let eot = model.eot_token();
    let backend = lanes.backend();
    let mut queue: VecDeque<Window> = windows.into();
    // The answers of complete windows whose earlier windows are not complete, by first token.
    let mut early: Vec<BTreeMap<usize, Vec<Answer>>> = vec![BTreeMap::new(); blocks.len()];
    // The first token of each block that the encoder did not take.
    let mut next = vec![0usize; blocks.len()];
    let mut slots: Vec<Option<Running>> = (0..lanes.slots()).map(|_| None).collect();
    let per_slot = lanes.per_slot();
    loop {
        for (slot, s) in slots.iter_mut().enumerate() {
            if s.is_some() {
                continue;
            }
            while let Some(w) = queue.pop_front() {
                if let Some(b) = blocks.get(w.block)
                    && !b.enc.stopped()
                {
                    lanes.reset(slot);
                    let rows = w.rows(&b.tokens, eot);
                    *s = Some(Running { window: w, rows, done: 0, answers: Vec::new(), coded: 0 });
                    break;
                }
            }
        }
        if slots.iter().all(Option::is_none) {
            return Ok(());
        }
        let mut inputs = Vec::new();
        let mut queries = Vec::new();
        let mut owners = Vec::new();
        for (slot, s) in slots.iter().enumerate() {
            let Some(r) = s else { continue };
            for &(input, query) in r.rows.iter().skip(r.done).take(per_slot) {
                inputs.push((slot, input));
                // NOTE: a restart row codes nothing; it asks for the interval of
                // `<|endoftext|>` and the answer is dropped.
                queries.push(Query::Encode(query.unwrap_or(eot)));
                owners.push((slot, query.is_some()));
            }
        }
        let answers = lanes.coded(&inputs, &queries)?;
        if answers.len() != inputs.len() {
            return Err(backend.fault("the coder op gave too few answers"));
        }
        for (&(slot, keep), a) in owners.iter().zip(answers) {
            if let Some(Some(r)) = slots.get_mut(slot) {
                r.done += 1;
                if keep {
                    r.answers.push(a);
                }
            }
        }
        // The first unfinished window of each block: its answers go to the encoder now. A
        // complete window then lets the windows after it go too. Two rounds are sufficient:
        // a window that becomes first in round 1 is coded in round 2.
        for _ in 0..2 {
            for s in &mut slots {
                let Some(r) = s.as_mut() else { continue };
                let w = r.window;
                let (Some(b), Some(n)) = (blocks.get_mut(w.block), next.get_mut(w.block)) else {
                    continue;
                };
                if *n == w.first + r.coded {
                    let fresh = r.answers.get(r.coded..).unwrap_or(&[]);
                    code_answers(b, fresh, *n, backend)?;
                    r.coded = r.answers.len();
                    *n = w.first + r.coded;
                }
                let complete = r.done >= r.rows.len();
                if complete && *n == w.first + r.coded && r.coded == w.len {
                    *s = None;
                    let map = early.get_mut(w.block).ok_or(TacoError::WorkerFailed)?;
                    while let Some(later) = map.remove(n) {
                        code_answers(b, &later, *n, backend)?;
                        *n += later.len();
                    }
                } else if complete {
                    // NOTE: a complete window that is not first waits with its answers.
                    if let Some(r) = s.take() {
                        let map = early.get_mut(w.block).ok_or(TacoError::WorkerFailed)?;
                        map.insert(w.first, r.answers);
                    }
                }
            }
        }
        // The windows of a block whose encoders stopped leave their slots.
        for s in &mut slots {
            if s.as_ref()
                .is_some_and(|r| blocks.get(r.window.block).is_some_and(|b| b.enc.stopped()))
            {
                *s = None;
            }
        }
    }
}

/// Codes `answers`, the intervals of the tokens `first..`, in the encoder of `block`, until
/// the encoder stops.
fn code_answers(
    block: &mut BlockWork,
    answers: &[Answer],
    first: usize,
    backend: Backend,
) -> Result<()> {
    for (i, a) in answers.iter().enumerate() {
        if block.enc.stopped() {
            return Ok(());
        }
        if block.tokens.get(first + i).copied() != Some(a.token) {
            return Err(backend.fault("the coder op answered for another token"));
        }
        block.enc.push_interval(a.interval);
    }
    Ok(())
}
