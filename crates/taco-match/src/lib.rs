//! Token match model and integer mixer for Taco: long repeats in logs (ADR 0010).
//!
//! Logs repeat templates, ids and timestamps over long distances: further than the 1,024
//! tokens of the GPT-2 context, and across its context restarts (plan §7.7). The
//! [`MatchModel`] finds the longest earlier occurrence of the current context in the block and
//! predicts the token that followed it. The [`Mixer`] gives that token a probability from the
//! GPT-2 probability, the match length and what it learned in the block, and changes the
//! HardShell weights into the final weights for `taco_coder::FreqTable::from_weights`.
//!
//! The crate is a kernel crate: no floats, no I/O, no clock, no randomness, no global state.
//! Each block starts with [`MatchModel::reset`] and [`Mixer::reset`], so blocks stay
//! independent (plan §9).
//!
//! # The order of the calls for one block (ADR 0010 section 4)
//!
//! For the tokens `t[0..n]` of a block, with the GPT-2 weights `w[i]` for token `i`:
//!
//! 1. `model.reset()` and `mixer.reset()`.
//! 2. For each `i`: `mixer.mix(&mut w[i], model.predict())`, make the table from `w[i]`, code
//!    `t[i]`, then `mixer.update(t[i])` and `model.observe(t[i])`.
//!
//! The decoder makes the same calls in the same order with the decoded tokens.
//!
//! PLAN(§3): rules 1, 2 and 5. Every output weight is at least 1.

#![deny(clippy::float_arithmetic)]

mod error;
mod logistic;
mod matcher;
mod mixer;
mod skip;
mod types;

pub use error::MatchError;
pub use matcher::{MAX_ORDER, MAX_ORDERS, MatchModel, MatchParams};
pub use mixer::{BUCKETS, Checkpoint, Mixer, MixerParams, bucket, rule_constants};
pub use skip::{SkipCode, SkipParams, skip_miss_weights};
pub use types::{MatchLen, Prediction, TokenId};
