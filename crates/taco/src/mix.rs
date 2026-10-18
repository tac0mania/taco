//! The coding part of the `logs` profile: the match model and the mixer of one block
//! (ADR 0010).
//!
//! PLAN(§4): the part sits between HardShell and `FreqTable::from_weights`. It changes the
//! u64 weights of HardShell only. For each token `t[i]` of a block, the encoder and the decoder
//! make the same calls in the same order (ADR 0010 section 4):
//!
//! 1. At the block start: a new [`MixState`] (the match model and the mixer start empty; this
//!    is the `reset` of ADR 0010 section 4).
//! 2. [`MixState::mix`] with the HardShell weights for `t[i]` gives the final weights. The
//!    coder makes its table from them.
//! 3. After `t[i]` is coded or decoded: [`MixState::observe`] (`Mixer::update`, then
//!    `MatchModel::observe`).
//!
//! The rule version of the container fixes the constants. This module knows only mixer rule
//! version 1 ([`MixState::new`]).
//!
//! DETERMINISM: `taco-match` is a kernel crate (integers only). The state depends only on the
//! tokens of the block, so a lane of a `Batch` and a single session give the same bits.

use hardshell::TokenId;
use taco_format::MixerRule;
use taco_match::{MatchModel, MatchParams, Mixer, MixerParams};

use crate::error::Result;

/// The match model, the mixer and a work buffer for the weights of one block.
///
/// Memory: 24 MiB of match tables (mixer rule version 1), 144 KiB of logistic tables, one
/// `u32` for each token of the block, and one `u64` for each token of the vocabulary.
#[derive(Clone, Debug)]
pub(crate) struct MixState {
    model: MatchModel,
    mixer: Mixer,
    weights: Vec<u64>,
}

impl MixState {
    /// The empty state of mixer rule `rule`, for the start of a block. A new state is the
    /// state after `reset` (a `taco-match` test proves it).
    ///
    /// Cost: one zeroed allocation of the match tables (the system gives zero pages, so only
    /// the entries that the block uses cost memory), and the logistic tables (about 4,096
    /// `u128` divisions).
    pub(crate) fn new(rule: MixerRule) -> Self {
        let (match_params, mixer_params) = rule_constants(rule);
        Self {
            model: MatchModel::new(match_params),
            mixer: Mixer::new(mixer_params),
            weights: Vec::new(),
        }
    }

    /// The final weights for the next token: the HardShell weights `hardshell`, changed by the
    /// prediction of the match model (ADR 0010 section 3). Every weight is at least 1.
    ///
    /// Cost: one copy of the weights, and one or two passes over them.
    ///
    /// # Errors
    ///
    /// [`crate::TacoError::Match`] if the weights are empty or the predicted token is not in
    /// them (an internal fault: the match model predicts only tokens that it observed).
    pub(crate) fn mix(&mut self, hardshell: &[u64]) -> Result<&[u64]> {
        self.weights.clear();
        self.weights.extend_from_slice(hardshell);
        self.mixer.mix(&mut self.weights, self.model.predict())?;
        Ok(&self.weights)
    }

    /// Adds the token that the coder coded or decoded (ADR 0010 section 4, step 2).
    pub(crate) fn observe(&mut self, token: TokenId) {
        let t = taco_match::TokenId::new(token.get());
        self.mixer.update(t);
        self.model.observe(t);
    }
}

/// The constants of a mixer rule version (ADR 0010 sections 2 and 3).
fn rule_constants(_rule: MixerRule) -> (MatchParams, MixerParams) {
    // NOTE: `MixerRule::from_u8` makes version 1 only, so the rule is version 1. A new rule
    // version adds its constants here and a value in `taco-format`.
    (MatchParams::V1, MixerParams::V1)
}
