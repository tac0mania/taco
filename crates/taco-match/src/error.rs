//! The error type of the crate: one variant for each fault that a caller can act on.

/// A fault in the parameters or in the input that a caller gave to the match model or the mixer.
///
/// Errors carry ids, counts and lengths. They never carry the data that the caller codes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum MatchError {
    /// The parameters break a rule of [`MatchParams`](crate::MatchParams) or
    /// [`MixerParams`](crate::MixerParams).
    #[error("invalid parameters: {reason}")]
    InvalidParams {
        /// The rule that the parameters break.
        reason: &'static str,
    },
    /// The predicted token is not inside the weights that the caller gave.
    #[error("the predicted token {token} is not in a vocabulary of {vocab} tokens")]
    TokenOutOfRange {
        /// The predicted token id.
        token: u32,
        /// The number of weights.
        vocab: usize,
    },
    /// The caller gave no weights.
    #[error("the weights are empty")]
    EmptyWeights,
    /// A rollback to a checkpoint of a speculative run that ended (or of another run).
    #[error("the checkpoint is not a point of the open speculative run")]
    StaleCheckpoint,
}
