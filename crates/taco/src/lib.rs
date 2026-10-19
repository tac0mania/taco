//! Taco: the library. It connects the tokenizer, HardShell, the range coder and the container.
//!
//! PLAN(§4): the data flow.
//!
//! ```text
//! fold:   input -> blocks -> tokenizer -> token ids
//!         for each token: HardShell -> weights -> frequency table -> range coder
//!         -> container (header, model hash, checksum, blocks)
//! unfold: container -> check the model hash -> for each block:
//!         HardShell -> frequency table -> range decoder -> token id -> bytes
//!         -> verify the checksum -> output
//! ```
//!
//! # Use
//!
//! ```no_run
//! # fn main() -> Result<(), taco::TacoError> {
//! # let model_file: Vec<u8> = Vec::new();
//! # let merges: Vec<(taco::TokenId, taco::TokenId)> = Vec::new();
//! let model = taco::Model::load(&model_file)?;
//! let tokenizer = taco::Tokenizer::for_vocabulary(model.vocabulary(), model.eot_token(), &merges)?;
//! let folded = taco::fold(&model, &tokenizer, b"some text", &taco::FoldOptions::new())?;
//! let restored = taco::unfold(Some(&model), &folded.container, &taco::UnfoldOptions::new())?;
//! assert_eq!(restored, b"some text");
//! # Ok(())
//! # }
//! ```
//!
//! # The parts
//!
//! - [`Tokenizer`]: GPT-2 byte-level BPE (plan §10), a port of `tools/baseline/bpe.py`. Only
//!   the encoder uses it.
//! - [`fold`] and [`unfold`]: the pipeline. Each block has its own HardShell session, so blocks
//!   can run on several threads ([`Threads`]). The output does not depend on the number of
//!   threads.
//! - The stored bytes of a coded block are the token count (a LEB128 varint), then the range
//!   coder bytes (ADR 0007).
//! - Profiles ([`FoldProfile`], ADR 0012): `text` codes with HardShell alone. `logs` also codes
//!   each block with the match model and the mixer of `taco-match` between HardShell and the
//!   frequency table (ADR 0010), and stores that form (block kind 3, "mixed") when it is
//!   smaller. `auto` selects for the encoder. [`unfold`] reads the profile and the coding parts
//!   from the header, so the caller never names a profile to decode.
//!
//! # Block size
//!
//! The default block size of the profile `text` is [`DEFAULT_BLOCK_SIZE`], 16 KiB. The
//! profiles `logs` and `auto` use [`LOGS_BLOCK_SIZE`], 64 KiB (measured, see its note). The
//! reasons for 16 KiB (est.; plan §13 question 3 asks for the measurement):
//!
//! - The context of GPT-2 starts again after 1,024 tokens in any case (§7.7). A block of
//!   16 KiB of English has about 5,000 tokens (phase 0: 3.1 bytes for each token on English,
//!   measured). Thus a block boundary adds about one context start for each five.
//! - The fixed cost of a block is about 8 bytes: 5 bytes in the block table, 2 bytes of token
//!   count, and the coder flush (est.). At 2 bits per byte, a block of 16 KiB codes to about
//!   4 KiB, so the cost is about 0.2 % (est.).
//! - A file of 1 MiB has 64 blocks: enough work for the threads of a CPU, and a binary region
//!   costs at most 16 KiB of raw blocks around it.
//!
//! UNVERIFIED: the ratio and the speed at other block sizes are not measured. Phase 2 measures
//! them with GPT-2 (plan §13 question 3).

// NOTE: the prediction path (HardShell, the frequency tables, the coder) is in the kernel
// crates. This crate only calls it, and it has no floats either.
#![deny(clippy::float_arithmetic)]

mod backend;
mod codec;
mod error;
mod fold;
mod gpu_fold;
mod gpu_unfold;
mod lanes;
mod mix;
mod parallel;
mod payload;
pub mod stream;
pub mod tokenizer;

/// The container crate (`taco-format`), for callers that read or check `.taco` files
/// directly (for example the conformance tests of `taco-cli`).
pub use taco_format as format;
mod unfold;

pub use backend::{Backend, BackendChoice, BackendNote, Engine, with_engine};
pub use codec::FREQ_BITS;
pub use error::{Result, TacoError};
pub use fold::{FoldOptions, FoldProfile, FoldStats, Folded, fold, fold_with};
pub use hardshell::TokenId;
pub use hardshell::gpt2::{Model, Overlap};
pub use parallel::Threads;
pub use taco_format::{BlockSize, Checksum, FormatError, ModelHash, Profile};
pub use tokenizer::{MergeFault, Tokenizer};
pub use unfold::{UnfoldOptions, container_overlap, unfold, unfold_with};

/// The default number of input bytes in each block of the profile `text`: 16 KiB. See the
/// crate documentation.
// NOTE: the compiler evaluates this constant. A panic here is a build error, not a run-time
// panic, and 16 KiB is not 0.
#[allow(clippy::panic)]
pub const DEFAULT_BLOCK_SIZE: BlockSize = {
    let Ok(b) = BlockSize::new(16 * 1024) else { panic!("the block size is not 0") };
    b
};

/// The default number of input bytes in each block of the profiles `logs` and `auto`: 64 KiB.
///
/// NOTE: with the overlap `K = 512` on the 18 logs of `tools/data/logs`, blocks of 64 KiB gave
/// 363,234 bytes against 444,436 bytes with blocks of 16 KiB (-18.3 %, measured,
/// `docs/research/2026-10-19-overlap-cli.md`): the match model sees more of each log. The cost:
/// 23 % more CPU time for each byte in fold, and fewer blocks for the threads and the lanes of a
/// small file (ADR 0012, addendum).
// NOTE: the compiler evaluates this constant. A panic here is a build error, not a run-time
// panic, and 64 KiB is not 0.
#[allow(clippy::panic)]
pub const LOGS_BLOCK_SIZE: BlockSize = {
    let Ok(b) = BlockSize::new(64 * 1024) else { panic!("the block size is not 0") };
    b
};

/// The `.taco` container versions that [`unfold`] and [`stream`] read (ADR 0006, ADR 0011).
pub const CONTAINER_READS: [u16; 2] = [taco_format::VERSION_1, taco_format::VERSION];

/// The `.taco` container version that [`fold`] and [`stream`] write (ADR 0011).
pub const CONTAINER_WRITES: u16 = taco_format::VERSION;

/// The `.hsm` model-file versions that [`Model::load`] reads.
pub const MODEL_FILE_READS: [u32; 2] = [hardshell::hsm::VERSION_1, hardshell::hsm::VERSION_2];

/// The model hash of `model` as the container stores it.
///
/// `hardshell` and `taco-format` each have a `ModelHash` with the same 32 bytes; neither
/// crate depends on the other.
#[must_use]
pub fn container_hash(model: &Model) -> ModelHash {
    ModelHash::from_bytes(*model.hash().as_bytes())
}
