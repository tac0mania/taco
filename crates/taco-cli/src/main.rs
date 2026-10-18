//! The `taco` command: `taco fold` and `taco unfold`.
//!
//! The command has no compression logic. It reads the files, calls `taco::fold` or
//! `taco::unfold`, writes the output, and prints a one-line summary to stderr.

// NOTE: a command prints its summary and its errors to stderr; the library never prints.
#![allow(clippy::print_stderr)]

mod args;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use args::{Command, Job, Mode, USAGE};
use taco::{BlockSize, FoldOptions, Model, Threads, TokenId, Tokenizer, UnfoldOptions};

type CliResult<T> = Result<T, Box<dyn Error>>;

/// `x / y` with 3 decimals, from integers: "1.234". `y` is not 0.
fn ratio(x: u64, y: u64) -> String {
    let milli = u128::from(x) * 1000 / u128::from(y.max(1));
    format!("{}.{:03}", milli / 1000, milli % 1000)
}

/// The model path from `--model`, else from `TACO_MODEL`.
fn model_path(job: &Job) -> Option<PathBuf> {
    job.model.clone().or_else(|| std::env::var_os("TACO_MODEL").map(PathBuf::from))
}

fn load_model(path: &Path) -> CliResult<Model> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read the model {}: {e}", path.display()))?;
    Ok(Model::load(&bytes)?)
}

/// The BPE merges of the model file, as token-id pairs. A file without merges gives an empty
/// list: the tokenizer then uses single-byte tokens only.
fn merges_of(model: &Model) -> Vec<(TokenId, TokenId)> {
    model.vocabulary().merges().unwrap_or(&[]).iter().map(|m| (m.left(), m.right())).collect()
}

fn threads(job: &Job) -> Threads {
    job.threads.and_then(Threads::new).unwrap_or_else(Threads::available)
}

fn write_output(job: &Job, bytes: &[u8]) -> CliResult<()> {
    if !job.force && job.output.exists() {
        return Err(format!("{} exists: use -f to replace it", job.output.display()).into());
    }
    std::fs::write(&job.output, bytes)
        .map_err(|e| format!("cannot write {}: {e}", job.output.display()))?;
    Ok(())
}

fn run(job: &Job) -> CliResult<()> {
    let start = Instant::now();
    let input = std::fs::read(&job.input)
        .map_err(|e| format!("cannot read {}: {e}", job.input.display()))?;
    let model = match model_path(job) {
        Some(p) => Some(load_model(&p)?),
        None if job.mode == Mode::Fold => {
            return Err("fold needs a model: use --model or set TACO_MODEL".into());
        }
        None => None,
    };
    let loaded = start.elapsed();
    match (job.mode, &model) {
        (Mode::Fold, Some(model)) => {
            let merges = merges_of(model);
            if merges.is_empty() {
                eprintln!("taco: warning: the model file has no merges; single-byte tokens only");
            }
            let tokenizer =
                Tokenizer::for_vocabulary(model.vocabulary(), model.eot_token(), &merges)?;
            let mut options = FoldOptions::new().with_threads(threads(job));
            if let Some(bs) = job.block_size {
                options = options.with_block_size(BlockSize::new(bs)?);
            }
            let folded = taco::fold(model, &tokenizer, &input, &options)?;
            write_output(job, &folded.container)?;
            let s = folded.stats;
            eprintln!(
                "taco: fold {} -> {}: {} -> {} bytes, ratio {}, {} bits per byte, {} of {} \
                 blocks coded, {} tokens, {:.1?} (model load {:.1?})",
                job.input.display(),
                job.output.display(),
                s.original_len,
                s.container_len,
                ratio(s.original_len, s.container_len),
                ratio(s.container_len * 8, s.original_len),
                s.coded_blocks,
                s.blocks,
                s.tokens,
                start.elapsed(),
                loaded,
            );
        }
        (Mode::Unfold, model) => {
            let options = UnfoldOptions::new().with_threads(threads(job));
            let restored = taco::unfold(model.as_ref(), &input, &options)?;
            write_output(job, &restored)?;
            eprintln!(
                "taco: unfold {} -> {}: {} -> {} bytes, checksum verified, {:.1?} (model load \
                 {:.1?})",
                job.input.display(),
                job.output.display(),
                input.len(),
                restored.len(),
                start.elapsed(),
                loaded,
            );
        }
        (Mode::Fold, None) => return Err("fold needs a model".into()),
    }
    Ok(())
}

fn main() -> ExitCode {
    match args::parse(std::env::args_os().skip(1)) {
        Ok(Command::Help) => {
            eprint!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Run(job)) => match run(&job) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("taco: error: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("taco: {e}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ratio;

    #[test]
    fn ratio_has_three_decimals() {
        assert_eq!(ratio(1, 3), "0.333");
        assert_eq!(ratio(8000, 1000), "8.000");
        assert_eq!(ratio(5, 0), "5.000");
    }
}
