//! The `taco` command: `taco fold` and `taco unfold`.
//!
//! The command has no compression logic. It reads the files, opens the backend
//! (`taco::with_engine`), calls `taco::fold_with` or `taco::unfold_with`, writes the output,
//! and prints a one-line summary to stderr.

// NOTE: a command prints its summary and its errors to stderr; the library never prints.
#![allow(clippy::print_stderr)]

mod args;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use args::{BackendArg, Command, Job, Mode, ProfileArg, USAGE};
use taco::{
    BackendChoice, BlockSize, Engine, FoldOptions, FoldProfile, Model, Overlap, Profile, Threads,
    TokenId, Tokenizer, UnfoldOptions,
};

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

/// The model file and the model. A GPU backend loads the file again into GPU memory.
fn load_model(path: &Path) -> CliResult<(Vec<u8>, Model)> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read the model {}: {e}", path.display()))?;
    let model = Model::load(&bytes)?;
    Ok((bytes, model))
}

fn backend_choice(arg: BackendArg) -> BackendChoice {
    match arg {
        BackendArg::Auto => BackendChoice::Auto,
        BackendArg::Cuda => BackendChoice::Cuda,
        BackendArg::Gpu => BackendChoice::Gpu,
        BackendArg::Cpu => BackendChoice::Cpu,
    }
}

/// The backend and its device, for the summary: "cuda (NVIDIA ...)".
fn backend_text(engine: &Engine<'_>) -> String {
    format!("{} ({})", engine.backend(), engine.device())
}

/// With `--backend auto`, says why a GPU that this machine has was not used (one line). A
/// backend that the machine does not have (no driver, no device) is not a note.
fn print_notes(job: &Job, engine: &Engine<'_>) {
    let failed: Vec<String> = engine
        .notes()
        .iter()
        .filter(|n| !n.missing)
        .map(|n| format!("{}: {}", n.backend, n.reason))
        .collect();
    if job.backend == BackendArg::Auto && !failed.is_empty() {
        eprintln!("taco: note: a GPU backend was not used: {}", failed.join("; "));
    }
}

/// The BPE merges of the model file, as token-id pairs. A file without merges gives an empty
/// list: the tokenizer then uses single-byte tokens only.
fn merges_of(model: &Model) -> Vec<(TokenId, TokenId)> {
    model.vocabulary().merges().unwrap_or(&[]).iter().map(|m| (m.left(), m.right())).collect()
}

fn fold_profile(arg: ProfileArg) -> FoldProfile {
    match arg {
        ProfileArg::Text => FoldProfile::Text,
        ProfileArg::Logs => FoldProfile::Logs,
        ProfileArg::Auto => FoldProfile::Auto,
    }
}

fn profile_name(profile: Profile) -> &'static str {
    match profile {
        Profile::Logs => "logs",
        _ => "text",
    }
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
        (Mode::Fold, Some((file, model))) => {
            let merges = merges_of(model);
            if merges.is_empty() {
                eprintln!("taco: warning: the model file has no merges; single-byte tokens only");
            }
            let tokenizer =
                Tokenizer::for_vocabulary(model.vocabulary(), model.eot_token(), &merges)?;
            let mut options = FoldOptions::new()
                .with_threads(threads(job))
                .with_profile(fold_profile(job.profile));
            if let Some(bs) = job.block_size {
                options = options.with_block_size(BlockSize::new(bs)?);
            }
            if let Some(k) = job.overlap {
                options = options.with_overlap(Overlap::new(k));
            }
            // NOTE: a K that the model refuses is an error before a GPU opens.
            options.effective_overlap(model)?;
            // NOTE: the profiles logs and auto fold on the CPU on every engine (ADR 0016), so
            // they do not open a GPU.
            let choice =
                if options.uses_gpu() { backend_choice(job.backend) } else { BackendChoice::Cpu };
            let (folded, backend) = taco::with_engine(choice, file, model, |engine| {
                print_notes(job, engine);
                let f = taco::fold_with(engine, model, &tokenizer, &input, &options)?;
                Ok((f, backend_text(engine)))
            })?;
            write_output(job, &folded.container)?;
            let s = folded.stats;
            eprintln!(
                "taco: fold {} -> {}: {} -> {} bytes, ratio {}, {} bits per byte, profile {}, \
                 blocks of {} bytes, overlap {}, {} of {} blocks coded and {} mixed (profile \
                 text: {} bytes), {} tokens, backend {}, {:.1?} (model load {:.1?})",
                job.input.display(),
                job.output.display(),
                s.original_len,
                s.container_len,
                ratio(s.original_len, s.container_len),
                ratio(s.container_len * 8, s.original_len),
                profile_name(s.profile),
                options.effective_block_size().get(),
                s.overlap.get(),
                s.coded_blocks,
                s.blocks,
                s.mixed_blocks,
                s.text_container_len,
                s.tokens,
                backend,
                start.elapsed(),
                loaded,
            );
        }
        (Mode::Unfold, model) => {
            let options = UnfoldOptions::new().with_threads(threads(job));
            let (restored, backend) = match model {
                Some((file, model)) => {
                    taco::with_engine(backend_choice(job.backend), file, model, |engine| {
                        print_notes(job, engine);
                        let r = taco::unfold_with(engine, Some(model), &input, &options)?;
                        Ok((r, backend_text(engine)))
                    })?
                }
                // NOTE: a container in the raw-only mode needs no model and no backend.
                None => (taco::unfold(None, &input, &options)?, "cpu (no model)".to_owned()),
            };
            write_output(job, &restored)?;
            eprintln!(
                "taco: unfold {} -> {}: {} -> {} bytes, checksum verified, backend {}, {:.1?} \
                 (model load {:.1?})",
                job.input.display(),
                job.output.display(),
                input.len(),
                restored.len(),
                backend,
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
