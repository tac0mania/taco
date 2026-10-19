//! The `taco` command: `taco fold`, `taco unfold` and `taco --version`.
//!
//! The command has no compression logic. It opens the input and the output (files, stdin or
//! stdout), opens the backend (`taco::with_engine`), calls the stream fold or the stream
//! unfold of `taco::stream` (ADR 0018), and prints a progress line and a one-line summary to
//! stderr.

// NOTE: a command prints its summary and its errors to stderr; the library never prints.
#![allow(clippy::print_stderr)]

mod args;
mod files;
mod meter;

use std::error::Error;
use std::fs::File;
use std::io::{self, BufWriter, IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use args::{BackendArg, Command, Job, Mode, Place, ProfileArg, USAGE};
use files::PendingFile;
use meter::Meter;
use taco::stream::{StreamFold, StreamUnfold};
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
    if job.backend == BackendArg::Auto && !failed.is_empty() && !job.quiet {
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

/// The input of a job: a reader and its length, if the input is a regular file.
struct Input {
    reader: Box<dyn Read>,
    /// The file, if the input is a regular file (it can seek).
    file: Option<File>,
    len: Option<u64>,
}

fn open_input(place: &Place) -> CliResult<Input> {
    match place {
        Place::Std => Ok(Input { reader: Box::new(io::stdin().lock()), file: None, len: None }),
        Place::File(p) => {
            let f = File::open(p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
            let meta = f.metadata().map_err(|e| format!("cannot read {}: {e}", p.display()))?;
            if meta.is_file() {
                Ok(Input { reader: Box::new(io::empty()), file: Some(f), len: Some(meta.len()) })
            } else {
                // NOTE: a named pipe or a device: read it as a stream.
                Ok(Input { reader: Box::new(f), file: None, len: None })
            }
        }
    }
}

/// Refuses to write a container to a terminal without `-f` (as gzip and zstd do).
fn check_stdout(job: &Job) -> CliResult<()> {
    if job.mode == Mode::Fold && !job.force && io::stdout().is_terminal() {
        return Err("stdout is a terminal: give -o, a pipe, or -f to write the container".into());
    }
    Ok(())
}

/// Prints the version, the commit, the target and the formats (model-distribution §6.6).
#[allow(clippy::print_stdout)]
fn print_version() {
    let list = |v: &[String]| v.join(", ");
    let reads: Vec<String> = taco::CONTAINER_READS.iter().map(u16::to_string).collect();
    let models: Vec<String> = taco::MODEL_FILE_READS.iter().map(u32::to_string).collect();
    println!(
        "taco {} ({}, {})",
        env!("CARGO_PKG_VERSION"),
        option_env!("TACO_COMMIT").unwrap_or("unknown"),
        env!("TACO_TARGET"),
    );
    println!(
        "container: reads versions {}; writes version {}",
        list(&reads),
        taco::CONTAINER_WRITES
    );
    println!("model files: reads versions {}", list(&models));
}

fn fold_options(job: &Job, model: &Model) -> CliResult<FoldOptions> {
    let mut options =
        FoldOptions::new().with_threads(threads(job)).with_profile(fold_profile(job.profile));
    if let Some(bs) = job.block_size {
        options = options.with_block_size(BlockSize::new(bs)?);
    }
    if let Some(k) = job.overlap {
        options = options.with_overlap(Overlap::new(k));
    }
    // NOTE: a K that the model refuses is an error before a GPU opens.
    options.effective_overlap(model)?;
    Ok(options)
}

fn fold(job: &Job, file: &[u8], model: &Model, start: Instant) -> CliResult<()> {
    let merges = merges_of(model);
    if merges.is_empty() {
        eprintln!("taco: warning: the model file has no merges; single-byte tokens only");
    }
    let tokenizer = Tokenizer::for_vocabulary(model.vocabulary(), model.eot_token(), &merges)?;
    let options = fold_options(job, model)?;
    let input = open_input(&job.input)?;
    if job.output == Place::Std {
        check_stdout(job)?;
    }
    let mut pending = match &job.output {
        Place::File(p) => Some(PendingFile::create(p, job.force)?),
        Place::Std => None,
    };
    let meter = Meter::new(!job.quiet && io::stderr().is_terminal(), "fold", input.len);
    let progress = |p| meter.update(p);
    let loaded = start.elapsed();
    // NOTE: the profiles logs and auto fold on the CPU on every engine (ADR 0016), so they do
    // not open a GPU.
    let choice = if options.uses_gpu() { backend_choice(job.backend) } else { BackendChoice::Cpu };
    let (stats, backend) = taco::with_engine(choice, file, model, |engine| {
        print_notes(job, engine);
        let fold = StreamFold::new(engine, model, &tokenizer, &options).with_progress(&progress);
        let Input { reader, file: in_file, len } = input;
        let reader: Box<dyn Read> = match in_file {
            Some(f) => Box::new(f),
            None => reader,
        };
        let r = match (&mut pending, len) {
            // ADR 0018 §1.1: a file of known length into a file: seek back, no spool.
            (Some(out), Some(len)) => fold.to_seekable(reader, len, out.writer()),
            (Some(out), None) => {
                let dir = out.dir();
                fold.to_writer(reader, None, out.writer(), &dir)
            }
            (None, len) => {
                let mut stdout = BufWriter::new(io::stdout().lock());
                fold.to_writer(reader, len, &mut stdout, &std::env::temp_dir())
            }
        };
        Ok((r, backend_text(engine)))
    })?;
    meter.clear();
    let s = stats?;
    if let Some(out) = pending {
        out.commit()?;
    }
    if !job.quiet {
        eprintln!(
            "taco: fold {} -> {}: {} -> {} bytes, ratio {}, {} bits per byte, profile {}, \
             blocks of {} bytes, overlap {}, {} of {} blocks coded and {} mixed (profile \
             text: {} bytes), {} tokens, backend {}, {:.1?} (model load {:.1?})",
            job.input.show("stdin"),
            job.output.show("stdout"),
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
    Ok(())
}

fn unfold(job: &Job, model: Option<&(Vec<u8>, Model)>, start: Instant) -> CliResult<()> {
    let options = UnfoldOptions::new().with_threads(threads(job));
    let input = open_input(&job.input)?;
    let mut pending = match &job.output {
        Place::File(p) => Some(PendingFile::create(p, job.force)?),
        Place::Std => None,
    };
    let meter = Meter::new(!job.quiet && io::stderr().is_terminal(), "unfold", input.len);
    let progress = |p| meter.update(p);
    let loaded = start.elapsed();
    let run = |engine: &Engine<'_>, model: Option<&Model>| {
        let unfold = StreamUnfold::new(engine, model, &options).with_progress(&progress);
        let Input { reader, file, .. } = input;
        let mut stdout;
        let out: &mut dyn io::Write = match &mut pending {
            Some(p) => p.writer(),
            None => {
                stdout = BufWriter::new(io::stdout().lock());
                &mut stdout
            }
        };
        match file {
            Some(f) => unfold.from_seekable(f, out),
            None => unfold.from_reader(reader, out),
        }
    };
    let (stats, backend) = match model {
        Some((file, model)) => {
            taco::with_engine(backend_choice(job.backend), file, model, |engine| {
                print_notes(job, engine);
                Ok((run(engine, Some(model)), backend_text(engine)))
            })?
        }
        // NOTE: a container in the raw-only mode needs no model and no backend.
        None => (run(&Engine::cpu(), None), "cpu (no model)".to_owned()),
    };
    meter.clear();
    let s = stats?;
    if let Some(out) = pending {
        out.commit()?;
    }
    if !job.quiet {
        eprintln!(
            "taco: unfold {} -> {}: {} -> {} bytes, container version {}, checksum verified, \
             backend {}, {:.1?} (model load {:.1?})",
            job.input.show("stdin"),
            job.output.show("stdout"),
            s.container_len,
            s.original_len,
            s.version,
            backend,
            start.elapsed(),
            loaded,
        );
    }
    Ok(())
}

fn run(job: &Job) -> CliResult<()> {
    let start = Instant::now();
    let model = match model_path(job) {
        Some(p) => Some(load_model(&p)?),
        None if job.mode == Mode::Fold => {
            return Err("fold needs a model: use --model or set TACO_MODEL".into());
        }
        None => None,
    };
    match (job.mode, &model) {
        (Mode::Fold, Some((file, model))) => fold(job, file, model, start),
        (Mode::Unfold, model) => unfold(job, model.as_ref(), start),
        (Mode::Fold, None) => Err("fold needs a model".into()),
    }
}

fn main() -> ExitCode {
    match args::parse(std::env::args_os().skip(1)) {
        Ok(Command::Help) => {
            eprint!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            print_version();
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
    use crate::files::dir_of;
    use std::path::Path;

    #[test]
    fn ratio_has_three_decimals() {
        assert_eq!(ratio(1, 3), "0.333");
        assert_eq!(ratio(8000, 1000), "8.000");
        assert_eq!(ratio(5, 0), "5.000");
    }

    #[test]
    fn the_directory_of_a_bare_name_is_the_current_directory() {
        assert_eq!(dir_of(Path::new("x.taco")), Path::new("."));
        assert_eq!(dir_of(Path::new("a/x.taco")), Path::new("a"));
    }
}
