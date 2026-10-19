//! The command line: `taco fold`, `taco unfold` and `taco --version`, parsed with the standard
//! library only.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The text of `taco --help`.
pub(crate) const USAGE: &str = "\
usage:
  taco fold <input> [-o <output>] [-c] [--model <path.hsm>] [--profile text|logs|auto]
            [--block-size <bytes>] [--overlap <tokens>] [--backend auto|cuda|gpu|cpu]
            [--threads <n>] [-f] [-q]
  taco unfold <input.taco> [-o <output>] [-c] [--model <path.hsm>]
              [--backend auto|cuda|gpu|cpu] [--threads <n>] [-f] [-q]
  taco --version

  fold      compress <input> into <input>.taco
  unfold    restore <input.taco> into the input name without .taco

  The input - is stdin. Then the default output is stdout. The output - is stdout.
  Files of any size stream with bounded memory. fold from stdin or to stdout keeps the
  blocks in a temporary file until the input ends. unfold to stdout writes the data
  before the checksum is known: check the exit status.

options:
  -o, --output <path>     the output file (- for stdout)
  -c, --stdout            write to stdout
  -m, --model <path.hsm>  the model file (default: the TACO_MODEL environment variable)
  --profile <name>        fold only: text (GPT-2 alone), logs (GPT-2, the match model and
                          the mixer) or auto (default: logs if a block gets smaller with
                          it, else text). unfold reads the profile from the file.
  --block-size <bytes>    the bytes in each block (fold only; default 16384 for text,
                          65536 for logs and auto; at most 67108864)
  --overlap <tokens>      fold only: when the model context is full, start it again with
                          the last <tokens> tokens (0 to the context length - 2). Default:
                          0 for text, half of the context for logs and auto (512 for
                          GPT-2). unfold reads it from the file.
  --backend <name>        where the model runs: cuda (NVIDIA), gpu (wgpu: Metal, Vulkan,
                          DX12, GL), cpu, or auto (default: the first of cuda, gpu, cpu that
                          passes a self-test). The output is the same on every backend.
                          fold with the profile logs or auto runs on the CPU.
  --threads <n>           worker threads (default: all CPUs); the output does not change
  -f, --force             replace an output file that exists; write a container to a
                          terminal
  -q, --quiet             no progress line and no summary on stderr (errors only)
  -V, --version           show the version, the commit and the supported formats
  -h, --help              show this text
";

/// What the user asked for.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Command {
    /// Show the usage text.
    Help,
    /// Show the version.
    Version,
    /// Compress or restore a file.
    Run(Job),
}

/// The direction of a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Fold,
    Unfold,
}

/// The `--profile` of fold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ProfileArg {
    Text,
    Logs,
    /// The default (ADR 0012).
    #[default]
    Auto,
}

/// The `--backend` of fold and unfold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum BackendArg {
    /// The default (ADR 0016).
    #[default]
    Auto,
    Cuda,
    Gpu,
    Cpu,
}

/// Where the data comes from or goes to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// A file.
    File(PathBuf),
    /// stdin (input) or stdout (output): the name `-`.
    Std,
}

impl Place {
    fn of(name: OsString) -> Self {
        if name == "-" { Self::Std } else { Self::File(PathBuf::from(name)) }
    }

    /// The name for messages: the path, `stdin` or `stdout`.
    pub(crate) fn show(&self, std_name: &str) -> String {
        match self {
            Self::File(p) => p.display().to_string(),
            Self::Std => std_name.to_owned(),
        }
    }
}

/// One `fold` or `unfold` job, with all defaults applied except the model and the threads.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Job {
    pub(crate) mode: Mode,
    pub(crate) input: Place,
    pub(crate) output: Place,
    /// `--model`, if given. The caller uses `TACO_MODEL` else.
    pub(crate) model: Option<PathBuf>,
    pub(crate) block_size: Option<u32>,
    /// `--overlap`, fold only. `None`: the default of the profile.
    pub(crate) overlap: Option<u16>,
    pub(crate) threads: Option<usize>,
    pub(crate) force: bool,
    /// `--quiet`: no progress and no summary.
    pub(crate) quiet: bool,
    /// Fold only. The default is [`ProfileArg::Auto`].
    pub(crate) profile: ProfileArg,
    /// The default is [`BackendArg::Auto`].
    pub(crate) backend: BackendArg,
}

/// The default output name: `<input>.taco` for fold, the input without `.taco` for unfold.
fn default_output(mode: Mode, input: &Path) -> Result<PathBuf, String> {
    match mode {
        Mode::Fold => {
            let mut s = input.as_os_str().to_owned();
            s.push(".taco");
            Ok(PathBuf::from(s))
        }
        Mode::Unfold => match input.extension() {
            Some(ext) if ext == "taco" && input.file_stem().is_some() => {
                Ok(input.with_extension(""))
            }
            _ => Err("the input name does not end with .taco: give the output with -o".into()),
        },
    }
}

fn number<T: std::str::FromStr>(name: &str, value: Option<OsString>) -> Result<T, String> {
    let v = value.ok_or(format!("{name} needs a value"))?;
    v.to_str().and_then(|s| s.parse().ok()).ok_or(format!("{name}: not a valid number"))
}

/// Parses the arguments after the program name.
pub(crate) fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let mode = match args.next() {
        None => return Err("no command: use fold or unfold".into()),
        Some(a) if a == "fold" => Mode::Fold,
        Some(a) if a == "unfold" => Mode::Unfold,
        Some(a) if a == "-h" || a == "--help" || a == "help" => return Ok(Command::Help),
        Some(a) if a == "-V" || a == "--version" || a == "version" => {
            return Ok(Command::Version);
        }
        Some(a) => return Err(format!("unknown command {:?}: use fold or unfold", a)),
    };
    let (mut input, mut output, mut model) = (None, None, None);
    let (mut block_size, mut threads, mut force, mut quiet) = (None, None, false, false);
    let (mut overlap, mut to_stdout) = (None, false);
    let mut profile = ProfileArg::default();
    let mut backend = BackendArg::default();
    while let Some(a) = args.next() {
        match a.to_str() {
            Some("-h" | "--help") => return Ok(Command::Help),
            Some("-V" | "--version") => return Ok(Command::Version),
            Some("-o" | "--output") => output = Some(args.next().ok_or("-o needs a path")?),
            Some("-c" | "--stdout") => to_stdout = true,
            Some("-m" | "--model") => model = Some(args.next().ok_or("--model needs a path")?),
            Some("--block-size") if mode == Mode::Fold => {
                block_size = Some(number("--block-size", args.next())?);
            }
            Some("--overlap") if mode == Mode::Fold => {
                overlap = Some(number("--overlap", args.next())?);
            }
            Some("--profile") if mode == Mode::Fold => {
                profile = match args.next().as_deref().and_then(std::ffi::OsStr::to_str) {
                    Some("text") => ProfileArg::Text,
                    Some("logs") => ProfileArg::Logs,
                    Some("auto") => ProfileArg::Auto,
                    _ => return Err("--profile needs text, logs or auto".into()),
                };
            }
            Some("--backend") => {
                backend = match args.next().as_deref().and_then(std::ffi::OsStr::to_str) {
                    Some("auto") => BackendArg::Auto,
                    Some("cuda") => BackendArg::Cuda,
                    Some("gpu") => BackendArg::Gpu,
                    Some("cpu") => BackendArg::Cpu,
                    _ => return Err("--backend needs auto, cuda, gpu or cpu".into()),
                };
            }
            Some("--threads") => threads = Some(number("--threads", args.next())?),
            Some("-f" | "--force") => force = true,
            Some("-q" | "--quiet") => quiet = true,
            Some(s) if s.starts_with('-') && s.len() > 1 => {
                return Err(format!("unknown option {s}"));
            }
            _ if input.is_none() => input = Some(a),
            _ => return Err("more than one input file".into()),
        }
    }
    let input = Place::of(input.ok_or("no input file")?);
    let output = match (output, to_stdout, &input) {
        (Some(_), true, _) => return Err("give -o or -c, not both".into()),
        (Some(o), false, _) => Place::of(o),
        (None, true, _) | (None, false, Place::Std) => Place::Std,
        (None, false, Place::File(p)) => Place::File(default_output(mode, p)?),
    };
    if block_size == Some(0) {
        return Err("--block-size must be at least 1".into());
    }
    if threads == Some(0) {
        return Err("--threads must be at least 1".into());
    }
    Ok(Command::Run(Job {
        mode,
        input,
        output,
        model: model.map(PathBuf::from),
        block_size,
        overlap,
        threads,
        force,
        quiet,
        profile,
        backend,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Result<Command, String> {
        parse(args.iter().map(OsString::from))
    }

    fn job(args: &[&str]) -> Job {
        match run(args) {
            Ok(Command::Run(j)) => j,
            other => panic!("{other:?}"),
        }
    }

    fn file(p: &str) -> Place {
        Place::File(PathBuf::from(p))
    }

    #[test]
    fn default_output_names() {
        assert_eq!(job(&["fold", "a.txt"]).output, file("a.txt.taco"));
        assert_eq!(job(&["unfold", "a.txt.taco"]).output, file("a.txt"));
        assert_eq!(job(&["unfold", "dir/x.taco"]).output, file("dir/x"));
        assert!(run(&["unfold", "a.txt"]).is_err());
        assert!(run(&["unfold", ".taco"]).is_err());
    }

    #[test]
    fn options_parse() {
        let j = job(&["fold", "in", "-o", "out", "--model", "m.hsm", "--block-size", "4096"]);
        assert_eq!(j.output, file("out"));
        assert_eq!(j.model, Some(PathBuf::from("m.hsm")));
        assert_eq!(j.block_size, Some(4096));
        let j = job(&["unfold", "x.taco", "--threads", "3", "-f"]);
        assert_eq!((j.threads, j.force, j.quiet), (Some(3), true, false));
        assert!(job(&["fold", "a", "-q"]).quiet);
        assert!(job(&["unfold", "a.taco", "--quiet"]).quiet);
    }

    #[test]
    fn bad_arguments_are_errors() {
        assert!(run(&[]).is_err());
        assert!(run(&["squash", "x"]).is_err());
        assert!(run(&["fold"]).is_err());
        assert!(run(&["fold", "a", "b"]).is_err());
        assert!(run(&["fold", "a", "--block-size", "0"]).is_err());
        assert!(run(&["fold", "a", "--block-size", "x"]).is_err());
        assert!(run(&["fold", "a", "--bogus"]).is_err());
        assert!(run(&["unfold", "a.taco", "--block-size", "5"]).is_err());
        assert!(run(&["fold", "a", "-o"]).is_err());
        assert!(run(&["fold", "a", "-o", "x", "-c"]).is_err());
        assert_eq!(run(&["--help"]), Ok(Command::Help));
        assert_eq!(run(&["fold", "-h"]), Ok(Command::Help));
    }

    #[test]
    fn version_parses_first_and_as_an_option() {
        assert_eq!(run(&["--version"]), Ok(Command::Version));
        assert_eq!(run(&["-V"]), Ok(Command::Version));
        assert_eq!(run(&["fold", "a", "--version"]), Ok(Command::Version));
    }

    #[test]
    fn profile_parses_for_fold_only() {
        assert_eq!(job(&["fold", "a"]).profile, ProfileArg::Auto);
        assert_eq!(job(&["fold", "a", "--profile", "text"]).profile, ProfileArg::Text);
        assert_eq!(job(&["fold", "a", "--profile", "logs"]).profile, ProfileArg::Logs);
        assert_eq!(job(&["fold", "a", "--profile", "auto"]).profile, ProfileArg::Auto);
        assert!(run(&["fold", "a", "--profile", "json"]).is_err());
        assert!(run(&["fold", "a", "--profile"]).is_err());
        assert!(run(&["unfold", "a.taco", "--profile", "logs"]).is_err());
    }

    #[test]
    fn backend_parses_for_fold_and_unfold() {
        assert_eq!(job(&["fold", "a"]).backend, BackendArg::Auto);
        assert_eq!(job(&["fold", "a", "--backend", "cuda"]).backend, BackendArg::Cuda);
        assert_eq!(job(&["unfold", "a.taco", "--backend", "gpu"]).backend, BackendArg::Gpu);
        assert_eq!(job(&["unfold", "a.taco", "--backend", "cpu"]).backend, BackendArg::Cpu);
        assert_eq!(job(&["fold", "a", "--backend", "auto"]).backend, BackendArg::Auto);
        assert!(run(&["fold", "a", "--backend", "tpu"]).is_err());
        assert!(run(&["fold", "a", "--backend"]).is_err());
    }

    #[test]
    fn overlap_parses_for_fold_only() {
        assert_eq!(job(&["fold", "a"]).overlap, None);
        assert_eq!(job(&["fold", "a", "--overlap", "0"]).overlap, Some(0));
        assert_eq!(job(&["fold", "a", "--overlap", "512"]).overlap, Some(512));
        assert_eq!(job(&["fold", "a", "--overlap", "65535"]).overlap, Some(65_535));
        assert!(run(&["fold", "a", "--overlap", "65536"]).is_err());
        assert!(run(&["fold", "a", "--overlap", "-1"]).is_err());
        assert!(run(&["fold", "a", "--overlap"]).is_err());
        assert!(run(&["unfold", "a.taco", "--overlap", "512"]).is_err());
    }

    #[test]
    fn a_lone_dash_is_stdin_and_stdout() {
        let j = job(&["fold", "-", "-o", "x"]);
        assert_eq!((j.input, j.output), (Place::Std, file("x")));
        // stdin gives stdout by default, for fold and unfold.
        assert_eq!(job(&["fold", "-"]).output, Place::Std);
        assert_eq!(job(&["unfold", "-"]).output, Place::Std);
        assert_eq!(job(&["unfold", "a.taco", "-o", "-"]).output, Place::Std);
        let j = job(&["fold", "a.txt", "-c"]);
        assert_eq!((j.input, j.output), (file("a.txt"), Place::Std));
    }
}
