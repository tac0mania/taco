//! The command line: `taco fold` and `taco unfold`, parsed with the standard library only.

use std::ffi::OsString;
use std::path::PathBuf;

/// The text of `taco --help`.
pub(crate) const USAGE: &str = "\
usage:
  taco fold <input> [-o <output>] [--model <path.hsm>] [--profile text|logs|auto]
            [--block-size <bytes>] [--threads <n>] [-f]
  taco unfold <input.taco> [-o <output>] [--model <path.hsm>] [--threads <n>] [-f]

  fold      compress <input> into <input>.taco
  unfold    restore <input.taco> into the input name without .taco

options:
  -o, --output <path>     the output file
  -m, --model <path.hsm>  the model file (default: the TACO_MODEL environment variable)
  --profile <name>        fold only: text (GPT-2 alone), logs (GPT-2, the match model and
                          the mixer) or auto (default: logs if a block gets smaller with
                          it, else text). unfold reads the profile from the file.
  --block-size <bytes>    the bytes in each block (fold only, default 16384)
  --threads <n>           worker threads (default: all CPUs); the output does not change
  -f, --force             replace an output file that exists
  -h, --help              show this text
";

/// What the user asked for.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Command {
    /// Show the usage text.
    Help,
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

/// One `fold` or `unfold` job, with all defaults applied except the model and the threads.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Job {
    pub(crate) mode: Mode,
    pub(crate) input: PathBuf,
    pub(crate) output: PathBuf,
    /// `--model`, if given. The caller uses `TACO_MODEL` else.
    pub(crate) model: Option<PathBuf>,
    pub(crate) block_size: Option<u32>,
    pub(crate) threads: Option<usize>,
    pub(crate) force: bool,
    /// Fold only. The default is [`ProfileArg::Auto`].
    pub(crate) profile: ProfileArg,
}

/// The default output name: `<input>.taco` for fold, the input without `.taco` for unfold.
fn default_output(mode: Mode, input: &std::path::Path) -> Result<PathBuf, String> {
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
        Some(a) => return Err(format!("unknown command {:?}: use fold or unfold", a)),
    };
    let (mut input, mut output, mut model) = (None, None, None);
    let (mut block_size, mut threads, mut force) = (None, None, false);
    let mut profile = ProfileArg::default();
    while let Some(a) = args.next() {
        match a.to_str() {
            Some("-h" | "--help") => return Ok(Command::Help),
            Some("-o" | "--output") => output = Some(args.next().ok_or("-o needs a path")?),
            Some("-m" | "--model") => model = Some(args.next().ok_or("--model needs a path")?),
            Some("--block-size") if mode == Mode::Fold => {
                block_size = Some(number("--block-size", args.next())?);
            }
            Some("--profile") if mode == Mode::Fold => {
                profile = match args.next().as_deref().and_then(std::ffi::OsStr::to_str) {
                    Some("text") => ProfileArg::Text,
                    Some("logs") => ProfileArg::Logs,
                    Some("auto") => ProfileArg::Auto,
                    _ => return Err("--profile needs text, logs or auto".into()),
                };
            }
            Some("--threads") => threads = Some(number("--threads", args.next())?),
            Some("-f" | "--force") => force = true,
            Some(s) if s.starts_with('-') && s.len() > 1 => {
                return Err(format!("unknown option {s}"));
            }
            _ if input.is_none() => input = Some(a),
            _ => return Err("more than one input file".into()),
        }
    }
    let input = PathBuf::from(input.ok_or("no input file")?);
    let output = match output {
        Some(o) => PathBuf::from(o),
        None => default_output(mode, &input)?,
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
        threads,
        force,
        profile,
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

    #[test]
    fn default_output_names() {
        assert_eq!(job(&["fold", "a.txt"]).output, PathBuf::from("a.txt.taco"));
        assert_eq!(job(&["unfold", "a.txt.taco"]).output, PathBuf::from("a.txt"));
        assert_eq!(job(&["unfold", "dir/x.taco"]).output, PathBuf::from("dir/x"));
        assert!(run(&["unfold", "a.txt"]).is_err());
        assert!(run(&["unfold", ".taco"]).is_err());
    }

    #[test]
    fn options_parse() {
        let j = job(&["fold", "in", "-o", "out", "--model", "m.hsm", "--block-size", "4096"]);
        assert_eq!(j.output, PathBuf::from("out"));
        assert_eq!(j.model, Some(PathBuf::from("m.hsm")));
        assert_eq!(j.block_size, Some(4096));
        let j = job(&["unfold", "x.taco", "--threads", "3", "-f"]);
        assert_eq!((j.threads, j.force), (Some(3), true));
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
        assert_eq!(run(&["--help"]), Ok(Command::Help));
        assert_eq!(run(&["fold", "-h"]), Ok(Command::Help));
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
    fn a_lone_dash_is_an_input_name() {
        assert_eq!(job(&["fold", "-", "-o", "x"]).input, PathBuf::from("-"));
    }
}
