//! The `taco` binary: fold and unfold files with the tiny test model of the `taco` crate.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Tests fail loudly.

#[path = "../../taco/tests/common/mod.rs"]
mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn taco(args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_taco")).args(args).env_remove("TACO_MODEL").output().unwrap()
}

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn fold_then_unfold_restores_the_file() {
    let d = dir("cli_round_trip");
    let model = d.join("tiny.hsm");
    std::fs::write(&model, common::model_file(1)).unwrap();
    let input = d.join("text.txt");
    let data = common::ENGLISH.repeat(4);
    std::fs::write(&input, &data).unwrap();

    let out = taco(&[
        "fold".as_ref(),
        &input,
        "--model".as_ref(),
        &model,
        "--block-size".as_ref(),
        "700".as_ref(),
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let summary = String::from_utf8_lossy(&out.stderr);
    assert!(summary.contains("bits per byte"), "{summary}");
    let folded = d.join("text.txt.taco");
    assert!(std::fs::metadata(&folded).unwrap().len() < data.len() as u64);

    // The output exists: unfold refuses to replace it without -f.
    let out = taco(&["unfold".as_ref(), &folded, "--model".as_ref(), &model]);
    assert!(!out.status.success());
    std::fs::remove_file(&input).unwrap();
    let out = taco(&["unfold".as_ref(), &folded, "--model".as_ref(), &model]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(std::fs::read(&input).unwrap(), data.as_bytes());

    // Without a model, unfold names the model that the container needs.
    let other = d.join("other");
    let out = taco(&["unfold".as_ref(), &folded, "-o".as_ref(), &other]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("needs the model"));
}

#[test]
fn every_profile_round_trips_and_unfold_needs_no_profile() {
    let d = dir("cli_profiles");
    let model = d.join("tiny.hsm");
    std::fs::write(&model, common::model_file(1)).unwrap();
    let input = d.join("text.txt");
    let data = common::ENGLISH.repeat(3);
    std::fs::write(&input, &data).unwrap();
    for profile in ["text", "logs", "auto"] {
        let folded = d.join(format!("{profile}.taco"));
        let out = taco(&[
            "fold".as_ref(),
            &input,
            "-o".as_ref(),
            &folded,
            "--model".as_ref(),
            &model,
            "--profile".as_ref(),
            profile.as_ref(),
            "--block-size".as_ref(),
            "512".as_ref(),
            // NOTE: the default overlap of logs and auto is half of the context (12 for the
            // tiny model); text gets the same K, so the three containers are comparable.
            "--overlap".as_ref(),
            "12".as_ref(),
        ]);
        let summary = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{summary}");
        assert!(summary.contains("overlap 12"), "{summary}");
        if profile != "auto" {
            assert!(summary.contains(&format!("profile {profile}")), "{summary}");
        }
        let back = d.join(format!("{profile}.out"));
        let out =
            taco(&["unfold".as_ref(), &folded, "-o".as_ref(), &back, "--model".as_ref(), &model]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(std::fs::read(&back).unwrap(), data.as_bytes());
    }
    let read = |name: &str| std::fs::read(d.join(name)).unwrap();
    let (text, logs, auto) = (read("text.taco"), read("logs.taco"), read("auto.taco"));
    assert!(logs.len() <= text.len());
    // Auto: the logs container if it is smaller, else the text container.
    assert_eq!(auto, if logs.len() < text.len() { logs } else { text });
    let out = taco(&["fold".as_ref(), &input, "--profile".as_ref(), "json".as_ref()]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn bad_arguments_exit_with_code_2() {
    let out = taco(&["squash".as_ref()]);
    assert_eq!(out.status.code(), Some(2));
    let out = taco(&["fold".as_ref(), "x".as_ref()]);
    // A missing model or file is an error of the run (code 1), not of the arguments.
    assert_eq!(out.status.code(), Some(1));
}

/// The context overlap `K` of a container: bit 1 of the coding parts (offset 90) and bytes
/// 94 and 95 (ADR 0015 §4).
fn overlap_of(container: &[u8]) -> Option<u16> {
    (container[90] & 2 != 0).then(|| u16::from_le_bytes([container[94], container[95]]))
}

#[test]
fn overlap_defaults_by_profile_and_round_trips() {
    let d = dir("cli_overlap");
    let model = d.join("tiny.hsm");
    std::fs::write(&model, common::model_file(1)).unwrap();
    let input = d.join("text.txt");
    let data = common::ENGLISH.repeat(3);
    std::fs::write(&input, &data).unwrap();
    // (profile, --overlap, the K of the container). The tiny model has a context of 24.
    let cases: [(&str, Option<&str>, Option<u16>); 6] = [
        ("text", None, None),
        ("logs", None, Some(12)),
        ("auto", None, Some(12)),
        ("logs", Some("0"), None),
        ("text", Some("22"), Some(22)),
        ("logs", Some("1"), Some(1)),
    ];
    for (i, (profile, k, want)) in cases.into_iter().enumerate() {
        let folded = d.join(format!("{i}.taco"));
        let mut args: Vec<&Path> = vec![
            "fold".as_ref(),
            &input,
            "-o".as_ref(),
            &folded,
            "--model".as_ref(),
            &model,
            "--profile".as_ref(),
            profile.as_ref(),
            "--block-size".as_ref(),
            "700".as_ref(),
        ];
        if let Some(k) = k {
            args.extend::<[&Path; 2]>(["--overlap".as_ref(), k.as_ref()]);
        }
        let out = taco(&args);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let bytes = std::fs::read(&folded).unwrap();
        assert_eq!(overlap_of(&bytes), want, "case {i}");
        let back = d.join(format!("{i}.out"));
        let out =
            taco(&["unfold".as_ref(), &folded, "-o".as_ref(), &back, "--model".as_ref(), &model]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(std::fs::read(&back).unwrap(), data.as_bytes());
    }
    // K > n_ctx - 2 is refused before a byte is written.
    let bad = d.join("bad.taco");
    let out = taco(&[
        "fold".as_ref(),
        &input,
        "-o".as_ref(),
        &bad,
        "--model".as_ref(),
        &model,
        "--overlap".as_ref(),
        "23".as_ref(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("overlap 23"));
    assert!(!bad.exists());
}

/// Runs `taco` with `stdin` as its standard input.
fn taco_with_stdin(args: &[&Path], stdin: &[u8]) -> Output {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new(env!("CARGO_BIN_EXE_taco"))
        .args(args)
        .env_remove("TACO_MODEL")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut pipe = child.stdin.take().unwrap();
    let data = stdin.to_vec();
    let writer = std::thread::spawn(move || {
        // NOTE: taco can stop to read early on an error; a broken pipe is then expected.
        let _ignored = pipe.write_all(&data);
    });
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap();
    out
}

#[test]
fn pipes_give_the_bytes_of_files() {
    let d = dir("cli_pipes");
    let model = d.join("tiny.hsm");
    std::fs::write(&model, common::model_file(1)).unwrap();
    let input = d.join("text.txt");
    let data = common::ENGLISH.repeat(5);
    std::fs::write(&input, &data).unwrap();
    let m: &Path = "--model".as_ref();
    let bs: &[&Path] = &["--block-size".as_ref(), "300".as_ref()];

    // The reference: a file to a file (the method "seek back").
    let file_out = d.join("file.taco");
    let mut args = vec!["fold".as_ref(), input.as_path(), "-o".as_ref(), &file_out, m, &model];
    args.extend_from_slice(bs);
    let out = taco(&args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let reference = std::fs::read(&file_out).unwrap();
    // No temporary file stays in the directory.
    let names: Vec<_> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(names.len(), 3, "{names:?}");

    // stdin to a file, and a file to stdout with -o - and with -c.
    let pipe_out = d.join("pipe.taco");
    let mut args =
        vec!["fold".as_ref(), "-".as_ref(), "-o".as_ref(), pipe_out.as_path(), m, &model];
    args.extend_from_slice(bs);
    let out = taco_with_stdin(&args, data.as_bytes());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(std::fs::read(&pipe_out).unwrap(), reference);
    for to_stdout in [&["-o".as_ref(), "-".as_ref()][..], &["-c".as_ref()][..]] {
        let mut args = vec!["fold".as_ref(), input.as_path(), m, &model];
        args.extend_from_slice(to_stdout);
        args.extend_from_slice(bs);
        let out = taco(&args);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(out.stdout, reference);
    }
    // stdin to stdout (the default output of stdin).
    let mut args = vec!["fold".as_ref(), "-".as_ref(), m, &model];
    args.extend_from_slice(bs);
    let out = taco_with_stdin(&args, data.as_bytes());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.stdout, reference);

    // unfold: stdin to stdout, and a file to stdout.
    let out = taco_with_stdin(&["unfold".as_ref(), "-".as_ref(), m, &model], &reference);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.stdout, data.as_bytes());
    let out = taco(&["unfold".as_ref(), &file_out, "-c".as_ref(), m, &model]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.stdout, data.as_bytes());

    // A truncated container from a pipe is an error with the exit status 1.
    let out = taco_with_stdin(
        &["unfold".as_ref(), "-".as_ref(), m, &model],
        &reference[..reference.len() - 1],
    );
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn a_failed_unfold_leaves_no_output_file() {
    let d = dir("cli_no_partial");
    let model = d.join("tiny.hsm");
    std::fs::write(&model, common::model_file(1)).unwrap();
    let input = d.join("text.txt");
    std::fs::write(&input, common::ENGLISH.repeat(3)).unwrap();
    let folded = d.join("text.txt.taco");
    let out = taco(&["fold".as_ref(), &input, "--model".as_ref(), &model]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let mut bytes = std::fs::read(&folded).unwrap();
    bytes[60] ^= 1; // the checksum
    std::fs::write(&folded, &bytes).unwrap();
    let restored = d.join("restored.txt");
    let out =
        taco(&["unfold".as_ref(), &folded, "-o".as_ref(), &restored, "--model".as_ref(), &model]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("checksum"));
    assert!(!restored.exists());
    let names: Vec<_> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(names.len(), 3, "no temporary file stays: {names:?}");
}

#[test]
fn quiet_prints_nothing_and_version_names_the_formats() {
    let d = dir("cli_quiet");
    let model = d.join("tiny.hsm");
    std::fs::write(&model, common::model_file(1)).unwrap();
    let input = d.join("a.txt");
    std::fs::write(&input, common::ENGLISH).unwrap();
    let out = taco(&["fold".as_ref(), &input, "--model".as_ref(), &model, "-q".as_ref()]);
    assert!(out.status.success());
    // Warnings stay; the summary and the progress line go.
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.lines().all(|l| l.starts_with("taco: warning:")), "{err}");

    let out = taco(&["--version".as_ref()]);
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.starts_with(&format!("taco {} (", env!("CARGO_PKG_VERSION"))), "{text}");
    assert!(text.contains("container: reads versions 1, 2; writes version 2"), "{text}");
    assert!(text.contains("model files: reads versions 1, 2"), "{text}");
    assert_eq!(taco(&["-V".as_ref()]).stdout, text.as_bytes());
}

#[test]
fn fold_refuses_a_block_size_above_the_stream_limit() {
    let d = dir("cli_limit");
    let model = d.join("tiny.hsm");
    std::fs::write(&model, common::model_file(1)).unwrap();
    let input = d.join("a.txt");
    std::fs::write(&input, common::ENGLISH).unwrap();
    let out = taco(&[
        "fold".as_ref(),
        &input,
        "--model".as_ref(),
        &model,
        "--block-size".as_ref(),
        "67108865".as_ref(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("limit"));
    assert!(!d.join("a.txt.taco").exists());
}
