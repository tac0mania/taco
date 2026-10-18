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
        ]);
        let summary = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{summary}");
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
