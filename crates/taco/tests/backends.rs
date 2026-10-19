//! Every backend gives the CPU container byte for byte, and unfolds every container (ADR 0016).
//!
//! The GPU tests run on each GPU backend that this machine has (CUDA, wgpu) and say "skip" for
//! a backend that it does not have. They use the tiny model of `tests/common` (a context of 24
//! tokens, so every block of more than 23 tokens has several context windows) and the golden
//! vectors of `tests/container.rs` and `tests/logs_profile.rs`.
//!
//! The context overlap (ADR 0015): CUDA and wgpu give the CPU container for every `K` of
//! the tests (0, 1, 12, 22), and unfold every container.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Tests fail loudly.
#![allow(clippy::print_stderr, clippy::cast_possible_truncation)] // Messages, test data.

mod common;

use common::{CHINESE, ENGLISH, JSON, LOGS, Rng, mixed, model_file, tiny};
use sha2::{Digest, Sha256};
use taco::{
    Backend, BackendChoice, BlockSize, Engine, FoldOptions, FoldProfile, Overlap, TacoError,
    Threads, UnfoldOptions, fold, fold_with, unfold, unfold_with, with_engine,
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The options with no context overlap (the goldens of ADR 0010 and ADR 0016).
fn options(block_size: u32, profile: FoldProfile, threads: usize) -> FoldOptions {
    FoldOptions::new()
        .with_block_size(BlockSize::new(block_size).unwrap())
        .with_profile(profile)
        .with_threads(Threads::new(threads).unwrap())
        .with_overlap(Overlap::NONE)
}

/// The overlaps of the tests: none, the smallest, half of the context of 24 (the default of
/// the profile logs) and the largest (`n_ctx - 2`).
const OVERLAPS: [u16; 4] = [0, 1, 12, 22];

/// The input of the golden vector of `tests/container.rs` (profile text, blocks of 512).
fn text_golden_input() -> Vec<u8> {
    let mut data = ENGLISH.repeat(3).into_bytes();
    data.extend_from_slice(CHINESE.as_bytes());
    data.extend(Rng::new(3).bytes(700));
    data.extend_from_slice(ENGLISH.as_bytes());
    data
}

/// English words in a random order (as `tests/logs_profile.rs`).
fn shuffled_words(seed: u64, n: usize) -> Vec<u8> {
    let words: Vec<&str> = ENGLISH.split(' ').collect();
    let mut rng = Rng::new(seed);
    let mut s = String::new();
    while s.len() < n {
        s.push_str(words[(rng.next_u64() % words.len() as u64) as usize]);
        s.push(' ');
    }
    s.truncate(n);
    s.into_bytes()
}

/// The input of the logs golden vector of `tests/logs_profile.rs` (profile logs, blocks of 512).
fn logs_golden_input() -> Vec<u8> {
    let mut data = shuffled_words(9, 1024);
    data.extend(Rng::new(3).bytes(512));
    data.extend_from_slice(&JSON.repeat(4).as_bytes()[..512]);
    data.extend_from_slice(&ENGLISH.as_bytes()[..100]);
    data
}

/// The cases: (name, input, block size, profile).
fn cases() -> Vec<(&'static str, Vec<u8>, u32, FoldProfile)> {
    let mut logs = LOGS.repeat(6).into_bytes();
    logs.extend(shuffled_words(4, 3000));
    vec![
        ("empty", Vec::new(), 512, FoldProfile::Text),
        ("one byte", b"a".to_vec(), 512, FoldProfile::Logs),
        ("english", ENGLISH.repeat(5).into_bytes(), 700, FoldProfile::Text),
        ("english, blocks of 7", ENGLISH.as_bytes().to_vec(), 7, FoldProfile::Text),
        ("english logs", ENGLISH.repeat(5).into_bytes(), 700, FoldProfile::Logs),
        ("mixed file", mixed(40_000), 16 * 1024, FoldProfile::Text),
        ("mixed file auto", mixed(20_000), 1000, FoldProfile::Auto),
        ("log lines", logs.clone(), 1000, FoldProfile::Logs),
        ("log lines text", logs, 4096, FoldProfile::Text),
        ("random bytes", Rng::new(5).bytes(3000), 1000, FoldProfile::Auto),
    ]
}

/// Runs `check` on each GPU backend of this machine. Gives the names of the backends that ran.
fn on_each_gpu(check: impl Fn(&Engine<'_>)) -> Vec<Backend> {
    let t = tiny();
    let mut ran = Vec::new();
    for choice in [BackendChoice::Cuda, BackendChoice::Gpu] {
        let r = with_engine(choice, &t.file, &t.model, |engine| {
            eprintln!("backend {} ({})", engine.backend(), engine.device());
            check(engine);
            Ok(engine.backend())
        });
        match r {
            Ok(b) => ran.push(b),
            Err(TacoError::Backend { backend, reason }) => {
                eprintln!("skip {backend}: {reason}");
            }
            Err(e) => panic!("{choice:?}: {e}"),
        }
    }
    ran
}

#[test]
fn every_gpu_backend_gives_the_cpu_containers_and_restores_them() {
    let t = tiny();
    let ran = on_each_gpu(|engine| {
        for (name, data, bs, profile) in cases() {
            for (threads, k) in [1, 4].into_iter().flat_map(|n| OVERLAPS.map(|k| (n, k))) {
                let opts = options(bs, profile, threads).with_overlap(Overlap::new(k));
                let cpu = fold(&t.model, &t.tokenizer, &data, &opts).unwrap();
                let uopts = UnfoldOptions::new().with_threads(Threads::new(threads).unwrap());
                let gpu = fold_with(engine, &t.model, &t.tokenizer, &data, &opts).unwrap();
                assert_eq!(
                    hex(&Sha256::digest(&gpu.container)),
                    hex(&Sha256::digest(&cpu.container)),
                    "{name}: {} fold, {threads} threads, K = {k}",
                    engine.backend()
                );
                // The profiles logs and auto fold on the CPU on every engine (ADR 0016 §5).
                let folded_on =
                    if profile == FoldProfile::Text { engine.backend() } else { Backend::Cpu };
                assert_eq!(gpu.stats.backend, folded_on, "{name}");
                let back = unfold_with(engine, Some(&t.model), &cpu.container, &uopts).unwrap();
                assert_eq!(back, data, "{name}: {} unfold, K = {k}", engine.backend());
                assert_eq!(unfold(Some(&t.model), &gpu.container, &uopts).unwrap(), data);
            }
        }
    });
    eprintln!("ran on: {ran:?}");
}

#[test]
fn every_gpu_backend_gives_the_golden_containers() {
    let t = tiny();
    on_each_gpu(|engine| {
        let data = text_golden_input();
        let f =
            fold_with(engine, &t.model, &t.tokenizer, &data, &options(512, FoldProfile::Text, 2))
                .unwrap();
        // CONFORMANCE: the golden container of `tests/container.rs`.
        assert_eq!(
            hex(&Sha256::digest(&f.container)),
            "0b0ba3f1bc8579247476163489ed3155a7828916d27e92f3435e9a8e74157c07"
        );
        let uopts = UnfoldOptions::new();
        assert_eq!(unfold_with(engine, Some(&t.model), &f.container, &uopts).unwrap(), data);
        let data = logs_golden_input();
        let f =
            fold_with(engine, &t.model, &t.tokenizer, &data, &options(512, FoldProfile::Logs, 2))
                .unwrap();
        // CONFORMANCE: the logs golden container of `tests/logs_profile.rs`.
        assert_eq!(
            hex(&Sha256::digest(&f.container)),
            "7262257e6d2875ae393dd2474acc8a00dd1b8adf829eeef75b0ca719f73b55ec"
        );
        assert_eq!(unfold_with(engine, Some(&t.model), &f.container, &uopts).unwrap(), data);
        // The goldens with an overlap (ADR 0015): text with K = 12, and the defaults of the
        // profile logs (K = 12 for the tiny model), on every GPU backend.
        let goldens = [
            (text_golden_input(), FoldProfile::Text, Some(Overlap::new(12)), GOLDEN_TEXT_K12),
            (logs_golden_input(), FoldProfile::Logs, None, GOLDEN_LOGS_DEFAULT),
        ];
        for (data, profile, k, want) in goldens {
            let mut opts = options(512, profile, 2);
            opts.overlap = k;
            let cpu = fold(&t.model, &t.tokenizer, &data, &opts).unwrap();
            assert_eq!(hex(&Sha256::digest(&cpu.container)), want);
            assert_eq!(unfold_with(engine, Some(&t.model), &cpu.container, &uopts).unwrap(), data);
            let f = fold_with(engine, &t.model, &t.tokenizer, &data, &opts).unwrap();
            assert_eq!(hex(&Sha256::digest(&f.container)), want);
            assert_eq!(unfold_with(engine, Some(&t.model), &f.container, &uopts).unwrap(), data);
        }
        // NOTE: a line for each backend, so a crash of a driver at exit does not hide the result.
        eprintln!("goldens equal on {}", engine.backend());
    });
}

/// CONFORMANCE: the text golden with K = 12 (`tests/container.rs`).
const GOLDEN_TEXT_K12: &str = "d705d4b6971fe351f29d14cd5723d3b2a79ced03facae32910bc13d09f3ce340";
/// CONFORMANCE: the logs golden with the defaults of the profile (`tests/logs_profile.rs`).
const GOLDEN_LOGS_DEFAULT: &str =
    "584b60fbd89d3f689cdd75161d43be973ef3c87294680fae1bd0d6f3f7066835";

#[test]
fn a_gpu_refuses_a_model_file_that_is_not_the_loaded_model() {
    let t = tiny();
    let other = model_file(2);
    for choice in [BackendChoice::Cuda, BackendChoice::Gpu] {
        match with_engine(choice, &other, &t.model, |_| Ok(())) {
            Err(TacoError::ModelHashMismatch { .. }) => {}
            Err(TacoError::Backend { backend, reason }) => eprintln!("skip {backend}: {reason}"),
            other => panic!("{choice:?}: {other:?}"),
        }
    }
}

#[test]
fn the_cpu_and_auto_engines_give_the_cpu_container() {
    let t = tiny();
    let data = text_golden_input();
    let opts = options(512, FoldProfile::Text, 1);
    let cpu = fold(&t.model, &t.tokenizer, &data, &opts).unwrap();
    let chosen = with_engine(BackendChoice::Cpu, &t.file, &t.model, |e| {
        assert_eq!(e.backend(), Backend::Cpu);
        fold_with(e, &t.model, &t.tokenizer, &data, &opts)
    })
    .unwrap();
    assert_eq!(chosen.container, cpu.container);
    // Auto works on every machine: a GPU if one passes the self-test, else the CPU.
    let auto = with_engine(BackendChoice::Auto, &t.file, &t.model, |e| {
        eprintln!("auto selected {} ({}); notes {:?}", e.backend(), e.device(), e.notes());
        fold_with(e, &t.model, &t.tokenizer, &data, &opts)
    })
    .unwrap();
    assert_eq!(auto.container, cpu.container);
}
