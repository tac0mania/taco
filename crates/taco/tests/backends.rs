//! Every backend gives the CPU container byte for byte, and unfolds every container (ADR 0016).
//!
//! The GPU tests run on each GPU backend that this machine has (CUDA, wgpu) and say "skip" for
//! a backend that it does not have (no driver, no device, no adapter: `missing`). A backend
//! that the machine has must work: a failed self-test or any other fault fails the test. They
//! use the tiny model of `tests/common` (a context of 24
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

/// `Some(reason)` if this machine does not have the backend of `choice`: no CUDA driver or no
/// CUDA device (`hardshell_cuda::Error::NoDevice`), or no wgpu adapter with compute shaders
/// (`hardshell_gpu::Error::NoAdapter`). Only then does a GPU test skip. Every other fault of a
/// backend that the machine has (a failed self-test, wrong bits, a driver error, no memory, a
/// device below compute capability 8.0) fails the test, so a GPU with wrong bits is never
/// skipped.
fn missing(choice: BackendChoice) -> Option<String> {
    match choice {
        BackendChoice::Cuda => match hardshell_cuda::CudaDevice::open(0) {
            Err(e @ hardshell_cuda::Error::NoDevice { .. }) => Some(e.to_string()),
            _ => None,
        },
        BackendChoice::Gpu => {
            match hardshell_gpu::GpuDevice::open(&hardshell_gpu::AdapterChoice::BestHardware) {
                Err(e @ hardshell_gpu::Error::NoAdapter { .. }) => Some(e.to_string()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The GPU backends of the tests.
const GPUS: [BackendChoice; 2] = [BackendChoice::Cuda, BackendChoice::Gpu];

/// Runs `check` on each GPU backend of this machine. Gives the names of the backends that ran.
/// A backend that the machine does not have ([`missing`]) is skipped; any other error of
/// [`with_engine`] fails.
fn on_each_gpu(check: impl Fn(&Engine<'_>)) -> Vec<Backend> {
    let t = tiny();
    let mut ran = Vec::new();
    for choice in GPUS {
        if let Some(reason) = missing(choice) {
            eprintln!("skip {choice:?}: the machine does not have it: {reason}");
            continue;
        }
        let r = with_engine(choice, &t.file, &t.model, |engine| {
            eprintln!("backend {} ({})", engine.backend(), engine.device());
            check(engine);
            Ok(engine.backend())
        });
        match r {
            Ok(b) => ran.push(b),
            Err(e) => panic!("{choice:?}: the machine has the backend, but it failed: {e}"),
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
                // The profiles logs and auto fold on CUDA (the mixed coder op) and on the CPU
                // on wgpu (ADR 0016 §5).
                let folded_on = if profile == FoldProfile::Text || engine.backend() == Backend::Cuda
                {
                    engine.backend()
                } else {
                    Backend::Cpu
                };
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
    for choice in GPUS {
        if let Some(reason) = missing(choice) {
            eprintln!("skip {choice:?}: the machine does not have it: {reason}");
            continue;
        }
        match with_engine(choice, &other, &t.model, |_| Ok(())) {
            Err(TacoError::ModelHashMismatch { .. }) => {}
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

#[test]
fn the_spread_rule_depends_only_on_the_length_and_restores_on_every_backend() {
    let spread = FoldOptions::new().with_spread(true);
    let size = |o: &FoldOptions, n: usize| o.block_size_for(n).get();
    assert_eq!(size(&spread, 0), 4096);
    assert_eq!(size(&spread, 24_603), 4096);
    assert_eq!(size(&spread, 426_754), 8192);
    assert_eq!(size(&spread, 64 * 16 * 1024), 16 * 1024);
    assert_eq!(size(&spread, 2_663_455), 16 * 1024);
    assert_eq!(size(&FoldOptions::new(), 24_603), 16 * 1024);
    assert_eq!(size(&spread.with_block_size(BlockSize::new(700).unwrap()), 24_603), 700);
    assert_eq!(size(&spread.with_profile(FoldProfile::Logs), 24_603), 64 * 1024);
    let t = tiny();
    let data = mixed(40_000);
    let opts = spread.with_threads(Threads::new(4).unwrap());
    let cpu = fold(&t.model, &t.tokenizer, &data, &opts).unwrap();
    assert_eq!(cpu.stats.block_size.map(BlockSize::get), Some(4096));
    assert_eq!(cpu.stats.blocks, 10);
    assert_eq!(unfold(Some(&t.model), &cpu.container, &UnfoldOptions::new()).unwrap(), data);
    on_each_gpu(|engine| {
        let gpu = fold_with(engine, &t.model, &t.tokenizer, &data, &opts).unwrap();
        assert_eq!(gpu.container, cpu.container, "{}", engine.backend());
        let back = unfold_with(engine, Some(&t.model), &gpu.container, &UnfoldOptions::new());
        assert_eq!(back.unwrap(), data);
    });
}

/// The peak resident memory of this process (`VmHWM`, Linux), in KiB.
#[cfg(target_os = "linux")]
fn peak_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmHWM:")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

/// The fold side of security finding F2: the GPU fold of the profile `logs` makes the mixer
/// state of a block (144 KiB of logistic tables and 24 MiB of zeroed match tables) when the
/// block enters a slot, and drops it when the block leaves. 5,000 blocks of 2 bytes must not
/// need 5,000 states at the same time (about 0.7 GiB, est.). The container is the CPU
/// container.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "needs a GPU and the free timing lock"]
fn many_logs_blocks_on_a_gpu_fold_with_bounded_memory() {
    let t = tiny();
    let data = shuffled_words(11, 10_000);
    let opts = options(2, FoldProfile::Logs, 4);
    let cpu = fold(&t.model, &t.tokenizer, &data, &opts).unwrap();
    assert_eq!(cpu.stats.blocks, 5_000);
    let before = peak_kib();
    let ran = on_each_gpu(|engine| {
        // NOTE: the growth of the engine alone, for the record.
        eprintln!("engine alone grew the peak by {} KiB", peak_kib().saturating_sub(before));
        let gpu = fold_with(engine, &t.model, &t.tokenizer, &data, &opts).unwrap();
        assert_eq!(gpu.container, cpu.container, "{}", engine.backend());
        let grew = peak_kib().saturating_sub(before);
        eprintln!("fold grew the peak by {grew} KiB");
        assert!(grew < 512 * 1024, "{}: the peak memory grew by {grew} KiB", engine.backend());
    });
    eprintln!("ran on: {ran:?}");
}
