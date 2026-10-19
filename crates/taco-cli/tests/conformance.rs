//! The conformance vectors of container version 2 (plan §11, ADR 0011 addendum "Version 2 is
//! frozen"): small inputs folded with `conformance/tiny.hsm`, for the profiles `text` and
//! `logs`, with and without the context overlap, and with raw, zstd, coded and mixed blocks.
//!
//! `conformance/v2/vectors.tsv` lists each vector. For each one, this test checks the SHA-256
//! of the committed input and container, folds the input again and compares the bytes (with
//! `taco::fold` and with the stream fold), and unfolds the committed container (with
//! `taco::unfold` and both stream readers).
//!
//! To make the files again (only when the lead accepts a format change):
//! `TACO_WRITE_CONFORMANCE=1 cargo test -p taco-cli --test conformance -- --ignored`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Tests fail loudly.
#![allow(clippy::cast_possible_truncation)] // Test sizes are small.

#[path = "../../taco/tests/common/mod.rs"]
mod common;

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use common::{CODE, ENGLISH, JSON, LOGS, Rng};

use taco::stream::{StreamFold, StreamUnfold};
use taco::{
    BlockSize, Engine, FoldOptions, FoldProfile, Model, Overlap, TokenId, Tokenizer, UnfoldOptions,
    fold, unfold,
};

/// The SHA-256 of `conformance/tiny.hsm` (conformance/README.md).
const TINY_HSM_SHA256: &str = "eb78b89a4f0f17c7245865aed22f86512a49e0272b7583a831c47e1a401418f8";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance")
}

fn hex(bytes: &[u8]) -> String {
    taco::Checksum::of(bytes).to_string()
}

struct TinyHsm {
    model: Model,
    tokenizer: Tokenizer,
}

fn tiny_hsm() -> &'static TinyHsm {
    static T: OnceLock<TinyHsm> = OnceLock::new();
    T.get_or_init(|| {
        let file = std::fs::read(root().join("tiny.hsm")).unwrap();
        assert_eq!(hex(&file), TINY_HSM_SHA256, "conformance/tiny.hsm changed");
        let model = Model::load(&file).unwrap();
        let merges: Vec<(TokenId, TokenId)> = model
            .vocabulary()
            .merges()
            .unwrap_or(&[])
            .iter()
            .map(|m| (m.left(), m.right()))
            .collect();
        let tokenizer =
            Tokenizer::for_vocabulary(model.vocabulary(), model.eot_token(), &merges).unwrap();
        TinyHsm { model, tokenizer }
    })
}

/// One vector: its name, the fold options and the input.
struct Spec {
    name: &'static str,
    profile: FoldProfile,
    block_size: Option<u32>,
    overlap: Option<u16>,
    input: fn() -> Vec<u8>,
}

fn english() -> Vec<u8> {
    ENGLISH.repeat(4).into_bytes()
}

fn logs() -> Vec<u8> {
    LOGS.repeat(8).into_bytes()
}

fn kinds_mix() -> Vec<u8> {
    let mut d = ENGLISH.repeat(2).into_bytes();
    d.extend(Rng::new(5).bytes(300));
    d.extend_from_slice(&b"abc".repeat(120));
    d.extend_from_slice(CODE.as_bytes());
    d.extend_from_slice(JSON.as_bytes());
    d
}

/// Bytes that the tiny model predicts well: at each step, a token from those whose weight is
/// at least `1 / spread` of the largest weight, picked with a seeded integer generator. The
/// weights of `conformance/tiny.hsm` are random, so English text codes larger than zstd;
/// this input gives coded and mixed blocks. Each block of `block` bytes starts the model
/// again, as the fold does. DETERMINISM: integers only.
fn predicted(seed: u64, len: usize, spread: u64) -> Vec<u8> {
    let t = tiny_hsm();
    let vocab = t.model.vocabulary();
    let eot = t.model.eot_token();
    let mut rng = Rng::new(seed);
    let mut session = hardshell::gpt2::Session::new(&t.model);
    let mut out = Vec::new();
    let mut weights: Vec<u64> = session.start_block().unwrap().to_vec();
    let mut steps = 0;
    while out.len() < len {
        let max = weights.iter().copied().max().unwrap_or(0);
        let good: Vec<u32> = (0..weights.len() as u32)
            .filter(|&i| {
                let id = TokenId::new(i);
                id != eot
                    && weights[i as usize].saturating_mul(spread) >= max
                    && vocab.token(id).is_some_and(|b| !b.is_empty())
            })
            .collect();
        let pick = TokenId::new(good[(rng.next_u64() % good.len() as u64) as usize]);
        out.extend_from_slice(vocab.token(pick).unwrap());
        steps += 1;
        // The context of the model is 64 tokens: start again before it is full.
        weights = if steps % 60 == 0 {
            session.reset();
            session.start_block().unwrap().to_vec()
        } else {
            session.step(pick).unwrap().to_vec()
        };
    }
    out.truncate(len);
    out
}

fn predicted_text() -> Vec<u8> {
    predicted(21, 1200, 16)
}

/// Predicted bytes with repeated lines: the match model of the profile `logs` helps.
fn predicted_logs() -> Vec<u8> {
    let line = predicted(22, 90, 2);
    let mut d = Vec::new();
    for i in 0..14u8 {
        d.extend_from_slice(&line);
        d.extend_from_slice(format!(" id={i}\n").as_bytes());
    }
    d
}

/// One input with every block kind in one container: predicted bytes (coded), random bytes
/// (raw), a repeated pattern (zstd), and predicted lines (mixed, with the profile `logs`).
fn all_kinds() -> Vec<u8> {
    let mut d = predicted(23, 600, 16);
    d.extend(Rng::new(5).bytes(300));
    d.extend_from_slice(&b"abc".repeat(100));
    d.extend(predicted_logs());
    d
}

fn auto_mix() -> Vec<u8> {
    let mut d = LOGS.repeat(4).into_bytes();
    d.extend_from_slice(ENGLISH.repeat(2).as_bytes());
    d
}

/// The vectors. A change here changes `conformance/v2/vectors.tsv`.
const SPECS: &[Spec] = &[
    Spec {
        name: "empty",
        profile: FoldProfile::Text,
        block_size: None,
        overlap: None,
        input: Vec::new,
    },
    Spec {
        name: "one_byte",
        profile: FoldProfile::Text,
        block_size: None,
        overlap: None,
        input: || b"x".to_vec(),
    },
    Spec {
        name: "random_raw",
        profile: FoldProfile::Text,
        block_size: Some(256),
        overlap: None,
        input: || Rng::new(11).bytes(1000),
    },
    Spec {
        name: "text_k0",
        profile: FoldProfile::Text,
        block_size: Some(96),
        overlap: Some(0),
        input: english,
    },
    Spec {
        name: "text_k20",
        profile: FoldProfile::Text,
        block_size: Some(96),
        overlap: Some(20),
        input: english,
    },
    Spec {
        name: "text_kinds",
        profile: FoldProfile::Text,
        block_size: Some(96),
        overlap: None,
        input: kinds_mix,
    },
    Spec {
        name: "logs_k0",
        profile: FoldProfile::Logs,
        block_size: Some(256),
        overlap: Some(0),
        input: logs,
    },
    Spec {
        name: "logs_default_k",
        profile: FoldProfile::Logs,
        block_size: Some(256),
        overlap: None,
        input: logs,
    },
    Spec {
        name: "logs_kinds",
        profile: FoldProfile::Logs,
        block_size: Some(96),
        overlap: None,
        input: kinds_mix,
    },
    Spec {
        name: "auto",
        profile: FoldProfile::Auto,
        block_size: Some(200),
        overlap: None,
        input: auto_mix,
    },
    Spec {
        name: "predicted_text_k0",
        profile: FoldProfile::Text,
        block_size: Some(300),
        overlap: Some(0),
        input: predicted_text,
    },
    Spec {
        name: "predicted_text_k32",
        profile: FoldProfile::Text,
        block_size: Some(300),
        overlap: Some(32),
        input: predicted_text,
    },
    Spec {
        name: "predicted_logs_k0",
        profile: FoldProfile::Logs,
        block_size: Some(400),
        overlap: Some(0),
        input: predicted_logs,
    },
    Spec {
        name: "predicted_logs_default_k",
        profile: FoldProfile::Logs,
        block_size: Some(400),
        overlap: None,
        input: predicted_logs,
    },
    Spec {
        name: "predicted_auto",
        profile: FoldProfile::Auto,
        block_size: Some(400),
        overlap: None,
        input: predicted_logs,
    },
    Spec {
        name: "all_kinds_text",
        profile: FoldProfile::Text,
        block_size: Some(300),
        overlap: Some(16),
        input: all_kinds,
    },
    Spec {
        name: "all_kinds_logs",
        profile: FoldProfile::Logs,
        block_size: Some(300),
        overlap: None,
        input: all_kinds,
    },
];

fn options(spec: &Spec) -> FoldOptions {
    let mut o = FoldOptions::new().with_profile(spec.profile);
    if let Some(b) = spec.block_size {
        o = o.with_block_size(BlockSize::new(b).unwrap());
    }
    if let Some(k) = spec.overlap {
        o = o.with_overlap(Overlap::new(k));
    }
    o
}

fn profile_name(p: FoldProfile) -> &'static str {
    match p {
        FoldProfile::Text => "text",
        FoldProfile::Logs => "logs",
        _ => "auto",
    }
}

/// The kinds of the blocks of a container, as text: "raw,zstd,coded".
fn kinds_of(container: &[u8]) -> String {
    let parsed = taco_format_parse(container);
    if parsed.is_empty() { "-".to_owned() } else { parsed.join(",") }
}

/// The block kinds, read from the block table (version 2: the table starts at byte 104).
fn taco_format_parse(container: &[u8]) -> Vec<&'static str> {
    let count = u32::from_le_bytes(container[84..88].try_into().unwrap()) as usize;
    (0..count)
        .map(|i| match container[104 + 5 * i] {
            0 => "raw",
            1 => "coded",
            2 => "zstd",
            3 => "mixed",
            _ => "unknown",
        })
        .collect()
}

/// One line of `vectors.tsv`.
fn line(spec: &Spec, input: &[u8], container: &[u8]) -> String {
    let opt = |v: Option<String>| v.unwrap_or_else(|| "default".to_owned());
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        spec.name,
        profile_name(spec.profile),
        opt(spec.block_size.map(|b| b.to_string())),
        opt(spec.overlap.map(|k| k.to_string())),
        input.len(),
        hex(input),
        container.len(),
        hex(container),
        kinds_of(container)
    )
}

const HEADER: &str = "# name\tprofile\tblock_size\toverlap\tinput_len\tinput_sha256\tcontainer_len\tcontainer_sha256\tblock_kinds";

#[test]
#[ignore = "writes conformance/v2; run only when the lead accepts new vectors"]
fn write_the_vectors() {
    if std::env::var_os("TACO_WRITE_CONFORMANCE").is_none() {
        return;
    }
    let t = tiny_hsm();
    let dir = root().join("v2");
    std::fs::create_dir_all(&dir).unwrap();
    let mut lines = vec![HEADER.to_owned()];
    for spec in SPECS {
        let input = (spec.input)();
        let c = fold(&t.model, &t.tokenizer, &input, &options(spec)).unwrap().container;
        std::fs::write(dir.join(format!("{}.in", spec.name)), &input).unwrap();
        std::fs::write(dir.join(format!("{}.taco", spec.name)), &c).unwrap();
        let l = line(spec, &input, &c);
        eprintln!("{l}");
        lines.push(l);
    }
    lines.push(String::new());
    std::fs::write(dir.join("vectors.tsv"), lines.join("\n")).unwrap();
}

#[test]
fn every_vector_folds_to_its_bytes_and_unfolds_to_its_input() {
    let t = tiny_hsm();
    let engine = Engine::cpu();
    let dir = root().join("v2");
    let manifest = std::fs::read_to_string(dir.join("vectors.tsv")).unwrap();
    let rows: Vec<&str> =
        manifest.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).collect();
    assert_eq!(rows.len(), SPECS.len(), "vectors.tsv and SPECS differ");
    let mut all_kinds = Vec::new();
    for (row, spec) in rows.iter().zip(SPECS) {
        let input = std::fs::read(dir.join(format!("{}.in", spec.name))).unwrap();
        let container = std::fs::read(dir.join(format!("{}.taco", spec.name))).unwrap();
        // CONFORMANCE: the committed files and the manifest agree.
        assert_eq!(*row, line(spec, &input, &container), "vector {}", spec.name);
        assert_eq!(input, (spec.input)(), "the input of {} changed", spec.name);

        // fold gives the same bytes, in memory and as a stream.
        let o = options(spec);
        let folded = fold(&t.model, &t.tokenizer, &input, &o).unwrap().container;
        assert_eq!(hex(&folded), hex(&container), "fold of {}", spec.name);
        let mut streamed = Vec::new();
        StreamFold::new(&engine, &t.model, &t.tokenizer, &o)
            .to_writer(&input[..], None, &mut streamed, &std::env::temp_dir())
            .unwrap();
        assert_eq!(hex(&streamed), hex(&container), "stream fold of {}", spec.name);

        // unfold restores the input, in memory and from both stream readers.
        let u = UnfoldOptions::new();
        assert_eq!(unfold(Some(&t.model), &container, &u).unwrap(), input, "{}", spec.name);
        let s = StreamUnfold::new(&engine, Some(&t.model), &u);
        let mut out = Vec::new();
        s.from_reader(&container[..], &mut out).unwrap();
        assert_eq!(out, input, "stream unfold of {}", spec.name);
        let mut out = Vec::new();
        s.from_seekable(Cursor::new(&container), &mut out).unwrap();
        assert_eq!(out, input, "stream unfold (file) of {}", spec.name);
        all_kinds.extend(taco_format_parse(&container));
    }
    // The vectors hold every block kind of version 2.
    for kind in ["raw", "zstd", "coded", "mixed"] {
        assert!(all_kinds.contains(&kind), "no {kind} block in the vectors");
    }
}

/// The version 1 vector (`conformance/taco-v1-tiny.taco`) still unfolds, in memory and as a
/// stream (plan §3 rule 8). Its model is the test model of `crates/taco/tests/common`.
#[test]
fn the_version_1_vector_still_unfolds() {
    const V1: &[u8] =
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../conformance/taco-v1-tiny.taco"));
    assert_eq!(hex(V1), "c3ff718b1c9682589836078b36d936836077744b03b1cc8af3a6ced0d5821b8a");
    let t = common::tiny();
    let restored = unfold(Some(&t.model), V1, &UnfoldOptions::new()).unwrap();
    let mut golden = ENGLISH.repeat(3).into_bytes();
    golden.extend_from_slice(common::CHINESE.as_bytes());
    golden.extend(Rng::new(3).bytes(700));
    golden.extend_from_slice(ENGLISH.as_bytes());
    assert_eq!(restored, golden);
    let engine = Engine::cpu();
    let mut out = Vec::new();
    let s = StreamUnfold::new(&engine, Some(&t.model), &UnfoldOptions::new())
        .from_reader(V1, &mut out)
        .unwrap();
    assert_eq!((s.version, out), (1, golden));
}
