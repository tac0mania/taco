//! The stream fold and the stream unfold of `taco::stream` (ADR 0018) give the bytes of
//! `taco::fold` and `taco::unfold`, with both output methods, over many segments and table
//! groups, and refuse damaged input.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Tests fail loudly.
#![allow(clippy::cast_possible_truncation)]

#[path = "../../taco/tests/common/mod.rs"]
mod common;

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use common::{CODE, ENGLISH, LOGS, Rng, mixed, tiny};
use taco::stream::{
    MAX_BLOCK_SIZE, Progress, StreamError, StreamFold, StreamOptions, StreamUnfold, segment_blocks,
};
use taco::{
    BlockSize, Engine, FoldOptions, FoldProfile, FormatError, Overlap, TacoError, Threads,
    UnfoldOptions, fold, unfold,
};

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn bs(n: u32) -> BlockSize {
    BlockSize::new(n).unwrap()
}

/// Folds `data` with both stream methods, checks that both give the container of
/// `taco::fold`, and gives it.
fn stream_fold_matches(data: &[u8], options: &FoldOptions, spool: &Path) -> Vec<u8> {
    let t = tiny();
    let engine = Engine::cpu();
    let expected = fold(&t.model, &t.tokenizer, data, options).unwrap();
    let f = StreamFold::new(&engine, &t.model, &t.tokenizer, options);
    let mut seek = Cursor::new(Vec::new());
    let stats = f.to_seekable(data, data.len() as u64, &mut seek).unwrap();
    assert_eq!(seek.into_inner(), expected.container, "seek back");
    assert_eq!(stats, expected.stats);
    for len in [Some(data.len() as u64), None] {
        let mut out = Vec::new();
        let stats = f.to_writer(data, len, &mut out, spool).unwrap();
        assert_eq!(out, expected.container, "spool, length {len:?}");
        assert_eq!(stats, expected.stats);
    }
    // The spool deletes its temporary files.
    assert_eq!(std::fs::read_dir(spool).unwrap().count(), 0);
    expected.container
}

/// Unfolds `container` with both stream readers and checks the restored data.
fn stream_unfold_matches(container: &[u8], data: &[u8]) {
    let t = tiny();
    let engine = Engine::cpu();
    for threads in [1, 3] {
        let options = UnfoldOptions::new().with_threads(Threads::new(threads).unwrap());
        let u = StreamUnfold::new(&engine, Some(&t.model), &options);
        let mut out = Vec::new();
        let s = u.from_seekable(Cursor::new(container), &mut out).unwrap();
        assert_eq!(out, data, "from_seekable");
        assert_eq!((s.original_len, s.container_len), (data.len() as u64, container.len() as u64));
        let mut out = Vec::new();
        u.from_reader(container, &mut out).unwrap();
        assert_eq!(out, data, "from_reader");
    }
}

#[test]
fn stream_fold_gives_the_bytes_of_fold_for_every_profile() {
    let spool = dir("stream_profiles");
    let mut inputs: Vec<Vec<u8>> = vec![
        Vec::new(),
        b"x".to_vec(),
        ENGLISH.repeat(5).into_bytes(),
        LOGS.repeat(6).into_bytes(),
        CODE.repeat(3).into_bytes(),
        Rng::new(9).bytes(2000),
    ];
    inputs.push(mixed(5000));
    for data in &inputs {
        for profile in [FoldProfile::Text, FoldProfile::Logs, FoldProfile::Auto] {
            for block in [64, 500] {
                let options = FoldOptions::new()
                    .with_profile(profile)
                    .with_block_size(bs(block))
                    .with_threads(Threads::new(2).unwrap());
                let c = stream_fold_matches(data, &options, &spool);
                stream_unfold_matches(&c, data);
            }
        }
    }
}

#[test]
fn stream_fold_over_many_segments_and_table_groups() {
    // Blocks of 7 bytes: 256 blocks in each segment, and more than 4,096 table entries.
    assert_eq!(segment_blocks(bs(7)), 256);
    let spool = dir("stream_segments");
    let data = ENGLISH.repeat(140).into_bytes();
    let blocks = data.len().div_ceil(7);
    assert!(blocks > 4096 + 256, "{blocks} blocks");
    let options = FoldOptions::new()
        .with_block_size(bs(7))
        .with_overlap(Overlap::new(3))
        .with_threads(Threads::new(4).unwrap());
    let c = stream_fold_matches(&data, &options, &spool);
    stream_unfold_matches(&c, &data);
}

#[test]
fn segments_hold_at_most_64_mib_or_one_block() {
    assert_eq!(segment_blocks(bs(16 * 1024)), 256);
    assert_eq!(segment_blocks(bs(1024 * 1024)), 64);
    assert_eq!(segment_blocks(MAX_BLOCK_SIZE), 1);
    assert_eq!(segment_blocks(bs(u32::MAX)), 1);
}

#[test]
fn progress_reaches_the_input_length() {
    let t = tiny();
    let engine = Engine::cpu();
    let data = ENGLISH.repeat(3).into_bytes();
    let options = FoldOptions::new().with_block_size(bs(100));
    let (input, output, calls) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let progress = |p: Progress| {
        input.fetch_max(p.input_bytes, Ordering::Relaxed);
        output.fetch_max(p.output_bytes, Ordering::Relaxed);
        calls.fetch_add(1, Ordering::Relaxed);
    };
    let f = StreamFold::new(&engine, &t.model, &t.tokenizer, &options).with_progress(&progress);
    let mut out = Cursor::new(Vec::new());
    let stats = f.to_seekable(&data[..], data.len() as u64, &mut out).unwrap();
    assert_eq!(input.load(Ordering::Relaxed), data.len() as u64);
    assert_eq!(output.load(Ordering::Relaxed), stats.container_len);
    assert!(calls.load(Ordering::Relaxed) > stats.blocks, "one call for each block and one");
}

#[test]
fn stream_fold_refuses_an_input_of_another_length() {
    let t = tiny();
    let engine = Engine::cpu();
    let data = ENGLISH.as_bytes();
    let options = FoldOptions::new().with_block_size(bs(64));
    let f = StreamFold::new(&engine, &t.model, &t.tokenizer, &options);
    for claimed in [data.len() as u64 - 1, data.len() as u64 + 1] {
        let mut out = Cursor::new(Vec::new());
        match f.to_seekable(data, claimed, &mut out) {
            Err(StreamError::Taco(TacoError::Format(FormatError::InputLengthMismatch {
                expected,
                found,
            }))) => assert_eq!(expected, claimed, "found {found}"),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn the_block_size_limit_applies_to_fold_and_unfold() {
    let t = tiny();
    let engine = Engine::cpu();
    let data = ENGLISH.as_bytes();
    let options = FoldOptions::new().with_block_size(bs(200));
    let limits = StreamOptions::new().with_max_block_size(bs(100));
    let f = StreamFold::new(&engine, &t.model, &t.tokenizer, &options).with_limits(limits);
    let mut out = Vec::new();
    let r = f.to_writer(data, None, &mut out, &dir("stream_limit"));
    assert!(matches!(r, Err(StreamError::BlockSizeLimit { block_size: 200, limit: 100 })), "{r:?}");
    let big = FoldOptions::new().with_block_size(bs(MAX_BLOCK_SIZE.get() + 1));
    let f = StreamFold::new(&engine, &t.model, &t.tokenizer, &big);
    assert!(matches!(
        f.to_writer(data, None, &mut out, &dir("stream_limit2")),
        Err(StreamError::BlockSizeLimit { .. })
    ));

    let c = fold(&t.model, &t.tokenizer, data, &options).unwrap().container;
    let u = StreamUnfold::new(&engine, Some(&t.model), &UnfoldOptions::new()).with_limits(limits);
    let mut out = Vec::new();
    assert!(matches!(
        u.from_reader(&c[..], &mut out),
        Err(StreamError::BlockSizeLimit { block_size: 200, limit: 100 })
    ));
    assert!(out.is_empty());
}

#[test]
fn damaged_input_is_an_error_and_a_file_is_refused_before_any_output() {
    let t = tiny();
    let engine = Engine::cpu();
    let data = ENGLISH.repeat(4).into_bytes();
    let c = fold(&t.model, &t.tokenizer, &data, &FoldOptions::new().with_block_size(bs(100)))
        .unwrap()
        .container;
    let u = StreamUnfold::new(&engine, Some(&t.model), &UnfoldOptions::new());

    // A missing last byte and an extra byte.
    let short = &c[..c.len() - 1];
    let mut long = c.clone();
    long.push(0);
    for bad in [short, &long[..]] {
        let mut out = Vec::new();
        let r = u.from_seekable(Cursor::new(bad), &mut out);
        assert!(
            matches!(
                r,
                Err(StreamError::Taco(TacoError::Format(FormatError::WrongTotalLength { .. })))
            ),
            "{r:?}"
        );
        assert!(out.is_empty(), "a file is refused before any output");
        let mut out = Vec::new();
        let r = u.from_reader(bad, &mut out);
        assert!(
            matches!(
                r,
                Err(StreamError::Taco(TacoError::Format(FormatError::WrongTotalLength { .. })))
            ),
            "{r:?}"
        );
    }

    // A header that claims more blocks than the file holds.
    let mut claims = c.clone();
    claims[40..48].copy_from_slice(&u64::from(u32::MAX).to_le_bytes());
    claims[80..84].copy_from_slice(&1u32.to_le_bytes());
    claims[84..88].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut out = Vec::new();
    let no_limit = StreamOptions::new().with_max_block_size(bs(u32::MAX));
    let r = StreamUnfold::new(&engine, Some(&t.model), &UnfoldOptions::new())
        .with_limits(no_limit)
        .from_reader(&claims[..], &mut out);
    assert!(
        matches!(r, Err(StreamError::Taco(TacoError::Format(FormatError::Truncated { .. })))),
        "{r:?}"
    );

    // A damaged checksum: an error, never a silent wrong output.
    let mut sum = c.clone();
    sum[50] ^= 1;
    let mut out = Vec::new();
    let r = u.from_reader(&sum[..], &mut out);
    assert!(matches!(r, Err(StreamError::Taco(TacoError::ChecksumMismatch { .. }))), "{r:?}");

    // Truncated headers and a bad magic.
    for n in [0, 3, 5, 50, 103] {
        let mut out = Vec::new();
        assert!(u.from_reader(&c[..n], &mut out).is_err(), "{n} bytes");
        assert!(u.from_seekable(Cursor::new(&c[..n]), &mut out).is_err(), "{n} bytes");
    }
}

#[test]
fn the_version_1_container_still_unfolds_from_a_stream() {
    const V1: &[u8] =
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../conformance/taco-v1-tiny.taco"));
    let t = tiny();
    let expected = unfold(Some(&t.model), V1, &UnfoldOptions::new()).unwrap();
    stream_unfold_matches(V1, &expected);
    let engine = Engine::cpu();
    let s = StreamUnfold::new(&engine, Some(&t.model), &UnfoldOptions::new())
        .from_reader(V1, &mut Vec::new())
        .unwrap();
    assert_eq!(s.version, 1);
}

#[test]
fn a_raw_only_container_unfolds_without_a_model() {
    // A container in the raw-only mode: the zstd vector C of taco-format is made there; here
    // the empty container of version 2 (ADR 0011, 104 bytes) is enough.
    let mut c = Vec::from(&b"TACO\x02\x00\x00\x00"[..]);
    c.extend_from_slice(&[0; 32]);
    c.extend_from_slice(&0u64.to_le_bytes());
    c.extend_from_slice(taco::Checksum::of(&[]).as_bytes());
    c.extend_from_slice(&65_536u32.to_le_bytes());
    c.extend_from_slice(&0u32.to_le_bytes());
    c.extend_from_slice(&[0; 16]);
    let engine = Engine::cpu();
    let u = StreamUnfold::new(&engine, None, &UnfoldOptions::new());
    let mut out = Vec::new();
    u.from_reader(&c[..], &mut out).unwrap();
    assert!(out.is_empty());
    u.from_seekable(Cursor::new(&c), &mut out).unwrap();
}

/// The format reads mixer rule version 2 (ADR 0017), but this library does not decode it yet:
/// both unfold paths refuse it before the first block, never decode it with rule 1.
#[test]
fn a_mixer_rule_2_container_is_refused_by_this_decoder() {
    use taco::format::{Block, CodingParts, Container, ContainerParams, MixerRule};
    let t = tiny();
    let data = LOGS.as_bytes();
    let params = ContainerParams::new(taco::container_hash(&t.model), bs(1000))
        .with_profile(taco::Profile::Logs)
        .with_coding(CodingParts::with_mixer(MixerRule::V2));
    let blocks: Vec<Block> = bs(1000).split(data).map(|_| Block::Mixed(vec![1, 0])).collect();
    let c = Container::build(&params, data, &blocks).unwrap();
    assert_eq!(Container::parse(&c).unwrap().header().coding_parts().mixer(), Some(MixerRule::V2));
    let refused = TacoError::Format(FormatError::UnknownMixerRule { rule: 2 });
    assert_eq!(unfold(Some(&t.model), &c, &UnfoldOptions::new()), Err(refused.clone()));
    let engine = Engine::cpu();
    let mut out = Vec::new();
    match StreamUnfold::new(&engine, Some(&t.model), &UnfoldOptions::new())
        .from_reader(&c[..], &mut out)
    {
        Err(StreamError::Taco(e)) => assert_eq!(e, refused),
        other => panic!("{other:?}"),
    }
    assert!(out.is_empty());
}

/// The speed of the stream fold and unfold against `taco::fold` and `taco::unfold` (ADR 0018,
/// measurements). Run by hand with a model and an input:
/// `TACO_SPEED_MODEL=gpt2.hsm TACO_SPEED_INPUT=file TACO_SPEED_PROFILE=text
/// cargo test --release -p taco-cli --test stream -- --ignored --nocapture`.
#[test]
#[ignore = "a measurement; needs TACO_SPEED_MODEL and TACO_SPEED_INPUT"]
fn stream_speed_against_memory() {
    let (Some(m), Some(i)) =
        (std::env::var_os("TACO_SPEED_MODEL"), std::env::var_os("TACO_SPEED_INPUT"))
    else {
        return;
    };
    let file = std::fs::read(m).unwrap();
    let model = taco::Model::load(&file).unwrap();
    let merges: Vec<(taco::TokenId, taco::TokenId)> =
        model.vocabulary().merges().unwrap_or(&[]).iter().map(|m| (m.left(), m.right())).collect();
    let tokenizer =
        taco::Tokenizer::for_vocabulary(model.vocabulary(), model.eot_token(), &merges).unwrap();
    let data = std::fs::read(i).unwrap();
    let profile = match std::env::var("TACO_SPEED_PROFILE").as_deref() {
        Ok("logs") => FoldProfile::Logs,
        _ => FoldProfile::Text,
    };
    let threads = Threads::new(6).unwrap();
    let options = FoldOptions::new().with_profile(profile).with_threads(threads);
    let engine = Engine::cpu();
    let t0 = std::time::Instant::now();
    let a = fold(&model, &tokenizer, &data, &options).unwrap().container;
    let t1 = std::time::Instant::now();
    let mut b = Vec::new();
    StreamFold::new(&engine, &model, &tokenizer, &options)
        .to_writer(&data[..], None, &mut b, &std::env::temp_dir())
        .unwrap();
    let t2 = std::time::Instant::now();
    assert_eq!(a, b);
    let u = UnfoldOptions::new().with_threads(threads);
    let r = unfold(Some(&model), &a, &u).unwrap();
    let t3 = std::time::Instant::now();
    let mut s = Vec::new();
    StreamUnfold::new(&engine, Some(&model), &u).from_reader(&a[..], &mut s).unwrap();
    let t4 = std::time::Instant::now();
    assert_eq!((r.as_slice(), s.as_slice()), (&data[..], &data[..]));
    eprintln!(
        "speed: {} bytes, {} blocks; fold memory {:.1?}, stream {:.1?}; unfold memory {:.1?}, \
         stream {:.1?}",
        data.len(),
        data.len().div_ceil(options.effective_block_size().as_usize()),
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3
    );
}
