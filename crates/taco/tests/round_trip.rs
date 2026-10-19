//! Round trips of every input class of plan §11.2 through `fold` and `unfold`, with the tiny
//! model of `common`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Tests fail loudly.
#![allow(clippy::print_stderr, clippy::cast_possible_truncation)] // Measurements, test data.

mod common;

use common::{ARABIC, CHINESE, CODE, ENGLISH, HINDI, JSON, LOGS, Rng, mixed, random_letters, tiny};
use proptest::prelude::*;
use taco::{BlockSize, FoldOptions, Folded, Overlap, Threads, UnfoldOptions, fold, unfold};
use taco_format::{HEADER_LEN, TABLE_ENTRY_LEN, worst_case_len};

fn options(block_size: u32) -> FoldOptions {
    FoldOptions::new().with_block_size(BlockSize::new(block_size).unwrap())
}

fn fold_with(data: &[u8], block_size: u32) -> Folded {
    let t = tiny();
    fold(&t.model, &t.tokenizer, data, &options(block_size)).unwrap()
}

/// Folds and unfolds `data`, checks the output and the growth bound, and gives the result.
fn round_trip(data: &[u8], block_size: u32) -> Folded {
    let t = tiny();
    let folded = fold_with(data, block_size);
    let restored = unfold(Some(&t.model), &folded.container, &UnfoldOptions::new()).unwrap();
    assert_eq!(restored, data, "block size {block_size}");
    let bound = worst_case_len(data.len() as u64, BlockSize::new(block_size).unwrap()).unwrap();
    assert!(folded.container.len() as u64 <= bound);
    assert_eq!(folded.stats.container_len, folded.container.len() as u64);
    folded
}

/// Each class with the default block size and with small blocks (many blocks and restarts).
fn check_class(data: &[u8]) {
    for bs in [16 * 1024, 1000, 64, 7, 1] {
        round_trip(data, bs);
    }
}

#[test]
fn empty_input_is_the_header_only() {
    let f = round_trip(b"", 16 * 1024);
    assert_eq!(f.container.len(), HEADER_LEN);
    assert_eq!(f.stats.blocks, 0);
}

#[test]
fn one_byte_round_trips_raw() {
    for b in [0u8, b'a', 0xFF] {
        let f = round_trip(&[b], 16 * 1024);
        assert_eq!(f.stats.coded_blocks, 0);
    }
}

#[test]
fn english_round_trips_and_codes_smaller() {
    let data = ENGLISH.repeat(8);
    check_class(data.as_bytes());
    // The text is the same paragraph 8 times, so zstd wins over the random tiny model.
    let f = round_trip(data.as_bytes(), 16 * 1024);
    assert_eq!((f.stats.coded_blocks, f.stats.zstd_blocks), (0, 1));
    assert!(f.container.len() < data.len(), "{} >= {}", f.container.len(), data.len());
    // One paragraph has no repeats for zstd to find, and the model codes it smaller.
    let one = round_trip(ENGLISH.as_bytes(), 16 * 1024);
    assert_eq!((one.stats.coded_blocks, one.stats.zstd_blocks), (1, 0));
    assert!(one.container.len() < ENGLISH.len(), "{} >= {}", one.container.len(), ENGLISH.len());
    eprintln!(
        "English: {} bytes -> {} bytes (zstd), {} bytes -> {} bytes (coded), {} tokens",
        data.len(),
        f.container.len(),
        ENGLISH.len(),
        one.container.len(),
        one.stats.tokens
    );
}

#[test]
fn other_languages_round_trip() {
    for s in [HINDI, CHINESE, ARABIC] {
        check_class(s.repeat(3).as_bytes());
    }
}

#[test]
fn code_json_and_logs_round_trip() {
    for s in [CODE, JSON, LOGS] {
        check_class(s.repeat(3).as_bytes());
    }
}

#[test]
fn repeated_text_round_trips() {
    check_class(b"dasdasdasdada");
    check_class("dasdasdasdada".repeat(100).as_bytes());
}

#[test]
fn random_letters_round_trip() {
    check_class(&random_letters(5, 3000));
}

#[test]
fn random_bytes_are_stored_raw() {
    let data = Rng::new(9).bytes(40_000);
    let f = round_trip(&data, 16 * 1024);
    assert_eq!((f.stats.coded_blocks, f.stats.zstd_blocks), (0, 0));
    // Plan §3 rule 6: the growth is the header and 5 bytes for each block only.
    assert_eq!(f.container.len(), HEADER_LEN + TABLE_ENTRY_LEN * 3 + data.len());
    check_class(&data[..3000]);
}

/// Binary data with a structure, like the cells of a spreadsheet file: records of 16 bytes
/// with a counter, a small value, a type code and zero padding.
fn records(n: u32) -> Vec<u8> {
    let mut rng = Rng::new(11);
    let mut out = Vec::new();
    for i in 0..n {
        out.extend_from_slice(&i.to_le_bytes());
        out.extend_from_slice(&((rng.next_u64() % 8) as u32).to_le_bytes());
        out.extend_from_slice(&[0x02, 0x01, 0xFE, 0x00, 0, 0, 0, 0]);
    }
    out
}

#[test]
fn structured_binary_is_stored_as_zstd() {
    // ADR 0011: the model does not compress binary data, zstd does.
    let data = records(2048);
    let f = round_trip(&data, 16 * 1024);
    assert_eq!((f.stats.blocks, f.stats.zstd_blocks, f.stats.coded_blocks), (2, 2, 0));
    assert!(f.container.len() < data.len() / 4, "{} bytes", f.container.len());
    let parsed = taco_format::Container::parse(&f.container).unwrap();
    assert!(parsed.blocks().iter().all(|b| b.kind() == taco_format::BlockKind::Zstd));
    eprintln!("records: {} bytes -> {} bytes", data.len(), f.container.len());
    check_class(&data[..3000]);
}

#[test]
fn all_256_byte_values_round_trip() {
    let all: Vec<u8> = (0..=255u8).collect();
    check_class(&all);
    let mut rev = all.clone();
    rev.reverse();
    check_class(&[all, rev].concat());
}

#[test]
fn invalid_utf8_round_trips() {
    let mut data = ENGLISH.as_bytes().to_vec();
    for (i, b) in [(3, 0xFF), (40, 0xC3), (41, 0x28), (100, 0xED), (101, 0xA0), (102, 0x80)] {
        data.insert(i, b);
    }
    check_class(&data);
    // A UTF-8 character cut by a block boundary.
    let s = CHINESE.repeat(4);
    for bs in [2, 4, 5, 100] {
        round_trip(s.as_bytes(), bs);
    }
}

#[test]
fn large_mixed_file_round_trips() {
    let start = std::time::Instant::now();
    let data = mixed(256 * 1024);
    // PERF: the debug build of HardShell takes about 300 us for each step of the tiny model
    // (measured: 126 s for this test with one thread; 3.8 s in a release build). Thus the test
    // uses all threads of the machine. The output does not depend on the thread count.
    let t = tiny();
    let threads = Threads::available();
    let f = fold(&t.model, &t.tokenizer, &data, &options(16 * 1024).with_threads(threads)).unwrap();
    let uopts = UnfoldOptions::new().with_threads(threads);
    assert_eq!(unfold(Some(&t.model), &f.container, &uopts).unwrap(), data);
    // The text parts repeat, so zstd or the model makes them smaller; the noise stays raw.
    let smaller = f.stats.coded_blocks + f.stats.zstd_blocks;
    assert!(smaller > 0 && smaller < f.stats.blocks, "{:?}", f.stats);
    let bound = worst_case_len(data.len() as u64, BlockSize::new(16 * 1024).unwrap()).unwrap();
    assert!(f.container.len() as u64 <= bound);
    eprintln!(
        "mixed: {} bytes -> {} bytes, {} coded and {} zstd of {} blocks, {} tokens, {:?}",
        data.len(),
        f.container.len(),
        f.stats.coded_blocks,
        f.stats.zstd_blocks,
        f.stats.blocks,
        f.stats.tokens,
        start.elapsed()
    );
}

#[test]
fn every_input_class_round_trips_with_an_overlap() {
    // ADR 0015: the profile text with K > 0. The tiny model has a context of 24, so each block
    // of more than 23 tokens restarts, and K = 22 is the largest valid overlap.
    let t = tiny();
    let classes: [&[u8]; 6] = [
        ENGLISH.as_bytes(),
        CHINESE.as_bytes(),
        CODE.as_bytes(),
        LOGS.as_bytes(),
        &Rng::new(8).bytes(500),
        &random_letters(9, 700),
    ];
    for data in classes {
        for k in [1, 5, 12, 22] {
            for bs in [16 * 1024, 300, 7] {
                let opts = options(bs).with_overlap(Overlap::new(k));
                let f = fold(&t.model, &t.tokenizer, data, &opts).unwrap();
                let back = unfold(Some(&t.model), &f.container, &UnfoldOptions::new()).unwrap();
                assert_eq!(back, data, "K = {k}, blocks of {bs}");
            }
        }
    }
}

#[test]
fn output_is_the_same_for_one_and_many_threads() {
    let t = tiny();
    let data = mixed(12 * 1024);
    let one = fold(&t.model, &t.tokenizer, &data, &options(1024)).unwrap();
    assert_eq!(one.stats.blocks, 12);
    for n in [2, 3, 8] {
        let opts = options(1024).with_threads(Threads::new(n).unwrap());
        let many = fold(&t.model, &t.tokenizer, &data, &opts).unwrap();
        assert_eq!(many, one, "{n} threads");
        let uopts = UnfoldOptions::new().with_threads(Threads::new(n).unwrap());
        assert_eq!(unfold(Some(&t.model), &one.container, &uopts).unwrap(), data);
    }
    // The same with an overlap (ADR 0015): the batch lanes of unfold restart independently.
    let k = Overlap::new(12);
    let one = fold(&t.model, &t.tokenizer, &data, &options(1024).with_overlap(k)).unwrap();
    for n in [2, 8] {
        let opts = options(1024).with_threads(Threads::new(n).unwrap()).with_overlap(k);
        assert_eq!(fold(&t.model, &t.tokenizer, &data, &opts).unwrap(), one, "{n} threads, K");
        let uopts = UnfoldOptions::new().with_threads(Threads::new(n).unwrap());
        assert_eq!(unfold(Some(&t.model), &one.container, &uopts).unwrap(), data);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn fuzz_bytes_round_trip(
        data in proptest::collection::vec(any::<u8>(), 0..400),
        bs in 1u32..300,
    ) {
        round_trip(&data, bs);
    }

    #[test]
    fn fuzz_text_round_trips(s in "[a-z ,.'\n]{0,300}|\\PC{0,120}", bs in 1u32..200) {
        round_trip(s.as_bytes(), bs);
    }
}
