//! The profile `logs` (ADR 0010, ADR 0012): round trips of every input class, the golden
//! container, the auto rule, threads, and damage. The tiny model of `common` is the model.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Tests fail loudly.
#![allow(clippy::print_stderr, clippy::cast_possible_truncation)] // Measurements, test data.

mod common;

use common::{ARABIC, CHINESE, CODE, ENGLISH, HINDI, JSON, LOGS, Rng, mixed, random_letters, tiny};
use proptest::prelude::*;
use sha2::{Digest, Sha256};
use taco::{
    BlockSize, FoldOptions, FoldProfile, Folded, Profile, TacoError, Threads, UnfoldOptions, fold,
    unfold,
};
use taco_format::{
    BlockKind, CodingParts, Container, HEADER_LEN, MixerRule, TABLE_ENTRY_LEN, worst_case_len,
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn options(block_size: u32, profile: FoldProfile) -> FoldOptions {
    FoldOptions::new().with_block_size(BlockSize::new(block_size).unwrap()).with_profile(profile)
}

fn fold_with(data: &[u8], block_size: u32, profile: FoldProfile) -> Folded {
    let t = tiny();
    fold(&t.model, &t.tokenizer, data, &options(block_size, profile)).unwrap()
}

fn kinds(container: &[u8]) -> Vec<BlockKind> {
    Container::parse(container).unwrap().blocks().iter().map(|b| b.kind()).collect()
}

/// Folds `data` with the profiles logs and auto, unfolds both, and checks the output, the
/// growth bound, the header and the relation to the profile text.
fn round_trip(data: &[u8], block_size: u32) -> Folded {
    let t = tiny();
    let text = fold_with(data, block_size, FoldProfile::Text);
    let logs = fold_with(data, block_size, FoldProfile::Logs);
    let auto = fold_with(data, block_size, FoldProfile::Auto);
    for f in [&logs, &auto] {
        let restored = unfold(Some(&t.model), &f.container, &UnfoldOptions::new()).unwrap();
        assert_eq!(restored, data, "block size {block_size}");
        let bs = BlockSize::new(block_size).unwrap();
        assert!(f.container.len() as u64 <= worst_case_len(data.len() as u64, bs).unwrap());
        assert_eq!(f.stats.container_len, f.container.len() as u64);
        // The text candidate of each block is the block of the profile text.
        assert_eq!(f.stats.text_container_len, text.stats.container_len);
    }
    // A block is never larger with the profile logs.
    assert!(logs.container.len() <= text.container.len());
    let header = *Container::parse(&logs.container).unwrap().header();
    assert_eq!(header.profile(), Profile::Logs);
    assert_eq!(header.coding_parts(), CodingParts::with_mixer(MixerRule::V1));
    assert_eq!(logs.stats.profile, Profile::Logs);
    // Auto: the text container if no block is mixed, else the logs container.
    if logs.stats.mixed_blocks == 0 {
        assert_eq!(auto.container, text.container);
        assert_eq!(auto.stats.profile, Profile::Text);
    } else {
        assert_eq!(auto.container, logs.container);
        assert_eq!(auto.stats.profile, Profile::Logs);
    }
    assert_eq!(text.stats.text_container_len, text.stats.container_len);
    assert_eq!(text.stats.mixed_blocks, 0);
    logs
}

fn check_class(data: &[u8]) {
    for bs in [16 * 1024, 1000, 64, 7] {
        round_trip(data, bs);
    }
}

/// Log lines of a few templates with changing fields, `n` bytes.
fn log_lines(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    let mut ts = 203_615u64;
    while out.len() < n {
        ts += rng.next_u64() % 3;
        let line = match rng.next_u64() % 4 {
            0 => format!(
                "081109 {ts} INFO dfs.DataNode$PacketResponder: PacketResponder {} for block \
                 blk_{} terminating\n",
                rng.next_u64() % 3,
                rng.next_u64() % 1_000_000
            ),
            1 => format!(
                "081109 {ts} INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap \
                 updated: 10.250.{}.{}:50010 is added\n",
                rng.next_u64() % 20,
                rng.next_u64() % 256
            ),
            2 => format!(
                "081109 {ts} WARN dfs.DataNode: the disk is {}% full\n",
                rng.next_u64() % 100
            ),
            _ => format!(
                "081109 {ts} INFO dfs.DataNode$DataXceiver: Receiving block blk_{} src: /10.251.{}.{}\n",
                rng.next_u64() % 1_000_000,
                rng.next_u64() % 40,
                rng.next_u64() % 256
            ),
        };
        out.extend_from_slice(line.as_bytes());
    }
    out.truncate(n);
    out
}

/// English words of [`ENGLISH`] in a random order, `n` bytes. The tiny model codes words
/// better than single bytes, and the match model finds the repeated words, so the mixed form
/// wins in blocks of 512 bytes (measured: 512 bytes give 354 bytes with the profile logs and
/// 365 bytes with the profile text). zstd wins on most longer repeats with this random model.
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

#[test]
fn empty_input_is_the_header_only() {
    let f = round_trip(b"", 16 * 1024);
    assert_eq!(f.container.len(), HEADER_LEN);
}

#[test]
fn one_byte_round_trips() {
    for b in [0u8, b'a', 0xFF] {
        for bs in [1, 16 * 1024] {
            round_trip(&[b], bs);
        }
    }
}

#[test]
fn english_and_other_languages_round_trip() {
    check_class(ENGLISH.as_bytes());
    check_class(ENGLISH.repeat(4).as_bytes());
    for s in [HINDI, CHINESE, ARABIC] {
        check_class(s.repeat(3).as_bytes());
    }
}

#[test]
fn code_json_and_logs_round_trip() {
    for s in [CODE, JSON, LOGS] {
        check_class(s.repeat(3).as_bytes());
    }
    check_class(&log_lines(1, 6000));
    check_class(&shuffled_words(1, 3000));
}

#[test]
fn repeated_text_random_letters_and_bytes_round_trip() {
    check_class(b"dasdasdasdada");
    check_class("dasdasdasdada".repeat(100).as_bytes());
    check_class(&random_letters(5, 3000));
    check_class(&Rng::new(9).bytes(3000));
    let all: Vec<u8> = (0..=255u8).collect();
    check_class(&all);
    let mut data = ENGLISH.as_bytes().to_vec();
    for (i, b) in [(3, 0xFF), (40, 0xC3), (41, 0x28), (100, 0xED), (101, 0xA0), (102, 0x80)] {
        data.insert(i, b);
    }
    check_class(&data);
    round_trip(&data[..200], 1);
}

#[test]
fn logs_profile_makes_repeated_words_smaller_than_text() {
    let data = shuffled_words(2, 4000);
    let logs = round_trip(&data, 512);
    let text = fold_with(&data, 512, FoldProfile::Text);
    assert!(logs.stats.mixed_blocks > 0, "{:?}", logs.stats);
    assert!(logs.container.len() < text.container.len());
    eprintln!(
        "words: {} bytes -> text {} bytes, logs {} bytes ({:?})",
        data.len(),
        text.container.len(),
        logs.container.len(),
        kinds(&logs.container)
    );
}

#[test]
fn large_mixed_file_round_trips() {
    let data = [mixed(64 * 1024), log_lines(3, 32 * 1024), shuffled_words(3, 32 * 1024)].concat();
    let t = tiny();
    let threads = Threads::available();
    let opts = options(512, FoldProfile::Logs).with_threads(threads);
    let f = fold(&t.model, &t.tokenizer, &data, &opts).unwrap();
    let uopts = UnfoldOptions::new().with_threads(threads);
    assert_eq!(unfold(Some(&t.model), &f.container, &uopts).unwrap(), data);
    assert!(f.stats.mixed_blocks > 0, "{:?}", f.stats);
    eprintln!("mixed file: {:?}", f.stats);
}

/// The input of the logs golden vector, in blocks of 512: shuffled words (mixed), random
/// bytes (raw), repeated JSON (zstd), and 100 bytes of English (coded).
fn golden_input() -> Vec<u8> {
    let mut data = shuffled_words(9, 1024);
    data.extend(Rng::new(3).bytes(512));
    data.extend_from_slice(&JSON.repeat(4).as_bytes()[..512]);
    data.extend_from_slice(&ENGLISH.as_bytes()[..100]);
    data
}

#[test]
fn golden_logs_container_hash() {
    let t = tiny();
    let data = golden_input();
    let f = fold(&t.model, &t.tokenizer, &data, &options(512, FoldProfile::Logs)).unwrap();
    let k = kinds(&f.container);
    eprintln!(
        "logs golden: input {} bytes, container {} bytes, kinds {k:?}, sha256 {}",
        data.len(),
        f.container.len(),
        hex(&Sha256::digest(&f.container))
    );
    // The vector covers the four kinds of block of version 2.
    assert_eq!(
        k,
        [BlockKind::Mixed, BlockKind::Zstd, BlockKind::Raw, BlockKind::Zstd, BlockKind::Coded]
    );
    assert_eq!(f.stats.text_container_len, 1339);
    // CONFORMANCE: a change to these values changes the format, the model file, the coder or
    // the mixer rule (ADR 0010: a new mixer rule version).
    assert_eq!(f.container.len(), 1328, "container length");
    assert_eq!(
        hex(&Sha256::digest(&f.container)),
        "7262257e6d2875ae393dd2474acc8a00dd1b8adf829eeef75b0ca719f73b55ec"
    );
    for threads in [1, 3] {
        let opts = UnfoldOptions::new().with_threads(Threads::new(threads).unwrap());
        assert_eq!(unfold(Some(&t.model), &f.container, &opts).unwrap(), data);
    }
}

#[test]
fn output_is_the_same_for_one_and_many_threads() {
    let t = tiny();
    let data = [shuffled_words(4, 3 * 1024), mixed(3 * 1024)].concat();
    for profile in [FoldProfile::Logs, FoldProfile::Auto] {
        let one = fold(&t.model, &t.tokenizer, &data, &options(512, profile)).unwrap();
        assert_eq!(one.stats.blocks, 12);
        assert!(one.stats.mixed_blocks > 0 && one.stats.coded_blocks + one.stats.zstd_blocks > 0);
        for n in [2, 3, 8] {
            let opts = options(512, profile).with_threads(Threads::new(n).unwrap());
            let many = fold(&t.model, &t.tokenizer, &data, &opts).unwrap();
            assert_eq!(many, one, "{n} threads");
            let uopts = UnfoldOptions::new().with_threads(Threads::new(n).unwrap());
            assert_eq!(unfold(Some(&t.model), &one.container, &uopts).unwrap(), data);
        }
    }
}

#[test]
fn a_flipped_bit_never_gives_wrong_output() {
    let t = tiny();
    let data = shuffled_words(5, 1200);
    let f = fold_with(&data, 400, FoldProfile::Logs);
    assert!(f.stats.mixed_blocks > 0, "{:?}", f.stats);
    let parsed = Container::parse(&f.container).unwrap();
    let table_end = HEADER_LEN + TABLE_ENTRY_LEN * parsed.blocks().len();
    let mut errors = 0;
    for pos in 0..f.container.len() {
        let bit = pos % 8;
        let mut bad = f.container.clone();
        bad[pos] ^= 1 << bit;
        match unfold(Some(&t.model), &bad, &UnfoldOptions::new()) {
            Ok(out) => assert_eq!(out, data, "byte {pos} bit {bit}: wrong output"),
            Err(_) => errors += 1,
        }
    }
    eprintln!("flips: {errors} of {} gave an error; the others the original", f.container.len());
    assert!(errors >= table_end);
}

#[test]
fn damaged_mixed_blocks_and_header_fields_are_errors() {
    let t = tiny();
    let data = shuffled_words(9, 512);
    let f = fold_with(&data, 512, FoldProfile::Logs);
    assert_eq!(kinds(&f.container), [BlockKind::Mixed]);
    // The first range coder byte of the mixed block (a token count of 2 varint bytes).
    let mut bad = f.container.clone();
    bad[HEADER_LEN + TABLE_ENTRY_LEN + 2] ^= 0x40;
    let r = unfold(Some(&t.model), &bad, &UnfoldOptions::new());
    assert!(
        matches!(
            r,
            Err(TacoError::CorruptBlock { block: 0, .. } | TacoError::ChecksumMismatch { .. })
        ),
        "{r:?}"
    );
    // Without the coding part, the mixed block is refused; with an unknown rule, the header.
    for (pos, value) in [(90, 0), (92, 2), (92, 0), (88, 0), (90, 3)] {
        let mut bad = f.container.clone();
        bad[pos] = value;
        let r = unfold(Some(&t.model), &bad, &UnfoldOptions::new());
        assert!(matches!(r, Err(TacoError::Format(_))), "byte {pos} = {value}: {r:?}");
    }
    // A mixed block read as coded (kind 1) gives an error, not wrong output.
    let mut bad = f.container.clone();
    bad[HEADER_LEN] = 1;
    assert!(unfold(Some(&t.model), &bad, &UnfoldOptions::new()).is_err());
}

#[test]
fn a_text_container_of_a_coded_block_read_as_mixed_is_an_error() {
    // The coded payload of the profile text, with the header and kind of the profile logs.
    let t = tiny();
    let f = fold_with(ENGLISH.as_bytes(), 4096, FoldProfile::Text);
    assert_eq!(kinds(&f.container), [BlockKind::Coded]);
    let mut bad = f.container.clone();
    bad[88] = 1; // profile logs
    bad[90] = 1; // the match model and the mixer
    bad[92] = 1; // rule version 1
    bad[HEADER_LEN] = 3; // mixed
    assert!(unfold(Some(&t.model), &bad, &UnfoldOptions::new()).is_err());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn fuzz_bytes_round_trip(
        data in proptest::collection::vec(any::<u8>(), 0..300),
        bs in 1u32..300,
    ) {
        round_trip(&data, bs);
    }

    #[test]
    fn fuzz_text_round_trips(s in "[a-z ,.'\n]{0,300}|\\PC{0,120}", bs in 1u32..200) {
        round_trip(s.as_bytes(), bs);
    }
}
