//! The safety rules of `unfold` (plan §3 rules 3, 4, 6) and the golden conformance vector.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Tests fail loudly.
#![allow(clippy::print_stderr, clippy::cast_possible_truncation)] // Measurements, test data.

mod common;

use common::{CHINESE, ENGLISH, Rng, merge_ids, model_file, tiny};
use sha2::{Digest, Sha256};
use taco::{
    BlockSize, FoldOptions, FoldProfile, Model, Overlap, TacoError, TokenId, Tokenizer,
    UnfoldOptions, container_hash, container_overlap, fold, unfold,
};
use taco_format::{BlockKind, Container, HEADER_LEN, HEADER_LEN_V1, Profile, VERSION, VERSION_1};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn options(block_size: u32) -> FoldOptions {
    FoldOptions::new().with_block_size(BlockSize::new(block_size).unwrap())
}

/// The input of the golden vector: English, Chinese and random bytes, in blocks of 512.
fn golden_input() -> Vec<u8> {
    let mut data = ENGLISH.repeat(3).into_bytes();
    data.extend_from_slice(CHINESE.as_bytes());
    data.extend(Rng::new(3).bytes(700));
    data.extend_from_slice(ENGLISH.as_bytes());
    data
}

#[test]
fn golden_container_hash() {
    let t = tiny();
    let data = golden_input();
    let f = fold(&t.model, &t.tokenizer, &data, &options(512)).unwrap();
    let parsed = Container::parse(&f.container).unwrap();
    let kinds: Vec<BlockKind> = parsed.blocks().iter().map(|b| b.kind()).collect();
    eprintln!(
        "golden: model {}, input {} bytes, container {} bytes, kinds {kinds:?}, sha256 {}",
        hex(&Sha256::digest(&t.file)),
        data.len(),
        f.container.len(),
        hex(&Sha256::digest(&f.container))
    );
    // The vector covers the three kinds of block of version 2 (ADR 0011).
    assert_eq!(
        kinds,
        [BlockKind::Zstd, BlockKind::Zstd, BlockKind::Raw, BlockKind::Zstd, BlockKind::Coded]
    );
    assert_eq!(parsed.header().version(), VERSION);
    assert_eq!(parsed.header().profile(), Profile::Text);
    assert_eq!((f.stats.coded_blocks, f.stats.zstd_blocks), (1, 3));
    // CONFORMANCE: a change to these values changes the format, the model file or the coder.
    assert_eq!(
        hex(&Sha256::digest(&t.file)),
        "5ba63e7c69063caa1430de0df5a9b3ed76db63de1a779d5ecce25484c88de4ff",
        "the tiny model file changed (common/mod.rs or the .hsm writer)"
    );
    // NOTE: the zstd frames depend on the zstd library version (ADR 0011 §6: zstd 1.5.7,
    // level 19). The version 1 vector of this input is `golden_v1_container_still_unfolds`.
    assert_eq!(f.container.len(), 1660, "container length");
    assert_eq!(
        hex(&Sha256::digest(&f.container)),
        "0b0ba3f1bc8579247476163489ed3155a7828916d27e92f3435e9a8e74157c07"
    );
    assert_eq!(unfold(Some(&t.model), &f.container, &UnfoldOptions::new()).unwrap(), data);
}

#[test]
fn golden_container_is_the_same_with_an_explicit_overlap_of_0() {
    // `--overlap 0` is the rule of ADR 0004 section 9: the golden vector of the profile text
    // does not change, and the container has no overlap bit.
    let t = tiny();
    let data = golden_input();
    let opts = options(512).with_overlap(Overlap::NONE);
    let f = fold(&t.model, &t.tokenizer, &data, &opts).unwrap();
    assert_eq!(f, fold(&t.model, &t.tokenizer, &data, &options(512)).unwrap());
    assert_eq!(
        hex(&Sha256::digest(&f.container)),
        "0b0ba3f1bc8579247476163489ed3155a7828916d27e92f3435e9a8e74157c07"
    );
    assert_eq!(&f.container[90..96], &[0; 6]);
    assert_eq!(container_overlap(&f.container).unwrap(), Overlap::NONE);
}

/// The overlap of the text golden vector with an overlap: half of the context of 24.
const GOLDEN_K: Overlap = Overlap::new(12);

#[test]
fn golden_overlap_container_hash() {
    // ADR 0015: the input of the golden vector, the profile text, `--overlap 12`.
    let t = tiny();
    let data = golden_input();
    let f = fold(&t.model, &t.tokenizer, &data, &options(512).with_overlap(GOLDEN_K)).unwrap();
    let parsed = Container::parse(&f.container).unwrap();
    let kinds: Vec<BlockKind> = parsed.blocks().iter().map(|b| b.kind()).collect();
    eprintln!(
        "golden, overlap 12: input {} bytes, container {} bytes, kinds {kinds:?}, sha256 {}",
        data.len(),
        f.container.len(),
        hex(&Sha256::digest(&f.container))
    );
    assert_eq!(parsed.header().profile(), Profile::Text);
    assert_eq!(&f.container[90..96], &[2, 0, 0, 0, 12, 0]);
    assert_eq!(container_overlap(&f.container).unwrap(), GOLDEN_K);
    assert_eq!(f.stats.overlap, GOLDEN_K);
    assert!(f.stats.coded_blocks > 0, "{kinds:?}");
    // CONFORMANCE: a change to these values changes the format, the model file, the coder or
    // the context rule (ADR 0015).
    assert_eq!(f.container.len(), 1660, "container length");
    assert_eq!(
        hex(&Sha256::digest(&f.container)),
        "d705d4b6971fe351f29d14cd5723d3b2a79ced03facae32910bc13d09f3ce340"
    );
    for threads in [1, 3] {
        let opts = UnfoldOptions::new().with_threads(taco::Threads::new(threads).unwrap());
        assert_eq!(unfold(Some(&t.model), &f.container, &opts).unwrap(), data);
    }
}

#[test]
fn the_overlap_changes_the_coded_bytes_after_the_first_restart_only() {
    // A block of fewer than n_ctx = 24 tokens never restarts, so K does not change it.
    let t = tiny();
    let short = &ENGLISH.as_bytes()[..20];
    let a = fold(&t.model, &t.tokenizer, short, &options(4096)).unwrap();
    let b = fold(&t.model, &t.tokenizer, short, &options(4096).with_overlap(GOLDEN_K)).unwrap();
    assert_eq!(a.container[104..], b.container[104..]);
    // A long block restarts, and the stored bytes differ.
    let long = ENGLISH.as_bytes();
    let a = fold(&t.model, &t.tokenizer, long, &options(4096)).unwrap();
    let b = fold(&t.model, &t.tokenizer, long, &options(4096).with_overlap(GOLDEN_K)).unwrap();
    assert_eq!(a.stats.coded_blocks, 1);
    assert_ne!(a.container[104..], b.container[104..]);
    eprintln!(
        "ENGLISH, one block: K = 0 {} bytes, K = 12 {} bytes",
        a.container.len(),
        b.container.len()
    );
}

#[test]
fn every_valid_overlap_round_trips_and_a_larger_one_is_refused() {
    let t = tiny();
    let data = [ENGLISH.as_bytes(), CHINESE.as_bytes()].concat();
    let n_ctx = u16::try_from(t.model.n_ctx()).unwrap();
    for k in 0..=n_ctx - 2 {
        for profile in [FoldProfile::Text, FoldProfile::Logs] {
            let opts = options(700).with_overlap(Overlap::new(k)).with_profile(profile);
            let f = fold(&t.model, &t.tokenizer, &data, &opts).unwrap();
            assert_eq!(container_overlap(&f.container).unwrap(), Overlap::new(k));
            let back = unfold(Some(&t.model), &f.container, &UnfoldOptions::new()).unwrap();
            assert_eq!(back, data, "K = {k}, {profile:?}");
        }
    }
    // Fold refuses K > n_ctx - 2 (ADR 0015 §1).
    let opts = options(700).with_overlap(Overlap::new(n_ctx - 1));
    assert_eq!(
        fold(&t.model, &t.tokenizer, &data, &opts).unwrap_err(),
        TacoError::Model(hardshell::Error::InvalidOverlap { overlap: n_ctx - 1, n_ctx: 24 })
    );
    // Unfold refuses it before a block decodes, also for a container that only claims it.
    let f = fold(&t.model, &t.tokenizer, &data, &options(700).with_overlap(GOLDEN_K)).unwrap();
    for bad_k in [n_ctx - 1, n_ctx, u16::MAX] {
        let mut bad = f.container.clone();
        bad[94..96].copy_from_slice(&bad_k.to_le_bytes());
        assert_eq!(
            unfold(Some(&t.model), &bad, &UnfoldOptions::new()).unwrap_err(),
            TacoError::Model(hardshell::Error::InvalidOverlap { overlap: bad_k, n_ctx: 24 })
        );
    }
}

#[test]
fn a_wrong_overlap_in_the_header_is_an_error_never_wrong_output() {
    // The checksum does not cover the header. A K that is valid but wrong gives other tokens,
    // and the checksum or the block length refuses them (ADR 0015, consequences).
    let t = tiny();
    let data = ENGLISH.as_bytes();
    let f = fold(&t.model, &t.tokenizer, data, &options(4096).with_overlap(GOLDEN_K)).unwrap();
    assert_eq!(f.stats.coded_blocks, 1);
    for k in [1u16, 2, 11, 13, 22] {
        let mut bad = f.container.clone();
        bad[94..96].copy_from_slice(&k.to_le_bytes());
        let r = unfold(Some(&t.model), &bad, &UnfoldOptions::new());
        assert!(
            matches!(r, Err(TacoError::CorruptBlock { .. } | TacoError::ChecksumMismatch { .. })),
            "K = {k}: {r:?}"
        );
    }
    // The bit cleared (K = 0) with the K bytes in place is refused by the parser.
    let mut bad = f.container.clone();
    bad[90] = 0;
    assert!(matches!(
        unfold(Some(&t.model), &bad, &UnfoldOptions::new()),
        Err(TacoError::Format(_))
    ));
}

/// The golden container of `golden_input` as the version 1 code wrote it (commit `dc621ce`,
/// ADR 0007). Plan §3 rule 8: the reader of version 2 still reads it.
const GOLDEN_V1: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../conformance/taco-v1-tiny.taco"));

#[test]
fn golden_v1_container_still_unfolds() {
    assert_eq!(GOLDEN_V1.len(), 1837);
    assert_eq!(
        hex(&Sha256::digest(GOLDEN_V1)),
        "c3ff718b1c9682589836078b36d936836077744b03b1cc8af3a6ced0d5821b8a",
        "conformance/taco-v1-tiny.taco changed"
    );
    let parsed = Container::parse(GOLDEN_V1).unwrap();
    assert_eq!(parsed.header().version(), VERSION_1);
    assert_eq!(parsed.header().byte_len(), HEADER_LEN_V1);
    assert_eq!(parsed.header().profile(), Profile::Text);
    let kinds: Vec<BlockKind> = parsed.blocks().iter().map(|b| b.kind()).collect();
    assert_eq!(
        kinds,
        [BlockKind::Coded, BlockKind::Coded, BlockKind::Raw, BlockKind::Raw, BlockKind::Coded]
    );
    let t = tiny();
    for threads in [1, 3] {
        let opts = UnfoldOptions::new().with_threads(taco::Threads::new(threads).unwrap());
        assert_eq!(unfold(Some(&t.model), GOLDEN_V1, &opts).unwrap(), golden_input());
    }
}

#[test]
fn decode_refuses_unknown_model_hash() {
    let t = tiny();
    let f = fold(&t.model, &t.tokenizer, ENGLISH.as_bytes(), &options(1024)).unwrap();
    let other = Model::load(&model_file(2)).unwrap();
    assert_ne!(container_hash(&other), container_hash(&t.model));
    match unfold(Some(&other), &f.container, &UnfoldOptions::new()) {
        Err(TacoError::ModelHashMismatch { container, loaded }) => {
            assert_eq!(container, container_hash(&t.model));
            assert_eq!(loaded, container_hash(&other));
        }
        r => panic!("{r:?}"),
    }
    assert_eq!(
        unfold(None, &f.container, &UnfoldOptions::new()),
        Err(TacoError::ModelRequired { container: container_hash(&t.model) })
    );
}

#[test]
fn decode_refuses_unknown_model_hash_with_only_raw_blocks() {
    // ADR 0006 section 7.3: a model hash that is not zero needs that model.
    let t = tiny();
    let data = Rng::new(4).bytes(300);
    let f = fold(&t.model, &t.tokenizer, &data, &options(100)).unwrap();
    assert_eq!(f.stats.coded_blocks, 0);
    let other = Model::load(&model_file(2)).unwrap();
    assert!(matches!(
        unfold(Some(&other), &f.container, &UnfoldOptions::new()),
        Err(TacoError::ModelHashMismatch { .. })
    ));
}

#[test]
fn a_flipped_bit_never_gives_wrong_output() {
    let t = tiny();
    let data = ENGLISH.as_bytes();
    let f = fold(&t.model, &t.tokenizer, data, &options(128)).unwrap();
    assert!(f.stats.coded_blocks > 0);
    let parsed = Container::parse(&f.container).unwrap();
    let coded_start = HEADER_LEN + 5 * parsed.blocks().len();
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
    eprintln!(
        "flips: {errors} of {} gave an error; the others gave the original bytes",
        f.container.len()
    );
    // Each flip in the header or the table is an error. In a coded block, a flip in the last
    // bits of the stream can leave the decoded tokens the same.
    assert!(errors >= coded_start);
}

#[test]
fn a_damaged_coded_block_is_an_error() {
    let t = tiny();
    let data = ENGLISH.as_bytes();
    let f = fold(&t.model, &t.tokenizer, data, &options(4096)).unwrap();
    assert_eq!(f.stats.coded_blocks, 1);
    // Byte 1 of the payload is the first byte of the range coder stream (the token count of
    // this block has 2 varint bytes, so byte 2 is the first coder byte).
    let first_coder_byte = HEADER_LEN + 5 + 2;
    let mut bad = f.container.clone();
    bad[first_coder_byte] ^= 0x40;
    let r = unfold(Some(&t.model), &bad, &UnfoldOptions::new());
    assert!(
        matches!(
            r,
            Err(TacoError::CorruptBlock { block: 0, .. } | TacoError::ChecksumMismatch { .. })
        ),
        "{r:?}"
    );
}

#[test]
fn a_damaged_checksum_is_an_error() {
    let t = tiny();
    let f = fold(&t.model, &t.tokenizer, ENGLISH.as_bytes(), &options(1024)).unwrap();
    let mut bad = f.container.clone();
    bad[48] ^= 1;
    assert!(matches!(
        unfold(Some(&t.model), &bad, &UnfoldOptions::new()),
        Err(TacoError::ChecksumMismatch { .. })
    ));
}

#[test]
fn a_truncated_container_is_an_error() {
    let t = tiny();
    let f = fold(&t.model, &t.tokenizer, ENGLISH.as_bytes(), &options(1024)).unwrap();
    for cut in [1, 10, f.container.len() / 2] {
        let short = &f.container[..f.container.len() - cut];
        assert!(matches!(
            unfold(Some(&t.model), short, &UnfoldOptions::new()),
            Err(TacoError::Format(_))
        ));
    }
}

#[test]
fn unfold_respects_the_output_limit() {
    let t = tiny();
    let f = fold(&t.model, &t.tokenizer, ENGLISH.as_bytes(), &options(1024)).unwrap();
    let n = ENGLISH.len() as u64;
    let opts = UnfoldOptions::new().with_max_output_len(Some(n - 1));
    assert_eq!(
        unfold(Some(&t.model), &f.container, &opts),
        Err(TacoError::OutputTooLarge { original_len: n, limit: n - 1 })
    );
    let opts = UnfoldOptions::new().with_max_output_len(Some(n));
    assert_eq!(unfold(Some(&t.model), &f.container, &opts).unwrap(), ENGLISH.as_bytes());
}

#[test]
fn fold_refuses_a_tokenizer_of_another_vocabulary() {
    let t = tiny();
    let bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let small = Tokenizer::new(bytes.iter().map(Vec::as_slice), None, &[]).unwrap();
    assert_eq!(
        fold(&t.model, &small, b"abc", &FoldOptions::new()).unwrap_err(),
        TacoError::TokenizerMismatch { tokenizer: 256, model: t.model.vocab_size() }
    );
}

#[test]
fn fold_refuses_tokens_that_do_not_restore_the_block() {
    // A tokenizer with the right size but with the bytes of 'a' and 'b' exchanged.
    let t = tiny();
    let mut vocab = common::vocabulary();
    vocab.swap(usize::from(b'a'), usize::from(b'b'));
    let eot = TokenId::new((vocab.len() - 1) as u32);
    let wrong = Tokenizer::new(vocab.iter().map(Vec::as_slice), Some(eot), &[]).unwrap();
    assert_eq!(
        fold(&t.model, &wrong, b"xyz abc", &FoldOptions::new()).unwrap_err(),
        TacoError::TokenizerRoundTrip { block: 0 }
    );
    // The same tokenizer without the exchange works, also without merges.
    let plain = Tokenizer::for_vocabulary(t.model.vocabulary(), t.model.eot_token(), &[]).unwrap();
    let f = fold(&t.model, &plain, b"xyz abc", &FoldOptions::new()).unwrap();
    assert_eq!(unfold(Some(&t.model), &f.container, &UnfoldOptions::new()).unwrap(), b"xyz abc");
    assert_eq!(merge_ids().len(), common::MERGES.len());
}
