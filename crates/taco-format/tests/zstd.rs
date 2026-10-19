//! The zstd block (ADR 0011 §5): the decoded size is bounded by the original length of the
//! block, the decoder does not allocate from a size in the frame, and damage to a zstd frame
//! gives an error or the exact original data, never other data.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::cast_possible_truncation,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{Craft, MODEL, UnfoldError, bs, fold, phrases, unfold};
use proptest::prelude::*;
use taco_format::{
    Block, BlockKind, BlockRef, Checksum, Container, ContainerParams, FormatError, ModelHash,
    ZSTD_WINDOW_LOG_MAX, compress_zstd,
};
use zstd_safe::{CCtx, CParameter};

/// A version 2 container in the raw-only mode with one zstd block: `frame` as the stored
/// bytes, `original_len` as the original length and the block size. The checksum is of
/// `data`.
fn one_block(frame: &[u8], original_len: u32, data: &[u8]) -> Vec<u8> {
    Craft {
        model_hash: [0; 32],
        original_len: u64::from(original_len),
        checksum: *Checksum::of(data).as_bytes(),
        block_size: original_len,
        block_count: 1,
        entries: vec![(2, u32::try_from(frame.len()).unwrap())],
        payload: frame.to_vec(),
        ..Craft::default()
    }
    .v2(0)
    .bytes()
}

/// Decodes the only block of `bytes` with [`BlockRef::decode_zstd_into`].
fn decode_only(bytes: &[u8]) -> (Result<(), FormatError>, Vec<u8>) {
    let c = Container::parse(bytes).unwrap();
    let b: &BlockRef<'_> = &c.blocks()[0];
    let mut out = Vec::new();
    (b.decode_zstd_into(&mut out), out)
}

/// The reason of a [`FormatError::CorruptZstdBlock`].
fn reason(r: Result<(), FormatError>) -> &'static str {
    match r {
        Err(FormatError::CorruptZstdBlock { block: 0, reason }) => reason,
        other => panic!("expected a corrupt zstd block, got {other:?}"),
    }
}

/// A hand-made zstd frame (RFC 8878): no single segment, a window of 1 KiB, a 4-byte content
/// size `content_size`, and one last RLE block of `rle_len` copies of `byte`.
fn rle_frame(content_size: u32, rle_len: u32, byte: u8) -> Vec<u8> {
    let mut f = vec![0x28, 0xB5, 0x2F, 0xFD];
    // Frame header descriptor: content size field of 4 bytes (flag 2), no single segment, no
    // checksum, no dictionary id.
    f.push(0b1000_0000);
    // Window descriptor: exponent 0, mantissa 0, so the window is 2^10 bytes.
    f.push(0);
    f.extend_from_slice(&content_size.to_le_bytes());
    // Block header: last block, type RLE (1), size rle_len.
    let header = 1 | (1 << 1) | (rle_len << 3);
    f.extend_from_slice(&header.to_le_bytes()[..3]);
    f.push(byte);
    f
}

#[test]
fn hand_made_rle_frame_is_valid() {
    let frame = rle_frame(1_000, 1_000, b'r');
    let bytes = one_block(&frame, 1_000, &[b'r'; 1_000]);
    assert_eq!(unfold(&bytes, ModelHash::RAW_ONLY).unwrap(), vec![b'r'; 1_000]);
}

#[test]
fn content_size_must_be_the_original_length() {
    let data = phrases(1, 300);
    let frame = compress_zstd(&data).unwrap();
    for len in [frame.len() as u32 + 1, 299, 301, 1_000] {
        let bytes = one_block(&frame, len, &data);
        let (r, out) = decode_only(&bytes);
        assert_eq!(reason(r), "the frame content size is not the original length");
        assert!(out.is_empty());
    }
}

#[test]
fn frame_with_no_content_size_is_refused() {
    let data = phrases(2, 300);
    let mut cctx = CCtx::create();
    cctx.set_parameter(CParameter::ContentSizeFlag(false)).unwrap();
    let mut frame = vec![0u8; 1_000];
    let n = cctx.compress2(frame.as_mut_slice(), &data).unwrap();
    frame.truncate(n);
    let bytes = one_block(&frame, 300, &data);
    assert_eq!(reason(decode_only(&bytes).0), "the frame header has no content size");
}

#[test]
fn false_content_size_cannot_make_more_or_fewer_bytes() {
    // The frame says 1,000 bytes and gives 500, or says 500 and gives 1,000.
    for (claim, real) in [(1_000, 500), (500, 1_000)] {
        let frame = rle_frame(claim, real, b'x');
        let bytes = one_block(&frame, claim, &vec![b'x'; claim as usize]);
        let (r, out) = decode_only(&bytes);
        assert!(matches!(r, Err(FormatError::CorruptZstdBlock { block: 0, .. })), "{r:?}");
        assert!(out.is_empty());
        assert!(out.capacity() <= 2 * claim as usize, "{}", out.capacity());
    }
}

#[test]
fn huge_original_length_is_not_allocated() {
    // A header of 64 MiB (the largest block size) in one block and a frame of 14 bytes that
    // says 64 MiB but gives 1,000 bytes. The decoder must not reserve 64 MiB.
    let claim = taco_format::MAX_BLOCK_SIZE.get();
    let frame = rle_frame(claim, 1_000, b'h');
    assert_eq!(frame.len(), 14);
    let bytes = one_block(&frame, claim, b"");
    // A block of 4 GiB: the parser refuses the block size (ADR 0011, addendum §4).
    let huge = one_block(&rle_frame(u32::MAX, 1_000, b'h'), u32::MAX, b"");
    assert!(matches!(Container::parse(&huge), Err(FormatError::BlockSizeTooLarge { .. })));
    let (r, out) = decode_only(&bytes);
    assert!(matches!(r, Err(FormatError::CorruptZstdBlock { block: 0, .. })), "{r:?}");
    assert!(out.capacity() < 1 << 20, "{}", out.capacity());
}

#[test]
fn window_larger_than_8_mib_is_refused() {
    // 9 MiB of zeros in one single-segment frame: the window is the content size, 9 MiB.
    let len = 9 << 20;
    let data = vec![0u8; len];
    let mut cctx = CCtx::create();
    cctx.set_parameter(CParameter::CompressionLevel(1)).unwrap();
    cctx.set_parameter(CParameter::WindowLog(24)).unwrap();
    let mut frame = vec![0u8; zstd_safe::compress_bound(len)];
    let n = cctx.compress2(frame.as_mut_slice(), &data).unwrap();
    frame.truncate(n);
    let bytes = one_block(&frame, len as u32, &data);
    let (r, out) = decode_only(&bytes);
    assert!(matches!(r, Err(FormatError::CorruptZstdBlock { block: 0, .. })), "{r:?}");
    assert!(out.is_empty());
    assert_eq!(ZSTD_WINDOW_LOG_MAX, 23);
}

#[test]
fn encoder_output_of_a_9_mib_block_decodes() {
    // Level 19 uses a window of at most 8 MiB, so its frames always decode.
    let len: usize = 9 << 20;
    let data: Vec<u8> = (0..len).map(|i| (i / 4096) as u8 ^ (i % 7) as u8).collect();
    let frame = compress_zstd(&data).unwrap();
    println!("9 MiB block: zstd frame of {} bytes", frame.len());
    let bytes = one_block(&frame, len as u32, &data);
    let (r, out) = decode_only(&bytes);
    r.unwrap();
    assert!(out == data);
}

#[test]
fn not_one_frame_is_refused() {
    let data = phrases(4, 400);
    let frame = compress_zstd(&data).unwrap();
    // A skippable frame (magic 0x184D2A50) of the same length.
    let mut skippable = vec![0x50, 0x2A, 0x4D, 0x18];
    skippable.extend_from_slice(&u32::try_from(frame.len() - 8).unwrap().to_le_bytes());
    skippable.resize(frame.len(), 0);
    let bytes = one_block(&skippable, 400, &data);
    assert_eq!(
        reason(decode_only(&bytes).0),
        "the stored bytes do not start with the zstd frame magic"
    );
    // A frame with one more byte after it.
    let long = [frame.as_slice(), &[0]].concat();
    assert_eq!(reason(decode_only(&one_block(&long, 400, &data)).0), "bytes follow the zstd frame");
    // A truncated frame.
    let short = &frame[..frame.len() - 1];
    assert!(decode_only(&one_block(short, 400, &data)).0.is_err());
}

#[test]
fn every_single_flip_of_a_zstd_container_gives_an_error_or_the_original() {
    let data = phrases(8, 600);
    let bytes = fold(&data, bs(300), ModelHash::RAW_ONLY);
    let c = Container::parse(&bytes).unwrap();
    assert!(c.blocks().iter().all(|b| b.kind() == BlockKind::Zstd));
    let (mut errors, mut same) = (0, 0);
    for pos in 0..bytes.len() {
        for x in [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80] {
            let mut damaged = bytes.clone();
            damaged[pos] ^= x;
            match unfold(&damaged, MODEL) {
                Ok(restored) => {
                    assert_eq!(restored, data, "flip {x:#04x} at {pos} gave other data");
                    same += 1;
                }
                Err(_) => errors += 1,
            }
        }
    }
    // NOTE: the profile byte (offset 88) can change from text to logs. The toy unfold does not
    // use the profile, so that flip gives the original data.
    println!("{} bytes, {errors} errors, {same} outputs equal to the original", bytes.len());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1_024))]

    /// Any data round-trips through zstd blocks, and the zstd block is never larger than raw.
    #[test]
    fn zstd_blocks_round_trip(
        seed in any::<u64>(),
        len in 0usize..3_000,
        block_size in 1u32..1_500,
    ) {
        let data = phrases(seed, len);
        let bytes = fold(&data, bs(block_size), ModelHash::RAW_ONLY);
        prop_assert_eq!(unfold(&bytes, ModelHash::RAW_ONLY).unwrap(), data.clone());
        let blocks: Vec<Block> =
            bs(block_size).split(&data).map(|b| Block::Raw.with_zstd(b).unwrap()).collect();
        let params = ContainerParams::new(ModelHash::RAW_ONLY, bs(block_size));
        prop_assert_eq!(Container::build(&params, &data, &blocks).unwrap(), bytes);
    }

    /// Random damage to the stored bytes of a zstd block gives an error or the original data.
    #[test]
    fn damaged_zstd_payload_never_gives_other_data(
        seed in any::<u64>(),
        flips in prop::collection::vec((any::<usize>(), 1u8..=255), 1..6),
        cut in any::<bool>(),
    ) {
        let data = phrases(seed, 2_000);
        let bytes = fold(&data, bs(1_000), ModelHash::RAW_ONLY);
        let start = 104 + 2 * 5;
        let mut damaged = bytes.clone();
        for &(p, x) in &flips {
            let i = start + p % (bytes.len() - start);
            damaged[i] ^= x;
        }
        if cut {
            damaged.pop();
        }
        match unfold(&damaged, ModelHash::RAW_ONLY) {
            Ok(restored) => prop_assert_eq!(restored, data),
            Err(UnfoldError::Format(_)) => {}
            Err(e) => prop_assert!(false, "{:?}", e),
        }
    }
}
