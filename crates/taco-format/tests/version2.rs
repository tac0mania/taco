//! Version 2 (ADR 0011): the fields that version 2 adds, the zstd block kind in the block
//! table, and the boundary between version 1 and version 2.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{Craft, MODEL, bs, fold, unfold};
use taco_format::{
    Block, BlockKind, Checksum, CodingParts, Container, ContainerParams, FormatError, HEADER_LEN,
    MixerRule, ModelHash, Profile, VERSION, compress_zstd,
};

/// 40 bytes in blocks of 20: block 0 is zstd, block 1 is raw.
fn data() -> Vec<u8> {
    [vec![b'q'; 20], b"0123456789abcdefghij".to_vec()].concat()
}

/// The zstd frame of block 0 of [`data`].
fn frame() -> Vec<u8> {
    compress_zstd(&data()[..20]).unwrap()
}

/// A valid version 2 container of [`data`]: zstd, raw. Profile `profile`.
fn sample(profile: u8) -> Craft {
    let d = data();
    let f = frame();
    let len = u32::try_from(f.len()).unwrap();
    Craft {
        original_len: 40,
        checksum: *Checksum::of(&d).as_bytes(),
        block_size: 20,
        block_count: 2,
        entries: vec![(2, len), (0, 20)],
        payload: [f.as_slice(), &d[20..]].concat(),
        ..Craft::default()
    }
    .v2(profile)
}

fn parse_err(c: &Craft) -> FormatError {
    Container::parse(&c.bytes()).expect_err("the container must be refused")
}

#[test]
fn sample_is_valid_and_gives_its_fields() {
    assert!(frame().len() < 20, "{}", frame().len());
    for (byte, profile) in [(0, Profile::Text), (1, Profile::Logs)] {
        let bytes = sample(byte).bytes();
        let c = Container::parse(&bytes).unwrap();
        let h = c.header();
        assert_eq!(h.version(), VERSION);
        assert_eq!(h.byte_len(), HEADER_LEN);
        assert_eq!(h.profile(), profile);
        assert_eq!(h.model_hash(), MODEL);
        let kinds: Vec<BlockKind> = c.blocks().iter().map(|b| b.kind()).collect();
        assert_eq!(kinds, [BlockKind::Zstd, BlockKind::Raw]);
        assert_eq!(c.blocks()[0].stored(), frame().as_slice());
        assert_eq!(unfold(&bytes, MODEL).unwrap(), data());
    }
}

#[test]
fn writer_output_is_the_hand_made_container() {
    let blocks = [Block::Zstd(frame()), Block::Raw];
    for (byte, profile) in [(0, Profile::Text), (1, Profile::Logs)] {
        let params = ContainerParams::new(MODEL, bs(20)).with_profile(profile);
        let built = Container::build(&params, &data(), &blocks).unwrap();
        assert_eq!(built, sample(byte).bytes());
    }
}

#[test]
fn version_2_header_needs_104_bytes() {
    let bytes = sample(0).bytes();
    for len in 6..HEADER_LEN {
        let e = Container::parse(&bytes[..len]).unwrap_err();
        assert!(matches!(e, FormatError::Truncated { part: "header", needed: 104, .. }), "{e:?}");
    }
    // Before the version is known, the need is the shortest header.
    for len in 0..6 {
        let e = Container::parse(&bytes[..len]).unwrap_err();
        assert!(matches!(e, FormatError::Truncated { part: "header", needed: 88, .. }), "{e:?}");
    }
}

#[test]
fn profile_must_be_text_or_logs() {
    for profile in [2, 3, 0x80, u8::MAX] {
        assert_eq!(parse_err(&sample(profile)), FormatError::UnknownProfile { profile });
    }
}

#[test]
fn reserved_byte_coding_parts_and_coding_parameters_must_be_zero() {
    let mut c = sample(0);
    c.section[1] = 1;
    assert_eq!(parse_err(&c), FormatError::ReservedNotZero { field: "reserved byte 89" });
    // Bit 0 is the match model and the mixer (ADR 0010); the other bits are unknown.
    for (lo, hi, parts) in [(2, 0, 2), (0, 0x80, 0x8000), (0xFF, 0xFF, u16::MAX)] {
        let mut c = sample(1);
        c.section[2] = lo;
        c.section[3] = hi;
        assert_eq!(parse_err(&c), FormatError::UnknownCodingParts { parts });
    }
    for i in 4..16 {
        let mut c = sample(0);
        c.section[i] = 0x10;
        assert_eq!(parse_err(&c), FormatError::ReservedNotZero { field: "coding parameters" });
    }
}

#[test]
fn flags_must_be_zero_in_version_2() {
    for flags in [1, 0x8000, u16::MAX] {
        assert_eq!(parse_err(&Craft { flags, ..sample(0) }), FormatError::UnknownFlags { flags });
    }
}

#[test]
fn version_1_refuses_the_zstd_kind() {
    // The same blocks with a version 1 header: kind 2 does not exist in version 1.
    let v1 = Craft { version: 1, section: Vec::new(), ..sample(0) };
    assert_eq!(parse_err(&v1), FormatError::UnknownBlockKind { block: 0, kind: 2 });
}

#[test]
fn version_2_refuses_kinds_above_three() {
    for kind in [4, 5, 0x80, u8::MAX] {
        let mut c = mixed_sample();
        c.entries[0].0 = kind;
        assert_eq!(parse_err(&c), FormatError::UnknownBlockKind { block: 0, kind });
    }
}

/// [`sample`] with the profile `logs`, the match model and the mixer (rule version 1), and
/// block 1 stored mixed (the fixed bytes 7, 8, 9 in place of a real payload).
fn mixed_sample() -> Craft {
    let d = data();
    let f = frame();
    let len = u32::try_from(f.len()).unwrap();
    let mut c = Craft {
        entries: vec![(2, len), (3, 3)],
        payload: [f.as_slice(), &[7, 8, 9]].concat(),
        ..sample(1)
    };
    c.section[2] = 1;
    c.section[4] = 1;
    assert_eq!(c.checksum, *Checksum::of(&d).as_bytes());
    c
}

#[test]
fn mixed_block_parses_with_the_coding_part() {
    let bytes = mixed_sample().bytes();
    let c = Container::parse(&bytes).unwrap();
    assert_eq!(c.header().profile(), Profile::Logs);
    assert_eq!(c.header().coding_parts(), CodingParts::with_mixer(MixerRule::V1));
    assert_eq!(c.header().coding_parts().mixer(), Some(MixerRule::V1));
    let kinds: Vec<BlockKind> = c.blocks().iter().map(|b| b.kind()).collect();
    assert_eq!(kinds, [BlockKind::Zstd, BlockKind::Mixed]);
    assert_eq!(c.blocks()[1].stored(), &[7, 8, 9]);
    assert!(BlockKind::Mixed.needs_model());
    // The writer gives the same bytes.
    let params = ContainerParams::new(MODEL, bs(20))
        .with_profile(Profile::Logs)
        .with_coding(CodingParts::with_mixer(MixerRule::V1));
    let built =
        Container::build(&params, &data(), &[Block::Zstd(frame()), Block::Mixed(vec![7, 8, 9])])
            .unwrap();
    assert_eq!(built, bytes);
    assert_eq!(&built[90..93], &[1, 0, 1]);
}

#[test]
fn mixed_block_needs_the_coding_part() {
    let mut c = mixed_sample();
    c.section[2] = 0;
    c.section[4] = 0;
    assert_eq!(parse_err(&c), FormatError::MixedWithoutCodingPart { block: 1 });
    let params = ContainerParams::new(MODEL, bs(20)).with_profile(Profile::Logs);
    assert_eq!(
        Container::build(&params, &data(), &[Block::Zstd(frame()), Block::Mixed(vec![1])]),
        Err(FormatError::MixedWithoutCodingPart { block: 1 })
    );
}

#[test]
fn coding_part_needs_a_known_mixer_rule() {
    for rule in [0, 2, 3, 0x80, u8::MAX] {
        let mut c = mixed_sample();
        c.section[4] = rule;
        assert_eq!(parse_err(&c), FormatError::UnknownMixerRule { rule });
    }
    for i in 5..16 {
        let mut c = mixed_sample();
        c.section[i] = 1;
        assert_eq!(parse_err(&c), FormatError::ReservedNotZero { field: "coding parameters" });
    }
}

#[test]
fn text_profile_refuses_the_coding_part() {
    let mut c = mixed_sample();
    c.section[0] = 0;
    assert_eq!(parse_err(&c), FormatError::CodingPartsNotInProfile { profile: 0, parts: 1 });
    let params =
        ContainerParams::new(MODEL, bs(20)).with_coding(CodingParts::with_mixer(MixerRule::V1));
    assert_eq!(
        Container::build(&params, &data(), &[Block::Zstd(frame()), Block::Raw]),
        Err(FormatError::CodingPartsNotInProfile { profile: 0, parts: 1 })
    );
}

#[test]
fn logs_profile_without_the_coding_part_is_valid() {
    // ADR 0011 vector C is such a container (raw-only mode, zstd blocks).
    let bytes = sample(1).bytes();
    assert_eq!(Container::parse(&bytes).unwrap().header().coding_parts(), CodingParts::NONE);
}

#[test]
fn mixed_block_must_be_smaller_than_its_original() {
    for stored in [20, 21, u32::MAX] {
        let mut c = mixed_sample();
        c.entries[1].1 = stored;
        assert_eq!(
            parse_err(&c),
            FormatError::CodedNotSmaller {
                block: 1,
                original_len: 20,
                stored_len: u64::from(stored)
            }
        );
    }
    // A mixed block of 0 bytes is valid, as a coded block of 0 bytes (ADR 0006 §6.4).
    let mut c = mixed_sample();
    c.entries[1].1 = 0;
    c.payload.truncate(frame().len());
    assert!(Container::parse(&c.bytes()).is_ok());
}

#[test]
fn raw_only_mode_refuses_a_mixed_block() {
    let c = Craft { model_hash: [0; 32], ..mixed_sample() };
    assert_eq!(parse_err(&c), FormatError::CodedBlockInRawOnly { block: 1 });
    let params = ContainerParams::new(ModelHash::RAW_ONLY, bs(20))
        .with_profile(Profile::Logs)
        .with_coding(CodingParts::with_mixer(MixerRule::V1));
    assert_eq!(
        Container::build(&params, &data(), &[Block::Zstd(frame()), Block::Mixed(vec![1])]),
        Err(FormatError::CodedBlockInRawOnly { block: 1 })
    );
}

#[test]
fn version_1_refuses_the_mixed_kind() {
    let mut v1 = Craft { version: 1, section: Vec::new(), ..sample(0) };
    v1.entries[0].0 = 3;
    assert_eq!(parse_err(&v1), FormatError::UnknownBlockKind { block: 0, kind: 3 });
}

#[test]
fn version_2_header_with_the_version_1_length_is_refused() {
    // A version 2 field with the 88-byte header of version 1: the table is read at 104, so the
    // container does not agree with itself.
    let c = Craft { section: Vec::new(), ..sample(0) };
    assert!(Container::parse(&c.bytes()).is_err());
}

#[test]
fn zstd_block_must_be_smaller_than_its_original() {
    for stored in [20, 21, u32::MAX] {
        let mut c = sample(0);
        c.entries[0].1 = stored;
        assert_eq!(
            parse_err(&c),
            FormatError::ZstdNotSmaller {
                block: 0,
                original_len: 20,
                stored_len: u64::from(stored)
            }
        );
    }
}

#[test]
fn raw_only_mode_permits_zstd_and_refuses_coded() {
    let raw_only = Craft { model_hash: [0; 32], ..sample(1) };
    let bytes = raw_only.bytes();
    let c = Container::parse(&bytes).unwrap();
    assert!(!c.header().needs_model());
    assert_eq!(unfold(&bytes, ModelHash::RAW_ONLY).unwrap(), data());
    let mut coded = Craft { model_hash: [0; 32], ..sample(1) };
    coded.entries[0].0 = 1;
    assert_eq!(parse_err(&coded), FormatError::CodedBlockInRawOnly { block: 0 });
}

#[test]
fn zstd_block_needs_no_model_but_the_container_hash_still_counts() {
    // A container with a model hash needs that model, even if no block is coded.
    let bytes = sample(0).bytes();
    let other = ModelHash::from_bytes([1; 32]);
    assert!(matches!(unfold(&bytes, other), Err(common::UnfoldError::Format(_))));
    assert!(!BlockKind::Zstd.needs_model());
}

#[test]
fn decode_zstd_refuses_another_kind() {
    let bytes = sample(0).bytes();
    let c = Container::parse(&bytes).unwrap();
    let mut out = Vec::new();
    assert_eq!(c.blocks()[1].decode_zstd_into(&mut out), Err(FormatError::NotZstd { block: 1 }));
    assert!(out.is_empty());
}

#[test]
fn writer_writes_version_2_also_without_a_zstd_block() {
    let bytes = fold(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", bs(10), MODEL);
    let c = Container::parse(&bytes).unwrap();
    assert_eq!(c.header().version(), VERSION);
    assert!(c.blocks().iter().all(|b| b.kind() == BlockKind::Coded));
}
