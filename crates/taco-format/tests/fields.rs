//! One test for each field and each rule of the reader: a valid value passes, and each kind
//! of bad value gives its typed error.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{Craft, MODEL, bs, fold, unfold};
use taco_format::{
    BlockKind, Checksum, Container, FormatError, HEADER_LEN, MAGIC, ModelHash, VERSION,
};

/// A valid container of 10 original bytes in blocks of 4: raw, coded (2 bytes), raw.
fn sample() -> Craft {
    Craft {
        original_len: 10,
        checksum: *Checksum::of(b"abcdxxxxij").as_bytes(),
        block_size: 4,
        block_count: 3,
        entries: vec![(0, 4), (1, 2), (0, 2)],
        payload: b"abcd\x04xij".to_vec(),
        ..Craft::default()
    }
}

fn parse_err(c: &Craft) -> FormatError {
    Container::parse(&c.bytes()).expect_err("the container must be refused")
}

#[test]
fn sample_is_valid_and_gives_its_fields() {
    let bytes = sample().bytes();
    let c = Container::parse(&bytes).unwrap();
    let h = c.header();
    assert_eq!(h.version(), VERSION);
    assert_eq!(h.model_hash(), MODEL);
    assert!(h.needs_model());
    assert_eq!(h.original_len(), 10);
    assert_eq!(h.checksum(), Checksum::of(b"abcdxxxxij"));
    assert_eq!(h.block_size(), bs(4));
    assert_eq!(h.block_count(), 3);
    let b = c.blocks();
    assert_eq!(b.len(), 3);
    let kinds: Vec<BlockKind> = b.iter().map(|b| b.kind()).collect();
    assert_eq!(kinds, [BlockKind::Raw, BlockKind::Coded, BlockKind::Raw]);
    let offsets: Vec<u64> = b.iter().map(|b| b.original_offset()).collect();
    assert_eq!(offsets, [0, 4, 8]);
    let lens: Vec<u32> = b.iter().map(|b| b.original_len()).collect();
    assert_eq!(lens, [4, 4, 2]);
    assert_eq!(b[0].stored(), b"abcd");
    assert_eq!(b[1].stored(), b"\x04x");
    assert_eq!(b[2].stored(), b"ij");
    assert_eq!(b[2].index(), 2);
    // The stored bytes are slices of the input: no copy.
    assert!(core::ptr::eq(b[2].stored().as_ptr(), bytes[bytes.len() - 2..].as_ptr()));
}

#[test]
fn magic_must_be_taco() {
    assert_eq!(parse_err(&Craft { magic: *b"TACP", ..sample() }), FormatError::BadMagic);
    assert_eq!(parse_err(&Craft { magic: *b"ZIP\0", ..sample() }), FormatError::BadMagic);
    assert_eq!(Container::parse(b"PK").unwrap_err(), FormatError::BadMagic);
    assert_eq!(MAGIC, *b"TACO");
}

#[test]
fn short_input_with_the_magic_prefix_is_truncated() {
    for len in 0..HEADER_LEN {
        let bytes = sample().bytes();
        let e = Container::parse(&bytes[..len]).unwrap_err();
        assert!(matches!(e, FormatError::Truncated { part: "header", needed: 88, .. }), "{e:?}");
    }
}

#[test]
fn unknown_version_is_refused_before_other_fields() {
    for version in [0, 2, 3, 0x0100, u16::MAX] {
        assert_eq!(
            parse_err(&Craft { version, ..sample() }),
            FormatError::UnsupportedVersion { version }
        );
        // A later version can have another header: 6 bytes are enough to refuse it.
        let mut bytes = b"TACO".to_vec();
        bytes.extend_from_slice(&version.to_le_bytes());
        assert_eq!(
            Container::parse(&bytes).unwrap_err(),
            FormatError::UnsupportedVersion { version }
        );
    }
}

#[test]
fn flags_must_be_zero() {
    for flags in [1, 2, 0x8000, u16::MAX] {
        assert_eq!(parse_err(&Craft { flags, ..sample() }), FormatError::UnknownFlags { flags });
    }
}

#[test]
fn block_size_must_not_be_zero() {
    assert_eq!(parse_err(&Craft { block_size: 0, ..sample() }), FormatError::ZeroBlockSize);
    let empty = Craft { block_size: 0, ..Craft::default() };
    assert_eq!(parse_err(&empty), FormatError::ZeroBlockSize);
}

#[test]
fn block_count_must_be_the_rounded_up_quotient() {
    for block_count in [0, 2, 4, u32::MAX] {
        assert_eq!(
            parse_err(&Craft { block_count, ..sample() }),
            FormatError::WrongBlockCount {
                original_len: 10,
                block_size: 4,
                expected: 3,
                found: u64::from(block_count)
            }
        );
    }
    // Empty data has no block.
    let e = parse_err(&Craft { block_count: 1, entries: vec![(0, 0)], ..Craft::default() });
    assert!(matches!(e, FormatError::WrongBlockCount { expected: 0, found: 1, .. }), "{e:?}");
}

#[test]
fn original_length_must_agree_with_the_blocks() {
    // 13 bytes need 4 blocks.
    let e = parse_err(&Craft { original_len: 13, ..sample() });
    assert!(matches!(e, FormatError::WrongBlockCount { expected: 4, found: 3, .. }), "{e:?}");
    // 9 bytes: the last raw block has 1 original byte, but stores 2.
    assert_eq!(
        parse_err(&Craft { original_len: 9, ..sample() }),
        FormatError::RawLengthMismatch { block: 2, original_len: 1, stored_len: 2 }
    );
    // The original length is a u64: a huge value needs too many blocks for the table.
    let e = parse_err(&Craft { original_len: u64::MAX, ..sample() });
    assert!(matches!(e, FormatError::WrongBlockCount { .. }), "{e:?}");
}

#[test]
fn kind_must_be_raw_or_coded() {
    for kind in [2, 3, 0x80, u8::MAX] {
        let mut c = sample();
        c.entries[1].0 = kind;
        assert_eq!(parse_err(&c), FormatError::UnknownBlockKind { block: 1, kind });
    }
}

#[test]
fn raw_block_stores_exactly_its_original_length() {
    for stored in [0, 3, 5, u32::MAX] {
        let mut c = sample();
        c.entries[0].1 = stored;
        assert_eq!(
            parse_err(&c),
            FormatError::RawLengthMismatch {
                block: 0,
                original_len: 4,
                stored_len: u64::from(stored)
            }
        );
    }
}

#[test]
fn coded_block_must_be_smaller_than_its_original() {
    for stored in [4, 5, u32::MAX] {
        let mut c = sample();
        c.entries[1].1 = stored;
        assert_eq!(
            parse_err(&c),
            FormatError::CodedNotSmaller {
                block: 1,
                original_len: 4,
                stored_len: u64::from(stored)
            }
        );
    }
}

#[test]
fn coded_block_of_zero_bytes_is_valid() {
    let mut c = sample();
    c.entries[1].1 = 0;
    c.payload = b"abcdij".to_vec();
    let bytes = c.bytes();
    let parsed = Container::parse(&bytes).unwrap();
    assert!(parsed.blocks()[1].stored().is_empty());
    assert_eq!(parsed.blocks()[2].stored(), b"ij");
}

#[test]
fn raw_only_mode_has_no_coded_block() {
    let c = Craft { model_hash: [0; 32], ..sample() };
    assert_eq!(parse_err(&c), FormatError::CodedBlockInRawOnly { block: 1 });
    let raw = Craft {
        model_hash: [0; 32],
        entries: vec![(0, 4), (0, 4), (0, 2)],
        payload: b"abcdxxxxij".to_vec(),
        ..sample()
    };
    let bytes = raw.bytes();
    let parsed = Container::parse(&bytes).unwrap();
    assert!(!parsed.header().needs_model());
    assert_eq!(parsed.header().model_hash(), ModelHash::RAW_ONLY);
}

#[test]
fn table_outside_the_input_is_truncated_and_not_allocated() {
    // The header claims 2^32 - 1 blocks of 1 byte. The input has only the header.
    let c = Craft {
        original_len: u64::from(u32::MAX),
        block_size: 1,
        block_count: u32::MAX,
        ..sample()
    };
    let mut bytes = c.bytes();
    bytes.truncate(HEADER_LEN + 7);
    assert_eq!(
        Container::parse(&bytes).unwrap_err(),
        FormatError::Truncated {
            part: "block table",
            needed: 88 + 5 * u64::from(u32::MAX),
            found: 95
        }
    );
    // Each length inside the table is short too.
    let full = sample().bytes();
    for len in HEADER_LEN..HEADER_LEN + 15 {
        let e = Container::parse(&full[..len]).unwrap_err();
        assert!(matches!(e, FormatError::Truncated { part: "block table", .. }), "{e:?}");
    }
}

#[test]
fn input_must_end_at_the_last_block() {
    let full = sample().bytes();
    assert_eq!(full.len(), 88 + 15 + 8);
    for len in HEADER_LEN + 15..full.len() {
        assert_eq!(
            Container::parse(&full[..len]).unwrap_err(),
            FormatError::WrongTotalLength { expected: 111, found: len as u64 }
        );
    }
    for extra in 1..4 {
        let mut long = full.clone();
        long.extend(std::iter::repeat_n(0, extra));
        assert_eq!(
            Container::parse(&long).unwrap_err(),
            FormatError::WrongTotalLength { expected: 111, found: 111 + extra as u64 }
        );
    }
}

#[test]
fn empty_container_is_the_header_only() {
    let c = Craft { checksum: *Checksum::of(&[]).as_bytes(), ..Craft::default() };
    let bytes = c.bytes();
    assert_eq!(bytes.len(), HEADER_LEN);
    let parsed = Container::parse(&bytes).unwrap();
    assert_eq!(parsed.header().block_count(), 0);
    assert!(parsed.blocks().is_empty());
    parsed.header().verify_checksum(&[]).unwrap();
    assert_eq!(
        Container::parse(&[bytes.as_slice(), &[0]].concat()).unwrap_err(),
        FormatError::WrongTotalLength { expected: 88, found: 89 }
    );
}

#[test]
fn checksum_helper_refuses_wrong_length_and_wrong_bytes() {
    let bytes = sample().bytes();
    let c = Container::parse(&bytes).unwrap();
    let h = c.header();
    h.verify_checksum(b"abcdxxxxij").unwrap();
    assert_eq!(
        h.verify_checksum(b"abcdxxxxi"),
        Err(FormatError::RestoredLengthMismatch { expected: 10, found: 9 })
    );
    assert_eq!(
        h.verify_checksum(b"abcdxxxxiJ"),
        Err(FormatError::ChecksumMismatch {
            expected: Checksum::of(b"abcdxxxxij"),
            found: Checksum::of(b"abcdxxxxiJ"),
        })
    );
}

#[test]
fn model_hash_helper_refuses_another_model() {
    let bytes = sample().bytes();
    let c = Container::parse(&bytes).unwrap();
    c.header().verify_model_hash(&MODEL).unwrap();
    let other = ModelHash::from_bytes([0x5B; 32]);
    assert_eq!(
        c.header().verify_model_hash(&other),
        Err(FormatError::ModelHashMismatch { container: MODEL, loaded: other })
    );
    // A real model hash never matches the raw-only "model".
    assert!(c.header().verify_model_hash(&ModelHash::RAW_ONLY).is_err());
    // A raw-only container needs no model, so any loaded model passes.
    let raw = fold(b"hello", bs(2), ModelHash::RAW_ONLY);
    let raw = Container::parse(&raw).unwrap();
    raw.header().verify_model_hash(&other).unwrap();
    raw.header().verify_model_hash(&ModelHash::RAW_ONLY).unwrap();
}

#[test]
fn errors_carry_no_user_data() {
    let e = FormatError::ChecksumMismatch {
        expected: Checksum::of(b"secret"),
        found: Checksum::of(b"secreT"),
    };
    let text = e.to_string();
    assert!(!text.contains("secret"));
    assert!(text.contains(&Checksum::of(b"secret").to_string()));
}

#[test]
fn unfold_with_another_model_fails() {
    let data = b"aaaaaaaabbbbbbbbcccccccc".repeat(10);
    let bytes = fold(&data, bs(16), MODEL);
    assert_eq!(unfold(&bytes, MODEL).unwrap(), data);
    let other = ModelHash::from_bytes([1; 32]);
    assert!(unfold(&bytes, other).is_err());
}
