//! The rules that the freeze of version 2 added (ADR 0011, addendum "Version 2 is frozen"):
//! the largest block size, for the reader and the writer of both versions.

#![allow(clippy::unwrap_used, reason = "test code: a failure must stop the test")]

mod common;

use common::{Craft, MODEL, bs};
use taco_format::{
    Block, Container, ContainerParams, FormatError, MAX_BLOCK_SIZE, ModelHash, StreamWriter,
};

const MAX: u32 = 64 * 1024 * 1024;

/// A container of one raw block of 1 byte, with the block size `block_size`.
fn one_byte(version: u16, block_size: u32) -> Vec<u8> {
    let c = Craft {
        model_hash: [0; 32],
        original_len: 1,
        checksum: *taco_format::Checksum::of(b"z").as_bytes(),
        block_size,
        block_count: 1,
        entries: vec![(0, 1)],
        payload: b"z".to_vec(),
        ..Craft::default()
    };
    if version == 2 { c.v2(0).bytes() } else { c.bytes() }
}

#[test]
fn the_maximum_block_size_is_64_mib() {
    assert_eq!(MAX_BLOCK_SIZE.get(), MAX);
}

#[test]
fn the_reader_refuses_a_block_size_above_the_maximum_in_both_versions() {
    for version in [1, 2] {
        assert!(Container::parse(&one_byte(version, MAX)).is_ok(), "version {version}");
        for block_size in [MAX + 1, u32::MAX] {
            assert_eq!(
                Container::parse(&one_byte(version, block_size)).unwrap_err(),
                FormatError::BlockSizeTooLarge { block_size, max: MAX },
                "version {version}"
            );
        }
    }
}

#[test]
fn the_writer_refuses_a_block_size_above_the_maximum() {
    let ok = ContainerParams::new(ModelHash::RAW_ONLY, bs(MAX));
    assert!(Container::build(&ok, b"z", &[Block::Raw]).is_ok());
    assert!(StreamWriter::new(ok, None).is_ok());
    let big = ContainerParams::new(MODEL, bs(MAX + 1));
    let refused = FormatError::BlockSizeTooLarge { block_size: MAX + 1, max: MAX };
    assert_eq!(Container::build(&big, b"z", &[Block::Raw]).unwrap_err(), refused);
    assert_eq!(StreamWriter::new(big, None).unwrap_err(), refused);
}
