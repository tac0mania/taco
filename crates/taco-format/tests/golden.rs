//! Golden vectors: fixed containers with their SHA-256 (plan §11). A change to the layout
//! changes a hash, and it needs a new container version (plan §3 rule 8).
//!
//! The version 1 vectors are hand-made with the layout of ADR 0006, because the writer writes
//! version 2 only. Their hashes are the hashes of ADR 0006. The reader must still read them.
//!
//! The coded blocks hold fixed bytes, not the output of a real coder: the vector tests the
//! container only. The zstd blocks hold real zstd frames. Their bytes depend on the zstd
//! library version (ADR 0011 §6), so the frames have their own hashes: if a frame hash fails,
//! the zstd library changed, not the layout.
//!
//! NOTE: a separate Python copy of the version 1 layout gave the same two version 1 hashes on
//! 2026-10-18 (measured).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{Craft, bs, phrases, splitmix64, unfold};
use sha2::{Digest, Sha256};
use taco_format::{
    Block, BlockKind, Checksum, Container, ContainerParams, ModelHash, Profile, VERSION, VERSION_1,
    ZSTD_LIBRARY_VERSION, compress_zstd, zstd_library_version,
};

/// SHA-256 of version 1 vector A (ADR 0006).
const VECTOR_A_SHA256: &str = "c1c07402a5dae2993c596345a696d35b7c1c0502fe304d31fa33d4f5ecefeeec";
/// The length of version 1 vector A: 88 + 4 * 5 + 100 + 7 + 0 + 50.
const VECTOR_A_LEN: usize = 265;
/// SHA-256 of the version 1 container of empty data in the raw-only mode, block size 65,536.
const VECTOR_EMPTY_SHA256: &str =
    "2f8a2052a618a4ca1adfc67aa07836c7d4fda0b10738142e9327228ddd0bcdf7";

/// SHA-256 of version 2 vector B.
const VECTOR_B_SHA256: &str = "18d7b8a97cf26ffc0a28d7e4575f870ed7463cabefded8450ee7f912d723e8d9";
/// SHA-256 of the zstd frame of block 2 of vector B (zstd 1.5.7, level 19).
const VECTOR_B_FRAME_SHA256: &str =
    "4a62d79659c63a4dbc2ea9236d5001543baea7f608f045b4563b8262b11fe83d";
/// SHA-256 of version 2 vector C.
const VECTOR_C_SHA256: &str = "72dcd44f9b87a9e0fd7f4be6b6e8de252597ec98eac4b5e0bc5215b06ccb7a59";
/// SHA-256 of the version 2 container of empty data in the raw-only mode, block size 65,536.
const VECTOR_EMPTY_V2_SHA256: &str =
    "03de05c54a2e192c7dd046526ef8a3c9882b16465b9ab577b42c3477e130c390";

/// The model hash of vectors A and B: bytes 0, 1, ..., 31.
fn model() -> ModelHash {
    ModelHash::from_bytes(core::array::from_fn(|i| u8::try_from(i).unwrap()))
}

/// 350 bytes: byte `i` is `(i * 7 + 3) mod 256`.
fn vector_a_data() -> Vec<u8> {
    (0..350u32).map(|i| u8::try_from((i * 7 + 3) % 256).unwrap()).collect()
}

/// Version 1 vector A, hand-made: blocks of 100 bytes: raw, coded `"coded-1"`, coded with 0
/// bytes, raw (50 bytes).
fn vector_a() -> Vec<u8> {
    let data = vector_a_data();
    Craft {
        model_hash: *model().as_bytes(),
        original_len: 350,
        checksum: *Checksum::of(&data).as_bytes(),
        block_size: 100,
        block_count: 4,
        entries: vec![(0, 100), (1, 7), (1, 0), (0, 50)],
        payload: [&data[..100], b"coded-1", &data[300..]].concat(),
        ..Craft::default()
    }
    .bytes()
}

/// 450 bytes: 100 pseudo-random bytes (seed 11), 100 bytes of `i mod 251`, 150 bytes of
/// phrases (seed 5), 100 bytes of `0x41`.
fn vector_b_data() -> Vec<u8> {
    let mut s = 11;
    let mut data: Vec<u8> = (0..100).map(|_| splitmix64(&mut s).to_le_bytes()[0]).collect();
    data.extend((0..100u32).map(|i| u8::try_from(i % 251).unwrap()));
    data.extend(phrases(5, 150));
    data.extend([0x41; 100]);
    data
}

/// The zstd frame of block 2 of vector B: original bytes 200 to 299.
fn vector_b_frame() -> Vec<u8> {
    compress_zstd(&vector_b_data()[200..300]).unwrap()
}

/// Version 2 vector B: model hash bytes 0 to 31, profile text, blocks of 100 bytes: raw,
/// coded `"coded-1"`, zstd (a real frame of phrases), coded with 0 bytes, zstd (a real frame
/// of 50 bytes of `0x41`).
fn vector_b() -> Vec<u8> {
    let data = vector_b_data();
    let blocks = [
        Block::Raw,
        Block::Coded(b"coded-1".to_vec()),
        Block::Zstd(vector_b_frame()),
        Block::Coded(Vec::new()),
        Block::Zstd(compress_zstd(&data[400..]).unwrap()),
    ];
    Container::build(&ContainerParams::new(model(), bs(100)), &data, &blocks).unwrap()
}

/// 1,000 bytes of phrases (seed 9).
fn vector_c_data() -> Vec<u8> {
    phrases(9, 1_000)
}

/// Version 2 vector C: the raw-only mode, profile logs, blocks of 256 bytes, each block raw or
/// zstd by [`Block::choose_smallest`].
fn vector_c() -> Vec<u8> {
    let data = vector_c_data();
    let blocks: Vec<Block> = bs(256)
        .split(&data)
        .map(|b| Block::choose_smallest(b.len(), None, Some(compress_zstd(b).unwrap())))
        .collect();
    let params = ContainerParams::new(ModelHash::RAW_ONLY, bs(256)).with_profile(Profile::Logs);
    Container::build(&params, &data, &blocks).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn golden_vectors_use_zstd_1_5_7() {
    assert_eq!(zstd_library_version(), ZSTD_LIBRARY_VERSION);
    assert_eq!(ZSTD_LIBRARY_VERSION, 10_507);
}

#[test]
fn vector_a_has_its_version_1_hash() {
    let bytes = vector_a();
    println!("vector A: {} bytes, sha256 {}", bytes.len(), hex(&bytes));
    assert_eq!(bytes.len(), VECTOR_A_LEN);
    assert_eq!(hex(&bytes), VECTOR_A_SHA256);
}

#[test]
fn vector_a_parses_as_version_1_into_its_blocks() {
    let bytes = vector_a();
    let c = Container::parse(&bytes).unwrap();
    let data = vector_a_data();
    assert_eq!(c.header().version(), VERSION_1);
    assert_eq!(c.header().profile(), Profile::Text);
    assert_eq!(c.header().model_hash(), model());
    assert_eq!(c.header().checksum(), Checksum::of(&data));
    let kinds: Vec<BlockKind> = c.blocks().iter().map(|b| b.kind()).collect();
    assert_eq!(kinds, [BlockKind::Raw, BlockKind::Coded, BlockKind::Coded, BlockKind::Raw]);
    assert_eq!(c.blocks()[0].stored(), &data[..100]);
    assert_eq!(c.blocks()[1].stored(), b"coded-1");
    assert!(c.blocks()[2].stored().is_empty());
    assert_eq!(c.blocks()[3].stored(), &data[300..]);
}

#[test]
fn empty_vector_has_its_version_1_hash_and_parses() {
    let bytes = Craft {
        model_hash: [0; 32],
        checksum: *Checksum::of(&[]).as_bytes(),
        block_size: 65_536,
        ..Craft::default()
    }
    .bytes();
    println!("empty vector, version 1: {} bytes, sha256 {}", bytes.len(), hex(&bytes));
    assert_eq!(bytes.len(), 88);
    assert_eq!(hex(&bytes), VECTOR_EMPTY_SHA256);
    let c = Container::parse(&bytes).unwrap();
    assert_eq!(c.header().version(), VERSION_1);
    assert!(c.blocks().is_empty());
    assert_eq!(unfold(&bytes, ModelHash::RAW_ONLY).unwrap(), b"");
}

#[test]
fn vector_b_has_its_hash() {
    let frame = vector_b_frame();
    let bytes = vector_b();
    println!("vector B frame: {} bytes, sha256 {}", frame.len(), hex(&frame));
    println!("vector B: {} bytes, sha256 {}", bytes.len(), hex(&bytes));
    assert_eq!(hex(&frame), VECTOR_B_FRAME_SHA256);
    assert_eq!(hex(&bytes), VECTOR_B_SHA256);
}

#[test]
fn vector_b_parses_and_decodes() {
    let bytes = vector_b();
    let data = vector_b_data();
    let c = Container::parse(&bytes).unwrap();
    assert_eq!(c.header().version(), VERSION);
    assert_eq!(c.header().byte_len(), 104);
    assert_eq!(c.header().profile(), Profile::Text);
    let kinds: Vec<BlockKind> = c.blocks().iter().map(|b| b.kind()).collect();
    assert_eq!(
        kinds,
        [BlockKind::Raw, BlockKind::Coded, BlockKind::Zstd, BlockKind::Coded, BlockKind::Zstd]
    );
    assert_eq!(c.blocks()[2].stored(), vector_b_frame().as_slice());
    let mut out = Vec::new();
    c.blocks()[2].decode_zstd_into(&mut out).unwrap();
    c.blocks()[4].decode_zstd_into(&mut out).unwrap();
    assert_eq!(out, [&data[200..300], &data[400..]].concat());
}

#[test]
fn vector_c_has_its_hash_and_round_trips() {
    let bytes = vector_c();
    println!("vector C: {} bytes, sha256 {}", bytes.len(), hex(&bytes));
    assert_eq!(hex(&bytes), VECTOR_C_SHA256);
    let c = Container::parse(&bytes).unwrap();
    assert_eq!(c.header().profile(), Profile::Logs);
    assert!(!c.header().needs_model());
    assert!(c.blocks().iter().all(|b| b.kind() == BlockKind::Zstd));
    assert_eq!(unfold(&bytes, ModelHash::RAW_ONLY).unwrap(), vector_c_data());
}

#[test]
fn empty_vector_has_its_version_2_hash() {
    let bytes =
        Container::build(&ContainerParams::new(ModelHash::RAW_ONLY, bs(65_536)), &[], &[]).unwrap();
    println!("empty vector, version 2: {} bytes, sha256 {}", bytes.len(), hex(&bytes));
    assert_eq!(bytes.len(), 104);
    assert_eq!(hex(&bytes), VECTOR_EMPTY_V2_SHA256);
}
