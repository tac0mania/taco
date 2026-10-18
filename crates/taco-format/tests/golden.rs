//! Golden vectors: fixed containers with their SHA-256 (plan §11). A change to the layout
//! changes a hash, and it needs a new container version (plan §3 rule 8).
//!
//! The coded blocks hold fixed bytes, not the output of a real coder: the vector tests the
//! container only.
//!
//! NOTE: a separate Python copy of the layout in the crate documentation gave the same two
//! hashes on 2026-10-18 (measured).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::bs;
use sha2::{Digest, Sha256};
use taco_format::{Block, BlockKind, Checksum, Container, ContainerParams, ModelHash};

/// SHA-256 of the container of [`vector_a`].
const VECTOR_A_SHA256: &str = "c1c07402a5dae2993c596345a696d35b7c1c0502fe304d31fa33d4f5ecefeeec";
/// The length of the container of [`vector_a`]: 88 + 4 * 5 + 100 + 7 + 0 + 50.
const VECTOR_A_LEN: usize = 265;
/// SHA-256 of the container of empty data in the raw-only mode, block size 65,536.
const VECTOR_EMPTY_SHA256: &str =
    "2f8a2052a618a4ca1adfc67aa07836c7d4fda0b10738142e9327228ddd0bcdf7";

/// 350 bytes: byte `i` is `(i * 7 + 3) mod 256`.
fn vector_a_data() -> Vec<u8> {
    (0..350u32).map(|i| u8::try_from((i * 7 + 3) % 256).unwrap()).collect()
}

/// Model hash bytes 0, 1, ..., 31. Blocks of 100 bytes: raw, coded `"coded-1"`, coded with
/// 0 bytes, raw (50 bytes).
fn vector_a() -> Vec<u8> {
    let model = ModelHash::from_bytes(core::array::from_fn(|i| u8::try_from(i).unwrap()));
    let blocks =
        [Block::Raw, Block::Coded(b"coded-1".to_vec()), Block::Coded(Vec::new()), Block::Raw];
    Container::build(&ContainerParams::new(model, bs(100)), &vector_a_data(), &blocks).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn vector_a_has_its_hash() {
    let bytes = vector_a();
    println!("vector A: {} bytes, sha256 {}", bytes.len(), hex(&bytes));
    assert_eq!(bytes.len(), VECTOR_A_LEN);
    assert_eq!(hex(&bytes), VECTOR_A_SHA256);
}

#[test]
fn vector_a_parses_into_its_blocks() {
    let bytes = vector_a();
    let c = Container::parse(&bytes).unwrap();
    let data = vector_a_data();
    assert_eq!(c.header().checksum(), Checksum::of(&data));
    let kinds: Vec<BlockKind> = c.blocks().iter().map(|b| b.kind()).collect();
    assert_eq!(kinds, [BlockKind::Raw, BlockKind::Coded, BlockKind::Coded, BlockKind::Raw]);
    assert_eq!(c.blocks()[0].stored(), &data[..100]);
    assert_eq!(c.blocks()[1].stored(), b"coded-1");
    assert!(c.blocks()[2].stored().is_empty());
    assert_eq!(c.blocks()[3].stored(), &data[300..]);
}

#[test]
fn empty_vector_has_its_hash() {
    let bytes =
        Container::build(&ContainerParams::new(ModelHash::RAW_ONLY, bs(65_536)), &[], &[]).unwrap();
    println!("empty vector: {} bytes, sha256 {}", bytes.len(), hex(&bytes));
    assert_eq!(bytes.len(), 88);
    assert_eq!(hex(&bytes), VECTOR_EMPTY_SHA256);
}
