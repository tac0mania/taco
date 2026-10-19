//! Regression tests for the findings of docs/security.md in `unfold`: hostile containers that
//! state much more work or memory than their own length.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    reason = "test code: a failure must stop the test with a message"
)]

use std::time::{Duration, Instant};

use taco::{Model, TacoError, UnfoldOptions, container_hash, unfold};
use taco_format::FormatError;

/// The committed tiny model (conformance/README.md).
const TINY: &[u8] = include_bytes!("../../../conformance/tiny.hsm");

/// A zstd frame of RLE blocks: 4 bytes for each 128 KiB of output (RFC 8878 §3.1.1.2).
fn rle_frame(content: u32, byte: u8) -> Vec<u8> {
    // Magic; descriptor: a 4-byte content size, not single segment; window 2^20 bytes.
    let mut f = vec![0x28, 0xB5, 0x2F, 0xFD, 0x80, 10 << 3];
    f.extend_from_slice(&content.to_le_bytes());
    let mut left = content;
    while left > 0 {
        let n = left.min(128 * 1024);
        left -= n;
        let header = u32::from(left == 0) | (1 << 1) | (n << 3);
        f.extend_from_slice(&header.to_le_bytes()[..3]);
        f.push(byte);
    }
    f
}

/// A version 2 container from its fields, with no checks. `parts` is `(profile, coding parts,
/// mixer rule)`.
fn craft(
    model: [u8; 32],
    original_len: u64,
    block_size: u32,
    parts: (u8, u16, u8),
    blocks: &[(u8, Vec<u8>)],
) -> Vec<u8> {
    let mut out = b"TACO".to_vec();
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&model);
    out.extend_from_slice(&original_len.to_le_bytes());
    out.extend_from_slice(&[0x11; 32]);
    out.extend_from_slice(&block_size.to_le_bytes());
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    out.extend_from_slice(&[parts.0, 0]);
    out.extend_from_slice(&parts.1.to_le_bytes());
    let mut params = [0u8; 12];
    params[0] = parts.2;
    out.extend_from_slice(&params);
    for (kind, stored) in blocks {
        out.push(*kind);
        out.extend_from_slice(&(stored.len() as u32).to_le_bytes());
    }
    for (_, stored) in blocks {
        out.extend_from_slice(stored);
    }
    out
}

/// Finding F1, the defence of the library: a container of 32 KiB that states 1 GiB is refused
/// before any block decodes when the caller sets a limit. The bomb uses 16 zstd blocks of
/// 64 MiB, the largest block size that a reader accepts (ADR 0011).
#[test]
fn output_limit_refuses_a_zstd_bomb_before_it_decodes() {
    let block = 64u32 << 20;
    let len = 1u32 << 30;
    let blocks: Vec<(u8, Vec<u8>)> =
        (0..len / block).map(|_| (2, rle_frame(block, b'A'))).collect();
    let stored: usize = blocks.iter().map(|(_, f)| f.len()).sum();
    assert!(stored < 33_000, "{stored}");
    let bomb = craft([0; 32], u64::from(len), block, (0, 0, 0), &blocks);
    let start = Instant::now();
    let limit = UnfoldOptions::new().with_max_output_len(Some(1 << 20));
    match unfold(None, &bomb, &limit) {
        Err(TacoError::OutputTooLarge { original_len, limit }) => {
            assert_eq!((original_len, limit), (1 << 30, 1 << 20));
        }
        other => panic!("{other:?}"),
    }
    assert!(start.elapsed() < Duration::from_secs(1));
}

/// 20,000 mixed blocks of 2 bytes each, in a container of 120 KB (finding F2). Each stored
/// block is the token count 1 and no coder bytes.
fn many_mixed_blocks(model: &Model) -> Vec<u8> {
    let n = 20_000u32;
    let blocks: Vec<(u8, Vec<u8>)> = (0..n).map(|_| (3, vec![0x01])).collect();
    let hash = *container_hash(model).as_bytes();
    craft(hash, u64::from(n) * 2, 2, (1, 1, 1), &blocks)
}

/// Finding F2, the CPU path: the decoder makes the mixer states of one group of blocks at a
/// time and stops at the first damaged block, so the time is small.
#[test]
fn many_mixed_blocks_on_the_cpu_fail_fast() {
    let model = Model::load(TINY).unwrap();
    let bytes = many_mixed_blocks(&model);
    assert!(bytes.len() < 130_000);
    let start = Instant::now();
    let r = unfold(Some(&model), &bytes, &UnfoldOptions::new());
    assert!(
        matches!(r, Err(TacoError::CorruptBlock { .. } | TacoError::ChecksumMismatch { .. })),
        "{r:?}"
    );
    assert!(start.elapsed() < Duration::from_secs(20), "{:?}", start.elapsed());
}

/// The peak resident memory of this process (`VmHWM`, Linux), in KiB.
#[cfg(target_os = "linux")]
fn peak_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmHWM:")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

/// Finding F2, the GPU path: `decode_on_gpu` makes a `BlockDecoder`, with a new `MixState`
/// (the logistic tables, 144 KiB, and 24 MiB of zeroed match tables), for every mixed block
/// before the first slot runs. 20,000 blocks of 2 bytes (a container of 120 KB) then need about
/// 2.8 GiB of resident memory and 480 GiB of address space (est.). The required behaviour: the
/// memory does not grow with the number of blocks; make the state when a block enters a slot.
/// Repaired in `gpu_unfold.rs`: the state is made in the slot, and at most `MAX_MIXED_SLOTS`
/// (12) mixed blocks decode at the same time (gpucli2). On a machine without a GPU the test
/// does nothing.
#[cfg(target_os = "linux")]
#[test]
fn many_mixed_blocks_on_a_gpu_use_bounded_memory() {
    use taco::{BackendChoice, unfold_with, with_engine};
    let model = Model::load(TINY).unwrap();
    let bytes = many_mixed_blocks(&model);
    let before = peak_kib();
    for choice in [BackendChoice::Cuda, BackendChoice::Gpu] {
        let r = with_engine(choice, TINY, &model, |engine| {
            unfold_with(engine, Some(&model), &bytes, &UnfoldOptions::new())
        });
        if let Err(TacoError::Backend { .. }) = r {
            continue;
        }
        assert!(r.is_err(), "the blocks are damaged");
        let grew = peak_kib().saturating_sub(before);
        assert!(grew < 512 * 1024, "{choice:?}: the peak memory grew by {grew} KiB");
    }
}

/// Finding F5 (a property of the format, documented): a coded block of 3 bytes (only the token
/// count) decodes its full stated length. The range decoder reads zero bytes after the end, so
/// each step gives token 0 (one byte) and the block decodes to the stated length. Only the
/// checksum at the end finds the damage. Thus the work is one model step for each stated
/// byte, not for each stored byte.
#[test]
fn coded_block_of_three_bytes_costs_one_step_for_each_stated_byte() {
    let model = Model::load(TINY).unwrap();
    let n = 20_000u32;
    let mut count = Vec::new();
    let mut v = n;
    while v >= 0x80 {
        count.push((v & 0x7F) as u8 | 0x80);
        v >>= 7;
    }
    count.push(v as u8);
    assert_eq!(count.len(), 3);
    let hash = *container_hash(&model).as_bytes();
    let bytes = craft(hash, u64::from(n), n, (0, 0, 0), &[(1, count)]);
    assert!(bytes.len() < 120);
    let start = Instant::now();
    let r = unfold(Some(&model), &bytes, &UnfoldOptions::new());
    assert!(matches!(r, Err(TacoError::ChecksumMismatch { .. })), "{r:?}");
    let took = start.elapsed();
    // NOTE: the time is the time of 20,000 steps of the tiny model; it is printed for the
    // record in docs/security.md.
    #[allow(clippy::print_stderr, reason = "a measurement for docs/security.md")]
    {
        eprintln!("F5: a container of {} bytes cost {took:?} for {n} model steps", bytes.len());
    }
}

/// The zstd error of a frame that states more than its block, for the documentation of F1: the
/// decoder never gives more than the block length.
#[test]
fn zstd_frame_longer_than_its_block_is_refused() {
    let frame = rle_frame(4096, b'A');
    let bytes = craft([0; 32], 4000, 4000, (0, 0, 0), &[(2, frame)]);
    match unfold(None, &bytes, &UnfoldOptions::new()) {
        Err(TacoError::Format(FormatError::CorruptZstdBlock { .. })) => {}
        other => panic!("{other:?}"),
    }
}
