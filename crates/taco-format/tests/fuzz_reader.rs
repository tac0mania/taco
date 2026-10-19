//! Long-run fuzz of the container reader (docs/security.md, target T1), and the bound of the
//! zstd block decoder on a hostile frame.
//!
//! Each case comes from one `u64` seed: a valid container of the toy coder of `common` (raw,
//! coded and zstd blocks, versions 1 and 2, the raw-only mode), then one to four mutations
//! (bit flips, truncation, extra bytes, a length field set to a large value, a table entry
//! changed, two table entries swapped, a hash changed). The required behaviour: the parser
//! gives a typed error or a container that obeys the layout rules, and an unfold gives an
//! error or exactly the original data.
//!
//! ```text
//! TACO_FUZZ_CASES=5000000 TACO_FUZZ_SEED=1 cargo test --release -p taco-format \
//!     --test fuzz_reader -- --ignored --nocapture
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::cast_possible_truncation,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{MODEL, bs, fold, phrases, splitmix64, unfold};
use taco_format::{BlockKind, Container, HEADER_LEN, ModelHash, TABLE_ENTRY_LEN};

/// Values that find off-by-one and overflow faults.
const LARGE: [u64; 14] = [
    0,
    1,
    2,
    255,
    256,
    65_535,
    65_536,
    (1 << 31) - 1,
    1 << 31,
    u32::MAX as u64,
    1 << 32,
    i64::MAX as u64,
    1 << 63,
    u64::MAX,
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        splitmix64(&mut self.0)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// A valid container of the toy coder and its original data.
fn valid(rng: &mut Rng) -> (Vec<u8>, Vec<u8>) {
    let len = rng.below(600) as usize;
    let data = match rng.below(3) {
        0 => phrases(rng.next(), len),
        1 => (0..len).map(|i| (i / (1 + rng.below(40) as usize)) as u8).collect(),
        _ => (0..len).map(|_| rng.next() as u8).collect(),
    };
    let model = if rng.below(4) == 0 { ModelHash::RAW_ONLY } else { MODEL };
    let top = [8u64, 64, 300][rng.below(3) as usize];
    let block = 1 + rng.below(top) as u32;
    let mut bytes = fold(&data, bs(block), model);
    // A version 1 copy when the container permits it (no zstd block).
    if rng.below(4) == 0 {
        let c = Container::parse(&bytes).unwrap();
        if c.blocks().iter().all(|b| b.kind() != BlockKind::Zstd) {
            let mut v1 = bytes[..88].to_vec();
            v1[4] = 1;
            v1.extend_from_slice(&bytes[HEADER_LEN..]);
            bytes = v1;
        }
    }
    (data, bytes)
}

fn set_le(bytes: &mut [u8], at: usize, value: u64, size: usize) {
    if let Some(dst) = bytes.get_mut(at..at + size) {
        dst.copy_from_slice(&value.to_le_bytes()[..size]);
    }
}

fn mutate(rng: &mut Rng, bytes: &mut Vec<u8>) {
    if bytes.is_empty() {
        bytes.push(rng.next() as u8);
        return;
    }
    let header = if bytes.get(4) == Some(&1) { 88 } else { HEADER_LEN };
    let blocks = bytes.len().saturating_sub(header) / TABLE_ENTRY_LEN;
    let pos = rng.below(bytes.len() as u64) as usize;
    match rng.below(10) {
        0..=2 => bytes[pos] ^= 1 << rng.below(8),
        3 => bytes.truncate(pos),
        4 => bytes.extend((0..1 + rng.below(6)).map(|_| rng.next() as u8)),
        5 => {
            let (at, size) =
                [(40usize, 8usize), (80, 4), (84, 4), (90, 2), (94, 2)][rng.below(5) as usize];
            set_le(bytes, at, LARGE[rng.below(LARGE.len() as u64) as usize], size);
        }
        6 if blocks > 0 => {
            // A table entry: the kind or the stored length.
            let e = header + TABLE_ENTRY_LEN * rng.below(blocks.min(64) as u64) as usize;
            if rng.below(2) == 0 {
                set_le(bytes, e, rng.below(6), 1);
            } else {
                set_le(bytes, e + 1, LARGE[rng.below(LARGE.len() as u64) as usize], 4);
            }
        }
        7 if blocks > 1 => {
            // Two table entries swapped.
            let n = blocks.min(64) as u64;
            let (i, j) = (rng.below(n) as usize, rng.below(n) as usize);
            let (a, b) = (header + TABLE_ENTRY_LEN * i, header + TABLE_ENTRY_LEN * j);
            if let (Some(x), Some(y)) =
                (bytes.get(a..a + 5).map(<[u8]>::to_vec), bytes.get(b..b + 5).map(<[u8]>::to_vec))
            {
                bytes[a..a + 5].copy_from_slice(&y);
                bytes[b..b + 5].copy_from_slice(&x);
            }
        }
        8 => {
            // A hash: the model hash or the checksum.
            let at = if rng.below(2) == 0 { 8 } else { 48 };
            let fill = rng.below(2) == 0;
            for k in 0..32 {
                let v = if fill { 0 } else { rng.next() as u8 };
                set_le(bytes, at + k, u64::from(v), 1);
            }
        }
        _ => bytes.insert(pos, rng.next() as u8),
    }
}

/// One case. Gives `true` if the damaged container unfolded (to the original data).
fn run_case(seed: u64) -> bool {
    let mut rng = Rng(seed);
    let (data, mut bytes) = valid(&mut rng);
    for _ in 0..1 + rng.below(4) {
        mutate(&mut rng, &mut bytes);
    }
    if let Ok(c) = Container::parse(&bytes) {
        let stored: usize = c.blocks().iter().map(|b| b.stored().len()).sum();
        let h = c.header();
        assert_eq!(
            bytes.len(),
            h.byte_len() + TABLE_ENTRY_LEN * c.blocks().len() + stored,
            "seed {seed}"
        );
        assert_eq!(
            c.blocks().len() as u64,
            h.original_len().div_ceil(u64::from(h.block_size().get()))
        );
        let total: u64 = c.blocks().iter().map(|b| u64::from(b.original_len())).sum();
        assert_eq!(total, h.original_len(), "seed {seed}");
        for b in c.blocks() {
            match b.kind() {
                BlockKind::Raw => assert_eq!(b.stored().len(), b.original_len() as usize),
                _ => assert!(b.stored().len() < b.original_len() as usize, "seed {seed}"),
            }
        }
    }
    match unfold(&bytes, MODEL) {
        Ok(out) => {
            assert!(out == data, "seed {seed}: other data");
            true
        }
        Err(_) => false,
    }
}

#[test]
fn short_run_of_the_long_harness() {
    for seed in 0..3_000 {
        run_case(seed);
    }
}

/// A zstd frame of RLE blocks: 4 bytes for each 128 KiB of output (RFC 8878 §3.1.1.2).
fn rle_frame(content: u32, byte: u8) -> Vec<u8> {
    // Magic; descriptor: a 4-byte content size, not single segment; window 2^20 bytes.
    let mut f = vec![0x28, 0xB5, 0x2F, 0xFD, 0x80, 10 << 3];
    f.extend_from_slice(&content.to_le_bytes());
    let mut left = content;
    while left > 0 {
        let n = left.min(128 * 1024);
        left -= n;
        let last = u32::from(left == 0);
        let header = last | (1 << 1) | (n << 3);
        f.extend_from_slice(&header.to_le_bytes()[..3]);
        f.push(byte);
    }
    f
}

#[test]
fn zstd_rle_frame_output_is_bounded_by_the_block_length() {
    // NOTE: docs/security.md finding F1. A frame of about 2 KiB gives 64 MiB: the decoder obeys
    // the block length, but the block length is the only bound. The caller must limit the
    // original length (taco::UnfoldOptions::max_output_len).
    let content = 64 << 20;
    let frame = rle_frame(content, b'A');
    assert!(frame.len() < 2_100, "{}", frame.len());
    let data = vec![b'A'; content as usize];
    let blocks = [taco_format::Block::Zstd(frame.clone())];
    let params = taco_format::ContainerParams::new(ModelHash::RAW_ONLY, bs(content));
    let bytes = Container::build(&params, &data, &blocks).unwrap();
    assert!(bytes.len() < 2_300);
    let c = Container::parse(&bytes).unwrap();
    let mut out = Vec::new();
    c.blocks()[0].decode_zstd_into(&mut out).unwrap();
    assert_eq!(out.len(), content as usize);
}

/// The number of cases and the first seed of a long run.
fn long_run_params(default_cases: u64) -> (u64, u64) {
    let get = |name: &str, default: u64| {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    (get("TACO_FUZZ_CASES", default_cases), get("TACO_FUZZ_SEED", 1))
}

#[test]
#[ignore = "long run: start it with --ignored (docs/security.md)"]
fn long_run_reader() {
    let (cases, first) = long_run_params(1_000_000);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    let (ok, failed) = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                s.spawn(move || {
                    let (mut ok, mut failed) = (0u64, Vec::new());
                    let mut i = k;
                    while i < cases {
                        let seed = first.wrapping_add(i);
                        match std::panic::catch_unwind(|| run_case(seed)) {
                            Ok(true) => ok += 1,
                            Ok(false) => {}
                            Err(_) => failed.push(seed),
                        }
                        if k == 0 && i % (threads * 100_000) == 0 {
                            eprintln!("fuzz_reader: {i} of {cases}");
                        }
                        i += threads;
                    }
                    (ok, failed)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).fold(
            (0, Vec::new()),
            |(a, mut f), (b, g)| {
                f.extend(g);
                (a + b, f)
            },
        )
    });
    println!(
        "fuzz_reader: {cases} cases from seed {first}: {ok} gave the original data, the others an error; {} failed",
        failed.len()
    );
    assert!(failed.is_empty(), "failed seeds: {failed:?}");
}
