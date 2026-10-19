//! Fuzz tests of `unfold` on damaged containers, and of the round trip on hostile original
//! data (docs/security.md, targets T1 and T2), with the model `conformance/tiny.hsm`.
//!
//! A pool of valid containers (raw, zstd, coded and mixed blocks; the profiles `text` and
//! `logs`; overlaps 0 to 62; versions 1 and 2) is made one time. Each case damages one
//! container with one to three mutations. Most mutations know the layout: they change a header
//! field, a table entry, a block (with the lengths made to agree again, so the damage reaches
//! the block decoders), swap two blocks, or set a length field, the token count or a hash to a
//! large or wrong value.
//!
//! The required behaviour: `unfold` gives a typed error or exactly the original data. It never
//! panics, never gives other data, and each case is fast (the time of the slowest case is
//! printed). The round-trip cases fold random, invalid UTF-8, long-run and Unicode data, and
//! unfold must restore it byte for byte.
//!
//! Each case comes from one `u64` seed. The long run is `#[ignore]`:
//!
//! ```text
//! TACO_FUZZ_CASES=2000000 TACO_FUZZ_SEED=1 cargo test --release -p taco --test fuzz_unfold \
//!     -- --ignored --nocapture
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

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use proptest::prelude::*;
use taco::{
    BlockSize, FoldOptions, FoldProfile, Model, Overlap, TokenId, Tokenizer, UnfoldOptions, fold,
    unfold,
};
use taco_format::Container;

/// The committed tiny model (conformance/README.md).
const TINY: &[u8] = include_bytes!("../../../conformance/tiny.hsm");

/// The splitmix64 generator: integer only, the same on every machine.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// Values that find off-by-one and overflow faults.
const LARGE: [u64; 18] = [
    0,
    1,
    2,
    63,
    64,
    255,
    256,
    65_535,
    65_536,
    (1 << 31) - 1,
    1 << 31,
    u32::MAX as u64 - 1,
    u32::MAX as u64,
    1 << 32,
    1 << 40,
    i64::MAX as u64,
    1 << 63,
    u64::MAX,
];

struct Tiny {
    model: Model,
    tokenizer: Tokenizer,
}

fn tiny() -> &'static Tiny {
    static T: OnceLock<Tiny> = OnceLock::new();
    T.get_or_init(|| {
        let model = Model::load(TINY).unwrap();
        let merges: Vec<(TokenId, TokenId)> = model
            .vocabulary()
            .merges()
            .unwrap_or(&[])
            .iter()
            .map(|m| (m.left(), m.right()))
            .collect();
        let tokenizer =
            Tokenizer::for_vocabulary(model.vocabulary(), model.eot_token(), &merges).unwrap();
        Tiny { model, tokenizer }
    })
}

/// The bytes of the merged (multi-byte) tokens of the tiny model. The random model codes
/// single bytes at about 8 bits each, so only text of merged tokens codes smaller than raw.
fn merged_tokens() -> Vec<Vec<u8>> {
    let m = &tiny().model;
    (256..m.vocab_size() as u32)
        .filter_map(|i| m.vocabulary().token(TokenId::new(i)))
        .filter(|b| b.len() > 1)
        .map(<[u8]>::to_vec)
        .collect()
}

/// Original data for a valid container or a round trip, of the class `class`.
fn data(rng: &mut Rng, class: u64, len: usize) -> Vec<u8> {
    let merged = merged_tokens();
    let mut out = Vec::new();
    match class {
        // Random merged tokens: coded blocks.
        0 => {
            while out.len() < len {
                out.extend_from_slice(rng.pick(&merged));
            }
        }
        // A few phrases of merged tokens, repeated: zstd and mixed blocks.
        1 => {
            let phrases: Vec<Vec<u8>> = (0..3)
                .map(|_| (0..4 + rng.below(8)).flat_map(|_| rng.pick(&merged).clone()).collect())
                .collect();
            while out.len() < len {
                out.extend_from_slice(rng.pick(&phrases));
                if rng.below(4) == 0 {
                    out.extend_from_slice(rng.pick(&merged));
                }
            }
        }
        // Random bytes: raw blocks.
        2 => out = rng.bytes(len),
        // Invalid UTF-8 and lone continuation bytes.
        3 => {
            let parts: [&[u8]; 6] =
                [b"\xC3", b"\xE2\x82", b"\xF0\x9F\x98", b"\x80\x80", b"\xFF\xFE", b"\xED\xA0\x80"];
            while out.len() < len {
                out.extend_from_slice(rng.pick(&parts));
                out.extend_from_slice(rng.pick(&merged));
            }
        }
        // One long run of one class: letters, white space, digits, other characters.
        4 => {
            let unit: &[u8] = rng.pick(&[
                &b"a"[..],
                b" ",
                b"\n",
                b"\r\n",
                b"7",
                b"!",
                b"\xC3\xA9",
                b"\xE4\xB8\xAD",
                b"'s",
            ]);
            while out.len() < len {
                out.extend_from_slice(unit);
            }
        }
        // Unicode text with spaces and contractions.
        _ => {
            let words: [&str; 8] =
                ["don't", " we'll", "\u{939}\u{93F}", " 汉字", "\t\t", "  x", "\u{2028}", " é"];
            while out.len() < len {
                out.extend_from_slice(rng.pick(&words).as_bytes());
            }
        }
    }
    out.truncate(len);
    out
}

/// A valid container and its original data.
struct Valid {
    data: Vec<u8>,
    container: Vec<u8>,
}

fn options(rng: &mut Rng) -> FoldOptions {
    let profile = *rng.pick(&[FoldProfile::Text, FoldProfile::Logs, FoldProfile::Auto]);
    let block = *rng.pick(&[16u32, 64, 100, 256, 512, 1000]);
    let k = *rng.pick(&[0u16, 0, 1, 12, 32, 62]);
    FoldOptions::new()
        .with_profile(profile)
        .with_block_size(BlockSize::new(block).unwrap())
        .with_overlap(Overlap::new(k))
}

/// The pool of valid containers, made one time: it holds every block kind.
fn pool() -> &'static [Valid] {
    static P: OnceLock<Vec<Valid>> = OnceLock::new();
    P.get_or_init(|| {
        let t = tiny();
        let mut rng = Rng(0x7AC0_F022);
        let mut pool = Vec::new();
        for i in 0..40u64 {
            let len = *rng.pick(&[0usize, 1, 40, 300, 900, 2_000]);
            let data = data(&mut rng, i % 3, len);
            let container =
                fold(&t.model, &t.tokenizer, &data, &options(&mut rng)).unwrap().container;
            pool.push(Valid { data, container });
        }
        let mut kinds = [false; 4];
        for v in &pool {
            for b in Container::parse(&v.container).unwrap().blocks() {
                kinds[usize::from(b.kind().as_u8())] = true;
            }
        }
        assert_eq!(kinds, [true; 4], "the pool must hold raw, coded, zstd and mixed blocks");
        pool
    })
}

/// A container taken apart: the header bytes and, for each block, its kind and stored bytes.
struct Parts {
    header: Vec<u8>,
    blocks: Vec<(u8, Vec<u8>)>,
}

fn take_apart(bytes: &[u8]) -> Parts {
    let c = Container::parse(bytes).unwrap();
    Parts {
        header: bytes[..c.header().byte_len()].to_vec(),
        blocks: c.blocks().iter().map(|b| (b.kind().as_u8(), b.stored().to_vec())).collect(),
    }
}

fn put_together(p: &Parts) -> Vec<u8> {
    let mut out = p.header.clone();
    for (kind, stored) in &p.blocks {
        out.push(*kind);
        out.extend_from_slice(&(stored.len() as u32).to_le_bytes());
    }
    for (_, stored) in &p.blocks {
        out.extend_from_slice(stored);
    }
    out
}

fn set_le(bytes: &mut [u8], at: usize, value: u64, size: usize) {
    if let Some(dst) = bytes.get_mut(at..at + size) {
        dst.copy_from_slice(&value.to_le_bytes()[..size]);
    }
}

fn varint(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let low = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(low);
            return out;
        }
        out.push(low | 0x80);
    }
}

/// One mutation of a container. Some mutations work on the parts and put them together, so
/// that the layout stays valid and the damage reaches the decoders.
fn mutate(rng: &mut Rng, bytes: &mut Vec<u8>) {
    let parts = Container::parse(bytes).ok().map(|_| take_apart(bytes));
    let kind = rng.below(14);
    match (kind, parts) {
        (0..=2, _) | (_, None) => {
            if bytes.is_empty() {
                bytes.push(0);
            }
            let pos = rng.below(bytes.len() as u64) as usize;
            match rng.below(6) {
                0..=2 => bytes[pos] ^= 1 << rng.below(8),
                3 => bytes[pos] = rng.next() as u8,
                4 => bytes.truncate(pos),
                _ => {
                    let n = 1 + rng.below(8) as usize;
                    bytes.extend(rng.bytes(n));
                }
            }
        }
        (3, Some(_)) => {
            // A header field set to a large or a near value.
            let (at, size) = *rng.pick(&[
                (4usize, 2usize),
                (6, 2),
                (40, 8),
                (80, 4),
                (84, 4),
                (88, 1),
                (89, 1),
                (90, 2),
                (92, 1),
                (93, 1),
                (94, 2),
                (96, 8),
            ]);
            let old = bytes.get(at..at + size).map_or(0, |b| {
                let mut a = [0u8; 8];
                a[..size].copy_from_slice(b);
                u64::from_le_bytes(a)
            });
            let v = match rng.below(3) {
                0 => old.wrapping_add(1),
                1 => old.wrapping_sub(1),
                _ => *rng.pick(&LARGE),
            };
            set_le(bytes, at, v, size);
        }
        (4, Some(mut p)) => {
            // The original length and the block count changed together.
            let bs = u64::from(u32::from_le_bytes(p.header[80..84].try_into().unwrap()));
            let len = *rng.pick(&LARGE);
            set_le(&mut p.header, 40, len, 8);
            set_le(&mut p.header, 84, len.div_ceil(bs).min(u64::from(u32::MAX)), 4);
            if rng.below(2) == 0 {
                let new_bs = *rng.pick(&[1u64, 2, bs / 2, bs * 2, u64::from(u32::MAX)]);
                set_le(&mut p.header, 80, new_bs.max(1), 4);
                set_le(&mut p.header, 84, len.div_ceil(new_bs.max(1)).min(u64::from(u32::MAX)), 4);
            }
            *bytes = put_together(&p);
        }
        (5, Some(mut p)) if !p.blocks.is_empty() => {
            // A block kind changed; the stored bytes stay.
            let i = rng.below(p.blocks.len() as u64) as usize;
            p.blocks[i].0 = *rng.pick(&[0u8, 1, 2, 3, 4, 255]);
            *bytes = put_together(&p);
        }
        (6, Some(mut p)) if !p.blocks.is_empty() => {
            // Stored bytes added, removed or changed, with the table length to agree.
            let i = rng.below(p.blocks.len() as u64) as usize;
            let s = &mut p.blocks[i].1;
            match rng.below(4) {
                0 => s.truncate(rng.below(s.len() as u64 + 1) as usize),
                1 => {
                    let n = 1 + rng.below(4) as usize;
                    s.extend(rng.bytes(n));
                }
                2 => {
                    let n = s.len();
                    *s = rng.bytes(n);
                }
                _ => {
                    if !s.is_empty() {
                        let at = rng.below(s.len() as u64) as usize;
                        s[at] ^= 1 << rng.below(8);
                    }
                }
            }
            *bytes = put_together(&p);
        }
        (7, Some(mut p)) if p.blocks.len() >= 2 => {
            // Two blocks swapped (kind and bytes).
            let n = p.blocks.len() as u64;
            let (i, j) = (rng.below(n) as usize, rng.below(n) as usize);
            p.blocks.swap(i, j);
            *bytes = put_together(&p);
        }
        (8, Some(mut p)) if !p.blocks.is_empty() => {
            // The token count of a coded or mixed block set to another value.
            let i = rng.below(p.blocks.len() as u64) as usize;
            let s = &p.blocks[i].1;
            let rest_at = s.iter().position(|&b| b & 0x80 == 0).map_or(s.len(), |x| x + 1);
            let rest = s.get(rest_at..).unwrap_or(&[]).to_vec();
            let count = *rng.pick(&LARGE);
            let mut new = if rng.below(4) == 0 { vec![0x80, 0x00] } else { varint(count) };
            new.extend_from_slice(&rest);
            p.blocks[i].1 = new;
            *bytes = put_together(&p);
        }
        (9, Some(mut p)) => {
            // A wrong hash: the model hash zero (raw-only) or other, or another checksum.
            let (at, len) = *rng.pick(&[(8usize, 32usize), (48, 32)]);
            let fill = if rng.below(2) == 0 { vec![0u8; len] } else { rng.bytes(len) };
            p.header[at..at + len].copy_from_slice(&fill);
            *bytes = put_together(&p);
        }
        (10, Some(mut p)) if p.header.len() == 104 => {
            // A version 2 container written as version 1.
            p.header.truncate(88);
            p.header[4] = 1;
            *bytes = put_together(&p);
        }
        (11, Some(mut p)) if !p.blocks.is_empty() => {
            // A block duplicated or removed, with the header lengths to agree.
            let old_n = p.blocks.len() as u64;
            let i = rng.below(old_n) as usize;
            if rng.below(2) == 0 {
                let b = p.blocks[i].clone();
                p.blocks.insert(i, b);
            } else {
                p.blocks.remove(i);
            }
            let bs = u64::from(u32::from_le_bytes(p.header[80..84].try_into().unwrap()));
            let len = u64::from_le_bytes(p.header[40..48].try_into().unwrap());
            // The last block keeps its length; the other blocks have the block size.
            let last = len - (old_n - 1) * bs;
            let n = p.blocks.len() as u64;
            let new_len = if n == 0 { 0 } else { (n - 1) * bs + last };
            set_le(&mut p.header, 40, new_len, 8);
            set_le(&mut p.header, 84, n, 4);
            *bytes = put_together(&p);
        }
        (12, Some(mut p)) if p.header.len() == 104 => {
            // The coding parts and the overlap K.
            let parts = *rng.pick(&[0u64, 1, 2, 3, 4]);
            set_le(&mut p.header, 90, parts, 2);
            set_le(&mut p.header, 94, *rng.pick(&[0u64, 1, 12, 62, 63, 64, 65_535]), 2);
            p.header[92] = u8::from(parts & 1 == 1) * *rng.pick(&[1u8, 1, 2]);
            p.header[88] = *rng.pick(&[0u8, 1, 1, 2]);
            *bytes = put_together(&p);
        }
        (_, Some(mut p)) if !p.blocks.is_empty() => {
            // A zstd frame header byte (descriptor, window, content size) changed.
            let i = rng.below(p.blocks.len() as u64) as usize;
            let s = &mut p.blocks[i].1;
            if s.len() > 4 {
                let at = 4 + rng.below((s.len() as u64 - 4).min(10)) as usize;
                s[at] =
                    if rng.below(2) == 0 { rng.next() as u8 } else { s[at] ^ (1 << rng.below(8)) };
            }
            *bytes = put_together(&p);
        }
        _ => {}
    }
}

/// The counts of a run: results by name, and the slowest case.
#[derive(Default)]
struct Tally {
    results: BTreeMap<String, u64>,
    slowest: Duration,
    slowest_seed: u64,
}

impl Tally {
    fn add(&mut self, other: Self) {
        for (k, v) in other.results {
            *self.results.entry(k).or_default() += v;
        }
        if other.slowest > self.slowest {
            self.slowest = other.slowest;
            self.slowest_seed = other.slowest_seed;
        }
    }
}

/// The name of the error variant (the text before its fields).
fn variant(e: &impl std::fmt::Debug) -> String {
    let text = format!("{e:?}");
    let end = text.find([' ', '(', '{']).unwrap_or(text.len());
    let mut name = text[..end].to_owned();
    // The inner error of a wrapper variant.
    if let Some(inner) = text.get(end + 1..)
        && matches!(name.as_str(), "Format" | "Model" | "Coder" | "Match")
    {
        let e2 = inner.find([' ', '(', '{', ')']).unwrap_or(inner.len());
        name = format!("{name}::{}", &inner[..e2]);
    }
    name
}

/// One damage case. Panics (with the seed) on a wrong output.
fn damage_case(seed: u64, tally: &mut Tally) {
    let t = tiny();
    let pool = pool();
    let mut rng = Rng(seed);
    let valid = rng.pick(pool);
    let mut bytes = valid.container.clone();
    for _ in 0..1 + rng.below(3) {
        mutate(&mut rng, &mut bytes);
    }
    let limit = if rng.below(4) == 0 { None } else { Some(1 << 20) };
    let options = UnfoldOptions::new().with_max_output_len(limit);
    let model = if rng.below(8) == 0 { None } else { Some(&t.model) };
    let start = Instant::now();
    let result = unfold(model, &bytes, &options);
    let took = start.elapsed();
    if took > tally.slowest {
        tally.slowest = took;
        tally.slowest_seed = seed;
    }
    let name = match &result {
        Ok(out) => {
            assert!(
                *out == valid.data,
                "seed {seed}: unfold gave other data ({} bytes)",
                out.len()
            );
            "Ok (the original data)".to_owned()
        }
        Err(e) => {
            // An error message is short and names offsets, lengths and hashes only.
            assert!(e.to_string().len() <= 400, "seed {seed}: long error {e}");
            variant(e)
        }
    };
    *tally.results.entry(name).or_default() += 1;
}

/// One round trip of hostile original data. Panics (with the seed) if the data does not come
/// back.
fn round_trip_case(seed: u64, tally: &mut Tally) {
    let t = tiny();
    let mut rng = Rng(seed);
    let len = *rng.pick(&[0usize, 1, 2, 17, 200, 700, 1_500]);
    let class = rng.below(6);
    let data = data(&mut rng, class, len);
    let start = Instant::now();
    let folded = fold(&t.model, &t.tokenizer, &data, &options(&mut rng)).unwrap();
    let back = unfold(Some(&t.model), &folded.container, &UnfoldOptions::new()).unwrap();
    assert!(back == data, "seed {seed}: the round trip changed the data");
    let took = start.elapsed();
    if took > tally.slowest {
        tally.slowest = took;
        tally.slowest_seed = seed;
    }
    *tally.results.entry(format!("round trip, class {class}")).or_default() += 1;
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn damaged_container_gives_an_error_or_the_original(seed in any::<u64>()) {
        damage_case(seed, &mut Tally::default());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn hostile_original_data_round_trips(seed in any::<u64>()) {
        round_trip_case(seed, &mut Tally::default());
    }
}

/// The number of cases and the first seed of a long run, from `TACO_FUZZ_CASES` and
/// `TACO_FUZZ_SEED`. `TACO_FUZZ_TRACE=1` prints each seed before its case.
fn long_run_params(default_cases: u64) -> (u64, u64, bool) {
    let get = |name: &str, default: u64| {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    (
        get("TACO_FUZZ_CASES", default_cases),
        get("TACO_FUZZ_SEED", 1),
        get("TACO_FUZZ_TRACE", 0) == 1,
    )
}

/// Runs `case` for `cases` seeds on all processors, and lists the results and the seeds that
/// failed. A panic does not stop the run.
fn long_run(name: &str, cases: u64, first: u64, trace: bool, case: fn(u64, &mut Tally)) {
    pool();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    let results = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                s.spawn(move || {
                    let mut tally = Tally::default();
                    let mut failed = Vec::new();
                    let mut i = k;
                    while i < cases {
                        let seed = first.wrapping_add(i);
                        if trace {
                            eprintln!("seed {seed}");
                        }
                        let mut one = Tally::default();
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            case(seed, &mut one)
                        })) {
                            Ok(()) => tally.add(one),
                            Err(_) => failed.push(seed),
                        }
                        if k == 0 && i % (threads * 20_000) == 0 {
                            eprintln!("{name}: {i} of {cases}");
                        }
                        i += threads;
                    }
                    (tally, failed)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    let mut tally = Tally::default();
    let mut failed = Vec::new();
    for (t, f) in results {
        tally.add(t);
        failed.extend(f);
    }
    println!("{name}: {cases} cases from seed {first}, {} failed", failed.len());
    for (k, v) in &tally.results {
        println!("  {v:>10}  {k}");
    }
    println!("  slowest case: {:?} (seed {})", tally.slowest, tally.slowest_seed);
    assert!(failed.is_empty(), "failed seeds: {failed:?}");
}

#[test]
#[ignore = "long run: start it with --ignored (docs/security.md)"]
fn long_run_unfold_damage() {
    let (cases, first, trace) = long_run_params(1_000_000);
    long_run("fuzz_unfold damage", cases, first, trace, damage_case);
}

#[test]
#[ignore = "long run: start it with --ignored (docs/security.md)"]
fn long_run_round_trip() {
    let (cases, first, trace) = long_run_params(20_000);
    long_run("fuzz_unfold round trip", cases, first, trace, round_trip_case);
}
