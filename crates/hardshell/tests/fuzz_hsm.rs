//! Fuzz tests of the `.hsm` reader and of the model on damaged model files (docs/security.md,
//! target T3).
//!
//! Each case damages `conformance/tiny.hsm` with one or more mutations:
//!
//! - bytes: bit flips, set bytes, truncation, extra bytes, an inserted or removed byte, and a
//!   `u32` or `u64` set to a large value, with a bias to the preamble, the header, the
//!   vocabulary and the merges;
//! - the JSON header: a number set to a large or a small value, a string changed, a field
//!   removed, added or duplicated, a shape or the tensor list changed, the constants of format
//!   version 2. The other sections then move, so that the layout stays canonical and the
//!   reader checks the next rule;
//! - tensor values: entries set to extreme values (`i8::MIN`, `i32::MAX`, ...).
//!
//! The required behaviour: [`HsmFile::parse`] and [`Model::load`] give a typed error or a
//! value, and never panic. A model that loads gives a vector of `vocab_size` weights for each
//! step, and every path (the reference, the portable and SIMD sessions, `step_many`, a batch
//! lane) gives the same weights, bit for bit (CLAUDE.md §1 rule 10). The weights obey the
//! documented bound: the largest weight is at least 1 and the sum is at most `2^60`.
//!
//! Each case comes from one `u64` seed. The long run is `#[ignore]`:
//!
//! ```text
//! TACO_FUZZ_CASES=1000000 TACO_FUZZ_SEED=1 cargo test -p hardshell --test fuzz_hsm -- \
//!     --ignored --nocapture
//! ```
//!
//! NOTE: run the long run in the dev profile (overflow checks on), not with `--release`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test code: a failure must stop the test with a message"
)]

use hardshell::Error;
use hardshell::gpt2::{
    Batch, BatchInput, Model, Overlap, ReferenceSession, Session, SessionConfig,
};
use hardshell::hsm::HsmFile;
use hardshell::kernels::Kernels;
use hardshell::types::TokenId;
use proptest::prelude::*;
use serde_json::{Value, json};

/// The committed tiny model (306,368 bytes, conformance/README.md).
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
}

/// Values that find off-by-one and overflow faults.
const LARGE: [u64; 22] = [
    0,
    1,
    2,
    63,
    64,
    65,
    127,
    128,
    255,
    256,
    65_535,
    65_536,
    1 << 20,
    (1 << 20) + 1,
    (1 << 31) - 1,
    1 << 31,
    u32::MAX as u64,
    1 << 32,
    1 << 53,
    i64::MAX as u64,
    1 << 63,
    u64::MAX,
];

fn header_len(f: &[u8]) -> usize {
    u32::from_le_bytes(f[12..16].try_into().unwrap()) as usize
}

fn header_value(f: &[u8]) -> Value {
    serde_json::from_slice(&f[16..16 + header_len(f)]).unwrap()
}

fn align64(x: u64) -> u64 {
    x.div_ceil(64) * 64
}

/// Adds `delta` to every `offset` field of the header, where the sum is a `u64`.
fn shift_offsets(v: &mut Value, delta: i128) {
    let shift = |x: &mut Value| {
        if let Some(n) = x.as_u64()
            && let Ok(m) = u64::try_from(i128::from(n) + delta)
        {
            *x = json!(m);
        }
    };
    if let Some(o) = v.pointer_mut("/vocabulary/offset") {
        shift(o);
    }
    if let Some(o) = v.pointer_mut("/merges/offset") {
        shift(o);
    }
    if let Some(ts) = v.get_mut("tensors").and_then(Value::as_array_mut)
        && !ts.is_empty()
    {
        for t in ts {
            if let Some(o) = t.get_mut("offset") {
                shift(o);
            }
        }
    }
}

/// The file `orig` with the header `text`: the sections after the header move by the change of
/// `align64(16 + header_len)`, and the offsets in `v` move with them (`relocate`), so that a
/// header edit does not by itself break the layout.
fn relayout(
    orig: &[u8],
    v: &Value,
    relocate: bool,
    edit_text: &dyn Fn(String) -> String,
) -> Vec<u8> {
    let old_vocab = align64(16 + header_len(orig) as u64);
    let mut delta: i128 = 0;
    let mut text = String::new();
    for _ in 0..8 {
        let mut w = v.clone();
        if relocate {
            shift_offsets(&mut w, delta);
        }
        text = edit_text(serde_json::to_string(&w).unwrap());
        let new_delta = i128::from(align64(16 + text.len() as u64)) - i128::from(old_vocab);
        if new_delta == delta || !relocate {
            break;
        }
        delta = new_delta;
    }
    let mut out = orig[..12].to_vec();
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
    out.resize(align64(out.len() as u64) as usize, 0);
    out.extend_from_slice(&orig[old_vocab as usize..]);
    out
}

/// Every JSON path to a value of the header, as a pointer.
fn paths(v: &Value, at: String, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                paths(x, format!("{at}/{k}"), out);
            }
        }
        Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                paths(x, format!("{at}/{i}"), out);
            }
        }
        _ => {}
    }
    out.push(at);
}

/// One mutation of the JSON header value.
fn mutate_value(rng: &mut Rng, v: &mut Value) {
    let mut all = Vec::new();
    paths(v, String::new(), &mut all);
    let path = rng.pick(&all).clone();
    let strings = ["i8", "i16", "i32", "u64", "f32", "", "gpt2", "wte", "h.0.mlp.gelu", "x"];
    match rng.below(12) {
        0..=3 => {
            let n = *rng.pick(&LARGE);
            if let Some(x) = v.pointer_mut(&path) {
                *x = match rng.below(6) {
                    0 => json!(-1),
                    1 => json!(1.5),
                    2 => x
                        .as_u64()
                        .map_or(json!(n), |o| json!(o.wrapping_add(rng.below(3)).wrapping_sub(1))),
                    _ => json!(n),
                };
            }
        }
        4 => {
            if let Some(x) = v.pointer_mut(&path) {
                *x = match rng.below(4) {
                    0 => json!("a".repeat(1 + rng.below(300) as usize)),
                    1 => json!("\u{e9}t\u{e9}"),
                    _ => json!(rng.pick(&strings)),
                };
            }
        }
        5 => {
            // Remove a field or an array entry.
            if let Some((parent, key)) = path.rsplit_once('/')
                && let Some(p) = v.pointer_mut(parent)
            {
                match p {
                    Value::Object(m) => {
                        m.remove(key);
                    }
                    Value::Array(a) => {
                        if let Ok(i) = key.parse::<usize>()
                            && i < a.len()
                        {
                            a.remove(i);
                        }
                    }
                    _ => {}
                }
            }
        }
        6 => {
            // An unknown field, or a version 2 constant.
            let c = v.pointer_mut("/model/constants").and_then(Value::as_object_mut);
            if let Some(c) = c {
                match rng.below(3) {
                    0 => c.insert("ln_f_bits".into(), json!(*rng.pick(&[8u64, 16, 0, 255]))),
                    1 => c.insert("ln_beta_frac".into(), json!(rng.below(20))),
                    _ => c.insert("bogus".into(), json!(1)),
                };
            }
        }
        7 => {
            // A shape with another number of dimensions or another dimension.
            if let Some(ts) = v.get_mut("tensors").and_then(Value::as_array_mut)
                && !ts.is_empty()
            {
                let i = rng.below(ts.len() as u64) as usize;
                if let Some(s) = ts[i].get_mut("shape").and_then(Value::as_array_mut) {
                    match rng.below(4) {
                        0 => s.push(json!(*rng.pick(&LARGE))),
                        1 => {
                            s.pop();
                        }
                        2 => s.reverse(),
                        _ => s.clear(),
                    }
                }
            }
        }
        8 => {
            // Swap, duplicate or remove tensor entries.
            if let Some(ts) = v.get_mut("tensors").and_then(Value::as_array_mut)
                && !ts.is_empty()
            {
                let n = ts.len() as u64;
                let (i, j) = (rng.below(n) as usize, rng.below(n) as usize);
                match rng.below(3) {
                    0 => ts.swap(i, j),
                    1 => {
                        let t = ts[i].clone();
                        ts.insert(j, t);
                    }
                    _ => {
                        ts.remove(i);
                    }
                }
            }
        }
        9 => {
            // A shift on a tensor, with a value in 0..=255 (the rescale shift is a u8).
            if let Some(ts) = v.get_mut("tensors").and_then(Value::as_array_mut)
                && !ts.is_empty()
            {
                let i = rng.below(ts.len() as u64) as usize;
                let s = *rng.pick(&[0u64, 1, 31, 62, 63, 64, 126, 127, 128, 200, 255, 256]);
                if let Some(t) = ts[i].as_object_mut() {
                    t.insert("shift".into(), json!(s));
                }
            }
        }
        10 => {
            // Remove the merges section from the header (the bytes stay).
            if let Some(m) = v.as_object_mut() {
                m.remove("merges");
            }
        }
        _ => {
            // Dimensions of the model that agree with each other but not with the tensors.
            let field = *rng.pick(&[
                "n_layer",
                "n_head",
                "d_model",
                "d_ff",
                "vocab_size",
                "n_ctx",
                "eot_token",
            ]);
            if let Some(x) = v.pointer_mut(&format!("/model/{field}")) {
                *x = json!(*rng.pick(&LARGE) & 0xFFFF_FFFF);
            }
        }
    }
}

/// A text edit of the serialised header: a duplicate field, a deep nesting, or nothing.
fn text_edit(kind: u64) -> impl Fn(String) -> String {
    move |s: String| match kind {
        0 => s.replacen("{\"architecture\"", "{\"n_layer\":2,\"architecture\"", 1),
        1 => s.replacen("\"shape\":[", "\"shape\":[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1", 1),
        2 => format!("{s}   "),
        3 => s.replacen(':', ": ", 3),
        _ => s,
    }
}

/// The tensors of the header: name, dtype, offset and length.
fn tensor_spans(f: &[u8]) -> Vec<(String, String, usize, usize)> {
    header_value(f)["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t["name"].as_str().unwrap().to_owned(),
                t["dtype"].as_str().unwrap().to_owned(),
                t["offset"].as_u64().unwrap() as usize,
                t["length"].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

/// Sets tensor entries to extreme or random values. The layout does not change.
fn mutate_tensors(rng: &mut Rng, f: &mut [u8]) {
    let spans = tensor_spans(TINY);
    for _ in 0..1 + rng.below(4) {
        let (_, dtype, offset, length) = rng.pick(&spans).clone();
        let size = match dtype.as_str() {
            "i8" => 1,
            "i16" => 2,
            "i32" => 4,
            _ => 8,
        };
        let extremes: Vec<Vec<u8>> = match size {
            1 => vec![vec![0x80], vec![0x7F], vec![0], vec![0xFF], vec![0x81]],
            2 => [i16::MIN, i16::MAX, -i16::MAX, 0, -1, 1]
                .iter()
                .map(|v| v.to_le_bytes().to_vec())
                .collect(),
            4 => [i32::MIN, i32::MAX, -i32::MAX, 0, -1, 1, 1 << 30, -(1 << 30)]
                .iter()
                .map(|v| v.to_le_bytes().to_vec())
                .collect(),
            _ => [u64::MAX, 0, 1, 1 << 40, (1 << 40) + 1, 1 << 63]
                .iter()
                .map(|v| v.to_le_bytes().to_vec())
                .collect(),
        };
        let count = length / size;
        let all = rng.below(5) == 0;
        let n = if all { count } else { 1 + rng.below(8) as usize };
        let fill = rng.pick(&extremes).clone();
        for k in 0..n {
            let i = if all { k } else { rng.below(count as u64) as usize };
            let at = offset + i * size;
            let value = if all || rng.below(2) == 0 {
                fill.clone()
            } else if rng.below(2) == 0 {
                rng.pick(&extremes).clone()
            } else {
                (0..size).map(|_| rng.next() as u8).collect()
            };
            f[at..at + size].copy_from_slice(&value);
        }
    }
}

/// One byte-level damage. Half of the positions are in the first 6,400 bytes (the preamble,
/// the header, the vocabulary and the merges of the tiny model).
fn mutate_bytes(rng: &mut Rng, f: &mut Vec<u8>) {
    if f.is_empty() {
        f.push(rng.next() as u8);
        return;
    }
    let len = f.len() as u64;
    let pos = if rng.below(2) == 0 { rng.below(len.min(6_400)) } else { rng.below(len) } as usize;
    match rng.below(9) {
        0..=2 => f[pos] ^= 1 << rng.below(8),
        3 => f[pos] = rng.next() as u8,
        4 => f.truncate(pos),
        5 => f.extend(
            (0..1 + rng.below(70)).map(|_| if rng.below(2) == 0 { 0 } else { rng.next() as u8 }),
        ),
        6 => f.insert(pos, rng.next() as u8),
        7 => {
            f.remove(pos);
        }
        _ => {
            let v = *rng.pick(&LARGE);
            let bytes = if rng.below(2) == 0 {
                (v as u32).to_le_bytes().to_vec()
            } else {
                v.to_le_bytes().to_vec()
            };
            let end = (pos + bytes.len()).min(f.len());
            f[pos..end].copy_from_slice(&bytes[..end - pos]);
        }
    }
}

/// A damaged copy of the tiny model for `seed`. Also gives a short name of the mutations.
fn damaged(seed: u64) -> (Vec<u8>, Vec<&'static str>) {
    let mut rng = Rng(seed);
    let mut f = TINY.to_vec();
    let mut kinds = Vec::new();
    match rng.below(10) {
        0..=3 => {
            for _ in 0..1 + rng.below(3) {
                mutate_bytes(&mut rng, &mut f);
            }
            kinds.push("bytes");
        }
        4..=6 => {
            let mut v = header_value(&f);
            for _ in 0..1 + rng.below(2) {
                mutate_value(&mut rng, &mut v);
            }
            let edit = text_edit(rng.below(12));
            f = relayout(&f, &v, rng.below(5) != 0, &edit);
            kinds.push("header");
            if rng.below(4) == 0 {
                mutate_bytes(&mut rng, &mut f);
                kinds.push("bytes");
            }
        }
        _ => {
            mutate_tensors(&mut rng, &mut f);
            kinds.push("tensors");
        }
    }
    (f, kinds)
}

/// The weights of every path for `tokens`, compared with the reference. Gives the number of
/// steps that ran.
fn check_paths(seed: u64, model: &Model, rng: &mut Rng) -> usize {
    let vocab = model.vocab_size() as u64;
    let n_ctx = model.n_ctx() as u64;
    let k = *rng.pick(&[0u64, 0, 1, n_ctx / 2, n_ctx - 2]);
    let overlap = Overlap::new(k as u16);
    model.check_overlap(overlap).unwrap();
    let count = 1 + rng.below(n_ctx + 12) as usize;
    let tokens: Vec<TokenId> = (0..count).map(|_| TokenId::new(rng.below(vocab) as u32)).collect();

    let mut reference = ReferenceSession::with_overlap(model, overlap);
    let mut want: Vec<Vec<u64>> = vec![reference.start_block().unwrap().to_vec()];
    for &t in &tokens {
        want.push(reference.step(t).unwrap().to_vec());
    }
    for w in &want {
        assert_eq!(w.len() as u64, vocab, "seed {seed}");
        let sum: u128 = w.iter().map(|&x| u128::from(x)).sum();
        assert!(w.iter().any(|&x| x >= 1) && sum <= 1 << 60, "seed {seed}: weight bound");
    }
    for kernels in Kernels::available() {
        let config = SessionConfig::new().with_kernels(kernels).with_overlap(overlap);
        let mut s = Session::with_config(model, config);
        assert_eq!(s.start_block().unwrap(), want[0].as_slice(), "seed {seed} {}", kernels.name());
        for (i, &t) in tokens.iter().enumerate() {
            assert_eq!(
                s.step(t).unwrap(),
                want[i + 1].as_slice(),
                "seed {seed} {} step {i}",
                kernels.name()
            );
        }
        // The prefill path: many known tokens in one call.
        let mut p = Session::with_config(model, config.with_chunk(8));
        p.start_block().unwrap();
        for (c, chunk) in tokens.chunks(8).enumerate() {
            let got = p.step_many(chunk).unwrap();
            for (i, w) in got.iter().enumerate() {
                let at = c * 8 + i;
                assert_eq!(
                    w,
                    want[at + 1].as_slice(),
                    "seed {seed} {} step_many {at}",
                    kernels.name()
                );
            }
        }
    }
    // A batch lane, next to a lane with other tokens.
    let mut batch = Batch::new(model, 2).with_overlap(overlap);
    batch.run(&[Some(BatchInput::Start), Some(BatchInput::Start)]).unwrap();
    assert_eq!(batch.weights(0).unwrap(), want[0].as_slice(), "seed {seed} batch start");
    for (i, &t) in tokens.iter().enumerate() {
        let other =
            (i % 3 != 2).then(|| BatchInput::Token(TokenId::new((i as u32 * 7) % vocab as u32)));
        batch.run(&[Some(BatchInput::Token(t)), other]).unwrap();
        assert_eq!(batch.weights(0).unwrap(), want[i + 1].as_slice(), "seed {seed} batch step {i}");
    }
    // Tokens outside the vocabulary are refused.
    assert!(matches!(
        reference.step(TokenId::new(vocab as u32)),
        Err(Error::TokenOutOfRange { .. })
    ));
    count
}

/// The result of one case, for the counts of the long run.
#[derive(Clone, Copy, Debug, Default)]
struct Outcome {
    parse_refused: u64,
    load_refused: u64,
    loaded: u64,
    steps: u64,
}

/// One fuzz case. Panics (with the seed) if the reader or the model breaks a rule.
fn run_case(seed: u64) -> Outcome {
    let (f, _) = damaged(seed);
    let mut out = Outcome::default();
    let parsed = HsmFile::parse(&f);
    let loaded = Model::load(&f);
    match (&parsed, &loaded) {
        (Err(_), Ok(_)) => {
            panic!("seed {seed}: Model::load accepted a file that HsmFile::parse refused")
        }
        (Err(_), Err(_)) => out.parse_refused = 1,
        (Ok(_), Err(_)) => out.load_refused = 1,
        (Ok(h), Ok(m)) => {
            // An accepted file has one hash: the SHA-256 of all its bytes.
            assert_eq!(h.hash(), m.hash(), "seed {seed}");
            out.loaded = 1;
            let mut rng = Rng(seed ^ 0xA11CE);
            out.steps = check_paths(seed, m, &mut rng) as u64;
        }
    }
    if let Err(e) = &parsed {
        // An error message is short: it never copies a large part of the file.
        let text = e.to_string();
        assert!(text.len() <= 600, "seed {seed}: error message of {} bytes", text.len());
    }
    out
}

#[test]
fn relayout_without_edits_gives_a_valid_file() {
    // NOTE: serde_json sorts the keys, so the bytes differ from the writer, but the file is
    // valid and has the same tensors.
    let v = header_value(TINY);
    let same = relayout(TINY, &v, true, &|s| s);
    assert_eq!(same.len(), TINY.len());
    assert_eq!(same[header_len(TINY) + 16..], TINY[header_len(TINY) + 16..]);
    Model::load(&same).unwrap();
    // A longer header moves every section by 64 bytes, and the file is still valid.
    let longer = relayout(TINY, &v, true, &|s| format!("{s}{}", " ".repeat(100)));
    assert_eq!(longer.len(), TINY.len() + 128);
    Model::load(&longer).unwrap();
}

#[test]
fn every_tensor_extreme_is_refused_or_gives_equal_paths() {
    // Each tensor in turn, all entries set to each extreme value of its type.
    let spans = tensor_spans(TINY);
    let mut loaded = 0;
    for (index, (_, dtype, offset, length)) in spans.iter().enumerate() {
        let values: Vec<Vec<u8>> = match dtype.as_str() {
            "i8" => vec![vec![0x80], vec![0x7F]],
            "i16" => vec![i16::MIN.to_le_bytes().to_vec(), i16::MAX.to_le_bytes().to_vec()],
            "i32" => vec![i32::MIN.to_le_bytes().to_vec(), i32::MAX.to_le_bytes().to_vec()],
            _ => vec![u64::MAX.to_le_bytes().to_vec(), (1u64 << 40).to_le_bytes().to_vec()],
        };
        for (vi, v) in values.iter().enumerate() {
            let mut f = TINY.to_vec();
            for chunk in f[*offset..offset + length].chunks_exact_mut(v.len()) {
                chunk.copy_from_slice(v);
            }
            if let Ok(m) = Model::load(&f) {
                loaded += 1;
                let mut rng = Rng((index * 2 + vi) as u64);
                check_paths(index as u64, &m, &mut rng);
            }
        }
    }
    assert!(loaded > 50, "{loaded}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn damaged_model_files_are_refused_or_safe(seed in any::<u64>()) {
        run_case(seed);
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

/// The long randomized run, on all processors. Each thread takes every `n`-th seed. A panic
/// does not stop the run: the seeds that failed are listed at the end.
#[test]
#[ignore = "long run: start it with --ignored (docs/security.md)"]
fn long_run_hsm() {
    let (cases, first, trace) = long_run_params(100_000);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    let results = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                s.spawn(move || {
                    let mut total = Outcome::default();
                    let mut failed = Vec::new();
                    let mut i = k;
                    while i < cases {
                        let seed = first.wrapping_add(i);
                        if trace {
                            eprintln!("seed {seed}");
                        }
                        match std::panic::catch_unwind(|| run_case(seed)) {
                            Ok(o) => {
                                total.parse_refused += o.parse_refused;
                                total.load_refused += o.load_refused;
                                total.loaded += o.loaded;
                                total.steps += o.steps;
                            }
                            Err(_) => failed.push(seed),
                        }
                        if k == 0 && i % (threads * 10_000) == 0 {
                            eprintln!("fuzz_hsm: {i} of {cases}");
                        }
                        i += threads;
                    }
                    (total, failed)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    let mut total = Outcome::default();
    let mut failed = Vec::new();
    for (o, f) in results {
        total.parse_refused += o.parse_refused;
        total.load_refused += o.load_refused;
        total.loaded += o.loaded;
        total.steps += o.steps;
        failed.extend(f);
    }
    println!(
        "fuzz_hsm: {cases} cases from seed {first}: {} refused by the reader, {} refused by \
         Model::load, {} loaded ({} steps on every path), {} failed",
        total.parse_refused,
        total.load_refused,
        total.loaded,
        total.steps,
        failed.len()
    );
    assert!(failed.is_empty(), "failed seeds: {failed:?}");
}
