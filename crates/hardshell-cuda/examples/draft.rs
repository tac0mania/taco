//! The speed of a verified draft on the CUDA GPU (ADR 0017): each call runs `k` rows for each
//! of `S` slots (the known token and `k - 1` draft tokens), with the coder op (plain) or the
//! mixed coder op (the logs profile), then truncates each slot.
//!
//!   draft <model.hsm> [slots...]     tokens/s for draft lengths 1, 4, 8, 16 and 32
//!
//! The slots decode at positions 256 to 1,000 (a slot goes back to 256 with a truncation when
//! its context is nearly full), so the attention reads a mid-window context. Each call keeps
//! all rows (a truncation of 0 rows) except at the wrap; the cost of a call does not depend on
//! the number of kept rows. "rows/s" is `S * k` rows per call; "calls/s" is the step rate of
//! the decoder. The real tokens/s of `taco unfold` is `calls/s * S * (tokens per pass)`, with
//! the tokens per pass of the measurement of `tools/logs/skip.py` (ADR 0017).
#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    missing_docs
)]
#![allow(clippy::cast_precision_loss, clippy::float_arithmetic, clippy::cast_possible_truncation)]

use std::time::Instant;

use hardshell::TokenId;
use hardshell_cuda::{CoderQuery, CudaBatch, CudaDevice, CudaModel, MixInput};

/// A token of slot `s` at stream position `i` (a fixed pseudo-random text).
fn tok(s: usize, i: usize) -> TokenId {
    let x = (s as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (i as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    TokenId::new(((x >> 17) % 50_000) as u32)
}

/// The first and the last position of the measured rows.
const LOW: usize = 256;
const HIGH: usize = 1000;

fn run(gpu: &CudaModel<'_>, slots: usize, k: usize, mixed: bool) -> Option<(f64, f64)> {
    let mut batch = CudaBatch::new(gpu, slots, (slots * k).max(LOW)).ok()?;
    // Fill each slot to the position LOW.
    for s in 0..slots {
        let mut inputs = vec![(s, gpu.eot_token())];
        inputs.extend((1..LOW).map(|i| (s, tok(s, i))));
        batch.run_without_readback(&inputs).unwrap();
    }
    let mut len = vec![LOW; slots];
    let call = |batch: &mut CudaBatch<'_>, len: &mut Vec<usize>| {
        let mut inputs = Vec::with_capacity(slots * k);
        let mut queries = Vec::with_capacity(slots * k);
        let mut tables = Vec::with_capacity(slots * k);
        for (s, &l) in len.iter().enumerate() {
            for j in 0..k {
                inputs.push((s, tok(s, l + j)));
                let q = if j == 0 {
                    CoderQuery::Decode((l as u64 * 7919) % (1 << 32))
                } else {
                    CoderQuery::Encode(tok(s, l + j + 1))
                };
                queries.push(q);
                tables.push(MixInput::Mixed(Some(tok(s, l + j + 1))));
            }
        }
        if mixed {
            batch.run_coded_mixed(&inputs, &queries, &tables, 32, |_, _| 60_000).unwrap();
        } else {
            batch.run_coded(&inputs, &queries, 32).unwrap();
        }
        for (s, l) in len.iter_mut().enumerate() {
            *l += k;
            if *l + k > HIGH {
                *l = LOW;
            }
            batch.truncate(s, *l).unwrap();
        }
    };
    for _ in 0..8 {
        call(&mut batch, &mut len);
    }
    // About 2 s of calls for each shape (at least 16 calls).
    let t0 = Instant::now();
    let mut n = 0;
    while n < 16 || t0.elapsed().as_secs_f64() < 2.0 {
        call(&mut batch, &mut len);
        n += 1;
    }
    let sec = t0.elapsed().as_secs_f64();
    Some((n as f64 / sec, (n * slots * k) as f64 / sec))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: draft <model.hsm> [slots...]");
    let bytes = std::fs::read(path).unwrap();
    let mut sizes: Vec<usize> = args[2..].iter().map(|s| s.parse().unwrap()).collect();
    if sizes.is_empty() {
        sizes = vec![1, 8, 32];
    }
    let dev = CudaDevice::open(0).unwrap();
    println!("device: {} {:?}", dev.name(), dev.compute_capability());
    let gpu = CudaModel::load(&dev, &bytes).unwrap();
    for mixed in [false, true] {
        let op = if mixed { "mixed coder op" } else { "coder op" };
        for &s in &sizes {
            for k in [1usize, 4, 8, 16, 32] {
                match run(&gpu, s, k, mixed) {
                    Some((calls, rows)) => println!(
                        "{op:15} slots {s:3} draft {k:2}: {calls:7.1} calls/s, {rows:8.0} rows/s, \
                         {:6.2} ms/call",
                        1000.0 / calls
                    ),
                    None => println!("{op:15} slots {s:3} draft {k:2}: not possible (memory)"),
                }
            }
        }
    }
}
