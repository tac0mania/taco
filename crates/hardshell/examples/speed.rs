//! Measures the speed of the HardShell paths on one thread, with no other work per step.
//!
//! ```text
//! cargo run --release -p hardshell --example speed -- MODEL.hsm WINDOWS.txt \
//!     [--path reference|step|chunk|batch] [--n N] [--limit STEPS] [--windows W] [--digest]
//! ```
//!
//! `WINDOWS.txt` has the format of `eval_bits`. Each path runs the first `W` windows (default
//! 1): `reference` and `step` one token at a time, `chunk` with `Session::step_many` of `N`
//! tokens, and `batch` with a `Batch` of `N` lanes (lane `j` runs window `j mod W`). The
//! output is the steps per second. With `--digest`, it also prints a SHA-256 of all weights of
//! each window, which is the same for every path (the hash costs time: do not measure the
//! speed with it).

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::cast_precision_loss)]

use std::error::Error;
use std::time::Instant;

use hardshell::TokenId;
use hardshell::gpt2::{Batch, BatchInput, Model, ReferenceSession, Session, SessionConfig};
use sha2::{Digest, Sha256};

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

fn read_windows(path: &str, limit: usize, count: usize) -> AnyResult<Vec<Vec<TokenId>>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let (_, ids) = line.split_once('\t').ok_or("a window line needs a tab")?;
        let toks: Vec<TokenId> = ids
            .split_ascii_whitespace()
            .take(limit)
            .map(|s| s.parse::<u32>().map(TokenId::new))
            .collect::<Result<_, _>>()?;
        out.push(toks);
        if out.len() == count {
            break;
        }
    }
    if out.is_empty() {
        return Err("no windows".into());
    }
    Ok(out)
}

/// The digest of the weights of each window, in window order, for the sequential paths.
fn sequential(
    model: &Model,
    wins: &[Vec<TokenId>],
    path: &str,
    n: usize,
    digest: bool,
) -> AnyResult<Vec<Sha256>> {
    let mut out = Vec::new();
    for toks in wins {
        let mut h = Sha256::new();
        let mut put = |w: &[u64]| {
            if digest {
                w.iter().for_each(|x| h.update(x.to_le_bytes()));
            } else {
                std::hint::black_box(w.first());
            }
        };
        let inputs = toks.get(..toks.len().saturating_sub(1)).unwrap_or(&[]);
        match path {
            "reference" => {
                let mut s = ReferenceSession::new(model);
                put(s.start_block()?);
                for &t in inputs {
                    put(s.step(t)?);
                }
            }
            "step" => {
                let mut s = Session::new(model);
                put(s.start_block()?);
                for &t in inputs {
                    put(s.step(t)?);
                }
            }
            _ => {
                let mut s = Session::with_config(model, SessionConfig::new().with_chunk(n));
                put(s.start_block()?);
                for part in inputs.chunks(n) {
                    s.step_many(part)?.iter().for_each(&mut put);
                }
            }
        }
        out.push(h);
    }
    Ok(out)
}

/// `n` lanes in lockstep; lane `j` runs window `j mod W`. Returns the digest of each window
/// (from its first lane) and the number of steps.
fn batched(
    model: &Model,
    wins: &[Vec<TokenId>],
    n: usize,
    digest: bool,
) -> AnyResult<(Vec<Sha256>, usize)> {
    let mut b = Batch::new(model, n);
    let mut hs: Vec<Sha256> = (0..n).map(|_| Sha256::new()).collect();
    let lane_win = |j: usize| &wins[j % wins.len()];
    let longest = (0..n).map(|j| lane_win(j).len()).max().unwrap_or(0);
    let mut steps = 0;
    for i in 0..longest {
        let inputs: Vec<Option<BatchInput>> = (0..n)
            .map(|j| match i {
                0 => Some(BatchInput::Start),
                _ => lane_win(j).get(i).map(|_| BatchInput::Token(lane_win(j)[i - 1])),
            })
            .collect();
        b.run(&inputs)?;
        for (j, inp) in inputs.iter().enumerate() {
            if inp.is_some() {
                let w = b.weights(j).ok_or("no weights")?;
                if digest {
                    w.iter().for_each(|x| hs[j].update(x.to_le_bytes()));
                }
                steps += 1;
            }
        }
    }
    let firsts = hs.into_iter().take(wins.len()).collect();
    Ok((firsts, steps))
}

fn main() -> AnyResult<()> {
    let mut it = std::env::args().skip(1);
    let (mut pos, mut path, mut n, mut limit, mut count) =
        (Vec::new(), "step".to_owned(), 32usize, usize::MAX, 1usize);
    let mut digest = false;
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or("an option needs a value");
        match a.as_str() {
            "--path" => path = val()?,
            "--n" => n = val()?.parse()?,
            "--limit" => limit = val()?.parse()?,
            "--windows" => count = val()?.parse()?,
            "--digest" => digest = true,
            _ => pos.push(a),
        }
    }
    let [model_path, windows] = <[String; 2]>::try_from(pos).map_err(|_| "usage: see the docs")?;
    let model = Model::load(&std::fs::read(model_path)?)?;
    let wins = read_windows(&windows, limit, count)?;
    let t0 = Instant::now();
    let (digests, steps) = if path == "batch" {
        batched(&model, &wins, n.max(1), digest)?
    } else {
        let steps = wins.iter().map(Vec::len).sum();
        (sequential(&model, &wins, &path, n.max(1), digest)?, steps)
    };
    let secs = t0.elapsed().as_secs_f64();
    for (i, h) in digests.into_iter().enumerate().filter(|_| digest) {
        let d: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        println!("window {i} digest {d}");
    }
    println!(
        "path {path} n {n}: {steps} steps in {secs:.2} s: {:.1} steps/s (one thread)",
        steps as f64 / secs
    );
    Ok(())
}
