//! Measures the ideal coded size of token windows with a `.hsm` model: the quality of the
//! integer model (plan §12, phase 3).
//!
//! ```text
//! cargo run --release -p hardshell --example eval_bits -- MODEL.hsm WINDOWS.txt \
//!     [--threads N] [--digest] [--limit STEPS] [--chunk N] [--reference] [--overlap K]
//! ```
//!
//! `--overlap K` sets the context rule of ADR 0015: a context restart keeps the last `K`
//! tokens (default 0, the rule of ADR 0004 section 9). The example then also counts the
//! restart rows (`<|endoftext|>` and the `K` kept tokens of each restart): the extra compute.
//!
//! `--chunk N` runs the window with `Session::step_many`, `N` tokens at a time (the prefill
//! path). `--reference` runs it with `ReferenceSession` (the scalar reference). The weights,
//! and thus the digests, are the same for every choice.
//!
//! `WINDOWS.txt` has one window on each line: a label, a tab, then the token ids, separated by
//! spaces. A line that starts with `#` is a comment. For each window the example calls
//! `Session::start_block` (the weights for token 0), then `Session::step` with each token
//! except the last (plan §7.7). Thus the model predicts every token of the window.
//!
//! For each token `t` with weights `w`, the example adds:
//!
//! - the ideal size `-log2(w[t] / sum(w))`, if `w[t] > 0` (a weight of 0 is counted apart);
//! - the table size `-log2(f[t] / 2^32)`, where `f` is the frequency table of plan §5.1 with
//!   `FREQ_BITS = 32`. This is the size that the range coder gives, to about 0.02 % (phase 0).
//!
//! The example uses `f64` for the logarithms only. It is a measurement tool, not the
//! prediction path: the weights come from HardShell, in integers.
//!
//! Windows are independent, so the example runs them on `N` threads. Each window has its own
//! session, the results go into a slot for each window, and the totals are summed in window
//! order. Thus the output does not depend on the thread count. `--digest` also prints the
//! SHA-256 of all weights of each window, so two runs can be compared bit for bit, and the
//! SHA-256 of the table frequency `f[t]` of each coded token (little-endian `u64`, in token
//! order, all windows in file order), which a GPU run can compare.

#![allow(clippy::print_stdout, clippy::print_stderr)]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::error::Error;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use hardshell::TokenId;
use hardshell::gpt2::{Model, Overlap, ReferenceSession, Session, SessionConfig};
use sha2::{Digest, Sha256};

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// The total of a frequency table: `2^FREQ_BITS` with `FREQ_BITS = 32` (plan §5.1).
const TOTAL: u128 = 1 << 32;

struct Window {
    label: String,
    tokens: Vec<TokenId>,
}

#[derive(Clone, Default)]
struct Outcome {
    bytes: usize,
    ideal_bits: f64,
    zero_weight: usize,
    table_bits: f64,
    seconds: f64,
    /// The rows of the context restarts: `<|endoftext|>` and the kept tokens.
    restart_rows: usize,
    digest: Option<String>,
    /// The table frequencies of the coded tokens, little-endian `u64`, in order.
    freqs: Vec<u8>,
}

struct Args {
    model: String,
    windows: String,
    threads: usize,
    digest: bool,
    limit: Option<usize>,
    chunk: usize,
    reference: bool,
    overlap: Overlap,
}

fn parse_args() -> AnyResult<Args> {
    let mut it = std::env::args().skip(1);
    let mut pos = Vec::new();
    let mut threads = std::thread::available_parallelism().map_or(1, usize::from);
    let mut digest = false;
    let mut limit = None;
    let mut chunk = 1;
    let mut reference = false;
    let mut overlap = 0u16;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--threads" => threads = it.next().ok_or("--threads needs a value")?.parse()?,
            "--digest" => digest = true,
            "--limit" => limit = Some(it.next().ok_or("--limit needs a value")?.parse()?),
            "--chunk" => chunk = it.next().ok_or("--chunk needs a value")?.parse()?,
            "--reference" => reference = true,
            "--overlap" => overlap = it.next().ok_or("--overlap needs a value")?.parse()?,
            _ => pos.push(a),
        }
    }
    let [model, windows] = <[String; 2]>::try_from(pos)
        .map_err(|_| "usage: eval_bits MODEL.hsm WINDOWS.txt [--threads N] [--digest]")?;
    Ok(Args {
        model,
        windows,
        threads: threads.max(1),
        digest,
        limit,
        chunk: chunk.max(1),
        reference,
        overlap: Overlap::new(overlap),
    })
}

fn read_windows(path: &str, limit: Option<usize>) -> AnyResult<Vec<Window>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let (label, ids) = line.split_once('\t').ok_or("a window line needs a tab")?;
        let mut tokens = ids
            .split_ascii_whitespace()
            .map(|s| s.parse::<u32>().map(TokenId::new))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(n) = limit {
            tokens.truncate(n);
        }
        if tokens.is_empty() {
            return Err(format!("window {label} has no tokens").into());
        }
        out.push(Window { label: label.to_owned(), tokens });
    }
    Ok(out)
}

/// The frequency of `token` in the plan §5.1 table of `weights`. Integer only, `u128`.
fn table_freq(weights: &[u64], token: usize) -> u128 {
    let v = weights.len() as u128;
    let mut sum: u128 = weights.iter().map(|&w| u128::from(w)).sum();
    let all_zero = sum == 0;
    if all_zero {
        sum = v;
    }
    let w_of = |i: usize| if all_zero { 1 } else { u128::from(weights[i]) };
    let f_of = |i: usize| 1 + w_of(i) * (TOTAL - v) / sum;
    // The remainder goes to the largest weight, the lowest id on a tie (plan §5.1 step 3).
    let mut best = 0;
    for (i, &w) in weights.iter().enumerate() {
        if w > weights[best] {
            best = i;
        }
    }
    let f = f_of(token);
    if token == best {
        let total: u128 = (0..weights.len()).map(f_of).sum();
        f + (TOTAL - total)
    } else {
        f
    }
}

/// Calls `f(i, weights)` with the weights that predict token `i` of the window, in order.
fn run_window(
    model: &Model,
    w: &Window,
    args: &Args,
    mut f: impl FnMut(usize, &[u64]) -> AnyResult<()>,
) -> AnyResult<()> {
    let toks = &w.tokens;
    let inputs = toks.get(..toks.len() - 1).unwrap_or(&[]);
    if args.reference {
        let mut s = ReferenceSession::with_overlap(model, args.overlap);
        f(0, s.start_block()?)?;
        for (i, &t) in inputs.iter().enumerate() {
            f(i + 1, s.step(t)?)?;
        }
        return Ok(());
    }
    let cfg = SessionConfig::new().with_chunk(args.chunk).with_overlap(args.overlap);
    let mut s = Session::with_config(model, cfg);
    f(0, s.start_block()?)?;
    let mut i = 1;
    for part in inputs.chunks(args.chunk) {
        for weights in s.step_many(part)?.iter() {
            f(i, weights)?;
            i += 1;
        }
    }
    Ok(())
}

fn measure(model: &Model, w: &Window, args: &Args) -> AnyResult<Outcome> {
    let t0 = Instant::now();
    let mut out = Outcome::default();
    let mut hasher = args.digest.then(Sha256::new);
    let mut bytes = Vec::new();
    run_window(model, w, args, |i, weights| {
        let tok = w.tokens[i];
        let t = tok.get() as usize;
        let wt = *weights.get(t).ok_or("token id out of range")?;
        let sum: u128 = weights.iter().map(|&x| u128::from(x)).sum();
        if wt > 0 {
            out.ideal_bits -= (wt as f64 / sum as f64).log2();
        } else {
            out.zero_weight += 1;
        }
        let f = table_freq(weights, t);
        out.table_bits -= (f as f64 / TOTAL as f64).log2();
        if args.digest {
            out.freqs.extend_from_slice(&(f as u64).to_le_bytes());
        }
        if let Some(h) = hasher.as_mut() {
            for &x in weights {
                h.update(x.to_le_bytes());
            }
        }
        bytes.extend_from_slice(model.vocabulary().token(tok).ok_or("token without bytes")?);
        Ok(())
    })?;
    out.bytes = bytes.len();
    out.restart_rows = restart_rows(model.n_ctx(), args.overlap.tokens(), w.tokens.len());
    out.seconds = t0.elapsed().as_secs_f64();
    out.digest = hasher.map(|h| h.finalize().iter().map(|b| format!("{b:02x}")).collect());
    Ok(out)
}

/// The restart rows of a window of `n` coded tokens: the session gets `n - 1` tokens after
/// the start; the first context holds `n_ctx - 1` of them, each later context `n_ctx - 1 - k`
/// new ones, and each restart runs `k + 1` rows without weights (ADR 0015).
fn restart_rows(n_ctx: usize, k: usize, n: usize) -> usize {
    let given = n.saturating_sub(1);
    let first = n_ctx.saturating_sub(1);
    let per = n_ctx.saturating_sub(1 + k).max(1);
    if given <= first { 0 } else { (given - first).div_ceil(per) * (k + 1) }
}

fn main() -> AnyResult<()> {
    let args = parse_args()?;
    let t0 = Instant::now();
    let bytes = std::fs::read(&args.model)?;
    let model = Model::load(&bytes)?;
    drop(bytes);
    let windows = read_windows(&args.windows, args.limit)?;
    let steps: usize = windows.iter().map(|w| w.tokens.len()).sum();
    eprintln!(
        "model {} ({}), {} windows, {steps} steps, {} threads, chunk {}, overlap {}{}, loaded in {:.1} s",
        args.model,
        model.hash(),
        windows.len(),
        args.threads,
        args.chunk,
        args.overlap.get(),
        if args.reference { ", reference session" } else { "" },
        t0.elapsed().as_secs_f64()
    );
    let slots: Vec<Mutex<Option<AnyResult<Outcome>>>> =
        windows.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t1 = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..args.threads.min(windows.len()) {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(w) = windows.get(i) else { break };
                    let r = measure(&model, w, &args);
                    let k = done.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!(
                        "  [{k}/{}] {} {:.1} s",
                        windows.len(),
                        w.label,
                        r.as_ref().map_or(0.0, |o| o.seconds)
                    );
                    if let Some(slot) = slots.get(i)
                        && let Ok(mut g) = slot.lock()
                    {
                        *g = Some(r);
                    }
                }
            });
        }
    });
    let wall = t1.elapsed().as_secs_f64();
    println!("# model\t{}\t{}", args.model, model.hash());
    println!("# overlap\t{}", args.overlap.get());
    println!(
        "# label\ttokens\tbytes\tideal_bits\tzero_weight\ttable_bits\tseconds\trestart_rows\tdigest"
    );
    let mut total = Outcome::default();
    let mut combined = Sha256::new();
    let mut freq_digest = Sha256::new();
    for (w, slot) in windows.iter().zip(slots) {
        let o =
            slot.into_inner().map_err(|_| "a worker panicked")?.ok_or("a window is missing")??;
        println!(
            "{}\t{}\t{}\t{:.6}\t{}\t{:.6}\t{:.1}\t{}\t{}",
            w.label,
            w.tokens.len(),
            o.bytes,
            o.ideal_bits,
            o.zero_weight,
            o.table_bits,
            o.seconds,
            o.restart_rows,
            o.digest.as_deref().unwrap_or("-")
        );
        freq_digest.update(&o.freqs);
        total.restart_rows += o.restart_rows;
        total.bytes += o.bytes;
        total.ideal_bits += o.ideal_bits;
        total.zero_weight += o.zero_weight;
        total.table_bits += o.table_bits;
        if let Some(d) = &o.digest {
            combined.update(d.as_bytes());
        }
    }
    let combined: String = combined.finalize().iter().map(|b| format!("{b:02x}")).collect();
    println!(
        "TOTAL\t{steps}\t{}\t{:.6}\t{}\t{:.6}\t{wall:.1}\t{}\t{}",
        total.bytes,
        total.ideal_bits,
        total.zero_weight,
        total.table_bits,
        total.restart_rows,
        if args.digest { combined.as_str() } else { "-" }
    );
    if args.digest {
        let f: String = freq_digest.finalize().iter().map(|b| format!("{b:02x}")).collect();
        println!("# table frequency digest {f}");
    }
    println!(
        "# bits per byte: ideal {:.6}, table {:.6}; f64 bits of the totals: ideal {:016x}, table {:016x}",
        total.ideal_bits / total.bytes as f64,
        total.table_bits / total.bytes as f64,
        total.ideal_bits.to_bits(),
        total.table_bits.to_bits()
    );
    println!(
        "# {steps} steps in {wall:.1} s on {} threads: {:.1} steps/s",
        args.threads,
        steps as f64 / wall
    );
    Ok(())
}
