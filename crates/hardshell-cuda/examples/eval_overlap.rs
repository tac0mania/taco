//! Measures the coded size of whole blocks with the context rule of ADR 0015 on the CUDA GPU:
//! bits per byte for each context overlap `K`, and the extra rows that the restarts cost.
//!
//! ```text
//! cargo run --release -p hardshell-cuda --example eval_overlap -- MODEL.hsm \
//!     --overlap 0,256,512,768 [--slots 32] [--chunk 16] [--native] [--positions] WINDOWS.txt...
//! ```
//!
//! `--positions` (not with `--native`) also prints the mean table size of a coded token for
//! each group of 128 context positions: the line `# positions FILE K` and then
//! `first_position tokens bits_per_token` for each group.
//!
//! `WINDOWS.txt` has the format of `crates/hardshell/examples/eval_bits.rs`: one block on each
//! line, a label, a tab, then the token ids of the block (tokenized alone, ADR 0007 §2.1).
//!
//! The model predicts every token of a block: `<|endoftext|>` predicts token 0, and each
//! token predicts the next. For each token `t`, the example adds the table size
//! `32 - log2(f[t])`, where `f` is the frequency table of plan §5.1 (`FREQ_BITS = 32`), which
//! the GPU coder op gives (`CudaBatch::run_coded`, equal to `taco_coder::FreqTable`).
//!
//! # Two ways to run a block
//!
//! - The default: each context window of a block is a job of its own. Window 0 is
//!   `<|endoftext|>` and the first `n_ctx - 1` tokens; window `w > 0` is `<|endoftext|>`, the
//!   last `K` tokens before it (rows without a result), then its own `n_ctx - 1 - K` tokens.
//!   By the definition of the rule (ADR 0015, `tests/context.rs` in `hardshell`), a window
//!   gives the same weights as the restart path, and windows are independent, so the jobs fill
//!   all slots of the GPU.
//! - `--native`: each block is one job in one slot, through the restart path of
//!   `CudaBatch::set_overlap`. It is slower (one slot for each block) and checks the default.
//!
//! Both print the SHA-256 of the table frequency of every coded token (little-endian `u64`,
//! in token order, blocks in file order), so the two ways and the CPU (`eval_bits --digest`)
//! can be compared bit for bit.
//!
//! The example uses `f64` for the logarithms only. It is a measurement tool, not the
//! prediction path.
#![allow(clippy::print_stdout, clippy::print_stderr, clippy::unwrap_used, missing_docs)]
#![allow(clippy::cast_precision_loss, clippy::float_arithmetic, clippy::cast_possible_truncation)]

use std::collections::VecDeque;
use std::error::Error;
use std::time::Instant;

use hardshell::TokenId;
use hardshell::gpt2::Overlap;
use hardshell_cuda::{CoderQuery, CudaBatch, CudaDevice, CudaModel};
use sha2::{Digest, Sha256};

type AnyResult<T> = Result<T, Box<dyn Error>>;

/// `FREQ_BITS` of plan §5.1.
const FREQ_BITS: u32 = 32;

/// One block of a windows file.
struct Block {
    tokens: Vec<TokenId>,
    bytes: usize,
}

/// One job: a fresh context in one slot. `inputs[i]` runs at position `i`; `coded[i]` is the
/// token that the weights of input `i` predict, or `None` for a row without a result.
struct Job {
    block: usize,
    inputs: Vec<TokenId>,
    coded: Vec<Option<TokenId>>,
}

struct Args {
    model: String,
    files: Vec<String>,
    overlaps: Vec<u16>,
    slots: usize,
    chunk: usize,
    native: bool,
    positions: bool,
}

fn parse_args() -> AnyResult<Args> {
    let mut it = std::env::args().skip(1);
    let mut pos = Vec::new();
    let mut overlaps = vec![0];
    let (mut slots, mut chunk, mut native, mut positions) = (32, 16, false, false);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--overlap" => {
                let v = it.next().ok_or("--overlap needs a value")?;
                overlaps = v.split(',').map(str::parse).collect::<Result<_, _>>()?;
            }
            "--slots" => slots = it.next().ok_or("--slots needs a value")?.parse()?,
            "--chunk" => chunk = it.next().ok_or("--chunk needs a value")?.parse()?,
            "--native" => native = true,
            "--positions" => positions = true,
            _ => pos.push(a),
        }
    }
    if pos.len() < 2 {
        return Err("usage: eval_overlap MODEL.hsm [--overlap K,K] WINDOWS.txt...".into());
    }
    let model = pos.remove(0);
    let (slots, chunk) = (slots.max(1), chunk.max(1));
    Ok(Args { model, files: pos, overlaps, slots, chunk, native, positions })
}

fn read_blocks(path: &str, vocab: &dyn Fn(TokenId) -> usize) -> AnyResult<Vec<Block>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let (_, ids) = line.split_once('\t').ok_or("a window line needs a tab")?;
        let tokens = ids
            .split_ascii_whitespace()
            .map(|s| s.parse::<u32>().map(TokenId::new))
            .collect::<Result<Vec<_>, _>>()?;
        let bytes = tokens.iter().map(|&t| vocab(t)).sum();
        out.push(Block { tokens, bytes });
    }
    Ok(out)
}

/// The window jobs of a block of tokens `t` (the default way).
///
/// The session inputs are `u[0] = <|endoftext|>` and `u[j] = t[j - 1]`; the weights after
/// input `j` predict `t[j]`. Window 0 holds the inputs `0..n_ctx`. A later window starts at
/// input `s`: `<|endoftext|>`, the kept inputs `u[s - k..s]` (rows without a result), then the
/// inputs `u[s..e]` with `e = min(s + n_ctx - 1 - k, n)`. This is the restart of ADR 0015 at
/// input `s`: the last `k` inputs before it are at the positions `1..=k`.
fn window_jobs(b: usize, t: &[TokenId], n_ctx: usize, k: usize, eot: TokenId) -> Vec<Job> {
    let n = t.len();
    let u = |j: usize| if j == 0 { eot } else { t[j - 1] };
    let mut jobs = Vec::new();
    let end0 = n.min(n_ctx);
    jobs.push(Job {
        block: b,
        inputs: (0..end0).map(u).collect(),
        coded: t[..end0].iter().map(|&x| Some(x)).collect(),
    });
    let mut s = end0;
    while s < n {
        let e = (s + n_ctx - 1 - k).min(n);
        let mut inputs = vec![eot];
        inputs.extend((s - k..e).map(u));
        let mut coded = vec![None; 1 + k];
        coded.extend(t[s..e].iter().map(|&x| Some(x)));
        debug_assert!(inputs.len() == coded.len() && inputs.len() <= n_ctx);
        jobs.push(Job { block: b, inputs, coded });
        s = e;
    }
    jobs
}

/// The native job of a block: one slot, the restart path of the batch.
fn native_job(b: usize, t: &[TokenId], eot: TokenId) -> Job {
    let mut inputs = vec![eot];
    inputs.extend_from_slice(&t[..t.len() - 1]);
    Job { block: b, inputs, coded: t.iter().map(|&x| Some(x)).collect() }
}

/// The results of one file and one overlap.
#[derive(Default)]
struct Totals {
    tokens: usize,
    bytes: usize,
    bits: f64,
    rows: usize,
}

/// Runs the jobs in `slots` slots, `chunk` inputs of each slot in each call. Gives the
/// frequency of each coded token, in block order and token order, and the inputs that ran
/// (the native path runs the restart rows in addition).
fn run_jobs(
    batch: &mut CudaBatch<'_>,
    jobs: &[Job],
    blocks: usize,
    chunk: usize,
) -> AnyResult<(Vec<Vec<u64>>, usize)> {
    let slots = batch.slots();
    let mut freqs: Vec<Vec<u64>> = vec![Vec::new(); blocks];
    // The jobs of one block end in order, so the frequencies are in token order. Each job
    // writes into its own part: the part of a job is known when it starts.
    let mut parts: Vec<Vec<Option<u64>>> = jobs.iter().map(|j| vec![None; j.coded.len()]).collect();
    let mut queue: VecDeque<usize> = (0..jobs.len()).collect();
    let mut active: Vec<Option<(usize, usize)>> = vec![None; slots]; // (job, next input)
    let mut rows = 0;
    loop {
        for (slot, a) in active.iter_mut().enumerate() {
            if a.is_none()
                && let Some(j) = queue.pop_front()
            {
                batch.reset(slot);
                *a = Some((j, 0));
            }
        }
        if active.iter().all(Option::is_none) {
            break;
        }
        let (mut inputs, mut queries, mut map) = (Vec::new(), Vec::new(), Vec::new());
        for (slot, a) in active.iter_mut().enumerate() {
            let Some((j, next)) = a else { continue };
            let job = &jobs[*j];
            let end = (*next + chunk).min(job.inputs.len());
            for i in *next..end {
                inputs.push((slot, job.inputs[i]));
                let q = job.coded[i].unwrap_or(TokenId::new(0));
                queries.push(CoderQuery::Encode(q));
                map.push((*j, i));
            }
            *next = end;
        }
        let got = batch.run_coded(&inputs, &queries, FREQ_BITS)?;
        rows += inputs.len();
        for (iv, &(j, i)) in got.iter().zip(&map) {
            if jobs[j].coded[i].is_some() {
                parts[j][i] = Some(iv.freq);
            }
        }
        for a in &mut active {
            if let Some((j, next)) = *a
                && next >= jobs[j].inputs.len()
            {
                *a = None;
            }
        }
    }
    for (job, part) in jobs.iter().zip(parts) {
        freqs[job.block].extend(part.into_iter().flatten());
    }
    Ok((freqs, rows))
}

fn main() -> AnyResult<()> {
    let args = parse_args()?;
    let dev = CudaDevice::open(0)?;
    let bytes = std::fs::read(&args.model)?;
    let model = CudaModel::load(&dev, &bytes)?;
    let cpu = hardshell::gpt2::Model::load(&bytes)?;
    drop(bytes);
    let n_ctx = model.n_ctx();
    let eot = model.eot_token();
    let vocab = |t: TokenId| cpu.vocabulary().token(t).map_or(0, <[u8]>::len);
    eprintln!("device {}, model {} ({})", dev.name(), args.model, model.hash());
    println!(
        "# file\toverlap\tway\tblocks\ttokens\tbytes\ttable_bits\tbpb\trows\textra_rows\tseconds\tfreq_digest"
    );
    for file in &args.files {
        let blocks = read_blocks(file, &vocab)?;
        for &k in &args.overlaps {
            let t0 = Instant::now();
            let (slots, rows_per_pass) = if args.native {
                (blocks.len().min(args.slots), blocks.len().min(args.slots) * args.chunk)
            } else {
                (args.slots, args.slots * args.chunk)
            };
            let mut batch = CudaBatch::new(&model, slots, rows_per_pass)?;
            let mut jobs = Vec::new();
            for (b, blk) in blocks.iter().enumerate() {
                if blk.tokens.is_empty() {
                    continue;
                }
                if args.native {
                    jobs.push(native_job(b, &blk.tokens, eot));
                } else {
                    jobs.extend(window_jobs(b, &blk.tokens, n_ctx, usize::from(k), eot));
                }
            }
            if args.native {
                batch.set_overlap(Overlap::new(k))?;
            }
            let (freqs, rows) = run_jobs(&mut batch, &jobs, blocks.len(), args.chunk)?;
            let mut t = Totals::default();
            let mut h = Sha256::new();
            for (blk, f) in blocks.iter().zip(&freqs) {
                assert_eq!(f.len(), blk.tokens.len(), "a token without a frequency");
                t.tokens += f.len();
                t.bytes += blk.bytes;
                let mut bits = 0.0;
                for &x in f {
                    bits += f64::from(FREQ_BITS) - (x as f64).log2();
                    h.update(x.to_le_bytes());
                }
                t.bits += bits;
            }
            // The restart rows of the native path (they answer no input): for each block,
            // `ceil((n - n_ctx) / (n_ctx - 1 - k))` restarts of `k + 1` rows.
            t.rows = rows;
            if args.native {
                let k = usize::from(k);
                for blk in &blocks {
                    let over = blk.tokens.len().saturating_sub(n_ctx);
                    t.rows += over.div_ceil(n_ctx - 1 - k) * (k + 1);
                }
            }
            let digest: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
            if args.positions && !args.native {
                print_positions(file, k, &jobs, &freqs, n_ctx);
            }
            println!(
                "{file}\t{k}\t{}\t{}\t{}\t{}\t{:.3}\t{:.6}\t{}\t{:.4}\t{:.1}\t{digest}",
                if args.native { "native" } else { "windows" },
                blocks.len(),
                t.tokens,
                t.bytes,
                t.bits,
                t.bits / t.bytes as f64,
                t.rows,
                t.rows as f64 / t.tokens as f64 - 1.0,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

/// Prints the mean table size of a coded token for each group of 128 context positions. In
/// the default way, input `i` of a job runs at position `i`, and the coded tokens of a block
/// are in job order.
fn print_positions(file: &str, k: u16, jobs: &[Job], freqs: &[Vec<u64>], n_ctx: usize) {
    const GROUP: usize = 128;
    let mut sum = vec![(0.0f64, 0usize); n_ctx.div_ceil(GROUP)];
    let mut next = vec![0usize; freqs.len()];
    for job in jobs {
        for (pos, c) in job.coded.iter().enumerate() {
            if c.is_none() {
                continue;
            }
            let i = next[job.block];
            next[job.block] += 1;
            let f = freqs[job.block][i];
            let g = &mut sum[pos / GROUP];
            g.0 += f64::from(FREQ_BITS) - (f as f64).log2();
            g.1 += 1;
        }
    }
    println!("# positions {file} {k}");
    for (g, &(bits, n)) in sum.iter().enumerate() {
        if n > 0 {
            println!("# {}\t{n}\t{:.4}", g * GROUP, bits / n as f64);
        }
    }
}
