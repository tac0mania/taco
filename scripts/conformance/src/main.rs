//! The cross-machine conformance run of CI (plan §11.3, CLAUDE.md §1 rule 7, docs/ci.md).
//!
//! ```text
//! cp Cargo.lock scripts/conformance/Cargo.lock
//! cargo run --manifest-path scripts/conformance/Cargo.toml -- --out <directory>
//! ```
//!
//! The program makes fixed inputs (`inputs.rs`), runs them through `conformance/tiny.hsm` on
//! the CPU, and writes the SHA-256 of every output to `<directory>/SHA256SUMS`. Two machines
//! conform when their `SHA256SUMS` files are equal. For each input it hashes:
//!
//! - `hardshell/<input>.weights`: all HardShell weights, in the order of the coder (the order
//!   of `crates/taco/src/codec.rs`: the weights of `start_block`, then of each token except
//!   the last; the context starts again after 64 tokens);
//! - `coder/<input>.coded`: the range coder stream of the tokens with tables from these
//!   weights (`FreqTable::from_weights`, `taco::FREQ_BITS`);
//! - `coder/<input>.mixed`: the same with the match model and the mixer of the profile
//!   `logs` between the weights and the table (mixer rule version 1, ADR 0010);
//! - `taco/<case>.taco`: the container of `taco::fold` for each case of [`CASES`].
//!
//! NOTE: the tiny model has random weights, so a coded block of `taco::fold` is almost never
//! smaller than zstd or raw. Thus the containers alone do not test HardShell; the
//! `hardshell/` and `coder/` lines do.
//!
//! On this machine the program also checks, and fails if one check fails:
//!
//! - the SHA-256 of the committed conformance files (a damaged checkout stops the run);
//! - the step path with the fastest kernels of this CPU (NEON, AVX2), the step path with the
//!   portable kernels, the prefill path (`step_many`) and the reference session give the
//!   same weights (CLAUDE.md §1 rule 10);
//! - the range decoder gives the tokens back;
//! - `fold` with 1 and with 3 threads gives the same container, and `unfold` restores the
//!   input byte for byte.

mod inputs;

use std::error::Error;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use hardshell::gpt2::{ReferenceSession, Session, SessionConfig};
use hardshell::kernels::Kernels;
use sha2::{Digest, Sha256};
use taco::{
    BlockSize, FoldOptions, FoldProfile, Model, Threads, TokenId, Tokenizer, UnfoldOptions,
};
use taco_coder::{Decoder, Encoder, FreqBits, FreqTable, Symbol};
use taco_match::{MatchModel, MatchParams, Mixer, MixerParams};

type Res<T> = Result<T, Box<dyn Error>>;

/// The published SHA-256 of the committed conformance files (`conformance/README.md`).
const VECTORS: &[(&str, &str)] = &[
    ("conformance/tiny.hsm", "eb78b89a4f0f17c7245865aed22f86512a49e0272b7583a831c47e1a401418f8"),
    (
        "conformance/taco-v1-tiny.taco",
        "c3ff718b1c9682589836078b36d936836077744b03b1cc8af3a6ced0d5821b8a",
    ),
];

/// The `taco::fold` cases: (input, profile, block size; `None` is the default of 16 KiB).
const CASES: &[(&str, FoldProfile, Option<u32>)] = &[
    ("empty.bin", FoldProfile::Text, None),
    ("dasdas.txt", FoldProfile::Text, None),
    ("byte_cycle_1k.bin", FoldProfile::Text, Some(256)),
    ("random_3000.bin", FoldProfile::Auto, Some(1024)),
    ("english_12k.txt", FoldProfile::Text, None),
    ("english_12k.txt", FoldProfile::Text, Some(1024)),
    ("utf8_4k.txt", FoldProfile::Text, Some(1024)),
    ("server_24k.log", FoldProfile::Text, Some(4096)),
    ("server_24k.log", FoldProfile::Logs, Some(4096)),
    ("server_24k.log", FoldProfile::Auto, Some(4096)),
    ("nginx_8k.jsonl", FoldProfile::Logs, Some(2048)),
    ("records_8k.bin", FoldProfile::Auto, Some(2048)),
];

/// The prefill chunk of the encoder (`FOLD_CHUNK` in `crates/taco/src/fold.rs`).
const CHUNK: usize = 32;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn sha_u64(words: &[u64]) -> String {
    let mut h = Sha256::new();
    for w in words {
        h.update(w.to_le_bytes());
    }
    hex(&h.finalize())
}

/// The tokens that the trace feeds to the model: every token except the last. The weights of
/// `start_block` predict the first token, so `n` tokens need `n` weight rows.
fn fed(tokens: &[TokenId]) -> &[TokenId] {
    tokens.split_last().map_or(&[], |(_, rest)| rest)
}

/// The weights of the step path with `kernels`, in the order of the coder.
fn trace_step(model: &Model, tokens: &[TokenId], kernels: Kernels) -> Res<Vec<u64>> {
    let mut out = Vec::new();
    if tokens.is_empty() {
        return Ok(out);
    }
    let mut s = Session::with_config(model, SessionConfig::new().with_kernels(kernels));
    out.extend_from_slice(s.start_block()?);
    for &t in fed(tokens) {
        out.extend_from_slice(s.step(t)?);
    }
    Ok(out)
}

/// The weights of the prefill path (`step_many`, chunks of [`CHUNK`] tokens).
fn trace_prefill(model: &Model, tokens: &[TokenId]) -> Res<Vec<u64>> {
    let mut out = Vec::new();
    if tokens.is_empty() {
        return Ok(out);
    }
    let mut s = Session::with_config(model, SessionConfig::new().with_chunk(CHUNK));
    out.extend_from_slice(s.start_block()?);
    for part in fed(tokens).chunks(CHUNK) {
        for w in s.step_many(part)?.iter() {
            out.extend_from_slice(w);
        }
    }
    Ok(out)
}

/// The weights of the reference session (the oracle, ADR 0013).
fn trace_reference(model: &Model, tokens: &[TokenId]) -> Res<Vec<u64>> {
    let mut out = Vec::new();
    if tokens.is_empty() {
        return Ok(out);
    }
    let mut s = ReferenceSession::new(model);
    out.extend_from_slice(s.start_block()?);
    for &t in fed(tokens) {
        out.extend_from_slice(s.step(t)?);
    }
    Ok(out)
}

/// The match model and the mixer of one block (mixer rule version 1, `crates/taco/src/mix.rs`).
struct Mix {
    model: MatchModel,
    mixer: Mixer,
    weights: Vec<u64>,
}

impl Mix {
    fn new() -> Self {
        Self {
            model: MatchModel::new(MatchParams::V1),
            mixer: Mixer::new(MixerParams::V1),
            weights: Vec::new(),
        }
    }

    fn table(&mut self, hardshell: &[u64], bits: FreqBits) -> Res<FreqTable> {
        self.weights.clear();
        self.weights.extend_from_slice(hardshell);
        self.mixer.mix(&mut self.weights, self.model.predict())?;
        Ok(FreqTable::from_weights(&self.weights, bits)?)
    }

    fn observe(&mut self, token: TokenId) {
        let t = taco_match::TokenId::new(token.get());
        self.mixer.update(t);
        self.model.observe(t);
    }
}

/// The tables of the coder for each token: from the weights, or through the mixer.
struct Tables<'a> {
    rows: std::slice::ChunksExact<'a, u64>,
    mix: Option<Mix>,
    bits: FreqBits,
}

impl<'a> Tables<'a> {
    fn new(weights: &'a [u64], vocab: usize, mixed: bool) -> Res<Self> {
        Ok(Self {
            rows: weights.chunks_exact(vocab.max(1)),
            mix: mixed.then(Mix::new),
            bits: FreqBits::new(taco::FREQ_BITS)?,
        })
    }

    /// The table of the next token.
    fn next_table(&mut self) -> Res<FreqTable> {
        let row = self.rows.next().ok_or("the trace has fewer rows than tokens")?;
        match &mut self.mix {
            Some(m) => m.table(row, self.bits),
            None => Ok(FreqTable::from_weights(row, self.bits)?),
        }
    }

    /// Tells the mixer the token that was coded or decoded.
    fn observe(&mut self, token: TokenId) {
        if let Some(m) = &mut self.mix {
            m.observe(token);
        }
    }
}

/// Codes `tokens` with the tables of `weights`, then decodes the stream and checks that the
/// tokens come back. Gives the stream.
fn code(weights: &[u64], vocab: usize, tokens: &[TokenId], mixed: bool) -> Res<Vec<u8>> {
    let mut tables = Tables::new(weights, vocab, mixed)?;
    let mut enc = Encoder::new();
    for &t in tokens {
        enc.encode(&tables.next_table()?, Symbol::new(t.get()))?;
        tables.observe(t);
    }
    let stream = enc.finish();
    let mut tables = Tables::new(weights, vocab, mixed)?;
    let mut dec = Decoder::new(&stream);
    for &t in tokens {
        let got = dec.decode(&tables.next_table()?);
        if got.get() != t.get() {
            return Err(
                format!("the decoder gave token {} for token {}", got.get(), t.get()).into()
            );
        }
        tables.observe(t);
    }
    Ok(stream)
}

fn profile_name(p: FoldProfile) -> &'static str {
    match p {
        FoldProfile::Logs => "logs",
        FoldProfile::Auto => "auto",
        _ => "text",
    }
}

struct Args {
    out: PathBuf,
    root: PathBuf,
}

fn args() -> Res<Args> {
    let mut out = None;
    let mut it = std::env::args_os().skip(1);
    while let Some(a) = it.next() {
        match a.to_str() {
            Some("--out") => out = it.next().map(PathBuf::from),
            _ => return Err("usage: taco-conformance --out <directory>".into()),
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    Ok(Args { out: out.ok_or("usage: taco-conformance --out <directory>")?, root })
}

fn write(path: &Path, data: &[u8]) -> Res<()> {
    std::fs::write(path, data).map_err(|e| format!("cannot write {}: {e}", path.display()).into())
}

fn main() -> Res<()> {
    let args = args()?;
    std::fs::create_dir_all(args.out.join("taco"))?;
    let mut sums: Vec<(String, String)> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut report = String::new();

    for (rel, want) in VECTORS {
        let got = sha(&std::fs::read(args.root.join(rel))?);
        if got != *want {
            return Err(format!("{rel} has SHA-256 {got}, not {want}: damaged checkout?").into());
        }
        let name = rel.rsplit('/').next().unwrap_or(rel);
        sums.push((format!("vector/{name}"), got));
    }

    let model_file = std::fs::read(args.root.join("conformance/tiny.hsm"))?;
    let model = Model::load(&model_file)?;
    let merges: Vec<(TokenId, TokenId)> =
        model.vocabulary().merges().unwrap_or(&[]).iter().map(|m| (m.left(), m.right())).collect();
    let tokenizer = Tokenizer::for_vocabulary(model.vocabulary(), model.eot_token(), &merges)?;
    let vocab = model.vocab_size();
    let fast = Kernels::detect();
    writeln!(
        report,
        "machine: {} {}, kernels {fast:?}, model tiny.hsm ({} merges)",
        std::env::consts::OS,
        std::env::consts::ARCH,
        merges.len()
    )?;
    writeln!(
        report,
        "{:<20} {:>7} {:>7} {:>8} {:>8}",
        "input", "bytes", "tokens", "coded", "mixed"
    )?;

    let inputs = inputs::all();
    for (name, data) in &inputs {
        sums.push((format!("input/{name}"), sha(data)));
        let tokens = tokenizer.encode(data);
        let weights = trace_step(&model, &tokens, fast)?;
        for (path, other) in [
            ("portable kernels", trace_step(&model, &tokens, Kernels::portable())?),
            ("prefill", trace_prefill(&model, &tokens)?),
            ("reference session", trace_reference(&model, &tokens)?),
        ] {
            if other != weights {
                failures.push(format!("{name}: the {path} gives other weights than {fast:?}"));
            }
        }
        let coded = code(&weights, vocab, &tokens, false)?;
        let mixed = code(&weights, vocab, &tokens, true)?;
        sums.push((format!("hardshell/{name}.weights"), sha_u64(&weights)));
        sums.push((format!("coder/{name}.coded"), sha(&coded)));
        sums.push((format!("coder/{name}.mixed"), sha(&mixed)));
        writeln!(
            report,
            "{name:<20} {:>7} {:>7} {:>8} {:>8}",
            data.len(),
            tokens.len(),
            coded.len(),
            mixed.len()
        )?;
    }

    writeln!(report, "\n{:<36} {:>7} {:>7}  blocks", "fold case", "bytes", "taco")?;
    let three = Threads::new(3).ok_or("3 threads")?;
    let two = Threads::new(2).ok_or("2 threads")?;
    for &(name, profile, bs) in CASES {
        let data = inputs.iter().find(|(n, _)| *n == name).map(|(_, d)| d).ok_or(name)?;
        let mut opts = FoldOptions::new().with_profile(profile);
        if let Some(b) = bs {
            opts = opts.with_block_size(BlockSize::new(b)?);
        }
        let tag = format!(
            "{name}.{}.{}",
            profile_name(profile),
            bs.map_or_else(|| "bdefault".to_owned(), |b| format!("b{b}"))
        );
        let one = taco::fold(&model, &tokenizer, data, &opts.with_threads(Threads::ONE))?;
        let many = taco::fold(&model, &tokenizer, data, &opts.with_threads(three))?;
        if many.container != one.container {
            failures.push(format!("{tag}: 1 thread and 3 threads give different containers"));
        }
        let back =
            taco::unfold(Some(&model), &one.container, &UnfoldOptions::new().with_threads(two))?;
        if back != *data {
            failures.push(format!("{tag}: unfold did not restore the input"));
        }
        write(&args.out.join("taco").join(format!("{tag}.taco")), &one.container)?;
        sums.push((format!("taco/{tag}.taco"), sha(&one.container)));
        let s = one.stats;
        let raw = s.blocks - s.coded_blocks - s.mixed_blocks - s.zstd_blocks;
        writeln!(
            report,
            "{tag:<36} {:>7} {:>7}  {} ({} coded, {} mixed, {} zstd, {raw} raw)",
            s.original_len,
            s.container_len,
            s.blocks,
            s.coded_blocks,
            s.mixed_blocks,
            s.zstd_blocks
        )?;
    }

    sums.sort();
    let list: String = sums.iter().map(|(k, v)| format!("{v}  {k}\n")).collect();
    write(&args.out.join("SHA256SUMS"), list.as_bytes())?;
    write(&args.out.join("summary.txt"), report.as_bytes())?;
    print!("{report}");
    println!("\nwrote {} hashes to {}", sums.len(), args.out.join("SHA256SUMS").display());
    if failures.is_empty() {
        println!("conformance: PASS on this machine (compare SHA256SUMS with other machines)");
        Ok(())
    } else {
        for f in &failures {
            eprintln!("conformance: FAIL: {f}");
        }
        Err(format!("{} checks failed", failures.len()).into())
    }
}
