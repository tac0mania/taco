//! GPT-2 on the GPU: equivalence with the CPU reference on real text, and speed.
//!
//!   gpt2 equiv <model.hsm> <text files...>    SHA-256 of all weights of each window, CPU and GPU
//!   gpt2 bench <model.hsm> [batch sizes]      tokens/s for each batch size
//!
//! `HARDSHELL_GPU_ADAPTER` selects the adapter (name or backend). The text is tokenized with
//! the longest vocabulary match: deterministic, not the canonical GPT-2 tokenization, which is
//! enough for an equivalence test.
#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    missing_docs
)]
#![allow(clippy::cast_precision_loss, clippy::float_arithmetic, clippy::cast_possible_truncation)]

use std::collections::HashMap;
use std::time::Instant;

use hardshell::TokenId;
use hardshell::gpt2::{Model, Session};
use hardshell_gpu::{AdapterChoice, CoderInterval, CoderQuery, GpuBatch, GpuDevice, GpuModel};
use sha2::{Digest, Sha256};

fn device() -> GpuDevice {
    let choice = match std::env::var("HARDSHELL_GPU_ADAPTER") {
        Ok(n) if !n.is_empty() => AdapterChoice::NameContains(n),
        _ => AdapterChoice::BestHardware,
    };
    let d = GpuDevice::open(&choice).expect("no GPU adapter");
    println!("adapter: {}", d.adapter());
    d
}

/// Longest-match tokenization with the vocabulary of the model.
fn tokenize(model: &Model, text: &[u8]) -> Vec<TokenId> {
    let mut map: HashMap<&[u8], u32> = HashMap::new();
    let mut longest = 1;
    for id in 0..model.vocab_size() as u32 {
        if id == model.eot_token().get() {
            continue;
        }
        if let Some(t) = model.vocabulary().token(TokenId::new(id)) {
            longest = longest.max(t.len());
            map.entry(t).or_insert(id);
        }
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let mut n = longest.min(text.len() - i);
        loop {
            if let Some(&id) = map.get(&text[i..i + n]) {
                out.push(TokenId::new(id));
                i += n;
                break;
            }
            n -= 1;
            assert!(n > 0, "a byte without a token");
        }
    }
    out
}

fn hex(h: &[u8]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

fn equiv(bytes: &[u8], files: &[String]) {
    let cpu = Model::load(bytes).unwrap();
    println!("model {}", cpu.hash());
    let dev = device();
    let gpu = GpuModel::load(&dev, bytes).unwrap();
    // Windows: 256 tokens from each file, and one window of 1,100 tokens (the context restarts
    // at 1,024).
    let mut windows: Vec<(String, Vec<TokenId>)> = Vec::new();
    for f in files {
        let t = tokenize(&cpu, &std::fs::read(f).unwrap());
        let name = f.rsplit('/').next().unwrap_or(f).to_owned();
        windows.push((name, t.iter().copied().take(256).collect()));
    }
    if let Some(f) = files.first() {
        let t = tokenize(&cpu, &std::fs::read(f).unwrap());
        windows.push(("long-1100".into(), t.into_iter().take(1100).collect()));
    }
    // GPU: all windows in one batch, one slot each; a slot leaves the step when its window ends.
    let t0 = Instant::now();
    let mut batch = GpuBatch::new(&gpu, windows.len()).unwrap();
    let mut gh: Vec<Sha256> = windows.iter().map(|_| Sha256::new()).collect();
    let first: Vec<(usize, TokenId)> = (0..windows.len()).map(|s| (s, gpu.eot_token())).collect();
    batch.step(&first).unwrap();
    for (s, h) in gh.iter_mut().enumerate() {
        batch.weights(s).unwrap().iter().for_each(|x| h.update(x.to_le_bytes()));
    }
    let max_len = windows.iter().map(|w| w.1.len()).max().unwrap_or(0);
    for i in 0..max_len {
        let inputs: Vec<(usize, TokenId)> =
            windows.iter().enumerate().filter_map(|(s, w)| w.1.get(i).map(|&t| (s, t))).collect();
        batch.step(&inputs).unwrap();
        for &(s, _) in &inputs {
            batch.weights(s).unwrap().iter().for_each(|x| gh[s].update(x.to_le_bytes()));
        }
    }
    let gpu_s = t0.elapsed().as_secs_f64();
    // GPU coder op: the encode interval of the next token at every step of every window.
    let next = |w: &[TokenId], i: usize| w.get(i).copied().unwrap_or(TokenId::new(0));
    let mut coded = GpuBatch::new(&gpu, windows.len()).unwrap();
    let mut giv: Vec<Vec<CoderInterval>> = windows.iter().map(|_| Vec::new()).collect();
    let q0: Vec<CoderQuery> = windows.iter().map(|w| CoderQuery::Encode(next(&w.1, 0))).collect();
    for (s, iv) in coded.step_coded(&first, &q0, 32).unwrap().into_iter().enumerate() {
        giv[s].push(iv);
    }
    for i in 0..max_len {
        let inputs: Vec<(usize, TokenId)> =
            windows.iter().enumerate().filter_map(|(s, w)| w.1.get(i).map(|&t| (s, t))).collect();
        let queries: Vec<CoderQuery> =
            inputs.iter().map(|&(s, _)| CoderQuery::Encode(next(&windows[s].1, i + 1))).collect();
        for (&(s, _), iv) in inputs.iter().zip(coded.step_coded(&inputs, &queries, 32).unwrap()) {
            giv[s].push(iv);
        }
    }
    let bits = taco_coder::FreqBits::new(32).unwrap();
    let cpu_iv = |w: &[u64], t: TokenId| {
        let table = taco_coder::FreqTable::from_weights(w, bits).unwrap();
        let mut start = 0;
        let mut freq = 0;
        for (i, f) in table.freqs().enumerate() {
            if i == t.get() as usize {
                freq = f.get();
                break;
            }
            start += f.get();
        }
        (start, freq)
    };
    let mut iv_diff = 0usize;
    let mut iv_count = 0usize;
    // CPU reference, one session.
    let t0 = Instant::now();
    let mut sess = Session::new(&cpu);
    let mut all_equal = true;
    let mut tokens = 0;
    for ((name, w), g) in windows.iter().zip(gh) {
        let mut h = Sha256::new();
        let s = windows.iter().position(|x| x.0 == *name).unwrap();
        let wt = sess.start_block().unwrap();
        wt.iter().for_each(|x| h.update(x.to_le_bytes()));
        let check = |k: usize, wt: &[u64], d: &mut usize| {
            let (st, fr) = cpu_iv(wt, next(w, k));
            let g = giv[s][k];
            if g.start != st || g.freq != fr || g.symbol != next(w, k) {
                *d += 1;
            }
        };
        check(0, wt, &mut iv_diff);
        iv_count += 1;
        for (k, &t) in w.iter().enumerate() {
            let wt = sess.step(t).unwrap();
            wt.iter().for_each(|x| h.update(x.to_le_bytes()));
            check(k + 1, wt, &mut iv_diff);
            iv_count += 1;
        }
        tokens += w.len() + 1;
        let (c, g) = (hex(&h.finalize()), hex(&g.finalize()));
        let ok = c == g;
        all_equal &= ok;
        println!(
            "window {name:<16} {:>5} steps  cpu {c}  gpu {g}  {}",
            w.len() + 1,
            if ok { "EQUAL" } else { "DIFFERENT" }
        );
    }
    let cpu_s = t0.elapsed().as_secs_f64();
    println!(
        "{tokens} steps; gpu {gpu_s:.1} s (batch {}), cpu {cpu_s:.1} s (1 thread)",
        windows.len()
    );
    println!(
        "coder intervals (GPU coder op against the CPU FreqTable, 32 bits): {iv_count} checked, {iv_diff} different"
    );
    println!(
        "{}",
        if all_equal && iv_diff == 0 { "ALL WINDOWS EQUAL" } else { "SOME WINDOWS DIFFERENT" }
    );
    assert!(all_equal && iv_diff == 0);
}

fn bench(bytes: &[u8], sizes: &[usize]) {
    let dev = device();
    let gpu = GpuModel::load(&dev, bytes).unwrap();
    println!(
        "weights on the GPU: {:.1} MB; one slot: {:.1} MB (est.)",
        gpu.weight_bytes() as f64 / 1e6,
        gpu.slot_bytes() as f64 / 1e6
    );
    for &b in sizes {
        let mut batch = match GpuBatch::new(&gpu, b) {
            Ok(x) => x,
            Err(e) => {
                println!("batch {b}: not possible: {e}");
                continue;
            }
        };
        let all: Vec<usize> = (0..b).collect();
        let tok = |s: usize, i: usize| TokenId::new(((s * 7919 + i * 104_729) % 50_000) as u32);
        let step = |batch: &mut GpuBatch<'_>, i: usize| {
            let inputs: Vec<(usize, TokenId)> = all.iter().map(|&s| (s, tok(s, i))).collect();
            batch.step(&inputs).unwrap();
        };
        let mut i = 0;
        for _ in 0..3 {
            step(&mut batch, i);
            i += 1;
        }
        let n = 16;
        let t0 = Instant::now();
        for _ in 0..n {
            step(&mut batch, i);
            i += 1;
        }
        let early = t0.elapsed().as_secs_f64();
        // Fill the context to position 512, then measure again.
        while i < 512 {
            step(&mut batch, i);
            i += 1;
        }
        let t0 = Instant::now();
        for _ in 0..n {
            step(&mut batch, i);
            i += 1;
        }
        let late = t0.elapsed().as_secs_f64();
        // The same steps without the read back of the weights.
        let t0 = Instant::now();
        for _ in 0..n {
            let inputs: Vec<(usize, TokenId)> = all.iter().map(|&s| (s, tok(s, i))).collect();
            batch.step_without_readback(&inputs).unwrap();
            i += 1;
        }
        let no_read = t0.elapsed().as_secs_f64();
        let inputs: Vec<(usize, TokenId)> = all.iter().map(|&s| (s, tok(s, i))).collect();
        if let Some(p) = batch.profile_step(&inputs).unwrap() {
            let period = 1.0;
            let total: u64 = p.iter().map(|x| x.1).sum();
            let parts: Vec<String> =
                p.iter().map(|(l, t)| format!("{l} {:.2}", *t as f64 * period / 1e6)).collect();
            println!(
                "  batch {b}: GPU time by op (ms, timestamps): total {:.2}; {}",
                total as f64 * period / 1e6,
                parts.join(", ")
            );
        }
        // The same steps with the coder op on the GPU: 16 bytes read back for each token.
        let t0 = Instant::now();
        for _ in 0..n {
            let inputs: Vec<(usize, TokenId)> = all.iter().map(|&s| (s, tok(s, i))).collect();
            let queries: Vec<CoderQuery> =
                all.iter().map(|&s| CoderQuery::Encode(tok(s, i + 1))).collect();
            std::hint::black_box(batch.step_coded(&inputs, &queries, 32).unwrap());
            i += 1;
        }
        let coded = t0.elapsed().as_secs_f64();
        println!(
            "  batch {b}: {:.1} ms per step with the coder op on the GPU ({:.1} tokens/s)",
            1e3 * coded / n as f64,
            (b * n) as f64 / coded
        );
        let bits = taco_coder::FreqBits::new(32).unwrap();
        let t0 = Instant::now();
        for s in 0..b {
            let t = taco_coder::FreqTable::from_weights(batch.weights(s).unwrap(), bits).unwrap();
            std::hint::black_box(t);
        }
        println!(
            "  batch {b}: FreqTable::from_weights on the CPU: {:.0} us per table (1 thread)",
            1e6 * t0.elapsed().as_secs_f64() / b as f64
        );
        let tps = |s: f64| (b * n) as f64 / s;
        println!(
            "  batch {b}: {:.1} ms per step without the read back ({:.1} tokens/s)",
            1e3 * no_read / n as f64,
            tps(no_read)
        );
        println!(
            "batch {b:>4}: {:>8.1} tokens/s at positions 3-19, {:>8.1} tokens/s at 512-528; {:.1} ms per step; GPU memory {:.0} MB (weights + batch, measured allocations)",
            tps(early),
            tps(late),
            1e3 * early / n as f64,
            (gpu.weight_bytes() + batch.gpu_bytes()) as f64 / 1e6
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[2]).unwrap();
    match args[1].as_str() {
        "equiv" => equiv(&bytes, &args[3..]),
        "bench" => {
            let sizes: Vec<usize> = if args.len() > 3 {
                args[3..].iter().map(|s| s.parse().unwrap()).collect()
            } else {
                vec![1, 8, 32, 128]
            };
            bench(&bytes, &sizes);
        }
        _ => panic!("usage: gpt2 equiv|bench <model.hsm> ..."),
    }
}
