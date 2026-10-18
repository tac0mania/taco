//! GPT-2 on the CUDA GPU: equivalence with the CPU reference on real text, and speed.
//!
//!   gpt2 equiv <model.hsm> <text files...>   per-window SHA-256 of all weights: CPU, CUDA decode,
//!                                            CUDA prefill; and every coder interval
//!   gpt2 bench <model.hsm> [batch sizes]     decode tokens/s, prefill tokens/s, time by op
//!
//! The text is tokenized with the longest vocabulary match: deterministic, not the canonical
//! GPT-2 tokenization, which is enough for an equivalence test.
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
use hardshell_cuda::{CoderQuery, CudaBatch, CudaDevice, CudaModel};
use sha2::{Digest, Sha256};

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

fn hash_rows(w: &[u64]) -> String {
    let mut h = Sha256::new();
    w.iter().for_each(|x| h.update(x.to_le_bytes()));
    hex(&h.finalize())
}

/// The CPU reference for one window: the SHA-256 of all weights, and the coder interval
/// (start, freq) of the next token at each step.
fn cpu_window(cpu: &Model, w: &[TokenId]) -> (String, Vec<(u64, u64)>) {
    let bits = taco_coder::FreqBits::new(32).unwrap();
    let next = |i: usize| w.get(i).copied().unwrap_or(TokenId::new(0));
    let mut s = Session::new(cpu);
    let mut h = Sha256::new();
    let mut ivs = Vec::new();
    let mut take = |wt: &[u64], i: usize, h: &mut Sha256| {
        wt.iter().for_each(|x| h.update(x.to_le_bytes()));
        let t = taco_coder::FreqTable::from_weights(wt, bits).unwrap();
        let (mut start, mut freq) = (0, 0);
        for (k, f) in t.freqs().enumerate() {
            if k == next(i).get() as usize {
                freq = f.get();
                break;
            }
            start += f.get();
        }
        ivs.push((start, freq));
    };
    take(s.start_block().unwrap(), 0, &mut h);
    for (i, &t) in w.iter().enumerate() {
        take(s.step(t).unwrap(), i + 1, &mut h);
    }
    (hex(&h.finalize()), ivs)
}

fn equiv(bytes: &[u8], files: &[String]) {
    let cpu = Model::load(bytes).unwrap();
    println!("model {}", cpu.hash());
    let dev = CudaDevice::open(0).unwrap();
    println!("device: {} {:?}", dev.name(), dev.compute_capability());
    let gpu = CudaModel::load(&dev, bytes).unwrap();
    let mut windows: Vec<(String, Vec<TokenId>)> = Vec::new();
    for f in files {
        let t = tokenize(&cpu, &std::fs::read(f).unwrap());
        windows.push((
            f.rsplit('/').next().unwrap_or(f).to_owned(),
            t.into_iter().take(256).collect(),
        ));
    }
    if let Some(f) = files.first() {
        let t = tokenize(&cpu, &std::fs::read(f).unwrap());
        windows.push(("long-1100".into(), t.into_iter().take(1100).collect()));
    }
    let v = gpu.vocab_size();
    let eot = gpu.eot_token();
    // CUDA decode: one slot for each window, one token for each slot in each call.
    let t0 = Instant::now();
    let mut batch = CudaBatch::new(&gpu, windows.len(), windows.len() + 1).unwrap();
    let mut dh: Vec<Sha256> = windows.iter().map(|_| Sha256::new()).collect();
    let max_len = windows.iter().map(|w| w.1.len()).max().unwrap_or(0);
    for i in 0..=max_len {
        let inputs: Vec<(usize, TokenId)> = windows
            .iter()
            .enumerate()
            .filter_map(
                |(s, w)| if i == 0 { Some((s, eot)) } else { w.1.get(i - 1).map(|&t| (s, t)) },
            )
            .collect();
        let out = batch.run_weights(&inputs).unwrap();
        for (k, &(s, _)) in inputs.iter().enumerate() {
            out[k * v..(k + 1) * v].iter().for_each(|x| dh[s].update(x.to_le_bytes()));
        }
    }
    let decode: Vec<String> = dh.into_iter().map(|h| hex(&h.finalize())).collect();
    let decode_s = t0.elapsed().as_secs_f64();
    // CUDA prefill: each window in one call (passes of up to 512 rows).
    let t0 = Instant::now();
    let mut pre = CudaBatch::new(&gpu, 1, 512).unwrap();
    let prefill: Vec<String> = windows
        .iter()
        .map(|(_, w)| {
            pre.reset(0);
            let mut inputs = vec![(0, eot)];
            inputs.extend(w.iter().map(|&t| (0, t)));
            hash_rows(&pre.run_weights(&inputs).unwrap())
        })
        .collect();
    let prefill_s = t0.elapsed().as_secs_f64();
    // CUDA coder op in prefill: the interval of the next token at each step.
    let mut coded = CudaBatch::new(&gpu, 1, 512).unwrap();
    let giv: Vec<Vec<(u64, u64)>> = windows
        .iter()
        .map(|(_, w)| {
            coded.reset(0);
            let mut inputs = vec![(0, eot)];
            inputs.extend(w.iter().map(|&t| (0, t)));
            let queries: Vec<CoderQuery> = (0..inputs.len())
                .map(|i| CoderQuery::Encode(w.get(i).copied().unwrap_or(TokenId::new(0))))
                .collect();
            coded
                .run_coded(&inputs, &queries, 32)
                .unwrap()
                .iter()
                .map(|iv| (iv.start, iv.freq))
                .collect()
        })
        .collect();
    // CPU reference, one thread for each window.
    let t0 = Instant::now();
    let cpu_res: Vec<(String, Vec<(u64, u64)>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = windows.iter().map(|(_, w)| sc.spawn(|| cpu_window(&cpu, w))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let cpu_s = t0.elapsed().as_secs_f64();
    let mut ok = true;
    let mut steps = 0;
    let mut iv_diff = 0;
    for (i, (name, w)) in windows.iter().enumerate() {
        let (c, civ) = &cpu_res[i];
        let same = *c == decode[i] && *c == prefill[i];
        ok &= same;
        iv_diff += civ.iter().zip(&giv[i]).filter(|(a, b)| a != b).count()
            + civ.len().abs_diff(giv[i].len());
        steps += w.len() + 1;
        println!(
            "window {name:<14} {:>5} steps  cpu {c}  decode {}  prefill {}  {}",
            w.len() + 1,
            &decode[i][..16],
            &prefill[i][..16],
            if same { "EQUAL" } else { "DIFFERENT" }
        );
    }
    println!(
        "{steps} steps: cuda decode {decode_s:.1} s, cuda prefill {prefill_s:.1} s, cpu {cpu_s:.1} s ({} threads)",
        windows.len()
    );
    println!(
        "coder intervals (CUDA coder op against the CPU FreqTable, 32 bits): {steps} checked, {iv_diff} different"
    );
    let all = ok && iv_diff == 0;
    println!("{}", if all { "ALL WINDOWS EQUAL" } else { "SOME WINDOWS DIFFERENT" });
    assert!(all);
}

fn tok(s: usize, i: usize) -> TokenId {
    TokenId::new(((s * 7919 + i * 104_729) % 50_000) as u32)
}

fn bench(bytes: &[u8], sizes: &[usize]) {
    let dev = CudaDevice::open(0).unwrap();
    let (free0, total) = dev.memory().unwrap();
    println!(
        "device: {} {:?}; memory free {:.0} MB of {:.0} MB",
        dev.name(),
        dev.compute_capability(),
        free0 as f64 / 1e6,
        total as f64 / 1e6
    );
    let gpu = CudaModel::load(&dev, bytes).unwrap();
    println!(
        "weights on the GPU: {:.1} MB; KV cache {:.1} MB per slot",
        gpu.weight_bytes() as f64 / 1e6,
        gpu.cache_bytes_per_slot() as f64 / 1e6
    );
    for &b in sizes {
        let mut batch = match CudaBatch::new(&gpu, b, b) {
            Ok(x) => x,
            Err(e) => {
                println!("batch {b}: not possible: {e}");
                continue;
            }
        };
        let used = free0 - dev.memory().unwrap().0;
        let all: Vec<usize> = (0..b).collect();
        let inputs =
            |i: usize| -> Vec<(usize, TokenId)> { all.iter().map(|&s| (s, tok(s, i))).collect() };
        let queries = |i: usize| -> Vec<CoderQuery> {
            all.iter().map(|&s| CoderQuery::Encode(tok(s, i + 1))).collect()
        };
        let mut i = 0;
        let coded = |batch: &mut CudaBatch<'_>, i: &mut usize, n: usize| {
            let t0 = Instant::now();
            for _ in 0..n {
                batch.run_coded(&inputs(*i), &queries(*i), 32).unwrap();
                *i += 1;
            }
            t0.elapsed().as_secs_f64()
        };
        coded(&mut batch, &mut i, 3);
        let n = 16;
        let early = coded(&mut batch, &mut i, n);
        let t0 = Instant::now();
        for _ in 0..4 {
            batch.run_weights(&inputs(i)).unwrap();
            i += 1;
        }
        let full = t0.elapsed().as_secs_f64() / 4.0;
        while i < 512 {
            batch.run_without_readback(&inputs(i)).unwrap();
            i += 1;
        }
        let late = coded(&mut batch, &mut i, n);
        let tps = |s: f64| (b * n) as f64 / s;
        println!(
            "decode batch {b:>4}: {:>9.0} tokens/s at positions 3-19, {:>9.0} at 512-528 (coder op, 24 bytes back per token); {:.2} ms per step; with all weights read back {:.0} tokens/s; GPU memory {:.0} MB (measured)",
            tps(early),
            tps(late),
            1e3 * early / n as f64,
            b as f64 / full,
            (used as f64 + gpu.weight_bytes() as f64) / 1e6
        );
        if b == *sizes.iter().max().unwrap_or(&b) {
            let times: Vec<(&'static str, f64)> = batch
                .profile(&inputs(i))
                .unwrap()
                .into_iter()
                .map(|(l, ns)| (l, ns as f64 * 1e-9))
                .collect();
            let tot: f64 = times.iter().map(|x| x.1).sum();
            let parts: Vec<String> =
                times.iter().map(|(l, t)| format!("{l} {:.2}", t * 1e3)).collect();
            println!(
                "  time by op at batch {b}, position {i} (ms, GPU timer marks): total {:.2}; {}",
                tot * 1e3,
                parts.join(", ")
            );
        }
    }
    // A full window: every slot decodes positions 0 to 1,023 (the whole context of GPT-2).
    if let Some(&b) = sizes.iter().max() {
        let mut batch = CudaBatch::new(&gpu, b, b).unwrap();
        let all: Vec<usize> = (0..b).collect();
        let t0 = Instant::now();
        for i in 0..gpu.n_ctx() {
            let inputs: Vec<(usize, TokenId)> = all.iter().map(|&s| (s, tok(s, i))).collect();
            let queries: Vec<CoderQuery> =
                all.iter().map(|&s| CoderQuery::Encode(tok(s, i + 1))).collect();
            batch.run_coded(&inputs, &queries, 32).unwrap();
        }
        let s = t0.elapsed().as_secs_f64();
        println!(
            "decode batch {b}, full window (positions 0 to {}): {:.0} tokens/s ({:.1} s, coder op)",
            gpu.n_ctx() - 1,
            (b * gpu.n_ctx()) as f64 / s,
            s
        );
    }
    // Prefill: windows of 512 tokens, 8 slots, passes of 512 rows.
    for (slots, len, rows) in [(1usize, 512usize, 512usize), (8, 512, 512), (8, 1024, 1024)] {
        let mut batch = match CudaBatch::new(&gpu, slots, rows) {
            Ok(x) => x,
            Err(e) => {
                println!("prefill {slots} x {len}: not possible: {e}");
                continue;
            }
        };
        let mut inputs = Vec::new();
        for s in 0..slots {
            inputs.extend((0..len).map(|i| (s, tok(s, i))));
        }
        let queries: Vec<CoderQuery> =
            inputs.iter().map(|&(s, _)| CoderQuery::Encode(tok(s, 1))).collect();
        batch
            .run_coded(&inputs[..rows.min(inputs.len())], &queries[..rows.min(inputs.len())], 32)
            .unwrap();
        (0..slots).for_each(|s| batch.reset(s));
        let t0 = Instant::now();
        batch.run_coded(&inputs, &queries, 32).unwrap();
        let s = t0.elapsed().as_secs_f64();
        println!(
            "prefill {slots} slots x {len} tokens (passes of {rows} rows): {:.0} tokens/s ({:.1} ms), coder op",
            inputs.len() as f64 / s,
            s * 1e3
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
