//! Dump the facts of `logs.gpt2pass` with hardshell-cuda (prefill of each window).
//!
//! Usage: `logs-dump MODEL.hsm IN.bin OUT.bin`
//!
//! IN: `u32 windows`, then for each window `u32 n`, `n` u32 tokens (the targets), and `n * 8`
//! i32 candidates (-1 pads). The model input of a window is `<|endoftext|>` and the first
//! `n - 1` tokens (plan §7.7). OUT, for each token in input order (all little-endian):
//! `W u64` (weight sum), `wa u64`, `fa u64` (the frequency of the target in
//! `FreqTable::from_weights(w, 32)`), `top_id 8 x u32`, `top_w 8 x u64`, `wc 8 x u64`.
//! NOTE: a measurement program with a clock and threads; it is not the prediction path.

use std::io::{Read, Write};
use std::time::Instant;

use hardshell::TokenId;
use hardshell_cuda::{CudaBatch, CudaDevice, CudaModel};
use taco_coder::{FreqBits, FreqTable, Symbol};

const TOP: usize = 8;
const CAND: usize = 8;
const ROW_BYTES: usize = 8 * 3 + 4 * TOP + 8 * TOP + 8 * CAND;

fn u32_at(b: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}

fn facts(w: &[u64], target: usize, cand: &[i32], out: &mut Vec<u8>) {
    let total: u64 = w.iter().sum();
    let table = FreqTable::from_weights(w, FreqBits::new(32).unwrap()).unwrap();
    let fa = table.freq(Symbol::new(target as u32)).unwrap().get();
    // The 8 largest weights, the lowest id first on a tie.
    let mut top: Vec<(u64, u32)> = Vec::with_capacity(TOP + 1);
    for (i, &x) in w.iter().enumerate() {
        if top.len() < TOP || x > top[TOP - 1].0 {
            let pos = top.partition_point(|&(y, _)| y >= x);
            top.insert(pos, (x, i as u32));
            top.truncate(TOP);
        }
    }
    out.extend_from_slice(&total.to_le_bytes());
    out.extend_from_slice(&w[target].to_le_bytes());
    out.extend_from_slice(&fa.to_le_bytes());
    for &(_, i) in &top {
        out.extend_from_slice(&i.to_le_bytes());
    }
    for &(x, _) in &top {
        out.extend_from_slice(&x.to_le_bytes());
    }
    for &c in cand {
        let v = if c >= 0 { w[c as usize] } else { 0 };
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).unwrap();
    let mut inp = Vec::new();
    std::fs::File::open(&args[2]).unwrap().read_to_end(&mut inp).unwrap();
    let mut out_file = std::fs::File::create(&args[3]).unwrap();
    let dev = CudaDevice::open(0).unwrap();
    let gpu = CudaModel::load(&dev, &bytes).unwrap();
    eprintln!("device {} model {}", dev.name(), gpu.hash());
    let v = gpu.vocab_size();
    let eot = gpu.eot_token();
    let mut batch = CudaBatch::new(&gpu, 1, 1024).unwrap();
    let mut p = 0usize;
    let windows = u32_at(&inp, &mut p) as usize;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    let t0 = Instant::now();
    let mut tokens_done = 0usize;
    for wi in 0..windows {
        let n = u32_at(&inp, &mut p) as usize;
        let toks: Vec<u32> = (0..n).map(|_| u32_at(&inp, &mut p)).collect();
        let cand: Vec<i32> = (0..n * CAND).map(|_| u32_at(&inp, &mut p) as i32).collect();
        batch.reset(0);
        let mut inputs = vec![(0usize, eot)];
        inputs.extend(toks[..n - 1].iter().map(|&t| (0usize, TokenId::new(t))));
        let w = batch.run_weights(&inputs).unwrap();
        let chunk = n.div_ceil(threads);
        let parts: Vec<Vec<u8>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|k| {
                    let (w, toks, cand) = (&w, &toks, &cand);
                    s.spawn(move || {
                        let mut o = Vec::with_capacity(chunk * ROW_BYTES);
                        for r in k * chunk..((k + 1) * chunk).min(n) {
                            facts(&w[r * v..(r + 1) * v], toks[r] as usize,
                                  &cand[r * CAND..(r + 1) * CAND], &mut o);
                        }
                        o
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for part in parts {
            out_file.write_all(&part).unwrap();
        }
        tokens_done += n;
        if wi % 20 == 0 || wi + 1 == windows {
            let el = t0.elapsed().as_secs_f64();
            eprintln!("{}/{} windows, {:.0} tokens/s, {:.1} min", wi + 1, windows,
                      tokens_done as f64 / el, el / 60.0);
        }
    }
}
