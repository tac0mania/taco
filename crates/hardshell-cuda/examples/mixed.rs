//! The mixed coder op (profile `logs`, ADR 0010) on real data: equal coded bytes, and speed.
//!
//!   mixed equiv <model.hsm> <bytes> <overlap K> <files...>
//!       For the first <bytes> bytes of each file: the two coded byte strings of the logs
//!       encoder (GPT-2 alone and mixed) of the CPU path (Session, MatchModel, Mixer::mix,
//!       FreqTable, Encoder) and of the GPU path (run_coded_dual in prefill passes,
//!       Encoder::encode_interval) must be equal; then the GPU decodes both (run_coded_mixed with
//!       decode queries, mixed and plain blocks in one call, one token for each slot in each
//!       step) and must give the tokens again.
//!   mixed bench <model.hsm> <file> [decode batch sizes]
//!       tokens/s of run_coded (text), run_coded_mixed (logs) and run_weights (the read back):
//!       prefill (fold) and decode (unfold).
//!
//! The text is tokenized with the longest vocabulary match (as `examples/gpt2.rs`):
//! deterministic, not the canonical GPT-2 tokenization, which is enough for an equivalence test.
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
use hardshell::gpt2::{Model, Overlap, Session, SessionConfig};
use hardshell_cuda::{CoderQuery, CudaBatch, CudaDevice, CudaModel, MixInput, MixStats};
use taco_coder::{CodeInterval, Decoder, Encoder, FreqBits, FreqTable, Symbol};
use taco_match::{MatchModel, MatchParams, Mixer, MixerParams, Prediction};

const BITS: u32 = 32;

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

fn mt(t: TokenId) -> taco_match::TokenId {
    taco_match::TokenId::new(t.get())
}

/// The host step of the mixer from the statistics only: gives `q` and leaves the mixer in the
/// state of `Mixer::mix(w, Some(pred))` for the weights `w` of the statistics.
///
/// HACK: `taco_match::Mixer` has no public mix from statistics (MIXED-INTEGRATION.md asks the
/// owner for `Mixer::mix_stats`). The pending state of `mix` depends only on `pg`, the bucket of
/// the length and the agreement bit, so a short vector with the same three values gives the same
/// state: `v[p] = pg` and a total of 2^16, the rest after `p` in parts of at most `pg` (agree), or
/// one larger part after `p` (disagree), or, if `pg = 2^15`, an equal part before `p` (a tie
/// that a lower id wins). The `peek` check below proves the same `q`.
fn mix_from_stats(mixer: &mut Mixer, st: MixStats, pred: Prediction) -> u32 {
    let p = pred.token.get() as usize;
    let total = u128::from(st.total);
    let agree = st.top.get() as usize == p;
    let q = mixer.peek(st.predicted, total, pred.length, agree);
    // NOTE: rule 2, the expression of `Mixer::mix`.
    let pg =
        (u128::from(st.predicted) << 16).checked_div(total).unwrap_or(1).clamp(1, 65_535) as u64;
    let mut v = vec![0u64; p + 1];
    v[p] = pg;
    let rest = 65_536 - pg;
    if agree {
        let mut left = rest;
        while left > 0 {
            let x = left.min(pg);
            v.push(x);
            left -= x;
        }
    } else if rest > pg {
        v.push(rest);
    } else {
        assert!(rest == pg && p > 0, "statistics that no weights give");
        v[p - 1] = rest;
    }
    assert_eq!(mixer.peek(pg, 65_536, pred.length, agree), q, "the short vector gives another q");
    mixer.mix(&mut v, Some(pred)).unwrap();
    q
}

/// The two coded byte strings of `tokens` on the CPU path of the logs profile (ADR 0010
/// section 5, rule 7.1): GPT-2 alone and mixed, from the same HardShell weights. Also the number
/// of correct match predictions.
fn cpu_encode(cpu: &Model, tokens: &[TokenId], overlap: Overlap) -> (Vec<u8>, Vec<u8>, usize) {
    let bits = FreqBits::new(BITS).unwrap();
    let cfg = SessionConfig::new().with_chunk(32).with_overlap(overlap);
    let mut s = Session::with_config(cpu, cfg);
    let mut model = MatchModel::new(MatchParams::V1);
    let mut mixer = Mixer::new(MixerParams::V1);
    let mut plain = Encoder::new();
    let mut enc = Encoder::new();
    let mut hits = 0;
    let mut code = |w: &[u64], t: TokenId| {
        let table = FreqTable::from_weights(w, bits).unwrap();
        plain.encode(&table, Symbol::new(t.get())).unwrap();
        let mut w = w.to_vec();
        let pred = model.predict();
        hits += usize::from(pred.is_some_and(|p| p.token == mt(t)));
        mixer.mix(&mut w, pred).unwrap();
        let table = FreqTable::from_weights(&w, bits).unwrap();
        enc.encode(&table, Symbol::new(t.get())).unwrap();
        mixer.update(mt(t));
        model.observe(mt(t));
    };
    if let Some(&first) = tokens.first() {
        code(s.start_block().unwrap(), first);
        let mut i = 1;
        for chunk in tokens[..tokens.len() - 1].chunks(32) {
            let ws = s.step_many(chunk).unwrap();
            for w in ws.iter() {
                code(w, tokens[i]);
                i += 1;
            }
        }
        assert_eq!(i, tokens.len());
    }
    (plain.finish(), enc.finish(), hits)
}

/// One file of the GPU path: its tokens, predictions, mixer and the two encoders.
struct Lane {
    tokens: Vec<TokenId>,
    preds: Vec<Option<Prediction>>,
    mixer: Mixer,
    plain: Encoder,
    enc: Encoder,
}

/// The two coded byte strings of each token list on the GPU (`run_coded_dual`): one slot for
/// each list, `rows` rows of each slot in each call (prefill), the mixer of each slot in the
/// host step.
fn gpu_encode(
    gpu: &CudaModel<'_>,
    lists: &[Vec<TokenId>],
    overlap: Overlap,
    rows: usize,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let bits = FreqBits::new(BITS).unwrap();
    let mut lanes: Vec<Lane> = lists
        .iter()
        .map(|t| {
            // NOTE: the match model depends only on the tokens, so fold can run it first.
            let mut m = MatchModel::new(MatchParams::V1);
            let preds = t
                .iter()
                .map(|&x| {
                    let p = m.predict();
                    m.observe(mt(x));
                    p
                })
                .collect();
            Lane {
                tokens: t.clone(),
                preds,
                mixer: Mixer::new(MixerParams::V1),
                plain: Encoder::new(),
                enc: Encoder::new(),
            }
        })
        .collect();
    let mut batch = CudaBatch::new(gpu, lists.len(), rows * lists.len()).unwrap();
    batch.set_overlap(overlap).unwrap();
    let eot = gpu.eot_token();
    let longest = lists.iter().map(Vec::len).max().unwrap_or(0);
    let mut at = 0;
    while at < longest {
        // (lane, token index) of each input.
        let mut owner = Vec::new();
        let mut inputs = Vec::new();
        let mut tokens = Vec::new();
        let mut preds = Vec::new();
        for (s, l) in lanes.iter().enumerate() {
            for j in at..(at + rows).min(l.tokens.len()) {
                inputs.push((s, if j == 0 { eot } else { l.tokens[j - 1] }));
                tokens.push(l.tokens[j]);
                preds.push(l.preds[j].map(|p| TokenId::new(p.token.get())));
                owner.push((s, j));
            }
        }
        let ivs = batch
            .run_coded_dual(&inputs, &tokens, &preds, BITS, |i, st| {
                let (s, j) = owner[i];
                let l = &mut lanes[s];
                let q = mix_from_stats(&mut l.mixer, st, l.preds[j].unwrap());
                // NOTE: fold knows the token, so the mixer takes it at once. A row without a
                // prediction has no call; its `update` does nothing on the CPU (no pending state).
                l.mixer.update(mt(l.tokens[j]));
                q
            })
            .unwrap();
        for (&(s, j), (pi, mi)) in owner.iter().zip(&ivs) {
            let l = &mut lanes[s];
            assert_eq!(pi.symbol, l.tokens[j], "the coder op answered for another token");
            assert_eq!(mi.symbol, l.tokens[j], "the mixed op answered for another token");
            l.plain.encode_interval(CodeInterval::new(pi.start, pi.freq, bits).unwrap());
            l.enc.encode_interval(CodeInterval::new(mi.start, mi.freq, bits).unwrap());
        }
        at += rows;
    }
    lanes.into_iter().map(|l| (l.plain.finish(), l.enc.finish())).collect()
}

/// One block to decode: its bytes, its token count, and the kind (mixed or GPT-2 alone).
struct Coded {
    bytes: Vec<u8>,
    count: usize,
    mixed: bool,
}

/// Decodes each block on the GPU: one slot for each, one token for each slot in each step, the
/// mixed and the plain blocks in one call. Gives the tokens and the seconds.
fn gpu_decode(gpu: &CudaModel<'_>, coded: &[Coded], overlap: Overlap) -> (Vec<Vec<TokenId>>, f64) {
    let bits = FreqBits::new(BITS).unwrap();
    let n = coded.len();
    let mut batch = CudaBatch::new(gpu, n, n.max(2)).unwrap();
    batch.set_overlap(overlap).unwrap();
    let mut decs: Vec<Decoder<'_>> = coded.iter().map(|c| Decoder::new(&c.bytes)).collect();
    let mut models: Vec<MatchModel> = (0..n).map(|_| MatchModel::new(MatchParams::V1)).collect();
    let mut mixers: Vec<Mixer> = (0..n).map(|_| Mixer::new(MixerParams::V1)).collect();
    let mut out: Vec<Vec<TokenId>> = vec![Vec::new(); n];
    let eot = gpu.eot_token();
    let t0 = Instant::now();
    loop {
        let active: Vec<usize> = (0..n).filter(|&s| out[s].len() < coded[s].count).collect();
        if active.is_empty() {
            break;
        }
        let inputs: Vec<(usize, TokenId)> =
            active.iter().map(|&s| (s, out[s].last().copied().unwrap_or(eot))).collect();
        let queries: Vec<CoderQuery> =
            active.iter().map(|&s| CoderQuery::Decode(decs[s].target(bits))).collect();
        let preds: Vec<Option<Prediction>> = active
            .iter()
            .map(|&s| if coded[s].mixed { models[s].predict() } else { None })
            .collect();
        let tables: Vec<MixInput> = active
            .iter()
            .zip(&preds)
            .map(|(&s, p)| {
                if coded[s].mixed {
                    MixInput::Mixed(p.map(|p| TokenId::new(p.token.get())))
                } else {
                    MixInput::Plain
                }
            })
            .collect();
        let ivs = batch
            .run_coded_mixed(&inputs, &queries, &tables, BITS, |i, st| {
                mix_from_stats(&mut mixers[active[i]], st, preds[i].unwrap())
            })
            .unwrap();
        for (k, &s) in active.iter().enumerate() {
            let iv = ivs[k];
            decs[s].decode_interval(CodeInterval::new(iv.start, iv.freq, bits).unwrap()).unwrap();
            if coded[s].mixed {
                mixers[s].update(mt(iv.symbol));
                models[s].observe(mt(iv.symbol));
            }
            out[s].push(iv.symbol);
        }
    }
    (out, t0.elapsed().as_secs_f64())
}

fn equiv(bytes: &[u8], len: usize, overlap: Overlap, files: &[String]) {
    let cpu = Model::load(bytes).unwrap();
    println!("model {}", cpu.hash());
    let dev = CudaDevice::open(0).unwrap();
    println!(
        "device: {} {:?}; overlap K = {}",
        dev.name(),
        dev.compute_capability(),
        overlap.get()
    );
    let gpu = CudaModel::load(&dev, bytes).unwrap();
    let lists: Vec<Vec<TokenId>> = files
        .iter()
        .map(|f| {
            let b = std::fs::read(f).unwrap();
            tokenize(&cpu, &b[..len.min(b.len())])
        })
        .collect();
    let t0 = Instant::now();
    let cpu_res: Vec<(Vec<u8>, Vec<u8>, usize)> = std::thread::scope(|sc| {
        let hs: Vec<_> = lists.iter().map(|t| sc.spawn(|| cpu_encode(&cpu, t, overlap))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let cpu_s = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let gpu_res = gpu_encode(&gpu, &lists, overlap, 512);
    let gpu_s = t0.elapsed().as_secs_f64();
    // Each file twice: the mixed block and the block of GPT-2 alone, in one batch.
    let coded: Vec<Coded> = gpu_res
        .iter()
        .zip(&lists)
        .flat_map(|((plain, mixed), t)| {
            [
                Coded { bytes: mixed.clone(), count: t.len(), mixed: true },
                Coded { bytes: plain.clone(), count: t.len(), mixed: false },
            ]
        })
        .collect();
    let (decoded, dec_s) = gpu_decode(&gpu, &coded, overlap);
    let mut ok = true;
    let total: usize = lists.iter().map(Vec::len).sum();
    for (i, f) in files.iter().enumerate() {
        let name = f.rsplit('/').next().unwrap_or(f);
        let (cp, cm, hits) = &cpu_res[i];
        let (gp, gm) = &gpu_res[i];
        let same = cp == gp && cm == gm;
        let back = decoded[2 * i] == lists[i] && decoded[2 * i + 1] == lists[i];
        ok &= same && back;
        println!(
            "{name:<20} {:>6} tokens {:>6} hits  text {:>6} B  mixed cpu {:>6} B gpu {:>6} B  {}  decode {}",
            lists[i].len(),
            hits,
            gp.len(),
            cm.len(),
            gm.len(),
            if same { "EQUAL" } else { "DIFFERENT" },
            if back { "RESTORED" } else { "WRONG" }
        );
    }
    println!(
        "{total} tokens: cpu encode {cpu_s:.1} s ({} threads), gpu encode (dual) {gpu_s:.1} s, \
         gpu decode of {} blocks {dec_s:.1} s",
        files.len(),
        coded.len()
    );
    println!("{}", if ok { "ALL EQUAL AND RESTORED" } else { "SOME DIFFERENT" });
    assert!(ok);
}

/// The rate of `tokens` tokens in `s` seconds.
fn rate(tokens: usize, s: f64) -> String {
    format!("{:>8.0} tokens/s", tokens as f64 / s)
}

fn bench(bytes: &[u8], file: &str, sizes: &[usize]) {
    let cpu = Model::load(bytes).unwrap();
    let dev = CudaDevice::open(0).unwrap();
    println!("device: {} {:?}", dev.name(), dev.compute_capability());
    let gpu = CudaModel::load(&dev, bytes).unwrap();
    let toks = tokenize(&cpu, &std::fs::read(file).unwrap());
    let tok = |s: usize, i: usize| toks[(s * 997 + i) % toks.len()];
    // NOTE: a host mixer step of the real cost class: `peek` is the probability part of
    // `Mixer::mix` (O(1)); the integration uses `Mixer::mix_stats`, of the same cost (est.).
    let mixer = Mixer::new(MixerParams::V1);
    let pred_of = |s: usize, i: usize| MixInput::Mixed(Some(tok(s, i + 1)));
    let len = taco_match::MatchLen::new(20).unwrap();
    // NOTE: the GPU clock rises under load (45 W limit); 2 s of work first, not timed.
    let mut warm = CudaBatch::new(&gpu, 1, 512).unwrap();
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < 2.0 {
        let inputs: Vec<(usize, TokenId)> = (0..512).map(|i| (0, tok(0, i))).collect();
        warm.reset(0);
        warm.run_without_readback(&inputs).unwrap();
    }
    drop(warm);
    // Prefill (fold): `slots` slots, 512 rows each, passes of 512 rows; 3 rounds.
    for &(slots, rows) in &[(1usize, 512usize), (8, 512)] {
        let mut b = CudaBatch::new(&gpu, slots, 512).unwrap();
        let mk = |round: usize| {
            let mut inputs = Vec::new();
            let mut queries = Vec::new();
            let mut preds = Vec::new();
            for s in 0..slots {
                for i in 0..rows {
                    let j = round * rows + i;
                    inputs.push((s, tok(s, j)));
                    queries.push(CoderQuery::Encode(tok(s, j + 1)));
                    preds.push(pred_of(s, j));
                }
            }
            (inputs, queries, preds)
        };
        let n = slots * rows;
        let mut line = format!("prefill {slots} x {rows}:");
        for what in ["coded", "mixed", "dual", "weights"] {
            for s in 0..slots {
                b.reset(s);
            }
            let mut t0 = Instant::now();
            for round in 0..=3 {
                if round == 1 {
                    // NOTE: round 0 records the graphs of the pass shapes; it is not timed.
                    t0 = Instant::now();
                }
                let (inputs, queries, preds) = mk(round);
                match what {
                    "coded" => drop(b.run_coded(&inputs, &queries, BITS).unwrap()),
                    "mixed" => drop(
                        b.run_coded_mixed(&inputs, &queries, &preds, BITS, |_, st| {
                            mixer.peek(st.predicted, u128::from(st.total), len, false)
                        })
                        .unwrap(),
                    ),
                    "dual" => {
                        let tokens: Vec<TokenId> = queries
                            .iter()
                            .map(|q| match *q {
                                CoderQuery::Encode(t) => t,
                                _ => unreachable!(),
                            })
                            .collect();
                        let p: Vec<Option<TokenId>> = tokens.iter().map(|&t| Some(t)).collect();
                        drop(
                            b.run_coded_dual(&inputs, &tokens, &p, BITS, |_, st| {
                                mixer.peek(st.predicted, u128::from(st.total), len, false)
                            })
                            .unwrap(),
                        );
                    }
                    _ => drop(b.run_weights(&inputs).unwrap()),
                }
            }
            line += &format!("  {what} {}", rate(3 * n, t0.elapsed().as_secs_f64()));
        }
        println!("{line}");
    }
    // Decode (unfold): `b` slots, one row each per step, positions 0 to 127.
    for &slots in sizes {
        let mut batch = match CudaBatch::new(&gpu, slots, slots.max(2)) {
            Ok(x) => x,
            Err(e) => {
                println!("decode {slots}: not possible: {e}");
                continue;
            }
        };
        let steps = 128;
        let mut line = format!("decode batch {slots:>3}:");
        for what in ["coded", "mixed", "weights"] {
            for s in 0..slots {
                batch.reset(s);
            }
            let steps = if what == "weights" { 16 } else { steps };
            let mut t0 = Instant::now();
            for i in 0..=steps {
                if i == 1 {
                    // NOTE: step 0 records the graph; it is not timed.
                    t0 = Instant::now();
                }
                let inputs: Vec<(usize, TokenId)> = (0..slots).map(|s| (s, tok(s, i))).collect();
                let queries: Vec<CoderQuery> = (0..slots)
                    .map(|s| CoderQuery::Decode(u64::from(tok(s, i).get()) << 16))
                    .collect();
                match what {
                    "coded" => drop(batch.run_coded(&inputs, &queries, BITS).unwrap()),
                    "mixed" => {
                        let preds: Vec<MixInput> = (0..slots).map(|s| pred_of(s, i)).collect();
                        drop(
                            batch
                                .run_coded_mixed(&inputs, &queries, &preds, BITS, |_, st| {
                                    mixer.peek(st.predicted, u128::from(st.total), len, false)
                                })
                                .unwrap(),
                        );
                    }
                    _ => drop(batch.run_weights(&inputs).unwrap()),
                }
            }
            line += &format!("  {what} {}", rate(slots * steps, t0.elapsed().as_secs_f64()));
        }
        println!("{line}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("");
    let bytes = std::fs::read(args.get(2).expect("model path")).unwrap();
    match mode {
        "equiv" => {
            let len: usize = args.get(3).expect("bytes").parse().unwrap();
            let k: u16 = args.get(4).expect("overlap").parse().unwrap();
            equiv(&bytes, len, Overlap::new(k), &args[5..]);
        }
        "bench" => {
            let file = args.get(3).expect("file");
            let sizes: Vec<usize> = args[4..].iter().map(|s| s.parse().unwrap()).collect();
            let sizes = if sizes.is_empty() { vec![8, 32, 64, 128, 224] } else { sizes };
            bench(&bytes, file, &sizes);
        }
        _ => panic!("usage: mixed equiv|bench ..."),
    }
}
