//! Model-file version 2 ("wide", ADR 0004 addendum B) on the CUDA GPU against the CPU
//! reference: `i16` GELU tables, `ln_f_bits = 16` and `ln_beta_frac`. The golden vector of
//! version 2, the reference session at every step, each rule alone and in pairs, batches with
//! prefill, the coder op, the context overlap, other shapes, and the real wide models (ignored by
//! default). A one-bit difference fails.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::print_stderr,
    clippy::panic
)]
#![allow(clippy::print_stdout)]

#[path = "../../hardshell/tests/common/mod.rs"]
mod common;
mod support;

use common::builder::{build, build_wide, tiny_dims};
use common::reference::Dims;
use hardshell::TokenId;
use hardshell::gpt2::{Model, Overlap, ReferenceSession, Session, SessionConfig};
use hardshell::hsm::{HsmFile, HsmWriter, TensorData};
use hardshell::tensor::Tensor;
use hardshell_cuda::{CoderInterval, CoderQuery, CudaBatch, CudaModel, CudaSession};
use sha2::{Digest, Sha256};
use support::device;
use taco_coder::{FreqBits, FreqTable};

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<TokenId> {
    let mut r = common::Rng::new(seed);
    (0..n).map(|_| TokenId::new(r.below(vocab as u64) as u32)).collect()
}

fn hex(h: &[u8]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

/// The tiny wide model of `crates/hardshell/tests/wide.rs` (`golden_conformance_hash_of_version_2`).
const GOLDEN_V2_MODEL: &str = "201f9fd6df4aa0467657380c3890bdb25330510d7253d0fcf5f6cd9211e6af04";
/// The SHA-256 of all its weights for the 150 tokens of the vector, from the reference session.
const GOLDEN_V2_OUTPUT: &str = "64108429f3ce999899c1df4da4d8c364138b1315143d08d2ab01ab7b2dde2eb8";

/// The rules of version 2 that a test model uses.
#[derive(Clone, Copy, Debug)]
struct Rules {
    gelu16: bool,
    lnf16: bool,
    frac8: bool,
}

/// All seven combinations of the three rules of version 2.
fn all_rules() -> Vec<Rules> {
    (1..8u8).map(|i| Rules { gelu16: i & 1 != 0, lnf16: i & 2 != 0, frac8: i & 4 != 0 }).collect()
}

fn i32_data(t: &TensorData) -> Vec<i32> {
    match t {
        TensorData::I32(t) => t.data().to_vec(),
        _ => panic!("not an i32 tensor"),
    }
}

fn i32_tensor(v: Vec<i32>) -> TensorData {
    TensorData::I32(Tensor::new(&[v.len()], v).unwrap())
}

/// A model of `dims` with the given rules of version 2: the tensors of the wide model of
/// `build_wide`, with the parts of the version-1 model of `build` (the same float weights) for
/// the rules that are off. A LayerNorm beta takes the units of the rule `ln_beta_frac`.
fn variant(dims: Dims, seed: u64, rules: Rules) -> Vec<u8> {
    let wide = build_wide(dims, seed, 64).bytes;
    if rules.gelu16 && rules.lnf16 && rules.frac8 {
        return wide;
    }
    let narrow = build(dims, seed, 64).bytes;
    let (wf, nf) = (HsmFile::parse(&wide).unwrap(), HsmFile::parse(&narrow).unwrap());
    let mut w = HsmWriter::new(wf.config().clone());
    let voc = wf.vocabulary();
    w.set_vocabulary(
        (0..voc.len() as u32).map(|i| voc.token(TokenId::new(i)).unwrap().to_vec()).collect(),
    );
    if rules.lnf16 {
        w.set_ln_f_bits(16);
    }
    if rules.frac8 {
        w.set_ln_beta_frac(8);
    }
    for name in wf.tensor_names() {
        let is_mlp = name.contains(".mlp.gelu") || name.contains(".mlp.c_proj.");
        let is_lnf = name == "ln_f.g" || name == "ln_f.rescale" || name == "wte.rescale_logits";
        let from_narrow = (is_mlp && !rules.gelu16) || (is_lnf && !rules.lnf16);
        let src = if from_narrow { &nf } else { &wf };
        let t = src.tensor(name).unwrap();
        let mut data = t.data().clone();
        let is_beta = name.ends_with("ln_1.b") || name.ends_with("ln_2.b") || name == "ln_f.b";
        if is_beta {
            // The wide model has beta in units of 2^-8 output units of the wide scales. ln_1
            // and ln_2 have the same output scale in both models; ln_f has the scale of its
            // output type.
            let v = if name == "ln_f.b" && !rules.lnf16 {
                let b = i32_data(nf.tensor(name).unwrap().data());
                if rules.frac8 { b.iter().map(|x| x * 256).collect() } else { b }
            } else {
                let b = i32_data(t.data());
                if rules.frac8 { b } else { b.iter().map(|x| x.div_euclid(256)).collect() }
            };
            data = i32_tensor(v);
        }
        w.add_tensor(name.clone(), data, t.shift());
    }
    w.finish().unwrap()
}

/// The reference session for `toks` after the start of a block, with the overlap `k`.
fn reference_weights(cpu: &Model, toks: &[TokenId], k: u16) -> Vec<Vec<u64>> {
    let mut r = ReferenceSession::with_overlap(cpu, Overlap::new(k));
    let mut want = vec![r.start_block().unwrap().to_vec()];
    for &t in toks {
        want.push(r.step(t).unwrap().to_vec());
    }
    want
}

/// The CUDA paths for one block of `toks` against `want` (the start row, then a row for each
/// token): a session (decode), and a batch of one slot in passes of `rows` rows (prefill), on
/// the fused path and on the op-level path.
fn cuda_paths_equal(gpu: &CudaModel<'_>, toks: &[TokenId], k: u16, rows: usize, want: &[Vec<u64>]) {
    let o = Overlap::new(k);
    let mut s = CudaSession::with_overlap(gpu, o).unwrap();
    assert_eq!(s.start_block().unwrap(), &want[0][..], "decode start, k {k}");
    for (i, &t) in toks.iter().enumerate() {
        assert_eq!(s.step(t).unwrap(), &want[i + 1][..], "decode step {i}, k {k}");
    }
    let v = gpu.vocab_size();
    let mut inputs = vec![(0, gpu.eot_token())];
    inputs.extend(toks.iter().map(|&t| (0, t)));
    for op_level in [false, true] {
        let mut batch = CudaBatch::new(gpu, 1, rows).unwrap();
        batch.set_op_level(op_level);
        batch.set_overlap(o).unwrap();
        let got = batch.run_weights(&inputs).unwrap();
        for (i, w) in want.iter().enumerate() {
            assert_eq!(&got[i * v..(i + 1) * v], &w[..], "prefill row {i}, k {k}, op {op_level}");
        }
    }
}

#[test]
fn golden_conformance_hash_of_version_2_on_cuda() {
    let Some(dev) = device("golden_conformance_hash_of_version_2_on_cuda") else { return };
    let b = build_wide(tiny_dims(), 0x7ac0_0001, 64);
    let m = CudaModel::load(&dev, &b.bytes).unwrap();
    assert_eq!(format!("{}", m.hash()), GOLDEN_V2_MODEL);
    assert!(m.is_wide());
    let toks = tokens(0x7ac0_0002, 150, 300);
    let mut s = CudaSession::new(&m).unwrap();
    let mut h = Sha256::new();
    for x in s.start_block().unwrap() {
        h.update(x.to_le_bytes());
    }
    for &t in &toks {
        for x in s.step(t).unwrap() {
            h.update(x.to_le_bytes());
        }
    }
    assert_eq!(hex(&h.finalize()), GOLDEN_V2_OUTPUT, "decode");
    // The same vector in one call: passes of 40, 7 and 512 rows, fused and op-level.
    let mut inputs = vec![(0, m.eot_token())];
    inputs.extend(toks.iter().map(|&t| (0, t)));
    for rows in [40, 7, 512] {
        for op_level in [false, true] {
            let mut batch = CudaBatch::new(&m, 1, rows).unwrap();
            batch.set_op_level(op_level);
            let mut h = Sha256::new();
            for x in batch.run_weights(&inputs).unwrap() {
                h.update(x.to_le_bytes());
            }
            assert_eq!(hex(&h.finalize()), GOLDEN_V2_OUTPUT, "prefill {rows} op {op_level}");
        }
    }
}

#[test]
fn wide_model_equals_the_reference_session_with_and_without_overlap() {
    // n_ctx 64 and 300 tokens: four restarts with K = 0, eight with K = 32, many with K = 62.
    let Some(dev) = device("wide_model_equals_the_reference_session") else { return };
    let b = build_wide(tiny_dims(), 0x7ac0_0005, 64);
    let cpu = Model::load(&b.bytes).unwrap();
    let gpu = CudaModel::load(&dev, &b.bytes).unwrap();
    let toks = tokens(0x7ac0_0006, 300, 300);
    for k in [0u16, 32, 62] {
        let want = reference_weights(&cpu, &toks, k);
        cuda_paths_equal(&gpu, &toks, k, 40, &want);
    }
}

#[test]
fn each_rule_of_version_2_alone_and_in_pairs_equals_the_reference() {
    let Some(dev) = device("each_rule_of_version_2_alone_and_in_pairs") else { return };
    for rules in all_rules() {
        let bytes = variant(tiny_dims(), 0x7ac0_0007, rules);
        let cpu = Model::load(&bytes).unwrap();
        assert!(cpu.is_wide(), "{rules:?}");
        let gpu = CudaModel::load(&dev, &bytes).unwrap();
        let toks = tokens(0x7ac0_0008, 140, 300);
        for (k, rows) in [(0u16, 37usize), (20, 8)] {
            let want = reference_weights(&cpu, &toks, k);
            cuda_paths_equal(&gpu, &toks, k, rows, &want);
        }
        eprintln!("rules {rules:?}: equal");
    }
}

/// CPU sessions and one GPU batch with the same token streams; each call gives every slot 0 to
/// 3 tokens (decode and prefill together), some slots start a new block.
fn batch_against_sessions(bytes: &[u8], slots: usize, calls: usize, seed: u64, max_rows: usize) {
    let Some(dev) = device("batch_against_sessions") else { return };
    let cpu = Model::load(bytes).unwrap();
    let gpu = CudaModel::load(&dev, bytes).unwrap();
    for op_level in [false, true] {
        let mut sessions: Vec<Session<'_>> = (0..slots).map(|_| Session::new(&cpu)).collect();
        let mut batch = CudaBatch::new(&gpu, slots, max_rows).unwrap();
        batch.set_op_level(op_level);
        let mut r = common::Rng::new(seed ^ 0x5eed);
        let v = gpu.vocab_size();
        for call in 0..calls {
            let (mut inputs, mut wants) = (Vec::new(), Vec::new());
            for _ in 0..3 {
                for (slot, s) in sessions.iter_mut().enumerate() {
                    if r.below(3) == 0 {
                        continue;
                    }
                    let token = if r.below(97) == 0 && !inputs.iter().any(|&(x, _)| x == slot) {
                        batch.reset(slot);
                        s.reset();
                        gpu.eot_token()
                    } else {
                        TokenId::new(r.below(v as u64) as u32)
                    };
                    wants.push(s.step(token).unwrap().to_vec());
                    inputs.push((slot, token));
                }
            }
            let got = batch.run_weights(&inputs).unwrap();
            for (i, want) in wants.iter().enumerate() {
                assert_eq!(
                    &got[i * v..(i + 1) * v],
                    &want[..],
                    "call {call} input {i} op {op_level}"
                );
            }
            for (slot, s) in sessions.iter().enumerate() {
                assert_eq!(batch.position(slot), s.position(), "call {call} slot {slot}");
            }
        }
    }
}

#[test]
fn wide_batches_with_prefill_equal_independent_sessions() {
    let b = build_wide(tiny_dims(), 0x7ac0_0009, 64).bytes;
    // Passes of up to 9 rows (dp4a products of 2n <= 18 plane rows), of at most 16 rows, and of
    // up to 200 rows (tensor-core products with 128-row tiles, 400 plane rows).
    batch_against_sessions(&b, 3, 40, 21, 512);
    batch_against_sessions(&b, 12, 30, 22, 16);
    batch_against_sessions(&b, 64, 12, 23, 200);
}

fn cpu_interval(w: &[u64], bits: u32, q: CoderQuery) -> CoderInterval {
    let t = FreqTable::from_weights(w, FreqBits::new(bits).unwrap()).unwrap();
    let freqs: Vec<u64> = t.freqs().map(|f| f.get()).collect();
    let mut starts = Vec::with_capacity(freqs.len());
    let mut acc = 0;
    for f in &freqs {
        starts.push(acc);
        acc += f;
    }
    let i = match q {
        CoderQuery::Encode(t) => t.get() as usize,
        CoderQuery::Decode(x) => starts.partition_point(|&s| s <= x) - 1,
        _ => unreachable!(),
    };
    CoderInterval {
        symbol: TokenId::new(i as u32),
        start: starts[i],
        freq: freqs[i],
        is_last: i + 1 == freqs.len(),
    }
}

/// `run_coded` of random calls on `slots` slots against the CPU table, with the overlap `k`.
fn coder_against_cpu(bytes: &[u8], slots: usize, calls: usize, k: u16, op_level: bool) {
    let Some(dev) = device("coder_against_cpu") else { return };
    let cpu = Model::load(bytes).unwrap();
    let gpu = CudaModel::load(&dev, bytes).unwrap();
    let cfg = SessionConfig::new().with_overlap(Overlap::new(k));
    let mut sessions: Vec<Session<'_>> =
        (0..slots).map(|_| Session::with_config(&cpu, cfg)).collect();
    let mut batch = CudaBatch::new(&gpu, slots, 2 * slots).unwrap();
    batch.set_op_level(op_level);
    batch.set_overlap(Overlap::new(k)).unwrap();
    let mut r = common::Rng::new(0xc0de ^ slots as u64 ^ u64::from(k));
    let vocab = gpu.vocab_size() as u64;
    for call in 0..calls {
        let bits = [32u32, 12, 9][call % 3];
        let total = 1u64 << bits;
        let (mut inputs, mut queries, mut wants) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..2 {
            for (slot, s) in sessions.iter_mut().enumerate() {
                if r.below(2) == 0 && !inputs.is_empty() {
                    continue;
                }
                let token = if call == 0 && !inputs.iter().any(|&(x, _)| x == slot) {
                    cpu.eot_token()
                } else {
                    TokenId::new(r.below(vocab) as u32)
                };
                let w = s.step(token).unwrap();
                let q = match r.below(4) {
                    0 => CoderQuery::Decode(0),
                    1 => CoderQuery::Decode(total - 1),
                    2 => CoderQuery::Decode(r.below(total)),
                    _ => CoderQuery::Encode(TokenId::new(r.below(vocab) as u32)),
                };
                wants.push(cpu_interval(w, bits, q));
                inputs.push((slot, token));
                queries.push(q);
            }
        }
        let got = batch.run_coded(&inputs, &queries, bits).unwrap();
        assert_eq!(got, wants, "call {call} bits {bits} k {k} op {op_level}");
    }
}

#[test]
fn wide_coder_intervals_equal_the_cpu_table() {
    let b = build_wide(tiny_dims(), 31, 64).bytes;
    // 4 slots: passes of at most 8 rows (dp4a, i64 logits); 40 slots: passes of up to 80 rows
    // (tensor cores; a wide model has no compact logits).
    coder_against_cpu(&b, 4, 60, 0, false);
    coder_against_cpu(&b, 40, 20, 0, false);
    coder_against_cpu(&b, 4, 30, 0, true);
    coder_against_cpu(&b, 5, 60, 20, false);
}

#[test]
fn wide_models_of_other_shapes_equal_the_reference() {
    // head_dim 32 and n_ctx 30; head_dim 12; d_model 1,280 (the norm reads the row from memory,
    // norm_row); a vocabulary of 1,000 and d_ff 512.
    let Some(dev) = device("wide_models_of_other_shapes") else { return };
    let shapes = [
        (Dims { n_layer: 1, n_head: 2, d_model: 64, d_ff: 128, vocab: 300, n_ctx: 30 }, 75, 16),
        (Dims { n_layer: 3, n_head: 4, d_model: 48, d_ff: 96, vocab: 257, n_ctx: 21 }, 50, 7),
        (Dims { n_layer: 1, n_head: 5, d_model: 1280, d_ff: 256, vocab: 300, n_ctx: 24 }, 40, 16),
        (Dims { n_layer: 2, n_head: 2, d_model: 128, d_ff: 512, vocab: 1000, n_ctx: 40 }, 90, 64),
    ];
    for (i, (dims, steps, rows)) in shapes.into_iter().enumerate() {
        for rules in [
            Rules { gelu16: true, lnf16: true, frac8: true },
            Rules { gelu16: true, lnf16: true, frac8: false },
        ] {
            let bytes = variant(dims, 70 + i as u64, rules);
            let cpu = Model::load(&bytes).unwrap();
            let gpu = CudaModel::load(&dev, &bytes).unwrap();
            let toks = tokens(0xa77e ^ i as u64, steps, dims.vocab);
            let want = reference_weights(&cpu, &toks, 0);
            cuda_paths_equal(&gpu, &toks, 0, rows, &want);
        }
    }
}

/// The token ids of `tools/data/out/eval_windows.txt` (the GPT-2 vocabulary).
fn eval_tokens(n: usize) -> Vec<TokenId> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/data/out/eval_windows.txt");
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| {
            let ids = l.split_once('\t').unwrap().1.to_owned();
            ids.split_whitespace().map(|t| TokenId::new(t.parse().unwrap())).collect::<Vec<_>>()
        })
        .take(n)
        .collect()
}

/// The real wide models (`gpt2_wide.hsm` 9aa394b1…, `distil_logs_wide.hsm` d12e5f34…; weights
/// are not in the repository): the paths in `TACO_WIDE_MODELS`, separated by commas. For each
/// model and for K = 0 and K = 512: 2,100 tokens (two restarts) on the CPU fast session; CUDA
/// decode, prefill (fused and op-level) and the coder op against it; the reference session for
/// the first 1,100 tokens (one restart).
#[test]
#[ignore = "needs TACO_WIDE_MODELS (paths of wide .hsm files) and a CUDA device"]
fn real_wide_models_on_cuda_equal_the_cpu() {
    let Some(dev) = device("real_wide_models_on_cuda_equal_the_cpu") else { return };
    let list = std::env::var("TACO_WIDE_MODELS").expect("TACO_WIDE_MODELS");
    let toks = eval_tokens(2100);
    for path in list.split(',') {
        let bytes = std::fs::read(path).unwrap();
        let cpu = Model::load(&bytes).unwrap();
        let gpu = CudaModel::load(&dev, &bytes).unwrap();
        println!("{path}: model {} wide {}", cpu.hash(), cpu.is_wide());
        assert!(cpu.is_wide());
        for k in [0u16, 512] {
            let o = Overlap::new(k);
            let cfg = SessionConfig::new().with_chunk(32).with_overlap(o);
            let mut s = Session::with_config(&cpu, cfg);
            let mut want = vec![s.start_block().unwrap().to_vec()];
            for part in toks.chunks(32) {
                want.extend(s.step_many(part).unwrap().iter().map(<[u64]>::to_vec));
            }
            let mut h = Sha256::new();
            want.iter().flatten().for_each(|x| h.update(x.to_le_bytes()));
            println!("  k {k}: {} tokens, CPU digest {}", toks.len(), hex(&h.finalize()));
            cuda_paths_equal(&gpu, &toks, k, 64, &want);
            // Prefill in passes of 512 rows (tensor-core products of 1,024 plane rows).
            let v = gpu.vocab_size();
            let mut batch = CudaBatch::new(&gpu, 1, 512).unwrap();
            batch.set_overlap(o).unwrap();
            let mut inputs = vec![(0, gpu.eot_token())];
            inputs.extend(toks.iter().map(|&t| (0, t)));
            let got = batch.run_weights(&inputs).unwrap();
            for (i, w) in want.iter().enumerate() {
                assert_eq!(&got[i * v..(i + 1) * v], &w[..], "prefill 512, row {i}, k {k}");
            }
            // The coder op in prefill: the interval of the next token at each row.
            batch.reset(0);
            let queries: Vec<CoderQuery> = (0..inputs.len())
                .map(|i| CoderQuery::Encode(toks.get(i).copied().unwrap_or(TokenId::new(0))))
                .collect();
            let got = batch.run_coded(&inputs, &queries, 32).unwrap();
            let wants: Vec<CoderInterval> =
                want.iter().zip(&queries).map(|(w, &q)| cpu_interval(w, 32, q)).collect();
            assert_eq!(got, wants, "coder intervals, k {k}");
            println!(
                "  k {k}: CUDA decode, prefill (64 and 512 rows) and {} intervals equal",
                wants.len()
            );
            // The oracle (slow): the first 1,100 tokens, one restart.
            let mut r = ReferenceSession::with_overlap(&cpu, o);
            assert_eq!(r.start_block().unwrap(), &want[0][..]);
            for (i, &t) in toks.iter().take(1100).enumerate() {
                assert_eq!(r.step(t).unwrap(), &want[i + 1][..], "oracle step {i}, k {k}");
            }
            println!("  k {k}: the reference session equals the CPU session (1,100 tokens)");
        }
    }
}
