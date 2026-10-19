//! Tests for hsm format version 2 (ADR 0004 addendum B): `i16` GELU outputs and `i16` final
//! LayerNorm outputs. The file rules, the kernels, the reference session against the float
//! model, and the conformance vector of version 2.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::print_stdout)]

mod common;

use common::builder::{build, build_wide, tiny_dims};
use common::{Rng, kl_bits, pow2, softmax};
use hardshell::gpt2::{Batch, BatchInput, Model, ReferenceSession, Session, SessionConfig};
use hardshell::hsm::{
    Constants, HsmFile, HsmWriter, ModelConfig, TensorData, VERSION_1, VERSION_2,
};
use hardshell::kernels::{Kernels, matvec_i8_i16};
use hardshell::layernorm::{LayerNorm, LnConstants};
use hardshell::tensor::{Matrix, Tensor};
use hardshell::{Error, TokenId};
use proptest::prelude::*;
use sha2::{Digest, Sha256};

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<TokenId> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| TokenId::new(r.below(vocab as u64) as u32)).collect()
}

/// All weights of a block with the reference session: the start, then one step per token.
fn run_reference(m: &Model, toks: &[TokenId]) -> Vec<Vec<u64>> {
    let mut s = ReferenceSession::new(m);
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for &t in toks {
        out.push(s.step(t).unwrap().to_vec());
    }
    out
}

fn small_config() -> ModelConfig {
    ModelConfig {
        architecture: "gpt2".into(),
        n_layer: 1,
        n_head: 1,
        d_model: 4,
        d_ff: 8,
        vocab_size: 2,
        n_ctx: 4,
        eot_token: 1,
        constants: Constants { ln_k: 48, ln_j: 8, ln_eps: 1, softmax_p: 30 },
    }
}

fn small_writer() -> HsmWriter {
    let mut w = HsmWriter::new(small_config());
    w.set_vocabulary(vec![b"a".to_vec(), b"<|endoftext|>".to_vec()]);
    w
}

#[test]
fn the_writer_selects_the_version_from_the_parts() {
    let v1 = small_writer().finish().unwrap();
    let f = HsmFile::parse(&v1).unwrap();
    assert_eq!((f.version(), f.ln_f_bits()), (VERSION_1, 8));

    let mut w = small_writer();
    w.set_ln_f_bits(16);
    let f = HsmFile::parse(&w.finish().unwrap()).unwrap();
    assert_eq!((f.version(), f.ln_f_bits()), (VERSION_2, 16));

    let mut w = small_writer();
    w.add_tensor("h.0.mlp.gelu", TensorData::I16(Tensor::new(&[256], vec![0; 256]).unwrap()), None);
    let f = HsmFile::parse(&w.finish().unwrap()).unwrap();
    assert_eq!((f.version(), f.ln_f_bits()), (VERSION_2, 8));

    let mut w = small_writer();
    w.set_ln_beta_frac(8);
    let f = HsmFile::parse(&w.finish().unwrap()).unwrap();
    assert_eq!((f.version(), f.ln_f_bits(), f.ln_beta_frac()), (VERSION_2, 8, 8));

    // An i8 GELU table and an i16 tensor with another name keep version 1.
    let mut w = small_writer();
    w.add_tensor("h.0.mlp.gelu", TensorData::I8(Tensor::new(&[256], vec![0; 256]).unwrap()), None);
    w.add_tensor("h.0.ln_1.g", TensorData::I16(Tensor::new(&[4], vec![0; 4]).unwrap()), None);
    assert_eq!(HsmFile::parse(&w.finish().unwrap()).unwrap().version(), VERSION_1);
}

#[test]
fn the_reader_refuses_a_version_that_the_parts_do_not_require() {
    let version_err = |bytes: &[u8]| match HsmFile::parse(bytes) {
        Err(Error::InvalidHeader { field, .. }) => assert_eq!(field, "version"),
        other => panic!("expected a version error, got {other:?}"),
    };
    let mut v1 = small_writer().finish().unwrap();
    v1[8] = 2;
    version_err(&v1);
    let mut w = small_writer();
    w.set_ln_f_bits(16);
    let mut v2 = w.finish().unwrap();
    v2[8] = 1;
    version_err(&v2);
    v2[8] = 3;
    assert_eq!(HsmFile::parse(&v2).unwrap_err(), Error::UnsupportedVersion { version: 3 });
}

#[test]
fn the_reader_refuses_ln_f_bits_other_than_16() {
    for bits in [0u8, 7, 12, 32] {
        let mut w = small_writer();
        w.set_ln_f_bits(bits);
        match w.finish() {
            Err(Error::InvalidHeader { field, .. }) => {
                assert_eq!(field, "model.constants.ln_f_bits");
            }
            other => panic!("ln_f_bits {bits}: expected an error, got {other:?}"),
        }
    }
    // 8 is the default: no header field, version 1.
    let mut w = small_writer();
    w.set_ln_f_bits(8);
    assert_eq!(w.finish().unwrap(), small_writer().finish().unwrap());
}

#[test]
fn the_reader_refuses_ln_beta_frac_above_16() {
    for frac in [17u8, 32, 255] {
        let mut w = small_writer();
        w.set_ln_beta_frac(frac);
        match w.finish() {
            Err(Error::InvalidHeader { field, .. }) => {
                assert_eq!(field, "model.constants.ln_beta_frac");
            }
            other => panic!("ln_beta_frac {frac}: expected an error, got {other:?}"),
        }
    }
    // 0 is the default: no header field, version 1.
    let mut w = small_writer();
    w.set_ln_beta_frac(0);
    assert_eq!(w.finish().unwrap(), small_writer().finish().unwrap());
}

#[test]
fn a_version_1_model_does_not_change() {
    let narrow = build(tiny_dims(), 0x7ac0_0001, 64);
    let f = HsmFile::parse(&narrow.bytes).unwrap();
    assert_eq!((f.version(), f.ln_f_bits()), (VERSION_1, 8));
    let m = Model::load(&narrow.bytes).unwrap();
    assert!(!m.is_wide());
    assert_eq!(m.ln_f_bits(), 8);
}

#[test]
fn a_wide_model_loads_as_version_2() {
    let b = build_wide(tiny_dims(), 0x7ac0_0001, 64);
    let f = HsmFile::parse(&b.bytes).unwrap();
    assert_eq!((f.version(), f.ln_f_bits()), (VERSION_2, 16));
    let m = Model::load(&b.bytes).unwrap();
    assert!(m.is_wide());
    assert_eq!(m.ln_f_bits(), 16);
    for w in run_reference(&m, &tokens(5, 20, 300)) {
        assert_eq!(w.len(), 300);
        assert!(w.iter().any(|&x| x >= 1));
        assert!(w.iter().map(|&x| u128::from(x)).sum::<u128>() <= 1 << 60);
    }
}

/// Mean KL divergence (bits per token) and top-1 agreement of the reference session against
/// the float model, over `blocks` blocks of 64 positions.
fn compare_with_float(
    bytes: &[u8],
    float: &common::reference::FloatModel,
    eot: usize,
    seed: u64,
) -> (f64, f64) {
    let m = Model::load(bytes).unwrap();
    let (mut kl_sum, mut count, mut agree) = (0.0, 0usize, 0usize);
    for blk in 0..4u64 {
        let toks = tokens(seed * 100 + blk, 63, 300);
        let ints = run_reference(&m, &toks);
        let mut ids = vec![eot];
        ids.extend(toks.iter().map(|t| t.get() as usize));
        for (w, l) in ints.iter().zip(&float.forward(&ids, None)) {
            let total: f64 = w.iter().map(|&x| x as f64).sum();
            let q: Vec<f64> = w.iter().map(|&x| (x as f64 / total).max(pow2(-32))).collect();
            let p = softmax(l);
            kl_sum += kl_bits(&p, &q);
            let am = |v: &[f64]| (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
            if am(&p) == am(&q) {
                agree += 1;
            }
            count += 1;
        }
    }
    (kl_sum / count as f64, agree as f64 / count as f64)
}

#[test]
fn a_wide_model_is_close_to_the_float_reference() {
    for seed in [11u64, 12, 13] {
        let narrow = build(tiny_dims(), seed, 64);
        let wide = build_wide(tiny_dims(), seed, 64);
        let (kl_n, top_n) = compare_with_float(&narrow.bytes, &narrow.float, narrow.eot, seed);
        let (kl_w, top_w) = compare_with_float(&wide.bytes, &wide.float, wide.eot, seed);
        println!(
            "tiny model seed {seed}: mean KL i8 {kl_n:.5}, i16 {kl_w:.5} bits/token; top-1 i8 \
             {:.2}%, i16 {:.2}% (measured, 4 x 64 positions)",
            100.0 * top_n,
            100.0 * top_w
        );
        assert!(kl_w < 0.05, "mean KL {kl_w}");
        // Measured (3 seeds): the version 2 rules are closer to the float model.
        assert!(kl_w < kl_n, "i16 KL {kl_w} is not below i8 KL {kl_n}");
        assert!(top_w > 0.9, "top-1 agreement {top_w}");
    }
}

#[test]
fn the_fast_paths_equal_the_reference_on_a_wide_model() {
    // The step path, the prefill path (several chunk sizes) and the batch path, with every
    // kernel selection, against the reference session. 2 * 64 + 7 tokens: two restarts.
    let b = build_wide(tiny_dims(), 3, 64);
    let m = Model::load(&b.bytes).unwrap();
    let toks = tokens(4, 2 * 64 + 7, 300);
    let want = run_reference(&m, &toks);
    for k in Kernels::available() {
        let mut s = Session::with_config(&m, SessionConfig::new().with_kernels(k));
        let mut got = vec![s.start_block().unwrap().to_vec()];
        for &t in &toks {
            got.push(s.step(t).unwrap().to_vec());
        }
        assert_eq!(got, want, "step path, kernels {}", k.name());
        for chunk in [1usize, 7, 64] {
            let cfg = SessionConfig::new().with_chunk(chunk).with_kernels(k);
            let mut s = Session::with_config(&m, cfg);
            let mut got = vec![s.start_block().unwrap().to_vec()];
            for part in toks.chunks(chunk) {
                got.extend(s.step_many(part).unwrap().iter().map(<[u64]>::to_vec));
            }
            assert_eq!(got, want, "prefill path, chunk {chunk}, kernels {}", k.name());
        }
        let lanes = 3;
        let mut bt = Batch::with_kernels(&m, lanes, k);
        let mut got: Vec<Vec<Vec<u64>>> = vec![Vec::new(); lanes];
        for i in 0..=toks.len() {
            let input = if i == 0 { BatchInput::Start } else { BatchInput::Token(toks[i - 1]) };
            bt.run(&vec![Some(input); lanes]).unwrap();
            for (j, g) in got.iter_mut().enumerate() {
                g.push(bt.weights(j).unwrap().to_vec());
            }
        }
        for g in got {
            assert_eq!(g, want, "batch path, kernels {}", k.name());
        }
    }
}

/// The conformance vector of format version 2 (plan §11): the tiny wide model of seed
/// `0x7ac0_0001`, 150 tokens (the context restart is in it), and the SHA-256 over all weights
/// as little-endian `u64`, in step order, from the reference session.
#[test]
fn golden_conformance_hash_of_version_2() {
    let b = build_wide(tiny_dims(), 0x7ac0_0001, 64);
    let m = Model::load(&b.bytes).unwrap();
    let mut h = Sha256::new();
    for w in run_reference(&m, &tokens(0x7ac0_0002, 150, 300)) {
        for x in w {
            h.update(x.to_le_bytes());
        }
    }
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    println!("tiny wide model {} output {got}", m.hash());
    assert_eq!(format!("{}", m.hash()), GOLDEN_V2_MODEL);
    assert_eq!(got, GOLDEN_V2_OUTPUT);
}

/// The SHA-256 of the tiny wide model file (measured on the Mac, 2026-10-19).
const GOLDEN_V2_MODEL: &str = "201f9fd6df4aa0467657380c3890bdb25330510d7253d0fcf5f6cd9211e6af04";
/// The SHA-256 of its weights (measured on the Mac, 2026-10-19).
const GOLDEN_V2_OUTPUT: &str = "64108429f3ce999899c1df4da4d8c364138b1315143d08d2ab01ab7b2dde2eb8";

proptest! {
    #[test]
    fn matmul_i8_i16_equals_the_reference(
        rows in 1usize..6, cols in 1usize..700, n in 1usize..4, seed in any::<u64>(),
        extreme in any::<bool>(),
    ) {
        let mut r = Rng::new(seed);
        let w: Vec<i8> = (0..rows * cols)
            .map(|_| if extreme { -128 } else { (r.below(256) as u8).cast_signed() })
            .collect();
        let x: Vec<i16> = (0..n * cols)
            .map(|_| if extreme { i16::MIN } else { (r.below(65_536) as u16).cast_signed() })
            .collect();
        let m = Matrix::new(rows, cols, w.clone()).unwrap();
        let mut want = vec![0i64; n * rows];
        for (xr, wr) in x.chunks_exact(cols).zip(want.chunks_exact_mut(rows)) {
            matvec_i8_i16(&m, xr, wr).unwrap();
        }
        for k in Kernels::available() {
            let mut got = vec![0i64; n * rows];
            k.matmul_i8_i16(&w, &x, cols, &mut got).unwrap();
            prop_assert_eq!(&got, &want);
        }
    }

    #[test]
    fn layernorm_beta_frac_with_whole_units_equals_version_1(
        x in proptest::collection::vec(-(1i32 << 30)..(1i32 << 30), 1..40),
        g in any::<i16>(), b in -(1i32 << 14)..(1i32 << 14), m in 1i32.., s in 16u8..100,
        frac in 1u8..=16,
    ) {
        // A beta of whole output units: beta * 2^F in the finer units gives the same output,
        // because beta * 2^s is a multiple of 2^s (one rounding = rounding, then the add).
        let n = x.len();
        let c = LnConstants::new(48, 8, 1).unwrap();
        let v1 = LayerNorm::new(vec![g; n], vec![b; n], m, s).unwrap();
        let fine = vec![b << frac; n];
        let Ok(v2) = LayerNorm::with_beta_frac(vec![g; n], fine, m, s, frac) else {
            prop_assume!(false);
            unreachable!()
        };
        let (mut o1, mut o2) = (vec![0i8; n], vec![0i8; n]);
        v1.apply(c, &x, &mut o1).unwrap();
        v2.apply(c, &x, &mut o2).unwrap();
        prop_assert_eq!(o1, o2);
    }

    #[test]
    fn layernorm_beta_frac_rounds_one_time(
        x in proptest::collection::vec(-(1i32 << 30)..(1i32 << 30), 2..40),
        g in any::<i16>(), b in any::<i32>(), m in 1i32.., s in 16u8..=100, frac in 1u8..=16,
    ) {
        // The output is floor((t * m + b * 2^(s-F) + 2^(s-1)) / 2^s), clamped, with
        // t = (x - mean) * inv * gamma, calculated here with i128 and the LayerNorm steps.
        prop_assume!(s - frac <= 92);
        let n = x.len();
        let c = LnConstants::new(48, 8, 1).unwrap();
        let ln = LayerNorm::with_beta_frac(vec![g; n], vec![b; n], m, s, frac).unwrap();
        let mut out = vec![0i16; n];
        ln.apply_i16(c, &x, &mut out).unwrap();
        let sum: i64 = x.iter().map(|&v| i64::from(v)).sum();
        let mean = sum.div_euclid(i64::try_from(n).unwrap());
        let ss: u128 = x.iter().map(|&v| ((i64::from(v) - mean).unsigned_abs() as u128).pow(2)).sum();
        let var = ss / n as u128;
        let root = hardshell::int::isqrt_u128((var << 16) + 1).max(1);
        let inv = i128::try_from((1u128 << 48) / root).unwrap();
        for (&v, &o) in x.iter().zip(&out) {
            let t = i128::from(i64::from(v) - mean) * inv * i128::from(g);
            let p = t * i128::from(m) + (i128::from(b) << (s - frac)) + (1i128 << (s - 1));
            let want = (p >> s).clamp(-32767, 32767);
            prop_assert_eq!(i128::from(o), want);
        }
    }

    #[test]
    fn layernorm_i8_output_is_the_clamp_of_the_i16_output(
        x in proptest::collection::vec(-(1i32 << 30)..(1i32 << 30), 1..40),
        g in any::<i16>(), b in -(1i32 << 20)..(1i32 << 20), m in 1i32.., s in 0u8..60,
    ) {
        let n = x.len();
        let ln = LayerNorm::new(vec![g; n], vec![b; n], m, s).unwrap();
        let c = LnConstants::new(48, 8, 1).unwrap();
        let (mut o8, mut o16) = (vec![0i8; n], vec![0i16; n]);
        ln.apply(c, &x, &mut o8).unwrap();
        ln.apply_i16(c, &x, &mut o16).unwrap();
        for (a, w) in o8.iter().zip(&o16) {
            prop_assert!((-32767..=32767).contains(w));
            prop_assert_eq!(i16::from(*a), (*w).clamp(-127, 127));
        }
    }
}
