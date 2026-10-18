//! Tests for the integer GPT-2: loading, determinism, the float comparison and the golden
//! conformance hash.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::print_stdout)]

mod common;

use common::builder::{build, tiny_dims};
use common::{Rng, kl_bits, pow2, softmax};
use hardshell::gpt2::{Model, Session};
use hardshell::{Error, TokenId};
use sha2::{Digest, Sha256};

fn tokens(seed: u64, n: usize, vocab: usize) -> Vec<TokenId> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| TokenId::new(r.below(vocab as u64) as u32)).collect()
}

/// All weights of a block: the start, then one step for each token.
fn run(s: &mut Session<'_>, toks: &[TokenId]) -> Vec<Vec<u64>> {
    let mut out = vec![s.start_block().unwrap().to_vec()];
    for &t in toks {
        out.push(s.step(t).unwrap().to_vec());
    }
    out
}

#[test]
fn tiny_model_loads_and_gives_valid_weights() {
    let b = build(tiny_dims(), 1, 64);
    let m = Model::load(&b.bytes).unwrap();
    assert_eq!(m.vocab_size(), 300);
    assert_eq!(m.n_ctx(), 64);
    assert_eq!(m.eot_token(), TokenId::new(299));
    assert_eq!(m.vocabulary().token(TokenId::new(299)), Some(&b"<|endoftext|>"[..]));
    let mut s = Session::new(&m);
    for w in run(&mut s, &tokens(2, 20, 300)) {
        assert_eq!(w.len(), 300);
        assert!(w.iter().any(|&x| x >= 1));
        assert!(w.iter().map(|&x| u128::from(x)).sum::<u128>() <= 1 << 60);
    }
}

#[test]
fn step_refuses_a_token_outside_the_vocabulary() {
    let b = build(tiny_dims(), 1, 16);
    let m = Model::load(&b.bytes).unwrap();
    let mut s = Session::new(&m);
    assert_eq!(
        s.step(TokenId::new(300)).unwrap_err(),
        Error::TokenOutOfRange { token: 300, vocab_size: 300 }
    );
    assert_eq!(s.position(), 0);
}

#[test]
fn sessions_are_deterministic_and_reset_is_a_fresh_start() {
    let b = build(tiny_dims(), 3, 64);
    let m = Model::load(&b.bytes).unwrap();
    let toks = tokens(4, 40, 300);
    let mut s1 = Session::new(&m);
    let mut s2 = Session::new(&m);
    let a = run(&mut s1, &toks);
    assert_eq!(a, run(&mut s2, &toks));
    // Repeated run on the same session after reset (start_block resets).
    assert_eq!(a, run(&mut s1, &toks));
    // A different history, then reset: the same as fresh.
    let mut s3 = Session::new(&m);
    run(&mut s3, &tokens(5, 17, 300));
    s3.reset();
    assert_eq!(s3.position(), 0);
    assert_eq!(a, run(&mut s3, &toks));
    // A second load of the same bytes gives the same model.
    let m2 = Model::load(&b.bytes).unwrap();
    assert_eq!(m2.hash(), m.hash());
    assert_eq!(a, run(&mut Session::new(&m2), &toks));
}

#[test]
fn full_context_starts_again_with_endoftext() {
    let d = tiny_dims();
    let b = build(d, 6, 64);
    let m = Model::load(&b.bytes).unwrap();
    let toks = tokens(7, 2 * d.n_ctx, 300);
    let mut s = Session::new(&m);
    let all = run(&mut s, &toks);
    // start_block used position 0; tokens 0..62 fill positions 1..63. Token 63 restarts.
    let restart_at = d.n_ctx - 1;
    let mut fresh = Session::new(&m);
    fresh.start_block().unwrap();
    let w = fresh.step(toks[restart_at]).unwrap().to_vec();
    assert_eq!(all[restart_at + 1], w);
    // The second restart is at token 63 + 1 + 62 = 126; then token 127 is at position 2.
    let second = restart_at + d.n_ctx - 1;
    fresh.start_block().unwrap();
    assert_eq!(all[second + 1], fresh.step(toks[second]).unwrap().to_vec());
    assert_eq!(s.position(), 3);
}

#[test]
fn load_refuses_missing_extra_and_wrong_tensors() {
    let b = build(tiny_dims(), 1, 8);
    let f = hardshell::hsm::HsmFile::parse(&b.bytes).unwrap();
    let rebuild = |drop: &str, extra: bool, wrong: Option<&str>| {
        let mut w = hardshell::hsm::HsmWriter::new(f.config().clone());
        w.set_vocabulary(
            (0..300).map(|i| f.vocabulary().token(TokenId::new(i)).unwrap().to_vec()).collect(),
        );
        for name in f.tensor_names() {
            if name == drop {
                continue;
            }
            let t = f.tensor(name).unwrap();
            if Some(name.as_str()) == wrong {
                let n = t.data().dims().iter().product::<usize>();
                w.add_tensor(
                    name.clone(),
                    hardshell::hsm::TensorData::I8(
                        hardshell::tensor::Tensor::new(&[n], vec![0; n]).unwrap(),
                    ),
                    None,
                );
            } else {
                w.add_tensor(name.clone(), t.data().clone(), t.shift());
            }
        }
        if extra {
            w.add_tensor(
                "extra",
                hardshell::hsm::TensorData::I8(
                    hardshell::tensor::Tensor::new(&[1], vec![0]).unwrap(),
                ),
                None,
            );
        }
        let bytes = w.finish().unwrap();
        Model::load(&bytes).expect_err("the model loaded")
    };
    assert_eq!(
        rebuild("h.1.mlp.gelu", false, None),
        Error::MissingTensor { name: "h.1.mlp.gelu".into() }
    );
    assert!(matches!(rebuild("", true, None), Error::UnexpectedTensor { .. }));
    assert!(matches!(rebuild("", false, Some("wte")), Error::TensorMismatch { .. }));
    assert!(matches!(rebuild("", false, Some("h.0.ln_1.g")), Error::TensorMismatch { .. }));
    assert!(matches!(
        rebuild("", false, Some("h.1.attn.score.rescale")),
        Error::TensorMismatch { .. }
    ));
}

/// Mean KL(float || integer) in bits per token, the largest KL, and the top-1 agreement.
fn compare_with_float(seed: u64, n_seq: usize) -> (f64, f64, f64) {
    let d = tiny_dims();
    let b = build(d, seed, 64);
    let m = Model::load(&b.bytes).unwrap();
    let mut s = Session::new(&m);
    let (mut kl_sum, mut kl_max, mut agree, mut count) = (0.0, 0.0f64, 0usize, 0usize);
    for q in 0..n_seq {
        let toks = tokens(1000 + seed * 100 + q as u64, d.n_ctx - 1, d.vocab);
        let ints = run(&mut s, &toks);
        let mut ids = vec![b.eot];
        ids.extend(toks.iter().map(|t| t.get() as usize));
        let floats = b.float.forward(&ids, None);
        for (w, l) in ints.iter().zip(&floats) {
            let total: f64 = w.iter().map(|&x| x as f64).sum();
            let qd: Vec<f64> = w.iter().map(|&x| (x as f64 / total).max(pow2(-32))).collect();
            let p = softmax(l);
            let kl = kl_bits(&p, &qd);
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            let am = |v: &[f64]| (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
            let wi = (0..w.len()).fold(0, |b, i| if w[i] > w[b] { i } else { b });
            if am(&p) == wi {
                agree += 1;
            }
            count += 1;
        }
    }
    (kl_sum / count as f64, kl_max, agree as f64 / count as f64)
}

#[test]
fn integer_model_is_close_to_the_float_reference() {
    let mut worst_mean: f64 = 0.0;
    for seed in [11u64, 12, 13] {
        let (kl, kl_max, top1) = compare_with_float(seed, 4);
        println!(
            "tiny model seed {seed}: mean KL {kl:.5} bits/token, max KL {kl_max:.5}, top-1 agreement {:.2}% (measured, 4 x 64 positions)",
            100.0 * top1
        );
        worst_mean = worst_mean.max(kl);
        assert!(top1 > 0.9, "top-1 agreement {top1}");
    }
    assert!(worst_mean < 0.05, "mean KL {worst_mean}");
}

/// The first HardShell conformance vector (plan §11): a fixed tiny model, a fixed token
/// sequence of 150 tokens (more than the context of 64, so the restart is in it), and the
/// SHA-256 over all weights as little-endian `u64`, in step order.
#[test]
fn golden_conformance_hash() {
    let b = build(tiny_dims(), 0x7ac0_0001, 64);
    let model_hash = format!("{}", Model::load(&b.bytes).unwrap().hash());
    let m = Model::load(&b.bytes).unwrap();
    let mut s = Session::new(&m);
    let mut h = Sha256::new();
    for w in run(&mut s, &tokens(0x7ac0_0002, 150, 300)) {
        for x in w {
            h.update(x.to_le_bytes());
        }
    }
    let out: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    println!("golden model hash {model_hash}");
    println!("golden output hash {out}");
    assert_eq!(model_hash, GOLDEN_MODEL);
    assert_eq!(out, GOLDEN_OUTPUT);
}

/// The SHA-256 of the golden model file (measured on the Mac, ARM, 2026-10-18). If this
/// differs on another machine, the test builder made a different file: check the builder, not
/// HardShell.
const GOLDEN_MODEL: &str = "e3055785bc4509abf7b017e52a0f99030a508cfe0810293e7a5efdccdf2d7bb9";
/// The SHA-256 of all weights (measured on the Mac, ARM, 2026-10-18).
const GOLDEN_OUTPUT: &str = "d4063522517854fdcab605c3d3159bd3ac9a15eb92fa6721ff4a276658f90078";
