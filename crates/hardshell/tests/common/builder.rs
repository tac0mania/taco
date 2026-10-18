//! Builds a random GPT-2 model from a seed: float weights, `i8` quantisation with one scale for
//! each row, calibration with the float reference, and a `.hsm` file. This is a small Rust
//! version of the converter rules in ADR 0004.

use hardshell::hsm::{Constants, HsmWriter, ModelConfig, TensorData};
use hardshell::tensor::Tensor;

use super::reference::{Calib, Dims, FBlock, FloatModel, Lin, QMat};
use super::{Rng, exp_table_i32, exp_table_u64, gelu_table, pow2, quant_mult, quant_mult_rows};

pub(crate) const LN_K: u8 = 48;
pub(crate) const LN_J: u8 = 8;
pub(crate) const SOFTMAX_P: u8 = 30;
/// One score unit and one logit unit: 2^-8 (ADR 0004).
pub(crate) const DELTA_LOG2: i32 = -8;
pub(crate) const TABLE_LEN: usize = 8192;
/// The converter puts the largest calibration residual at 2^22 units (ADR 0004 section 1).
pub(crate) const RES_TARGET_LOG2: i32 = 22;

pub(crate) struct Built {
    pub(crate) bytes: Vec<u8>,
    pub(crate) float: FloatModel,
    pub(crate) eot: usize,
}

fn quantise_rows(rng: &mut Rng, rows: usize, cols: usize, std: f64) -> QMat {
    let mut q = Vec::with_capacity(rows * cols);
    let mut scale = Vec::with_capacity(rows);
    let mut row = vec![0.0; cols];
    for _ in 0..rows {
        for v in &mut row {
            *v = std * rng.normal();
        }
        let max = row.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let s = if max > 0.0 { max / 127.0 } else { 1.0 };
        q.extend(row.iter().map(|v| (v / s).round().clamp(-127.0, 127.0) as i8));
        scale.push(s);
    }
    QMat { q, scale, rows, cols }
}

fn floats(rng: &mut Rng, n: usize, mean: f64, std: f64) -> Vec<f64> {
    (0..n).map(|_| mean + std * rng.normal()).collect()
}

fn lin(rng: &mut Rng, out: usize, inp: usize, std: f64) -> Lin {
    Lin { w: quantise_rows(rng, out, inp, std), b: floats(rng, out, 0.0, 0.02) }
}

/// A random float model with quantised weight matrices.
fn random_model(d: Dims, rng: &mut Rng) -> FloatModel {
    let dm = d.d_model as f64;
    let wte_std = 2.5 / dm.sqrt();
    let wte = quantise_rows(rng, d.vocab, d.d_model, wte_std);
    let wpe = quantise_rows(rng, d.n_ctx, d.d_model, 0.3 * wte_std);
    let blocks = (0..d.n_layer)
        .map(|_| FBlock {
            ln1_g: floats(rng, d.d_model, 1.0, 0.1),
            ln1_b: floats(rng, d.d_model, 0.0, 0.02),
            c_attn: lin(rng, 3 * d.d_model, d.d_model, 1.5 / dm.sqrt()),
            c_proj: lin(rng, d.d_model, d.d_model, 0.5 / dm.sqrt()),
            ln2_g: floats(rng, d.d_model, 1.0, 0.1),
            ln2_b: floats(rng, d.d_model, 0.0, 0.02),
            fc: lin(rng, d.d_ff, d.d_model, 1.0 / dm.sqrt()),
            proj: lin(rng, d.d_model, d.d_ff, 0.5 / (d.d_ff as f64).sqrt()),
        })
        .collect();
    FloatModel {
        dims: d,
        wte,
        wpe,
        blocks,
        lnf_g: floats(rng, d.d_model, 1.0, 0.1),
        lnf_b: floats(rng, d.d_model, 0.0, 0.02),
    }
}

/// The vocabulary: the 256 single bytes, then two-byte tokens, then `<|endoftext|>` last.
fn vocabulary(n: usize) -> Vec<Vec<u8>> {
    assert!(n > 256);
    let mut v: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let mut i = 0usize;
    while v.len() < n - 1 {
        v.push(vec![b'a' + (i % 26) as u8, b'a' + (i / 26 % 26) as u8]);
        i += 1;
    }
    v.push(b"<|endoftext|>".to_vec());
    v
}

fn i32_tensor(v: Vec<i32>) -> TensorData {
    let n = v.len();
    TensorData::I32(Tensor::new(&[n], v).unwrap())
}

fn i8_matrix(m: &QMat) -> TensorData {
    TensorData::I8(Tensor::new(&[m.rows, m.cols], m.q.clone()).unwrap())
}

/// Quantised LayerNorm tensors: gamma (i16), beta (i32, output units), rescale (m, s).
fn add_layernorm(w: &mut HsmWriter, p: &str, g: &[f64], b: &[f64], s_out: f64) {
    let s_gamma = g.iter().fold(0.0f64, |a, v| a.max(v.abs())) / 32767.0;
    let gq: Vec<i16> = g.iter().map(|v| (v / s_gamma).round() as i16).collect();
    let bq: Vec<i32> = b.iter().map(|v| (v / s_out).round() as i32).collect();
    let (m, s) = quant_mult(s_gamma / (pow2(i32::from(LN_K) - i32::from(LN_J)) * s_out));
    w.add_tensor(format!("{p}.g"), TensorData::I16(Tensor::new(&[gq.len()], gq).unwrap()), None);
    w.add_tensor(format!("{p}.b"), i32_tensor(bq), None);
    w.add_tensor(format!("{p}.rescale"), i32_tensor(vec![m]), Some(s));
}

/// Quantised linear tensors. `s_in` is the input scale; `s_out[r]` the output scale of row r.
fn add_linear(w: &mut HsmWriter, p: &str, l: &Lin, s_in: f64, s_out: &dyn Fn(usize) -> f64) {
    let rows = l.w.rows;
    let acc_scale: Vec<f64> = (0..rows).map(|r| s_in * l.w.scale[r]).collect();
    let bq: Vec<i32> = (0..rows).map(|r| (l.b[r] / acc_scale[r]).round() as i32).collect();
    let ratios: Vec<f64> = (0..rows).map(|r| acc_scale[r] / s_out(r)).collect();
    let (m, s) = quant_mult_rows(&ratios);
    w.add_tensor(format!("{p}.w"), i8_matrix(&l.w), None);
    w.add_tensor(format!("{p}.b"), i32_tensor(bq), None);
    w.add_tensor(format!("{p}.rescale"), i32_tensor(m), Some(s));
}

/// Builds a model: random weights from `seed`, calibration on `calib_tokens` (which start with
/// the `<|endoftext|>` token), and the `.hsm` bytes.
pub(crate) fn build(d: Dims, seed: u64, calib_len: usize) -> Built {
    let mut rng = Rng::new(seed);
    let float = random_model(d, &mut rng);
    let eot = d.vocab - 1;
    let mut calib_tokens = vec![eot];
    while calib_tokens.len() < calib_len.min(d.n_ctx) {
        calib_tokens.push(rng.below(d.vocab as u64) as usize);
    }
    let mut c = Calib::default();
    float.forward(&calib_tokens, Some(&mut c));

    let s_res = c.res / pow2(RES_TARGET_LOG2);
    let delta = pow2(DELTA_LOG2);
    let hd = d.d_model / d.n_head;
    let eps =
        ((super::reference::LN_EPS / (s_res * s_res)) * pow2(2 * i32::from(LN_J))).round().max(1.0);
    let config = ModelConfig {
        architecture: "gpt2".into(),
        n_layer: d.n_layer as u32,
        n_head: d.n_head as u32,
        d_model: d.d_model as u32,
        d_ff: d.d_ff as u32,
        vocab_size: d.vocab as u32,
        n_ctx: d.n_ctx as u32,
        eot_token: eot as u32,
        constants: Constants { ln_k: LN_K, ln_j: LN_J, ln_eps: eps as u64, softmax_p: SOFTMAX_P },
    };
    let mut w = HsmWriter::new(config);
    w.set_vocabulary(vocabulary(d.vocab));

    let s_lnf = c.lnf / 127.0;
    w.add_tensor("wte", i8_matrix(&float.wte), None);
    let embed: Vec<f64> = float.wte.scale.iter().map(|s| s / s_res).collect();
    let (m, s) = quant_mult_rows(&embed);
    w.add_tensor("wte.rescale_embed", i32_tensor(m), Some(s));
    let logit: Vec<f64> = float.wte.scale.iter().map(|s| s * s_lnf / delta).collect();
    let (m, s) = quant_mult_rows(&logit);
    w.add_tensor("wte.rescale_logits", i32_tensor(m), Some(s));
    w.add_tensor("wpe", i8_matrix(&float.wpe), None);
    let pos: Vec<f64> = float.wpe.scale.iter().map(|s| s / s_res).collect();
    let (m, s) = quant_mult_rows(&pos);
    w.add_tensor("wpe.rescale", i32_tensor(m), Some(s));

    for (l, b) in float.blocks.iter().enumerate() {
        let p = format!("h.{l}");
        let (s_ln1, s_ln2) = (c.ln1[l] / 127.0, c.ln2[l] / 127.0);
        let (s_q, s_k, s_v) = (c.q[l] / 127.0, c.k[l] / 127.0, c.v[l] / 127.0);
        let s_att = c.att[l] / 127.0;
        let s_gin = c.fc[l] / 32767.0;
        let s_gout = c.gelu[l] / 127.0;
        add_layernorm(&mut w, &format!("{p}.ln_1"), &b.ln1_g, &b.ln1_b, s_ln1);
        let dm = d.d_model;
        let qkv_scale = move |r: usize| {
            if r < dm {
                s_q
            } else if r < 2 * dm {
                s_k
            } else {
                s_v
            }
        };
        add_linear(&mut w, &format!("{p}.attn.c_attn"), &b.c_attn, s_ln1, &qkv_scale);
        let (m, s) = quant_mult(s_q * s_k / ((hd as f64).sqrt() * delta));
        w.add_tensor(format!("{p}.attn.score.rescale"), i32_tensor(vec![m]), Some(s));
        let (m, s) = quant_mult(s_v / (pow2(i32::from(SOFTMAX_P)) * s_att));
        w.add_tensor(format!("{p}.attn.out.rescale"), i32_tensor(vec![m]), Some(s));
        add_linear(&mut w, &format!("{p}.attn.c_proj"), &b.c_proj, s_att, &|_| s_res);
        add_layernorm(&mut w, &format!("{p}.ln_2"), &b.ln2_g, &b.ln2_b, s_ln2);
        add_linear(&mut w, &format!("{p}.mlp.c_fc"), &b.fc, s_ln2, &|_| s_gin);
        let g = gelu_table(16, s_gin, s_gout);
        w.add_tensor(
            format!("{p}.mlp.gelu"),
            TensorData::I8(Tensor::new(&[g.len()], g).unwrap()),
            None,
        );
        add_linear(&mut w, &format!("{p}.mlp.c_proj"), &b.proj, s_gout, &|_| s_res);
    }
    add_layernorm(&mut w, "ln_f", &float.lnf_g, &float.lnf_b, s_lnf);
    w.add_tensor("attn_exp", i32_tensor(exp_table_i32(30, delta, TABLE_LEN)), None);
    let t = exp_table_u64(40, delta, TABLE_LEN);
    w.add_tensor("logits_exp", TensorData::U64(Tensor::new(&[t.len()], t).unwrap()), None);
    Built { bytes: w.finish().unwrap(), float, eot }
}

/// The tiny test model of the brief: 2 layers, width 64, 4 heads, vocabulary 300, context 64.
pub(crate) fn tiny_dims() -> Dims {
    Dims { n_layer: 2, n_head: 4, d_model: 64, d_ff: 256, vocab: 300, n_ctx: 64 }
}

/// GPT-2 small shapes (published): 12 layers, width 768, 12 heads, vocabulary 50,257.
pub(crate) fn gpt2_small_dims() -> Dims {
    Dims { n_layer: 12, n_head: 12, d_model: 768, d_ff: 3072, vocab: 50_257, n_ctx: 1024 }
}
