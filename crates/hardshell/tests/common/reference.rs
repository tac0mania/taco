//! The float reference forward pass of GPT-2, with the dequantised `i8` weights of the model
//! (`q * scale`) and float LayerNorm, GELU and softmax. It also records the largest activation
//! values for calibration.

/// The dimensions of a model.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dims {
    pub(crate) n_layer: usize,
    pub(crate) n_head: usize,
    pub(crate) d_model: usize,
    pub(crate) d_ff: usize,
    pub(crate) vocab: usize,
    pub(crate) n_ctx: usize,
}

/// A matrix of `i8` values with one float scale for each row: the dequantised value is
/// `q[r][c] * scale[r]`.
#[derive(Clone, Debug)]
pub(crate) struct QMat {
    pub(crate) q: Vec<i8>,
    pub(crate) scale: Vec<f64>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl QMat {
    pub(crate) fn row(&self, r: usize) -> Vec<f64> {
        let s = self.scale[r];
        self.q[r * self.cols..(r + 1) * self.cols].iter().map(|&v| f64::from(v) * s).collect()
    }

    pub(crate) fn matvec(&self, x: &[f64]) -> Vec<f64> {
        (0..self.rows)
            .map(|r| {
                let row = &self.q[r * self.cols..(r + 1) * self.cols];
                let mut sum = 0.0;
                for (&w, &v) in row.iter().zip(x) {
                    sum += f64::from(w) * v;
                }
                sum * self.scale[r]
            })
            .collect()
    }
}

/// A linear layer with dequantised weights and a float bias.
#[derive(Clone, Debug)]
pub(crate) struct Lin {
    pub(crate) w: QMat,
    pub(crate) b: Vec<f64>,
}

impl Lin {
    pub(crate) fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.w.matvec(x).iter().zip(&self.b).map(|(a, b)| a + b).collect()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FBlock {
    pub(crate) ln1_g: Vec<f64>,
    pub(crate) ln1_b: Vec<f64>,
    pub(crate) c_attn: Lin,
    pub(crate) c_proj: Lin,
    pub(crate) ln2_g: Vec<f64>,
    pub(crate) ln2_b: Vec<f64>,
    pub(crate) fc: Lin,
    pub(crate) proj: Lin,
}

#[derive(Clone, Debug)]
pub(crate) struct FloatModel {
    pub(crate) dims: Dims,
    pub(crate) wte: QMat,
    pub(crate) wpe: QMat,
    pub(crate) blocks: Vec<FBlock>,
    pub(crate) lnf_g: Vec<f64>,
    pub(crate) lnf_b: Vec<f64>,
}

/// The largest absolute activation values, for calibration.
#[derive(Clone, Debug, Default)]
pub(crate) struct Calib {
    pub(crate) res: f64,
    pub(crate) ln1: Vec<f64>,
    pub(crate) q: Vec<f64>,
    pub(crate) k: Vec<f64>,
    pub(crate) v: Vec<f64>,
    pub(crate) att: Vec<f64>,
    pub(crate) ln2: Vec<f64>,
    pub(crate) fc: Vec<f64>,
    pub(crate) gelu: Vec<f64>,
    pub(crate) lnf: f64,
}

fn amax(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |a, v| a.max(v.abs()))
}

pub(crate) const LN_EPS: f64 = 1e-5;

pub(crate) fn layernorm(x: &[f64], g: &[f64], b: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    x.iter().zip(g).zip(b).map(|((v, g), b)| (v - mean) * inv * g + b).collect()
}

impl FloatModel {
    /// The logits for each position of `tokens` (one context, no restart). If `calib` is
    /// given, it records the largest activations.
    pub(crate) fn forward(&self, tokens: &[usize], mut calib: Option<&mut Calib>) -> Vec<Vec<f64>> {
        let d = self.dims;
        let hd = d.d_model / d.n_head;
        let inv_sqrt = 1.0 / (hd as f64).sqrt();
        if let Some(c) = calib.as_deref_mut() {
            let z = vec![0.0; d.n_layer];
            *c = Calib {
                ln1: z.clone(),
                q: z.clone(),
                k: z.clone(),
                v: z.clone(),
                att: z.clone(),
                ln2: z.clone(),
                fc: z.clone(),
                gelu: z,
                ..Calib::default()
            };
        }
        let mut kc: Vec<Vec<Vec<f64>>> = vec![Vec::new(); d.n_layer];
        let mut vc: Vec<Vec<Vec<f64>>> = vec![Vec::new(); d.n_layer];
        let mut out = Vec::with_capacity(tokens.len());
        for (t, &tok) in tokens.iter().enumerate() {
            let mut x: Vec<f64> =
                self.wte.row(tok).iter().zip(self.wpe.row(t)).map(|(a, b)| a + b).collect();
            let rec = |c: &mut Calib, x: &[f64]| c.res = c.res.max(amax(x));
            if let Some(c) = calib.as_deref_mut() {
                rec(c, &x);
            }
            for (l, blk) in self.blocks.iter().enumerate() {
                let a = layernorm(&x, &blk.ln1_g, &blk.ln1_b);
                let qkv = blk.c_attn.apply(&a);
                let (q, kv) = qkv.split_at(d.d_model);
                let (k, v) = kv.split_at(d.d_model);
                kc[l].push(k.to_vec());
                vc[l].push(v.to_vec());
                let mut att = vec![0.0; d.d_model];
                for h in 0..d.n_head {
                    let o = h * hd;
                    let scores: Vec<f64> = kc[l]
                        .iter()
                        .map(|kj| {
                            let mut s = 0.0;
                            for c in 0..hd {
                                s += q[o + c] * kj[o + c];
                            }
                            s * inv_sqrt
                        })
                        .collect();
                    let p = super::softmax(&scores);
                    for (pj, vj) in p.iter().zip(&vc[l]) {
                        for c in 0..hd {
                            att[o + c] += pj * vj[o + c];
                        }
                    }
                }
                let proj = blk.c_proj.apply(&att);
                for (xv, p) in x.iter_mut().zip(&proj) {
                    *xv += p;
                }
                let a2 = layernorm(&x, &blk.ln2_g, &blk.ln2_b);
                let fc = blk.fc.apply(&a2);
                let g: Vec<f64> = fc.iter().map(|&v| super::gelu(v)).collect();
                let mp = blk.proj.apply(&g);
                if let Some(c) = calib.as_deref_mut() {
                    c.ln1[l] = c.ln1[l].max(amax(&a));
                    c.q[l] = c.q[l].max(amax(q));
                    c.k[l] = c.k[l].max(amax(k));
                    c.v[l] = c.v[l].max(amax(v));
                    c.att[l] = c.att[l].max(amax(&att));
                    c.ln2[l] = c.ln2[l].max(amax(&a2));
                    c.fc[l] = c.fc[l].max(amax(&fc));
                    c.gelu[l] = c.gelu[l].max(amax(&g));
                    rec(c, &x);
                }
                for (xv, p) in x.iter_mut().zip(&mp) {
                    *xv += p;
                }
                if let Some(c) = calib.as_deref_mut() {
                    rec(c, &x);
                }
            }
            let a = layernorm(&x, &self.lnf_g, &self.lnf_b);
            if let Some(c) = calib.as_deref_mut() {
                c.lnf = c.lnf.max(amax(&a));
            }
            out.push(self.wte.matvec(&a));
        }
        out
    }
}
