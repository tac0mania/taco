//! A tiny GPT-2 model for the tests, made with integers only, and test inputs.
//!
//! The model has a byte-level vocabulary (256 single bytes, 12 merges, `<|endoftext|>`), 2
//! layers, width 16 and a context of 24 tokens, so the tests are fast and each block of more
//! than 23 tokens starts the context again (plan §7.7).
//!
//! DETERMINISM: every weight comes from a seeded integer generator, and the exponent tables
//! come from an integer recurrence. No float is used, so the model file and the golden hashes
//! are the same on every machine.
//!
//! The output layer prefers frequent English bytes: the final LayerNorm has a bias `u`, and the
//! embedding row of each token has a component along `u` that grows with the frequency of the
//! token. Thus English text codes smaller than its raw bytes, and random bytes do not.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(dead_code, clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]

use std::sync::OnceLock;

use hardshell::hsm::{Constants, HsmWriter, ModelConfig, TensorData};
use hardshell::tensor::Tensor;
use taco::{Model, TokenId, Tokenizer};

pub(crate) const D_MODEL: usize = 16;
pub(crate) const N_HEAD: usize = 2;
pub(crate) const D_FF: usize = 32;
pub(crate) const N_LAYER: usize = 2;
pub(crate) const N_CTX: usize = 24;
pub(crate) const TABLE_LEN: usize = 8192;
/// round(2^40 * exp(-1/256)), calculated one time (published value of exp, rounded).
const EXP_STEP: u64 = 1_095_225_038_176;

/// The merges, in rank order, as byte strings: (left, right).
pub(crate) const MERGES: &[(&[u8], &[u8])] = &[
    (b"t", b"h"),
    (b"h", b"e"),
    (b"th", b"e"),
    (b" ", b"the"),
    (b"i", b"n"),
    (b"e", b"r"),
    (b"a", b"n"),
    (b" ", b"a"),
    (b"d", b"a"),
    (b"da", b"s"),
    (b"o", b"n"),
    (b" ", b"s"),
];

/// splitmix64: a seeded integer generator.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `lo..=hi`.
    pub(crate) fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as i64
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u64() as u8).collect()
    }
}

/// The vocabulary: 256 single bytes, the merge results, then `<|endoftext|>`.
pub(crate) fn vocabulary() -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    for (a, b) in MERGES {
        v.push([*a, *b].concat());
    }
    v.push(b"<|endoftext|>".to_vec());
    v
}

/// The merges as token-id pairs.
pub(crate) fn merge_ids() -> Vec<(TokenId, TokenId)> {
    let v = vocabulary();
    let id = |b: &[u8]| TokenId::new(v.iter().position(|t| t == b).unwrap() as u32);
    MERGES.iter().map(|(a, b)| (id(a), id(b))).collect()
}

/// How much the output layer prefers a token: -4 (rare) to 4 (frequent).
fn strength(token: &[u8]) -> i64 {
    match token {
        b" the" | b" " | b"e" => 4,
        [c] if b"taoinshr".contains(c) => 3,
        [c] if c.is_ascii_lowercase() => 2,
        [c] if b".,\n'\"".contains(c) => 2,
        [c] if c.is_ascii_uppercase() || c.is_ascii_digit() => 1,
        [c] if c.is_ascii_graphic() => 0,
        [c] if c.is_ascii() => -2,
        [_] => -3,
        b"<|endoftext|>" => -4,
        _ => 3,
    }
}

fn i8_tensor(rows: usize, cols: usize, data: Vec<i8>) -> TensorData {
    TensorData::I8(Tensor::new(&[rows, cols], data).unwrap())
}

fn i32_vec(data: Vec<i32>) -> TensorData {
    let n = data.len();
    TensorData::I32(Tensor::new(&[n], data).unwrap())
}

/// `n` rescale multipliers of 2^30.
fn unit_rescale(n: usize) -> TensorData {
    i32_vec(vec![1 << 30; n])
}

fn random_i8(rng: &mut Rng, n: usize, lo: i64, hi: i64) -> Vec<i8> {
    (0..n).map(|_| rng.range(lo, hi) as i8).collect()
}

fn random_i32(rng: &mut Rng, n: usize, lo: i64, hi: i64) -> Vec<i32> {
    (0..n).map(|_| rng.range(lo, hi) as i32).collect()
}

/// A LayerNorm with gamma near 1024 and the output `z * 2^(76 - s)` plus `beta`.
fn add_layernorm(w: &mut HsmWriter, rng: &mut Rng, prefix: &str, beta: Vec<i32>, s: u8) {
    let g: Vec<i16> = (0..D_MODEL).map(|_| rng.range(900, 1100) as i16).collect();
    w.add_tensor(format!("{prefix}.g"), TensorData::I16(Tensor::new(&[D_MODEL], g).unwrap()), None);
    w.add_tensor(format!("{prefix}.b"), i32_vec(beta), None);
    w.add_tensor(format!("{prefix}.rescale"), unit_rescale(1), Some(s));
}

fn add_linear(w: &mut HsmWriter, rng: &mut Rng, prefix: &str, rows: usize, cols: usize, s: u8) {
    w.add_tensor(
        format!("{prefix}.w"),
        i8_tensor(rows, cols, random_i8(rng, rows * cols, -64, 63)),
        None,
    );
    w.add_tensor(format!("{prefix}.b"), i32_vec(random_i32(rng, rows, -100, 100)), None);
    w.add_tensor(format!("{prefix}.rescale"), unit_rescale(rows), Some(s));
}

/// `table[i] = floor(table[i - 1] * exp(-1/256))` in 2^-40 units, from `first`.
fn exp_table(first: u64) -> Vec<u64> {
    let mut t = Vec::with_capacity(TABLE_LEN);
    let mut e = first;
    for _ in 0..TABLE_LEN {
        t.push(e);
        e = ((u128::from(e) * u128::from(EXP_STEP)) >> 40) as u64;
    }
    t
}

/// The bytes of the tiny model file for `seed`.
pub(crate) fn model_file(seed: u64) -> Vec<u8> {
    let vocab = vocabulary();
    let v = vocab.len();
    let mut rng = Rng::new(seed);
    let config = ModelConfig {
        architecture: "gpt2".into(),
        n_layer: N_LAYER as u32,
        n_head: N_HEAD as u32,
        d_model: D_MODEL as u32,
        d_ff: D_FF as u32,
        vocab_size: v as u32,
        n_ctx: N_CTX as u32,
        eot_token: (v - 1) as u32,
        constants: Constants { ln_k: 48, ln_j: 8, ln_eps: 1, softmax_p: 30 },
    };
    let mut w = HsmWriter::new(config);
    w.set_vocabulary(vocab.clone());

    // The direction of the output bias, one sign for each channel.
    let u: Vec<i64> = (0..D_MODEL).map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 }).collect();
    let mut wte = Vec::with_capacity(v * D_MODEL);
    for tok in &vocab {
        let s = strength(tok);
        for &uj in &u {
            wte.push((uj * s * 6 + rng.range(-20, 20)).clamp(-127, 127) as i8);
        }
    }
    w.add_tensor("wte", i8_tensor(v, D_MODEL, wte), None);
    w.add_tensor("wte.rescale_embed", unit_rescale(v), Some(20));
    w.add_tensor("wte.rescale_logits", unit_rescale(v), Some(33));
    w.add_tensor(
        "wpe",
        i8_tensor(N_CTX, D_MODEL, random_i8(&mut rng, N_CTX * D_MODEL, -40, 40)),
        None,
    );
    w.add_tensor("wpe.rescale", unit_rescale(N_CTX), Some(21));
    for l in 0..N_LAYER {
        let p = format!("h.{l}");
        let beta = random_i32(&mut rng, D_MODEL, -5, 5);
        add_layernorm(&mut w, &mut rng, &format!("{p}.ln_1"), beta, 75);
        add_linear(&mut w, &mut rng, &format!("{p}.attn.c_attn"), 3 * D_MODEL, D_MODEL, 37);
        w.add_tensor(format!("{p}.attn.score.rescale"), unit_rescale(1), Some(32));
        w.add_tensor(format!("{p}.attn.out.rescale"), unit_rescale(1), Some(60));
        add_linear(&mut w, &mut rng, &format!("{p}.attn.c_proj"), D_MODEL, D_MODEL, 27);
        let beta = random_i32(&mut rng, D_MODEL, -5, 5);
        add_layernorm(&mut w, &mut rng, &format!("{p}.ln_2"), beta, 75);
        add_linear(&mut w, &mut rng, &format!("{p}.mlp.c_fc"), D_FF, D_MODEL, 37);
        // A ReLU-like table: x for x >= 0, floor(x / 4) for x < 0 (arithmetic shift).
        let gelu: Vec<i8> =
            (-128i32..128).map(|x| if x >= 0 { x as i8 } else { (x >> 2) as i8 }).collect();
        w.add_tensor(
            format!("{p}.mlp.gelu"),
            TensorData::I8(Tensor::new(&[256], gelu).unwrap()),
            None,
        );
        add_linear(&mut w, &mut rng, &format!("{p}.mlp.c_proj"), D_MODEL, D_FF, 27);
    }
    let beta: Vec<i32> = u.iter().map(|&uj| (uj * 40) as i32).collect();
    add_layernorm(&mut w, &mut rng, "ln_f", beta, 76);
    let attn: Vec<i32> = exp_table(1 << 30).into_iter().map(|e| e as i32).collect();
    w.add_tensor("attn_exp", i32_vec(attn), None);
    let logits = exp_table(1 << 40);
    w.add_tensor("logits_exp", TensorData::U64(Tensor::new(&[TABLE_LEN], logits).unwrap()), None);
    w.finish().unwrap()
}

/// The tiny model of seed 1 and its tokenizer, made one time for each test binary.
pub(crate) struct Tiny {
    pub(crate) file: Vec<u8>,
    pub(crate) model: Model,
    pub(crate) tokenizer: Tokenizer,
}

pub(crate) fn tiny() -> &'static Tiny {
    static T: OnceLock<Tiny> = OnceLock::new();
    T.get_or_init(|| {
        let file = model_file(1);
        let model = Model::load(&file).unwrap();
        let tokenizer =
            Tokenizer::for_vocabulary(model.vocabulary(), model.eot_token(), &merge_ids()).unwrap();
        Tiny { file, model, tokenizer }
    })
}

/// English prose (public domain style text written for the tests).
pub(crate) const ENGLISH: &str = "It was the best of times and the worst of times. The sun rose over the \
hills, and the farmers went out into the fields as they had done for a hundred years. Nobody in \
the village thought that this morning was different from the others, but the old miller stood \
at his door and looked at the road for a long time. Then he went in, and he shut the door.\n";

pub(crate) const HINDI: &str = "हिन्दी एक भारतीय भाषा है। यह देवनागरी लिपि में लिखी जाती है। भारत में \
करोड़ों लोग हिन्दी बोलते हैं।\n";
pub(crate) const CHINESE: &str = "汉字是中文的书写系统。我们今天学习压缩算法，明天学习语言模型。\n";
pub(crate) const ARABIC: &str = "اللغة العربية لغة جميلة، وهي من أكثر اللغات انتشارا في العالم.\n";
pub(crate) const CODE: &str = "fn main() {\n    let xs: Vec<u32> = (0..10).map(|x| x * x).collect();\n    \
for x in &xs {\n        println!(\"{x}\");\n    }\n}\n";
pub(crate) const JSON: &str = "{\"id\": 1850006136, \"user\": {\"name\": \"taco\", \"followers\": 42}, \
\"tags\": [\"compression\", \"gpt2\"], \"ratio\": 1.992, \"ok\": true, \"next\": null}\n";
pub(crate) const LOGS: &str = "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 \
for block blk_38865049064139660 terminating\n081109 203807 222 INFO dfs.DataNode$PacketResponder: \
PacketResponder 0 for block blk_-6952295868487656571 terminating\n[Sun Dec 04 04:47:44 2005] \
[notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties\n";

/// `n` random lower-case letters and spaces.
pub(crate) fn random_letters(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| b"abcdefghijklmnopqrstuvwxyz "[(rng.next_u64() % 27) as usize]).collect()
}

/// A mixed file of `n` bytes: mostly English, then code, JSON, logs, text of other languages,
/// random letters and random bytes, and one region of 20,000 random bytes in the middle (so
/// that at least one block of 16 KiB is raw).
pub(crate) fn mixed(n: usize) -> Vec<u8> {
    let mut rng = Rng::new(77);
    let parts: [&[u8]; 6] = [
        CODE.as_bytes(),
        JSON.as_bytes(),
        LOGS.as_bytes(),
        HINDI.as_bytes(),
        CHINESE.as_bytes(),
        ARABIC.as_bytes(),
    ];
    let mut out = Vec::with_capacity(n);
    let mut noise_done = false;
    while out.len() < n {
        if !noise_done && out.len() >= n / 2 {
            out.extend(rng.bytes(20_000));
            noise_done = true;
        }
        match rng.next_u64() % 24 {
            i @ 0..=5 => out.extend_from_slice(parts[i as usize]),
            6 => out.extend(rng.bytes(64)),
            7 => out.extend(random_letters(rng.next_u64(), 100)),
            _ => out.extend_from_slice(ENGLISH.as_bytes()),
        }
    }
    out.truncate(n);
    out
}
