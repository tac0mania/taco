//! The tokenizer with the real GPT-2 vocabulary and merges: canonical equality with the Python
//! reference (`tools/baseline/bpe.py`), and the round trip on the corpus and on random bytes.
//!
//! The tests need `tools/data/model/vocab.json` and `merges.txt` (`python -m baseline.fetch`).
//! The canonical test also needs the reference ids: see `crates/taco/scripts/bpe_reference.py`
//! and the variable `TACO_BPE_REFERENCE_DIR`. A test without its files prints why and passes.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Tests fail loudly.
#![allow(clippy::print_stderr, clippy::cast_possible_truncation)] // Skip messages, test data.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use proptest::prelude::*;
use taco::{TokenId, Tokenizer};

/// The GPT-2 vocabulary (token id -> bytes) and the tokenizer.
struct Gpt2 {
    tokens: Vec<Vec<u8>>,
    tokenizer: Tokenizer,
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The GPT-2 map from a byte to a printable character (`bytes_to_unicode` in bpe.py).
fn byte_decoder() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = (u32::from(b'!')..=u32::from(b'~')).collect();
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter().zip(&cs).map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8)).collect()
}

fn load_gpt2() -> Option<Gpt2> {
    let dir = repo().join("tools/data/model");
    let vocab = std::fs::read_to_string(dir.join("vocab.json")).ok()?;
    let merges = std::fs::read_to_string(dir.join("merges.txt")).ok()?;
    let vocab: HashMap<String, u32> = serde_json::from_str(&vocab).unwrap();
    let dec = byte_decoder();
    let mut tokens = vec![Vec::new(); vocab.len()];
    for (text, &id) in &vocab {
        if text != "<|endoftext|>" {
            tokens[id as usize] = text.chars().map(|c| dec[&c]).collect();
        }
    }
    let pairs: Vec<(TokenId, TokenId)> = merges
        .lines()
        .skip(1)
        .filter(|l| !l.is_empty())
        .map(|l| {
            let (a, b) = l.split_once(' ').unwrap();
            (TokenId::new(vocab[a]), TokenId::new(vocab[b]))
        })
        .collect();
    let eot = TokenId::new(vocab["<|endoftext|>"]);
    let tokenizer = Tokenizer::new(tokens.iter().map(Vec::as_slice), Some(eot), &pairs).unwrap();
    assert_eq!(tokenizer.merge_count(), 50_000);
    Some(Gpt2 { tokens, tokenizer })
}

fn gpt2() -> Option<&'static Gpt2> {
    static G: OnceLock<Option<Gpt2>> = OnceLock::new();
    let g = G.get_or_init(load_gpt2).as_ref();
    if g.is_none() {
        eprintln!("SKIPPED: tools/data/model/vocab.json or merges.txt is missing");
    }
    g
}

fn decode(g: &Gpt2, ids: &[TokenId]) -> Vec<u8> {
    ids.iter().flat_map(|t| g.tokens[t.get() as usize].iter().copied()).collect()
}

fn ids(g: &Gpt2, data: &[u8]) -> Vec<u32> {
    g.tokenizer.encode(data).iter().map(|t| t.get()).collect()
}

fn read_ids(path: &Path) -> Vec<u32> {
    let b = std::fs::read(path).unwrap();
    b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

fn corpus_files() -> Option<Vec<PathBuf>> {
    let dir = repo().join("tools/data/corpus");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        eprintln!("SKIPPED: {} is missing (run python -m baseline.fetch)", dir.display());
        return None;
    };
    let mut files: Vec<PathBuf> = rd.map(|e| e.unwrap().path()).filter(|p| p.is_file()).collect();
    files.sort();
    Some(files)
}

#[test]
fn known_gpt2_ids() {
    let Some(g) = gpt2() else { return };
    assert_eq!(ids(g, b"Hello world"), [15496, 995]);
    assert_eq!(ids(g, b" Paris"), [6342]);
}

#[test]
fn corpus_round_trips_with_gpt2_merges() {
    let Some(g) = gpt2() else { return };
    let Some(files) = corpus_files() else { return };
    for f in files {
        let data = std::fs::read(&f).unwrap();
        assert_eq!(decode(g, &g.tokenizer.encode(&data)), data, "{}", f.display());
    }
}

#[test]
fn canonical_equality_with_the_python_reference() {
    let Some(g) = gpt2() else { return };
    let Some(reference) = std::env::var_os("TACO_BPE_REFERENCE_DIR").map(PathBuf::from) else {
        eprintln!("SKIPPED: TACO_BPE_REFERENCE_DIR is not set (see scripts/bpe_reference.py)");
        return;
    };
    let mut cases: Vec<(String, Vec<u8>, Vec<u32>)> = Vec::new();
    if let Some(files) = corpus_files() {
        for f in files {
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            let ids_path = reference.join("corpus").join(format!("{name}.ids"));
            if ids_path.is_file() {
                cases.push((name, std::fs::read(&f).unwrap(), read_ids(&ids_path)));
            }
        }
    }
    let gen_dir = reference.join("generated");
    for i in 0.. {
        let bin = gen_dir.join(format!("{i}.bin"));
        if !bin.is_file() {
            break;
        }
        let ids_path = gen_dir.join(format!("{i}.ids"));
        cases.push((format!("generated/{i}"), std::fs::read(&bin).unwrap(), read_ids(&ids_path)));
    }
    assert!(!cases.is_empty(), "no reference files in {}", reference.display());
    let (mut same_cases, mut same_tokens, mut total_tokens) = (0usize, 0usize, 0usize);
    let mut different = Vec::new();
    for (name, data, want) in &cases {
        let got = ids(g, data);
        total_tokens += want.len();
        same_tokens += got.iter().zip(want).take_while(|(a, b)| a == b).count();
        if &got == want {
            same_cases += 1;
        } else {
            different.push(name.clone());
        }
    }
    eprintln!(
        "canonical: {same_cases} of {} inputs equal; {same_tokens} of {total_tokens} tokens \
         equal before the first difference",
        cases.len()
    );
    assert!(different.is_empty(), "different tokens for: {different:?}");
}

/// Text of several languages, with the special cases of the pre-tokenizer.
const MULTILINGUAL: &[&str] = &[
    "The quick brown fox jumps over the lazy dog. Don't stop; we'll see.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "हिन्दी एक भारतीय भाषा है। यह देवनागरी लिपि में लिखी जाती है।",
    "汉字是中文的书写系统。我们今天学习压缩。",
    "اللغة العربية لغة جميلة، وهي لغة القرآن.",
    "日本語のテキストとカタカナ、ひらがな。",
    "emoji 😀🎉 👍🏽 family 👨\u{200d}👩\u{200d}👧 ❤\u{fe0f}",
    "fn main() {\n    println!(\"{}\", 42);\n}\n\t\ttabs  and   spaces   \n\n\n",
    "{\"key\": [1, 2.5, -3e10], \"nested\": {\"a\": null}}",
    "2026-10-18 14:43:32 INFO dfs.DataNode: Receiving block blk_-1608999687919862906",
    "dasdasdasdada dasdasdasdada",
    "\u{a0}\u{85}\u{2028}\u{3000} odd spaces \u{205f}x",
];

#[test]
fn multilingual_round_trips() {
    let Some(g) = gpt2() else { return };
    for s in MULTILINGUAL {
        assert_eq!(decode(g, &g.tokenizer.encode(s.as_bytes())), s.as_bytes());
    }
    let all: Vec<u8> = (0..=255u8).collect();
    assert_eq!(decode(g, &g.tokenizer.encode(&all)), all);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn random_bytes_round_trip(data in proptest::collection::vec(any::<u8>(), 0..2048)) {
        if let Some(g) = gpt2() {
            prop_assert_eq!(decode(g, &g.tokenizer.encode(&data)), data);
        }
    }

    #[test]
    fn random_text_round_trips(s in "\\PC{0,300}") {
        if let Some(g) = gpt2() {
            prop_assert_eq!(decode(g, &g.tokenizer.encode(s.as_bytes())), s.as_bytes());
        }
    }
}
