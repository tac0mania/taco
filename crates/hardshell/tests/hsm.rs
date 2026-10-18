//! Tests for the `.hsm` reader and writer: round trip, and refusal of every bad file.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(clippy::cast_possible_truncation)]

use hardshell::Error;
use hardshell::hsm::{Constants, HsmFile, HsmWriter, ModelConfig, TensorData};
use hardshell::tensor::Tensor;
use hardshell::types::TokenId;
use proptest::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn config() -> ModelConfig {
    ModelConfig {
        architecture: "gpt2".into(),
        n_layer: 2,
        n_head: 4,
        d_model: 8,
        d_ff: 32,
        vocab_size: 5,
        n_ctx: 16,
        eot_token: 4,
        constants: Constants { ln_k: 48, ln_j: 8, ln_eps: 1234, softmax_p: 30 },
    }
}

fn vocab() -> Vec<Vec<u8>> {
    vec![b"a".to_vec(), b"bc".to_vec(), vec![], vec![0xff, 0x00], b"<|endoftext|>".to_vec()]
}

fn sample_file() -> Vec<u8> {
    let mut w = HsmWriter::new(config());
    w.set_vocabulary(vocab());
    w.add_tensor(
        "w",
        TensorData::I8(Tensor::new(&[2, 3], vec![1, -2, 3, -4, 5, -128]).unwrap()),
        None,
    );
    w.add_tensor("g", TensorData::I16(Tensor::new(&[3], vec![-1, 0, 32767]).unwrap()), None);
    w.add_tensor("m", TensorData::I32(Tensor::new(&[2], vec![1 << 30, -7]).unwrap()), Some(31));
    w.add_tensor("t", TensorData::U64(Tensor::new(&[1], vec![u64::MAX]).unwrap()), None);
    w.finish().unwrap()
}

fn header_len(f: &[u8]) -> usize {
    u32::from_le_bytes(f[12..16].try_into().unwrap()) as usize
}

fn header_value(f: &[u8]) -> Value {
    serde_json::from_slice(&f[16..16 + header_len(f)]).unwrap()
}

/// Rebuilds `orig` with a changed header. `edit_value` changes the JSON value, `edit_text`
/// changes the serialised text. All sections move by a multiple of 64 bytes, so the layout
/// stays canonical unless an edit changes it.
fn rebuild(
    orig: &[u8],
    edit_value: impl Fn(&mut Value),
    edit_text: impl Fn(String) -> String,
) -> Vec<u8> {
    let base = header_value(orig);
    let vocab_offset = base["vocabulary"]["offset"].as_u64().unwrap() as usize;
    for k in 0..64u64 {
        let mut v = base.clone();
        let shift = 64 * k;
        let off = &mut v["vocabulary"]["offset"];
        *off = json!(off.as_u64().unwrap() + shift);
        for t in v["tensors"].as_array_mut().unwrap() {
            t["offset"] = json!(t["offset"].as_u64().unwrap() + shift);
        }
        edit_value(&mut v);
        let text = edit_text(serde_json::to_string(&v).unwrap());
        let room = vocab_offset + shift as usize - 16;
        if text.len() <= room {
            let mut out = orig[..12].to_vec();
            out.extend_from_slice(&u32::try_from(room).unwrap().to_le_bytes());
            out.extend_from_slice(text.as_bytes());
            out.resize(16 + room, b' ');
            out.extend_from_slice(&orig[vocab_offset..]);
            return out;
        }
    }
    panic!("header does not fit");
}

#[test]
fn round_trip_keeps_every_part() {
    let f = sample_file();
    let h = HsmFile::parse(&f).unwrap();
    assert_eq!(h.config(), &config());
    for (i, t) in vocab().iter().enumerate() {
        assert_eq!(h.vocabulary().token(TokenId::new(i as u32)), Some(&t[..]));
    }
    assert_eq!(h.vocabulary().token(TokenId::new(5)), None);
    assert_eq!(h.tensor_names(), ["w", "g", "m", "t"]);
    let m = h.tensor("m").unwrap();
    assert_eq!(m.shift(), Some(31));
    assert_eq!(m.data(), &TensorData::I32(Tensor::new(&[2], vec![1 << 30, -7]).unwrap()));
    assert_eq!(h.tensor("w").unwrap().data().dims(), [2, 3]);
    let digest: [u8; 32] = Sha256::digest(&f).into();
    assert_eq!(h.hash().as_bytes(), &digest);
    // Every section is aligned to 64 bytes and the header is compact JSON.
    let v = header_value(&f);
    assert_eq!(v["vocabulary"]["offset"].as_u64().unwrap() % 64, 0);
    for t in v["tensors"].as_array().unwrap() {
        assert_eq!(t["offset"].as_u64().unwrap() % 64, 0);
    }
    assert!(!String::from_utf8_lossy(&f[16..16 + header_len(&f)]).trim_end().contains(": "));
}

#[test]
fn writer_output_is_the_same_every_time() {
    assert_eq!(sample_file(), sample_file());
}

#[test]
fn rebuild_without_edits_is_accepted() {
    let f = rebuild(&sample_file(), |_| {}, |t| t);
    HsmFile::parse(&f).unwrap();
}

#[test]
fn reader_refuses_bad_magic_and_version() {
    let mut f = sample_file();
    f[0] = b'X';
    assert_eq!(HsmFile::parse(&f).unwrap_err(), Error::BadMagic);
    let mut f = sample_file();
    f[8] = 2;
    assert_eq!(HsmFile::parse(&f).unwrap_err(), Error::UnsupportedVersion { version: 2 });
}

#[test]
fn reader_refuses_every_truncation() {
    let f = sample_file();
    for n in 0..f.len() {
        assert!(HsmFile::parse(&f[..n]).is_err(), "prefix of {n} bytes was accepted");
    }
}

#[test]
fn reader_refuses_trailing_bytes() {
    let mut f = sample_file();
    f.push(0);
    assert!(matches!(HsmFile::parse(&f), Err(Error::Layout { .. })));
}

fn assert_json_error(f: &[u8]) {
    match HsmFile::parse(f) {
        Err(Error::HeaderJson { .. }) => {}
        other => panic!("expected a JSON error, got {other:?}"),
    }
}

#[test]
fn reader_refuses_floats_in_the_header() {
    let f = sample_file();
    assert_json_error(&rebuild(&f, |v| v["model"]["n_layer"] = json!(2.0), |t| t));
    assert_json_error(&rebuild(&f, |_| {}, |t| t.replace("\"eot_token\":4", "\"eot_token\":4e0")));
    assert_json_error(&rebuild(&f, |_| {}, |t| t.replace("\"ln_eps\":1234", "\"ln_eps\":1.234e3")));
    assert_json_error(&rebuild(&f, |v| v["tensors"][0]["shape"] = json!([2.0, 3]), |t| t));
    assert_json_error(&rebuild(&f, |v| v["model"]["n_head"] = json!(-4), |t| t));
    assert_json_error(&rebuild(
        &f,
        |v| v["model"]["n_head"] = json!(18_446_744_073_709_551_616.0),
        |t| t,
    ));
}

#[test]
fn reader_refuses_unknown_missing_and_duplicate_fields() {
    let f = sample_file();
    assert_json_error(&rebuild(&f, |v| v["model"]["dropout"] = json!(0), |t| t));
    assert_json_error(&rebuild(&f, |v| v["extra"] = json!(1), |t| t));
    assert_json_error(&rebuild(
        &f,
        |v| drop(v["model"].as_object_mut().unwrap().remove("d_ff")),
        |t| t,
    ));
    assert_json_error(&rebuild(
        &f,
        |_| {},
        |t| t.replace("\"n_layer\":2", "\"n_layer\":2,\"n_layer\":2"),
    ));
    assert_json_error(&rebuild(&f, |v| v["tensors"][1]["scale"] = json!(1), |t| t));
}

#[test]
fn reader_refuses_a_header_that_is_not_utf8() {
    let f = sample_file();
    let g = rebuild(&f, |_| {}, |t| t.replace("\"gpt2\"", "\"gpt\u{e9}\""));
    let mut g2 = g.clone();
    let pos = g2.windows(2).position(|w| w == [0xc3, 0xa9]).unwrap();
    g2[pos] = 0xff;
    assert_eq!(HsmFile::parse(&g2).unwrap_err(), Error::HeaderNotUtf8);
    // The valid UTF-8 form is refused for its name.
    assert!(matches!(HsmFile::parse(&g), Err(Error::InvalidHeader { .. })));
}

#[test]
fn reader_refuses_bad_model_fields() {
    let f = sample_file();
    let inv = |g: Vec<u8>| {
        assert!(
            matches!(HsmFile::parse(&g), Err(Error::InvalidHeader { .. })),
            "{:?}",
            HsmFile::parse(&g).err()
        )
    };
    inv(rebuild(&f, |v| v["model"]["eot_token"] = json!(5), |t| t));
    inv(rebuild(&f, |v| v["model"]["vocab_size"] = json!(0), |t| t));
    inv(rebuild(&f, |v| v["model"]["constants"]["ln_k"] = json!(49), |t| t));
    inv(rebuild(&f, |v| v["model"]["constants"]["ln_eps"] = json!(0), |t| t));
    inv(rebuild(&f, |v| v["model"]["constants"]["softmax_p"] = json!(31), |t| t));
    inv(rebuild(&f, |v| v["model"]["architecture"] = json!("GPT2"), |t| t));
    inv(rebuild(&f, |v| v["tensors"][0]["name"] = json!(""), |t| t));
}

#[test]
fn reader_refuses_bad_tensor_entries() {
    let f = sample_file();
    let mismatch = |g: Vec<u8>| match HsmFile::parse(&g) {
        Err(Error::TensorMismatch { .. } | Error::InvalidShape { .. }) => {}
        other => panic!("expected a tensor error, got {other:?}"),
    };
    mismatch(rebuild(&f, |v| v["tensors"][0]["dtype"] = json!("f32"), |t| t));
    mismatch(rebuild(
        &f,
        |v| v["tensors"][0]["shape"] = json!([3, 2, 1]),
        |t| t.replace("[3,2,1]", "[3,2,2]"),
    ));
    mismatch(rebuild(&f, |v| v["tensors"][0]["shape"] = json!([6, 0]), |t| t));
    mismatch(rebuild(&f, |v| v["tensors"][0]["shape"] = json!([]), |t| t));
    mismatch(rebuild(&f, |v| v["tensors"][0]["shift"] = json!(3), |t| t));
    mismatch(rebuild(&f, |v| v["tensors"][1]["length"] = json!(4), |t| t));
    assert!(matches!(
        HsmFile::parse(&rebuild(&f, |v| v["tensors"][1]["name"] = json!("w"), |t| t)),
        Err(Error::UnexpectedTensor { .. })
    ));
}

#[test]
fn reader_refuses_a_non_canonical_layout() {
    let f = sample_file();
    let layout = |g: Vec<u8>| {
        assert!(
            matches!(HsmFile::parse(&g), Err(Error::Layout { .. })),
            "{:?}",
            HsmFile::parse(&g).err()
        )
    };
    layout(rebuild(
        &f,
        |v| v["tensors"][1]["offset"] = json!(v["tensors"][1]["offset"].as_u64().unwrap() + 64),
        |t| t,
    ));
    layout(rebuild(
        &f,
        |v| v["tensors"][1]["offset"] = json!(v["tensors"][1]["offset"].as_u64().unwrap() - 64),
        |t| t,
    ));
    layout(rebuild(
        &f,
        |v| v["vocabulary"]["offset"] = json!(v["vocabulary"]["offset"].as_u64().unwrap() + 64),
        |t| t,
    ));
    layout(rebuild(
        &f,
        |v| v["tensors"][3]["offset"] = json!(v["tensors"][3]["offset"].as_u64().unwrap() + 1),
        |t| t,
    ));
}

#[test]
fn reader_refuses_non_zero_padding() {
    let f = sample_file();
    let v = header_value(&f);
    // The gap after tensor "w" (6 bytes) before tensor "g".
    let w_end = (v["tensors"][0]["offset"].as_u64().unwrap() + 6) as usize;
    let mut g = f.clone();
    g[w_end] = 1;
    assert_eq!(HsmFile::parse(&g).unwrap_err(), Error::NonZeroPadding { offset: w_end as u64 });
    // The gap between the header and the vocabulary.
    let voff = v["vocabulary"]["offset"].as_u64().unwrap() as usize;
    let hl = header_len(&f);
    if 16 + hl < voff {
        let mut g = f.clone();
        g[16 + hl] = 1;
        assert!(matches!(HsmFile::parse(&g), Err(Error::NonZeroPadding { .. })));
    }
}

#[test]
fn reader_refuses_a_bad_vocabulary() {
    let f = sample_file();
    let vocab_err = |g: Vec<u8>| {
        assert!(
            matches!(HsmFile::parse(&g), Err(Error::Vocabulary { .. } | Error::Layout { .. })),
            "{:?}",
            HsmFile::parse(&g).err()
        )
    };
    // One token fewer than vocab_size says.
    vocab_err(rebuild(&f, |v| v["model"]["vocab_size"] = json!(6), |t| t));
    // A token length that runs past the section.
    let voff = header_value(&f)["vocabulary"]["offset"].as_u64().unwrap() as usize;
    let mut g = f.clone();
    g[voff] = 200;
    vocab_err(g);
    // Extra bytes in the section.
    vocab_err(rebuild(
        &f,
        |v| v["model"]["vocab_size"] = json!(4),
        |t| t.replace("\"eot_token\":4", "\"eot_token\":3"),
    ));
}

#[test]
fn writer_refuses_what_the_reader_refuses() {
    let mut w = HsmWriter::new(config());
    w.set_vocabulary(vec![b"x".to_vec(); 4]);
    assert!(w.finish().is_err());
    let mut w = HsmWriter::new(config());
    w.set_vocabulary(vocab());
    w.add_tensor("Bad", TensorData::I8(Tensor::new(&[1], vec![0]).unwrap()), None);
    assert!(w.finish().is_err());
    let mut w = HsmWriter::new(config());
    w.set_vocabulary(vec![vec![0u8; 70_000], vec![], vec![], vec![], vec![]]);
    assert!(matches!(w.finish(), Err(Error::TooLarge { .. })));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn reader_never_panics_on_damaged_files(pos in any::<usize>(), byte in any::<u8>(), cut in any::<usize>()) {
        let mut f = sample_file();
        let p = pos % f.len();
        f[p] = byte;
        let n = f.len() - cut % 8;
        // Any result is permitted; the reader must not panic. An accepted file has its hash.
        if let Ok(h) = HsmFile::parse(&f[..n]) {
            let digest: [u8; 32] = Sha256::digest(&f[..n]).into();
            let got = *h.hash().as_bytes();
            prop_assert_eq!(got, digest);
        }
    }
}
