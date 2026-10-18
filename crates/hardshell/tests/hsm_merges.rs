//! Tests for the optional merges section of the `.hsm` file: round trip, the hash, and refusal
//! of every bad section.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Test helpers fail loudly.
#![allow(clippy::cast_possible_truncation)]

use hardshell::Error;
use hardshell::gpt2::{Model, Session};
use hardshell::hsm::{Constants, HsmFile, HsmWriter, MergePair, ModelConfig, TensorData};
use hardshell::tensor::Tensor;
use hardshell::types::TokenId;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn config(vocab_size: u32) -> ModelConfig {
    ModelConfig {
        architecture: "gpt2".into(),
        n_layer: 1,
        n_head: 1,
        d_model: 4,
        d_ff: 8,
        vocab_size,
        n_ctx: 4,
        eot_token: vocab_size - 1,
        constants: Constants { ln_k: 48, ln_j: 8, ln_eps: 1, softmax_p: 30 },
    }
}

/// Tokens: "a", "b", "c", "ab", "abc", "<|endoftext|>".
fn vocab() -> Vec<Vec<u8>> {
    vec![
        b"a".to_vec(),
        b"b".to_vec(),
        b"c".to_vec(),
        b"ab".to_vec(),
        b"abc".to_vec(),
        b"<|endoftext|>".to_vec(),
    ]
}

fn t(id: u32) -> TokenId {
    TokenId::new(id)
}

fn writer(vocab: Vec<Vec<u8>>, merges: Vec<(TokenId, TokenId)>) -> HsmWriter {
    let mut w = HsmWriter::new(config(vocab.len() as u32));
    w.set_vocabulary(vocab);
    w.set_merges(merges);
    w.add_tensor("w", TensorData::I8(Tensor::new(&[3], vec![1, -2, 3]).unwrap()), None);
    w
}

fn with_merges() -> Vec<u8> {
    writer(vocab(), vec![(t(0), t(1)), (t(3), t(2))]).finish().unwrap()
}

fn header(f: &[u8]) -> Value {
    let n = u32::from_le_bytes(f[12..16].try_into().unwrap()) as usize;
    serde_json::from_slice(&f[16..16 + n]).unwrap()
}

#[test]
fn merges_round_trip_in_rank_order_with_the_merged_token() {
    let f = with_merges();
    let h = HsmFile::parse(&f).unwrap();
    let m = h.merges().unwrap();
    assert_eq!(m.len(), 2);
    assert_eq!((m[0].left(), m[0].right(), m[0].merged()), (t(0), t(1), t(3)));
    assert_eq!((m[1].left(), m[1].right(), m[1].merged()), (t(3), t(2), t(4)));
    assert_eq!(h.vocabulary().merges(), Some(m));
    // The section is between the vocabulary and the tensors, aligned, 8 bytes for each pair.
    let v = header(&f);
    let off = v["merges"]["offset"].as_u64().unwrap();
    assert_eq!(off % 64, 0);
    assert_eq!(v["merges"]["length"].as_u64().unwrap(), 16);
    assert!(off >= v["vocabulary"]["offset"].as_u64().unwrap());
    assert!(v["tensors"][0]["offset"].as_u64().unwrap() >= off + 16);
    let o = off as usize;
    assert_eq!(&f[o..o + 16], &[0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0]);
    // The header lists the fields in schema order.
    let text = String::from_utf8_lossy(&f[16..]).to_string();
    let (vp, mp, tp) = (
        text.find("\"vocabulary\"").unwrap(),
        text.find("\"merges\"").unwrap(),
        text.find("\"tensors\"").unwrap(),
    );
    assert!(vp < mp && mp < tp);
}

#[test]
fn a_file_without_merges_has_no_section_and_no_field() {
    let f = writer(vocab(), vec![]).finish().unwrap();
    let h = HsmFile::parse(&f).unwrap();
    assert!(h.merges().is_none());
    assert!(header(&f).get("merges").is_none());
    assert_eq!(h.tensor("w").unwrap().data().dims(), [3]);
}

#[test]
fn the_model_hash_covers_the_merges() {
    let a = with_merges();
    let b = writer(vocab(), vec![(t(0), t(1))]).finish().unwrap();
    let c = writer(vocab(), vec![]).finish().unwrap();
    let ha = HsmFile::parse(&a).unwrap().hash();
    let hb = HsmFile::parse(&b).unwrap().hash();
    let hc = HsmFile::parse(&c).unwrap().hash();
    assert_ne!(ha, hb);
    assert_ne!(ha, hc);
    assert_ne!(hb, hc);
    let digest: [u8; 32] = Sha256::digest(&a).into();
    assert_eq!(ha.as_bytes(), &digest);
    assert_eq!(a, with_merges(), "the writer output is the same every time");
}

fn refused_by_writer(vocab: Vec<Vec<u8>>, merges: Vec<(TokenId, TokenId)>) -> Error {
    writer(vocab, merges).finish().unwrap_err()
}

fn assert_merge_error(e: &Error, rank: usize) {
    match e {
        Error::InvalidHeader { field, .. } => assert_eq!(field, &format!("merges[{rank}]")),
        other => panic!("expected a merges error, got {other:?}"),
    }
}

#[test]
fn reader_refuses_a_token_id_outside_the_vocabulary() {
    assert_merge_error(&refused_by_writer(vocab(), vec![(t(0), t(1)), (t(6), t(0))]), 1);
    assert_merge_error(&refused_by_writer(vocab(), vec![(t(0), t(u32::MAX))]), 0);
}

#[test]
fn reader_refuses_a_part_without_bytes() {
    let mut v = vocab();
    v[2] = vec![];
    // "a" + "" = "a" is a token, but an empty part is not a merge.
    assert_merge_error(&refused_by_writer(v, vec![(t(0), t(2))]), 0);
}

#[test]
fn reader_refuses_a_duplicate_pair() {
    assert_merge_error(&refused_by_writer(vocab(), vec![(t(0), t(1)), (t(0), t(1))]), 1);
}

#[test]
fn reader_refuses_joined_bytes_that_are_not_one_token() {
    // "ba" is not a token.
    assert_merge_error(&refused_by_writer(vocab(), vec![(t(1), t(0))]), 0);
    // Two tokens have the bytes "ab".
    let mut v = vocab();
    v[2] = b"ab".to_vec();
    assert_merge_error(&refused_by_writer(v, vec![(t(0), t(1))]), 0);
}

/// Rebuilds `f` with a changed header text of the same length.
fn edit_header(f: &[u8], from: &str, to: &str) -> Vec<u8> {
    assert_eq!(from.len(), to.len());
    let n = u32::from_le_bytes(f[12..16].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&f[16..16 + n]).unwrap();
    assert!(text.contains(from), "{from} is not in the header");
    let mut out = f.to_vec();
    out[16..16 + n].copy_from_slice(text.replacen(from, to, 1).as_bytes());
    out
}

#[test]
fn reader_refuses_a_bad_merges_length_or_offset() {
    let f = with_merges();
    // A length that is not a multiple of 8.
    let g = edit_header(&f, "\"length\":16}", "\"length\":15}");
    assert!(HsmFile::parse(&g).is_err());
    // An empty section: omit the field instead.
    let g = edit_header(&f, "\"length\":16}", "\"length\":0 }");
    match HsmFile::parse(&g) {
        Err(Error::InvalidHeader { field, .. }) => assert_eq!(field, "merges.length"),
        other => panic!("expected a merges.length error, got {other:?}"),
    }
    // An offset that is not align64 of the vocabulary end.
    let off = header(&f)["merges"]["offset"].as_u64().unwrap();
    let (from, to) =
        (format!("\"merges\":{{\"offset\":{off}"), format!("\"merges\":{{\"offset\":{}", off + 64));
    assert_eq!(from.len(), to.len(), "the test needs offsets with the same number of digits");
    let g = edit_header(&f, &from, &to);
    assert!(matches!(HsmFile::parse(&g), Err(Error::Layout { .. })));
}

#[test]
fn reader_refuses_non_zero_padding_before_the_merges() {
    let f = with_merges();
    let v = header(&f);
    let vend = (v["vocabulary"]["offset"].as_u64().unwrap()
        + v["vocabulary"]["length"].as_u64().unwrap()) as usize;
    let off = v["merges"]["offset"].as_u64().unwrap() as usize;
    assert!(vend < off, "the test needs a gap between the vocabulary and the merges");
    let mut g = f.clone();
    g[vend] = 7;
    assert_eq!(HsmFile::parse(&g).unwrap_err(), Error::NonZeroPadding { offset: vend as u64 });
}

#[test]
fn reader_refuses_every_truncation_of_a_file_with_merges() {
    let f = with_merges();
    for n in 0..f.len() {
        assert!(HsmFile::parse(&f[..n]).is_err(), "prefix of {n} bytes was accepted");
    }
}

#[test]
fn reader_refuses_a_damaged_merge_id() {
    let f = with_merges();
    let off = header(&f)["merges"]["offset"].as_u64().unwrap() as usize;
    let mut g = f.clone();
    g[off + 4] = 2; // (a, b) becomes (a, c): "ac" is not a token.
    assert_merge_error(&HsmFile::parse(&g).unwrap_err(), 0);
}

#[test]
fn merge_pair_is_a_small_copy_type() {
    fn is_copy<T: Copy>() {}
    is_copy::<MergePair>();
    assert_eq!(hardshell::hsm::MERGE_PAIR_LEN, 8);
}

/// The committed conformance model loads, has merges, and runs (plan §11.1).
#[test]
fn the_tiny_conformance_model_has_merges_and_runs() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../conformance/tiny.hsm");
    let bytes = std::fs::read(path).unwrap();
    let model = Model::load(&bytes).unwrap();
    let merges = model.vocabulary().merges().unwrap();
    assert!(!merges.is_empty());
    for (i, b) in (0..=255u8).enumerate() {
        assert_eq!(model.vocabulary().token(TokenId::new(i as u32)), Some(&[b][..]));
    }
    let mut s = Session::new(&model);
    let w = s.start_block().unwrap().to_vec();
    assert_eq!(w.len(), model.vocab_size());
    assert!(w.iter().any(|&x| x > 0));
    let w2 = s.step(merges[0].merged()).unwrap();
    assert_eq!(w2.len(), model.vocab_size());
}

/// The Python writer (`tools/convert/hsmfile.py`) and `HsmWriter` make the same bytes: the
/// tiny model, read and written again with `HsmWriter`, is the same file.
#[test]
fn hsm_writer_makes_the_same_bytes_as_the_python_writer() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../conformance/tiny.hsm");
    let bytes = std::fs::read(path).unwrap();
    let h = HsmFile::parse(&bytes).unwrap();
    let mut w = HsmWriter::new(h.config().clone());
    let v = h.vocabulary();
    w.set_vocabulary(
        (0..v.len()).map(|i| v.token(TokenId::new(i as u32)).unwrap().to_vec()).collect(),
    );
    w.set_merges(h.merges().unwrap().iter().map(|m| (m.left(), m.right())).collect());
    for name in h.tensor_names() {
        let t = h.tensor(name).unwrap();
        w.add_tensor(name.clone(), t.data().clone(), t.shift());
    }
    assert_eq!(w.finish().unwrap(), bytes);
}
