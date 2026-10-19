//! Regression tests for the findings of docs/security.md in the `.hsm` reader.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Tests fail loudly.

use hardshell::Error;
use hardshell::hsm::HsmFile;

/// The committed tiny model (conformance/README.md).
const TINY: &[u8] = include_bytes!("../../../conformance/tiny.hsm");

fn old_header_len() -> usize {
    usize::try_from(u32::from_le_bytes(TINY[12..16].try_into().unwrap())).unwrap()
}

/// The tiny model with `text` in place of its header. The sections do not move, so only a
/// header that the JSON parser refuses is a useful test.
fn with_header(text: &str) -> Vec<u8> {
    let mut f = TINY[..12].to_vec();
    f.extend_from_slice(&u32::try_from(text.len()).unwrap().to_le_bytes());
    f.extend_from_slice(text.as_bytes());
    f.extend_from_slice(&TINY[16 + old_header_len()..]);
    f
}

fn header_text() -> String {
    String::from_utf8(TINY[16..16 + old_header_len()].to_vec()).unwrap()
}

/// Finding F3: a serde message quotes a field name of the header. Before the fix, an unknown
/// field of 100,000 bytes gave an error message of more than 100,000 bytes.
#[test]
fn json_error_message_is_short_for_a_long_unknown_field() {
    for name in ["a".repeat(100_000), "\u{e9}".repeat(50_000)] {
        let text = header_text().replacen('{', &format!("{{\"{name}\":1,"), 1);
        match HsmFile::parse(&with_header(&text)) {
            Err(e @ Error::HeaderJson { .. }) => {
                let shown = e.to_string();
                assert!(shown.len() < 300, "the message has {} bytes", shown.len());
            }
            other => panic!("{other:?}"),
        }
    }
}

/// Finding F3: a string where the schema has a number is quoted too.
#[test]
fn json_error_message_is_short_for_a_long_string_value() {
    let long = format!("\"n_layer\":\"{}\"", "x".repeat(70_000));
    let text = header_text().replacen("\"n_layer\":2", &long, 1);
    assert_ne!(text, header_text());
    match HsmFile::parse(&with_header(&text)) {
        Err(e @ Error::HeaderJson { .. }) => {
            let shown = e.to_string();
            assert!(shown.len() < 300, "the message has {} bytes", shown.len());
        }
        other => panic!("{other:?}"),
    }
}
