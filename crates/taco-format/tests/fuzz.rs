//! Fuzz-style property tests: damaged containers never panic the parser, and an unfold of a
//! damaged container gives an error or the exact original data, never other data.
//!
//! A change in a stored block parses, because the parser does not know the block data. The
//! checksum helper then refuses the restored data (plan §3 rule 4).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{MODEL, UnfoldError, bs, fold, unfold};
use proptest::prelude::*;
use taco_format::{BlockKind, Container, FormatError, HEADER_LEN, ModelHash, TABLE_ENTRY_LEN};

/// Data with runs, so that the container has raw and coded blocks.
fn runs() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec((any::<u8>(), 1usize..30, any::<bool>()), 0..40).prop_map(|parts| {
        let mut out = Vec::new();
        for (v, n, run) in parts {
            if run {
                out.extend(std::iter::repeat_n(v, n));
            } else {
                out.extend((0..n).map(|i| v.wrapping_add(i as u8)));
            }
        }
        out
    })
}

/// One damage to a container.
#[derive(Clone, Debug)]
enum Damage {
    /// XOR one byte with a non-zero value. The position is taken modulo the length.
    Flip(usize, u8),
    /// Remove bytes from the end. The count is taken modulo the length, plus 1.
    Truncate(usize),
    /// Add bytes at the end.
    Append(Vec<u8>),
    /// Insert one byte. The position is taken modulo the length.
    Insert(usize, u8),
    /// Remove one byte. The position is taken modulo the length.
    Remove(usize),
}

fn damage() -> impl Strategy<Value = Damage> {
    prop_oneof![
        4 => (any::<usize>(), 1u8..=255).prop_map(|(p, x)| Damage::Flip(p, x)),
        1 => any::<usize>().prop_map(Damage::Truncate),
        1 => prop::collection::vec(any::<u8>(), 1..8).prop_map(Damage::Append),
        1 => (any::<usize>(), any::<u8>()).prop_map(|(p, b)| Damage::Insert(p, b)),
        1 => any::<usize>().prop_map(Damage::Remove),
    ]
}

fn apply(bytes: &[u8], d: &Damage) -> Vec<u8> {
    let mut out = bytes.to_vec();
    match *d {
        Damage::Flip(p, x) => out[p % bytes.len()] ^= x,
        Damage::Truncate(n) => out.truncate(bytes.len() - 1 - n % bytes.len()),
        Damage::Append(ref extra) => out.extend_from_slice(extra),
        Damage::Insert(p, b) => out.insert(p % bytes.len(), b),
        Damage::Remove(p) => {
            out.remove(p % bytes.len());
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4_096))]

    /// Any damage gives an error or the exact original data.
    #[test]
    fn damaged_container_gives_an_error_or_the_original(
        data in runs(),
        block_size in 1u32..200,
        raw_only in any::<bool>(),
        d in damage(),
    ) {
        let model = if raw_only { ModelHash::RAW_ONLY } else { MODEL };
        let bytes = fold(&data, bs(block_size), model);
        let damaged = apply(&bytes, &d);
        if let Ok(restored) = unfold(&damaged, MODEL) {
            prop_assert_eq!(restored, data);
        }
        // A change of the length always fails in the parser: the header and the table give
        // the exact length.
        if matches!(d, Damage::Truncate(_) | Damage::Append(_)) {
            prop_assert!(Container::parse(&damaged).is_err());
        }
    }

    /// A flip in the magic, the version or the flags always fails in the parser. A flip in a
    /// raw block parses, and the checksum helper refuses the restored data.
    #[test]
    fn flip_is_caught_where_it_lands(
        data in runs(),
        block_size in 1u32..200,
        pos in any::<usize>(),
        x in 1u8..=255,
    ) {
        let bytes = fold(&data, bs(block_size), MODEL);
        let pos = pos % bytes.len();
        let damaged = apply(&bytes, &Damage::Flip(pos, x));
        let parsed = Container::parse(&damaged);
        if pos < 8 {
            prop_assert!(parsed.is_err());
            return Ok(());
        }
        let original = Container::parse(&bytes).unwrap();
        let payload_start = HEADER_LEN + TABLE_ENTRY_LEN * original.blocks().len();
        if pos < payload_start {
            return Ok(());
        }
        // The damage is in the stored bytes: the layout is the same.
        let parsed = parsed.unwrap();
        prop_assert_eq!(parsed.header(), original.header());
        let block = original
            .blocks()
            .iter()
            .find(|b| {
                let start = b.stored().as_ptr() as usize - bytes.as_ptr() as usize;
                (start..start + b.stored().len()).contains(&pos)
            })
            .unwrap();
        let e = unfold(&damaged, MODEL).unwrap_err();
        if block.kind() == BlockKind::Raw {
            prop_assert!(
                matches!(e, UnfoldError::Format(FormatError::ChecksumMismatch { .. })),
                "{:?}", e
            );
        }
    }

    /// Random bytes, with or without a valid start, never panic the parser.
    #[test]
    fn random_bytes_never_panic(
        prefix in any::<bool>(),
        body in prop::collection::vec(any::<u8>(), 0..400),
    ) {
        let mut bytes = if prefix { b"TACO\x01\x00\x00\x00".to_vec() } else { Vec::new() };
        bytes.extend_from_slice(&body);
        if let Ok(c) = Container::parse(&bytes) {
            // A container that parses obeys the layout rules.
            let stored: usize = c.blocks().iter().map(|b| b.stored().len()).sum();
            prop_assert_eq!(bytes.len(), HEADER_LEN + TABLE_ENTRY_LEN * c.blocks().len() + stored);
        }
    }
}

#[test]
fn every_single_flip_of_a_small_container_is_caught() {
    let data = b"aaaaaaaaaaaabcdefgh-zzzzzzzzzzzz".repeat(3);
    for model in [MODEL, ModelHash::RAW_ONLY] {
        let bytes = fold(&data, bs(16), model);
        for pos in 0..bytes.len() {
            for x in [0x01, 0x80, 0xFF] {
                let damaged = apply(&bytes, &Damage::Flip(pos, x));
                assert!(unfold(&damaged, MODEL).is_err(), "flip {x:#04x} at {pos} was not caught");
            }
        }
    }
}
