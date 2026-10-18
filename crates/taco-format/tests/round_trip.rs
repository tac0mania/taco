//! Round trips: every container that the writer makes parses again with the same fields and
//! blocks, and the toy fold and unfold restore the data. Includes the edge layouts (no block,
//! one block, a short last block) and the worst-case growth (plan §9).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    reason = "test code: a failure must stop the test with a message"
)]

mod common;

use common::{MODEL, bs, fold, splitmix64, unfold};
use proptest::prelude::*;
use taco_format::{
    Block, BlockKind, BlockSize, Checksum, Container, ContainerParams, HEADER_LEN, ModelHash,
    TABLE_ENTRY_LEN, worst_case_len,
};

/// Data with runs, so that the toy coder makes some blocks smaller and some larger.
fn runs() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec((any::<u8>(), 1usize..40, any::<bool>()), 0..80).prop_map(|parts| {
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

/// Checks that `bytes` parses into the header and blocks that `build` received.
fn check_layout(bytes: &[u8], data: &[u8], params: &ContainerParams, blocks: &[Block]) {
    let c = Container::parse(bytes).unwrap();
    let h = c.header();
    assert_eq!(h.model_hash(), params.model_hash);
    assert_eq!(h.block_size(), params.block_size);
    assert_eq!(h.original_len(), data.len() as u64);
    assert_eq!(h.checksum(), Checksum::of(data));
    assert_eq!(c.blocks().len(), blocks.len());
    assert_eq!(h.block_count() as usize, blocks.len());

    let mut stored_sum = 0;
    let pieces = params.block_size.split(data);
    for (i, ((parsed, given), piece)) in c.blocks().iter().zip(blocks).zip(pieces).enumerate() {
        assert_eq!(parsed.index() as usize, i);
        assert_eq!(parsed.kind(), given.kind());
        assert_eq!(parsed.original_len() as usize, piece.len());
        assert_eq!(parsed.original_offset(), i as u64 * u64::from(params.block_size.get()));
        match given {
            Block::Raw => assert_eq!(parsed.stored(), piece),
            Block::Coded(coded) => assert_eq!(parsed.stored(), coded.as_slice()),
            _ => unreachable!("version 1 has two kinds"),
        }
        stored_sum += parsed.stored().len();
    }
    assert_eq!(bytes.len(), HEADER_LEN + TABLE_ENTRY_LEN * blocks.len() + stored_sum);
    let worst = worst_case_len(data.len() as u64, params.block_size).unwrap();
    assert!(bytes.len() as u64 <= worst);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Random layouts: random data, block size and kind of each block. A coded block has
    /// random bytes, shorter than its original bytes.
    #[test]
    fn any_layout_round_trips(
        data in prop::collection::vec(any::<u8>(), 0..2_000),
        block_size in 1u32..300,
        raw_only in any::<bool>(),
        seed in any::<u64>(),
    ) {
        let block_size = BlockSize::new(block_size).unwrap();
        let model = if raw_only { ModelHash::RAW_ONLY } else { MODEL };
        let mut s = seed;
        let blocks: Vec<Block> = block_size
            .split(&data)
            .map(|piece| {
                let x = splitmix64(&mut s);
                if raw_only || x.is_multiple_of(3) {
                    Block::Raw
                } else {
                    let len = usize::try_from(x >> 8).unwrap() % piece.len();
                    Block::Coded((0..len).map(|_| splitmix64(&mut s) as u8).collect())
                }
            })
            .collect();
        let params = ContainerParams::new(model, block_size);
        let bytes = Container::build(&params, &data, &blocks).unwrap();
        check_layout(&bytes, &data, &params, &blocks);
    }

    /// The toy fold and unfold restore any data, with and without a model.
    #[test]
    fn fold_then_unfold_restores_the_data(
        data in runs(),
        block_size in 1u32..400,
        raw_only in any::<bool>(),
    ) {
        let model = if raw_only { ModelHash::RAW_ONLY } else { MODEL };
        let bytes = fold(&data, bs(block_size), model);
        prop_assert_eq!(unfold(&bytes, MODEL).unwrap(), data);
    }

    /// With every block raw, the container is exactly the worst case: 88 + 5 per block.
    #[test]
    fn all_raw_is_exactly_the_worst_case(
        data in prop::collection::vec(any::<u8>(), 0..3_000),
        block_size in 1u32..5_000,
    ) {
        let block_size = bs(block_size);
        let bytes = fold(&data, block_size, ModelHash::RAW_ONLY);
        let blocks = data.len().div_ceil(block_size.as_usize());
        prop_assert_eq!(bytes.len(), 88 + 5 * blocks + data.len());
        prop_assert_eq!(bytes.len() as u64, worst_case_len(data.len() as u64, block_size).unwrap());
    }

    /// With the raw fallback, no container is larger than the worst case.
    #[test]
    fn fallback_never_exceeds_the_worst_case(
        data in prop::collection::vec(any::<u8>(), 0..3_000),
        block_size in 1u32..500,
    ) {
        let bytes = fold(&data, bs(block_size), MODEL);
        let worst = worst_case_len(data.len() as u64, bs(block_size)).unwrap();
        prop_assert!(bytes.len() as u64 <= worst);
    }
}

fn round_trip(data: &[u8], block_size: u32) -> Vec<u8> {
    let bytes = fold(data, bs(block_size), MODEL);
    assert_eq!(unfold(&bytes, MODEL).unwrap(), data);
    bytes
}

#[test]
fn empty_data_has_no_block() {
    let bytes = round_trip(&[], 64);
    assert_eq!(bytes.len(), 88);
    assert_eq!(Container::parse(&bytes).unwrap().blocks().len(), 0);
}

#[test]
fn one_byte_is_one_raw_block() {
    let bytes = round_trip(b"x", 64);
    assert_eq!(bytes.len(), 88 + 5 + 1);
    let c = Container::parse(&bytes).unwrap();
    assert_eq!(c.blocks()[0].kind(), BlockKind::Raw);
}

#[test]
fn one_block_holds_data_smaller_than_the_block_size() {
    let data = vec![7u8; 1000];
    let bytes = round_trip(&data, u32::MAX);
    let c = Container::parse(&bytes).unwrap();
    assert_eq!(c.blocks().len(), 1);
    assert_eq!(c.blocks()[0].kind(), BlockKind::Coded);
    assert_eq!(c.blocks()[0].original_len(), 1000);
}

#[test]
fn last_block_is_shorter() {
    let data = vec![3u8; 1000];
    let bytes = round_trip(&data, 300);
    let c = Container::parse(&bytes).unwrap();
    let lens: Vec<u32> = c.blocks().iter().map(|b| b.original_len()).collect();
    assert_eq!(lens, [300, 300, 300, 100]);
}

#[test]
fn exact_multiple_has_no_short_block() {
    let data = vec![3u8; 900];
    let bytes = round_trip(&data, 300);
    let c = Container::parse(&bytes).unwrap();
    assert_eq!(c.blocks().len(), 3);
    assert!(c.blocks().iter().all(|b| b.original_len() == 300));
}

#[test]
fn block_size_one_stores_each_byte_raw() {
    let data = b"abc";
    let bytes = round_trip(data, 1);
    assert_eq!(bytes.len(), 88 + 3 * 5 + 3);
}

#[test]
fn mixed_blocks_round_trip() {
    let mut data = vec![0u8; 500];
    let mut s = 1;
    data.extend((0..500).map(|_| splitmix64(&mut s) as u8));
    data.extend(vec![9u8; 500]);
    let bytes = round_trip(&data, 250);
    let c = Container::parse(&bytes).unwrap();
    let kinds: Vec<BlockKind> = c.blocks().iter().map(|b| b.kind()).collect();
    assert_eq!(
        kinds,
        [
            BlockKind::Coded,
            BlockKind::Coded,
            BlockKind::Raw,
            BlockKind::Raw,
            BlockKind::Coded,
            BlockKind::Coded
        ]
    );
}
