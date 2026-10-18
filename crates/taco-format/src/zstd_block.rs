//! The zstd block kind (ADR 0011): the encoder of a zstd block and the bounded decoder.
//!
//! PLAN(§9): a zstd block stores exactly one zstd frame (RFC 8878) of the original bytes of
//! the block. The frame needs no model, so this crate decodes it. The frame header must give
//! the content size, and the content size must be the original length of the block. The
//! decoder never allocates from a size in the frame. It writes the output in chunks of at most
//! [`DECODE_CHUNK`] bytes and stops when the output is longer than the original length.
//!
//! DETERMINISM: a conforming zstd decoder gives the same bytes on every machine. The encoder
//! output depends on the version of the zstd library and on [`ZSTD_LEVEL`]. Cargo.toml pins
//! the library, and a test checks the linked version ([`ZSTD_LIBRARY_VERSION`]).

use zstd_safe::{CCtx, CParameter, DCtx, DParameter, InBuffer, OutBuffer};

use crate::error::{FormatError, Result};

/// The zstd compression level of the encoder (ADR 0011 §6). On blocks of 16 KiB, level 19 was
/// 3% to 7% smaller than level 9 on `kennedy.xls`, `HDFS_2k.log` and `alice29.txt`, and about
/// 10 times slower (measured, zstd 1.5.7 CLI, Mac). Levels 20 to 22 gave the same size on
/// blocks of 8 and 16 KiB of `kennedy.xls` (measured) and need a window of up to 128 MiB on
/// large blocks.
pub const ZSTD_LEVEL: i32 = 19;

/// The largest zstd window that the decoder accepts: `2^23` bytes (8 MiB). Level 19 never uses
/// a larger window (zstd 1.5.7 parameter table), so the encoder output always decodes.
pub const ZSTD_WINDOW_LOG_MAX: u32 = 23;

/// The version of the zstd C library that the conformance vectors use: 1.5.7, as
/// `major * 10000 + minor * 100 + patch`.
pub const ZSTD_LIBRARY_VERSION: u32 = 10_507;

/// The largest piece of output of one call to the zstd decoder, in bytes. The decoder copies
/// each piece into the output.
// PERF: one extra copy of each decoded byte. The decoder with the copy gave 58 to 766 MB/s on
// blocks of 8 and 16 KiB (measured, Mac, release build), much faster than a coded block.
const DECODE_CHUNK: usize = 1 << 17;

/// The first 4 bytes of a zstd frame (RFC 8878 §3.1.1), little-endian `0xFD2FB528`.
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// The version of the zstd library that this crate uses, as
/// `major * 10000 + minor * 100 + patch`. Compare it with [`ZSTD_LIBRARY_VERSION`].
#[must_use]
pub fn zstd_library_version() -> u32 {
    zstd_safe::version_number()
}

/// A [`FormatError::Zstd`] with the name of a zstd error code.
fn library_error(code: usize) -> FormatError {
    FormatError::Zstd { reason: zstd_safe::get_error_name(code) }
}

/// Compresses the original bytes of one block into one zstd frame, at [`ZSTD_LEVEL`], with the
/// content size in the frame header, no frame checksum and no dictionary id.
///
/// The result can be larger than `original`. [`crate::Block::choose_smallest`] then stores the
/// block raw. Cost: one zstd compression of `original` (1.5 to 5.4 MB/s on one thread at level
/// 19, measured on the Mac, release build) and one allocation of
/// `zstd_safe::compress_bound(original.len())` bytes.
///
/// # Errors
///
/// [`FormatError::Zstd`] if the zstd library fails, for example with no memory.
pub fn compress_zstd(original: &[u8]) -> Result<Vec<u8>> {
    let mut cctx =
        CCtx::try_create().ok_or(FormatError::Zstd { reason: "no memory for a zstd context" })?;
    for param in [
        CParameter::CompressionLevel(ZSTD_LEVEL),
        CParameter::ContentSizeFlag(true),
        CParameter::ChecksumFlag(false),
        CParameter::DictIdFlag(false),
    ] {
        cctx.set_parameter(param).map_err(library_error)?;
    }
    let mut out = vec![0u8; zstd_safe::compress_bound(original.len())];
    let len = cctx.compress2(out.as_mut_slice(), original).map_err(library_error)?;
    out.truncate(len);
    Ok(out)
}

/// Checks the frame structure of the stored bytes of zstd block `block` (ADR 0011 §5): one
/// zstd frame, no byte after it, and a content size equal to `original_len`. It does not
/// decode the frame. Cost: a walk over the block headers of the frame.
///
/// # Errors
///
/// [`FormatError::CorruptZstdBlock`] if a rule is not obeyed.
pub(crate) fn check_frame(stored: &[u8], original_len: u32, block: u32) -> Result<()> {
    let corrupt = |reason| FormatError::CorruptZstdBlock { block, reason };
    if stored.first_chunk::<4>() != Some(&ZSTD_MAGIC) {
        return Err(corrupt("the stored bytes do not start with the zstd frame magic"));
    }
    match zstd_safe::find_frame_compressed_size(stored) {
        Ok(len) if len == stored.len() => {}
        Ok(_) => return Err(corrupt("bytes follow the zstd frame")),
        Err(code) => return Err(corrupt(zstd_safe::get_error_name(code))),
    }
    match zstd_safe::get_frame_content_size(stored) {
        Ok(Some(size)) if size == u64::from(original_len) => Ok(()),
        Ok(Some(_)) => Err(corrupt("the frame content size is not the original length")),
        Ok(None) => Err(corrupt("the frame header has no content size")),
        Err(_) => Err(corrupt("the frame header is not valid")),
    }
}

/// Decodes the stored bytes of zstd block `block` and appends exactly `original_len` bytes to
/// `out`. On an error, `out` keeps its old length.
///
/// The memory is bounded: the zstd window is at most `2^ZSTD_WINDOW_LOG_MAX` bytes, and the
/// output grows only with the bytes that the frame really gives, up to `original_len`.
///
/// Cost: one zstd decode of the block and one copy of the output.
///
/// # Errors
///
/// - [`FormatError::CorruptZstdBlock`] if the frame does not obey ADR 0011 §5, the zstd
///   decoder refuses it, or it gives more or fewer bytes than `original_len`.
/// - [`FormatError::Zstd`] if the zstd library fails, for example with no memory.
pub(crate) fn decode_into(
    stored: &[u8],
    original_len: u32,
    block: u32,
    out: &mut Vec<u8>,
) -> Result<()> {
    let start = out.len();
    let result = decode_append(stored, original_len, block, out);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

/// The body of [`decode_into`]. It can leave a part of the output in `out` on an error.
fn decode_append(stored: &[u8], original_len: u32, block: u32, out: &mut Vec<u8>) -> Result<()> {
    let corrupt = |reason| FormatError::CorruptZstdBlock { block, reason };
    check_frame(stored, original_len, block)?;
    let want = usize::try_from(original_len)
        .map_err(|_| FormatError::LengthOverflow { part: "zstd block" })?;
    let mut dctx =
        DCtx::try_create().ok_or(FormatError::Zstd { reason: "no memory for a zstd context" })?;
    dctx.set_parameter(DParameter::WindowLogMax(ZSTD_WINDOW_LOG_MAX)).map_err(library_error)?;

    // INVARIANT: want >= 1 (a block has at least 1 byte), so the chunk is not empty and each
    // call can make progress.
    let mut chunk = vec![0u8; want.clamp(1, DECODE_CHUNK)];
    let mut input = InBuffer::around(stored);
    let mut produced = 0usize;
    loop {
        let before = input.pos();
        let (hint, len) = {
            let mut output = OutBuffer::around(chunk.as_mut_slice());
            let hint = dctx
                .decompress_stream(&mut output, &mut input)
                .map_err(|code| corrupt(zstd_safe::get_error_name(code)))?;
            (hint, output.pos())
        };
        produced = produced
            .checked_add(len)
            .filter(|&p| p <= want)
            .ok_or(corrupt("the frame gives more bytes than the original length"))?;
        out.extend_from_slice(chunk.get(..len).ok_or(corrupt("the zstd output is too long"))?);
        if hint == 0 {
            break;
        }
        if len == 0 && input.pos() == before {
            return Err(corrupt("the zstd frame is truncated"));
        }
    }
    if produced != want {
        return Err(corrupt("the frame gives fewer bytes than the original length"));
    }
    if input.pos() != stored.len() {
        return Err(corrupt("bytes follow the zstd frame"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(n: usize) -> Vec<u8> {
        b"the quick brown fox jumps over the lazy dog. ".iter().copied().cycle().take(n).collect()
    }

    #[test]
    fn linked_library_is_the_pinned_version() {
        assert_eq!(zstd_library_version(), ZSTD_LIBRARY_VERSION);
    }

    #[test]
    fn compressed_frame_decodes_to_the_original() {
        let data = text(5_000);
        let frame = compress_zstd(&data).unwrap();
        assert!(frame.len() < 200, "{}", frame.len());
        let mut out = b"prefix".to_vec();
        decode_into(&frame, 5_000, 0, &mut out).unwrap();
        assert_eq!(&out[..6], b"prefix");
        assert_eq!(&out[6..], data.as_slice());
    }

    #[test]
    fn decoder_writes_in_chunks_larger_than_one_call() {
        let data = text(3 * DECODE_CHUNK + 17);
        let frame = compress_zstd(&data).unwrap();
        let mut out = Vec::new();
        decode_into(&frame, u32::try_from(data.len()).unwrap(), 0, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn wrong_original_length_is_refused_and_out_is_unchanged() {
        let frame = compress_zstd(&text(100)).unwrap();
        for len in [1, 99, 101, u32::MAX] {
            let mut out = vec![7u8; 3];
            let e = decode_into(&frame, len, 4, &mut out).unwrap_err();
            assert!(matches!(e, FormatError::CorruptZstdBlock { block: 4, .. }), "{e:?}");
            assert_eq!(out, [7, 7, 7]);
        }
    }

    #[test]
    fn frame_rules_are_checked() {
        let frame = compress_zstd(&text(100)).unwrap();
        check_frame(&frame, 100, 0).unwrap();
        let reason = |bytes: &[u8]| match check_frame(bytes, 100, 0) {
            Err(FormatError::CorruptZstdBlock { reason, .. }) => reason,
            other => panic!("{other:?}"),
        };
        assert_eq!(reason(&[]), "the stored bytes do not start with the zstd frame magic");
        assert_eq!(reason(&frame[1..]), "the stored bytes do not start with the zstd frame magic");
        assert_eq!(reason(&[frame.as_slice(), &[0]].concat()), "bytes follow the zstd frame");
        assert_eq!(reason(&[frame.as_slice(), &frame].concat()), "bytes follow the zstd frame");
        assert!(check_frame(&frame[..frame.len() - 1], 100, 0).is_err());
    }
}
