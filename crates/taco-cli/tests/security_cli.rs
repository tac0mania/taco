//! Regression tests for the findings of docs/security.md in the `taco` command. Copied from
//! the fuzz agent (finding F1) and made to pass by the stream unfold (ADR 0018) and the
//! largest block size of the format (ADR 0011, addendum §4).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // Tests fail loudly.

/// A zstd frame of RLE blocks: 4 bytes for each 128 KiB of output (RFC 8878 §3.1.1.2).
#[cfg(target_os = "linux")]
fn rle_frame(content: u32, byte: u8) -> Vec<u8> {
    let mut f = vec![0x28, 0xB5, 0x2F, 0xFD, 0x80, 10 << 3];
    f.extend_from_slice(&content.to_le_bytes());
    let mut left = content;
    while left > 0 {
        let n = left.min(128 * 1024);
        left -= n;
        let header = u32::from(left == 0) | (1 << 1) | (n << 3);
        f.extend_from_slice(&header.to_le_bytes()[..3]);
        f.push(byte);
    }
    f
}

/// A raw-only version 2 container of `blocks` zstd blocks of `block_size` bytes each. The
/// checksum is wrong, so a correct unfold ends with a checksum error.
#[cfg(target_os = "linux")]
fn zstd_bomb(block_size: u32, blocks: u32) -> Vec<u8> {
    let frame = rle_frame(block_size, b'A');
    let mut out = b"TACO\x02\x00\x00\x00".to_vec();
    out.extend_from_slice(&[0; 32]);
    out.extend_from_slice(&(u64::from(block_size) * u64::from(blocks)).to_le_bytes());
    out.extend_from_slice(&[0x11; 32]);
    out.extend_from_slice(&block_size.to_le_bytes());
    out.extend_from_slice(&blocks.to_le_bytes());
    out.extend_from_slice(&[0; 16]);
    for _ in 0..blocks {
        out.push(2);
        out.extend_from_slice(&u32::try_from(frame.len()).unwrap().to_le_bytes());
    }
    for _ in 0..blocks {
        out.extend_from_slice(&frame);
    }
    out
}

/// Runs `taco unfold` on `bomb` with 450 MiB of address space and gives the exit code and
/// stderr. Panics if the command stopped on a signal.
#[cfg(target_os = "linux")]
fn unfold_limited(name: &str, bomb: &[u8]) -> (Option<i32>, String) {
    use std::os::unix::process::ExitStatusExt;
    let d = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&d).unwrap();
    let input = d.join("bomb.taco");
    std::fs::write(&input, bomb).unwrap();
    let output = d.join("bomb.out");
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg("ulimit -v 460800; exec \"$0\" unfold \"$1\" -o \"$2\" -f --threads 4")
        .arg(env!("CARGO_BIN_EXE_taco"))
        .arg(&input)
        .arg(&output)
        .env_remove("TACO_MODEL")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.signal(), None, "the command stopped on a signal: {stderr}");
    // The checksum is wrong: no output file stays (ADR 0018 §3.4).
    assert!(!output.exists());
    (out.status.code(), stderr)
}

/// Finding F1: a container of 10 KB that states 300 MiB in one block. The format refuses the
/// block size (more than 64 MiB) before any allocation.
#[cfg(target_os = "linux")]
#[test]
fn unfold_of_a_small_container_that_states_much_data_does_not_abort() {
    let bomb = zstd_bomb(300 << 20, 1);
    assert!(bomb.len() < 10_000, "{}", bomb.len());
    let (code, stderr) = unfold_limited("security_bomb", &bomb);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stderr.contains("block size"), "{stderr}");
}

/// Finding F1 with valid block sizes: 320 MiB in 5 blocks of 64 MiB, from 10 KB. The stream
/// unfold writes each segment when it is decoded, so the memory does not grow with the
/// original length. The command ends with the checksum error.
#[cfg(target_os = "linux")]
#[test]
fn unfold_of_many_large_zstd_blocks_streams_and_does_not_abort() {
    let bomb = zstd_bomb(64 << 20, 5);
    assert!(bomb.len() < 12_000, "{}", bomb.len());
    let (code, stderr) = unfold_limited("security_bomb_blocks", &bomb);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stderr.contains("checksum"), "{stderr}");
}
