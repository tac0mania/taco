//! The integer helpers and the i8 matrix product on the GPU, compared bit for bit with
//! `hardshell::int` and `hardshell::kernels` on random shapes and extreme values.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap, clippy::print_stderr)]

mod support;

use hardshell::int::{self, Divisor};
use hardshell::kernels::matvec_i8;
use hardshell::tensor::Matrix;
use hardshell_gpu::{GpuOps, IntOp, MatKernel, SHADER_SOURCES};
use support::{Rng, device};

fn w64(x: u64) -> [u32; 2] {
    [x as u32, (x >> 32) as u32]
}

fn w128(x: u128) -> [u32; 4] {
    [x as u32, (x >> 32) as u32, (x >> 64) as u32, (x >> 96) as u32]
}

fn case(parts: &[u32]) -> [u32; 8] {
    let mut c = [0u32; 8];
    c[..parts.len()].copy_from_slice(parts);
    c
}

fn signed_i64(r: &mut Rng) -> i64 {
    let v = r.bits(63) as i64;
    if r.below(2) == 0 { v } else { -v }
}

fn signed_i128(r: &mut Rng, max_bits: u32) -> i128 {
    // A magnitude below 2^max_bits (max_bits <= 127), with a random sign.
    let raw = (u128::from(r.next()) << 64 | u128::from(r.next())) >> (128 - max_bits);
    let v = (raw >> r.below(u64::from(max_bits))) as i128;
    if r.below(2) == 0 { v } else { -v }
}

/// Runs `op` on all cases and compares with `expect`.
fn check(ops: &GpuOps<'_>, op: IntOp, cases: &[[u32; 8]], expect: impl Fn(&[u32; 8]) -> [u32; 4]) {
    eprintln!("  {op:?}: {} cases", cases.len());
    let got = ops.int_op(op, cases).unwrap();
    assert_eq!(got.len(), cases.len());
    for (c, g) in cases.iter().zip(&got) {
        assert_eq!(*g, expect(c), "{op:?} on {c:x?}");
    }
}

fn i64_of(c: &[u32; 8], i: usize) -> i64 {
    (u64::from(c[i]) | (u64::from(c[i + 1]) << 32)) as i64
}

fn u128_of(c: &[u32; 8]) -> u128 {
    (0..4).fold(0u128, |a, i| a | (u128::from(c[i]) << (32 * i)))
}

#[test]
fn int_helpers_equal_hardshell_int() {
    let Some(dev) = device("int_helpers_equal_hardshell_int") else { return };
    let ops = GpuOps::new(&dev).unwrap();
    let mut r = Rng::new(0x7ac0_9a11);
    let n = 4096;

    // rescale: extremes, then random values.
    let mut cases = Vec::new();
    for x in [0i64, 1, -1, 3, -3, -5, i64::MIN, i64::MAX, 1 << 40, -(1 << 40)] {
        for m in [0i32, 1, -1, i32::MIN, i32::MAX, 3] {
            for s in [0u32, 1, 2, 31, 32, 63, 64, 94, 95, 96, 200, 255] {
                let [a, b] = w64(x as u64);
                cases.push(case(&[a, b, m as u32, s]));
            }
        }
    }
    for _ in 0..n {
        let [a, b] = w64(signed_i64(&mut r) as u64);
        let m = (r.bits(32) as u32) as i32;
        cases.push(case(&[a, b, m as u32, r.below(100) as u32]));
    }
    check(&ops, IntOp::Rescale, &cases, |c| {
        w128(int::rescale(i64_of(c, 0), c[2] as i32, c[3] as u8) as u128)
    });

    // floor_div by 1..=2^16.
    let mut cases = Vec::new();
    for a in [0i64, 1, -1, 7, -7, i64::MIN, i64::MAX, -(1 << 47), (1 << 47) - 1] {
        for d in [1u32, 2, 3, 768, 65_535, 65_536] {
            let [x, y] = w64(a as u64);
            cases.push(case(&[x, y, d]));
        }
    }
    for _ in 0..n {
        let [x, y] = w64(signed_i64(&mut r) as u64);
        cases.push(case(&[x, y, 1 + r.below(65_536) as u32]));
    }
    check(&ops, IntOp::FloorDivSmall, &cases, |c| {
        let q = int::floor_div(i64_of(c, 0), Divisor::new(i64::from(c[2])).unwrap());
        let [a, b] = w64(q as u64);
        [a, b, 0, 0]
    });

    // isqrt over the full u128 range, and around squares.
    let mut cases = Vec::new();
    for x in [0u128, 1, 2, 3, 4, 15, 16, u128::MAX, 1 << 126, (1 << 126) - 1, 1 << 97] {
        cases.push(case(&w128(x)));
    }
    for _ in 0..n {
        let s = u128::from(r.bits(64));
        let x = match r.below(3) {
            0 => s * s,
            1 => (s * s).wrapping_sub(1),
            _ => u128::from(r.next()) << 64 | u128::from(r.next()) >> r.below(64),
        };
        cases.push(case(&w128(x)));
    }
    check(&ops, IntOp::Isqrt, &cases, |c| w128(int::isqrt_u128(u128_of(c))));

    // u64 / u64.
    let mut cases = Vec::new();
    for (a, b) in [(0u64, 1u64), (u64::MAX, 1), (u64::MAX, u64::MAX), (1 << 62, 3), (5, 7)] {
        let [x, y] = w64(a);
        let [z, w] = w64(b);
        cases.push(case(&[x, y, z, w]));
    }
    for _ in 0..n {
        let [x, y] = w64(r.bits(64));
        let [z, w] = w64(r.bits(64).max(1));
        cases.push(case(&[x, y, z, w]));
    }
    check(&ops, IntOp::DivU64, &cases, |c| {
        let q = (i64_of(c, 0) as u64) / (i64_of(c, 2) as u64);
        let [a, b] = w64(q);
        [a, b, 0, 0]
    });

    // clamp_wide.
    let mut cases = Vec::new();
    for _ in 0..n {
        let x = signed_i128(&mut r, 127);
        let lo = signed_i64(&mut r);
        let hi =
            if r.below(8) == 0 { signed_i64(&mut r) } else { lo.saturating_add(r.bits(62) as i64) };
        let mut c = [0u32; 8];
        c[..4].copy_from_slice(&w128(x as u128));
        c[4..6].copy_from_slice(&w64(lo as u64));
        c[6..8].copy_from_slice(&w64(hi as u64));
        cases.push(c);
    }
    check(&ops, IntOp::ClampWide, &cases, |c| {
        let x = u128_of(c) as i128;
        let v = int::clamp_wide(x, i64_of(c, 4), i64_of(c, 6));
        let [a, b] = w64(v as u64);
        [a, b, 0, 0]
    });

    // u64 * u64.
    let mut cases = Vec::new();
    for _ in 0..n {
        let [x, y] = w64(r.bits(64));
        let [z, w] = w64(r.bits(64));
        cases.push(case(&[x, y, z, w]));
    }
    cases.push(case(&[u32::MAX, u32::MAX, u32::MAX, u32::MAX]));
    check(&ops, IntOp::MulU64, &cases, |c| {
        w128(u128::from(i64_of(c, 0) as u64) * u128::from(i64_of(c, 2) as u64))
    });

    // The rounding of rescale_wide, |p| < 2^126.
    let mut cases = Vec::new();
    for _ in 0..n {
        let p = signed_i128(&mut r, 126);
        let mut c = case(&w128(p as u128));
        c[4] = r.below(140) as u32;
        cases.push(c);
    }
    for p in [-(1i128 << 125), (1i128 << 125) - 1, -1, 1, 0] {
        for s in [0u32, 1, 64, 125, 126, 127, 200] {
            let mut c = case(&w128(p as u128));
            c[4] = s;
            cases.push(c);
        }
    }
    check(&ops, IntOp::ShiftRound, &cases, |c| {
        w128(int::rescale_wide(u128_of(c) as i128, 1, c[4] as u8) as u128)
    });

    // u128 / n for x < 2^96.
    let mut cases = Vec::new();
    for _ in 0..n {
        let x = (u128::from(r.next()) << 32 | u128::from(r.next() as u32)) >> r.below(96);
        let mut c = case(&w128(x));
        c[4] = 1 + r.below(65_536) as u32;
        cases.push(c);
    }
    check(&ops, IntOp::DivSmallU128, &cases, |c| w128(u128_of(c) / u128::from(c[4])));

    // The packed dot product, with every extreme byte.
    let mut cases = Vec::new();
    for _ in 0..n {
        cases.push(case(&[r.next() as u32, r.next() as u32]));
    }
    cases.push(case(&[0x8080_8080, 0x8080_8080]));
    cases.push(case(&[0x7f7f_7f7f, 0x8080_8080]));
    check(&ops, IntOp::Dot4, &cases, |c| {
        let a = c[0].to_le_bytes();
        let b = c[1].to_le_bytes();
        let s: i32 = (0..4).map(|i| i32::from(a[i] as i8) * i32::from(b[i] as i8)).sum();
        [s as u32, 0, 0, 0]
    });
}

fn cpu_matmul(w: &Matrix<i8>, a: &[i8], m: usize) -> Vec<i32> {
    let mut out = vec![0i32; m * w.rows()];
    for (row, o) in a.chunks(w.cols()).zip(out.chunks_mut(w.rows())) {
        matvec_i8(w, row, o).unwrap();
    }
    out
}

#[test]
fn matmul_equals_hardshell_kernels() {
    let Some(dev) = device("matmul_equals_hardshell_kernels") else { return };
    let ops = GpuOps::new(&dev).unwrap();
    let mut r = Rng::new(0x07ac_03a7);
    let shapes = [
        (1usize, 1usize, 1usize),
        (5, 3, 2),
        (64, 64, 1),
        (70, 97, 9),
        (300, 768, 3),
        (129, 200, 64),
        (257, 1000, 70),
        (31, 3072, 8),
        (2, 4100, 130),
    ];
    for (n, k, m) in shapes {
        let w = Matrix::new(n, k, (0..n * k).map(|_| r.i8()).collect()).unwrap();
        let a: Vec<i8> = (0..m * k).map(|_| r.i8()).collect();
        let want = cpu_matmul(&w, &a, m);
        for kernel in [MatKernel::Matvec, MatKernel::Gemm] {
            let got = ops.matmul_i8(&w, &a, m, Some(kernel)).unwrap();
            assert_eq!(got, want, "shape n {n} k {k} m {m}, {kernel:?}");
        }
    }
}

#[test]
fn matmul_is_exact_at_the_accumulator_bound() {
    let Some(dev) = device("matmul_is_exact_at_the_accumulator_bound") else { return };
    let ops = GpuOps::new(&dev).unwrap();
    // All products +2^14 or -2^14 * 127 / 128, at the longest dot product: sums of +-2^30.
    let k = hardshell::kernels::MAX_DOT_LEN_I32;
    for (wv, av) in [(-128i8, -128i8), (127, -128), (-128, 127), (127, 127)] {
        let w = Matrix::new(3, k, vec![wv; 3 * k]).unwrap();
        let a = vec![av; 2 * k];
        let want = cpu_matmul(&w, &a, 2);
        for kernel in [MatKernel::Matvec, MatKernel::Gemm] {
            assert_eq!(
                ops.matmul_i8(&w, &a, 2, Some(kernel)).unwrap(),
                want,
                "{wv} {av} {kernel:?}"
            );
        }
    }
}

#[test]
fn shaders_use_no_floats() {
    // PLAN(§3 rule 1): no floating-point type or literal in any shader of the prediction path.
    let float_types = ["f16", "f32", "f64", "vec2f", "vec3f", "vec4f", "vec2h", "vec3h", "vec4h"];
    for (name, src) in SHADER_SOURCES {
        for (n, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for word in code.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
                assert!(
                    !float_types.contains(&word)
                        && !(word.starts_with("mat")
                            && word[3..].starts_with(|c: char| c.is_ascii_digit())),
                    "{name}:{} uses a float type: {word}",
                    n + 1
                );
            }
            let b = code.as_bytes();
            for i in 1..b.len().saturating_sub(1) {
                assert!(
                    !(b[i] == b'.' && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit()),
                    "{name}:{} has a float literal",
                    n + 1
                );
            }
        }
    }
}
