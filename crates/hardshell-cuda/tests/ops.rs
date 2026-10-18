//! The i8 matrix product on the GPU (tensor cores and dp4a) against `hardshell::kernels`, and
//! checks of the embedded PTX that run without a GPU.

#![allow(clippy::unwrap_used, clippy::print_stderr)]

mod support;

use hardshell::kernels::matvec_i8;
use hardshell::tensor::Matrix;
use hardshell_cuda::{KERNEL_SOURCE, MatKernel, PTX_SOURCE_LINE, matmul_i8, matmul_i8_split, ptx};
use sha2::{Digest, Sha256};
use support::{Rng, device};

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
    let mut r = Rng::new(0x07ac_0cda);
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
        (2304, 768, 256),
        (50, 768, 1000),
    ];
    for (n, k, m) in shapes {
        let w = Matrix::new(n, k, (0..n * k).map(|_| r.i8()).collect()).unwrap();
        let a: Vec<i8> = (0..m * k).map(|_| r.i8()).collect();
        let want = cpu_matmul(&w, &a, m);
        for kernel in [MatKernel::TensorCore, MatKernel::Dp4a] {
            assert_eq!(
                matmul_i8(&dev, &w, &a, m, kernel).unwrap(),
                want,
                "n {n} k {k} m {m} {kernel:?}"
            );
        }
    }
}

#[test]
fn matmul_is_exact_at_the_accumulator_bound() {
    let Some(dev) = device("matmul_is_exact_at_the_accumulator_bound") else { return };
    let k = hardshell::kernels::MAX_DOT_LEN_I32;
    for (wv, av) in [(-128i8, -128i8), (127, -128), (-128, 127), (127, 127)] {
        let w = Matrix::new(3, k, vec![wv; 3 * k]).unwrap();
        let a = vec![av; 2 * k];
        let want = cpu_matmul(&w, &a, 2);
        for kernel in [MatKernel::TensorCore, MatKernel::Dp4a] {
            assert_eq!(matmul_i8(&dev, &w, &a, 2, kernel).unwrap(), want, "{wv} {av} {kernel:?}");
        }
        let got = matmul_i8_split(&dev, &w, &a, 2, MatKernel::TensorCore, 5).unwrap();
        assert_eq!(got, want, "{wv} {av} split");
    }
}

#[test]
fn embedded_ptx_is_made_from_the_embedded_source() {
    // A stale PTX fails here, also on a machine without NVRTC or a GPU.
    let digest: String =
        Sha256::digest(KERNEL_SOURCE.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    let line = ptx().lines().find(|l| l.starts_with(PTX_SOURCE_LINE)).unwrap();
    assert_eq!(&line[PTX_SOURCE_LINE.len()..], digest, "run tools/cuda/build_ptx.py");
}

#[test]
fn embedded_ptx_has_no_float_instructions() {
    // PLAN(§3 rule 1): no float type in an instruction or a register of the kernels.
    for (n, line) in ptx().lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        for bad in [".f16", ".f32", ".f64", ".bf16", ".tf32", ".e4m3", ".e5m2"] {
            assert!(!code.contains(bad), "PTX line {}: {line}", n + 1);
        }
    }
}
