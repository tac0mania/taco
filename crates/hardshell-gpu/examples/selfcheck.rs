//! Runs each integer helper and the matrix product on the selected adapter, and prints the
//! progress. Use it to find a driver that fails. `HARDSHELL_GPU_ADAPTER` selects the adapter.
#![allow(clippy::print_stdout, clippy::unwrap_used, clippy::cast_possible_truncation)]
use hardshell_gpu::{AdapterChoice, GpuDevice, GpuOps, IntOp, MatKernel};

fn main() {
    let choice = match std::env::var("HARDSHELL_GPU_ADAPTER") {
        Ok(n) if !n.is_empty() => AdapterChoice::NameContains(n),
        _ => AdapterChoice::BestHardware,
    };
    let dev = GpuDevice::open(&choice).unwrap();
    println!("adapter {}", dev.adapter());
    let ops = GpuOps::new(&dev).unwrap();
    println!("compiled");
    for op in [
        IntOp::Rescale,
        IntOp::FloorDivSmall,
        IntOp::Isqrt,
        IntOp::DivU64,
        IntOp::ClampWide,
        IntOp::MulU64,
        IntOp::ShiftRound,
        IntOp::DivSmallU128,
        IntOp::Dot4,
    ] {
        for n in [4usize, 64, 65, 1000, 5000] {
            let cases: Vec<[u32; 8]> = (0..n as u32)
                .map(|i| [i.wrapping_mul(2654435761), i, 5 + i, 1, 7, 0, 9, 0])
                .collect();
            let r = ops.int_op(op, &cases).unwrap();
            println!("{op:?} n {n} {:?}", r[n - 1]);
        }
    }
    let w = hardshell::tensor::Matrix::new(2, 4, vec![1i8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    for k in [MatKernel::Matvec, MatKernel::Gemm] {
        println!("{k:?} {:?}", ops.matmul_i8(&w, &[1, 1, 1, 1], 1, Some(k)).unwrap());
    }
}
