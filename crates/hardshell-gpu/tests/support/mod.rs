//! Helpers of the GPU tests: the device choice and a small random generator.
#![allow(dead_code, clippy::print_stderr, clippy::panic, clippy::cast_possible_truncation)]

use std::sync::OnceLock;

use hardshell_gpu::{AdapterChoice, Error, GpuDevice};

/// Opens the device of the tests. `HARDSHELL_GPU_ADAPTER` selects an adapter by name or backend
/// (for example `llvmpipe`, `D3D12`, `Metal`); without it, the best hardware adapter. Gives
/// `None`, with a message, if there is no adapter: the test then skips.
///
/// NOTE: the tests share one device for the whole process, and never drop it. The Mesa D3D12
/// OpenGL driver of WSL2 (measured, 2026-10-18) crashes when a device drops on a test thread.
pub(crate) fn device(test: &str) -> Option<GpuDevice> {
    static DEVICE: OnceLock<Option<GpuDevice>> = OnceLock::new();
    let dev = DEVICE.get_or_init(|| open(test)).clone();
    if let Some(d) = &dev {
        eprintln!("{test}: GPU adapter {}", d.adapter());
    }
    dev
}

fn open(test: &str) -> Option<GpuDevice> {
    let choice = match std::env::var("HARDSHELL_GPU_ADAPTER") {
        Ok(name) if !name.is_empty() => AdapterChoice::NameContains(name),
        _ => AdapterChoice::BestHardware,
    };
    match GpuDevice::open(&choice) {
        Ok(d) => Some(d),
        Err(Error::NoAdapter { .. }) => {
            eprintln!("{test}: SKIPPED, no GPU adapter found ({choice:?})");
            None
        }
        Err(e) => panic!("{test}: the device did not open: {e}"),
    }
}

/// splitmix64.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub(crate) fn i8(&mut self) -> i8 {
        (self.next() as u8).cast_signed()
    }

    /// A value with a random number of significant bits, so that small and large values occur.
    pub(crate) fn bits(&mut self, max_bits: u32) -> u64 {
        let b = self.below(u64::from(max_bits) + 1) as u32;
        if b == 0 { 0 } else { self.next() >> (64 - b) }
    }
}
