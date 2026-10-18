//! Helpers of the CUDA tests: the device and a small random generator.
#![allow(dead_code, clippy::print_stderr, clippy::panic, clippy::cast_possible_truncation)]

use hardshell_cuda::{CudaDevice, Error};

/// Opens CUDA device 0. Gives `None`, with a message, if there is no driver or no device (the
/// Mac): the test then skips.
pub(crate) fn device(test: &str) -> Option<CudaDevice> {
    match CudaDevice::open(0) {
        Ok(d) => {
            eprintln!("{test}: CUDA device {} (compute {:?})", d.name(), d.compute_capability());
            Some(d)
        }
        Err(Error::NoDevice { reason }) => {
            eprintln!("{test}: SKIPPED, no CUDA device ({reason})");
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
}
