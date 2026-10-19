//! A machine with the CUDA driver but no visible device (`CUDA_VISIBLE_DEVICES=""`) gives
//! `Error::NoDevice`, so the caller (and the tests, `support::device`) can use another backend.
//!
//! The variable must be set before the driver starts (`cuInit`), so the test runs the check in
//! a child process of this test binary with the variable set.

#![allow(clippy::unwrap_used, clippy::panic, clippy::print_stderr)]

use hardshell_cuda::{CudaDevice, Error};

/// The variable that tells the child process to run the check.
const CHILD: &str = "HARDSHELL_CUDA_NO_DEVICE_CHILD";

#[test]
fn hidden_devices_give_no_device() {
    if std::env::var_os(CHILD).is_some() {
        // The child: no visible device, with or without a driver.
        match CudaDevice::open(0) {
            Err(Error::NoDevice { reason }) => eprintln!("no device, as expected: {reason}"),
            Err(e) => panic!("another error than NoDevice: {e}"),
            Ok(d) => panic!("a device opened with CUDA_VISIBLE_DEVICES empty: {}", d.name()),
        }
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args(["--exact", "hidden_devices_give_no_device", "--nocapture"])
        .env(CHILD, "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stderr);
    eprintln!("{text}");
    assert!(out.status.success(), "the child failed: {text}");
    assert!(text.contains("no device, as expected"), "{text}");
}
