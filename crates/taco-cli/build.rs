//! Gives the CLI its target triple (`TACO_TARGET`) for `taco --version`, and rebuilds the CLI
//! when the release commit (`TACO_COMMIT`, from `scripts/release/build.sh`) changes.

// NOTE: a build script talks to cargo through stdout.
#![allow(clippy::print_stdout)]

fn main() {
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_owned());
    println!("cargo:rustc-env=TACO_TARGET={target}");
    println!("cargo:rerun-if-env-changed=TACO_COMMIT");
    println!("cargo:rerun-if-changed=build.rs");
}
