//! Lists the GPU adapters that wgpu finds on this machine.
#![allow(clippy::print_stdout)]
fn main() {
    let r = hardshell_gpu::adapters();
    println!("{} adapters", r.len());
    for a in r {
        println!("{a}");
    }
}
