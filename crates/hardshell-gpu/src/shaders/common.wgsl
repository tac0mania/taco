// Shared declarations of the model kernels.

// The rows of one forward pass. Row `r` runs the token `info[r].y` of the slot `info[r].x` at
// the position `info[r].z`.
struct Rows {
    count: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    info: array<vec4<u32>>,
}

// The residual limit R = 2^30 - 1 (hardshell::gpt2::RESIDUAL_LIMIT).
const RES: i32 = 1073741823i;
