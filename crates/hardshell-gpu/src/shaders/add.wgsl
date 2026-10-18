// The `add` op: x[c] = clamp(x[c] + delta[c], lo, hi), for i32 values.

struct AddParams {
    n: u32,
    x_stride: u32,
    d_stride: u32,
    _p0: u32,
    lo: i32,
    hi: i32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<i32>;
@group(0) @binding(1) var<storage, read> delta: array<i32>;
@group(0) @binding(2) var<uniform> ap: AddParams;
@group(0) @binding(3) var<storage, read> rows: Rows;

// Grid: (ceil(n / 64), rows).
@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let row = gid.y;
    if (row >= rows.count || c >= ap.n) { return; }
    let i = row * ap.x_stride + c;
    let s = u64_add(i64_from_i32(x[i]), i64_from_i32(delta[row * ap.d_stride + c]));
    x[i] = clamp_to_i32(i128_from_i64(s), ap.lo, ap.hi);
}
