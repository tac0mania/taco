// The output part of the `linear` op, after the matrix product (matmul.wgsl):
// y[r] = clamp(rescale(acc[r] + b[r], m[r], s), lo, hi). The output is packed i8 (linear_i8) or
// i32 (linear_i32).

struct LinParams {
    n: u32,
    acc_stride: u32,
    // Words (linear_i8) or values (linear_i32) in each row of the output.
    out_stride: u32,
    shift: u32,
    lo: i32,
    hi: i32,
    has_bias: u32,
    _p0: u32,
}

@group(0) @binding(0) var<storage, read> acc: array<i32>;
@group(0) @binding(1) var<storage, read> bias: array<i32>;
@group(0) @binding(2) var<storage, read> mult: array<i32>;
@group(0) @binding(3) var<uniform> lp: LinParams;
@group(0) @binding(4) var<storage, read> rows: Rows;
@group(0) @binding(5) var<storage, read_write> lout: array<u32>;

fn value(row: u32, r: u32) -> i32 {
    var v = i64_from_i32(acc[row * lp.acc_stride + r]);
    if (lp.has_bias != 0u) { v = u64_add(v, i64_from_i32(bias[r])); }
    return clamp_to_i32(rescale(v, mult[r], lp.shift), lp.lo, lp.hi);
}

// Grid: (ceil(n / 256), rows); each invocation makes one word of four i8 values.
@compute @workgroup_size(64)
fn linear_i8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let row = gid.y;
    if (row >= rows.count || w * 4u >= lp.n) { return; }
    var word = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let r = w * 4u + k;
        if (r < lp.n) { word = word | ((bitcast<u32>(value(row, r)) & 0xffu) << (8u * k)); }
    }
    lout[row * lp.out_stride + w] = word;
}

// Grid: (ceil(n / 64), rows).
@compute @workgroup_size(64)
fn linear_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let row = gid.y;
    if (row >= rows.count || r >= lp.n) { return; }
    lout[row * lp.out_stride + r] = bitcast<u32>(value(row, r));
}
