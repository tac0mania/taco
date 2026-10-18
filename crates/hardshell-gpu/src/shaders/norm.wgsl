// The `norm` op in mode `layer`: hardshell::layernorm::LayerNorm::apply (plan §7.3).
//
// 1. mean = floor_div(sum(x), n).
// 2. var = floor(sum((x - mean)^2) / n).
// 3. inv = floor(2^K / isqrt(var * 2^(2J) + eps)).
// 4. y = clamp(rescale_wide((x - mean) * inv * gamma, m, s) + beta, -127, 127), packed i8.
//
// One workgroup for each row. The sums are exact integer sums (the bounds are in
// hardshell::layernorm), so the order of the tree sum does not change them.

struct NormParams {
    n: u32,
    x_stride: u32,
    out_kw: u32,
    k: u32,
    j: u32,
    eps_lo: u32,
    eps_hi: u32,
    m: i32,
    s: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<i32>;
@group(0) @binding(1) var<storage, read> gamma: array<i32>;
@group(0) @binding(2) var<storage, read> beta: array<i32>;
@group(0) @binding(3) var<uniform> np: NormParams;
@group(0) @binding(4) var<storage, read> rows: Rows;
@group(0) @binding(5) var<storage, read_write> nout: array<u32>;

const NT: u32 = 256u;
var<workgroup> red64: array<vec2<u32>, 256>;
var<workgroup> red128: array<vec4<u32>, 256>;
var<workgroup> shared_mean: vec2<u32>;
var<workgroup> shared_inv: vec2<u32>;

fn one(row: u32, c: u32, mean: vec2<u32>, inv: vec2<u32>) -> u32 {
    let centred = u64_sub(i64_from_i32(x[row * np.x_stride + c]), mean);
    let g = gamma[c];
    let mag_g = select(bitcast<u32>(g), 0u - bitcast<u32>(g), g < 0);
    let mag_m = select(bitcast<u32>(np.m), 0u - bitcast<u32>(np.m), np.m < 0);
    // INVARIANT: |centred| < 2^32, inv <= 2^48, |g| <= 2^15 and |m| <= 2^31, so the product of
    // the magnitudes is below 2^126: it is exact, as the product of hardshell::layernorm is.
    var p = u128_mul64(i64_abs(centred), inv);
    p = u128_mul32(p, mag_g);
    p = u128_mul32(p, mag_m);
    if (i64_is_neg(centred) != ((g < 0) != (np.m < 0))) { p = u128_neg(p); }
    let r = u128_add(shift_round(p, np.s, 126u), i128_from_i32(beta[c]));
    return bitcast<u32>(clamp_act_i8(r)) & 0xffu;
}

// Grid: (rows, 1).
@compute @workgroup_size(256)
fn norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = wg.x;
    let live = row < rows.count;
    let base = row * np.x_stride;
    // Step 1: the sum.
    var s = vec2<u32>(0u, 0u);
    if (live) {
        for (var c = t; c < np.n; c = c + NT) { s = u64_add(s, i64_from_i32(x[base + c])); }
    }
    red64[t] = s;
    workgroupBarrier();
    for (var h = NT / 2u; h > 0u; h = h / 2u) {
        if (t < h) { red64[t] = u64_add(red64[t], red64[t + h]); }
        workgroupBarrier();
    }
    if (t == 0u) { shared_mean = i64_floor_div_small(red64[0], np.n); }
    workgroupBarrier();
    let mean = shared_mean;
    // Step 2: the sum of squares.
    var q = vec4<u32>(0u, 0u, 0u, 0u);
    if (live) {
        for (var c = t; c < np.n; c = c + NT) {
            let d = i64_abs(u64_sub(i64_from_i32(x[base + c]), mean));
            q = u128_add(q, u128_mul64(d, d));
        }
    }
    red128[t] = q;
    workgroupBarrier();
    for (var h = NT / 2u; h > 0u; h = h / 2u) {
        if (t < h) { red128[t] = u128_add(red128[t], red128[t + h]); }
        workgroupBarrier();
    }
    // Step 3.
    if (t == 0u) {
        let v = u128_div_small(red128[0], np.n);
        let arg = u128_add(u128_shl(v, 2u * np.j), vec4<u32>(np.eps_lo, np.eps_hi, 0u, 0u));
        var root = isqrt_u128(arg);
        // INVARIANT: eps >= 1, so root >= 1; root < 2^49 fits in a u64.
        if (root.x == 0u && root.y == 0u) { root = vec4<u32>(1u, 0u, 0u, 0u); }
        shared_inv = u64_div(u64_shl(vec2<u32>(1u, 0u), np.k), root.xy);
    }
    workgroupBarrier();
    let inv = shared_inv;
    // Step 4.
    if (live) {
        for (var w = t; w * 4u < np.n; w = w + NT) {
            var word = 0u;
            for (var k = 0u; k < 4u; k = k + 1u) {
                let c = w * 4u + k;
                if (c < np.n) { word = word | (one(row, c, mean, inv) << (8u * k)); }
            }
            nout[row * np.out_kw + w] = word;
        }
    }
}
