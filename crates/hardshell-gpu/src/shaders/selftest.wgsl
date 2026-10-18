// Runs one integer helper of int.wgsl on many inputs, so that a test can compare each result
// with the Rust function in hardshell::int. Each case has 8 input words and 4 output words.

struct TestParams {
    op: u32,
    count: u32,
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<storage, read> tin: array<u32>;
@group(0) @binding(1) var<storage, read_write> tout: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> tp: TestParams;

fn arg64(base: u32, i: u32) -> vec2<u32> { return vec2<u32>(tin[base + 2u * i], tin[base + 2u * i + 1u]); }
fn arg128(base: u32) -> vec4<u32> { return vec4<u32>(tin[base], tin[base + 1u], tin[base + 2u], tin[base + 3u]); }

@compute @workgroup_size(64)
fn selftest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= tp.count) { return; }
    let b = c * 8u;
    var r = vec4<u32>(0u, 0u, 0u, 0u);
    switch tp.op {
        // rescale(x: i64, m: i32, s) -> i128
        case 0u: { r = rescale(arg64(b, 0u), bitcast<i32>(tin[b + 2u]), tin[b + 3u]); }
        // floor_div(a: i64, n) -> i64, 1 <= n <= 2^16
        case 1u: { let q = i64_floor_div_small(arg64(b, 0u), tin[b + 2u]); r = vec4<u32>(q, 0u, 0u); }
        // isqrt_u128(x) -> u128
        case 2u: { r = isqrt_u128(arg128(b)); }
        // u64 / u64 -> u64
        case 3u: { let q = u64_div(arg64(b, 0u), arg64(b, 1u)); r = vec4<u32>(q, 0u, 0u); }
        // clamp_wide(x: i128, lo: i64, hi: i64) -> i64
        case 4u: { let q = clamp_wide(arg128(b), arg64(b, 2u), arg64(b, 3u)); r = vec4<u32>(q, 0u, 0u); }
        // u64 * u64 -> u128
        case 5u: { r = u128_mul64(arg64(b, 0u), arg64(b, 1u)); }
        // shift_round(p: i128, s, 126) (the rule of rescale_wide)
        case 6u: { r = shift_round(arg128(b), tin[b + 4u], 126u); }
        // u128 / n, n <= 2^16
        case 7u: { r = u128_div_small(arg128(b), tin[b + 4u]); }
        // dot4I8Packed(a, b)
        case 8u: { r = vec4<u32>(bitcast<u32>(dot4I8Packed(tin[b], tin[b + 1u])), 0u, 0u, 0u); }
        default: {}
    }
    tout[c] = r;
}
