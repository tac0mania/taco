// The `embed` op (hardshell::gpt2 step 1):
// x[j] = clamp(rescale(wte[tok][j], embed_m[tok], es) + rescale(wpe[pos][j], pos_m[pos], ps), -R, R).

struct EmbedParams {
    d: u32,
    // Words in each row of `wte` and `wpe`.
    kw: u32,
    es: u32,
    ps: u32,
    // i32 values in each row of `x`.
    x_stride: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read> wte: array<u32>;
@group(0) @binding(1) var<storage, read> embed_m: array<i32>;
@group(0) @binding(2) var<storage, read> wpe: array<u32>;
@group(0) @binding(3) var<storage, read> pos_m: array<i32>;
@group(0) @binding(4) var<uniform> ep: EmbedParams;
@group(0) @binding(5) var<storage, read> rows: Rows;
@group(0) @binding(6) var<storage, read_write> x: array<i32>;

// Grid: (ceil(d / 64), rows).
@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let row = gid.y;
    if (row >= rows.count || j >= ep.d) { return; }
    let tok = rows.info[row].y;
    let pos = rows.info[row].z;
    let te = byte_i8(wte[tok * ep.kw + j / 4u], j % 4u);
    let pe = byte_i8(wpe[pos * ep.kw + j / 4u], j % 4u);
    let a = rescale(i64_from_i32(te), embed_m[tok], ep.es);
    let b = rescale(i64_from_i32(pe), pos_m[pos], ep.ps);
    x[row * ep.x_stride + j] = clamp_to_i32(u128_add(a, b), -RES, RES);
}
