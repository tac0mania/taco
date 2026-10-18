// The `table` op (the GELU of hardshell::gelu): y = t[clamp(x, -offset, len - 1 - offset) + offset],
// with i32 inputs and packed i8 outputs (four values in each word).

struct TableParams {
    n: u32,
    in_stride: u32,
    // Words in each row of the output.
    out_kw: u32,
    offset: u32,
    len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read> tin: array<i32>;
@group(0) @binding(1) var<storage, read> t: array<i32>;
@group(0) @binding(2) var<uniform> tp: TableParams;
@group(0) @binding(3) var<storage, read> rows: Rows;
@group(0) @binding(4) var<storage, read_write> tout: array<u32>;

fn lookup(x: i32) -> u32 {
    // INVARIANT: offset <= 2^15 and len <= 2^16, so these i32 values cannot overflow.
    let lo = -i32(tp.offset);
    let hi = i32(tp.len) - 1i - i32(tp.offset);
    let i = u32(clamp(x, lo, hi) + i32(tp.offset));
    return bitcast<u32>(t[i]) & 0xffu;
}

// Grid: (ceil(n / 256), rows); each invocation makes one output word.
@compute @workgroup_size(64)
fn table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let row = gid.y;
    if (row >= rows.count || w * 4u >= tp.n) { return; }
    var word = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let c = w * 4u + k;
        if (c < tp.n) { word = word | (lookup(tin[row * tp.in_stride + c]) << (8u * k)); }
    }
    tout[row * tp.out_kw + w] = word;
}
