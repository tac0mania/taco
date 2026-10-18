// The output part of the `logits` op (hardshell::gpt2 steps 4 and 5, plan §7.6), after the
// matrix product with `wte`:
//   logit[v] = clamp(rescale(acc[v], m[v], s), -2^40, 2^40)
//   weight[v] = logits_exp[min(max(logit) - logit[v], len - 1)]

struct LogitParams {
    vocab: u32,
    acc_stride: u32,
    shift: u32,
    table_len: u32,
}

@group(0) @binding(0) var<storage, read> acc: array<i32>;
@group(0) @binding(1) var<storage, read> mult: array<i32>;
@group(0) @binding(2) var<uniform> lp: LogitParams;
@group(0) @binding(3) var<storage, read> rows: Rows;
@group(0) @binding(4) var<storage, read_write> logit: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> row_max: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> logits_exp: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read_write> weights: array<vec2<u32>>;

// Grid: (ceil(vocab / 64), rows).
@compute @workgroup_size(64)
fn logits_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    let row = gid.y;
    if (row >= rows.count || v >= lp.vocab) { return; }
    let r = rescale(i64_from_i32(acc[row * lp.acc_stride + v]), mult[v], lp.shift);
    logit[row * lp.vocab + v] = clamp_wide(r, u64_neg(vec2<u32>(0u, 256u)), vec2<u32>(0u, 256u));
}

var<workgroup> red64: array<vec2<u32>, 256>;

// Grid: (rows, 1).
@compute @workgroup_size(256)
fn logits_max(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = wg.x;
    var mx = u64_neg(vec2<u32>(0u, 256u));
    if (row < rows.count) {
        for (var v = t; v < lp.vocab; v = v + 256u) { mx = i64_max(mx, logit[row * lp.vocab + v]); }
    }
    red64[t] = mx;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k / 2u) {
        if (t < k) { red64[t] = i64_max(red64[t], red64[t + k]); }
        workgroupBarrier();
    }
    if (t == 0u && row < rows.count) { row_max[row] = red64[0]; }
}

// Grid: (ceil(vocab / 64), rows).
@compute @workgroup_size(64)
fn logits_weights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    let row = gid.y;
    if (row >= rows.count || v >= lp.vocab) { return; }
    let d = u64_sub(row_max[row], logit[row * lp.vocab + v]);
    let last = vec2<u32>(lp.table_len - 1u, 0u);
    let idx = select(d, last, u64_lt(last, d));
    weights[row * lp.vocab + v] = logits_exp[idx.x];
}
