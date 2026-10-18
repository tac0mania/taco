// The `attention` op: hardshell::gpt2 steps 2.2 (cache) and 2.3, with the softmax of
// hardshell::softmax (plan §7.5).
//
// kv_store: writes k and v of each row into the cache of its slot at its position.
// attend: for each row and head, over the positions j <= pos:
//   score[j] = clamp(rescale(q . k[j], score_m, score_s), -2^40, 2^40)
//   e[j] = table[min(max(score) - score[j], len - 1)]
//   p[j] = floor(e[j] * 2^P / sum(e))
//   out[c] = clamp(rescale(sum_j p[j] * v[j][c], out_m, out_s), -127, 127)
//
// DETERMINISM: max and sum are exact integer reductions. p[j] uses an exact reciprocal with one
// correction step (see `prob`), so it equals the floor division of the CPU path. The sum of
// p * v is split in two i32 sums whose bounds are proved at `attend`.

struct AttnParams {
    // Words in each row of the cache (d / 4).
    d4: u32,
    // Words in each row of `qkv`.
    qkv_stride: u32,
    // Words in each head (head_dim / 4).
    hd4: u32,
    n_ctx: u32,
    score_m: i32,
    score_s: u32,
    out_m: i32,
    out_s: u32,
    p: u32,
    table_len: u32,
    // Words in each row of `att`.
    att_stride: u32,
    n_head: u32,
}

@group(0) @binding(0) var<storage, read> qkv: array<u32>;
@group(0) @binding(1) var<storage, read_write> kc: array<u32>;
@group(0) @binding(2) var<storage, read_write> vc: array<u32>;
@group(0) @binding(3) var<storage, read_write> sc: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> attn_exp: array<i32>;
@group(0) @binding(5) var<uniform> ap: AttnParams;
@group(0) @binding(6) var<storage, read> rows: Rows;
@group(0) @binding(7) var<storage, read_write> att: array<u32>;

// Grid: (ceil(d4 / 64), rows).
@compute @workgroup_size(64)
fn kv_store(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let row = gid.y;
    if (row >= rows.count || w >= ap.d4) { return; }
    let slot = rows.info[row].x;
    let pos = rows.info[row].z;
    let at = (slot * ap.n_ctx + pos) * ap.d4 + w;
    kc[at] = qkv[row * ap.qkv_stride + ap.d4 + w];
    vc[at] = qkv[row * ap.qkv_stride + 2u * ap.d4 + w];
}

const AT: u32 = 64u;
var<workgroup> red64: array<vec2<u32>, 64>;
var<workgroup> q_h: array<u32, 64>;
var<workgroup> outs: array<i32, 256>;
var<workgroup> recip: vec2<u32>;
var<workgroup> total: vec2<u32>;

// floor(e * 2^P / s) for e < 2^31, 1 <= s < 2^62 and m = floor(2^(P+32) / s).
//
// est = floor(e * m / 2^32). Then est <= e * 2^P / s, and e * 2^P / s - e * m / 2^32 < e / 2^32
// < 1, so the exact quotient is est or est + 1. One exact check selects it.
fn prob(e: u32, s: vec2<u32>, m: vec2<u32>) -> u32 {
    let lo = mul32(e, m.x);
    let hi = mul32(e, m.y);
    let est = u64_add(hi, vec2<u32>(lo.y, 0u)).x;
    // INVARIANT: est * s <= e * 2^P < 2^61, so the wrapped u64 products are exact, and the
    // remainder is below 2s.
    let num = u64_shl(vec2<u32>(e, 0u), ap.p);
    let rem = u64_sub(num, u64_mul(vec2<u32>(est, 0u), s));
    return select(est, est + 1u, !u64_lt(rem, s));
}

// Grid: (n_head, rows). The sums of p * v: sum(p) <= 2^30, so sum((p >> 6) * v) has a
// magnitude of at most 2^24 * 128 = 2^31 (an i32; the values of v are in -127..127, so the sum is
// below 2^31), and sum((p & 63) * v) <= n_ctx * 63 * 127 < 2^29 for n_ctx <= 2^16.
@compute @workgroup_size(64)
fn attend(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let h = wg.x;
    let row = wg.y;
    let live = row < rows.count;
    var slot = 0u;
    var n = 0u;
    if (live) {
        slot = rows.info[row].x;
        n = rows.info[row].z + 1u;
    }
    let sbase = (row * ap.n_head + h) * ap.n_ctx;
    if (t < ap.hd4 && live) { q_h[t] = qkv[row * ap.qkv_stride + h * ap.hd4 + t]; }
    workgroupBarrier();
    // Scores and their maximum.
    let neg_limit = u64_neg(vec2<u32>(0u, 256u));
    let pos_limit = vec2<u32>(0u, 256u);
    var mx = neg_limit;
    for (var j = t; j < n; j = j + AT) {
        let kb = (slot * ap.n_ctx + j) * ap.d4 + h * ap.hd4;
        var dot = 0i;
        for (var w = 0u; w < ap.hd4; w = w + 1u) { dot = dot + dot4I8Packed(q_h[w], kc[kb + w]); }
        let s = clamp_wide(rescale(i64_from_i32(dot), ap.score_m, ap.score_s), neg_limit, pos_limit);
        sc[sbase + j] = s;
        mx = i64_max(mx, s);
    }
    red64[t] = mx;
    workgroupBarrier();
    for (var k = AT / 2u; k > 0u; k = k / 2u) {
        if (t < k) { red64[t] = i64_max(red64[t], red64[t + k]); }
        workgroupBarrier();
    }
    let row_max = red64[0];
    workgroupBarrier();
    // Exponents and their sum.
    let last = vec2<u32>(ap.table_len - 1u, 0u);
    var sum = vec2<u32>(0u, 0u);
    for (var j = t; j < n; j = j + AT) {
        let d = u64_sub(row_max, sc[sbase + j]);
        let idx = select(d, last, u64_lt(last, d));
        let e = bitcast<u32>(attn_exp[idx.x]);
        sc[sbase + j] = vec2<u32>(e, 0u);
        sum = u64_add(sum, vec2<u32>(e, 0u));
    }
    red64[t] = sum;
    workgroupBarrier();
    for (var k = AT / 2u; k > 0u; k = k / 2u) {
        if (t < k) { red64[t] = u64_add(red64[t], red64[t + k]); }
        workgroupBarrier();
    }
    if (t == 0u) {
        // INVARIANT: the maximum score takes table[0] >= 1, so the sum is at least 1.
        var s = red64[0];
        if (s.x == 0u && s.y == 0u) { s = vec2<u32>(1u, 0u); }
        total = s;
        recip = u64_div(u64_shl(vec2<u32>(1u, 0u), ap.p + 32u), s);
    }
    workgroupBarrier();
    let s = total;
    let m = recip;
    for (var j = t; j < n; j = j + AT) {
        sc[sbase + j] = vec2<u32>(prob(sc[sbase + j].x, s, m), 0u);
    }
    storageBarrier();
    workgroupBarrier();
    // sum_j p[j] * v[j][c] for the channels of this invocation.
    let hd = ap.hd4 * 4u;
    for (var c = t; c < hd; c = c + AT) {
        let ch = h * hd + c;
        var acc_hi = 0i;
        var acc_lo = 0i;
        for (var j = 0u; j < n; j = j + 1u) {
            let p = sc[sbase + j].x;
            let v = byte_i8(vc[(slot * ap.n_ctx + j) * ap.d4 + ch / 4u], ch % 4u);
            acc_hi = acc_hi + i32(p >> 6u) * v;
            acc_lo = acc_lo + i32(p & 63u) * v;
        }
        let a = u64_add(u64_shl(i64_from_i32(acc_hi), 6u), i64_from_i32(acc_lo));
        outs[c] = clamp_act_i8(rescale(a, ap.out_m, ap.out_s));
    }
    workgroupBarrier();
    if (t < ap.hd4 && live) {
        var word = 0u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            word = word | ((bitcast<u32>(outs[t * 4u + k]) & 0xffu) << (8u * k));
        }
        att[row * ap.att_stride + h * ap.hd4 + t] = word;
    }
}
