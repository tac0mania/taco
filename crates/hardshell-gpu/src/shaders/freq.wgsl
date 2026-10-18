// The coder interval of one symbol: the rule of taco_coder::FreqTable::from_weights (plan §5.1)
// on the weights of one row, without a copy of the table to the host.
//
// With V symbols, the weight sum W and the total T = 2^bits:
// 1. If W is 0, every weight is 1.
// 2. f[i] = 1 + floor(w[i] * (T - V) / W).
// 3. The remainder T - sum(f) goes to the symbol with the largest weight (the lowest id on a tie).
// The start of a symbol is the sum of the frequencies of the symbols before it.
//
// A query encodes (the interval of a known token) or decodes (the symbol whose interval holds a
// target value below T, as FreqTable::find). One workgroup for each row. Each invocation owns a
// contiguous chunk of symbols. All sums are exact u64 sums, so their order does not matter.

struct FreqParams {
    vocab: u32,
    total_lo: u32,
    total_hi: u32,
    // T - V, below 2^32.
    spare: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> fp: FreqParams;
@group(0) @binding(3) var<storage, read> rows: Rows;
@group(0) @binding(4) var<storage, read_write> intervals: array<vec4<u32>>;

const FT: u32 = 256u;
var<workgroup> red_w: array<vec2<u32>, 256>;
var<workgroup> red_i: array<u32, 256>;
var<workgroup> chunk: array<vec2<u32>, 256>;
var<workgroup> prefix: array<vec2<u32>, 257>;
var<workgroup> wsum: vec2<u32>;
var<workgroup> all_zero: u32;
var<workgroup> largest: u32;
var<workgroup> rem: vec2<u32>;

// floor(w * spare / W) for w <= W: below 2^32. The long division starts at p >> 32, which is
// below W because the quotient is below 2^32; then 32 rounds give the 32 quotient bits.
fn quot(w: vec2<u32>, big_w: vec2<u32>) -> u32 {
    let p = u128_mul32(vec4<u32>(w, 0u, 0u), fp.spare);
    var r = vec2<u32>(p.y, p.z);
    var q = 0u;
    for (var i = 31i; i >= 0i; i = i - 1i) {
        let top = (r.y & SIGN) != 0u;
        r = vec2<u32>((r.x << 1u) | ((p.x >> u32(i)) & 1u), (r.y << 1u) | (r.x >> 31u));
        q = q << 1u;
        if (top || !u64_lt(r, big_w)) {
            r = u64_sub(r, big_w);
            q = q | 1u;
        }
    }
    return q;
}

// The frequency of symbol i of `base`, with the remainder on the largest symbol.
fn freq_of(base: u32, i: u32) -> vec2<u32> {
    var w = weights[base + i];
    if (all_zero != 0u) { w = vec2<u32>(1u, 0u); }
    var f = u64_add(vec2<u32>(quot(w, wsum), 0u), vec2<u32>(1u, 0u));
    if (i == largest) { f = u64_add(f, rem); }
    return f;
}

// Grid: (rows, 1).
@compute @workgroup_size(256)
fn coder_interval(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = wg.x;
    let live = row < rows.count;
    let base = row * fp.vocab;
    let c = (fp.vocab + FT - 1u) / FT;
    let lo = min(t * c, fp.vocab);
    let hi = min(lo + c, fp.vocab);
    // Pass 1: the weight sum, and the largest weight with the lowest id.
    var s = vec2<u32>(0u, 0u);
    var bw = vec2<u32>(0u, 0u);
    var bi = select(0xffffffffu, lo, lo < hi);
    if (live) {
        for (var i = lo; i < hi; i = i + 1u) {
            let w = weights[base + i];
            s = u64_add(s, w);
            if (u64_lt(bw, w)) {
                bw = w;
                bi = i;
            }
        }
    }
    chunk[t] = s;
    red_w[t] = bw;
    red_i[t] = bi;
    workgroupBarrier();
    for (var h = FT / 2u; h > 0u; h = h / 2u) {
        if (t < h) {
            chunk[t] = u64_add(chunk[t], chunk[t + h]);
            let ow = red_w[t + h];
            let oi = red_i[t + h];
            if (u64_lt(red_w[t], ow) || (red_w[t].x == ow.x && red_w[t].y == ow.y && oi < red_i[t])) {
                red_w[t] = ow;
                red_i[t] = oi;
            }
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        let total_w = chunk[0];
        let zero = total_w.x == 0u && total_w.y == 0u;
        all_zero = select(0u, 1u, zero);
        // Rule 1: every weight is 1, and symbol 0 is the largest.
        wsum = select(total_w, vec2<u32>(fp.vocab, 0u), zero);
        largest = select(red_i[0], 0u, zero);
        rem = vec2<u32>(0u, 0u);
    }
    workgroupBarrier();
    // Pass 2: the sum of f in each chunk, before the remainder.
    var cs = vec2<u32>(0u, 0u);
    if (live) {
        for (var i = lo; i < hi; i = i + 1u) { cs = u64_add(cs, freq_of(base, i)); }
    }
    chunk[t] = cs;
    workgroupBarrier();
    if (t == 0u) {
        var acc = vec2<u32>(0u, 0u);
        for (var j = 0u; j < FT; j = j + 1u) {
            prefix[j] = acc;
            acc = u64_add(acc, chunk[j]);
        }
        // INVARIANT: sum(f) <= V + (T - V) = T, so the remainder is not negative.
        rem = u64_sub(vec2<u32>(fp.total_lo, fp.total_hi), acc);
        // The remainder moves every start after the largest symbol.
        for (var j = 0u; j < FT; j = j + 1u) {
            if (largest < j * c) { prefix[j] = u64_add(prefix[j], rem); }
        }
        prefix[FT] = vec2<u32>(fp.total_lo, fp.total_hi);
    }
    workgroupBarrier();
    if (!live || lo >= hi) { return; }
    // Pass 3: the query.
    let qr = queries[row];
    var mine = false;
    if (qr.x == 0u) {
        mine = qr.y >= lo && qr.y < hi;
    } else {
        let x = vec2<u32>(qr.y, 0u);
        var next = prefix[FT];
        if (hi < fp.vocab) { next = prefix[t + 1u]; }
        mine = !u64_lt(x, prefix[t]) && u64_lt(x, next);
    }
    if (!mine) { return; }
    var cum = prefix[t];
    for (var i = lo; i < hi; i = i + 1u) {
        let f = freq_of(base, i);
        let end = u64_add(cum, f);
        let hit = select(i == qr.y, u64_lt(vec2<u32>(qr.y, 0u), end), qr.x != 0u);
        if (hit) {
            intervals[row] = vec4<u32>(i, cum.x, f.x, f.y);
            return;
        }
        cum = end;
    }
}
