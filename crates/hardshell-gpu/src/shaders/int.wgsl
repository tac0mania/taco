// The integer rules of hardshell::int, in WGSL, with 32-bit words only.
//
// WGSL has no 64-bit or 128-bit integers. This file emulates them with u32 words:
// - u64 and i64 are vec2<u32>(low word, high word), two's complement for i64.
// - i128 is vec4<u32>(word 0 = lowest, ..., word 3 = highest), two's complement.
//
// DETERMINISM: every operation here is a u32 operation whose result the WGSL specification
// defines on every device: addition, subtraction and multiplication wrap modulo 2^32, shifts
// use an amount below 32, and comparisons are exact. No function uses i32 arithmetic that can
// overflow, no float and no division by a value that can be zero. Thus each function gives the
// result of the matching Rust function in hardshell::int, on every GPU.
//
// PLAN(§7.2): rescale, clamp, floor division, square root and the table look-up rules.

const SIGN: u32 = 0x80000000u;
const ONES: u32 = 0xffffffffu;

// ---------- 64-bit ----------

fn u64_from(x: u32) -> vec2<u32> { return vec2<u32>(x, 0u); }

fn i64_from_i32(x: i32) -> vec2<u32> {
    return vec2<u32>(bitcast<u32>(x), select(0u, ONES, x < 0));
}

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

fn u64_neg(a: vec2<u32>) -> vec2<u32> { return u64_sub(vec2<u32>(0u, 0u), a); }

fn u64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

fn i64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
    return u64_lt(vec2<u32>(a.x, a.y ^ SIGN), vec2<u32>(b.x, b.y ^ SIGN));
}

fn i64_is_neg(a: vec2<u32>) -> bool { return (a.y & SIGN) != 0u; }

fn i64_max(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(a, b, i64_lt(a, b)); }

// |a| as a u64. |i64::MIN| = 2^63 fits.
fn i64_abs(a: vec2<u32>) -> vec2<u32> { return select(a, u64_neg(a), i64_is_neg(a)); }

// The full 64-bit product of two u32 values, from 16-bit halves.
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu;
    let ah = a >> 16u;
    let bl = b & 0xffffu;
    let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    // INVARIANT: each partial product is below 2^32; `mid` can carry one bit.
    let mid = lh + hl;
    let mid_carry = select(0u, 0x10000u, mid < lh);
    let lo = ll + (mid << 16u);
    let c = select(0u, 1u, lo < ll);
    return vec2<u32>(lo, hh + (mid >> 16u) + mid_carry + c);
}

// a * b modulo 2^64.
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = mul32(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}

// a << s for s in 0..64, modulo 2^64.
fn u64_shl(a: vec2<u32>, s: u32) -> vec2<u32> {
    if (s == 0u) { return a; }
    if (s >= 64u) { return vec2<u32>(0u, 0u); }
    if (s >= 32u) { return vec2<u32>(0u, a.x << (s - 32u)); }
    return vec2<u32>(a.x << s, (a.y << s) | (a.x >> (32u - s)));
}

// Bit `i` (0..64) of a.
fn u64_bit(a: vec2<u32>, i: u32) -> u32 {
    if (i < 32u) { return (a.x >> i) & 1u; }
    return (a.y >> (i - 32u)) & 1u;
}

// floor(n / d) for d >= 1: the long division, one bit in each round. Cost: 64 rounds. Use it
// once for each row, not for each element.
fn u64_div(n: vec2<u32>, d: vec2<u32>) -> vec2<u32> {
    var q = vec2<u32>(0u, 0u);
    var r = vec2<u32>(0u, 0u);
    for (var i = 63i; i >= 0i; i = i - 1i) {
        let bit = u64_bit(n, u32(i));
        // INVARIANT: r < d <= 2^64 - 1 before the shift, so 2r + 1 can need 65 bits only when
        // the top bit of r is set; then 2r + 1 >= d surely, and the wrapped difference is exact.
        let top = (r.y & SIGN) != 0u;
        r = vec2<u32>((r.x << 1u) | bit, (r.y << 1u) | (r.x >> 31u));
        if (top || !u64_lt(r, d)) {
            r = u64_sub(r, d);
            if (i >= 32i) { q.y = q.y | (1u << u32(i - 32i)); } else { q.x = q.x | (1u << u32(i)); }
        }
    }
    return q;
}

// floor_div(a, n) for an i64 `a` and 1 <= n <= 2^16 (hardshell::int::floor_div). Cost: four
// u32 divisions.
fn i64_floor_div_small(a: vec2<u32>, n: u32) -> vec2<u32> {
    let neg = i64_is_neg(a);
    let m = i64_abs(a);
    // Divide the magnitude by n with 16-bit digits: each partial dividend is below 2^32.
    var q = vec2<u32>(0u, 0u);
    var r = 0u;
    let digits = array<u32, 4>(m.y >> 16u, m.y & 0xffffu, m.x >> 16u, m.x & 0xffffu);
    var qd = array<u32, 4>(0u, 0u, 0u, 0u);
    for (var i = 0u; i < 4u; i = i + 1u) {
        // INVARIANT: r < n <= 2^16, so (r << 16) | digit < 2^32.
        let cur = (r << 16u) | digits[i];
        qd[i] = cur / n;
        r = cur - qd[i] * n;
    }
    q = vec2<u32>((qd[2] << 16u) | qd[3], (qd[0] << 16u) | qd[1]);
    if (!neg) { return q; }
    // floor(-m / n) = -(q + 1) when the remainder is not zero.
    let qq = select(q, u64_add(q, vec2<u32>(1u, 0u)), r != 0u);
    return u64_neg(qq);
}

// ---------- 128-bit ----------

fn i128_from_i64(a: vec2<u32>) -> vec4<u32> {
    let s = select(0u, ONES, i64_is_neg(a));
    return vec4<u32>(a.x, a.y, s, s);
}

fn i128_from_i32(x: i32) -> vec4<u32> { return i128_from_i64(i64_from_i32(x)); }

fn u128_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var r: vec4<u32>;
    var c = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let s1 = a[i] + b[i];
        let c1 = select(0u, 1u, s1 < a[i]);
        let s2 = s1 + c;
        let c2 = select(0u, 1u, s2 < s1);
        r[i] = s2;
        c = c1 + c2;
    }
    return r;
}

fn u128_neg(a: vec4<u32>) -> vec4<u32> {
    return u128_add(vec4<u32>(~a.x, ~a.y, ~a.z, ~a.w), vec4<u32>(1u, 0u, 0u, 0u));
}

fn i128_is_neg(a: vec4<u32>) -> bool { return (a.w & SIGN) != 0u; }

fn i128_lt(a: vec4<u32>, b: vec4<u32>) -> bool {
    let aw = a.w ^ SIGN;
    let bw = b.w ^ SIGN;
    if (aw != bw) { return aw < bw; }
    if (a.z != b.z) { return a.z < b.z; }
    if (a.y != b.y) { return a.y < b.y; }
    return a.x < b.x;
}

// a * b modulo 2^128, for a u128 `a` and a u32 `b`.
fn u128_mul32(a: vec4<u32>, b: u32) -> vec4<u32> {
    var r: vec4<u32>;
    var carry = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let p = mul32(a[i], b);
        let s = p.x + carry;
        r[i] = s;
        // INVARIANT: p <= (2^32 - 1)^2, so p + carry < 2^64 and the high word cannot wrap.
        carry = p.y + select(0u, 1u, s < p.x);
    }
    return r;
}

// a * b modulo 2^128, for a u64 `a` and a u64 `b`.
fn u128_mul64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let lo = u128_mul32(vec4<u32>(a.x, a.y, 0u, 0u), b.x);
    let hi = u128_mul32(vec4<u32>(a.x, a.y, 0u, 0u), b.y);
    return u128_add(lo, vec4<u32>(0u, hi.x, hi.y, hi.z));
}

// floor(a / 2^s) for s in 1..128, with the sign bit copied (an arithmetic shift).
fn i128_sar(a: vec4<u32>, s: u32) -> vec4<u32> {
    let fill = select(0u, ONES, i128_is_neg(a));
    let ws = s >> 5u;
    let bs = s & 31u;
    var w = array<u32, 6>(a.x, a.y, a.z, a.w, fill, fill);
    var r: vec4<u32>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = min(i + ws, 4u);
        let lo = w[j];
        let hi = w[min(j + 1u, 5u)];
        if (bs == 0u) { r[i] = lo; } else { r[i] = (lo >> bs) | (hi << (32u - bs)); }
    }
    return r;
}

// 2^s as an i128, for s in 0..127.
fn u128_pow2(s: u32) -> vec4<u32> {
    var r = vec4<u32>(0u, 0u, 0u, 0u);
    r[s >> 5u] = 1u << (s & 31u);
    return r;
}

// hardshell::int::shift_round: floor((p + 2^(s-1)) / 2^s), 0 for s > max_shift, p for s = 0.
fn shift_round(p: vec4<u32>, s: u32, max_shift: u32) -> vec4<u32> {
    if (s == 0u) { return p; }
    if (s > max_shift) { return vec4<u32>(0u, 0u, 0u, 0u); }
    return i128_sar(u128_add(p, u128_pow2(s - 1u)), s);
}

// hardshell::int::rescale(x, m, s): exact (x * m + 2^(s-1)) >> s for an i64 `x` and an i32 `m`.
fn rescale(x: vec2<u32>, m: i32, s: u32) -> vec4<u32> {
    let mag_m = select(bitcast<u32>(m), 0u - bitcast<u32>(m), m < 0);
    // INVARIANT: |x| <= 2^63 and |m| <= 2^31, so the product of the magnitudes is below 2^95
    // and the sign-magnitude product is the exact two's complement product.
    var p = u128_mul32(vec4<u32>(i64_abs(x), 0u, 0u), mag_m);
    if (i64_is_neg(x) != (m < 0)) { p = u128_neg(p); }
    return shift_round(p, s, 95u);
}

// hardshell::int::clamp_wide(x, lo, hi) for i64 limits; the result is an i64.
fn clamp_wide(x: vec4<u32>, lo: vec2<u32>, hi: vec2<u32>) -> vec2<u32> {
    let lo_w = i128_from_i64(lo);
    let hi_w = i128_from_i64(hi);
    var upper = x;
    if (!i128_lt(x, hi_w)) { upper = hi_w; }
    var c = lo_w;
    if (i128_lt(lo_w, upper)) { c = upper; }
    return vec2<u32>(c.x, c.y);
}

// clamp_wide to i32 limits, as an i32.
fn clamp_to_i32(x: vec4<u32>, lo: i32, hi: i32) -> i32 {
    let c = clamp_wide(x, i64_from_i32(lo), i64_from_i32(hi));
    return bitcast<i32>(c.x);
}

// hardshell::int::clamp_act_i8: clamp to [-127, 127].
fn clamp_act_i8(x: vec4<u32>) -> i32 { return clamp_to_i32(x, -127i, 127i); }

// floor(sqrt(x)) for a u128 (hardshell::int::isqrt_u128), the same digit-by-digit method.
fn isqrt_u128(x: vec4<u32>) -> vec4<u32> {
    var bit = u128_pow2(126u);
    var bit_s = 126u;
    // The largest power of 4 that is not more than x.
    loop {
        if (bit_s == 0u || !u128_lt(x, bit)) { break; }
        bit_s = bit_s - 2u;
        bit = u128_pow2(bit_s);
    }
    if (u128_lt(x, bit)) { return vec4<u32>(0u, 0u, 0u, 0u); }
    var rem = x;
    var res = vec4<u32>(0u, 0u, 0u, 0u);
    loop {
        let trial = u128_add(res, bit);
        if (!u128_lt(rem, trial)) {
            rem = u128_add(rem, u128_neg(trial));
            res = u128_add(u128_shr1(res), bit);
        } else {
            res = u128_shr1(res);
        }
        if (bit_s < 2u) { break; }
        bit_s = bit_s - 2u;
        bit = u128_pow2(bit_s);
    }
    return res;
}

fn u128_lt(a: vec4<u32>, b: vec4<u32>) -> bool {
    if (a.w != b.w) { return a.w < b.w; }
    if (a.z != b.z) { return a.z < b.z; }
    if (a.y != b.y) { return a.y < b.y; }
    return a.x < b.x;
}

fn u128_shr1(a: vec4<u32>) -> vec4<u32> {
    return vec4<u32>((a.x >> 1u) | (a.y << 31u), (a.y >> 1u) | (a.z << 31u),
                     (a.z >> 1u) | (a.w << 31u), a.w >> 1u);
}

// a << s for s in 0..128, modulo 2^128.
fn u128_shl(a: vec4<u32>, s: u32) -> vec4<u32> {
    let ws = s >> 5u;
    let bs = s & 31u;
    var w = array<u32, 4>(a.x, a.y, a.z, a.w);
    var r = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (i < ws) { continue; }
        let lo = w[i - ws];
        var below = 0u;
        if (i > ws) { below = w[i - ws - 1u]; }
        if (bs == 0u) { r[i] = lo; } else { r[i] = (lo << bs) | (below >> (32u - bs)); }
    }
    return r;
}

// floor(x / n) for a u128 `x` below 2^96 and 1 <= n <= 2^16, with 16-bit digits.
fn u128_div_small(x: vec4<u32>, n: u32) -> vec4<u32> {
    var r = 0u;
    var q = vec4<u32>(0u, 0u, 0u, 0u);
    for (var wi = 0i; wi < 4i; wi = wi + 1i) {
        let i = u32(3i - wi);
        let hi_d = (r << 16u) | (x[i] >> 16u);
        let qh = hi_d / n;
        r = hi_d - qh * n;
        let lo_d = (r << 16u) | (x[i] & 0xffffu);
        let ql = lo_d / n;
        r = lo_d - ql * n;
        q[i] = (qh << 16u) | ql;
    }
    return q;
}

// The low byte `k` (0..3) of a packed word, as a signed value.
fn byte_i8(word: u32, k: u32) -> i32 {
    return bitcast<i32>(word << (24u - 8u * k)) >> 24u;
}
