// HardShell on NVIDIA GPUs: the op kernels of the integer GPT-2 step (hardshell::gpt2), in CUDA C.
//
// tools/cuda/build_ptx.py compiles this file with NVRTC to PTX for compute_80 (Ampere and
// later). The PTX is committed in kernels/hardshell.ptx and embedded in the crate; at run time
// only the driver (libcuda) loads it. No user installs NVRTC or the CUDA toolkit (ADR 0008).
//
// DETERMINISM: every kernel uses only integer instructions. No float, no atomics. Each rule is
// the rule of the matching function in hardshell::int: rescale, clamp, floor division, isqrt and
// the clamped table look-up. Signed overflow is undefined in C++, so every signed operation has
// a bound (INVARIANT) that excludes it; shifts of negative values use unsigned operations.
// Integer sums are exact and associative, so the order of a reduction does not change it.

typedef signed char i8;
typedef int i32;
typedef unsigned int u32;
typedef long long i64;
typedef unsigned long long u64;
typedef __int128 i128;
typedef unsigned __int128 u128;

#define RES 1073741823
#define LIMIT40 (1LL << 40)

// ---------------------------------------------------------------- hardshell::int

// floor(x / 2^s) for s in 1..127, without a shift of a negative value.
__device__ __forceinline__ i128 sar128(i128 x, u32 s) {
    // INVARIANT: for x < 0, ~x = -x - 1 >= 0.
    return x >= 0 ? (i128)((u128)x >> s) : ~(i128)((u128)(~x) >> s);
}

// hardshell::int::shift_round: floor((p + 2^(s-1)) / 2^s); 0 for s > max_shift; p for s = 0.
__device__ __forceinline__ i128 shift_round(i128 p, u32 s, u32 max_shift) {
    if (s == 0) return p;
    if (s > max_shift) return 0;
    // INVARIANT: |p| < 2^max_shift and max_shift <= 126, so p + 2^(s-1) < 2^127.
    return sar128(p + ((i128)1 << (s - 1)), s);
}

// hardshell::int::rescale(x, m, s). INVARIANT: |x * m| <= 2^94.
__device__ __forceinline__ i128 rescale(i64 x, i32 m, u32 s) {
    return shift_round((i128)x * (i128)m, s, 95);
}

// hardshell::int::rescale(x, m, s) for |x * m| < 2^62, in i64. For s > 62 the exact result is
// 0: 2^(s-1) >= 2^62 > |p|, so 0 < p + 2^(s-1) < 2^s. For s <= 62, p + 2^(s-1) < 2^63.
__device__ __forceinline__ i64 rescale_small(i64 x, i32 m, u32 s) {
    const i64 p = x * (i64)m;
    if (s == 0) return p;
    if (s > 62) return 0;
    const i64 q = p + ((i64)1 << (s - 1));
    // INVARIANT: for q < 0, ~q = -q - 1 >= 0.
    return q >= 0 ? (i64)((u64)q >> s) : ~(i64)((u64)(~q) >> s);
}

// hardshell::int::clamp_wide.
__device__ __forceinline__ i64 clamp_wide(i128 x, i64 lo, i64 hi) {
    i128 upper = x < (i128)hi ? x : (i128)hi;
    return (i64)(upper > (i128)lo ? upper : (i128)lo);
}

__device__ __forceinline__ i32 clamp_act_i8(i128 x) { return (i32)clamp_wide(x, -127, 127); }

// hardshell::int::floor_div for b >= 1. C++ division truncates toward zero.
__device__ __forceinline__ i64 floor_div(i64 a, i64 b) {
    i64 q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

// hardshell::int::isqrt_u128: the digit-by-digit method.
__device__ u128 isqrt_u128(u128 x) {
    if (x == 0) return 0;
    // The largest power of 4 that is not more than x (the first loop of hardshell::int).
    const u64 hi = (u64)(x >> 64);
    const int top = hi != 0 ? 127 - __clzll(hi) : 63 - __clzll((u64)x);
    u128 bit = (u128)1 << (top & ~1);
    u128 rem = x, res = 0;
    while (bit != 0) {
        u128 trial = res + bit;
        if (rem >= trial) {
            rem -= trial;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return res;
}

// floor(x / n) for x < 2^96 and 1 <= n <= 2^16, with u64 steps only.
__device__ u128 udiv128_small(u128 x, u64 n) {
    u64 hi = (u64)(x >> 64), lo = (u64)x;
    u64 qh = hi / n, r = hi % n;
    // INVARIANT: r < 2^16, so each partial dividend is below 2^48.
    u64 d1 = (r << 32) | (lo >> 32);
    u64 q1 = d1 / n;
    r = d1 % n;
    u64 d0 = (r << 32) | (lo & 0xffffffffULL);
    u64 q0 = d0 / n;
    return ((u128)qh << 64) | (u128)((q1 << 32) | q0);
}

__device__ __forceinline__ i32 byte_at(const i8* p, u64 i) { return (i32)p[i]; }

// The low byte of `x` as a signed value, with defined integer operations only.
__device__ __forceinline__ i64 sbyte(u32 x) {
    i64 v = (i64)(x & 0xffu);
    return v - ((v & 0x80) << 1);
}

// ---------------------------------------------------------------- matrix products

// out[m][n] = sum_k a[m][k] * w[n][k] on int8 tensor cores: mma.sync m16n8k32 with exact s32
// accumulation. Both operands are row-major with k contiguous; a row has kw words (4 bytes each),
// kw is a multiple of 16, and padding bytes are zero. The buffers of `a` and `w` hold their rows
// rounded up to a multiple of 64, so every load is in bounds.
//
// INVARIANT: K <= 2^16 and |a*w| <= 2^14, so every s32 sum is at most 2^30 (no overflow), as
// hardshell::kernels::MAX_DOT_LEN_I32 states. The tile order does not change an exact sum.
//
// Block: 4 warps; tile 64 (m) x 64 (n); each warp makes 64 x 16. A stage holds 64 bytes of k for
// both tiles, loaded with cp.async (16 bytes for each copy) into two buffers, so the next load
// runs during the products. The rows have 20 words in shared memory (16 + 4), so the fragment
// loads of the 32 lanes hit 32 different banks.
//
// Split-K: block z of gridDim.z sums the stages [z * S / Z, (z + 1) * S / Z) and writes its
// partial sums to out + z * split_stride. The output op adds the Z partials. Each partial and
// the total are exact i32 sums below 2^30 in magnitude (the INVARIANT above), so the split does not
// change a result. Grid: (ceil(m / 64), ceil(n / 64), Z).
#define LDS 20
extern "C" __global__ void __launch_bounds__(128) gemm_tiled(const u32* __restrict__ w, const u32* __restrict__ a, i32* __restrict__ out,
                                      u32 n, u32 kw, u32 out_stride, u32 m, u32 split_stride) {
    __shared__ __align__(16) u32 as_[2][64 * LDS];
    __shared__ __align__(16) u32 ws_[2][64 * LDS];
    const u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const u32 g = lane >> 2, t = lane & 3;
    const u32 m0 = blockIdx.x * 64, n0 = blockIdx.y * 64;
    const u32 stages = kw / 16, z = blockIdx.z, zs = gridDim.z;
    const u32 s_begin = (u32)((u64)stages * z / zs), s_end = (u32)((u64)stages * (z + 1) / zs);
    i32 acc[4][2][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0;
    // Each thread copies 2 x 16 bytes of each tile: row r = (tid + 128 * h) / 4, quarter q.
    auto load = [&](u32 stage, u32 buf) {
#pragma unroll
        for (u32 h = 0; h < 2; h++) {
            u32 idx = tid + 128 * h, r = idx >> 2, q = idx & 3;
            const u32* ga = a + (u64)(m0 + r) * kw + stage * 16 + q * 4;
            const u32* gw = w + (u64)(n0 + r) * kw + stage * 16 + q * 4;
            u32 sa = (u32)__cvta_generic_to_shared(&as_[buf][r * LDS + q * 4]);
            u32 sw = (u32)__cvta_generic_to_shared(&ws_[buf][r * LDS + q * 4]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(ga));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sw), "l"(gw));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    if (s_begin < s_end) load(s_begin, 0);
    for (u32 st = s_begin; st < s_end; st++) {
        const u32 buf = (st - s_begin) & 1;
        if (st + 1 < s_end) {
            load(st + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const u32* A = as_[buf];
        const u32* W = ws_[buf] + warp * 16 * LDS;
#pragma unroll
        for (u32 kk = 0; kk < 16; kk += 8) {
            u32 b[2][2];
#pragma unroll
            for (int j = 0; j < 2; j++) {
                b[j][0] = W[(j * 8 + g) * LDS + kk + t];
                b[j][1] = W[(j * 8 + g) * LDS + kk + 4 + t];
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const u32* r0 = A + (i * 16 + g) * LDS + kk;
                const u32* r1 = r0 + 8 * LDS;
                u32 a0 = r0[t], a1 = r1[t], a2 = r0[4 + t], a3 = r1[4 + t];
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[j][0]), "r"(b[j][1]));
                }
            }
        }
        __syncthreads();
    }
    i32* o = out + (u64)z * split_stride;
    const u32 nw = n0 + warp * 16;
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            u32 col = nw + j * 8 + t * 2;
            u32 r0 = m0 + i * 16 + g, r1 = r0 + 8;
            if (r0 < m) {
                if (col < n) o[(u64)r0 * out_stride + col] = acc[i][j][0];
                if (col + 1 < n) o[(u64)r0 * out_stride + col + 1] = acc[i][j][1];
            }
            if (r1 < m) {
                if (col < n) o[(u64)r1 * out_stride + col] = acc[i][j][2];
                if (col + 1 < n) o[(u64)r1 * out_stride + col + 1] = acc[i][j][3];
            }
        }
    }
}

// The same product with dp4a: one warp for each row of `w`, all rows of `a` (m <= 8) in one
// pass. Grid: (ceil(n / 8), ceil(m / 8)); block: 256 threads (8 warps).
extern "C" __global__ void matvec_dp4a(const u32* __restrict__ w, const u32* __restrict__ a, i32* __restrict__ out,
                                       u32 n, u32 kw, u32 out_stride, u32 m) {
    const u32 lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const u32 row = blockIdx.x * 8 + warp;
    const u32 m0 = blockIdx.y * 8;
    if (row >= n) return;
    const u32 cnt = m - m0 < 8 ? m - m0 : 8;
    i32 acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (u32 k = lane; k < kw; k += 32) {
        i32 wv = (i32)w[(u64)row * kw + k];
#pragma unroll
        for (u32 r = 0; r < 8; r++)
            if (r < cnt) acc[r] = __dp4a(wv, (i32)a[(u64)(m0 + r) * kw + k], acc[r]);
    }
#pragma unroll
    for (u32 r = 0; r < 8; r++) {
        i32 v = acc[r];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0 && r < cnt) out[(u64)(m0 + r) * out_stride + row] = v;
    }
}

// ---------------------------------------------------------------- rows of a forward pass

// rows[4 * r + 0..3] = slot, token, position, unused.

// The `embed` op. Grid: (ceil(d / 128), rows); block 128.
extern "C" __global__ void embed(const i8* __restrict__ wte, const i32* __restrict__ embed_m, const i8* __restrict__ wpe,
                                 const i32* __restrict__ pos_m, const u32* __restrict__ rows, i32* __restrict__ x,
                                 u32 d, u32 row_bytes, u32 es, u32 ps) {
    u32 j = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (j >= d) return;
    u32 tok = rows[4 * r + 1], pos = rows[4 * r + 2];
    i128 v = rescale(byte_at(wte, (u64)tok * row_bytes + j), embed_m[tok], es) +
             rescale(byte_at(wpe, (u64)pos * row_bytes + j), pos_m[pos], ps);
    x[(u64)r * d + j] = (i32)clamp_wide(v, -RES, RES);
}

// The `norm` op in mode `layer` (hardshell::layernorm). One block of 256 for each row.
extern "C" __global__ void __launch_bounds__(256) layer_norm(const i32* __restrict__ x, const i32* __restrict__ gamma, const i32* __restrict__ beta,
                                i8* __restrict__ out, u32 n, u32 out_stride, u32 k, u32 j, u64 eps, i32 m, u32 s) {
    __shared__ i64 red64[256];
    __shared__ u128 red128[256];
    __shared__ i64 mean_s;
    __shared__ i128 inv_s;
    const u32 t = threadIdx.x, row = blockIdx.x;
    const i32* xr = x + (u64)row * n;
    i64 sum = 0;
    // INVARIANT: |sum| <= 2^31 * 2^16 (hardshell::layernorm bounds).
    for (u32 c = t; c < n; c += 256) sum += xr[c];
    red64[t] = sum;
    __syncthreads();
    for (u32 h = 128; h > 0; h >>= 1) {
        if (t < h) red64[t] += red64[t + h];
        __syncthreads();
    }
    if (t == 0) mean_s = floor_div(red64[0], (i64)n);
    __syncthreads();
    const i64 mean = mean_s;
    u128 sq = 0;
    for (u32 c = t; c < n; c += 256) {
        i64 dv = (i64)xr[c] - mean;
        u64 a = dv < 0 ? (u64)(-dv) : (u64)dv;
        sq += (u128)a * a;
    }
    red128[t] = sq;
    __syncthreads();
    for (u32 h = 128; h > 0; h >>= 1) {
        if (t < h) red128[t] += red128[t + h];
        __syncthreads();
    }
    if (t == 0) {
        u128 var = udiv128_small(red128[0], n);
        u128 root = isqrt_u128((var << (2 * j)) + (u128)eps);
        if (root == 0) root = 1;
        // INVARIANT: root < 2^49 and k <= 48.
        inv_s = (i128)(((u64)1 << k) / (u64)root);
    }
    __syncthreads();
    const i128 inv = inv_s;
    for (u32 c = t; c < n; c += 256) {
        i128 centred = (i128)((i64)xr[c] - mean);
        // INVARIANT: |centred * inv * gamma| < 2^95 and |m| <= 2^31: the product is below 2^126.
        i128 p = centred * inv * (i128)gamma[c] * (i128)m;
        out[(u64)row * out_stride + c] = (i8)clamp_act_i8(shift_round(p, s, 126) + (i128)beta[c]);
    }
}

// The output part of the `linear` op: y = clamp(rescale(acc + b, m[r], s), lo, hi), as i8
// (out_i8) or i32 (out_i32). `splits` partial sums of the product are at acc + z * split_stride;
// their total is the exact product (see gemm_tiled). Grid: (ceil(n / 128), rows).
extern "C" __global__ void linear_out(const i32* __restrict__ acc, const i32* __restrict__ bias, const i32* __restrict__ mult,
                                      i8* __restrict__ out_i8, i32* __restrict__ out_i32, u32 n, u32 acc_stride, u32 out_stride,
                                      u32 shift, i32 lo, i32 hi, u32 splits, u32 split_stride) {
    u32 r = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (r >= n) return;
    i64 v = (i64)bias[r];
    // INVARIANT: the sum of the partials is the dot product, |dot| <= 2^30.
    for (u32 z = 0; z < splits; z++) v += (i64)acc[(u64)z * split_stride + (u64)row * acc_stride + r];
    i32 y = (i32)clamp_wide(rescale(v, mult[r], shift), lo, hi);
    if (out_i8) out_i8[(u64)row * out_stride + r] = (i8)y;
    else out_i32[(u64)row * out_stride + r] = y;
}

// The `add` op: x = clamp(x + delta, -R, R). Grid: (ceil(n / 128), rows).
extern "C" __global__ void add_clamp(i32* __restrict__ x, const i32* __restrict__ delta, u32 n, u32 d_stride) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (c >= n) return;
    i64 s = (i64)x[(u64)row * n + c] + (i64)delta[(u64)row * d_stride + c];
    x[(u64)row * n + c] = (i32)clamp_wide(s, -RES, RES);
}

// The `table` op (GELU): y = t[clamp(x, -offset, len - 1 - offset) + offset], i8 output.
extern "C" __global__ void table_lookup(const i32* __restrict__ in, const i8* __restrict__ tab, i8* __restrict__ out,
                                        u32 n, u32 in_stride, u32 out_stride, i32 offset, i32 len) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (c >= n) return;
    i32 v = in[(u64)row * in_stride + c];
    // INVARIANT: offset <= 2^15 and len <= 2^16.
    i32 lo = -offset, hi = len - 1 - offset;
    v = v < lo ? lo : (v > hi ? hi : v);
    out[(u64)row * out_stride + c] = tab[v + offset];
}

// ---------------------------------------------------------------- attention

// The cache: [slot][head][n_ctx][head_dim] bytes for each layer, so the keys and values of one
// head are contiguous. kv_store copies k and v of each row. Grid: (ceil(d / 128), rows).
extern "C" __global__ void kv_store(const i8* __restrict__ qkv, i8* __restrict__ kc, i8* __restrict__ vc, const u32* __restrict__ rows,
                                    u32 d, u32 qkv_stride, u32 n_ctx, u32 head_dim) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (c >= d) return;
    u32 slot = rows[4 * row], pos = rows[4 * row + 2];
    u32 h = c / head_dim, n_head = d / head_dim;
    u64 at = (((u64)slot * n_head + h) * n_ctx + pos) * head_dim + c % head_dim;
    kc[at] = qkv[(u64)row * qkv_stride + d + c];
    vc[at] = qkv[(u64)row * qkv_stride + 2 * d + c];
}

#define ATT_T 128
// Causal attention for one (head, row) with the softmax of hardshell::softmax.
// Grid: (n_head, rows); block ATT_T (128). `sc` holds n_ctx i64 values for each (row, head).
//
// Scores: one invocation for each position, with 16-byte loads of the key. Sum of p * v:
// invocation t takes the word t % hw (4 channels) of the positions t / hw, t / hw + 128 / hw, ...,
// so the loads of a value row are contiguous; then a tree over the position groups.
extern "C" __global__ void __launch_bounds__(ATT_T) attend(const i8* __restrict__ qkv, const i8* __restrict__ kc, const i8* __restrict__ vc,
                                  i64* __restrict__ sc, const i32* __restrict__ attn_exp, const u32* __restrict__ rows, i8* __restrict__ att,
                                  u32 d, u32 qkv_stride, u32 head_dim, u32 n_ctx, u32 n_head, i32 score_m, u32 score_s,
                                  i32 out_m, u32 out_s, u32 p, u32 table_len, u32 att_stride) {
    __shared__ i64 red[ATT_T];
    __shared__ __align__(16) u32 qw[64];
    __shared__ i64 part[4 * ATT_T];
    __shared__ i64 s_sh[1024];
    __shared__ u32 planes[256][4];
    const u32 t = threadIdx.x, h = blockIdx.x, row = blockIdx.y;
    const u32 slot = rows[4 * row], n = rows[4 * row + 2] + 1;
    // The scores of a context of at most 1,024 positions stay in shared memory.
    i64* s = n_ctx <= 1024 ? s_sh : sc + ((u64)row * n_head + h) * n_ctx;
    const u32 hw = head_dim / 4;
    if (t < hw) qw[t] = ((const u32*)(qkv + (u64)row * qkv_stride + h * head_dim))[t];
    __syncthreads();
    const i8* kb = kc + ((u64)slot * n_head + h) * n_ctx * head_dim;
    i64 mx = -LIMIT40;
    for (u32 jj = t; jj < n; jj += ATT_T) {
        const u32* kr = (const u32*)(kb + (u64)jj * head_dim);
        i32 dot = 0;
        u32 w = 0;
        if ((head_dim & 15) == 0) {
            for (; w < hw; w += 4) {
                uint4 k4 = *(const uint4*)(kr + w);
                dot = __dp4a((i32)qw[w], (i32)k4.x, dot);
                dot = __dp4a((i32)qw[w + 1], (i32)k4.y, dot);
                dot = __dp4a((i32)qw[w + 2], (i32)k4.z, dot);
                dot = __dp4a((i32)qw[w + 3], (i32)k4.w, dot);
            }
        }
        for (; w < hw; w++) dot = __dp4a((i32)qw[w], (i32)kr[w], dot);
        // INVARIANT: |dot| <= head_dim * 2^14 <= 2^22 and |m| <= 2^31, so |dot * m| < 2^62.
        i64 v = rescale_small(dot, score_m, score_s);
        v = v < -LIMIT40 ? -LIMIT40 : (v > LIMIT40 ? LIMIT40 : v);
        s[jj] = v;
        mx = v > mx ? v : mx;
    }
    red[t] = mx;
    __syncthreads();
    for (u32 k = ATT_T / 2; k > 0; k >>= 1) {
        if (t < k) red[t] = red[t + k] > red[t] ? red[t + k] : red[t];
        __syncthreads();
    }
    const i64 row_max = red[0];
    __syncthreads();
    i64 sum = 0;
    for (u32 jj = t; jj < n; jj += ATT_T) {
        // INVARIANT: scores are in +-2^40, so the difference fits and is >= 0.
        i64 dd = row_max - s[jj];
        i64 idx = dd < (i64)table_len - 1 ? dd : (i64)table_len - 1;
        i64 e = attn_exp[idx];
        s[jj] = e;
        // INVARIANT: n <= 2^16 entries below 2^31.
        sum += e;
    }
    red[t] = sum;
    __syncthreads();
    for (u32 k = ATT_T / 2; k > 0; k >>= 1) {
        if (t < k) red[t] += red[t + k];
        __syncthreads();
    }
    u64 total = (u64)red[0];
    if (total == 0) total = 1;
    // p[j] = floor(e * 2^P / total) with an exact reciprocal: m = floor(2^(P+32) / total) and
    // est = floor(e * m / 2^32). Then est <= e * 2^P / total and the error is below e / 2^32 < 1,
    // so the exact quotient is est or est + 1; one exact check selects it.
    const u64 recip = ((u64)1 << (p + 32)) / total;
    for (u32 jj = t; jj < n; jj += ATT_T) {
        // INVARIANT: e < 2^31 and p <= 30, so e << p < 2^61 and est * total <= e << p.
        const u64 e = (u64)s[jj];
        u64 est = (u64)(((u128)e * recip) >> 32);
        const u64 num = e << p;
        if (num - est * total >= total) est += 1;
        s[jj] = (i64)est;
    }
    __syncthreads();
    const i8* vb = vc + ((u64)slot * n_head + h) * n_ctx * head_dim;
    const u32 groups = ATT_T / hw, grp = t / hw, w = t % hw;
    i64 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (n_ctx <= 1024 && (n_ctx & 3) == 0) {
        // The byte planes of p: planes[q][b] holds byte b of p[4q .. 4q + 3] (0 after n).
        const u32 nq = (n + 3) / 4;
        for (u32 q = t; q < nq; q += ATT_T) {
            u32 pw[4] = {0, 0, 0, 0};
#pragma unroll
            for (u32 k = 0; k < 4; k++) {
                u32 pj = 4 * q + k < n ? (u32)s[4 * q + k] : 0;
#pragma unroll
                for (u32 b = 0; b < 4; b++) pw[b] |= ((pj >> (8 * b)) & 0xffu) << (8 * k);
            }
#pragma unroll
            for (u32 b = 0; b < 4; b++) planes[q][b] = pw[b];
        }
        __syncthreads();
        // sum_j p[j] * v[j][c] = sum_b 2^(8b) * sum_j p_b[j] * v[j][c]: dp4a of an unsigned byte
        // plane and four signed values. INVARIANT: each i32 sum has at most n <= 2^16 terms of
        // magnitude <= 255 * 128 < 2^15, so it stays in -2^31 .. 2^31 - 1.
        i32 acc[4][4];
#pragma unroll
        for (int c = 0; c < 4; c++)
#pragma unroll
            for (int b = 0; b < 4; b++) acc[c][b] = 0;
        if (grp < groups) {
            for (u32 q = grp; q < nq; q += groups) {
                const u32* base = (const u32*)(vb + (u64)(4 * q) * head_dim) + w;
                // INVARIANT: 4q + 3 < n_ctx (n_ctx is a multiple of 4), so the loads are inside
                // the cache of the slot; values after n meet a zero plane byte.
                u32 x0 = base[0], x1 = base[hw], x2 = base[2 * hw], x3 = base[3 * hw];
                u32 t0 = __byte_perm(x0, x1, 0x5140), t1 = __byte_perm(x2, x3, 0x5140);
                u32 t2 = __byte_perm(x0, x1, 0x7362), t3 = __byte_perm(x2, x3, 0x7362);
                u32 y[4] = {__byte_perm(t0, t1, 0x5410), __byte_perm(t0, t1, 0x7632),
                            __byte_perm(t2, t3, 0x5410), __byte_perm(t2, t3, 0x7632)};
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const u32 pl = planes[q][b];
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        asm("dp4a.u32.s32 %0, %1, %2, %0;" : "+r"(acc[c][b]) : "r"(pl), "r"(y[c]));
                    }
                }
            }
        }
        // INVARIANT: |acc| < 2^31, so acc * 2^24 < 2^55.
        a0 = (i64)acc[0][0] + (i64)acc[0][1] * 256 + (i64)acc[0][2] * 65536 + (i64)acc[0][3] * 16777216;
        a1 = (i64)acc[1][0] + (i64)acc[1][1] * 256 + (i64)acc[1][2] * 65536 + (i64)acc[1][3] * 16777216;
        a2 = (i64)acc[2][0] + (i64)acc[2][1] * 256 + (i64)acc[2][2] * 65536 + (i64)acc[2][3] * 16777216;
        a3 = (i64)acc[3][0] + (i64)acc[3][1] * 256 + (i64)acc[3][2] * 65536 + (i64)acc[3][3] * 16777216;
    } else if (grp < groups) {
        // INVARIANT: sum(p) <= 2^30 and |v| <= 128, so each sum is at most 2^37.
        for (u32 jj = grp; jj < n; jj += groups) {
            // INVARIANT: 0 <= p <= 2^30 fits in i32, so each product is a widening multiply.
            const i32 pj = (i32)s[jj];
            const u32 v4 = ((const u32*)(vb + (u64)jj * head_dim))[w];
            a0 += (i64)pj * (i32)sbyte(v4);
            a1 += (i64)pj * (i32)sbyte(v4 >> 8);
            a2 += (i64)pj * (i32)sbyte(v4 >> 16);
            a3 += (i64)pj * (i32)sbyte(v4 >> 24);
        }
    }
    if (grp < groups) {
        part[4 * t] = a0;
        part[4 * t + 1] = a1;
        part[4 * t + 2] = a2;
        part[4 * t + 3] = a3;
    }
    __syncthreads();
    for (u32 c = t; c < head_dim; c += ATT_T) {
        i64 tot = 0;
        for (u32 gg = 0; gg < groups; gg++) tot += part[4 * (gg * hw) + c];
        att[(u64)row * att_stride + h * head_dim + c] = (i8)clamp_act_i8(rescale(tot, out_m, out_s));
    }
}

// Writes the GPU global timer (nanoseconds) to out[i]: a profiler mark in the stream. It changes
// no model value. One thread.
extern "C" __global__ void stamp(u64* __restrict__ out, u32 i) {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    out[i] = t;
}

// ---------------------------------------------------------------- logits and the coder

// logit = clamp(rescale(acc, m[v], s), -2^40, 2^40). Grid: (ceil(vocab / 128), rows).
extern "C" __global__ void logits_scale(const i32* __restrict__ acc, const i32* __restrict__ mult, i64* __restrict__ logit,
                                        u32 vocab, u32 acc_stride, u32 shift) {
    u32 v = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (v >= vocab) return;
    logit[(u64)row * vocab + v] =
        clamp_wide(rescale(acc[(u64)row * acc_stride + v], mult[v], shift), -LIMIT40, LIMIT40);
}

// weight = logits_exp[min(max - logit, len - 1)]. One block of 256 for each row.
extern "C" __global__ void __launch_bounds__(256) logits_weights(const i64* __restrict__ logit, const u64* __restrict__ logits_exp,
                                          u64* __restrict__ weights, u32 vocab, u32 table_len) {
    __shared__ i64 red[256];
    const u32 t = threadIdx.x, row = blockIdx.x;
    const i64* l = logit + (u64)row * vocab;
    i64 mx = -LIMIT40;
    for (u32 v = t; v < vocab; v += 256) mx = l[v] > mx ? l[v] : mx;
    red[t] = mx;
    __syncthreads();
    for (u32 k = 128; k > 0; k >>= 1) {
        if (t < k) red[t] = red[t + k] > red[t] ? red[t + k] : red[t];
        __syncthreads();
    }
    const i64 m = red[0];
    for (u32 v = t; v < vocab; v += 256) {
        i64 dd = m - l[v];
        i64 idx = dd < (i64)table_len - 1 ? dd : (i64)table_len - 1;
        weights[(u64)row * vocab + v] = logits_exp[idx];
    }
}

// floor(x / d) for a u128 x and a u64 d >= 1 with x / d < 2^64: 128 rounds of long division.
// Use it once for each row.
__device__ u64 udiv128_64(u128 x, u64 d) {
    u128 r = 0;
    u64 q = 0;
    for (int i = 127; i >= 0; i--) {
        r = (r << 1) | (u128)((x >> i) & 1);
        q <<= 1;
        if (r >= d) {
            r -= d;
            q |= 1;
        }
    }
    return q;
}

// floor(w * spare / big_w) for w <= big_w, with frac = floor(spare * 2^64 / big_w) when
// big_w >= 2^32 (frac then fits in u64). est = floor(w * frac / 2^64) is at most the quotient,
// and the error is below w / 2^64 < 1, so one exact check finds the quotient.
__device__ __forceinline__ u64 quot_fast(u64 w, u64 big_w, u64 spare, u64 frac) {
    u64 est = (u64)(((u128)w * frac) >> 64);
    // INVARIANT: est * big_w <= w * spare < 2^96.
    if ((u128)w * spare - (u128)est * big_w >= (u128)big_w) est += 1;
    return est;
}

// floor(w * spare / big_w) for w <= big_w: below 2^32. 32 rounds of long division.
__device__ __forceinline__ u64 quot(u64 w, u64 big_w, u64 spare) {
    u128 pr = (u128)w * spare;
    u64 r = (u64)(pr >> 32);
    u32 lo = (u32)pr;
    u64 q = 0;
    for (int i = 31; i >= 0; i--) {
        bool top = (r >> 63) != 0;
        r = (r << 1) | ((lo >> i) & 1u);
        q <<= 1;
        if (top || r >= big_w) {
            r -= big_w;
            q |= 1;
        }
    }
    return q;
}

// The coder interval (taco_coder::FreqTable::from_weights, plan §5.1) of one query for each
// row: kind 0 encodes a token, kind 1 decodes a target. Output: symbol, start, freq (u64).
// One block of 256 for each row. Each of the 8 warps owns a contiguous chunk of symbols; the 32
// lanes of a warp read consecutive symbols. All sums are exact u64 sums.
__device__ __forceinline__ u64 freq_at(const u64* w, u32 i, bool zero, u64 wsum, u64 spare, u64 frac, u32 largest, u64 rem) {
    const u64 wi = zero ? 1 : w[i];
    return 1 + (frac ? quot_fast(wi, wsum, spare, frac) : quot(wi, wsum, spare)) + (i == largest ? rem : 0);
}

extern "C" __global__ void __launch_bounds__(256) coder_interval(const u64* __restrict__ weights, const u32* __restrict__ queries,
                                          u64* __restrict__ out, u32 vocab, u64 total) {
    __shared__ u64 cs[256];
    __shared__ u64 bw[256];
    __shared__ u32 bi[256];
    __shared__ u64 wprefix[9];
    __shared__ u64 wsum_s, rem_s, frac_s;
    __shared__ u32 largest_s, zero_s;
    const u32 t = threadIdx.x, row = blockIdx.x, lane = t & 31, warp = t >> 5;
    const u64* w = weights + (u64)row * vocab;
    const u32 c = (vocab + 7) / 8;
    const u32 lo = min(warp * c, vocab), hi = min(lo + c, vocab);
    const u64 spare = total - vocab;
    // Pass 1: the weight sum, and the largest weight with the lowest id.
    u64 s = 0, best = 0;
    u32 besti = 0xffffffffu;
    for (u32 i = lo + lane; i < hi; i += 32) {
        const u64 wi = w[i];
        s += wi;
        // NOTE: each lane sees its ids in increasing order, so `>` keeps the lowest id of a tie.
        if (besti == 0xffffffffu || wi > best) {
            best = wi;
            besti = i;
        }
    }
    cs[t] = s;
    bw[t] = best;
    bi[t] = besti;
    __syncthreads();
    for (u32 h = 128; h > 0; h >>= 1) {
        if (t < h) {
            cs[t] += cs[t + h];
            const bool other_valid = bi[t + h] != 0xffffffffu;
            const bool mine_valid = bi[t] != 0xffffffffu;
            if (other_valid && (!mine_valid || bw[t + h] > bw[t] || (bw[t + h] == bw[t] && bi[t + h] < bi[t]))) {
                bw[t] = bw[t + h];
                bi[t] = bi[t + h];
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        // INVARIANT: weights <= 2^40 and vocab <= 2^20, so the u64 sum cannot overflow.
        zero_s = cs[0] == 0;
        wsum_s = cs[0] == 0 ? vocab : cs[0];
        // Rule 1: with all weights 0, symbol 0 is the largest. Otherwise the largest weight
        // (> 0) with the lowest id, as the CPU loop that keeps the first strictly larger weight.
        largest_s = cs[0] == 0 ? 0 : bi[0];
        rem_s = 0;
        // NOTE: frac = 0 selects the long division (a weight sum below 2^32).
        frac_s = wsum_s >= ((u64)1 << 32) ? udiv128_64((u128)spare << 64, wsum_s) : 0;
    }
    __syncthreads();
    const u64 wsum = wsum_s, frac = frac_s;
    const u32 largest = largest_s;
    const bool zero = zero_s != 0;
    // Pass 2: the sum of f in each warp chunk, before the remainder.
    u64 f_sum = 0;
    for (u32 i = lo + lane; i < hi; i += 32) f_sum += freq_at(w, i, zero, wsum, spare, frac, largest, 0);
    for (int off = 16; off > 0; off >>= 1) f_sum += __shfl_xor_sync(0xffffffffu, f_sum, off);
    if (lane == 0) cs[warp] = f_sum;
    __syncthreads();
    if (t == 0) {
        u64 acc = 0;
        for (u32 j = 0; j < 8; j++) {
            wprefix[j] = acc;
            acc += cs[j];
        }
        // INVARIANT: sum(f) <= vocab + spare = total.
        rem_s = total - acc;
        for (u32 j = 0; j < 8; j++)
            if (largest < j * c) wprefix[j] += rem_s;
        wprefix[8] = total;
    }
    __syncthreads();
    if (lo >= hi) return;
    const u32 kind = queries[4 * row], q = queries[4 * row + 1];
    bool mine;
    if (kind == 0) {
        mine = q >= lo && q < hi;
    } else {
        const u64 next = hi < vocab ? wprefix[warp + 1] : total;
        mine = (u64)q >= wprefix[warp] && (u64)q < next;
    }
    if (!mine) return;
    // Pass 3, in the owning warp only: groups of 32 symbols with an inclusive warp scan.
    const u64 rem = rem_s;
    u64 cum = wprefix[warp];
    for (u32 base = lo; base < hi; base += 32) {
        const u32 i = base + lane;
        const u64 f = i < hi ? freq_at(w, i, zero, wsum, spare, frac, largest, rem) : 0;
        u64 incl = f;
        for (int off = 1; off < 32; off <<= 1) {
            const u64 v = __shfl_up_sync(0xffffffffu, incl, off);
            if ((int)lane >= off) incl += v;
        }
        const u64 start = cum + incl - f;
        const bool hit = i < hi && (kind == 0 ? i == q : ((u64)q >= start && (u64)q < start + f));
        const unsigned ballot = __ballot_sync(0xffffffffu, hit);
        if (ballot != 0) {
            if (hit) {
                out[3 * row] = i;
                out[3 * row + 1] = start;
                out[3 * row + 2] = f;
            }
            return;
        }
        cum += __shfl_sync(0xffffffffu, incl, 31);
    }
}
