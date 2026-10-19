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
typedef unsigned char u8;
typedef short i16;
typedef unsigned short u16;
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

// clamp(rescale(x, m, s), lo, hi) with lo <= hi, for |x * m| <= 2^94: the expression of the op
// kernels, for the fused kernels. For |x| < 2^31, |x * m| < 2^62, so rescale_small gives the value
// of rescale (see rescale_small), with i64 steps only; otherwise the i128 form runs. Both branches
// give the same value; tests/fused.rs compares the fused kernels with the op kernels, which use
// the i128 form.
__device__ __forceinline__ i64 rescale_clamp(i64 x, i32 m, u32 s, i64 lo, i64 hi) {
    if (x > -2147483648LL && x < 2147483648LL) {
        const i64 v = rescale_small(x, m, s);
        return v < lo ? lo : (v > hi ? hi : v);
    }
    return clamp_wide(rescale(x, m, s), lo, hi);
}

// hardshell::int::floor_div for b >= 1. C++ division truncates toward zero.
__device__ __forceinline__ i64 floor_div(i64 a, i64 b) {
    i64 q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

// floor(sqrt(x)) for x < 2^64: the digit-by-digit method of isqrt_u128 in u64 steps. The method
// gives floor(sqrt(x)) for every x, so both functions give the same value.
__device__ u64 isqrt_u64(u64 x) {
    if (x == 0) return 0;
    u64 bit = (u64)1 << ((63 - __clzll(x)) & ~1);
    // INVARIANT: res < 2^32 and bit <= 2^62, so res + bit does not overflow.
    u64 rem = x, res = 0;
    while (bit != 0) {
        const u64 trial = res + bit;
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

// hardshell::int::isqrt_u128: the digit-by-digit method (in u64 steps for x < 2^64).
__device__ u128 isqrt_u128(u128 x) {
    if ((u64)(x >> 64) == 0) return (u128)isqrt_u64((u64)x);
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
    // NOTE: one u64 division for x < 2^64 (the same quotient).
    if (hi == 0) return (u128)(lo / n);
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

__device__ __forceinline__ i64 warp_max_i64(i64 v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const i64 o = __shfl_xor_sync(0xffffffffu, v, off);
        v = o > v ? o : v;
    }
    return v;
}

__device__ __forceinline__ i64 warp_sum_i64(i64 v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
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

// cp.async of 16 bytes; with src_bytes 0 it writes 16 zero bytes and reads nothing.
__device__ __forceinline__ void cp_async16(u32 dst, const void* src, u32 src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}

// Four 8 x 8 matrices of 16-bit lanes from shared memory (here: 8 rows of 16 i8 values each).
__device__ __forceinline__ void ldmatrix_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// A tile in shared memory has rows of 64 bytes (4 chunks of 16 bytes). Chunk c of row r is at
// chunk c ^ ((r >> 1) & 3), so the 8 rows of one ldmatrix read hit 32 different banks.
__device__ __forceinline__ u32 swz(u32 r, u32 c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); }

// The same product as gemm_tiled with larger tiles: block tile BM (m) x BN (n), warp tile WM x WN,
// STAGES stages of 64 bytes of k in flight (cp.async), fragments with ldmatrix. Rows of `a` at m
// or more and rows of `w` at n or more are read as zero (cp.async zero fill), so the buffers need
// no padding rows. The exactness argument and the split-K rule are the ones of gemm_tiled.
// Grid: (ceil(m / BM), ceil(n / BN), Z); block: 32 * (BM / WM) * (BN / WN) threads.
//
// With LOGITS, the epilogue is the logits op (logits_scale) of each product and the first part of
// logits_weights, fused, in a compact form for the coder op (coder_d16). For each row and each
// tile of BN columns: the tile max tm of the logits into tmax[row][tile], and for each column
// d = min(tm - logit, last) as u16 into d16 (row stride out_stride), with last = len - 1 of the
// logits table. Then min(max - logit, last) = min((max - tm) + d, last) for the row max `max`:
// if d < last, d = tm - logit; else tm - logit >= last, and both sides are `last`.
// This needs one split (gridDim.z = 1).
template <int BM, int BN, int WM, int WN, int STAGES, bool LOGITS>
__device__ __forceinline__ void gemm_tc_body(const u32* __restrict__ w, const u32* __restrict__ a, i32* __restrict__ out, u32 n,
                                             u32 kw, u32 out_stride, u32 m, u32 split_stride, u8* smem, const i32* __restrict__ mult,
                                             u32 shift, u16* __restrict__ d16, i64* __restrict__ tmax, u32 last) {
    constexpr u32 WARPS_M = BM / WM, THREADS = 32 * (BM / WM) * (BN / WN);
    constexpr u32 MI = WM / 16, NJ = WN / 8;
    constexpr u32 A_BYTES = BM * 64, STAGE_BYTES = (BM + BN) * 64;
    const u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const u32 wm = warp % WARPS_M, wn = warp / WARPS_M;
    const u32 g = lane >> 2, t = lane & 3;
    const u32 m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const u32 stages = kw / 16, z = blockIdx.z, zs = gridDim.z;
    const u32 s_begin = (u32)((u64)stages * z / zs), s_end = (u32)((u64)stages * (z + 1) / zs);
    const u32 sbase = (u32)__cvta_generic_to_shared(smem);
    i32 acc[MI][NJ][4];
#pragma unroll
    for (u32 i = 0; i < MI; i++)
#pragma unroll
        for (u32 j = 0; j < NJ; j++)
#pragma unroll
            for (u32 c = 0; c < 4; c++) acc[i][j][c] = 0;
    auto load = [&](u32 stage, u32 buf) {
        const u32 sa = sbase + buf * STAGE_BYTES, sw = sa + A_BYTES;
#pragma unroll
        for (u32 idx = tid; idx < BM * 4; idx += THREADS) {
            const u32 r = idx >> 2, c = idx & 3, gr = m0 + r;
            const u32* src = a + (u64)(gr < m ? gr : 0) * kw + stage * 16 + c * 4;
            cp_async16(sa + swz(r, c), src, gr < m ? 16 : 0);
        }
#pragma unroll
        for (u32 idx = tid; idx < BN * 4; idx += THREADS) {
            const u32 r = idx >> 2, c = idx & 3, gr = n0 + r;
            const u32* src = w + (u64)(gr < n ? gr : 0) * kw + stage * 16 + c * 4;
            cp_async16(sw + swz(r, c), src, gr < n ? 16 : 0);
        }
    };
#pragma unroll
    for (u32 s = 0; s + 1 < STAGES; s++) {
        if (s_begin + s < s_end) load(s_begin + s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (u32 st = s_begin; st < s_end; st++) {
        // The group of stage st is complete when at most STAGES - 2 newer groups are pending.
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
        __syncthreads();
        // The buffer of stage st - 1 is free: every thread passed the barrier after its products.
        const u32 nx = st + STAGES - 1;
        if (nx < s_end) load(nx, (nx - s_begin) % STAGES);
        asm volatile("cp.async.commit_group;\n" ::);
        const u32 sa = sbase + ((st - s_begin) % STAGES) * STAGE_BYTES, sw = sa + A_BYTES;
#pragma unroll
        for (u32 kk = 0; kk < 2; kk++) {
            u32 af[MI][4], bf[NJ][2];
#pragma unroll
            for (u32 i = 0; i < MI; i++)
                ldmatrix_x4(sa + swz(wm * WM + i * 16 + (lane & 15), 2 * kk + (lane >> 4)), af[i][0], af[i][1], af[i][2], af[i][3]);
#pragma unroll
            for (u32 jp = 0; jp < NJ / 2; jp++)
                ldmatrix_x4(sw + swz(wn * WN + jp * 16 + (lane & 7) + 8 * (lane >> 4), 2 * kk + ((lane >> 3) & 1)), bf[2 * jp][0],
                            bf[2 * jp][1], bf[2 * jp + 1][0], bf[2 * jp + 1][1]);
#pragma unroll
            for (u32 i = 0; i < MI; i++)
#pragma unroll
                for (u32 j = 0; j < NJ; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    if (LOGITS) {
        // logit(acc, col) = clamp(rescale(acc, mult[col], shift), -2^40, 2^40), the logits_scale
        // expression. INVARIANT: |acc| <= 2^30 and |m| < 2^31, so |acc * m| < 2^61 and
        // rescale_small gives the value of rescale (see rescale_small).
        auto logit_of = [&](i32 v, u32 cc) -> i64 {
            const i64 x = rescale_small((i64)v, mult[cc], shift);
            return x < -LIMIT40 ? -LIMIT40 : (x > LIMIT40 ? LIMIT40 : x);
        };
        // The tile max of each row: the values of this thread, the 4 lanes of a row (xor 1, 2),
        // then the WARPS_N warps of a row through shared memory (the pipeline buffers are free).
        constexpr u32 WARPS_N = BN / WN;
        i64* red = (i64*)smem;
        __syncthreads();
#pragma unroll
        for (u32 i = 0; i < MI; i++) {
#pragma unroll
            for (u32 h = 0; h < 2; h++) {
                const u32 rl = wm * WM + i * 16 + g + 8 * h;
                i64 mx = -LIMIT40;
#pragma unroll
                for (u32 j = 0; j < NJ; j++) {
                    const u32 col = n0 + wn * WN + j * 8 + t * 2;
                    if (col < n) mx = max(mx, logit_of(acc[i][j][2 * h], col));
                    if (col + 1 < n) mx = max(mx, logit_of(acc[i][j][2 * h + 1], col + 1));
                }
                mx = max(mx, (i64)__shfl_xor_sync(0xffffffffu, mx, 1));
                mx = max(mx, (i64)__shfl_xor_sync(0xffffffffu, mx, 2));
                if (t == 0) red[rl * WARPS_N + wn] = mx;
            }
        }
        __syncthreads();
#pragma unroll
        for (u32 i = 0; i < MI; i++) {
#pragma unroll
            for (u32 h = 0; h < 2; h++) {
                const u32 rl = wm * WM + i * 16 + g + 8 * h, r = m0 + rl;
                i64 tm = red[rl * WARPS_N];
#pragma unroll
                for (u32 k = 1; k < WARPS_N; k++) tm = max(tm, red[rl * WARPS_N + k]);
                if (r >= m) continue;
                if (wn == 0 && t == 0) tmax[(u64)r * gridDim.y + blockIdx.y] = tm;
#pragma unroll
                for (u32 j = 0; j < NJ; j++) {
#pragma unroll
                    for (u32 e = 0; e < 2; e++) {
                        const u32 cc = n0 + wn * WN + j * 8 + t * 2 + e;
                        if (cc >= n) continue;
                        // INVARIANT: logit <= tm, and both are in +-2^40.
                        const i64 dd = tm - logit_of(acc[i][j][2 * h + e], cc);
                        d16[(u64)r * out_stride + cc] = (u16)(dd < (i64)last ? dd : (i64)last);
                    }
                }
            }
        }
        return;
    }
    i32* o = out + (u64)z * split_stride;
#pragma unroll
    for (u32 i = 0; i < MI; i++) {
#pragma unroll
        for (u32 j = 0; j < NJ; j++) {
            const u32 col = n0 + wn * WN + j * 8 + t * 2;
            const u32 r0 = m0 + wm * WM + i * 16 + g, r1 = r0 + 8;
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

// 128 x 128 tiles, 8 warps of 64 x 32, 3 stages (48 KB of shared memory).
extern "C" __global__ void __launch_bounds__(256, 2) gemm_tc128(const u32* __restrict__ w, const u32* __restrict__ a, i32* __restrict__ out,
                                                                u32 n, u32 kw, u32 out_stride, u32 m, u32 split_stride) {
    __shared__ __align__(128) u8 smem[3 * 256 * 64];
    gemm_tc_body<128, 128, 64, 32, 3, false>(w, a, out, n, kw, out_stride, m, split_stride, smem, nullptr, 0, nullptr, nullptr, 0);
}

extern "C" __global__ void __launch_bounds__(256, 2) gemm_tc128_logits(const u32* __restrict__ w, const u32* __restrict__ a,
                                                                       const i32* __restrict__ mult, u16* __restrict__ d16,
                                                                       i64* __restrict__ tmax, u32 n, u32 kw, u32 out_stride, u32 m,
                                                                       u32 shift, u32 last) {
    __shared__ __align__(128) u8 smem[3 * 256 * 64];
    gemm_tc_body<128, 128, 64, 32, 3, true>(w, a, nullptr, n, kw, out_stride, m, 0, smem, mult, shift, d16, tmax, last);
}

// 64 x 128 tiles, 8 warps of 32 x 32, 4 stages (48 KB of shared memory).
extern "C" __global__ void __launch_bounds__(256, 2) gemm_tc64(const u32* __restrict__ w, const u32* __restrict__ a, i32* __restrict__ out,
                                                               u32 n, u32 kw, u32 out_stride, u32 m, u32 split_stride) {
    __shared__ __align__(128) u8 smem[4 * 192 * 64];
    gemm_tc_body<64, 128, 32, 32, 4, false>(w, a, out, n, kw, out_stride, m, split_stride, smem, nullptr, 0, nullptr, nullptr, 0);
}

extern "C" __global__ void __launch_bounds__(256, 2) gemm_tc64_logits(const u32* __restrict__ w, const u32* __restrict__ a,
                                                                      const i32* __restrict__ mult, u16* __restrict__ d16,
                                                                      i64* __restrict__ tmax, u32 n, u32 kw, u32 out_stride, u32 m,
                                                                      u32 shift, u32 last) {
    __shared__ __align__(128) u8 smem[4 * 192 * 64];
    gemm_tc_body<64, 128, 32, 32, 4, true>(w, a, nullptr, n, kw, out_stride, m, 0, smem, mult, shift, d16, tmax, last);
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

// PERF: 128 threads for each row: 224 rows of 256 threads need two waves on 30 SMs (48 warps
// each), 128 threads one (est.; not timed alone, see docs/research/2026-10-18-gpu-backend.md section 7).
#define NORM_T 128
#define NORM_W (NORM_T / 32)

// The exact u128 sum over a warp, with u64 halves and a carry.
__device__ __forceinline__ u128 warp_sum_u128(u128 v) {
    u64 lo = (u64)v, hi = (u64)(v >> 64);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const u64 olo = __shfl_xor_sync(0xffffffffu, lo, off), ohi = __shfl_xor_sync(0xffffffffu, hi, off);
        const u64 nlo = lo + olo;
        hi = hi + ohi + (nlo < lo ? 1 : 0);
        lo = nlo;
    }
    return ((u128)hi << 64) | (u128)lo;
}

// The value of step 4 of the norm before the output clamp, for the product
// pr = centred * inv * gamma * m. frac 0 (model-file version 1): shift_round(pr, s, 126) + beta, two
// roundings. frac F > 0 (version 2, ln_beta_frac, ADR 0004 addendum B): beta is in units of 2^-F
// output units and enters before the one rounding, shift_round(pr + beta * 2^(s - F), s, 126):
// hardshell::int::rescale_wide_offset.
// INVARIANT: |pr| < 2^126 (see norm_row). For F > 0 the loader checked 1 <= F <= s <= 120 and
// s - F <= 92, so |beta * 2^(s - F)| < 2^123, and |pr + c| + 2^(s - 1) < 2^127: no step overflows.
__device__ __forceinline__ i128 norm_value(i128 pr, i32 beta, u32 s, u32 frac) {
    if (frac == 0) return shift_round(pr, s, 126) + (i128)beta;
    return shift_round(pr + (i128)beta * ((i128)1 << (s - frac)), s, 126);
}

// Stores the i16 activation v (-32768..32767) of column c as two i8 planes (the operand form of the
// i8 x i16 product, wide_dot): hi = floor(v / 256) and lo = (v mod 256) - 128, so that
// v = 256 * hi + lo + 128. Both are in -128..127.
__device__ __forceinline__ void plane_store(i8* __restrict__ hi, i8* __restrict__ lo, u64 c, i32 v) {
    // INVARIANT: -32768 <= v <= 32767, so 0 <= u < 2^16 and both bytes are in 0..255.
    const u32 u = (u32)(v + 32768);
    hi[c] = (i8)((i32)(u >> 8) - 128);
    lo[c] = (i8)((i32)(u & 255u) - 128);
}

// The output of the norm for column c: an i8 activation (clamp to -127..127) into orow, or, with
// lorow set (ln_f_bits = 16), an i16 activation (clamp to -32767..32767, LayerNorm::apply_i16) as two
// planes: hi into orow, lo into lorow (plane_store).
__device__ __forceinline__ void norm_store(i8* __restrict__ orow, i8* __restrict__ lorow, u32 c, i128 y) {
    if (lorow == nullptr) {
        orow[c] = (i8)clamp_act_i8(y);
        return;
    }
    plane_store(orow, lorow, c, (i32)clamp_wide(y, -32767, 32767));
}

// The `norm` op in mode `layer` (hardshell::layernorm) for one row `xr` of n values, by a block of
// NORM_T threads. Thread t reads only the columns t, t + NORM_T, ..., so a caller that wrote these
// columns of xr in the same thread needs no barrier before the call.
__device__ void norm_row(const i32* __restrict__ xr, const i32* __restrict__ gamma, const i32* __restrict__ beta, i8* __restrict__ orow,
                         i8* __restrict__ lorow, u32 n, u32 k, u32 j, u64 eps, i32 m, u32 s, u32 frac) {
    __shared__ i64 red64[NORM_W];
    __shared__ u128 red128[NORM_W];
    __shared__ i128 inv_s;
    const u32 t = threadIdx.x, lane = t & 31, warp = t >> 5;
    i64 sum = 0;
    // INVARIANT: |sum| <= 2^31 * 2^16 (hardshell::layernorm bounds).
    for (u32 c = t; c < n; c += NORM_T) sum += xr[c];
    sum = warp_sum_i64(sum);
    if (lane == 0) red64[warp] = sum;
    __syncthreads();
    i64 tot = 0;
#pragma unroll
    for (u32 w = 0; w < NORM_W; w++) tot += red64[w];
    const i64 mean = floor_div(tot, (i64)n);
    u128 sq = 0;
    for (u32 c = t; c < n; c += NORM_T) {
        // INVARIANT: |x| <= 2^30 - 1 (the residual clamp) and |mean| <= 2^30 - 1, so |dv| < 2^31
        // and dv^2 < 2^62.
        const i64 dv = (i64)xr[c] - mean;
        const u64 av = dv < 0 ? (u64)(-dv) : (u64)dv;
        sq += (u128)(av * av);
    }
    sq = warp_sum_u128(sq);
    if (lane == 0) red128[warp] = sq;
    __syncthreads();
    if (t == 0) {
        u128 tq = 0;
#pragma unroll
        for (u32 w = 0; w < NORM_W; w++) tq += red128[w];
        u128 var = udiv128_small(tq, n);
        u128 root = isqrt_u128((var << (2 * j)) + (u128)eps);
        if (root == 0) root = 1;
        // INVARIANT: root < 2^49 and k <= 48.
        inv_s = (i128)(((u64)1 << k) / (u64)root);
    }
    __syncthreads();
    const i128 inv = inv_s;
    for (u32 c = t; c < n; c += NORM_T) {
        const i128 centred = (i128)((i64)xr[c] - mean);
        // INVARIANT: |centred * inv * gamma| < 2^95 and |m| <= 2^31: the product is below 2^126.
        const i128 pr = centred * inv * (i128)gamma[c] * (i128)m;
        norm_store(orow, lorow, c, norm_value(pr, beta[c], s, frac));
    }
}

// The values of a row that one thread keeps in registers in norm_regs: rows of at most
// NORM_T * NORM_V values.
#define NORM_V 8

// norm_row for a row whose values are in registers: v[i] is column t + i * NORM_T (0 after n).
// The same expressions as norm_row, with no global read of the row. The product of the last step
// is centred * inv * (gamma * m): the same integer (integer products are associative), with
// |gamma * m| < 2^62 in i64 and |centred * inv| < 2^79 (|centred| < 2^31, inv < 2^48).
__device__ void norm_regs(const i32 (&v)[NORM_V], const i32* __restrict__ gamma, const i32* __restrict__ beta, i8* __restrict__ orow,
                          i8* __restrict__ lorow, u32 n, u32 k, u32 j, u64 eps, i32 m, u32 s, u32 frac) {
    __shared__ i64 red64[NORM_W];
    __shared__ u128 red128[NORM_W];
    __shared__ i128 inv_s;
    const u32 t = threadIdx.x, lane = t & 31, warp = t >> 5;
    i64 sum = 0;
    // INVARIANT: |sum| <= 2^31 * 2^16 (hardshell::layernorm bounds).
#pragma unroll
    for (u32 i = 0; i < NORM_V; i++) sum += v[i];
    sum = warp_sum_i64(sum);
    if (lane == 0) red64[warp] = sum;
    __syncthreads();
    i64 tot = 0;
#pragma unroll
    for (u32 w = 0; w < NORM_W; w++) tot += red64[w];
    const i64 mean = floor_div(tot, (i64)n);
    u128 sq = 0;
#pragma unroll
    for (u32 i = 0; i < NORM_V; i++) {
        if (t + i * NORM_T < n) {
            // INVARIANT: |dv| < 2^31 and dv^2 < 2^62 (see norm_row).
            const i64 dv = (i64)v[i] - mean;
            const u64 av = dv < 0 ? (u64)(-dv) : (u64)dv;
            sq += (u128)(av * av);
        }
    }
    sq = warp_sum_u128(sq);
    if (lane == 0) red128[warp] = sq;
    __syncthreads();
    if (t == 0) {
        u128 tq = 0;
#pragma unroll
        for (u32 w = 0; w < NORM_W; w++) tq += red128[w];
        u128 var = udiv128_small(tq, n);
        u128 root = isqrt_u128((var << (2 * j)) + (u128)eps);
        if (root == 0) root = 1;
        // INVARIANT: root < 2^49 and k <= 48.
        inv_s = (i128)(((u64)1 << k) / (u64)root);
    }
    __syncthreads();
    const i64 inv = (i64)inv_s;
#pragma unroll
    for (u32 i = 0; i < NORM_V; i++) {
        const u32 c = t + i * NORM_T;
        if (c < n) {
            const i128 ci = (i128)((i64)v[i] - mean) * (i128)inv;
            // INVARIANT: |gamma * m| < 2^62, and the product is below 2^126 (see norm_row).
            const i128 pr = ci * (i128)((i64)gamma[c] * (i64)m);
            norm_store(orow, lorow, c, norm_value(pr, beta[c], s, frac));
        }
    }
}

// The `norm` op in mode `layer`. One block of NORM_T for each row. `frac` is ln_beta_frac (norm_value).
// With wide = 0 the output is i8 (row r at out + r * out_stride). With wide = 1 (ln_f_bits = 16) it
// is i16 in two planes (plane_store): the hi plane of row r at row r of `out`, the lo plane at row
// gridDim.x + r, so the planes of a pass of n rows are the 2n operand rows of one i8 product.
extern "C" __global__ void __launch_bounds__(NORM_T) layer_norm(const i32* __restrict__ x, const i32* __restrict__ gamma, const i32* __restrict__ beta,
                                                                i8* __restrict__ out, u32 n, u32 out_stride, u32 k, u32 j, u64 eps, i32 m, u32 s,
                                                                u32 frac, u32 wide) {
    const u32 row = blockIdx.x, t = threadIdx.x;
    const i32* xr = x + (u64)row * n;
    i8* orow = out + (u64)row * out_stride;
    i8* lorow = wide ? out + ((u64)gridDim.x + row) * out_stride : nullptr;
    if (n <= NORM_T * NORM_V) {
        i32 v[NORM_V];
#pragma unroll
        for (u32 i = 0; i < NORM_V; i++) v[i] = t + i * NORM_T < n ? xr[t + i * NORM_T] : 0;
        norm_regs(v, gamma, beta, orow, lorow, n, k, j, eps, m, s, frac);
        return;
    }
    norm_row(xr, gamma, beta, orow, lorow, n, k, j, eps, m, s, frac);
}

// The sum of the split-K partials and the bias: the exact value of linear_out before the rescale.
// INVARIANT: the sum of the partials is the dot product, |dot| <= 2^30.
__device__ __forceinline__ i64 linear_value(const i32* __restrict__ acc, const i32* __restrict__ bias, u32 r, u32 row, u32 acc_stride,
                                            u32 splits, u32 split_stride) {
    i64 v = (i64)bias[r];
    for (u32 z = 0; z < splits; z++) v += (i64)acc[(u64)z * split_stride + (u64)row * acc_stride + r];
    return v;
}

// Fused: the output of a linear op with an i32 result (linear_out), the residual add (add_clamp)
// and the norm of the new residual row (layer_norm), in one block of NORM_T for each row. Each
// step is the expression of its op kernel, so the result is the result of the three op kernels
// (a test runs both paths). Grid: (rows, 1).
extern "C" __global__ void __launch_bounds__(NORM_T) residual_norm(const i32* __restrict__ acc, const i32* __restrict__ bias,
                                                                   const i32* __restrict__ mult, u32 shift, u32 splits, u32 split_stride,
                                                                   i32* __restrict__ x, const i32* __restrict__ gamma, const i32* __restrict__ beta,
                                                                   i8* __restrict__ out, u32 n, u32 out_stride, u32 k, u32 j, u64 eps, i32 nm,
                                                                   u32 ns, u32 frac) {
    const u32 t = threadIdx.x, row = blockIdx.x;
    i32* xr = x + (u64)row * n;
    if (n <= NORM_T * NORM_V) {
        i32 v[NORM_V];
#pragma unroll
        for (u32 i = 0; i < NORM_V; i++) {
            const u32 c = t + i * NORM_T;
            v[i] = 0;
            if (c < n) {
                const i64 lv = linear_value(acc, bias, c, row, n, splits, split_stride);
                const i64 y = rescale_clamp(lv, mult[c], shift, -2147483648LL, 2147483647LL);
                v[i] = (i32)clamp_wide((i128)((i64)xr[c] + y), -RES, RES);
                xr[c] = v[i];
            }
        }
        norm_regs(v, gamma, beta, out + (u64)row * out_stride, nullptr, n, k, j, eps, nm, ns, frac);
        return;
    }
    for (u32 c = t; c < n; c += NORM_T) {
        const i64 v = linear_value(acc, bias, c, row, n, splits, split_stride);
        const i64 y = rescale_clamp(v, mult[c], shift, -2147483648LL, 2147483647LL);
        xr[c] = (i32)clamp_wide((i128)((i64)xr[c] + y), -RES, RES);
    }
    norm_row(xr, gamma, beta, out + (u64)row * out_stride, nullptr, n, k, j, eps, nm, ns, frac);
}

// The output part of the `linear` op: y = clamp(rescale(acc + b, m[r], s), lo, hi), as i8
// (out_i8) or i32 (out_i32). `splits` partial sums of the product are at acc + z * split_stride;
// their total is the exact product (see gemm_tiled). Grid: (ceil(n / 128), rows).
extern "C" __global__ void linear_out(const i32* __restrict__ acc, const i32* __restrict__ bias, const i32* __restrict__ mult,
                                      i8* __restrict__ out_i8, i32* __restrict__ out_i32, u32 n, u32 acc_stride, u32 out_stride,
                                      u32 shift, i32 lo, i32 hi, u32 splits, u32 split_stride) {
    u32 r = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (r >= n) return;
    const i64 v = linear_value(acc, bias, r, row, acc_stride, splits, split_stride);
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

// Fused: the output of c_fc (linear_out with the range lo..hi) and the GELU table (table_lookup),
// the expressions of the two op kernels. Grid: (ceil(n / 128), rows).
extern "C" __global__ void linear_gelu(const i32* __restrict__ acc, const i32* __restrict__ bias, const i32* __restrict__ mult,
                                       const i8* __restrict__ tab, i8* __restrict__ out, u32 n, u32 out_stride, u32 shift, i32 lo,
                                       i32 hi, i32 offset, i32 len, u32 splits, u32 split_stride) {
    const u32 r = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (r >= n) return;
    const i64 v = linear_value(acc, bias, r, row, n, splits, split_stride);
    i32 y = (i32)rescale_clamp(v, mult[r], shift, lo, hi);
    // INVARIANT: offset <= 2^15 and len <= 2^16.
    const i32 tlo = -offset, thi = len - 1 - offset;
    y = y < tlo ? tlo : (y > thi ? thi : y);
    out[(u64)row * out_stride + r] = tab[y + offset];
}

// ---------------------------------------------------------------- i16 activations (model-file version 2)

// The i8 x i16 product of ADR 0004 addendum B (hardshell::kernels::matvec_i8_i16) with the i8
// products above. The producer of an i16 activation row writes it as two i8 planes (plane_store):
// v = 256 * hi + lo + 128. For a pass of n rows, the hi planes are the operand rows 0..n and the
// lo planes the rows n..2n of one i8 product with 2n rows. Then for weight row r
//   dot(w_r, v) = 256 * dot(w_r, hi) + dot(w_r, lo) + 128 * wsum[r],   wsum[r] = sum_k w_r[k],
// an identity of integers. The model loader calculates wsum exactly (the padding of w is 0).
//
// INVARIANT: the planes are in -128..127 and |w| <= 128, so the i8 product keeps its bound (each
// partial sum at most 2^30, see gemm_tiled). |wsum| <= K * 2^7 <= 2^23 and |dot(w, v)| <= K * 2^22
// <= 2^38 (K <= 2^16), so every step below is exact in i64.
__device__ __forceinline__ i64 wide_dot(const i32* __restrict__ acc, const i32* __restrict__ wsum, u32 r, u32 row, u32 rows,
                                        u32 acc_stride, u32 splits, u32 split_stride) {
    i64 hi = 0, lo = 0;
    for (u32 z = 0; z < splits; z++) {
        const i32* p = acc + (u64)z * split_stride;
        hi += (i64)p[(u64)row * acc_stride + r];
        lo += (i64)p[((u64)rows + row) * acc_stride + r];
    }
    return 256 * hi + lo + 128 * (i64)wsum[r];
}

// The `linear` op with an i16 input (mlp.c_proj of a model with an i16 GELU table): the output
// part y = clamp(rescale(dot + b, m[r], s), lo, hi), i32 output, as linear_out. Before the residual
// add the range is the i32 range (see the batch module). Grid: (ceil(n / 128), rows); the product
// has 2 * rows rows (wide_dot).
// INVARIANT: |dot + b| < 2^39 and |m| <= 2^31, so |(dot + b) * m| < 2^70 (the bound of rescale).
extern "C" __global__ void linear_out_wide(const i32* __restrict__ acc, const i32* __restrict__ bias, const i32* __restrict__ mult,
                                           const i32* __restrict__ wsum, i32* __restrict__ out, u32 n, u32 out_stride, u32 shift, i32 lo,
                                           i32 hi, u32 splits, u32 split_stride) {
    const u32 r = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (r >= n) return;
    const i64 v = (i64)bias[r] + wide_dot(acc, wsum, r, row, gridDim.y, n, splits, split_stride);
    out[(u64)row * out_stride + r] = (i32)clamp_wide(rescale(v, mult[r], shift), lo, hi);
}

// The `table` op with i16 outputs (an i16 GELU table): y = t[clamp(x, -offset, len - 1 - offset) +
// offset], stored as two planes (plane_store): the hi plane of row r at row r of `out`, the lo
// plane at row gridDim.y + r. Grid: (ceil(n / 128), rows).
extern "C" __global__ void table_wide(const i32* __restrict__ in, const i16* __restrict__ tab, i8* __restrict__ out, u32 n,
                                      u32 in_stride, u32 out_stride, i32 offset, i32 len) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (c >= n) return;
    i32 v = in[(u64)row * in_stride + c];
    // INVARIANT: offset <= 2^15 and len <= 2^16.
    const i32 lo = -offset, hi = len - 1 - offset;
    v = v < lo ? lo : (v > hi ? hi : v);
    plane_store(out + (u64)row * out_stride, out + ((u64)gridDim.y + row) * out_stride, c, (i32)tab[v + offset]);
}

// The exact i8 x i16 product with the bias, dot + b as i64 (no rescale): the value that
// linear_out_wide and logits_scale_wide rescale. For the tests of the product (ops.rs).
// Grid: (ceil(n / 128), rows).
extern "C" __global__ void wide_sum(const i32* __restrict__ acc, const i32* __restrict__ bias, const i32* __restrict__ wsum,
                                    i64* __restrict__ out, u32 n, u32 splits, u32 split_stride) {
    const u32 r = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (r >= n) return;
    out[(u64)row * n + r] = (i64)bias[r] + wide_dot(acc, wsum, r, row, gridDim.y, n, splits, split_stride);
}

// ---------------------------------------------------------------- attention

// The cache of one layer has a region of ctx4 * head_dim bytes for each (slot, head); ctx4 is
// n_ctx rounded up to a multiple of 4.
// - Keys: position j at j * head_dim, so the key of a position is contiguous.
// - Values: in groups of 4 positions (quads). Byte (j, c) is at (j & ~3) * head_dim + 4 * c +
//   (j & 3), so one word holds channel c of four consecutive positions: the operand of dp4a in
//   the sum of p * v, with no transposition in the kernel.
// kv_store copies k and v of each row. Grid: (ceil(d / 128), rows).
extern "C" __global__ void kv_store(const i8* __restrict__ qkv, i8* __restrict__ kc, i8* __restrict__ vc, const u32* __restrict__ rows,
                                    u32 d, u32 qkv_stride, u32 ctx4, u32 head_dim) {
    u32 c = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (c >= d) return;
    u32 slot = rows[4 * row], pos = rows[4 * row + 2];
    u32 h = c / head_dim, cc = c % head_dim, n_head = d / head_dim;
    const u64 base = ((u64)slot * n_head + h) * ctx4 * head_dim;
    kc[base + (u64)pos * head_dim + cc] = qkv[(u64)row * qkv_stride + d + c];
    vc[base + (u64)(pos & ~3u) * head_dim + 4 * cc + (pos & 3)] = qkv[(u64)row * qkv_stride + 2 * d + c];
}

// Fused: the output of c_attn (linear_out with the range -127..127, into qkv) and kv_store (k and
// v into the cache layout above). Grid: (ceil(3 * d / 128), rows).
extern "C" __global__ void linear_qkv(const i32* __restrict__ acc, const i32* __restrict__ bias, const i32* __restrict__ mult,
                                      i8* __restrict__ qkv, i8* __restrict__ kc, i8* __restrict__ vc, const u32* __restrict__ rows, u32 d,
                                      u32 qkv_stride, u32 shift, u32 splits, u32 split_stride, u32 ctx4, u32 head_dim) {
    const u32 r = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (r >= 3 * d) return;
    const i64 v = linear_value(acc, bias, r, row, 3 * d, splits, split_stride);
    const i8 y = (i8)rescale_clamp(v, mult[r], shift, -127, 127);
    qkv[(u64)row * qkv_stride + r] = y;
    if (r < d) return;
    const u32 slot = rows[4 * row], pos = rows[4 * row + 2];
    const u32 c = r < 2 * d ? r - d : r - 2 * d;
    const u32 h = c / head_dim, cc = c % head_dim, n_head = d / head_dim;
    const u64 base = ((u64)slot * n_head + h) * ctx4 * head_dim;
    if (r < 2 * d) kc[base + (u64)pos * head_dim + cc] = y;
    else vc[base + (u64)(pos & ~3u) * head_dim + 4 * cc + (pos & 3)] = y;
}

#define ATT_T 128
#define ATT_W (ATT_T / 32)
// The scores and the byte planes of a context of at most ATT_CTX positions stay in shared memory.
#define ATT_CTX 1024

// The score of one dot product: clamp(rescale(dot, m, s), -2^40, 2^40).
// INVARIANT: |dot| <= head_dim * 2^14 <= 2^22 and |m| < 2^31, so |dot * m| < 2^53 < 2^62 (the
// bound of rescale_small).
__device__ __forceinline__ i64 att_score(i32 dot, i32 m, u32 s) {
    i64 v = rescale_small((i64)dot, m, s);
    return v < -LIMIT40 ? -LIMIT40 : (v > LIMIT40 ? LIMIT40 : v);
}

// Causal attention for one (head, row) with the softmax of hardshell::softmax.
// Grid: (n_head, rows); block ATT_T (128). For ctx4 > ATT_CTX, `sc` holds 12 * ctx4 bytes for each
// (row, head): ctx4 i32 values (in the first 8 * ctx4 bytes), then ctx4 / 4 plane words of 16 bytes.
//
// 1. Dots. With head_dim / 16 = lk a power of two, lk lanes read one key together (16 bytes
//    each, so a warp reads 32 / lk keys of contiguous memory), and add their dp4a sums with
//    shuffles. Each thread has 4 loads in flight. Otherwise one thread reads one key in words.
//    The dot of each key goes to shared memory (i32), with the largest and the smallest dot.
// 2. The row max of the scores is the larger of the scores of the largest and the smallest dot:
//    att_score is monotonic in the dot (below). Then e = table[min(max - s, len - 1)] with
//    s = att_score(dot), and the sum of e: warp shuffles.
// 3. p = floor(e * 2^P / total) with an exact reciprocal (see below); the byte planes of p.
// 4. sum_j p[j] * v[j][c] = sum_b 2^(8b) * sum_j p_b[j] * v[j][c]: dp4a of a plane word (bytes of
//    p of a quad) and a value word (channel c of the same quad). Each thread takes 4 channels
//    (16 bytes) of every `groups`-th quad; then a sum over the groups.
//
// DETERMINISM: att_score(dot) = clamp(floor((dot * m + 2^(s-1)) / 2^s), -2^40, 2^40) (or dot * m
// for s = 0, or 0 for s > 62). Each step is non-decreasing in dot * m, and dot * m is
// non-decreasing in dot for m >= 0 and non-increasing for m < 0. A monotonic function of the dot
// takes its maximum over the dots at the largest or at the smallest dot, so max(att_score(max_j
// dot_j), att_score(min_j dot_j)) is the maximum over all scores, for every sign of m.
extern "C" __global__ void __launch_bounds__(ATT_T, 8) attend(const i8* __restrict__ qkv, const i8* __restrict__ kc, const i8* __restrict__ vc,
                                  u8* __restrict__ sc, const i32* __restrict__ attn_exp, const u32* __restrict__ rows, i8* __restrict__ att,
                                  u32 d, u32 qkv_stride, u32 head_dim, u32 ctx4, u32 n_head, i32 score_m, u32 score_s,
                                  i32 out_m, u32 out_s, u32 p, u32 table_len, u32 att_stride) {
    // The dots, then the values e (both i32), of ATT_CTX positions; after the planes, the i64
    // partial sums of step 4 (groups * head_dim = 4 * ATT_T = ATT_CTX / 2 values).
    __shared__ __align__(16) i64 s_sh[ATT_CTX / 2];
    __shared__ __align__(16) uint4 pl_sh[ATT_CTX / 4];
    __shared__ __align__(16) u32 qw[64];
    __shared__ i32 red_hi[ATT_W], red_lo[ATT_W];
    __shared__ i64 red_sum[ATT_W];
    const u32 t = threadIdx.x, lane = t & 31, warp = t >> 5, h = blockIdx.x, row = blockIdx.y;
    const u32 slot = rows[4 * row], n = rows[4 * row + 2] + 1;
    i32* s = (i32*)s_sh;
    uint4* planes = pl_sh;
    if (ctx4 > ATT_CTX) {
        u8* g = sc + ((u64)row * n_head + h) * 12 * (u64)ctx4;
        s = (i32*)g;
        planes = (uint4*)(g + 8 * (u64)ctx4);
    }
    const u32 hw = head_dim / 4;
    if (t < hw) qw[t] = ((const u32*)(qkv + (u64)row * qkv_stride + h * head_dim))[t];
    __syncthreads();
    const u64 region = (u64)ctx4 * head_dim;
    const i8* kb = kc + ((u64)slot * n_head + h) * region;
    // INVARIANT: |dot| <= head_dim * 2^14 <= 2^22 (see att_score), so the i32 limits are outside
    // every dot; n >= 1, so both are replaced.
    i32 hi = -2147483647 - 1, lo = 2147483647;
    const u32 lk = hw / 4;
    if ((hw & 3) == 0 && (lk & (lk - 1)) == 0) {
        // INVARIANT: head_dim <= 256, so lk <= 16 and a key group stays inside a warp.
        const u32 sub = t & (lk - 1), per = ATT_T / lk;
        const uint4 q4 = ((const uint4*)qw)[sub];
        // NOTE: the bound of the loop is the same for all threads, so every lane takes part in
        // each shuffle.
        for (u32 base = 0; base < n; base += 4 * per) {
            uint4 k4[4];
#pragma unroll
            for (u32 u = 0; u < 4; u++) {
                const u32 j = base + u * per + t / lk;
                k4[u] = j < n ? *(const uint4*)(kb + (u64)j * head_dim + 16 * sub) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (u32 u = 0; u < 4; u++) {
                i32 dot = __dp4a((i32)q4.x, (i32)k4[u].x, 0);
                dot = __dp4a((i32)q4.y, (i32)k4[u].y, dot);
                dot = __dp4a((i32)q4.z, (i32)k4[u].z, dot);
                dot = __dp4a((i32)q4.w, (i32)k4[u].w, dot);
                for (u32 off = 1; off < lk; off <<= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
                const u32 j = base + u * per + t / lk;
                if (j < n) {
                    if (sub == 0) s[j] = dot;
                    hi = dot > hi ? dot : hi;
                    lo = dot < lo ? dot : lo;
                }
            }
        }
    } else {
        for (u32 j = t; j < n; j += ATT_T) {
            const u32* kr = (const u32*)(kb + (u64)j * head_dim);
            i32 dot = 0;
            for (u32 w = 0; w < hw; w++) dot = __dp4a((i32)qw[w], (i32)kr[w], dot);
            s[j] = dot;
            hi = dot > hi ? dot : hi;
            lo = dot < lo ? dot : lo;
        }
    }
    hi = __reduce_max_sync(0xffffffffu, hi);
    lo = __reduce_min_sync(0xffffffffu, lo);
    if (lane == 0) {
        red_hi[warp] = hi;
        red_lo[warp] = lo;
    }
    __syncthreads();
#pragma unroll
    for (u32 w = 0; w < ATT_W; w++) {
        hi = red_hi[w] > hi ? red_hi[w] : hi;
        lo = red_lo[w] < lo ? red_lo[w] : lo;
    }
    const i64 s_hi = att_score(hi, score_m, score_s), s_lo = att_score(lo, score_m, score_s);
    const i64 row_max = s_hi > s_lo ? s_hi : s_lo;
    i64 sum = 0;
    for (u32 j = t; j < n; j += ATT_T) {
        // INVARIANT: scores are in +-2^40 and row_max is the largest, so the difference fits and
        // is >= 0.
        const i64 dd = row_max - att_score(s[j], score_m, score_s);
        const i64 idx = dd < (i64)table_len - 1 ? dd : (i64)table_len - 1;
        // INVARIANT: the table entries are in 0 .. 2^31 - 1, so e fits in i32.
        const i32 e = attn_exp[idx];
        s[j] = e;
        // INVARIANT: n <= 2^16 entries below 2^31.
        sum += e;
    }
    sum = warp_sum_i64(sum);
    if (lane == 0) red_sum[warp] = sum;
    __syncthreads();
    u64 total = 0;
#pragma unroll
    for (u32 w = 0; w < ATT_W; w++) total += (u64)red_sum[w];
    // INVARIANT: the maximum score takes the first entry, which is >= 1, so total >= 1.
    if (total == 0) total = 1;
    // p[j] = floor(e * 2^P / total) with an exact reciprocal: recip = floor(2^(P+32) / total) and
    // est = floor(e * recip / 2^32). e <= total, so e * recip <= 2^(P+32) <= 2^62 fits in u64.
    // est <= e * 2^P / total, and the error is below e / 2^32 < 1, so the exact quotient is est
    // or est + 1; one exact check selects it.
    const u64 recip = ((u64)1 << (p + 32)) / total;
    const u32 nq = (n + 3) / 4;
    for (u32 q = t; q < nq; q += ATT_T) {
        u32 pw[4] = {0, 0, 0, 0};
#pragma unroll
        for (u32 k = 0; k < 4; k++) {
            u32 pj = 0;
            if (4 * q + k < n) {
                // INVARIANT: e < 2^31 and p <= 30, so e << p < 2^61 and est * total <= e << p.
                const u64 e = (u64)(u32)s[4 * q + k];
                u64 est = (e * recip) >> 32;
                if ((e << p) - est * total >= total) est += 1;
                pj = (u32)est;
            }
#pragma unroll
            for (u32 b = 0; b < 4; b++) pw[b] |= ((pj >> (8 * b)) & 0xffu) << (8 * k);
        }
        planes[q] = make_uint4(pw[0], pw[1], pw[2], pw[3]);
    }
    __syncthreads();
    const i8* vb = vc + ((u64)slot * n_head + h) * region;
    // Each thread takes 16 bytes (4 channels) of a quad: tpq threads for each quad.
    const u32 tpq = head_dim / 4, groups = ATT_T / tpq, grp = t / tpq, w4 = t % tpq;
    // INVARIANT: each i32 sum has at most n <= 2^16 terms of magnitude <= 255 * 128 < 2^15, so it
    // stays in -2^31 .. 2^31 - 1.
    i32 acc[4][4];
#pragma unroll
    for (int c = 0; c < 4; c++)
#pragma unroll
        for (int b = 0; b < 4; b++) acc[c][b] = 0;
    if (grp < groups) {
#pragma unroll 4
        for (u32 q = grp; q < nq; q += groups) {
            // INVARIANT: 4q + 3 < ctx4, so the load is inside the region of the slot; values
            // after n meet a zero plane byte. The address is a multiple of 16: head_dim and
            // ctx4 are multiples of 4.
            const uint4 v4 = *(const uint4*)(vb + (u64)q * 4 * head_dim + 16 * w4);
            const uint4 pq = planes[q];
            const u32 vv[4] = {v4.x, v4.y, v4.z, v4.w};
            const u32 pp[4] = {pq.x, pq.y, pq.z, pq.w};
#pragma unroll
            for (int b = 0; b < 4; b++)
#pragma unroll
                for (int c = 0; c < 4; c++) asm("dp4a.u32.s32 %0, %1, %2, %0;" : "+r"(acc[c][b]) : "r"(pp[b]), "r"(vv[c]));
        }
    }
    // The values e are not read after the planes, so the sums over the groups use s_sh.
    // INVARIANT: groups * head_dim <= 4 * ATT_T = ATT_CTX / 2.
    i64* part = s_sh;
    if (grp < groups) {
#pragma unroll
        for (int c = 0; c < 4; c++)
            // INVARIANT: |acc| < 2^31, so acc * 2^24 < 2^55.
            part[grp * head_dim + 4 * w4 + c] =
                (i64)acc[c][0] + (i64)acc[c][1] * 256 + (i64)acc[c][2] * 65536 + (i64)acc[c][3] * 16777216;
    }
    __syncthreads();
    for (u32 c = t; c < head_dim; c += ATT_T) {
        i64 tot = 0;
        for (u32 gg = 0; gg < groups; gg++) tot += part[gg * head_dim + c];
        att[(u64)row * att_stride + h * head_dim + c] = (i8)clamp_act_i8(rescale(tot, out_m, out_s));
    }
}

// ---------------------------------------------------------------- attention of row groups (prefill)

#define ATG_ROWS 16
#define ATG_KEYS 32
// Words in a row of the p buffer: 32 keys and 4 words of padding.
#define ATG_PS 36
// The most n-tiles (8 channels each) of a warp: head_dim <= 128.
#define ATG_NT 4

__device__ __forceinline__ void mma_s8s8(i32 (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_u8s8(i32 (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The byte planes of four values x.x .. x.w: plane b holds byte b of each value, in value order.
__device__ __forceinline__ void planes_of(uint4 x, u32& b0, u32& b1, u32& b2, u32& b3) {
    const u32 lo01 = __byte_perm(x.x, x.y, 0x5140), hi01 = __byte_perm(x.x, x.y, 0x7362);
    const u32 lo23 = __byte_perm(x.z, x.w, 0x5140), hi23 = __byte_perm(x.z, x.w, 0x7362);
    b0 = __byte_perm(lo01, lo23, 0x5410);
    b1 = __byte_perm(lo01, lo23, 0x7632);
    b2 = __byte_perm(hi01, hi23, 0x5410);
    b3 = __byte_perm(hi01, hi23, 0x7632);
}

// p = floor(e * 2^P / total) for e = table[min(row_max - att_score(dot), len - 1)], with the
// exact reciprocal of `attend` (step 3 there).
__device__ __forceinline__ u32 att_p(i32 dot, i64 row_max, u64 total, u64 recip, const i32* __restrict__ attn_exp, i32 m, u32 s,
                                     u32 p, u32 table_len) {
    // INVARIANT: row_max is the largest score of the row, so 0 <= dd <= 2^41.
    const i64 dd = row_max - att_score(dot, m, s);
    const i64 idx = dd < (i64)table_len - 1 ? dd : (i64)table_len - 1;
    const u64 e = (u64)(u32)attn_exp[idx];
    // INVARIANT: e < 2^31 and p <= 30 (see attend).
    u64 est = (e * recip) >> 32;
    if ((e << p) - est * total >= total) est += 1;
    return (u32)est;
}

// Causal attention for a group of up to 16 rows of one slot at consecutive positions p0 ..
// p0 + cnt - 1 (prefill), for one head, with int8 tensor cores. The result of each row is the
// result of `attend` for that row, bit for bit:
//
// - The dots q . k are mma.sync m16n8k32 s8 x s8 products with exact s32 sums: the integers of
//   the dp4a sums of `attend`.
// - Three passes over the keys, in chunks of 32 keys (warp w takes keys 8w .. 8w + 7 of a chunk):
//   (A) the largest and the smallest dot of each row, and the row max as in `attend`; (B) the sum
//   of e; (C) p of each key, its byte planes, and sum_j p_b[j] * v[j][c] as mma.sync u8 x s8
//   products (A operand: the planes of 16 rows; B operand: the value quads of the cache layout,
//   one word = 4 keys of one channel). The pass recomputes the dots; integer products give the
//   same dots each time.
// - Each sum of p_b * v has at most n <= 2^16 terms of magnitude <= 255 * 128 < 2^15, so the
//   s32 sums are exact (the INVARIANT of `attend`). Integer sums are associative, so the chunk
//   order does not change them.
//
// Keys at or after n of a row get p = 0; the cache bytes there are not read for a key, and the
// values there meet a zero plane byte. Grid: (n_head, groups); block 128 (4 warps). `groups` holds
// (first row, count) for each group. Needs head_dim to be a multiple of 32 and at most 128.
extern "C" __global__ void __launch_bounds__(128) attend_rows(const i8* __restrict__ qkv, const i8* __restrict__ kc, const i8* __restrict__ vc,
                                                               const u32* __restrict__ groups, const i32* __restrict__ attn_exp,
                                                               const u32* __restrict__ rows, i8* __restrict__ att, u32 qkv_stride,
                                                               u32 head_dim, u32 ctx4, u32 n_head, i32 score_m, u32 score_s, i32 out_m,
                                                               u32 out_s, u32 p, u32 table_len, u32 att_stride) {
    __shared__ __align__(16) u32 ps[2][ATG_ROWS * ATG_PS];
    __shared__ i32 red_hi[4][ATG_ROWS], red_lo[4][ATG_ROWS];
    __shared__ i64 red_sum[4][ATG_ROWS];
    const u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const u32 h = blockIdx.x, first = groups[2 * blockIdx.y], cnt = groups[2 * blockIdx.y + 1];
    const u32 slot = rows[4 * first], p0 = rows[4 * first + 2];
    // INVARIANT: the rows of the group are at p0 .. p0 + cnt - 1 < n_ctx <= ctx4.
    const u32 nkeys = p0 + cnt, hw = head_dim / 4, ksteps = head_dim / 32, nts = head_dim / 32;
    const u64 region = (u64)ctx4 * head_dim;
    const u32* kb = (const u32*)(kc + ((u64)slot * n_head + h) * region);
    const u32* vb = (const u32*)(vc + ((u64)slot * n_head + h) * region);
    // The rows of this thread in the fragments: g and g + 8.
    const u32 r0 = g, r1 = g + 8;
    const bool ok0 = r0 < cnt, ok1 = r1 < cnt;
    const u32 n0 = p0 + r0 + 1, n1 = p0 + r1 + 1;
    u32 qa[ATG_NT][4];
    {
        // INVARIANT: qkv_stride and head_dim are multiples of 4, so the words are aligned.
        const u32* q0 = (const u32*)(qkv + (u64)(first + r0) * qkv_stride + h * head_dim);
        const u32* q1 = (const u32*)(qkv + (u64)(first + r1) * qkv_stride + h * head_dim);
#pragma unroll
        for (u32 kk = 0; kk < ATG_NT; kk++) {
            const bool k_ok = kk < ksteps;
            qa[kk][0] = k_ok && ok0 ? q0[kk * 8 + t] : 0;
            qa[kk][1] = k_ok && ok1 ? q1[kk * 8 + t] : 0;
            qa[kk][2] = k_ok && ok0 ? q0[kk * 8 + 4 + t] : 0;
            qa[kk][3] = k_ok && ok1 ? q1[kk * 8 + 4 + t] : 0;
        }
    }
    // The dots of the 16 rows with keys c0 + 8 * warp .. + 7: c[0], c[1] for row g and keys
    // c0 + 8 * warp + 2t, + 1; c[2], c[3] for row g + 8.
    auto dots = [&](u32 c0, i32 (&c)[4]) {
        c[0] = c[1] = c[2] = c[3] = 0;
        const u32 key = c0 + 8 * warp + g;
        const bool kok = key < nkeys;
        const u32* kr = kb + (u64)key * hw;
#pragma unroll
        for (u32 kk = 0; kk < ATG_NT; kk++) {
            if (kk < ksteps) {
                const u32 b0 = kok ? kr[kk * 8 + t] : 0, b1 = kok ? kr[kk * 8 + 4 + t] : 0;
                mma_s8s8(c, qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
            }
        }
    };
    // (A) The largest and the smallest dot of each row.
    i32 hi0 = -2147483647 - 1, lo0 = 2147483647, hi1 = hi0, lo1 = lo0;
    for (u32 c0 = 0; c0 < nkeys; c0 += ATG_KEYS) {
        i32 c[4];
        dots(c0, c);
        const u32 j = c0 + 8 * warp + 2 * t;
        if (ok0 && j < n0) { hi0 = max(hi0, c[0]); lo0 = min(lo0, c[0]); }
        if (ok0 && j + 1 < n0) { hi0 = max(hi0, c[1]); lo0 = min(lo0, c[1]); }
        if (ok1 && j < n1) { hi1 = max(hi1, c[2]); lo1 = min(lo1, c[2]); }
        if (ok1 && j + 1 < n1) { hi1 = max(hi1, c[3]); lo1 = min(lo1, c[3]); }
    }
#pragma unroll
    for (u32 off = 1; off < 4; off <<= 1) {
        hi0 = max(hi0, __shfl_xor_sync(0xffffffffu, hi0, off));
        lo0 = min(lo0, __shfl_xor_sync(0xffffffffu, lo0, off));
        hi1 = max(hi1, __shfl_xor_sync(0xffffffffu, hi1, off));
        lo1 = min(lo1, __shfl_xor_sync(0xffffffffu, lo1, off));
    }
    if (t == 0) {
        red_hi[warp][r0] = hi0;
        red_lo[warp][r0] = lo0;
        red_hi[warp][r1] = hi1;
        red_lo[warp][r1] = lo1;
    }
    __syncthreads();
#pragma unroll
    for (u32 w = 0; w < 4; w++) {
        hi0 = max(hi0, red_hi[w][r0]);
        lo0 = min(lo0, red_lo[w][r0]);
        hi1 = max(hi1, red_hi[w][r1]);
        lo1 = min(lo1, red_lo[w][r1]);
    }
    // The row max of `attend` (see its DETERMINISM note). A row that is not in the group has no
    // dot; it gets 0 and takes no part below.
    auto row_max_of = [&](bool ok, i32 hi, i32 lo) -> i64 {
        if (!ok) return 0;
        const i64 a = att_score(hi, score_m, score_s), b = att_score(lo, score_m, score_s);
        return a > b ? a : b;
    };
    const i64 rm0 = row_max_of(ok0, hi0, lo0), rm1 = row_max_of(ok1, hi1, lo1);
    // (B) The sum of e of each row.
    auto e_of = [&](i32 dot, i64 rm) -> i64 {
        const i64 dd = rm - att_score(dot, score_m, score_s);
        const i64 idx = dd < (i64)table_len - 1 ? dd : (i64)table_len - 1;
        return (i64)attn_exp[idx];
    };
    i64 sum0 = 0, sum1 = 0;
    for (u32 c0 = 0; c0 < nkeys; c0 += ATG_KEYS) {
        i32 c[4];
        dots(c0, c);
        const u32 j = c0 + 8 * warp + 2 * t;
        // INVARIANT: at most 2^16 values e below 2^31 for each row.
        if (ok0 && j < n0) sum0 += e_of(c[0], rm0);
        if (ok0 && j + 1 < n0) sum0 += e_of(c[1], rm0);
        if (ok1 && j < n1) sum1 += e_of(c[2], rm1);
        if (ok1 && j + 1 < n1) sum1 += e_of(c[3], rm1);
    }
#pragma unroll
    for (u32 off = 1; off < 4; off <<= 1) {
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, off);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, off);
    }
    if (t == 0) {
        red_sum[warp][r0] = sum0;
        red_sum[warp][r1] = sum1;
    }
    __syncthreads();
    u64 tot0 = 0, tot1 = 0;
#pragma unroll
    for (u32 w = 0; w < 4; w++) {
        tot0 += (u64)red_sum[w][r0];
        tot1 += (u64)red_sum[w][r1];
    }
    // INVARIANT: the maximum score takes the first entry, which is >= 1 (see attend).
    if (tot0 == 0) tot0 = 1;
    if (tot1 == 0) tot1 = 1;
    const u64 rc0 = ((u64)1 << (p + 32)) / tot0, rc1 = ((u64)1 << (p + 32)) / tot1;
    // (C) p, the planes, and the products with the values.
    i32 acc[4][ATG_NT][4];
#pragma unroll
    for (u32 b = 0; b < 4; b++)
#pragma unroll
        for (u32 nt = 0; nt < ATG_NT; nt++)
#pragma unroll
            for (u32 x = 0; x < 4; x++) acc[b][nt][x] = 0;
    u32 buf = 0;
    for (u32 c0 = 0; c0 < nkeys; c0 += ATG_KEYS, buf ^= 1) {
        i32 c[4];
        dots(c0, c);
        const u32 j = c0 + 8 * warp + 2 * t;
        u32* P = ps[buf];
        const u32 col = 8 * warp + 2 * t;
        P[r0 * ATG_PS + col] = ok0 && j < n0 ? att_p(c[0], rm0, tot0, rc0, attn_exp, score_m, score_s, p, table_len) : 0;
        P[r0 * ATG_PS + col + 1] = ok0 && j + 1 < n0 ? att_p(c[1], rm0, tot0, rc0, attn_exp, score_m, score_s, p, table_len) : 0;
        P[r1 * ATG_PS + col] = ok1 && j < n1 ? att_p(c[2], rm1, tot1, rc1, attn_exp, score_m, score_s, p, table_len) : 0;
        P[r1 * ATG_PS + col + 1] = ok1 && j + 1 < n1 ? att_p(c[3], rm1, tot1, rc1, attn_exp, score_m, score_s, p, table_len) : 0;
        // NOTE: two buffers, so one barrier for each chunk is enough: a warp writes a buffer again
        // only after the barrier of the next chunk, which all warps reach after their reads.
        __syncthreads();
        // A operands: plane b of p for rows g, g + 8 and keys 4t .. 4t + 3, 16 + 4t .. 16 + 4t + 3.
        u32 pa[4][4];
        planes_of(*(const uint4*)&P[r0 * ATG_PS + 4 * t], pa[0][0], pa[1][0], pa[2][0], pa[3][0]);
        planes_of(*(const uint4*)&P[r1 * ATG_PS + 4 * t], pa[0][1], pa[1][1], pa[2][1], pa[3][1]);
        planes_of(*(const uint4*)&P[r0 * ATG_PS + 16 + 4 * t], pa[0][2], pa[1][2], pa[2][2], pa[3][2]);
        planes_of(*(const uint4*)&P[r1 * ATG_PS + 16 + 4 * t], pa[0][3], pa[1][3], pa[2][3], pa[3][3]);
        // B operands: the value words of quads c0 / 4 + t and c0 / 4 + 4 + t (keys 4t .., 16 + 4t ..)
        // for channel g of each n-tile of this warp. The word of (quad Q, channel ch) is at
        // Q * head_dim + ch (the cache layout of kv_store).
        const u32 qa_ = c0 / 4 + t, qb_ = qa_ + 4;
#pragma unroll
        for (u32 nt = 0; nt < ATG_NT; nt++) {
            if (nt < nts) {
                const u32 ch = warp * 8 * nts + nt * 8 + g;
                const u32 b0 = 4 * qa_ < ctx4 ? vb[(u64)qa_ * head_dim + ch] : 0;
                const u32 b1 = 4 * qb_ < ctx4 ? vb[(u64)qb_ * head_dim + ch] : 0;
#pragma unroll
                for (u32 b = 0; b < 4; b++) mma_u8s8(acc[b][nt], pa[b][0], pa[b][1], pa[b][2], pa[b][3], b0, b1);
            }
        }
    }
    // The output: channels 2t, 2t + 1 of each n-tile, rows g (acc 0, 1) and g + 8 (acc 2, 3).
#pragma unroll
    for (u32 nt = 0; nt < ATG_NT; nt++) {
        if (nt >= nts) continue;
#pragma unroll
        for (u32 x = 0; x < 4; x++) {
            const bool ok = x < 2 ? ok0 : ok1;
            if (!ok) continue;
            const u32 r = x < 2 ? r0 : r1, ch = warp * 8 * nts + nt * 8 + 2 * t + (x & 1);
            // INVARIANT: |acc| < 2^31, so acc * 2^24 < 2^55.
            const i64 tot = (i64)acc[0][nt][x] + (i64)acc[1][nt][x] * 256 + (i64)acc[2][nt][x] * 65536 +
                            (i64)acc[3][nt][x] * 16777216;
            att[(u64)(first + r) * att_stride + h * head_dim + ch] = (i8)clamp_act_i8(rescale(tot, out_m, out_s));
        }
    }
}

// Writes the GPU global timer (nanoseconds) to out[i]: a profiler mark in the stream. It changes
// no model value. One thread.
extern "C" __global__ void stamp(u64* __restrict__ out, u32 i) {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    out[i] = t;
}

// Reads n16 * 16 bytes of `p` and writes one word for each thread: a measurement of the memory
// bandwidth. It changes no model value. Grid: any; block: any.
extern "C" __global__ void read_bandwidth(const uint4* __restrict__ p, u64 n16, u32* __restrict__ out) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x, step = (u64)gridDim.x * blockDim.x;
    u32 x = 0;
    for (u64 i = tid; i < n16; i += step) {
        const uint4 v = p[i];
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    out[tid] = x;
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

// The logits with an i16 final LayerNorm (ln_f_bits = 16): logit = clamp(rescale(dot, m[v], s),
// -2^40, 2^40) with dot the i8 x i16 product of wte and the planes of ln_f (wide_dot, one split).
// Grid: (ceil(vocab / 128), rows). INVARIANT: |dot| < 2^39, so |dot * m| < 2^70.
extern "C" __global__ void logits_scale_wide(const i32* __restrict__ acc, const i32* __restrict__ mult, const i32* __restrict__ wsum,
                                             i64* __restrict__ logit, u32 vocab, u32 shift) {
    const u32 v = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (v >= vocab) return;
    const i64 dot = wide_dot(acc, wsum, v, row, gridDim.y, vocab, 1, 0);
    logit[(u64)row * vocab + v] = clamp_wide(rescale(dot, mult[v], shift), -LIMIT40, LIMIT40);
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
//
// The weights come from `src`: the weights of the logits op (coder_interval), or, fused, the
// logits and the table of logits_weights (coder_logits): weight = table[min(max - logit, len - 1)],
// the expression of logits_weights, with the row max of a first pass.
// The tiles of the compact logits that coder_d16 keeps in shared memory: a vocabulary of at most
// CODER_TILES * 128 tokens.
#define CODER_TILES 2048

struct WeightSrc {
    const u64* w;
    const i64* l;
    const u16* d;
    const u32* off;
    const u64* tab;
    i64 max;
    u32 len;
    __device__ __forceinline__ u64 at(u32 i) const {
        if (w) return w[i];
        i64 idx;
        if (d) {
            // INVARIANT: off <= len - 1 and d <= len - 1 <= 2^16 - 1, so the sum fits.
            const i64 s = (i64)off[i >> 7] + (i64)d[i];
            idx = s < (i64)len - 1 ? s : (i64)len - 1;
        } else {
            const i64 dd = max - l[i];
            idx = dd < (i64)len - 1 ? dd : (i64)len - 1;
        }
        return tab[idx];
    }
};

template <class Src>
__device__ __forceinline__ u64 freq_at(const Src& w, u32 i, bool zero, u64 wsum, u64 spare, u64 frac, u32 largest, u64 rem) {
    const u64 wi = zero ? 1 : w.at(i);
    return 1 + (frac ? quot_fast(wi, wsum, spare, frac) : quot(wi, wsum, spare)) + (i == largest ? rem : 0);
}

template <class Src>
__device__ void coder_body(const Src& w, const u32* __restrict__ queries, u64* __restrict__ out, u32 vocab, u64 total) {
    __shared__ u64 cs[256];
    __shared__ u64 bw[256];
    __shared__ u32 bi[256];
    __shared__ u64 wprefix[9];
    __shared__ u64 wsum_s, rem_s, frac_s;
    __shared__ u32 largest_s, zero_s;
    const u32 t = threadIdx.x, row = blockIdx.x, lane = t & 31, warp = t >> 5;
    const u32 c = (vocab + 7) / 8;
    const u32 lo = min(warp * c, vocab), hi = min(lo + c, vocab);
    const u64 spare = total - vocab;
    // Pass 1: the weight sum, and the largest weight with the lowest id.
    u64 s = 0, best = 0;
    u32 besti = 0xffffffffu;
    for (u32 i = lo + lane; i < hi; i += 32) {
        const u64 wi = w.at(i);
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
        // INVARIANT: the u64 sum cannot overflow: GPT-2 weights are at most 2^40 and vocab <= 2^20;
        // the mixed weights of MixedSrc are below 2^62 with a sum below 2^63 (ADR 0010 section 3).
        // quot and quot_fast are exact for every big_w < 2^64.
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

extern "C" __global__ void __launch_bounds__(256) coder_interval(const u64* __restrict__ weights, const u32* __restrict__ queries,
                                                                 u64* __restrict__ out, u32 vocab, u64 total) {
    WeightSrc w = {weights + (u64)blockIdx.x * vocab, nullptr, nullptr, nullptr, nullptr, 0, 0};
    coder_body(w, queries, out, vocab, total);
}

// The weights of one row from the logits and the table of logits_weights: pass 0 finds the row max
// (as logits_weights). One block of 256 for each row.
__device__ __forceinline__ WeightSrc src_logits(const i64* __restrict__ logit, const u64* __restrict__ logits_exp, u32 vocab,
                                                u32 table_len) {
    __shared__ i64 red[8];
    const u32 t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const i64* l = logit + (u64)blockIdx.x * vocab;
    i64 mx = -LIMIT40;
    for (u32 v = t; v < vocab; v += 256) mx = l[v] > mx ? l[v] : mx;
    mx = warp_max_i64(mx);
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    i64 m = red[0];
#pragma unroll
    for (u32 k = 1; k < 8; k++) m = red[k] > m ? red[k] : m;
    WeightSrc w = {nullptr, l, nullptr, nullptr, logits_exp, m, table_len};
    return w;
}

// The weights of one row from the compact logits of gemm_tc*_logits (d16, tmax with `tiles` tiles of
// 128 columns for each row). The row max is the max of the tile maxima; the weight of column i is
// table[min((max - tm) + d, len - 1)] (see gemm_tc_body), the expression of logits_weights.
// off[tile] = min(max - tm, len - 1) keeps the sum small; it does not change the clamped index. One
// block of 256 for each row; tiles <= CODER_TILES.
__device__ __forceinline__ WeightSrc src_d16(const u16* __restrict__ d16, const i64* __restrict__ tmax,
                                             const u64* __restrict__ logits_exp, u32 vocab, u32 tiles, u32 table_len) {
    __shared__ i64 red[8];
    __shared__ u32 off[CODER_TILES];
    const u32 t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const i64* tm = tmax + (u64)blockIdx.x * tiles;
    i64 mx = -LIMIT40;
    for (u32 k = t; k < tiles; k += 256) mx = tm[k] > mx ? tm[k] : mx;
    mx = warp_max_i64(mx);
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    i64 m = red[0];
#pragma unroll
    for (u32 k = 1; k < 8; k++) m = red[k] > m ? red[k] : m;
    for (u32 k = t; k < tiles; k += 256) {
        // INVARIANT: every tile max is at most the row max, both in +-2^40.
        const i64 dd = m - tm[k];
        off[k] = (u32)(dd < (i64)table_len - 1 ? dd : (i64)table_len - 1);
    }
    __syncthreads();
    WeightSrc w = {nullptr, nullptr, d16 + (u64)blockIdx.x * vocab, off, logits_exp, m, table_len};
    return w;
}

// Fused: logits_weights and coder_interval. The passes of the coder read each weight from the logit
// and the table. One block of 256 for each row.
extern "C" __global__ void __launch_bounds__(256) coder_logits(const i64* __restrict__ logit, const u64* __restrict__ logits_exp,
                                                               const u32* __restrict__ queries, u64* __restrict__ out, u32 vocab,
                                                               u32 table_len, u64 total) {
    const WeightSrc w = src_logits(logit, logits_exp, vocab, table_len);
    coder_body(w, queries, out, vocab, total);
}

// Fused: logits_weights and coder_interval on the compact logits of gemm_tc*_logits (src_d16).
extern "C" __global__ void __launch_bounds__(256) coder_d16(const u16* __restrict__ d16, const i64* __restrict__ tmax, const u64* __restrict__ logits_exp,
                                                            const u32* __restrict__ queries, u64* __restrict__ out, u32 vocab, u32 tiles,
                                                            u32 table_len, u64 total) {
    const WeightSrc w = src_d16(d16, tmax, logits_exp, vocab, tiles, table_len);
    coder_body(w, queries, out, vocab, total);
}

// ---------------------------------------------------------------- the mixer of the logs profile

// The mixer of ADR 0010 section 3 (taco_match::Mixer) runs in two parts with a host step between
// them. The match model and the adaptive state of the mixer stay on the host.
//
// 1. mix_stats_*: for each row, out = (W, w[p], top) of the GPT-2 weights w: W = sum(w), w[p] for
//    the prediction p of the row (0 without a prediction), and top = the token with the largest
//    weight, the lowest id on a tie, or 0 if every weight is 0 (rule 1 to 3 of Mixer::mix: the
//    loop that keeps the first strictly larger weight, starting at token 0 with weight 0).
// 2. The host computes q from (W, w[p], top == p) with the mixer state, then R = max(W - w[p], 1),
//    m = 2^16 - q, k = max(0, bitlen(R) + 16 - 62) and w'[p] = max(1, (R * q) >> k) (rule 5).
// 3. coder_mixed_*: the coder interval of from_weights(w'), with w'[i] = max(1, (w[i] * m) >> k)
//    for i != p, and w'[i] = max(1, w[i]) for a row without a prediction (rule 1). A plain row
//    (a block of GPT-2 alone in the same pass) has w' = w: the table of coder_interval.
//
// The mix record of a row: 8 u32 words: p (MIX_NONE: no prediction; MIX_PLAIN: a plain row), k,
// m, 0, w'[p] (low word, high word), 0, 0. The stats kernel reads only p.
#define MIX_NONE 0xffffffffu
#define MIX_PLAIN 0xfffffffeu

struct MixedSrc {
    WeightSrc base;
    u32 p, k;
    u64 m, wp;
    __device__ __forceinline__ u64 at(u32 i) const {
        const u64 w = base.at(i);
        if (p == MIX_PLAIN) return w;
        if (p == MIX_NONE) return w > 1 ? w : 1;
        if (i == p) return wp;
        // INVARIANT: w <= R for i != p, m <= 2^16 and k = max(0, bitlen(R) + 16 - 62) < 128, so
        // (w * m) >> k < 2^62 and fits in u64: the value of the u128 and u64 loops of Mixer::mix.
        const u64 v = (u64)(((u128)w * m) >> k);
        return v > 1 ? v : 1;
    }
};

__device__ __forceinline__ MixedSrc mixed_of(const WeightSrc& base, const u32* __restrict__ mix) {
    const u32* r = mix + 8 * (u64)blockIdx.x;
    MixedSrc s;
    s.base = base;
    s.p = r[0];
    s.k = r[1];
    s.m = r[2];
    s.wp = (u64)r[4] | ((u64)r[5] << 32);
    return s;
}

// Part 1. One block of 256 for each row. Each thread reads the ids t, t + 256, ... in increasing
// order, so `>` keeps the lowest id of a tie; the reduction keeps the lower id of two equal maxima.
// INVARIANT: GPT-2 weights are at most 2^40 and vocab <= 2^20, so the u64 sum is exact (the u128
// sum of Mixer::mix has the same value).
__device__ void mix_stats_body(const WeightSrc& w, const u32* __restrict__ mix, u64* __restrict__ out, u32 vocab) {
    __shared__ u64 cs[256];
    __shared__ u64 bw[256];
    __shared__ u32 bi[256];
    const u32 t = threadIdx.x, row = blockIdx.x;
    u64 s = 0, best = 0;
    u32 besti = 0xffffffffu;
    for (u32 i = t; i < vocab; i += 256) {
        const u64 wi = w.at(i);
        s += wi;
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
        const u32 p = mix[8 * (u64)row];
        out[3 * (u64)row] = cs[0];
        out[3 * (u64)row + 1] = p < vocab ? w.at(p) : 0;
        out[3 * (u64)row + 2] = cs[0] == 0 ? 0 : bi[0];
    }
}

extern "C" __global__ void __launch_bounds__(256) mix_stats_weights(const u64* __restrict__ weights, const u32* __restrict__ mix,
                                                                    u64* __restrict__ out, u32 vocab) {
    WeightSrc w = {weights + (u64)blockIdx.x * vocab, nullptr, nullptr, nullptr, nullptr, 0, 0};
    mix_stats_body(w, mix, out, vocab);
}

extern "C" __global__ void __launch_bounds__(256) mix_stats_logits(const i64* __restrict__ logit, const u64* __restrict__ logits_exp,
                                                                   const u32* __restrict__ mix, u64* __restrict__ out, u32 vocab,
                                                                   u32 table_len) {
    const WeightSrc w = src_logits(logit, logits_exp, vocab, table_len);
    mix_stats_body(w, mix, out, vocab);
}

extern "C" __global__ void __launch_bounds__(256) mix_stats_d16(const u16* __restrict__ d16, const i64* __restrict__ tmax,
                                                                const u64* __restrict__ logits_exp, const u32* __restrict__ mix,
                                                                u64* __restrict__ out, u32 vocab, u32 tiles, u32 table_len) {
    const WeightSrc w = src_d16(d16, tmax, logits_exp, vocab, tiles, table_len);
    mix_stats_body(w, mix, out, vocab);
}

// Part 3: the coder interval of the mixed weights (coder_body with MixedSrc).
extern "C" __global__ void __launch_bounds__(256) coder_mixed_weights(const u64* __restrict__ weights, const u32* __restrict__ mix,
                                                                      const u32* __restrict__ queries, u64* __restrict__ out, u32 vocab,
                                                                      u64 total) {
    WeightSrc base = {weights + (u64)blockIdx.x * vocab, nullptr, nullptr, nullptr, nullptr, 0, 0};
    const MixedSrc w = mixed_of(base, mix);
    coder_body(w, queries, out, vocab, total);
}

extern "C" __global__ void __launch_bounds__(256) coder_mixed_logits(const i64* __restrict__ logit, const u64* __restrict__ logits_exp,
                                                                     const u32* __restrict__ mix, const u32* __restrict__ queries,
                                                                     u64* __restrict__ out, u32 vocab, u32 table_len, u64 total) {
    const MixedSrc w = mixed_of(src_logits(logit, logits_exp, vocab, table_len), mix);
    coder_body(w, queries, out, vocab, total);
}

extern "C" __global__ void __launch_bounds__(256) coder_mixed_d16(const u16* __restrict__ d16, const i64* __restrict__ tmax,
                                                                  const u64* __restrict__ logits_exp, const u32* __restrict__ mix,
                                                                  const u32* __restrict__ queries, u64* __restrict__ out, u32 vocab,
                                                                  u32 tiles, u32 table_len, u64 total) {
    const MixedSrc w = mixed_of(src_d16(d16, tmax, logits_exp, vocab, tiles, table_len), mix);
    coder_body(w, queries, out, vocab, total);
}
