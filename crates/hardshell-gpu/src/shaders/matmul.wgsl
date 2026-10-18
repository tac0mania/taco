// i8 x i8 -> i32 matrix products with packed 4 x 8-bit dot products (plan §7.1).
//
// out[m][n] = sum_k a[m][k] * w[n][k]. Both operands are row-major with k contiguous, four i8
// values in each u32 word (byte 0 = the lowest k). A row has `kw` words; `kw` is a multiple of
// 16 (64 bytes) and the padding bytes are zero, so they add nothing.
//
// DETERMINISM: dot4I8Packed gives the exact sum of four products of signed bytes. Every
// accumulator is an i32 sum of at most 2^16 products of magnitude at most 2^14, so its
// magnitude is at most 2^30 and it cannot overflow (hardshell::kernels::MAX_DOT_LEN_I32). Integer
// addition is associative, so the order of the sums does not change the result.

struct MatParams {
    // The number of rows of `w` (output columns).
    n: u32,
    // Words in each row of `a` and `w`.
    kw: u32,
    // The distance between two rows of `out`, in i32 values.
    out_stride: u32,
    // The number of rows of `a`. The row table gives it for a model step; this field is used
    // when `use_rows` is 0.
    m: u32,
    use_rows: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;
@group(0) @binding(3) var<uniform> p: MatParams;
@group(0) @binding(4) var<storage, read> rows: Rows;

fn row_count() -> u32 {
    if (p.use_rows != 0u) { return rows.count; }
    return p.m;
}

// ---------- Few rows: one pass over `w` serves up to 8 rows of `a` ----------

const MV_ROWS: u32 = 8u;
const MV_LANES: u32 = 32u;
var<workgroup> mv_red: array<i32, 2048>;

// Workgroup: 8 rows of `w`, 32 invocations for each row. Grid: (ceil(n / 8), ceil(m / 8)).
@compute @workgroup_size(256)
fn matvec(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let local_row = li / MV_LANES;
    let lane = li % MV_LANES;
    let n = wg.x * MV_ROWS + local_row;
    let m0 = wg.y * MV_ROWS;
    let m_total = row_count();
    let cnt = min(MV_ROWS, m_total - min(m0, m_total));
    var acc = array<i32, 8>(0i, 0i, 0i, 0i, 0i, 0i, 0i, 0i);
    if (n < p.n) {
        let wbase = n * p.kw;
        for (var k = lane; k < p.kw; k = k + MV_LANES) {
            let wv = w[wbase + k];
            for (var r = 0u; r < cnt; r = r + 1u) {
                acc[r] = acc[r] + dot4I8Packed(a[(m0 + r) * p.kw + k], wv);
            }
        }
    }
    for (var r = 0u; r < MV_ROWS; r = r + 1u) {
        mv_red[(r * 8u + local_row) * MV_LANES + lane] = acc[r];
    }
    workgroupBarrier();
    // Tree sum over the 32 lanes of each (row of a, row of w) pair.
    for (var step = MV_LANES / 2u; step > 0u; step = step / 2u) {
        if (lane < step) {
            for (var r = 0u; r < MV_ROWS; r = r + 1u) {
                let i = (r * 8u + local_row) * MV_LANES + lane;
                mv_red[i] = mv_red[i] + mv_red[i + step];
            }
        }
        workgroupBarrier();
    }
    if (lane < cnt && n < p.n) {
        out[(m0 + lane) * p.out_stride + n] = mv_red[(lane * 8u + local_row) * MV_LANES];
    }
}

// ---------- Many rows: 64 x 64 tiles ----------

const T: u32 = 64u;
const KT: u32 = 16u;
const LD: u32 = 17u;
var<workgroup> as_tile: array<u32, 1088>;
var<workgroup> ws_tile: array<u32, 1088>;

// Workgroup: a 64 x 64 tile of `out`; each invocation makes 4 x 4 values.
// Grid: (ceil(n / 64), ceil(m / 64)).
@compute @workgroup_size(16, 16)
fn gemm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let m0 = wg.y * T;
    let n0 = wg.x * T;
    let m_total = row_count();
    var acc: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0i; }
    for (var k0 = 0u; k0 < p.kw; k0 = k0 + KT) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = li + i * 256u;
            let r = idx / KT;
            let c = idx % KT;
            var av = 0u;
            if (m0 + r < m_total) { av = a[(m0 + r) * p.kw + k0 + c]; }
            as_tile[r * LD + c] = av;
            var wv = 0u;
            if (n0 + r < p.n) { wv = w[(n0 + r) * p.kw + k0 + c]; }
            ws_tile[r * LD + c] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < KT; kk = kk + 1u) {
            var av: array<u32, 4>;
            var wv: array<u32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                av[i] = as_tile[(lid.y + 16u * i) * LD + kk];
                wv[i] = ws_tile[(lid.x + 16u * i) * LD + kk];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + dot4I8Packed(av[i], wv[j]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + lid.y + 16u * i;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let n = n0 + lid.x + 16u * j;
            if (m < m_total && n < p.n) { out[m * p.out_stride + n] = acc[i * 4u + j]; }
        }
    }
}
