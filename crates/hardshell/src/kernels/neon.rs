//! aarch64 NEON kernels with the `sdot` instruction (FEAT_DotProd, `vdotq_s32`).
//!
//! `sdot` multiplies four groups of four `i8` pairs and adds each group to one `i32` lane.
//! The products and the sums are exact (no saturation). `fast::Kernels` calls these functions
//! only after `is_aarch64_feature_detected!("dotprod")` returned true.
//!
//! INVARIANT: a lane adds a subset of the products of one dot product, so the bound of `fast`
//! holds: no lane, no partial sum and no tail sum overflows an `i32`.
//!
//! NOTE: Miri cannot run these intrinsics. `tests/fast_kernels.rs` compares this module with
//! the reference kernels on random shapes and extreme values, on every aarch64 test run. The
//! portable kernels contain no `unsafe` code.
#![allow(unsafe_code)]

use core::arch::aarch64::{
    int8x16_t, vaddq_s32, vaddvq_s32, vdotq_s32, vdupq_n_s32, vld1q_s8, vpaddq_s32, vst1q_s32,
};

use super::fast::Shape;
use super::portable;

/// A proof that the CPU has the feature `dotprod`. Only [`DotProd::detect`] makes one, so the safe
/// functions below can call the `#[target_feature]` functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct DotProd(());

impl DotProd {
    /// The proof, if the CPU has the feature. Cost: one CPU feature query.
    pub(super) fn detect() -> Option<Self> {
        if std::arch::is_aarch64_feature_detected!("dotprod") { Some(Self(())) } else { None }
    }
}

/// The dot product of `a` and `b`, which have the same length.
pub(super) fn dot(_cpu: DotProd, a: &[i8], b: &[i8]) -> i32 {
    // SAFETY: a `DotProd` value exists only after the run-time check found `dotprod`.
    unsafe { dot_impl(a, b) }
}

/// `out[r][o] = w[o] . x[r]` for the checked `shape`.
pub(super) fn matmul(_cpu: DotProd, w: &[i8], x: &[i8], shape: Shape, out: &mut [i32]) {
    // SAFETY: a `DotProd` value exists only after the run-time check found `dotprod`. The
    // functions below check every slice length before they read.
    unsafe { matmul_impl(w, x, shape, out) }
}

/// The number of `i8` values in one NEON register.
const W: usize = 16;

/// Loads 16 values. NEON is part of the aarch64 base architecture, so this needs no feature.
#[inline]
fn load(chunk: &[i8; W]) -> int8x16_t {
    // SAFETY: `chunk` is a reference to exactly 16 `i8` values, so the 16-byte load reads
    // only valid memory. `vld1q_s8` has no alignment requirement.
    unsafe { vld1q_s8(chunk.as_ptr()) }
}

/// The dot product of `a` and `b`, which have the same length.
///
/// # Safety
/// The CPU must have FEAT_DotProd.
#[target_feature(enable = "dotprod")]
unsafe fn dot_impl(a: &[i8], b: &[i8]) -> i32 {
    let (a64, a_rest) = a.as_chunks::<64>();
    let (b64, b_rest) = b.as_chunks::<64>();
    // NOTE: four accumulators hide the latency of `sdot`.
    let mut acc = [vdupq_n_s32(0); 4];
    for (x, y) in a64.iter().zip(b64) {
        let (xs, ys) = (x.as_chunks::<W>().0, y.as_chunks::<W>().0);
        for (s, (xc, yc)) in acc.iter_mut().zip(xs.iter().zip(ys)) {
            *s = vdotq_s32(*s, load(xc), load(yc));
        }
    }
    let (a16, a_tail) = a_rest.as_chunks::<W>();
    let (b16, b_tail) = b_rest.as_chunks::<W>();
    let mut s0 = acc[0];
    for (xc, yc) in a16.iter().zip(b16) {
        s0 = vdotq_s32(s0, load(xc), load(yc));
    }
    let v = vaddq_s32(vaddq_s32(s0, acc[1]), vaddq_s32(acc[2], acc[3]));
    vaddvq_s32(v) + portable::dot(a_tail, b_tail)
}

/// One tile: the dot products of `RW` weight rows with `RX` input rows, over `cols` values.
/// Each weight register is used `RX` times and each input register `RW` times.
///
/// # Safety
/// The CPU must have FEAT_DotProd. The function checks the slice lengths itself: it reads at
/// most `min(cols, shortest slice)` values of each slice.
#[target_feature(enable = "dotprod")]
unsafe fn tile<const RW: usize, const RX: usize>(
    w: [&[i8]; RW],
    x: [&[i8]; RX],
    cols: usize,
) -> [[i32; RX]; RW] {
    let len = w.iter().chain(x.iter()).map(|s| s.len()).fold(cols, usize::min);
    let blocks = len / W;
    let wp = w.map(<[i8]>::as_ptr);
    let xp = x.map(<[i8]>::as_ptr);
    let mut acc = [[vdupq_n_s32(0); RX]; RW];
    for c in 0..blocks {
        let off = c * W;
        // SAFETY: every slice has at least `len >= blocks * 16 >= off + 16` values, so each
        // 16-byte load at `off` stays inside its slice.
        let wv = wp.map(|p| unsafe { vld1q_s8(p.add(off)) });
        // SAFETY: as above.
        let xv = xp.map(|p| unsafe { vld1q_s8(p.add(off)) });
        for (row, &wi) in acc.iter_mut().zip(&wv) {
            for (s, &xj) in row.iter_mut().zip(&xv) {
                *s = vdotq_s32(*s, wi, xj);
            }
        }
    }
    let tail = blocks * W;
    let mut out = [[0i32; RX]; RW];
    for (o_row, a_row) in out.iter_mut().zip(&acc) {
        if RX == 4 {
            // NOTE: pairwise adds give the four sums of one weight row in one register.
            let v = vpaddq_s32(vpaddq_s32(a_row[0], a_row[1]), vpaddq_s32(a_row[2], a_row[3]));
            let mut sums = [0i32; 4];
            // SAFETY: `sums` has 4 values, so the 16-byte store writes only valid memory.
            unsafe { vst1q_s32(sums.as_mut_ptr(), v) };
            for (o, &v) in o_row.iter_mut().zip(&sums) {
                *o = v;
            }
        } else {
            for (o, &a) in o_row.iter_mut().zip(a_row) {
                *o = vaddvq_s32(a);
            }
        }
    }
    if tail < len {
        for (o_row, w_row) in out.iter_mut().zip(&w) {
            let w_tail = w_row.get(tail..len).unwrap_or(&[]);
            for (o, x_row) in o_row.iter_mut().zip(&x) {
                *o += portable::dot(w_tail, x_row.get(tail..len).unwrap_or(&[]));
            }
        }
    }
    out
}

/// Writes a tile into `out` (`n` rows of `rows` values) at weight row `o` and input row `r`.
fn store<const RW: usize, const RX: usize>(
    out: &mut [i32],
    rows: usize,
    o: usize,
    r: usize,
    t: &[[i32; RX]; RW],
) {
    for (i, t_row) in t.iter().enumerate() {
        for (j, &v) in t_row.iter().enumerate() {
            if let Some(slot) = out.get_mut((r + j) * rows + o + i) {
                *slot = v;
            }
        }
    }
}

/// All tiles of `RW` weight rows that start at row `o`, for every input row.
///
/// # Safety
/// The CPU must have FEAT_DotProd.
#[target_feature(enable = "dotprod")]
unsafe fn row_group<const RW: usize>(w: &[i8], x: &[i8], s: Shape, o: usize, out: &mut [i32]) {
    let wr: [&[i8]; RW] = core::array::from_fn(|i| s.w_row(w, o + i));
    let mut r = 0;
    // NOTE: a tile of 4 x 6 gives 123 GMAC/s and 4 x 4 gives 77 to 114 GMAC/s (measured, M2,
    // 3,072 x 768 weights, 32 input rows, other load on the machine).
    while r + 6 <= s.n {
        let xr: [&[i8]; 6] = core::array::from_fn(|j| s.x_row(x, r + j));
        // SAFETY: the caller has FEAT_DotProd; `tile` checks the lengths.
        let t = unsafe { tile::<RW, 6>(wr, xr, s.cols) };
        store(out, s.rows, o, r, &t);
        r += 6;
    }
    while r + 4 <= s.n {
        let xr: [&[i8]; 4] = core::array::from_fn(|j| s.x_row(x, r + j));
        // SAFETY: the caller has FEAT_DotProd; `tile` checks the lengths.
        let t = unsafe { tile::<RW, 4>(wr, xr, s.cols) };
        store(out, s.rows, o, r, &t);
        r += 4;
    }
    let xr = |j: usize| s.x_row(x, r + j);
    // SAFETY (all three arms): the caller has FEAT_DotProd; `tile` checks the lengths.
    match s.n - r {
        3 => store(out, s.rows, o, r, &unsafe { tile::<RW, 3>(wr, [xr(0), xr(1), xr(2)], s.cols) }),
        2 => store(out, s.rows, o, r, &unsafe { tile::<RW, 2>(wr, [xr(0), xr(1)], s.cols) }),
        1 => store(out, s.rows, o, r, &unsafe { tile::<RW, 1>(wr, [xr(0)], s.cols) }),
        _ => {}
    }
}

/// `out[r][o] = w[o] . x[r]` for the checked `shape`.
///
/// Weight rows go in groups of 4, and the input rows go in groups of 6 (then 4, then the rest)
/// inside each group, so each weight register serves up to six input rows. NOTE: for one input row, groups of 4
/// weight rows read 38.4 GB/s and groups of 8 read 36.0 GB/s (measured, M2, `wte`).
///
/// # Safety
/// The CPU must have FEAT_DotProd.
#[target_feature(enable = "dotprod")]
unsafe fn matmul_impl(w: &[i8], x: &[i8], shape: Shape, out: &mut [i32]) {
    let mut o = 0;
    while o + 4 <= shape.rows {
        // SAFETY: the caller has FEAT_DotProd.
        unsafe { row_group::<4>(w, x, shape, o, out) };
        o += 4;
    }
    while o < shape.rows {
        // SAFETY: the caller has FEAT_DotProd.
        unsafe { row_group::<1>(w, x, shape, o, out) };
        o += 1;
    }
}
