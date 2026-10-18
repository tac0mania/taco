//! x86_64 AVX2 kernels.
//!
//! Each step sign-extends 16 `i8` values to 16 `i16` values (`vpmovsxbw`), then multiplies
//! pairs and adds each pair into one `i32` lane (`vpmaddwd`). The inputs are in `-128..=127`,
//! so a pair sum has a magnitude of at most `2 * 2^14 = 2^15`, and `vpmaddwd` is exact.
//!
//! NOTE: `vpmaddubsw` (`_mm256_maddubs_epi16`) saturates its `i16` result, so this module does
//! not use it. `fast::Kernels` calls these functions only after
//! `is_x86_feature_detected!("avx2")` returned true.
//!
//! INVARIANT: a lane adds a subset of the products of one dot product, so the bound of `fast`
//! holds: no lane, no partial sum and no tail sum overflows an `i32`.
//!
//! NOTE: Miri cannot run these intrinsics. `tests/fast_kernels.rs` compares this module with
//! the reference kernels on random shapes and extreme values, on every x86_64 test run.
#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_shuffle_epi32,
    _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
    _mm256_madd_epi16, _mm256_setzero_si256,
};

use super::fast::Shape;
use super::portable;

/// A proof that the CPU has the feature `avx2`. Only [`Avx2::detect`] makes one, so the safe
/// functions below can call the `#[target_feature]` functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Avx2(());

impl Avx2 {
    /// The proof, if the CPU has the feature. Cost: one CPU feature query.
    pub(super) fn detect() -> Option<Self> {
        if std::arch::is_x86_feature_detected!("avx2") { Some(Self(())) } else { None }
    }
}

/// The dot product of `a` and `b`, which have the same length.
pub(super) fn dot(_cpu: Avx2, a: &[i8], b: &[i8]) -> i32 {
    // SAFETY: a `Avx2` value exists only after the run-time check found `avx2`.
    unsafe { dot_impl(a, b) }
}

/// `out[r][o] = w[o] . x[r]` for the checked `shape`.
pub(super) fn matmul(_cpu: Avx2, w: &[i8], x: &[i8], shape: Shape, out: &mut [i32]) {
    // SAFETY: a `Avx2` value exists only after the run-time check found `avx2`. The
    // functions below check every slice length before they read.
    unsafe { matmul_impl(w, x, shape, out) }
}

/// The number of `i8` values in one step.
const W: usize = 16;

/// Loads 16 `i8` values at `p` and sign-extends them to 16 `i16` values.
///
/// # Safety
/// The CPU must have AVX2, and `p` must point to 16 readable `i8` values.
#[target_feature(enable = "avx2")]
unsafe fn load_i16(p: *const i8) -> __m256i {
    // SAFETY: the caller gives 16 readable bytes at `p`; `loadu` has no alignment requirement.
    let v = unsafe { _mm_loadu_si128(p.cast::<__m128i>()) };
    _mm256_cvtepi8_epi16(v)
}

/// The sum of the 8 `i32` lanes of `v`.
#[target_feature(enable = "avx2")]
fn hsum(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    _mm_cvtsi128_si32(s)
}

/// The dot product of `a` and `b`, which have the same length.
///
/// # Safety
/// The CPU must have AVX2.
#[target_feature(enable = "avx2")]
unsafe fn dot_impl(a: &[i8], b: &[i8]) -> i32 {
    let (a32, a_rest) = a.as_chunks::<32>();
    let (b32, b_rest) = b.as_chunks::<32>();
    // NOTE: two accumulators hide the latency of `vpmaddwd`.
    let (mut s0, mut s1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    for (xc, yc) in a32.iter().zip(b32) {
        // SAFETY: AVX2 is present; each chunk has 32 values, so offsets 0 and 16 are valid.
        let (x0, y0, x1, y1) = unsafe {
            let (px, py) = (xc.as_ptr(), yc.as_ptr());
            (load_i16(px), load_i16(py), load_i16(px.add(W)), load_i16(py.add(W)))
        };
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(x0, y0));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(x1, y1));
    }
    let (a16, a_tail) = a_rest.as_chunks::<W>();
    let (b16, b_tail) = b_rest.as_chunks::<W>();
    for (xc, yc) in a16.iter().zip(b16) {
        // SAFETY: AVX2 is present; each chunk is a reference to exactly 16 values.
        let (x, y) = unsafe { (load_i16(xc.as_ptr()), load_i16(yc.as_ptr())) };
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(x, y));
    }
    hsum(_mm256_add_epi32(s0, s1)) + portable::dot(a_tail, b_tail)
}

/// One tile: the dot products of `RW` weight rows with `RX` input rows, over `cols` values.
///
/// # Safety
/// The CPU must have AVX2. The function checks the slice lengths itself: it reads at most
/// `min(cols, shortest slice)` values of each slice.
#[target_feature(enable = "avx2")]
unsafe fn tile<const RW: usize, const RX: usize>(
    w: [&[i8]; RW],
    x: [&[i8]; RX],
    cols: usize,
) -> [[i32; RX]; RW] {
    let len = w.iter().chain(x.iter()).map(|s| s.len()).fold(cols, usize::min);
    let blocks = len / W;
    let wp = w.map(<[i8]>::as_ptr);
    let xp = x.map(<[i8]>::as_ptr);
    let mut acc = [[_mm256_setzero_si256(); RX]; RW];
    for c in 0..blocks {
        let off = c * W;
        // SAFETY: AVX2 is present, and every slice has at least `len >= off + 16` values.
        let xv = xp.map(|p| unsafe { load_i16(p.add(off)) });
        for (row, &p) in acc.iter_mut().zip(&wp) {
            // SAFETY: as above.
            let wi = unsafe { load_i16(p.add(off)) };
            for (s, &xj) in row.iter_mut().zip(&xv) {
                *s = _mm256_add_epi32(*s, _mm256_madd_epi16(wi, xj));
            }
        }
    }
    let tail = blocks * W;
    let mut out = [[0i32; RX]; RW];
    for ((o_row, a_row), w_row) in out.iter_mut().zip(&acc).zip(&w) {
        let w_tail = w_row.get(tail..len).unwrap_or(&[]);
        for ((o, &a), x_row) in o_row.iter_mut().zip(a_row).zip(&x) {
            let x_tail = x_row.get(tail..len).unwrap_or(&[]);
            *o = hsum(a) + portable::dot(w_tail, x_tail);
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

/// All tiles of `RW` weight rows that start at row `o`, for every input row. AVX2 has 16
/// vector registers, so a tile has at most 8 accumulators.
///
/// # Safety
/// The CPU must have AVX2.
#[target_feature(enable = "avx2")]
unsafe fn row_group<const RW: usize>(w: &[i8], x: &[i8], s: Shape, o: usize, out: &mut [i32]) {
    let wr: [&[i8]; RW] = core::array::from_fn(|i| s.w_row(w, o + i));
    let mut r = 0;
    while r + 2 <= s.n {
        // SAFETY: the caller has AVX2; `tile` checks the lengths.
        let t = unsafe { tile::<RW, 2>(wr, [s.x_row(x, r), s.x_row(x, r + 1)], s.cols) };
        store(out, s.rows, o, r, &t);
        r += 2;
    }
    if r < s.n {
        // SAFETY: the caller has AVX2; `tile` checks the lengths.
        let t = unsafe { tile::<RW, 1>(wr, [s.x_row(x, r)], s.cols) };
        store(out, s.rows, o, r, &t);
    }
}

/// `out[r][o] = w[o] . x[r]` for the checked `shape`: groups of 4 weight rows, and 2 input
/// rows in each tile.
///
/// # Safety
/// The CPU must have AVX2.
#[target_feature(enable = "avx2")]
unsafe fn matmul_impl(w: &[i8], x: &[i8], shape: Shape, out: &mut [i32]) {
    let mut o = 0;
    while o + 4 <= shape.rows {
        // SAFETY: the caller has AVX2.
        unsafe { row_group::<4>(w, x, shape, o, out) };
        o += 4;
    }
    while o < shape.rows {
        // SAFETY: the caller has AVX2.
        unsafe { row_group::<1>(w, x, shape, o, out) };
        o += 1;
    }
}
