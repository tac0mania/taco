//! The fast kernels: one entry point for each kernel, with a run-time choice of the instruction
//! set (NEON with `sdot` on aarch64, AVX2 on x86_64, or portable Rust on every machine).
//!
//! DETERMINISM: every variant calculates the exact integer result, so every variant gives the
//! same bits as the reference kernels of the parent module (CLAUDE.md §1 rule 10). The proof
//! is the bound argument below and the property tests (`tests/fast_kernels.rs`), which compare
//! each variant with the reference on random shapes and on the extreme values -128 and 127.
//!
//! INVARIANT: the `i32` bound of `MAX_DOT_LEN_I32` holds for every grouping of the sum. Each
//! product has a magnitude of at most `2^14`. A SIMD lane, a partial sum of lanes, or a scalar
//! tail adds a subset of the products of one dot product, so its magnitude is at most
//! `len * 2^14 <= 2^30 < 2^31`. Thus no partial sum overflows, and the order of the additions
//! does not change the result.

use super::{MAX_DOT_LEN_I32, MAX_DOT_LEN_I64, portable};
use crate::error::{Error, Result};
use crate::int::{clamp, clamp_wide, rescale};

/// The instruction set of a [`Kernels`] value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Isa {
    /// Portable Rust. The compiler can vectorise it, but it uses no `unsafe` code.
    Portable,
    /// aarch64 NEON with the `sdot` instruction (FEAT_DotProd).
    #[cfg(target_arch = "aarch64")]
    NeonDot(super::neon::DotProd),
    /// x86_64 AVX2: sign extension to `i16`, then `vpmaddwd` (exact, it does not saturate).
    #[cfg(target_arch = "x86_64")]
    Avx2(super::avx2::Avx2),
}

/// A selection of kernel implementations. All selections give the same bits.
///
/// A SIMD selection holds a proof value of its SIMD module, which only the run-time check of
/// the CPU feature makes. Thus the SIMD code always runs on a CPU that has the feature. This
/// module has no `unsafe` code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Kernels {
    isa: Isa,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::detect()
    }
}

impl Kernels {
    /// The fastest kernels for this CPU. Cost: one CPU feature query (the standard library
    /// keeps the answer).
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        if let Some(cpu) = super::neon::DotProd::detect() {
            return Self { isa: Isa::NeonDot(cpu) };
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(cpu) = super::avx2::Avx2::detect() {
            return Self { isa: Isa::Avx2(cpu) };
        }
        Self::portable()
    }

    /// The portable kernels: safe Rust only, on every machine.
    #[must_use]
    pub const fn portable() -> Self {
        Self { isa: Isa::Portable }
    }

    /// Every selection that this CPU can run: the portable kernels first, then the SIMD
    /// kernels if the CPU has the feature. Tests use this list to compare all of them.
    #[must_use]
    pub fn available() -> Vec<Self> {
        let mut all = vec![Self::portable()];
        let best = Self::detect();
        if best != Self::portable() {
            all.push(best);
        }
        all
    }

    /// A short name of the instruction set: `"portable"`, `"neon-dotprod"` or `"avx2"`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self.isa {
            Isa::Portable => "portable",
            #[cfg(target_arch = "aarch64")]
            Isa::NeonDot(_) => "neon-dotprod",
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2(_) => "avx2",
        }
    }

    /// The dot product of two `i8` vectors with an `i32` accumulator: the same result as
    /// [`super::dot_i8`]. If the lengths differ, the extra values of the longer vector are not
    /// used. The caller keeps the length at most [`MAX_DOT_LEN_I32`]. Cost: one pass.
    #[must_use]
    pub fn dot_i8(self, a: &[i8], b: &[i8]) -> i32 {
        let n = a.len().min(b.len());
        let (a, b) = (a.get(..n).unwrap_or(&[]), b.get(..n).unwrap_or(&[]));
        match self.isa {
            Isa::Portable => portable::dot(a, b),
            #[cfg(target_arch = "aarch64")]
            Isa::NeonDot(cpu) => super::neon::dot(cpu, a, b),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2(cpu) => super::avx2::dot(cpu, a, b),
        }
    }

    /// The matrix product `out[r][o] = w[o] . x[r]`, with `i32` accumulators.
    ///
    /// `w` holds `rows` rows of `cols` values, `x` holds `n` rows of `cols` values, and `out`
    /// holds `n` rows of `rows` values, all row-major. Every value of `out` is the exact dot
    /// product, the same as [`super::dot_i8`]. With `n` rows of `x`, each weight is read from
    /// memory once for all of them. Cost: `rows * cols * n` multiply-adds.
    ///
    /// # Errors
    /// [`Error::DotTooLong`] if `cols > MAX_DOT_LEN_I32`; [`Error::LengthMismatch`] if `cols`
    /// is 0 or does not divide `w.len()` or `x.len()`, or if `out.len()` is not
    /// `n * rows`.
    pub fn matmul_i8(self, w: &[i8], x: &[i8], cols: usize, out: &mut [i32]) -> Result<()> {
        let rows = if cols > 0 && w.len().is_multiple_of(cols) { w.len() / cols } else { 0 };
        if rows == 0 && !w.is_empty() {
            return Err(Error::LengthMismatch { expected: cols, found: w.len() });
        }
        self.matmul_i8_strided(MatRef { data: w, rows, stride: cols }, x, cols, out)
    }

    /// The same product as [`Kernels::matmul_i8`], but row `o` of the weights is
    /// `w.data[o * w.stride..o * w.stride + cols]`: the weights can be the first `cols` columns
    /// of a wider matrix.
    ///
    /// # Errors
    /// [`Error::DotTooLong`] if `cols > MAX_DOT_LEN_I32`; [`Error::LengthMismatch`] if `cols`
    /// is 0, `w.stride < cols`, `w.data` is too short, `cols` does not divide `x.len()`, or
    /// `out.len()` is not `n * w.rows`.
    pub fn matmul_i8_strided(
        self,
        w: MatRef<'_>,
        x: &[i8],
        cols: usize,
        out: &mut [i32],
    ) -> Result<()> {
        if cols > MAX_DOT_LEN_I32 {
            return Err(Error::DotTooLong { len: cols, max: MAX_DOT_LEN_I32 });
        }
        let MatRef { data, rows, stride } = w;
        let need = if rows == 0 { 0 } else { (rows - 1) * stride + cols };
        if cols == 0 || stride < cols || data.len() < need || !x.len().is_multiple_of(cols) {
            return Err(Error::LengthMismatch { expected: need, found: data.len() });
        }
        let n = x.len() / cols;
        if out.len() != rows * n {
            return Err(Error::LengthMismatch { expected: rows * n, found: out.len() });
        }
        let shape = Shape { rows, cols, n, stride };
        let w = data;
        match self.isa {
            Isa::Portable => portable::matmul(w, x, shape, out),
            #[cfg(target_arch = "aarch64")]
            Isa::NeonDot(cpu) => super::neon::matmul(cpu, w, x, shape, out),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2(cpu) => super::avx2::matmul(cpu, w, x, shape, out),
        }
        Ok(())
    }
}

impl Kernels {
    /// The matrix product `out[r][o] = w[o] . x[r]` for `i8` weights and `i16` activations,
    /// with `i64` results: the same values as [`super::dot_i8_i16`]. This is the product of
    /// the `i16` activations of hsm format version 2 (ADR 0004 addendum B).
    ///
    /// The shapes are the shapes of [`Kernels::matmul_i8`]. Every selection uses the portable
    /// code (`portable::matmul_i8_i16`).
    ///
    /// PERF: no SIMD variant yet. The portable loop sums blocks of 256 products in `i32`, which
    /// the compiler can vectorise.
    ///
    /// # Errors
    /// [`Error::DotTooLong`] if `cols > MAX_DOT_LEN_I64`; [`Error::LengthMismatch`] if `cols`
    /// is 0 or does not divide `w.len()` or `x.len()`, or if `out.len()` is not `n * rows`.
    pub fn matmul_i8_i16(self, w: &[i8], x: &[i16], cols: usize, out: &mut [i64]) -> Result<()> {
        if cols > MAX_DOT_LEN_I64 {
            return Err(Error::DotTooLong { len: cols, max: MAX_DOT_LEN_I64 });
        }
        if cols == 0 || !w.len().is_multiple_of(cols) || !x.len().is_multiple_of(cols) {
            return Err(Error::LengthMismatch { expected: cols, found: w.len() });
        }
        let (rows, n) = (w.len() / cols, x.len() / cols);
        if out.len() != rows * n {
            return Err(Error::LengthMismatch { expected: rows * n, found: out.len() });
        }
        portable::matmul_i8_i16(w, x, cols, out);
        Ok(())
    }
}

/// A borrowed `i8` matrix of `rows` rows. Row `o` starts at `o * stride` in `data`.
#[derive(Clone, Copy, Debug)]
pub struct MatRef<'a> {
    /// The values.
    pub data: &'a [i8],
    /// The number of rows.
    pub rows: usize,
    /// The distance between the starts of two rows, in values.
    pub stride: usize,
}

/// The checked shape of one matrix product. `w` has `rows` rows of `cols` values at a distance
/// of `stride`, `x` has `n * cols` values and `out` has `n * rows` values.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shape {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) n: usize,
    /// The distance between two weight rows (`>= cols`).
    pub(super) stride: usize,
}

impl Shape {
    /// Weight row `o`, or an empty slice if it is outside `w`.
    pub(super) fn w_row(self, w: &[i8], o: usize) -> &[i8] {
        w.get(o * self.stride..o * self.stride + self.cols).unwrap_or(&[])
    }

    /// Input row `r`, or an empty slice if it is outside `x`.
    pub(super) fn x_row(self, x: &[i8], r: usize) -> &[i8] {
        x.get(r * self.cols..(r + 1) * self.cols).unwrap_or(&[])
    }
}

/// `clamp(rescale(x, m, s), lo, hi)` (plan §7.2), with a fast path in `i64`.
///
/// The result is the same as `int::clamp_wide(int::rescale(x, m, s), lo, hi)` for every input.
/// The fast path is taken when `x * m` and `x * m + 2^(s-1)` fit in an `i64` and
/// `1 <= s <= 62`: then the `i64` arithmetic shift is the same floor division as the `i128`
/// shift of `rescale`. Other inputs use `rescale` itself. A property test compares the two.
/// Cost: one 64-bit multiplication on the fast path.
#[must_use]
#[inline]
pub fn rescale_clamp(x: i64, m: i32, s: u8, lo: i64, hi: i64) -> i64 {
    if s == 0 {
        if let Some(p) = x.checked_mul(i64::from(m)) {
            return clamp(p, lo, hi);
        }
    } else if s <= 62 {
        // INVARIANT: 1 <= s <= 62, so `1 << (s - 1)` and the shift are in range.
        if let Some(q) = x.checked_mul(i64::from(m)).and_then(|p| p.checked_add(1 << (s - 1))) {
            return clamp(q >> s, lo, hi);
        }
    }
    clamp_wide(rescale(x, m, s), lo, hi)
}
