//! Owned tensors of `i8`, `i16`, `i32` and `u64` values with a validated shape.
//!
//! A [`Tensor`] holds its values in row-major order. Its constructor checks that the number of
//! values agrees with the shape, so a kernel can trust the shape. A [`Matrix`] is a tensor with
//! two dimensions, `[rows, cols]`.

use crate::error::{Error, Result};

/// The largest number of dimensions of a tensor.
pub const MAX_DIMS: usize = 4;

/// The element type of a tensor, as the model file names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
}

impl DType {
    /// The size of one element in bytes.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::U64 => 8,
        }
    }

    /// The name of the type in the model file: `"i8"`, `"i16"`, `"i32"` or `"u64"`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::U64 => "u64",
        }
    }

    /// The type with this model-file name, or `None`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "i8" => Some(Self::I8),
            "i16" => Some(Self::I16),
            "i32" => Some(Self::I32),
            "u64" => Some(Self::U64),
            _ => None,
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for u64 {}
}

/// An element type that a tensor can hold. Only `i8`, `i16`, `i32` and `u64` implement it.
pub trait Element: Copy + Default + PartialEq + core::fmt::Debug + sealed::Sealed {
    /// The model-file type of this element.
    const DTYPE: DType;
    /// Reads one element from little-endian bytes. `bytes` has exactly `DTYPE.size()` bytes.
    fn read_le(bytes: &[u8]) -> Self;
    /// Appends the little-endian bytes of this element.
    fn write_le(self, out: &mut Vec<u8>);
}

macro_rules! element {
    ($t:ty, $d:expr) => {
        impl Element for $t {
            const DTYPE: DType = $d;
            fn read_le(bytes: &[u8]) -> Self {
                let mut a = [0u8; core::mem::size_of::<$t>()];
                // NOTE: the caller gives exactly `size_of` bytes (chunks_exact). With a
                // shorter slice the missing bytes stay zero; nothing panics.
                for (d, s) in a.iter_mut().zip(bytes) {
                    *d = *s;
                }
                <$t>::from_le_bytes(a)
            }
            fn write_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}
element!(i8, DType::I8);
element!(i16, DType::I16);
element!(i32, DType::I32);
element!(u64, DType::U64);

/// A validated shape: 1 to [`MAX_DIMS`] dimensions, each at least 1, with an element count
/// that fits in `usize`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
    numel: usize,
}

impl Shape {
    /// Validates a shape. Cost: one pass over the dimensions.
    ///
    /// # Errors
    /// [`Error::InvalidShape`] if there are no dimensions, too many, a zero dimension, or if
    /// the element count overflows `usize` or is more than `isize::MAX`.
    pub fn new(dims: &[usize]) -> Result<Self> {
        if dims.is_empty() {
            return Err(Error::InvalidShape { reason: "a shape needs at least one dimension" });
        }
        if dims.len() > MAX_DIMS {
            return Err(Error::InvalidShape { reason: "a shape has at most 4 dimensions" });
        }
        let mut numel: usize = 1;
        for &d in dims {
            if d == 0 {
                return Err(Error::InvalidShape { reason: "a dimension is zero" });
            }
            numel = numel
                .checked_mul(d)
                .ok_or(Error::InvalidShape { reason: "the element count overflows" })?;
        }
        if isize::try_from(numel).is_err() {
            return Err(Error::InvalidShape { reason: "the element count is too large" });
        }
        Ok(Self { dims: dims.to_vec(), numel })
    }

    /// The dimensions, outermost first.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// The number of elements: the product of the dimensions.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.numel
    }
}

/// An owned tensor with a validated shape. The values are in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor<T: Element> {
    shape: Shape,
    data: Vec<T>,
}

impl<T: Element> Tensor<T> {
    /// Makes a tensor. Cost: one shape check; the data is moved, not copied.
    ///
    /// # Errors
    /// [`Error::InvalidShape`] for a bad shape, [`Error::LengthMismatch`] if `data.len()` is not
    /// the element count of the shape.
    pub fn new(dims: &[usize], data: Vec<T>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if data.len() != shape.numel() {
            return Err(Error::LengthMismatch { expected: shape.numel(), found: data.len() });
        }
        Ok(Self { shape, data })
    }

    /// The shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The values in row-major order.
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Takes the values out of the tensor.
    #[must_use]
    pub fn into_data(self) -> Vec<T> {
        self.data
    }
}

/// An owned matrix `[rows, cols]` in row-major order. Row `r` is `data[r * cols..(r + 1) * cols]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<T: Element> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Element> Matrix<T> {
    /// Makes a matrix from a tensor with exactly two dimensions.
    ///
    /// # Errors
    /// [`Error::InvalidShape`] if the tensor does not have two dimensions.
    pub fn from_tensor(t: Tensor<T>) -> Result<Self> {
        match *t.shape.dims() {
            [rows, cols] => Ok(Self { rows, cols, data: t.data }),
            _ => Err(Error::InvalidShape { reason: "a matrix needs exactly two dimensions" }),
        }
    }

    /// Makes a matrix from its dimensions and values.
    ///
    /// # Errors
    /// The errors of [`Tensor::new`].
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self> {
        Self::from_tensor(Tensor::new(&[rows, cols], data)?)
    }

    /// The number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns: the length of each row.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The values in row-major order.
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Row `r`, or `None` if `r >= rows`. Cost: no copy.
    #[must_use]
    pub fn row(&self, r: usize) -> Option<&[T]> {
        let start = r.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.data.get(start..end)
    }

    /// An iterator over the rows, in order.
    pub fn row_iter(&self) -> impl ExactSizeIterator<Item = &[T]> + '_ {
        self.data.chunks_exact(self.cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_refuses_bad_dimensions() {
        assert!(Shape::new(&[]).is_err());
        assert!(Shape::new(&[1, 0]).is_err());
        assert!(Shape::new(&[1, 1, 1, 1, 1]).is_err());
        assert!(Shape::new(&[usize::MAX, 2]).is_err());
        assert_eq!(Shape::new(&[3, 4]).unwrap().numel(), 12);
    }

    #[test]
    fn tensor_refuses_wrong_length() {
        assert_eq!(
            Tensor::<i8>::new(&[2, 3], vec![0; 5]),
            Err(Error::LengthMismatch { expected: 6, found: 5 })
        );
    }

    #[test]
    fn matrix_rows_are_row_major() {
        let m = Matrix::new(2, 3, vec![1i8, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(m.row(1), Some(&[4i8, 5, 6][..]));
        assert_eq!(m.row(2), None);
        assert_eq!(m.row_iter().count(), 2);
        assert!(Matrix::from_tensor(Tensor::new(&[6], vec![0i8; 6]).unwrap()).is_err());
    }

    #[test]
    fn elements_round_trip_little_endian() {
        let mut out = Vec::new();
        (-2i16).write_le(&mut out);
        0x0102_0304i32.write_le(&mut out);
        assert_eq!(out, [0xfe, 0xff, 4, 3, 2, 1]);
        assert_eq!(i16::read_le(&out[0..2]), -2);
        assert_eq!(i32::read_le(&out[2..6]), 0x0102_0304);
        assert_eq!(u64::read_le(&u64::MAX.to_le_bytes()), u64::MAX);
    }
}
