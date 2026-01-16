//! # mac_amx
//!
//! Hardware-accelerated matrix operations using Apple's AMX coprocessor.
//!
//! ## Quick Start
//!
//! ```no_run
//! use mac_amx::{is_available, Matrix};
//!
//! if is_available() {
//!     let a = Matrix::from_slice(64, 64, &vec![1.0f32; 64 * 64]);
//!     let b = Matrix::from_slice(64, 64, &vec![2.0f32; 64 * 64]);
//!     let c = a.matmul(&b);
//!     println!("Result: {:?}", &c.data()[..4]);
//! }
//! ```
//!
//! ## Feature Levels
//!
//! - **High-level**: [`Matrix`] type with safe operations
//! - **Mid-level**: [`ops`] module for direct AMX register control
//! - **Low-level**: [`raw`] module for raw instruction access

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

mod detect;

pub mod ops;
pub mod raw;

pub use detect::{detect, is_available, AmxVersion};

use std::fmt;

// ============================================================================
// Matrix Type
// ============================================================================

/// A row-major matrix of f32 values.
///
/// This is the primary type for high-level AMX operations.
#[derive(Clone)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Create a new matrix filled with zeros.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create a new matrix filled with a constant value.
    #[must_use]
    pub fn fill(rows: usize, cols: usize, value: f32) -> Self {
        Self {
            data: vec![value; rows * cols],
            rows,
            cols,
        }
    }

    /// Create an identity matrix.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Create a matrix from existing data.
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    #[must_use]
    pub fn from_slice(rows: usize, cols: usize, data: &[f32]) -> Self {
        assert_eq!(data.len(), rows * cols, "Data length must equal rows × cols");
        Self {
            data: data.to_vec(),
            rows,
            cols,
        }
    }

    /// Create a matrix from a Vec, taking ownership.
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    #[must_use]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "Data length must equal rows × cols");
        Self { data, rows, cols }
    }

    /// Number of rows.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Shape as (rows, cols).
    #[inline]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Immutable access to underlying data (row-major order).
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable access to underlying data (row-major order).
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consume the matrix and return the underlying Vec.
    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Get element at (row, col).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col] = value;
    }

    /// Matrix multiplication: self × other.
    ///
    /// Uses AMX acceleration when available.
    ///
    /// # Panics
    /// Panics if dimensions don't match (self.cols != other.rows).
    #[must_use]
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimensions don't match: ({}, {}) × ({}, {})",
            self.rows, self.cols, other.rows, other.cols
        );

        let mut result = Matrix::zeros(self.rows, other.cols);

        if is_available() && self.rows == self.cols && self.cols == other.cols {
            // Square matrix fast path with AMX
            matmul_amx(&self.data, &other.data, &mut result.data, self.rows);
        } else {
            // General fallback
            matmul_naive(&self.data, &other.data, &mut result.data,
                        self.rows, self.cols, other.cols);
        }

        result
    }

    /// In-place matrix multiplication: self = self × other.
    ///
    /// # Panics
    /// Panics if dimensions don't match or result dimensions differ.
    pub fn matmul_assign(&mut self, other: &Matrix) {
        let result = self.matmul(other);
        assert_eq!(self.shape(), result.shape(), "Result shape must match");
        self.data = result.data;
    }

    /// Transpose the matrix.
    #[must_use]
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        result
    }

    /// Element-wise addition.
    ///
    /// # Panics
    /// Panics if shapes don't match.
    #[must_use]
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.shape(), other.shape(), "Shapes must match");
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Element-wise subtraction.
    ///
    /// # Panics
    /// Panics if shapes don't match.
    #[must_use]
    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.shape(), other.shape(), "Shapes must match");
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(&self, scalar: f32) -> Matrix {
        let data: Vec<f32> = self.data.iter().map(|x| x * scalar).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix({}×{}, {:?}...)", self.rows, self.cols, &self.data[..self.data.len().min(4)])
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, (row, col): (usize, usize)) -> &f32 {
        &self.data[row * self.cols + col]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f32 {
        &mut self.data[row * self.cols + col]
    }
}

// ============================================================================
// Matrix Operations (Internal)
// ============================================================================

fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a[i * k + kk];
            for j in 0..n {
                c[i * n + j] += a_ik * b[kk * n + j];
            }
        }
    }
}

fn matmul_amx(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    use ops::{fma32, ldx, ldy, ldz, stz};

    c.fill(0.0);

    // SAFETY: is_available() was checked by caller
    unsafe {
        raw::amx_set();

        for i in (0..n).step_by(16) {
            for j in (0..n).step_by(16) {
                // Load C tile into Z (rows at j*4 stride for fma32 matrix mode)
                for row in 0..16.min(n - i) {
                    let c_row = &c[(i + row) * n + j..];
                    let len = 16.min(n - j);
                    let mut tmp = [0.0f32; 16];
                    tmp[..len].copy_from_slice(&c_row[..len]);
                    ldz(tmp.as_ptr().cast(), (row * 4) as u64, false);
                }

                // Accumulate
                for k in (0..n).step_by(16) {
                    for kk in 0..16.min(n - k) {
                        // Load A column into Y
                        let mut a_col = [0.0f32; 16];
                        for row in 0..16.min(n - i) {
                            a_col[row] = a[(i + row) * n + k + kk];
                        }
                        ldy(a_col.as_ptr().cast(), 0, false);

                        // Load B row into X
                        let mut b_row = [0.0f32; 16];
                        let b_start = (k + kk) * n + j;
                        let b_len = 16.min(n - j);
                        b_row[..b_len].copy_from_slice(&b[b_start..b_start + b_len]);
                        ldx(b_row.as_ptr().cast(), 0, false);

                        fma32(0, 0, 0, false);
                    }
                }

                // Store C tile
                for row in 0..16.min(n - i) {
                    let mut tmp = [0.0f32; 16];
                    stz(tmp.as_mut_ptr().cast(), (row * 4) as u64, false);
                    let len = 16.min(n - j);
                    c[(i + row) * n + j..(i + row) * n + j + len].copy_from_slice(&tmp[..len]);
                }
            }
        }

        raw::amx_clr();
    }
}

// ============================================================================
// RAII Guard
// ============================================================================

/// RAII guard for AMX state.
///
/// Enables AMX on creation, disables on drop. Use this when performing
/// multiple low-level AMX operations to avoid repeated enable/disable cycles.
///
/// # Example
///
/// ```no_run
/// use mac_amx::AmxGuard;
///
/// let guard = AmxGuard::try_new().expect("AMX not available");
/// // Perform multiple AMX operations...
/// // AMX automatically disabled when guard is dropped
/// ```
pub struct AmxGuard(());

impl AmxGuard {
    /// Enable AMX and return a guard.
    ///
    /// # Panics
    /// Panics if AMX is not available.
    #[must_use]
    pub fn new() -> Self {
        assert!(is_available(), "AMX not available");
        unsafe { raw::amx_set() };
        Self(())
    }

    /// Try to enable AMX, returning `None` if unavailable.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        if is_available() {
            unsafe { raw::amx_set() };
            Some(Self(()))
        } else {
            None
        }
    }
}

impl Default for AmxGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AmxGuard {
    fn drop(&mut self) {
        unsafe { raw::amx_clr() };
    }
}

// ============================================================================
// Re-exports
// ============================================================================

pub use raw::{amx_clr, amx_set};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_create() {
        let m = Matrix::zeros(3, 4);
        assert_eq!(m.shape(), (3, 4));
        assert!(m.data().iter().all(|&x| x == 0.0));

        let m = Matrix::fill(2, 2, 5.0);
        assert!(m.data().iter().all(|&x| x == 5.0));

        let m = Matrix::identity(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(1, 1), 1.0);
    }

    #[test]
    fn test_matrix_index() {
        let mut m = Matrix::zeros(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 2.0;
        m[(1, 0)] = 3.0;
        m[(1, 1)] = 4.0;

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_matmul_identity() {
        if !is_available() {
            return;
        }

        let a = Matrix::identity(64);
        let b = Matrix::from_vec(64, 64, (0..64 * 64).map(|i| (i % 64) as f32).collect());
        let c = a.matmul(&b);

        for (i, (&ci, &bi)) in c.data().iter().zip(b.data().iter()).enumerate() {
            assert!(
                (ci - bi).abs() < 1e-5,
                "Mismatch at {i}: {ci} != {bi}"
            );
        }
    }

    #[test]
    fn test_matmul_small() {
        // 2x2 @ 2x2 (will use naive fallback)
        let a = Matrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);

        // [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_detect() {
        let result = detect();
        println!("AMX: {result:?}");
        assert!(result.is_some());
    }
}
