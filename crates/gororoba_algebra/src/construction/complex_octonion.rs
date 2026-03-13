//! Complexified Octonion algebra ($\mathbb{O}_\mathbb{C}$ or Bi-Octonions).
//!
//! Elements: $A + iB$, where $A, B \in \mathbb{O}$ and $i^2 = -1$.
//! The complex imaginary unit $i$ commutes with all octonions.
//!
//! Product:
//! $(A + iB)(C + iD) = (AC - BD) + i(AD + BC)$
//!
//! This algebra is 16-dimensional over $\mathbb{R}$, but differs from the Sedenions
//! because the Sedenion $e_8$ anti-commutes with $e_1 \dots e_7$, whereas the
//! complex $i$ commutes with them.

use crate::physics::octonion_field::{oct_conjugate, oct_multiply};
use num_complex::Complex;

/// A Complexified Octonion $X = A + iB$, where $A, B \in \mathbb{O}$.
#[derive(Debug, Clone, Copy)]
pub struct ComplexOctonion {
    pub real: [f64; 8],
    pub imag: [f64; 8],
}

impl ComplexOctonion {
    pub fn zero() -> Self {
        Self {
            real: [0.0; 8],
            imag: [0.0; 8],
        }
    }

    pub fn new(real: [f64; 8], imag: [f64; 8]) -> Self {
        Self { real, imag }
    }

    /// Octonion conjugate: $(A + iB)^* = A^* + iB^*$
    pub fn conjugate(&self) -> Self {
        Self {
            real: oct_conjugate(&self.real),
            imag: oct_conjugate(&self.imag),
        }
    }

    /// Complex conjugate: $\overline{A + iB} = A - iB$
    pub fn complex_conjugate(&self) -> Self {
        let mut neg_imag = self.imag;
        for x in &mut neg_imag {
            *x = -*x;
        }
        Self {
            real: self.real,
            imag: neg_imag,
        }
    }

    /// Hermitian conjugate (both): $(A + iB)^\dagger = A^* - iB^*$
    pub fn hermitian_conjugate(&self) -> Self {
        self.conjugate().complex_conjugate()
    }

    /// Add two complex octonions.
    pub fn add(&self, other: &Self) -> Self {
        let mut r = [0.0; 8];
        let mut i = [0.0; 8];
        for k in 0..8 {
            r[k] = self.real[k] + other.real[k];
            i[k] = self.imag[k] + other.imag[k];
        }
        Self { real: r, imag: i }
    }

    /// Subtract two complex octonions.
    pub fn sub(&self, other: &Self) -> Self {
        let mut r = [0.0; 8];
        let mut i = [0.0; 8];
        for k in 0..8 {
            r[k] = self.real[k] - other.real[k];
            i[k] = self.imag[k] - other.imag[k];
        }
        Self { real: r, imag: i }
    }

    /// Multiply two complex octonions: $(A + iB)(C + iD) = (AC - BD) + i(AD + BC)$
    pub fn multiply(&self, other: &Self) -> Self {
        let ac = oct_multiply(&self.real, &other.real);
        let bd = oct_multiply(&self.imag, &other.imag);
        let ad = oct_multiply(&self.real, &other.imag);
        let bc = oct_multiply(&self.imag, &other.real);

        let mut r = [0.0; 8];
        let mut i = [0.0; 8];
        for k in 0..8 {
            r[k] = ac[k] - bd[k];
            i[k] = ad[k] + bc[k];
        }
        Self { real: r, imag: i }
    }

    /// Complex-valued squared norm.
    /// In $\mathbb{O}_\mathbb{C}$, the quadratic form is $\sum (A_k + iB_k)^2$.
    pub fn norm_sq_complex(&self) -> Complex<f64> {
        let mut re = 0.0;
        let mut im = 0.0;
        for k in 0..8 {
            let z = Complex::new(self.real[k], self.imag[k]);
            let z2 = z * z;
            re += z2.re;
            im += z2.im;
        }
        Complex::new(re, im)
    }

    /// Scalar multiplication by a complex number.
    pub fn mul_scalar(&self, c: Complex<f64>) -> Self {
        let mut r = [0.0; 8];
        let mut i = [0.0; 8];
        for k in 0..8 {
            let z = Complex::new(self.real[k], self.imag[k]) * c;
            r[k] = z.re;
            i[k] = z.im;
        }
        Self { real: r, imag: i }
    }
}
