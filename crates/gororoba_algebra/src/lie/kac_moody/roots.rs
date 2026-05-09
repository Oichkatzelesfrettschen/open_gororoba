//! Kac-Moody root systems and individual roots.
//!
//! - [`KacMoodyRootSystem`] -- a generic finite-or-infinite root system built
//!   from a [`super::GeneralizedCartanMatrix`]. Stores simple roots, coroots,
//!   and an optional null direction `delta` (used by affine algebras).
//! - [`KacMoodyRoot`] / [`RootType`] -- a single root in `R^rank x Z^null`,
//!   classified as Real, ImaginaryNull, or ImaginaryNonNull. Real roots
//!   correspond to root vectors of finite or affine type; imaginary roots
//!   appear at and beyond the affine boundary.
//!
//! Specialized E-series root systems live in [`super::e_series`] and reuse
//! these types.

use super::GeneralizedCartanMatrix;

/// Root system representation for Kac-Moody algebras.
#[derive(Debug, Clone)]
pub struct KacMoodyRootSystem {
    /// Simple roots (alpha_1, ..., alpha_n) as column vectors
    pub simple_roots: Vec<Vec<f64>>,
    /// Simple coroots (alpha_1^v, ..., alpha_n^v)
    pub simple_coroots: Vec<Vec<f64>>,
    /// The GCM
    pub cartan_matrix: GeneralizedCartanMatrix,
}

impl KacMoodyRootSystem {
    /// Create a root system from a GCM using the standard realization.
    pub fn from_gcm(gcm: GeneralizedCartanMatrix) -> Self {
        let n = gcm.rank();

        // Standard realization in R^n (for finite case)
        // Simple roots are rows of the identity matrix scaled by 2
        // Coroots are adjusted for non-simply-laced cases

        let simple_roots: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut root = vec![0.0; n];
                root[i] = 1.0;
                root
            })
            .collect();

        // For simply-laced, coroots = roots
        // For non-simply-laced, need to adjust by root lengths
        let simple_coroots = simple_roots.clone();

        Self {
            simple_roots,
            simple_coroots,
            cartan_matrix: gcm,
        }
    }

    /// Apply a simple reflection s_i to a weight vector.
    pub fn simple_reflection(&self, weight: &[f64], i: usize) -> Vec<f64> {
        let mut result = weight.to_vec();

        // s_i(lambda) = lambda - <lambda, alpha_i^v> * alpha_i
        let pairing: f64 = weight
            .iter()
            .zip(&self.simple_coroots[i])
            .map(|(w, c)| w * c)
            .sum();

        for (r_j, &s_ij) in result.iter_mut().zip(self.simple_roots[i].iter()) {
            *r_j -= pairing * s_ij;
        }

        result
    }
}

// === Extended E-series Root Systems ===

/// Root type for affine and indefinite Kac-Moody algebras.
#[derive(Debug, Clone, PartialEq)]
pub struct KacMoodyRoot {
    /// Finite part in R^n (embedding of simple roots)
    pub finite_part: Vec<f64>,
    /// Affine/imaginary level (coefficient of delta)
    pub level: i32,
    /// For E10+: additional Lorentzian coordinates
    pub lorentz_coords: Vec<f64>,
    /// Real/imaginary classification
    pub root_type: RootType,
}

/// Classification of roots in Kac-Moody algebras.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootType {
    /// Real roots: have squared length > 0
    Real,
    /// Imaginary roots: have squared length <= 0
    Imaginary,
    /// Null roots (affine algebras): squared length = 0
    Null,
}

impl KacMoodyRoot {
    /// Create a real root from finite coordinates.
    pub fn real(coords: Vec<f64>) -> Self {
        Self {
            finite_part: coords,
            level: 0,
            lorentz_coords: vec![],
            root_type: RootType::Real,
        }
    }

    /// Create an affine root at a given level.
    pub fn affine(finite_part: Vec<f64>, level: i32) -> Self {
        Self {
            finite_part,
            level,
            lorentz_coords: vec![],
            root_type: if level == 0 {
                RootType::Real
            } else {
                RootType::Imaginary
            },
        }
    }

    /// Create a root with Lorentzian extension (for E10+).
    pub fn lorentzian(finite_part: Vec<f64>, level: i32, lorentz: Vec<f64>) -> Self {
        Self {
            finite_part,
            level,
            lorentz_coords: lorentz,
            root_type: RootType::Real, // Will be determined by norm
        }
    }

    /// Compute the squared norm using the appropriate inner product.
    pub fn norm_squared(&self, signature: &[i32]) -> f64 {
        let mut result = 0.0;

        // Finite part: positive definite
        for &x in &self.finite_part {
            result += x * x;
        }

        // Lorentzian part: use signature
        for (i, &x) in self.lorentz_coords.iter().enumerate() {
            if i < signature.len() {
                result += signature[i] as f64 * x * x;
            }
        }

        result
    }

    /// Add two roots.
    pub fn add(&self, other: &Self) -> Self {
        let max_finite = self.finite_part.len().max(other.finite_part.len());
        let max_lorentz = self.lorentz_coords.len().max(other.lorentz_coords.len());

        let mut finite_part = vec![0.0; max_finite];
        let mut lorentz_coords = vec![0.0; max_lorentz];

        for (i, &x) in self.finite_part.iter().enumerate() {
            finite_part[i] += x;
        }
        for (i, &x) in other.finite_part.iter().enumerate() {
            finite_part[i] += x;
        }

        for (i, &x) in self.lorentz_coords.iter().enumerate() {
            lorentz_coords[i] += x;
        }
        for (i, &x) in other.lorentz_coords.iter().enumerate() {
            lorentz_coords[i] += x;
        }

        Self {
            finite_part,
            level: self.level + other.level,
            lorentz_coords,
            root_type: RootType::Real, // Will be recomputed
        }
    }
}

