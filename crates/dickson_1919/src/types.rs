//! Core types for the Cayley-Dickson construction (Steps 6-10).
//!
//! Provides the CayleyDicksonLevel enum, inverse computation,
//! and enriched property reports that compose with hurwitz_1898.
//!
//! Mirrors: DicksonCDProcess.v Parts I-V.

use std::fmt;

use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};

/// A level in the Cayley-Dickson tower.
///
/// Each level doubles the dimension and loses one algebraic property.
///
/// Mirrors: DicksonCDProcess.v property hierarchy (p.159).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CayleyDicksonLevel {
    Real,            // dim 1
    Complex,         // dim 2
    Quaternion,      // dim 4
    Octonion,        // dim 8
    Sedenion,        // dim 16
    Pathion,         // dim 32
    Chingon,         // dim 64
    Routon,          // dim 128
    Voudon,          // dim 256
    Higher(usize),   // dim = 2^n for n >= 9
}

impl CayleyDicksonLevel {
    /// Dimension of this CD level.
    pub const fn dim(&self) -> usize {
        match self {
            Self::Real => 1,
            Self::Complex => 2,
            Self::Quaternion => 4,
            Self::Octonion => 8,
            Self::Sedenion => 16,
            Self::Pathion => 32,
            Self::Chingon => 64,
            Self::Routon => 128,
            Self::Voudon => 256,
            Self::Higher(d) => *d,
        }
    }

    /// Create from dimension (must be a power of 2).
    pub fn from_dim(dim: usize) -> Option<Self> {
        if dim == 0 || (dim & (dim - 1)) != 0 {
            return None; // not a power of 2
        }
        Some(match dim {
            1 => Self::Real,
            2 => Self::Complex,
            4 => Self::Quaternion,
            8 => Self::Octonion,
            16 => Self::Sedenion,
            32 => Self::Pathion,
            64 => Self::Chingon,
            128 => Self::Routon,
            256 => Self::Voudon,
            d => Self::Higher(d),
        })
    }

    /// Name of this CD level.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Real => "Real",
            Self::Complex => "Complex",
            Self::Quaternion => "Quaternion",
            Self::Octonion => "Octonion",
            Self::Sedenion => "Sedenion",
            Self::Pathion => "Pathion",
            Self::Chingon => "Chingon",
            Self::Routon => "Routon",
            Self::Voudon => "Voudon",
            Self::Higher(_) => "Higher",
        }
    }

    /// Is this level commutative? (Only R and C.)
    pub const fn is_commutative(&self) -> bool {
        matches!(self, Self::Real | Self::Complex)
    }

    /// Is this level associative? (Only R, C, H.)
    pub const fn is_associative(&self) -> bool {
        matches!(self, Self::Real | Self::Complex | Self::Quaternion)
    }

    /// Is this level alternative? (Only R, C, H, O.)
    pub const fn is_alternative(&self) -> bool {
        matches!(
            self,
            Self::Real | Self::Complex | Self::Quaternion | Self::Octonion
        )
    }

    /// Is this level a division algebra? (Only R, C, H, O.)
    pub const fn is_division(&self) -> bool {
        self.is_alternative() // same set: dims 1, 2, 4, 8
    }

    /// Does this level have zero divisors? (Dim >= 16.)
    pub const fn has_zero_divisors(&self) -> bool {
        !self.is_division()
    }

    /// Is this level a composition algebra? (Same as division for CD tower.)
    pub fn is_composition(&self) -> bool {
        hurwitz_1898::hurwitz_radon::is_composition_algebra_dim(self.dim())
    }

    /// The Hurwitz-Radon number for this level.
    pub fn hurwitz_radon(&self) -> usize {
        hurwitz_1898::hurwitz_radon::hurwitz_radon(self.dim())
    }

    /// Full Hurwitz classification for this level.
    pub fn hurwitz_classification(&self) -> hurwitz_1898::HurwitzClassification {
        hurwitz_1898::classify(self.dim())
    }
}

impl fmt::Display for CayleyDicksonLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (dim {})", self.name(), self.dim())
    }
}

/// Compute the inverse of a CD element: x^{-1} = conj(x) / norm(x).
///
/// Returns None if the element has zero norm (is a zero divisor or zero).
/// Only valid for division algebra dimensions (1, 2, 4, 8), but computes
/// for any dimension.
///
/// Mirrors: DicksonCDProcess.v dickson_quat_right_inverse,
///          Dickson (1919) sect.3 (quaternion inverse),
///          Dickson (1919) sect.4 (octonion inverse).
pub fn cd_inverse(x: &[f64]) -> Option<Vec<f64>> {
    let norm = cd_norm_sq(x);
    if norm.abs() < 1e-15 {
        return None;
    }
    let conj = cd_conjugate(x);
    let inv_norm = 1.0 / norm;
    Some(conj.iter().map(|c| c * inv_norm).collect())
}

/// Verify that x * x^{-1} = e_0 (the identity element).
///
/// Returns the maximum component deviation from e_0.
pub fn verify_inverse(x: &[f64]) -> Option<f64> {
    let inv = cd_inverse(x)?;
    let product = cd_multiply(x, &inv);
    let mut max_err = 0.0_f64;
    for (i, &v) in product.iter().enumerate() {
        let expected = if i == 0 { 1.0 } else { 0.0 };
        max_err = max_err.max((v - expected).abs());
    }
    // Also check left inverse
    let left_product = cd_multiply(&inv, x);
    for (i, &v) in left_product.iter().enumerate() {
        let expected = if i == 0 { 1.0 } else { 0.0 };
        max_err = max_err.max((v - expected).abs());
    }
    Some(max_err)
}

/// Enhanced property report combining Dickson and Hurwitz classifications.
///
/// Mirrors: DicksonCDProcess.v property hierarchy + HurwitzTheorem.v.
#[derive(Debug, Clone)]
pub struct PropertyReport {
    pub level: CayleyDicksonLevel,
    pub commutative: bool,
    pub associative: bool,
    pub alternative: bool,
    pub division: bool,
    pub has_zero_divisors: bool,
    pub composition: bool,
    pub hurwitz_radon: usize,
}

impl PropertyReport {
    /// Generate a full property report for a CD level.
    pub fn for_level(level: CayleyDicksonLevel) -> Self {
        Self {
            level,
            commutative: level.is_commutative(),
            associative: level.is_associative(),
            alternative: level.is_alternative(),
            division: level.is_division(),
            has_zero_divisors: level.has_zero_divisors(),
            composition: level.is_composition(),
            hurwitz_radon: level.hurwitz_radon(),
        }
    }

    /// Generate for a dimension.
    pub fn for_dim(dim: usize) -> Option<Self> {
        CayleyDicksonLevel::from_dim(dim).map(Self::for_level)
    }
}

impl fmt::Display for PropertyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: comm={} assoc={} alt={} div={} zd={} comp={} rho={}",
            self.level,
            if self.commutative { "Y" } else { "N" },
            if self.associative { "Y" } else { "N" },
            if self.alternative { "Y" } else { "N" },
            if self.division { "Y" } else { "N" },
            if self.has_zero_divisors { "Y" } else { "N" },
            if self.composition { "Y" } else { "N" },
            self.hurwitz_radon,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_from_dim() {
        assert_eq!(CayleyDicksonLevel::from_dim(1), Some(CayleyDicksonLevel::Real));
        assert_eq!(CayleyDicksonLevel::from_dim(8), Some(CayleyDicksonLevel::Octonion));
        assert_eq!(CayleyDicksonLevel::from_dim(16), Some(CayleyDicksonLevel::Sedenion));
        assert_eq!(CayleyDicksonLevel::from_dim(3), None); // not power of 2
        assert_eq!(CayleyDicksonLevel::from_dim(0), None);
    }

    #[test]
    fn test_property_hierarchy() {
        let r = CayleyDicksonLevel::Real;
        assert!(r.is_commutative() && r.is_associative() && r.is_division());

        let h = CayleyDicksonLevel::Quaternion;
        assert!(!h.is_commutative() && h.is_associative() && h.is_division());

        let o = CayleyDicksonLevel::Octonion;
        assert!(!o.is_commutative() && !o.is_associative() && o.is_alternative() && o.is_division());

        let s = CayleyDicksonLevel::Sedenion;
        assert!(!s.is_alternative() && !s.is_division() && s.has_zero_divisors());
    }

    #[test]
    fn test_inverse_quaternion() {
        let q = vec![1.0, 2.0, 3.0, 4.0]; // 1 + 2i + 3j + 4k
        let err = verify_inverse(&q).unwrap();
        assert!(err < 1e-10, "quaternion inverse should work: err={err}");
    }

    #[test]
    fn test_inverse_octonion() {
        let o = vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]; // 1 + e2 + e7
        let err = verify_inverse(&o).unwrap();
        assert!(err < 1e-10, "octonion inverse should work: err={err}");
    }

    #[test]
    fn test_inverse_sedenion_zd() {
        // ZD element: e3 + e10
        let mut x = vec![0.0; 16];
        x[3] = 1.0;
        x[10] = 1.0;
        // This has nonzero norm so inverse exists, but it's not a division algebra
        let inv = cd_inverse(&x);
        assert!(inv.is_some(), "nonzero elements have inverses by formula");
        // However the RIGHT inverse won't satisfy x * x^{-1} = 1 perfectly
        // in non-alternative algebras
    }

    #[test]
    fn test_property_report_display() {
        let r = PropertyReport::for_level(CayleyDicksonLevel::Octonion);
        let s = format!("{r}");
        assert!(s.contains("Octonion"));
        assert!(s.contains("alt=Y"));
        assert!(s.contains("zd=N"));
    }

    #[test]
    fn test_property_report_full_tower() {
        for dim in [1, 2, 4, 8, 16, 32, 64] {
            let r = PropertyReport::for_dim(dim).unwrap();
            if dim <= 8 {
                assert!(r.division, "dim {dim} should be division");
                assert!(!r.has_zero_divisors);
            } else {
                assert!(!r.division, "dim {dim} should NOT be division");
                assert!(r.has_zero_divisors);
            }
        }
    }
}
