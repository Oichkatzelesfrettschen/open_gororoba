//! Core types for composition algebra theory (Steps 1-3).
//!
//! These types make hurwitz_1898 usable as a LIBRARY that downstream
//! crates (dickson_1919, schafer_1945, wilmot_2025) can depend on.
//!
//! Mirrors: HurwitzTheorem.v Parts I-VI.

use std::fmt;

// Re-export cd_kernel fundamentals (Step 5)
pub use cd_kernel::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};

/// Classification of a dimension by Hurwitz composition theory.
///
/// Mirrors: HurwitzTheorem.v combined summary.
#[derive(Debug, Clone, PartialEq)]
pub struct HurwitzClassification {
    /// The dimension being classified.
    pub dim: usize,
    /// Whether square composition (||xy|| = ||x|| * ||y||) holds.
    pub is_composition: bool,
    /// The Hurwitz-Radon number rho(n).
    pub hurwitz_radon: usize,
    /// Deficit: dim - rho(dim). Zero iff composition holds.
    pub deficit: usize,
    /// Why this dimension is/isn't a composition algebra.
    pub reason: &'static str,
}

impl fmt::Display for HurwitzClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dim={}: {} (rho={}, deficit={}) -- {}",
            self.dim,
            if self.is_composition {
                "COMPOSITION"
            } else {
                "not composition"
            },
            self.hurwitz_radon,
            self.deficit,
            self.reason
        )
    }
}

/// A system of Clifford generators B_1, ..., B_{n-1} at dimension n.
///
/// These are n x n real skew-symmetric matrices satisfying:
///   B_i^2 = -I
///   B_i * B_k = -B_k * B_i  (i != k)
///
/// Mirrors: HurwitzTheorem.v Part I.5, CliffordSystem Module Type.
#[derive(Debug, Clone)]
pub struct CliffordSystem {
    /// Matrix dimension n.
    pub dim: usize,
    /// Number of generators (= n - 1).
    pub num_generators: usize,
    /// The generator matrices, each stored as a flat n*n row-major array.
    pub generators: Vec<Vec<f64>>,
}

/// Result of verifying Clifford relations.
#[derive(Debug, Clone, PartialEq)]
pub struct CliffordVerification {
    /// B_i^2 = -I for all i.
    pub squares_ok: bool,
    /// B_i * B_k = -B_k * B_i for all i != k.
    pub anticommutation_ok: bool,
    /// B_i is skew-symmetric for all i.
    pub skew_symmetric_ok: bool,
    /// All relations satisfied.
    pub all_ok: bool,
}

impl CliffordSystem {
    /// Verify all Clifford relations.
    ///
    /// Mirrors: HurwitzTheorem.v CliffordSystem Module Type axioms.
    pub fn verify(&self, atol: f64) -> CliffordVerification {
        let (sq, ac, sk) =
            crate::clifford::check_clifford_relations(self.dim, &self.generators, atol);
        CliffordVerification {
            squares_ok: sq,
            anticommutation_ok: ac,
            skew_symmetric_ok: sk,
            all_ok: sq && ac && sk,
        }
    }
}

impl fmt::Display for CliffordVerification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Clifford: squares={}, anticomm={}, skew={}{}",
            if self.squares_ok { "OK" } else { "FAIL" },
            if self.anticommutation_ok {
                "OK"
            } else {
                "FAIL"
            },
            if self.skew_symmetric_ok { "OK" } else { "FAIL" },
            if self.all_ok { " [ALL PASS]" } else { " [FAIL]" }
        )
    }
}

/// Classify a dimension by Hurwitz composition theory.
///
/// This is the main entry point for downstream crates.
///
/// Mirrors: HurwitzTheorem.v combined Parts I-IV.
pub fn classify(dim: usize) -> HurwitzClassification {
    let rho = crate::hurwitz_radon::hurwitz_radon(dim);
    let is_comp = rho >= dim;
    let deficit = if is_comp { 0 } else { dim - rho };
    let reason = match dim {
        0 => "zero dimension",
        1 => "trivial (R)",
        2 => "complex norm multiplicativity (Brahmagupta-Fibonacci)",
        4 => "quaternion norm multiplicativity (Euler four-square)",
        8 => "octonion norm multiplicativity (Degen-Graves eight-square)",
        d if !d.is_multiple_of(2) => "odd: det(B_i)^2 = (-1)^n < 0 impossible",
        6 => "refined bound: 2^4=16 > n*(n-1)/2=15 skew-sym matrices",
        _ if !crate::dimension_bound::basic_bound_holds(dim) => "basic bound: 2^{n-2} > n^2",
        _ => "eliminated by combined Hurwitz criteria",
    };
    HurwitzClassification {
        dim,
        is_composition: is_comp,
        hurwitz_radon: rho,
        deficit,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_composition_dims() {
        for d in [1, 2, 4, 8] {
            let c = classify(d);
            assert!(c.is_composition, "dim {d} should be composition");
            assert_eq!(c.deficit, 0);
        }
    }

    #[test]
    fn test_classify_non_composition() {
        for d in [3, 5, 6, 7, 9, 10, 16, 32] {
            let c = classify(d);
            assert!(!c.is_composition, "dim {d} should NOT be composition");
            assert!(c.deficit > 0);
        }
    }

    #[test]
    fn test_classify_display() {
        let c = classify(8);
        let s = format!("{c}");
        assert!(s.contains("COMPOSITION"));
        assert!(s.contains("rho=8"));
    }

    #[test]
    fn test_clifford_quaternion() {
        let gens = crate::clifford::quaternion_clifford_generators();
        let sys = CliffordSystem {
            dim: 4,
            num_generators: 3,
            generators: gens,
        };
        let v = sys.verify(1e-10);
        assert!(v.all_ok);
    }
}
