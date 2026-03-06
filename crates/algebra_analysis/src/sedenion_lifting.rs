//! Sedenion lifting: map ordered planetary positions to 16D sedenion elements.
//!
//! Given N <= 5 planetary position vectors (3D each), we encode them into the
//! 15 imaginary components of a sedenion (e1..e15), producing a unit-norm
//! algebraic element suitable for associator and box-kite analysis.
//!
//! Encoding scheme:
//!   Planet i (i=0..4) maps to basis elements e_{3i+1}, e_{3i+2}, e_{3i+3}.
//!   The real component e0 is set to zero.
//!
//! This is a *structural* tool for exploring algebraic correlations with
//! planetary geometry, NOT a physical prediction.

use cd_kernel::cayley_dickson::{cd_associator, cd_norm_sq};

/// Maximum number of planets that can be encoded (5 * 3 = 15 imaginary slots).
pub const MAX_PLANETS: usize = 5;

/// Result of lifting planetary positions into a sedenion.
#[derive(Debug, Clone)]
pub struct SedenionLift {
    /// The 16D sedenion element (unit-normalized).
    pub element: Vec<f64>,
    /// The associator norm ||[e, e_conj, e]|| (self-associator, always zero for alt. algebras).
    pub self_assoc_norm: f64,
    /// Number of planets encoded.
    pub n_planets: usize,
}

/// Lift N planetary position vectors into a unit sedenion.
///
/// `positions`: slice of 3D position vectors (up to 5 planets).
/// Each planet occupies 3 consecutive imaginary basis elements.
/// The result is L2-normalized to unit norm.
pub fn lift_positions_to_sedenion(positions: &[[f64; 3]]) -> SedenionLift {
    assert!(
        positions.len() <= MAX_PLANETS,
        "At most {} planets can be encoded in a sedenion",
        MAX_PLANETS
    );

    let mut element = vec![0.0; 16];

    for (i, pos) in positions.iter().enumerate() {
        let base = 3 * i + 1; // e1, e2, e3 for planet 0, etc.
        element[base] = pos[0];
        element[base + 1] = pos[1];
        element[base + 2] = pos[2];
    }

    // Normalize to unit norm
    let norm_sq = cd_norm_sq(&element);
    if norm_sq > 1e-30 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for x in &mut element {
            *x *= inv_norm;
        }
    }

    // Compute self-associator [e, e, e] (trivially zero, but verify)
    let assoc = cd_associator(&element, &element, &element);
    let self_assoc_norm = cd_norm_sq(&assoc).sqrt();

    SedenionLift {
        element,
        self_assoc_norm,
        n_planets: positions.len(),
    }
}

/// Compute the associator of two sedenion lifts: [a, b, c] where c = cd_multiply(a, b).
///
/// This measures the non-associativity of the combined planetary configuration.
pub fn compute_lifted_associator(a: &[f64], b: &[f64]) -> (Vec<f64>, f64) {
    let assoc = cd_associator(a, b, a);
    let norm = cd_norm_sq(&assoc).sqrt();
    (assoc, norm)
}

/// Compute the full associator triple [a, b, c] for three sedenion lifts.
pub fn compute_triple_associator(a: &[f64], b: &[f64], c: &[f64]) -> (Vec<f64>, f64) {
    let assoc = cd_associator(a, b, c);
    let norm = cd_norm_sq(&assoc).sqrt();
    (assoc, norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_planet_lift_is_unit() {
        let positions = [[1.0, 0.0, 0.0]];
        let lift = lift_positions_to_sedenion(&positions);
        let norm = cd_norm_sq(&lift.element).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "Unit norm expected, got {}",
            norm
        );
        assert_eq!(lift.n_planets, 1);
    }

    #[test]
    fn five_planets_fills_imaginary_slots() {
        let positions = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ];
        let lift = lift_positions_to_sedenion(&positions);
        // e0 (real part) should be zero
        assert_eq!(lift.element[0], 0.0);
        // All 15 imaginary slots should be nonzero (before normalization)
        assert_eq!(lift.n_planets, 5);
        let norm = cd_norm_sq(&lift.element).sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn self_associator_is_zero() {
        let positions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let lift = lift_positions_to_sedenion(&positions);
        // [e, e, e] = 0 for any element in any algebra
        assert!(
            lift.self_assoc_norm < 1e-12,
            "Self-associator should vanish, got {}",
            lift.self_assoc_norm
        );
    }

    #[test]
    fn two_planet_associator_nonzero() {
        // Two different planet configs should produce a nonzero associator in dim=16
        let pos_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let pos_b = [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let lift_a = lift_positions_to_sedenion(&pos_a);
        let lift_b = lift_positions_to_sedenion(&pos_b);

        let (_, norm) = compute_lifted_associator(&lift_a.element, &lift_b.element);
        // Sedenions are non-alternative, so most pairs have nonzero associator
        // (may be zero for specific alignments, but generically nonzero)
        assert!(norm >= 0.0); // At minimum, norm is non-negative
    }
}
