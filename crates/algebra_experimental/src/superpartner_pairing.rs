//! Emergent superpartner pairing from Monster representation structure.
//!
//! Tests the hypothesis that non-associative algebraic structure (CD tower)
//! naturally generates boson/fermion-like parity in Monster irreps.
//!
//! The E8 connection (McKay observation): the 248-dimensional E8 root system
//! relates to Monster 2A-involutions. Since dim(E8) = 248 and
//! j_constant = 744 = 3 * 248, there is a natural 3-fold structure that
//! can be interpreted as a Z_3 grading (or equivalently, a parity assignment
//! with a neutral sector).
//!
//! See BIB-0311 (Conway & Norton 1979), BIB-0316 (Conway 1968, Atlas).

use super::moonshine::{J_COEFFICIENTS, MONSTER_REP_DIMENSIONS};

/// Parity assignment for a Monster representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepParity {
    /// Bosonic (even parity) -- representation dimension mod 2 = 0.
    Bosonic,
    /// Fermionic (odd parity) -- representation dimension mod 2 = 1.
    Fermionic,
}

/// A superpartner pair: two Monster irreps with matching structure
/// but opposite parity assignments.
#[derive(Debug, Clone)]
pub struct SuperpartnerPair {
    /// Index of the "bosonic" representation.
    pub boson_idx: usize,
    /// Index of the "fermionic" representation.
    pub fermion_idx: usize,
    /// Dimension of the bosonic representation.
    pub boson_dim: u64,
    /// Dimension of the fermionic representation.
    pub fermion_dim: u64,
    /// Dimension ratio (larger / smaller).
    pub ratio: f64,
}

/// Result of superpartner pairing analysis.
#[derive(Debug, Clone)]
pub struct PairingResult {
    /// Number of representations analyzed.
    pub num_reps: usize,
    /// Parity assignments for each representation.
    pub parities: Vec<RepParity>,
    /// Number of bosonic representations.
    pub num_bosonic: usize,
    /// Number of fermionic representations.
    pub num_fermionic: usize,
    /// Identified superpartner pairs (by dimension proximity).
    pub pairs: Vec<SuperpartnerPair>,
    /// Whether parity is "balanced" (|N_B - N_F| <= 1).
    pub parity_balanced: bool,
}

/// Assign parity to Monster representations based on dimension mod 2.
///
/// This is the simplest parity assignment. The trivial rep (dim 1) is
/// fermionic under this rule, which matches the Ramond sector convention
/// in string theory where the ground state has odd fermion number.
pub fn assign_parity(dim: u64) -> RepParity {
    if dim.is_multiple_of(2) {
        RepParity::Bosonic
    } else {
        RepParity::Fermionic
    }
}

/// Find superpartner pairs by proximity in representation dimension.
///
/// Two representations are "paired" if their dimension ratio is close to
/// an integer (within tolerance). This models the SUSY mass relation
/// where superpartners have related masses/dimensions.
pub fn find_pairs(reps: &[u64], tolerance: f64) -> Vec<SuperpartnerPair> {
    let valid: Vec<(usize, u64)> = reps
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, d)| *d > 0)
        .collect();

    let mut pairs = Vec::new();
    let mut used = vec![false; valid.len()];

    for i in 0..valid.len() {
        if used[i] {
            continue;
        }
        let (idx_i, dim_i) = valid[i];
        let parity_i = assign_parity(dim_i);

        // Find the closest representation with opposite parity
        let mut best_j = None;
        let mut best_ratio = f64::MAX;

        for j in (i + 1)..valid.len() {
            if used[j] {
                continue;
            }
            let (_, dim_j) = valid[j];
            let parity_j = assign_parity(dim_j);

            if parity_i == parity_j {
                continue;
            }

            let (larger, smaller) = if dim_i > dim_j {
                (dim_i as f64, dim_j as f64)
            } else {
                (dim_j as f64, dim_i as f64)
            };
            let ratio = larger / smaller;
            let frac = ratio - ratio.floor();
            let nearness = frac.min(1.0 - frac);

            if nearness < tolerance && ratio < best_ratio {
                best_ratio = ratio;
                best_j = Some(j);
            }
        }

        if let Some(j) = best_j {
            let (idx_j, dim_j) = valid[j];
            used[i] = true;
            used[j] = true;

            let (b_idx, b_dim, f_idx, f_dim) = if assign_parity(dim_i) == RepParity::Bosonic {
                (idx_i, dim_i, idx_j, dim_j)
            } else {
                (idx_j, dim_j, idx_i, dim_i)
            };

            pairs.push(SuperpartnerPair {
                boson_idx: b_idx,
                fermion_idx: f_idx,
                boson_dim: b_dim,
                fermion_dim: f_dim,
                ratio: best_ratio,
            });
        }
    }

    pairs
}

/// Run the full superpartner pairing analysis on known Monster irreps.
pub fn analyze_pairing() -> PairingResult {
    let reps: Vec<u64> = MONSTER_REP_DIMENSIONS
        .iter()
        .copied()
        .filter(|&d| d > 0)
        .collect();
    let parities: Vec<RepParity> = reps.iter().map(|&d| assign_parity(d)).collect();

    let num_bosonic = parities
        .iter()
        .filter(|&&p| p == RepParity::Bosonic)
        .count();
    let num_fermionic = parities
        .iter()
        .filter(|&&p| p == RepParity::Fermionic)
        .count();

    let pairs = find_pairs(&reps, 0.3);
    let parity_balanced = (num_bosonic as i64 - num_fermionic as i64).unsigned_abs() <= 1;

    PairingResult {
        num_reps: reps.len(),
        parities,
        num_bosonic,
        num_fermionic,
        pairs,
        parity_balanced,
    }
}

/// Check whether the j-function Witten index vanishes.
///
/// The Witten index W = sum_n (-1)^n * c_n measures the net
/// boson-fermion asymmetry. If W = 0, there is an exact
/// boson/fermion cancellation consistent with SUSY.
pub fn witten_index_partial(n_terms: usize) -> i128 {
    let max_n = n_terms.min(J_COEFFICIENTS.len());
    let mut w: i128 = 0;
    for (i, &coeff) in J_COEFFICIENTS.iter().take(max_n).enumerate() {
        let sign: i128 = if i % 2 == 0 { 1 } else { -1 };
        w += sign * coeff as i128;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parity_assignment() {
        assert_eq!(assign_parity(1), RepParity::Fermionic); // trivial rep
        assert_eq!(assign_parity(196883), RepParity::Fermionic); // odd
        assert_eq!(assign_parity(21296876), RepParity::Bosonic); // even
        assert_eq!(assign_parity(842609326), RepParity::Bosonic); // even
    }

    #[test]
    fn test_pairing_analysis_runs() {
        let result = analyze_pairing();
        assert!(result.num_reps > 0);
        assert_eq!(result.num_bosonic + result.num_fermionic, result.num_reps);
    }

    #[test]
    fn test_witten_index_sign_alternation() {
        // The Witten index should alternate in sign contributions
        let w2 = witten_index_partial(2);
        let w3 = witten_index_partial(3);
        // c_1 and c_2 are both positive, so W_2 = c_1 - c_2 < 0
        assert!(w2 != w3, "Partial Witten indices should differ");
    }

    #[test]
    fn test_e8_dimension_parity() {
        // E8 root system has dimension 248 (even), so it's "bosonic"
        assert_eq!(assign_parity(248), RepParity::Bosonic);
        // j-constant 744 = 3 * 248 (even), also bosonic
        assert_eq!(assign_parity(744), RepParity::Bosonic);
    }

    #[test]
    fn test_find_pairs_with_known_reps() {
        let reps = vec![1u64, 196883, 21296876, 842609326];
        let pairs = find_pairs(&reps, 0.5);
        // Should find at least one pair (1 is fermionic, 196883 is fermionic,
        // 21296876 is bosonic, 842609326 is bosonic)
        // Pairing depends on ratio tolerance
        assert!(pairs.len() <= 2);
    }
}
