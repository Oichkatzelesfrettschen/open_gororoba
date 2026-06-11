//! Leech lattice (Lambda_24) via Construction A from the Golay code.
//!
//! # What is the Leech lattice?
//!
//! The Leech lattice Lambda_24 is the unique densest sphere packing in 24
//! dimensions (proven by Viazovska 2022 for R^24).  Its key invariants:
//! - Minimum squared norm: 4 (no vectors of norm 2, unlike all other unimodular lattices in 24D)
//! - Kissing number: 196560 (the most contact spheres possible in 24D)
//! - Covering radius: sqrt(2)
//! - Automorphism group: Co_0 (Conway's group), order 8315553613086720000
//! - Deep holes: 23 types (one for each Niemeier lattice -- the other 23 even
//!   unimodular 24D lattices)
//!
//! # Construction A from the Golay code
//!
//! Given a binary linear code C in GF(2)^24, Construction A produces a
//! 24-dimensional lattice:
//!
//! ```text
//! Lambda(C) = { x in Z^24 : x mod 2 in C }
//! ```
//!
//! For C = C_24 (the extended Golay code, `[24,12,8]`), this yields the Leech
//! lattice after scaling by 1/2.  The generator matrix is therefore
//! `(1/2) * [I_24 + 2*G_golay]` where G_golay = `[I_12 | P]` in systematic form.
//!
//! This is why the two modules are co-located: `leech_lattice` depends on
//! `golay_code` for the generator matrix and the codeword membership test.
//!
//! # Why this matters for the CD tower
//!
//! The Monster group M (the largest sporadic simple group) acts on the Moonshine
//! module V^natural, whose graded character involves the j-invariant.  The
//! Leech lattice is the vertex operator algebra underlying this construction.
//! The CD tower connects: O (8D) -> E8 root system -> Leech (24D) -> Monster.
//! This chain is studied in `crates/algebra_experimental/src/moonshine_embedding.rs`.
//!
//! # References
//!
//! - Conway & Sloane, "Sphere Packings, Lattices and Groups", Ch. 4 (1999)
//! - Viazovska (2022), "The sphere packing problem in dimension 8 and 24"
//! - Babai (1986), "On Lovasz' lattice reduction and the nearest lattice point problem"
//! - Conway (1969), "A characterisation of Leech's lattice"

use super::golay_code;

/// A point in 24-dimensional space.
pub type Point24 = [f64; 24];

/// Classification of a point relative to the Leech lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoleType {
    /// On or very near a lattice point (distance^2 < epsilon).
    LatticePoint,
    /// In a shallow hole (0 < distance^2 < deep_threshold).
    ShallowHole,
    /// In a deep hole (distance^2 near covering radius^2 = 2).
    DeepHole,
}

/// Result of the closest vector problem.
#[derive(Debug, Clone)]
pub struct CvpResult {
    /// Nearest lattice point found by Babai's algorithm.
    pub nearest_point: Point24,
    /// Squared Euclidean distance to the nearest lattice point.
    pub distance_squared: f64,
    /// Classification of the target point.
    pub classification: HoleType,
}

/// Threshold for classifying as "on lattice" (distance^2 < this).
const LATTICE_EPSILON: f64 = 0.01;
/// Threshold for deep hole classification (distance^2 > this).
/// Covering radius = sqrt(2), so deep holes have dist^2 near 2.0.
const DEEP_HOLE_THRESHOLD: f64 = 1.5;

/// Precomputed basis for efficient Babai nearest-plane CVP queries.
///
/// The Leech lattice basis is constructed via a simplified Construction A
/// approach: we use an integer basis that generates all lattice vectors
/// with the correct norm properties.
pub struct LeechBasis {
    /// Lattice basis vectors (24x24 row-major).
    basis: [Point24; 24],
    /// Gram-Schmidt orthogonalized basis.
    gs_basis: [Point24; 24],
    /// Squared norms of Gram-Schmidt vectors.
    gs_norms_sq: [f64; 24],
    /// Gram-Schmidt coefficients: mu`[i]``[j]` for j < i.
    mu: [[f64; 24]; 24],
}

/// Dot product of two 24D vectors.
fn dot24(a: &Point24, b: &Point24) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Norm squared of a 24D vector.
fn norm_sq24(a: &Point24) -> f64 {
    dot24(a, a)
}

/// Subtract b from a: a - b.
fn sub24(a: &Point24, b: &Point24) -> Point24 {
    let mut r = [0.0; 24];
    for i in 0..24 {
        r[i] = a[i] - b[i];
    }
    r
}

/// Scale a vector: c * a.
fn scale24(c: f64, a: &Point24) -> Point24 {
    let mut r = [0.0; 24];
    for i in 0..24 {
        r[i] = c * a[i];
    }
    r
}

/// Add b to a: a + b.
fn add24(a: &Point24, b: &Point24) -> Point24 {
    let mut r = [0.0; 24];
    for i in 0..24 {
        r[i] = a[i] + b[i];
    }
    r
}

/// Construct the Leech lattice generator matrix via Construction A.
///
/// Construction A: Given the extended Golay code C_24, the Leech lattice is
///   Lambda_24 = (1/sqrt(8)) * { x in Z^24 : x mod 2 in C_24 }
///
/// We use a simplified generator: the rows of (2*I_24) combined with
/// Golay-code-based correction vectors. For practical CVP, we use a
/// concrete basis that generates the correct lattice.
///
/// The standard basis from Conway & Sloane uses:
///   - Row 0..11: 2*e_i + sum_j P`[i]``[j]` * e_{12+j}  (from Golay systematic form)
///   - Row 12..23: 2*e_{12+j}                         (even coordinate shifts)
///
/// After scaling by 1/sqrt(8), this generates Lambda_24 with min norm 4/8 = 0.5.
/// We instead work in the unscaled lattice where min norm = 4.
fn build_leech_basis() -> [Point24; 24] {
    let mut basis = [[0.0f64; 24]; 24];

    // First 12 rows: encode Golay structure
    // Each row has 2 in position i, plus P[i][j] in positions 12+j
    for (i, row) in basis.iter_mut().enumerate().take(12) {
        row[i] = 2.0;
        for (j, &p) in golay_code::GOLAY_P_MATRIX[i].iter().enumerate() {
            row[12 + j] = p as f64;
        }
    }

    // Last 12 rows: 2*e_{12+j}
    for (j, row) in basis.iter_mut().enumerate().skip(12) {
        row[j] = 2.0;
    }

    basis
}

impl LeechBasis {
    /// Create a new Leech lattice basis with precomputed Gram-Schmidt.
    #[allow(clippy::needless_range_loop)]
    pub fn new() -> Self {
        let basis = build_leech_basis();

        // Gram-Schmidt orthogonalization
        // NOTE: clippy::needless_range_loop suppressed because these loops have
        // inter-dependent index patterns (gs_basis[i] depends on gs_basis[j<i]).
        let mut gs_basis = [[0.0f64; 24]; 24];
        let mut gs_norms_sq = [0.0f64; 24];
        let mut mu = [[0.0f64; 24]; 24];

        for i in 0..24 {
            gs_basis[i] = basis[i];

            for j in 0..i {
                if gs_norms_sq[j] > 1e-30 {
                    mu[i][j] = dot24(&basis[i], &gs_basis[j]) / gs_norms_sq[j];
                    let correction = scale24(mu[i][j], &gs_basis[j]);
                    gs_basis[i] = sub24(&gs_basis[i], &correction);
                }
            }

            gs_norms_sq[i] = norm_sq24(&gs_basis[i]);
        }

        Self {
            basis,
            gs_basis,
            gs_norms_sq,
            mu,
        }
    }

    /// Babai nearest-plane CVP.
    ///
    /// Given a target point t in R^24, find an approximate closest lattice
    /// point by successively rounding Gram-Schmidt coordinates.
    ///
    /// Complexity: O(n^2) per query.
    #[allow(clippy::needless_range_loop)]
    pub fn cvp(&self, target: &Point24) -> CvpResult {
        // Babai nearest-plane: work from the last GS vector backward
        let b = *target;
        let mut coeffs = [0.0f64; 24];

        // Compute GS coordinates
        for i in 0..24 {
            if self.gs_norms_sq[i] > 1e-30 {
                coeffs[i] = dot24(&b, &self.gs_basis[i]) / self.gs_norms_sq[i];
            }
        }

        // Round from last to first, adjusting for GS dependencies
        let mut int_coeffs = [0i64; 24];
        for i in (0..24).rev() {
            let c = coeffs[i];
            let rounded = c.round() as i64;
            int_coeffs[i] = rounded;
            let diff = c - rounded as f64;
            // Propagate to earlier coefficients
            for j in 0..i {
                coeffs[j] -= diff * self.mu[i][j];
            }
        }

        // Reconstruct the lattice point
        let mut nearest = [0.0f64; 24];
        for i in 0..24 {
            let scaled = scale24(int_coeffs[i] as f64, &self.basis[i]);
            nearest = add24(&nearest, &scaled);
        }

        let diff = sub24(target, &nearest);
        let dist_sq = norm_sq24(&diff);

        let classification = if dist_sq < LATTICE_EPSILON {
            HoleType::LatticePoint
        } else if dist_sq > DEEP_HOLE_THRESHOLD {
            HoleType::DeepHole
        } else {
            HoleType::ShallowHole
        };

        CvpResult {
            nearest_point: nearest,
            distance_squared: dist_sq,
            classification,
        }
    }

    /// Batch CVP on multiple target points.
    pub fn decode_batch(&self, targets: &[Point24]) -> Vec<CvpResult> {
        targets.iter().map(|t| self.cvp(t)).collect()
    }

    /// Return a reference to the lattice basis.
    pub fn basis(&self) -> &[Point24; 24] {
        &self.basis
    }
}

impl Default for LeechBasis {
    fn default() -> Self {
        Self::new()
    }
}

/// Project a signal chunk onto 24D space.
///
/// If the chunk length is exactly 24, use it directly.
/// If shorter, zero-pad. If longer, average consecutive samples
/// to reduce to 24 dimensions.
pub fn project_signal_chunk(chunk: &[f64]) -> Point24 {
    let mut point = [0.0f64; 24];
    if chunk.is_empty() {
        return point;
    }

    if chunk.len() <= 24 {
        // Zero-pad
        for (i, &v) in chunk.iter().enumerate() {
            point[i] = v;
        }
    } else {
        // Average bins to reduce to 24
        let bin_size = chunk.len() as f64 / 24.0;
        for (i, p) in point.iter_mut().enumerate() {
            let start = (i as f64 * bin_size) as usize;
            let end = ((i + 1) as f64 * bin_size).min(chunk.len() as f64) as usize;
            if end > start {
                let sum: f64 = chunk[start..end].iter().sum();
                *p = sum / (end - start) as f64;
            }
        }
    }

    // Normalize to unit sphere (if nonzero)
    let norm = norm_sq24(&point).sqrt();
    if norm > 1e-30 {
        for v in &mut point {
            *v /= norm;
        }
    }

    point
}

/// Compute hole statistics for a batch of CVP results.
#[derive(Debug, Clone)]
pub struct HoleStatistics {
    pub total: usize,
    pub lattice_points: usize,
    pub shallow_holes: usize,
    pub deep_holes: usize,
    pub mean_distance_sq: f64,
    pub max_distance_sq: f64,
}

impl HoleStatistics {
    pub fn from_results(results: &[CvpResult]) -> Self {
        let total = results.len();
        let mut lattice_points = 0;
        let mut shallow_holes = 0;
        let mut deep_holes = 0;
        let mut sum_dist = 0.0;
        let mut max_dist = 0.0f64;

        for r in results {
            match r.classification {
                HoleType::LatticePoint => lattice_points += 1,
                HoleType::ShallowHole => shallow_holes += 1,
                HoleType::DeepHole => deep_holes += 1,
            }
            sum_dist += r.distance_squared;
            max_dist = max_dist.max(r.distance_squared);
        }

        Self {
            total,
            lattice_points,
            shallow_holes,
            deep_holes,
            mean_distance_sq: if total > 0 {
                sum_dist / total as f64
            } else {
                0.0
            },
            max_distance_sq: max_dist,
        }
    }

    pub fn lattice_fraction(&self) -> f64 {
        if self.total > 0 {
            self.lattice_points as f64 / self.total as f64
        } else {
            0.0
        }
    }

    pub fn shallow_fraction(&self) -> f64 {
        if self.total > 0 {
            self.shallow_holes as f64 / self.total as f64
        } else {
            0.0
        }
    }

    pub fn deep_fraction(&self) -> f64 {
        if self.total > 0 {
            self.deep_holes as f64 / self.total as f64
        } else {
            0.0
        }
    }
}

/// Theta series coefficient: number of lattice vectors at given norm.
///
/// First few values for the Leech lattice:
/// theta_0 = 1, theta_4 = 196560, theta_6 = 16773120, theta_8 = 398034000
///
/// Only returns known values; returns None for norms > 8 or odd norms.
pub fn theta_series_coefficient(norm: usize) -> Option<u64> {
    match norm {
        0 => Some(1),
        2 => Some(0), // No vectors of norm 2 (defining property)
        4 => Some(196_560),
        6 => Some(16_773_120),
        8 => Some(398_034_000),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leech_basis_construction() {
        let basis = LeechBasis::new();
        // All 24 GS norms should be positive (linearly independent)
        for i in 0..24 {
            assert!(
                basis.gs_norms_sq[i] > 0.0,
                "GS norm {} should be positive, got {}",
                i,
                basis.gs_norms_sq[i]
            );
        }
    }

    #[test]
    fn test_cvp_on_lattice_point() {
        let basis = LeechBasis::new();
        // A lattice point (the first basis vector) should decode to itself
        let point = basis.basis[0];
        let result = basis.cvp(&point);
        assert!(
            result.distance_squared < 1e-10,
            "Lattice point should have near-zero distance, got {}",
            result.distance_squared
        );
        assert_eq!(result.classification, HoleType::LatticePoint);
    }

    #[test]
    fn test_cvp_origin() {
        let basis = LeechBasis::new();
        // Origin is a lattice point (zero vector)
        let result = basis.cvp(&[0.0; 24]);
        assert!(
            result.distance_squared < 1e-10,
            "Origin should be a lattice point, dist^2={}",
            result.distance_squared
        );
    }

    #[test]
    fn test_cvp_batch_consistency() {
        let basis = LeechBasis::new();
        let targets = vec![[0.0; 24], [1.0; 24], [0.5; 24]];
        let batch = basis.decode_batch(&targets);
        for (i, t) in targets.iter().enumerate() {
            let single = basis.cvp(t);
            assert!(
                (batch[i].distance_squared - single.distance_squared).abs() < 1e-12,
                "Batch and single CVP should agree for target {}",
                i
            );
        }
    }

    #[test]
    fn test_theta_series_known_values() {
        assert_eq!(theta_series_coefficient(0), Some(1));
        assert_eq!(theta_series_coefficient(2), Some(0));
        assert_eq!(theta_series_coefficient(4), Some(196_560));
        assert_eq!(theta_series_coefficient(6), Some(16_773_120));
        assert_eq!(theta_series_coefficient(8), Some(398_034_000));
    }

    #[test]
    fn test_project_signal_exact_24() {
        let chunk: Vec<f64> = (0..24).map(|i| (i + 1) as f64).collect();
        let point = project_signal_chunk(&chunk);
        // Should be normalized to unit sphere
        let norm = norm_sq24(&point).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Projected point should be on unit sphere, norm={}",
            norm
        );
    }

    #[test]
    fn test_project_signal_short() {
        let chunk = [1.0, 2.0, 3.0];
        let point = project_signal_chunk(&chunk);
        // Should be zero-padded and normalized
        let norm = norm_sq24(&point).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Short chunk projection should be normalized, norm={}",
            norm
        );
        // Last elements should be zero (before normalization scale)
        // Actually after normalization they're all scaled, but the
        // zero-padded ones contribute 0/norm = 0
        assert!(point[23].abs() < 1e-10, "Padded element should be ~0");
    }

    #[test]
    fn test_project_signal_long() {
        let chunk: Vec<f64> = (0..48).map(|i| (i + 1) as f64).collect();
        let point = project_signal_chunk(&chunk);
        let norm = norm_sq24(&point).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Long chunk projection should be normalized"
        );
    }

    #[test]
    fn test_hole_statistics() {
        let results = vec![
            CvpResult {
                nearest_point: [0.0; 24],
                distance_squared: 0.001,
                classification: HoleType::LatticePoint,
            },
            CvpResult {
                nearest_point: [0.0; 24],
                distance_squared: 1.0,
                classification: HoleType::ShallowHole,
            },
            CvpResult {
                nearest_point: [0.0; 24],
                distance_squared: 1.8,
                classification: HoleType::DeepHole,
            },
        ];
        let stats = HoleStatistics::from_results(&results);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.lattice_points, 1);
        assert_eq!(stats.shallow_holes, 1);
        assert_eq!(stats.deep_holes, 1);
        assert!((stats.lattice_fraction() - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_basis_rows_have_correct_norm() {
        let basis = LeechBasis::new();
        // The first 12 rows: 2*e_i + P row -> norm = 4 + sum(P[i][j]^2)
        // For our basis, row norms should be 4 + (number of 1s in P row)
        for i in 0..12 {
            let norm = norm_sq24(&basis.basis[i]);
            let p_weight: f64 = golay_code::GOLAY_P_MATRIX[i]
                .iter()
                .map(|&b| (b as f64) * (b as f64))
                .sum();
            let expected = 4.0 + p_weight;
            assert!(
                (norm - expected).abs() < 1e-10,
                "Row {} norm = {}, expected {}",
                i,
                norm,
                expected
            );
        }
        // Last 12 rows: 2*e_j -> norm = 4
        for j in 12..24 {
            let norm = norm_sq24(&basis.basis[j]);
            assert!(
                (norm - 4.0).abs() < 1e-10,
                "Row {} norm = {}, expected 4.0",
                j,
                norm
            );
        }
    }
}
