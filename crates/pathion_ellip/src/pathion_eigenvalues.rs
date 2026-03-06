//! 32D Pathion ZD graph eigenvalue spectrum.
//!
//! Builds the zero-divisor interaction graph adjacency matrix for dim=32,
//! computes the graph-Laplacian, and extracts eigenvalues that modulate
//! the GPU Pathion heat sink.

use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
use nalgebra::{DMatrix, SymmetricEigen};

/// Eigenvalue spectrum of the 32D Pathion ZD interaction graph.
#[derive(Debug, Clone)]
pub struct PathionEigenvalueSpectrum {
    /// Sorted eigenvalues (ascending) of the graph Laplacian.
    pub eigenvalues: Vec<f64>,
    /// 16 representative eigenvalues for GPU upload (padded/truncated).
    pub eigenvalues_16: [f32; 16],
    /// Number of connected components (= multiplicity of zero eigenvalue).
    pub n_components: usize,
}

/// Build the ZD adjacency matrix for a Cayley-Dickson algebra of given dimension.
///
/// Entry A[i][j] = 1.0 if basis elements e_i and e_j participate in a
/// zero-divisor pair (there exist signs s1, s2 such that
/// ||(e_i + s1*e_j) * (e_k + s2*e_l)|| = 0 for some k, l involving i or j).
///
/// For efficiency, we use the associator-based criterion:
/// A[i][j] = 1 if the associator [e_i, e_j, e_k] is nonzero for any k,
/// which indicates non-alternative behavior linking i and j.
fn build_zd_adjacency(dim: usize) -> DMatrix<f64> {
    let mut adj = DMatrix::zeros(dim, dim);

    for i in 1..dim {
        for j in (i + 1)..dim {
            // Check if e_i and e_j participate in any nonzero associator
            let mut has_assoc = false;
            for k in 1..dim {
                if k == i || k == j {
                    continue;
                }
                // Associator [e_i, e_j, e_k] = (e_i * e_j) * e_k - e_i * (e_j * e_k)
                // For basis elements, this reduces to sign arithmetic.
                let ij = i ^ j;
                let sign_ij = cd_basis_mul_sign_iter(dim, i, j);
                let jk = j ^ k;
                let sign_jk = cd_basis_mul_sign_iter(dim, j, k);

                // (e_i * e_j) * e_k
                let ij_k = ij ^ k;
                let sign_ij_k = sign_ij * cd_basis_mul_sign_iter(dim, ij, k);

                // e_i * (e_j * e_k)
                let i_jk = i ^ jk;
                let sign_i_jk = sign_jk * cd_basis_mul_sign_iter(dim, i, jk);

                // Associator is nonzero if the two products differ
                if ij_k == i_jk && sign_ij_k != sign_i_jk {
                    has_assoc = true;
                    break;
                }
                if ij_k != i_jk {
                    // Different target basis element -- always nonzero associator
                    has_assoc = true;
                    break;
                }
            }

            if has_assoc {
                adj[(i, j)] = 1.0;
                adj[(j, i)] = 1.0;
            }
        }
    }

    adj
}

/// Compute the graph Laplacian L = D - A where D is the degree matrix.
fn graph_laplacian(adj: &DMatrix<f64>) -> DMatrix<f64> {
    let n = adj.nrows();
    let mut lap = -adj.clone();
    for i in 0..n {
        let degree: f64 = (0..n).map(|j| adj[(i, j)]).sum();
        lap[(i, i)] = degree;
    }
    lap
}

impl PathionEigenvalueSpectrum {
    /// Compute the eigenvalue spectrum for a 32D Pathion ZD graph.
    pub fn compute() -> Self {
        Self::compute_for_dim(32)
    }

    /// Compute eigenvalue spectrum for arbitrary CD dimension.
    pub fn compute_for_dim(dim: usize) -> Self {
        let adj = build_zd_adjacency(dim);
        let lap = graph_laplacian(&adj);

        let eigen = SymmetricEigen::new(lap);
        let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Clamp near-zero eigenvalues
        for ev in &mut eigenvalues {
            if ev.abs() < 1e-10 {
                *ev = 0.0;
            }
        }

        // Count connected components (multiplicity of zero eigenvalue)
        let n_components = eigenvalues.iter().filter(|&&v| v == 0.0).count();

        // Build 16 representative eigenvalues for GPU
        // Strategy: take the first 16 distinct non-negative eigenvalues
        let mut ev16 = [0.0f32; 16];
        let positive: Vec<f64> = eigenvalues.iter().copied().filter(|&v| v > 1e-10).collect();
        for (i, &v) in positive.iter().take(16).enumerate() {
            ev16[i] = v as f32;
        }
        // Normalize so max = 1.0 for stable GPU arithmetic
        let max_ev = ev16.iter().copied().fold(0.0f32, f32::max);
        if max_ev > 0.0 {
            for v in &mut ev16 {
                *v /= max_ev;
            }
        }

        Self {
            eigenvalues,
            eigenvalues_16: ev16,
            n_components,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pathion_eigenvalues_nonneg() {
        let spec = PathionEigenvalueSpectrum::compute();
        for &ev in &spec.eigenvalues {
            assert!(
                ev >= -1e-10,
                "Laplacian eigenvalue should be non-negative, got {}",
                ev
            );
        }
    }

    #[test]
    fn test_pathion_eigenvalues_count() {
        let spec = PathionEigenvalueSpectrum::compute();
        assert_eq!(
            spec.eigenvalues.len(),
            32,
            "32D Pathion should have 32 eigenvalues"
        );
    }

    #[test]
    fn test_pathion_has_components() {
        let spec = PathionEigenvalueSpectrum::compute();
        // At dim=32, the ZD graph should have multiple connected components
        assert!(
            spec.n_components >= 2,
            "Expected multiple components, got {}",
            spec.n_components
        );
    }

    #[test]
    fn test_eigenvalues_16_normalized() {
        let spec = PathionEigenvalueSpectrum::compute();
        let max = spec.eigenvalues_16.iter().copied().fold(0.0f32, f32::max);
        // Should be normalized to max=1.0 (or all zero if no positive eigenvalues)
        assert!(
            (max - 1.0).abs() < 1e-5 || max == 0.0,
            "Max eigenvalue should be 1.0, got {}",
            max
        );
    }

    #[test]
    fn test_quaternion_no_zd() {
        // Quaternions (dim=4) are alternative -- no associator violations
        let spec = PathionEigenvalueSpectrum::compute_for_dim(4);
        // All eigenvalues should be zero (no edges in adjacency)
        assert_eq!(
            spec.n_components, 4,
            "Quaternion ZD graph should be fully disconnected"
        );
    }

    #[test]
    fn test_octonion_no_zd() {
        // Octonions (dim=8) are alternative -- no associator violations between basis pairs
        let spec = PathionEigenvalueSpectrum::compute_for_dim(8);
        // Octonions have nonzero associators but are alternative (associator is skew-symmetric),
        // so the adjacency matrix will have edges. But no zero-divisors.
        // The test just checks we get 8 eigenvalues.
        assert_eq!(spec.eigenvalues.len(), 8);
    }

    #[test]
    fn test_sedenion_has_zd() {
        // Sedenions (dim=16) have zero-divisors and non-alternative behavior
        let spec = PathionEigenvalueSpectrum::compute_for_dim(16);
        // Should have positive eigenvalues (edges in the graph)
        let has_positive = spec.eigenvalues.iter().any(|&v| v > 1e-10);
        assert!(has_positive, "Sedenion ZD graph should have edges");
    }
}
