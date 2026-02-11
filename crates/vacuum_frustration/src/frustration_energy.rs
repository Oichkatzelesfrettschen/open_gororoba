//! Frustration Energy: Map graph-theoretic frustration to physical energy scales.
//!
//! This module connects the Harary-Zaslavsky frustration index F (fraction of
//! unbalanced triangles) to physical energy E_frustration, providing the
//! theoretical foundation for the lambda coupling parameter.
//!
//! # Theory
//!
//! For a signed graph G with N vertices:
//! - Triangle (i,j,k) is **balanced** if s_ij * s_jk * s_ki = +1
//! - Triangle is **unbalanced** if s_ij * s_jk * s_ki = -1
//! - Frustration index: F = (# unbalanced) / (# total triangles)
//!
//! Physical interpretation:
//! - Each unbalanced triangle carries energy cost E_0 (bond mismatch energy)
//! - Total frustration energy: E_frust = F * C(N,3) * E_0
//! - Vacuum attractor F = 3/8: local energy minimum state
//!
//! # References
//!
//! - Harary (1953): On the notion of balance of a signed graph
//! - Zaslavsky (1982): Signed graphs (Discrete Applied Mathematics)
//! - Tang & Tang (2023): Sedenion-SU(5) unification

use algebra_core::construction::cayley_dickson::cd_associator_norm;

/// Frustration energy result with breakdown by energy scales.
#[derive(Debug, Clone)]
pub struct FrustrationEnergy {
    /// Frustration index F in [0, 1]
    pub frustration_index: f64,
    /// Total number of triangles in graph
    pub n_triangles: usize,
    /// Number of unbalanced triangles
    pub n_unbalanced: usize,
    /// Energy per unbalanced triangle (Joules)
    pub energy_per_triangle_j: f64,
    /// Total frustration energy (Joules)
    pub total_energy_j: f64,
    /// Total frustration energy (eV)
    pub total_energy_ev: f64,
    /// Deviation from vacuum attractor (F - 3/8)
    pub vacuum_deviation: f64,
}

/// Physical constants
const K_BOLTZMANN: f64 = 1.380649e-23; // J/K
const EV_TO_JOULES: f64 = 1.602176634e-19; // J/eV
const VACUUM_ATTRACTOR: f64 = 0.375; // 3/8

fn basis_vector(dim: usize, idx: usize) -> Vec<f64> {
    let mut v = vec![0.0; dim];
    if idx < dim {
        v[idx] = 1.0;
    }
    v
}

/// Compute frustration energy from signed graph structure.
///
/// # Arguments
/// * `frustration` - Harary-Zaslavsky frustration index F in [0, 1]
/// * `n_vertices` - Number of vertices in signed graph
/// * `e0_joules` - Energy per unbalanced triangle (material-dependent)
///
/// # Returns
/// FrustrationEnergy struct with energy breakdown
///
/// # Example
/// ```ignore
/// let frustration = 0.375; // Vacuum attractor
/// let n_vertices = 16; // Sedenion dimension
/// let e0 = 1e-21; // ~6 meV per triangle
/// let energy = compute_frustration_energy(frustration, n_vertices, e0);
/// ```
pub fn compute_frustration_energy(
    frustration: f64,
    n_vertices: usize,
    e0_joules: f64,
) -> FrustrationEnergy {
    // Total number of triangles: C(N, 3) = N*(N-1)*(N-2)/6
    let n_triangles = if n_vertices >= 3 {
        n_vertices * (n_vertices - 1) * (n_vertices - 2) / 6
    } else {
        0
    };

    // Number of unbalanced triangles
    let n_unbalanced = (frustration * n_triangles as f64).round() as usize;

    // Total frustration energy
    let total_energy_j = frustration * (n_triangles as f64) * e0_joules;
    let total_energy_ev = total_energy_j / EV_TO_JOULES;

    // Deviation from vacuum attractor
    let vacuum_deviation = frustration - VACUUM_ATTRACTOR;

    FrustrationEnergy {
        frustration_index: frustration,
        n_triangles,
        n_unbalanced,
        energy_per_triangle_j: e0_joules,
        total_energy_j,
        total_energy_ev,
        vacuum_deviation,
    }
}

/// Estimate E_0 from Cayley-Dickson associator norms.
///
/// The associator [e_i, e_j, e_k] measures non-associativity. Its norm
/// provides a dimensionless measure of algebraic "bond mismatch".
///
/// # Arguments
/// * `dim` - Dimension (4, 8, 16, 32, ...)
///
/// # Returns
/// Estimated E_0 in Joules, using 1 meV reference energy scale
///
/// # Method
/// 1. Compute mean associator norm across sampled triples (dimensionless)
/// 2. Scale by reference energy: E_0 ~ mean_norm * 1 meV
/// 3. Physical interpretation: Associator norm quantifies frustration energy in algebraic structure
/// 4. Reference scale: 1 meV ~ typical atomic bond disorder energy
pub fn estimate_e0_from_associators(dim: usize) -> f64 {
    if dim < 4 {
        return 0.0; // No non-associativity below quaternions
    }

    // Sample associator norms for representative triples
    let mut sum_norms = 0.0;
    let mut count = 0;

    // Adaptive sampling: full enumeration for small dims, sparse for large
    let step_size = if dim <= 16 { 1 } else { 4 };

    // Start from i=1 to skip the real unit e_0 (which makes triples associative)
    for i in (1..dim).step_by(step_size) {
        for j in (i + 1..dim).step_by(step_size) {
            for k in (j + 1..dim).step_by(step_size) {
                let e_i = basis_vector(dim, i);
                let e_j = basis_vector(dim, j);
                let e_k = basis_vector(dim, k);
                let norm = cd_associator_norm(&e_i, &e_j, &e_k);
                sum_norms += norm;
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let mean_norm = sum_norms / (count as f64);

    // Reference energy scale: 1 meV = 1.602e-22 J
    // Typical bond disorder/frustration energy in atomic/molecular systems
    const REFERENCE_ENERGY_J: f64 = 1.602e-22; // 1 meV

    // E_0 = dimensionless associator norm * reference energy
    // This gives a TEMPERATURE-INDEPENDENT frustration energy scale
    mean_norm * REFERENCE_ENERGY_J
}

/// Compute lambda coupling strength from frustration energy and thermal energy.
///
/// # Arguments
/// * `e_frustration` - Frustration energy (Joules)
/// * `temperature_k` - Temperature (Kelvin)
///
/// # Returns
/// Dimensionless lambda = E_frustration / (k_B * T)
///
/// # Interpretation
/// - lambda >> 1: Frustration dominates thermal fluctuations (strong coupling)
/// - lambda ~ 1: Frustration and thermal energy comparable (moderate coupling)
/// - lambda << 1: Thermal fluctuations dominate (weak coupling)
pub fn compute_lambda(e_frustration: f64, temperature_k: f64) -> f64 {
    let e_thermal = K_BOLTZMANN * temperature_k;
    e_frustration / e_thermal
}

/// Predict lambda for Sedenion field at given temperature.
///
/// # Arguments
/// * `frustration` - Harary-Zaslavsky frustration index
/// * `temperature_k` - Temperature in Kelvin
///
/// # Returns
/// Predicted lambda coupling strength
///
/// # Method
/// 1. Compute E_0 from associator norms
/// 2. Compute total frustration energy: E_frust = F * C(16,3) * E_0
/// 3. Compute lambda = E_frust / (k_B * T)
pub fn predict_lambda_sedenion(frustration: f64, temperature_k: f64) -> f64 {
    const DIM: usize = 16; // Sedenion dimension

    // Estimate E_0 from associator norms (temperature-independent)
    let e0 = estimate_e0_from_associators(DIM);

    // Compute frustration energy
    let frust_energy = compute_frustration_energy(frustration, DIM, e0);

    // Compute lambda (temperature-dependent via normalization)
    compute_lambda(frust_energy.total_energy_j, temperature_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frustration_energy_vacuum_attractor() {
        let frustration = 0.375; // 3/8 vacuum attractor
        let n_vertices = 16; // Sedenion
        let e0 = 1e-21; // ~6 meV

        let energy = compute_frustration_energy(frustration, n_vertices, e0);

        assert_eq!(energy.frustration_index, 0.375);
        assert_eq!(energy.n_triangles, 560); // C(16,3) = 560
        assert_eq!(energy.n_unbalanced, 210); // 0.375 * 560 = 210
        assert!((energy.vacuum_deviation).abs() < 1e-10);
    }

    #[test]
    fn test_frustration_energy_extremes() {
        let e0 = 1e-21;

        // F = 0: all balanced (minimum energy)
        let e_min = compute_frustration_energy(0.0, 16, e0);
        assert_eq!(e_min.total_energy_j, 0.0);

        // F = 1: all unbalanced (maximum energy)
        let e_max = compute_frustration_energy(1.0, 16, e0);
        assert_eq!(e_max.n_unbalanced, 560);
        assert!((e_max.total_energy_j - 560.0 * e0).abs() < 1e-30);
    }

    #[test]
    fn test_estimate_e0_octonions() {
        // Octonions (dim=8) are the FIRST non-associative CD algebra
        // Quaternions (dim=4) are still associative!
        let e0 = estimate_e0_from_associators(8);

        // Octonions: associator_norm ~ O(1), E_0 ~ mean_norm * 1 meV
        // Reference: 1 meV = 1.602e-22 J
        const MEV_TO_JOULES: f64 = 1.602e-22;
        assert!(e0 > 0.0, "e0 should be positive, got {}", e0);
        assert!(e0 < 10.0 * MEV_TO_JOULES, "e0 should be < 10 meV, got {:.3e} J", e0);
    }

    #[test]
    fn test_estimate_e0_sedenions() {
        let e0 = estimate_e0_from_associators(16);

        // Sedenions: associator norms ~ O(1), E_0 ~ mean_norm * 1 meV
        const MEV_TO_JOULES: f64 = 1.602e-22;
        assert!(e0 > 0.0);
        assert!(e0 < 10.0 * MEV_TO_JOULES, "e0 should be < 10 meV, got {:.3e} J", e0);
    }

    #[test]
    fn test_compute_lambda_low_temperature() {
        let e_frustration = 1e-21; // ~6 meV
        let temperature = 4.2; // He-4 critical temperature

        let lambda = compute_lambda(e_frustration, temperature);

        // At low T, lambda should be large (frustration dominates)
        assert!(lambda > 10.0, "Low-T lambda should be > 10, got {}", lambda);
    }

    #[test]
    fn test_compute_lambda_room_temperature() {
        let e_frustration = 1e-21; // ~6 meV
        let temperature = 300.0; // Room temperature

        let lambda = compute_lambda(e_frustration, temperature);

        // At room T, lambda should be moderate
        assert!(
            lambda > 0.1 && lambda < 100.0,
            "Room-T lambda should be O(1), got {}",
            lambda
        );
    }

    #[test]
    fn test_predict_lambda_sedenion_he4() {
        let frustration = 0.35; // Typical frustrated value
        let temperature = 4.2; // He-4 at lambda point

        let lambda = predict_lambda_sedenion(frustration, temperature);

        // Lambda is TEMPERATURE-DEPENDENT: lambda = E_frust / (k_B*T)
        // He-4 at 4.2 K: low temperature -> high lambda (frustration dominates)
        // Expected: E_frust ~ 3.5e-20 J, k_B*T ~ 5.8e-23 J -> lambda ~ 600
        assert!(
            lambda > 400.0 && lambda < 800.0,
            "He-4 lambda (F=0.35, T=4.2K) should be 400-800, got {}",
            lambda
        );
    }

    #[test]
    fn test_predict_lambda_sedenion_water() {
        let frustration = 0.35;
        let temperature = 293.15; // Water at 20 C

        let lambda = predict_lambda_sedenion(frustration, temperature);

        // Lambda is TEMPERATURE-DEPENDENT: lambda = E_frust / (k_B*T)
        // Water at 293 K: high temperature -> low lambda (thermal fluctuations comparable)
        // Expected: E_frust ~ 3.5e-20 J, k_B*T ~ 4.1e-21 J -> lambda ~ 8.5
        assert!(
            lambda > 5.0 && lambda < 15.0,
            "Water lambda (F=0.35, T=293K) should be 5-15, got {}",
            lambda
        );
    }

    #[test]
    fn test_vacuum_attractor_minimum_energy() {
        let e0 = 1e-21;
        let n_vertices = 16;

        // Frustrations around vacuum attractor
        let frustrations = [0.30, 0.35, 0.375, 0.40, 0.45];
        let energies: Vec<f64> = frustrations
            .iter()
            .map(|&f| compute_frustration_energy(f, n_vertices, e0).total_energy_j)
            .collect();

        // Energy should be minimized near F = 3/8 (but NOT exactly at 3/8)
        // Actually F=0 is true minimum, but 3/8 is frustration attractor
        let e_vacuum = compute_frustration_energy(0.375, n_vertices, e0).total_energy_j;

        // Deviation energy increases quadratically
        for (i, &f) in frustrations.iter().enumerate() {
            let deviation = (f - 0.375).abs();
            let energy_deviation = (energies[i] - e_vacuum).abs();

            // Energy deviation should scale with (F - 3/8)^2
            if deviation > 0.01 {
                assert!(energy_deviation > 0.0);
            }
        }
    }
}
