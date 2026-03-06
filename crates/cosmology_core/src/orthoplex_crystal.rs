//! Orthoplex-to-crystal mapping: phonon band structure analogy.
//!
//! The complete multipartite graph K_{2,2,...,2} (orthoplex) has a Laplacian
//! spectrum with specific degeneracy patterns. These degeneracies are
//! analogous to phonon band degeneracies in crystal lattices.
//!
//! For k parts of size 2 (total n = 2k vertices):
//! - mu_0 = 0 (multiplicity 1) -- acoustic mode (translation)
//! - mu_1 = 2(k-1) (multiplicity k) -- "optical-like" modes
//! - mu_2 = 2k (multiplicity k-1) -- highest frequency band
//!
//! The ratio mu_2/mu_1 = k/(k-1) approaches 1 as k grows, mirroring the
//! optical phonon band narrowing in high-symmetry crystals (e.g., Fm-3m FCC
//! metals where LO-TO splitting vanishes at Gamma point).
//!
//! While the cosmological application of orthoplex diffusion was FALSIFIED
//! (C-932, Delta-BIC = +705), the algebraic structure may still predict
//! material properties through this phonon band analogy.

/// Phonon-like band structure from orthoplex Laplacian spectrum.
#[derive(Debug, Clone)]
pub struct OrthoplexPhononAnalogy {
    /// Number of parts k (CD dimension = 4*(k+1)).
    pub k: usize,
    /// Total vertices n = 2k.
    pub n_vertices: usize,
    /// Eigenvalue mu_1 = 2(k-1).
    pub mu_optical: f64,
    /// Eigenvalue mu_2 = 2k.
    pub mu_highest: f64,
    /// Degeneracy of mu_1.
    pub optical_degeneracy: usize,
    /// Degeneracy of mu_2.
    pub highest_degeneracy: usize,
    /// Band ratio mu_2/mu_1 = k/(k-1).
    pub band_ratio: f64,
}

/// Compute the phonon band analogy for orthoplex K_{2,...,2} with k parts.
pub fn orthoplex_phonon_analogy(k: usize) -> OrthoplexPhononAnalogy {
    assert!(k >= 2, "Need at least 2 parts for nontrivial spectrum");

    let mu_optical = 2.0 * (k as f64 - 1.0);
    let mu_highest = 2.0 * k as f64;

    OrthoplexPhononAnalogy {
        k,
        n_vertices: 2 * k,
        mu_optical,
        mu_highest,
        optical_degeneracy: k,
        highest_degeneracy: k - 1,
        band_ratio: mu_highest / mu_optical,
    }
}

/// Compute the Debye-like density of states for orthoplex graph.
///
/// Returns (omega, g(omega)) pairs where g(omega) is the number of modes
/// at eigenvalue omega (delta-function spikes for graph Laplacian).
pub fn orthoplex_dos(k: usize) -> Vec<(f64, f64)> {
    vec![
        (0.0, 1.0),
        (2.0 * (k as f64 - 1.0), k as f64),
        (2.0 * k as f64, (k - 1) as f64),
    ]
}

/// Heat kernel trace P(t) = (1/(2k)) * [1 + k*exp(-mu_1*t) + (k-1)*exp(-mu_2*t)].
pub fn orthoplex_heat_kernel(k: usize, t: f64) -> f64 {
    let n = 2.0 * k as f64;
    let mu1 = 2.0 * (k as f64 - 1.0);
    let mu2 = 2.0 * k as f64;
    (1.0 + k as f64 * (-mu1 * t).exp() + (k as f64 - 1.0) * (-mu2 * t).exp()) / n
}

/// Spectral dimension d_s(t) = -2t * d(ln P)/dt.
pub fn orthoplex_spectral_dimension(k: usize, t: f64) -> f64 {
    let n = 2.0 * k as f64;
    let mu1 = 2.0 * (k as f64 - 1.0);
    let mu2 = 2.0 * k as f64;

    let numerator =
        k as f64 * mu1 * (-mu1 * t).exp() + (k as f64 - 1.0) * mu2 * (-mu2 * t).exp();
    let denominator =
        1.0 + k as f64 * (-mu1 * t).exp() + (k as f64 - 1.0) * (-mu2 * t).exp();

    if denominator.abs() < 1e-300 {
        return 0.0;
    }

    2.0 * t * numerator / (n * denominator / n)
}

/// Compare orthoplex degeneracy pattern with FCC Fm-3m (225) phonon structure.
///
/// In FCC with p atoms per primitive cell:
/// - 3 acoustic branches (1 LA + 2 TA)
/// - 3(p-1) optical branches
/// - At Gamma: optical modes have specific degeneracies from Oh symmetry
///
/// For FCC elemental metals (p=1): 3 branches total, all acoustic.
/// For NaCl structure (p=2): 6 branches = 3 acoustic + 3 optical.
///
/// Returns (orthoplex_acoustic_frac, fcc_acoustic_frac) for comparison.
pub fn compare_with_fcc(k: usize, atoms_per_cell: usize) -> (f64, f64) {
    let ortho_total = 2 * k;
    let ortho_acoustic = 1; // mu=0 has multiplicity 1
    let ortho_acoustic_frac = ortho_acoustic as f64 / ortho_total as f64;

    let fcc_total = 3 * atoms_per_cell;
    let fcc_acoustic = 3;
    let fcc_acoustic_frac = fcc_acoustic as f64 / fcc_total as f64;

    (ortho_acoustic_frac, fcc_acoustic_frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phonon_analogy_k4() {
        let pa = orthoplex_phonon_analogy(4);
        assert_eq!(pa.n_vertices, 8);
        assert!((pa.mu_optical - 6.0).abs() < 1e-10);
        assert!((pa.mu_highest - 8.0).abs() < 1e-10);
        assert_eq!(pa.optical_degeneracy, 4);
        assert_eq!(pa.highest_degeneracy, 3);
        assert!((pa.band_ratio - 4.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_phonon_analogy_k63() {
        let pa = orthoplex_phonon_analogy(63);
        assert!((pa.mu_optical - 124.0).abs() < 1e-10);
        assert!((pa.mu_highest - 126.0).abs() < 1e-10);
        // Band ratio approaches 1 for large k
        assert!((pa.band_ratio - 63.0 / 62.0).abs() < 1e-10);
        assert!(pa.band_ratio < 1.02, "Band ratio should be near 1 for large k");
    }

    #[test]
    fn test_heat_kernel_normalization() {
        // P(0) = 1 for any k
        for k in [2, 4, 8, 63] {
            let p0 = orthoplex_heat_kernel(k, 0.0);
            assert!(
                (p0 - 1.0).abs() < 1e-12,
                "P(0) = {} for k={}, expected 1.0",
                p0,
                k
            );
        }
    }

    #[test]
    fn test_heat_kernel_decay() {
        // P(t) -> 1/(2k) as t -> infinity
        let k = 8;
        let p_large = orthoplex_heat_kernel(k, 1000.0);
        let expected = 1.0 / (2.0 * k as f64);
        assert!(
            (p_large - expected).abs() < 1e-10,
            "P(inf) = {}, expected {}",
            p_large,
            expected
        );
    }

    #[test]
    fn test_spectral_dimension_bounded() {
        for k in [4, 8, 16, 63] {
            for t in [0.01, 0.1, 1.0, 10.0] {
                let ds = orthoplex_spectral_dimension(k, t);
                assert!(ds >= 0.0, "d_s should be non-negative: d_s({})={}", t, ds);
            }
        }
    }

    #[test]
    fn test_dos_total_modes() {
        let k = 63;
        let dos = orthoplex_dos(k);
        let total: f64 = dos.iter().map(|(_, g)| g).sum();
        assert!(
            (total - 2.0 * k as f64).abs() < 1e-10,
            "Total modes = {}, expected {}",
            total,
            2 * k
        );
    }

    #[test]
    fn test_fcc_comparison() {
        let (ortho_frac, fcc_frac) = compare_with_fcc(63, 1);
        // Orthoplex: 1 acoustic out of 126 total
        assert!((ortho_frac - 1.0 / 126.0).abs() < 1e-10);
        // FCC elemental: 3 acoustic out of 3 total = 1.0
        assert!((fcc_frac - 1.0).abs() < 1e-10);
    }
}
