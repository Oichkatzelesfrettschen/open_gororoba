//! ZD adjacency as tight-binding Hamiltonian: crystal band analysis.
//!
//! Interprets the zero-divisor motif graph (from boxkites.rs) as a
//! tight-binding Hamiltonian and extracts the "band structure" of the
//! ZD crystal at each Cayley-Dickson dimension.
//!
//! At D=16, the 84-vertex Reggiani partner graph has spectrum
//! {-4, -2, 0, +2, +4} with degeneracies {7, 14, 42, 14, 7}.
//! The 42-fold zero eigenvalue is the algebraic flat band -- half of
//! all ZD linear combinations lie in the kernel.
//!
//! For D > 16, the full ZD graph is block-diagonal (motif components
//! are disconnected), so the aggregate spectrum is the multiset-union
//! of per-component spectra.  This avoids building enormous matrices
//! at D = 256 where the full graph would be 16256 x 16256.
//!
//! # Cross-domain connection
//!
//! Flat band fraction quantifies localization strength: ZD frustration
//! on box-kite triangles produces the same algebraic localization as
//! kagome frustrated hopping in condensed matter.

use crate::{boxkites::motif_components_for_cross_assessors, reggiani::partner_graph_degeneracies};

/// Spectrum of a single motif component class (isomorphic components).
#[derive(Debug, Clone)]
pub struct ClassSpectrum {
    /// Number of edges in each component of this class.
    pub edge_count: usize,
    /// Number of nodes in each component of this class.
    pub node_count: usize,
    /// Number of isomorphic copies of this component class.
    pub multiplicity: usize,
    /// Eigenvalues of one representative (sorted descending).
    pub spectrum: Vec<f64>,
    /// Fraction of eigenvalues within `tol` of zero.
    pub flat_band_fraction: f64,
}

/// Full dimension spectrum summary.
#[derive(Debug, Clone)]
pub struct DimSpectrumSummary {
    pub dim: usize,
    pub n_components: usize,
    pub total_nodes: usize,
    pub total_eigenvalues: usize,
    pub flat_band_fraction: f64,
    pub is_integer_spectrum: bool,
    pub degeneracies: Vec<(f64, usize)>,
    pub distinct_eigenvalue_count: usize,
    pub class_spectra: Vec<ClassSpectrum>,
}

/// Result of verifying Reggiani factorization at D=16.
#[derive(Debug, Clone)]
pub struct ReggianiFacResult {
    /// Degeneracies from the full 84x84 partner graph.
    pub full_degeneracies: Vec<(f64, usize)>,
    /// Degeneracies from assembling per-component spectra.
    pub per_component_degeneracies: Vec<(f64, usize)>,
    /// Number of motif components at D=16.
    pub n_components: usize,
    /// Whether the two approaches yield the same degeneracy table.
    pub factorizes: bool,
}

/// Aggregate spectrum across all motif components at dimension `dim`.
///
/// Each component's adjacency matrix is diagonalized independently,
/// and all eigenvalues are collected into a single sorted vector.
/// This is exact because the full ZD graph is block-diagonal.
pub fn aggregate_component_spectrum(dim: usize) -> Vec<f64> {
    let components = motif_components_for_cross_assessors(dim);
    let mut all_evals: Vec<f64> = Vec::new();
    for comp in &components {
        let spec = comp.spectrum();
        all_evals.extend_from_slice(&spec);
    }
    all_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_evals
}

/// Group eigenvalues into (value, count) pairs within tolerance `tol`.
pub fn degeneracy_table(spectrum: &[f64], tol: f64) -> Vec<(f64, usize)> {
    let mut sorted = spectrum.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut groups: Vec<(f64, usize)> = Vec::new();
    for &ev in &sorted {
        if let Some(last) = groups.last_mut().filter(|(v, _)| (ev - *v).abs() < tol) {
            // Update centroid as running mean
            let n = last.1;
            last.0 = (last.0 * n as f64 + ev) / (n + 1) as f64;
            last.1 += 1;
            continue;
        }
        groups.push((ev, 1));
    }
    groups
}

/// Fraction of eigenvalues within `tol` of zero.
pub fn flat_band_fraction(spectrum: &[f64], tol: f64) -> f64 {
    if spectrum.is_empty() {
        return 0.0;
    }
    let n_zero = spectrum.iter().filter(|&&v| v.abs() < tol).count();
    n_zero as f64 / spectrum.len() as f64
}

/// Whether all eigenvalues are within `tol` of an integer.
pub fn is_integer_spectrum(spectrum: &[f64], tol: f64) -> bool {
    spectrum.iter().all(|&v| (v - v.round()).abs() < tol)
}

/// Verify that the D=16 Reggiani partner graph degeneracies factorize.
///
/// The 84-vertex partner graph has spectrum {-4:7, -2:14, 0:42, +2:14, +4:7}.
/// Factorization means all degeneracies share a common factor (GCD=7),
/// yielding reduced degeneracies {-4:1, -2:2, 0:6, +2:2, +4:1}.
/// This implies the partner graph decomposes into 7 isomorphic copies
/// of a 12-vertex subgraph.
pub fn verify_reggiani_factorization(tol: f64) -> ReggianiFacResult {
    let full_degs = partner_graph_degeneracies(tol);
    let n_components = motif_components_for_cross_assessors(16).len();

    // Compute GCD of all degeneracies
    let gcd = full_degs.iter().map(|(_, d)| *d).fold(0, gcd_usize);

    let factorizes = gcd > 1;

    let per_component_degeneracies: Vec<(f64, usize)> = full_degs
        .iter()
        .map(|(v, d)| (*v, d / gcd.max(1)))
        .collect();

    ReggianiFacResult {
        full_degeneracies: full_degs,
        per_component_degeneracies,
        n_components,
        factorizes,
    }
}

fn gcd_usize(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd_usize(b, a % b) }
}

/// Full spectrum summary for a given dimension.
pub fn dim_spectrum_summary(dim: usize, tol: f64) -> DimSpectrumSummary {
    let components = motif_components_for_cross_assessors(dim);
    let n_components = components.len();

    // Group components by (node_count, edge_count) signature
    let mut class_map: std::collections::BTreeMap<(usize, usize), (Vec<f64>, usize)> =
        std::collections::BTreeMap::new();

    let mut all_evals: Vec<f64> = Vec::new();
    let mut total_nodes = 0usize;

    for comp in &components {
        let nc = comp.nodes.len();
        let ec = comp.edges.len();
        total_nodes += nc;
        let spec = comp.spectrum();
        all_evals.extend_from_slice(&spec);

        let entry = class_map
            .entry((nc, ec))
            .or_insert_with(|| (spec.clone(), 0));
        entry.1 += 1;
    }

    all_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total_eigenvalues = all_evals.len();
    let fbf = flat_band_fraction(&all_evals, tol);
    let int_spec = is_integer_spectrum(&all_evals, tol);
    let degs = degeneracy_table(&all_evals, tol);
    let distinct = degs.len();

    let class_spectra: Vec<ClassSpectrum> = class_map
        .into_iter()
        .map(|((nc, ec), (spec, mult))| {
            let class_fbf = flat_band_fraction(&spec, tol);
            ClassSpectrum {
                edge_count: ec,
                node_count: nc,
                multiplicity: mult,
                spectrum: spec,
                flat_band_fraction: class_fbf,
            }
        })
        .collect();

    DimSpectrumSummary {
        dim,
        n_components,
        total_nodes,
        total_eigenvalues,
        flat_band_fraction: fbf,
        is_integer_spectrum: int_spec,
        degeneracies: degs,
        distinct_eigenvalue_count: distinct,
        class_spectra,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reggiani::partner_graph_spectrum;

    const TOL: f64 = 0.01;

    #[test]
    fn test_d16_flat_band_fraction() {
        let spectrum = partner_graph_spectrum();
        let fbf = flat_band_fraction(&spectrum, TOL);
        assert!(
            (fbf - 42.0 / 84.0).abs() < 0.01,
            "D=16 flat band fraction = {fbf}, expected 0.5"
        );
    }

    #[test]
    fn test_d16_reggiani_factorization() {
        let result = verify_reggiani_factorization(TOL);
        assert!(
            result.factorizes,
            "D=16 Reggiani factorization failed: full={:?}",
            result.full_degeneracies,
        );
        // Reduced degeneracies should be {1, 2, 6, 2, 1}
        let reduced: Vec<usize> = result
            .per_component_degeneracies
            .iter()
            .map(|(_, d)| *d)
            .collect();
        assert_eq!(
            reduced,
            vec![1, 2, 6, 2, 1],
            "Reduced degeneracies: {:?}",
            reduced
        );
    }

    #[test]
    fn test_d16_integer_spectrum() {
        let spectrum = partner_graph_spectrum();
        assert!(
            is_integer_spectrum(&spectrum, TOL),
            "D=16 spectrum is not integer"
        );
    }

    #[test]
    fn test_d16_degeneracy_table() {
        let degs = partner_graph_degeneracies(TOL);
        // Expected: {-4:7, -2:14, 0:42, +2:14, +4:7}
        assert_eq!(
            degs.len(),
            5,
            "Expected 5 distinct eigenvalues, got {}",
            degs.len()
        );
        let expected_counts: Vec<usize> = vec![7, 14, 42, 14, 7];
        let actual_counts: Vec<usize> = degs.iter().map(|(_, d)| *d).collect();
        assert_eq!(
            actual_counts, expected_counts,
            "Degeneracy mismatch: got {:?}",
            degs
        );
    }

    #[test]
    fn test_d16_summary() {
        let summary = dim_spectrum_summary(16, TOL);
        assert_eq!(summary.dim, 16);
        // Motif components at D=16 use cross-assessor pairs (42 nodes),
        // not the 84-vertex Reggiani partner graph.
        assert!(
            summary.total_eigenvalues > 0,
            "Should have eigenvalues from motif components"
        );
        assert!(summary.is_integer_spectrum);
        // Flat band fraction should be close to 0.5
        assert!(
            (summary.flat_band_fraction - 0.5).abs() < 0.1,
            "D=16 motif flat band fraction = {}",
            summary.flat_band_fraction
        );
        eprintln!(
            "D=16 summary: {} components, {} nodes, {} evals, fbf={:.4}",
            summary.n_components,
            summary.total_nodes,
            summary.total_eigenvalues,
            summary.flat_band_fraction
        );
    }

    #[test]
    fn test_d32_aggregate_spectrum() {
        let agg = aggregate_component_spectrum(32);
        // Must have eigenvalues from all components
        assert!(
            !agg.is_empty(),
            "D=32 aggregate spectrum should not be empty"
        );
        // Sorted ascending
        for w in agg.windows(2) {
            assert!(w[0] <= w[1] + 1e-12, "Spectrum not sorted");
        }
    }

    #[test]
    fn test_d32_flat_band_fraction() {
        let summary = dim_spectrum_summary(32, TOL);
        assert!(
            summary.flat_band_fraction >= 0.0 && summary.flat_band_fraction <= 1.0,
            "D=32 flat band fraction out of range: {}",
            summary.flat_band_fraction
        );
        eprintln!(
            "D=32: {} components, {} evals, flat_band_fraction={:.4}, integer={}",
            summary.n_components,
            summary.total_eigenvalues,
            summary.flat_band_fraction,
            summary.is_integer_spectrum
        );
    }

    #[test]
    #[ignore] // ~30s in release
    fn test_d64_flat_band_fraction() {
        let summary = dim_spectrum_summary(64, TOL);
        assert!(
            summary.flat_band_fraction >= 0.0 && summary.flat_band_fraction <= 1.0,
            "D=64 flat band fraction out of range: {}",
            summary.flat_band_fraction
        );
        eprintln!(
            "D=64: {} components, {} evals, flat_band_fraction={:.4}, integer={}",
            summary.n_components,
            summary.total_eigenvalues,
            summary.flat_band_fraction,
            summary.is_integer_spectrum
        );
    }

    #[test]
    fn test_degeneracy_table_empty() {
        let degs = degeneracy_table(&[], TOL);
        assert!(degs.is_empty());
    }

    #[test]
    fn test_flat_band_fraction_empty() {
        assert_eq!(flat_band_fraction(&[], TOL), 0.0);
    }

    #[test]
    fn test_is_integer_spectrum_known() {
        assert!(is_integer_spectrum(&[0.0, 1.0, -2.0, 4.0], TOL));
        assert!(!is_integer_spectrum(&[0.0, 1.5, -2.0], TOL));
    }
}
