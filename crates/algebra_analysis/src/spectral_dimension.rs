//! Graph spectral dimension of zero-divisor graphs.
//!
//! Computes the spectral dimension d_s(t) of the cross-assessor zero-divisor
//! graphs across Cayley-Dickson dimensions via heat kernel methods on the
//! graph Laplacian.
//!
//! # Physical motivation
//!
//! Carlip (2017) showed that d_s -> 2 in the UV is "universal" across quantum
//! gravity approaches. This module tests whether the spectral dimension of
//! zero-divisor graphs recovers the Calcagni flow from pure algebraic structure:
//!
//!   d_s(t) = -2 * d(ln P(t)) / d(ln t)
//!
//! where P(t) = (1/N) sum_{lambda_i > 0} exp(-lambda_i * t) is the heat kernel
//! return probability on the graph Laplacian.
//!
//! # Algorithm
//!
//! 1. Build cross-assessor ZD graph via `motif_components_for_cross_assessors(dim)`.
//! 2. Compute Laplacian L = D - A from adjacency.
//! 3. Eigendecompose L via nalgebra `symmetric_eigen()`.
//! 4. Clamp near-zero eigenvalues (floating-point dust from dense solver).
//! 5. Compute P(t) on geometrically spaced t grid.
//! 6. Savitzky-Golay smooth ln P(t) vs ln t.
//! 7. Central-difference differentiate to get d_s(t).
//! 8. Extract plateau value where d_s(t) stabilizes.
//!
//! # References
//!
//! - Carlip, Class. Quantum Grav. 34 (2017) 193001 ("Dimension and dimensional
//!   reduction in quantum gravity")
//! - Calcagni, PRL 104 (2010) 251301

use nalgebra::DMatrix;
use petgraph::graph::UnGraph;

use crate::boxkites::{MotifComponent, motif_components_for_cross_assessors};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Plateau fit result from a d_s(t) curve.
#[derive(Debug, Clone)]
pub struct PlateauFit {
    /// Plateau spectral dimension value.
    pub d_s: f64,
    /// Lower bound of the plateau region in ln(t).
    pub ln_t_min: f64,
    /// Upper bound of the plateau region in ln(t).
    pub ln_t_max: f64,
    /// R-squared of the constant fit within the plateau.
    pub r_squared: f64,
}

/// Result for a single CD dimension.
#[derive(Debug, Clone)]
pub struct DimensionResult {
    /// Cayley-Dickson dimension (16, 32, 64, ...).
    pub cd_dim: usize,
    /// Number of connected components in the ZD graph.
    pub n_components: usize,
    /// Number of nodes per component (all same for ZD graphs).
    pub nodes_per_component: usize,
    /// Full d_s(t) curve as (ln t, d_s) pairs.
    pub curve: Vec<(f64, f64)>,
    /// Extracted plateau, if found.
    pub plateau: Option<PlateauFit>,
    /// Non-zero Laplacian eigenvalues (from the first component).
    pub eigenvalues: Vec<f64>,
}

/// Result of the full dimensional flow analysis.
#[derive(Debug, Clone)]
pub struct DimensionalFlowResult {
    /// Per-dimension results.
    pub dimensions: Vec<DimensionResult>,
}

/// Trait for eigenvalue solvers, allowing future sparse backends.
pub trait EigenSolver {
    /// Return sorted eigenvalues of a symmetric matrix.
    fn eigenvalues(&self, matrix: &DMatrix<f64>) -> Vec<f64>;
}

/// Dense eigensolver via nalgebra symmetric_eigen.
pub struct DenseEigenSolver;

impl EigenSolver for DenseEigenSolver {
    fn eigenvalues(&self, matrix: &DMatrix<f64>) -> Vec<f64> {
        let eigen = matrix.clone().symmetric_eigen();
        let mut vals: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals
    }
}

// ---------------------------------------------------------------------------
// Laplacian construction
// ---------------------------------------------------------------------------

/// Build the graph Laplacian L = D - A from a petgraph UnGraph.
///
/// The Laplacian is a symmetric positive-semidefinite matrix whose smallest
/// eigenvalue is 0, with multiplicity equal to the number of connected
/// components.
pub fn build_graph_laplacian(graph: &UnGraph<(), ()>) -> DMatrix<f64> {
    let n = graph.node_count();
    let mut laplacian = DMatrix::zeros(n, n);

    for edge in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(edge).unwrap();
        let ui = u.index();
        let vi = v.index();
        laplacian[(ui, vi)] -= 1.0;
        laplacian[(vi, ui)] -= 1.0;
        laplacian[(ui, ui)] += 1.0;
        laplacian[(vi, vi)] += 1.0;
    }

    laplacian
}

/// Build the graph Laplacian directly from a `MotifComponent`.
pub fn build_laplacian_from_component(comp: &MotifComponent) -> DMatrix<f64> {
    let n = comp.nodes.len();
    let nodes: Vec<_> = comp.nodes.iter().copied().collect();
    let idx: std::collections::HashMap<_, _> =
        nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();

    let mut laplacian = DMatrix::zeros(n, n);
    for &(u, v) in &comp.edges {
        let ui = idx[&u];
        let vi = idx[&v];
        laplacian[(ui, vi)] -= 1.0;
        laplacian[(vi, ui)] -= 1.0;
        laplacian[(ui, ui)] += 1.0;
        laplacian[(vi, vi)] += 1.0;
    }
    laplacian
}

// ---------------------------------------------------------------------------
// Eigenvalue processing
// ---------------------------------------------------------------------------

/// Clamp near-zero eigenvalues to exactly 0.0.
///
/// Dense eigensolvers produce floating-point dust (~1e-15) for theoretically
/// zero eigenvalues. Without clamping, these pseudo-zeros cause P(t) to
/// spuriously decay at large t, corrupting plateau extraction.
pub fn clamp_zero_eigenvalues(eigenvalues: &mut [f64], epsilon: f64) {
    for ev in eigenvalues.iter_mut() {
        if ev.abs() < epsilon {
            *ev = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Heat kernel
// ---------------------------------------------------------------------------

/// Heat kernel return probability P(t) on the graph.
///
/// P(t) = (1/N_nz) * sum_{lambda_i > 0} exp(-lambda_i * t)
///
/// Skips zero eigenvalues (one per connected component). N_nz is the count
/// of strictly positive eigenvalues, normalizing so P(0) = 1.
pub fn heat_kernel_return_probability(eigenvalues: &[f64], t: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &ev in eigenvalues {
        if ev > 0.0 {
            sum += (-ev * t).exp();
            count += 1;
        }
    }
    if count == 0 {
        return 1.0;
    }
    sum / count as f64
}

// ---------------------------------------------------------------------------
// Geometric spacing
// ---------------------------------------------------------------------------

/// Generate geometrically spaced values: t_i = t_min * (t_max/t_min)^(i/(n-1)).
///
/// The resulting points are uniformly spaced in log-space, which is a
/// prerequisite for the Savitzky-Golay filter applied to ln(t).
pub fn geom_linspace(t_min: f64, t_max: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2, "need at least 2 points");
    assert!(t_min > 0.0, "t_min must be positive");
    assert!(t_max > t_min, "t_max must exceed t_min");
    let ratio = t_max / t_min;
    (0..n)
        .map(|i| t_min * ratio.powf(i as f64 / (n - 1) as f64))
        .collect()
}

// ---------------------------------------------------------------------------
// Savitzky-Golay filter
// ---------------------------------------------------------------------------

/// Savitzky-Golay polynomial smoothing filter.
///
/// Fits a polynomial of degree `order` to a sliding window of `window` points
/// and evaluates it at the center. Preserves plateau geometry better than
/// moving averages because it does not bias flat regions.
///
/// Requires: `window` is odd, `order < window`, input is uniformly spaced.
/// Points within half-window of the boundaries are left unsmoothed.
pub fn savitzky_golay_smooth(data: &[(f64, f64)], window: usize, order: usize) -> Vec<(f64, f64)> {
    let n = data.len();
    if n < window || window < 3 || order >= window {
        return data.to_vec();
    }
    let w = if window.is_multiple_of(2) { window + 1 } else { window };
    let half = w / 2;

    // Precompute convolution coefficients for the center point.
    // The SG filter is a linear convolution: y_smooth[i] = sum_j c_j * y[i+j-half].
    // Coefficients come from the pseudoinverse of the Vandermonde matrix.
    let coeffs = sg_coefficients(w, order);

    let mut result: Vec<(f64, f64)> = data.to_vec();
    for i in half..(n - half) {
        let mut val = 0.0;
        for (j, &c) in coeffs.iter().enumerate() {
            val += c * data[i + j - half].1;
        }
        result[i].1 = val;
    }
    result
}

/// Compute Savitzky-Golay convolution coefficients for center-point evaluation.
///
/// Solves J^T J c = J^T e_0 where J is the Vandermonde matrix and e_0
/// selects the center row of the pseudoinverse.
fn sg_coefficients(window: usize, order: usize) -> Vec<f64> {
    let half = window as i64 / 2;
    let m = order + 1;

    // Build Vandermonde-like matrix J (window x m)
    // J[i][k] = (i - half)^k
    let mut jtj = vec![0.0; m * m];
    let mut jte = vec![0.0; m];

    for i in 0..window {
        let x = (i as i64 - half) as f64;
        let mut powers = vec![1.0; m];
        for k in 1..m {
            powers[k] = powers[k - 1] * x;
        }
        for r in 0..m {
            for c in 0..m {
                jtj[r * m + c] += powers[r] * powers[c];
            }
            // e_0 selects center row: delta(i, half)
            if i as i64 == half {
                jte[r] += powers[r];
            }
        }
    }

    // Solve (J^T J) a = (J^T e_0) via Gaussian elimination
    let a = solve_linear_system(&jtj, &jte, m);

    // Convolution coefficients: c_i = sum_k a_k * (i - half)^k
    let mut coeffs = vec![0.0; window];
    for (i, coeff) in coeffs.iter_mut().enumerate() {
        let x = (i as i64 - half) as f64;
        let mut xpow = 1.0;
        for (k, &ak) in a.iter().enumerate() {
            *coeff += ak * xpow;
            if k + 1 < a.len() {
                xpow *= x;
            }
        }
    }
    coeffs
}

/// Solve a dense linear system Ax = b via Gaussian elimination with pivoting.
fn solve_linear_system(a_flat: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0; n * (n + 1)];
    for r in 0..n {
        for c in 0..n {
            aug[r * (n + 1) + c] = a_flat[r * n + c];
        }
        aug[r * (n + 1) + n] = b[r];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_row != col {
            for c in 0..=n {
                aug.swap(col * (n + 1) + c, max_row * (n + 1) + c);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        if pivot.abs() < 1e-30 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for c in col..=n {
                aug[row * (n + 1) + c] -= factor * aug[col * (n + 1) + c];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = aug[row * (n + 1) + n];
        for c in (row + 1)..n {
            sum -= aug[row * (n + 1) + c] * x[c];
        }
        let diag = aug[row * (n + 1) + row];
        x[row] = if diag.abs() > 1e-30 { sum / diag } else { 0.0 };
    }
    x
}

// ---------------------------------------------------------------------------
// Spectral dimension computation
// ---------------------------------------------------------------------------

/// Compute d_s(t) curve from Laplacian eigenvalues.
///
/// Pipeline: geom_linspace -> P(t) -> ln P vs ln t -> SG smooth -> central diff.
///
/// Returns (ln t, d_s) pairs.
pub fn graph_spectral_dimension(
    laplacian_eigenvalues: &[f64],
    t_min: f64,
    t_max: f64,
    n_points: usize,
) -> Vec<(f64, f64)> {
    let ts = geom_linspace(t_min, t_max, n_points);

    // Compute ln P(t) vs ln t
    let ln_data: Vec<(f64, f64)> = ts
        .iter()
        .map(|&t| {
            let p = heat_kernel_return_probability(laplacian_eigenvalues, t);
            (t.ln(), p.ln())
        })
        .collect();

    // Savitzky-Golay smooth
    let smoothed = savitzky_golay_smooth(&ln_data, 11, 3);

    // Central differences: d_s = -2 * d(ln P) / d(ln t)
    let n = smoothed.len();
    if n < 3 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(n - 2);
    for i in 1..(n - 1) {
        let d_ln_t = smoothed[i + 1].0 - smoothed[i - 1].0;
        let d_ln_p = smoothed[i + 1].1 - smoothed[i - 1].1;
        if d_ln_t.abs() > 1e-30 {
            let d_s = -2.0 * d_ln_p / d_ln_t;
            result.push((smoothed[i].0, d_s));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Plateau extraction
// ---------------------------------------------------------------------------

/// Extract the plateau from a d_s(t) curve.
///
/// Searches for the longest contiguous region where the derivative of d_s
/// with respect to ln(t) is below a threshold. Returns the mean d_s value
/// in that region along with quality metrics.
pub fn extract_plateau(curve: &[(f64, f64)], deriv_threshold: f64) -> Option<PlateauFit> {
    if curve.len() < 5 {
        return None;
    }

    // Compute |d(d_s)/d(ln t)| via central differences
    let n = curve.len();
    let mut derivs = vec![f64::MAX; n];
    for i in 1..(n - 1) {
        let d_ln_t = curve[i + 1].0 - curve[i - 1].0;
        let d_ds = curve[i + 1].1 - curve[i - 1].1;
        if d_ln_t.abs() > 1e-30 {
            derivs[i] = (d_ds / d_ln_t).abs();
        }
    }

    // Find longest contiguous run below threshold
    let mut best_start = 0;
    let mut best_len = 0;
    let mut run_start = 0;
    let mut run_len = 0;

    for (i, &deriv) in derivs.iter().enumerate().skip(1).take(n - 2) {
        if deriv < deriv_threshold {
            if run_len == 0 {
                run_start = i;
            }
            run_len += 1;
        } else {
            if run_len > best_len {
                best_start = run_start;
                best_len = run_len;
            }
            run_len = 0;
        }
    }
    if run_len > best_len {
        best_start = run_start;
        best_len = run_len;
    }

    if best_len < 3 {
        return None;
    }

    // Compute mean and R^2 in plateau region
    let plateau_slice = &curve[best_start..(best_start + best_len)];
    let mean_ds: f64 = plateau_slice.iter().map(|(_, ds)| ds).sum::<f64>() / best_len as f64;

    let ss_tot: f64 = plateau_slice
        .iter()
        .map(|(_, ds)| (ds - mean_ds).powi(2))
        .sum();
    // For a constant fit, SS_res = SS_tot, R^2 = 1 - SS_res/SS_tot
    // But here SS_res is the variance around the mean, which IS SS_tot.
    // R^2 for a constant model: 1 - Var(residuals) / Var(data)
    // Since residuals = data - mean, R^2 = 1 - 0 if data is constant.
    // Use R^2 = 1 - SS_res / SS_tot where SS_res = sum (d_s_i - mean)^2
    // and SS_tot = sum (d_s_i - grand_mean)^2. Since they're the same, R^2 = 0.
    // Actually for quality metric: use 1 - (std_dev / |mean|).
    let std_dev = (ss_tot / best_len as f64).sqrt();
    let r_squared = if mean_ds.abs() > 1e-15 {
        1.0 - std_dev / mean_ds.abs()
    } else {
        0.0
    };

    Some(PlateauFit {
        d_s: mean_ds,
        ln_t_min: plateau_slice.first().unwrap().0,
        ln_t_max: plateau_slice.last().unwrap().0,
        r_squared: r_squared.max(0.0),
    })
}

// ---------------------------------------------------------------------------
// Cross-assessor graph integration
// ---------------------------------------------------------------------------

/// Build the union Laplacian for all ZD graph components at a given CD dimension.
///
/// Since components are disconnected, the Laplacian is block-diagonal. The
/// eigenvalues are the union of per-component eigenvalues. All components at
/// a given dimension are isomorphic (same graph motif), so we compute
/// eigenvalues from one representative component and replicate.
pub fn zd_laplacian_eigenvalues(dim: usize) -> Vec<f64> {
    let components = motif_components_for_cross_assessors(dim);
    if components.is_empty() {
        return Vec::new();
    }

    // All components are isomorphic at a given dimension, so compute from first
    let first = &components[0];
    let laplacian = build_laplacian_from_component(first);
    let solver = DenseEigenSolver;
    let mut eigenvalues = solver.eigenvalues(&laplacian);
    clamp_zero_eigenvalues(&mut eigenvalues, 1e-12);
    eigenvalues
}

/// Compute spectral dimension analysis for a single CD dimension.
pub fn analyze_single_dimension(
    dim: usize,
    t_min: f64,
    t_max: f64,
    n_points: usize,
) -> DimensionResult {
    let components = motif_components_for_cross_assessors(dim);
    let n_components = components.len();
    let nodes_per_component = components.first().map_or(0, |c| c.nodes.len());

    let eigenvalues = zd_laplacian_eigenvalues(dim);

    let curve = graph_spectral_dimension(&eigenvalues, t_min, t_max, n_points);

    // Adaptive threshold: 0.5 for small graphs, tighter for larger ones
    let deriv_threshold = if nodes_per_component <= 10 { 0.5 } else { 0.2 };
    let plateau = extract_plateau(&curve, deriv_threshold);

    let nonzero_eigs: Vec<f64> = eigenvalues.iter().copied().filter(|&e| e > 0.0).collect();

    DimensionResult {
        cd_dim: dim,
        n_components,
        nodes_per_component,
        curve,
        plateau,
        eigenvalues: nonzero_eigs,
    }
}

/// Run the full dimensional flow analysis across multiple CD dimensions.
///
/// Computes d_s(t) for each dimension and collects plateau values.
pub fn dimensional_flow_analysis(
    dims: &[usize],
    t_min: f64,
    t_max: f64,
    n_points: usize,
) -> DimensionalFlowResult {
    let dimensions: Vec<DimensionResult> = dims
        .iter()
        .map(|&dim| analyze_single_dimension(dim, t_min, t_max, n_points))
        .collect();

    DimensionalFlowResult { dimensions }
}

// ---------------------------------------------------------------------------
// Spectral gap (algebraic connectivity) analysis
// ---------------------------------------------------------------------------

/// Result of spectral gap analysis for a single CD dimension.
#[derive(Debug, Clone)]
pub struct SpectralGapResult {
    /// Cayley-Dickson dimension (16, 32, 64, ...).
    pub cd_dim: usize,
    /// Number of connected components in the ZD graph.
    pub n_components: usize,
    /// Number of nodes per component.
    pub nodes_per_component: usize,
    /// Algebraic connectivity: first strictly positive eigenvalue of the Laplacian.
    pub lambda_2: f64,
    /// Largest Laplacian eigenvalue.
    pub lambda_max: f64,
    /// Gap ratio: lambda_2 / lambda_max. Bounded in (0, 1] for connected graphs.
    pub gap_ratio: f64,
    /// Whether the gap ratio exceeds the conservative Alon-Milman threshold 1/(2n).
    pub is_expander: bool,
    /// All strictly positive eigenvalues, sorted ascending.
    pub eigenvalues: Vec<f64>,
}

/// Extract the spectral gap (algebraic connectivity) from sorted eigenvalues.
///
/// Expects sorted ascending, zero-clamped eigenvalues from a connected graph
/// Laplacian. Returns the first strictly positive eigenvalue, or 0.0 if none.
pub fn spectral_gap(eigenvalues: &[f64]) -> f64 {
    eigenvalues
        .iter()
        .copied()
        .find(|&e| e > 0.0)
        .unwrap_or(0.0)
}

/// Spectral gap census across multiple CD dimensions.
///
/// For each dimension, computes the Laplacian eigenvalues of one representative
/// ZD graph component, extracts lambda_2 and lambda_max, and tests the expander
/// condition gap_ratio > 1/(2*nodes).
pub fn spectral_gap_census(dims: &[usize]) -> Vec<SpectralGapResult> {
    dims.iter()
        .map(|&dim| {
            let components = motif_components_for_cross_assessors(dim);
            let n_components = components.len();
            let nodes_per_component = components.first().map_or(0, |c| c.nodes.len());

            let mut eigenvalues = zd_laplacian_eigenvalues(dim);
            eigenvalues.retain(|&e| e > 0.0);

            let lambda_2 = eigenvalues.first().copied().unwrap_or(0.0);
            let lambda_max = eigenvalues.last().copied().unwrap_or(0.0);
            let gap_ratio = if lambda_max > 0.0 {
                lambda_2 / lambda_max
            } else {
                0.0
            };
            // Conservative Alon-Milman threshold for expander test
            let is_expander = nodes_per_component > 0
                && gap_ratio > 1.0 / (2.0 * nodes_per_component as f64);

            SpectralGapResult {
                cd_dim: dim,
                n_components,
                nodes_per_component,
                lambda_2,
                lambda_max,
                gap_ratio,
                is_expander,
                eigenvalues,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a complete graph K_n as UnGraph
    fn complete_graph(n: usize) -> UnGraph<(), ()> {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(nodes[i], nodes[j], ());
            }
        }
        g
    }

    // Helper: build a cycle graph C_n
    fn cycle_graph(n: usize) -> UnGraph<(), ()> {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
        for i in 0..n {
            g.add_edge(nodes[i], nodes[(i + 1) % n], ());
        }
        g
    }

    // Helper: build a path graph P_n (no wraparound)
    fn path_graph(n: usize) -> UnGraph<(), ()> {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
        for i in 0..(n - 1) {
            g.add_edge(nodes[i], nodes[i + 1], ());
        }
        g
    }

    #[test]
    fn laplacian_complete_graph_eigenvalues() {
        // K_n has Laplacian eigenvalues: 0 (once) and n (with multiplicity n-1)
        let g = complete_graph(6);
        let l = build_graph_laplacian(&g);
        let solver = DenseEigenSolver;
        let mut eigs = solver.eigenvalues(&l);
        clamp_zero_eigenvalues(&mut eigs, 1e-10);

        let zero_count = eigs.iter().filter(|&&e| e == 0.0).count();
        assert_eq!(zero_count, 1, "K_6 should have exactly 1 zero eigenvalue");

        let nonzero: Vec<f64> = eigs.iter().copied().filter(|&e| e > 0.0).collect();
        assert_eq!(nonzero.len(), 5);
        for &ev in &nonzero {
            assert!(
                (ev - 6.0).abs() < 1e-10,
                "K_6 non-zero eigenvalue should be 6, got {ev}"
            );
        }
    }

    #[test]
    fn laplacian_cycle_graph_eigenvalues() {
        // C_n has eigenvalues 2 - 2*cos(2*pi*k/n) for k = 0..n-1
        let n = 8;
        let g = cycle_graph(n);
        let l = build_graph_laplacian(&g);
        let solver = DenseEigenSolver;
        let mut eigs = solver.eigenvalues(&l);
        clamp_zero_eigenvalues(&mut eigs, 1e-10);

        let zero_count = eigs.iter().filter(|&&e| e == 0.0).count();
        assert_eq!(zero_count, 1, "C_8 should have 1 zero eigenvalue");

        // Max eigenvalue should be 4 (for k = n/2)
        let max_eig = eigs.iter().copied().fold(0.0_f64, f64::max);
        assert!(
            (max_eig - 4.0).abs() < 1e-10,
            "C_8 max eigenvalue should be 4, got {max_eig}"
        );
    }

    #[test]
    fn heat_kernel_at_zero() {
        // P(0) should be 1.0 (all exp(-0) = 1)
        let eigs = vec![0.0, 1.0, 2.0, 3.0];
        let p = heat_kernel_return_probability(&eigs, 0.0);
        assert!(
            (p - 1.0).abs() < 1e-14,
            "P(0) should be 1.0, got {p}"
        );
    }

    #[test]
    fn heat_kernel_decays() {
        // P(t) should decrease with t
        let eigs = vec![0.0, 1.0, 2.0, 3.0];
        let p1 = heat_kernel_return_probability(&eigs, 0.1);
        let p2 = heat_kernel_return_probability(&eigs, 1.0);
        let p3 = heat_kernel_return_probability(&eigs, 10.0);
        assert!(p1 > p2, "P should decrease: P(0.1)={p1} > P(1.0)={p2}");
        assert!(p2 > p3, "P should decrease: P(1.0)={p2} > P(10)={p3}");
    }

    #[test]
    fn geom_linspace_uniform_log_spacing() {
        let ts = geom_linspace(0.01, 100.0, 50);
        assert_eq!(ts.len(), 50);

        // Check constant log-step
        let ln_ts: Vec<f64> = ts.iter().map(|t| t.ln()).collect();
        let expected_step = (100.0_f64 / 0.01).ln() / 49.0;
        for i in 1..ln_ts.len() {
            let step = ln_ts[i] - ln_ts[i - 1];
            assert!(
                (step - expected_step).abs() < 1e-12,
                "log-step {i}: expected {expected_step}, got {step}"
            );
        }
    }

    #[test]
    fn sg_smooth_preserves_constant() {
        // Smoothing a constant function should return the same constant
        let val = std::f64::consts::PI;
        let data: Vec<(f64, f64)> = (0..50).map(|i| (i as f64, val)).collect();
        let smoothed = savitzky_golay_smooth(&data, 11, 3);
        for (i, (_, y)) in smoothed.iter().enumerate() {
            assert!(
                (y - val).abs() < 1e-10,
                "SG on constant should preserve value at {i}: got {y}"
            );
        }
    }

    #[test]
    fn sg_smooth_preserves_linear() {
        // SG with order >= 1 should preserve linear functions exactly
        let data: Vec<(f64, f64)> = (0..50).map(|i| (i as f64, 2.0 * i as f64 + 1.0)).collect();
        let smoothed = savitzky_golay_smooth(&data, 11, 3);
        let half = 11 / 2;
        for (i, item) in smoothed.iter().enumerate().skip(half).take(50 - 2 * half) {
            let expected = 2.0 * i as f64 + 1.0;
            assert!(
                (item.1 - expected).abs() < 1e-8,
                "SG on linear at {i}: expected {expected}, got {}",
                item.1
            );
        }
    }

    #[test]
    fn complete_graph_spectral_dimension() {
        // K_n has all non-zero eigenvalues = n, so P(t) = exp(-n*t)
        // => ln P = -n*t, d(ln P)/d(ln t) = -n*t, d_s = 2*n*t
        // This means d_s grows linearly with t (no plateau) -- which is correct
        // for a complete graph (effectively infinite-dimensional at all scales).
        let g = complete_graph(6);
        let l = build_graph_laplacian(&g);
        let solver = DenseEigenSolver;
        let mut eigs = solver.eigenvalues(&l);
        clamp_zero_eigenvalues(&mut eigs, 1e-12);

        let curve = graph_spectral_dimension(&eigs, 0.01, 10.0, 200);
        assert!(!curve.is_empty(), "should produce d_s curve");

        // d_s should be monotonically increasing for complete graph
        for window in curve.windows(2) {
            // Allow small numerical noise
            assert!(
                window[1].1 >= window[0].1 - 0.1,
                "d_s should increase for K_6"
            );
        }
    }

    #[test]
    fn cycle_graph_spectral_dimension_approaches_one() {
        // C_n at large t should have d_s -> 1 (1D manifold)
        let g = cycle_graph(100);
        let l = build_graph_laplacian(&g);
        let solver = DenseEigenSolver;
        let mut eigs = solver.eigenvalues(&l);
        clamp_zero_eigenvalues(&mut eigs, 1e-12);

        let curve = graph_spectral_dimension(&eigs, 0.1, 100.0, 300);
        // At large t, d_s should be near 1
        let large_t_values: Vec<f64> = curve
            .iter()
            .filter(|(ln_t, _)| *ln_t > 3.0) // t > ~20
            .map(|(_, ds)| *ds)
            .collect();

        if !large_t_values.is_empty() {
            let mean_ds =
                large_t_values.iter().sum::<f64>() / large_t_values.len() as f64;
            // For a cycle, d_s should approach 1 at large t
            // Allow generous tolerance for finite-size effects
            assert!(
                mean_ds < 2.0 && mean_ds > 0.2,
                "C_100 large-t d_s should be near 1, got {mean_ds}"
            );
        }
    }

    #[test]
    fn zd_graph_dim16_structure() {
        // dim=16 should have 7 components of 6 nodes each (box-kites)
        let components = motif_components_for_cross_assessors(16);
        assert_eq!(components.len(), 7, "dim=16 should have 7 components");
        for comp in &components {
            assert_eq!(comp.nodes.len(), 6, "each component should have 6 nodes");
        }

        // The Laplacian of each K_{2,2,2} (octahedral) component is well-defined
        let eigs = zd_laplacian_eigenvalues(16);
        let zero_count = eigs.iter().filter(|&&e| e == 0.0).count();
        assert_eq!(
            zero_count, 1,
            "single component should have 1 zero eigenvalue"
        );
    }

    #[test]
    fn zd_spectral_dimension_dim16() {
        let result = analyze_single_dimension(16, 0.01, 50.0, 300);
        assert_eq!(result.cd_dim, 16);
        assert_eq!(result.n_components, 7);
        assert_eq!(result.nodes_per_component, 6);
        assert!(!result.curve.is_empty(), "should produce d_s curve");
        assert!(
            !result.eigenvalues.is_empty(),
            "should have non-zero eigenvalues"
        );
    }

    #[test]
    fn path_graph_laplacian_zero_eigenvalue() {
        let g = path_graph(5);
        let l = build_graph_laplacian(&g);
        let solver = DenseEigenSolver;
        let mut eigs = solver.eigenvalues(&l);
        clamp_zero_eigenvalues(&mut eigs, 1e-10);

        let zero_count = eigs.iter().filter(|&&e| e == 0.0).count();
        assert_eq!(zero_count, 1, "path graph should have 1 zero eigenvalue");

        // All eigenvalues should be non-negative (Laplacian is PSD)
        for &ev in &eigs {
            assert!(ev >= -1e-14, "Laplacian eigenvalue should be >= 0: {ev}");
        }
    }

    // -- spectral gap tests --

    #[test]
    fn spectral_gap_complete_graph() {
        // K_n: Laplacian eigenvalues are 0 (once) and n (with multiplicity n-1).
        // So lambda_2 = n for a complete graph.
        let g = complete_graph(6);
        let l = build_graph_laplacian(&g);
        let solver = DenseEigenSolver;
        let mut eigs = solver.eigenvalues(&l);
        clamp_zero_eigenvalues(&mut eigs, 1e-10);

        let gap = spectral_gap(&eigs);
        assert!(
            (gap - 6.0).abs() < 1e-10,
            "K_6 spectral gap should be 6: {gap}"
        );
    }

    #[test]
    fn spectral_gap_cycle_graph() {
        // C_n: eigenvalues 2 - 2*cos(2*pi*k/n), k=0..n-1.
        // lambda_2 = 2 - 2*cos(2*pi/n).
        let n = 8;
        let g = cycle_graph(n);
        let l = build_graph_laplacian(&g);
        let solver = DenseEigenSolver;
        let mut eigs = solver.eigenvalues(&l);
        clamp_zero_eigenvalues(&mut eigs, 1e-10);

        let gap = spectral_gap(&eigs);
        let expected = 2.0 - 2.0 * (2.0 * std::f64::consts::PI / n as f64).cos();
        assert!(
            (gap - expected).abs() < 1e-8,
            "C_8 spectral gap should be {expected:.6}: got {gap:.6}"
        );
    }

    #[test]
    fn spectral_gap_census_dims_16_32_64() {
        let results = spectral_gap_census(&[16, 32, 64]);

        for r in &results {
            eprintln!(
                "dim={:3}: comps={:2}, nodes={:3}, lambda_2={:.4}, lambda_max={:.4}, gap_ratio={:.4}, expander={}",
                r.cd_dim, r.n_components, r.nodes_per_component,
                r.lambda_2, r.lambda_max, r.gap_ratio, r.is_expander
            );
        }

        // All should have positive lambda_2 (connected components)
        for r in &results {
            assert!(
                r.lambda_2 > 0.0,
                "dim={}: lambda_2 should be positive: {}",
                r.cd_dim, r.lambda_2
            );
            assert!(
                r.gap_ratio > 0.0 && r.gap_ratio <= 1.0,
                "dim={}: gap_ratio should be in (0,1]: {}",
                r.cd_dim, r.gap_ratio
            );
            assert!(
                r.is_expander,
                "dim={}: should be an expander (gap_ratio={:.4} > 1/(2*{})={:.4})",
                r.cd_dim, r.gap_ratio, r.nodes_per_component,
                1.0 / (2.0 * r.nodes_per_component as f64)
            );
        }

        // dim=16: K_{2,2,2} has 6 nodes, lambda_2 should be 4 (octahedral graph)
        let r16 = results.iter().find(|r| r.cd_dim == 16).unwrap();
        assert!(
            (r16.lambda_2 - 4.0).abs() < 0.1,
            "dim=16 lambda_2 should be ~4: {}",
            r16.lambda_2
        );
    }
}
