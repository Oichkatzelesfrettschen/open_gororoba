//! Nearest-Neighbor Spacing Distribution (NNSD) analysis of ZD graph Laplacian spectra.
//!
//! Diagnoses quantum chaos vs integrability in the zero-divisor graph Laplacian
//! eigenvalue spectra using Random Matrix Theory (RMT) statistics.
//!
//! # Background
//!
//! In RMT, the distribution of nearest-neighbor eigenvalue spacings distinguishes:
//! - **Poisson** P(s) = exp(-s): integrable systems, no level repulsion.
//! - **GOE (Wigner-Dyson)** P(s) = (pi*s/2) exp(-pi*s^2/4): chaotic, level repulsion.
//! - **Brody** parameter q in `[0,1]` interpolates: q=0 Poisson, q=1 GOE.
//!
//! # Spectrum unfolding
//!
//! Raw spacings are meaningless because the macroscopic density of states varies.
//! We use Gaussian KDE unfolding, which handles the exact massive degeneracies
//! in ZD graph Laplacian spectra (near-complete multipartite structure) without
//! the Runge oscillations of polynomial staircase methods.

use crate::{
    boxkites::{MotifComponent, motif_components_for_cross_assessors},
    spectral_dimension::{
        DenseEigenSolver, EigenSolver, build_laplacian_from_component, clamp_zero_eigenvalues,
    },
};
use std::{collections::BTreeMap, f64::consts::PI};

/// Minimum number of eigenvalues required for meaningful NNSD analysis.
const MIN_EIGENVALUES: usize = 30;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Classification of spacing statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpacingVerdict {
    /// Degenerate integrable: super-Poisson statistics from exact algebraic
    /// symmetries forcing massive eigenvalue degeneracies (variance >> 1.0).
    /// Hallmark of deterministic, highly symmetric graphs (Ramanujan-like).
    DegenerateIntegrable,
    /// Poisson-like: integrable, no level repulsion (q < 0.3, variance ~ 1.0).
    Poisson,
    /// Intermediate: partial level repulsion (0.3 <= q <= 0.7).
    Intermediate,
    /// GOE Wigner-Dyson: chaotic, strong level repulsion (q > 0.7).
    GoeWigner,
}

/// Threshold above which spacing variance indicates super-Poisson statistics
/// from exact algebraic degeneracies rather than generic integrability.
const SUPER_POISSON_VARIANCE_THRESHOLD: f64 = 1.5;

impl std::fmt::Display for SpacingVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DegenerateIntegrable => write!(f, "DegenerateIntegrable"),
            Self::Poisson => write!(f, "Poisson"),
            Self::Intermediate => write!(f, "Intermediate"),
            Self::GoeWigner => write!(f, "GOE_Wigner"),
        }
    }
}

/// Full NNSD analysis result for a set of eigenvalues.
#[derive(Debug, Clone)]
pub struct NnsdResult {
    /// Cayley-Dickson dimension.
    pub cd_dim: usize,
    /// Number of positive eigenvalues analyzed.
    pub n_eigenvalues: usize,
    /// Unfolded spacings.
    pub spacings: Vec<f64>,
    /// Mean of unfolded spacings (should be ~1.0 after unfolding).
    pub mean_spacing: f64,
    /// Variance of unfolded spacings.
    pub spacing_variance: f64,
    /// Histogram bin heights (probability density).
    pub histogram: Vec<f64>,
    /// Histogram bin centers.
    pub bin_centers: Vec<f64>,
    /// Sum of squared errors vs Poisson.
    pub poisson_sse: f64,
    /// Sum of squared errors vs GOE Wigner surmise.
    pub goe_sse: f64,
    /// Kolmogorov-Smirnov statistic vs Poisson CDF.
    pub ks_poisson: f64,
    /// Kolmogorov-Smirnov statistic vs GOE CDF.
    pub ks_goe: f64,
    /// Brody parameter q in [0, 1].
    pub brody_q: f64,
    /// Classification based on Brody q.
    pub verdict: SpacingVerdict,
}

/// NNSD result for a single motif class (components grouped by edge count).
#[derive(Debug, Clone)]
pub struct MotifClassNnsd {
    /// Cayley-Dickson dimension.
    pub cd_dim: usize,
    /// Number of edges in this motif class (topology identifier).
    pub motif_edge_count: usize,
    /// Number of components in this class.
    pub n_components_in_class: usize,
    /// NNSD analysis result (None if too few eigenvalues).
    pub nnsd: Option<NnsdResult>,
}

// ---------------------------------------------------------------------------
// Error function (Abramowitz & Stegun 7.1.26)
// ---------------------------------------------------------------------------

/// Error function erf(x) via Abramowitz & Stegun formula 7.1.26.
///
/// Maximum absolute error: 1.5e-7.
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ---------------------------------------------------------------------------
// Log-Gamma (Lanczos approximation)
// ---------------------------------------------------------------------------

/// Natural logarithm of the Gamma function via Lanczos approximation (g=7).
///
/// Accurate to ~15 digits for positive real arguments.
pub fn ln_gamma(z: f64) -> f64 {
    // Lanczos coefficients (g=7, n=9)
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    const G: f64 = 7.0;

    let z = z - 1.0;
    let mut x = C[0];
    for (i, &c) in C.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

// ---------------------------------------------------------------------------
// Reference distributions
// ---------------------------------------------------------------------------

/// Poisson spacing distribution: P(s) = exp(-s).
pub fn poisson_spacing(s: f64) -> f64 {
    (-s).exp()
}

/// GOE Wigner surmise: P(s) = (pi*s/2) * exp(-pi*s^2/4).
pub fn wigner_surmise_goe(s: f64) -> f64 {
    (PI * s / 2.0) * (-PI * s * s / 4.0).exp()
}

/// Brody distribution: P(s, q) = (q+1)*b*s^q * exp(-b*s^{q+1}).
///
/// where b = Gamma((q+2)/(q+1))^{q+1}.
/// At q=0 this is Poisson, at q=1 it approximates the GOE Wigner surmise.
pub fn brody_distribution(s: f64, q: f64) -> f64 {
    let ratio = (q + 2.0) / (q + 1.0);
    let ln_g = ln_gamma(ratio);
    let b = (ln_g * (q + 1.0)).exp();
    (q + 1.0) * b * s.powf(q) * (-b * s.powf(q + 1.0)).exp()
}

/// Variance of the Brody distribution as a function of q.
///
/// var(q) = Gamma((q+3)/(q+1)) / [Gamma((q+2)/(q+1))]^2 - 1
///
/// Monotonically decreasing: 1.0 at q=0 (Poisson), ~0.273 at q=1 (GOE).
pub fn brody_variance(q: f64) -> f64 {
    let r1 = (q + 3.0) / (q + 1.0);
    let r2 = (q + 2.0) / (q + 1.0);
    let ln_g1 = ln_gamma(r1);
    let ln_g2 = ln_gamma(r2);
    (ln_g1 - 2.0 * ln_g2).exp() - 1.0
}

// ---------------------------------------------------------------------------
// Spectrum unfolding (Gaussian KDE)
// ---------------------------------------------------------------------------

/// Unfold an eigenvalue spectrum via Gaussian KDE.
///
/// Maps eigenvalues through a smoothed empirical CDF:
///   e_i = (1/2N) * sum_j [1 + erf((lambda_i - lambda_j) / (sigma * sqrt(2)))]
///
/// The unfolded values have approximately unit mean spacing.
/// O(N^2) complexity, acceptable for N <= 253 (dim=512).
pub fn unfold_spectrum_kde(eigenvalues: &[f64], sigma: f64) -> Vec<f64> {
    let n = eigenvalues.len();
    if n == 0 || sigma <= 0.0 {
        return Vec::new();
    }
    let inv_2n = 1.0 / (2.0 * n as f64);
    let inv_sqrt2_sigma = 1.0 / (sigma * std::f64::consts::SQRT_2);

    eigenvalues
        .iter()
        .map(|&lam_i| {
            let sum: f64 = eigenvalues
                .iter()
                .map(|&lam_j| 1.0 + erf((lam_i - lam_j) * inv_sqrt2_sigma))
                .sum();
            sum * inv_2n
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Histogram and scoring
// ---------------------------------------------------------------------------

/// Build a normalized spacing histogram.
///
/// Range: [0, 4*mean_spacing], `n_bins` bins. Normalized so the integral
/// (sum * bin_width) approximates 1.0.
///
/// Returns (bin_centers, probability_density).
pub fn build_spacing_histogram(spacings: &[f64], n_bins: usize) -> (Vec<f64>, Vec<f64>) {
    if spacings.is_empty() || n_bins == 0 {
        return (Vec::new(), Vec::new());
    }
    let mean = spacings.iter().sum::<f64>() / spacings.len() as f64;
    let s_max = 4.0 * mean.max(1e-15);
    let bin_width = s_max / n_bins as f64;

    let mut counts = vec![0usize; n_bins];
    for &s in spacings {
        let idx = ((s / bin_width) as usize).min(n_bins - 1);
        counts[idx] += 1;
    }

    let norm = spacings.len() as f64 * bin_width;
    let centers: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * bin_width).collect();
    let density: Vec<f64> = counts.iter().map(|&c| c as f64 / norm).collect();
    (centers, density)
}

/// Sum of squared errors between a histogram and a reference distribution.
pub fn histogram_sse(centers: &[f64], histogram: &[f64], reference_fn: impl Fn(f64) -> f64) -> f64 {
    centers
        .iter()
        .zip(histogram.iter())
        .map(|(&s, &p)| {
            let ref_p = reference_fn(s);
            (p - ref_p).powi(2)
        })
        .sum()
}

/// Kolmogorov-Smirnov statistic: max |F_empirical(s) - F_reference(s)|.
pub fn ks_statistic(data: &[f64], reference_cdf: impl Fn(f64) -> f64) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut max_d = 0.0_f64;
    for (i, &s) in sorted.iter().enumerate() {
        let f_emp = (i + 1) as f64 / n as f64;
        let f_ref = reference_cdf(s);
        max_d = max_d.max((f_emp - f_ref).abs());
        // Also check the left-continuous empirical CDF
        let f_emp_left = i as f64 / n as f64;
        max_d = max_d.max((f_emp_left - f_ref).abs());
    }
    max_d
}

// ---------------------------------------------------------------------------
// Brody fitting (method of moments)
// ---------------------------------------------------------------------------

/// Fit the Brody parameter q by matching observed variance.
///
/// Uses bisection on q in [0, 1] to match the observed variance of spacings
/// to `brody_variance(q)`. Since `brody_variance` is monotonically decreasing,
/// bisection is guaranteed to converge. 64 iterations ~ 1e-19 precision.
pub fn fit_brody_parameter(spacings: &[f64]) -> f64 {
    if spacings.len() < 2 {
        return 0.0;
    }
    let mean = spacings.iter().sum::<f64>() / spacings.len() as f64;
    if mean <= 0.0 {
        return 0.0;
    }
    let variance = spacings
        .iter()
        .map(|&s| {
            let normed = s / mean;
            (normed - 1.0).powi(2)
        })
        .sum::<f64>()
        / spacings.len() as f64;

    // brody_variance(0) ~ 1.0, brody_variance(1) ~ 0.273
    if variance >= brody_variance(0.0) {
        return 0.0;
    }
    if variance <= brody_variance(1.0) {
        return 1.0;
    }

    // Bisection: find q such that brody_variance(q) == variance
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        if brody_variance(mid) > variance {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Classify spacing statistics from Brody parameter and normalized variance.
///
/// Super-Poisson (variance >> 1.0) indicates exact algebraic degeneracies
/// producing many zero spacings with intermittent large gaps. This is
/// distinct from generic Poisson integrability (variance ~ 1.0).
fn classify_verdict(q: f64, normalized_variance: f64) -> SpacingVerdict {
    if q < 0.3 && normalized_variance > SUPER_POISSON_VARIANCE_THRESHOLD {
        SpacingVerdict::DegenerateIntegrable
    } else if q < 0.3 {
        SpacingVerdict::Poisson
    } else if q > 0.7 {
        SpacingVerdict::GoeWigner
    } else {
        SpacingVerdict::Intermediate
    }
}

/// Full NNSD analysis pipeline for a set of eigenvalues.
///
/// Returns `None` if there are fewer than `MIN_EIGENVALUES` positive eigenvalues.
///
/// Steps:
/// 1. Filter to positive eigenvalues.
/// 2. Unfold via Gaussian KDE (sigma = 0.1 * spectral range).
/// 3. Compute spacings from unfolded values.
/// 4. Build histogram and compute SSE vs Poisson and GOE.
/// 5. Fit Brody parameter via method-of-moments.
/// 6. Compute KS statistics vs Poisson and GOE CDFs.
/// 7. Classify verdict based on Brody q.
pub fn nnsd_analysis(eigenvalues: &[f64], cd_dim: usize, n_bins: usize) -> Option<NnsdResult> {
    // Filter to positive eigenvalues, sorted ascending
    let mut positive: Vec<f64> = eigenvalues.iter().copied().filter(|&e| e > 0.0).collect();
    positive.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if positive.len() < MIN_EIGENVALUES {
        return None;
    }

    // KDE unfolding
    let range = positive.last().unwrap() - positive.first().unwrap();
    let sigma = 0.1 * range.max(1e-15);
    let unfolded = unfold_spectrum_kde(&positive, sigma);

    // Spacings from unfolded values
    let spacings: Vec<f64> = unfolded.windows(2).map(|w| w[1] - w[0]).collect();
    if spacings.is_empty() {
        return None;
    }

    let mean_spacing = spacings.iter().sum::<f64>() / spacings.len() as f64;
    let spacing_variance = if mean_spacing > 0.0 {
        spacings
            .iter()
            .map(|&s| (s / mean_spacing - 1.0).powi(2))
            .sum::<f64>()
            / spacings.len() as f64
    } else {
        0.0
    };

    // Normalize spacings by mean for histogram/fitting
    let normed: Vec<f64> = if mean_spacing > 0.0 {
        spacings.iter().map(|&s| s / mean_spacing).collect()
    } else {
        spacings.clone()
    };

    // Histogram
    let (bin_centers, histogram) = build_spacing_histogram(&normed, n_bins);

    // SSE vs reference distributions
    let poisson_sse = histogram_sse(&bin_centers, &histogram, poisson_spacing);
    let goe_sse = histogram_sse(&bin_centers, &histogram, wigner_surmise_goe);

    // Brody fit
    let brody_q = fit_brody_parameter(&spacings);

    // KS statistics
    let ks_poisson = ks_statistic(&normed, |s| 1.0 - (-s).exp());
    let ks_goe = ks_statistic(&normed, |s| 1.0 - (-PI * s * s / 4.0).exp());

    let verdict = classify_verdict(brody_q, spacing_variance);

    Some(NnsdResult {
        cd_dim,
        n_eigenvalues: positive.len(),
        spacings: normed,
        mean_spacing,
        spacing_variance,
        histogram,
        bin_centers,
        poisson_sse,
        goe_sse,
        ks_poisson,
        ks_goe,
        brody_q,
        verdict,
    })
}

/// NNSD census across multiple CD dimensions.
///
/// Computes eigenvalues from one representative ZD graph component per dimension
/// and runs the full NNSD pipeline. Skips dimensions where the component is
/// too small (fewer than MIN_EIGENVALUES positive eigenvalues).
pub fn nnsd_census(dims: &[usize]) -> Vec<NnsdResult> {
    dims.iter()
        .filter_map(|&dim| {
            let eigenvalues = eigenvalues_for_dim(dim);
            nnsd_analysis(&eigenvalues, dim, 20)
        })
        .collect()
}

/// NNSD analysis grouped by motif class (edge count) at a given CD dimension.
///
/// For dim >= 128, ZD graph components fall into multiple non-isomorphic motif
/// classes distinguished by edge count. This function groups components by
/// edge count, takes one representative per class, and runs NNSD analysis on
/// each class independently.
pub fn nnsd_by_motif_class(dim: usize) -> Vec<MotifClassNnsd> {
    let components = motif_components_for_cross_assessors(dim);

    // Group by edge count (proxy for isomorphism class)
    let mut classes: BTreeMap<usize, Vec<&MotifComponent>> = BTreeMap::new();
    for comp in &components {
        classes.entry(comp.edges.len()).or_default().push(comp);
    }

    classes
        .into_iter()
        .map(|(edge_count, comps)| {
            let n_components_in_class = comps.len();

            // Compute eigenvalues from one representative
            let rep = comps[0];
            let laplacian = build_laplacian_from_component(rep);
            let solver = DenseEigenSolver;
            let mut eigs = solver.eigenvalues(&laplacian);
            clamp_zero_eigenvalues(&mut eigs, 1e-12);

            let nnsd = nnsd_analysis(&eigs, dim, 20);

            MotifClassNnsd {
                cd_dim: dim,
                motif_edge_count: edge_count,
                n_components_in_class,
                nnsd,
            }
        })
        .collect()
}

/// Compute positive eigenvalues for a single ZD graph component at given dim.
fn eigenvalues_for_dim(dim: usize) -> Vec<f64> {
    let components = motif_components_for_cross_assessors(dim);
    if components.is_empty() {
        return Vec::new();
    }
    let laplacian = build_laplacian_from_component(&components[0]);
    let solver = DenseEigenSolver;
    let mut eigs = solver.eigenvalues(&laplacian);
    clamp_zero_eigenvalues(&mut eigs, 1e-12);
    eigs
}

// ---------------------------------------------------------------------------
// Motif graph export
// ---------------------------------------------------------------------------

/// Graph-theoretic invariants for a motif component.
#[derive(Debug, Clone)]
pub struct MotifInvariants {
    /// Cayley-Dickson dimension.
    pub cd_dim: usize,
    /// Number of vertices.
    pub n_vertices: usize,
    /// Number of edges.
    pub n_edges: usize,
    /// Sorted degree sequence.
    pub degree_sequence: Vec<usize>,
    /// Whether the graph is regular (all degrees equal).
    pub is_regular: bool,
    /// Degree if regular, 0 otherwise.
    pub regularity_degree: usize,
    /// Diameter (longest shortest path).
    pub diameter: usize,
    /// Girth (shortest cycle length, usize::MAX if acyclic).
    pub girth: usize,
    /// Number of triangles.
    pub triangle_count: usize,
    /// Number of parts if K_{2,2,...,2} multipartite, 0 otherwise.
    pub k2_parts: usize,
    /// Adjacency spectrum (sorted descending).
    pub spectrum: Vec<f64>,
    /// Number of components sharing this motif class.
    pub n_components_in_class: usize,
}

/// Compute graph-theoretic invariants for a motif component.
pub fn motif_invariants(comp: &MotifComponent, n_components_in_class: usize) -> MotifInvariants {
    let degree_sequence = comp.degree_sequence();
    let is_regular =
        !degree_sequence.is_empty() && degree_sequence.first() == degree_sequence.last();
    let regularity_degree = if is_regular {
        degree_sequence.first().copied().unwrap_or(0)
    } else {
        0
    };

    MotifInvariants {
        cd_dim: comp.dim,
        n_vertices: comp.nodes.len(),
        n_edges: comp.edges.len(),
        degree_sequence,
        is_regular,
        regularity_degree,
        diameter: comp.diameter(),
        girth: comp.girth(),
        triangle_count: comp.triangle_count(),
        k2_parts: comp.k2_multipartite_part_count(),
        spectrum: comp.spectrum(),
        n_components_in_class,
    }
}

/// Export a motif component as a Graphviz .dot string.
///
/// Nodes are labeled with local indices 0..N-1 (BTreeSet ordering of the
/// original CrossPair nodes). Graph-level attributes encode the full set of
/// isomorphism invariants for identification.
pub fn export_motif_graphviz(comp: &MotifComponent, inv: &MotifInvariants) -> String {
    let nodes: Vec<_> = comp.nodes.iter().copied().collect();
    let idx: std::collections::HashMap<(usize, usize), usize> =
        nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();

    let mut dot = String::new();
    dot.push_str("graph motif {\n");

    // Graph-level attributes as comments and label
    dot.push_str(&format!(
        "  // CD dim={}, edges={}, components_in_class={}\n",
        inv.cd_dim, inv.n_edges, inv.n_components_in_class
    ));
    dot.push_str(&format!(
        "  // vertices={}, diameter={}, girth={}, triangles={}\n",
        inv.n_vertices,
        inv.diameter,
        if inv.girth == usize::MAX {
            "inf".to_string()
        } else {
            inv.girth.to_string()
        },
        inv.triangle_count
    ));
    if inv.is_regular {
        dot.push_str(&format!("  // regular: degree={}\n", inv.regularity_degree));
    } else {
        dot.push_str(&format!(
            "  // degree_sequence: {:?}\n",
            inv.degree_sequence
        ));
    }
    if inv.k2_parts > 0 {
        dot.push_str(&format!(
            "  // complete multipartite K_{{2,2,...,2}} with {} parts\n",
            inv.k2_parts
        ));
    }

    // Spectrum as comment (truncated to 3 decimal places)
    let spec_str: Vec<String> = inv.spectrum.iter().map(|&v| format!("{v:.3}")).collect();
    dot.push_str(&format!(
        "  // adjacency_spectrum: [{}]\n",
        spec_str.join(", ")
    ));

    // Graph label with key invariants
    dot.push_str(&format!(
        "  label=\"dim={} edges={} V={} diam={} girth={} tri={}{}\"\n",
        inv.cd_dim,
        inv.n_edges,
        inv.n_vertices,
        inv.diameter,
        if inv.girth == usize::MAX {
            "inf".to_string()
        } else {
            inv.girth.to_string()
        },
        inv.triangle_count,
        if inv.k2_parts > 0 {
            format!(" K_{{2x{}}}", inv.k2_parts)
        } else {
            String::new()
        },
    ));
    dot.push_str("  labelloc=t\n");
    dot.push_str("  node [shape=circle, width=0.3, fixedsize=true]\n");
    dot.push_str('\n'.to_string().as_str());

    // Nodes with both local index and original CrossPair as tooltip
    for (i, &(a, b)) in nodes.iter().enumerate() {
        dot.push_str(&format!("  {i} [tooltip=\"({a},{b})\"]\n"));
    }
    dot.push_str('\n'.to_string().as_str());

    // Edges
    for &(u, v) in &comp.edges {
        let ui = idx[&u];
        let vi = idx[&v];
        dot.push_str(&format!("  {ui} -- {vi}\n"));
    }

    dot.push_str("}\n");
    dot
}

/// Export all distinct motif classes at a given dimension as .dot files.
///
/// Returns a vector of (edge_count, dot_string, invariants) triples, one per
/// motif class. Components are grouped by edge count, and one representative
/// per class is exported.
pub fn export_motif_classes(dim: usize) -> Vec<(usize, String, MotifInvariants)> {
    let components = motif_components_for_cross_assessors(dim);

    let mut classes: BTreeMap<usize, Vec<&MotifComponent>> = BTreeMap::new();
    for comp in &components {
        classes.entry(comp.edges.len()).or_default().push(comp);
    }

    classes
        .into_iter()
        .map(|(edge_count, comps)| {
            let n_in_class = comps.len();
            let rep = comps[0];
            let inv = motif_invariants(rep, n_in_class);
            let dot = export_motif_graphviz(rep, &inv);
            (edge_count, dot, inv)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wigner_surmise_normalizes() {
        // Trapezoidal integral of wigner_surmise_goe over [0, 8] should be ~1.0
        let n = 10_000;
        let s_max = 8.0;
        let ds = s_max / n as f64;
        let integral: f64 = (0..n)
            .map(|i| {
                let s = (i as f64 + 0.5) * ds;
                wigner_surmise_goe(s) * ds
            })
            .sum();
        assert!(
            (integral - 1.0).abs() < 0.001,
            "GOE Wigner integral should be ~1.0, got {integral}"
        );
    }

    #[test]
    fn poisson_normalizes() {
        // Integral of exp(-s) from 0 to 20 should be ~1.0
        let n = 10_000;
        let s_max = 20.0;
        let ds = s_max / n as f64;
        let integral: f64 = (0..n)
            .map(|i| {
                let s = (i as f64 + 0.5) * ds;
                poisson_spacing(s) * ds
            })
            .sum();
        assert!(
            (integral - 1.0).abs() < 1e-6,
            "Poisson integral should be ~1.0, got {integral}"
        );
    }

    #[test]
    fn brody_q0_is_poisson() {
        for &s in &[0.5, 1.0, 2.0, 3.0] {
            let brody = brody_distribution(s, 0.0);
            let poisson = poisson_spacing(s);
            assert!(
                (brody - poisson).abs() < 1e-10,
                "Brody(s={s}, q=0) = {brody} should equal Poisson = {poisson}"
            );
        }
    }

    #[test]
    fn brody_q1_approximates_goe() {
        // Brody at q=1 is close to but not identical to the Wigner surmise
        for &s in &[0.5, 1.0, 1.5, 2.0] {
            let brody = brody_distribution(s, 1.0);
            let goe = wigner_surmise_goe(s);
            assert!(
                (brody - goe).abs() < 0.05,
                "Brody(s={s}, q=1) = {brody:.4} should approximate GOE = {goe:.4}"
            );
        }
    }

    #[test]
    fn ln_gamma_known_values() {
        // Gamma(1) = 1 => ln_gamma(1) = 0
        assert!(
            ln_gamma(1.0).abs() < 1e-12,
            "ln_gamma(1) should be 0: {}",
            ln_gamma(1.0)
        );
        // Gamma(2) = 1 => ln_gamma(2) = 0
        assert!(
            ln_gamma(2.0).abs() < 1e-12,
            "ln_gamma(2) should be 0: {}",
            ln_gamma(2.0)
        );
        // Gamma(5) = 24 => ln_gamma(5) = ln(24)
        assert!(
            (ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10,
            "ln_gamma(5) should be ln(24): {}",
            ln_gamma(5.0)
        );
        // Gamma(0.5) = sqrt(pi) => ln_gamma(0.5) = 0.5*ln(pi)
        assert!(
            (ln_gamma(0.5) - 0.5 * PI.ln()).abs() < 1e-10,
            "ln_gamma(0.5) should be 0.5*ln(pi): {}",
            ln_gamma(0.5)
        );
    }

    #[test]
    fn brody_variance_monotone() {
        let v0 = brody_variance(0.0);
        let v1 = brody_variance(1.0);
        assert!(
            (v0 - 1.0).abs() < 1e-8,
            "brody_variance(0) should be ~1.0: {v0}"
        );
        assert!(
            (v1 - 0.273).abs() < 0.01,
            "brody_variance(1) should be ~0.273: {v1}"
        );

        // Strictly decreasing
        let mut prev = v0;
        for i in 1..=10 {
            let q = i as f64 * 0.1;
            let v = brody_variance(q);
            assert!(
                v < prev,
                "brody_variance should decrease: v({:.1}) = {v} >= v({:.1}) = {prev}",
                q,
                q - 0.1
            );
            prev = v;
        }
    }

    #[test]
    fn erf_known_values() {
        assert!(erf(0.0).abs() < 1e-10, "erf(0) should be 0: {}", erf(0.0));
        assert!(
            (erf(1.0) - 0.8427007929).abs() < 2e-7,
            "erf(1) should be ~0.8427: {}",
            erf(1.0)
        );
        assert!(
            (erf(-1.0) + 0.8427007929).abs() < 2e-7,
            "erf(-1) should be ~-0.8427: {}",
            erf(-1.0)
        );
        // erf(3) = 0.99998 (not exactly 1.0)
        assert!(erf(3.0) > 0.9999, "erf(3) should be > 0.9999: {}", erf(3.0));
    }

    #[test]
    fn kde_unfold_uniform_spectrum() {
        // Evenly spaced eigenvalues should produce nearly uniform spacings
        // in the interior (edges have KDE boundary effects)
        let eigs: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let sigma = 0.1 * (eigs.last().unwrap() - eigs.first().unwrap());
        let unfolded = unfold_spectrum_kde(&eigs, sigma);

        let spacings: Vec<f64> = unfolded.windows(2).map(|w| w[1] - w[0]).collect();
        // Trim boundary-affected spacings (first 5 and last 5)
        let interior = &spacings[5..spacings.len() - 5];
        let mean = interior.iter().sum::<f64>() / interior.len() as f64;

        for (i, &s) in interior.iter().enumerate() {
            assert!(
                (s - mean).abs() < 0.3 * mean,
                "uniform spectrum interior spacing[{i}] = {s} too far from mean {mean}"
            );
        }
    }

    #[test]
    fn kde_unfold_degenerate_graceful() {
        // All-equal eigenvalues: should produce all-zero spacings, no panic/NaN
        let eigs = vec![3.0; 20];
        let sigma = 0.1;
        let unfolded = unfold_spectrum_kde(&eigs, sigma);

        let spacings: Vec<f64> = unfolded.windows(2).map(|w| w[1] - w[0]).collect();
        for &s in &spacings {
            assert!(!s.is_nan(), "degenerate spacings should not be NaN");
            assert!(s.abs() < 1e-10, "degenerate spacings should be ~0: {s}");
        }
    }

    #[test]
    fn fit_brody_poisson_sample() {
        // Generate 1000 exponential(1) samples (Poisson spacing distribution)
        // Using inverse CDF: s = -ln(1 - u) with deterministic u
        let n = 1000;
        let spacings: Vec<f64> = (0..n)
            .map(|i| {
                let u = (i as f64 + 0.5) / n as f64;
                -(1.0 - u).ln()
            })
            .collect();

        let q = fit_brody_parameter(&spacings);
        assert!(q < 0.15, "Poisson sample should give q < 0.15, got {q}");
    }

    #[test]
    fn fit_brody_goe_sample() {
        // Generate Wigner-distributed spacings via inverse CDF
        // F(s) = 1 - exp(-pi*s^2/4) => s = sqrt(-4*ln(1-u)/pi)
        let n = 1000;
        let spacings: Vec<f64> = (0..n)
            .map(|i| {
                let u = (i as f64 + 0.5) / n as f64;
                (-4.0 * (1.0 - u).ln() / PI).sqrt()
            })
            .collect();

        let q = fit_brody_parameter(&spacings);
        assert!(q > 0.85, "GOE sample should give q > 0.85, got {q}");
    }
}
