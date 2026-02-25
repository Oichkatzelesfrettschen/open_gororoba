//! Optical Glauber model for nucleus-nucleus collisions.
//!
//! Computes geometric quantities as functions of impact parameter b:
//! - T_AB(b): nuclear overlap function
//! - Npart(b): number of participants
//! - A_perp(b): transverse overlap area
//! - L(b): average path length through the medium
//! - epsilon(b): spatial eccentricity
//!
//! Uses Gauss-Legendre quadrature for 2D integrals over the transverse
//! plane and bisection for centrality-to-impact-parameter mapping.
//!
//! Reference: Arleo & Falmagne arXiv:2411.13258, Sec. II;
//!            Miller, Reygers, Sanders, Steinberg, Ann. Rev. Nucl. Part. Sci. 57 (2007) 205.

use crate::nucleus::{NucleusParams, thickness_function};
use gauss_quad::GaussLegendre;
use std::f64::consts::PI;

/// Inelastic nucleon-nucleon cross-sections (mb) at various sqrt(s).
/// 1 mb = 0.1 fm^2.
#[derive(Debug, Clone, Copy)]
pub struct SigmaNN {
    /// sqrt(s) label (GeV).
    pub sqrt_s_gev: f64,
    /// sigma_NN^inel in mb.
    pub sigma_mb: f64,
}

impl SigmaNN {
    /// sigma_NN for sqrt(s) = 200 GeV (RHIC).
    #[must_use]
    pub fn rhic_200() -> Self {
        Self {
            sqrt_s_gev: 200.0,
            sigma_mb: 42.0,
        }
    }

    /// sigma_NN for sqrt(s) = 2.76 TeV (LHC Run 1).
    #[must_use]
    pub fn lhc_2760() -> Self {
        Self {
            sqrt_s_gev: 2760.0,
            sigma_mb: 64.0,
        }
    }

    /// sigma_NN for sqrt(s) = 5.02 TeV (LHC Run 2).
    #[must_use]
    pub fn lhc_5020() -> Self {
        Self {
            sqrt_s_gev: 5020.0,
            sigma_mb: 67.6,
        }
    }

    /// sigma_NN for sqrt(s) = 5.44 TeV (Xe-Xe).
    #[must_use]
    pub fn lhc_5440() -> Self {
        Self {
            sqrt_s_gev: 5440.0,
            sigma_mb: 68.0,
        }
    }

    /// Convert sigma from mb to fm^2.
    #[must_use]
    pub fn sigma_fm2(&self) -> f64 {
        self.sigma_mb * 0.1
    }
}

/// Result of Glauber calculations for a given centrality bin.
#[derive(Debug, Clone)]
pub struct CentralityBinGeometry {
    /// Centrality bin edges (e.g. 0.0..0.05 for 0-5%).
    pub cent_lo: f64,
    pub cent_hi: f64,
    /// Impact parameter range [b_lo, b_hi] in fm.
    pub b_lo: f64,
    pub b_hi: f64,
    /// Average number of participants.
    pub n_part: f64,
    /// Transverse overlap area in fm^2.
    pub a_perp: f64,
    /// Average path length in fm.
    pub l_avg: f64,
    /// Spatial eccentricity epsilon = (<y^2>-<x^2>)/(<y^2>+<x^2>).
    pub eccentricity: f64,
}

/// Nuclear overlap function T_AB(b) for two nuclei separated by impact parameter b.
///
/// T_AB(b) = integral T_A^phys(s) * T_B^phys(|s - b|) d^2s
///
/// where T_A^phys = A * T_A^norm is the unnormalized (physical) thickness
/// function satisfying integral T_A^phys d^2s = A.
///
/// Evaluated via 2D Gauss-Legendre quadrature on a square grid
/// covering [-r_max, r_max]^2 with b_vec = (b, 0).
///
/// Units: fm^{-2}. For Pb-Pb at b=0, T_AB ~ 30 fm^{-2}.
#[must_use]
#[allow(clippy::needless_range_loop)] // 2D quadrature stencil: index used for paired (node, weight)
pub fn overlap_function(b: f64, nuc_a: &NucleusParams, nuc_b: &NucleusParams, n_gl: usize) -> f64 {
    let gl = GaussLegendre::new(n_gl).expect("Gauss-Legendre init");
    let pairs = gl.as_node_weight_pairs();

    let a_a = nuc_a.a as f64;
    let a_b = nuc_b.a as f64;

    // Integration domain: [-r_max, r_max] in both x and y
    let r_max = nuc_a.r_a.max(nuc_b.r_a + b) + 0.5;

    let mut sum = 0.0;
    for &(node_i, wx) in pairs {
        let x = r_max * node_i;
        for &(node_j, wy) in pairs {
            let y = r_max * node_j;

            let s_a = (x * x + y * y).sqrt();
            let s_b = ((x - b) * (x - b) + y * y).sqrt();

            // Physical (unnormalized) thickness: T^phys = A * T^norm
            let ta = a_a * thickness_function(s_a, nuc_a);
            let tb = a_b * thickness_function(s_b, nuc_b);

            sum += wx * wy * ta * tb;
        }
    }

    // Jacobian: mapping [-1,1]^2 -> [-r_max,r_max]^2 gives factor r_max^2
    sum * r_max * r_max
}

/// Inelastic probability at impact parameter b.
///
/// P(b) = 1 - exp(-sigma_NN * T_AB(b))
///
/// where T_AB is built from unnormalized (physical) thickness functions.
#[must_use]
pub fn inelastic_probability(
    b: f64,
    sigma: &SigmaNN,
    nuc_a: &NucleusParams,
    nuc_b: &NucleusParams,
    n_gl: usize,
) -> f64 {
    let tab = overlap_function(b, nuc_a, nuc_b, n_gl);
    let exponent = sigma.sigma_fm2() * tab;
    1.0 - (-exponent).exp()
}

/// Total inelastic cross-section sigma_inel = integral 2*pi*b * P(b) db.
///
/// Integrated from b=0 to b_max = R_A + R_B + few fm (where P(b) vanishes).
/// Uses composite Simpson's rule with n_steps points.
#[must_use]
pub fn total_inelastic_cross_section(
    sigma: &SigmaNN,
    nuc_a: &NucleusParams,
    nuc_b: &NucleusParams,
    n_gl: usize,
    n_steps: usize,
) -> f64 {
    let b_max = nuc_a.r_a + nuc_b.r_a + 3.0;
    let db = b_max / n_steps as f64;

    // Simpson's rule
    let mut sum = 0.0;
    for i in 0..=n_steps {
        let b = i as f64 * db;
        let integrand = 2.0 * PI * b * inelastic_probability(b, sigma, nuc_a, nuc_b, n_gl);
        let w = if i == 0 || i == n_steps {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * integrand;
    }
    sum * db / 3.0
}

/// Build a table of cumulative inelastic cross-section vs impact parameter.
/// Returns (b_values, cumulative_sigma) where cumulative_sigma[i] =
/// integral_0^{b_values[i]} 2*pi*b' * P(b') db'.
fn build_sigma_cdf(
    sigma: &SigmaNN,
    nuc_a: &NucleusParams,
    nuc_b: &NucleusParams,
    n_gl: usize,
    n_steps: usize,
) -> (Vec<f64>, Vec<f64>) {
    let b_max = nuc_a.r_a + nuc_b.r_a + 3.0;
    let db = b_max / n_steps as f64;

    let mut b_vals = Vec::with_capacity(n_steps + 1);
    let mut cdf = Vec::with_capacity(n_steps + 1);
    let mut running = 0.0;

    for i in 0..=n_steps {
        let b = i as f64 * db;
        b_vals.push(b);

        if i > 0 {
            // Trapezoidal step
            let b_prev = (i - 1) as f64 * db;
            let f_prev =
                2.0 * PI * b_prev * inelastic_probability(b_prev, sigma, nuc_a, nuc_b, n_gl);
            let f_curr = 2.0 * PI * b * inelastic_probability(b, sigma, nuc_a, nuc_b, n_gl);
            running += 0.5 * (f_prev + f_curr) * db;
        }
        cdf.push(running);
    }

    (b_vals, cdf)
}

/// Map centrality percentile to impact parameter by inverting the CDF.
///
/// Centrality c = sigma_inel(b) / sigma_inel_total.
/// Given c, find b such that sigma_inel(b) = c * sigma_inel_total.
fn centrality_to_b(c: f64, b_vals: &[f64], cdf: &[f64], sigma_total: f64) -> f64 {
    let target = c * sigma_total;
    // Linear interpolation search
    for i in 1..b_vals.len() {
        if cdf[i] >= target {
            let frac = (target - cdf[i - 1]) / (cdf[i] - cdf[i - 1]);
            return b_vals[i - 1] + frac * (b_vals[i] - b_vals[i - 1]);
        }
    }
    *b_vals.last().unwrap_or(&0.0)
}

/// Number of participants at impact parameter b for symmetric A+A collisions.
///
/// N_part(b) = 2 * A * integral T_A(s) * [1 - (1 - sigma_NN * T_A(|s-b|))^A] d^2s
///
/// In the optical limit:
/// N_part(b) = 2 * integral T_A(s) * [1 - exp(-sigma_NN * A * T_B(|s-b|))] d^2s
///
/// Note: This integral is over the UNNORMALIZED thickness function.
/// Our T_A is normalized to 1, so we need to multiply by A.
#[must_use]
pub fn n_part_at_b(b: f64, sigma: &SigmaNN, nuc: &NucleusParams, n_gl: usize) -> f64 {
    let gl = GaussLegendre::new(n_gl).expect("Gauss-Legendre init");
    let pairs = gl.as_node_weight_pairs();

    let r_max = nuc.r_a + b + 0.5;
    let a = nuc.a as f64;

    let mut sum = 0.0;
    for &(node_i, wx) in pairs {
        let x = r_max * node_i;
        for &(node_j, wy) in pairs {
            let y = r_max * node_j;

            let s_a = (x * x + y * y).sqrt();
            let s_b = ((x - b) * (x - b) + y * y).sqrt();

            // T_A and T_B (normalized to 1), multiply by A to get physical thickness
            let ta_phys = a * thickness_function(s_a, nuc);
            let tb_phys = a * thickness_function(s_b, nuc);

            // Contribution from nucleus A side:
            // T_A(s) * [1 - exp(-sigma_NN * T_B(|s-b|))]
            let contrib_a = ta_phys * (1.0 - (-sigma.sigma_fm2() * tb_phys).exp());

            // Contribution from nucleus B side (by symmetry for A+A):
            // T_B(|s-b|) * [1 - exp(-sigma_NN * T_A(s))]
            let contrib_b = tb_phys * (1.0 - (-sigma.sigma_fm2() * ta_phys).exp());

            sum += wx * wy * (contrib_a + contrib_b);
        }
    }

    sum * r_max * r_max
}

/// Transverse overlap area for two hard spheres of radii R_A, R_B at impact parameter b.
///
/// A_perp(b) = area of intersection of two circles of radii R_A, R_B
/// separated by distance b.
#[must_use]
pub fn overlap_area(b: f64, r_a: f64, r_b: f64) -> f64 {
    if b >= r_a + r_b {
        return 0.0;
    }
    if b <= (r_a - r_b).abs() {
        // One circle inside the other
        let r_min = r_a.min(r_b);
        return PI * r_min * r_min;
    }
    // Standard lens area formula
    let d1 = (b * b + r_a * r_a - r_b * r_b) / (2.0 * b);
    let d2 = b - d1;
    r_a * r_a * (d1 / r_a).acos() + r_b * r_b * (d2 / r_b).acos()
        - b * ((r_a * r_a - d1 * d1).max(0.0)).sqrt()
}

/// Average path length through the overlap region at impact parameter b.
///
/// L(b) = integral |s| * T_A(s) * T_B(|s-b|) d^2s / integral T_A(s) * T_B(|s-b|) d^2s
///
/// This is the T_AB-weighted average distance from the collision axis,
/// giving a measure of how far partons travel through the medium.
#[must_use]
pub fn avg_path_length(b: f64, nuc_a: &NucleusParams, nuc_b: &NucleusParams, n_gl: usize) -> f64 {
    let gl = GaussLegendre::new(n_gl).expect("Gauss-Legendre init");
    let pairs = gl.as_node_weight_pairs();

    let r_max = nuc_a.r_a.max(nuc_b.r_a + b) + 0.5;

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for &(node_i, wx) in pairs {
        let x = r_max * node_i;
        for &(node_j, wy) in pairs {
            let y = r_max * node_j;

            let s_a = (x * x + y * y).sqrt();
            let s_b = ((x - b) * (x - b) + y * y).sqrt();

            let product = thickness_function(s_a, nuc_a) * thickness_function(s_b, nuc_b);
            let w = wx * wy;

            // Average distance from the midpoint of the two centers (x=b/2, y=0)
            let dx = x - b / 2.0;
            let dist = (dx * dx + y * y).sqrt();

            num += w * dist * product;
            den += w * product;
        }
    }

    if den.abs() < 1e-30 {
        return 0.0;
    }

    // The ratio num/den gives <|s|> in fm (the r_max^2 Jacobians cancel
    // between numerator and denominator).
    num / den
}

/// Spatial eccentricity at impact parameter b.
///
/// epsilon = (<y^2> - <x^2>) / (<y^2> + <x^2>)
///
/// where moments are T_AB-weighted: <f> = integral f * T_A * T_B d^2s / integral T_A * T_B d^2s.
/// x is along the impact parameter direction (reaction plane).
#[must_use]
#[allow(clippy::needless_range_loop)] // 2D quadrature stencil: index used for paired (node, weight)
pub fn eccentricity(b: f64, nuc_a: &NucleusParams, nuc_b: &NucleusParams, n_gl: usize) -> f64 {
    let gl = GaussLegendre::new(n_gl).expect("Gauss-Legendre init");
    let pairs = gl.as_node_weight_pairs();

    let r_max = nuc_a.r_a.max(nuc_b.r_a + b) + 0.5;

    // First pass: compute centroid <x>, <y>
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_x2 = 0.0_f64;
    let mut sum_y2 = 0.0_f64;
    let mut sum_w = 0.0_f64;
    for &(node_i, wx) in pairs {
        let x = r_max * node_i;
        for &(node_j, wy) in pairs {
            let y = r_max * node_j;

            let s_a = (x * x + y * y).sqrt();
            let s_b = ((x - b) * (x - b) + y * y).sqrt();

            let product = thickness_function(s_a, nuc_a) * thickness_function(s_b, nuc_b);
            let w = wx * wy * product;

            sum_x += w * x;
            sum_y += w * y;
            sum_x2 += w * x * x;
            sum_y2 += w * y * y;
            sum_w += w;
        }
    }

    if sum_w.abs() < 1e-30 {
        return 0.0;
    }

    // Centered second moments: var(x) = <x^2> - <x>^2
    let mean_x = sum_x / sum_w;
    let mean_y = sum_y / sum_w;
    let var_x = sum_x2 / sum_w - mean_x * mean_x;
    let var_y = sum_y2 / sum_w - mean_y * mean_y;

    if (var_x + var_y).abs() < 1e-30 {
        return 0.0;
    }

    (var_y - var_x) / (var_y + var_x)
}

/// Compute Glauber geometry for standard centrality bins in a symmetric A+A collision.
///
/// Returns a vector of [`CentralityBinGeometry`] for each centrality bin.
/// The centrality bins are specified as percentile edges (e.g. `[0.0, 0.05, 0.10, ...]`).
///
/// # Arguments
/// * `cent_edges` - Centrality bin edges as fractions (0.0 to 1.0)
/// * `sigma` - Inelastic NN cross-section
/// * `nuc` - Nucleus parameters (symmetric A+A assumed)
/// * `n_gl` - Gauss-Legendre quadrature order (40-60 recommended)
/// * `n_b_steps` - Number of steps for impact parameter integration (200-500)
pub fn compute_centrality_bins(
    cent_edges: &[f64],
    sigma: &SigmaNN,
    nuc: &NucleusParams,
    n_gl: usize,
    n_b_steps: usize,
) -> Vec<CentralityBinGeometry> {
    // Build CDF of sigma_inel vs b
    let (b_vals, cdf) = build_sigma_cdf(sigma, nuc, nuc, n_gl, n_b_steps);
    let sigma_total = *cdf.last().unwrap_or(&1.0);

    let mut results = Vec::new();

    for w in cent_edges.windows(2) {
        let c_lo = w[0];
        let c_hi = w[1];

        let b_lo = centrality_to_b(c_lo, &b_vals, &cdf, sigma_total);
        let b_hi = centrality_to_b(c_hi, &b_vals, &cdf, sigma_total);

        // Average quantities over the centrality bin using several b samples
        let n_samples = 20;
        let db = (b_hi - b_lo) / n_samples as f64;

        let mut npart_sum = 0.0;
        let mut aperp_sum = 0.0;
        let mut lavg_sum = 0.0;
        let mut ecc_sum = 0.0;
        let mut weight_sum = 0.0;

        for k in 0..n_samples {
            let b_k = b_lo + (k as f64 + 0.5) * db;
            // Weight by 2*pi*b * P(b) (probability of this b)
            let p_b = inelastic_probability(b_k, sigma, nuc, nuc, n_gl);
            let weight = 2.0 * PI * b_k * p_b * db;

            npart_sum += weight * n_part_at_b(b_k, sigma, nuc, n_gl);
            aperp_sum += weight * overlap_area(b_k, nuc.r_a, nuc.r_a);
            lavg_sum += weight * avg_path_length(b_k, nuc, nuc, n_gl);
            ecc_sum += weight * eccentricity(b_k, nuc, nuc, n_gl);
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            results.push(CentralityBinGeometry {
                cent_lo: c_lo,
                cent_hi: c_hi,
                b_lo,
                b_hi,
                n_part: npart_sum / weight_sum,
                a_perp: aperp_sum / weight_sum,
                l_avg: lavg_sum / weight_sum,
                eccentricity: ecc_sum / weight_sum,
            });
        }
    }

    results
}

/// Standard centrality bin edges used in ALICE/CMS analyses.
/// Returns edges for 0-5%, 5-10%, 10-20%, 20-30%, ..., 70-80%, 80-90%.
#[must_use]
pub fn standard_centrality_edges() -> Vec<f64> {
    vec![
        0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nucleus::NucleusParams;

    #[test]
    fn test_overlap_headon() {
        // T_AB(b=0) for Pb+Pb should be maximal
        let pb = NucleusParams::pb208();
        let tab_0 = overlap_function(0.0, &pb, &pb, 48);
        // For two identical hard spheres at b=0 with physical (unnormalized) thickness:
        // T_AB(0) = A^2 * integral (T_A^norm(s))^2 d^2s
        // For Pb-Pb (A=208): T_AB(0) ~ 350 fm^{-2}
        // (equivalent to ~35 mb^{-1} in the standard convention where 1 mb = 0.1 fm^2)
        assert!(tab_0 > 0.0, "T_AB(0) should be positive, got {}", tab_0);
        assert!(
            tab_0 > 200.0 && tab_0 < 500.0,
            "T_AB(0) = {} fm^-2 (expected ~350)",
            tab_0
        );
    }

    #[test]
    fn test_overlap_decreases_with_b() {
        let pb = NucleusParams::pb208();
        let tab_0 = overlap_function(0.0, &pb, &pb, 40);
        let tab_5 = overlap_function(5.0, &pb, &pb, 40);
        let tab_10 = overlap_function(10.0, &pb, &pb, 40);
        assert!(
            tab_0 > tab_5,
            "T_AB should decrease: T_AB(0)={} > T_AB(5)={}",
            tab_0,
            tab_5
        );
        assert!(
            tab_5 > tab_10,
            "T_AB should decrease: T_AB(5)={} > T_AB(10)={}",
            tab_5,
            tab_10
        );
    }

    #[test]
    fn test_overlap_vanishes_large_b() {
        let pb = NucleusParams::pb208();
        // Beyond 2*R_A, overlap should vanish for hard spheres
        let tab = overlap_function(2.0 * pb.r_a + 1.0, &pb, &pb, 40);
        assert!(tab < 1e-2, "T_AB at large b should vanish, got {}", tab);
    }

    #[test]
    fn test_overlap_area_headon() {
        let r = 6.639;
        let area = overlap_area(0.0, r, r);
        let expected = PI * r * r;
        assert!(
            (area - expected).abs() / expected < 1e-10,
            "A_perp(b=0) = {} (expected {})",
            area,
            expected
        );
    }

    #[test]
    fn test_overlap_area_no_overlap() {
        let r = 6.639;
        let area = overlap_area(2.0 * r + 1.0, r, r);
        assert!(
            area.abs() < 1e-10,
            "A_perp at large b = {} (expected 0)",
            area
        );
    }

    #[test]
    fn test_eccentricity_headon() {
        // At b=0, eccentricity should be ~0 by symmetry
        let pb = NucleusParams::pb208();
        let ecc = eccentricity(0.0, &pb, &pb, 40);
        assert!(
            ecc.abs() < 0.05,
            "eccentricity(b=0) = {} (expected ~0)",
            ecc
        );
    }

    #[test]
    fn test_eccentricity_peripheral() {
        // At moderate b, eccentricity should be positive (y^2 > x^2)
        let pb = NucleusParams::pb208();
        let ecc = eccentricity(8.0, &pb, &pb, 40);
        assert!(ecc > 0.0, "eccentricity(b=8) = {} (expected > 0)", ecc);
        assert!(ecc < 1.0, "eccentricity should be < 1, got {}", ecc);
    }

    #[test]
    fn test_npart_headon_pbpb() {
        // Npart at b=0 for Pb-Pb should be close to 2*A = 416
        let pb = NucleusParams::pb208();
        let sigma = SigmaNN::lhc_5020();
        let npart = n_part_at_b(0.0, &sigma, &pb, 48);
        // At b=0 nearly all nucleons participate: Npart ~ 400-416
        assert!(
            npart > 380.0 && npart < 420.0,
            "Npart(b=0) = {} (expected ~400-416)",
            npart
        );
    }

    #[test]
    fn test_npart_decreases_with_b() {
        let pb = NucleusParams::pb208();
        let sigma = SigmaNN::lhc_5020();
        let npart_0 = n_part_at_b(0.0, &sigma, &pb, 40);
        let npart_5 = n_part_at_b(5.0, &sigma, &pb, 40);
        let npart_10 = n_part_at_b(10.0, &sigma, &pb, 40);
        assert!(npart_0 > npart_5, "Npart should decrease with b");
        assert!(npart_5 > npart_10, "Npart should decrease with b");
    }

    #[test]
    fn test_inelastic_probability_limits() {
        let pb = NucleusParams::pb208();
        let sigma = SigmaNN::lhc_5020();
        // At b=0, P should be ~1
        let p0 = inelastic_probability(0.0, &sigma, &pb, &pb, 40);
        assert!(p0 > 0.99, "P(b=0) = {} (expected ~1)", p0);
        // At large b, P should be ~0
        let p_far = inelastic_probability(20.0, &sigma, &pb, &pb, 40);
        assert!(p_far < 0.01, "P(b=20) = {} (expected ~0)", p_far);
    }
}
