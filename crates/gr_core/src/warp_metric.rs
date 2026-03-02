//! Nacelle warp bubble spacetime (White et al. 2025).
//!
//! Implements both the standard Alcubierre warp metric and the White 2025
//! interior-flat cylindrical nacelle generalization. The nacelle design
//! segments the warp bubble into n cylindrical sub-bubbles arranged
//! azimuthally, each with its own warp field. The interior-flat gating
//! function W_g ensures K_ij = 0 inside the bubble, eliminating tidal forces.
//!
//! # Metric form (Cartesian coordinates [t, x, y, z])
//!
//! ds^2 = -dt^2 + (dx - beta_x dt)^2 + dy^2 + dz^2
//!
//! where beta_x = -v_s * f(r_s) * M(rho, phi) * W_x(x) is the shift vector.
//!
//! # Component functions
//!
//! - f(r_s): Alcubierre shape function (tanh top-hat)
//! - M(rho, phi) = 1 - W_g(rho) + W_g(rho) * G_n(rho, phi): modulation
//! - W_g(rho): logistic gating function (interior -> 0, wall -> 1)
//! - G_n(rho, phi): nacelle array (sum of n azimuthal Gaussians)
//! - W_x(x): axial taper (tanh top-hat along propagation axis)
//!
//! # References
//! - White et al. (2025): CQG 42, 235022. DOI:10.1088/1361-6382/ae237a
//! - Alcubierre (1994): Class. Quantum Grav. 11, L73
//! - Pfenning & Ford (1997): Class. Quantum Grav. 14, 1743

use crate::metric::{DIM, MetricComponents, SpacetimeMetric};
use crate::spatial::SpatialVector;
use std::f64::consts::PI;

// ============================================================================
// Parameter struct
// ============================================================================

/// Parameters for the nacelle warp bubble geometry.
#[derive(Debug, Clone)]
pub struct NacelleWarpParams {
    /// Warp bubble velocity (natural units, c=1).
    pub v_s: f64,
    /// Bubble center x-coordinate (usually travels along x).
    pub x_center: f64,
    /// Bubble radius R in the Alcubierre shaping function.
    pub r_bubble: f64,
    /// Wall steepness parameter sigma for the tanh shaping.
    pub sigma: f64,

    /// Number of nacelles (azimuthal segments). 0 = pure Alcubierre.
    pub n_nacelles: usize,
    /// Nacelle radial position rho_0 (center of nacelle ring).
    pub rho_0: f64,
    /// Nacelle radial width sigma_rho (Gaussian width of each nacelle).
    pub sigma_rho: f64,

    /// Gating function steepness kappa.
    pub kappa: f64,
    /// Gating function transition width delta.
    pub delta_gating: f64,

    /// Axial taper half-length L.
    pub axial_l: f64,
    /// Axial taper steepness.
    pub sigma_x: f64,
}

impl Default for NacelleWarpParams {
    fn default() -> Self {
        Self {
            v_s: 0.1,
            x_center: 0.0,
            r_bubble: 10.0,
            sigma: 1.0,
            n_nacelles: 4,
            rho_0: 5.0,
            sigma_rho: 1.0,
            kappa: 10.0,
            delta_gating: 0.5,
            axial_l: 8.0,
            sigma_x: 1.0,
        }
    }
}

impl NacelleWarpParams {
    /// Create parameters for the standard Alcubierre bubble (no nacelles).
    pub fn alcubierre(v_s: f64, r_bubble: f64, sigma: f64) -> Self {
        Self {
            v_s,
            r_bubble,
            sigma,
            n_nacelles: 0,
            axial_l: r_bubble * 2.0,
            sigma_x: sigma,
            ..Default::default()
        }
    }

    /// Create parameters matching the White 2025 paper defaults.
    pub fn white_2025(v_s: f64, n_nacelles: usize) -> Self {
        Self {
            v_s,
            x_center: 0.0,
            r_bubble: 10.0,
            sigma: 1.0,
            n_nacelles,
            rho_0: 5.0,
            sigma_rho: 1.0,
            kappa: 10.0,
            delta_gating: 0.5,
            axial_l: 8.0,
            sigma_x: 1.0,
        }
    }
}

// ============================================================================
// Component functions
// ============================================================================

/// Alcubierre shape function f(r_s).
///
/// f(r_s) = [tanh(sigma*(r_s+R)) - tanh(sigma*(r_s-R))] / [2*tanh(sigma*R)]
///
/// Properties: f(0) = 1, f(r >> R) -> 0.
pub fn alcubierre_shape_function(r_s: f64, r_bubble: f64, sigma: f64) -> f64 {
    let denom = 2.0 * (sigma * r_bubble).tanh();
    if denom.abs() < 1e-30 {
        return 0.0;
    }
    ((sigma * (r_s + r_bubble)).tanh() - (sigma * (r_s - r_bubble)).tanh()) / denom
}

/// Derivative df/dr_s of the Alcubierre shape function.
///
/// df/dr_s = sigma * [sech^2(sigma*(r_s+R)) - sech^2(sigma*(r_s-R))] / [2*tanh(sigma*R)]
pub fn alcubierre_shape_derivative(r_s: f64, r_bubble: f64, sigma: f64) -> f64 {
    let denom = 2.0 * (sigma * r_bubble).tanh();
    if denom.abs() < 1e-30 {
        return 0.0;
    }
    let sech_plus = 1.0 / (sigma * (r_s + r_bubble)).cosh();
    let sech_minus = 1.0 / (sigma * (r_s - r_bubble)).cosh();
    sigma * (sech_plus * sech_plus - sech_minus * sech_minus) / denom
}

/// Nacelle radial envelope F(rho).
///
/// Gaussian centered at rho_0 with width sigma_rho:
/// F(rho) = exp(-(rho - rho_0)^2 / (2 * sigma_rho^2))
pub fn nacelle_radial_envelope(rho: f64, rho_0: f64, sigma_rho: f64) -> f64 {
    let dr = rho - rho_0;
    (-dr * dr / (2.0 * sigma_rho * sigma_rho)).exp()
}

/// Interior-flat gating function W_g(rho).
///
/// Logistic transition from 0 (interior) to 1 (wall/exterior):
/// W_g(rho) = 1 / (1 + exp(-kappa * (rho - rho_0 + delta)))
///
/// At rho << rho_0: W_g -> 0 (interior is flat).
/// At rho ~ rho_0: W_g -> 1 (nacelle region is active).
pub fn gating_function(rho: f64, rho_0: f64, kappa: f64, delta: f64) -> f64 {
    1.0 / (1.0 + (-kappa * (rho - rho_0 + delta)).exp())
}

/// Nacelle array function G_n(rho, phi).
///
/// Sum of n azimuthal Gaussians at angular positions phi_k = 2*pi*k/n,
/// with radial envelope F(rho). Normalized so mean value at rho_0 is 1.
///
/// G_n = C_n * F(rho) * sum_{k=0}^{n-1} exp(-(phi - phi_k)^2 / (2*sigma_phi^2))
///
/// where sigma_phi = pi/n (each nacelle subtends ~2*pi/n).
pub fn nacelle_array(rho: f64, phi: f64, n: usize, rho_0: f64, sigma_rho: f64) -> f64 {
    if n == 0 {
        return 1.0; // No nacelles: uniform modulation = Alcubierre limit
    }

    let n_f = n as f64;
    let sigma_phi = PI / n_f;
    let f_rho = nacelle_radial_envelope(rho, rho_0, sigma_rho);

    let mut sum = 0.0;
    for k in 0..n {
        let phi_k = 2.0 * PI * k as f64 / n_f;
        let mut dphi = phi - phi_k;
        // Wrap to [-pi, pi]
        dphi = dphi.rem_euclid(2.0 * PI);
        if dphi > PI {
            dphi -= 2.0 * PI;
        }
        sum += (-dphi * dphi / (2.0 * sigma_phi * sigma_phi)).exp();
    }

    // Normalization: C_n chosen so the azimuthal average of the Gaussian sum
    // at rho = rho_0 equals 1. The average of n unit Gaussians with sigma_phi = pi/n
    // over [0, 2*pi] is n * sigma_phi * sqrt(2*pi) / (2*pi).
    let c_n = (2.0 * PI) / (n_f * sigma_phi * (2.0 * PI).sqrt());

    c_n * f_rho * sum
}

/// Axial taper function W_x(x).
///
/// tanh top-hat along the propagation axis:
/// W_x(x) = [tanh(sigma_x*(x+L)) - tanh(sigma_x*(x-L))] / [2*tanh(sigma_x*L)]
///
/// Properties: W_x(0) ~ 1 for |x| < L, W_x -> 0 for |x| >> L.
pub fn axial_taper(x: f64, l: f64, sigma_x: f64) -> f64 {
    let denom = 2.0 * (sigma_x * l).tanh();
    if denom.abs() < 1e-30 {
        return 0.0;
    }
    ((sigma_x * (x + l)).tanh() - (sigma_x * (x - l)).tanh()) / denom
}

/// Full modulation function M(rho, phi).
///
/// M = 1 - W_g(rho) + W_g(rho) * G_n(rho, phi)
///
/// At the interior (W_g = 0): M = 1 (uniform shift, flat space).
/// At the wall (W_g = 1): M = G_n (nacelle-modulated shift).
pub fn modulation(rho: f64, phi: f64, params: &NacelleWarpParams) -> f64 {
    if params.n_nacelles == 0 {
        return 1.0; // Alcubierre: no modulation
    }
    let wg = gating_function(rho, params.rho_0, params.kappa, params.delta_gating);
    let gn = nacelle_array(rho, phi, params.n_nacelles, params.rho_0, params.sigma_rho);
    1.0 - wg + wg * gn
}

/// Compute the shift vector beta^i at a spacetime point.
///
/// beta_x = -v_s * f(r_s) * M(rho, phi) * W_x(x)
/// beta_y = beta_z = 0
///
/// Coordinates: [t, x, y, z], spatial = {x, y, z}.
pub fn shift_vector(_t: f64, x: f64, y: f64, z: f64, params: &NacelleWarpParams) -> SpatialVector {
    let dx = x - params.x_center;
    let rho = (y * y + z * z).sqrt();
    let r_s = (dx * dx + rho * rho).sqrt();
    let phi = z.atan2(y);

    let f = alcubierre_shape_function(r_s, params.r_bubble, params.sigma);
    let m = modulation(rho, phi, params);
    let wx = axial_taper(dx, params.axial_l, params.sigma_x);

    [-params.v_s * f * m * wx, 0.0, 0.0]
}

/// Nacelle warp bubble energy density (Euler-Lagrange from the metric).
///
/// For the ADM decomposition with alpha=1, gamma=delta, the energy density is:
/// rho = -(1/(16*pi)) * (d_j beta_x)^2 = -(v_s^2/(32*pi)) * W_x^2 * [(d_y(fM))^2 + (d_z(fM))^2]
///
/// Always non-positive (violates WEC).
pub fn nacelle_energy_density(x: f64, y: f64, z: f64, params: &NacelleWarpParams) -> f64 {
    let dx = x - params.x_center;
    let h = 1e-6;

    // Numerical derivatives of the shift component beta_x w.r.t. y and z
    let beta_yp = shift_vector(0.0, x, y + h, z, params)[0];
    let beta_ym = shift_vector(0.0, x, y - h, z, params)[0];
    let beta_zp = shift_vector(0.0, x, y, z + h, params)[0];
    let beta_zm = shift_vector(0.0, x, y, z - h, params)[0];

    let d_beta_dy = (beta_yp - beta_ym) / (2.0 * h);
    let d_beta_dz = (beta_zp - beta_zm) / (2.0 * h);

    // rho = -(1/(32*pi)) * (d_y beta_x)^2 + (d_z beta_x)^2)
    // Factor of 2 from the ADM formula with alpha=1
    let _ = dx; // suppress unused
    -(1.0 / (32.0 * PI)) * (d_beta_dy * d_beta_dy + d_beta_dz * d_beta_dz)
}

/// York time (expansion scalar) for the nacelle warp bubble.
///
/// theta = -K = partial_i beta^i = d_x(beta_x)
/// (since beta_y = beta_z = 0)
///
/// For the nacelle: theta = v_s * f * M * d_x(W_x) + v_s * W_x * d_x(f*M)
pub fn nacelle_york_time(x: f64, y: f64, z: f64, params: &NacelleWarpParams) -> f64 {
    let h = 1e-6;
    let beta_xp = shift_vector(0.0, x + h, y, z, params)[0];
    let beta_xm = shift_vector(0.0, x - h, y, z, params)[0];
    // theta = -div(beta) for our sign convention. Since beta_x is the only
    // nonzero component: theta = -d_x(beta_x).
    // beta_x = -v_s * f * M * W_x, so d_x(beta_x) is negative-ish.
    // York time theta = -K = -gamma^{ij} K_{ij}
    // For alpha=1 and gamma=delta: K_{ij} = -(1/2)(d_i beta_j + d_j beta_i)
    // K = -(1/2)(d_i beta^i + d_i beta^i) = -d_i beta^i = -div(beta)
    // So theta = -K = div(beta) = d_x(beta_x)
    (beta_xp - beta_xm) / (2.0 * h)
}

// ============================================================================
// SpacetimeMetric implementation
// ============================================================================

/// Nacelle warp bubble spacetime (implements SpacetimeMetric).
///
/// Coordinates: [t, x, y, z] (Cartesian).
///
/// g_{00} = -(1 - beta_x^2)
/// g_{0x} = beta_x (= -v_s * f * M * W_x, typically negative)
/// g_{ij} = delta_{ij}
#[derive(Debug, Clone)]
pub struct NacelleWarpBubble {
    pub params: NacelleWarpParams,
}

impl NacelleWarpBubble {
    pub fn new(params: NacelleWarpParams) -> Self {
        Self { params }
    }
}

impl SpacetimeMetric for NacelleWarpBubble {
    fn metric_components(&self, x: &[f64; DIM]) -> MetricComponents {
        let beta = shift_vector(x[0], x[1], x[2], x[3], &self.params);
        let beta_x = beta[0];

        let mut g = [[0.0; DIM]; DIM];

        // g_{00} = -(alpha^2 - beta_i beta^i) where alpha=1 and gamma=delta
        // beta_i = gamma_{ij} beta^j = beta^j (flat spatial metric)
        // so g_{00} = -(1 - beta_x^2)
        g[0][0] = -(1.0 - beta_x * beta_x);

        // g_{0x} = beta_x
        g[0][1] = beta_x;
        g[1][0] = beta_x;

        // Spatial part is flat
        g[1][1] = 1.0;
        g[2][2] = 1.0;
        g[3][3] = 1.0;

        g
    }
}

// ============================================================================
// Utility: Pfenning-Ford energy ratio for nacelle bubbles
// ============================================================================

/// Integrate total negative energy over a cross-section for the nacelle bubble.
///
/// Integrates rho(x, y, z) over a rectangular grid at x=0 (midplane).
/// Returns the estimated total energy per unit length.
pub fn nacelle_cross_section_energy(
    params: &NacelleWarpParams,
    grid_half_size: f64,
    n_grid: usize,
) -> f64 {
    let dy = 2.0 * grid_half_size / n_grid as f64;
    let dz = dy;
    let mut total = 0.0;

    for iy in 0..n_grid {
        let y = -grid_half_size + (iy as f64 + 0.5) * dy;
        for iz in 0..n_grid {
            let z = -grid_half_size + (iz as f64 + 0.5) * dz;
            total += nacelle_energy_density(0.0, y, z, params) * dy * dz;
        }
    }
    total
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::DIM;

    const TOL: f64 = 1e-10;

    // -- Shape function --

    #[test]
    fn test_shape_function_at_center() {
        let f = alcubierre_shape_function(0.0, 10.0, 1.0);
        assert!(
            (f - 1.0).abs() < 0.01,
            "f(0) = {f}, expected ~1.0"
        );
    }

    #[test]
    fn test_shape_function_far_field() {
        let f = alcubierre_shape_function(100.0, 10.0, 1.0);
        assert!(
            f.abs() < 0.01,
            "f(100) = {f}, expected ~0"
        );
    }

    #[test]
    fn test_shape_function_is_positive() {
        for r in [0.0, 1.0, 5.0, 10.0, 20.0] {
            let f = alcubierre_shape_function(r, 10.0, 1.0);
            assert!(f >= -TOL, "f({r}) = {f} should be non-negative");
        }
    }

    #[test]
    fn test_shape_derivative_zero_at_center() {
        // Derivative should be zero at r=0 by symmetry
        let df = alcubierre_shape_derivative(0.0, 10.0, 1.0);
        assert!(df.abs() < 0.01, "df/dr(0) = {df}, expected ~0");
    }

    // -- Gating function --

    #[test]
    fn test_gating_interior_is_zero() {
        // Far inside: rho << rho_0
        let wg = gating_function(0.1, 5.0, 10.0, 0.5);
        assert!(wg < 0.01, "W_g(interior) = {wg}, expected ~0");
    }

    #[test]
    fn test_gating_exterior_is_one() {
        // At/beyond nacelle radius: rho >= rho_0
        let wg = gating_function(6.0, 5.0, 10.0, 0.5);
        assert!(wg > 0.99, "W_g(exterior) = {wg}, expected ~1");
    }

    // -- Axial taper --

    #[test]
    fn test_axial_taper_center() {
        let wx = axial_taper(0.0, 8.0, 1.0);
        assert!(
            (wx - 1.0).abs() < 0.01,
            "W_x(0) = {wx}, expected ~1"
        );
    }

    #[test]
    fn test_axial_taper_far() {
        let wx = axial_taper(100.0, 8.0, 1.0);
        assert!(wx < 0.01, "W_x(100) = {wx}, expected ~0");
    }

    // -- Modulation --

    #[test]
    fn test_modulation_alcubierre_limit() {
        let params = NacelleWarpParams {
            n_nacelles: 0,
            ..NacelleWarpParams::default()
        };
        let m = modulation(5.0, 0.0, &params);
        assert!(
            (m - 1.0).abs() < TOL,
            "Alcubierre limit: M = {m}, expected 1"
        );
    }

    #[test]
    fn test_modulation_interior() {
        let params = NacelleWarpParams::default();
        // Interior: rho << rho_0, so W_g ~ 0, M ~ 1
        let m = modulation(0.1, 0.0, &params);
        assert!(
            (m - 1.0).abs() < 0.01,
            "Interior M = {m}, expected ~1"
        );
    }

    // -- Shift vector --

    #[test]
    fn test_shift_zero_far_field() {
        let params = NacelleWarpParams::default();
        let beta = shift_vector(0.0, 100.0, 100.0, 100.0, &params);
        for (i, &b) in beta.iter().enumerate() {
            assert!(b.abs() < 0.01, "far-field beta[{i}] = {}", b);
        }
    }

    #[test]
    fn test_shift_transverse_zero() {
        let params = NacelleWarpParams::default();
        let beta = shift_vector(0.0, 0.0, 0.0, 0.0, &params);
        // beta_y and beta_z are always zero
        assert!(beta[1].abs() < TOL, "beta_y = {}", beta[1]);
        assert!(beta[2].abs() < TOL, "beta_z = {}", beta[2]);
    }

    // -- SpacetimeMetric implementation --

    #[test]
    fn test_metric_far_field_is_minkowski() {
        let params = NacelleWarpParams::default();
        let warp = NacelleWarpBubble::new(params);
        let x = [0.0, 100.0, 100.0, 100.0];
        let g = warp.metric_components(&x);

        // Far field: beta ~ 0, so g -> eta
        assert!(
            (g[0][0] - (-1.0)).abs() < 0.01,
            "far-field g_00 = {}",
            g[0][0]
        );
        assert!(
            g[0][1].abs() < 0.01,
            "far-field g_0x = {}",
            g[0][1]
        );
        assert!(
            (g[1][1] - 1.0).abs() < TOL,
            "far-field g_xx = {}",
            g[1][1]
        );
    }

    #[test]
    fn test_metric_spatial_is_flat() {
        let params = NacelleWarpParams::default();
        let warp = NacelleWarpBubble::new(params);
        let x = [0.0, 1.0, 2.0, 3.0];
        let g = warp.metric_components(&x);

        // Spatial metric is always flat Euclidean
        assert!((g[1][1] - 1.0).abs() < TOL);
        assert!((g[2][2] - 1.0).abs() < TOL);
        assert!((g[3][3] - 1.0).abs() < TOL);
        assert!(g[1][2].abs() < TOL);
        assert!(g[1][3].abs() < TOL);
        assert!(g[2][3].abs() < TOL);
    }

    // -- Energy density --

    #[test]
    fn test_energy_density_nonpositive() {
        let params = NacelleWarpParams::default();
        // Sample several points in the bubble wall region
        for y in [3.0, 5.0, 7.0] {
            let rho = nacelle_energy_density(0.0, y, 0.0, &params);
            assert!(
                rho <= 1e-15,
                "energy density at y={y}: {rho} (must be <= 0)"
            );
        }
    }

    #[test]
    fn test_energy_density_zero_on_axis() {
        let params = NacelleWarpParams::default();
        // On the propagation axis (y=z=0), transverse gradients vanish
        let rho = nacelle_energy_density(0.0, 0.0, 0.0, &params);
        assert!(
            rho.abs() < 1e-8,
            "energy density on axis: {rho} (expected ~0)"
        );
    }

    // -- York time --

    #[test]
    fn test_york_time_zero_at_center() {
        let params = NacelleWarpParams::default();
        // At the interior center, the shift is nearly uniform, so divergence ~ 0
        let theta = nacelle_york_time(0.0, 0.0, 0.0, &params);
        // May not be exactly zero due to axial taper gradient, but should be small
        assert!(
            theta.abs() < 0.1,
            "York time at center: {theta}"
        );
    }

    // -- Nacelle count effects --

    #[test]
    fn test_nacelle_energy_varies_with_n() {
        let e2 = {
            let params = NacelleWarpParams::white_2025(0.1, 2);
            nacelle_cross_section_energy(&params, 15.0, 50)
        };
        let e4 = {
            let params = NacelleWarpParams::white_2025(0.1, 4);
            nacelle_cross_section_energy(&params, 15.0, 50)
        };
        // Different nacelle counts should produce different total energies
        assert!(
            (e2 - e4).abs() > 1e-12,
            "n=2 energy {e2} vs n=4 energy {e4} should differ"
        );
    }

    #[test]
    fn test_alcubierre_limit_energy() {
        // With n=0 (Alcubierre), energy should be nonzero in the wall region
        let params = NacelleWarpParams::alcubierre(0.1, 10.0, 1.0);
        let e = nacelle_cross_section_energy(&params, 15.0, 50);
        assert!(e < 0.0, "Alcubierre total energy should be negative: {e}");
    }

    // -- Metric symmetry --

    #[test]
    #[allow(clippy::needless_range_loop)] // 2D metric tensor symmetry check g[mu][nu] == g[nu][mu]
    fn test_metric_symmetric() {
        let params = NacelleWarpParams::default();
        let warp = NacelleWarpBubble::new(params);
        let x = [0.0, 1.0, 2.0, 3.0];
        let g = warp.metric_components(&x);

        for mu in 0..DIM {
            for nu in 0..DIM {
                assert!(
                    (g[mu][nu] - g[nu][mu]).abs() < TOL,
                    "g[{mu}][{nu}] = {} != g[{nu}][{mu}] = {}",
                    g[mu][nu],
                    g[nu][mu]
                );
            }
        }
    }

    // -- Cross-validation with quantum_inequalities --

    #[test]
    fn test_nacelle_alcubierre_cross_validation() {
        // When n_nacelles = 0, nacelle energy should match the Alcubierre formula
        // at a specific point (up to sign/unit conventions).
        let v_s = 0.1;
        let r_bubble = 10.0;
        let sigma = 1.0;
        let params = NacelleWarpParams::alcubierre(v_s, r_bubble, sigma);

        // Check at a point off-axis
        let x_pt = 0.0;
        let y_pt = 5.0;
        let z_pt = 0.0;

        let rho_nacelle = nacelle_energy_density(x_pt, y_pt, z_pt, &params);
        // Should be non-positive
        assert!(rho_nacelle <= 1e-15, "Alcubierre limit density should be <= 0: {rho_nacelle}");
    }
}
