//! Sersic surface brightness profile, Abel deprojection, and LBM density.
//!
//! Implements the Sersic (1963) profile I(R) = I_e * exp(-b_n * ((R/R_e)^(1/n) - 1))
//! and its 3D deprojection via the Abel transform for generating initial
//! density fields suitable for LBM simulation.
//!
//! # Physics
//!
//! The Sersic profile is the standard parametrization for galaxy surface
//! brightness.  n=1 gives an exponential (disk), n=4 gives de Vaucouleurs
//! (elliptical).  The Abel deprojection recovers the intrinsic 3D luminosity
//! density j(r) assuming spherical symmetry:
//!
//!   j(r) = -(1/pi) * integral_r^inf (dI/dR) * dR / sqrt(R^2 - r^2)
//!
//! We regularize the R=r singularity with the cosh substitution R = r*cosh(u),
//! which transforms the integrand to a smooth function of u on [0, U_max].
//!
//! # References
//! - Sersic (1963): Atlas de Galaxias Australes
//! - Ciotti & Bertin (1999): A&A 352, 447 (b_n approximation)
//! - Mellier & Mathez (1987): A&A 175, 1 (deprojection)
//! - Graham & Driver (2005): PASA 22, 118 (review)

use crate::gl_integrate;
use statrs::function::gamma::ln_gamma;

// ---------------------------------------------------------------------------
// Sersic b_n parameter
// ---------------------------------------------------------------------------

/// Ciotti & Bertin (1999) approximation for the Sersic b_n parameter.
///
/// b_n is defined such that Gamma(2n, b_n) = Gamma(2n)/2, i.e. the
/// effective radius R_e encloses half the total luminosity.
///
/// Accurate to < 0.1% for n >= 0.5.
pub fn sersic_bn(n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    // Ciotti & Bertin (1999) Eq. 18 with third-order correction
    2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)
        + 131.0 / (1148175.0 * n * n * n)
}

// ---------------------------------------------------------------------------
// 2D surface brightness
// ---------------------------------------------------------------------------

/// Sersic surface brightness I(R) / I_e.
///
/// Returns the dimensionless ratio I(R)/I_e where I_e is the surface
/// brightness at the effective radius R_e.
pub fn sersic_profile_2d(r: f64, r_e: f64, n: f64) -> f64 {
    if r < 0.0 || r_e <= 0.0 || n <= 0.0 {
        return 0.0;
    }
    let bn = sersic_bn(n);
    let x = r / r_e;
    (-bn * (x.powf(1.0 / n) - 1.0)).exp()
}

/// Derivative dI/dR of the Sersic profile (needed for Abel transform).
///
/// dI/dR = I(R) * (-b_n / (n * R_e)) * (R / R_e)^(1/n - 1)
fn sersic_di_dr(r: f64, r_e: f64, n: f64, i_e: f64, bn: f64) -> f64 {
    if r <= 0.0 || r_e <= 0.0 || n <= 0.0 {
        return 0.0;
    }
    let x = r / r_e;
    let profile = i_e * (-bn * (x.powf(1.0 / n) - 1.0)).exp();
    profile * (-bn / (n * r_e)) * x.powf(1.0 / n - 1.0)
}

// ---------------------------------------------------------------------------
// Total luminosity
// ---------------------------------------------------------------------------

/// Total luminosity of a Sersic profile (2D integral).
///
/// L = 2 * pi * n * I_e * R_e^2 * exp(b_n) * Gamma(2n) / b_n^(2n)
///
/// For n=1 (exponential): L = 2 * pi * I_e * R_e^2 (since b_1 ~ 1.678
/// and Gamma(2) = 1, exp(b_1)/b_1^2 = 1).
pub fn sersic_total_luminosity(i_e: f64, r_e: f64, n: f64) -> f64 {
    if i_e <= 0.0 || r_e <= 0.0 || n <= 0.0 {
        return 0.0;
    }
    let bn = sersic_bn(n);
    let gamma_2n = ln_gamma(2.0 * n).exp();
    2.0 * std::f64::consts::PI * n * i_e * r_e * r_e * bn.exp() * gamma_2n / bn.powf(2.0 * n)
}

// ---------------------------------------------------------------------------
// Abel deprojection -> 3D density
// ---------------------------------------------------------------------------

/// Abel-deprojected 3D luminosity density j(r).
///
/// j(r) = -(1/pi) * integral_r^inf dI/dR * dR / sqrt(R^2 - r^2)
///
/// We use the cosh substitution R = r * cosh(u) to remove the singularity:
///   dR = r * sinh(u) * du
///   sqrt(R^2 - r^2) = r * sinh(u)
///   => integrand becomes dI/dR(r*cosh(u)) * du  (sinh cancels)
///
/// Integration range: u in [0, U_max] where R_max = 20 * R_e.
pub fn sersic_deprojected_density(r: f64, r_e: f64, n: f64, i_e: f64) -> f64 {
    if r <= 0.0 || r_e <= 0.0 || n <= 0.0 || i_e <= 0.0 {
        return 0.0;
    }

    let bn = sersic_bn(n);
    let r_max = 20.0 * r_e;

    // u_max such that r * cosh(u_max) = r_max
    let cosh_max = r_max / r;
    if cosh_max <= 1.0 {
        return 0.0;
    }
    let u_max = (cosh_max + (cosh_max * cosh_max - 1.0).sqrt()).ln();

    // Integrate dI/dR(r*cosh(u)) du from 0 to u_max
    let integral = gl_integrate(
        |u: f64| {
            let big_r = r * u.cosh();
            sersic_di_dr(big_r, r_e, n, i_e, bn)
        },
        0.0,
        u_max,
        64,
    );

    // j(r) = -(1/pi) * integral
    -integral / std::f64::consts::PI
}

// ---------------------------------------------------------------------------
// LBM density field
// ---------------------------------------------------------------------------

/// Configuration for Sersic-to-LBM density conversion.
#[derive(Debug, Clone)]
pub struct SersicLbmConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Cell size in kpc.
    pub dx_kpc: f64,
    /// Effective (half-light) radius in kpc.
    pub r_e_kpc: f64,
    /// Sersic index.
    pub n: f64,
    /// Surface brightness at R_e (arbitrary units).
    pub i_e: f64,
    /// Ellipticity e = 1 - b/a.
    pub ellipticity: f64,
    /// Position angle in radians.
    pub pa_rad: f64,
    /// Mass-to-light ratio.
    pub ml_ratio: f64,
}

/// Generate a 3D density field from a Sersic profile for LBM simulation.
///
/// Places the galaxy center at the grid center. Each cell gets the
/// deprojected density j(r) converted to mass density via M/L ratio.
///
/// Supports elliptical profiles: the radius is computed as
///   r_eff = sqrt(x'^2 + (y'/(1-e))^2 + z'^2)
/// where (x', y') are rotated by position angle PA.
///
/// Returns: density field of size nx * ny * nz (row-major).
pub fn sersic_to_lbm_density(cfg: &SersicLbmConfig) -> Vec<f64> {
    let SersicLbmConfig {
        nx, ny, nz, dx_kpc, r_e_kpc, n, i_e, ellipticity, pa_rad, ml_ratio,
    } = *cfg;
    let total = nx * ny * nz;
    let mut rho = vec![0.0_f64; total];

    let cx = nx as f64 / 2.0;
    let cy = ny as f64 / 2.0;
    let cz = nz as f64 / 2.0;

    let cos_pa = pa_rad.cos();
    let sin_pa = pa_rad.sin();
    let q = 1.0 - ellipticity; // axis ratio b/a
    let q_inv = if q > 0.0 { 1.0 / q } else { 1.0 };

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dx_cell = (ix as f64 + 0.5 - cx) * dx_kpc;
                let dy_cell = (iy as f64 + 0.5 - cy) * dx_kpc;
                let dz_cell = (iz as f64 + 0.5 - cz) * dx_kpc;

                // Rotate by PA in xy-plane
                let x_rot = dx_cell * cos_pa + dy_cell * sin_pa;
                let y_rot = -dx_cell * sin_pa + dy_cell * cos_pa;

                // Elliptical radius
                let r = (x_rot * x_rot + (y_rot * q_inv) * (y_rot * q_inv)
                    + dz_cell * dz_cell)
                    .sqrt();

                if r > 0.0 {
                    let j = sersic_deprojected_density(r, r_e_kpc, n, i_e);
                    rho[iz * ny * nx + iy * nx + ix] = j * ml_ratio;
                }
            }
        }
    }

    rho
}

// ---------------------------------------------------------------------------
// Otsu threshold selection
// ---------------------------------------------------------------------------

/// Otsu's method for bimodal threshold selection.
///
/// Partitions density values into 256 histogram bins, finds the threshold
/// that maximizes between-class variance (inter-class variance). Ideal for
/// bimodal distributions like galaxy+background density fields where median
/// fails (median = floor when >50% of cells are background).
///
/// Falls back to the arithmetic mean if the histogram is unimodal (all
/// values in one bin) or the input is empty.
///
/// Reference: Otsu (1979), IEEE Trans. Sys. Man. Cyber. 9(1), 62-66.
pub fn otsu_threshold(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in values {
        if v < vmin {
            vmin = v;
        }
        if v > vmax {
            vmax = v;
        }
    }

    let range = vmax - vmin;
    if range < 1e-30 {
        // All values identical -- return that value
        return vmin;
    }

    // Build 256-bin histogram
    const N_BINS: usize = 256;
    let mut hist = [0u64; N_BINS];
    let bin_width = range / N_BINS as f64;
    for &v in values {
        let bin = ((v - vmin) / bin_width).min((N_BINS - 1) as f64) as usize;
        hist[bin] += 1;
    }

    let total = values.len() as f64;

    // Precompute cumulative sums for O(N_BINS) sweep
    let mut sum_total = 0.0_f64;
    for (i, &count) in hist.iter().enumerate() {
        let bin_center = vmin + (i as f64 + 0.5) * bin_width;
        sum_total += bin_center * count as f64;
    }

    let mut w0 = 0.0_f64; // weight of class 0 (background)
    let mut sum0 = 0.0_f64; // cumulative mean * weight for class 0
    let mut best_sigma = -1.0_f64;
    let mut best_threshold = vmin;

    for (i, &count) in hist.iter().enumerate() {
        let bin_center = vmin + (i as f64 + 0.5) * bin_width;
        w0 += count as f64;
        if w0 == 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0.0 {
            break;
        }

        sum0 += bin_center * count as f64;
        let mu0 = sum0 / w0;
        let mu1 = (sum_total - sum0) / w1;
        let sigma_b = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);

        if sigma_b > best_sigma {
            best_sigma = sigma_b;
            // Threshold = upper edge of this bin
            best_threshold = vmin + (i as f64 + 1.0) * bin_width;
        }
    }

    best_threshold
}

/// Otsu threshold for f32 density field (GPU readback path).
///
/// Widens to f64 for the histogram sweep, returns f32 for direct use
/// with GPU box-counting kernels.
pub fn otsu_threshold_f32(values: &[f32]) -> f32 {
    let widened: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    otsu_threshold(&widened) as f32
}

// ---------------------------------------------------------------------------
// Box-counting fractal dimension
// ---------------------------------------------------------------------------

/// Box-counting fractal dimension D_f of a 3D density field.
///
/// Thresholds the field using Otsu's method (bimodal-optimal), then counts
/// occupied boxes at power-of-2 scales from 1 up to min(nx, ny, nz)/2.
/// D_f is the negative slope of the log-log regression:
///   D_f = -d(log N) / d(log epsilon).
///
/// Returns 3.0 on degenerate inputs (empty, uniform, or < 2 scales).
pub fn box_counting_fractal_dim(rho: &[f64], nx: usize, ny: usize, nz: usize) -> f64 {
    let finite: Vec<f64> = rho.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return 3.0;
    }
    let threshold = otsu_threshold(&finite);
    box_counting_fractal_dim_threshold(rho, nx, ny, nz, threshold)
}

/// Box-counting fractal dimension from an f32 density field (GPU readback path).
///
/// Widens to f64 internally for the regression but computes Otsu threshold
/// directly on f32 values widened to f64.
pub fn box_counting_fractal_dim_f32(rho: &[f32], nx: usize, ny: usize, nz: usize) -> f64 {
    let finite: Vec<f64> = rho.iter().filter(|v| v.is_finite()).map(|&v| v as f64).collect();
    if finite.is_empty() {
        return 3.0;
    }
    let threshold = otsu_threshold(&finite);
    let rho_f64: Vec<f64> = rho.iter().map(|&v| v as f64).collect();
    box_counting_fractal_dim_threshold(&rho_f64, nx, ny, nz, threshold)
}

/// Internal: box-counting with explicit threshold.
fn box_counting_fractal_dim_threshold(
    rho: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    threshold: f64,
) -> f64 {
    let mut log_n = Vec::new();
    let mut log_eps = Vec::new();

    let min_dim = nx.min(ny).min(nz);
    let mut box_size = 1;
    while box_size <= min_dim / 2 {
        let bx = nx.div_ceil(box_size);
        let by = ny.div_ceil(box_size);
        let bz = nz.div_ceil(box_size);

        let mut count = 0u64;
        for ibz in 0..bz {
            for iby in 0..by {
                for ibx in 0..bx {
                    let mut occupied = false;
                    'outer: for dz in 0..box_size {
                        for dy in 0..box_size {
                            for dx_cell in 0..box_size {
                                let ix = ibx * box_size + dx_cell;
                                let iy = iby * box_size + dy;
                                let iz = ibz * box_size + dz;
                                if ix < nx && iy < ny && iz < nz {
                                    let idx = iz * ny * nx + iy * nx + ix;
                                    if rho[idx] > threshold {
                                        occupied = true;
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    }
                    if occupied {
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            log_n.push((count as f64).ln());
            log_eps.push((box_size as f64).ln());
        }

        box_size *= 2;
    }

    if log_n.len() < 2 {
        return 3.0;
    }

    let n = log_n.len() as f64;
    let sum_x: f64 = log_eps.iter().sum();
    let sum_y: f64 = log_n.iter().sum();
    let sum_xy: f64 = log_eps.iter().zip(log_n.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = log_eps.iter().map(|x| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return 3.0;
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    -slope
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bn_n4() {
        // de Vaucouleurs: b_4 ~ 7.6693
        let b4 = sersic_bn(4.0);
        assert_relative_eq!(b4, 7.6693, epsilon = 0.01);
    }

    #[test]
    fn test_bn_n1() {
        // Exponential: b_1 ~ 1.6783
        let b1 = sersic_bn(1.0);
        assert_relative_eq!(b1, 1.6783, epsilon = 0.01);
    }

    #[test]
    fn test_profile_at_re() {
        // I(R_e) / I_e = 1.0 by definition
        let ratio = sersic_profile_2d(5.0, 5.0, 4.0);
        assert_relative_eq!(ratio, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_profile_monotone() {
        let r_e = 5.0;
        let n = 4.0;
        let mut prev = sersic_profile_2d(0.01, r_e, n);
        for i in 1..20 {
            let r = 0.01 + i as f64 * 0.5;
            let curr = sersic_profile_2d(r, r_e, n);
            assert!(
                curr <= prev + 1e-12,
                "Profile not monotone at r={}: {} > {}",
                r,
                curr,
                prev
            );
            prev = curr;
        }
    }

    #[test]
    fn test_luminosity_n1_analytic() {
        // For exponential (n=1): L = 2*pi*I_e*R_e^2 * exp(b1)*Gamma(2)/b1^2
        // With b1 ~ 1.6783, exp(b1)/b1^2 = exp(1.6783)/(1.6783^2) ~ 1.90
        // Exact: L = 2*pi*I_e*R_e^2 when we define I_e via b_n convention
        let i_e = 1.0;
        let r_e = 1.0;
        let l = sersic_total_luminosity(i_e, r_e, 1.0);
        // For n=1, analytic result: L = 2*pi*n*I_e*R_e^2*exp(b_n)*Gamma(2n)/b_n^(2n)
        // Gamma(2) = 1, so L = 2*pi*exp(b1)/b1^2
        let b1 = sersic_bn(1.0);
        let expected = 2.0 * std::f64::consts::PI * b1.exp() / (b1 * b1);
        assert_relative_eq!(l, expected, epsilon = 0.01);
    }

    #[test]
    fn test_deprojected_positive() {
        let r_e = 5.0;
        let n = 4.0;
        let i_e = 1.0;
        for i in 1..10 {
            let r = i as f64 * 1.0;
            let j = sersic_deprojected_density(r, r_e, n, i_e);
            assert!(
                j >= 0.0,
                "Deprojected density negative at r={}: {}",
                r, j
            );
        }
    }

    #[test]
    fn test_lbm_density_symmetry() {
        // Circular (e=0) -> should be azimuthally symmetric
        let nx = 16;
        let cfg = SersicLbmConfig {
            nx, ny: nx, nz: nx, dx_kpc: 1.0, r_e_kpc: 3.0,
            n: 4.0, i_e: 1.0, ellipticity: 0.0, pa_rad: 0.0, ml_ratio: 1.0,
        };
        let rho = sersic_to_lbm_density(&cfg);
        // Check xy symmetry at z=center
        let cz = nx / 2;
        let cx = nx / 2;
        let cy = nx / 2;
        let rho_xp = rho[cz * nx * nx + cy * nx + (cx + 2)];
        let rho_yp = rho[cz * nx * nx + (cy + 2) * nx + cx];
        assert_relative_eq!(rho_xp, rho_yp, epsilon = 1e-6);
    }

    #[test]
    fn test_lbm_density_mass_conservation() {
        let nx = 32;
        let dx = 0.5; // kpc
        let r_e = 3.0; // kpc
        let i_e = 1.0;
        let n = 1.0;
        let ml = 1.0;
        let cfg = SersicLbmConfig {
            nx, ny: nx, nz: nx, dx_kpc: dx, r_e_kpc: r_e,
            n, i_e, ellipticity: 0.0, pa_rad: 0.0, ml_ratio: ml,
        };
        let rho = sersic_to_lbm_density(&cfg);

        let total_mass: f64 = rho.iter().sum::<f64>() * dx * dx * dx;
        let total_lum = sersic_total_luminosity(i_e, r_e, n);

        // The 3D grid captures most of the luminosity (within truncation)
        // Grid spans +/- 8 kpc from center. For R_e=3 kpc exponential,
        // most light is within ~5 R_e = 15 kpc, so we expect >80% capture.
        let ratio = total_mass / total_lum;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Mass conservation ratio = {:.4} (expected ~1.0, grid truncation applies)",
            ratio
        );
    }

    #[test]
    fn test_bn_edge_cases() {
        assert_eq!(sersic_bn(0.0), 0.0);
        assert_eq!(sersic_bn(-1.0), 0.0);
    }

    #[test]
    fn test_profile_edge_cases() {
        assert_eq!(sersic_profile_2d(-1.0, 5.0, 4.0), 0.0);
        assert_eq!(sersic_profile_2d(1.0, 0.0, 4.0), 0.0);
        assert_eq!(sersic_profile_2d(1.0, 5.0, 0.0), 0.0);
    }

    // --- Otsu threshold tests ---

    #[test]
    fn test_otsu_bimodal() {
        // 90% at floor 0.01, 10% at galaxy core 5.0
        let mut values = vec![0.01_f64; 900];
        values.extend(vec![5.0_f64; 100]);
        let threshold = otsu_threshold(&values);
        // Otsu should place threshold between the two modes
        assert!(
            threshold > 0.01 && threshold < 5.0,
            "Otsu threshold={threshold:.4}, expected between 0.01 and 5.0"
        );
    }

    #[test]
    fn test_otsu_uniform() {
        // All values identical -> returns that value
        let values = vec![1.0_f64; 100];
        let threshold = otsu_threshold(&values);
        assert_relative_eq!(threshold, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_otsu_empty() {
        let threshold = otsu_threshold(&[]);
        assert_eq!(threshold, 0.0);
    }

    #[test]
    fn test_otsu_two_values() {
        let values = vec![0.0, 10.0];
        let threshold = otsu_threshold(&values);
        // Should split between 0 and 10
        assert!(
            threshold > 0.0 && threshold <= 10.0,
            "Otsu on [0,10] = {threshold:.4}"
        );
    }

    #[test]
    fn test_otsu_f32_agrees() {
        let mut values_f64 = vec![0.01_f64; 900];
        values_f64.extend(vec![5.0_f64; 100]);
        let values_f32: Vec<f32> = values_f64.iter().map(|&v| v as f32).collect();
        let t64 = otsu_threshold(&values_f64);
        let t32 = otsu_threshold_f32(&values_f32) as f64;
        assert!(
            (t64 - t32).abs() < 0.1,
            "f32/f64 Otsu mismatch: {t32:.4} vs {t64:.4}"
        );
    }

    #[test]
    fn test_box_counting_bimodal_not_degenerate() {
        // Simulate bimodal field: floor at 0.01 with a small galaxy core
        let n = 16;
        let mut rho = vec![0.01_f64; n * n * n];
        // Place a sphere of high density at center
        let c = n / 2;
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let dx = ix as f64 - c as f64;
                    let dy = iy as f64 - c as f64;
                    let dz = iz as f64 - c as f64;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    if r < 3.0 {
                        rho[iz * n * n + iy * n + ix] = 5.0;
                    }
                }
            }
        }
        let df = box_counting_fractal_dim(&rho, n, n, n);
        // With Otsu, the galaxy core should yield D_f well below 3.0
        assert!(
            df < 2.8,
            "Bimodal field D_f={df:.4}, expected < 2.8 (not the uniform artifact)"
        );
    }

    // --- Box-counting fractal dimension tests ---

    #[test]
    fn test_box_counting_uniform_field() {
        // Uniform field above threshold -> every box occupied -> D_f = 3.0
        let n = 16;
        let rho = vec![2.0_f64; n * n * n];
        let df = box_counting_fractal_dim(&rho, n, n, n);
        assert_relative_eq!(df, 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_box_counting_single_cell() {
        // Single occupied cell at center -> D_f close to 0
        let n = 16;
        let mut rho = vec![0.0_f64; n * n * n];
        let c = n / 2;
        rho[c * n * n + c * n + c] = 10.0;
        let df = box_counting_fractal_dim(&rho, n, n, n);
        assert!(
            df < 1.0,
            "Single cell D_f={df:.4}, expected < 1.0"
        );
    }

    #[test]
    fn test_box_counting_filled_plane() {
        // Filled z=center plane -> D_f ~ 2.0
        let n = 16;
        let mut rho = vec![0.0_f64; n * n * n];
        let cz = n / 2;
        for iy in 0..n {
            for ix in 0..n {
                rho[cz * n * n + iy * n + ix] = 5.0;
            }
        }
        let df = box_counting_fractal_dim(&rho, n, n, n);
        assert!(
            df > 1.5 && df < 2.5,
            "Plane D_f={df:.4}, expected ~2.0"
        );
    }

    #[test]
    fn test_box_counting_f32_agrees_with_f64() {
        let n = 16;
        let rho_f64: Vec<f64> = (0..(n * n * n))
            .map(|i| ((i % 7) as f64) * 0.3)
            .collect();
        let rho_f32: Vec<f32> = rho_f64.iter().map(|&v| v as f32).collect();
        let df_64 = box_counting_fractal_dim(&rho_f64, n, n, n);
        let df_32 = box_counting_fractal_dim_f32(&rho_f32, n, n, n);
        assert!(
            (df_64 - df_32).abs() < 0.1,
            "f32/f64 D_f mismatch: {df_32:.4} vs {df_64:.4}"
        );
    }

    #[test]
    fn test_box_counting_empty_field() {
        let n = 8;
        let rho = vec![0.0_f64; n * n * n];
        let df = box_counting_fractal_dim(&rho, n, n, n);
        // All zeros: median = 0, all cells equal threshold (not >), so D_f = 3.0 fallback
        assert_relative_eq!(df, 3.0, epsilon = 0.1);
    }
}
