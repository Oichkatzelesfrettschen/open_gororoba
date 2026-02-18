//! Tabulated real-axis optical data (n, k) with Kramers-Kronig transform.
//!
//! The KK relation for the imaginary-axis dielectric function:
//!
//!   eps(i*xi) = 1 + (2/pi) * integral_0^inf omega*eps2(omega)/(omega^2+xi^2) d omega
//!
//! where eps2(omega) = Im[eps(omega)] = 2*n(omega)*k(omega).
//!
//! This follows from Cauchy's theorem applied to eps(omega)-1, which is analytic
//! in the upper half-plane with Im[eps] >= 0 (passivity). The imaginary-axis
//! dielectric function is real, smooth, and monotone-decreasing -- properties that
//! make Gauss-Legendre quadrature of the Lifshitz formula exponentially convergent
//! in n_matsubara.
//!
//! # Integration strategy
//!
//! The semi-infinite integral over [0, inf) is split:
//!
//!   1. Tabulated range [omega_min, omega_max]: trapezoidal rule.
//!   2. IR tail [0, omega_min]: Drude analytical (if DrudeParams given) or
//!      1/omega extrapolation scaled to eps2_min (Drude-like metallic tail).
//!   3. UV tail [omega_max, inf]: 1/omega^3 free-electron extrapolation
//!      (Thomas-Reiche-Kuhn: interband contribution decays beyond the last oscillator).
//!      Analytical closed form via partial fractions.
//!
//! # Data sources
//!
//! Johnson & Christy (1972) -- Gold, Silver, Copper:
//!   P. B. Johnson and R. W. Christy, Phys. Rev. B 6, 4370-4379 (1972).
//!   Room temperature, 49 data points, 0.64-6.60 eV (1937-188 nm).
//!   Retrieved from refractiveindex.info database (CC0 public domain).

use crate::optical_database::{DrudeParams, EV_TO_RADS, K_B_EV, E_CHARGE, C};
use gauss_quad::GaussLegendre;
use std::f64::consts::PI;

// ============================================================================
// Johnson & Christy 1972 (PRB 6, 4370) room-temperature data
// Energy in eV ascending (converted from wavelength um: E = 1.2398419843 / lambda_um)
// Retrieved from: https://refractiveindex.info/database (CC0 public domain)
// ============================================================================

const JC_AU_EV: [f64; 49] = [
    0.640084, 0.770088, 0.890052, 1.019607, 1.139561, 1.260002,
    1.389957, 1.509977, 1.640003, 1.759889, 1.879973, 2.010120,
    2.129947, 2.260011, 2.380192, 2.500185, 2.630127, 2.749705,
    2.880005, 2.999860, 3.119884, 3.249913, 3.370052, 3.500401,
    3.619977, 3.740096, 3.869669, 3.990480, 4.120445, 4.240226,
    4.359501, 4.490554, 4.610792, 4.739457, 4.860219, 4.979285,
    5.110643, 5.229194, 5.360320, 5.481176, 5.600009, 5.729399,
    5.851071, 5.980907, 6.098583, 6.220983, 6.348397, 6.470992,
    6.598414,
];
const JC_AU_N: [f64; 49] = [
    0.920000, 0.560000, 0.430000, 0.350000, 0.270000, 0.220000,
    0.170000, 0.160000, 0.140000, 0.130000, 0.140000, 0.210000,
    0.290000, 0.430000, 0.620000, 1.040000, 1.310000, 1.380000,
    1.450000, 1.460000, 1.470000, 1.460000, 1.480000, 1.500000,
    1.480000, 1.480000, 1.540000, 1.530000, 1.530000, 1.490000,
    1.470000, 1.430000, 1.380000, 1.350000, 1.330000, 1.330000,
    1.320000, 1.320000, 1.300000, 1.310000, 1.300000, 1.300000,
    1.300000, 1.300000, 1.330000, 1.330000, 1.340000, 1.320000,
    1.280000,
];
const JC_AU_K: [f64; 49] = [
    13.780000, 11.210000, 9.519000, 8.145000, 7.150000, 6.350000,
    5.663000, 5.083000, 4.542000, 4.103000, 3.697000, 3.272000,
    2.863000, 2.455000, 2.081000, 1.833000, 1.849000, 1.914000,
    1.948000, 1.958000, 1.952000, 1.933000, 1.895000, 1.866000,
    1.871000, 1.883000, 1.898000, 1.893000, 1.889000, 1.878000,
    1.869000, 1.847000, 1.803000, 1.749000, 1.688000, 1.631000,
    1.577000, 1.536000, 1.497000, 1.460000, 1.427000, 1.387000,
    1.350000, 1.304000, 1.277000, 1.251000, 1.226000, 1.203000,
    1.188000,
];

const JC_AG_EV: [f64; 49] = [
    0.640084, 0.770088, 0.890052, 1.019607, 1.139561, 1.260002,
    1.389957, 1.509977, 1.640003, 1.759889, 1.879973, 2.010120,
    2.129947, 2.260011, 2.380192, 2.500185, 2.630127, 2.749705,
    2.880005, 2.999860, 3.119884, 3.249913, 3.370052, 3.500401,
    3.619977, 3.740096, 3.869669, 3.990480, 4.120445, 4.240226,
    4.359501, 4.490554, 4.610792, 4.739457, 4.860219, 4.979285,
    5.110643, 5.229194, 5.360320, 5.481176, 5.600009, 5.729399,
    5.851071, 5.980907, 6.098583, 6.220983, 6.348397, 6.470992,
    6.598414,
];
const JC_AG_N: [f64; 49] = [
    0.240000, 0.150000, 0.130000, 0.090000, 0.040000, 0.040000,
    0.040000, 0.040000, 0.030000, 0.040000, 0.050000, 0.060000,
    0.050000, 0.060000, 0.050000, 0.050000, 0.050000, 0.040000,
    0.040000, 0.050000, 0.050000, 0.050000, 0.070000, 0.100000,
    0.140000, 0.170000, 0.810000, 1.130000, 1.340000, 1.390000,
    1.410000, 1.410000, 1.380000, 1.350000, 1.330000, 1.310000,
    1.300000, 1.280000, 1.280000, 1.260000, 1.250000, 1.220000,
    1.200000, 1.180000, 1.150000, 1.140000, 1.120000, 1.100000,
    1.070000,
];
const JC_AG_K: [f64; 49] = [
    14.080000, 11.850000, 10.100000, 8.828000, 7.795000, 6.992000,
    6.312000, 5.727000, 5.242000, 4.838000, 4.483000, 4.152000,
    3.858000, 3.586000, 3.324000, 3.093000, 2.869000, 2.657000,
    2.462000, 2.275000, 2.070000, 1.864000, 1.657000, 1.419000,
    1.142000, 0.829000, 0.392000, 0.616000, 0.964000, 1.161000,
    1.264000, 1.331000, 1.372000, 1.387000, 1.393000, 1.389000,
    1.378000, 1.367000, 1.357000, 1.344000, 1.342000, 1.336000,
    1.325000, 1.312000, 1.296000, 1.277000, 1.255000, 1.232000,
    1.212000,
];

const JC_CU_EV: [f64; 49] = [
    0.640084, 0.770088, 0.890052, 1.019607, 1.139561, 1.260002,
    1.389957, 1.509977, 1.640003, 1.759889, 1.879973, 2.010120,
    2.129947, 2.260011, 2.380192, 2.500185, 2.630127, 2.749705,
    2.880005, 2.999860, 3.119884, 3.249913, 3.370052, 3.500401,
    3.619977, 3.740096, 3.869669, 3.990480, 4.120445, 4.240226,
    4.359501, 4.490554, 4.610792, 4.739457, 4.860219, 4.979285,
    5.110643, 5.229194, 5.360320, 5.481176, 5.600009, 5.729399,
    5.851071, 5.980907, 6.098583, 6.220983, 6.348397, 6.470992,
    6.598414,
];
const JC_CU_N: [f64; 49] = [
    1.090000, 0.760000, 0.600000, 0.480000, 0.360000, 0.320000,
    0.300000, 0.260000, 0.240000, 0.210000, 0.220000, 0.300000,
    0.700000, 1.020000, 1.180000, 1.220000, 1.250000, 1.240000,
    1.250000, 1.280000, 1.320000, 1.330000, 1.360000, 1.370000,
    1.360000, 1.340000, 1.380000, 1.380000, 1.400000, 1.420000,
    1.450000, 1.460000, 1.450000, 1.410000, 1.410000, 1.370000,
    1.340000, 1.280000, 1.230000, 1.180000, 1.130000, 1.080000,
    1.040000, 1.010000, 0.990000, 0.980000, 0.970000, 0.950000,
    0.940000,
];
const JC_CU_K: [f64; 49] = [
    13.430000, 11.120000, 9.439000, 8.245000, 7.217000, 6.421000,
    5.768000, 5.180000, 4.665000, 4.205000, 3.747000, 3.205000,
    2.704000, 2.577000, 2.608000, 2.564000, 2.483000, 2.397000,
    2.305000, 2.207000, 2.116000, 2.045000, 1.975000, 1.916000,
    1.864000, 1.821000, 1.783000, 1.729000, 1.679000, 1.633000,
    1.633000, 1.646000, 1.668000, 1.691000, 1.741000, 1.783000,
    1.799000, 1.802000, 1.792000, 1.768000, 1.737000, 1.699000,
    1.651000, 1.599000, 1.550000, 1.493000, 1.440000, 1.388000,
    1.337000,
];

// ============================================================================
// Struct and methods
// ============================================================================

/// Tabulated real-axis optical data (n, k measurements from experiment).
///
/// Holds experimental data for the real and imaginary parts of the refractive
/// index n + i*k. Used with `epsilon_imaginary_kk()` to compute the imaginary-
/// axis dielectric function needed by the Lifshitz formula for Casimir forces.
///
/// # Physical Conventions
///
/// The dielectric function: eps(omega) = (n + i*k)^2 = n^2 - k^2 + 2*n*k*i
/// Loss function (imaginary part): eps2(omega) = 2*n(omega)*k(omega)
///
/// # Note on Data Range
///
/// The J&C data covers 0.64-6.60 eV. The KK transform integrates over all
/// positive frequencies using:
///   - Drude or 1/omega extrapolation below 0.64 eV (metallic IR tail)
///   - Power-law 1/omega^3 extrapolation above 6.60 eV (UV free-electron limit)
#[derive(Debug, Clone)]
pub struct TabulatedNK {
    /// Human-readable source reference.
    pub source: &'static str,
    /// Photon energies in eV, monotonically ascending.
    pub energy_ev: &'static [f64],
    /// Real part of the refractive index n(E).
    pub n: &'static [f64],
    /// Extinction coefficient k(E) (imaginary part of the refractive index).
    pub k: &'static [f64],
}

impl TabulatedNK {
    /// Imaginary part of dielectric function: eps2(E) = 2*n*k at energy E (eV).
    ///
    /// Uses linear interpolation between tabulated points. Clamps to endpoint values
    /// outside the tabulated range (safe for integration purposes).
    pub fn eps2_at_ev(&self, e_ev: f64) -> f64 {
        let n_val = interp_linear(self.energy_ev, self.n, e_ev);
        let k_val = interp_linear(self.energy_ev, self.k, e_ev);
        2.0 * n_val * k_val
    }

    /// Compute epsilon(i*xi) via Kramers-Kronig transform.
    ///
    /// # Parameters
    ///
    /// - `xi_rads`: Imaginary frequency in rad/s. Must be > 0.
    ///   (Corresponds to Matsubara frequencies xi_n = 2*pi*n*kT/hbar, n >= 1.)
    /// - `drude_ir`: Optional Drude parameters for the IR tail (omega < omega_min).
    ///   If None, uses 1/omega power-law extrapolation scaled to eps2(omega_min).
    ///   For metals, passing Drude parameters gives a more physical IR tail.
    ///
    /// # Mathematics
    ///
    /// eps(i*xi) = 1 + (2/pi) * I
    /// where I = I_tab + I_UV + I_IR
    ///
    /// I_tab: trapezoidal integration over [omega_min, omega_max]
    ///
    /// I_UV (omega > omega_max): 1/omega^3 extrapolation.
    ///   eps2(omega) = eps2_max * (omega_max/omega)^3
    ///   Analytical: eps2_max * omega_max^3 / xi^2 * [1/omega_max - (pi/2 - arctan(omega_max/xi))/xi]
    ///
    /// I_IR (omega < omega_min, Drude case):
    ///   eps2_Drude(omega) = omega_p^2 * gamma / (omega * (omega^2 + gamma^2))
    ///   Analytical: omega_p^2 * gamma / (xi^2 - gamma^2) *
    ///               [arctan(omega_min/gamma)/gamma - arctan(omega_min/xi)/xi]
    ///
    /// I_IR (omega < omega_min, 1/omega extrapolation):
    ///   eps2(omega) ~ eps2_min * omega_min / omega
    ///   Analytical: eps2_min * omega_min / xi * arctan(omega_min / xi)
    pub fn epsilon_imaginary_kk(&self, xi_rads: f64, drude_ir: Option<&DrudeParams>) -> f64 {
        debug_assert!(xi_rads > 0.0, "epsilon_imaginary_kk requires xi > 0");
        let xi = xi_rads.max(1.0); // guard against zero (1 rad/s << all physical scales)

        let n_pts = self.energy_ev.len();
        let omega_min = self.energy_ev[0] * EV_TO_RADS;
        let omega_max = self.energy_ev[n_pts - 1] * EV_TO_RADS;

        // Build eps2 = 2nk and omega arrays in rad/s
        let omega_tab: Vec<f64> = self.energy_ev.iter().map(|&e| e * EV_TO_RADS).collect();
        let eps2_tab: Vec<f64> = (0..n_pts).map(|i| 2.0 * self.n[i] * self.k[i]).collect();
        let eps2_min = eps2_tab[0];
        let eps2_max = eps2_tab[n_pts - 1];

        // --- 1. Trapezoidal integral over tabulated range ---
        let mut integral = 0.0;
        for i in 0..n_pts - 1 {
            let w0 = omega_tab[i];
            let w1 = omega_tab[i + 1];
            let f0 = w0 * eps2_tab[i] / (w0 * w0 + xi * xi);
            let f1 = w1 * eps2_tab[i + 1] / (w1 * w1 + xi * xi);
            integral += 0.5 * (f0 + f1) * (w1 - w0);
        }

        // --- 2. UV tail: eps2(omega) = eps2_max * (omega_max/omega)^3 for omega > omega_max ---
        //
        // Integral_omega_max^inf (omega / omega^2) * eps2_max*(omega_max/omega)^3 / (1 + (xi/omega)^2) domega
        //    = eps2_max * omega_max^3 * integral_omega_max^inf 1/(omega^2 * (omega^2 + xi^2)) domega
        //
        // Using 1/(omega^2*(omega^2+xi^2)) = (1/xi^2)*(1/omega^2 - 1/(omega^2+xi^2)):
        //    = eps2_max * omega_max^3 / xi^2 * [1/omega_max - arctan(omega_max/xi)/xi - pi/2/xi + arctan(omega_max/xi)/xi]
        // Wait, let me redo:
        //    integral_a^inf [1/omega^2 - 1/(omega^2+xi^2)] domega
        //    = [-1/omega]_a^inf - [arctan(omega/xi)/xi]_a^inf
        //    = 1/a - (pi/(2*xi) - arctan(a/xi)/xi)
        //    = 1/a - pi/(2*xi) + arctan(a/xi)/xi
        let r_uv = omega_max / xi;
        let uv_tail = eps2_max * omega_max.powi(3) / (xi * xi)
            * (1.0 / omega_max - PI / (2.0 * xi) + r_uv.atan() / xi);
        integral += uv_tail;

        // --- 3. IR tail ---
        let ir_tail = if let Some(drude) = drude_ir {
            // Drude: eps2_Drude(omega) = omega_p^2 * gamma / (omega * (omega^2 + gamma^2))
            // integral_0^{omega_min} omega * eps2_Drude / (omega^2 + xi^2) domega
            //   = omega_p^2 * gamma * integral_0^{omega_min} 1/((omega^2+gamma^2)(omega^2+xi^2)) domega
            //
            // Partial fractions (gamma != xi):
            //   1/((omega^2+gamma^2)(omega^2+xi^2)) = 1/(xi^2-gamma^2) * [1/(omega^2+gamma^2) - 1/(omega^2+xi^2)]
            // integral_0^x dw/(w^2+c^2) = arctan(x/c)/c
            //
            // => omega_p^2 * gamma / (xi^2-gamma^2) * [arctan(om/gamma)/gamma - arctan(om/xi)/xi]
            let op = drude.omega_p_ev * EV_TO_RADS;
            let gam = drude.gamma_ev * EV_TO_RADS;
            let g2 = gam * gam;
            let x2 = xi * xi;
            if (g2 - x2).abs() > 1e-8 * (g2 + x2) {
                op * op * gam / (x2 - g2)
                    * ((omega_min / gam).atan() / gam - (omega_min / xi).atan() / xi)
            } else {
                // gamma ~ xi: use the equal-argument limit via L'Hopital.
                // As g -> xi: d/d(xi^2-gamma^2) ... result:
                //   omega_p^2 / (2*gamma) * d/d_gamma [arctan(om/gamma)/gamma]
                //   where d/dc[arctan(x/c)/c] = -x/(c*(c^2+x^2)) - arctan(x/c)/c^2
                let x = omega_min;
                let g = gam;
                op * op * gam / (2.0 * x2)
                    * (-(x / (g * (g * g + x * x))) - (x / g).atan() / (g * g))
            }
        } else {
            // 1/omega extrapolation: eps2(omega) ~ eps2_min * omega_min / omega
            // integral_0^{omega_min} (omega * eps2_min * omega_min / omega) / (omega^2+xi^2) domega
            //   = eps2_min * omega_min * integral_0^{omega_min} 1/(omega^2+xi^2) domega
            //   = eps2_min * omega_min / xi * arctan(omega_min / xi)
            eps2_min * omega_min / xi * (omega_min / xi).atan()
        };
        integral += ir_tail;

        1.0 + (2.0 / PI) * integral
    }

    /// Number of data points.
    pub fn n_points(&self) -> usize {
        self.energy_ev.len()
    }

    /// Energy range covered [E_min, E_max] in eV.
    pub fn energy_range_ev(&self) -> (f64, f64) {
        (self.energy_ev[0], self.energy_ev[self.energy_ev.len() - 1])
    }
}

// ============================================================================
// Constructor functions
// ============================================================================

/// Johnson & Christy (1972) data for gold (Au).
///
/// 49 data points from 0.64 to 6.60 eV at room temperature.
/// Source: Phys. Rev. B 6, 4370-4379 (1972); refractiveindex.info (CC0).
pub fn gold_jc_nk() -> TabulatedNK {
    TabulatedNK {
        source: "Johnson & Christy, Phys. Rev. B 6, 4370 (1972) -- Gold (Au)",
        energy_ev: &JC_AU_EV,
        n: &JC_AU_N,
        k: &JC_AU_K,
    }
}

/// Johnson & Christy (1972) data for silver (Ag).
///
/// 49 data points from 0.64 to 6.60 eV at room temperature.
/// Source: Phys. Rev. B 6, 4370-4379 (1972); refractiveindex.info (CC0).
pub fn silver_jc_nk() -> TabulatedNK {
    TabulatedNK {
        source: "Johnson & Christy, Phys. Rev. B 6, 4370 (1972) -- Silver (Ag)",
        energy_ev: &JC_AG_EV,
        n: &JC_AG_N,
        k: &JC_AG_K,
    }
}

/// Johnson & Christy (1972) data for copper (Cu).
///
/// 49 data points from 0.64 to 6.60 eV at room temperature.
/// Source: Phys. Rev. B 6, 4370-4379 (1972); refractiveindex.info (CC0).
pub fn copper_jc_nk() -> TabulatedNK {
    TabulatedNK {
        source: "Johnson & Christy, Phys. Rev. B 6, 4370 (1972) -- Copper (Cu)",
        energy_ev: &JC_CU_EV,
        n: &JC_CU_N,
        k: &JC_CU_K,
    }
}

/// Get J&C 1972 tabulated data by material name.
///
/// Returns Some(TabulatedNK) for "gold"/"au", "silver"/"ag", "copper"/"cu".
/// Returns None for other materials (parametric model only).
pub fn get_tabulated_nk(name: &str) -> Option<TabulatedNK> {
    match name.to_lowercase().as_str() {
        "gold" | "au" => Some(gold_jc_nk()),
        "silver" | "ag" => Some(silver_jc_nk()),
        "copper" | "cu" => Some(copper_jc_nk()),
        _ => None,
    }
}

// ============================================================================
// Lifshitz formula using tabulated KK data
// ============================================================================

/// Fresnel reflection coefficient r_TE for imaginary frequencies.
///
/// p is the rescaled lateral wavevector (p >= 1), eps is the dielectric function.
/// s = sqrt(p^2 + eps - 1), r_TE = (p - s) / (p + s).
#[inline]
fn tab_r_te(p: f64, eps: f64) -> f64 {
    let s = (p * p + eps - 1.0).max(0.0).sqrt();
    (p - s) / (p + s)
}

/// Fresnel reflection coefficient r_TM for imaginary frequencies.
///
/// r_TM = (eps*p - s) / (eps*p + s).
#[inline]
fn tab_r_tm(p: f64, eps: f64) -> f64 {
    let s = (p * p + eps - 1.0).max(0.0).sqrt();
    (eps * p - s) / (eps * p + s)
}

/// Casimir energy per unit area using tabulated KK n,k data and the Lifshitz formula.
///
/// Implements the full Lifshitz formula at finite temperature T using Matsubara
/// imaginary frequencies xi_n = 2*pi*n*kT/hbar (n = 0, 1, 2, ...).
///
/// For each n>=1 Matsubara frequency, eps(i*xi_n) is computed via the KK transform
/// of the tabulated real-axis n,k data, then fed into the Lifshitz p-integral.
/// The n=0 quasi-static term uses the Drude convention (r_TE=0 at xi=0).
///
/// This implementation mirrors `casimir_lifshitz_energy()` in optical_database.rs
/// but replaces `DrudeLorentzParams::epsilon_imaginary()` with `TabulatedNK::epsilon_imaginary_kk()`.
///
/// # Parameters
///
/// - `nk1`, `nk2`: Tabulated n,k data for the two plates.
/// - `drude_ir1`, `drude_ir2`: Drude parameters for IR tail (recommended for metals,
///   covering the Drude free-electron contribution below 0.64 eV).
/// - `separation_m`: Gap between the plates in meters.
/// - `temperature_k`: Temperature in Kelvin (300 for room temperature).
/// - `n_matsubara`: Number of Matsubara terms (500 typical for convergence at room T).
/// - `n_gauss`: Gauss-Legendre points for the p integration (48 typical).
///
/// # Returns
///
/// Energy per unit area in J/m^2. Negative value indicates attraction.
#[allow(clippy::too_many_arguments)]
pub fn casimir_lifshitz_energy_tabulated(
    nk1: &TabulatedNK,
    nk2: &TabulatedNK,
    drude_ir1: Option<&DrudeParams>,
    drude_ir2: Option<&DrudeParams>,
    separation_m: f64,
    temperature_k: f64,
    n_matsubara: usize,
    n_gauss: usize,
) -> f64 {
    let d = separation_m;
    // kT in Joules: kT[eV] * q[C/eV] = kT[J]
    let k_b_t_si = K_B_EV * temperature_k * E_CHARGE;
    let global_pref = k_b_t_si / (4.0 * PI * PI);
    // xi_1 = 2*pi*kT/hbar in rad/s
    let xi_unit = 2.0 * PI * K_B_EV * temperature_k * EV_TO_RADS;

    let gl = GaussLegendre::new(n_gauss).expect("GL degree must be >= 1");
    let mut energy = 0.0_f64;

    // ------------------------------------------------------------------
    // n=0 quasi-static term (Drude: r_TE=0, TM uses static permittivity)
    // E_n0 = global_pref * (1/2) * (1/(4d^2)) * integral_0^{u_max} u * ln(1 - r_TM1*r_TM2*e^{-u}) du
    // Variable u = 2*k_perp*d, k_perp dk_perp = u/(4d^2) du.
    // ------------------------------------------------------------------
    {
        // xi_dc = 1 rad/s << any physical frequency (~10^14 rad/s for gamma_Au)
        // gives eps(i*xi_dc) ~ eps_static (effectively the DC limit).
        let xi_dc = 1.0_f64;
        let eps1_dc = nk1.epsilon_imaginary_kk(xi_dc, drude_ir1);
        let eps2_dc = nk2.epsilon_imaginary_kk(xi_dc, drude_ir2);
        // Static TM: r_TM = (eps_s - 1)/(eps_s + 1). Metals: eps_s -> inf, r_TM -> 1.
        let r_tm1 = ((eps1_dc - 1.0) / (eps1_dc + 1.0)).clamp(0.0, 1.0);
        let r_tm2 = ((eps2_dc - 1.0) / (eps2_dc + 1.0)).clamp(0.0, 1.0);
        let r_prod = r_tm1 * r_tm2;
        let u_max = 40.0_f64;
        let n0_int = gl.integrate(0.0, u_max, |u| {
            let g = 1.0 - r_prod * (-u).exp();
            if g > 0.0 { u * g.ln() } else { 0.0 }
        });
        // n=0 gets weight 1/2 in the Matsubara sum (prime notation).
        energy += global_pref * 0.5 * n0_int / (4.0 * d * d);
    }

    // ------------------------------------------------------------------
    // n >= 1 Matsubara terms
    // I_n = (xi_n/c)^2 * integral_1^{p_max} p * [f_TE + f_TM] dp
    // ------------------------------------------------------------------
    for n in 1..=n_matsubara {
        let xi_n = n as f64 * xi_unit;
        // Exponential at p=1: if > e^{-100} the term is negligible.
        let decay_1 = 2.0 * xi_n * d / C;
        if decay_1 > 100.0 {
            break;
        }
        let eps1 = nk1.epsilon_imaginary_kk(xi_n, drude_ir1).max(1.0);
        let eps2 = nk2.epsilon_imaginary_kk(xi_n, drude_ir2).max(1.0);
        // Truncate p-integral: integrand ~ exp(-decay_1*p) < e^{-40} when p > 1 + 40/decay_1.
        let p_max = (1.0 + 40.0 / decay_1.max(1e-10)).min(1.0e4_f64);
        let pref_n = (xi_n / C) * (xi_n / C);
        let int_n = gl.integrate(1.0, p_max, |p| {
            let r_te1 = tab_r_te(p, eps1);
            let r_te2 = tab_r_te(p, eps2);
            let r_tm1 = tab_r_tm(p, eps1);
            let r_tm2 = tab_r_tm(p, eps2);
            let decay = (-decay_1 * p).exp();
            let g_te = 1.0 - r_te1 * r_te2 * decay;
            let g_tm = 1.0 - r_tm1 * r_tm2 * decay;
            let f_te = if g_te > 0.0 { g_te.ln() } else { 0.0 };
            let f_tm = if g_tm > 0.0 { g_tm.ln() } else { 0.0 };
            p * (f_te + f_tm) // correct p dp measure
        });
        energy += global_pref * pref_n * int_n;
    }

    energy
}

// ============================================================================
// Physical properties helper
// ============================================================================

/// Hardness and mechanical properties for a material.
///
/// Data from CRC Handbook of Chemistry and Physics, Kittel, and primary literature.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalProperties {
    /// Mohs hardness scale (1=talc, 10=diamond).
    pub mohs_hardness: Option<f64>,
    /// Debye temperature in Kelvin (phonon cutoff frequency).
    pub debye_temperature_k: Option<f64>,
    /// Bulk (compression) modulus in GPa.
    pub bulk_modulus_gpa: Option<f64>,
    /// Density in g/cm^3 at room temperature.
    pub density_g_cm3: Option<f64>,
    /// Lattice constant in Angstroms (a-axis; see crystal_notes for HCP/tetragonal).
    pub lattice_constant_ang: Option<f64>,
}

/// Return well-established physical properties for named elements/compounds.
///
/// Data sources: CRC Handbook (2023), Kittel "Introduction to Solid State Physics",
/// NIST webbook, and material-specific references noted in comments.
///
/// Returns None for materials without well-established property data in the database.
pub fn get_physical_properties(name: &str) -> Option<PhysicalProperties> {
    let name_lower = name.to_lowercase();
    match name_lower.as_str() {
        // ----------------------------------------------------------------
        // Noble metals
        // ----------------------------------------------------------------
        "gold" | "au" => Some(PhysicalProperties {
            mohs_hardness: Some(2.5),
            debye_temperature_k: Some(165.0),
            bulk_modulus_gpa: Some(180.0),
            density_g_cm3: Some(19.32),
            lattice_constant_ang: Some(4.079), // FCC a
        }),
        "silver" | "ag" => Some(PhysicalProperties {
            mohs_hardness: Some(2.5),
            debye_temperature_k: Some(225.0),
            bulk_modulus_gpa: Some(100.0),
            density_g_cm3: Some(10.49),
            lattice_constant_ang: Some(4.086), // FCC a
        }),
        "copper" | "cu" => Some(PhysicalProperties {
            mohs_hardness: Some(3.0),
            debye_temperature_k: Some(343.0),
            bulk_modulus_gpa: Some(140.0),
            density_g_cm3: Some(8.96),
            lattice_constant_ang: Some(3.615), // FCC a
        }),
        // ----------------------------------------------------------------
        // Common metals
        // ----------------------------------------------------------------
        "aluminum" | "al" => Some(PhysicalProperties {
            mohs_hardness: Some(2.75),
            debye_temperature_k: Some(428.0),
            bulk_modulus_gpa: Some(76.0),
            density_g_cm3: Some(2.70),
            lattice_constant_ang: Some(4.046), // FCC a
        }),
        "beryllium" | "be" => Some(PhysicalProperties {
            mohs_hardness: Some(5.5),
            debye_temperature_k: Some(1440.0),
            bulk_modulus_gpa: Some(130.0),
            density_g_cm3: Some(1.85),
            lattice_constant_ang: Some(2.286), // HCP a-axis
        }),
        "chromium" | "cr" => Some(PhysicalProperties {
            mohs_hardness: Some(8.5),
            debye_temperature_k: Some(630.0),
            bulk_modulus_gpa: Some(160.0),
            density_g_cm3: Some(7.19),
            lattice_constant_ang: Some(2.885), // BCC a
        }),
        "nickel" | "ni" => Some(PhysicalProperties {
            mohs_hardness: Some(4.0),
            debye_temperature_k: Some(450.0),
            bulk_modulus_gpa: Some(180.0),
            density_g_cm3: Some(8.91),
            lattice_constant_ang: Some(3.523), // FCC a
        }),
        "palladium" | "pd" => Some(PhysicalProperties {
            mohs_hardness: Some(4.75),
            debye_temperature_k: Some(274.0),
            bulk_modulus_gpa: Some(180.0),
            density_g_cm3: Some(12.02),
            lattice_constant_ang: Some(3.890), // FCC a
        }),
        "platinum" | "pt" => Some(PhysicalProperties {
            mohs_hardness: Some(4.25),
            debye_temperature_k: Some(240.0),
            bulk_modulus_gpa: Some(230.0),
            density_g_cm3: Some(21.45),
            lattice_constant_ang: Some(3.924), // FCC a
        }),
        "titanium" | "ti" => Some(PhysicalProperties {
            mohs_hardness: Some(6.0),
            debye_temperature_k: Some(420.0),
            bulk_modulus_gpa: Some(105.0),
            density_g_cm3: Some(4.51),
            lattice_constant_ang: Some(2.951), // HCP a-axis (c=4.683)
        }),
        "tungsten" | "w" => Some(PhysicalProperties {
            mohs_hardness: Some(7.5),
            debye_temperature_k: Some(400.0),
            bulk_modulus_gpa: Some(310.0),
            density_g_cm3: Some(19.25),
            lattice_constant_ang: Some(3.165), // BCC a
        }),
        // ----------------------------------------------------------------
        // Semiconductors
        // ----------------------------------------------------------------
        "silicon" | "si" => Some(PhysicalProperties {
            mohs_hardness: Some(7.0),
            debye_temperature_k: Some(645.0),
            bulk_modulus_gpa: Some(97.6),
            density_g_cm3: Some(2.33),
            lattice_constant_ang: Some(5.431), // diamond cubic a
        }),
        "germanium" | "ge" => Some(PhysicalProperties {
            mohs_hardness: Some(6.0),
            debye_temperature_k: Some(374.0),
            bulk_modulus_gpa: Some(75.0),
            density_g_cm3: Some(5.32),
            lattice_constant_ang: Some(5.658), // diamond cubic a
        }),
        // ----------------------------------------------------------------
        // Dielectrics
        // ----------------------------------------------------------------
        "silica" | "sio2" | "glass" => Some(PhysicalProperties {
            mohs_hardness: Some(5.5),  // fused silica (amorphous)
            debye_temperature_k: Some(470.0), // approximate
            bulk_modulus_gpa: Some(37.0),
            density_g_cm3: Some(2.20),
            lattice_constant_ang: None, // amorphous
        }),
        "quartz" | "crystalline_sio2" => Some(PhysicalProperties {
            mohs_hardness: Some(7.0), // crystalline quartz
            debye_temperature_k: Some(470.0),
            bulk_modulus_gpa: Some(37.0),
            density_g_cm3: Some(2.65), // quartz crystal slightly denser than fused
            lattice_constant_ang: Some(4.913), // hexagonal a-axis
        }),
        "alumina" | "al2o3" | "sapphire" => Some(PhysicalProperties {
            mohs_hardness: Some(9.0),
            debye_temperature_k: Some(1047.0), // Kittel
            bulk_modulus_gpa: Some(250.0),
            density_g_cm3: Some(3.99),
            lattice_constant_ang: Some(4.785), // rhombohedral (hexagonal) a-axis
        }),
        "diamond" | "c_diamond" => Some(PhysicalProperties {
            mohs_hardness: Some(10.0),
            debye_temperature_k: Some(1860.0),
            bulk_modulus_gpa: Some(443.0),
            density_g_cm3: Some(3.51),
            lattice_constant_ang: Some(3.567), // diamond cubic a
        }),
        "silicon_nitride" | "si3n4" => Some(PhysicalProperties {
            mohs_hardness: Some(8.5),
            debye_temperature_k: None, // not well-tabulated for polycrystalline
            bulk_modulus_gpa: Some(240.0),
            density_g_cm3: Some(3.17),
            lattice_constant_ang: None, // polymorphic (alpha/beta)
        }),
        "tio2" | "rutile" | "titanium_dioxide" => Some(PhysicalProperties {
            mohs_hardness: Some(6.0),
            debye_temperature_k: Some(760.0),
            bulk_modulus_gpa: Some(211.0),
            density_g_cm3: Some(4.23),
            lattice_constant_ang: Some(4.594), // rutile tetragonal a-axis
        }),
        // ----------------------------------------------------------------
        // Perovskites / Titanates
        // ----------------------------------------------------------------
        "srtio3" | "strontium_titanate" => Some(PhysicalProperties {
            mohs_hardness: Some(6.0),
            debye_temperature_k: Some(495.0),
            bulk_modulus_gpa: Some(174.0),
            density_g_cm3: Some(5.12),
            lattice_constant_ang: Some(3.905), // cubic perovskite a
        }),
        "tio" | "titanium_monoxide" => Some(PhysicalProperties {
            mohs_hardness: None,
            debye_temperature_k: None,
            bulk_modulus_gpa: Some(145.0),
            density_g_cm3: Some(4.89),
            lattice_constant_ang: Some(4.177), // rocksalt a (may vary with stoichiometry)
        }),
        // ----------------------------------------------------------------
        // Tungsten oxides
        // ----------------------------------------------------------------
        "wo3" | "tungsten_trioxide" | "tungsten_oxide" => Some(PhysicalProperties {
            mohs_hardness: Some(2.5),
            debye_temperature_k: None,
            bulk_modulus_gpa: Some(28.0),
            density_g_cm3: Some(7.16),
            lattice_constant_ang: Some(7.309), // monoclinic a-axis
        }),
        "cawo4" | "calcium_tungstate" | "scheelite_ca" => Some(PhysicalProperties {
            mohs_hardness: Some(4.5),
            debye_temperature_k: Some(195.0), // Nikl et al.
            bulk_modulus_gpa: Some(72.0),
            density_g_cm3: Some(6.06),
            lattice_constant_ang: Some(5.243), // tetragonal a-axis
        }),
        "pbwo4" | "lead_tungstate" | "scheelite_pb" => Some(PhysicalProperties {
            mohs_hardness: Some(2.75),
            debye_temperature_k: None,
            bulk_modulus_gpa: Some(68.0),
            density_g_cm3: Some(8.23),
            lattice_constant_ang: Some(5.456), // tetragonal a-axis
        }),
        _ => None,
    }
}

// ============================================================================
// Private helpers
// ============================================================================

/// Linear interpolation on a sorted ascending grid.
///
/// Clamps to ys[0] below xs[0] and ys[n-1] above xs[n-1].
fn interp_linear(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let i = xs.partition_point(|&xi| xi <= x).saturating_sub(1).min(n - 2);
    let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] * (1.0 - t) + ys[i + 1] * t
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const AU_DRUDE: DrudeParams = DrudeParams { omega_p_ev: 9.0, gamma_ev: 0.072, eps_inf: 1.0 };
    const AG_DRUDE: DrudeParams = DrudeParams { omega_p_ev: 9.01, gamma_ev: 0.021, eps_inf: 1.0 };
    // xi_1 at 300 K: 2*pi * kT/hbar = 2*pi * (8.617e-5 eV * EV_TO_RADS) * 300
    // = 2*pi * 300 * 8.617e-5 * 1.519e15 rad/s ~ 2.46e13 rad/s
    const XI1_RADS: f64 = 2.0 * PI * 300.0 * 8.617_333_262e-5 * 1.519_267_447e15;

    #[test]
    fn test_kk_au_large_eps_at_xi1() {
        // Au at xi_1 (first Matsubara ~0.16 eV at 300K) should have eps >> 1 due to Drude
        let au = gold_jc_nk();
        let eps = au.epsilon_imaginary_kk(XI1_RADS, Some(&AU_DRUDE));
        assert!(eps > 50.0, "Au eps(i*xi_1) should be >> 1 (metallic Drude), got {eps:.2}");
    }

    #[test]
    fn test_kk_au_monotone_decreasing() {
        // eps(i*xi) must be monotonically decreasing with xi (dispersion relations)
        let au = gold_jc_nk();
        let xi_evs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let eps: Vec<f64> = xi_evs
            .iter()
            .map(|&xi_ev| au.epsilon_imaginary_kk(xi_ev * EV_TO_RADS, Some(&AU_DRUDE)))
            .collect();
        for i in 0..eps.len() - 1 {
            assert!(
                eps[i] >= eps[i + 1],
                "eps(i*xi) not monotone at index {i}: {:.4} vs {:.4}",
                eps[i], eps[i + 1]
            );
        }
    }

    #[test]
    fn test_kk_approaches_one_at_high_xi() {
        // At very high xi >> omega_p, all polarization screens out: eps(i*xi) -> 1
        let au = gold_jc_nk();
        let xi_high = 500.0 * EV_TO_RADS; // 500 eV >> 9 eV plasma
        let eps = au.epsilon_imaginary_kk(xi_high, Some(&AU_DRUDE));
        assert!(eps < 1.05, "At 500 eV, Au eps(i*xi) should be ~1, got {eps:.4}");
    }

    #[test]
    fn test_kk_ag_vs_au_higher_at_low_xi() {
        // Ag has lower gamma (0.021 eV) vs Au (0.072 eV), so larger Drude contribution
        // Ag should have LARGER eps(i*xi_1) than Au (less scattering -> larger DC eps)
        let au = gold_jc_nk();
        let ag = silver_jc_nk();
        let eps_au = au.epsilon_imaginary_kk(XI1_RADS, Some(&AU_DRUDE));
        let eps_ag = ag.epsilon_imaginary_kk(XI1_RADS, Some(&AG_DRUDE));
        assert!(
            eps_ag > eps_au,
            "Ag should have higher eps(i*xi_1) than Au (lower gamma), got Ag={eps_ag:.1} Au={eps_au:.1}"
        );
    }

    #[test]
    fn test_eps2_at_ev_interpolation() {
        let au = gold_jc_nk();
        // At 0.64 eV: n=0.92, k=13.78 -> eps2 = 2*0.92*13.78 = 25.36
        let eps2 = au.eps2_at_ev(0.640084);
        let expected = 2.0 * 0.92 * 13.78;
        assert!((eps2 - expected).abs() < 0.01, "eps2 at 0.64 eV: got {eps2:.4} expected {expected:.4}");
    }

    #[test]
    fn test_interp_linear_boundary_cases() {
        let xs = [1.0_f64, 2.0, 3.0];
        let ys = [10.0_f64, 20.0, 30.0];
        assert_eq!(interp_linear(&xs, &ys, 0.5), 10.0, "below min -> clamp");
        assert_eq!(interp_linear(&xs, &ys, 4.0), 30.0, "above max -> clamp");
        let mid = interp_linear(&xs, &ys, 1.5);
        assert!((mid - 15.0).abs() < 1e-10, "midpoint interpolation");
        let mid2 = interp_linear(&xs, &ys, 2.5);
        assert!((mid2 - 25.0).abs() < 1e-10, "second midpoint");
    }

    #[test]
    fn test_casimir_jc_au_negative_and_finite() {
        // Casimir energy should be negative (attractive) and physically reasonable
        let au = gold_jc_nk();
        let d = 100e-9_f64;
        let e = casimir_lifshitz_energy_tabulated(
            &au, &au, Some(&AU_DRUDE), Some(&AU_DRUDE),
            d, 300.0, 100, 24,
        );
        assert!(e < 0.0, "Casimir energy must be attractive (negative), got {e:.3e}");
        // Ideal: -hbar*c*pi^2/(720*d^3) ~ -1.3e-3 J/m^2 at 100 nm
        assert!(e > -1.0e-2, "Should not exceed 10x ideal, got {e:.3e}");
    }

    #[test]
    fn test_physical_properties_gold() {
        let props = get_physical_properties("gold").unwrap();
        assert!((props.mohs_hardness.unwrap() - 2.5).abs() < 0.1);
        assert!((props.debye_temperature_k.unwrap() - 165.0).abs() < 5.0);
        assert!((props.density_g_cm3.unwrap() - 19.32).abs() < 0.1);
    }

    #[test]
    fn test_physical_properties_diamond() {
        let props = get_physical_properties("diamond").unwrap();
        assert_eq!(props.mohs_hardness, Some(10.0));
        assert!(props.debye_temperature_k.unwrap() > 1800.0);
        assert!(props.bulk_modulus_gpa.unwrap() > 400.0);
    }
}
