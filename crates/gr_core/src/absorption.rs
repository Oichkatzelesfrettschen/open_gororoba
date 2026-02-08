//! Absorption models for radiative transfer in astrophysical plasmas.
//!
//! Three absorption mechanisms relevant to black hole environments:
//!
//! 1. Synchrotron self-absorption (SSA): alpha ~ B^2 / nu^2
//!    Dominates at low radio frequencies in magnetized plasma.
//!
//! 2. Free-free (bremsstrahlung): alpha ~ n_e^2 T^{-3/2} / nu^2
//!    Dominates in hot ionized gas at intermediate frequencies.
//!
//! 3. Compton scattering: alpha ~ n_e * sigma (Thomson or Klein-Nishina)
//!    Dominates at X-ray/gamma-ray energies.
//!
//! References:
//!   - Rybicki & Lightman (1979): Radiative Processes in Astrophysics, Ch. 5-7
//!   - Longair (2011): High Energy Astrophysics, Ch. 6

use crate::constants::*;
use std::f64::consts::PI;

/// Planck constant [erg s].
const H_PLANCK: f64 = 6.626_070_15e-27;

/// Compton wavelength of the electron [cm].
const LAMBDA_COMPTON: f64 = 2.426_310_239e-10;

// ============================================================================
// Synchrotron self-absorption
// ============================================================================

/// Synchrotron self-absorption (SSA) coefficient [cm^-1].
///
/// alpha_ssa = (n_e sigma_T / 2) (nu_c / nu)^2 * relativistic_correction
///
/// where nu_c = eB/(2 pi m_e c) is the cyclotron frequency and the
/// relativistic correction is (1 + 2.4 theta) with theta = kT/(m_e c^2).
///
/// SSA dominates at low radio frequencies in magnetized plasma.
pub fn synchrotron_self_absorption(nu: f64, b_gauss: f64, n_e: f64, temp_k: f64) -> f64 {
    let nu_c = E_CHARGE_CGS * b_gauss.abs()
        / (2.0 * PI * M_ELECTRON_CGS * C_CGS);

    let alpha_ssa = (n_e * SIGMA_THOMSON / 2.0) * (nu_c * nu_c) / (nu * nu);

    // Relativistic correction: theta = kT / (m_e c^2)
    let theta = K_B_CGS * temp_k / (M_ELECTRON_CGS * C_CGS * C_CGS);
    let rel_factor = 1.0 + 2.4 * theta;

    alpha_ssa * rel_factor
}

// ============================================================================
// Free-free (bremsstrahlung) absorption
// ============================================================================

/// Free-free (bremsstrahlung) absorption coefficient [cm^-1].
///
/// alpha_ff = K_ff n_e^2 g_ff / (T^{3/2} nu^2)
///
/// where K_ff = 3.68e8 cm^5 K^{3/2} s^{-2} and g_ff is the Gaunt
/// factor approximated as ln(lambda_D / lambda_c + 1).
///
/// Assumes hydrogen plasma with n_i = n_e (quasi-neutrality).
pub fn free_free_absorption(nu: f64, n_e: f64, temp_k: f64) -> f64 {
    let k_ff = 3.68e8; // prefactor [cm^5 / K^{3/2} / s^{-2}]

    // Debye length [cm]
    let lambda_d = (K_B_CGS * temp_k
        / (4.0 * PI * n_e * E_CHARGE_CGS * E_CHARGE_CGS))
    .sqrt();

    // Approximate Gaunt factor
    let gaunt = (lambda_d / LAMBDA_COMPTON + 1.0).ln();

    let t_32 = temp_k.powf(1.5);
    let alpha_ff = k_ff * n_e * n_e * gaunt / (t_32 * nu * nu);

    alpha_ff.max(0.0)
}

// ============================================================================
// Compton scattering absorption
// ============================================================================

/// Compton absorption coefficient [cm^-1].
///
/// alpha = n_e * sigma(E) where sigma transitions from Thomson to
/// Klein-Nishina as photon energy approaches m_e c^2.
///
/// Three regimes:
/// - x < 0.1: Thomson limit, sigma = sigma_T
/// - 0.1 < x < 10: intermediate, sigma ~ sigma_T / (1 + 2.7x)
/// - x > 10: Klein-Nishina, sigma ~ sigma_T * 0.2(1 + x/2) / x
///
/// where x = h nu / (m_e c^2) is the dimensionless photon energy.
pub fn compton_absorption(nu: f64, n_e: f64) -> f64 {
    let h_nu = H_PLANCK * nu;
    let me_c2 = M_ELECTRON_CGS * C_CGS * C_CGS;
    let x = h_nu / me_c2;

    let sigma = if x < 0.1 {
        // Thomson limit
        SIGMA_THOMSON
    } else if x < 10.0 {
        // Intermediate
        SIGMA_THOMSON / (1.0 + 2.7 * x)
    } else {
        // Klein-Nishina limit
        SIGMA_THOMSON * 0.2 * (1.0 + x / 2.0) / x
    };

    n_e * sigma
}

// ============================================================================
// Combined absorption
// ============================================================================

/// Which absorption mechanism dominates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsorptionMode {
    /// Synchrotron self-absorption (low radio frequencies)
    SynchrotronSelfAbsorption,
    /// Free-free / bremsstrahlung (intermediate frequencies)
    FreeFree,
    /// Compton scattering (X-ray / gamma-ray)
    Compton,
}

/// Total absorption coefficient (SSA + free-free + Compton) [cm^-1].
pub fn total_absorption(nu: f64, b_gauss: f64, n_e: f64, temp_k: f64) -> f64 {
    synchrotron_self_absorption(nu, b_gauss, n_e, temp_k)
        + free_free_absorption(nu, n_e, temp_k)
        + compton_absorption(nu, n_e)
}

/// Identify which absorption mechanism dominates at given conditions.
pub fn dominant_mode(nu: f64, b_gauss: f64, n_e: f64, temp_k: f64) -> AbsorptionMode {
    let a_ssa = synchrotron_self_absorption(nu, b_gauss, n_e, temp_k);
    let a_ff = free_free_absorption(nu, n_e, temp_k);
    let a_comp = compton_absorption(nu, n_e);

    if a_ssa >= a_ff && a_ssa >= a_comp {
        AbsorptionMode::SynchrotronSelfAbsorption
    } else if a_ff >= a_comp {
        AbsorptionMode::FreeFree
    } else {
        AbsorptionMode::Compton
    }
}

/// Frequency where optical depth = 1 for a given path length [Hz].
///
/// Uses bisection on log(nu) to find where alpha(nu) * path_length = 1.
/// Returns the threshold frequency for the specified absorption mode.
pub fn optical_depth_threshold_frequency(
    b_gauss: f64,
    n_e: f64,
    temp_k: f64,
    path_length_cm: f64,
    mode: AbsorptionMode,
) -> f64 {
    let mut nu_low = 1e8_f64;  // 100 MHz
    let mut nu_high = 1e20_f64; // 100 EeV

    for _ in 0..60 {
        let nu_mid = (nu_low * nu_high).sqrt(); // geometric mean

        let alpha = match mode {
            AbsorptionMode::SynchrotronSelfAbsorption => {
                synchrotron_self_absorption(nu_mid, b_gauss, n_e, temp_k)
            }
            AbsorptionMode::FreeFree => free_free_absorption(nu_mid, n_e, temp_k),
            AbsorptionMode::Compton => compton_absorption(nu_mid, n_e),
        };

        let tau = alpha * path_length_cm;

        if tau > 1.0 {
            // Still opaque: threshold is at higher frequency
            nu_low = nu_mid;
        } else {
            // Already transparent: threshold is at lower frequency
            nu_high = nu_mid;
        }
    }

    (nu_low * nu_high).sqrt()
}

// ============================================================================
// Radiative transfer
// ============================================================================

/// Formal solution of the radiative transfer equation along a ray.
///
/// I_nu(s) = I_0 exp(-tau) + S_nu (1 - exp(-tau))
///
/// where tau = alpha * ds is the optical depth and S_nu = j_nu / alpha_nu
/// is the source function. For thermal emission, S_nu = B_nu(T).
///
/// Limits:
/// - Optically thin (tau << 1): I ~ I_0 + j * ds
/// - Optically thick (tau >> 1): I ~ S_nu (LTE)
pub fn radiative_transfer_step(
    i_in: f64,
    j_nu: f64,
    alpha_nu: f64,
    ds: f64,
) -> f64 {
    let tau = alpha_nu * ds;

    if tau < 1e-8 {
        // Optically thin limit: avoid numerical issues
        i_in + j_nu * ds
    } else if tau > 50.0 {
        // Optically thick: saturates to source function
        j_nu / alpha_nu
    } else {
        let exp_neg_tau = (-tau).exp();
        let source = j_nu / alpha_nu;
        i_in * exp_neg_tau + source * (1.0 - exp_neg_tau)
    }
}

/// Planck function B_nu(T) [erg s^-1 cm^-2 Hz^-1 sr^-1].
///
/// B_nu = (2 h nu^3 / c^2) / (exp(h nu / kT) - 1)
pub fn planck_function(nu: f64, temp_k: f64) -> f64 {
    let x = H_PLANCK * nu / (K_B_CGS * temp_k);
    if x > 500.0 {
        return 0.0; // Avoid overflow in exp
    }
    let numerator = 2.0 * H_PLANCK * nu * nu * nu / (C_CGS * C_CGS);
    let denominator = x.exp() - 1.0;
    if denominator <= 0.0 {
        return 0.0;
    }
    numerator / denominator
}

// ============================================================================
// Plasma frequency
// ============================================================================

/// Plasma frequency [Hz].
///
/// nu_p = (n_e * e^2 / (pi * m_e))^{1/2}
///
/// The frequency below which electromagnetic waves cannot propagate
/// through a plasma. Photons with nu < nu_p are reflected.
///
/// Typical values:
/// - Solar corona (n_e ~ 1e8 cm^-3): nu_p ~ 90 MHz
/// - ISM (n_e ~ 0.03 cm^-3): nu_p ~ 1.6 kHz
/// - Tokamak (n_e ~ 1e14 cm^-3): nu_p ~ 90 GHz
///
/// The plasma frequency also determines the dispersion delay of radio
/// pulses through ionized media: delta_t ~ DM / nu^2, where DM = integral(n_e dl).
///
/// # Arguments
/// * `n_e` - Electron number density [cm^-3]
pub fn plasma_frequency(n_e: f64) -> f64 {
    // nu_p = sqrt(n_e * e^2 / (pi * m_e))  in CGS
    (n_e * E_CHARGE_CGS * E_CHARGE_CGS / (PI * M_ELECTRON_CGS)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- SSA --

    #[test]
    fn test_ssa_positive() {
        let a = synchrotron_self_absorption(1e9, 1.0, 1e3, 1e8);
        assert!(a > 0.0);
    }

    #[test]
    fn test_ssa_scales_inversely_with_nu_squared() {
        let a1 = synchrotron_self_absorption(1e9, 1.0, 1e3, 1e8);
        let a2 = synchrotron_self_absorption(2e9, 1.0, 1e3, 1e8);
        // alpha ~ 1/nu^2 -> a1/a2 ~ 4
        assert!((a1 / a2 - 4.0).abs() < 0.01, "ratio = {}", a1 / a2);
    }

    #[test]
    fn test_ssa_scales_with_b_squared() {
        let a1 = synchrotron_self_absorption(1e9, 1.0, 1e3, 1e8);
        let a2 = synchrotron_self_absorption(1e9, 2.0, 1e3, 1e8);
        // alpha ~ B^2 (via nu_c^2)
        assert!((a2 / a1 - 4.0).abs() < 0.01, "ratio = {}", a2 / a1);
    }

    #[test]
    fn test_ssa_relativistic_correction_increases() {
        let a_cold = synchrotron_self_absorption(1e9, 1.0, 1e3, 1e6);
        let a_hot = synchrotron_self_absorption(1e9, 1.0, 1e3, 1e10);
        assert!(a_hot > a_cold, "hotter plasma should absorb more");
    }

    // -- Free-free --

    #[test]
    fn test_ff_positive() {
        let a = free_free_absorption(1e9, 1e6, 1e7);
        assert!(a > 0.0);
    }

    #[test]
    fn test_ff_scales_inversely_with_nu_squared() {
        let a1 = free_free_absorption(1e9, 1e6, 1e7);
        let a2 = free_free_absorption(2e9, 1e6, 1e7);
        let ratio = a1 / a2;
        assert!((ratio - 4.0).abs() < 0.1, "ratio = {ratio}");
    }

    #[test]
    fn test_ff_scales_with_ne_squared() {
        let a1 = free_free_absorption(1e9, 1e6, 1e7);
        let a2 = free_free_absorption(1e9, 2e6, 1e7);
        // Gaunt factor depends weakly on n_e (via Debye length), so not exact 4x
        let ratio = a2 / a1;
        assert!(ratio > 3.5 && ratio < 4.5, "ratio = {ratio}");
    }

    // -- Compton --

    #[test]
    fn test_compton_thomson_limit() {
        // Low frequency: sigma = sigma_T
        let a = compton_absorption(1e9, 1e3); // radio
        let expected = 1e3 * SIGMA_THOMSON;
        assert!((a - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_compton_kn_suppression() {
        // Very high frequency: Klein-Nishina suppresses cross-section
        let a_low = compton_absorption(1e9, 1e3); // radio
        let a_high = compton_absorption(1e21, 1e3); // gamma-ray
        assert!(a_high < a_low, "KN should suppress absorption");
    }

    #[test]
    fn test_compton_scales_linearly_with_ne() {
        let a1 = compton_absorption(1e9, 1e3);
        let a2 = compton_absorption(1e9, 2e3);
        assert!((a2 / a1 - 2.0).abs() < 1e-10);
    }

    // -- Total / mode --

    #[test]
    fn test_total_equals_sum() {
        let nu = 1e12;
        let b = 10.0;
        let ne = 1e8;
        let t = 1e9;
        let total = total_absorption(nu, b, ne, t);
        let sum = synchrotron_self_absorption(nu, b, ne, t)
            + free_free_absorption(nu, ne, t)
            + compton_absorption(nu, ne);
        assert!((total - sum).abs() < 1e-30);
    }

    #[test]
    fn test_ssa_dominates_low_freq() {
        // Very strong B-field (1e4 G), low density (1 cm^-3), hot plasma (1e10 K).
        // SSA scales as B^2 * n_e, FF scales as n_e^2 / T^{3/2}.
        // Low n_e suppresses FF's n_e^2 term; high T further suppresses FF;
        // high B boosts SSA via nu_c^2 ~ B^2.
        let mode = dominant_mode(1e8, 1e4, 1.0, 1e10);
        assert_eq!(mode, AbsorptionMode::SynchrotronSelfAbsorption);
    }

    #[test]
    fn test_compton_dominates_high_freq() {
        // High frequency, weak B -> Compton dominates
        let mode = dominant_mode(1e18, 0.001, 1e8, 1e6);
        assert_eq!(mode, AbsorptionMode::Compton);
    }

    // -- Optical depth threshold --

    #[test]
    fn test_threshold_frequency_reasonable() {
        // Strong B (1000 G) and high density (1e8 cm^-3) with long path (1e15 cm)
        // ensure the SSA optical depth exceeds 1 at low frequencies and falls
        // below 1 at high frequencies, so the bisection converges.
        let nu_th = optical_depth_threshold_frequency(
            1000.0, 1e8, 1e9, 1e15,
            AbsorptionMode::SynchrotronSelfAbsorption,
        );
        // Should be between 100 MHz and 100 GHz
        assert!(nu_th > 1e8 && nu_th < 1e11, "nu_th = {nu_th}");
    }

    // -- Radiative transfer --

    #[test]
    fn test_rt_optically_thin() {
        // tau << 1: I ~ I_0 + j * ds
        let i_out = radiative_transfer_step(1.0, 0.01, 1e-10, 1.0);
        assert!((i_out - 1.01).abs() < 0.001, "I = {i_out}");
    }

    #[test]
    fn test_rt_optically_thick() {
        // tau >> 1: I -> j/alpha (source function)
        let i_out = radiative_transfer_step(0.0, 1e5, 1.0, 100.0);
        let source = 1e5 / 1.0;
        assert!((i_out - source).abs() / source < 0.01, "I = {i_out}");
    }

    #[test]
    fn test_rt_kirchhoff() {
        // Kirchhoff's law: in thermal equilibrium, I_nu = B_nu(T)
        // If j_nu = alpha_nu * B_nu, then the solution should approach B_nu
        let t = 1e7; // 10 million K
        let nu = 1e14; // infrared
        let b_nu = planck_function(nu, t);
        let alpha = 1.0;
        let j_nu = alpha * b_nu;
        let i_out = radiative_transfer_step(0.0, j_nu, alpha, 100.0);
        assert!((i_out - b_nu).abs() / b_nu < 0.01, "Kirchhoff violated");
    }

    // -- Planck function --

    #[test]
    fn test_planck_positive() {
        let b = planck_function(1e14, 1e4);
        assert!(b > 0.0);
    }

    #[test]
    fn test_planck_rayleigh_jeans() {
        // Low frequency limit: B_nu ~ 2 nu^2 kT / c^2
        let nu = 1e8; // 100 MHz, at T=1e4 K: h*nu/kT ~ 5e-13, very small
        let t = 1e4;
        let b = planck_function(nu, t);
        let rj = 2.0 * nu * nu * K_B_CGS * t / (C_CGS * C_CGS);
        assert!((b - rj).abs() / rj < 0.01, "B = {b}, RJ = {rj}");
    }

    #[test]
    fn test_planck_peak_wien() {
        // Wien's displacement law: nu_peak ~ 5.879e10 * T
        // At T = 5778 K (Sun): nu_peak ~ 3.4e14 Hz
        let t = 5778.0;
        let nu_peak = 5.879e10 * t;
        // B_nu should be near maximum around this frequency
        let b_peak = planck_function(nu_peak, t);
        let b_low = planck_function(nu_peak / 10.0, t);
        let b_high = planck_function(nu_peak * 10.0, t);
        assert!(b_peak > b_low && b_peak > b_high);
    }

    // -- Plasma frequency --

    #[test]
    fn test_plasma_frequency_solar_corona() {
        // Solar corona: n_e ~ 1e8 cm^-3, nu_p ~ 90 MHz
        let nu_p = plasma_frequency(1e8);
        assert!(
            nu_p > 80e6 && nu_p < 100e6,
            "solar corona nu_p = {nu_p:.2e}, expected ~90 MHz"
        );
    }

    #[test]
    fn test_plasma_frequency_ism() {
        // ISM: n_e ~ 0.03 cm^-3, nu_p ~ 1.6 kHz
        let nu_p = plasma_frequency(0.03);
        assert!(
            nu_p > 1.0e3 && nu_p < 2.0e3,
            "ISM nu_p = {nu_p:.2e}, expected ~1.6 kHz"
        );
    }

    #[test]
    fn test_plasma_frequency_scaling() {
        // nu_p ~ sqrt(n_e), so quadrupling density doubles frequency
        let nu1 = plasma_frequency(1e6);
        let nu4 = plasma_frequency(4e6);
        assert!(
            (nu4 / nu1 - 2.0).abs() < 1e-10,
            "nu_p should scale as sqrt(n_e): ratio = {}",
            nu4 / nu1
        );
    }

    #[test]
    fn test_plasma_frequency_positive() {
        assert!(plasma_frequency(1.0) > 0.0);
        assert!(plasma_frequency(1e20) > 0.0);
    }
}
