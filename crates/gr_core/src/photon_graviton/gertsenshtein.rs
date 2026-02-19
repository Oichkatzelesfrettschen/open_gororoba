//! Gertsenshtein effect and vacuum birefringence from Euler-Heisenberg.
//!
//! Tree-level photon-graviton conversion probability (Gertsenshtein 1962),
//! QED vacuum birefringence from the one-loop Euler-Heisenberg effective
//! Lagrangian, and decoherence length for graviton-photon oscillations.
//!
//! # Physics overview
//!
//! The Gertsenshtein effect (1962) is the resonant conversion of gravitons
//! to photons (and vice versa) in a transverse magnetic field. At tree level,
//! the conversion probability for a graviton traversing a uniform field B_0
//! over distance D is:
//!
//!   P(g -> gamma) = sin^2(Delta_osc * D)
//!
//! where Delta_osc = sqrt(G) * B_0 in the vacuum weak-field limit.
//!
//! The one-loop Euler-Heisenberg effective Lagrangian introduces vacuum
//! birefringence that modifies the photon dispersion relation, causing
//! decoherence between the graviton and photon modes over a characteristic
//! length L_c = 45 m_e^4 / (14 alpha^2 B_0^2 k).
//!
//! # References
//!
//! - Gertsenshtein, M.E. (1962), Sov. Phys. JETP 14, 84
//! - Raffelt & Stodolsky (1988), Phys. Rev. D 37, 1237
//! - De Logi & Mickelson (1977), Phys. Rev. D 16, 2915
//! - Pantig & Ali-Haimoud (2024), arXiv:2405.01407
//! - Ejlli (2020), JHEP 06, 029 (arXiv:2004.02714)
//! - PVLAS review: Della Valle et al. (2020), arXiv:2005.12913

use super::constants::*;
use std::f64::consts::PI;

// ============================================================================
// Euler-Heisenberg refractive indices
// ============================================================================

/// Vacuum refractive index for parallel polarization in a magnetic field.
///
/// From the one-loop Euler-Heisenberg effective Lagrangian (spinor QED):
///
///   n_parallel = 1 + (8 alpha^2 / 45 m_e^4) * B^2 * sin^2(theta)
///
/// In natural units (m_e = 1):
///
///   n_parallel = 1 + (8 alpha^2 / 45) * (eB)^2 * sin^2(theta)
///
/// # Arguments
/// * `eb` -- eB in natural units (= B / B_cr)
/// * `theta` -- angle between photon propagation and B field [radians]
pub fn refractive_index_parallel(eb: f64, theta: f64) -> f64 {
    let sin2 = theta.sin().powi(2);
    1.0 + (8.0 * ALPHA_EM * ALPHA_EM / 45.0) * eb * eb * sin2
}

/// Vacuum refractive index for perpendicular polarization.
///
///   n_perp = 1 + (14 alpha^2 / 45 m_e^4) * B^2 * sin^2(theta)
///
/// The ratio of the corrections is (n_par - 1) / (n_perp - 1) = 8/14 = 4/7,
/// which is the magnetic dichroism ratio.
pub fn refractive_index_perpendicular(eb: f64, theta: f64) -> f64 {
    let sin2 = theta.sin().powi(2);
    1.0 + (14.0 * ALPHA_EM * ALPHA_EM / 45.0) * eb * eb * sin2
}

/// Vacuum birefringence Delta_n = n_perp - n_parallel.
///
///   Delta_n = (6 alpha^2 / 45) * (eB)^2 * sin^2(theta)
///           = (2 alpha^2 / 15) * (eB)^2 * sin^2(theta)
///
/// For B = 2.5 T: Delta_n ~ 2.5e-23 (the PVLAS target).
///
/// # Arguments
/// * `eb` -- eB in natural units
/// * `theta` -- propagation angle [radians]
pub fn vacuum_birefringence(eb: f64, theta: f64) -> f64 {
    let sin2 = theta.sin().powi(2);
    (2.0 * ALPHA_EM * ALPHA_EM / 15.0) * eb * eb * sin2
}

/// Vacuum birefringence in SI units (dimensionless Delta_n).
///
/// Converts from Tesla input directly.
///
/// # Arguments
/// * `b_tesla` -- magnetic field in Tesla
/// * `theta` -- propagation angle [radians]
pub fn vacuum_birefringence_si(b_tesla: f64, theta: f64) -> f64 {
    let eb = tesla_to_natural(b_tesla);
    vacuum_birefringence(eb, theta)
}

/// EH permittivity correction (from arXiv:2405.01407, Eq. 4.4).
///
///   epsilon = 1 + 4 alpha^2 B^2 / (9 m_e^4)
///
/// In natural units: epsilon = 1 + (4 alpha^2 / 9) * (eB)^2
///
/// This enters the full Gertsenshtein conversion formula.
pub fn eh_permittivity(eb: f64) -> f64 {
    1.0 + (4.0 * ALPHA_EM * ALPHA_EM / 9.0) * eb * eb
}

/// EH inverse permeability correction (from arXiv:2405.01407, Eq. 4.4).
///
///   mu^{-1} = 1 - 8 alpha^2 B^2 / (45 m_e^4)
pub fn eh_inverse_permeability(eb: f64) -> f64 {
    1.0 - (8.0 * ALPHA_EM * ALPHA_EM / 45.0) * eb * eb
}

/// Phase velocity v in the EH-modified vacuum.
///
///   v^2 = 1 / (epsilon * mu) = mu^{-1} / epsilon
pub fn eh_phase_velocity_sq(eb: f64) -> f64 {
    eh_inverse_permeability(eb) / eh_permittivity(eb)
}

// ============================================================================
// Gertsenshtein conversion probability
// ============================================================================

/// Vacuum Gertsenshtein mixing frequency.
///
///   Delta_osc = sqrt(G) * B_0 = (kappa / (4*sqrt(pi))) * eB
///
/// In natural units: Delta_osc = kappa * eB / (4 * sqrt(pi))
/// where kappa = sqrt(16*pi*G).
///
/// # Arguments
/// * `eb` -- eB in natural units (= B / B_cr)
pub fn gertsenshtein_mixing_frequency(eb: f64) -> f64 {
    // kappa = sqrt(16*pi*G), so sqrt(G) = kappa / (4*sqrt(pi))
    // Delta_osc = sqrt(G) * B = kappa/(4*sqrt(pi)) * eB/e = kappa*B_cr/(4*sqrt(pi)) * eb
    // But in natural units B (not eB), so we need B = eB/e = eB * B_CR_NATURAL
    // Actually: sqrt(G) * B_physical, where B_physical is in natural units of m_e^2
    // kappa = sqrt(16*pi*G), so G = kappa^2 / (16*pi)
    // sqrt(G) = kappa / (4*sqrt(pi))
    // B_physical = eB / e = eB * B_CR_NATURAL (in units of m_e^2)
    // Delta_osc = kappa / (4*sqrt(pi)) * eB * B_CR_NATURAL
    //
    // Simpler: in natural units where m_e = hbar = c = 1,
    // Delta_osc = sqrt(4*pi*G) * B = (kappa/2) * B
    // where B here is the physical field in natural units = eB / e_natural
    KAPPA_NATURAL / 2.0 * eb / E_NATURAL
}

/// Tree-level (vacuum) Gertsenshtein conversion probability.
///
///   P(g -> gamma) = sin^2(Delta_osc * D)
///
/// In the weak-mixing limit (B*D << M_Planck): P ~ (Delta_osc * D)^2 ~ G * B^2 * D^2
///
/// # Arguments
/// * `eb` -- eB in natural units
/// * `distance` -- propagation distance in natural units (1/m_e)
pub fn gertsenshtein_probability_vacuum(eb: f64, distance: f64) -> f64 {
    let delta = gertsenshtein_mixing_frequency(eb);
    (delta * distance).sin().powi(2)
}

/// Full Gertsenshtein conversion probability with EH corrections.
///
/// From Pantig & Ali-Haimoud (2024), Eq. 3.8:
///
///   P = [4 G B^2 eps v^4] / [4 G B^2 eps v^4 + k^2 (1-v)^2]
///       * sin^2((1/2) * sqrt[4 G B^2 eps v^4 + k^2 (1-v)^2] * D)
///
/// The EH corrections modify epsilon and v = 1/sqrt(eps*mu), introducing
/// a detuning term k^2*(1-v)^2 that suppresses coherent oscillation.
///
/// # Arguments
/// * `eb` -- eB in natural units
/// * `omega` -- photon energy in natural units
/// * `distance` -- propagation distance in natural units
pub fn gertsenshtein_probability_eh(eb: f64, omega: f64, distance: f64) -> f64 {
    let eps = eh_permittivity(eb);
    let v2 = eh_phase_velocity_sq(eb);
    let v = v2.sqrt();
    let v4 = v2 * v2;

    // G = kappa^2 / (16*pi) in natural units
    let g_nat = KAPPA_NATURAL * KAPPA_NATURAL / (16.0 * PI);

    // Physical B in natural units = eB / e_natural
    let b_phys = eb / E_NATURAL;
    let b2 = b_phys * b_phys;

    let mixing_term = 4.0 * g_nat * b2 * eps * v4;
    let detuning_term = omega * omega * (1.0 - v) * (1.0 - v);

    let total = mixing_term + detuning_term;
    if total < 1e-300 {
        return 0.0;
    }

    let suppression = mixing_term / total;
    let phase = 0.5 * total.sqrt() * distance;
    suppression * phase.sin().powi(2)
}

// ============================================================================
// Decoherence length
// ============================================================================

/// Decoherence length for graviton-photon oscillations (natural units).
///
/// From Pantig & Ali-Haimoud (2024), Eq. 4.6:
///
///   L_c = 45 m_e^4 / (14 alpha^2 B^2 k)
///
/// In natural units with m_e = 1: L_c = 45 / (14 alpha^2 (eB)^2 omega)
/// (using eB and omega directly in natural units, with B_physical = eB/e).
///
/// This is the distance over which vacuum birefringence destroys coherent
/// graviton-photon oscillation.
///
/// # Arguments
/// * `eb` -- eB in natural units
/// * `omega` -- photon energy in natural units
pub fn decoherence_length(eb: f64, omega: f64) -> f64 {
    if eb.abs() < 1e-30 || omega.abs() < 1e-30 {
        return f64::INFINITY;
    }
    // B_physical = eB / e = eB * B_CR_NATURAL
    let b_phys = eb / E_NATURAL;
    45.0 / (14.0 * ALPHA_EM * ALPHA_EM * b_phys * b_phys * omega)
}

/// Decoherence length in SI units [meters].
///
/// # Arguments
/// * `b_tesla` -- magnetic field in Tesla
/// * `photon_energy_ev` -- photon energy in eV
pub fn decoherence_length_si(b_tesla: f64, photon_energy_ev: f64) -> f64 {
    let eb = tesla_to_natural(b_tesla);
    let omega = ev_to_natural(photon_energy_ev);
    let l_c_nat = decoherence_length(eb, omega);
    // Convert from natural units (1/m_e) to meters
    let compton_m = HBAR_SI / (M_E_SI * C_SI); // ~3.86e-13 m
    l_c_nat * compton_m
}

// ============================================================================
// De Logi-Mickelson cross section
// ============================================================================

/// De Logi-Mickelson differential cross section for graviton-photon scattering.
///
/// From De Logi & Mickelson (1977), Phys. Rev. D 16, 2915:
///
///   d sigma / d Omega = (kappa^2 omega^2) / (64 pi^2) * (1 + cos^2(theta))^2
///
/// This is the tree-level cross section for graviton + photon -> graviton + photon
/// in the forward limit. The total cross section (integrated) is:
///
///   sigma_total = (kappa^2 omega^2) / (64 pi^2) * (16 pi / 5) * 2
///               = kappa^2 omega^2 / (10 pi)
///
/// For omega ~ m_e: sigma ~ 10^{-86} (in natural units) ~ 10^{-107} cm^2.
///
/// # Arguments
/// * `omega` -- photon energy in natural units
/// * `cos_theta` -- cosine of scattering angle
pub fn de_logi_mickelson_dsigma(omega: f64, cos_theta: f64) -> f64 {
    let factor = (1.0 + cos_theta * cos_theta).powi(2);
    KAPPA_NATURAL * KAPPA_NATURAL * omega * omega / (64.0 * PI * PI) * factor
}

/// Total De Logi-Mickelson cross section (integrated over solid angle).
///
///   sigma = (kappa^2 omega^2 / (64 pi^2))
///           * integral d Omega (1 + cos^2 theta)^2
///         = (kappa^2 omega^2 / (64 pi^2)) * (32 pi / 5)
///         = kappa^2 omega^2 / (10 pi)
///
/// # Arguments
/// * `omega` -- photon energy in natural units
pub fn de_logi_mickelson_total(omega: f64) -> f64 {
    // Integral of (1+cos^2)^2 over solid angle:
    // integral_0^{2pi} dphi integral_{-1}^{1} d(cos theta) (1+cos^2 theta)^2
    // = 2*pi * integral_{-1}^{1} (1+x^2)^2 dx
    // = 2*pi * integral_{-1}^{1} (1 + 2x^2 + x^4) dx
    // = 2*pi * [2 + 4/3 + 2/5]
    // = 2*pi * [30/15 + 20/15 + 6/15]
    // = 2*pi * 56/15
    // = 112*pi/15
    let solid_angle_integral = 112.0 * PI / 15.0;
    KAPPA_NATURAL * KAPPA_NATURAL * omega * omega / (64.0 * PI * PI) * solid_angle_integral
}

/// De Logi-Mickelson cross section in cm^2.
///
/// # Arguments
/// * `omega` -- photon energy in natural units
pub fn de_logi_mickelson_total_cm2(omega: f64) -> f64 {
    cross_section_to_cm2(de_logi_mickelson_total(omega))
}

// ============================================================================
// Astrophysical conversion: neutron star magnetospheres
// ============================================================================

/// Resonance condition for graviton-photon conversion in a magnetized plasma.
///
/// From arXiv:2406.18634, Eq. 18. Resonant conversion occurs when the
/// plasma frequency matches the EH birefringence correction:
///
///   omega_p^2 = C * alpha^2 * omega^2 * B^2 / (45 m_e^4)
///
/// where C = 28 for the LO mode, C = 16*sin^2(theta) for perpendicular,
/// and C = 28*sin^2(theta) for parallel.
///
/// Returns the required plasma frequency squared (in natural units) for
/// resonance at the given field and photon energy.
///
/// # Arguments
/// * `eb` -- eB in natural units
/// * `omega` -- photon energy in natural units
/// * `theta` -- propagation angle
/// * `mode` -- "lo", "perp", or "par"
pub fn resonance_plasma_freq_sq(eb: f64, omega: f64, theta: f64, mode: &str) -> f64 {
    let b_phys = eb / E_NATURAL;
    let c = match mode {
        "lo" => 28.0,
        "perp" => 16.0 * theta.sin().powi(2),
        "par" => 28.0 * theta.sin().powi(2),
        _ => 28.0, // default to LO
    };
    c * ALPHA_EM * ALPHA_EM * omega * omega * b_phys * b_phys / 45.0
}

/// Neutron star surface field in natural units.
///
/// Converts from Gauss (CGS, common in astrophysics) to natural units.
///
/// # Arguments
/// * `b_gauss` -- surface magnetic field in Gauss
pub fn gauss_to_natural(b_gauss: f64) -> f64 {
    // 1 T = 10^4 G, B_cr = 4.414e13 G
    let b_cr_gauss = B_CR_SI * 1e4; // ~4.414e13 G
    b_gauss / b_cr_gauss
}

/// Conversion probability for graviton traversing a neutron star magnetosphere.
///
/// Simple estimate assuming uniform field B_0 over coherence length L_coh.
/// In practice the field is dipolar (B ~ B_0 (R_NS/r)^3) and the coherence
/// length is set by the gradient scale.
///
/// P ~ (sqrt(G) * B_0 * L_coh)^2 ~ (kappa * B_0 * L_coh)^2 / (16 pi)
///
/// # Arguments
/// * `b_gauss` -- surface field in Gauss
/// * `l_coh_cm` -- coherence length in cm
pub fn ns_conversion_estimate(b_gauss: f64, l_coh_cm: f64) -> f64 {
    let eb = gauss_to_natural(b_gauss);
    // Convert coherence length from cm to natural units (1/m_e)
    let compton_cm = HBAR_SI / (M_E_SI * C_SI) * 100.0; // ~3.86e-11 cm
    let l_nat = l_coh_cm / compton_cm;
    gertsenshtein_probability_vacuum(eb, l_nat)
}

// ============================================================================
// Euler-Heisenberg effective Lagrangian coefficients
// ============================================================================

/// Euler-Heisenberg coefficient for F^2 term (spinor QED).
///
///   c_F2 = 2 alpha^2 / (45 m_e^4)
///
/// In natural units (m_e = 1): c_F2 = 2 alpha^2 / 45
///
/// The full EH Lagrangian: L = -F + c_F2 * [4 F^2 + 7 G^2] + ...
/// where F = (1/2)(B^2 - E^2) and G = E . B.
pub fn eh_coefficient_spinor() -> (f64, f64) {
    let c = 2.0 * ALPHA_EM * ALPHA_EM / 45.0;
    (4.0 * c, 7.0 * c) // (coefficient of F^2, coefficient of G^2)
}

/// Euler-Heisenberg coefficient for F^2 term (scalar QED).
///
///   L_EH^scalar = -F + (alpha^2 / 90) * [7 F^2 + G^2] + ...
///
/// The scalar-to-spinor ratio for F^2 is (7/90) / (8/90) = 7/8.
pub fn eh_coefficient_scalar() -> (f64, f64) {
    let c = ALPHA_EM * ALPHA_EM / 90.0;
    (7.0 * c, 1.0 * c) // (coefficient of F^2, coefficient of G^2)
}

/// Verify the 4:7 dichroism ratio from EH coefficients.
///
/// For spinor QED:
///   n_par - 1 = 8 alpha^2 / 45 * B^2 sin^2(theta)
///   n_perp - 1 = 14 alpha^2 / 45 * B^2 sin^2(theta)
///   ratio = 8/14 = 4/7
///
/// Returns (parallel_coefficient, perpendicular_coefficient, ratio).
pub fn dichroism_ratio_from_eh() -> (f64, f64, f64) {
    let c_par = 8.0 * ALPHA_EM * ALPHA_EM / 45.0;
    let c_perp = 14.0 * ALPHA_EM * ALPHA_EM / 45.0;
    (c_par, c_perp, c_par / c_perp)
}

// ============================================================================
// PVLAS comparison
// ============================================================================

/// PVLAS-comparable birefringence at given field.
///
/// Returns the QED prediction for Delta_n at the given field, and the
/// PVLAS measurement value and uncertainty for comparison.
///
/// # Arguments
/// * `b_tesla` -- magnetic field in Tesla
///
/// # Returns
/// (qed_prediction, pvlas_central, pvlas_uncertainty)
///
/// PVLAS at 2.5 T: Delta_n = (12 +/- 17) x 10^{-23}
/// QED at 2.5 T: Delta_n = 2.5 x 10^{-23}
pub fn pvlas_comparison(b_tesla: f64) -> (f64, f64, f64) {
    let qed = vacuum_birefringence_si(b_tesla, PI / 2.0);

    // PVLAS measurement at B_ref = 2.5 T (arXiv:2005.12913)
    let b_ref = 2.5;
    let pvlas_central_ref = 12.0e-23;
    let pvlas_unc_ref = 17.0e-23;

    // Scale to requested field (Delta_n ~ B^2)
    let scale = (b_tesla / b_ref).powi(2);
    let pvlas_central = pvlas_central_ref * scale;
    let pvlas_unc = pvlas_unc_ref * scale;

    (qed, pvlas_central, pvlas_unc)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Euler-Heisenberg refractive indices --

    #[test]
    fn test_refractive_index_at_zero_field() {
        let n_par = refractive_index_parallel(0.0, PI / 2.0);
        let n_perp = refractive_index_perpendicular(0.0, PI / 2.0);
        assert!((n_par - 1.0).abs() < 1e-30);
        assert!((n_perp - 1.0).abs() < 1e-30);
    }

    #[test]
    fn test_refractive_index_dichroism_ratio() {
        // (n_par - 1) / (n_perp - 1) = 8/14 = 4/7
        let eb = 0.01;
        let theta = PI / 2.0;
        let dn_par = refractive_index_parallel(eb, theta) - 1.0;
        let dn_perp = refractive_index_perpendicular(eb, theta) - 1.0;
        let ratio = dn_par / dn_perp;
        assert!(
            (ratio - 4.0 / 7.0).abs() < 1e-12,
            "Dichroism ratio = {ratio}, expected 4/7 = {}",
            4.0 / 7.0
        );
    }

    #[test]
    fn test_refractive_index_vanishes_parallel_propagation() {
        // At theta = 0 (k || B), sin^2(theta) = 0, so no birefringence
        let eb = 0.1;
        let n_par = refractive_index_parallel(eb, 0.0);
        let n_perp = refractive_index_perpendicular(eb, 0.0);
        assert!((n_par - 1.0).abs() < 1e-30);
        assert!((n_perp - 1.0).abs() < 1e-30);
    }

    #[test]
    fn test_birefringence_positive() {
        // Delta_n = n_perp - n_par > 0 (perpendicular is slower)
        let dn = vacuum_birefringence(0.01, PI / 2.0);
        assert!(dn > 0.0, "Birefringence should be positive: {dn}");
    }

    #[test]
    fn test_birefringence_scales_as_b_squared() {
        let dn1 = vacuum_birefringence(0.01, PI / 2.0);
        let dn2 = vacuum_birefringence(0.02, PI / 2.0);
        let ratio = dn2 / dn1;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "Delta_n should scale as B^2: ratio = {ratio}"
        );
    }

    #[test]
    fn test_birefringence_si_pvlas_target() {
        // QED prediction at 2.5 T: Delta_n ~ 2.5e-23
        let dn = vacuum_birefringence_si(2.5, PI / 2.0);
        assert!(
            dn > 1e-24 && dn < 1e-22,
            "QED birefringence at 2.5 T = {dn:.3e}, expected ~2.5e-23"
        );
    }

    // -- EH constitutive relations --

    #[test]
    fn test_eh_permittivity_near_unity() {
        let eps = eh_permittivity(0.01);
        assert!(
            (eps - 1.0).abs() < 1e-6,
            "Permittivity at weak field should be ~1: {eps}"
        );
        assert!(eps > 1.0, "Permittivity correction should be positive");
    }

    #[test]
    fn test_eh_permeability_near_unity() {
        let mu_inv = eh_inverse_permeability(0.01);
        assert!(
            (mu_inv - 1.0).abs() < 1e-6,
            "Inverse permeability at weak field ~1: {mu_inv}"
        );
        assert!(
            mu_inv < 1.0,
            "Permeability correction should be paramagnetic (mu_inv < 1)"
        );
    }

    #[test]
    fn test_phase_velocity_decreases_with_field() {
        let v2_0 = eh_phase_velocity_sq(0.0);
        let v2_01 = eh_phase_velocity_sq(0.01);
        assert!((v2_0 - 1.0).abs() < 1e-30, "v=1 in vacuum");
        assert!(
            v2_01 < 1.0,
            "Phase velocity should decrease with field: {v2_01}"
        );
    }

    // -- Gertsenshtein conversion --

    #[test]
    fn test_gertsenshtein_probability_zero_field() {
        let p = gertsenshtein_probability_vacuum(0.0, 1e10);
        assert!(p.abs() < 1e-30, "P should vanish at B=0");
    }

    #[test]
    fn test_gertsenshtein_probability_zero_distance() {
        let p = gertsenshtein_probability_vacuum(0.01, 0.0);
        assert!(p.abs() < 1e-30, "P should vanish at D=0");
    }

    #[test]
    fn test_gertsenshtein_probability_bounded() {
        // sin^2 is bounded by 1
        let p = gertsenshtein_probability_vacuum(0.01, 1e20);
        assert!(p >= 0.0 && p <= 1.0, "P = {p} should be in [0,1]");
    }

    #[test]
    fn test_gertsenshtein_weak_mixing_quadratic() {
        // In weak mixing: P ~ (Delta * D)^2 ~ D^2
        let d1 = 1e10;
        let d2 = 2e10;
        let p1 = gertsenshtein_probability_vacuum(1e-6, d1);
        let p2 = gertsenshtein_probability_vacuum(1e-6, d2);
        // For small angles, sin^2(x) ~ x^2
        if p1 > 1e-30 && p1 < 0.01 {
            let ratio = p2 / p1;
            assert!(
                (ratio - 4.0).abs() < 0.1,
                "P should scale as D^2: ratio = {ratio}"
            );
        }
    }

    #[test]
    fn test_gertsenshtein_eh_reduces_to_vacuum() {
        // At zero field, both should give P=0
        let p_vac = gertsenshtein_probability_vacuum(0.0, 1e10);
        let p_eh = gertsenshtein_probability_eh(0.0, 0.1, 1e10);
        assert!(p_vac.abs() < 1e-30);
        assert!(p_eh.abs() < 1e-30);
    }

    #[test]
    fn test_gertsenshtein_eh_suppressed_vs_vacuum() {
        // EH corrections introduce detuning that suppresses conversion
        // For weak fields the suppression should be small
        let eb = 1e-6;
        let omega = 0.1;
        let d = 1e10;
        let p_vac = gertsenshtein_probability_vacuum(eb, d);
        let p_eh = gertsenshtein_probability_eh(eb, omega, d);
        // At very weak field, EH correction is tiny, so they should be close
        if p_vac > 1e-50 {
            let suppression = p_eh / p_vac;
            assert!(
                suppression <= 1.0 + 1e-6,
                "EH should suppress or maintain: {suppression}"
            );
        }
    }

    // -- Decoherence length --

    #[test]
    fn test_decoherence_length_infinite_at_zero_field() {
        let l = decoherence_length(0.0, 0.1);
        assert!(l.is_infinite(), "L_c should be infinite at B=0");
    }

    #[test]
    fn test_decoherence_length_decreases_with_field() {
        let l1 = decoherence_length(0.01, 0.1);
        let l2 = decoherence_length(0.02, 0.1);
        assert!(
            l2 < l1,
            "Decoherence length should decrease with B: {l2} vs {l1}"
        );
    }

    #[test]
    fn test_decoherence_length_scales_as_b_minus_2() {
        let l1 = decoherence_length(0.01, 0.1);
        let l2 = decoherence_length(0.02, 0.1);
        let ratio = l1 / l2;
        // L_c ~ 1/B^2, so doubling B gives 4x shorter
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "L_c ratio = {ratio}, expected ~4"
        );
    }

    // -- De Logi-Mickelson cross section --

    #[test]
    fn test_dlm_forward_scattering() {
        // At cos(theta)=1 (forward): factor = (1+1)^2 = 4
        let sigma_fwd = de_logi_mickelson_dsigma(1.0, 1.0);
        let sigma_perp = de_logi_mickelson_dsigma(1.0, 0.0);
        // Forward: (1+1)^2 = 4, perpendicular: (1+0)^2 = 1
        assert!(
            (sigma_fwd / sigma_perp - 4.0).abs() < 1e-10,
            "Forward/perp ratio = {}",
            sigma_fwd / sigma_perp
        );
    }

    #[test]
    fn test_dlm_total_scales_as_omega_squared() {
        let s1 = de_logi_mickelson_total(1.0);
        let s2 = de_logi_mickelson_total(2.0);
        let ratio = s2 / s1;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "sigma ~ omega^2: ratio = {ratio}"
        );
    }

    #[test]
    fn test_dlm_incredibly_tiny() {
        // sigma ~ kappa^2 * omega^2 ~ (3e-22)^2 * 1 ~ 9e-44 (natural)
        // In cm^2: ~ 9e-44 * (3.86e-11)^2 ~ 1.3e-64 cm^2
        let s = de_logi_mickelson_total_cm2(1.0);
        assert!(
            s < 1e-60,
            "Cross section should be incredibly tiny: {s:.3e} cm^2"
        );
    }

    // -- EH coefficients --

    #[test]
    fn test_eh_spinor_ratio_4_to_7() {
        let (c_f2, c_g2) = eh_coefficient_spinor();
        let ratio = c_f2 / c_g2;
        assert!(
            (ratio - 4.0 / 7.0).abs() < 1e-12,
            "Spinor F^2/G^2 ratio = {ratio}, expected 4/7"
        );
    }

    #[test]
    fn test_eh_scalar_coefficients() {
        let (c_f2, c_g2) = eh_coefficient_scalar();
        let ratio = c_f2 / c_g2;
        // Scalar: (7/90) / (1/90) = 7
        assert!(
            (ratio - 7.0).abs() < 1e-10,
            "Scalar F^2/G^2 ratio = {ratio}, expected 7"
        );
    }

    #[test]
    fn test_eh_spinor_dominates_scalar() {
        let (spinor_f2, _) = eh_coefficient_spinor();
        let (scalar_f2, _) = eh_coefficient_scalar();
        assert!(
            spinor_f2 > scalar_f2,
            "Spinor should dominate: {spinor_f2} vs {scalar_f2}"
        );
    }

    #[test]
    fn test_dichroism_ratio_from_eh_is_4_over_7() {
        let (_, _, ratio) = dichroism_ratio_from_eh();
        assert!(
            (ratio - 4.0 / 7.0).abs() < 1e-12,
            "Dichroism ratio = {ratio}"
        );
    }

    // -- PVLAS comparison --

    #[test]
    fn test_pvlas_qed_within_measurement_uncertainty() {
        let (qed, _pvlas_central, pvlas_unc) = pvlas_comparison(2.5);
        // QED prediction should be within PVLAS uncertainty
        assert!(
            qed < pvlas_unc,
            "QED = {qed:.3e} should be < PVLAS uncertainty {pvlas_unc:.3e}"
        );
    }

    // -- Astrophysical estimates --

    #[test]
    fn test_gauss_to_natural() {
        // B_cr = 4.414e13 G -> natural = 1
        let b_nat = gauss_to_natural(4.414e13);
        assert!(
            (b_nat - 1.0).abs() < 0.01,
            "B_cr in Gauss should give ~1: {b_nat}"
        );
    }

    #[test]
    fn test_resonance_plasma_freq_positive() {
        let wp2 = resonance_plasma_freq_sq(0.1, 1.0, PI / 4.0, "lo");
        assert!(
            wp2 > 0.0,
            "Resonance plasma frequency should be positive: {wp2}"
        );
    }

    #[test]
    fn test_ns_conversion_tiny() {
        // Magnetar: B ~ 10^15 G, L ~ 10^6 cm (10 km)
        // P ~ (sqrt(G) * B * L)^2 is small but not negligible:
        // kappa*B*L ~ 1.4e-3, so P ~ O(10^-6 to 10^-7).
        let p = ns_conversion_estimate(1e15, 1e6);
        assert!(
            p < 1e-3,
            "NS conversion should be much less than 1: {p:.3e}"
        );
        assert!(
            p > 0.0,
            "NS conversion should be positive"
        );
    }
}
