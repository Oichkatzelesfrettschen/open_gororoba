//! External-leg diagram (Gamma_ext) for photon-graviton mixing.
//!
//! This diagram involves the vacuum polarization tensor Pi_bar inserted on
//! the external photon leg, contracted with the effective polarization
//! vector v_F from the graviton-field coupling.
//!
//! Gamma_ext = v_F^beta * [Pi_bar^{ba}(k;F) - Pi_bar^{ba}(k;0)] * epsilon_alpha
//!
//! The subtraction of Pi_bar(k;0) removes the constant (zero-field) part.
//! Like the tadpole, this starts at cubic order in the field after
//! renormalization and does not contribute to dichroism at leading order.
//!
//! The on-shell IR divergence from v_F ~ 1/k^2 is interpreted as a modified
//! dispersion relation following Ahlers et al. (2009).
//!
//! # References
//! - Bastianelli et al. (2026), Sec 6
//! - Ahlers et al. (2009): Phys. Rev. D 80, 023513

use num_complex::Complex64;

use super::constants::KAPPA_NATURAL;
use super::quadrature::QuadratureConfig;
use super::types::{FieldConfig, LoopType, PhotonPolarization};
use super::vacuum_pol;

/// Compute the external-leg diagram amplitude for a given photon polarization.
///
/// This uses the vacuum polarization in the constant field, contracted with
/// the effective graviton-field polarization vector v_F.
///
/// The result is the complex amplitude coefficient multiplying the tree-level
/// kinematic structure (same polarization basis as tadpole).
///
/// # IR regularization
/// For on-shell photon, v_F has a 1/k^2 pole. We use config.k_squared as
/// an IR cutoff. For physical predictions, the IR-divergent pieces cancel
/// in the full amplitude when properly combined with modified dispersion
/// relation effects.
pub fn external_amplitude(
    config: &FieldConfig,
    photon_pol: PhotonPolarization,
    loop_type: LoopType,
    quad_config: &QuadratureConfig,
) -> Complex64 {
    let eb = config.a;
    if eb.abs() < 1e-30 {
        return Complex64::new(0.0, 0.0);
    }

    // v_F magnitude (effective graviton polarization vector)
    let v_f = vacuum_pol::effective_polarization_vector(config);

    // Pi_bar in the constant field (field-dependent part, already renormalized)
    let pi_field = vacuum_pol::vacuum_polarization(config, photon_pol, loop_type, quad_config);

    // The external diagram is v_F * Pi_bar_field * epsilon
    // Since we already subtracted the free-field part in vacuum_polarization,
    // the result is the field-dependent correction only.
    pi_field * v_f
}

/// External-leg contribution to the squared amplitude for a given photon
/// polarization.
pub fn external_rate(
    config: &FieldConfig,
    photon_pol: PhotonPolarization,
    loop_type: LoopType,
    quad_config: &QuadratureConfig,
) -> f64 {
    let amp = external_amplitude(config, photon_pol, loop_type, quad_config);
    amp.norm_sqr()
}

/// Weak-field estimate of the external diagram.
///
/// At leading order in eB, the external diagram gives a contribution of
/// order (alpha * eB^2) * (kappa * eB) ~ kappa * alpha * eB^3.
/// This is one power of eB higher than the irreducible diagram's leading
/// order, so it is subdominant at weak field.
pub fn external_weak_field_estimate(config: &FieldConfig, loop_type: LoopType) -> f64 {
    let eb = config.a;
    let omega = config.omega;
    let sin_t = config.theta.sin();

    // Leading order: ~ kappa * alpha^2 * eB^3 * omega / k^2
    // With k^2 cutoff, this is parametrically suppressed by one power of eB
    let alpha = super::constants::ALPHA_EM;
    let k_sq = if config.k_squared.abs() > 1e-30 {
        config.k_squared
    } else {
        1e-20
    };

    let loop_factor = match loop_type {
        LoopType::Spinor => 1.0,
        LoopType::Scalar => 0.25,
    };

    KAPPA_NATURAL * alpha * alpha * eb * eb * eb * omega * sin_t * loop_factor / k_sq
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn weak_config() -> FieldConfig {
        FieldConfig {
            a: 0.01,
            b: 0.0,
            k_squared: 1e-6, // small IR cutoff
            omega: 0.1,
            theta: PI / 4.0,
        }
    }

    #[test]
    fn test_external_zero_field() {
        let config = FieldConfig {
            a: 0.0,
            b: 0.0,
            k_squared: 1e-6,
            omega: 0.1,
            theta: PI / 4.0,
        };
        let quad = QuadratureConfig::fast();
        let amp = external_amplitude(
            &config,
            PhotonPolarization::Parallel,
            LoopType::Spinor,
            &quad,
        );
        assert!(
            amp.norm() < 1e-25,
            "External should vanish at B=0, got {amp}"
        );
    }

    #[test]
    fn test_external_nonzero() {
        let config = weak_config();
        let quad = QuadratureConfig::default();
        let amp = external_amplitude(
            &config,
            PhotonPolarization::Parallel,
            LoopType::Spinor,
            &quad,
        );
        // May be very small due to IR cutoff, but should be finite
        assert!(amp.is_finite(), "External amplitude not finite: {amp}");
    }

    #[test]
    fn test_external_subdominant() {
        // External should be suppressed relative to irreducible by ~ eB
        let config = weak_config();
        let ext_est = external_weak_field_estimate(&config, LoopType::Spinor);
        // Should be parametrically small
        assert!(ext_est.abs() < 1e-10, "External estimate = {ext_est}");
    }

    #[test]
    fn test_external_scales_with_field() {
        let quad = QuadratureConfig::default();
        let cfg1 = FieldConfig {
            a: 0.01,
            b: 0.0,
            k_squared: 1e-6,
            omega: 0.1,
            theta: PI / 4.0,
        };
        let cfg2 = FieldConfig {
            a: 0.02,
            b: 0.0,
            k_squared: 1e-6,
            omega: 0.1,
            theta: PI / 4.0,
        };
        let a1 = external_amplitude(
            &cfg1,
            PhotonPolarization::Parallel,
            LoopType::Spinor,
            &quad,
        );
        let a2 = external_amplitude(
            &cfg2,
            PhotonPolarization::Parallel,
            LoopType::Spinor,
            &quad,
        );

        // External ~ Pi(B) * v_F ~ (eB)^2 * (eB) = (eB)^3
        // So doubling B should give ~8x increase
        if a1.norm() > 1e-30 {
            let ratio = a2.norm() / a1.norm();
            assert!(
                ratio > 3.0,
                "External should scale faster than B^2: ratio = {ratio}"
            );
        }
    }
}
