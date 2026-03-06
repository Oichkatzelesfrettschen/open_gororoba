//! Vacuum polarization tensor in a constant background field.
//!
//! The renormalized vacuum polarization Pi_bar^{ab}(k; F) enters the
//! external-leg diagram (Gamma_ext) of the photon-graviton amplitude.
//! For a pure magnetic field, the tensor decomposes into two independent
//! scalar functions: Pi_parallel and Pi_perpendicular.
//!
//! The computation follows Sec 3.3 of Part 3 (Eqs 3.22-3.23):
//!
//! Pi_bar = (e^2 / 8 pi^2) * integral dT/T * exp(-m^2 T)
//!        * integral_0^1 du * {det * I_tilde * exp(-T k.Phi.k) + subtraction}
//!
//! # References
//! - Bastianelli et al. (2026), Sec 3.3, Eqs 3.22-3.23
//! - Dittrich (1976): "One-loop effective Lagrangian in QED"
//! - Tsai (1974): Phys. Rev. D 10, 2699

use num_complex::Complex64;

use super::{
    constants::ALPHA_EM,
    quadrature::{self, QuadratureConfig},
    special_functions,
    types::{FieldConfig, LoopType, PhotonPolarization},
    worldline_greens,
};

/// Compute the vacuum polarization for a given photon polarization in
/// a pure magnetic field.
///
/// Returns the renormalized Pi_bar as a complex number.
///
/// For parallel (perpendicular) polarization, this gives the modification
/// of the photon propagator in the constant field. The tensor structure
/// is already projected onto the polarization basis.
///
/// # Physics
/// The integrand involves:
/// 1. Determinant factor (scalar or spinor)
/// 2. Tensor structure I_tilde from worldline contractions
/// 3. Exponential factor from the momentum-dependent Green's function
/// 4. Renormalization subtraction (zero-field part)
pub fn vacuum_polarization(
    config: &FieldConfig,
    photon_pol: PhotonPolarization,
    loop_type: LoopType,
    quad_config: &QuadratureConfig,
) -> Complex64 {
    let prefactor = ALPHA_EM / (2.0 * std::f64::consts::PI);
    let mass_sq = 1.0; // m_e = 1 in natural units

    let result = quadrature::double_integral(
        |t, u| {
            let z = config.a * t; // z = eB*T
            let v = 1.0 - 2.0 * u;

            // Determinant factor
            let det = match loop_type {
                LoopType::Scalar => special_functions::det_scalar_4d(z),
                LoopType::Spinor => special_functions::det_spinor_4d(z),
            };

            // Tensor structure I_tilde projected onto the given polarization.
            // For pure magnetic field:
            //
            // Parallel (E in B plane):
            //   I_par = SB12^2 + z*AB12 (scalar)
            //   I_par = SB12^2 + z*AB12 - SF12*AF12 (spinor, from GF term)
            //
            // Perpendicular (E perp to B):
            //   I_perp = AB12^2 + z*AB12_coincidence (scalar)
            //   I_perp = AB12^2 + z*AB_coincidence - SF12^2 (spinor, from GF)
            let sb = worldline_greens::sb12(z, u);
            let ab = worldline_greens::ab12(z, u);

            let i_tilde = match (photon_pol, loop_type) {
                (PhotonPolarization::Parallel, LoopType::Scalar) => {
                    // From SB12 contraction in the field plane
                    sb * sb + z * ab
                }
                (PhotonPolarization::Parallel, LoopType::Spinor) => {
                    let sf = worldline_greens::sf12(z, u);
                    let af = worldline_greens::af12(z, u);
                    sb * sb + z * ab - sf * af
                }
                (PhotonPolarization::Perpendicular, LoopType::Scalar) => {
                    let ab_coinc = worldline_greens::ab_coincidence(z);
                    ab * ab + z * ab_coinc
                }
                (PhotonPolarization::Perpendicular, LoopType::Spinor) => {
                    let sf = worldline_greens::sf12(z, u);
                    let ab_coinc = worldline_greens::ab_coincidence(z);
                    ab * ab + z * ab_coinc - sf * sf
                }
            };

            // Exponential factor from photon momentum
            let exp_arg = worldline_greens::exp_factor(z, u, config.omega, config.theta, t);

            // Renormalization subtraction: remove the z=0 (free field) contribution.
            // At z=0: SB12->v, AB12->0, SF12->1, AF12->v
            // For parallel scalar: v^2 + 0 = v^2
            // For parallel spinor: v^2 + 0 - v = v^2 - v = v(v-1)
            let i_tilde_free = match (photon_pol, loop_type) {
                (PhotonPolarization::Parallel, LoopType::Scalar) => v * v,
                (PhotonPolarization::Parallel, LoopType::Spinor) => v * v - v,
                (PhotonPolarization::Perpendicular, LoopType::Scalar) => 0.0,
                (PhotonPolarization::Perpendicular, LoopType::Spinor) => -1.0,
            };

            let exp_free = worldline_greens::exp_factor(0.0, u, config.omega, config.theta, t);

            let integrand = det * i_tilde * (-exp_arg).exp() - i_tilde_free * (-exp_free).exp();

            Complex64::new(integrand, 0.0)
        },
        mass_sq,
        1.0, // power = 1 (dT/T)
        quad_config,
    );

    result * prefactor
}

/// Vacuum polarization at zero momentum (k=0) in a pure magnetic field.
///
/// This is the static limit, relevant for modified dispersion relations.
/// Simpler than the full k-dependent version since exp_factor = 0.
pub fn vacuum_polarization_static(
    config: &FieldConfig,
    photon_pol: PhotonPolarization,
    loop_type: LoopType,
    quad_config: &QuadratureConfig,
) -> Complex64 {
    let mut static_config = *config;
    static_config.omega = 0.0;
    vacuum_polarization(&static_config, photon_pol, loop_type, quad_config)
}

/// Effective polarization vector v_F (Eq 3.8/6.2 of Part 3).
///
/// v_F = i * kappa * {epsilon_0, F} . k / k^2
///
/// For on-shell photon (k^2 = 0) this has an IR pole. We regularize by
/// setting k^2 = k_squared from the config (which should be a small IR cutoff
/// for on-shell computations).
///
/// For pure magnetic field, the contraction {epsilon_0, F}.k gives a factor
/// proportional to eB * omega * sin(theta).
///
/// Returns the effective polarization magnitude (real for pure magnetic field).
pub fn effective_polarization_vector(config: &FieldConfig) -> f64 {
    let eb = config.a;
    let omega = config.omega;
    let sin_t = config.theta.sin();

    // {epsilon_0, F}.k ~ eB * omega * sin(theta) for both graviton polarizations
    // The IR-regulated denominator:
    let k_sq = if config.k_squared.abs() > 1e-30 {
        config.k_squared
    } else {
        // Use a small IR cutoff for on-shell
        1e-20
    };

    super::constants::KAPPA_NATURAL * eb * omega * sin_t / k_sq
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn weak_field_config() -> FieldConfig {
        FieldConfig::pure_magnetic(0.01, 0.1, PI / 4.0)
    }

    #[test]
    fn test_vacuum_pol_zero_field() {
        // At B=0, the renormalized vacuum polarization should vanish
        // (we subtracted the free-field part).
        let config = FieldConfig::pure_magnetic(0.0, 0.1, PI / 4.0);
        let quad = QuadratureConfig::fast();
        let result = vacuum_polarization(
            &config,
            PhotonPolarization::Parallel,
            LoopType::Spinor,
            &quad,
        );
        assert!(result.norm() < 1e-10, "Pi(B=0) = {result}, expected ~0");
    }

    #[test]
    fn test_vacuum_pol_nonzero() {
        // At nonzero B, vacuum polarization should be nonzero
        let config = weak_field_config();
        let quad = QuadratureConfig::default();
        let result = vacuum_polarization(
            &config,
            PhotonPolarization::Parallel,
            LoopType::Spinor,
            &quad,
        );
        // Should be proportional to alpha * (eB)^2 at weak field
        assert!(
            result.norm() > 1e-20,
            "Pi should be nonzero at B > 0, got {result}"
        );
    }

    #[test]
    fn test_vacuum_pol_scales_with_field() {
        let quad = QuadratureConfig::default();
        let cfg1 = FieldConfig::pure_magnetic(0.01, 0.1, PI / 4.0);
        let cfg2 = FieldConfig::pure_magnetic(0.02, 0.1, PI / 4.0);
        let pi1 = vacuum_polarization(&cfg1, PhotonPolarization::Parallel, LoopType::Spinor, &quad);
        let pi2 = vacuum_polarization(&cfg2, PhotonPolarization::Parallel, LoopType::Spinor, &quad);
        // At weak field, Pi ~ (eB)^2, so doubling B should give ~4x
        if pi1.norm() > 1e-30 {
            let ratio = pi2.norm() / pi1.norm();
            assert!(
                (ratio - 4.0).abs() < 1.5,
                "Pi should scale ~B^2: ratio = {ratio}"
            );
        }
    }

    #[test]
    fn test_vacuum_pol_scalar_vs_spinor() {
        let config = weak_field_config();
        let quad = QuadratureConfig::default();
        let pi_scalar = vacuum_polarization(
            &config,
            PhotonPolarization::Parallel,
            LoopType::Scalar,
            &quad,
        );
        let pi_spinor = vacuum_polarization(
            &config,
            PhotonPolarization::Parallel,
            LoopType::Spinor,
            &quad,
        );
        // Both should be nonzero but different
        // Spinor typically larger due to extra spin structure
        assert!(pi_scalar.norm() > 0.0);
        assert!(pi_spinor.norm() > 0.0);
    }

    #[test]
    fn test_effective_polarization_vector() {
        let config = FieldConfig {
            a: 0.01,
            b: 0.0,
            k_squared: 1e-6, // IR cutoff
            omega: 0.1,
            theta: PI / 4.0,
        };
        let v_f = effective_polarization_vector(&config);
        // Should be kappa * eB * omega * sin(theta) / k^2
        assert!(v_f.abs() > 0.0);
        assert!(v_f.is_finite());
    }
}
