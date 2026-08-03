//! Photon-graviton mixing in constant electromagnetic fields.
//!
//! Full numerical implementation of the one-loop photon-graviton conversion
//! amplitude from the worldline formalism, following the three-paper series
//! by Bastianelli, Corradini, Huet, and Schubert (2005, 2007, 2026).
//!
//! The one-loop amplitude receives contributions from three diagrams:
//! 1. **Irreducible** -- photon and graviton couple directly to the loop
//! 2. **Tadpole** -- graviton couples to the vacuum energy (NEW in Part 3)
//! 3. **External** -- vacuum polarization on the external photon leg
//!
//! The tadpole does NOT contribute to magnetic dichroism; it only rescales
//! the overall rate. Dichroism arises purely from the irreducible diagram
//! (J_3 tensor structure) and gives a parallel:perpendicular ratio of 4:7
//! at weak field for spinor QED.
//!
//! # Natural units
//! All internal computations use hbar = c = 1, m_e = 1. SI conversion
//! functions are provided for final output.
//!
//! # Module organization
//! - `types` -- Core types (FieldConfig, HelicityChannel, LoopType, etc.)
//! - `constants` -- QED/gravitational constants in natural and SI units
//! - `special_functions` -- Bernoulli numbers, Hurwitz zeta, determinant factors
//! - `quadrature` -- Gauss-Legendre integration utilities
//! - `worldline_greens` -- Worldline Green's functions on the circle
//! - `tree_level` -- Tree-level reference amplitude
//! - `vacuum_pol` -- Vacuum polarization in constant field
//! - `tadpole` -- Tadpole diagram (3 representations)
//! - `irreducible` -- Irreducible one-loop diagram
//! - `external` -- External-leg diagram
//! - `ward` -- Ward identity verification
//! - `gertsenshtein` -- Gertsenshtein effect, vacuum birefringence, decoherence
//! - `benchmarks` -- External numerical benchmarks and strong-field tests
//!
//! # References
//! - Part 1: Bastianelli & Schubert (2005), gr-qc/0412095
//! - Part 2: Bastianelli, Schubert et al. (2007), 0710.5572
//! - Part 3: Bastianelli et al. (2026), arXiv:2601.23279
//! - Ahmadiniaz, N. et al. (2026), arXiv:2601.23279

pub mod benchmarks;
pub mod constants;
pub mod external;
pub mod external_tensor;
pub mod gertsenshtein;
pub mod irreducible;
pub mod irreducible_tensor;
pub mod one_photon;
pub mod quadrature;
pub mod special_functions;
pub mod tadpole;
pub mod tadpole_tensor;
pub mod tensor_integrands;
pub mod tensor_types;
pub mod tree_level;
pub mod types;
pub mod vacuum_pol;
pub mod vacuum_pol_tensor;
pub mod ward;
pub mod worldline_greens;
pub mod worldline_tensor;

use num_complex::Complex64;
use std::f64::consts::PI;

pub use quadrature::QuadratureConfig;
pub use types::*;

// ============================================================================
// Legacy SI-unit functions (preserved from the original photon_graviton.rs)
// ============================================================================

/// Electron mass `[kg]`
const M_E_SI: f64 = 9.109_383_701_5e-31;

/// Speed of light `[m/s]`
const C_SI: f64 = 2.997_924_58e8;

/// Elementary charge `[C]`
const E_SI: f64 = 1.602_176_634e-19;

/// Reduced Planck constant `[J*s]`
const HBAR_SI: f64 = 1.054_571_817e-34;

/// Gravitational constant [m^3 kg^-1 s^-2]
const G_SI: f64 = 6.674_30e-11;

/// Fine-structure constant (dimensionless)
const ALPHA_EM_LEGACY: f64 = 1.0 / 137.036;

/// Schwinger critical magnetic field B_cr = m_e^2 c^2 / (e hbar) `[T]`.
///
/// Above this field strength, vacuum pair production becomes significant.
/// B_cr ~ 4.41e9 T.
pub fn schwinger_critical_field() -> f64 {
    M_E_SI * M_E_SI * C_SI * C_SI / (E_SI * HBAR_SI)
}

/// Gravitational coupling constant kappa = sqrt(16 pi G / c^4) `[SI]`.
///
/// This sets the scale of gravitational interactions in linearized GR.
/// kappa ~ 2.04e-22 in SI units.
pub fn gravitational_coupling() -> f64 {
    (16.0 * PI * G_SI / (C_SI * C_SI * C_SI * C_SI)).sqrt()
}

/// One-loop photon-graviton mixing amplitude estimate.
///
/// The dimensionless amplitude ratio is:
///   A ~ (alpha/pi) * (B_lab / B_cr)^2
///
/// This is the suppression factor for photon-graviton conversion in
/// a laboratory magnetic field. For B_lab = 10 T, the ratio is ~ 1.2e-20,
/// making the effect unmeasurable.
///
/// Returns `(b_cr, kappa, amplitude_ratio)`.
pub fn mixing_amplitude_estimate(b_lab: f64) -> (f64, f64, f64) {
    let b_cr = schwinger_critical_field();
    let kappa = gravitational_coupling();
    let ratio = (ALPHA_EM_LEGACY / PI) * (b_lab / b_cr).powi(2);
    (b_cr, kappa, ratio)
}

// ============================================================================
// Full one-loop amplitude computation
// ============================================================================

/// Compute the full one-loop photon-graviton amplitude for a given helicity
/// channel, summing all three diagrams.
///
/// Returns a `FullAmplitude` with the decomposition into irreducible,
/// tadpole, and external contributions, plus the total and conversion
/// probability.
///
/// # Arguments
/// * `config` -- Background field configuration
/// * `loop_type` -- Scalar or spinor loop particle
/// * `channel` -- Helicity channel (photon pol x graviton pol)
/// * `quad_config` -- Quadrature settings
pub fn one_loop_amplitude(
    config: &FieldConfig,
    loop_type: LoopType,
    channel: HelicityChannel,
    quad_config: &QuadratureConfig,
) -> FullAmplitude {
    // Irreducible diagram (the dominant one)
    let irr = irreducible::irreducible_amplitude(config, channel, loop_type, quad_config);

    // Tadpole (proportional to tree level, so has the same helicity structure)
    let tad_coeff = tadpole::tadpole_proper_time(config, loop_type, quad_config);
    let tree = tree_level::tree_level_amplitude(config, channel);
    // Tadpole modifies the tree-level amplitude by a multiplicative correction
    let tad = tree * tad_coeff;

    // External leg (uses vacuum polarization)
    let ext = external::external_amplitude(config, channel.photon_pol(), loop_type, quad_config);
    // External has the same polarization structure as tree level
    let ext_with_grav = ext
        * match channel.graviton_pol() {
            GravitonPolarization::Plus => Complex64::new(config.theta.cos(), 0.0),
            GravitonPolarization::Cross => Complex64::new(1.0, 0.0),
        };

    let total = irr + tad + ext_with_grav;

    // Conversion probability: |M|^2 (in natural units)
    let conversion_probability = total.norm_sqr();

    FullAmplitude {
        irreducible: irr,
        tadpole: tad,
        external: ext_with_grav,
        total,
        conversion_probability,
    }
}

/// Compute the magnetic dichroism for one-loop photon-graviton conversion.
///
/// Dichroism measures the difference in conversion rates between photons
/// polarized parallel vs perpendicular to the magnetic field:
///
///   R = rate_parallel / rate_perpendicular
///
/// At weak field for spinor QED: R = (4/7)^2 ~ 0.327 (amplitude ratio 4/7).
/// At tree level: R = 1 (no dichroism).
///
/// # Arguments
/// * `config` -- Background field configuration (pure magnetic)
/// * `loop_type` -- Scalar or spinor loop
/// * `quad_config` -- Quadrature settings
pub fn magnetic_dichroism(
    config: &FieldConfig,
    loop_type: LoopType,
    quad_config: &QuadratureConfig,
) -> DichroismResult {
    // Sum over graviton polarizations for each photon polarization
    let mut rate_par = 0.0_f64;
    let mut rate_perp = 0.0_f64;

    for &ch in &HelicityChannel::ALL {
        let amp = one_loop_amplitude(config, loop_type, ch, quad_config);
        match ch.photon_pol() {
            PhotonPolarization::Parallel => rate_par += amp.conversion_probability,
            PhotonPolarization::Perpendicular => rate_perp += amp.conversion_probability,
        }
    }

    let ratio = if rate_perp > 0.0 {
        rate_par / rate_perp
    } else {
        1.0
    };

    DichroismResult {
        rate_parallel: rate_par,
        rate_perp,
        ratio,
    }
}

/// Quick diagnostic: verify all Ward identities pass.
pub fn verify_ward_identities(
    config: &FieldConfig,
    loop_type: LoopType,
    quad_config: &QuadratureConfig,
    tolerance: f64,
) -> Vec<WardCheckResult> {
    ward::full_ward_check(config, loop_type, quad_config, tolerance)
}

// ============================================================================
// Tests (legacy + new)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Legacy tests (preserved from original photon_graviton.rs) --

    #[test]
    fn test_schwinger_critical_field() {
        let b_cr = schwinger_critical_field();
        assert!(
            b_cr > 4.4e9 && b_cr < 4.5e9,
            "B_cr = {b_cr:.3e}, expected ~4.41e9 T"
        );
    }

    #[test]
    fn test_gravitational_coupling_tiny() {
        let kappa = gravitational_coupling();
        assert!(kappa > 0.0, "kappa must be positive");
        assert!(kappa < 1e-20, "kappa = {kappa:.3e}, expected << 1");
    }

    #[test]
    fn test_mixing_amplitude_negligible() {
        let (_b_cr, _kappa, ratio) = mixing_amplitude_estimate(10.0);
        assert!(
            ratio < 1e-15,
            "Amplitude ratio {ratio:.3e} unexpectedly large"
        );
    }

    #[test]
    fn test_mixing_amplitude_zero_field() {
        let (_b_cr, _kappa, ratio) = mixing_amplitude_estimate(0.0);
        assert!(ratio == 0.0, "Zero field should give zero mixing amplitude");
    }

    #[test]
    fn test_mixing_amplitude_scales_quadratically() {
        let (_, _, r1) = mixing_amplitude_estimate(1.0);
        let (_, _, r10) = mixing_amplitude_estimate(10.0);
        let scale = r10 / r1;
        assert!(
            (scale - 100.0).abs() < 1e-10,
            "Amplitude should scale as B^2: ratio = {scale}"
        );
    }

    // -- New integration tests --

    #[test]
    fn test_one_loop_amplitude_structure() {
        let config = FieldConfig::pure_magnetic(0.01, 0.1, PI / 4.0);
        let quad = QuadratureConfig::fast();
        let amp = one_loop_amplitude(
            &config,
            LoopType::Spinor,
            HelicityChannel::ParallelCross,
            &quad,
        );
        // Total should be the sum of the three diagrams
        let sum = amp.irreducible + amp.tadpole + amp.external;
        assert!(
            (amp.total - sum).norm() < 1e-25,
            "Total != sum of parts: {} vs {}",
            amp.total,
            sum
        );
    }

    #[test]
    fn test_one_loop_amplitude_zero_field() {
        let config = FieldConfig::pure_magnetic(0.0, 0.1, PI / 4.0);
        let quad = QuadratureConfig::fast();
        let amp = one_loop_amplitude(
            &config,
            LoopType::Spinor,
            HelicityChannel::ParallelCross,
            &quad,
        );
        assert!(amp.total.norm() < 1e-25, "Amplitude should vanish at B=0");
        assert!(
            amp.conversion_probability < 1e-50,
            "Conversion probability should vanish at B=0"
        );
    }

    #[test]
    fn test_one_loop_all_channels() {
        let config = FieldConfig::pure_magnetic(0.01, 0.1, PI / 4.0);
        let quad = QuadratureConfig::fast();
        for ch in HelicityChannel::ALL {
            let amp = one_loop_amplitude(&config, LoopType::Spinor, ch, &quad);
            assert!(amp.total.is_finite(), "Amplitude not finite for {:?}", ch);
        }
    }

    #[test]
    fn test_magnetic_dichroism_basic() {
        let config = FieldConfig::pure_magnetic(0.01, 0.1, PI / 4.0);
        let quad = QuadratureConfig::fast();
        let dich = magnetic_dichroism(&config, LoopType::Spinor, &quad);
        assert!(dich.rate_parallel >= 0.0);
        assert!(dich.rate_perp >= 0.0);
        assert!(dich.ratio.is_finite());
    }

    #[test]
    fn test_ward_identities_pass() {
        let config = FieldConfig::pure_magnetic(0.01, 0.1, PI / 4.0);
        let quad = QuadratureConfig::default();
        let results = verify_ward_identities(&config, LoopType::Spinor, &quad, 1e-8);
        for r in &results {
            assert!(
                r.passes,
                "Ward identity {:?} failed: residual = {}",
                r.identity, r.residual
            );
        }
    }

    #[test]
    fn test_conversion_probability_tiny() {
        // For laboratory fields (B << B_cr), conversion probability is negligible
        let config = FieldConfig::pure_magnetic(1e-9, 0.1, PI / 4.0);
        let quad = QuadratureConfig::fast();
        let amp = one_loop_amplitude(
            &config,
            LoopType::Spinor,
            HelicityChannel::ParallelCross,
            &quad,
        );
        assert!(
            amp.conversion_probability < 1e-50,
            "Conversion probability too large for lab field: {:.3e}",
            amp.conversion_probability
        );
    }
}
