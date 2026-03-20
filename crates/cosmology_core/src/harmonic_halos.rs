//! Harmonic Halos: 7-mode box-kite rotation curve modulation.
//!
//! Implements the SKA (2030) "Harmonic Halos" prediction: galactic rotation
//! curves acquire harmonic modulations from sedenion zero-divisor topology.
//!
//! # Physics
//!
//! The 7 sedenion box-kites define 7 angular symmetry modes on a sphere.
//! Each box-kite has a unique strut signature (the missing octonion index
//! 1..7). The 3 strut pairs per box-kite define preferred angular orientations
//! in the Cayley-Dickson tower. When projected onto 3D via mod-8 folding,
//! these give 7 harmonic modulation modes for galactic rotation curves.
//!
//! The base amplitude derives from the assessor fraction: 42 primitive
//! assessors out of 84 total zero-divisors gives a flat-band fraction of
//! exactly 0.5 (proven constant for dim >= 16).
//!
//! At `alpha_zd = 0`, all modulations vanish and standard NFW is recovered
//! exactly. The `alpha_zd` parameter controls forcing strength (typical
//! range 0.001--0.1 for sub-percent velocity modulations).
//!
//! Wavenumber: `k_n = 2*pi*n / (7 * r_s)` for mode n = 1..7, where r_s is
//! the NFW scale radius. The factor of 7 ties the fundamental wavelength to
//! the box-kite count.
//!
//! # Literature
//!
//! - de Marrais (2000): arXiv:math/0011260 (42 assessors, 7 box-kites)
//! - Reggiani (2024): partner graph spectrum {-4,-2,0,+2,+4}
//! - C-1313, C-1314: harmonic halo modulation claims

use std::f64::consts::PI;

use gr_core::adm_algebra_bridge::algebraic_stress_energy_correction;
use crate::nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass};

/// Gravitational constant in kpc^3 Msun^{-1} (km/s)^2 units.
///
/// G = 4.302e-3 pc (km/s)^2 Msun^{-1} = 4.302e-6 kpc (km/s)^2 Msun^{-1}.
const G_KPC_KMS2: f64 = 4.302e-6;

/// Number of box-kites in sedenions (algebraic constant, D=16).
const N_BOXKITES_D16: usize = 7;

/// Assessor fraction: 42 primitive assessors / 84 total ZDs = 0.5.
/// This is the flat-band fraction for dim >= 16 Cayley-Dickson algebras.
const ASSESSOR_FRACTION: f64 = 0.5;

/// Configuration for harmonic halo modulation.
#[derive(Debug, Clone)]
pub struct HarmonicHaloConfig {
    /// ZD forcing strength (0.0 = disabled, typical 0.001--0.1).
    pub alpha_zd: f64,
    /// Number of motif-component modes to include (default: dim/2 - 1).
    pub n_modes: usize,
    /// Total motif components for this CD dimension (= dim/2 - 1).
    pub n_components: usize,
    /// NFW scale radius in kpc (sets modulation wavelength).
    pub r_s_kpc: f64,
    /// Phase offsets per mode (from strut geometry / uniform distribution).
    pub phases: Vec<f64>,
    /// Assessor fraction (0.5 for D=16, conjectured universal).
    pub assessor_fraction: f64,
    /// Cayley-Dickson dimension for this config (16, 32, 64...).
    pub cd_dim: usize,
}

impl HarmonicHaloConfig {
    /// Create config for sedenions (D=16) with default phases.
    ///
    /// Each box-kite's strut signature (missing octonion index k = 1..7)
    /// maps to a phase offset phi_k = 2*pi*k / 7. This distributes the
    /// 7 modes uniformly around the unit circle.
    pub fn new(alpha_zd: f64, n_modes: usize, r_s_kpc: f64) -> Self {
        Self::new_cd(alpha_zd, n_modes, r_s_kpc, 16)
    }

    /// Create config for an arbitrary CD dimension.
    ///
    /// n_components = dim/2 - 1 from the PG(n-2,2) correspondence.
    /// Phases are distributed uniformly: phi_k = 2*pi*k / n_components.
    pub fn new_cd(alpha_zd: f64, n_modes: usize, r_s_kpc: f64, cd_dim: usize) -> Self {
        assert!(
            cd_dim >= 16 && cd_dim.is_power_of_two(),
            "CD dimension must be a power of 2 >= 16, got {cd_dim}"
        );
        let n_components = cd_dim / 2 - 1;
        let n_modes = n_modes.clamp(1, n_components);

        // Use the exact FBF values derived computationally
        let assessor_fraction = match cd_dim {
            16 => 0.5,
            32 => 4.0 / 7.0, // Pathion Anomaly!
            64 => 1854.0 / 3036.0, // Chingon FBF
            _ => ASSESSOR_FRACTION, // Fallback for higher dims or custom
        };

        Self {
            alpha_zd,
            n_modes,
            n_components,
            r_s_kpc,
            phases: default_phases(n_components),
            assessor_fraction,
            cd_dim,
        }
    }
}

/// Compute default phase offsets for N motif components.
///
/// Uniformly distributed: phi_k = 2*pi*k / N for k = 1..N.
/// At D=16 (N=7), this matches the box-kite strut signatures.
pub fn default_phases(n_components: usize) -> Vec<f64> {
    (1..=n_components)
        .map(|k| 2.0 * PI * k as f64 / n_components as f64)
        .collect()
}

/// Backward-compatible: default phases for sedenion box-kites (7 modes).
pub fn default_box_kite_phases() -> Vec<f64> {
    default_phases(N_BOXKITES_D16)
}

/// Multiplicative modulation factor for v_circ^2 at radius r.
///
/// Returns a factor F such that `v_circ_modulated^2 = v_circ_nfw^2 * F`.
///
/// F = 1.0 + alpha_zd * sum_{n=1}^{n_modes} a_n * cos(k_n * r + phi_n) * exp(-r / r_s)
///
/// where:
/// - k_n = 2*pi*n / (7 * r_s) -- box-kite wavenumber
/// - a_n = ASSESSOR_FRACTION * (1 / n) -- amplitude with 1/n falloff
/// - phi_n = phase offset from strut geometry
///
/// At alpha_zd = 0, returns exactly 1.0.
pub fn harmonic_halo_modulation(r_kpc: f64, config: &HarmonicHaloConfig) -> f64 {
    if config.alpha_zd == 0.0 || config.r_s_kpc <= 0.0 || r_kpc <= 0.0 {
        return 1.0;
    }

    let r_s = config.r_s_kpc;
    let n_comp = config.n_components as f64;
    let decay = (-r_kpc / r_s).exp();
    let mut modulation = 0.0_f64;

    for n in 1..=config.n_modes {
        let k_n = 2.0 * PI * n as f64 / (n_comp * r_s);
        let a_n = config.assessor_fraction / n as f64;
        let phi_n = config.phases[n - 1];
        modulation += a_n * (k_n * r_kpc + phi_n).cos() * decay;
    }

    1.0 + config.alpha_zd * modulation
}

/// Force-level modulation for LBM integration.
///
/// Computes the radial modulation factor for the gravitational force
/// at radius r_kpc. This is the derivative-level quantity: for a
/// potential Phi(r) with modulation, the force F = -dPhi/dr picks up
/// both the modulation value and its radial derivative.
///
/// For simplicity and numerical stability, we use the same multiplicative
/// modulation as the potential (equivalent to assuming the modulation
/// wavelength >> dr). This is valid when alpha_zd << 1.
///
/// At alpha_zd = 0, returns exactly 1.0.
pub fn harmonic_halo_force_modulation(r_kpc: f64, config: &HarmonicHaloConfig) -> f64 {
    harmonic_halo_modulation(r_kpc, config)
}

/// Circular velocity with harmonic halo modulation.
///
/// v_circ(r) = sqrt(G * M_enc(r) / r * F(r))
///
/// where F(r) is the harmonic modulation factor. At alpha_zd = 0,
/// this reduces to the standard NFW circular velocity exactly.
///
/// Returns velocity in km/s.
pub fn v_circ_with_halos(
    r_kpc: f64,
    m200_solar: f64,
    c200: f64,
    z: f64,
    config: &HarmonicHaloConfig,
) -> f64 {
    if r_kpc <= 0.0 {
        return 0.0;
    }

    let nfw = nfw_params_from_mass(m200_solar, z);
    // Use the c200 from the caller if it differs from the CMR default
    let nfw = if (nfw.c200 - c200).abs() > 0.01 {
        // Reconstruct with caller's c200
        crate::nfw_utils::NfwParams {
            r_s_kpc: nfw.r200_kpc / c200,
            rho_s_solar_per_kpc3: crate::nfw_utils::nfw_rho_s_from_c(c200, z),
            c200,
            r200_kpc: nfw.r200_kpc,
            m200_solar: nfw.m200_solar,
            inner_gradient_warning: nfw.inner_gradient_warning,
        }
    } else {
        nfw
    };

    let m_enc = nfw_enclosed_mass_from_params(r_kpc, &nfw);
    let v_sq_nfw = G_KPC_KMS2 * m_enc / r_kpc;
    let modulation = harmonic_halo_modulation(r_kpc, config);

    // Apply algebraic stress-energy correction
    let r_s = nfw.r200_kpc / nfw.c200; // Recalculate NFW scale radius
    let algebraic_correction =
        algebraic_stress_energy_correction(config.cd_dim, r_kpc, r_s, config.alpha_zd);

    // This is a direct addition to the mass term (or force term if thinking about acceleration)
    // Here we treat it as an effective mass density perturbation.
    // So M_effective = M_enc + M_alg, or v_sq_total = v_sq_nfw + v_sq_alg
    let v_sq_alg_correction = G_KPC_KMS2 * algebraic_correction * r_kpc;

    // Ensure non-negative under modulation (modulation can dip below 1.0
    // for negative cosine phases, but v_sq * modulation should stay positive
    // for physically reasonable alpha_zd values)
    (v_sq_nfw * modulation + v_sq_alg_correction).max(0.0).sqrt()
}

/// Standard NFW circular velocity (no modulation), in km/s.
pub fn v_circ_nfw(r_kpc: f64, m200_solar: f64, z: f64) -> f64 {
    if r_kpc <= 0.0 {
        return 0.0;
    }
    let nfw = nfw_params_from_mass(m200_solar, z);
    let m_enc = nfw_enclosed_mass_from_params(r_kpc, &nfw);
    (G_KPC_KMS2 * m_enc / r_kpc).max(0.0).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modulation_unity_at_alpha_zero() {
        let config = HarmonicHaloConfig::new(0.0, 7, 20.0);
        for r in [0.1, 1.0, 10.0, 50.0, 100.0] {
            let m = harmonic_halo_modulation(r, &config);
            assert!(
                (m - 1.0).abs() < 1e-15,
                "modulation at alpha_zd=0 should be 1.0, got {m} at r={r}"
            );
        }
    }

    #[test]
    fn test_modulation_decays_at_large_r() {
        let config = HarmonicHaloConfig::new(0.05, 7, 20.0);
        let m_near = (harmonic_halo_modulation(5.0, &config) - 1.0).abs();
        let m_far = (harmonic_halo_modulation(200.0, &config) - 1.0).abs();
        assert!(
            m_far < m_near,
            "modulation should decay: |m-1| at 200 kpc = {m_far} should be < {m_near} at 5 kpc"
        );
    }

    #[test]
    fn test_modulation_sub_percent_at_small_alpha() {
        let config = HarmonicHaloConfig::new(0.01, 7, 20.0);
        for r in [1.0, 5.0, 10.0, 20.0, 50.0] {
            let m = harmonic_halo_modulation(r, &config);
            let deviation = (m - 1.0).abs();
            assert!(
                deviation < 0.01,
                "deviation at alpha=0.01 should be < 1%, got {:.4}% at r={}",
                deviation * 100.0,
                r
            );
        }
    }

    #[test]
    fn test_v_circ_matches_nfw_at_alpha_zero() {
        let config = HarmonicHaloConfig::new(0.0, 7, 20.0);
        let m200 = 1e12;
        let z = 0.0;
        let nfw = nfw_params_from_mass(m200, z);

        for r in [1.0, 5.0, 10.0, 50.0, 100.0] {
            let v_halo = v_circ_with_halos(r, m200, nfw.c200, z, &config);
            let v_nfw = v_circ_nfw(r, m200, z);
            let rel_err = if v_nfw > 0.0 {
                (v_halo - v_nfw).abs() / v_nfw
            } else {
                0.0
            };
            assert!(
                rel_err < 1e-12,
                "v_circ_with_halos should match v_circ_nfw at alpha=0: rel_err={rel_err:.2e} at r={r}"
            );
        }
    }

    #[test]
    fn test_v_circ_nfw_physical_range() {
        // MW halo: v_circ at ~8 kpc should be ~200 km/s (Sun's orbit)
        let v = v_circ_nfw(8.0, 1e12, 0.0);
        assert!(
            v > 100.0 && v < 400.0,
            "v_circ(8 kpc, MW) = {v:.1} km/s, expected 100-400"
        );
    }

    #[test]
    fn test_seven_distinct_wavenumbers() {
        let r_s = 20.0;
        let mut wavenumbers = Vec::new();
        for n in 1..=7 {
            let k_n = 2.0 * PI * n as f64 / (7.0 * r_s);
            wavenumbers.push(k_n);
        }
        // All distinct
        for i in 0..wavenumbers.len() {
            for j in (i + 1)..wavenumbers.len() {
                assert!(
                    (wavenumbers[i] - wavenumbers[j]).abs() > 1e-10,
                    "wavenumbers k_{} = {:.6} and k_{} = {:.6} should be distinct",
                    i + 1,
                    wavenumbers[i],
                    j + 1,
                    wavenumbers[j]
                );
            }
        }
        // Verify k_n increases with n
        for i in 0..wavenumbers.len() - 1 {
            assert!(wavenumbers[i] < wavenumbers[i + 1]);
        }
    }

    #[test]
    fn test_default_phases_are_distinct() {
        let phases = default_box_kite_phases();
        assert_eq!(phases.len(), 7);
        for i in 0..phases.len() {
            for j in (i + 1)..phases.len() {
                assert!(
                    (phases[i] - phases[j]).abs() > 1e-10,
                    "phases[{i}] = {:.6} and phases[{j}] = {:.6} should be distinct",
                    phases[i],
                    phases[j]
                );
            }
        }
    }

    #[test]
    fn test_force_modulation_equals_potential_modulation() {
        let config = HarmonicHaloConfig::new(0.05, 7, 20.0);
        for r in [1.0, 10.0, 50.0] {
            let m_pot = harmonic_halo_modulation(r, &config);
            let m_force = harmonic_halo_force_modulation(r, &config);
            assert!(
                (m_pot - m_force).abs() < 1e-15,
                "force and potential modulation should match"
            );
        }
    }

    #[test]
    fn test_n_modes_clamping() {
        let config = HarmonicHaloConfig::new(0.05, 0, 20.0);
        assert_eq!(config.n_modes, 1, "n_modes=0 should clamp to 1");

        let config = HarmonicHaloConfig::new(0.05, 100, 20.0);
        assert_eq!(config.n_modes, 7, "n_modes=100 should clamp to 7");
    }

    #[test]
    fn test_modulation_at_r_zero_returns_unity() {
        let config = HarmonicHaloConfig::new(0.05, 7, 20.0);
        assert_eq!(harmonic_halo_modulation(0.0, &config), 1.0);
    }

    #[test]
    fn test_v_circ_at_r_zero() {
        let config = HarmonicHaloConfig::new(0.05, 7, 20.0);
        assert_eq!(v_circ_with_halos(0.0, 1e12, 10.0, 0.0, &config), 0.0);
        assert_eq!(v_circ_nfw(0.0, 1e12, 0.0), 0.0);
    }

    #[test]
    fn test_new_cd_dimensions() {
        // D=32 (pathion): 15 components
        let config32 = HarmonicHaloConfig::new_cd(0.05, 15, 20.0, 32);
        assert_eq!(config32.n_components, 15);
        assert_eq!(config32.n_modes, 15);
        assert_eq!(config32.phases.len(), 15);

        // D=64 (chingon): 31 components
        let config64 = HarmonicHaloConfig::new_cd(0.05, 31, 20.0, 64);
        assert_eq!(config64.n_components, 31);
        assert_eq!(config64.n_modes, 31);
        assert_eq!(config64.phases.len(), 31);

        // D=128 (routon): 63 components
        let config128 = HarmonicHaloConfig::new_cd(0.05, 63, 20.0, 128);
        assert_eq!(config128.n_components, 63);

        // D=256 (voudion): 127 components
        let config256 = HarmonicHaloConfig::new_cd(0.05, 127, 20.0, 256);
        assert_eq!(config256.n_components, 127);
    }

    #[test]
    fn test_modulation_unity_at_alpha_zero_d32() {
        let config = HarmonicHaloConfig::new_cd(0.0, 15, 20.0, 32);
        for r in [0.1, 1.0, 10.0, 50.0, 100.0] {
            let m = harmonic_halo_modulation(r, &config);
            assert!(
                (m - 1.0).abs() < 1e-15,
                "D=32 modulation at alpha_zd=0 should be 1.0, got {m} at r={r}"
            );
        }
    }

    #[test]
    fn test_default_phases_scale_with_dimension() {
        let p7 = default_phases(7);
        let p15 = default_phases(15);
        assert_eq!(p7.len(), 7);
        assert_eq!(p15.len(), 15);
        // Both should have last phase = 2*pi (approximately)
        assert!((p7[6] - 2.0 * PI).abs() < 1e-10);
        assert!((p15[14] - 2.0 * PI).abs() < 1e-10);
        // D=32 fundamental phase is smaller than D=16 fundamental
        assert!(p15[0] < p7[0]);
    }

    #[test]
    fn test_new_cd_backward_compatible_with_new() {
        // new() and new_cd(..., 16) should produce identical configs
        let c1 = HarmonicHaloConfig::new(0.05, 7, 20.0);
        let c2 = HarmonicHaloConfig::new_cd(0.05, 7, 20.0, 16);
        assert_eq!(c1.n_modes, c2.n_modes);
        assert_eq!(c1.n_components, c2.n_components);
        assert!((c1.assessor_fraction - c2.assessor_fraction).abs() < 1e-15);
        for (a, b) in c1.phases.iter().zip(c2.phases.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
