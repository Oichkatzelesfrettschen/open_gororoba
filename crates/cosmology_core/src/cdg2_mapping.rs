//! CDG-2 parameter mapping: spectral dimension at Perseus redshift and
//! dark halo falsification.
//!
//! Maps orthoplex diffusion model parameters to CDG-2 observables (Li et al.
//! 2025, arXiv:2506.15644). Used by the dark_halo_hunt binary to test whether
//! the ZD-mediated viscosity mechanism can produce structures consistent with
//! the baryon-free ultra-diffuse galaxy CDG-2 in Perseus.

use crate::{
    halo::{CDG2_DM_FRACTION_MIN, CDG2_STELLAR_MASS_SOLAR, PERSEUS_Z},
    orthoplex_diffusion::{diffusion_time, spectral_dimension_k22},
};

/// Spectral dimension at CDG-2 redshift (z = 0.0179).
///
/// d_s(z) evaluated at the Perseus cluster redshift where CDG-2 resides.
/// A larger d_s implies stronger ZD bottleneck effects at this epoch.
pub fn spectral_dim_at_cdg2(k: usize, alpha: f64, t_0: f64) -> f64 {
    let t = diffusion_time(PERSEUS_Z, alpha, t_0);
    spectral_dimension_k22(t, k)
}

/// Compute ZD bottleneck volume fraction from a GPU halo_mask readback.
///
/// The halo_mask array (from dark_halo_detector.wgsl) contains 0 (background)
/// or 1 (halo candidate) per grid cell. The volume fraction is simply
/// n_halo / n_total.
pub fn zd_bottleneck_volume_fraction(halo_mask: &[u32]) -> f64 {
    if halo_mask.is_empty() {
        return 0.0;
    }
    let n_halo = halo_mask.iter().filter(|&&v| v == 1).count();
    n_halo as f64 / halo_mask.len() as f64
}

/// Result of evaluating CDG-2 consistency with the orthoplex model.
#[derive(Clone, Debug)]
pub struct DarkHaloFalsificationResult {
    /// Volume fraction of ZD bottleneck cells in the LBM simulation.
    pub zd_volume_fraction: f64,
    /// Predicted baryon fraction from the volume fraction.
    /// f_baryon_pred = 1 - zd_volume_fraction (halos are baryon-empty).
    pub predicted_f_baryon: f64,
    /// Observed CDG-2 baryon fraction upper bound (1 - 0.9994 = 0.0006).
    pub observed_f_baryon: f64,
    /// |predicted - observed|.
    pub delta_f_baryon: f64,
    /// Whether the model is consistent: |pred - obs| < 2 * sigma.
    pub consistent: bool,
}

/// Evaluate CDG-2 consistency from ZD bottleneck volume fraction.
///
/// CDG-2 has a dark matter fraction >= 99.94% (Li et al. 2025), meaning
/// baryonic fraction <= 0.06%. The orthoplex model predicts that ZD
/// bottleneck regions trap baryons in gravitational wells while creating
/// baryon-empty halos. If the simulated volume fraction of such halos
/// is consistent with the observed baryon deficit, the model is supported.
///
/// The sigma used for the consistency check is 0.0003 (half the observed
/// baryon fraction, representing ~50% relative uncertainty on f_baryon).
pub fn evaluate_cdg2_consistency(volume_fraction: f64) -> DarkHaloFalsificationResult {
    let observed_f_baryon = 1.0 - CDG2_DM_FRACTION_MIN; // 0.0006
    let predicted_f_baryon = 1.0 - volume_fraction;
    let delta = (predicted_f_baryon - observed_f_baryon).abs();
    let sigma = 0.0003; // ~50% relative uncertainty

    DarkHaloFalsificationResult {
        zd_volume_fraction: volume_fraction,
        predicted_f_baryon,
        observed_f_baryon,
        delta_f_baryon: delta,
        consistent: delta < 2.0 * sigma,
    }
}

// Suppress unused import warnings for constants referenced in doc comments
// and used in evaluate_cdg2_consistency indirectly through CDG2_DM_FRACTION_MIN.
const _: f64 = CDG2_STELLAR_MASS_SOLAR;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_dim_at_cdg2_range() {
        // At Perseus z=0.0179, d_s should be positive and bounded for k=63
        let ds = spectral_dim_at_cdg2(63, 5.0, 0.01);
        assert!(ds >= 0.0, "d_s must be non-negative, got {ds}");
        assert!(ds < 10.0, "d_s must be bounded, got {ds}");
    }

    #[test]
    fn test_spectral_dim_at_cdg2_varies_with_alpha() {
        // Use physically motivated t_0=0.01 (Sprint 60 best-fit) to stay in
        // the regime where d_s is appreciable and varies with alpha.
        let ds1 = spectral_dim_at_cdg2(63, 1.0, 0.01);
        let ds2 = spectral_dim_at_cdg2(63, 10.0, 0.01);
        // Different alpha should give different d_s
        assert!(
            (ds1 - ds2).abs() > 1e-10,
            "d_s should vary with alpha: ds1={ds1}, ds2={ds2}"
        );
    }

    #[test]
    fn test_volume_fraction_empty() {
        assert_eq!(zd_bottleneck_volume_fraction(&[]), 0.0);
    }

    #[test]
    fn test_volume_fraction_all_halo() {
        let mask = vec![1u32; 1000];
        let vf = zd_bottleneck_volume_fraction(&mask);
        assert!((vf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_fraction_mixed() {
        let mut mask = vec![0u32; 1000];
        mask[0] = 1;
        mask[1] = 1;
        let vf = zd_bottleneck_volume_fraction(&mask);
        assert!((vf - 0.002).abs() < 1e-10);
    }

    #[test]
    fn test_cdg2_falsification_thresholds() {
        // Volume fraction ~0.9994 should be consistent (baryon fraction ~0.0006)
        let result = evaluate_cdg2_consistency(0.9994);
        assert!(
            result.consistent,
            "Volume fraction 0.9994 should be consistent with CDG-2"
        );
        assert!((result.observed_f_baryon - 0.0006).abs() < 1e-10);

        // Volume fraction 0 means all cells are background (baryon fraction = 1)
        let result_zero = evaluate_cdg2_consistency(0.0);
        assert!(
            !result_zero.consistent,
            "Volume fraction 0 should NOT be consistent with CDG-2"
        );

        // Volume fraction 0.5 means baryon fraction = 0.5 (way too high)
        let result_half = evaluate_cdg2_consistency(0.5);
        assert!(
            !result_half.consistent,
            "Volume fraction 0.5 should NOT be consistent with CDG-2"
        );
    }
}
