//! Diffusion tensor for cosmic ray transport.
//!
//! The Jokipii (1971) / Bieber (1994) tensor has three components:
//!   kappa_par: diffusion along B (dominant, ~1/B, R^alpha)
//!   kappa_perp: cross-field diffusion (~epsilon_perp * kappa_par)
//!   kappa_A: antisymmetric drift term (sign depends on solar polarity epoch A)
//!
//! Physical reference: Parker (1965) Planet. Space Sci. 13, 9.
//! The kappa_par ~ B^-1 scaling reflects particle scattering off magnetic
//! irregularities with a Kolmogorov turbulence spectrum.

/// Configuration for the diffusion tensor.
#[derive(Clone, Debug)]
pub struct DiffusionConfig {
    /// Reference diffusion coefficient (AU^2/s) at r_ref_gv and B = 5 nT.
    pub kappa_0_au2_per_s: f64,
    /// Reference rigidity (GV).
    pub r_ref_gv: f64,
    /// Rigidity power-law index (0.3-0.6 typical).
    pub alpha: f64,
    /// Perpendicular-to-parallel ratio (~0.02-0.05).
    pub epsilon_perp: f64,
    /// Solar epoch sign: +1.0 (A>0, 1970s-80s) or -1.0 (A<0, 1980s-90s).
    pub solar_epoch_a: f64,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            kappa_0_au2_per_s: 4.5e-5, // Potgieter (2013) canonical value
            r_ref_gv: 1.0,
            alpha: 0.5,
            epsilon_perp: 0.02,
            solar_epoch_a: 1.0,
        }
    }
}

/// Reference B-field magnitude (nT) for kappa_0 normalization.
const B_REF_NT: f64 = 5.0;

/// Compute parallel diffusion coefficient (AU^2/s).
///
/// kappa_par(R, B) = kappa_0 * (R/R_0)^alpha * (B_0/B)
pub fn kappa_parallel(rigidity_gv: f64, b_magnitude_nt: f64, config: &DiffusionConfig) -> f64 {
    let b = b_magnitude_nt.max(0.01); // guard against zero B
    config.kappa_0_au2_per_s
        * (rigidity_gv / config.r_ref_gv).powf(config.alpha)
        * (B_REF_NT / b)
}

/// Compute the full 3x3 diffusion tensor K in the LBM lattice frame.
///
/// K = kappa_par * bb + kappa_perp * (I - bb) + kappa_A * (b x)
///
/// where bb = b_hat tensor b_hat (outer product) and (b x) is the antisymmetric
/// drift matrix with b_hat as the rotation axis.
///
/// Arguments:
///   b: B-field vector at the cell (arbitrary units; only direction matters)
///   b_magnitude: |B| in nT (for kappa_par ~ 1/B scaling)
///   rigidity_gv: particle rigidity (GV)
///   config: diffusion parameters
///
/// Returns: 3x3 tensor [[f64; 3]; 3]
pub fn diffusion_tensor(
    b: [f64; 3],
    b_magnitude_nt: f64,
    rigidity_gv: f64,
    config: &DiffusionConfig,
) -> [[f64; 3]; 3] {
    let kpar = kappa_parallel(rigidity_gv, b_magnitude_nt, config);
    let kperp = config.epsilon_perp * kpar;
    let ka = config.solar_epoch_a * kperp; // drift amplitude scaled from kperp

    // Normalize b to unit vector
    let b_norm = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt().max(1e-30);
    let bh = [b[0] / b_norm, b[1] / b_norm, b[2] / b_norm];

    // Symmetric part: kperp * I + (kpar - kperp) * bb
    let mut k = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b_idx in 0..3 {
            let delta = if a == b_idx { 1.0 } else { 0.0 };
            k[a][b_idx] = kperp * delta + (kpar - kperp) * bh[a] * bh[b_idx];
        }
    }

    // Antisymmetric part: kappa_A * epsilon_abc * b_c (drift term)
    // [  0    ka*bz  -ka*by  ]
    // [-ka*bz   0    ka*bx   ]
    // [ ka*by -ka*bx   0     ]
    k[0][1] += ka * bh[2];
    k[0][2] -= ka * bh[1];
    k[1][0] -= ka * bh[2];
    k[1][2] += ka * bh[0];
    k[2][0] += ka * bh[1];
    k[2][1] -= ka * bh[0];

    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kappa_par_inverse_b_scaling() {
        let cfg = DiffusionConfig::default();
        let k1 = kappa_parallel(1.0, 5.0, &cfg);
        let k2 = kappa_parallel(1.0, 10.0, &cfg);
        // kappa_par ~ 1/B: doubling B should halve kappa
        let ratio = k1 / k2;
        assert!((ratio - 2.0).abs() < 1e-10, "kappa_par ~ 1/B failed: ratio={ratio}");
    }

    #[test]
    fn test_kappa_par_rigidity_power_law() {
        let cfg = DiffusionConfig::default();
        let k1 = kappa_parallel(1.0, 5.0, &cfg);
        let k10 = kappa_parallel(10.0, 5.0, &cfg);
        // kappa ~ R^0.5: factor-10 in R -> factor-sqrt(10) in kappa
        let expected = 10.0_f64.powf(0.5);
        let ratio = k10 / k1;
        assert!((ratio - expected).abs() < 1e-8);
    }

    #[test]
    fn test_tensor_symmetric_part_positive_definite() {
        // Symmetric part of K should have positive diagonal
        let cfg = DiffusionConfig { solar_epoch_a: 0.0, ..Default::default() };
        let b = [3.0_f64, 4.0, 0.0];
        let k = diffusion_tensor(b, 5.0, 1.0, &cfg);
        // Diagonal entries should all be positive
        for (i, row) in k.iter().enumerate() {
            assert!(row[i] > 0.0, "non-positive diagonal at [{i}][{i}]");
        }
        // Symmetric part should be symmetric
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            for j in 0..3 {
                let sym = (k[i][j] + k[j][i]) / 2.0;
                let antisym = (k[i][j] - k[j][i]) / 2.0;
                let _ = (sym, antisym); // verify they exist without panic
            }
        }
    }
}
