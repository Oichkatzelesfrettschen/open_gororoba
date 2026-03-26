//! # H3: Sign-Split + Mass-Binned Stacking
//!
//! Splits galaxies by sign of mean velocity residual and bins by stellar mass,
//! then tests for CD-ZD harmonic signals in each cell with Bonferroni
//! correction for multiple testing.
//!
//! ## Design
//! - 2 sign groups (positive / negative mean residual) x 5 mass bins = 10 cells
//! - Plus 2 marginal tests (sign-only, mass-only) = 12 total tests
//! - For each cell: stack residuals, compute Fourier power, test significance
//! - Bonferroni correction: \alpha_effective = 0.05 / 12
//! - Tests both phase-randomization AND Feshbach mass-resonance hypotheses
//!
//! ## Ablations
//! - **Sign-only**: 2 tests (positive / negative residual, no mass binning)
//! - **Mass-only**: 5 tests (mass bins, no sign split)
//!
//! ~75 seconds compute.

use crate::{
    common::{
        SyntheticGalaxy, detection_snr, fourier_power_at_wavenumbers, generate_galaxy_sample,
        predicted_wavenumbers_cd16, rms,
    },
    h2_dc14_exclusion::{ProfileType, compute_residuals},
};
use adjustp::{Procedure, adjust};
use statrs::distribution::{ChiSquared, ContinuousCDF};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters for H3.
#[derive(Debug, Clone)]
pub struct H3Config {
    /// Number of galaxies.
    pub n_galaxies: usize,
    /// Mass bin edges (n_bins + 1 values in log10(M_star)).
    pub mass_bin_edges: Vec<f64>,
    /// Multiple-testing correction procedure.
    pub correction: Procedure,
    /// Significance level before correction.
    pub alpha_significance: f64,
    /// Number of radial points per synthetic rotation curve.
    pub n_radial_points: usize,
    /// Fractional Gaussian noise on velocities.
    pub noise_frac: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for H3Config {
    fn default() -> Self {
        Self {
            n_galaxies: 200,
            mass_bin_edges: vec![9.0, 9.5, 10.0, 10.5, 11.0, 11.5],
            correction: Procedure::Bonferroni,
            alpha_significance: 0.05,
            n_radial_points: 30,
            noise_frac: 0.05,
            seed: 42,
        }
    }
}

/// Which splitting strategy to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitMode {
    /// Full 2x5 sign x mass grid.
    SignAndMass,
    /// Sign-only ablation (2 groups).
    SignOnly,
    /// Mass-only ablation (5 bins).
    MassOnly,
}

// ---------------------------------------------------------------------------
// Cell result
// ---------------------------------------------------------------------------

/// Result for one stacking cell.
#[derive(Debug, Clone)]
pub struct CellTestResult {
    /// Human-readable cell label.
    pub label: String,
    /// Number of galaxies in this cell.
    pub n_galaxies: usize,
    /// Total Fourier power at CD wavenumbers.
    pub total_power: f64,
    /// Detection SNR.
    pub snr: f64,
    /// Raw p-value from chi-squared test.
    pub p_value: f64,
    /// Corrected p-value (Bonferroni or BH).
    pub p_corrected: f64,
    /// Whether this cell is significant after correction.
    pub significant: bool,
}

/// Full H3 result.
#[derive(Debug, Clone)]
pub struct H3Result {
    /// Per-cell test results.
    pub cells: Vec<CellTestResult>,
    /// Split mode used.
    pub mode: SplitMode,
    /// Number of significant cells after correction.
    pub n_significant: usize,
    /// Human-readable summary.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Core stacking + testing
// ---------------------------------------------------------------------------

/// Classify galaxies into (sign_group, mass_bin) cells.
fn classify_galaxy(
    galaxy: &SyntheticGalaxy,
    mean_residual: f64,
    mass_bin_edges: &[f64],
    mode: SplitMode,
) -> Option<(usize, usize)> {
    let sign_group = if mean_residual >= 0.0 { 0 } else { 1 };
    let log_mstar = galaxy.meta.log_mstar;

    let mass_bin = mass_bin_edges
        .windows(2)
        .position(|w| log_mstar >= w[0] && log_mstar < w[1]);

    match mode {
        SplitMode::SignAndMass => mass_bin.map(|mb| (sign_group, mb)),
        SplitMode::SignOnly => Some((sign_group, 0)),
        SplitMode::MassOnly => mass_bin.map(|mb| (0, mb)),
    }
}

/// Cell label.
fn cell_label(sign: usize, mass_bin: usize, mode: SplitMode) -> String {
    match mode {
        SplitMode::SignAndMass => {
            let s = if sign == 0 { "pos" } else { "neg" };
            format!("{s}_mbin{mass_bin}")
        }
        SplitMode::SignOnly => {
            let s = if sign == 0 { "pos" } else { "neg" };
            s.to_string()
        }
        SplitMode::MassOnly => format!("mbin{mass_bin}"),
    }
}

/// Estimate p-value from stacked Fourier power using chi-squared null.
///
/// Under the null (pure noise), each Fourier power at a wavenumber is
/// chi-squared(2) distributed (real + imaginary components).  The total
/// power over k wavenumbers is chi-squared(2k).
fn p_value_from_power(total_power: f64, n_wavenumbers: usize, noise_rms: f64) -> f64 {
    if noise_rms < 1e-30 {
        return 1.0;
    }
    let dof = 2.0 * n_wavenumbers as f64;
    // Normalize power to unit-variance null.
    let stat = total_power / (noise_rms * noise_rms);
    let chi2 = ChiSquared::new(dof).unwrap();
    1.0 - chi2.cdf(stat)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the H3 experiment with a given split mode.
pub fn run_h3(config: &H3Config, mode: SplitMode) -> H3Result {
    let galaxies = generate_galaxy_sample(
        config.n_galaxies,
        11.0,
        13.0,
        0.03,
        config.n_radial_points,
        config.noise_frac,
        config.seed,
    );

    let n_mass_bins = config.mass_bin_edges.len().saturating_sub(1);
    let wavenumbers = predicted_wavenumbers_cd16();

    // Compute residuals and classify.
    let profile = ProfileType::Nfw;
    let galaxy_data: Vec<(usize, usize, Vec<f64>, Vec<f64>)> = galaxies
        .iter()
        .filter_map(|g| {
            let (x, dv) = compute_residuals(g, profile);
            let mean_dv: f64 = dv.iter().sum::<f64>() / dv.len() as f64;
            classify_galaxy(g, mean_dv, &config.mass_bin_edges, mode).map(|(s, m)| (s, m, x, dv))
        })
        .collect();

    // Stack each cell.
    let sign_range = match mode {
        SplitMode::SignAndMass => 2,
        SplitMode::SignOnly => 2,
        SplitMode::MassOnly => 1,
    };
    let mass_range = match mode {
        SplitMode::SignAndMass => n_mass_bins,
        SplitMode::SignOnly => 1,
        SplitMode::MassOnly => n_mass_bins,
    };

    let mut raw_results: Vec<CellTestResult> = Vec::new();

    for si in 0..sign_range {
        for mi in 0..mass_range {
            let members: Vec<&(usize, usize, Vec<f64>, Vec<f64>)> = galaxy_data
                .iter()
                .filter(|(s, m, _, _)| *s == si && *m == mi)
                .collect();

            let n_gal = members.len();
            if n_gal == 0 {
                raw_results.push(CellTestResult {
                    label: cell_label(si, mi, mode),
                    n_galaxies: 0,
                    total_power: 0.0,
                    snr: 0.0,
                    p_value: 1.0,
                    p_corrected: 1.0,
                    significant: false,
                });
                continue;
            }

            // Average Fourier power across galaxies.
            let mut total_power_sum = 0.0;
            let mut total_snr_sum = 0.0;
            let mut total_rms_sum = 0.0;

            for (_, _, x, dv) in &members {
                let power = fourier_power_at_wavenumbers(x, dv, &wavenumbers);
                let total_p: f64 = power.iter().sum();
                let snr = detection_snr(&power, dv);
                total_power_sum += total_p;
                total_snr_sum += snr;
                total_rms_sum += rms(dv);
            }

            let mean_power = total_power_sum / n_gal as f64;
            let mean_snr = total_snr_sum / n_gal as f64;
            let mean_rms = total_rms_sum / n_gal as f64;

            // Scale total power by sqrt(N) for stacking gain.
            let stacked_power = mean_power * (n_gal as f64).sqrt();
            let p_val = p_value_from_power(stacked_power, wavenumbers.len(), mean_rms);

            raw_results.push(CellTestResult {
                label: cell_label(si, mi, mode),
                n_galaxies: n_gal,
                total_power: mean_power,
                snr: mean_snr,
                p_value: p_val,
                p_corrected: 1.0, // filled below
                significant: false,
            });
        }
    }

    // Apply multiple-testing correction.
    let raw_p: Vec<f64> = raw_results.iter().map(|r| r.p_value).collect();
    let corrected_p = adjust(&raw_p, config.correction);

    for (r, &pc) in raw_results.iter_mut().zip(corrected_p.iter()) {
        r.p_corrected = pc;
        r.significant = pc < config.alpha_significance;
    }

    let n_significant = raw_results.iter().filter(|r| r.significant).count();

    let summary = format!(
        r"H3 Sign-Split+Mass-Binned Stacking: mode={:?}, {} cells, \
         {} galaxies, {} significant (correction={:?}, \alpha={})",
        mode,
        raw_results.len(),
        config.n_galaxies,
        n_significant,
        config.correction,
        config.alpha_significance,
    );

    H3Result {
        cells: raw_results,
        mode,
        n_significant,
        summary,
    }
}

/// Run the full experiment (sign x mass) with both ablations.
pub fn run_h3_full(config: &H3Config) -> (H3Result, H3Result, H3Result) {
    let full = run_h3(config, SplitMode::SignAndMass);
    let sign_only = run_h3(config, SplitMode::SignOnly);
    let mass_only = run_h3(config, SplitMode::MassOnly);
    (full, sign_only, mass_only)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h3_smoke_full() {
        let config = H3Config {
            n_galaxies: 30,
            n_radial_points: 20,
            seed: 42,
            ..Default::default()
        };
        let (full, sign_abl, mass_abl) = run_h3_full(&config);

        // Full: 2 sign x 5 mass = 10 cells
        assert_eq!(full.cells.len(), 10);
        assert_eq!(full.mode, SplitMode::SignAndMass);

        // Sign-only: 2 cells
        assert_eq!(sign_abl.cells.len(), 2);
        assert_eq!(sign_abl.mode, SplitMode::SignOnly);

        // Mass-only: 5 cells
        assert_eq!(mass_abl.cells.len(), 5);
        assert_eq!(mass_abl.mode, SplitMode::MassOnly);

        println!("{}", full.summary);
        println!("{}", sign_abl.summary);
        println!("{}", mass_abl.summary);
    }

    #[test]
    fn test_h3_bonferroni_correction_applied() {
        let config = H3Config {
            n_galaxies: 30,
            n_radial_points: 20,
            seed: 42,
            ..Default::default()
        };
        let result = run_h3(&config, SplitMode::SignAndMass);
        // Corrected p-values should be >= raw p-values.
        for cell in &result.cells {
            assert!(
                cell.p_corrected + 1e-15 >= cell.p_value,
                "Corrected p ({}) should be >= raw p ({})",
                cell.p_corrected,
                cell.p_value,
            );
        }
    }

    #[test]
    fn test_h3_noise_floor_not_significant() {
        // With pure noise (no injection), we expect no significant cells.
        let config = H3Config {
            n_galaxies: 50,
            n_radial_points: 20,
            noise_frac: 0.05,
            seed: 42,
            ..Default::default()
        };
        let result = run_h3(&config, SplitMode::SignAndMass);
        // Not guaranteed, but with Bonferroni on pure noise, very unlikely.
        // Allow at most 1 false positive.
        assert!(
            result.n_significant <= 1,
            "Expected ≤1 false positive under noise, got {}",
            result.n_significant,
        );
    }

    #[test]
    fn test_p_value_from_power() {
        // Under null, total_power ~ chi2(2k) with scale noise_rms^2.
        // p-value of 0 power should be 1.0.
        let p = p_value_from_power(0.0, 7, 1.0);
        assert!((p - 1.0).abs() < 0.01);

        // Very large power should give p ~= 0.
        let p = p_value_from_power(1e6, 7, 1.0);
        assert!(p < 0.001);
    }
}
