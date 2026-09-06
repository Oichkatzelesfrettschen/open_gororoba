//! Multi-galaxy D_f sweep pipeline: preparation, provenance, and analysis.
//!
//! Orchestrates the conversion of Euclid Q1 catalog entries into LBM-ready
//! density+force fields, tracks per-galaxy provenance, and computes
//! statistical summaries (bootstrap CI, morphological binning).
//!
//! # Pipeline
//!
//! 1. `prepare_galaxy()` converts `EuclidSersicParams` -> `GalaxySetup`
//!    (combined Sersic+NFW density, force field, threshold, metadata).
//! 2. Caller runs LBM evolution (CPU or GPU) on the density field.
//! 3. Caller measures D_f via `box_counting_fractal_dim[_f32]`.
//! 4. `GalaxyDfRecord` captures per-galaxy results with full provenance.
//! 5. `analyze_df_sweep()` produces `DfSweepSummary` with bootstrap CI.

use std::fmt;

#[cfg(feature = "euclid-catalog")]
use crate::euclid_morphology::EuclidSersicParams;
#[cfg(feature = "euclid-catalog")]
use crate::nfw_utils::{
    nfw_density_from_params, nfw_enclosed_mass_from_params, nfw_params_from_mass,
};
#[cfg(feature = "euclid-catalog")]
use crate::sersic::{SersicLbmConfig, sersic_to_lbm_density};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the multi-galaxy D_f sweep pipeline.
#[derive(Debug, Clone)]
pub struct GalaxyPipelineConfig {
    /// Grid dimension (cubic: g x g x g).
    pub grid_dim: usize,
    /// Number of LBM evolution steps.
    pub lbm_steps: usize,
    /// LBM relaxation time (hard floor: >= 1.5).
    pub tau: f64,
    /// Cell size in kpc.
    pub dx_kpc: f64,
    /// Maximum number of galaxies to process.
    pub max_galaxies: usize,
    /// Use morphological M/L instead of catalog M/L.
    pub morphological_ml: bool,
    /// ZD algebraic forcing strength (0.0 = disabled, typical 0.1).
    /// Adds topological confinement from flat band localization.
    pub alpha_zd: f64,
    /// Plummer softening length in units of dx_kpc (default 1.0).
    /// Replaces the singular r^-2 NFW cusp with r^-2 + eps^2, preventing
    /// the gravitational force from creating supersonic velocities at the
    /// halo center where the lattice cell cannot resolve the 1/r divergence.
    pub softening_eps: f64,
    /// Minimum density floor in LBM units (default 0.1).
    /// Controls the max density contrast (rho_peak / floor). At 200:1
    /// contrast, streaming across the density interface creates Ma > 1.0
    /// in a single step, causing compressibility instability regardless
    /// of collision operator (BGK or MRT).
    pub density_floor: f64,
    /// Number of box-kite harmonic modes for ZD forcing (1..=7, default 7).
    /// Only used when alpha_zd > 0.
    pub halo_n_modes: usize,
}

impl Default for GalaxyPipelineConfig {
    fn default() -> Self {
        Self {
            grid_dim: 64,
            lbm_steps: 200,
            tau: 1.5,
            dx_kpc: 1.0,
            max_galaxies: 100,
            morphological_ml: true,
            alpha_zd: 0.0,
            softening_eps: 1.0,
            density_floor: 0.1,
            halo_n_modes: 7,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-galaxy provenance record
// ---------------------------------------------------------------------------

/// Full provenance record for one galaxy's D_f measurement.
#[derive(Debug, Clone)]
pub struct GalaxyDfRecord {
    pub object_id: i64,
    pub sersic_n: f64,
    pub r_e_kpc: f64,
    pub photo_z: f64,
    pub log_stellar_mass: f64,
    pub m200_solar: f64,
    pub c200: f64,
    pub ml_ratio: f64,
    pub morphological_type: String,
    pub grid_dim: usize,
    pub lbm_steps: usize,
    pub tau: f64,
    pub df_initial: f64,
    pub df_final: f64,
    pub elapsed_ms: u64,
}

impl GalaxyDfRecord {
    /// CSV header line.
    pub fn csv_header() -> &'static str {
        "object_id,sersic_n,r_e_kpc,photo_z,log_stellar_mass,\
         m200_solar,c200,ml_ratio,morphological_type,\
         grid_dim,lbm_steps,tau,df_initial,df_final,elapsed_ms"
    }
}

impl fmt::Display for GalaxyDfRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{},{:.4},{:.4},{:.4},{:.4},\
             {:.4e},{:.4},{:.4},{},\
             {},{},{:.4},{:.4},{:.4},{}",
            self.object_id,
            self.sersic_n,
            self.r_e_kpc,
            self.photo_z,
            self.log_stellar_mass,
            self.m200_solar,
            self.c200,
            self.ml_ratio,
            self.morphological_type,
            self.grid_dim,
            self.lbm_steps,
            self.tau,
            self.df_initial,
            self.df_final,
            self.elapsed_ms,
        )
    }
}

// ---------------------------------------------------------------------------
// Galaxy setup (LBM-ready density + force field)
// ---------------------------------------------------------------------------

/// LBM-ready density and force field for one galaxy, plus metadata template.
pub struct GalaxySetup {
    /// Combined Sersic + NFW density (row-major, nx*ny*nz).
    pub rho_total: Vec<f64>,
    /// NFW gravitational force field (normalized, max |F| ~ 1e-4).
    pub force_field: Vec<[f64; 3]>,
    /// Median-density threshold for box-counting.
    pub threshold: f64,
    /// Partially filled record (df_initial/df_final/elapsed_ms to be filled by caller).
    pub record_template: GalaxyDfRecord,
}

// ---------------------------------------------------------------------------
// Galaxy preparation
// ---------------------------------------------------------------------------

/// Prepare one galaxy for LBM simulation.
///
/// Combines Sersic baryonic density with NFW dark matter halo, computes the
/// gravitational force field, and normalizes both to LBM-safe ranges.
///
/// Returns `None` if galaxy params are invalid (NaN, r_e=0, z<=0, etc.).
#[cfg(feature = "euclid-catalog")]
pub fn prepare_galaxy(
    entry: &EuclidSersicParams,
    config: &GalaxyPipelineConfig,
) -> Option<GalaxySetup> {
    let g = config.grid_dim;
    let total = g * g * g;

    // Validate inputs
    if !entry.n.is_finite()
        || entry.n <= 0.0
        || !entry.r_e_arcsec.is_finite()
        || entry.r_e_arcsec <= 0.0
        || entry.photo_z <= 0.0
    {
        return None;
    }

    let r_e_kpc = entry.r_e_kpc();
    if r_e_kpc <= 0.0 || !r_e_kpc.is_finite() {
        return None;
    }

    let ml = if config.morphological_ml {
        entry.ml_ratio_morphological()
    } else {
        let ml_raw = entry.ml_ratio();
        if ml_raw <= 0.0 || !ml_raw.is_finite() || ml_raw > 1e6 {
            return None;
        }
        ml_raw
    };

    let i_e = entry.i_e_solar_per_kpc2();
    if i_e <= 0.0 || !i_e.is_finite() {
        return None;
    }

    // 1. Baryonic density from Sersic profile
    let cfg = SersicLbmConfig {
        nx: g,
        ny: g,
        nz: g,
        dx_kpc: config.dx_kpc,
        r_e_kpc,
        n: entry.n,
        i_e,
        ellipticity: entry.ellipticity,
        pa_rad: entry.pa_deg.to_radians(),
        ml_ratio: ml,
    };
    let rho_baryon = sersic_to_lbm_density(&cfg);

    // 2. NFW dark matter density
    let m200 = entry.m200_solar();
    let nfw = nfw_params_from_mass(m200, entry.photo_z);
    if nfw.r_s_kpc <= 0.0 {
        return None;
    }

    let cx = g as f64 / 2.0;
    let cy = g as f64 / 2.0;
    let cz = g as f64 / 2.0;

    let eps = config.softening_eps * config.dx_kpc;
    let mut rho_dm = vec![0.0_f64; total];
    for iz in 0..g {
        for iy in 0..g {
            for ix in 0..g {
                let dx = (ix as f64 + 0.5 - cx) * config.dx_kpc;
                let dy = (iy as f64 + 0.5 - cy) * config.dx_kpc;
                let dz = (iz as f64 + 0.5 - cz) * config.dx_kpc;
                // Plummer-softened radius: eliminates 1/r NFW cusp
                let r = (dx * dx + dy * dy + dz * dz + eps * eps).sqrt();
                let idx = iz * g * g + iy * g + ix;
                rho_dm[idx] = nfw_density_from_params(r, &nfw);
            }
        }
    }

    // 3. Combine and normalize to LBM-safe range (max rho = 2.0)
    let max_combined: f64 = rho_baryon
        .iter()
        .zip(rho_dm.iter())
        .map(|(b, d)| b + d)
        .fold(0.0_f64, f64::max);

    if max_combined <= 0.0 || !max_combined.is_finite() {
        return None;
    }

    let scale = 2.0 / max_combined;
    let floor = config.density_floor;
    let mut rho_total = vec![0.0_f64; total];
    for i in 0..total {
        // Floor prevents rho=0 -> NaN and limits density contrast.
        // At 200:1 contrast (floor=0.01), streaming across the density
        // interface creates Ma > 1 in a single step. Floor >= 0.1
        // keeps contrast <= 20:1, maintaining Ma < 0.3 at interfaces.
        rho_total[i] = ((rho_baryon[i] + rho_dm[i]) * scale).max(floor);
    }

    // 4. Compute Otsu threshold for box-counting (replaces median which fails
    //    on bimodal distributions where >50% of cells sit at the density floor)
    let finite: Vec<f64> = rho_total
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let threshold = if finite.is_empty() {
        0.0
    } else {
        crate::sersic::otsu_threshold(&finite)
    };

    // 5. NFW gravitational force field with Plummer softening
    //    F(r) = -G * M_enc(r) / (r^2 + eps^2) * r_hat
    //    Softening prevents the 1/r^2 singularity from creating supersonic
    //    velocities at the halo center where a discrete lattice cell cannot
    //    resolve the continuous NFW cusp.
    let eps_sq = eps * eps;
    let mut raw_g = vec![[0.0_f64; 3]; total];
    let mut max_g = 0.0_f64;
    for iz in 0..g {
        for iy in 0..g {
            for ix in 0..g {
                let dx = (ix as f64 + 0.5 - cx) * config.dx_kpc;
                let dy = (iy as f64 + 0.5 - cy) * config.dx_kpc;
                let dz = (iz as f64 + 0.5 - cz) * config.dx_kpc;
                let r_sq = dx * dx + dy * dy + dz * dz;
                let r = r_sq.sqrt();
                let idx = iz * g * g + iy * g + ix;

                // Plummer-softened enclosed mass evaluated at physical r
                let r_phys = r.max(1e-10);
                let m_enc = nfw_enclosed_mass_from_params(r_phys, &nfw);
                // Softened force magnitude: M_enc / (r^2 + eps^2)
                let g_mag = m_enc / (r_sq + eps_sq);
                max_g = max_g.max(g_mag);

                if r > 1e-10 {
                    raw_g[idx] = [-g_mag * dx / r, -g_mag * dy / r, -g_mag * dz / r];
                }
            }
        }
    }

    let force_target = 1e-4;
    let force_norm = if max_g > 0.0 {
        force_target / max_g
    } else {
        0.0
    };
    let force_field: Vec<[f64; 3]> = raw_g
        .iter()
        .map(|g_vec| {
            [
                g_vec[0] * force_norm,
                g_vec[1] * force_norm,
                g_vec[2] * force_norm,
            ]
        })
        .collect();

    // 6. Optional ZD harmonic halo forcing (topological confinement)
    // Modulates NFW force field with 7-mode box-kite harmonic structure.
    // At alpha_zd=0, force_field is unchanged (modulation = 1.0 exactly).
    let force_field = if config.alpha_zd > 0.0 {
        let halo_config = crate::harmonic_halos::HarmonicHaloConfig::new(
            config.alpha_zd,
            config.halo_n_modes,
            nfw.r_s_kpc,
        );
        let mut ff = force_field;
        for iz in 0..g {
            for iy in 0..g {
                for ix in 0..g {
                    let dx = (ix as f64 + 0.5 - cx) * config.dx_kpc;
                    let dy = (iy as f64 + 0.5 - cy) * config.dx_kpc;
                    let dz = (iz as f64 + 0.5 - cz) * config.dx_kpc;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                    let idx = iz * g * g + iy * g + ix;

                    let m = crate::harmonic_halos::harmonic_halo_force_modulation(r, &halo_config);
                    ff[idx][0] *= m;
                    ff[idx][1] *= m;
                    ff[idx][2] *= m;
                }
            }
        }
        ff
    } else {
        force_field
    };

    let record_template = GalaxyDfRecord {
        object_id: entry.object_id,
        sersic_n: entry.n,
        r_e_kpc,
        photo_z: entry.photo_z,
        log_stellar_mass: entry.log_stellar_mass,
        m200_solar: m200,
        c200: nfw.c200,
        ml_ratio: ml,
        morphological_type: entry.morphological_type().to_string(),
        grid_dim: g,
        lbm_steps: config.lbm_steps,
        tau: config.tau,
        df_initial: 0.0,
        df_final: 0.0,
        elapsed_ms: 0,
    };

    Some(GalaxySetup {
        rho_total,
        force_field,
        threshold,
        record_template,
    })
}

// ---------------------------------------------------------------------------
// Sweep analysis
// ---------------------------------------------------------------------------

/// Summary statistics for a D_f sweep across multiple galaxies.
#[derive(Debug, Clone)]
pub struct DfSweepSummary {
    pub n_galaxies: usize,
    pub n_skipped: usize,
    pub overall_mean: f64,
    pub overall_std: f64,
    pub overall_ci_95: (f64, f64),
    pub disk_summary: Option<MorphBinSummary>,
    pub elliptical_summary: Option<MorphBinSummary>,
}

/// Summary for one morphological bin (disk or elliptical).
#[derive(Debug, Clone)]
pub struct MorphBinSummary {
    pub n: usize,
    pub mean: f64,
    pub std: f64,
    pub ci_95: (f64, f64),
}

/// Compute sweep summary from a set of galaxy D_f records.
///
/// Uses bootstrap percentile CI (10_000 resamples) for robust uncertainty.
pub fn analyze_df_sweep(records: &[GalaxyDfRecord], n_skipped: usize, seed: u64) -> DfSweepSummary {
    let df_vals: Vec<f64> = records.iter().map(|r| r.df_final).collect();

    let (overall_mean, overall_std, overall_ci) = if df_vals.is_empty() {
        (f64::NAN, f64::NAN, (f64::NAN, f64::NAN))
    } else {
        let mean = df_vals.iter().sum::<f64>() / df_vals.len() as f64;
        let var = df_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (df_vals.len() as f64 - 1.0).max(1.0);
        let ci = bootstrap_ci_simple(&df_vals, seed);
        (mean, var.sqrt(), ci)
    };

    let disk_vals: Vec<f64> = records
        .iter()
        .filter(|r| r.morphological_type == "disk")
        .map(|r| r.df_final)
        .collect();
    let disk_summary = morph_bin_summary(&disk_vals, seed.wrapping_add(1));

    let ell_vals: Vec<f64> = records
        .iter()
        .filter(|r| r.morphological_type == "elliptical")
        .map(|r| r.df_final)
        .collect();
    let elliptical_summary = morph_bin_summary(&ell_vals, seed.wrapping_add(2));

    DfSweepSummary {
        n_galaxies: records.len(),
        n_skipped,
        overall_mean,
        overall_std,
        overall_ci_95: overall_ci,
        disk_summary,
        elliptical_summary,
    }
}

fn morph_bin_summary(vals: &[f64], seed: u64) -> Option<MorphBinSummary> {
    if vals.is_empty() {
        return None;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var =
        vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (vals.len() as f64 - 1.0).max(1.0);
    let ci = bootstrap_ci_simple(vals, seed);
    Some(MorphBinSummary {
        n: vals.len(),
        mean,
        std: var.sqrt(),
        ci_95: ci,
    })
}

/// Simple bootstrap percentile CI (no external dep on stats_core for the lib crate).
fn bootstrap_ci_simple(data: &[f64], seed: u64) -> (f64, f64) {
    if data.len() < 2 {
        let v = data.first().copied().unwrap_or(f64::NAN);
        return (v, v);
    }

    let n = data.len();
    let n_bootstrap = 10_000;
    let mut stats = Vec::with_capacity(n_bootstrap);

    // Simple LCG for reproducibility without external crate
    let mut rng_state = seed;
    let lcg_next = |state: &mut u64| -> usize {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as usize) % n
    };

    for _ in 0..n_bootstrap {
        let sum: f64 = (0..n).map(|_| data[lcg_next(&mut rng_state)]).sum();
        stats.push(sum / n as f64);
    }

    stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lo = stats[(0.025 * n_bootstrap as f64) as usize];
    let hi = stats[((0.975 * n_bootstrap as f64) as usize).min(n_bootstrap - 1)];
    (lo, hi)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galaxy_df_record_csv_roundtrip() {
        let rec = GalaxyDfRecord {
            object_id: -861130966084299185,
            sersic_n: 1.23,
            r_e_kpc: 4.56,
            photo_z: 0.29,
            log_stellar_mass: 10.5,
            m200_solar: 1.2e12,
            c200: 8.5,
            ml_ratio: 2.0,
            morphological_type: "disk".to_string(),
            grid_dim: 64,
            lbm_steps: 200,
            tau: 1.5,
            df_initial: 2.85,
            df_final: 2.72,
            elapsed_ms: 84,
        };
        let header = GalaxyDfRecord::csv_header();
        let row = rec.to_string();
        assert_eq!(header.split(',').count(), row.split(',').count());
    }

    #[test]
    fn test_pipeline_config_default() {
        let cfg = GalaxyPipelineConfig::default();
        assert_eq!(cfg.grid_dim, 64);
        assert!(cfg.tau >= 1.5);
        assert!(cfg.morphological_ml);
    }

    #[test]
    fn test_analyze_df_sweep_basic() {
        let records: Vec<GalaxyDfRecord> = (0..20)
            .map(|i| GalaxyDfRecord {
                object_id: i,
                sersic_n: if i < 10 { 1.0 } else { 4.0 },
                r_e_kpc: 5.0,
                photo_z: 0.3,
                log_stellar_mass: 10.5,
                m200_solar: 1e12,
                c200: 8.0,
                ml_ratio: 3.0,
                morphological_type: if i < 10 {
                    "disk".to_string()
                } else {
                    "elliptical".to_string()
                },
                grid_dim: 64,
                lbm_steps: 200,
                tau: 1.5,
                df_initial: 2.85,
                df_final: 2.7 + (i as f64) * 0.005,
                elapsed_ms: 80,
            })
            .collect();

        let summary = analyze_df_sweep(&records, 3, 42);
        assert_eq!(summary.n_galaxies, 20);
        assert_eq!(summary.n_skipped, 3);
        assert!(summary.overall_mean > 2.5 && summary.overall_mean < 3.0);
        assert!(summary.overall_std > 0.0);
        assert!(summary.overall_ci_95.0 < summary.overall_ci_95.1);
        assert!(summary.disk_summary.is_some());
        assert!(summary.elliptical_summary.is_some());
        assert_eq!(summary.disk_summary.as_ref().unwrap().n, 10);
        assert_eq!(summary.elliptical_summary.as_ref().unwrap().n, 10);
    }

    #[test]
    fn test_analyze_df_sweep_empty() {
        let summary = analyze_df_sweep(&[], 5, 42);
        assert_eq!(summary.n_galaxies, 0);
        assert_eq!(summary.n_skipped, 5);
        assert!(summary.overall_mean.is_nan());
    }

    #[test]
    fn test_bootstrap_ci_deterministic() {
        let data = vec![2.7, 2.71, 2.72, 2.73, 2.74];
        let (lo1, hi1) = bootstrap_ci_simple(&data, 42);
        let (lo2, hi2) = bootstrap_ci_simple(&data, 42);
        assert_eq!(lo1, lo2);
        assert_eq!(hi1, hi2);
    }
}
