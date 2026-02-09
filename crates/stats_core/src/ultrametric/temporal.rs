//! Temporal cascade analysis for repeating transient sources.
//!
//! Analyzes the temporal structure of repeating Fast Radio Bursts (FRBs)
//! and other transient sources to detect Self-Organized Criticality (SOC)
//! patterns and ultrametric hierarchy in waiting-time distributions.
//!
//! # Physical Motivation
//!
//! If FRB emission is driven by SOC (e.g., magnetospheric reconnection
//! avalanches), the waiting times and energies should exhibit:
//! 1. Power-law distributions (scale invariance)
//! 2. Long-range temporal correlations (Hurst exponent H > 0.5)
//! 3. Hierarchical/ultrametric temporal structure (nested avalanches)
//!
//! # Algorithm
//!
//! 1. Extract waiting times dt_i = t_{i+1} - t_i from burst timestamps
//! 2. Compute waiting-time statistics (mean, median, coefficient of variation)
//! 3. Estimate Hurst exponent via R/S analysis (using `hurst` crate)
//! 4. Build distance matrix in (log_dt, log_energy) space
//! 5. Run ultrametric fraction test on the distance matrix
//! 6. Run dendrogram analysis on the distance matrix
//!
//! # References
//!
//! - Aschwanden (2011): SOC in astrophysics
//! - Cheng et al. (2020): FRB 121102 waiting time distributions
//! - Wang & Zhang (2019): FRB energy distributions

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::claims_gates::Verdict;

/// Statistics of waiting-time distribution.
#[derive(Debug, Clone)]
pub struct WaitingTimeStats {
    /// Number of bursts.
    pub n_bursts: usize,
    /// Number of waiting times (n_bursts - 1).
    pub n_intervals: usize,
    /// Mean waiting time (same units as input timestamps).
    pub mean_dt: f64,
    /// Median waiting time.
    pub median_dt: f64,
    /// Standard deviation of waiting times.
    pub std_dt: f64,
    /// Coefficient of variation (std/mean). CV >> 1 indicates clustered bursts.
    pub cv: f64,
    /// Minimum waiting time.
    pub min_dt: f64,
    /// Maximum waiting time.
    pub max_dt: f64,
}

/// Complete temporal cascade analysis result.
#[derive(Debug, Clone)]
pub struct CascadeAnalysis {
    /// Source identifier.
    pub source_id: String,
    /// Number of bursts analyzed.
    pub n_bursts: usize,
    /// Waiting time statistics.
    pub waiting_time_stats: WaitingTimeStats,
    /// Hurst exponent from R/S analysis of waiting times.
    /// H > 0.5 indicates long-range temporal correlations (persistence).
    /// H = 0.5 is random walk (no memory).
    /// H < 0.5 indicates anti-persistence.
    pub hurst_exponent: f64,
    /// Ultrametric fraction in (log_dt, log_energy) space.
    pub ultrametric_fraction: f64,
    /// Null ultrametric fraction (shuffled timestamps).
    pub null_fraction_mean: f64,
    /// P-value for ultrametric test.
    pub p_value: f64,
    /// Verdict.
    pub verdict: Verdict,
}

/// Compute waiting-time statistics from sorted burst timestamps.
pub fn waiting_time_statistics(timestamps: &[f64]) -> WaitingTimeStats {
    let n = timestamps.len();
    assert!(n >= 2, "Need at least 2 bursts for waiting times");

    let mut dts: Vec<f64> = timestamps
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&dt| dt > 0.0) // Filter out simultaneous bursts
        .collect();

    let n_intervals = dts.len();
    if n_intervals == 0 {
        return WaitingTimeStats {
            n_bursts: n,
            n_intervals: 0,
            mean_dt: 0.0,
            median_dt: 0.0,
            std_dt: 0.0,
            cv: 0.0,
            min_dt: 0.0,
            max_dt: 0.0,
        };
    }

    let mean_dt = dts.iter().sum::<f64>() / n_intervals as f64;
    let var_dt = dts.iter().map(|dt| (dt - mean_dt).powi(2)).sum::<f64>() / n_intervals as f64;
    let std_dt = var_dt.sqrt();

    dts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_dt = dts[n_intervals / 2];
    let min_dt = dts[0];
    let max_dt = dts[n_intervals - 1];

    let cv = if mean_dt > 0.0 { std_dt / mean_dt } else { 0.0 };

    WaitingTimeStats {
        n_bursts: n,
        n_intervals,
        mean_dt,
        median_dt,
        std_dt,
        cv,
        min_dt,
        max_dt,
    }
}

/// Estimate the Hurst exponent from a time series using R/S analysis.
///
/// Uses the `hurst` crate's corrected rescaled range method.
/// Returns H in [0, 1]. H > 0.5 indicates persistence (long memory).
///
/// Falls back to 0.5 (null hypothesis) if:
/// - Series is too short (< 10 points)
/// - The hurst crate panics internally (e.g., "TooSteep" on degenerate data)
/// - The result is NaN or infinite
pub fn estimate_hurst(series: &[f64]) -> f64 {
    if series.len() < 10 {
        return 0.5; // Not enough data; return null hypothesis
    }

    // hurst::rs_corrected can panic on degenerate inputs (constant series,
    // perfectly regular intervals) with "TooSteep". Catch this.
    let series_owned = series.to_vec();
    let result = std::panic::catch_unwind(|| hurst::rs_corrected(series_owned));

    match result {
        Ok(h) if h.is_finite() => h.clamp(0.0, 1.0),
        _ => 0.5, // Fallback to null if computation fails or panics
    }
}

/// Run the full temporal cascade analysis for a repeating source.
///
/// `source_id`: identifier for the source (e.g., "FRB20220912A").
/// `timestamps`: sorted burst arrival times (MJD or seconds).
/// `energies`: burst energies or fluences (same order as timestamps).
///             If empty or wrong length, only temporal analysis is performed.
/// `n_triples`: number of triples for ultrametric test.
/// `n_permutations`: permutations for null distribution.
/// `seed`: RNG seed.
pub fn analyze_temporal_cascade(
    source_id: &str,
    timestamps: &[f64],
    energies: &[f64],
    n_triples: usize,
    n_permutations: usize,
    seed: u64,
) -> CascadeAnalysis {
    let n = timestamps.len();
    assert!(n >= 3, "Need at least 3 bursts for cascade analysis");

    // 1. Waiting time statistics
    let wt_stats = waiting_time_statistics(timestamps);

    // 2. Hurst exponent on waiting times
    let waiting_times: Vec<f64> = timestamps
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&dt| dt > 0.0)
        .collect();
    let hurst_exp = estimate_hurst(&waiting_times);

    // 3. Build distance matrix in (log_dt, log_energy) space
    // Each burst i is represented by (log(dt_before), log(energy_i))
    // For the first burst, use dt_after instead of dt_before
    let has_energies = energies.len() == n;

    let points: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let log_dt = if i == 0 {
                if n > 1 {
                    (timestamps[1] - timestamps[0]).max(1e-10).ln()
                } else {
                    0.0
                }
            } else {
                (timestamps[i] - timestamps[i - 1]).max(1e-10).ln()
            };

            let log_e = if has_energies && energies[i] > 0.0 {
                energies[i].ln()
            } else {
                0.0
            };

            (log_dt, log_e)
        })
        .collect();

    // Compute pairwise distances in (log_dt, log_energy) space
    let n_pairs = n * (n - 1) / 2;
    let mut dist_matrix = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            let ddt = points[i].0 - points[j].0;
            let de = points[i].1 - points[j].1;
            dist_matrix.push((ddt * ddt + de * de).sqrt());
        }
    }

    // 4. Ultrametric fraction test
    let obs_frac = super::ultrametric_fraction_from_matrix(&dist_matrix, n, n_triples, seed);

    // 5. Null: shuffle timestamps (preserving energy associations)
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 3_000_000);
    let mut null_fracs = Vec::with_capacity(n_permutations);
    let mut shuffled_ts = timestamps.to_vec();

    for _ in 0..n_permutations {
        shuffled_ts.shuffle(&mut rng);
        shuffled_ts.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Recompute points with shuffled timestamps
        let null_points: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let log_dt = if i == 0 {
                    if n > 1 {
                        (shuffled_ts[1] - shuffled_ts[0]).max(1e-10).ln()
                    } else {
                        0.0
                    }
                } else {
                    (shuffled_ts[i] - shuffled_ts[i - 1]).max(1e-10).ln()
                };

                let log_e = if has_energies && energies[i] > 0.0 {
                    energies[i].ln()
                } else {
                    0.0
                };

                (log_dt, log_e)
            })
            .collect();

        let mut null_dists = Vec::with_capacity(n_pairs);
        for i in 0..n {
            for j in (i + 1)..n {
                let ddt = null_points[i].0 - null_points[j].0;
                let de = null_points[i].1 - null_points[j].1;
                null_dists.push((ddt * ddt + de * de).sqrt());
            }
        }

        let null_frac =
            super::ultrametric_fraction_from_matrix(&null_dists, n, n_triples, seed + 4_000_000);
        null_fracs.push(null_frac);
    }

    let null_mean = null_fracs.iter().sum::<f64>() / n_permutations as f64;

    // One-sided p-value: fraction of null >= observed
    let n_ge = null_fracs.iter().filter(|&&f| f >= obs_frac).count();
    let p_value = (n_ge as f64 + 1.0) / (n_permutations as f64 + 1.0);

    let verdict = if p_value < 0.05 {
        Verdict::Pass
    } else {
        Verdict::Fail
    };

    CascadeAnalysis {
        source_id: source_id.to_string(),
        n_bursts: n,
        waiting_time_stats: wt_stats,
        hurst_exponent: hurst_exp,
        ultrametric_fraction: obs_frac,
        null_fraction_mean: null_mean,
        p_value,
        verdict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waiting_time_statistics_basic() {
        let timestamps = vec![0.0, 1.0, 3.0, 6.0, 10.0];
        let stats = waiting_time_statistics(&timestamps);

        assert_eq!(stats.n_bursts, 5);
        assert_eq!(stats.n_intervals, 4);
        // dt = [1, 2, 3, 4], mean = 2.5
        assert!((stats.mean_dt - 2.5).abs() < 1e-10);
        assert!((stats.min_dt - 1.0).abs() < 1e-10);
        assert!((stats.max_dt - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_waiting_time_cv_clustered() {
        // Highly clustered bursts: short intervals interspersed with long gaps
        let timestamps = vec![0.0, 0.1, 0.2, 100.0, 100.1, 100.2];
        let stats = waiting_time_statistics(&timestamps);

        // CV should be >> 1 for clustered data
        assert!(
            stats.cv > 1.0,
            "Clustered bursts should have CV >> 1, got {}",
            stats.cv
        );
    }

    #[test]
    fn test_hurst_exponent_range() {
        // Random walk should give H ~ 0.5
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let series: Vec<f64> = (0..100).map(|_| rng.gen_range(0.0..1.0)).collect();

        let h = estimate_hurst(&series);
        assert!(
            h >= 0.0 && h <= 1.0,
            "Hurst exponent should be in [0,1], got {}",
            h
        );
    }

    #[test]
    fn test_cascade_analysis_smoke() {
        // Synthetic repeating source with regular intervals
        let timestamps: Vec<f64> = (0..20).map(|i| i as f64 * 1.0).collect();
        let energies: Vec<f64> = (0..20).map(|i| (i as f64 + 1.0) * 10.0).collect();

        let result = analyze_temporal_cascade(
            "TEST_SOURCE",
            &timestamps,
            &energies,
            1_000,
            20, // Small for speed
            42,
        );

        assert_eq!(result.source_id, "TEST_SOURCE");
        assert_eq!(result.n_bursts, 20);
        assert!(result.ultrametric_fraction >= 0.0 && result.ultrametric_fraction <= 1.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_cascade_no_energies() {
        // Should work without energy data
        let timestamps: Vec<f64> = (0..10).map(|i| i as f64 * 2.0).collect();

        let result = analyze_temporal_cascade(
            "NO_ENERGY",
            &timestamps,
            &[], // No energies
            500,
            10,
            42,
        );

        assert_eq!(result.n_bursts, 10);
        assert!(result.ultrametric_fraction >= 0.0);
    }
}
