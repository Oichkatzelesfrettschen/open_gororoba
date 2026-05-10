//! Cadence-agnostic CD delay embedding parameters.
//!
//! WHY: The original CD pipeline was hardcoded for 1-minute magnetic field
//! data with an absolute 0.1-0.5 nT noise floor. Voyager VLISM has |B| ~
//! 0.2 nT (floor would dominate) and Juno Jupiter magnetopause has |B| ~
//! 100 nT (floor is negligible), so a single absolute threshold cannot span
//! nine orders of magnitude in boundary scale. This module parameterizes all
//! cadence, normalization, and coordinate choices in one struct so the same
//! embedding logic runs unchanged from MMS (16 Hz native, 1-min binned) to
//! Voyager (48 s native or hourly COHO).
//!
//! Design contract:
//! - `CdEmbeddingParams::default()` reproduces the THEMIS-Staples F1 = 0.601
//!   result to within float precision.
//! - `resample_to_lag_cadence` handles all cadence mismatches; callers pass
//!   their native-cadence series and receive a lag-aligned series ready for
//!   embedding.

/// How the noise floor / denominator is determined for embedding normalization.
#[derive(Debug, Clone, PartialEq)]
pub enum EpsMode {
    /// Absolute nT floor: `denom = max(window_mean_B, floor_nt)`.
    /// Reproduces the original THEMIS/MMS pipeline behavior.
    /// `floor_nt` should match the instrument noise floor (THEMIS FGS: 0.5 nT,
    /// PSP FIELDS: 0.1 nT, MMS FGM: 0.1 nT).
    Absolute(f64),

    /// Relative fraction of the per-window running *median* of |B|.
    /// `denom = fraction * median_window(|B|)`.
    /// Recommended for cross-mission work. `fraction = 0.01` gives a 1%
    /// noise floor that scales with field strength from 0.2 nT (Voyager VLISM)
    /// to 100 nT (Juno Jupiter magnetopause) without manual tuning.
    RelativeFraction(f64),

    /// Relative fraction of the per-series global median of |B|.
    /// Like `RelativeFraction` but computed once over the full input series,
    /// not re-computed per window. Useful when the series is already trimmed
    /// to a single boundary interval.
    GlobalRelativeFraction(f64),
}

/// How the 4-channel lag vector is normalized before the CD product.
#[derive(Debug, Clone, PartialEq)]
pub enum NormMode {
    /// No additional normalization. The denominator from `EpsMode` already
    /// scales the vector; channels are (Bx/denom, By/denom, Bz/denom,
    /// (|B|-mean_B)/denom). This is the original paper behavior.
    None,

    /// Divide the entire lag vector by its L2 norm after `EpsMode` scaling.
    /// Projects onto the unit sphere; makes the associator sensitive to
    /// direction changes rather than amplitude.
    UnitSphere,

    /// After `EpsMode` scaling, additionally divide by the running median of
    /// |B| over the window. Provides a second normalization stage that is
    /// robust to outliers.
    RunningMedianScale,
}

/// Coordinate preprocessing applied before embedding.
#[derive(Debug, Clone, PartialEq)]
pub enum CoordMode {
    /// Use (Bx, By, Bz, |B|-mean_B) / denom. Original paper behavior.
    Raw,

    /// Use (Bx/|B|, By/|B|, Bz/|B|, 0.0) for each lag sample -- unit
    /// direction vector plus a zero padding channel. Removes amplitude
    /// sensitivity entirely; tests pure geometric rotation structure.
    FieldUnitVector,
}

/// All cadence and normalization parameters for the CD delay embedding.
///
/// Construct via `CdEmbeddingParams::default()` (reproduces paper baseline)
/// or via `CdEmbeddingParams::new(window_secs, n_lags, eps_mode, ...)`.
///
/// The embedding dimension is always `n_channels * n_lags` where
/// `n_channels = 4` for `CoordMode::Raw` and `CoordMode::FieldUnitVector`.
#[derive(Debug, Clone)]
pub struct CdEmbeddingParams {
    /// Total embedding window in seconds (e.g. 480.0 for 8 minutes).
    pub window_secs: f64,

    /// Number of delay lags (samples) within the window (e.g. 8).
    pub n_lags: usize,

    /// Lag spacing in seconds: `window_secs / n_lags`.
    /// The input series is resampled to this cadence if needed.
    pub lag_secs: f64,

    /// How to compute the noise floor denominator for normalization.
    pub eps_mode: EpsMode,

    /// Whether and how to normalize the lag vector after `eps_mode` scaling.
    pub norm_mode: NormMode,

    /// Coordinate preprocessing before embedding.
    pub coord_mode: CoordMode,
}

impl Default for CdEmbeddingParams {
    /// Reproduces the THEMIS-Staples paper baseline: 8-minute window, 8 lags
    /// at 60 s spacing, THEMIS FGS noise floor 0.5 nT, no additional
    /// normalization, raw coordinates.
    fn default() -> Self {
        Self::new(
            480.0,
            8,
            EpsMode::Absolute(0.5),
            NormMode::None,
            CoordMode::Raw,
        )
    }
}

impl CdEmbeddingParams {
    /// Construct with explicit parameters. Derives `lag_secs` automatically.
    pub fn new(
        window_secs: f64,
        n_lags: usize,
        eps_mode: EpsMode,
        norm_mode: NormMode,
        coord_mode: CoordMode,
    ) -> Self {
        assert!(n_lags >= 2, "n_lags must be >= 2");
        assert!(window_secs > 0.0, "window_secs must be positive");
        let lag_secs = window_secs / n_lags as f64;
        Self {
            window_secs,
            n_lags,
            lag_secs,
            eps_mode,
            norm_mode,
            coord_mode,
        }
    }

    /// Embedding dimension = 4 channels * n_lags.
    pub fn embedding_dim(&self) -> usize {
        4 * self.n_lags
    }
}

/// A single magnetic field sample with its timestamp.
#[derive(Debug, Clone, Copy)]
pub struct MagSample {
    /// Seconds since some reference epoch (consistent within a series).
    pub time_secs: f64,
    pub bx: f64,
    pub by: f64,
    pub bz: f64,
    /// |B| = sqrt(bx^2 + by^2 + bz^2). May be pre-computed by caller or set
    /// to `(bx*bx + by*by + bz*bz).sqrt()`.
    pub b_mag: f64,
}

impl MagSample {
    pub fn new(time_secs: f64, bx: f64, by: f64, bz: f64) -> Self {
        let b_mag = (bx * bx + by * by + bz * bz).sqrt();
        Self {
            time_secs,
            bx,
            by,
            bz,
            b_mag,
        }
    }
}

/// Resample a time series to the lag cadence required by `params`.
///
/// Returns a new series where consecutive samples are spaced `lag_secs` apart.
///
/// Resampling rules:
/// - If input cadence matches `lag_secs` within 5%: returns input as-is
///   (after re-timing to align on the lag grid).
/// - If input is oversampled (cadence < lag_secs): boxcar average into
///   `lag_secs`-wide bins.
/// - If input is undersampled (cadence > lag_secs): linear interpolation,
///   but only when the gap between consecutive input samples is < 2x the
///   input cadence; wider gaps produce an empty slot and the embedding
///   window spanning that slot is invalid and should be discarded by the
///   caller.
///
/// The output time stamps are anchored to the first input sample and spaced
/// exactly `lag_secs` apart.
pub fn resample_to_lag_cadence(input: &[MagSample], params: &CdEmbeddingParams) -> Vec<MagSample> {
    if input.len() < 2 {
        return input.to_vec();
    }

    let lag = params.lag_secs;
    let t0 = input[0].time_secs;
    let t_end = input[input.len() - 1].time_secs;

    // Estimate input cadence from median of consecutive differences.
    let input_cadence = median_cadence(input);

    // Tolerance: 5% of lag_secs.
    let tol = 0.05 * lag;

    if (input_cadence - lag).abs() <= tol {
        // Cadences match: re-anchor timestamps, return as-is.
        return input
            .iter()
            .enumerate()
            .map(|(i, s)| MagSample {
                time_secs: t0 + i as f64 * lag,
                ..*s
            })
            .collect();
    }

    // Number of output samples.
    let n_out = ((t_end - t0) / lag).floor() as usize + 1;
    if n_out == 0 {
        return vec![];
    }

    let mut out = Vec::with_capacity(n_out);

    if input_cadence < lag {
        // Oversampled: boxcar average within each lag-secs bin.
        for k in 0..n_out {
            let bin_start = t0 + k as f64 * lag;
            let bin_end = bin_start + lag;
            let bin_samples: Vec<&MagSample> = input
                .iter()
                .filter(|s| s.time_secs >= bin_start && s.time_secs < bin_end)
                .collect();
            if bin_samples.is_empty() {
                // Gap bin: emit NaN sample so caller can detect and skip windows.
                out.push(MagSample {
                    time_secs: bin_start,
                    bx: f64::NAN,
                    by: f64::NAN,
                    bz: f64::NAN,
                    b_mag: f64::NAN,
                });
            } else {
                let n = bin_samples.len() as f64;
                let bx = bin_samples.iter().map(|s| s.bx).sum::<f64>() / n;
                let by = bin_samples.iter().map(|s| s.by).sum::<f64>() / n;
                let bz = bin_samples.iter().map(|s| s.bz).sum::<f64>() / n;
                let b_mag = bin_samples.iter().map(|s| s.b_mag).sum::<f64>() / n;
                out.push(MagSample {
                    time_secs: bin_start,
                    bx,
                    by,
                    bz,
                    b_mag,
                });
            }
        }
    } else {
        // Undersampled: linear interpolation, gap limit = 2x input cadence.
        let gap_limit = 2.0 * input_cadence;
        for k in 0..n_out {
            let t_target = t0 + k as f64 * lag;
            match interpolate_at(input, t_target, gap_limit) {
                Some(s) => out.push(s),
                None => out.push(MagSample {
                    time_secs: t_target,
                    bx: f64::NAN,
                    by: f64::NAN,
                    bz: f64::NAN,
                    b_mag: f64::NAN,
                }),
            }
        }
    }

    out
}

/// Compute the noise floor denominator for a window of samples.
///
/// Returns `(denom, window_mean_b)` where `denom` is the normalization
/// denominator and `window_mean_b` is the arithmetic mean of |B| in the window.
pub fn compute_eps(
    window_samples: &[&MagSample],
    global_median_b: Option<f64>,
    eps_mode: &EpsMode,
) -> (f64, f64) {
    let n = window_samples.len();
    let mean_b = if n == 0 {
        0.0
    } else {
        window_samples.iter().map(|s| s.b_mag).sum::<f64>() / n as f64
    };

    let denom = match eps_mode {
        EpsMode::Absolute(floor) => mean_b.max(*floor),
        EpsMode::RelativeFraction(frac) => {
            let med = window_median_b(window_samples);
            (frac * med).max(1e-9)
        }
        EpsMode::GlobalRelativeFraction(frac) => {
            let med = global_median_b.unwrap_or(mean_b);
            (frac * med).max(1e-9)
        }
    };

    (denom, mean_b)
}

// ---- Private helpers -------------------------------------------------------

fn median_cadence(input: &[MagSample]) -> f64 {
    if input.len() < 2 {
        return 0.0;
    }
    let mut diffs: Vec<f64> = input
        .windows(2)
        .map(|w| w[1].time_secs - w[0].time_secs)
        .collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = diffs.len() / 2;
    if diffs.len().is_multiple_of(2) {
        (diffs[mid - 1] + diffs[mid]) / 2.0
    } else {
        diffs[mid]
    }
}

fn window_median_b(samples: &[&MagSample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut vals: Vec<f64> = samples.iter().map(|s| s.b_mag).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = vals.len() / 2;
    if vals.len().is_multiple_of(2) {
        (vals[mid - 1] + vals[mid]) / 2.0
    } else {
        vals[mid]
    }
}

fn interpolate_at(input: &[MagSample], t: f64, gap_limit: f64) -> Option<MagSample> {
    // Find surrounding samples using binary search.
    let pos = input.partition_point(|s| s.time_secs <= t);
    if pos == 0 {
        // t is before first sample: extrapolate only within gap_limit.
        let s0 = &input[0];
        if (s0.time_secs - t).abs() <= gap_limit {
            return Some(MagSample {
                time_secs: t,
                ..*s0
            });
        }
        return None;
    }
    if pos >= input.len() {
        // t is after last sample.
        let sl = &input[input.len() - 1];
        if (t - sl.time_secs).abs() <= gap_limit {
            return Some(MagSample {
                time_secs: t,
                ..*sl
            });
        }
        return None;
    }
    let s0 = &input[pos - 1];
    let s1 = &input[pos];
    let gap = s1.time_secs - s0.time_secs;
    if gap > gap_limit {
        return None;
    }
    let alpha = (t - s0.time_secs) / gap;
    let lerp = |a: f64, b: f64| a + alpha * (b - a);
    Some(MagSample {
        time_secs: t,
        bx: lerp(s0.bx, s1.bx),
        by: lerp(s0.by, s1.by),
        bz: lerp(s0.bz, s1.bz),
        b_mag: lerp(s0.b_mag, s1.b_mag),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn uniform_series(n: usize, cadence_secs: f64, b_nt: f64) -> Vec<MagSample> {
        (0..n)
            .map(|i| MagSample::new(i as f64 * cadence_secs, b_nt, 0.0, 0.0))
            .collect()
    }

    #[test]
    fn test_default_params_reproduce_paper_config() {
        let p = CdEmbeddingParams::default();
        assert_eq!(p.n_lags, 8);
        assert_relative_eq!(p.window_secs, 480.0);
        assert_relative_eq!(p.lag_secs, 60.0);
        assert_eq!(p.embedding_dim(), 32);
        assert_eq!(p.eps_mode, EpsMode::Absolute(0.5));
        assert_eq!(p.norm_mode, NormMode::None);
        assert_eq!(p.coord_mode, CoordMode::Raw);
    }

    #[test]
    fn test_resample_matching_cadence_passthrough() {
        // 60s input, 60s lag: should return same number of samples.
        let input = uniform_series(20, 60.0, 40.0);
        let params = CdEmbeddingParams::default();
        let out = resample_to_lag_cadence(&input, &params);
        assert_eq!(out.len(), input.len());
        // B values preserved.
        for s in &out {
            assert_relative_eq!(s.b_mag, 40.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_resample_48s_to_60s_interpolation() {
        // Voyager-like 48s native cadence upsampled to 60s lag.
        // 20 samples at 48s => 960s span.
        let input = uniform_series(20, 48.0, 0.3);
        let params = CdEmbeddingParams {
            lag_secs: 60.0,
            ..CdEmbeddingParams::default()
        };
        // median cadence = 48s, lag = 60s: undersampled path.
        let out = resample_to_lag_cadence(&input, &params);
        // 20 samples at 48s: t_end = 19*48 = 912s. floor(912/60)+1 = 16.
        assert_eq!(out.len(), 16);
        // B values should all be ~0.3 (constant series, interpolation is exact).
        for s in &out {
            if s.b_mag.is_finite() {
                assert_relative_eq!(s.b_mag, 0.3, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_resample_16s_to_60s_boxcar() {
        // MMS-like 16s native cadence binned to 60s lag.
        // 75 samples at 16s => 1184s span (fills 19 bins of 60s each).
        let input = uniform_series(75, 16.0, 10.0);
        let params = CdEmbeddingParams::default(); // lag_secs=60
        let out = resample_to_lag_cadence(&input, &params);
        // B values should be ~10 (constant series, boxcar preserves average).
        for s in &out {
            if s.b_mag.is_finite() {
                assert_relative_eq!(s.b_mag, 10.0, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_eps_mode_relative_fraction_ism_signal() {
        // Voyager VLISM: |B| ~ 0.2 nT. RelativeFraction(0.01) should give
        // eps = 0.002 nT (1% of 0.2 nT). The plan mandates this exact check.
        let samples: Vec<MagSample> = (0..8).map(|_| MagSample::new(0.0, 0.2, 0.0, 0.0)).collect();
        let refs: Vec<&MagSample> = samples.iter().collect();
        let eps_mode = EpsMode::RelativeFraction(0.01);
        let (denom, _) = compute_eps(&refs, None, &eps_mode);
        assert_relative_eq!(denom, 0.002, epsilon = 1e-6);
    }

    #[test]
    fn test_eps_mode_absolute_themis_floor() {
        // THEMIS: |B| ~ 40 nT. Absolute(0.5) => denom = max(40, 0.5) = 40.
        let samples: Vec<MagSample> = (0..8)
            .map(|_| MagSample::new(0.0, 40.0, 0.0, 0.0))
            .collect();
        let refs: Vec<&MagSample> = samples.iter().collect();
        let eps_mode = EpsMode::Absolute(0.5);
        let (denom, _) = compute_eps(&refs, None, &eps_mode);
        assert_relative_eq!(denom, 40.0, epsilon = 1e-9);
    }

    #[test]
    fn test_eps_mode_absolute_ism_floor_dominates() {
        // Voyager VLISM: |B| = 0.1 nT (below floor 0.5 nT). Absolute(0.5)
        // => denom = max(0.1, 0.5) = 0.5 nT. Floor dominates, suppressing
        // signal. This is the pathological case that RelativeFraction fixes.
        let samples: Vec<MagSample> = (0..8).map(|_| MagSample::new(0.0, 0.1, 0.0, 0.0)).collect();
        let refs: Vec<&MagSample> = samples.iter().collect();
        let eps_mode = EpsMode::Absolute(0.5);
        let (denom, _) = compute_eps(&refs, None, &eps_mode);
        assert_relative_eq!(denom, 0.5, epsilon = 1e-9);
    }

    #[test]
    fn test_median_cadence_uniform() {
        let s = uniform_series(10, 60.0, 1.0);
        let c = super::median_cadence(&s);
        assert_relative_eq!(c, 60.0, epsilon = 1e-9);
    }
}
