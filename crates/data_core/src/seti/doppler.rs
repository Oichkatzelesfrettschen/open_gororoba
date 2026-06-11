//! De-Doppler shift-and-add search for narrowband signals.
//!
//! Implements the core turboSETI algorithm in pure Rust: for each trial drift
//! rate, shift frequency channels by the expected Doppler offset at each time
//! step, integrate over time, then detect peaks above a noise threshold.
//!
//! Algorithm detail:
//! 1. For drift rate `r` (Hz/s), time step `t`, fine channel `c`:
//!    - shifted_c = c + round(r * t * tsamp / |foff_hz|)
//!    - Accumulate `data[t][shifted_c]` into integrated spectrum
//! 2. Normalize: spectrum = (integrated - median) / (1.4826 * MAD)
//! 3. Detect peaks above SNR threshold
//! 4. De-duplicate: keep strongest hit within a frequency window
//!
//! References:
//! - Enriquez et al. (2017) ApJ 849 104 (turboSETI algorithm)
//! - Perez et al. (2022) RNAAS 6 197 (BL 6EQUJ5 application)

/// Parameters controlling the Doppler drift search.
#[derive(Debug, Clone)]
pub struct DopplerSearchParams {
    /// Maximum drift rate magnitude (Hz/s). Standard: 4.0.
    pub max_drift: f64,
    /// Minimum drift rate magnitude (Hz/s). Standard: 0.0.
    pub min_drift: f64,
    /// SNR threshold for hit detection. Standard: 10.0.
    pub snr_threshold: f64,
}

impl Default for DopplerSearchParams {
    fn default() -> Self {
        Self {
            max_drift: 4.0,
            min_drift: 0.0,
            snr_threshold: 10.0,
        }
    }
}

/// A single narrowband signal candidate detected by the Doppler search.
#[derive(Debug, Clone)]
pub struct DopplerHit {
    /// Detected frequency (MHz) after de-drift correction.
    pub freq_mhz: f64,
    /// Doppler drift rate (Hz/s).
    pub drift_rate_hz_s: f64,
    /// Signal-to-noise ratio (MAD-normalized).
    pub snr: f64,
    /// Coarse channel index where this hit was found.
    pub coarse_channel: u32,
    /// Uncorrected frequency (MHz) at the first time sample.
    pub uncorrected_freq: f64,
    /// Integrated signal power (raw sum).
    pub total_power: f64,
    /// Number of time samples that contributed to the integration.
    pub n_time_samples: usize,
}

/// Results from searching one coarse channel.
#[derive(Debug, Clone)]
pub struct CoarseChannelResult {
    /// Coarse channel index.
    pub coarse_idx: u32,
    /// Number of hits detected.
    pub n_hits: usize,
    /// Detected hits (sorted by SNR descending).
    pub hits: Vec<DopplerHit>,
    /// Noise floor median (raw power units).
    pub noise_median: f64,
    /// Noise floor MAD (median absolute deviation).
    pub noise_mad: f64,
}

/// Compute median of a f64 slice (sorts a copy).
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute MAD (median absolute deviation) of a f64 slice.
fn mad(values: &[f64]) -> f64 {
    let med = median(values);
    let deviations: Vec<f64> = values.iter().map(|&x| (x - med).abs()).collect();
    median(&deviations)
}

/// Search a single coarse channel for Doppler-drifting signals.
///
/// # Arguments
/// * `data` - Flat f32 array of shape [n_time * n_fine], row-major (time-major)
/// * `n_time` - Number of time samples
/// * `n_fine` - Number of fine frequency channels (nfpc)
/// * `freqs_mhz` - Frequency array for fine channels (MHz), length n_fine
/// * `tsamp` - Time sample interval (seconds)
/// * `channel_width_hz` - Absolute channel width (Hz)
/// * `coarse_idx` - Coarse channel index (for tagging hits)
/// * `params` - Search parameters
#[allow(clippy::too_many_arguments)]
pub fn search_coarse_channel(
    data: &[f32],
    n_time: usize,
    n_fine: usize,
    freqs_mhz: &[f64],
    tsamp: f64,
    channel_width_hz: f64,
    coarse_idx: u32,
    params: &DopplerSearchParams,
) -> CoarseChannelResult {
    assert_eq!(data.len(), n_time * n_fine, "Data length mismatch");
    assert_eq!(freqs_mhz.len(), n_fine, "Frequency array length mismatch");

    if n_time < 2 || n_fine < 2 {
        return CoarseChannelResult {
            coarse_idx,
            n_hits: 0,
            hits: Vec::new(),
            noise_median: 0.0,
            noise_mad: 0.0,
        };
    }

    // Compute drift rate resolution and trial drift rates.
    // drift_resolution = channel_width_hz / total_observation_time
    let total_time = (n_time - 1) as f64 * tsamp;
    let drift_resolution = channel_width_hz / total_time;

    // Number of drift rate trials (both positive and negative)
    let n_drift_positive = if drift_resolution > 0.0 {
        (params.max_drift / drift_resolution).ceil() as i64
    } else {
        0
    };
    // Collect all hits across drift rates before de-duplication
    let mut all_hits: Vec<DopplerHit> = Vec::new();

    // Search both positive and negative drift rates
    for drift_idx in -n_drift_positive..=n_drift_positive {
        let drift_rate = drift_idx as f64 * drift_resolution;
        let drift_abs = drift_rate.abs();

        // Skip if below minimum drift rate
        if drift_abs < params.min_drift - drift_resolution * 0.5 {
            continue;
        }
        // Skip if above maximum
        if drift_abs > params.max_drift + drift_resolution * 0.5 {
            continue;
        }

        // Shift-and-add: integrate along the drift rate trajectory
        let mut integrated = vec![0.0f64; n_fine];
        let mut valid_counts = vec![0usize; n_fine];

        for t in 0..n_time {
            let time_offset = t as f64 * tsamp;
            // Channel shift for this time step (in fine channel units)
            let chan_shift = (drift_rate * time_offset / channel_width_hz).round() as i64;

            for c in 0..n_fine {
                let shifted_c = c as i64 + chan_shift;
                if shifted_c >= 0 && (shifted_c as usize) < n_fine {
                    let val = data[t * n_fine + shifted_c as usize];
                    if val.is_finite() {
                        integrated[c] += val as f64;
                        valid_counts[c] += 1;
                    }
                }
            }
        }

        // Normalize by count to get mean power per time sample
        for c in 0..n_fine {
            if valid_counts[c] > 0 {
                integrated[c] /= valid_counts[c] as f64;
            }
        }

        // Compute noise floor: median and MAD of the integrated spectrum
        let spec_median = median(&integrated);
        let spec_mad = mad(&integrated);

        // Guard against zero MAD (constant spectrum)
        // 1.4826 converts MAD to equivalent Gaussian sigma
        let sigma = 1.4826 * spec_mad;
        if sigma <= 0.0 {
            continue;
        }

        // Detect peaks above threshold
        for c in 0..n_fine {
            let snr = (integrated[c] - spec_median) / sigma;
            if snr >= params.snr_threshold {
                all_hits.push(DopplerHit {
                    freq_mhz: freqs_mhz[c],
                    drift_rate_hz_s: drift_rate,
                    snr,
                    coarse_channel: coarse_idx,
                    uncorrected_freq: freqs_mhz[c],
                    total_power: integrated[c] * valid_counts[c] as f64,
                    n_time_samples: valid_counts[c],
                });
            }
        }
    }

    // De-duplicate: within a window of +/- 2 channels and +/- 1 drift step,
    // keep only the hit with highest SNR.
    let freq_tolerance = 2.0 * channel_width_hz / 1e6; // MHz
    let drift_tolerance = 1.5 * drift_resolution;
    let hits = deduplicate_hits(&mut all_hits, freq_tolerance, drift_tolerance);

    // Compute overall noise floor from the zero-drift spectrum
    let mut zero_drift_spectrum = vec![0.0f64; n_fine];
    for t in 0..n_time {
        for c in 0..n_fine {
            let val = data[t * n_fine + c];
            if val.is_finite() {
                zero_drift_spectrum[c] += val as f64;
            }
        }
    }
    for v in &mut zero_drift_spectrum {
        *v /= n_time as f64;
    }
    let noise_med = median(&zero_drift_spectrum);
    let noise_m = mad(&zero_drift_spectrum);

    CoarseChannelResult {
        coarse_idx,
        n_hits: hits.len(),
        hits,
        noise_median: noise_med,
        noise_mad: noise_m,
    }
}

/// De-duplicate hits: keep the strongest-SNR hit within each frequency/drift cluster.
fn deduplicate_hits(
    hits: &mut [DopplerHit],
    freq_tolerance_mhz: f64,
    drift_tolerance_hz_s: f64,
) -> Vec<DopplerHit> {
    if hits.is_empty() {
        return Vec::new();
    }

    // Sort by SNR descending so we greedily keep the best
    hits.sort_by(|a, b| {
        b.snr
            .partial_cmp(&a.snr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<DopplerHit> = Vec::new();
    let mut suppressed = vec![false; hits.len()];

    for i in 0..hits.len() {
        if suppressed[i] {
            continue;
        }
        kept.push(hits[i].clone());

        // Suppress all weaker hits within tolerance
        for j in (i + 1)..hits.len() {
            if suppressed[j] {
                continue;
            }
            let freq_diff = (hits[i].freq_mhz - hits[j].freq_mhz).abs();
            let drift_diff = (hits[i].drift_rate_hz_s - hits[j].drift_rate_hz_s).abs();
            if freq_diff <= freq_tolerance_mhz && drift_diff <= drift_tolerance_hz_s {
                suppressed[j] = true;
            }
        }
    }

    kept
}

/// Inject a synthetic narrowband signal into a data array (for testing).
///
/// Injects a signal at `freq_chan` (fine channel index) with drift rate
/// `drift_hz_s`, amplitude `amplitude`, into the `data` array.
#[allow(clippy::too_many_arguments)]
pub fn inject_signal(
    data: &mut [f32],
    n_time: usize,
    n_fine: usize,
    freq_chan: usize,
    drift_hz_s: f64,
    tsamp: f64,
    channel_width_hz: f64,
    amplitude: f32,
) {
    for t in 0..n_time {
        let time_offset = t as f64 * tsamp;
        let chan_shift = (drift_hz_s * time_offset / channel_width_hz).round() as i64;
        let c = freq_chan as i64 + chan_shift;
        if c >= 0 && (c as usize) < n_fine {
            data[t * n_fine + c as usize] += amplitude;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};
    use rand_distr::{Distribution, Normal};

    fn make_test_freqs(n_fine: usize, fch1_mhz: f64, foff_mhz: f64) -> Vec<f64> {
        (0..n_fine)
            .map(|i| fch1_mhz + i as f64 * foff_mhz)
            .collect()
    }

    fn make_noise_data(n_time: usize, n_fine: usize, mean: f32, sigma: f32, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(mean as f64, sigma as f64).unwrap();
        (0..n_time * n_fine)
            .map(|_| dist.sample(&mut rng) as f32)
            .collect()
    }

    #[test]
    fn test_drift_resolution_medium_res() {
        // Medium-res: foff = 2861 Hz, n_time = 279, tsamp = 1.074s
        // total_time = 278 * 1.074 = 298.572s
        // drift_res = 2861 / 298.572 = 9.58 Hz/s
        let n_time = 279;
        let tsamp = 1.073741823999999;
        let channel_width_hz = 2861.0229;
        let total_time = (n_time - 1) as f64 * tsamp;
        let dr = channel_width_hz / total_time;
        assert!((dr - 9.58).abs() < 0.1, "Expected ~9.58 Hz/s, got {}", dr);
    }

    #[test]
    fn test_inject_recover_zero_drift() {
        let n_time = 64;
        let n_fine = 256;
        let tsamp = 1.0;
        let channel_width_hz = 2861.0;
        let fch1 = 1500.0;
        let foff = -channel_width_hz / 1e6;
        let freqs = make_test_freqs(n_fine, fch1, foff);

        // Gaussian noise background
        let mut data = make_noise_data(n_time, n_fine, 100.0, 5.0, 42);

        // Inject a bright signal at channel 128, zero drift, amplitude = 200
        inject_signal(
            &mut data,
            n_time,
            n_fine,
            128,
            0.0,
            tsamp,
            channel_width_hz,
            200.0,
        );

        let params = DopplerSearchParams {
            max_drift: 4.0,
            min_drift: 0.0,
            snr_threshold: 5.0,
        };

        let result = search_coarse_channel(
            &data,
            n_time,
            n_fine,
            &freqs,
            tsamp,
            channel_width_hz,
            0,
            &params,
        );

        assert!(
            result.n_hits > 0,
            "Should detect injected zero-drift signal, got {} hits",
            result.n_hits
        );

        // The strongest hit should be near channel 128
        let best = &result.hits[0];
        let expected_freq = freqs[128];
        assert!(
            (best.freq_mhz - expected_freq).abs() < (channel_width_hz / 1e6) * 3.0,
            "Strongest hit at {:.6} MHz, expected near {:.6} MHz",
            best.freq_mhz,
            expected_freq
        );
        assert!(
            best.drift_rate_hz_s.abs() < channel_width_hz / ((n_time - 1) as f64 * tsamp) * 1.5,
            "Drift rate should be near zero, got {} Hz/s",
            best.drift_rate_hz_s
        );
    }

    #[test]
    fn test_inject_recover_positive_drift() {
        let n_time = 64;
        let n_fine = 256;
        let tsamp = 1.0;
        let channel_width_hz = 100.0; // Finer resolution for drift test
        let fch1 = 1500.0;
        let foff = -channel_width_hz / 1e6;
        let freqs = make_test_freqs(n_fine, fch1, foff);

        let mut data = make_noise_data(n_time, n_fine, 100.0, 5.0, 123);

        // Inject signal at channel 128 with drift rate = 1.0 Hz/s
        let inject_drift = 1.0;
        inject_signal(
            &mut data,
            n_time,
            n_fine,
            128,
            inject_drift,
            tsamp,
            channel_width_hz,
            200.0,
        );

        let params = DopplerSearchParams {
            max_drift: 4.0,
            min_drift: 0.0,
            snr_threshold: 5.0,
        };

        let result = search_coarse_channel(
            &data,
            n_time,
            n_fine,
            &freqs,
            tsamp,
            channel_width_hz,
            0,
            &params,
        );

        assert!(result.n_hits > 0, "Should detect injected drifting signal");

        // Find the hit closest to the injected drift rate
        let best_drift_match = result
            .hits
            .iter()
            .min_by(|a, b| {
                (a.drift_rate_hz_s - inject_drift)
                    .abs()
                    .partial_cmp(&(b.drift_rate_hz_s - inject_drift).abs())
                    .unwrap()
            })
            .unwrap();

        let drift_tolerance = channel_width_hz / ((n_time - 1) as f64 * tsamp) * 2.0;
        assert!(
            (best_drift_match.drift_rate_hz_s - inject_drift).abs() < drift_tolerance,
            "Recovered drift {:.3} Hz/s should be within {:.3} of injected {:.3}",
            best_drift_match.drift_rate_hz_s,
            drift_tolerance,
            inject_drift
        );
    }

    #[test]
    fn test_noise_floor_no_false_positives() {
        let n_time = 64;
        let n_fine = 256;
        let tsamp = 1.0;
        let channel_width_hz = 2861.0;
        let fch1 = 1500.0;
        let foff = -channel_width_hz / 1e6;
        let freqs = make_test_freqs(n_fine, fch1, foff);

        // Pure Gaussian noise, no injected signal
        let data = make_noise_data(n_time, n_fine, 100.0, 5.0, 999);

        // High SNR threshold should yield zero hits
        let params = DopplerSearchParams {
            max_drift: 4.0,
            min_drift: 0.0,
            snr_threshold: 15.0,
        };

        let result = search_coarse_channel(
            &data,
            n_time,
            n_fine,
            &freqs,
            tsamp,
            channel_width_hz,
            0,
            &params,
        );

        // With SNR >= 15 and only 256 channels, false positive rate should be ~0
        assert!(
            result.n_hits <= 2,
            "Expect very few false positives with SNR >= 15, got {}",
            result.n_hits
        );
    }

    #[test]
    fn test_normalization_median_mad() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let med = median(&values);
        assert!((med - 5.5).abs() < 1e-10);

        let m = mad(&values);
        // Deviations from 5.5: [4.5, 3.5, 2.5, 1.5, 0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        // Sorted: [0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5]
        // Median of deviations = (2.5 + 2.5)/2 = 2.5
        assert!((m - 2.5).abs() < 1e-10, "Expected MAD=2.5, got {}", m);
    }

    #[test]
    fn test_deduplicate_hits() {
        let mut hits = vec![
            DopplerHit {
                freq_mhz: 1500.0,
                drift_rate_hz_s: 0.0,
                snr: 20.0,
                coarse_channel: 0,
                uncorrected_freq: 1500.0,
                total_power: 1000.0,
                n_time_samples: 64,
            },
            DopplerHit {
                freq_mhz: 1500.001,
                drift_rate_hz_s: 0.0,
                snr: 15.0,
                coarse_channel: 0,
                uncorrected_freq: 1500.001,
                total_power: 800.0,
                n_time_samples: 64,
            },
            DopplerHit {
                freq_mhz: 1600.0,
                drift_rate_hz_s: 1.0,
                snr: 12.0,
                coarse_channel: 0,
                uncorrected_freq: 1600.0,
                total_power: 600.0,
                n_time_samples: 64,
            },
        ];

        let kept = deduplicate_hits(&mut hits, 0.005, 2.0);
        // First two are within tolerance, third is separate
        assert_eq!(kept.len(), 2, "Should keep 2 clusters");
        assert!(
            (kept[0].snr - 20.0).abs() < 1e-10,
            "Best SNR should be first"
        );
    }

    #[test]
    fn test_empty_data() {
        let params = DopplerSearchParams::default();
        let result = search_coarse_channel(&[], 0, 0, &[], 1.0, 2861.0, 0, &params);
        assert_eq!(result.n_hits, 0);
    }

    #[test]
    fn test_default_params() {
        let p = DopplerSearchParams::default();
        assert!((p.max_drift - 4.0).abs() < 1e-10);
        assert!((p.min_drift - 0.0).abs() < 1e-10);
        assert!((p.snr_threshold - 10.0).abs() < 1e-10);
    }
}
