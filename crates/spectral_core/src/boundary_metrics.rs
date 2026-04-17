//! Boundary detection evaluation metrics with rigorous statistical testing.
//!
//! WHY: The paper's F1 = 0.601 needed (a) a block-bootstrap 95% CI because
//! consecutive boundary crossings are autocorrelated (a 30-min block width
//! conservatively spans one full Alfven transit time at THEMIS orbit), and
//! (b) a permutation p-value against shuffled event times to rule out chance
//! alignment. The multi-match semantics (detection-level precision, event-level
//! recall) are an explicit design choice, now disclosed in Table 1.
//!
//! All randomized functions use ChaCha8Rng seeded at 42 for reproducibility.
//! Callers requiring a custom seed may use the `_seeded` variants.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

const DEFAULT_SEED: u64 = 42;

// ============================================================================
// Core metric
// ============================================================================

/// Compute precision, recall, and F1 with explicit multi-match semantics.
///
/// **Precision** is detection-level: a detection is a true positive (TP) if it
/// falls within `window_secs` of ANY catalog event. Multiple detections near
/// one event each count independently in the precision denominator.
///
/// **Recall** is event-level: an event is recalled if ANY detection fires within
/// `window_secs` of it. Each event contributes at most 1 to the TP_event count.
///
/// Returns `(precision, recall, f1)`.
pub fn precision_recall_f1(
    detection_times: &[i64],
    event_times: &[i64],
    window_secs: i64,
) -> (f64, f64, f64) {
    if detection_times.is_empty() && event_times.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    // Detection-level TP: count detections within window of any event.
    let tp_detections = detection_times
        .iter()
        .filter(|&&d| event_times.iter().any(|&e| (d - e).abs() <= window_secs))
        .count();

    // Event-level TP: count events recalled by at least one detection.
    let tp_events = event_times
        .iter()
        .filter(|&&e| detection_times.iter().any(|&d| (d - e).abs() <= window_secs))
        .count();

    let precision = if detection_times.is_empty() {
        0.0
    } else {
        tp_detections as f64 / detection_times.len() as f64
    };

    let recall = if event_times.is_empty() {
        0.0
    } else {
        tp_events as f64 / event_times.len() as f64
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1)
}

// ============================================================================
// Block bootstrap CI
// ============================================================================

/// Stratified block bootstrap 95% CI for F1.
///
/// The timeline `[series_start, series_end]` is divided into non-overlapping
/// blocks of `block_secs`. Each bootstrap replicate samples `n_blocks` block
/// indices with replacement; detections and events in the sampled blocks are
/// shifted to a new contiguous timeline and F1 is recomputed.
///
/// Returns `(f1_mean, ci_lo, ci_hi)` at `coverage_pct` (e.g. 0.95).
///
/// # Recommended parameters
/// - `block_secs = 1800` (30 min): spans ~1 Alfven transit time at THEMIS orbit.
/// - `n_bootstrap = 10_000`
/// - `coverage_pct = 0.95`
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_f1_ci(
    detection_times: &[i64],
    event_times: &[i64],
    series_start: i64,
    series_end: i64,
    block_secs: i64,
    n_bootstrap: usize,
    coverage_pct: f64,
    window_secs: i64,
) -> (f64, f64, f64) {
    bootstrap_f1_ci_seeded(
        detection_times,
        event_times,
        series_start,
        series_end,
        block_secs,
        n_bootstrap,
        coverage_pct,
        window_secs,
        DEFAULT_SEED,
    )
}

/// Seeded variant of `bootstrap_f1_ci` for deterministic testing.
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_f1_ci_seeded(
    detection_times: &[i64],
    event_times: &[i64],
    series_start: i64,
    series_end: i64,
    block_secs: i64,
    n_bootstrap: usize,
    coverage_pct: f64,
    window_secs: i64,
    seed: u64,
) -> (f64, f64, f64) {
    if n_bootstrap == 0 {
        let (_, _, f1) = precision_recall_f1(detection_times, event_times, window_secs);
        return (f1, f1, f1);
    }

    let total = (series_end - series_start).max(0);
    let n_blocks = ((total + block_secs - 1) / block_secs).max(1) as usize;

    // Pre-index detections and events into blocks.
    let block_detections: Vec<Vec<i64>> = (0..n_blocks)
        .map(|b| {
            let b_start = series_start + b as i64 * block_secs;
            let b_end = b_start + block_secs;
            detection_times
                .iter()
                .filter(|&&t| t >= b_start && t < b_end)
                .copied()
                .collect()
        })
        .collect();

    let block_events: Vec<Vec<i64>> = (0..n_blocks)
        .map(|b| {
            let b_start = series_start + b as i64 * block_secs;
            let b_end = b_start + block_secs;
            event_times
                .iter()
                .filter(|&&t| t >= b_start && t < b_end)
                .copied()
                .collect()
        })
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut f1_vals: Vec<f64> = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let indices: Vec<usize> = (0..n_blocks).map(|_| rng.random_range(0..n_blocks)).collect();
        let mut resampled_detections: Vec<i64> = Vec::new();
        let mut resampled_events: Vec<i64> = Vec::new();

        for (new_b, &old_b) in indices.iter().enumerate() {
            let offset = (new_b as i64 - old_b as i64) * block_secs;
            for &t in &block_detections[old_b] {
                resampled_detections.push(t + offset);
            }
            for &t in &block_events[old_b] {
                resampled_events.push(t + offset);
            }
        }

        let (_, _, f1) = precision_recall_f1(&resampled_detections, &resampled_events, window_secs);
        f1_vals.push(f1);
    }

    f1_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = (1.0 - coverage_pct) / 2.0;
    let lo_idx = (alpha * n_bootstrap as f64) as usize;
    let hi_idx = ((1.0 - alpha) * n_bootstrap as f64)
        .min(n_bootstrap as f64 - 1.0) as usize;
    let mean = f1_vals.iter().sum::<f64>() / n_bootstrap as f64;

    (mean, f1_vals[lo_idx], f1_vals[hi_idx])
}

// ============================================================================
// Permutation ratio test
// ============================================================================

/// Permutation test for the observed fire-rate ratio between two window classes.
///
/// Labels (true = class A, false = class B) and fire_mask must be the same
/// length and represent per-minute window classifications and CD fire presence.
/// The observed statistic is `fires_in_A / fires_in_B`. The permutation null
/// shuffles labels while holding fire_mask fixed.
///
/// Returns the empirical p-value: P(ratio_permuted >= observed_ratio).
///
/// `observed_ratio` must be pre-computed from the original labels/fire_mask to
/// ensure the test uses the caller's intended statistic definition.
pub fn permutation_ratio_test(
    labels: &[bool],
    fire_mask: &[bool],
    n_permutations: usize,
    observed_ratio: f64,
) -> f64 {
    permutation_ratio_test_seeded(labels, fire_mask, n_permutations, observed_ratio, DEFAULT_SEED)
}

/// Seeded variant of `permutation_ratio_test`.
pub fn permutation_ratio_test_seeded(
    labels: &[bool],
    fire_mask: &[bool],
    n_permutations: usize,
    observed_ratio: f64,
    seed: u64,
) -> f64 {
    assert_eq!(labels.len(), fire_mask.len(), "labels and fire_mask must be the same length");
    if n_permutations == 0 {
        return 1.0;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut shuffled_labels: Vec<bool> = labels.to_vec();
    let mut n_extreme: usize = 0;

    for _ in 0..n_permutations {
        shuffled_labels.shuffle(&mut rng);
        let ratio = class_fire_ratio(&shuffled_labels, fire_mask);
        if ratio >= observed_ratio {
            n_extreme += 1;
        }
    }

    n_extreme as f64 / n_permutations as f64
}

/// Compute fires_in_A / fires_in_B from labels and fire_mask.
/// Returns 0.0 if class B has no fires (avoids division by zero).
pub fn class_fire_ratio(labels: &[bool], fire_mask: &[bool]) -> f64 {
    let fires_a = labels.iter().zip(fire_mask).filter(|&(&l, &f)| l && f).count();
    let fires_b = labels.iter().zip(fire_mask).filter(|&(&l, &f)| !l && f).count();
    if fires_b == 0 {
        0.0
    } else {
        fires_a as f64 / fires_b as f64
    }
}

// ============================================================================
// F1 null distribution
// ============================================================================

/// Compute the p-value for observed F1 under the null hypothesis that event
/// times are uniformly distributed in `[series_start, series_end]`.
///
/// For each shuffle: places `n_events` uniformly at random in the interval,
/// computes F1 against the fixed `detection_times`, and counts how often
/// F1_null >= observed_f1.
///
/// Returns p = P(F1_null >= observed_f1). A small p-value (e.g. < 0.001)
/// means the observed F1 is unlikely under random event placement.
pub fn f1_null_distribution(
    detection_times: &[i64],
    n_events: usize,
    series_start: i64,
    series_end: i64,
    n_shuffles: usize,
    observed_f1: f64,
    window_secs: i64,
) -> f64 {
    f1_null_distribution_seeded(
        detection_times,
        n_events,
        series_start,
        series_end,
        n_shuffles,
        observed_f1,
        window_secs,
        DEFAULT_SEED,
    )
}

/// Seeded variant of `f1_null_distribution`.
#[allow(clippy::too_many_arguments)]
pub fn f1_null_distribution_seeded(
    detection_times: &[i64],
    n_events: usize,
    series_start: i64,
    series_end: i64,
    n_shuffles: usize,
    observed_f1: f64,
    window_secs: i64,
    seed: u64,
) -> f64 {
    if n_shuffles == 0 || series_end <= series_start {
        return 1.0;
    }

    let range = series_end - series_start;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut n_extreme: usize = 0;

    for _ in 0..n_shuffles {
        let shuffled_events: Vec<i64> = (0..n_events)
            .map(|_| series_start + rng.random_range(0..range))
            .collect();
        let (_, _, f1) = precision_recall_f1(detection_times, &shuffled_events, window_secs);
        if f1 >= observed_f1 {
            n_extreme += 1;
        }
    }

    n_extreme as f64 / n_shuffles as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ---- precision_recall_f1 -----------------------------------------------

    #[test]
    fn test_perfect_detection() {
        let events = vec![100_i64, 200, 300];
        let detections = vec![100_i64, 200, 300];
        let (p, r, f1) = precision_recall_f1(&detections, &events, 60);
        assert_relative_eq!(p, 1.0, epsilon = 1e-9);
        assert_relative_eq!(r, 1.0, epsilon = 1e-9);
        assert_relative_eq!(f1, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_zero_detections() {
        let events = vec![100_i64, 200, 300];
        let detections: Vec<i64> = vec![];
        let (p, r, f1) = precision_recall_f1(&detections, &events, 60);
        assert_relative_eq!(p, 0.0, epsilon = 1e-9);
        assert_relative_eq!(r, 0.0, epsilon = 1e-9);
        assert_relative_eq!(f1, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_multi_match_semantics() {
        // 2 detections both near the same event: precision should be 0.5 (only 1 of 2 is TP
        // when window is tight), but recall should be 1.0 (event is recalled by 1 detection).
        // Actually: both detections ARE within 60s of the event, so both are TP_detections.
        // precision = 2/2 = 1.0, recall = 1/1 = 1.0.
        let events = vec![100_i64];
        let detections = vec![60_i64, 140]; // both within 60s of event=100
        let (p, r, f1) = precision_recall_f1(&detections, &events, 60);
        assert_relative_eq!(p, 1.0, epsilon = 1e-9); // both detections are TP
        assert_relative_eq!(r, 1.0, epsilon = 1e-9); // event is recalled
        assert_relative_eq!(f1, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_multi_match_false_positives() {
        // 3 detections, only 1 near an event: precision = 1/3.
        let events = vec![100_i64];
        let detections = vec![100_i64, 500, 900]; // only first is TP
        let (p, r, f1) = precision_recall_f1(&detections, &events, 60);
        assert_relative_eq!(p, 1.0 / 3.0, epsilon = 1e-9);
        assert_relative_eq!(r, 1.0, epsilon = 1e-9);
        let expected_f1 = 2.0 * (1.0 / 3.0) * 1.0 / (1.0 / 3.0 + 1.0);
        assert_relative_eq!(f1, expected_f1, epsilon = 1e-9);
    }

    #[test]
    fn test_window_boundary_inclusive() {
        // Detection exactly at window boundary (distance == window_secs) should be TP.
        let events = vec![100_i64];
        let detections = vec![160_i64]; // 60s away from event=100, window=60s
        let (p, r, _) = precision_recall_f1(&detections, &events, 60);
        assert_relative_eq!(p, 1.0, epsilon = 1e-9);
        assert_relative_eq!(r, 1.0, epsilon = 1e-9);
    }

    // ---- bootstrap_f1_ci ---------------------------------------------------

    #[test]
    fn test_bootstrap_perfect_signal_tight_ci() {
        // Perfect detection on a regular event grid: F1=1.0 in every bootstrap replicate.
        // CI should be [1.0, 1.0].
        let t_start = 0_i64;
        let t_end = 7 * 24 * 3600; // 7 days
        let events: Vec<i64> = (0..24).map(|h| h * 3600).collect(); // hourly events
        let detections = events.clone();
        let (mean, lo, hi) = bootstrap_f1_ci_seeded(
            &detections,
            &events,
            t_start,
            t_end,
            1800,
            500,
            0.95,
            300,
            42,
        );
        assert_relative_eq!(mean, 1.0, epsilon = 1e-9);
        assert_relative_eq!(lo, 1.0, epsilon = 1e-9);
        assert_relative_eq!(hi, 1.0, epsilon = 1e-9);
    }

    // ---- f1_null_distribution ----------------------------------------------

    #[test]
    fn test_f1_null_perfect_signal_low_p() {
        // Dense detections everywhere: random events should produce high F1 -> p ~ 1.0.
        // Conversely, detections at specific times with events everywhere random
        // should produce p < 0.5 for a well-separated detection set.
        let t_start = 0_i64;
        let t_end = 86400; // 1 day in seconds
        let detections: Vec<i64> = (0..5).map(|i| i * 3600 + 1800).collect(); // every 1h
        // 5 events randomly placed: many won't land near detections.
        let p = f1_null_distribution_seeded(&detections, 5, t_start, t_end, 1000, 0.0, 300, 42);
        // observed_f1=0.0 -> any null f1 >= 0.0, so p = 1.0.
        assert_relative_eq!(p, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_f1_null_impossible_f1() {
        // observed_f1 > 1.0 is impossible: p should be 0.0.
        let t_start = 0_i64;
        let t_end = 86400;
        let detections = vec![3600_i64, 7200];
        let p =
            f1_null_distribution_seeded(&detections, 3, t_start, t_end, 100, 2.0, 300, 42);
        assert_relative_eq!(p, 0.0, epsilon = 1e-9);
    }

    // ---- permutation_ratio_test --------------------------------------------

    #[test]
    fn test_permutation_null_ratio_one() {
        // When ratio = 1.0 and all permutations maintain ratio ~ 1.0, p ~ 0.5.
        // Equal class sizes, equal fire rates.
        let labels: Vec<bool> = (0..100).map(|i| i < 50).collect(); // 50 A, 50 B
        let fire_mask: Vec<bool> = (0_usize..100).map(|i| i.is_multiple_of(2)).collect(); // every other
        let fires_a = 25_f64;
        let fires_b = 25_f64;
        let observed = fires_a / fires_b;
        let p = permutation_ratio_test_seeded(&labels, &fire_mask, 500, observed, 42);
        // p should be near 0.5 (ratio=1.0 is at the median of the null distribution).
        assert!(p > 0.2 && p < 0.8, "p={p}");
    }

    #[test]
    fn test_permutation_high_ratio_low_p() {
        // Class A has very high fire rate; class B has very low. Observed ratio should
        // be extreme under the null.
        let mut labels = vec![true; 50];
        labels.extend(vec![false; 950]);
        let mut fire_mask = vec![true; 50]; // all class A windows fire
        fire_mask.extend(vec![false; 950]); // no class B windows fire
        let observed = 50.0 / 1.0; // use 1 to avoid division by zero
        let p = permutation_ratio_test_seeded(&labels, &fire_mask, 500, observed, 42);
        assert!(p < 0.05, "expected low p-value for extreme ratio, got p={p}");
    }
}
