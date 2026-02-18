//! ABACAD cadence filter for RFI discrimination.
//!
//! The Breakthrough Listen ABACAD strategy observes in alternating ON-OFF
//! pairs: A1 B1 A2 C1 A3 D1, where A is the target and B/C/D are nearby
//! calibrators at different sky positions. A genuine narrowband signal from
//! the target direction appears in ALL ON pointings and NONE of the OFF
//! pointings, because it tracks the target's sky position.
//!
//! This module matches Doppler hits across observations within a cadence
//! group, accounting for frequency drift between observations.
//!
//! References:
//! - Price et al. (2020) AJ 159 86 (BL ABACAD strategy)
//! - Enriquez et al. (2017) ApJ 849 104

use super::doppler::DopplerHit;

/// Hit list from a single observation.
#[derive(Debug, Clone)]
pub struct ObservationHits {
    /// Source/target name.
    pub source_name: String,
    /// "ON" or "OFF".
    pub pointing_type: String,
    /// Position in ABACAD sequence (1-6).
    pub cadence_pos: u32,
    /// Observation identifier.
    pub obs_id: String,
    /// Detected Doppler hits.
    pub hits: Vec<DopplerHit>,
}

/// A candidate event surviving the ABACAD cadence filter.
#[derive(Debug, Clone)]
pub struct CadenceEvent {
    /// Representative frequency (MHz), from the best ON detection.
    pub freq_mhz: f64,
    /// Drift rate (Hz/s), from the best ON detection.
    pub drift_rate_hz_s: f64,
    /// Mean SNR across all ON detections.
    pub snr_on_mean: f64,
    /// Maximum SNR across ON detections.
    pub snr_on_max: f64,
    /// Number of ON observations where the signal was detected.
    pub n_on_detections: usize,
    /// Number of OFF observations where the signal was detected (should be 0).
    pub n_off_detections: usize,
    /// The individual ON hits that matched.
    pub on_hits: Vec<DopplerHit>,
}

/// Apply the ABACAD cadence filter to hit lists from one cadence group.
///
/// A genuine signal must appear in ALL ON observations (typically 3)
/// and NONE of the OFF observations (typically 3) within the cadence.
///
/// # Arguments
/// * `observations` - Hit lists from all 6 observations in one cadence
/// * `freq_tolerance_mhz` - Maximum frequency difference for matching (MHz)
/// * `drift_tolerance_hz_s` - Maximum drift rate difference for matching (Hz/s)
///
/// # Returns
/// Candidate events that pass the filter (expected: empty for non-detection).
pub fn abacad_event_filter(
    observations: &[ObservationHits],
    freq_tolerance_mhz: f64,
    drift_tolerance_hz_s: f64,
) -> Vec<CadenceEvent> {
    let on_obs: Vec<&ObservationHits> = observations
        .iter()
        .filter(|o| o.pointing_type == "ON")
        .collect();
    let off_obs: Vec<&ObservationHits> = observations
        .iter()
        .filter(|o| o.pointing_type == "OFF")
        .collect();

    if on_obs.is_empty() {
        return Vec::new();
    }

    let n_on_required = on_obs.len();
    let mut events: Vec<CadenceEvent> = Vec::new();

    // Use the first ON observation as the reference.
    // For each hit in the reference, check if it appears in ALL other ON
    // observations and in NONE of the OFF observations.
    let reference = on_obs[0];
    let other_on: Vec<&&ObservationHits> = on_obs[1..].iter().collect();

    for ref_hit in &reference.hits {
        // Check all other ON observations
        let mut on_matches: Vec<DopplerHit> = vec![ref_hit.clone()];
        let mut all_on_matched = true;

        for on in &other_on {
            let matched = on.hits.iter().find(|h| {
                (h.freq_mhz - ref_hit.freq_mhz).abs() <= freq_tolerance_mhz
                    && (h.drift_rate_hz_s - ref_hit.drift_rate_hz_s).abs() <= drift_tolerance_hz_s
            });

            if let Some(m) = matched {
                on_matches.push(m.clone());
            } else {
                all_on_matched = false;
                break;
            }
        }

        if !all_on_matched {
            continue;
        }

        // Check that the signal is ABSENT in all OFF observations
        let n_off_detections = off_obs
            .iter()
            .filter(|off| {
                off.hits.iter().any(|h| {
                    (h.freq_mhz - ref_hit.freq_mhz).abs() <= freq_tolerance_mhz
                        && (h.drift_rate_hz_s - ref_hit.drift_rate_hz_s).abs()
                            <= drift_tolerance_hz_s
                })
            })
            .count();

        // Only pass if present in ALL ON and NONE of the OFF
        if n_off_detections > 0 {
            continue;
        }

        let snr_values: Vec<f64> = on_matches.iter().map(|h| h.snr).collect();
        let snr_mean = snr_values.iter().sum::<f64>() / snr_values.len() as f64;
        let snr_max = snr_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        events.push(CadenceEvent {
            freq_mhz: ref_hit.freq_mhz,
            drift_rate_hz_s: ref_hit.drift_rate_hz_s,
            snr_on_mean: snr_mean,
            snr_on_max: snr_max,
            n_on_detections: n_on_required,
            n_off_detections: 0,
            on_hits: on_matches,
        });
    }

    // De-duplicate events (in case hits from different drift rate trials converge)
    dedup_events(&mut events, freq_tolerance_mhz, drift_tolerance_hz_s)
}

/// De-duplicate events by keeping the highest-SNR event within each cluster.
fn dedup_events(
    events: &mut [CadenceEvent],
    freq_tol: f64,
    drift_tol: f64,
) -> Vec<CadenceEvent> {
    if events.is_empty() {
        return Vec::new();
    }

    events.sort_by(|a, b| {
        b.snr_on_max
            .partial_cmp(&a.snr_on_max)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<CadenceEvent> = Vec::new();
    let mut suppressed = vec![false; events.len()];

    for i in 0..events.len() {
        if suppressed[i] {
            continue;
        }
        kept.push(events[i].clone());
        for j in (i + 1)..events.len() {
            if suppressed[j] {
                continue;
            }
            if (events[i].freq_mhz - events[j].freq_mhz).abs() <= freq_tol
                && (events[i].drift_rate_hz_s - events[j].drift_rate_hz_s).abs() <= drift_tol
            {
                suppressed[j] = true;
            }
        }
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hit(freq: f64, drift: f64, snr: f64) -> DopplerHit {
        DopplerHit {
            freq_mhz: freq,
            drift_rate_hz_s: drift,
            snr,
            coarse_channel: 0,
            uncorrected_freq: freq,
            total_power: snr * 100.0,
            n_time_samples: 64,
        }
    }

    fn make_obs(name: &str, pointing: &str, pos: u32, obs_id: &str, hits: Vec<DopplerHit>) -> ObservationHits {
        ObservationHits {
            source_name: name.to_string(),
            pointing_type: pointing.to_string(),
            cadence_pos: pos,
            obs_id: obs_id.to_string(),
            hits,
        }
    }

    #[test]
    fn test_abacad_perfect_signal() {
        // Signal at 1420 MHz, 0.5 Hz/s drift, present in all 3 ON, absent in all 3 OFF
        let signal = make_hit(1420.0, 0.5, 20.0);
        let observations = vec![
            make_obs("TARGET", "ON",  1, "0016", vec![signal.clone()]),
            make_obs("CAL_A",  "OFF", 2, "0017", vec![]),
            make_obs("TARGET", "ON",  3, "0018", vec![signal.clone()]),
            make_obs("CAL_B",  "OFF", 4, "0019", vec![]),
            make_obs("TARGET", "ON",  5, "0020", vec![signal.clone()]),
            make_obs("CAL_C",  "OFF", 6, "0021", vec![]),
        ];

        let events = abacad_event_filter(&observations, 0.01, 1.0);
        assert_eq!(events.len(), 1, "Should detect 1 cadence event");
        assert_eq!(events[0].n_on_detections, 3);
        assert_eq!(events[0].n_off_detections, 0);
        assert!((events[0].freq_mhz - 1420.0).abs() < 0.01);
    }

    #[test]
    fn test_abacad_rfi_rejection() {
        // Signal in ON + OFF = RFI -> rejected
        let signal = make_hit(1420.0, 0.0, 25.0);
        let observations = vec![
            make_obs("TARGET", "ON",  1, "0016", vec![signal.clone()]),
            make_obs("CAL_A",  "OFF", 2, "0017", vec![signal.clone()]), // RFI also here
            make_obs("TARGET", "ON",  3, "0018", vec![signal.clone()]),
            make_obs("CAL_B",  "OFF", 4, "0019", vec![]),
            make_obs("TARGET", "ON",  5, "0020", vec![signal.clone()]),
            make_obs("CAL_C",  "OFF", 6, "0021", vec![]),
        ];

        let events = abacad_event_filter(&observations, 0.01, 1.0);
        assert_eq!(events.len(), 0, "RFI present in OFF should be rejected");
    }

    #[test]
    fn test_abacad_partial_detection() {
        // Signal in only 2/3 ON -> rejected (requires ALL ON)
        let signal = make_hit(1420.0, 0.5, 20.0);
        let observations = vec![
            make_obs("TARGET", "ON",  1, "0016", vec![signal.clone()]),
            make_obs("CAL_A",  "OFF", 2, "0017", vec![]),
            make_obs("TARGET", "ON",  3, "0018", vec![signal.clone()]),
            make_obs("CAL_B",  "OFF", 4, "0019", vec![]),
            make_obs("TARGET", "ON",  5, "0020", vec![]),  // Missing!
            make_obs("CAL_C",  "OFF", 6, "0021", vec![]),
        ];

        let events = abacad_event_filter(&observations, 0.01, 1.0);
        assert_eq!(events.len(), 0, "Partial ON detection should be rejected");
    }

    #[test]
    fn test_abacad_freq_tolerance() {
        // Signal drifts slightly between observations but within tolerance
        let observations = vec![
            make_obs("TARGET", "ON",  1, "0016", vec![make_hit(1420.000, 0.5, 20.0)]),
            make_obs("CAL_A",  "OFF", 2, "0017", vec![]),
            make_obs("TARGET", "ON",  3, "0018", vec![make_hit(1420.003, 0.5, 18.0)]),
            make_obs("CAL_B",  "OFF", 4, "0019", vec![]),
            make_obs("TARGET", "ON",  5, "0020", vec![make_hit(1420.006, 0.5, 19.0)]),
            make_obs("CAL_C",  "OFF", 6, "0021", vec![]),
        ];

        let events = abacad_event_filter(&observations, 0.01, 1.0);
        assert_eq!(events.len(), 1, "Drifted signal within tolerance should pass");

        // Same test with tight tolerance -> should fail
        let events_tight = abacad_event_filter(&observations, 0.001, 1.0);
        assert_eq!(events_tight.len(), 0, "Drifted signal outside tight tolerance should fail");
    }

    #[test]
    fn test_abacad_empty_cadence() {
        let observations = vec![
            make_obs("TARGET", "ON",  1, "0016", vec![]),
            make_obs("CAL_A",  "OFF", 2, "0017", vec![]),
            make_obs("TARGET", "ON",  3, "0018", vec![]),
            make_obs("CAL_B",  "OFF", 4, "0019", vec![]),
            make_obs("TARGET", "ON",  5, "0020", vec![]),
            make_obs("CAL_C",  "OFF", 6, "0021", vec![]),
        ];

        let events = abacad_event_filter(&observations, 0.01, 1.0);
        assert_eq!(events.len(), 0, "No hits -> no events");
    }

    #[test]
    fn test_abacad_no_observations() {
        let events = abacad_event_filter(&[], 0.01, 1.0);
        assert_eq!(events.len(), 0);
    }
}
