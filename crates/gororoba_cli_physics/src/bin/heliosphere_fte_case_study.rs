//! FTE case-study classifier -- response to JGR reviewer Major Comment 1.
//!
//! WHY: The reviewer correctly states that "process of elimination is not
//! proof." We must demonstrate empirically that the CD associator's
//! "extra" detections (those not matched to FPI density events) are genuine
//! rotational discontinuities (FTEs, KH waves, reconnection jets) rather
//! than compressive noise.
//!
//! WHAT: Reads the MMS evaluation JSON (which now stores cd_fire_hours) and
//! the MMS FGM minute-averaged B-field.  For each CD fire NOT matched to any
//! FPI event, extracts a +/-window_minutes context, then computes two
//! discriminants:
//!   (1) compressiveness: delta_B_rel = (|B|_max - |B|_min) / |B|_mean in window
//!   (2) rotation: angle (deg) between mean B-direction in pre-half and post-half
//! Classification:
//!   - rotational discontinuity: rotation > rot_threshold AND delta_B_rel < comp_threshold
//!   - compressive event:        rotation < rot_threshold AND delta_B_rel > comp_threshold
//!   - combined (FTE with compression): both thresholds exceeded
//!   - weak (low confidence):    both below threshold
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere-fte-case-study -- \
//!     --eval-json data/output/heliosphere/ablations/mms_sitl_labeled_eval.json \
//!     --data-dir data/external

use anyhow::{Context, Result};
use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime};
use clap::Parser;
use data_core::catalogs::mms::{MmsFgmMinuteRecord, parse_mms_fgm_hapi_csv_minutes};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-fte-case-study",
    about = "Classify CD-associator extra detections as rotational vs compressive (JGR reviewer response)"
)]
struct Cli {
    /// Path to the MMS labeled evaluation JSON (must have cd_fire_hours field).
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/mms_sitl_labeled_eval.json"
    )]
    eval_json: PathBuf,

    /// Data cache root (should contain mms/ subdirectory with FGM CSVs).
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,

    /// Half-width (minutes) of context window extracted around each fire.
    #[arg(long, default_value_t = 30)]
    window_minutes: usize,

    /// Rotation threshold (degrees) above which event is classed rotational.
    #[arg(long, default_value_t = 15.0)]
    rot_threshold: f64,

    /// Relative |B| range threshold above which event is classed compressive.
    #[arg(long, default_value_t = 0.30)]
    comp_threshold: f64,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/mms_fte_case_study.json"
    )]
    out_json: PathBuf,
}

// ============================================================================
// Evaluation JSON schema (only fields we need)
// ============================================================================

#[derive(Deserialize)]
struct EvalJson {
    start_date: String,
    n_days: u32,
    cd_fire_hours: Vec<f64>,
    sitl_events: Vec<SitlEventRaw>,
}

#[derive(Deserialize)]
struct SitlEventRaw {
    start: String,
    end: String,
    cd_matched: bool,
}

// ============================================================================
// Output structures
// ============================================================================

#[derive(Debug, Serialize)]
struct ExtraDetectionSummary {
    /// ISO timestamp of the CD fire.
    timestamp: String,
    /// Elapsed hours from eval reference midnight.
    elapsed_hours: f64,
    /// (|B|_max - |B|_min) / |B|_mean in the context window.
    compressiveness: f64,
    /// Angle (deg) between mean B in first half and second half of context window.
    rotation_deg: f64,
    /// Classification label.
    classification: String,
}

#[derive(Debug, Serialize)]
struct CaseStudyResults {
    eval_json: String,
    n_cd_fires: usize,
    n_matched_to_fpi: usize,
    n_extra: usize,
    rot_threshold_deg: f64,
    comp_threshold: f64,
    n_rotational: usize,
    n_compressive: usize,
    n_combined: usize,
    n_weak: usize,
    /// Fraction of extra detections that are rotational or combined.
    frac_topology_positive: f64,
    extra_detections: Vec<ExtraDetectionSummary>,
    ascii_summary: String,
}

// ============================================================================
// Helpers
// ============================================================================

/// Dot product of two 3-vectors.
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Mean 3-vector over a slice of FGM records, normalized to unit length.
/// Returns None if the slice is empty or the mean magnitude is zero.
fn mean_unit_b(records: &[&MmsFgmMinuteRecord]) -> Option<[f64; 3]> {
    if records.is_empty() {
        return None;
    }
    let n = records.len() as f64;
    let mx = records.iter().map(|r| r.bx_gse).sum::<f64>() / n;
    let my = records.iter().map(|r| r.by_gse).sum::<f64>() / n;
    let mz = records.iter().map(|r| r.bz_gse).sum::<f64>() / n;
    let mag = (mx * mx + my * my + mz * mz).sqrt();
    if mag < 1e-9 {
        return None;
    }
    Some([mx / mag, my / mag, mz / mag])
}

/// Compute angle (degrees) between two unit vectors.
fn angle_between_deg(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let cos_angle = dot3(a, b).clamp(-1.0, 1.0);
    cos_angle.acos().to_degrees()
}

fn classify(rotation_deg: f64, compressiveness: f64, rot_thr: f64, comp_thr: f64) -> &'static str {
    match (rotation_deg >= rot_thr, compressiveness >= comp_thr) {
        (true, false) => "rotational",
        (false, true) => "compressive",
        (true, true) => "combined",
        (false, false) => "weak",
    }
}

// ============================================================================
// main
// ============================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

    // -----------------------------------------------------------------------
    // Step 1: Load the evaluation JSON
    // -----------------------------------------------------------------------
    println!("[1/4] Loading evaluation JSON: {}", cli.eval_json.display());
    let eval_str = fs::read_to_string(&cli.eval_json)
        .with_context(|| format!("reading {}", cli.eval_json.display()))?;
    let eval: EvalJson = serde_json::from_str(&eval_str)?;

    let start = NaiveDate::parse_from_str(&eval.start_date, "%Y-%m-%d")
        .with_context(|| format!("parsing start_date: {}", eval.start_date))?;

    // Reference midnight: the start of day 1.
    let reference_midnight = start
        .and_hms_opt(0, 0, 0)
        .context("building reference midnight")?;

    println!(
        "  {} CD fires, {} SITL events",
        eval.cd_fire_hours.len(),
        eval.sitl_events.len()
    );

    // Parse SITL event time ranges.
    let sitl_ranges: Vec<(NaiveDateTime, NaiveDateTime, bool)> = eval
        .sitl_events
        .iter()
        .filter_map(|ev| {
            let s = NaiveDateTime::parse_from_str(&ev.start, "%Y-%m-%dT%H:%M:%S").ok()?;
            let e = NaiveDateTime::parse_from_str(&ev.end, "%Y-%m-%dT%H:%M:%S").ok()?;
            Some((s, e, ev.cd_matched))
        })
        .collect();

    // Identify "extra" fires: CD fires NOT inside any FPI event interval.
    let extra_hours: Vec<f64> = eval
        .cd_fire_hours
        .iter()
        .copied()
        .filter(|&h| {
            let t = reference_midnight + Duration::seconds((h * 3600.0) as i64);
            !sitl_ranges.iter().any(|(s, e, _)| t >= *s && t < *e)
        })
        .collect();

    println!(
        "  Extra (unmatched) CD fires: {}/{} ({:.1}%)",
        extra_hours.len(),
        eval.cd_fire_hours.len(),
        100.0 * extra_hours.len() as f64 / eval.cd_fire_hours.len().max(1) as f64
    );

    // -----------------------------------------------------------------------
    // Step 2: Load MMS FGM minute data
    // -----------------------------------------------------------------------
    println!(
        "[2/4] Loading MMS FGM minute data ({} days)...",
        eval.n_days
    );
    let mms_dir = cli.data_dir.join("mms");
    let mut all_minutes: Vec<MmsFgmMinuteRecord> = Vec::new();

    for day_offset in 0..eval.n_days {
        let date = start + Duration::days(day_offset as i64);
        // MMS FGM files are named {year}_{doy}_{doy}.csv to match the
        // heliosphere-mms-sitl-labeled cache convention.
        let doy = date.ordinal();
        let y = date.year();
        let fname = format!("mms1_fgm_srvy_l2_{y}_{doy}_{doy}.csv");
        let path = mms_dir.join(&fname);
        if !path.exists() {
            eprintln!("  Warning: {} not found", path.display());
            continue;
        }
        let content = fs::read_to_string(&path)?;
        let mut recs = parse_mms_fgm_hapi_csv_minutes(&content);
        println!("  {}: {} minute records", fname, recs.len());
        all_minutes.append(&mut recs);
    }

    all_minutes.sort_by(|a, b| {
        a.year
            .cmp(&b.year)
            .then(a.doy.cmp(&b.doy))
            .then(a.hour.cmp(&b.hour))
            .then(a.minute.cmp(&b.minute))
    });

    // Recompute elapsed hours from reference midnight.
    if all_minutes.is_empty() {
        anyhow::bail!("No FGM data loaded.");
    }
    let (ry, rd, rh, rm) = {
        let y = start.year() as u16;
        let d = start.ordinal() as u16;
        (y, d, 0u8, 0u8)
    };
    for rec in &mut all_minutes {
        let dy = (rec.year as f64 - ry as f64) * 365.25;
        let dd = rec.doy as f64 - rd as f64;
        rec.elapsed_hours = (dy + dd) * 24.0
            + (rec.hour as f64 - rh as f64)
            + (rec.minute as f64 - rm as f64) / 60.0;
    }

    println!("  Total: {} minute records", all_minutes.len());

    // -----------------------------------------------------------------------
    // Step 3: Classify each extra detection
    // -----------------------------------------------------------------------
    println!(
        "[3/4] Classifying {} extra detections...",
        extra_hours.len()
    );

    let half = cli.window_minutes;
    let half_h = half as f64 / 60.0;

    let mut summaries: Vec<ExtraDetectionSummary> = Vec::new();

    for &fire_h in &extra_hours {
        // Extract context window.
        let win_start_h = fire_h - half_h;
        let win_end_h = fire_h + half_h;
        let win_recs: Vec<&MmsFgmMinuteRecord> = all_minutes
            .iter()
            .filter(|r| r.elapsed_hours >= win_start_h && r.elapsed_hours <= win_end_h)
            .collect();

        if win_recs.len() < 4 {
            // Not enough context (e.g. near edge of data).
            continue;
        }

        // Compressiveness: relative |B| range in window.
        let bmags: Vec<f64> = win_recs.iter().map(|r| r.b_magnitude).collect();
        let b_min = bmags.iter().cloned().fold(f64::INFINITY, f64::min);
        let b_max = bmags.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let b_mean = bmags.iter().sum::<f64>() / bmags.len() as f64;
        let compressiveness = if b_mean > 0.0 {
            (b_max - b_min) / b_mean
        } else {
            0.0
        };

        // Rotation: angle between mean B in pre-half and post-half.
        let mid = win_recs.len() / 2;
        let pre_recs: Vec<&MmsFgmMinuteRecord> = win_recs[..mid].to_vec();
        let post_recs: Vec<&MmsFgmMinuteRecord> = win_recs[mid..].to_vec();
        let rotation_deg = match (mean_unit_b(&pre_recs), mean_unit_b(&post_recs)) {
            (Some(b_pre), Some(b_post)) => angle_between_deg(&b_pre, &b_post),
            _ => 0.0,
        };

        let label = classify(
            rotation_deg,
            compressiveness,
            cli.rot_threshold,
            cli.comp_threshold,
        );

        // Reconstruct approximate wall-clock timestamp.
        let t = reference_midnight + Duration::seconds((fire_h * 3600.0) as i64);
        summaries.push(ExtraDetectionSummary {
            timestamp: t.format("%Y-%m-%dT%H:%M").to_string(),
            elapsed_hours: fire_h,
            compressiveness,
            rotation_deg,
            classification: label.to_string(),
        });
    }

    // -----------------------------------------------------------------------
    // Step 4: Tabulate and report
    // -----------------------------------------------------------------------
    let n_rotational = summaries
        .iter()
        .filter(|s| s.classification == "rotational")
        .count();
    let n_compressive = summaries
        .iter()
        .filter(|s| s.classification == "compressive")
        .count();
    let n_combined = summaries
        .iter()
        .filter(|s| s.classification == "combined")
        .count();
    let n_weak = summaries
        .iter()
        .filter(|s| s.classification == "weak")
        .count();
    let topology_positive = n_rotational + n_combined;
    let frac_tp = topology_positive as f64 / summaries.len().max(1) as f64;

    let mut ascii = String::new();
    ascii.push_str("MMS FPI Extra-Detection Classification (JGR Reviewer Response)\n");
    ascii.push_str("================================================================\n");
    ascii.push_str(&format!(
        "Total extra (unmatched) CD detections analyzed: {}\n",
        summaries.len()
    ));
    ascii.push_str(&format!(
        "  Rotational (rotation>{:.0} deg, delta_B<{:.2}): {:3}  ({:.1}%)\n",
        cli.rot_threshold,
        cli.comp_threshold,
        n_rotational,
        100.0 * n_rotational as f64 / summaries.len().max(1) as f64
    ));
    ascii.push_str(&format!(
        "  Compressive (rotation<{:.0} deg, delta_B>{:.2}): {:3}  ({:.1}%)\n",
        cli.rot_threshold,
        cli.comp_threshold,
        n_compressive,
        100.0 * n_compressive as f64 / summaries.len().max(1) as f64
    ));
    ascii.push_str(&format!(
        "  Combined (both):                           {:3}  ({:.1}%)\n",
        n_combined,
        100.0 * n_combined as f64 / summaries.len().max(1) as f64
    ));
    ascii.push_str(&format!(
        "  Weak (below both thresholds):              {:3}  ({:.1}%)\n",
        n_weak,
        100.0 * n_weak as f64 / summaries.len().max(1) as f64
    ));
    ascii.push_str(&format!(
        "Topology-positive (rotational + combined): {}/{} = {:.1}%\n",
        topology_positive,
        summaries.len(),
        100.0 * frac_tp
    ));
    ascii.push_str("----------------------------------------------------------------\n");
    ascii.push_str(&format!(
        "{:<22} {:>7} {:>9} {:>13}\n",
        "Timestamp", "dB_rel", "Rot(deg)", "Class"
    ));
    ascii.push_str(&"-".repeat(56));
    ascii.push('\n');
    for s in &summaries {
        ascii.push_str(&format!(
            "{:<22} {:>7.3} {:>9.1} {:>13}\n",
            s.timestamp, s.compressiveness, s.rotation_deg, s.classification
        ));
    }

    println!("\n{}", ascii);

    let results = CaseStudyResults {
        eval_json: cli.eval_json.display().to_string(),
        n_cd_fires: eval.cd_fire_hours.len(),
        n_matched_to_fpi: eval.cd_fire_hours.len() - extra_hours.len(),
        n_extra: extra_hours.len(),
        rot_threshold_deg: cli.rot_threshold,
        comp_threshold: cli.comp_threshold,
        n_rotational,
        n_compressive,
        n_combined,
        n_weak,
        frac_topology_positive: frac_tp,
        extra_detections: summaries,
        ascii_summary: ascii,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&results)?)?;
    println!("[4/4] Wrote {}", cli.out_json.display());
    Ok(())
}
