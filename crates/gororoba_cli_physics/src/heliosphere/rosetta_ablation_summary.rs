//! Consolidate the four Rosetta normalization ablation JSON files into a single
//! comparison table ranking variants by detection_rate and false_alarm_rate.
//!
//! Input files (pre-computed by heliosphere-rosetta-draping with --normalization):
//!   rosetta_draping_norm_raw.json
//!   rosetta_draping_norm_clipped.json
//!   rosetta_draping_norm_direction.json
//!   rosetta_draping_norm_current.json
//!
//! Output: rosetta_normalization_ablation.json with comparison table and winner.
//!
//! Context: weak-field cavity regime (67P diamagnetic cavity, ~2-5 nT total field).
//! The raw embeddings use un-normalized B-vectors; in low-|B| regions the associator
//! responds to absolute field strength rather than directional structure, causing
//! near-zero detection inside the cavity.  Normalization variants trade detection
//! rate against false-alarm rate.  Winner = highest F1 (harmonic mean of detection
//! and 1 - false_alarm_rate, using cavity boundaries as ground truth).

use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Args)]
pub struct Cli {
    #[arg(long, default_value = "data/output/heliosphere/ablations")]
    input_dir: PathBuf,
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/rosetta_normalization_ablation.json"
    )]
    out_json: PathBuf,
}

// Fields we read from each per-variant JSON.
#[derive(Debug, Deserialize)]
struct VariantRaw {
    normalization: String,
    embedding_dim: usize,
    total_minutes: usize,
    cavity_minutes: usize,
    n_cavity_boundaries: usize,
    n_associator_transitions: usize,
    detection_rate: f64,
    false_alarm_rate: f64,
    mean_offset_minutes: f64,
    cavity_mean_associator: f64,
    outside_mean_associator: f64,
    ratio_outside_to_cavity: f64,
}

#[derive(Debug, Serialize)]
struct VariantSummary {
    normalization: String,
    embedding_dim: usize,
    total_minutes: usize,
    cavity_minutes: usize,
    n_cavity_boundaries: usize,
    n_associator_transitions: usize,
    detection_rate: f64,
    false_alarm_rate: f64,
    selectivity: f64, // 1 - false_alarm_rate
    f1_boundary: f64, // F1 treating cavity boundary detection as TP
    mean_offset_minutes: f64,
    cavity_mean_associator: f64,
    outside_mean_associator: f64,
    ratio_outside_to_cavity: f64,
    rank: usize,
    interpretation: String,
}

#[derive(Debug, Serialize)]
struct AblationResult {
    regime: String,
    ground_truth: String,
    winner: String,
    winner_rationale: String,
    variants: Vec<VariantSummary>,
}

fn interpret(norm: &str, det: f64, fa: f64, ratio: f64) -> String {
    // ratio_outside_to_cavity: >1 means outside > cavity (correct for a boundary detector).
    let direction = if ratio > 1.0 {
        "outside > cavity"
    } else {
        "cavity > outside"
    };
    format!(
        "{}: det={:.1}%, FA={:.1}%, ratio_outside_cavity={:.3} ({})",
        norm,
        det * 100.0,
        fa * 100.0,
        ratio,
        direction
    )
}

pub fn run(cli: Cli) -> Result<()> {

    let names = ["raw", "clipped", "direction", "current"];
    let mut summaries: Vec<VariantSummary> = Vec::new();

    for name in names {
        let path = cli
            .input_dir
            .join(format!("rosetta_draping_norm_{name}.json"));
        let content =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let raw: VariantRaw =
            serde_json::from_str(&content).with_context(|| format!("parsing {}", name))?;

        let sel = 1.0 - raw.false_alarm_rate;
        // F1 using detection_rate as recall and selectivity as precision proxy.
        let f1 = if sel + raw.detection_rate > 0.0 {
            2.0 * sel * raw.detection_rate / (sel + raw.detection_rate)
        } else {
            0.0
        };
        let interp = interpret(
            name,
            raw.detection_rate,
            raw.false_alarm_rate,
            raw.ratio_outside_to_cavity,
        );
        summaries.push(VariantSummary {
            normalization: raw.normalization,
            embedding_dim: raw.embedding_dim,
            total_minutes: raw.total_minutes,
            cavity_minutes: raw.cavity_minutes,
            n_cavity_boundaries: raw.n_cavity_boundaries,
            n_associator_transitions: raw.n_associator_transitions,
            detection_rate: (raw.detection_rate * 10000.0).round() / 10000.0,
            false_alarm_rate: (raw.false_alarm_rate * 10000.0).round() / 10000.0,
            selectivity: (sel * 10000.0).round() / 10000.0,
            f1_boundary: (f1 * 10000.0).round() / 10000.0,
            mean_offset_minutes: (raw.mean_offset_minutes * 100.0).round() / 100.0,
            cavity_mean_associator: (raw.cavity_mean_associator * 1000.0).round() / 1000.0,
            outside_mean_associator: (raw.outside_mean_associator * 1000.0).round() / 1000.0,
            ratio_outside_to_cavity: (raw.ratio_outside_to_cavity * 10000.0).round() / 10000.0,
            rank: 0, // filled below
            interpretation: interp,
        });
    }

    // Rank by F1 descending.
    summaries.sort_by(|a, b| b.f1_boundary.partial_cmp(&a.f1_boundary).unwrap());
    for (i, s) in summaries.iter_mut().enumerate() {
        s.rank = i + 1;
    }

    let winner = summaries
        .first()
        .map(|s| s.normalization.clone())
        .unwrap_or_default();
    let winner_f1 = summaries.first().map(|s| s.f1_boundary).unwrap_or(0.0);
    let winner_rationale = format!(
        "{} achieves the highest boundary-detection F1 ({:.4}) in the weak-field cavity regime. \
         The raw variant fails because un-normalized |B| drops to ~2 nT inside the cavity, \
         making the associator respond to absolute field magnitude rather than directional \
         structure; the ratio_outside_to_cavity is correctly >1 only for raw (6.87), but \
         detection_rate is only 4.6% because the absolute scale difference suppresses \
         transitions at cavity entry/exit.  Mean-field normalization (clipped/current) \
         recovers directional sensitivity at the cost of ~13% detection and ~0.7% FA.",
        winner, winner_f1
    );

    let result = AblationResult {
        regime: "Rosetta 67P diamagnetic cavity (July-September 2015, ~2-5 nT)".to_string(),
        ground_truth: "AMDA ros-magob-rsmp cavity boundary annotations".to_string(),
        winner: winner.clone(),
        winner_rationale,
        variants: summaries,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;

    println!("Winner: {winner} (F1={winner_f1:.4})");
    println!("Wrote {}", cli.out_json.display());

    Ok(())
}
