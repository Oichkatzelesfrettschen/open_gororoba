//! Consolidate per-mission FA attribution JSONs into a cross-mission summary table.
//!
//! Reads output files from heliosphere-fa-attribution (unified format) and generates
//! a single JSON table + ASCII summary comparing Precision / Recall / F1 and
//! false-alarm type distributions across THEMIS, Cluster, and MAVEN.
//!
//! WHY: The methods paper requires a cross-mission comparison table showing that
//! the CD associator generalises across very different plasma regimes.  Each
//! per-mission file is produced by the same binary with consistent field names,
//! making aggregation straightforward.
//!
//! NOTE on Recall > 1.0: The 2009w40 period has recall > 1.0 due to multi-match
//! bunching -- a single CD transition catches multiple curated crossings within the
//! 20-minute tolerance window.  This is a known measurement artefact of magnetopause
//! oscillation periods where the boundary dwells near threshold for extended intervals.
//! Such rows are flagged in the output with a multi_match_flag and excluded from
//! mission-level aggregates.
//!
//! Usage:
//!   heliosphere-fa-cross-mission-table
//!   heliosphere-fa-cross-mission-table --input-dir data/output/heliosphere/ablations

use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Args)]
pub struct Cli {
    /// Directory containing fa_attribution_*.json files.
    #[arg(long, default_value = "data/output/heliosphere/ablations")]
    input_dir: PathBuf,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/fa_cross_mission_table.json"
    )]
    out_json: PathBuf,
}

/// Subset of fields we read from each per-mission JSON.
#[derive(Debug, Deserialize)]
struct MissionRaw {
    start_date: String,
    end_date: String,
    #[serde(default)]
    mission: String,
    n_cd_transitions: usize,
    n_curated_matched: usize,
    n_curated_crossings: usize,
    n_extras: usize,
    precision: f64,
    recall: f64,
    f1: f64,
    type_counts: std::collections::BTreeMap<String, usize>,
}

/// One row in the output table.
#[derive(Debug, Serialize)]
struct TableRow {
    mission: String,
    period: String,
    n_curated_crossings: usize,
    n_cd_transitions: usize,
    n_extras: usize,
    tp: usize,
    fn_count: usize,
    precision: f64,
    recall: f64,
    f1: f64,
    dominant_fa_type: String,
    fa_type_fractions: std::collections::BTreeMap<String, f64>,
    multi_match_flag: bool,
    note: String,
}

/// Aggregate P/R/F1 across multiple rows (micro-averaging over events).
#[derive(Debug, Serialize)]
struct MissionAggregate {
    mission: String,
    n_periods: usize,
    total_curated_crossings: usize,
    total_cd_transitions: usize,
    total_tp: usize,
    micro_precision: f64,
    micro_recall: f64,
    micro_f1: f64,
    mean_f1: f64,
    std_f1: f64,
    pooled_fa_type_fractions: std::collections::BTreeMap<String, f64>,
}

#[derive(Debug, Serialize)]
struct CrossMissionTable {
    generated: String,
    per_period_rows: Vec<TableRow>,
    mission_aggregates: Vec<MissionAggregate>,
    comparison_note: String,
    ascii_table: String,
}

fn dominant_type(counts: &std::collections::BTreeMap<String, usize>) -> String {
    counts
        .iter()
        .filter(|(k, _)| *k != "Unclassified")
        .max_by_key(|(_, v)| *v)
        .map(|(k, _)| k.clone())
        .unwrap_or_else(|| "Unclassified".to_string())
}

fn type_fractions(
    counts: &std::collections::BTreeMap<String, usize>,
) -> std::collections::BTreeMap<String, f64> {
    let total: usize = counts.values().sum();
    if total == 0 {
        return counts.keys().map(|k| (k.clone(), 0.0)).collect();
    }
    counts
        .iter()
        .map(|(k, &v)| (k.clone(), (v as f64 / total as f64 * 100.0).round() / 100.0))
        .collect()
}

/// Build pooled type fractions by summing across rows.
fn pooled_type_fractions(rows: &[&TableRow]) -> std::collections::BTreeMap<String, f64> {
    let mut totals: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    let mut grand_total = 0usize;
    for row in rows {
        let n = row.n_extras;
        grand_total += n;
        for (k, &frac) in &row.fa_type_fractions {
            *totals.entry(k.clone()).or_insert(0.0) += frac * n as f64;
        }
    }
    if grand_total == 0 {
        return totals;
    }
    totals
        .iter_mut()
        .for_each(|(_, v)| *v = (*v / grand_total as f64 * 100.0).round() / 100.0);
    totals
}

fn build_ascii_table(rows: &[TableRow], aggregates: &[MissionAggregate]) -> String {
    let mut out = String::new();
    out.push_str("Cross-mission CD associator FA attribution (methods paper table)\n");
    out.push_str("=================================================================\n");
    out.push_str(&format!(
        "{:<24} {:>10} {:>5} {:>5} {:>7} {:>7} {:>7}  {:<22}\n",
        "Period", "Crossings", "TP", "FP", "Prec", "Rec", "F1", "Dominant FA type"
    ));
    out.push_str(&"-".repeat(95));
    out.push('\n');
    for row in rows {
        let flag = if row.multi_match_flag { "*" } else { " " };
        out.push_str(&format!(
            "{:<24} {:>10} {:>5} {:>5} {:>7.3} {:>7.3} {:>7.3}  {:<22} {}\n",
            row.period,
            row.n_curated_crossings,
            row.tp,
            row.n_extras,
            row.precision,
            row.recall,
            row.f1,
            row.dominant_fa_type,
            flag
        ));
    }
    out.push_str(&"-".repeat(95));
    out.push('\n');
    out.push_str(
        "* multi-match flag (recall>1 due to bunched crossings; excluded from aggregate)\n\n",
    );

    out.push_str("Mission-level micro-averages (excluding flagged rows):\n");
    out.push_str(&format!(
        "{:<12} {:>6} {:>5} {:>5} {:>7} {:>7} {:>7}  mean-F1\n",
        "Mission", "Xings", "TP", "FP", "mPrec", "mRec", "mF1"
    ));
    out.push_str(&"-".repeat(65));
    out.push('\n');
    for agg in aggregates {
        out.push_str(&format!(
            "{:<12} {:>6} {:>5} {:>5} {:>7.3} {:>7.3} {:>7.3}  {:.3}\n",
            agg.mission,
            agg.total_curated_crossings,
            agg.total_tp,
            agg.total_cd_transitions - agg.total_tp,
            agg.micro_precision,
            agg.micro_recall,
            agg.micro_f1,
            agg.mean_f1,
        ));
    }
    out
}

fn load_mission_json(path: &PathBuf, label: &str) -> Result<MissionRaw> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut raw: MissionRaw = serde_json::from_str(&content)
        .with_context(|| format!("parsing {}: {}", path.display(), label))?;
    if raw.mission.is_empty() {
        raw.mission = label.to_string();
    }
    Ok(raw)
}

pub fn run(cli: Cli) -> Result<()> {

    // File list: (path, label).
    let files: &[(&str, &str)] = &[
        ("fa_attribution_themis_2008w44.json", "THEMIS 2008w44"),
        ("fa_attribution_themis_2009w40.json", "THEMIS 2009w40"),
        ("fa_attribution_themis_2009w44.json", "THEMIS 2009w44"),
        ("fa_attribution_cluster_2002w01.json", "Cluster 2002w01"),
        ("fa_attribution_maven_2015.json", "MAVEN 2015"),
    ];

    let mut rows: Vec<TableRow> = Vec::new();

    for (fname, label) in files {
        let path = cli.input_dir.join(fname);
        if !path.exists() {
            eprintln!("  SKIP (missing): {}", path.display());
            continue;
        }
        let raw = load_mission_json(&path, label)?;

        let tp = raw.n_curated_matched;
        let fn_count = raw.n_curated_crossings.saturating_sub(tp);
        let multi_match = raw.recall > 1.001;
        let fracs = type_fractions(&raw.type_counts);
        let dom = dominant_type(&raw.type_counts);

        let period = format!("{} to {}", raw.start_date, raw.end_date);
        let note = if multi_match {
            "Recall>1: magnetopause oscillation bunching within 20-min window".to_string()
        } else {
            String::new()
        };

        rows.push(TableRow {
            mission: label.to_string(),
            period,
            n_curated_crossings: raw.n_curated_crossings,
            n_cd_transitions: raw.n_cd_transitions,
            n_extras: raw.n_extras,
            tp,
            fn_count,
            precision: raw.precision,
            recall: raw.recall.min(1.0),
            f1: raw.f1,
            dominant_fa_type: dom,
            fa_type_fractions: fracs,
            multi_match_flag: multi_match,
            note,
        });

        let flag = if multi_match { " [multi-match]" } else { "" };
        println!(
            "  {:20}  P={:.3}  R={:.3}  F1={:.3}  crossings={}{flag}",
            label, raw.precision, raw.recall, raw.f1, raw.n_curated_crossings
        );
    }

    // Mission-level aggregates.
    let mission_keys = &["THEMIS", "Cluster", "MAVEN"];
    let mut aggregates: Vec<MissionAggregate> = Vec::new();

    for key in mission_keys {
        let mission_rows: Vec<&TableRow> = rows
            .iter()
            .filter(|r| r.mission.starts_with(key) && !r.multi_match_flag)
            .collect();
        if mission_rows.is_empty() {
            continue;
        }

        let total_xings: usize = mission_rows.iter().map(|r| r.n_curated_crossings).sum();
        let total_tp: usize = mission_rows.iter().map(|r| r.tp).sum();
        let total_cd: usize = mission_rows.iter().map(|r| r.n_cd_transitions).sum();
        let micro_prec = if total_cd > 0 {
            total_tp as f64 / total_cd as f64
        } else {
            0.0
        };
        let micro_rec = if total_xings > 0 {
            total_tp as f64 / total_xings as f64
        } else {
            0.0
        };
        let micro_f1 = if micro_prec + micro_rec > 0.0 {
            2.0 * micro_prec * micro_rec / (micro_prec + micro_rec)
        } else {
            0.0
        };

        let f1s: Vec<f64> = mission_rows.iter().map(|r| r.f1).collect();
        let mean_f1 = f1s.iter().sum::<f64>() / f1s.len() as f64;
        let std_f1 = if f1s.len() > 1 {
            let var =
                f1s.iter().map(|f| (f - mean_f1).powi(2)).sum::<f64>() / (f1s.len() - 1) as f64;
            var.sqrt()
        } else {
            0.0
        };

        let pooled = pooled_type_fractions(&mission_rows);

        aggregates.push(MissionAggregate {
            mission: key.to_string(),
            n_periods: mission_rows.len(),
            total_curated_crossings: total_xings,
            total_cd_transitions: total_cd,
            total_tp,
            micro_precision: micro_prec,
            micro_recall: micro_rec,
            micro_f1,
            mean_f1,
            std_f1,
            pooled_fa_type_fractions: pooled,
        });
    }

    let ascii = build_ascii_table(&rows, &aggregates);
    println!("\n{}", ascii);

    let table = CrossMissionTable {
        generated: chrono::Utc::now().format("%Y-%m-%d").to_string(),
        per_period_rows: rows,
        mission_aggregates: aggregates,
        comparison_note: concat!(
            "All rows produced by heliosphere-fa-attribution with uniform parameters: ",
            "embedding_dim=32, match_tolerance=20min, max_bmag=200nT. ",
            "Multi-match rows excluded from micro-averages. ",
            "Dominant FA type excludes Unclassified bucket."
        )
        .to_string(),
        ascii_table: ascii,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&table)?)?;
    println!("Wrote {}", cli.out_json.display());

    Ok(())
}
