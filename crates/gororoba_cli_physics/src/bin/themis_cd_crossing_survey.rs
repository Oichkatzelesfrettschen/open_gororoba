//! Supervised THEMIS magnetopause crossing survey: CD associator early/late ratio.
//!
//! Uses ground-truth labeled crossings (Staples et al. 2020, v2 catalog) to measure
//! the CD associator early/late norm ratio in the 12 steps BEFORE each confirmed
//! magnetopause crossing. Compares against:
//!   - ACE unsupervised result (C-1618): 4.30x mean, 2.40x median
//!   - LBM BGK C-1617 baseline: 1.92x
//!
//! Method:
//!   1. Parse crossing catalog: filter probe "a", group by (year, doy)
//!   2. For each (year, doy) group: load FGM CSV, parse rows to (seconds_from_midnight, bx, by, bz)
//!   3. For each crossing: binary search for crossing time in FGM rows
//!   4. Extract BACK_ROWS rows before the crossing index
//!   5. Build 32D Takens embeddings (8-step x 4-channel: bx, by, bz, bmag), unit-normalized
//!   6. Compute batch_sliding_associator_norms_parallel on pre-crossing window
//!   7. Extract 12 norms immediately before the crossing position
//!   8. Compute early/late ratio (first 6 vs last 6)
//!
//! Physical interpretation: if early/late ratio > 1.0 systematically before ground-truth
//! crossings, the CD associator precursor is confirmed on labeled data, not just p95 peaks.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::Result;
use cd_kernel::batch_sliding_associator_norms_parallel;
use clap::Parser;
use serde::Serialize;

#[derive(Parser)]
#[command(name = "themis-cd-crossing-survey")]
#[command(about = "Supervised CD associator early/late ratio at THEMIS magnetopause crossings")]
struct Cli {
    /// Path to THEMIS crossing catalog TXT (Staples et al. 2020 v2).
    #[arg(
        long,
        default_value = "data/external/crossing_lists/themis_mp_crossings_v2.txt"
    )]
    crossings_txt: PathBuf,

    /// Directory containing th{probe}_fgm_{year}_{doy:03}.csv files.
    #[arg(long, default_value = "data/external/themis")]
    fgm_dir: PathBuf,

    /// Probe filter (single letter, e.g. "a").
    #[arg(long, default_value = "a")]
    probe: String,

    /// Takens embedding dimension (must be power of 2 and equal to steps*4).
    #[arg(long, default_value_t = 32)]
    embed_dim: usize,

    /// Steps per Takens window (embed_dim / 4 channels).
    #[arg(long, default_value_t = 8)]
    steps: usize,

    /// Precursor window length in norms (must be even, split equally early/late).
    #[arg(long, default_value_t = 12)]
    precursor_window: usize,

    /// FGM rows to load before each crossing (must be > steps + precursor_window + 2).
    #[arg(long, default_value_t = 120)]
    back_rows: usize,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/themis_cd_crossing_survey.json"
    )]
    out_json: PathBuf,
}

/// One crossing entry from the catalog.
struct Crossing {
    year: u32,
    doy: u32,
    secs_from_midnight: f64,
}

/// One FGM row parsed from the CSV.
struct FgmRow {
    secs: f64,
    bx: f64,
    by: f64,
    bz: f64,
}

#[derive(Debug, Serialize)]
struct CrossingResult {
    n_crossings_analyzed: usize,
    n_crossings_skipped: usize,
    early_late_ratio_mean: f64,
    early_late_ratio_median: f64,
    early_late_ratio_p25: f64,
    early_late_ratio_p75: f64,
    fraction_above_1_0: f64,
    fraction_above_1_1: f64,
    fraction_above_1_5: f64,
    /// Number of crossings where ratio > 1.0 (monotonic buildup expected)
    precursor_confirmed: bool,
}

#[derive(Debug, Serialize)]
struct SurveyOutput {
    lbm_reference_ratio: f64,
    ace_unsupervised_mean_ratio: f64,
    ace_unsupervised_median_ratio: f64,
    themis_supervised: CrossingResult,
    interpretation: String,
}

/// Parse "2007-04-21" -> day-of-year (1-based).
fn date_to_doy(year: u32, month: u32, day: u32) -> u32 {
    let leap = (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400);
    let days_in_month: [u32; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    days_in_month[..(month as usize - 1)].iter().sum::<u32>() + day
}

/// Parse "2007-04-21T23:35:24Z" -> (year, doy, secs_from_midnight).
fn parse_crossing_timestamp(ts: &str) -> Option<Crossing> {
    // Format: YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DDTHH:MM:SS.fffZ
    let t_pos = ts.find('T')?;
    let date_part = &ts[..t_pos];
    let time_part = ts[t_pos + 1..].trim_end_matches('Z');

    let date_fields: Vec<u32> = date_part
        .split('-')
        .filter_map(|s| s.parse().ok())
        .collect();
    if date_fields.len() < 3 {
        return None;
    }
    let (year, month, day) = (date_fields[0], date_fields[1], date_fields[2]);
    let doy = date_to_doy(year, month, day);

    let time_fields: Vec<&str> = time_part.split(':').collect();
    if time_fields.len() < 3 {
        return None;
    }
    let hh: f64 = time_fields[0].parse().ok()?;
    let mm: f64 = time_fields[1].parse().ok()?;
    let ss: f64 = time_fields[2].parse().ok()?;
    let secs_from_midnight = hh * 3600.0 + mm * 60.0 + ss;

    Some(Crossing {
        year,
        doy,
        secs_from_midnight,
    })
}

/// Parse FGM timestamp "2008-07-04T00:00:02.734Z" -> seconds from midnight.
fn parse_fgm_timestamp_secs(ts: &str) -> Option<f64> {
    let t_pos = ts.find('T')?;
    let time_part = ts[t_pos + 1..].trim_end_matches('Z');
    let fields: Vec<&str> = time_part.split(':').collect();
    if fields.len() < 3 {
        return None;
    }
    let hh: f64 = fields[0].parse().ok()?;
    let mm: f64 = fields[1].parse().ok()?;
    let ss: f64 = fields[2].parse().ok()?;
    Some(hh * 3600.0 + mm * 60.0 + ss)
}

/// Load all rows from a THEMIS FGM CSV file (Bx, By, Bz in GSE nT).
fn load_fgm_csv(path: &Path) -> Result<Vec<FgmRow>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = result?;
        if record.len() < 4 {
            continue;
        }
        let secs = match parse_fgm_timestamp_secs(&record[0]) {
            Some(s) => s,
            None => continue,
        };
        let bx: f64 = match record[1].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let by: f64 = match record[2].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let bz: f64 = match record[3].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if bx.is_finite() && by.is_finite() && bz.is_finite() {
            rows.push(FgmRow { secs, bx, by, bz });
        }
    }
    Ok(rows)
}

/// Build 32D Takens embedding from a slice of FGM rows.
/// Channels: bx, by, bz, bmag (unit-normalized).
fn build_embeddings(rows: &[FgmRow], steps: usize, embed_dim: usize) -> Vec<Vec<f64>> {
    let n_ch = 4;
    let n = rows.len();
    let mut embedded = Vec::with_capacity(n.saturating_sub(steps));
    for start in 0..n.saturating_sub(steps) {
        let mut v = vec![0.0f64; embed_dim];
        for s in 0..steps {
            let r = &rows[start + s];
            let bmag = (r.bx * r.bx + r.by * r.by + r.bz * r.bz).sqrt();
            v[s * n_ch] = r.bx;
            v[s * n_ch + 1] = r.by;
            v[s * n_ch + 2] = r.bz;
            v[s * n_ch + 3] = bmag;
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in v.iter_mut() {
                *x /= norm;
            }
            embedded.push(v);
        }
    }
    embedded
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    assert_eq!(
        cli.steps * 4,
        cli.embed_dim,
        "steps ({}) * 4 must equal embed_dim ({})",
        cli.steps,
        cli.embed_dim
    );
    assert!(cli.embed_dim.is_power_of_two());
    assert!(cli.precursor_window % 2 == 0);
    assert!(
        cli.back_rows > cli.steps + cli.precursor_window + 2,
        "back_rows too small for requested precursor analysis"
    );

    // -----------------------------------------------------------------------
    // 1. Parse and group crossings by (year, doy)
    // -----------------------------------------------------------------------
    eprintln!("Parsing crossing catalog: {}", cli.crossings_txt.display());
    let content = std::fs::read_to_string(&cli.crossings_txt)?;

    // Group: BTreeMap<(year, doy), Vec<secs_from_midnight>>
    let mut crossing_groups: BTreeMap<(u32, u32), Vec<f64>> = BTreeMap::new();
    let mut total_parsed = 0usize;

    for line in content.lines() {
        if line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 8 {
            continue;
        }
        let probe_field = fields[7].trim();
        if probe_field != cli.probe {
            continue;
        }
        if let Some(crossing) = parse_crossing_timestamp(fields[0].trim()) {
            crossing_groups
                .entry((crossing.year, crossing.doy))
                .or_default()
                .push(crossing.secs_from_midnight);
            total_parsed += 1;
        }
    }

    eprintln!(
        "Probe '{}' crossings parsed: {} across {} file groups",
        cli.probe,
        total_parsed,
        crossing_groups.len()
    );

    // -----------------------------------------------------------------------
    // 2. Process each (year, doy) group
    // -----------------------------------------------------------------------
    let mut all_ratios: Vec<f64> = Vec::new();
    let mut n_skipped = 0usize;
    let half = cli.precursor_window / 2;

    for ((year, doy), crossing_secs) in &crossing_groups {
        let fgm_path = cli
            .fgm_dir
            .join(format!("th{}_fgm_{}_{:03}.csv", cli.probe, year, doy));
        if !fgm_path.exists() {
            n_skipped += crossing_secs.len();
            continue;
        }

        let rows = match load_fgm_csv(&fgm_path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  WARN: {}: {}", fgm_path.display(), e);
                n_skipped += crossing_secs.len();
                continue;
            }
        };

        if rows.len() < cli.back_rows + cli.steps {
            n_skipped += crossing_secs.len();
            continue;
        }

        // Sort rows by time (should already be sorted, but enforce it).
        // rows is already sorted since FGM files are time-ordered.

        for &cross_secs in crossing_secs {
            // Binary search for the FGM row closest to the crossing time.
            let cross_row_idx = {
                let pos = rows.partition_point(|r| r.secs < cross_secs);
                if pos == 0 {
                    0
                } else if pos >= rows.len() {
                    rows.len() - 1
                } else if (rows[pos].secs - cross_secs).abs()
                    < (rows[pos - 1].secs - cross_secs).abs()
                {
                    pos
                } else {
                    pos - 1
                }
            };

            // Need back_rows rows before the crossing plus steps for the crossing embedding.
            if cross_row_idx < cli.back_rows {
                n_skipped += 1;
                continue;
            }

            // Extract the pre-crossing window of FGM rows.
            let window_start = cross_row_idx - cli.back_rows;
            let window_end = cross_row_idx; // exclusive: up to but not including the crossing row
            let window = &rows[window_start..window_end];

            // Build embeddings from this window.
            let embedded = build_embeddings(window, cli.steps, cli.embed_dim);

            // We need at least precursor_window + 2 embeddings to compute the norms.
            let n_emb = embedded.len();
            if n_emb < cli.precursor_window + 2 {
                n_skipped += 1;
                continue;
            }

            // Compute associator norms on the pre-crossing embeddings.
            let norms = batch_sliding_associator_norms_parallel(&embedded, cli.embed_dim);
            let n_norms = norms.len();

            if n_norms < cli.precursor_window {
                n_skipped += 1;
                continue;
            }

            // Extract the last precursor_window norms (closest to the crossing).
            let precursor_start = n_norms - cli.precursor_window;
            let early = &norms[precursor_start..precursor_start + half];
            let late = &norms[precursor_start + half..precursor_start + cli.precursor_window];

            let early_mean = early.iter().sum::<f64>() / half as f64;
            let late_mean = late.iter().sum::<f64>() / half as f64;

            if early_mean < 1e-20 {
                n_skipped += 1;
                continue;
            }

            all_ratios.push(late_mean / early_mean);
        }
    }

    let n_analyzed = all_ratios.len();
    if n_analyzed == 0 {
        anyhow::bail!("No crossings could be analyzed (all skipped or insufficient data)");
    }

    // -----------------------------------------------------------------------
    // 3. Compute summary statistics
    // -----------------------------------------------------------------------
    all_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let ratio_mean = all_ratios.iter().sum::<f64>() / n_analyzed as f64;
    let ratio_median = percentile_sorted(&all_ratios, 50.0);
    let ratio_p25 = percentile_sorted(&all_ratios, 25.0);
    let ratio_p75 = percentile_sorted(&all_ratios, 75.0);
    let frac_above_1_0 = all_ratios.iter().filter(|&&r| r > 1.0).count() as f64 / n_analyzed as f64;
    let frac_above_1_1 = all_ratios.iter().filter(|&&r| r > 1.1).count() as f64 / n_analyzed as f64;
    let frac_above_1_5 = all_ratios.iter().filter(|&&r| r > 1.5).count() as f64 / n_analyzed as f64;

    // Precursor confirmed: majority show buildup.
    let precursor_confirmed = frac_above_1_0 > 0.5;

    // -----------------------------------------------------------------------
    // 4. Report comparison table
    // -----------------------------------------------------------------------
    eprintln!("\n=== THEMIS Supervised CD Crossing Survey ===");
    eprintln!(
        "Crossings analyzed: {}  |  Skipped: {}",
        n_analyzed, n_skipped
    );
    eprintln!(
        "Early/late ratio: mean={:.4}  median={:.4}  IQR=[{:.4}, {:.4}]",
        ratio_mean, ratio_median, ratio_p25, ratio_p75
    );
    eprintln!(
        "Fraction ratio > 1.0: {:.1}%  > 1.1: {:.1}%  > 1.5: {:.1}%",
        frac_above_1_0 * 100.0,
        frac_above_1_1 * 100.0,
        frac_above_1_5 * 100.0
    );
    eprintln!(
        "Precursor confirmed (majority > 1.0): {}",
        precursor_confirmed
    );

    eprintln!("\n=== Cross-Domain Comparison ===");
    eprintln!("{:<45}  {:>10}  {:>10}", "Source", "Mean", "Median");
    eprintln!("{}", "-".repeat(70));
    eprintln!(
        "{:<45}  {:>10}  {:>10}",
        "LBM BGK D3Q19 (C-1617, synthetic)", "1.92x", "1.92x"
    );
    eprintln!(
        "{:<45}  {:>10}  {:>10}",
        "ACE 2004 solar wind (unsupervised, C-1618)",
        format!("{:.2}x", 4.30),
        format!("{:.2}x", 2.40)
    );
    eprintln!(
        "{:<45}  {:>10}  {:>10}",
        "THEMIS labeled crossings (supervised)",
        format!("{:.2}x", ratio_mean),
        format!("{:.2}x", ratio_median)
    );

    let interpretation = if precursor_confirmed && ratio_median > 1.0 {
        format!(
            concat!(
                "CONFIRMED: Supervised THEMIS crossings show CD associator buildup ",
                "({:.2}x mean, {:.2}x median, {:.1}% of crossings > 1.0 ratio). ",
                "Consistent with LBM C-1617 (1.92x) and ACE C-1618 (4.30x mean). ",
                "Regime boundary established: precursor is a general property of ",
                "continuous-dynamics systems at magnetopause discontinuities."
            ),
            ratio_mean,
            ratio_median,
            frac_above_1_0 * 100.0
        )
    } else {
        format!(
            concat!(
                "INCONCLUSIVE: ratio_mean={:.2}x, {:.1}% of crossings > 1.0. ",
                "May require longer precursor windows or different embedding cadence."
            ),
            ratio_mean,
            frac_above_1_0 * 100.0
        )
    };
    eprintln!("\n{}", interpretation);

    // -----------------------------------------------------------------------
    // 5. Write JSON output
    // -----------------------------------------------------------------------
    let output = SurveyOutput {
        lbm_reference_ratio: 1.92,
        ace_unsupervised_mean_ratio: 4.30,
        ace_unsupervised_median_ratio: 2.40,
        themis_supervised: CrossingResult {
            n_crossings_analyzed: n_analyzed,
            n_crossings_skipped: n_skipped,
            early_late_ratio_mean: ratio_mean,
            early_late_ratio_median: ratio_median,
            early_late_ratio_p25: ratio_p25,
            early_late_ratio_p75: ratio_p75,
            fraction_above_1_0: frac_above_1_0,
            fraction_above_1_1: frac_above_1_1,
            fraction_above_1_5: frac_above_1_5,
            precursor_confirmed,
        },
        interpretation,
    };

    if let Some(parent) = cli.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    eprintln!("\nWrote: {}", cli.out_json.display());

    Ok(())
}
