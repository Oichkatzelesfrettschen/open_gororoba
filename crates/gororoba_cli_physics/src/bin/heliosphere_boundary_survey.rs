//! Unified boundary survey: THEMIS, ARTEMIS, Cluster, Swarm, MAVEN.
//!
//! A single binary that runs the standard 32D CD associator boundary detection
//! pipeline against any of the supported missions. The analysis logic is
//! mission-agnostic: fetch minute-level FGM data, detect |B| gradient
//! boundaries, run 32D Takens embedding + CD associator, cross-compare.

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-boundary-survey",
    about = "Multi-mission boundary survey with 32D CD associator"
)]
struct Cli {
    /// Mission: themis-a, themis-b (artemis), themis-c (artemis),
    /// themis-d, themis-e, cluster-1..4, swarm-a, maven
    #[arg(long)]
    mission: String,

    /// Start date (YYYY-MM-DD).
    #[arg(long)]
    start_date: String,

    /// End date (YYYY-MM-DD).
    #[arg(long)]
    end_date: String,

    /// Embedding dimension.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// |B| gradient threshold (nT) for boundary detection.
    #[arg(long, default_value_t = 5.0)]
    bmag_gradient_threshold: f64,

    /// Window size (minutes) for boundary detection.
    #[arg(long, default_value_t = 10)]
    boundary_window_minutes: usize,

    /// Match tolerance (minutes) for cross-comparison.
    #[arg(long, default_value_t = 20)]
    match_tolerance_minutes: usize,

    /// Maximum |B| filter (nT). Records above this are excluded.
    #[arg(long, default_value_t = 10000.0)]
    max_bmag: f64,

    /// Curated crossing list file (TSV or space-delimited timestamps).
    /// If provided, crossings from this list replace |B|-gradient detection.
    /// Supports THEMIS V2 format (TIMESTAMP\tX\tY\tZ\t...\tPROBE) and
    /// MESSENGER format (start_time end_time per line).
    #[arg(long)]
    crossing_list: Option<PathBuf>,

    /// Probe filter for crossing list (e.g., "a" for THEMIS-A).
    #[arg(long)]
    crossing_probe: Option<String>,

    /// Output JSON path.
    #[arg(long, default_value = "data/output/heliosphere/ablations/boundary_survey.json")]
    out_json: PathBuf,

    /// Data cache directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
}

/// Generic minute-level magnetic field record (mission-agnostic).
struct MagMinute {
    year: u16,
    doy: u16,
    hour: u8,
    minute: u8,
    elapsed_hours: f64,
    bx: f64,
    by: f64,
    bz: f64,
    b_magnitude: f64,
}

#[derive(Debug, Serialize)]
struct SurveyResults {
    mission: String,
    start_date: String,
    end_date: String,
    embedding_dim: usize,
    total_minutes: usize,
    n_boundaries: usize,
    n_associator_transitions: usize,
    detection_rate: f64,
    false_alarm_rate: f64,
    mean_offset_minutes: f64,
    median_offset_minutes: f64,
    /// Curated crossing list results (if --crossing-list provided).
    curated_n_crossings: Option<usize>,
    curated_detection_rate: Option<f64>,
    curated_false_alarm_rate: Option<f64>,
    curated_mean_offset: Option<f64>,
    curated_median_offset: Option<f64>,
    global_mean_associator: f64,
    global_std_associator: f64,
    transition_threshold: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad start_date: {}", cli.start_date))?;
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")
        .with_context(|| format!("bad end_date: {}", cli.end_date))?;

    println!(
        "=== Boundary Survey: {} ===\n  {} to {}\n  {}D, |B| grad={:.1} nT\n",
        cli.mission, start, end, cli.embedding_dim, cli.bmag_gradient_threshold,
    );

    // --- Step 1: Fetch + parse to generic MagMinute ---
    println!("[1/4] Fetching {} data...", cli.mission);

    let all_minutes = fetch_and_parse(&cli, start, end)?;

    if all_minutes.is_empty() {
        anyhow::bail!("No data for {} in requested window.", cli.mission);
    }

    println!("  Total: {} minutes across {:.1} hours",
        all_minutes.len(),
        all_minutes.last().map(|r| r.elapsed_hours).unwrap_or(0.0));

    // --- Step 2: Detect boundaries ---
    println!("[2/4] Detecting |B| gradient boundaries...");

    let half = cli.boundary_window_minutes;
    let mut boundaries: Vec<usize> = Vec::new();

    if all_minutes.len() > half * 2 + 1 {
        for i in half..all_minutes.len().saturating_sub(half) {
            let pre: f64 = all_minutes[i.saturating_sub(half)..i]
                .iter().map(|r| r.b_magnitude).sum::<f64>() / half as f64;
            let post: f64 = all_minutes[i..(i + half).min(all_minutes.len())]
                .iter().map(|r| r.b_magnitude).sum::<f64>()
                / half.min(all_minutes.len() - i) as f64;
            let jump = (post - pre).abs();
            if jump > cli.bmag_gradient_threshold {
                let dominated = boundaries.last()
                    .is_some_and(|&prev| i.saturating_sub(prev) < cli.boundary_window_minutes);
                if !dominated { boundaries.push(i); }
            }
        }
    }

    println!("  Found {} boundaries", boundaries.len());

    // --- Step 3: 32D CD associator ---
    println!("[3/4] Computing {}D Takens embedding + CD associator...", cli.embedding_dim);

    let channels: usize = 4;
    let steps = cli.embedding_dim / channels;
    let window_rows = (steps - 1) + 1;

    let mut embedded: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();

    for w_start in 0..=all_minutes.len().saturating_sub(window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s).collect();
        let sum_b: f64 = sample_indices.iter().map(|&i| all_minutes[i].b_magnitude).sum();
        let mean_b = sum_b / steps as f64;
        if mean_b <= 0.0 || !mean_b.is_finite() { continue; }

        let mut v = vec![0.0; cli.embedding_dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let rec = &all_minutes[ri];
            v[s * channels] = rec.bx / mean_b;
            v[s * channels + 1] = rec.by / mean_b;
            v[s * channels + 2] = rec.bz / mean_b;
            v[s * channels + 3] = (rec.b_magnitude - mean_b) / mean_b;
        }
        embedded.push(v);
        embed_meta.push(*sample_indices.last().unwrap());
    }

    println!("  Embedded {} vectors", embedded.len());

    if embedded.len() < 3 {
        anyhow::bail!("Too few embedded vectors ({}) for associator", embedded.len());
    }

    let associators = cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);
    println!("  Computed {} associator norms", associators.len());

    let assoc_indices: Vec<usize> = (0..associators.len()).map(|k| embed_meta[k + 2]).collect();

    // Detect transitions
    let trans_window = cli.boundary_window_minutes.max(5);
    let global_mean: f64 = associators.iter().sum::<f64>() / associators.len() as f64;
    let global_std: f64 = {
        let var = associators.iter().map(|&a| (a - global_mean).powi(2)).sum::<f64>()
            / associators.len() as f64;
        var.sqrt()
    };
    let threshold = global_std * 1.5;

    println!("  Stats: mean={:.4}, std={:.4}, threshold={:.4}", global_mean, global_std, threshold);

    let mut transitions: Vec<usize> = Vec::new();
    if associators.len() > trans_window * 2 {
        for i in trans_window..associators.len().saturating_sub(trans_window) {
            let pre: f64 = associators[i.saturating_sub(trans_window)..i].iter().sum::<f64>()
                / trans_window as f64;
            let post: f64 = associators[i..(i + trans_window).min(associators.len())]
                .iter().sum::<f64>() / trans_window.min(associators.len() - i) as f64;
            if (post - pre).abs() > threshold {
                let dominated = transitions.last()
                    .is_some_and(|&prev| i.saturating_sub(prev) < trans_window);
                if !dominated { transitions.push(assoc_indices[i]); }
            }
        }
    }

    println!("  Found {} transitions", transitions.len());

    // --- Step 4: Cross-compare ---
    println!("[4/4] Cross-comparing...");

    let tol_hours = cli.match_tolerance_minutes as f64 / 60.0;
    let mut matched = 0usize;
    let mut offsets: Vec<f64> = Vec::new();

    for &bi in &boundaries {
        let b_h = all_minutes[bi].elapsed_hours;
        let hit = transitions.iter().any(|&ti| {
            (all_minutes[ti].elapsed_hours - b_h).abs() < tol_hours
        });
        if hit {
            matched += 1;
            let best_offset = transitions.iter()
                .map(|&ti| (all_minutes[ti].elapsed_hours - b_h).abs() * 60.0)
                .fold(f64::MAX, f64::min);
            offsets.push(best_offset);
        }
    }

    let matched_trans = transitions.iter()
        .filter(|&&ti| {
            let t_h = all_minutes[ti].elapsed_hours;
            boundaries.iter().any(|&bi| (all_minutes[bi].elapsed_hours - t_h).abs() < tol_hours)
        })
        .count();

    let det_rate = if boundaries.is_empty() { 0.0 } else { matched as f64 / boundaries.len() as f64 };
    let fa_rate = if transitions.is_empty() { 0.0 }
        else { (transitions.len() - matched_trans) as f64 / transitions.len() as f64 };

    offsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_off = if offsets.is_empty() { 0.0 } else { offsets.iter().sum::<f64>() / offsets.len() as f64 };
    let med_off = if offsets.is_empty() { 0.0 } else { offsets[offsets.len() / 2] };

    println!("\n=== {} Results (|B|-gradient) ===", cli.mission);
    println!("  Boundaries: {}, Transitions: {}", boundaries.len(), transitions.len());
    println!("  Detection rate: {}/{} = {:.1}%", matched, boundaries.len(), det_rate * 100.0);
    println!("  False alarm rate: {:.1}%", fa_rate * 100.0);
    println!("  Mean offset: {:.1} min, Median: {:.1} min", mean_off, med_off);

    // --- Curated crossing list comparison (if provided) ---
    let (cur_n, cur_det, cur_fa, cur_mean, cur_med) =
        if let Some(ref list_path) = cli.crossing_list {
            println!("\n--- Curated crossing list comparison ---");
            let curated_hours = parse_curated_crossings(
                list_path,
                cli.crossing_probe.as_deref(),
                &start,
                &end,
            )?;
            println!("  Curated crossings in window: {}", curated_hours.len());

            if curated_hours.is_empty() {
                (Some(0), Some(0.0), Some(0.0), Some(0.0), Some(0.0))
            } else {
                // Compare: for each curated crossing, is there a transition nearby?
                let mut c_matched = 0usize;
                let mut c_offsets: Vec<f64> = Vec::new();

                // First record's elapsed_hours is 0 at start of data
                let data_start_hours = all_minutes.first().map(|r| r.elapsed_hours).unwrap_or(0.0);
                let _ = data_start_hours; // used below in curated_hours mapping

                for &ch in &curated_hours {
                    let hit = transitions.iter().any(|&ti| {
                        (all_minutes[ti].elapsed_hours - ch).abs() < tol_hours
                    });
                    if hit {
                        c_matched += 1;
                        let best = transitions.iter()
                            .map(|&ti| (all_minutes[ti].elapsed_hours - ch).abs() * 60.0)
                            .fold(f64::MAX, f64::min);
                        c_offsets.push(best);
                    }
                }

                let c_matched_trans = transitions.iter()
                    .filter(|&&ti| {
                        let th = all_minutes[ti].elapsed_hours;
                        curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours)
                    })
                    .count();

                let c_det = c_matched as f64 / curated_hours.len() as f64;
                let c_fa = if transitions.is_empty() { 0.0 }
                    else { (transitions.len() - c_matched_trans) as f64 / transitions.len() as f64 };

                c_offsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let c_mean = if c_offsets.is_empty() { 0.0 }
                    else { c_offsets.iter().sum::<f64>() / c_offsets.len() as f64 };
                let c_med = if c_offsets.is_empty() { 0.0 }
                    else { c_offsets[c_offsets.len() / 2] };

                println!("  Curated detection rate: {}/{} = {:.1}%",
                    c_matched, curated_hours.len(), c_det * 100.0);
                println!("  Curated false alarm rate: {:.1}%", c_fa * 100.0);
                println!("  Curated mean offset: {:.1} min, median: {:.1} min", c_mean, c_med);

                (Some(curated_hours.len()), Some(c_det), Some(c_fa), Some(c_mean), Some(c_med))
            }
        } else {
            (None, None, None, None, None)
        };

    let results = SurveyResults {
        mission: cli.mission.clone(),
        start_date: cli.start_date.clone(),
        end_date: cli.end_date.clone(),
        embedding_dim: cli.embedding_dim,
        total_minutes: all_minutes.len(),
        n_boundaries: boundaries.len(),
        n_associator_transitions: transitions.len(),
        detection_rate: det_rate,
        false_alarm_rate: fa_rate,
        mean_offset_minutes: mean_off,
        median_offset_minutes: med_off,
        curated_n_crossings: cur_n,
        curated_detection_rate: cur_det,
        curated_false_alarm_rate: cur_fa,
        curated_mean_offset: cur_mean,
        curated_median_offset: cur_med,
        global_mean_associator: global_mean,
        global_std_associator: global_std,
        transition_threshold: threshold,
    };

    if let Some(parent) = cli.out_json.parent() { fs::create_dir_all(parent)?; }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&results)?)?;
    println!("\nWrote {}", cli.out_json.display());

    Ok(())
}

/// Dispatch to mission-specific fetcher and normalize to MagMinute.
fn fetch_and_parse(cli: &Cli, start: NaiveDate, end: NaiveDate) -> Result<Vec<MagMinute>> {
    let mission = cli.mission.to_lowercase();

    if mission.starts_with("themis-") || mission.starts_with("th") {
        fetch_themis(cli, &mission, start, end)
    } else if mission.starts_with("cluster-") {
        fetch_cluster(cli, &mission, start, end)
    } else if mission == "swarm-a" || mission == "swarm" {
        fetch_swarm(cli, start, end)
    } else if mission == "maven" {
        fetch_maven(cli, start, end)
    } else {
        anyhow::bail!("Unknown mission: {}. Use themis-a..e, cluster-1..4, swarm-a, maven", cli.mission);
    }
}

fn fetch_themis(cli: &Cli, mission: &str, start: NaiveDate, end: NaiveDate) -> Result<Vec<MagMinute>> {
    use data_core::catalogs::themis::{ThemisFgmProvider, parse_themis_fgm_hapi_csv_minutes};
    use data_core::fetcher::{DatasetProvider, FetchConfig};

    let probe = match mission {
        "themis-a" | "tha" => "THA",
        "themis-b" | "thb" | "artemis-p1" => "THB",
        "themis-c" | "thc" | "artemis-p2" => "THC",
        "themis-d" | "thd" => "THD",
        "themis-e" | "the" => "THE",
        other => anyhow::bail!("Unknown THEMIS probe: {other}"),
    };

    let config = FetchConfig { output_dir: cli.data_dir.clone(), skip_existing: true, verify_checksums: false };

    for doy_offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(doy_offset);
        let provider = ThemisFgmProvider {
            probe: probe.to_string(),
            year: date.year() as u16,
            doy_start: date.ordinal() as u16,
            doy_end: date.ordinal() as u16,
        };
        let _ = provider.fetch(&config);
    }

    let data_dir = cli.data_dir.join("themis");
    let mut all = Vec::new();
    for doy_offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(doy_offset);
        let fname = format!("{}_fgm_{:04}_{:03}.csv", probe.to_lowercase(), date.year(), date.ordinal());
        let path = data_dir.join(&fname);
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let records = parse_themis_fgm_hapi_csv_minutes(&content, probe);
            println!("  {} DOY {}: {} min", probe, date.ordinal(), records.len());
            for r in records {
                if r.b_magnitude <= cli.max_bmag {
                    all.push(MagMinute {
                        year: r.year, doy: r.doy, hour: r.hour, minute: r.minute,
                        elapsed_hours: r.elapsed_hours,
                        bx: r.bx_gse, by: r.by_gse, bz: r.bz_gse, b_magnitude: r.b_magnitude,
                    });
                }
            }
        }
    }
    recompute_elapsed(&mut all);
    Ok(all)
}

fn fetch_cluster(cli: &Cli, mission: &str, start: NaiveDate, end: NaiveDate) -> Result<Vec<MagMinute>> {
    use data_core::catalogs::cluster::{ClusterFgmProvider, parse_cluster_fgm_hapi_csv_minutes};
    use data_core::fetcher::{DatasetProvider, FetchConfig};

    let probe_id: u8 = mission.strip_prefix("cluster-")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let config = FetchConfig { output_dir: cli.data_dir.clone(), skip_existing: true, verify_checksums: false };

    for doy_offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(doy_offset);
        let provider = ClusterFgmProvider {
            probe_id,
            year: date.year() as u16,
            doy_start: date.ordinal() as u16,
            doy_end: date.ordinal() as u16,
        };
        let _ = provider.fetch(&config);
    }

    let data_dir = cli.data_dir.join("cluster");
    let mut all = Vec::new();
    for doy_offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(doy_offset);
        let fname = format!("c{}_fgm_spin_{:04}_{:03}.csv", probe_id, date.year(), date.ordinal());
        let path = data_dir.join(&fname);
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let records = parse_cluster_fgm_hapi_csv_minutes(&content, probe_id);
            println!("  C{} DOY {}: {} min", probe_id, date.ordinal(), records.len());
            for r in records {
                if r.b_magnitude <= cli.max_bmag {
                    all.push(MagMinute {
                        year: r.year, doy: r.doy, hour: r.hour, minute: r.minute,
                        elapsed_hours: r.elapsed_hours,
                        bx: r.bx_gse, by: r.by_gse, bz: r.bz_gse, b_magnitude: r.b_magnitude,
                    });
                }
            }
        }
    }
    recompute_elapsed(&mut all);
    Ok(all)
}

fn fetch_swarm(cli: &Cli, start: NaiveDate, end: NaiveDate) -> Result<Vec<MagMinute>> {
    use data_core::catalogs::swarm_mag::{SwarmMagProvider, parse_swarm_amda_mag_csv_minutes};
    use data_core::fetcher::{DatasetProvider, FetchConfig};

    let config = FetchConfig { output_dir: cli.data_dir.clone(), skip_existing: true, verify_checksums: false };

    // Fetch monthly
    let provider = SwarmMagProvider {
        year: start.year() as u16,
        month_start: start.month() as u8,
        month_end: end.month() as u8,
    };
    let data_dir = provider.fetch(&config)?;

    let mut all = Vec::new();
    for month in start.month()..=end.month() {
        let fname = format!("swarma_mag_{:04}_{:02}.csv", start.year(), month);
        let path = data_dir.join(&fname);
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let records = parse_swarm_amda_mag_csv_minutes(&content);
            println!("  Swarm-A {:04}-{:02}: {} min", start.year(), month, records.len());
            for r in records {
                if r.b_magnitude <= cli.max_bmag {
                    all.push(MagMinute {
                        year: r.year, doy: r.doy, hour: r.hour, minute: r.minute,
                        elapsed_hours: r.elapsed_hours,
                        bx: r.b_north, by: r.b_east, bz: r.b_center, b_magnitude: r.b_magnitude,
                    });
                }
            }
        }
    }
    recompute_elapsed(&mut all);
    Ok(all)
}

fn fetch_maven(cli: &Cli, start: NaiveDate, end: NaiveDate) -> Result<Vec<MagMinute>> {
    use data_core::catalogs::maven_mag::{MavenMagProvider, parse_maven_mag_hapi_csv_minutes};
    use data_core::fetcher::{DatasetProvider, FetchConfig};

    let config = FetchConfig { output_dir: cli.data_dir.clone(), skip_existing: true, verify_checksums: false };

    for doy_offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(doy_offset);
        let provider = MavenMagProvider {
            year: date.year() as u16,
            doy_start: date.ordinal() as u16,
            doy_end: date.ordinal() as u16,
        };
        let _ = provider.fetch(&config);
    }

    let data_dir = cli.data_dir.join("maven");
    let mut all = Vec::new();
    for doy_offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(doy_offset);
        let fname = format!("maven_mag_{:04}_{:03}.csv", date.year(), date.ordinal());
        let path = data_dir.join(&fname);
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let records = parse_maven_mag_hapi_csv_minutes(&content);
            println!("  MAVEN DOY {}: {} min", date.ordinal(), records.len());
            for r in records {
                if r.b_magnitude <= cli.max_bmag {
                    all.push(MagMinute {
                        year: r.year, doy: r.doy, hour: r.hour, minute: r.minute,
                        elapsed_hours: r.elapsed_hours,
                        bx: r.bx_ss, by: r.by_ss, bz: r.bz_ss, b_magnitude: r.b_magnitude,
                    });
                }
            }
        }
    }
    recompute_elapsed(&mut all);
    Ok(all)
}

/// Parse a curated crossing list file into elapsed-hours relative to `start`.
///
/// Supports two formats:
/// - THEMIS V2: tab-separated TIMESTAMP X Y Z ... PROBE (header + comment lines)
/// - MESSENGER: space-separated start_time end_time per line (no header)
fn parse_curated_crossings(
    path: &PathBuf,
    probe_filter: Option<&str>,
    start: &NaiveDate,
    end: &NaiveDate,
) -> Result<Vec<f64>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("read crossing list: {}", path.display()))?;

    let start_dt = start.and_hms_opt(0, 0, 0).unwrap();
    let end_dt = end.and_hms_opt(23, 59, 59).unwrap();

    let mut hours: Vec<f64> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("TIMESTAMP") {
            continue;
        }

        // Try THEMIS format: TIMESTAMP\tX\tY\tZ\t...\tPROBE
        let fields: Vec<&str> = trimmed.split('\t').collect();
        if fields.len() >= 2 {
            if let Some(filter) = probe_filter
                && fields.last().is_some_and(|pf| pf.trim().to_lowercase() != filter.to_lowercase())
            {
                continue;
            }
            if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(
                fields[0].trim().trim_end_matches('Z'),
                "%Y-%m-%dT%H:%M:%S",
            ) && dt >= start_dt && dt <= end_dt {
                let elapsed = (dt - start_dt).num_minutes() as f64 / 60.0;
                hours.push(elapsed);
            }
            continue;
        }

        // Try MESSENGER format: start_time end_time (space-separated)
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if !parts.is_empty() {
            let ts = parts[0].trim_end_matches('Z');
            if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%d/%H:%M:%S")
                && dt >= start_dt && dt <= end_dt
            {
                let elapsed = (dt - start_dt).num_minutes() as f64 / 60.0;
                hours.push(elapsed);
            }
        }
    }

    hours.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(hours)
}

fn recompute_elapsed(records: &mut [MagMinute]) {
    records.sort_by(|a, b| {
        a.year.cmp(&b.year).then(a.doy.cmp(&b.doy)).then(a.hour.cmp(&b.hour)).then(a.minute.cmp(&b.minute))
    });
    if records.is_empty() { return; }
    let (fy, fd, fh, fm) = (records[0].year, records[0].doy, records[0].hour, records[0].minute);
    for r in records.iter_mut() {
        r.elapsed_hours = (r.year as f64 - fy as f64) * 365.25 * 24.0
            + (r.doy as f64 - fd as f64) * 24.0
            + (r.hour as f64 - fh as f64)
            + (r.minute as f64 - fm as f64) / 60.0;
    }
}
