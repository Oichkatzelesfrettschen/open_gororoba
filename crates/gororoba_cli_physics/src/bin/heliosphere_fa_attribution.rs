//! False-alarm attribution: classify CD extra transitions by B-field signature.
//!
//! For each CD transition NOT in the curated crossing list, classifies it as:
//! - TD (tangential discontinuity): rotation > 30 deg, |B| change < 3 nT
//! - FTE candidate: bipolar B_N change > 5 nT, |B| ratio > 1.3
//! - Mirror mode: |B| CV > 0.3, rotation < 15 deg
//! - Partial crossing: |B| jump > 10 nT and rotation > 20 deg
//! - Compressive: |B| CV > 0.5
//! - Unclassified: none of the above

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Parser;
use data_core::{
    catalogs::{
        cluster::parse_cluster_fgm_hapi_csv_minutes,
        maven_mag::parse_maven_mag_hapi_csv_minutes,
        themis::parse_themis_fgm_hapi_csv_minutes,
    },
    crossing_lists::parse_crossing_list,
};
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "heliosphere-fa-attribution")]
struct Cli {
    #[arg(long)]
    start_date: String,
    #[arg(long)]
    end_date: String,
    /// Mission to analyze: "themis" (default), "cluster", or "maven".
    #[arg(long, default_value = "themis")]
    mission: String,
    /// THEMIS probe letter (a-e) or Cluster probe number (1-4).
    #[arg(long)]
    crossing_probe: Option<String>,
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,
    #[arg(long, default_value_t = 200.0)]
    max_bmag: f64,
    #[arg(long)]
    crossing_list: PathBuf,
    #[arg(long, default_value_t = 20)]
    match_tolerance_minutes: usize,
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/fa_attribution.json"
    )]
    out_json: PathBuf,
    #[arg(long, default_value = "data/external")]
    data_dir: String,
    /// If set, write a precision/recall/F1 curve by sweeping the detection threshold.
    #[arg(long)]
    out_pr_curve: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct Attribution {
    index: usize,
    doy: u16,
    hour: u8,
    minute: u8,
    associator_norm: f64,
    rotation_deg: f64,
    b_jump_nt: f64,
    cv_bmag: f64,
    bn_change: f64,
    event_type: String,
}

#[derive(Debug, Serialize)]
struct TypeStats {
    count: usize,
    fraction: f64,
    mean_rotation_deg: f64,
    mean_b_jump_nt: f64,
    mean_cv_bmag: f64,
    mean_associator_norm: f64,
    std_rotation_deg: f64,
    std_b_jump_nt: f64,
}

#[derive(Debug, Serialize)]
struct PrPoint {
    threshold_factor: f64,
    n_detected: usize,
    tp: usize,
    fp: usize,
    fn_count: usize,
    precision: f64,
    recall: f64,
    f1: f64,
}

#[derive(Debug, Serialize)]
struct AttributionResult {
    start_date: String,
    end_date: String,
    mission: String,
    n_cd_transitions: usize,
    n_curated_matched: usize,
    n_curated_crossings: usize,
    n_fn: usize,
    n_extras: usize,
    n_attributed: usize,
    attribution_fraction: f64,
    precision: f64,
    recall: f64,
    f1: f64,
    type_counts: BTreeMap<String, usize>,
    type_stats: BTreeMap<String, TypeStats>,
    events: Vec<Attribution>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad start: {}", cli.start_date))?;
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")
        .with_context(|| format!("bad end: {}", cli.end_date))?;

    // Load FGM for the selected mission.
    let mut bx = Vec::new();
    let mut by = Vec::new();
    let mut bz = Vec::new();
    let mut keys: Vec<(u16, u8, u8)> = Vec::new();
    let mut elapsed: Vec<f64> = Vec::new();

    match cli.mission.to_lowercase().as_str() {
        "cluster" => {
            let probe_id: u8 = cli
                .crossing_probe
                .as_deref()
                .unwrap_or("1")
                .parse()
                .unwrap_or(1)
                .clamp(1, 4);
            for offset in 0..=(end - start).num_days() {
                let date = start + chrono::Duration::days(offset);
                let path = format!(
                    "{}/cluster/c{}_fgm_spin_{:04}_{:03}.csv",
                    cli.data_dir,
                    probe_id,
                    date.year(),
                    date.ordinal()
                );
                if let Ok(content) = fs::read_to_string(&path) {
                    for r in parse_cluster_fgm_hapi_csv_minutes(&content, probe_id) {
                        if r.b_magnitude <= cli.max_bmag {
                            let day_diff = (r.doy as f64 - start.ordinal() as f64)
                                + (r.year as f64 - start.year() as f64) * 365.25;
                            let eh = day_diff * 24.0 + r.hour as f64 + r.minute as f64 / 60.0;
                            bx.push(r.bx_gse);
                            by.push(r.by_gse);
                            bz.push(r.bz_gse);
                            keys.push((r.doy, r.hour, r.minute));
                            elapsed.push(eh);
                        }
                    }
                }
            }
        }
        "maven" => {
            // MAVEN MAG (SS frame). Files: maven/maven_mag_{year}_{doy}.csv
            for offset in 0..=(end - start).num_days() {
                let date = start + chrono::Duration::days(offset);
                let path = format!(
                    "{}/maven/maven_mag_{:04}_{:03}.csv",
                    cli.data_dir,
                    date.year(),
                    date.ordinal()
                );
                if let Ok(content) = fs::read_to_string(&path) {
                    for r in parse_maven_mag_hapi_csv_minutes(&content) {
                        if r.b_magnitude <= cli.max_bmag {
                            let day_diff = (r.doy as f64 - start.ordinal() as f64)
                                + (r.year as f64 - start.year() as f64) * 365.25;
                            let eh = day_diff * 24.0 + r.hour as f64 + r.minute as f64 / 60.0;
                            // MAVEN MAG reports in SS (Sun-State) frame (bx_ss = toward Sun).
                            bx.push(r.bx_ss);
                            by.push(r.by_ss);
                            bz.push(r.bz_ss);
                            keys.push((r.doy, r.hour, r.minute));
                            elapsed.push(eh);
                        }
                    }
                }
            }
        }
        _ => {
            // Default: THEMIS
            let probe_upper = format!(
                "TH{}",
                cli.crossing_probe.as_deref().unwrap_or("A").to_uppercase()
            );
            for offset in 0..=(end - start).num_days() {
                let date = start + chrono::Duration::days(offset);
                let path = format!(
                    "{}/themis/{}_fgm_{:04}_{:03}.csv",
                    cli.data_dir,
                    probe_upper.to_lowercase(),
                    date.year(),
                    date.ordinal()
                );
                if let Ok(content) = fs::read_to_string(&path) {
                    for r in parse_themis_fgm_hapi_csv_minutes(&content, &probe_upper) {
                        if r.b_magnitude <= cli.max_bmag {
                            let day_diff = (r.doy as f64 - start.ordinal() as f64)
                                + (r.year as f64 - start.year() as f64) * 365.25;
                            let eh = day_diff * 24.0 + r.hour as f64 + r.minute as f64 / 60.0;
                            bx.push(r.bx_gse);
                            by.push(r.by_gse);
                            bz.push(r.bz_gse);
                            keys.push((r.doy, r.hour, r.minute));
                            elapsed.push(eh);
                        }
                    }
                }
            }
        }
    }

    let n = bx.len();
    println!("Loaded {} minutes", n);

    // Build 32D Takens embedding + CD associator
    let channels = 4usize;
    let steps = cli.embedding_dim / channels;
    let mut embedded: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();

    for w in 0..=n.saturating_sub(steps) {
        let sum_b: f64 = (0..steps)
            .map(|s| (bx[w + s].powi(2) + by[w + s].powi(2) + bz[w + s].powi(2)).sqrt())
            .sum();
        let mean_b = sum_b / steps as f64;
        if mean_b <= 0.01 || !mean_b.is_finite() {
            continue;
        }
        let mut v = vec![0.0; cli.embedding_dim];
        for s in 0..steps {
            let i = w + s;
            let bmag = (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt();
            v[s * channels] = bx[i] / mean_b;
            v[s * channels + 1] = by[i] / mean_b;
            v[s * channels + 2] = bz[i] / mean_b;
            v[s * channels + 3] = (bmag - mean_b) / mean_b;
        }
        embedded.push(v);
        embed_meta.push(w + steps - 1);
    }

    let norms = cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);
    let assoc_idx: Vec<usize> = (0..norms.len()).map(|k| embed_meta[k + 2]).collect();
    println!("Computed {} associator norms", norms.len());

    // Detect transitions
    let global_mean: f64 = norms.iter().sum::<f64>() / norms.len() as f64;
    let global_std: f64 = {
        let var = norms
            .iter()
            .map(|&a| (a - global_mean).powi(2))
            .sum::<f64>()
            / norms.len() as f64;
        var.sqrt()
    };
    let tw = 10usize;

    // Pre-compute (assoc_index, |post_mean - pre_mean|) for every position.
    // Used both for the main detection run (threshold=1.5) and the PR curve sweep.
    let mut all_changes: Vec<(usize, f64)> = Vec::new(); // (assoc_idx, change_magnitude)
    if norms.len() > tw * 2 {
        for i in tw..norms.len().saturating_sub(tw) {
            let pre: f64 = norms[i.saturating_sub(tw)..i].iter().sum::<f64>() / tw as f64;
            let n_post = tw.min(norms.len() - i);
            let post: f64 = norms[i..(i + n_post)].iter().sum::<f64>() / n_post as f64;
            let change = (post - pre).abs();
            all_changes.push((assoc_idx[i], change));
        }
    }

    // Helper: apply a threshold multiplier and collect (minute_index, norm) pairs.
    let detect_at = |factor: f64| -> Vec<(usize, f64)> {
        let thresh = global_std * factor;
        let mut out: Vec<(usize, f64)> = Vec::new();
        let mut last: Option<usize> = None;
        for (pos, &(mi, change)) in all_changes.iter().enumerate() {
            if change > thresh {
                let norm_pos = pos + tw; // position in norms array (offset by tw)
                if last.is_some_and(|prev| pos.saturating_sub(prev) < tw) {
                    continue;
                }
                let ni = norm_pos.min(norms.len() - 1);
                out.push((mi, norms[ni]));
                last = Some(pos);
            }
        }
        out
    };

    let transitions = detect_at(1.5);
    println!("CD transitions: {}", transitions.len());

    // Load curated crossings
    let crossing_content = fs::read_to_string(&cli.crossing_list)?;
    let events = parse_crossing_list(
        &crossing_content,
        cli.crossing_probe.as_deref(),
        &start,
        &end,
    );
    let curated_hours: Vec<f64> = events.iter().map(|e| e.elapsed_hours).collect();
    let n_curated = curated_hours.len();
    println!("Curated crossings: {}", n_curated);

    // Helper: match transitions against curated crossings.
    // Returns (matched_count, extras, fn_count).
    let tol_hours = cli.match_tolerance_minutes as f64 / 60.0;
    let match_transitions = |trans: &[(usize, f64)]| -> (usize, Vec<(usize, f64)>, usize) {
        let mut matched = 0usize;
        let mut extra_out: Vec<(usize, f64)> = Vec::new();
        for &(mi, norm) in trans {
            let t_h = elapsed[mi.min(n - 1)];
            if curated_hours.iter().any(|&ch| (ch - t_h).abs() < tol_hours) {
                matched += 1;
            } else {
                extra_out.push((mi, norm));
            }
        }
        // FN: curated crossings not within tolerance of any detected transition.
        let fn_count = curated_hours
            .iter()
            .filter(|&&ch| {
                !trans.iter().any(|&(mi, _)| {
                    let t_h = elapsed[mi.min(n - 1)];
                    (ch - t_h).abs() < tol_hours
                })
            })
            .count();
        (matched, extra_out, fn_count)
    };

    let (matched_count, extras, fn_count) = match_transitions(&transitions);
    println!("Matched: {}, Extras: {}, FN: {}", matched_count, extras.len(), fn_count);

    // Classify each extra by local B-field signature
    let w = 10usize;
    let mut attributions: Vec<Attribution> = Vec::new();
    let mut type_counts: BTreeMap<String, usize> = BTreeMap::new();

    for &(mi, norm) in &extras {
        if mi < w || mi + w >= n {
            continue;
        }

        let pre_bx: f64 = bx[mi.saturating_sub(w)..mi].iter().sum::<f64>() / w as f64;
        let pre_by: f64 = by[mi.saturating_sub(w)..mi].iter().sum::<f64>() / w as f64;
        let pre_bz: f64 = bz[mi.saturating_sub(w)..mi].iter().sum::<f64>() / w as f64;
        let post_bx: f64 = bx[mi..mi + w].iter().sum::<f64>() / w as f64;
        let post_by: f64 = by[mi..mi + w].iter().sum::<f64>() / w as f64;
        let post_bz: f64 = bz[mi..mi + w].iter().sum::<f64>() / w as f64;

        let bmag_local: Vec<f64> = (mi.saturating_sub(w)..mi + w)
            .filter(|&i| i < n)
            .map(|i| (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt())
            .collect();
        let pre_b: f64 = bmag_local[..w.min(bmag_local.len())].iter().sum::<f64>()
            / w.min(bmag_local.len()) as f64;
        let post_b: f64 = bmag_local[w.min(bmag_local.len())..].iter().sum::<f64>()
            / (bmag_local.len() - w.min(bmag_local.len())).max(1) as f64;

        let pre_mag = (pre_bx.powi(2) + pre_by.powi(2) + pre_bz.powi(2)).sqrt();
        let post_mag = (post_bx.powi(2) + post_by.powi(2) + post_bz.powi(2)).sqrt();

        let rotation = if pre_mag > 0.1 && post_mag > 0.1 {
            let cos_a = ((pre_bx * post_bx + pre_by * post_by + pre_bz * post_bz)
                / (pre_mag * post_mag))
                .clamp(-1.0, 1.0);
            cos_a.acos().to_degrees()
        } else {
            0.0
        };

        let b_jump = (post_b - pre_b).abs();
        let mean_b = bmag_local.iter().sum::<f64>() / bmag_local.len() as f64;
        let var_b = bmag_local
            .iter()
            .map(|&b| (b - mean_b).powi(2))
            .sum::<f64>()
            / bmag_local.len() as f64;
        let cv = if mean_b > 0.1 {
            var_b.sqrt() / mean_b
        } else {
            0.0
        };
        let bn_change = (post_bz - pre_bz).abs();

        let event_type = if rotation > 30.0 && b_jump < 3.0 {
            "TD"
        } else if bn_change > 5.0 && post_b / pre_b.max(0.1) > 1.3 {
            "FTE_candidate"
        } else if cv > 0.3 && rotation < 15.0 {
            "Mirror_mode"
        } else if b_jump > 10.0 && rotation > 20.0 {
            "Partial_crossing"
        } else if cv > 0.5 {
            "Compressive"
        } else {
            "Unclassified"
        };

        *type_counts.entry(event_type.to_string()).or_insert(0) += 1;

        let (doy, hour, minute) = keys[mi.min(n - 1)];
        attributions.push(Attribution {
            index: mi,
            doy,
            hour,
            minute,
            associator_norm: norm,
            rotation_deg: (rotation * 10.0).round() / 10.0,
            b_jump_nt: (b_jump * 10.0).round() / 10.0,
            cv_bmag: (cv * 1000.0).round() / 1000.0,
            bn_change: (bn_change * 10.0).round() / 10.0,
            event_type: event_type.to_string(),
        });
    }

    let n_attributed = attributions.len() - type_counts.get("Unclassified").copied().unwrap_or(0);
    let frac = n_attributed as f64 / attributions.len().max(1) as f64;

    // Per-type statistics (mean + std of key observables).
    let mut type_stats: BTreeMap<String, TypeStats> = BTreeMap::new();
    for (typ, &cnt) in &type_counts {
        let events_of_type: Vec<&Attribution> =
            attributions.iter().filter(|a| &a.event_type == typ).collect();
        let n = events_of_type.len().max(1) as f64;
        let mean_rot = events_of_type.iter().map(|a| a.rotation_deg).sum::<f64>() / n;
        let mean_bj = events_of_type.iter().map(|a| a.b_jump_nt).sum::<f64>() / n;
        let mean_cv = events_of_type.iter().map(|a| a.cv_bmag).sum::<f64>() / n;
        let mean_an = events_of_type.iter().map(|a| a.associator_norm).sum::<f64>() / n;
        let std_rot = (events_of_type
            .iter()
            .map(|a| (a.rotation_deg - mean_rot).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let std_bj = (events_of_type
            .iter()
            .map(|a| (a.b_jump_nt - mean_bj).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let total = attributions.len().max(1);
        type_stats.insert(
            typ.clone(),
            TypeStats {
                count: cnt,
                fraction: cnt as f64 / total as f64,
                mean_rotation_deg: (mean_rot * 10.0).round() / 10.0,
                mean_b_jump_nt: (mean_bj * 10.0).round() / 10.0,
                mean_cv_bmag: (mean_cv * 1000.0).round() / 1000.0,
                mean_associator_norm: (mean_an * 100.0).round() / 100.0,
                std_rotation_deg: (std_rot * 10.0).round() / 10.0,
                std_b_jump_nt: (std_bj * 10.0).round() / 10.0,
            },
        );
    }

    // Overall precision / recall / F1 at the default threshold.
    let tp = matched_count;
    let fp = extras.len();
    let prec = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let rec = if n_curated > 0 { tp as f64 / n_curated as f64 } else { 0.0 };
    let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };

    println!("\n=== Attribution Results ===");
    println!("  Total extras: {}", attributions.len());
    for (typ, count) in &type_counts {
        println!(
            "  {}: {} ({:.1}%)",
            typ,
            count,
            *count as f64 / attributions.len().max(1) as f64 * 100.0
        );
    }
    println!(
        "  ATTRIBUTED: {}/{} = {:.1}%",
        n_attributed,
        attributions.len(),
        frac * 100.0
    );
    println!(
        "  Precision={:.3}  Recall={:.3}  F1={:.3}",
        prec, rec, f1
    );

    let result = AttributionResult {
        start_date: cli.start_date.clone(),
        end_date: cli.end_date.clone(),
        mission: cli.mission.clone(),
        n_cd_transitions: transitions.len(),
        n_curated_matched: matched_count,
        n_curated_crossings: n_curated,
        n_fn: fn_count,
        n_extras: extras.len(),
        n_attributed,
        attribution_fraction: frac,
        precision: (prec * 1000.0).round() / 1000.0,
        recall: (rec * 1000.0).round() / 1000.0,
        f1: (f1 * 1000.0).round() / 1000.0,
        type_counts,
        type_stats,
        events: attributions,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;
    println!("\nWrote {}", cli.out_json.display());

    // Optional PR curve: sweep threshold_factor from 4.0 down to 0.1 (50 steps, log-spaced).
    if let Some(pr_path) = cli.out_pr_curve {
        let factors: Vec<f64> = (0..=50)
            .map(|k| 4.0_f64 * (0.1_f64 / 4.0_f64).powf(k as f64 / 50.0))
            .collect();

        let mut pr_curve: Vec<PrPoint> = Vec::with_capacity(factors.len());
        for factor in &factors {
            let trans = detect_at(*factor);
            let (tp, extra_pr, fn_pr) = match_transitions(&trans);
            let fp = extra_pr.len();
            let prec_pr = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 1.0 };
            let rec_pr = if n_curated > 0 { tp as f64 / n_curated as f64 } else { 0.0 };
            let f1_pr = if prec_pr + rec_pr > 0.0 {
                2.0 * prec_pr * rec_pr / (prec_pr + rec_pr)
            } else {
                0.0
            };
            pr_curve.push(PrPoint {
                threshold_factor: (factor * 1000.0).round() / 1000.0,
                n_detected: trans.len(),
                tp,
                fp,
                fn_count: fn_pr,
                precision: (prec_pr * 1000.0).round() / 1000.0,
                recall: (rec_pr * 1000.0).round() / 1000.0,
                f1: (f1_pr * 1000.0).round() / 1000.0,
            });
        }

        if let Some(parent) = pr_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&pr_path, serde_json::to_string_pretty(&pr_curve)?)?;
        println!("Wrote PR curve ({} points) to {}", pr_curve.len(), pr_path.display());
    }

    Ok(())
}
