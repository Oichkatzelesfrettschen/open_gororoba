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
    catalogs::themis::parse_themis_fgm_hapi_csv_minutes,
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
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,
    #[arg(long, default_value_t = 200.0)]
    max_bmag: f64,
    #[arg(long)]
    crossing_list: PathBuf,
    #[arg(long)]
    crossing_probe: Option<String>,
    #[arg(long, default_value_t = 20)]
    match_tolerance_minutes: usize,
    #[arg(long, default_value = "data/output/heliosphere/ablations/fa_attribution.json")]
    out_json: PathBuf,
    #[arg(long, default_value = "data/external")]
    data_dir: String,
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
struct AttributionResult {
    start_date: String,
    end_date: String,
    n_cd_transitions: usize,
    n_curated_matched: usize,
    n_extras: usize,
    n_attributed: usize,
    attribution_fraction: f64,
    type_counts: BTreeMap<String, usize>,
    events: Vec<Attribution>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad start: {}", cli.start_date))?;
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")
        .with_context(|| format!("bad end: {}", cli.end_date))?;

    let probe_upper = format!("TH{}", cli.crossing_probe.as_deref().unwrap_or("A").to_uppercase());

    // Load FGM
    let mut bx = Vec::new();
    let mut by = Vec::new();
    let mut bz = Vec::new();
    let mut keys: Vec<(u16, u8, u8)> = Vec::new();
    let mut elapsed: Vec<f64> = Vec::new();

    for offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(offset);
        let path = format!(
            "{}/themis/{}_fgm_{:04}_{:03}.csv",
            cli.data_dir, probe_upper.to_lowercase(), date.year(), date.ordinal()
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
        if mean_b <= 0.01 || !mean_b.is_finite() { continue; }
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
        let var = norms.iter().map(|&a| (a - global_mean).powi(2)).sum::<f64>() / norms.len() as f64;
        var.sqrt()
    };
    let threshold = global_std * 1.5;
    let tw = 10usize;

    let mut transitions: Vec<(usize, f64)> = Vec::new(); // (minute_index, norm)
    if norms.len() > tw * 2 {
        let mut last: Option<usize> = None;
        for i in tw..norms.len().saturating_sub(tw) {
            let pre: f64 = norms[i.saturating_sub(tw)..i].iter().sum::<f64>() / tw as f64;
            let post: f64 = norms[i..(i + tw).min(norms.len())].iter().sum::<f64>()
                / tw.min(norms.len() - i) as f64;
            if (post - pre).abs() > threshold {
                if last.is_some_and(|prev| i.saturating_sub(prev) < tw) { continue; }
                let mi = assoc_idx[i];
                transitions.push((mi, norms[i]));
                last = Some(i);
            }
        }
    }
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
    println!("Curated crossings: {}", curated_hours.len());

    // Classify transitions as matched or extra
    let tol_hours = cli.match_tolerance_minutes as f64 / 60.0;
    let mut matched_count = 0usize;
    let mut extras: Vec<(usize, f64)> = Vec::new();

    for &(mi, norm) in &transitions {
        let t_hours = elapsed[mi.min(n - 1)];
        let is_matched = curated_hours.iter().any(|&ch| (ch - t_hours).abs() < tol_hours);
        if is_matched {
            matched_count += 1;
        } else {
            extras.push((mi, norm));
        }
    }
    println!("Matched: {}, Extras: {}", matched_count, extras.len());

    // Classify each extra by local B-field signature
    let w = 10usize;
    let mut attributions: Vec<Attribution> = Vec::new();
    let mut type_counts: BTreeMap<String, usize> = BTreeMap::new();

    for &(mi, norm) in &extras {
        if mi < w || mi + w >= n { continue; }

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
        let var_b = bmag_local.iter().map(|&b| (b - mean_b).powi(2)).sum::<f64>()
            / bmag_local.len() as f64;
        let cv = if mean_b > 0.1 { var_b.sqrt() / mean_b } else { 0.0 };
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

    println!("\n=== Attribution Results ===");
    println!("  Total extras: {}", attributions.len());
    for (typ, count) in &type_counts {
        println!("  {}: {} ({:.1}%)", typ, count, *count as f64 / attributions.len().max(1) as f64 * 100.0);
    }
    println!("  ATTRIBUTED: {}/{} = {:.1}%", n_attributed, attributions.len(), frac * 100.0);

    let result = AttributionResult {
        start_date: cli.start_date.clone(),
        end_date: cli.end_date.clone(),
        n_cd_transitions: transitions.len(),
        n_curated_matched: matched_count,
        n_extras: extras.len(),
        n_attributed,
        attribution_fraction: frac,
        type_counts,
        events: attributions,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;
    println!("\nWrote {}", cli.out_json.display());

    Ok(())
}
