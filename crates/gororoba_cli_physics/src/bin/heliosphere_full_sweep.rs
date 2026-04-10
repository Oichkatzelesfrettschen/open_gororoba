//! Full-dataset sweep: run boundary survey across entire curated crossing database.
//!
//! Processes the THEMIS 2007-2016 curated crossing database week by week,
//! running the CD associator + baseline detectors on each week. Restartable:
//! skips weeks whose output JSON already exists.
//!
//! Output: per-week JSON files + aggregate summary.

use anyhow::{Context, Result};
use chrono::{Datelike, Duration, NaiveDate};
use clap::Parser;
use data_core::{
    catalogs::themis::parse_themis_fgm_hapi_csv_minutes,
    catalogs::themis_fetch::ThemisFgmProvider,
    crossing_lists::{match_crossings, parse_crossing_list},
    fetcher::{DatasetProvider, FetchConfig},
};
use serde::{Deserialize, Serialize};
use spectral_core::boundary_detectors::{
    bmag_gradient_crossings, magnetic_shear_crossings, pvi_crossings, spectral_entropy_crossings,
    wavelet_variance_crossings,
};
use std::{fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-full-sweep",
    about = "Full-dataset weekly sweep with curated labels"
)]
struct Cli {
    /// First year to sweep.
    #[arg(long, default_value_t = 2007)]
    year_start: i32,

    /// Last year to sweep.
    #[arg(long, default_value_t = 2016)]
    year_end: i32,

    /// Probe for THEMIS (a-e).
    #[arg(long, default_value = "a")]
    probe: String,

    /// Curated crossing list file.
    #[arg(
        long,
        default_value = "data/external/crossing_lists/themis_mp_crossings_v2.txt"
    )]
    crossing_list: PathBuf,

    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    #[arg(long, default_value_t = 200.0)]
    max_bmag: f64,

    #[arg(long, default_value_t = 20)]
    match_tolerance_minutes: usize,

    /// Output directory for per-week JSONs.
    #[arg(long, default_value = "data/output/heliosphere/full_sweep/themis")]
    out_dir: PathBuf,

    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,

    /// Force recompute all weeks (ignore cached JSONs).
    #[arg(long, default_value_t = false)]
    force_recompute: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct WeekResult {
    year: i32,
    week: u32,
    start_date: String,
    end_date: String,
    n_minutes: usize,
    n_curated: usize,
    cd_detection_rate: f64,
    cd_false_alarm_rate: f64,
    cd_n_transitions: usize,
    cd_n_extra: usize,
    cd_median_offset: f64,
    pvi_detection_rate: f64,
    pvi_false_alarm_rate: f64,
    bmag_detection_rate: f64,
    bmag_false_alarm_rate: f64,
    #[serde(default)]
    shear_detection_rate: f64,
    #[serde(default)]
    shear_false_alarm_rate: f64,
    #[serde(default)]
    wavelet_detection_rate: f64,
    #[serde(default)]
    wavelet_false_alarm_rate: f64,
    #[serde(default)]
    entropy_detection_rate: f64,
    #[serde(default)]
    entropy_false_alarm_rate: f64,
}

#[derive(Debug, Serialize)]
struct SweepSummary {
    total_weeks: usize,
    total_curated_crossings: usize,
    total_cd_transitions: usize,
    total_cd_extra: usize,
    aggregate_cd_detection: f64,
    aggregate_cd_false_alarm: f64,
    aggregate_pvi_detection: f64,
    aggregate_pvi_false_alarm: f64,
    aggregate_bmag_detection: f64,
    aggregate_bmag_false_alarm: f64,
    weeks: Vec<WeekResult>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    fs::create_dir_all(&cli.out_dir)?;

    // Load curated crossing list once
    let crossing_content = fs::read_to_string(&cli.crossing_list)
        .with_context(|| format!("read {}", cli.crossing_list.display()))?;

    let _config = FetchConfig {
        output_dir: cli.data_dir.clone(),
        skip_existing: true,
        verify_checksums: false,
    };

    let tol_hours = cli.match_tolerance_minutes as f64 / 60.0;
    let channels = 4usize;
    let steps = cli.embedding_dim / channels;
    // THEMIS probes: cli.probe is "a"-"e", provider needs "THA"-"THE"
    let probe_upper = format!("TH{}", cli.probe.to_uppercase());

    let mut all_weeks: Vec<WeekResult> = Vec::new();
    let mut total_curated = 0usize;
    let mut total_cd_matched = 0usize;
    let mut total_cd_trans = 0usize;
    let mut total_pvi_matched = 0usize;
    let mut total_pvi_trans = 0usize;
    let mut total_bmag_matched = 0usize;
    let mut total_bmag_trans = 0usize;

    // Iterate over weeks
    let mut current = NaiveDate::from_ymd_opt(cli.year_start, 1, 1).unwrap();
    let final_date = NaiveDate::from_ymd_opt(cli.year_end, 12, 31).unwrap();

    while current <= final_date {
        let week_end = current + Duration::days(6);
        let week_num = current.iso_week().week();

        let out_file = cli
            .out_dir
            .join(format!("week_{:04}_{:02}.json", current.year(), week_num));

        // Skip if already computed (restartable) unless --force-recompute
        if !cli.force_recompute
            && out_file.exists()
            && let Ok(content) = fs::read_to_string(&out_file)
            && let Ok(wr) = serde_json::from_str::<WeekResult>(&content)
        {
            total_curated += wr.n_curated;
            total_cd_trans += wr.cd_n_transitions;
            all_weeks.push(wr);
            current += Duration::days(7);
            continue;
        }

        // Parse curated crossings for this week
        let events = parse_crossing_list(&crossing_content, Some(&cli.probe), &current, &week_end);

        if events.is_empty() {
            // No curated crossings this week -- skip
            current += Duration::days(7);
            continue;
        }

        let curated_hours: Vec<f64> = events.iter().map(|e| e.elapsed_hours).collect();

        // Fetch + parse FGM data
        let mut bx = Vec::new();
        let mut by = Vec::new();
        let mut bz = Vec::new();
        let mut elapsed = Vec::new();

        for offset in 0..7 {
            let date = current + Duration::days(offset);
            if date > final_date {
                break;
            }

            let doy = date.ordinal();
            let fname = format!(
                "{}_fgm_{:04}_{:03}.csv",
                probe_upper.to_lowercase(),
                date.year(),
                doy
            );
            let dir = cli.data_dir.join("themis");
            fs::create_dir_all(&dir)?;
            let path = dir.join(&fname);

            if !path.exists() {
                let provider = ThemisFgmProvider {
                    probe: probe_upper.clone(),
                    year: date.year() as u16,
                    doy_start: doy as u16,
                    doy_end: doy as u16,
                };
                if let Err(e) = provider.fetch(&_config) {
                    eprintln!("  Fetch error DOY {}: {}", doy, e);
                    continue;
                }
            }

            if path.exists() {
                let content = fs::read_to_string(&path)?;
                for r in parse_themis_fgm_hapi_csv_minutes(&content, &probe_upper) {
                    if r.b_magnitude <= cli.max_bmag {
                        let day_off = (r.doy as f64 - current.ordinal() as f64)
                            + (r.year as f64 - current.year() as f64) * 365.25;
                        let eh = day_off * 24.0 + r.hour as f64 + r.minute as f64 / 60.0;
                        bx.push(r.bx_gse);
                        by.push(r.by_gse);
                        bz.push(r.bz_gse);
                        elapsed.push(eh);
                    }
                }
            }
        }

        let n = bx.len();
        if n < steps + 5 {
            current += Duration::days(7);
            continue;
        }

        // --- CD Associator ---
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

        if embedded.len() < 3 {
            current += Duration::days(7);
            continue;
        }

        let assoc_norms =
            cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);
        let assoc_idx: Vec<usize> = (0..assoc_norms.len()).map(|k| embed_meta[k + 2]).collect();

        let global_mean = assoc_norms.iter().sum::<f64>() / assoc_norms.len() as f64;
        let global_std = {
            let var = assoc_norms
                .iter()
                .map(|&a| (a - global_mean).powi(2))
                .sum::<f64>()
                / assoc_norms.len() as f64;
            var.sqrt()
        };
        let threshold = global_std * 1.5;
        let tw = 10usize;

        let mut cd_trans_hours: Vec<f64> = Vec::new();
        if assoc_norms.len() > tw * 2 {
            let mut last: Option<usize> = None;
            for i in tw..assoc_norms.len().saturating_sub(tw) {
                let pre: f64 = assoc_norms[i.saturating_sub(tw)..i].iter().sum::<f64>() / tw as f64;
                let post: f64 = assoc_norms[i..(i + tw).min(assoc_norms.len())]
                    .iter()
                    .sum::<f64>()
                    / tw.min(assoc_norms.len() - i) as f64;
                if (post - pre).abs() > threshold {
                    if last.is_some_and(|prev| i.saturating_sub(prev) < tw) {
                        continue;
                    }
                    let idx = assoc_idx[i];
                    if idx < elapsed.len() {
                        cd_trans_hours.push(elapsed[idx]);
                    }
                    last = Some(i);
                }
            }
        }

        // --- PVI ---
        let pvi_idx = pvi_crossings(&bx, &by, &bz, 1, 4.0, 10);
        let pvi_hours: Vec<f64> = pvi_idx.iter().map(|&i| elapsed[i.min(n - 1)]).collect();

        // --- |B| gradient ---
        let bmag_idx = bmag_gradient_crossings(&bx, &by, &bz, 10, 10.0);
        let bmag_hours: Vec<f64> = bmag_idx.iter().map(|&i| elapsed[i.min(n - 1)]).collect();

        // --- Magnetic shear ---
        let shear_idx = magnetic_shear_crossings(&bx, &by, &bz, 10, 45.0);
        let shear_hours: Vec<f64> = shear_idx.iter().map(|&i| elapsed[i.min(n - 1)]).collect();

        // --- Wavelet variance ---
        let wavelet_idx = wavelet_variance_crossings(&bx, &by, &bz, 30, 1.0);
        let wavelet_hours: Vec<f64> = wavelet_idx.iter().map(|&i| elapsed[i.min(n - 1)]).collect();

        // --- Spectral entropy ---
        let entropy_idx = spectral_entropy_crossings(&bx, &by, &bz, 30, 0.1);
        let entropy_hours: Vec<f64> = entropy_idx.iter().map(|&i| elapsed[i.min(n - 1)]).collect();

        // --- Match against curated ---
        let (cd_m, cd_off) = match_crossings(&curated_hours, &cd_trans_hours, tol_hours);
        let cd_rev = cd_trans_hours
            .iter()
            .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
            .count();

        let (pvi_m, _) = match_crossings(&curated_hours, &pvi_hours, tol_hours);
        let pvi_rev = pvi_hours
            .iter()
            .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
            .count();

        let (bmag_m, _) = match_crossings(&curated_hours, &bmag_hours, tol_hours);
        let bmag_rev = bmag_hours
            .iter()
            .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
            .count();

        let nc = curated_hours.len();
        let cd_det = cd_m as f64 / nc as f64;
        let cd_fa = if cd_trans_hours.is_empty() {
            0.0
        } else {
            (cd_trans_hours.len() - cd_rev) as f64 / cd_trans_hours.len() as f64
        };
        let cd_extra = cd_trans_hours.len().saturating_sub(cd_rev);
        let cd_med = if cd_off.is_empty() {
            0.0
        } else {
            cd_off[cd_off.len() / 2]
        };

        let pvi_det = pvi_m as f64 / nc as f64;
        let pvi_fa = if pvi_hours.is_empty() {
            0.0
        } else {
            (pvi_hours.len() - pvi_rev) as f64 / pvi_hours.len() as f64
        };

        let bmag_det = bmag_m as f64 / nc as f64;
        let bmag_fa = if bmag_hours.is_empty() {
            0.0
        } else {
            (bmag_hours.len() - bmag_rev) as f64 / bmag_hours.len() as f64
        };

        let (shear_m, _) = match_crossings(&curated_hours, &shear_hours, tol_hours);
        let shear_rev = shear_hours
            .iter()
            .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
            .count();
        let shear_det = shear_m as f64 / nc as f64;
        let shear_fa = if shear_hours.is_empty() {
            0.0
        } else {
            (shear_hours.len() - shear_rev) as f64 / shear_hours.len() as f64
        };

        let (wav_m, _) = match_crossings(&curated_hours, &wavelet_hours, tol_hours);
        let wav_rev = wavelet_hours
            .iter()
            .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
            .count();
        let wav_det = wav_m as f64 / nc as f64;
        let wav_fa = if wavelet_hours.is_empty() {
            0.0
        } else {
            (wavelet_hours.len() - wav_rev) as f64 / wavelet_hours.len() as f64
        };

        let (ent_m, _) = match_crossings(&curated_hours, &entropy_hours, tol_hours);
        let ent_rev = entropy_hours
            .iter()
            .filter(|&&th| curated_hours.iter().any(|&ch| (ch - th).abs() < tol_hours))
            .count();
        let ent_det = ent_m as f64 / nc as f64;
        let ent_fa = if entropy_hours.is_empty() {
            0.0
        } else {
            (entropy_hours.len() - ent_rev) as f64 / entropy_hours.len() as f64
        };

        let wr = WeekResult {
            year: current.year(),
            week: week_num,
            start_date: current.to_string(),
            end_date: week_end.to_string(),
            n_minutes: n,
            n_curated: nc,
            cd_detection_rate: cd_det,
            cd_false_alarm_rate: cd_fa,
            cd_n_transitions: cd_trans_hours.len(),
            cd_n_extra: cd_extra,
            cd_median_offset: cd_med,
            pvi_detection_rate: pvi_det,
            pvi_false_alarm_rate: pvi_fa,
            bmag_detection_rate: bmag_det,
            bmag_false_alarm_rate: bmag_fa,
            shear_detection_rate: shear_det,
            shear_false_alarm_rate: shear_fa,
            wavelet_detection_rate: wav_det,
            wavelet_false_alarm_rate: wav_fa,
            entropy_detection_rate: ent_det,
            entropy_false_alarm_rate: ent_fa,
        };

        println!(
            "  {:04}-W{:02}: {} curated, CD {:.0}%/{:.0}%FA, PVI {:.0}%/{:.0}%FA, |B| {:.0}%/{:.0}%FA, {} extra",
            current.year(),
            week_num,
            nc,
            cd_det * 100.0,
            cd_fa * 100.0,
            pvi_det * 100.0,
            pvi_fa * 100.0,
            bmag_det * 100.0,
            bmag_fa * 100.0,
            cd_extra,
        );

        fs::write(&out_file, serde_json::to_string_pretty(&wr)?)?;

        total_curated += nc;
        total_cd_matched += cd_m;
        total_cd_trans += cd_trans_hours.len();
        total_pvi_matched += pvi_m;
        total_pvi_trans += pvi_hours.len();
        total_bmag_matched += bmag_m;
        total_bmag_trans += bmag_hours.len();
        all_weeks.push(wr);

        current += Duration::days(7);
    }

    // Aggregate summary
    let agg_cd_det = if total_curated > 0 {
        total_cd_matched as f64 / total_curated as f64
    } else {
        0.0
    };
    let total_cd_extra: usize = all_weeks.iter().map(|w| w.cd_n_extra).sum();

    println!("\n=== SWEEP COMPLETE ===");
    println!("  Weeks processed: {}", all_weeks.len());
    println!("  Total curated crossings: {}", total_curated);
    println!("  Aggregate CD detection: {:.1}%", agg_cd_det * 100.0);
    println!("  Total CD extra transitions: {}", total_cd_extra);

    let summary = SweepSummary {
        total_weeks: all_weeks.len(),
        total_curated_crossings: total_curated,
        total_cd_transitions: total_cd_trans,
        total_cd_extra,
        aggregate_cd_detection: agg_cd_det,
        aggregate_cd_false_alarm: if total_cd_trans > 0 {
            (total_cd_trans - total_cd_matched) as f64 / total_cd_trans as f64
        } else {
            0.0
        },
        aggregate_pvi_detection: if total_curated > 0 {
            total_pvi_matched as f64 / total_curated as f64
        } else {
            0.0
        },
        aggregate_pvi_false_alarm: if total_pvi_trans > 0 {
            (total_pvi_trans - total_pvi_matched) as f64 / total_pvi_trans as f64
        } else {
            0.0
        },
        aggregate_bmag_detection: if total_curated > 0 {
            total_bmag_matched as f64 / total_curated as f64
        } else {
            0.0
        },
        aggregate_bmag_false_alarm: if total_bmag_trans > 0 {
            (total_bmag_trans - total_bmag_matched) as f64 / total_bmag_trans as f64
        } else {
            0.0
        },
        weeks: all_weeks,
    };

    let summary_path = cli.out_dir.join("sweep_summary.json");
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    println!("\nWrote {}", summary_path.display());

    Ok(())
}
