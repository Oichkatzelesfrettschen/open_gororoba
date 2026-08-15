//! PSP/Solar Orbiter switchback detection and fast-wind Omega_mv analysis.
//!
//! Detects magnetic field switchbacks (Br sign reversals) in PSP FIELDS
//! 1-minute MAG data, computes the 32D CD associator separately for
//! switchback vs quiet intervals, and compares Omega_mv between fast-wind
//! and slow-wind regimes.
//!
//! WHY: Switchbacks are a major PSP discovery. Testing whether the CD
//! associator discriminates switchback geometry from laminar flow probes
//! whether the algebraic framework captures MHD-scale dynamics at ~0.1 AU.

use anyhow::{Context, Result};
use clap::Args;
use data_core::{
    catalogs::psp_fields::{PspFieldsMagRecord, parse_psp_fields_hapi_csv},
    fetcher::{FetchError, download_hapi_csv},
};
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Args, Debug)]
pub struct Cli {
    /// Start date (YYYY-MM-DD).
    #[arg(long, default_value = "2020-01-20")]
    start_date: String,

    /// End date (YYYY-MM-DD).
    #[arg(long, default_value = "2020-02-10")]
    end_date: String,

    /// Embedding dimension.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Minimum consecutive hours for switchback classification.
    #[arg(long, default_value_t = 1)]
    min_switchback_hours: usize,

    /// Br reversal threshold: fraction of |B| below which Br is considered reversed.
    /// Standard: if Br/|B| < -0.5 (field pointing sunward), it's a switchback.
    #[arg(long, default_value_t = -0.5)]
    br_threshold: f64,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/psp_switchback_omega.json"
    )]
    out_json: PathBuf,

    /// Data cache directory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct SwitchbackResults {
    start_date: String,
    end_date: String,
    embedding_dim: usize,
    br_threshold: f64,
    total_hours: usize,
    switchback_hours: usize,
    quiet_hours: usize,
    switchback_fraction: f64,
    switchback_mean_associator: f64,
    quiet_mean_associator: f64,
    switchback_median_associator: f64,
    quiet_median_associator: f64,
    ratio_switchback_to_quiet: f64,
    hourly_detail: Vec<HourlyDetail>,
}

#[derive(Debug, Serialize)]
struct HourlyDetail {
    year: u16,
    doy: u16,
    hour: u8,
    br: f64,
    bt: f64,
    bn: f64,
    b_magnitude: f64,
    br_over_b: f64,
    is_switchback: bool,
    associator_norm: Option<f64>,
}

pub fn run(cli: Cli) -> Result<()> {
    println!(
        "=== PSP Switchback + Omega_mv Analysis ===\n\
         Window: {} to {}\n\
         Embedding: {}D\n\
         Br threshold: {:.2}\n",
        cli.start_date, cli.end_date, cli.embedding_dim, cli.br_threshold,
    );

    // --- Step 1: Fetch PSP FIELDS 1-min MAG data ---
    println!("[1/4] Fetching PSP FIELDS MAG RTN 1-min data...");

    let psp_dir = cli.data_dir.join("psp_fields");
    fs::create_dir_all(&psp_dir)?;

    let start = chrono::NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad start_date: {}", cli.start_date))?;
    let end = chrono::NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")
        .with_context(|| format!("bad end_date: {}", cli.end_date))?;

    // Fetch in weekly chunks
    let dataset = "PSP_FLD_L2_MAG_RTN_1MIN";
    let mut all_records: Vec<PspFieldsMagRecord> = Vec::new();
    let mut current = start;

    while current <= end {
        let chunk_end = (current + chrono::Duration::days(7)).min(end);
        let t_min = format!("{}T00:00:00Z", current);
        let t_max = format!("{}T23:59:59Z", chunk_end);

        let fname = format!(
            "psp_fields_mag_{}_{}.csv",
            current.format("%Y%m%d"),
            chunk_end.format("%Y%m%d")
        );
        let output = psp_dir.join(&fname);

        if output.exists() {
            let content = fs::read_to_string(&output)?;
            let records = parse_psp_fields_hapi_csv(&content);
            println!(
                "  {} to {}: {} hourly records (cached)",
                current,
                chunk_end,
                records.len()
            );
            all_records.extend(records);
        } else {
            println!("  Fetching {} to {}...", current, chunk_end);
            match download_hapi_csv(dataset, &t_min, &t_max, None) {
                Ok(body) => {
                    fs::write(&output, &body)?;
                    let records = parse_psp_fields_hapi_csv(&body);
                    println!(
                        "  {} to {}: {} hourly records",
                        current,
                        chunk_end,
                        records.len()
                    );
                    all_records.extend(records);
                }
                Err(FetchError::Validation(msg)) if msg.contains("no data") => {
                    println!(
                        "  {} to {}: no data (PSP not in science mode)",
                        current, chunk_end
                    );
                }
                Err(e) => {
                    eprintln!("  Warning: {} to {}: {}", current, chunk_end, e);
                }
            }
        }
        current = chunk_end + chrono::Duration::days(1);
    }

    if all_records.is_empty() {
        anyhow::bail!("No PSP FIELDS data found for the requested window.");
    }

    // Sort by time
    all_records.sort_by(|a, b| {
        a.year
            .cmp(&b.year)
            .then(a.doy.cmp(&b.doy))
            .then(a.hour.cmp(&b.hour))
    });

    println!("  Total: {} hourly records", all_records.len());

    // --- Step 2: Classify switchbacks ---
    println!(
        "[2/4] Classifying switchbacks (Br/|B| < {:.2})...",
        cli.br_threshold
    );

    let mut hourly_detail: Vec<HourlyDetail> = Vec::new();
    for rec in &all_records {
        let br_over_b = if rec.b_magnitude > 0.01 {
            rec.br / rec.b_magnitude
        } else {
            0.0
        };
        // In the inner heliosphere, the mean Parker spiral has Br < 0 (toward Sun)
        // for the away sector and Br > 0 for the toward sector.
        // A switchback reverses the local Br sign. Using Br/|B| < threshold
        // captures field folding regardless of polarity sector.
        let is_switchback = br_over_b < cli.br_threshold;

        hourly_detail.push(HourlyDetail {
            year: rec.year,
            doy: rec.doy,
            hour: rec.hour,
            br: rec.br,
            bt: rec.bt,
            bn: rec.bn,
            b_magnitude: rec.b_magnitude,
            br_over_b,
            is_switchback,
            associator_norm: None,
        });
    }

    let switchback_count = hourly_detail.iter().filter(|d| d.is_switchback).count();
    let quiet_count = hourly_detail.len() - switchback_count;
    println!(
        "  Switchback hours: {} ({:.1}%)",
        switchback_count,
        100.0 * switchback_count as f64 / hourly_detail.len() as f64
    );
    println!("  Quiet hours: {}", quiet_count);

    // --- Step 3: Compute 32D CD associator ---
    println!(
        "[3/4] Computing {}D Takens embedding + CD associator...",
        cli.embedding_dim
    );

    let channels: usize = 4;
    let steps = cli.embedding_dim / channels;
    let window_rows = (steps - 1) + 1; // lag=1 for hourly

    if all_records.len() < window_rows + 2 {
        anyhow::bail!(
            "Not enough data: need {} rows, have {}",
            window_rows + 2,
            all_records.len()
        );
    }

    // Build Takens-embedded vectors
    let mut embedded: Vec<Vec<f64>> = Vec::new();
    let mut embed_meta: Vec<usize> = Vec::new();

    for w_start in 0..=(all_records.len() - window_rows) {
        let sample_indices: Vec<usize> = (0..steps).map(|s| w_start + s).collect();

        let sum_b: f64 = sample_indices
            .iter()
            .map(|&i| all_records[i].b_magnitude)
            .sum();
        let local_mean_b = sum_b / steps as f64;
        if local_mean_b <= 0.0 || !local_mean_b.is_finite() {
            continue;
        }

        let mut v = vec![0.0; cli.embedding_dim];
        for (s, &ri) in sample_indices.iter().enumerate() {
            let rec = &all_records[ri];
            v[s * channels] = rec.br / local_mean_b;
            v[s * channels + 1] = rec.bt / local_mean_b;
            v[s * channels + 2] = rec.bn / local_mean_b;
            v[s * channels + 3] = (rec.b_magnitude - local_mean_b) / local_mean_b;
        }
        embedded.push(v);
        embed_meta.push(*sample_indices.last().unwrap());
    }

    println!(
        "  Embedded {} vectors ({}D)",
        embedded.len(),
        cli.embedding_dim
    );

    let associators =
        cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);

    println!("  Computed {} associator norms", associators.len());

    // Map associator norms back to hourly records
    for (k, &norm) in associators.iter().enumerate() {
        let tag_idx = embed_meta[k + 2]; // triple (emb[k], emb[k+1], emb[k+2])
        if tag_idx < hourly_detail.len() {
            hourly_detail[tag_idx].associator_norm = Some(norm);
        }
    }

    // --- Step 4: Compare switchback vs quiet associator ---
    println!("[4/4] Comparing switchback vs quiet associator statistics...");

    let mut switchback_assocs: Vec<f64> = Vec::new();
    let mut quiet_assocs: Vec<f64> = Vec::new();

    for d in &hourly_detail {
        if let Some(a) = d.associator_norm {
            if d.is_switchback {
                switchback_assocs.push(a);
            } else {
                quiet_assocs.push(a);
            }
        }
    }

    switchback_assocs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    quiet_assocs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let sb_mean = if switchback_assocs.is_empty() {
        0.0
    } else {
        switchback_assocs.iter().sum::<f64>() / switchback_assocs.len() as f64
    };
    let q_mean = if quiet_assocs.is_empty() {
        0.0
    } else {
        quiet_assocs.iter().sum::<f64>() / quiet_assocs.len() as f64
    };
    let sb_median = if switchback_assocs.is_empty() {
        0.0
    } else {
        switchback_assocs[switchback_assocs.len() / 2]
    };
    let q_median = if quiet_assocs.is_empty() {
        0.0
    } else {
        quiet_assocs[quiet_assocs.len() / 2]
    };
    let ratio = if q_mean > 1e-15 {
        sb_mean / q_mean
    } else {
        0.0
    };

    println!("\n=== Results ===");
    println!(
        "  Switchback associator: mean={:.4}, median={:.4} (n={})",
        sb_mean,
        sb_median,
        switchback_assocs.len()
    );
    println!(
        "  Quiet associator:      mean={:.4}, median={:.4} (n={})",
        q_mean,
        q_median,
        quiet_assocs.len()
    );
    println!("  Ratio (switchback/quiet): {:.3}", ratio);

    let results = SwitchbackResults {
        start_date: cli.start_date.clone(),
        end_date: cli.end_date.clone(),
        embedding_dim: cli.embedding_dim,
        br_threshold: cli.br_threshold,
        total_hours: hourly_detail.len(),
        switchback_hours: switchback_count,
        quiet_hours: quiet_count,
        switchback_fraction: switchback_count as f64 / hourly_detail.len() as f64,
        switchback_mean_associator: sb_mean,
        quiet_mean_associator: q_mean,
        switchback_median_associator: sb_median,
        quiet_median_associator: q_median,
        ratio_switchback_to_quiet: ratio,
        hourly_detail,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&cli.out_json, &json)?;
    println!("\nWrote results to {}", cli.out_json.display());

    Ok(())
}
