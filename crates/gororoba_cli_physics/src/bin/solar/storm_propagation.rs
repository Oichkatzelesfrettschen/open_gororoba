//! Solar storm propagation tracker: CME from SDO -> ACE -> Voyager ISM.
//!
//! Tracks a coronal mass ejection through three measurement points:
//! 1. SDO/HMI SHARP: source active region pre-flare magnetic evolution
//! 2. ACE at L1 (1 AU): ICME passage in MAG data (16-sec cadence)
//! 3. Voyager 2 PWS: ISM plasma wave response ~1 year later
//!
//! The CD associator provides a uniform phase-geometry diagnostic at each
//! stage, enabling direct comparison of magnetic disorder levels across
//! 8 orders of magnitude in heliocentric distance.
//!
//! Usage:
//!   solar storm-propagation \
//!     --ace-csv data/external/ace/ace_mag_sep2017.csv \
//!     --sharp-json data/external/sdo_aia/hmi_sharp_7115_x93.json \
//!     --v2-pws-csv data/external/voyager2/pws/v2_pws_hourly_2012_2026.csv \
//!     --harpnum 7115 --event-name "X9.3 Sep 2017"

use anyhow::{Context, Result};
use clap::Args;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Args)]
pub struct Cli {
    /// ACE MAG CSV from HAPI (timestamp, Magnitude, BGSM_x, BGSM_y, BGSM_z)
    #[arg(long)]
    ace_csv: PathBuf,

    /// SHARP JSON from JSOC (optional, for source AR analysis)
    #[arg(long)]
    sharp_json: Option<PathBuf>,

    /// SHARP HARPNUM
    #[arg(long, default_value_t = 7115)]
    harpnum: u32,

    /// V2 PWS hourly CSV (optional, for ISM response)
    #[arg(long)]
    v2_pws_csv: Option<PathBuf>,

    /// Event name for output labeling
    #[arg(long, default_value = "unnamed")]
    event_name: String,

    /// Embedding dimension (power of 2)
    #[arg(long, default_value_t = 32)]
    dim: usize,

    /// Output JSON
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/storm_propagation.json"
    )]
    out_json: PathBuf,
}

const ACE_CHANNELS: usize = 4; // |B|, Bx, By, Bz in GSM

#[derive(Debug, Clone)]
struct AceRecord {
    timestamp: String,
    magnitude: f64,
    bgsm: [f64; 3],
}

#[derive(Debug, Serialize)]
struct StormSegment {
    source: String,
    distance_au: f64,
    n_records: usize,
    n_windows: usize,
    cd_median: f64,
    cd_mean: f64,
    cd_max: f64,
    cd_p90: f64,
    peak_timestamp: String,
    quiet_median: f64,
    storm_ratio: f64,
}

#[derive(Debug, Serialize)]
struct PropagationResult {
    event_name: String,
    embedding_dim: usize,
    segments: Vec<StormSegment>,
}

fn parse_ace_csv(path: &std::path::Path) -> Result<Vec<AceRecord>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("Reading {}", path.display()))?;
    let mut records = Vec::new();
    for line in content.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }
        // Skip header-like lines
        if fields[1].contains("Magnitude") || fields[0].starts_with('#') {
            continue;
        }
        let magnitude: f64 = match fields[1].trim().parse() {
            Ok(v) if v > -1e30 => v,
            _ => continue,
        };
        let bx: f64 = match fields[2].trim().parse() {
            Ok(v) if v > -1e30 => v,
            _ => continue,
        };
        let by: f64 = match fields[3].trim().parse() {
            Ok(v) if v > -1e30 => v,
            _ => continue,
        };
        let bz: f64 = if fields.len() > 4 {
            match fields[4].trim().parse() {
                Ok(v) if v > -1e30 => v,
                _ => continue,
            }
        } else {
            continue;
        };
        records.push(AceRecord {
            timestamp: fields[0].to_string(),
            magnitude,
            bgsm: [bx, by, bz],
        });
    }
    Ok(records)
}

fn ace_embedding(records: &[AceRecord], start: usize, steps: usize) -> Option<Vec<f64>> {
    if start + steps > records.len() {
        return None;
    }
    let mut vec = Vec::with_capacity(ACE_CHANNELS * steps);
    for r in records.iter().skip(start).take(steps) {
        if !r.magnitude.is_finite() {
            return None;
        }
        vec.push(r.magnitude);
        vec.push(r.bgsm[0]);
        vec.push(r.bgsm[1]);
        vec.push(r.bgsm[2]);
    }
    Some(vec)
}

fn compute_cd_sliding(embeddings: &[Vec<f64>], dim: usize) -> Vec<f64> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    // Pad to target dim
    let padded: Vec<Vec<f64>> = embeddings
        .iter()
        .map(|v| {
            let mut p = v.clone();
            p.resize(dim, 0.0);
            p
        })
        .collect();

    // Use f32 fast path
    let padded_f32: Vec<Vec<f32>> = padded
        .iter()
        .map(|v| v.iter().map(|&x| x as f32).collect())
        .collect();

    let norms = cd_kernel::batch_sliding_associator_norms_f32(&padded_f32, dim);
    norms.into_iter().map(|n| n as f64).collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn run(cli: Cli) -> Result<()> {
    println!("=== Solar Storm Propagation: {} ===", cli.event_name);
    println!("  Embedding dim: {}D", cli.dim);

    let mut segments = Vec::new();

    // --- Segment 1: ACE at L1 (1 AU) ---
    println!("\n--- ACE MAG (1 AU) ---");
    let ace_records = parse_ace_csv(&cli.ace_csv)?;
    println!("  {} valid records from ACE MAG", ace_records.len());

    let steps = cli.dim / ACE_CHANNELS;
    let mut ace_embeddings = Vec::new();
    let mut ace_timestamps = Vec::new();
    for i in 0..ace_records.len().saturating_sub(steps) {
        if let Some(emb) = ace_embedding(&ace_records, i, steps) {
            ace_embeddings.push(emb);
            ace_timestamps.push(ace_records[i + steps - 1].timestamp.clone());
        }
    }
    println!(
        "  {} embedding windows ({}D)",
        ace_embeddings.len(),
        cli.dim
    );

    let ace_cd = compute_cd_sliding(&ace_embeddings, cli.dim);

    if !ace_cd.is_empty() {
        let mut sorted = ace_cd.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = percentile(&sorted, 0.5);
        let mean = ace_cd.iter().sum::<f64>() / ace_cd.len() as f64;
        let p90 = percentile(&sorted, 0.9);
        let max_val = sorted.last().copied().unwrap_or(0.0);

        // Find peak timestamp
        let max_idx = ace_cd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_ts = if max_idx + 1 < ace_timestamps.len() {
            ace_timestamps[max_idx + 1].clone()
        } else {
            "unknown".to_string()
        };

        // Quiet baseline: first quartile of data (pre-storm)
        let quiet_median = percentile(&sorted, 0.25);
        let storm_ratio = if quiet_median > 0.0 {
            median / quiet_median
        } else {
            0.0
        };

        println!(
            "  CD: median={:.3}, mean={:.3}, p90={:.3}, max={:.3}",
            median, mean, p90, max_val
        );
        println!("  Peak at: {}", peak_ts);
        println!(
            "  Quiet baseline (Q1): {:.3}, storm ratio: {:.1}x",
            quiet_median, storm_ratio
        );

        // Hourly CD profile for time-resolved output
        let records_per_hour = 3600 / 16; // 16-sec cadence -> 225 records/hour
        let n_hours = ace_cd.len() / records_per_hour.max(1);
        if n_hours > 1 {
            println!("  Hourly CD profile ({} hours):", n_hours);
            for h in 0..n_hours.min(24) {
                let start = h * records_per_hour;
                let end = ((h + 1) * records_per_hour).min(ace_cd.len());
                let slice = &ace_cd[start..end];
                if !slice.is_empty() {
                    let hr_mean = slice.iter().sum::<f64>() / slice.len() as f64;
                    let ts = if start + 1 < ace_timestamps.len() {
                        &ace_timestamps[start + 1]
                    } else {
                        "?"
                    };
                    let bar_len = (hr_mean / max_val * 40.0).ceil() as usize;
                    let bar: String = "#".repeat(bar_len.min(40));
                    println!("    {} A={:7.3} {}", &ts[..19.min(ts.len())], hr_mean, bar);
                }
            }
        }

        segments.push(StormSegment {
            source: "ACE_L1".to_string(),
            distance_au: 1.0,
            n_records: ace_records.len(),
            n_windows: ace_embeddings.len(),
            cd_median: median,
            cd_mean: mean,
            cd_max: max_val,
            cd_p90: p90,
            peak_timestamp: peak_ts,
            quiet_median,
            storm_ratio,
        });
    }

    // --- Segment 2: SDO SHARP (source AR, 0 AU) ---
    if let Some(sharp_path) = &cli.sharp_json {
        println!("\n--- SDO SHARP (Source AR, HARPNUM {}) ---", cli.harpnum);
        let ts = data_core::catalogs::sdo_hmi::load_sharp_json(sharp_path, cli.harpnum)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        println!(
            "  {} SHARP records at {}-min cadence",
            ts.records.len(),
            ts.cadence_minutes
        );

        let sharp_steps = 2; // 2-step Takens -> 30D, pad to 32
        let (embedded, indices) =
            data_core::catalogs::sdo_hmi::sharp_takens_embed(&ts, sharp_steps, 1);
        println!(
            "  {} embedding windows ({}D padded to {}D)",
            embedded.len(),
            data_core::catalogs::sdo_hmi::SHARP_CHANNELS * sharp_steps,
            cli.dim
        );

        let sharp_cd = compute_cd_sliding(&embedded, cli.dim);
        if !sharp_cd.is_empty() {
            let mut sorted = sharp_cd.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = percentile(&sorted, 0.5);
            let mean = sharp_cd.iter().sum::<f64>() / sharp_cd.len() as f64;
            let max_val = sorted.last().copied().unwrap_or(0.0);
            let p90 = percentile(&sorted, 0.9);

            let max_idx = sharp_cd
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let peak_ts = if max_idx + 1 < indices.len() {
                ts.records[indices[max_idx + 1]].t_rec.clone()
            } else {
                "unknown".to_string()
            };

            let quiet_median = percentile(&sorted, 0.25);
            let storm_ratio = if quiet_median > 0.0 {
                median / quiet_median
            } else {
                0.0
            };

            println!(
                "  CD: median={:.3}, mean={:.3}, p90={:.3}, max={:.3}",
                median, mean, p90, max_val
            );
            println!("  Peak at: {}", peak_ts);

            segments.push(StormSegment {
                source: format!("SDO_SHARP_{}", cli.harpnum),
                distance_au: 0.0,
                n_records: ts.records.len(),
                n_windows: embedded.len(),
                cd_median: median,
                cd_mean: mean,
                cd_max: max_val,
                cd_p90: p90,
                peak_timestamp: peak_ts,
                quiet_median,
                storm_ratio,
            });
        }
    }

    // --- Segment 3: V2 PWS ISM response ---
    if let Some(pws_path) = &cli.v2_pws_csv {
        println!("\n--- Voyager 2 PWS (ISM, ~125 AU) ---");
        let content = std::fs::read_to_string(pws_path)
            .with_context(|| format!("Reading {}", pws_path.display()))?;
        let mut pws_values: Vec<(String, Vec<f64>)> = Vec::new();
        for line in content.lines() {
            if line.starts_with('#') || line.starts_with("year") {
                continue;
            }
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 18 {
                continue;
            }
            let year: u16 = match fields[0].trim().parse() {
                Ok(v) => v,
                _ => continue,
            };
            // Focus on 2018-2020 for ~1 year delayed response to 2017 CME
            if !(2018..=2020).contains(&year) {
                continue;
            }
            let label = format!("{}-{:03}-{:02}", fields[0], fields[1], fields[2]);
            let channels: Vec<f64> = fields[3..19]
                .iter()
                .map(|f| f.trim().parse::<f64>().unwrap_or(f64::NAN))
                .collect();
            if channels.iter().all(|c| c.is_finite()) {
                pws_values.push((label, channels));
            }
        }
        println!(
            "  {} valid PWS hourly records (2018-2020)",
            pws_values.len()
        );

        if pws_values.len() >= 4 {
            // Build 16-channel embeddings with 2-step Takens -> 32D
            let pws_steps = 2;
            let pws_ch = 16;
            let mut embeddings = Vec::new();
            let mut timestamps = Vec::new();
            for i in 0..pws_values.len().saturating_sub(pws_steps) {
                let mut vec = Vec::with_capacity(pws_ch * pws_steps);
                let mut valid = true;
                for s in 0..pws_steps {
                    for c in 0..pws_ch {
                        let v = pws_values[i + s].1[c];
                        if !v.is_finite() {
                            valid = false;
                            break;
                        }
                        vec.push(v);
                    }
                    if !valid {
                        break;
                    }
                }
                if valid && vec.len() == pws_ch * pws_steps {
                    embeddings.push(vec);
                    timestamps.push(pws_values[i + pws_steps - 1].0.clone());
                }
            }
            println!(
                "  {} embedding windows ({}D padded to {}D)",
                embeddings.len(),
                pws_ch * pws_steps,
                cli.dim
            );

            let pws_cd = compute_cd_sliding(&embeddings, cli.dim);
            if !pws_cd.is_empty() {
                let mut sorted = pws_cd.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = percentile(&sorted, 0.5);
                let mean = pws_cd.iter().sum::<f64>() / pws_cd.len() as f64;
                let max_val = sorted.last().copied().unwrap_or(0.0);
                let p90 = percentile(&sorted, 0.9);

                let max_idx = pws_cd
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let peak_ts = if max_idx + 1 < timestamps.len() {
                    timestamps[max_idx + 1].clone()
                } else {
                    "unknown".to_string()
                };

                let quiet_median = percentile(&sorted, 0.25);
                let storm_ratio = if quiet_median > 0.0 {
                    median / quiet_median
                } else {
                    0.0
                };

                println!(
                    "  CD: median={:.3}, mean={:.3}, p90={:.3}, max={:.3}",
                    median, mean, p90, max_val
                );
                println!("  Peak at: {}", peak_ts);

                segments.push(StormSegment {
                    source: "V2_PWS_ISM".to_string(),
                    distance_au: 125.0,
                    n_records: pws_values.len(),
                    n_windows: embeddings.len(),
                    cd_median: median,
                    cd_mean: mean,
                    cd_max: max_val,
                    cd_p90: p90,
                    peak_timestamp: peak_ts,
                    quiet_median,
                    storm_ratio,
                });
            }
        }
    }

    // --- Summary ---
    println!("\n=== Propagation Summary: {} ===", cli.event_name);
    for seg in &segments {
        println!(
            "  {:20} ({:>6.1} AU): median={:.3}, max={:.3}, ratio={:.1}x",
            seg.source, seg.distance_au, seg.cd_median, seg.cd_max, seg.storm_ratio
        );
    }

    let result = PropagationResult {
        event_name: cli.event_name,
        embedding_dim: cli.dim,
        segments,
    };
    let json = serde_json::to_string_pretty(&result)?;
    if let Some(parent) = cli.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out_json, &json)?;
    println!("\nResults -> {}", cli.out_json.display());

    Ok(())
}
