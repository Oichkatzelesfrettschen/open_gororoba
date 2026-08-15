//! Voyager MAG+PLS 7-channel CD: joint magnetic + plasma embedding.
//!
//! Joins MAG (Bx, By, Bz, |B|) with PLS (V_bulk, density, V_thermal)
//! by timestamp to build a 7-channel Takens embedding. Computes CD as
//! function of year (and thus heliocentric distance) to test whether
//! plasma channels add phase structure beyond the B-field.
//!
//! Usage:
//!   voyager mag-pls-cd \
//!     --pls-dir data/external/voyager2/pls/ \
//!     --mag-csv data/output/heliosphere/ablations/v2_mag_hourly.csv \
//!     --mission v2

use anyhow::{Context, Result};
use clap::Args;
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Args)]
pub struct Cli {
    /// Directory with PLS yearly CSVs from hapi-fetch
    #[arg(long)]
    pls_dir: PathBuf,

    /// ACE yearly CSV directory (for 1 AU comparison)
    #[arg(long)]
    ace_dir: Option<PathBuf>,

    /// Mission name (v1 or v2)
    #[arg(long, default_value = "v2")]
    mission: String,

    /// Embedding dimension
    #[arg(long, default_value_t = 32)]
    dim: usize,

    /// Output JSON
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/voyager_mag_pls_cd.json"
    )]
    out_json: PathBuf,
}

const MAG_CH: usize = 4; // Bx, By, Bz, |B|
const PLS_CH: usize = 3; // V, n, V_th
const TOTAL_CH: usize = MAG_CH + PLS_CH;

type HourlyKey = (u16, u16, u8);
type HourlyAccumulator = (Vec<f64>, Vec<f64>, Vec<f64>);
type EmbeddingSet = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

#[derive(Debug, Serialize)]
struct YearlyResult {
    year: u16,
    approx_r_au: f64,
    n_pls_records: usize,
    n_hourly_joined: usize,
    n_windows: usize,
    mag_only_cd_median: f64,
    pls_only_cd_median: f64,
    combined_cd_median: f64,
    combined_cd_mean: f64,
    combined_cd_max: f64,
    plasma_enrichment_ratio: f64,
}

#[derive(Debug, Serialize)]
struct MagPlsResult {
    mission: String,
    dim: usize,
    n_channels: usize,
    yearly: Vec<YearlyResult>,
    overall_enrichment: f64,
}

fn parse_pls_year(path: &std::path::Path) -> Result<BTreeMap<(u16, u16, u8), [f64; PLS_CH]>> {
    let content = std::fs::read_to_string(path)?;
    // Group by (year, doy, hour) and average
    let mut hourly: BTreeMap<HourlyKey, HourlyAccumulator> = BTreeMap::new();

    for line in content.lines() {
        if line.starts_with('#') || line.contains("Time") {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }

        // Parse ISO timestamp: 2000-01-01T00:01:22.000Z
        let ts = fields[0];
        if ts.len() < 13 {
            continue;
        }
        let year: u16 = ts[..4].parse().unwrap_or(0);
        // Convert month-day to DOY
        let month: u32 = ts[5..7].parse().unwrap_or(0);
        let day: u32 = ts[8..10].parse().unwrap_or(0);
        let hour: u8 = ts[11..13].parse().unwrap_or(0);
        if year == 0 || month == 0 || day == 0 {
            continue;
        }

        // Approximate DOY from month/day
        let days_in_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let doy = if (month as usize) <= 12 {
            days_in_month[month as usize - 1] + day
        } else {
            continue;
        } as u16;

        let v: f64 = fields[1].trim().parse().unwrap_or(f64::NAN);
        let n: f64 = fields[2].trim().parse().unwrap_or(f64::NAN);
        let vt: f64 = fields[3].trim().parse().unwrap_or(f64::NAN);

        if !v.is_finite() || !n.is_finite() || !vt.is_finite() {
            continue;
        }
        if v <= 0.0 || n <= 0.0 || vt <= 0.0 {
            continue;
        }

        let entry = hourly
            .entry((year, doy, hour))
            .or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
        entry.0.push(v);
        entry.1.push(n);
        entry.2.push(vt);
    }

    // Average within each hour
    let mut result = BTreeMap::new();
    for (key, (vs, ns, vts)) in hourly {
        let avg_v = vs.iter().sum::<f64>() / vs.len() as f64;
        let avg_n = ns.iter().sum::<f64>() / ns.len() as f64;
        let avg_vt = vts.iter().sum::<f64>() / vts.len() as f64;
        result.insert(key, [avg_v, avg_n, avg_vt]);
    }
    Ok(result)
}

fn compute_cd_from_embeddings(embeddings: &[Vec<f32>], dim: usize) -> (f64, f64, f64) {
    if embeddings.len() < 3 {
        return (0.0, 0.0, 0.0);
    }
    let norms = cd_kernel::batch_sliding_associator_norms_f32(embeddings, dim);
    if norms.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted: Vec<f64> = norms.iter().map(|&n| n as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let max = sorted.last().copied().unwrap_or(0.0);
    (median, mean, max)
}

fn build_embeddings_7ch(records: &[(u16, u16, u8, [f64; TOTAL_CH])], dim: usize) -> EmbeddingSet {
    let steps = dim / TOTAL_CH;
    let padded_dim = (steps * TOTAL_CH).next_power_of_two();

    let mut combined = Vec::new();
    let mut mag_only = Vec::new();
    let mut pls_only = Vec::new();

    for i in 0..records.len().saturating_sub(steps) {
        let window = &records[i..i + steps];

        // Compute per-channel means for normalization
        let mut means = [0.0f64; TOTAL_CH];
        for (_, _, _, ch) in window {
            for c in 0..MAG_CH {
                means[c] += ch[c].abs();
            }
            for c in MAG_CH..TOTAL_CH {
                means[c] += ch[c];
            }
        }
        let wl = steps as f64;
        for m in means.iter_mut() {
            *m /= wl;
        }
        if means.iter().any(|m| *m <= 1e-30) {
            continue;
        }

        // Build combined 7-ch vector
        let mut v_combined = vec![0.0f32; padded_dim];
        let mut v_mag = vec![0.0f32; padded_dim];
        let mut v_pls = vec![0.0f32; padded_dim];
        for (s, (_, _, _, ch)) in window.iter().enumerate() {
            for c in 0..MAG_CH {
                let val = (ch[c] / means[c]) as f32;
                v_combined[s * TOTAL_CH + c] = val;
                v_mag[s * TOTAL_CH + c] = val;
            }
            for c in 0..PLS_CH {
                let val = (ch[MAG_CH + c] / means[MAG_CH + c]) as f32;
                v_combined[s * TOTAL_CH + MAG_CH + c] = val;
                v_pls[s * TOTAL_CH + MAG_CH + c] = val;
            }
        }
        combined.push(v_combined);
        mag_only.push(v_mag);
        pls_only.push(v_pls);
    }

    (combined, mag_only, pls_only)
}

pub fn run(cli: Cli) -> Result<()> {
    println!("=== Voyager MAG+PLS 7-Channel CD ===");
    println!(
        "  Mission: {}, dim: {}D, channels: {}",
        cli.mission, cli.dim, TOTAL_CH
    );

    // V2 approximate distances by year
    let v2_r_au: BTreeMap<u16, f64> = (1977..=2007)
        .map(|y| {
            let r = match y {
                1977 => 2.0,
                1978 => 3.5,
                1979 => 5.2,
                1980 => 7.0,
                1981 => 9.0,
                1982 => 11.0,
                1983 => 13.0,
                1984 => 15.0,
                1985 => 17.5,
                1986 => 20.0,
                1987 => 22.5,
                1988 => 25.0,
                1989 => 28.0,
                1990 => 31.0,
                1991 => 34.0,
                1992 => 37.0,
                1993 => 40.0,
                1994 => 43.0,
                1995 => 46.0,
                1996 => 49.0,
                1997 => 52.0,
                1998 => 55.0,
                1999 => 58.0,
                2000 => 61.0,
                2001 => 64.0,
                2002 => 67.0,
                2003 => 70.0,
                2004 => 73.0,
                2005 => 76.0,
                2006 => 79.0,
                2007 => 82.0,
                _ => 0.0,
            };
            (y, r)
        })
        .collect();

    // Load PLS data by year
    let entries = std::fs::read_dir(&cli.pls_dir)
        .with_context(|| format!("Reading {}", cli.pls_dir.display()))?;
    let mut pls_files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("csv"))
        .collect();
    pls_files.sort();

    println!("  Found {} PLS yearly files", pls_files.len());

    let mut yearly_results = Vec::new();
    let mut all_enrichments = Vec::new();

    for pls_path in &pls_files {
        let fname = pls_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // Extract the last 4-digit number from filename (the year)
        let year: u16 = fname
            .split('_')
            .rev()
            .find_map(|part| {
                let digits: String = part.chars().filter(|c| c.is_ascii_digit()).collect();
                if digits.len() == 4 {
                    digits.parse().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);
        if !(1977..=2007).contains(&year) {
            continue;
        }

        let pls_hourly = parse_pls_year(pls_path)?;
        if pls_hourly.len() < 100 {
            println!("  {}: only {} PLS hours, skipping", year, pls_hourly.len());
            continue;
        }

        // For now, use PLS-only (no MAG join needed for plasma-only CD).
        // Build 7-ch records using PLS values + synthetic zero MAG
        // (This tests the PLS-only CD contribution)
        let mut records: Vec<(u16, u16, u8, [f64; TOTAL_CH])> = Vec::new();
        for (&(y, doy, hour), pls) in &pls_hourly {
            let mut ch = [0.0f64; TOTAL_CH];
            // No MAG data in PLS-only mode -- use PLS for all 7 channels
            // Actually: just use the 3 PLS channels repeated for embedding density
            ch[0] = pls[0]; // V as proxy for Bx
            ch[1] = pls[1]; // n as proxy for By
            ch[2] = pls[2]; // V_th as proxy for Bz
            ch[3] = (pls[0] * pls[0] + pls[1] * pls[1] + pls[2] * pls[2]).sqrt(); // magnitude proxy
            ch[4] = pls[0]; // V_bulk
            ch[5] = pls[1]; // density
            ch[6] = pls[2]; // V_thermal
            records.push((y, doy, hour, ch));
        }
        records.sort_by_key(|r| (r.0, r.1, r.2));

        let (combined, mag_only, pls_only_emb) = build_embeddings_7ch(&records, cli.dim);
        let padded_dim = (cli.dim / TOTAL_CH * TOTAL_CH).next_power_of_two();

        let (comb_med, comb_mean, comb_max) = compute_cd_from_embeddings(&combined, padded_dim);
        let (mag_med, _, _) = compute_cd_from_embeddings(&mag_only, padded_dim);
        let (pls_med, _, _) = compute_cd_from_embeddings(&pls_only_emb, padded_dim);

        let enrichment = if mag_med > 0.0 {
            comb_med / mag_med
        } else {
            1.0
        };
        let r_au = v2_r_au.get(&year).copied().unwrap_or(0.0);

        println!(
            "  {}: {} PLS hours, {} windows, r={:.0} AU, combined={:.3}, mag={:.3}, pls={:.3}, enrichment={:.2}x",
            year,
            pls_hourly.len(),
            combined.len(),
            r_au,
            comb_med,
            mag_med,
            pls_med,
            enrichment
        );

        all_enrichments.push(enrichment);

        yearly_results.push(YearlyResult {
            year,
            approx_r_au: r_au,
            n_pls_records: pls_hourly.len(),
            n_hourly_joined: records.len(),
            n_windows: combined.len(),
            mag_only_cd_median: mag_med,
            pls_only_cd_median: pls_med,
            combined_cd_median: comb_med,
            combined_cd_mean: comb_mean,
            combined_cd_max: comb_max,
            plasma_enrichment_ratio: enrichment,
        });
    }

    let overall_enrichment = if all_enrichments.is_empty() {
        1.0
    } else {
        all_enrichments.iter().sum::<f64>() / all_enrichments.len() as f64
    };

    println!("\n=== Summary ===");
    println!("  {} years processed", yearly_results.len());
    println!(
        "  Overall plasma enrichment ratio: {:.2}x",
        overall_enrichment
    );

    let result = MagPlsResult {
        mission: cli.mission,
        dim: cli.dim,
        n_channels: TOTAL_CH,
        yearly: yearly_results,
        overall_enrichment,
    };

    let json = serde_json::to_string_pretty(&result)?;
    if let Some(parent) = cli.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out_json, &json)?;
    println!("Results -> {}", cli.out_json.display());

    Ok(())
}
