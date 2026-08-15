//! Cross-mission log-associator self-similarity statistics.
//!
//! WHY: If the raw staple associator of solar-wind magnetic field data is
//! self-similar across heliocentric distance, the log of the
//! median-normalized associator should carry mission-independent moments.
//! Comparing log-sigma, skew, and excess kurtosis from PSP (0.15 AU) out to
//! Voyager (80-100 AU) tests that directly and feeds the regime-boundary
//! analysis (C-1617/18).
//!
//! WHAT: Samples up to K files per mission from data/external, reads the
//! field vector with a per-mission reader (Voyager 33-column .asc,
//! named-column CSV, MESSENGER B_radial/B_tangential/B_normal CSV), computes
//! unnormalized joint associator norms over the raw 16D staple embedding,
//! and aggregates per-file `LogMomentStats` into one JSON keyed by mission.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin heliosphere -- selfsim-logstats \
//!     --external-dir data/external --out data/output/agg_selfsim.json

use anyhow::{Context, Result};
use clap::Args;
use gororoba_cli_physics::staple_associator::{
    LogMomentStats, joint_associator_norms, log_moment_stats, staple_embedding,
};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Args, Debug)]
pub struct Cli {
    /// Root of the external data tree (mission subdirectories).
    #[arg(long, default_value = "data/external")]
    external_dir: PathBuf,

    /// Output JSON path.
    #[arg(long, default_value = "data/output/agg_selfsim.json")]
    out: PathBuf,

    /// Files sampled per mission (evenly spaced over the sorted listing).
    #[arg(long, default_value_t = 40)]
    files_per_mission: usize,

    /// Row cap per file before embedding.
    #[arg(long, default_value_t = 20_000)]
    max_rows: usize,
}

#[derive(Clone, Copy)]
enum Reader {
    /// Whitespace-separated daily/hourly merge files; BR, BT, BN are
    /// columns 8..10 and 900+ magnitudes are fill values.
    VoyagerAsc,
    /// CSV whose header names the field columns (B_vec / BGSE / RTN /
    /// psp_fld prefixes); falls back to headerless columns 1..3.
    NamedCsv,
    /// CSV with B_radial / B_tangential / B_normal named columns.
    MessengerCsv,
}

struct MissionSpec {
    name: &'static str,
    distance_au: f64,
    reader: Reader,
    pattern: &'static str,
}

const MISSIONS: &[MissionSpec] = &[
    MissionSpec {
        name: "psp",
        distance_au: 0.15,
        reader: Reader::NamedCsv,
        pattern: "csv",
    },
    MissionSpec {
        name: "psp_fields",
        distance_au: 0.15,
        reader: Reader::NamedCsv,
        pattern: "csv",
    },
    MissionSpec {
        name: "messenger",
        distance_au: 0.39,
        reader: Reader::MessengerCsv,
        pattern: "csv",
    },
    MissionSpec {
        name: "ace",
        distance_au: 1.0,
        reader: Reader::NamedCsv,
        pattern: "csv",
    },
    MissionSpec {
        name: "themis",
        distance_au: 1.0,
        reader: Reader::NamedCsv,
        pattern: "csv",
    },
    MissionSpec {
        name: "cluster",
        distance_au: 1.0,
        reader: Reader::NamedCsv,
        pattern: "csv",
    },
    MissionSpec {
        name: "voyager1",
        distance_au: 100.0,
        reader: Reader::VoyagerAsc,
        pattern: "asc",
    },
    MissionSpec {
        name: "voyager2",
        distance_au: 80.0,
        reader: Reader::VoyagerAsc,
        pattern: "asc",
    },
];

/// Reject cached error pages saved under a data filename.
fn is_html(path: &Path) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return true;
    };
    let head = &bytes[..bytes.len().min(200)];
    let lower: Vec<u8> = head.to_ascii_lowercase();
    lower.windows(5).any(|w| w == b"<html")
        || head.windows(9).any(|w| w == b"<!DOCTYPE")
        || lower[..lower.len().min(60)]
            .windows(5)
            .any(|w| w == b"route")
        || head[..head.len().min(60)]
            .windows(14)
            .any(|w| w == b"does not exist")
}

fn read_voyager_asc(path: &Path) -> Option<Vec<[f64; 3]>> {
    let text = fs::read_to_string(path).ok()?;
    let mut rows = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 11 {
            continue;
        }
        let br: f64 = parts[8].parse().ok()?;
        let bt: f64 = parts[9].parse().ok()?;
        let bn: f64 = parts[10].parse().ok()?;
        if br.abs() > 900.0 || bt.abs() > 900.0 || bn.abs() > 900.0 {
            continue;
        }
        rows.push([br, bt, bn]);
    }
    Some(rows)
}

fn csv_columns(path: &Path, wanted: impl Fn(&str) -> bool) -> Option<Vec<[f64; 3]>> {
    let mut reader = csv::Reader::from_path(path).ok()?;
    let headers = reader.headers().ok()?.clone();
    let cols: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter(|(_, h)| wanted(h))
        .map(|(i, _)| i)
        .take(3)
        .collect();
    if cols.len() < 3 {
        return None;
    }
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.ok()?;
        let mut b = [f64::NAN; 3];
        for (slot, &c) in cols.iter().enumerate() {
            b[slot] = record.get(c)?.trim().parse().unwrap_or(f64::NAN);
        }
        rows.push(b);
    }
    Some(rows)
}

fn csv_positional(path: &Path) -> Option<Vec<[f64; 3]>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(path)
        .ok()?;
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.ok()?;
        let parse = |i: usize| -> f64 {
            record
                .get(i)
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(f64::NAN)
        };
        rows.push([parse(1), parse(2), parse(3)]);
    }
    Some(rows)
}

fn read_field(path: &Path, reader: Reader) -> Option<Vec<[f64; 3]>> {
    if is_html(path) {
        return None;
    }
    let rows = match reader {
        Reader::VoyagerAsc => read_voyager_asc(path)?,
        Reader::NamedCsv => csv_columns(path, |h| {
            (h.contains("B_vec")
                || h.contains("BGSE")
                || h.contains("RTN")
                || h.starts_with("psp_fld"))
                && !h.to_lowercase().contains("mag")
        })
        .or_else(|| csv_positional(path))?,
        Reader::MessengerCsv => csv_columns(path, |h| {
            h.contains("B_radial") || h.contains("B_tangential") || h.contains("B_normal")
        })?,
    };
    let clean: Vec<[f64; 3]> = rows
        .into_iter()
        .filter(|b| b.iter().all(|x| x.is_finite() && x.abs() < 1e4))
        .collect();
    (clean.len() > 60).then_some(clean)
}

#[derive(Serialize)]
struct MissionAggregate {
    distance_au: f64,
    records: Vec<LogMomentStats>,
}

fn sample_files(dir: &Path, extension: &str, count: usize) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == extension))
        .collect();
    files.sort();
    if files.len() <= count {
        return files;
    }
    // Evenly spaced sample over the sorted listing, endpoints included.
    // Indices truncate (not round) to match numpy linspace->int selection,
    // keeping the file sample identical to the reference survey.
    (0..count)
        .map(|i| {
            let idx = (i as f64 * (files.len() - 1) as f64 / (count - 1) as f64) as usize;
            files[idx].clone()
        })
        .collect()
}

pub fn run(cli: Cli) -> Result<()> {
    let mut aggregate: BTreeMap<&'static str, MissionAggregate> = BTreeMap::new();
    for spec in MISSIONS {
        let files = sample_files(
            &cli.external_dir.join(spec.name),
            spec.pattern,
            cli.files_per_mission,
        );
        let records: Vec<LogMomentStats> = files
            .par_iter()
            .filter_map(|path| {
                let mut rows = read_field(path, spec.reader)?;
                rows.truncate(cli.max_rows);
                let staples = staple_embedding(&rows);
                let assoc = joint_associator_norms(&staples, false);
                let stats = log_moment_stats(&assoc)?;
                (stats.n >= 30).then_some(stats)
            })
            .collect();
        if !records.is_empty() {
            aggregate.insert(
                spec.name,
                MissionAggregate {
                    distance_au: spec.distance_au,
                    records,
                },
            );
        }
    }

    fs::write(&cli.out, serde_json::to_string_pretty(&aggregate)?)
        .with_context(|| format!("write {}", cli.out.display()))?;

    for (name, agg) in &aggregate {
        let sigmas: Vec<f64> = agg.records.iter().map(|r| r.log_std).collect();
        let mean = sigmas.iter().sum::<f64>() / sigmas.len() as f64;
        let std =
            (sigmas.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sigmas.len() as f64).sqrt();
        println!(
            "{name:11} d={:6.1} ok={:3} logsigma={mean:.2}+-{std:.2}",
            agg.distance_au,
            agg.records.len()
        );
    }
    Ok(())
}
