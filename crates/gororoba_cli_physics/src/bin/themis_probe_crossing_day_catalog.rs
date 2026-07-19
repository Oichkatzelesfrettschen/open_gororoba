//! Per-probe THEMIS crossing-day catalog generator.
//!
//! The Staples et al. 2020 magnetopause crossing list
//! (doi:10.1029/2019JA027190) pools five THEMIS probes in one file. The
//! staples benchmark consumes per-probe day catalogs: each benchmark run
//! binds one probe, one set of daily FGM files, and one crossing list, so
//! probe replication stays independent instead of silently pooling.
//!
//! For the selected probe this binary emits two CSVs:
//!
//! 1. `<out-matched>`: `year,doy,path` rows for crossing days whose daily
//!    FGM cache file `th<p>_fgm_<year>_<doy>.csv` exists on disk. The
//!    schema matches the `--matched-files` input of
//!    `themis-staples-score-export`.
//! 2. `<out-days>`: one row per calendar day that has either a catalogued
//!    crossing or a cached data file:
//!    `probe,year,doy,date,n_crossings,first_crossing_utc,last_crossing_utc,data_file_present`.
//!    Rows with `n_crossings=0` are crossing-free control-day candidates;
//!    rows with `data_file_present=0` are the download work list for
//!    probe replication.
//!
//! Usage:
//!   themis-probe-crossing-day-catalog --probe c \
//!     --catalog data/external/crossing_lists/themis_mp_crossings_v2.txt \
//!     --data-dir data/external \
//!     --out-matched data/output/thc_matched_files.csv \
//!     --out-days data/output/thc_crossing_day_catalog.csv

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Parser;
use data_core::catalogs::themis::parse_staples_crossing_catalog;

#[derive(Parser)]
#[command(about = "Generate per-probe THEMIS crossing-day catalogs from the Staples 2020 list")]
struct Args {
    /// THEMIS probe letter: a, b, c, d, e.
    #[arg(long)]
    probe: String,

    /// Staples et al. 2020 crossing list (tab-separated txt).
    #[arg(long, default_value = "data/external/crossing_lists/themis_mp_crossings_v2.txt")]
    catalog: PathBuf,

    /// External data root containing the `themis/` daily FGM cache.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,

    /// Output CSV of crossing days with cached data (year,doy,path).
    #[arg(long)]
    out_matched: PathBuf,

    /// Output CSV of all crossing/data days with coverage flags.
    #[arg(long)]
    out_days: PathBuf,

    /// Inclusive catalog window start (the Staples list spans 2007-2016).
    #[arg(long, default_value = "2007-01-01")]
    start_date: String,

    /// Inclusive catalog window end.
    #[arg(long, default_value = "2016-12-31")]
    end_date: String,
}

/// Per-day accumulator: crossing count plus first/last crossing timestamps.
#[derive(Default)]
struct DayEntry {
    n_crossings: usize,
    first_crossing: Option<chrono::NaiveDateTime>,
    last_crossing: Option<chrono::NaiveDateTime>,
    data_file: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let probe_char = args
        .probe
        .trim()
        .to_lowercase()
        .chars()
        .next()
        .context("--probe must be a single letter (a-e)")?;
    if !('a'..='e').contains(&probe_char) {
        bail!("--probe must be one of a, b, c, d, e (got {probe_char})");
    }
    let stem = format!("th{probe_char}");

    let start = NaiveDate::parse_from_str(&args.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad --start-date {}", args.start_date))?;
    let end = NaiveDate::parse_from_str(&args.end_date, "%Y-%m-%d")
        .with_context(|| format!("bad --end-date {}", args.end_date))?;

    let content = fs::read_to_string(&args.catalog)
        .with_context(|| format!("reading {}", args.catalog.display()))?;
    // pad_minutes=0 keeps interval start == crossing timestamp, so each
    // event contributes exactly one point-in-time to its UTC day.
    let crossings = parse_staples_crossing_catalog(&content, probe_char, start, end, 0);
    if crossings.is_empty() {
        bail!(
            "no {} crossings parsed from {} in [{start}, {end}]",
            stem.to_uppercase(),
            args.catalog.display()
        );
    }

    let mut days: BTreeMap<(i32, u32), DayEntry> = BTreeMap::new();
    for ev in &crossings {
        let day = ev.start.date();
        let entry = days.entry((day.year(), day.ordinal())).or_default();
        entry.n_crossings += 1;
        let ts = ev.start;
        entry.first_crossing = Some(entry.first_crossing.map_or(ts, |cur| cur.min(ts)));
        entry.last_crossing = Some(entry.last_crossing.map_or(ts, |cur| cur.max(ts)));
    }
    let crossing_day_count = days.len();

    // Sweep the on-disk cache once: attach paths to crossing days and add
    // crossing-free data days as control-day candidate rows.
    let themis_dir = args.data_dir.join("themis");
    let prefix = format!("{stem}_fgm_");
    let mut data_day_count = 0usize;
    if themis_dir.is_dir() {
        for dirent in fs::read_dir(&themis_dir)? {
            let path = dirent?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(rest) = name.strip_prefix(&prefix).and_then(|r| r.strip_suffix(".csv"))
            else {
                continue;
            };
            let mut parts = rest.splitn(2, '_');
            let (Some(y), Some(d)) = (parts.next(), parts.next()) else {
                continue;
            };
            let (Ok(year), Ok(doy)) = (y.parse::<i32>(), d.parse::<u32>()) else {
                continue;
            };
            data_day_count += 1;
            days.entry((year, doy)).or_default().data_file = Some(path);
        }
    }

    let mut matched = String::from("year,doy,path\n");
    let mut day_rows = String::from(
        "probe,year,doy,date,n_crossings,first_crossing_utc,last_crossing_utc,data_file_present\n",
    );
    let mut matched_count = 0usize;
    let mut missing_data_days = 0usize;
    let mut control_candidate_days = 0usize;
    for (&(year, doy), entry) in &days {
        let date = NaiveDate::from_yo_opt(year, doy)
            .with_context(|| format!("invalid year/doy {year}/{doy}"))?;
        let fmt_ts = |ts: Option<chrono::NaiveDateTime>| {
            ts.map_or(String::new(), |t| format!("{}Z", t.format("%Y-%m-%dT%H:%M:%S")))
        };
        day_rows.push_str(&format!(
            "{stem},{year},{doy:03},{date},{},{},{},{}\n",
            entry.n_crossings,
            fmt_ts(entry.first_crossing),
            fmt_ts(entry.last_crossing),
            u8::from(entry.data_file.is_some()),
        ));
        match (&entry.data_file, entry.n_crossings) {
            (Some(path), 1..) => {
                matched.push_str(&format!("{year},{doy:03},{}\n", path.display()));
                matched_count += 1;
            }
            (None, 1..) => missing_data_days += 1,
            (Some(_), 0) => control_candidate_days += 1,
            (None, 0) => unreachable!("day rows originate from a crossing or a data file"),
        }
    }

    fs::write(&args.out_matched, matched)
        .with_context(|| format!("writing {}", args.out_matched.display()))?;
    fs::write(&args.out_days, day_rows)
        .with_context(|| format!("writing {}", args.out_days.display()))?;

    println!(
        "probe=TH{} crossings={} crossing_days={} data_days_on_disk={}\n\
         matched_days={} missing_data_days={} control_candidate_days={}\n\
         matched -> {}\n\
         days    -> {}",
        probe_char.to_ascii_uppercase(),
        crossings.len(),
        crossing_day_count,
        data_day_count,
        matched_count,
        missing_data_days,
        control_candidate_days,
        args.out_matched.display(),
        args.out_days.display(),
    );
    Ok(())
}
