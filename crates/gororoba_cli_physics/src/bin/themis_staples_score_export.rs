//! Export per-sample CD associator scores with Staples-catalog labels for
//! ROC-style benchmarking against conventional detectors.
//!
//! WHY: The thresholded detector in heliosphere-themis-staples-labeled
//! reports one P/R/F1 point per parameter set.  Method comparison for the
//! methods paper needs the full score distributions -- CD associator,
//! |B|-gradient, and rotation angle on identical samples with identical
//! ground-truth labels -- so ROC / PR curves can be traced at every
//! threshold.
//!
//! WHAT: For each THEMIS-A daily FGM CSV listed in the matched-files table,
//! builds the raw 16D staple embedding, computes normalized joint associator
//! norms plus the two conventional baselines, labels each sample within
//! +/- 2 minutes of a catalogued magnetopause crossing on that UTC day, and
//! concatenates everything into one flat CSV of
//! (file_id, assoc, dbdt, rot, bmag, label), where file_id names the
//! source daily file -- the intact cluster a file-aware bootstrap
//! resamples on. A sidecar <out>.files.csv maps file_id to path.
//!
//! HOW:
//!   cargo run --release -p gororoba_cli_physics \
//!     --bin themis-staples-score-export -- \
//!     --matched-files data/output/tha_matched_files.csv \
//!     --catalog data/output/cat_themis_a.csv \
//!     --out data/output/benchmark_scores.csv

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, DurationRound, Utc};
use clap::Parser;
use gororoba_cli_physics::{
    staple_associator::{joint_associator_norms, staple_embedding},
    staple_controls::{SparseCubicTensor, permute_channels, six_sample_baselines},
};
use rayon::prelude::*;
use std::{fs, io::Write, path::PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Export labeled CD associator + baseline scores for THEMIS-A")]
struct Args {
    /// CSV with a `path` column listing daily THEMIS-A FGM files.
    #[arg(long)]
    matched_files: PathBuf,

    /// Crossing catalog CSV with a `TIMESTAMP` column (Staples et al. 2020).
    #[arg(long)]
    catalog: PathBuf,

    /// Output CSV path.
    #[arg(long)]
    out: PathBuf,

    /// Half-width of the positive-label window around each crossing, minutes.
    #[arg(long, default_value_t = 2.0)]
    label_pad_minutes: f64,

    /// Minimum finite samples a daily file must carry to be scored.
    #[arg(long, default_value_t = 500)]
    min_samples: usize,
}

/// One scored sample; label is true within the crossing pad window.
/// The six trailing vectors are the equal-receptive-field controls:
/// classical six-sample statistics plus the two tensor controls
/// (sign-scrambled CD tensor, channel-permuted-input associator).
struct Scored {
    assoc: Vec<f64>,
    dbdt: Vec<f64>,
    rot: Vec<f64>,
    bmag: Vec<f64>,
    label: Vec<bool>,
    cumrot6: Vec<f64>,
    maxrot6: Vec<f64>,
    pvi6: Vec<f64>,
    gram6: Vec<f64>,
    scram: Vec<f64>,
    chperm: Vec<f64>,
}

fn parse_timestamp(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .or_else(|_| DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%:z"))
        .map(|t| t.with_timezone(&Utc))
        .ok()
}

fn read_catalog(path: &PathBuf) -> Result<Vec<DateTime<Utc>>> {
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("open catalog {}", path.display()))?;
    let ts_col = reader
        .headers()?
        .iter()
        .position(|h| h == "TIMESTAMP")
        .context("catalog lacks TIMESTAMP column")?;
    let mut out = Vec::new();
    for record in reader.records() {
        let record = record?;
        if let Some(t) = record.get(ts_col).and_then(parse_timestamp) {
            out.push(t);
        }
    }
    Ok(out)
}

fn score_file(
    path: &str,
    crossings: &[DateTime<Utc>],
    args: &Args,
    scrambled: &SparseCubicTensor,
    cd_tensor: &SparseCubicTensor,
) -> Option<Scored> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .ok()?;
    let mut times: Vec<DateTime<Utc>> = Vec::new();
    let mut rows: Vec<[f64; 3]> = Vec::new();
    for record in reader.records() {
        let record = record.ok()?;
        let t = record.get(0).and_then(parse_timestamp);
        let bx = record.get(1).and_then(|v| v.parse::<f64>().ok());
        let by = record.get(2).and_then(|v| v.parse::<f64>().ok());
        let bz = record.get(3).and_then(|v| v.parse::<f64>().ok());
        if let (Some(t), Some(bx), Some(by), Some(bz)) = (t, bx, by, bz)
            && bx.is_finite()
            && by.is_finite()
            && bz.is_finite()
        {
            times.push(t);
            rows.push([bx, by, bz]);
        }
    }
    // The emptiness check stands on its own: --min-samples 0 must still
    // skip a file that parses to zero valid rows rather than index times[0].
    if rows.is_empty() || rows.len() < args.min_samples {
        return None;
    }

    // Positive labels exist only when this UTC day has catalogued crossings.
    let day_start = times[0].duration_trunc(Duration::days(1)).ok()?;
    let day_end = day_start + Duration::days(1);
    let day_crossings: Vec<DateTime<Utc>> = crossings
        .iter()
        .copied()
        .filter(|t| *t >= day_start && *t < day_end)
        .collect();
    if day_crossings.is_empty() {
        return None;
    }

    let bmag: Vec<f64> = rows
        .iter()
        .map(|b| (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt())
        .collect();

    let staples = staple_embedding(&rows);
    let assoc = joint_associator_norms(&staples, true);
    let n = assoc.len();

    // Staple k spans samples k..k+3 and the associator joins staples
    // k, k+1, k+2, so score k aligns to sample timestamp k+4 -- one past
    // the middle staple's trailing sample, matching the reference export.
    let ts: Vec<DateTime<Utc>> = times[4..4 + n].to_vec();

    let pad = Duration::milliseconds((args.label_pad_minutes * 60_000.0) as i64);
    let label: Vec<bool> = ts
        .iter()
        .map(|t| day_crossings.iter().any(|c| (*t - *c).abs() <= pad))
        .collect();

    // Conventional baselines on the same alignment: |d|B|/dt| and the
    // rotation angle between consecutive unit field vectors.
    let dbdt: Vec<f64> = (0..n).map(|k| (bmag[k + 4] - bmag[k + 3]).abs()).collect();
    let rot: Vec<f64> = (0..n)
        .map(|k| {
            let (p, q) = (&rows[k + 3], &rows[k + 4]);
            let (mp, mq) = (bmag[k + 3] + 1e-30, bmag[k + 4] + 1e-30);
            let cosine = (p[0] * q[0] + p[1] * q[1] + p[2] * q[2]) / (mp * mq);
            cosine.clamp(-1.0, 1.0).acos()
        })
        .collect();
    let bmag_aligned: Vec<f64> = bmag[4..4 + n].to_vec();

    // Window k spans rows k..k+5 and indexes 1:1 with assoc. Rotation
    // and Gram volume use that local window; PVI additionally uses the
    // daily file's RMS increment to normalize its window numerator.
    let base = six_sample_baselines(&rows);
    let scram = scrambled.scores(&staples);
    let permuted_staples = staple_embedding(&permute_channels(&rows, [1, 2, 0]));
    let chperm = cd_tensor.scores(&permuted_staples);
    debug_assert_eq!(base.cum_rotation.len(), n);
    debug_assert_eq!(scram.len(), n);
    debug_assert_eq!(chperm.len(), n);

    let keep: Vec<usize> = (0..n)
        .filter(|&k| assoc[k].is_finite() && dbdt[k].is_finite() && rot[k].is_finite())
        .collect();
    Some(Scored {
        assoc: keep.iter().map(|&k| assoc[k]).collect(),
        dbdt: keep.iter().map(|&k| dbdt[k]).collect(),
        rot: keep.iter().map(|&k| rot[k]).collect(),
        bmag: keep.iter().map(|&k| bmag_aligned[k]).collect(),
        label: keep.iter().map(|&k| label[k]).collect(),
        cumrot6: keep.iter().map(|&k| base.cum_rotation[k]).collect(),
        maxrot6: keep.iter().map(|&k| base.max_rotation[k]).collect(),
        pvi6: keep.iter().map(|&k| base.max_pvi[k]).collect(),
        gram6: keep.iter().map(|&k| base.max_gram_volume[k]).collect(),
        scram: keep.iter().map(|&k| scram[k]).collect(),
        chperm: keep.iter().map(|&k| chperm[k]).collect(),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();

    let crossings = read_catalog(&args.catalog)?;
    anyhow::ensure!(!crossings.is_empty(), "catalog parsed to zero crossings");

    let mut reader = csv::Reader::from_path(&args.matched_files)
        .with_context(|| format!("open {}", args.matched_files.display()))?;
    let path_col = reader
        .headers()?
        .iter()
        .position(|h| h == "path")
        .context("matched-files table lacks a path column")?;
    let files: Vec<String> = reader
        .records()
        .filter_map(|r| r.ok().and_then(|r| r.get(path_col).map(str::to_owned)))
        .collect();

    // Pair every score block with its source path so the CSV carries the
    // cluster identity a file-aware bootstrap resamples on. rayon's
    // filter_map preserves input order on collect, so file_id is stable
    // across reruns of the same matched-files table.
    // Tensor controls are file-independent: build once, share across
    // the rayon workers. The scramble seed is fixed so the control
    // column is reproducible.
    let table = cd_kernel::mult_table::CdMultTable::generate(16);
    let cd_tensor = SparseCubicTensor::from_associator(&table);
    let scrambled = cd_tensor.sign_scrambled(42);

    let scored: Vec<(&String, Scored)> = files
        .par_iter()
        .filter_map(|f| score_file(f, &crossings, &args, &scrambled, &cd_tensor).map(|s| (f, s)))
        .collect();

    let total: usize = scored.iter().map(|(_, s)| s.assoc.len()).sum();
    let positives: usize = scored
        .iter()
        .map(|(_, s)| s.label.iter().filter(|&&l| l).count())
        .sum();

    let mut out = std::io::BufWriter::new(
        fs::File::create(&args.out).with_context(|| format!("create {}", args.out.display()))?,
    );
    writeln!(
        out,
        "file_id,assoc,dbdt,rot,bmag,label,cumrot6,maxrot6,pvi6,gram6,scram,chperm"
    )?;
    for (file_id, (_, s)) in scored.iter().enumerate() {
        for k in 0..s.assoc.len() {
            writeln!(
                out,
                "{},{:e},{:e},{:e},{:e},{},{:e},{:e},{:e},{:e},{:e},{:e}",
                file_id,
                s.assoc[k],
                s.dbdt[k],
                s.rot[k],
                s.bmag[k],
                u8::from(s.label[k]),
                s.cumrot6[k],
                s.maxrot6[k],
                s.pvi6[k],
                s.gram6[k],
                s.scram[k],
                s.chperm[k]
            )?;
        }
    }

    // Sidecar map ties each file_id back to its source path for
    // provenance and for event-level attribution later.
    let map_path = args.out.with_extension("files.csv");
    let mut map_out = std::io::BufWriter::new(
        fs::File::create(&map_path).with_context(|| format!("create {}", map_path.display()))?,
    );
    writeln!(map_out, "file_id,path")?;
    for (file_id, (path, _)) in scored.iter().enumerate() {
        writeln!(map_out, "{},{}", file_id, path)?;
    }

    println!(
        "files_ok={} total_pts={} positives={} ({:.2}%)",
        scored.len(),
        total,
        positives,
        100.0 * positives as f64 / total.max(1) as f64
    );
    Ok(())
}
