//! Shared substrate for the preregistered staple-associator controls:
//! the THEMIS-A daily-file loader with Staples-catalog labels, a
//! deterministic file-stratified subsample, layout-parameterized staple
//! embeddings, and rank-based ROC-AUC.
//!
//! Labels follow `themis-staples-score-export`: score index k joins
//! staples k, k+1, k+2, so it sits on sample timestamp k+4 and is positive
//! within `pad_minutes` of a catalogued crossing on that UTC day. Files
//! whose UTC day carries no catalogued crossing contribute nothing, and
//! a file with fewer than `min_samples` parsed rows is skipped, exactly
//! as the export does, so every experiment here scores the E-239 sample.

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, DurationRound, Utc};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::path::Path;

use crate::staple_associator::{STAPLE_DIM, STAPLE_LAGS};

/// One THEMIS-A daily file with its crossing labels on the score index.
pub struct LabeledFile {
    pub file_id: usize,
    pub path: String,
    pub rows: Vec<[f64; 3]>,
    /// `rows.len() - 5` entries; entry k labels the triple of staples k..k+2.
    pub labels: Vec<bool>,
}

impl LabeledFile {
    /// Number of score positions.
    pub fn score_len(&self) -> usize {
        self.rows.len().saturating_sub(STAPLE_LAGS + 1)
    }
}

pub fn parse_timestamp(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .or_else(|_| DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%:z"))
        .map(|t| t.with_timezone(&Utc))
        .ok()
}

/// Crossing timestamps from a Staples et al. (2020) catalog CSV.
pub fn read_catalog(path: &Path) -> Result<Vec<DateTime<Utc>>> {
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

/// Daily-file paths from the matched-files table (`path` column).
pub fn read_matched_paths(path: &Path) -> Result<Vec<String>> {
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("open {}", path.display()))?;
    let path_col = reader
        .headers()?
        .iter()
        .position(|h| h == "path")
        .context("matched-files table lacks a path column")?;
    Ok(reader
        .records()
        .filter_map(|r| r.ok().and_then(|r| r.get(path_col).map(str::to_owned)))
        .collect())
}

fn load_one(
    file_id: usize,
    path: &str,
    crossings: &[DateTime<Utc>],
    pad_minutes: f64,
    min_samples: usize,
) -> Option<LabeledFile> {
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
    if rows.is_empty() || rows.len() < min_samples || rows.len() < STAPLE_LAGS + 2 {
        return None;
    }
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
    let pad = Duration::milliseconds((pad_minutes * 60_000.0) as i64);
    let n = rows.len() - (STAPLE_LAGS + 1);
    let labels = times[4..4 + n]
        .iter()
        .map(|t| day_crossings.iter().any(|c| (*t - *c).abs() <= pad))
        .collect();
    Some(LabeledFile {
        file_id,
        path: path.to_string(),
        rows,
        labels,
    })
}

/// Load every matched file in parallel; file ids follow the table order.
pub fn load_labeled_files(
    matched_files: &Path,
    catalog: &Path,
    pad_minutes: f64,
    min_samples: usize,
) -> Result<Vec<LabeledFile>> {
    let crossings = read_catalog(catalog)?;
    anyhow::ensure!(!crossings.is_empty(), "catalog parsed to zero crossings");
    let paths = read_matched_paths(matched_files)?;
    let mut files: Vec<LabeledFile> = paths
        .par_iter()
        .enumerate()
        .filter_map(|(id, p)| load_one(id, p, &crossings, pad_minutes, min_samples))
        .collect();
    files.sort_by_key(|f| f.file_id);
    Ok(files)
}

/// Score indices kept for a file: every positive, and each negative with
/// probability `neg_fraction` from a ChaCha8 stream keyed on
/// (seed, file_id). Pooled AUC on this subsample is an unbiased estimate
/// of the full pooled AUC, because AUC compares every positive with an
/// independently thinned negative set.
pub fn stratified_keep(
    labels: &[bool],
    file_id: usize,
    seed: u64,
    neg_fraction: f64,
) -> Vec<usize> {
    let mut rng =
        ChaCha8Rng::seed_from_u64(seed ^ (file_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    labels
        .iter()
        .enumerate()
        .filter(|&(_, &label)| label || rng.random::<f64>() < neg_fraction)
        .map(|(k, _)| k)
        .collect()
}

/// Staple embedding under an explicit layout. Slot `s` of the 16-vector
/// (s = 4 * lag_slot + channel_slot) receives lag `lag_perm[lag_slot]`
/// of the four samples i-3..i and component `comp_perm[channel_slot]`
/// of (Bx, By, Bz); channel slot 3 is the magnitude and stays fixed.
/// The identity permutations reproduce `staple_embedding`.
pub fn staple_embedding_layout(
    rows: &[[f64; 3]],
    lag_perm: [usize; STAPLE_LAGS],
    comp_perm: [usize; 3],
) -> Vec<[f64; STAPLE_DIM]> {
    if rows.len() < STAPLE_LAGS {
        return Vec::new();
    }
    let feat: Vec<[f64; 4]> = rows
        .iter()
        .map(|b| {
            let mag = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
            [b[comp_perm[0]], b[comp_perm[1]], b[comp_perm[2]], mag]
        })
        .collect();
    (STAPLE_LAGS - 1..rows.len())
        .map(|i| {
            let mut v = [0.0_f64; STAPLE_DIM];
            for (lag_slot, &lag) in lag_perm.iter().enumerate() {
                v[4 * lag_slot..4 * lag_slot + 4]
                    .copy_from_slice(&feat[i - (STAPLE_LAGS - 1) + lag]);
            }
            v
        })
        .collect()
}

/// All permutations of 0..N in lexicographic order.
pub fn permutations<const N: usize>() -> Vec<[usize; N]> {
    fn go<const N: usize>(
        prefix: &mut Vec<usize>,
        used: &mut [bool; N],
        out: &mut Vec<[usize; N]>,
    ) {
        if prefix.len() == N {
            let mut arr = [0usize; N];
            arr.copy_from_slice(prefix);
            out.push(arr);
            return;
        }
        for i in 0..N {
            if !used[i] {
                used[i] = true;
                prefix.push(i);
                go(prefix, used, out);
                prefix.pop();
                used[i] = false;
            }
        }
    }
    let mut out = Vec::new();
    go::<N>(&mut Vec::with_capacity(N), &mut [false; N], &mut out);
    out
}

/// Average-rank Mann-Whitney ROC-AUC; ties share their mean rank.
pub fn rank_auc(scores: &[f64], labels: &[bool]) -> f64 {
    let n = scores.len();
    let n_pos = labels.iter().filter(|&&l| l).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.par_sort_unstable_by(|&a, &b| {
        scores[a as usize]
            .partial_cmp(&scores[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank_sum_pos = 0.0_f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[order[j + 1] as usize] == scores[order[i] as usize] {
            j += 1;
        }
        // ranks are 1-based; the tie block i..=j shares the mean rank
        let mean_rank = (i + j + 2) as f64 / 2.0;
        let pos_in_block = (i..=j).filter(|&t| labels[order[t] as usize]).count();
        rank_sum_pos += mean_rank * pos_in_block as f64;
        i = j + 1;
    }
    (rank_sum_pos - (n_pos as f64) * (n_pos as f64 + 1.0) / 2.0) / (n_pos as f64 * n_neg as f64)
}

/// Percentile of a sorted sample by linear interpolation.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staple_associator::staple_embedding;

    #[test]
    fn rank_auc_matches_hand_computation() {
        let scores = [0.1, 0.4, 0.35, 0.8];
        let labels = [false, false, true, true];
        // positives beat negatives in 3 of 4 pairs
        assert!((rank_auc(&scores, &labels) - 0.75).abs() < 1e-12);
        let tied = [0.5, 0.5, 0.5, 0.5];
        assert!((rank_auc(&tied, &labels) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn identity_layout_reproduces_the_canonical_embedding() {
        let rows: Vec<[f64; 3]> = (0..12)
            .map(|i| [i as f64, (i * i) as f64 * 0.1, 1.0 - i as f64 * 0.05])
            .collect();
        let a = staple_embedding(&rows);
        let b = staple_embedding_layout(&rows, [0, 1, 2, 3], [0, 1, 2]);
        assert_eq!(a, b);
        let c = staple_embedding_layout(&rows, [3, 2, 1, 0], [2, 0, 1]);
        assert_ne!(a, c);
        assert_eq!(c[0][0], rows[3][2]);
        assert_eq!(c[0][3], a[0][15]);
    }

    #[test]
    fn permutation_counts_are_factorials() {
        assert_eq!(permutations::<3>().len(), 6);
        assert_eq!(permutations::<4>().len(), 24);
        assert_eq!(permutations::<4>()[0], [0, 1, 2, 3]);
    }

    #[test]
    fn stratified_keep_keeps_every_positive_and_is_deterministic() {
        let labels: Vec<bool> = (0..1000).map(|i| i % 50 == 0).collect();
        let a = stratified_keep(&labels, 7, 42, 0.1);
        let b = stratified_keep(&labels, 7, 42, 0.1);
        assert_eq!(a, b);
        assert!(
            labels
                .iter()
                .enumerate()
                .filter(|(_, l)| **l)
                .all(|(k, _)| a.contains(&k))
        );
        let negatives = a.iter().filter(|&&k| !labels[k]).count();
        assert!(negatives > 50 && negatives < 150, "{negatives}");
    }
}
