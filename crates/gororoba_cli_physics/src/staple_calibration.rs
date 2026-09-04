//! Provenance-checked calibration data on a common preceding-sample mask.

use std::{collections::HashSet, fs::File, io::Read, path::Path};

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Duration, DurationRound, Utc};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::{
    staple_associator::{joint_associator_norms, staple_embedding},
    staple_controls::six_sample_baselines,
};

const HEADER: [&str; 12] = [
    "file_id", "assoc", "dbdt", "rot", "bmag", "label", "cumrot6", "maxrot6", "pvi6", "gram6",
    "scram", "chperm",
];

#[derive(Debug, Serialize)]
pub struct FileEvidence {
    pub id: u16,
    pub path: String,
    pub increment_square_sum: f64,
    pub increment_count: u64,
    pub sha256: String,
    pub finite_samples: usize,
    pub discarded_raw_rows: usize,
    pub scored_rows: usize,
    pub retained_rows: usize,
    pub first_timestamp: String,
    pub last_timestamp: String,
    pub min_cadence_milliseconds: i64,
    pub max_cadence_milliseconds: i64,
    pub nonpositive_cadence_count: usize,
    pub equal_timestamp_pair_count: usize,
    pub backward_timestamp_pair_count: usize,
    pub positive_submillisecond_pair_count: usize,
    pub kept_index_label_sha256: String,
}

pub struct PreparedDataset {
    pub features: Vec<f32>,
    pub labels: Vec<u8>,
    pub file_index: Vec<u16>,
    pub pvi_numerator: Vec<f64>,
    pub raw_assoc: Vec<f64>,
    pub daily_pvi: Vec<f64>,
    pub rolling_pvi: Vec<f64>,
    pub rolling_log_pvi: Vec<f32>,
    pub dbdt: Vec<f64>,
    pub files: Vec<FileEvidence>,
    pub evidence: serde_json::Value,
}

/// Preserve equal-time input order while rejecting exact backward timestamps.
/// Sample-count windows establish preceding parsed rows; equal timestamps alone
/// establish neither strict temporal precedence nor streaming availability.
pub fn validate_sample_order(files: &[FileEvidence]) -> Result<()> {
    for file in files {
        ensure!(
            file.backward_timestamp_pair_count == 0,
            "pre-window sample-order admission rejects {} backward timestamp pairs in file {}",
            file.backward_timestamp_pair_count,
            file.id
        );
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let length = file.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        digest.update(&buffer[..length]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn timestamp(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .or_else(|_| DateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%:z"))
        .map(|time| time.with_timezone(&Utc))
        .ok()
}

fn evidence_path(path: &Path, root: &Path, role: &str) -> String {
    if let Ok(relative) = path.strip_prefix(root) {
        relative.to_string_lossy().into_owned()
    } else if path.is_relative() {
        path.to_string_lossy().into_owned()
    } else {
        role.to_owned()
    }
}

fn validate_map(entries: &[(u16, String)]) -> Result<()> {
    ensure!(!entries.is_empty(), "empty exported file map");
    let mut paths = HashSet::new();
    for (index, (id, path)) in entries.iter().enumerate() {
        ensure!(
            usize::from(*id) == index,
            "exported file IDs must be dense and ordered: row {index}, ID {id}"
        );
        ensure!(paths.insert(path), "duplicate exported path {path}");
        ensure!(!path.is_empty(), "empty exported path");
    }
    Ok(())
}

/// Numerator consumes five increments; calibration precedes the first scored
/// increment in parsed row order, including distinct vectors at equal times.
fn rolling_score(increments: &[f64], index: usize, window: usize) -> Result<Option<(f64, f64)>> {
    ensure!(window > 0, "rolling context must contain increments");
    if index < window {
        return Ok(None);
    }
    ensure!(index + 5 <= increments.len(), "incomplete numerator window");
    let square_mean = increments[index - window..index]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        / window as f64;
    ensure!(
        square_mean.is_finite() && square_mean > 0.0,
        "invalid preceding-sample RMS at score {index}"
    );
    let numerator = increments[index..index + 5]
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let score = numerator / square_mean.sqrt();
    ensure!(score.is_finite(), "nonfinite rolling PVI at score {index}");
    Ok(Some((numerator, score)))
}

struct RawScores {
    evidence: FileEvidence,
    values: Vec<[f64; 8]>,
    labels: Vec<u8>,
    indices: Vec<usize>,
    rolling: Vec<Option<(f64, f64)>>,
}

fn reconstruct(
    root: &Path,
    id: u16,
    path: &str,
    crossings: &[DateTime<Utc>],
    window: usize,
) -> Result<RawScores> {
    let resolved = root.join(path);
    let raw_bytes = std::fs::read(&resolved)?;
    let sha256 = Sha256::digest(&raw_bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(raw_bytes.as_slice());
    let mut times = Vec::new();
    let mut rows = Vec::new();
    let mut discarded = 0;
    for record in reader.records() {
        let record = record?;
        let time = record.get(0).and_then(timestamp);
        let components: Option<Vec<f64>> = (1..4)
            .map(|index| record.get(index)?.parse::<f64>().ok())
            .collect();
        if let (Some(time), Some(components)) = (time, components)
            && components.iter().all(|value| value.is_finite())
        {
            times.push(time);
            rows.push([components[0], components[1], components[2]]);
        } else {
            discarded += 1;
        }
    }
    ensure!(
        rows.len() >= 500,
        "exported file {id} has fewer than 500 finite samples"
    );
    let day_start = times[0].duration_trunc(Duration::days(1))?;
    let day_crossings: Vec<_> = crossings
        .iter()
        .filter(|&&time| time >= day_start && time < day_start + Duration::days(1))
        .collect();
    ensure!(
        !day_crossings.is_empty(),
        "exported file {id} has zero daily crossings"
    );
    let magnitude: Vec<f64> = rows
        .iter()
        .map(|row| (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt())
        .collect();
    let increments: Vec<f64> = rows
        .windows(2)
        .map(|pair| {
            let delta = [
                pair[1][0] - pair[0][0],
                pair[1][1] - pair[0][1],
                pair[1][2] - pair[0][2],
            ];
            (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
        })
        .collect();
    let increment_square_sum: f64 = increments.iter().map(|value| value * value).sum();
    ensure!(
        increment_square_sum.is_finite() && increment_square_sum > 0.0,
        "invalid file RMS sufficient statistic in exported file {id}"
    );
    let assoc = joint_associator_norms(&staple_embedding(&rows), true);
    let controls = six_sample_baselines(&rows);
    let mut values = Vec::new();
    let mut labels = Vec::new();
    let mut indices = Vec::new();
    let mut rolling = Vec::new();
    let mut mapping = Sha256::new();
    for (index, &associator) in assoc.iter().enumerate() {
        let dbdt = (magnitude[index + 4] - magnitude[index + 3]).abs();
        let left = rows[index + 3];
        let right = rows[index + 4];
        let cosine = (left[0] * right[0] + left[1] * right[1] + left[2] * right[2])
            / ((magnitude[index + 3] + 1e-30) * (magnitude[index + 4] + 1e-30));
        let rotation = cosine.clamp(-1.0, 1.0).acos();
        if !(associator.is_finite() && dbdt.is_finite() && rotation.is_finite()) {
            continue;
        }
        let label = u8::from(
            day_crossings
                .iter()
                .any(|&&crossing| (times[index + 4] - crossing).abs() <= Duration::minutes(2)),
        );
        let calibrated = rolling_score(&increments, index, window)?;
        if calibrated.is_some() {
            mapping.update(id.to_le_bytes());
            mapping.update((index as u64).to_le_bytes());
            mapping.update(times[index + 4].timestamp_millis().to_le_bytes());
            mapping.update([label]);
        }
        values.push([
            associator,
            dbdt,
            rotation,
            magnitude[index + 4],
            controls.cum_rotation[index],
            controls.max_rotation[index],
            controls.max_pvi[index],
            controls.max_gram_volume[index],
        ]);
        labels.push(label);
        indices.push(index);
        rolling.push(calibrated);
    }
    let cadence: Vec<i64> = times
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).num_milliseconds())
        .collect();
    Ok(RawScores {
        evidence: FileEvidence {
            id,
            path: path.to_owned(),
            increment_square_sum,
            increment_count: increments.len() as u64,
            sha256,
            finite_samples: rows.len(),
            discarded_raw_rows: discarded,
            scored_rows: values.len(),
            retained_rows: rolling.iter().filter(|value| value.is_some()).count(),
            first_timestamp: times[0].to_rfc3339(),
            last_timestamp: times[times.len() - 1].to_rfc3339(),
            min_cadence_milliseconds: *cadence.iter().min().context("empty cadence")?,
            max_cadence_milliseconds: *cadence.iter().max().context("empty cadence")?,
            nonpositive_cadence_count: cadence.iter().filter(|&&value| value <= 0).count(),
            equal_timestamp_pair_count: times.windows(2).filter(|pair| pair[1] == pair[0]).count(),
            backward_timestamp_pair_count: times
                .windows(2)
                .filter(|pair| pair[1] < pair[0])
                .count(),
            positive_submillisecond_pair_count: times
                .windows(2)
                .filter(|pair| pair[1] > pair[0] && pair[1] - pair[0] < Duration::milliseconds(1))
                .count(),
            kept_index_label_sha256: mapping
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        },
        values,
        labels,
        indices,
        rolling,
    })
}

/// Reconstruct each exported daily file and reject score, label, or order drift.
pub fn prepare(
    input_root: &Path,
    scores: &Path,
    file_map: &Path,
    catalog: &Path,
    rolling_window: usize,
) -> Result<PreparedDataset> {
    ensure!(rolling_window > 0, "rolling context must be positive");
    let mut map_reader = csv::Reader::from_path(file_map)?;
    ensure!(
        map_reader.headers()?.iter().eq(["file_id", "path"]),
        "unexpected exported file-map header"
    );
    let entries: Vec<(u16, String)> = map_reader
        .records()
        .map(|record| {
            let record = record?;
            Ok((record[0].parse()?, record[1].to_owned()))
        })
        .collect::<Result<_>>()?;
    validate_map(&entries)?;
    let mut catalog_reader = csv::Reader::from_path(catalog)?;
    let timestamp_column = catalog_reader
        .headers()?
        .iter()
        .position(|header| header == "TIMESTAMP")
        .context("catalog lacks TIMESTAMP")?;
    let mut crossings = Vec::new();
    let mut discarded_catalog_rows = 0;
    for record in catalog_reader.records() {
        let record = record?;
        if let Some(time) = record.get(timestamp_column).and_then(timestamp) {
            crossings.push(time);
        } else {
            discarded_catalog_rows += 1;
        }
    }
    ensure!(!crossings.is_empty(), "catalog has zero parsed crossings");
    let mut reader = csv::Reader::from_path(scores)?;
    ensure!(
        reader.headers()?.iter().eq(HEADER),
        "unexpected retained score header"
    );
    let mut records = reader.records();
    let mut output = PreparedDataset {
        features: Vec::new(),
        labels: Vec::new(),
        file_index: Vec::new(),
        pvi_numerator: Vec::new(),
        raw_assoc: Vec::new(),
        daily_pvi: Vec::new(),
        rolling_pvi: Vec::new(),
        rolling_log_pvi: Vec::new(),
        dbdt: Vec::new(),
        files: Vec::new(),
        evidence: serde_json::Value::Null,
    };
    let mut row_count = 0;
    let mut positives_before_warmup = 0_usize;
    for batch in entries.chunks(16) {
        let reconstructed: Vec<Result<RawScores>> = batch
            .par_iter()
            .map(|(id, path)| {
                reconstruct(input_root, *id, path, &crossings, rolling_window)
                    .with_context(|| format!("reconstruct file {id}: {path}"))
            })
            .collect();
        for raw in reconstructed {
            let raw = raw?;
            for (position, expected) in raw.values.iter().enumerate() {
                let record = records
                    .next()
                    .context("score CSV ended before reconstructed samples")??;
                row_count += 1;
                let id: u16 = record[0].parse()?;
                ensure!(
                    id == raw.evidence.id,
                    "file/order drift at score row {row_count}"
                );
                let label: u8 = record[5].parse()?;
                ensure!(
                    matches!(&record[5], "0" | "1"),
                    "label must be exactly 0 or 1 at score row {row_count}"
                );
                ensure!(
                    label <= 1 && label == raw.labels[position],
                    "label drift at score row {row_count}"
                );
                positives_before_warmup += usize::from(label);
                for column in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11] {
                    let value: f64 = record[column].parse()?;
                    ensure!(
                        value.is_finite() && (column >= 10 || value >= 0.0),
                        "invalid score at row {row_count}, column {column}"
                    );
                }
                for (column, &recomputed) in [1, 2, 3, 4, 6, 7, 8, 9].iter().zip(expected) {
                    let retained: f64 = record[*column].parse()?;
                    ensure!(
                        recomputed.is_finite()
                            && (retained - recomputed).abs()
                                <= 1e-12 + 1e-10 * retained.abs().max(recomputed.abs()),
                        "numeric drift file {id} index {} column {column}: retained {retained}, recomputed {recomputed}",
                        raw.indices[position]
                    );
                }
                let Some((numerator, rolling)) = raw.rolling[position] else {
                    continue;
                };
                for feature in [
                    expected[1],
                    expected[2],
                    expected[4],
                    expected[5],
                    expected[6],
                    expected[7],
                    expected[0],
                ] {
                    output.features.push((feature + 1e-12).ln() as f32);
                }
                output.labels.push(label);
                output.file_index.push(id);
                output.pvi_numerator.push(numerator);
                output.raw_assoc.push(expected[0]);
                output.daily_pvi.push(expected[6]);
                output.rolling_pvi.push(rolling);
                output.rolling_log_pvi.push((rolling + 1e-12).ln() as f32);
                output.dbdt.push(expected[1]);
            }
            eprintln!(
                "validated file {}: {} score rows, {} retained",
                raw.evidence.id, raw.evidence.scored_rows, raw.evidence.retained_rows
            );
            output.files.push(raw.evidence);
        }
    }
    ensure!(
        records.next().is_none(),
        "score CSV has trailing rows outside exported file map"
    );
    let excluded_ids: Vec<u16> = output
        .files
        .iter()
        .filter(|file| file.retained_rows == 0)
        .map(|file| file.id)
        .collect();
    output.evidence = json!({
        "scores_path": evidence_path(scores, input_root, "scores-input"),
        "scores_sha256": hash_file(scores)?,
        "file_map_path": evidence_path(file_map, input_root, "file-map-input"),
        "file_map_sha256": hash_file(file_map)?,
        "catalog_path": evidence_path(catalog, input_root, "catalog-input"),
        "catalog_sha256": hash_file(catalog)?,
        "catalog_parsed_crossings": crossings.len(),
        "catalog_discarded_rows": discarded_catalog_rows,
        "exported_files": entries.len(),
        "rows_before_warmup": row_count,
        "rows_after_warmup": output.labels.len(),
        "positives_before_warmup": positives_before_warmup,
        "positives_after_warmup": output.labels.iter().map(|&label| usize::from(label)).sum::<usize>(),
        "warmup_excluded_rows": row_count-output.labels.len(),
        "excluded_file_ids": excluded_ids,
        "rolling_context_increments": rolling_window,
        "rolling_context": "increments j in [k-W,k); increment j joins samples j,j+1",
        "numerator": "maximum norm of increments j in [k,k+5)",
        "label_alignment": "sample k+4; inclusive 2 minute pad; crossings restricted to UTC day of first finite sample",
        "cadence_boundary": "preceding parsed finite rows; discarded rows can bridge cadence gaps; per-file extrema and historical nonpositive counts use truncated milliseconds; separate equality, backward and positive-submillisecond counts use exact timestamps",
        "equal_time_boundary": "Equal-time vectors remain distinct samples in input row order. Equal-time rows establish within-timestamp order only; strict temporal precedence and streaming availability remain unproven.",
        "mapping_hash_encoding": "per retained row: little-endian u16 file ID, u64 score index, i64 aligned Unix milliseconds, u8 label"
    });
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AdmissionFixture(std::path::PathBuf);

    impl Drop for AdmissionFixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).expect("remove owned admission fixture");
        }
    }

    fn admission_fixture() -> AdmissionFixture {
        admission_fixture_with_timestamp_delta(Duration::seconds(1))
    }

    fn admission_fixture_with_timestamp_delta(delta: Duration) -> AdmissionFixture {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "staples-calibration-admission-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let fixture = AdmissionFixture(directory);
        let start = timestamp("2016-09-01T00:00:00Z").unwrap();
        let mut raw = csv::WriterBuilder::new()
            .has_headers(false)
            .from_path(fixture.0.join("raw.csv"))
            .unwrap();
        for index in 0..520 {
            let coordinate = f64::from(index);
            let sample_time = if index == 300 {
                start + Duration::seconds(299) + delta
            } else {
                start + Duration::seconds(i64::from(index))
            };
            raw.write_record([
                sample_time.to_rfc3339(),
                (2.0 + (coordinate * 0.03).sin()).to_string(),
                (1.0 + (coordinate * 0.07).cos()).to_string(),
                (0.5 + coordinate * 0.002).to_string(),
            ])
            .unwrap();
        }
        raw.flush().unwrap();
        let crossing = start + Duration::seconds(400);
        std::fs::write(
            fixture.0.join("catalog.csv"),
            format!("TIMESTAMP\n{}\n", crossing.to_rfc3339()),
        )
        .unwrap();
        std::fs::write(fixture.0.join("map.csv"), "file_id,path\n0,raw.csv\n").unwrap();
        let reconstructed = reconstruct(&fixture.0, 0, "raw.csv", &[crossing], 256).unwrap();
        let mut scores = csv::Writer::from_path(fixture.0.join("scores.csv")).unwrap();
        scores.write_record(HEADER).unwrap();
        for (values, label) in reconstructed.values.iter().zip(reconstructed.labels) {
            scores
                .write_record([
                    "0".to_owned(),
                    values[0].to_string(),
                    values[1].to_string(),
                    values[2].to_string(),
                    values[3].to_string(),
                    label.to_string(),
                    values[4].to_string(),
                    values[5].to_string(),
                    values[6].to_string(),
                    values[7].to_string(),
                    "0".to_owned(),
                    "0".to_owned(),
                ])
                .unwrap();
        }
        scores.flush().unwrap();
        fixture
    }

    fn admit_fixture(fixture: &AdmissionFixture) -> Result<PreparedDataset> {
        prepare(
            &fixture.0,
            &fixture.0.join("scores.csv"),
            &fixture.0.join("map.csv"),
            &fixture.0.join("catalog.csv"),
            256,
        )
    }

    #[test]
    fn admission_checks_scores_before_the_common_warmup_mask() {
        let fixture = admission_fixture();
        let admitted = admit_fixture(&fixture).unwrap();
        assert_eq!(admitted.files[0].scored_rows, 515);
        assert_eq!(admitted.labels.len(), 259);
        let path = fixture.0.join("scores.csv");
        let original = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<String> = original.lines().map(str::to_owned).collect();
        let mut fields: Vec<String> = lines[1].split(',').map(str::to_owned).collect();
        fields[1] = "99999".to_owned();
        lines[1] = fields.join(",");
        std::fs::write(path, lines.join("\n") + "\n").unwrap();
        assert!(
            admit_fixture(&fixture)
                .err()
                .unwrap()
                .to_string()
                .contains("numeric drift")
        );
    }

    #[test]
    fn admission_rejects_label_changes_and_extra_rows() {
        let fixture = admission_fixture();
        let path = fixture.0.join("scores.csv");
        let original = std::fs::read_to_string(&path).unwrap();
        let mut lines: Vec<String> = original.lines().map(str::to_owned).collect();
        let mut fields: Vec<String> = lines[300].split(',').map(str::to_owned).collect();
        fields[5] = if fields[5] == "0" { "1" } else { "0" }.to_owned();
        lines[300] = fields.join(",");
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        assert!(
            admit_fixture(&fixture)
                .err()
                .unwrap()
                .to_string()
                .contains("label drift")
        );
        std::fs::write(
            &path,
            original.clone() + original.lines().last().unwrap() + "\n",
        )
        .unwrap();
        assert!(
            admit_fixture(&fixture)
                .err()
                .unwrap()
                .to_string()
                .contains("trailing rows")
        );
    }

    #[test]
    fn equal_time_distinct_vectors_remain_admitted_samples() {
        let ordinary = admit_fixture(&admission_fixture()).unwrap();
        let fixture = admission_fixture_with_timestamp_delta(Duration::zero());
        let data = admit_fixture(&fixture).unwrap();
        validate_sample_order(&data.files).unwrap();
        let mut raw = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(fixture.0.join("raw.csv"))
            .unwrap();
        let pair: Vec<csv::StringRecord> = raw
            .records()
            .skip(299)
            .take(2)
            .map(Result::unwrap)
            .collect();
        assert_eq!(pair[0][0], pair[1][0]);
        assert_ne!(&pair[0][1], &pair[1][1]);
        assert_eq!(data.features, ordinary.features);
        assert_eq!(data.labels, ordinary.labels);
        assert_eq!(data.files[0].finite_samples, 520);
        assert_eq!(data.files[0].increment_count, 519);
        assert_eq!(data.files[0].equal_timestamp_pair_count, 1);
        assert_eq!(data.files[0].backward_timestamp_pair_count, 0);
        assert_eq!(data.files[0].positive_submillisecond_pair_count, 0);
    }

    #[test]
    fn exact_cadence_gate_distinguishes_backward_and_submillisecond_pairs() {
        let backward = admit_fixture(&admission_fixture_with_timestamp_delta(
            Duration::nanoseconds(-1),
        ))
        .unwrap();
        assert_eq!(backward.files[0].backward_timestamp_pair_count, 1);
        assert_eq!(backward.files[0].equal_timestamp_pair_count, 0);
        assert_eq!(backward.files[0].min_cadence_milliseconds, 0);
        assert!(
            validate_sample_order(&backward.files)
                .unwrap_err()
                .to_string()
                .contains("backward timestamp")
        );
        let submillisecond = admit_fixture(&admission_fixture_with_timestamp_delta(
            Duration::microseconds(500),
        ))
        .unwrap();
        assert_eq!(
            submillisecond.files[0].positive_submillisecond_pair_count,
            1
        );
        assert_eq!(submillisecond.files[0].backward_timestamp_pair_count, 0);
        assert_eq!(submillisecond.files[0].equal_timestamp_pair_count, 0);
        assert_eq!(submillisecond.files[0].nonpositive_cadence_count, 1);
        validate_sample_order(&submillisecond.files).unwrap();
    }

    #[test]
    fn rolling_context_excludes_numerator_and_suffix() {
        let mut increments = vec![2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 99.0];
        let before = rolling_score(&increments, 2, 2).unwrap().unwrap();
        assert_eq!(before, (7.0, 3.5));
        increments[7] = 9999.0;
        assert_eq!(rolling_score(&increments, 2, 2).unwrap().unwrap(), before);
        increments[1] = 4.0;
        assert!(rolling_score(&increments, 2, 2).unwrap().unwrap().1 < before.1);
    }

    #[test]
    fn warmup_is_common_and_zero_context_fails() {
        assert!(rolling_score(&[1.0; 10], 1, 2).unwrap().is_none());
        assert!(rolling_score(&[1.0; 10], 2, 2).unwrap().is_some());
        assert!(rolling_score(&[0.0; 10], 2, 2).is_err());
        assert!(rolling_score(&[1.0; 10], 2, 0).is_err());
    }

    #[test]
    fn map_rejects_gaps_duplicates_and_reordering() {
        assert!(validate_map(&[(0, "a".into()), (1, "b".into())]).is_ok());
        assert!(validate_map(&[(0, "a".into()), (2, "b".into())]).is_err());
        assert!(validate_map(&[(0, "a".into()), (1, "a".into())]).is_err());
        assert!(validate_map(&[(1, "a".into()), (0, "b".into())]).is_err());
    }
}
