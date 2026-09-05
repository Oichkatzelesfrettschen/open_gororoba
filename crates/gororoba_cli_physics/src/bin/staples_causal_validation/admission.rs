//! Raw-row admission with closed timestamp batches and shared causal support.

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, Datelike, Utc};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    path::Path,
};

use super::{
    Config,
    evidence::digest,
    features::{Ensemble, Features, Sample},
};

pub(super) struct Row {
    pub(super) features: Features,
    pub(super) label: u8,
    pub(super) file: u16,
    pub(super) year: i32,
}

#[derive(Clone, Serialize)]
pub(super) struct FileEvidence {
    pub(super) id: u16,
    pub(super) path: String,
    pub(super) sha256: String,
    pub(super) date: String,
    pub(super) year: i32,
    pub(super) probe: String,
    pub(super) raw_rows: u64,
    pub(super) valid_vector_rows: u64,
    pub(super) closed_unique_batches: u64,
    pub(super) admitted_decisions: usize,
    pub(super) positive_decisions: usize,
    pub(super) first_timestamp: String,
    pub(super) last_timestamp: String,
    pub(super) minimum_positive_gap_nanos: Option<i64>,
    pub(super) maximum_gap_nanos: Option<i64>,
    pub(super) rejected: BTreeMap<String, u64>,
    pub(super) support_sha256: String,
    pub(super) support_path: String,
    pub(super) support_bytes: usize,
    #[serde(skip)]
    support_records: Vec<u8>,
    pub(super) label_sha256: String,
}

pub(super) struct Dataset {
    pub(super) rows: Vec<Row>,
    pub(super) files: Vec<FileEvidence>,
}

struct Batch {
    nanos: i64,
    raw_index: u64,
    raw_count: usize,
    vector: Option<[f64; 3]>,
}

fn timestamp(text: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text)
        .or_else(|_| DateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%:z"))
        .ok()
        .map(|value| value.with_timezone(&Utc))
}

fn count(evidence: &mut FileEvidence, reason: &str) {
    *evidence.rejected.entry(reason.to_owned()).or_default() += 1;
}

pub(super) fn catalog(path: &Path, probe: &str, expected: &str) -> Result<Vec<i64>> {
    let bytes = fs::read(path)?;
    ensure!(
        digest(&bytes) == expected,
        "catalog SHA256 differs from verified V2"
    );
    let text = std::str::from_utf8(&bytes)?;
    let mut crossings = Vec::new();
    for line in text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
    {
        if line.starts_with("TIMESTAMP") {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        ensure!(fields.len() == 8, "malformed V2 catalog row");
        let time = timestamp(fields[0])
            .context("invalid V2 timestamp")?
            .timestamp_nanos_opt()
            .context("catalog timestamp overflow")?;
        if fields[7] == probe {
            crossings.push(time);
        }
    }
    crossings.sort_unstable();
    ensure!(
        !crossings.is_empty(),
        "catalog has no crossings for selected probe"
    );
    Ok(crossings)
}

pub(super) fn file_map(path: &Path) -> Result<Vec<(u16, String)>> {
    let mut reader = csv::Reader::from_path(path)?;
    ensure!(
        reader.headers()?.iter().eq(["file_id", "path"]),
        "file map header mismatch"
    );
    let mut paths = BTreeSet::new();
    let mut entries = Vec::new();
    for record in reader.records() {
        let record = record?;
        ensure!(record.len() == 2, "file map row width mismatch");
        let id: u16 = record[0].parse()?;
        ensure!(
            usize::from(id) == entries.len(),
            "file IDs must be dense in plan order"
        );
        ensure!(
            !record[1].is_empty() && paths.insert(record[1].to_owned()),
            "missing or duplicate planned raw path"
        );
        entries.push((id, record[1].to_owned()));
    }
    ensure!(!entries.is_empty(), "empty file map");
    Ok(entries)
}

fn label(crossings: &[i64], nanos: i64, radius: i64) -> u8 {
    let position = crossings.partition_point(|&crossing| crossing < nanos - radius);
    u8::from(
        crossings
            .get(position)
            .is_some_and(|&crossing| crossing <= nanos + radius),
    )
}

fn parse_file(
    bytes: &[u8],
    id: u16,
    path: &str,
    probe: &str,
    config: &Config,
    crossings: &[i64],
    ensemble: &Ensemble,
) -> Result<(Vec<Row>, FileEvidence)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(bytes);
    let mut evidence = FileEvidence {
        id,
        path: path.to_owned(),
        sha256: digest(bytes),
        date: String::new(),
        year: 0,
        probe: probe.to_owned(),
        raw_rows: 0,
        valid_vector_rows: 0,
        closed_unique_batches: 0,
        admitted_decisions: 0,
        positive_decisions: 0,
        first_timestamp: String::new(),
        last_timestamp: String::new(),
        minimum_positive_gap_nanos: None,
        maximum_gap_nanos: None,
        rejected: BTreeMap::new(),
        support_sha256: String::new(),
        support_path: format!("supports/file-{id:04}.bin"),
        support_bytes: 0,
        support_records: Vec::new(),
        label_sha256: String::new(),
    };
    let mut history = VecDeque::<Sample>::new();
    let capacity = config.widths.iter().max().unwrap() + 7;
    let mut pending: Option<Batch> = None;
    let mut last_time: Option<i64> = None;
    let mut rows = Vec::new();
    let mut support_records = Vec::new();
    let mut label_digest = Sha256::new();
    for (record_index, record) in reader.records().enumerate() {
        let raw_index = record_index as u64;
        let record = match record {
            Ok(record) => record,
            Err(_) => {
                evidence.raw_rows += 1;
                count(&mut evidence, "malformed_csv");
                history.clear();
                pending = None;
                continue;
            }
        };
        if record_index == 0 && matches!(record.get(0), Some("Time" | "TIMESTAMP")) {
            let expected = format!("th{probe}_fgs_gse");
            ensure!(
                record.len() == 4
                    && (1..4).all(|column| record[column] == format!("{expected}_{}", column - 1)),
                "unexpected vector header"
            );
            count(&mut evidence, "header");
            continue;
        }
        evidence.raw_rows += 1;
        let Some(time) = record.get(0).and_then(timestamp) else {
            count(&mut evidence, "malformed_timestamp");
            history.clear();
            pending = None;
            continue;
        };
        let nanos = time
            .timestamp_nanos_opt()
            .context("raw timestamp overflow")?;
        let date = time.date_naive().to_string();
        if evidence.date.is_empty() {
            evidence.date = date.clone();
            evidence.year = time.year();
            evidence.first_timestamp = time.to_rfc3339();
        }
        ensure!(
            date == evidence.date,
            "file {id} contains more than one UTC day at raw row {raw_index}"
        );
        evidence.last_timestamp = time.to_rfc3339();
        let gap = last_time.map(|previous| nanos - previous);
        ensure!(
            gap.is_none_or(|gap| gap >= 0),
            "backward timestamp file {id} raw row {raw_index}"
        );
        if let Some(gap) = gap {
            if gap == 0 {
                count(&mut evidence, "equal_timestamp_pairs");
            } else {
                evidence.minimum_positive_gap_nanos = Some(
                    evidence
                        .minimum_positive_gap_nanos
                        .map_or(gap, |minimum| minimum.min(gap)),
                );
                evidence.maximum_gap_nanos = Some(
                    evidence
                        .maximum_gap_nanos
                        .map_or(gap, |maximum| maximum.max(gap)),
                );
            }
        }
        if gap != Some(0) {
            if let Some(batch) = pending.take() {
                if let (1, Some(vector)) = (batch.raw_count, batch.vector) {
                    history.push_back(Sample {
                        nanos: batch.nanos,
                        raw_index: batch.raw_index,
                        vector,
                    });
                    evidence.closed_unique_batches += 1;
                    if history.len() > capacity {
                        history.pop_front();
                    }
                } else {
                    history.clear();
                    count(&mut evidence, "closed_invalid_or_duplicate_batch");
                }
            }
            if gap.is_some_and(|gap| gap > config.maximum_gap_seconds * 1_000_000_000) {
                history.clear();
                count(&mut evidence, "long_gap");
            }
            if history.len() == capacity {
                let ordered = history.make_contiguous();
                let feature_start = capacity - 6;
                if ordered[capacity - 1].nanos - ordered[feature_start].nanos
                    <= config.maximum_feature_span_seconds * 1_000_000_000
                {
                    ensure!(
                        ordered[capacity - 1].nanos < nanos,
                        "feature dependency reaches decision timestamp"
                    );
                    let features = super::features::construct(
                        ordered,
                        &config.widths,
                        config.log_epsilon,
                        ensemble,
                    )?;
                    let target = label(
                        crossings,
                        nanos,
                        config.label_radius_seconds * 1_000_000_000,
                    );
                    support_records.extend_from_slice(&id.to_le_bytes());
                    for value in [
                        raw_index,
                        ordered[0].raw_index,
                        ordered[feature_start].raw_index,
                        ordered[capacity - 1].raw_index,
                        capacity as u64,
                    ] {
                        support_records.extend_from_slice(&value.to_le_bytes());
                    }
                    support_records.extend_from_slice(&nanos.to_le_bytes());
                    support_records.extend_from_slice(&ordered[capacity - 1].nanos.to_le_bytes());
                    support_records.push(target);
                    label_digest.update(nanos.to_le_bytes());
                    label_digest.update([target]);
                    rows.push(Row {
                        features,
                        label: target,
                        file: id,
                        year: time.year(),
                    });
                    evidence.admitted_decisions += 1;
                    evidence.positive_decisions += usize::from(target);
                } else {
                    count(&mut evidence, "feature_span");
                }
            } else {
                count(&mut evidence, "common_warmup");
            }
            pending = Some(Batch {
                nanos,
                raw_index,
                raw_count: 0,
                vector: None,
            });
        }
        let vector = if record.len() == 4 {
            let components: Option<Vec<f64>> = (1..4)
                .map(|column| record[column].parse::<f64>().ok())
                .collect();
            components
                .filter(|values| {
                    values
                        .iter()
                        .all(|value| value.is_finite() && *value != config.fill_value)
                })
                .map(|values| [values[0], values[1], values[2]])
        } else {
            None
        };
        if vector.is_some() {
            evidence.valid_vector_rows += 1;
        } else {
            count(&mut evidence, "invalid_or_fill_vector");
            history.clear();
        }
        // Timestamp decisions precede admission of the vector bearing that timestamp.
        if let Some(batch) = pending.as_mut() {
            batch.raw_count += 1;
            if batch.raw_count == 1 {
                batch.vector = vector;
            } else {
                batch.vector = None;
                history.clear();
            }
        } else {
            // An invalid timestamp row can interrupt a batch before another equal timestamp.
            pending = Some(Batch {
                nanos,
                raw_index,
                raw_count: 2,
                vector: None,
            });
            history.clear();
        }
        last_time = Some(nanos);
    }
    ensure!(!evidence.date.is_empty(), "file lacks a parsed timestamp");
    evidence.support_sha256 = digest(&support_records);
    evidence.support_bytes = support_records.len();
    evidence.support_records = support_records;
    evidence.label_sha256 = super::evidence::hex(&label_digest.finalize());
    Ok((rows, evidence))
}

pub(super) fn prepare(
    root: &Path,
    output: &Path,
    entries: &[(u16, String)],
    probe: &str,
    config: &Config,
    crossings: &[i64],
    ensemble: &Ensemble,
) -> Result<Dataset> {
    let mut dataset = Dataset {
        rows: Vec::new(),
        files: Vec::new(),
    };
    fs::create_dir_all(output.join("supports"))?;
    for batch in entries.chunks(config.threads) {
        let admitted: Vec<_> = batch
            .par_iter()
            .map(|(id, path)| {
                let bytes = fs::read(root.join(path))
                    .with_context(|| format!("read planned file {id}: {path}"))?;
                parse_file(&bytes, *id, path, probe, config, crossings, ensemble)
                    .with_context(|| format!("admit file {id}: {path}, sha256={}", digest(&bytes)))
            })
            .collect();
        for result in admitted {
            let (mut rows, mut file) = result?;
            super::evidence::preserve_bytes(
                &output.join(&file.support_path),
                &file.support_records,
            )?;
            file.support_records.clear();
            file.support_records.shrink_to_fit();
            eprintln!(
                "admitted file {} date {}: {} decisions, {} positives",
                file.id, file.date, file.admitted_decisions, file.positive_decisions
            );
            dataset.rows.append(&mut rows);
            dataset.files.push(file);
        }
    }
    ensure!(
        dataset.rows.len() <= u32::MAX as usize,
        "row index exceeds u32 capacity"
    );
    let expected: BTreeSet<_> = dataset
        .files
        .iter()
        .map(|file| format!("file-{:04}.bin", file.id))
        .collect();
    let observed: BTreeSet<_> = fs::read_dir(output.join("supports"))?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<_>>()?;
    ensure!(
        observed == expected,
        "support record set differs from admitted file plan"
    );
    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw(times: &[i64], fill: Option<usize>) -> Vec<u8> {
        times
            .iter()
            .enumerate()
            .map(|(index, seconds)| {
                let time = DateTime::from_timestamp(1_483_228_800 + seconds, 0).unwrap();
                let value = if fill == Some(index) {
                    -1e30
                } else {
                    index as f64 + 1.0
                };
                format!(
                    "{},{},{},{}\n",
                    time.to_rfc3339(),
                    value,
                    (index as f64 * 0.7).sin(),
                    (index as f64 * 0.4).cos()
                )
            })
            .collect::<String>()
            .into_bytes()
    }
    fn admit(bytes: &[u8]) -> Result<(Vec<Row>, FileEvidence)> {
        let config = crate::test_config();
        let ensemble = Ensemble::new(&config.control_seeds)?;
        parse_file(bytes, 0, "fixture", "a", &config, &[], &ensemble)
    }
    #[test]
    fn current_vector_and_suffix_cannot_change_earlier_decision() {
        let times: Vec<i64> = (0..18).collect();
        let normal = admit(&raw(&times, None)).unwrap();
        let changed = admit(&raw(&times, Some(11))).unwrap();
        assert_eq!(normal.0[0].features.tensors, changed.0[0].features.tensors);
        assert_eq!(
            normal.0[0].features.geometry,
            changed.0[0].features.geometry
        );
        assert_eq!(normal.0[0].features.pvi, changed.0[0].features.pvi);
        assert_eq!(changed.0.len(), 1);
    }
    #[test]
    fn duplicates_reset_future_history_without_revising_current_decision() {
        let mut times: Vec<i64> = (0..18).collect();
        times.insert(12, 11);
        let (rows, file) = admit(&raw(&times, None)).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(file.rejected["equal_timestamp_pairs"], 1);
        assert_eq!(file.rejected["closed_invalid_or_duplicate_batch"], 1);
    }
    #[test]
    fn gaps_fill_invalid_rows_and_backward_times_have_explicit_boundaries() {
        let mut times: Vec<i64> = (0..18).collect();
        for time in &mut times[10..] {
            *time += 40;
        }
        assert!(admit(&raw(&times, None)).unwrap().0.is_empty());
        let times: Vec<i64> = (0..18).collect();
        assert!(admit(&raw(&times, Some(9))).unwrap().0.is_empty());
        let mut bytes = raw(&times[..9], None);
        bytes.extend_from_slice(b"invalid,row\n");
        bytes.extend(raw(&times[9..], None));
        assert!(admit(&bytes).unwrap().0.is_empty());
        let mut backwards = times;
        backwards[11] = 4;
        assert!(admit(&raw(&backwards, None)).is_err());
    }
    #[test]
    fn all_widths_and_controls_share_one_admitted_row_set() {
        let (rows, file) = admit(&raw(&(0..18).collect::<Vec<_>>(), None)).unwrap();
        assert_eq!(rows.len(), 7);
        assert_eq!(file.admitted_decisions, 7);
        assert_eq!(file.support_records.len(), 7 * 59);
        for record in file.support_records.chunks_exact(59) {
            let decision = i64::from_le_bytes(record[42..50].try_into().unwrap());
            let latest = i64::from_le_bytes(record[50..58].try_into().unwrap());
            assert!(latest < decision);
            let first = u64::from_le_bytes(record[10..18].try_into().unwrap());
            let last = u64::from_le_bytes(record[26..34].try_into().unwrap());
            let count = u64::from_le_bytes(record[34..42].try_into().unwrap());
            assert_eq!(last - first + 1, count);
        }
        assert!(
            rows.iter()
                .all(|row| row.features.pvi.len() == 3 && row.features.tensors.len() == 20)
        );
    }
}
