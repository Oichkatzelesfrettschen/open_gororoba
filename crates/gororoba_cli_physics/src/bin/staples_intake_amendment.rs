//! Retrospective timestamp-ambiguity quarantine with byte-preserving provenance.

use anyhow::{Context, Result, ensure};
use chrono::DateTime;
use clap::Parser;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io::Write, path::PathBuf};

const MARKER: &[u8] = b"AMENDMENT_QUARANTINED";
const PROTOCOL_SHA256: &str = "1bcda67ce7af846e2c52d1e9affe554ceb55989baa50dce6d87673dbc37a32f6";
const AMENDMENT_ID: &str = "retrospective-backward-interval-quarantine";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    parent_manifest: PathBuf,
    #[arg(long)]
    input_root: PathBuf,
    #[arg(long)]
    protocol: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
}

#[derive(Clone, Serialize)]
struct Interval {
    start: i64,
    end: i64,
}

#[derive(Serialize)]
struct Replacement {
    raw_ordinal: usize,
    original_timestamp: String,
    original_byte_offset: usize,
    original_byte_length: usize,
    derived_byte_offset: usize,
    replacement_byte_length: usize,
}

struct Row {
    timestamp: i64,
    start: usize,
    end: usize,
}

struct Transformation {
    bytes: Vec<u8>,
    intervals: Vec<Interval>,
    replacements: Vec<Replacement>,
    rows: usize,
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn transform(bytes: &[u8], date: &str) -> Result<Transformation> {
    let mut rows = Vec::new();
    let mut offset = 0;
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        let content = line.strip_suffix(b"\n").unwrap_or(line);
        let content = content.strip_suffix(b"\r").unwrap_or(content);
        ensure!(
            !content.is_empty() && !content.contains(&b'"') && !content.contains(&b'\r'),
            "blank, quoted, or multiline layout at raw ordinal {}",
            rows.len() + 1
        );
        let fields: Vec<_> = content.split(|byte| *byte == b',').collect();
        ensure!(
            fields.len() == 4,
            "expected timestamp plus three vector literals"
        );
        let timestamp_literal = std::str::from_utf8(fields[0])?;
        ensure!(
            timestamp_literal.trim() == timestamp_literal,
            "timestamp whitespace"
        );
        let timestamp = DateTime::parse_from_rfc3339(timestamp_literal)
            .with_context(|| format!("malformed timestamp at raw ordinal {}", rows.len() + 1))?;
        ensure!(
            timestamp.offset().local_minus_utc() == 0 && timestamp.date_naive().to_string() == date,
            "timestamp outside declared UTC date"
        );
        for field in &fields[1..] {
            // Parsing validates the layout; the original vector bytes remain authoritative.
            ensure!(
                std::str::from_utf8(field)?
                    .trim()
                    .parse::<f64>()?
                    .is_finite(),
                "nonfinite vector field"
            );
        }
        ensure!(
            timestamp.timestamp_subsec_nanos() < 1_000_000_000,
            "leap-second encoding requires a separate time-scale policy"
        );
        let timestamp = timestamp
            .timestamp_nanos_opt()
            .context("timestamp outside i64 Unix nanoseconds")?;
        rows.push(Row {
            timestamp,
            start: offset,
            end: offset + fields[0].len(),
        });
        offset += line.len();
    }
    ensure!(!rows.is_empty(), "empty source body");
    let mut maximum = rows[0].timestamp;
    let mut intervals: Vec<Interval> = Vec::new();
    for row in &rows[1..] {
        if row.timestamp < maximum {
            intervals.push(Interval {
                start: row.timestamp,
                end: maximum,
            });
        }
        maximum = maximum.max(row.timestamp);
    }
    intervals.sort_by_key(|interval| interval.start);
    let mut merged: Vec<Interval> = Vec::new();
    for interval in intervals {
        if let Some(previous) = merged.last_mut()
            && interval.start <= previous.end
        {
            previous.end = previous.end.max(interval.end);
        } else {
            merged.push(interval);
        }
    }
    let mut derived = Vec::with_capacity(bytes.len());
    let mut cursor = 0;
    let mut replacements = Vec::new();
    let mut previous_valid = None;
    for (index, row) in rows.iter().enumerate() {
        if merged
            .iter()
            .any(|interval| row.timestamp >= interval.start && row.timestamp <= interval.end)
        {
            derived.extend_from_slice(&bytes[cursor..row.start]);
            replacements.push(Replacement {
                raw_ordinal: index + 1,
                original_timestamp: std::str::from_utf8(&bytes[row.start..row.end])?.to_owned(),
                original_byte_offset: row.start,
                original_byte_length: row.end - row.start,
                derived_byte_offset: derived.len(),
                replacement_byte_length: MARKER.len(),
            });
            derived.extend_from_slice(MARKER);
            cursor = row.end;
        } else {
            ensure!(
                previous_valid.is_none_or(|previous| previous <= row.timestamp),
                "remaining timestamp reversal"
            );
            previous_valid = Some(row.timestamp);
        }
    }
    derived.extend_from_slice(&bytes[cursor..]);
    verify_inverse(bytes, &derived, &replacements)?;
    Ok(Transformation {
        bytes: derived,
        intervals: merged,
        replacements,
        rows: rows.len(),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(&args)
}

fn validate_parent(parent: &Value, expected_dates: usize) -> Result<Vec<String>> {
    let dates: Vec<String> = serde_json::from_value(parent["planned_dates"].clone())?;
    let unique: BTreeSet<_> = dates.iter().collect();
    ensure!(
        !dates.is_empty() && dates.len() == expected_dates && unique.len() == dates.len(),
        "empty, duplicate, or incomplete parent date plan"
    );
    ensure!(
        parent["complete_accounting"] == true
            && parent["planned_count"] == dates.len()
            && parent["accounted_count"] == dates.len()
            && parent["plan"]["planned_dates"] == parent["planned_dates"]
            && parent["plan"]["planned_count"] == dates.len()
            && parent["probe"] == "d"
            && parent["plan"]["probe"] == parent["probe"]
            && parent["plan"]["catalog_sha256"] == parent["catalog_sha256"]
            && parent["plan"]["protocol_sha256"] == parent["protocol_sha256"],
        "parent accounting or nested identity mismatch"
    );
    let results = parent["results"]
        .as_array()
        .context("parent result array missing")?;
    ensure!(
        results.len() == dates.len(),
        "parent results differ from planned count"
    );
    for (date, result) in dates.iter().zip(results) {
        chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")?;
        ensure!(result["date"] == *date, "parent result date/order mismatch");
        ensure!(
            matches!(result["status"].as_str(), Some("admitted" | "failed")),
            "parent result status invalid"
        );
        let attempts = result["attempts"]
            .as_array()
            .context("parent attempts missing")?;
        ensure!(!attempts.is_empty(), "parent attempts empty");
        let first = &attempts[0];
        ensure!(
            first["http_status"] == 200
                && first["curl_exit"] == 0
                && first["raw_path"]
                    .as_str()
                    .is_some_and(|path| path.ends_with(&format!("/{date}/attempt-1.body")))
                && first["sha256"]
                    .as_str()
                    .is_some_and(|hash| hash.len() == 64
                        && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
                && first["bytes"].as_u64().is_some_and(|bytes| bytes > 0),
            "parent attempt-1 source mapping invalid"
        );
        for attempt in attempts {
            ensure!(
                attempt["sha256"] == first["sha256"] && attempt["bytes"] == first["bytes"],
                "conflicting parent attempt hashes or sizes"
            );
        }
        if result["status"] == "admitted" {
            ensure!(
                result["sha256"] == first["sha256"]
                    && result["bytes"] == first["bytes"]
                    && result["raw_path"] == first["raw_path"],
                "parent admitted association conflicts with attempts"
            );
        }
    }
    Ok(dates)
}

fn verify_source(bytes: &[u8], result: &Value) -> Result<()> {
    ensure!(
        result["attempts"][0]["sha256"] == digest(bytes)
            && result["attempts"][0]["bytes"] == bytes.len(),
        "materialized source hash/size differs from retained parent attempt"
    );
    Ok(())
}

fn run(args: &Args) -> Result<()> {
    let protocol_bytes = fs::read(&args.protocol)?;
    let protocol: toml::Value = toml::from_str(std::str::from_utf8(&protocol_bytes)?)?;
    validate_protocol(&protocol_bytes, &protocol)?;
    let parent_bytes = fs::read(&args.parent_manifest)?;
    ensure!(
        digest(&parent_bytes) == "be2d2040db55e1fc0e6e54aa6bccecd333de6424dfb0c76c802242257f135e20",
        "parent manifest differs from frozen identity"
    );
    let mut derived: Value = serde_json::from_slice(&parent_bytes)?;
    let dates = validate_parent(&derived, 166)?;
    let mut outputs = Vec::new();
    let mut ledger = Vec::new();
    let mut file_map = String::from("file_id,path\n");
    for (index, date) in dates.iter().enumerate() {
        let relative_path = format!("{date}/attempt-1.body");
        let original = fs::read(args.input_root.join(&relative_path))?;
        let result = &mut derived["results"][index];
        verify_source(&original, result)?;
        let transformed = transform(&original, date)?;
        let original_result = result.clone();
        let derived_digest = digest(&transformed.bytes);
        ledger.push(json!({
            "date": date,
            "raw_rows": transformed.rows,
            "original_sha256": digest(&original),
            "original_bytes": original.len(),
            "original_materialized_path": args.input_root.join(&relative_path),
            "original_manifest_result": original_result,
            "derived_path": relative_path,
            "derived_sha256": derived_digest,
            "derived_bytes": transformed.bytes.len(),
            "quarantined_rows": transformed.replacements.len(),
            "intervals": transformed.intervals,
            "replacements": transformed.replacements,
            "remaining_valid_timestamps_monotonic": true,
            "inverse_reconstruction_original_bytes_and_sha256_verified": true,
            "byte_preservation_scope": "Only the listed timestamp field ranges are replaced; vector literals, record order, record count, separators, and line endings retain their original bytes"
        }));
        result["parent_status"] = result["status"].clone();
        let attempts = result
            .as_object_mut()
            .unwrap()
            .remove("attempts")
            .context("parent attempts missing")?;
        result["parent_attempts"] = attempts;
        result["status"] = json!("admitted");
        result["admission_scope"] = json!(
            "derived timestamp-quarantine preflight; frozen feature admission and inference are separate"
        );
        result["raw_path"] = json!(relative_path);
        result["sha256"] = json!(derived_digest);
        result["bytes"] = json!(transformed.bytes.len());
        // Parent coverage describes original samples and remains separately identified.
        if let Some(coverage) = result.as_object_mut().unwrap().remove("coverage") {
            result["parent_coverage"] = coverage;
        }
        file_map.push_str(&format!("{index},{relative_path}\n"));
        outputs.push((relative_path, transformed.bytes));
    }
    let protocol_hash = digest(&protocol_bytes);
    let ledger_bytes = serde_json::to_vec_pretty(&json!({
        "schema_version": 1,
        "utility_source_sha256": digest(include_bytes!("staples_intake_amendment.rs")),
        "utility_executable_sha256": digest(&fs::read(std::env::current_exe()?)?),
        "parent_manifest_sha256": digest(&parent_bytes),
        "amendment_protocol_sha256": protocol_hash,
        "ordinal_base": 1,
        "byte_offset_base": 0,
        "interval_timestamp_units": "i64 Unix nanoseconds, leap seconds rejected",
        "planned_dates": dates,
        "dates": ledger
    }))?;
    derived["admitted_count"] = json!(dates.len());
    derived["failed_count"] = json!(0);
    derived["amendment"] = json!({
        "amendment_id": AMENDMENT_ID,
        "protocol_sha256": protocol_hash,
        "protocol_path": "protocol.toml",
        "parent_manifest_sha256": digest(&parent_bytes),
        "parent_manifest_path": "parent-manifest.json",
        "ledger_sha256": digest(&ledger_bytes),
        "ledger_path": "ledger.json",
        "file_map_sha256": digest(file_map.as_bytes()),
        "file_map_path": "file-map.csv",
        "selection_scope": "retrospective full-file timestamp-quality selection; causal feature construction remains subject to frozen runner admission",
        "original_experiment": "E-283",
        "original_experiment_status": "blocked_input_admission_backward_timestamp",
        "replication_count": 0,
        "discrimination_requirement": 0.005
    });
    let manifest_bytes = serde_json::to_vec_pretty(&derived)?;
    // Exclusive directory creation rejects every existing output, including partial runs.
    fs::create_dir(&args.out_dir)
        .context("output directory must be absent and its parent must exist")?;
    for (relative_path, bytes) in outputs {
        let output_path = args.out_dir.join(relative_path);
        fs::create_dir(output_path.parent().context("output parent missing")?)?;
        write_new(&output_path, &bytes)?;
    }
    write_new(&args.out_dir.join("parent-manifest.json"), &parent_bytes)?;
    write_new(&args.out_dir.join("protocol.toml"), &protocol_bytes)?;
    write_new(&args.out_dir.join("ledger.json"), &ledger_bytes)?;
    write_new(&args.out_dir.join("file-map.csv"), file_map.as_bytes())?;
    // The manifest is the completion artifact; earlier write failures leave it absent.
    write_new(&args.out_dir.join("derived-manifest.json"), &manifest_bytes)?;
    println!("{}", args.out_dir.join("derived-manifest.json").display());
    Ok(())
}

fn write_new(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn validate_protocol(bytes: &[u8], protocol: &toml::Value) -> Result<()> {
    ensure!(
        digest(bytes) == PROTOCOL_SHA256
            && protocol.get("amendment_id").and_then(toml::Value::as_str) == Some(AMENDMENT_ID),
        "amendment protocol differs from frozen hash or identity"
    );
    Ok(())
}

fn verify_inverse(original: &[u8], derived: &[u8], replacements: &[Replacement]) -> Result<()> {
    let mut reconstructed = Vec::with_capacity(original.len());
    let mut cursor = 0;
    for replacement in replacements {
        let start = replacement.derived_byte_offset;
        let end = start
            .checked_add(replacement.replacement_byte_length)
            .context("replacement range overflow")?;
        ensure!(
            start >= cursor && derived.get(start..end) == Some(MARKER),
            "derived replacement range mismatch"
        );
        let original_end = replacement
            .original_byte_offset
            .checked_add(replacement.original_byte_length)
            .context("original range overflow")?;
        ensure!(
            original.get(replacement.original_byte_offset..original_end)
                == Some(replacement.original_timestamp.as_bytes()),
            "original replacement range mismatch"
        );
        reconstructed.extend_from_slice(&derived[cursor..start]);
        ensure!(
            reconstructed.len() == replacement.original_byte_offset,
            "inverse reconstruction offset mismatch"
        );
        reconstructed.extend_from_slice(replacement.original_timestamp.as_bytes());
        cursor = end;
    }
    reconstructed.extend_from_slice(&derived[cursor..]);
    ensure!(
        reconstructed == original && digest(&reconstructed) == digest(original),
        "inverse reconstruction fails original byte/hash identity"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(seconds: &[u32]) -> Vec<u8> {
        seconds
            .iter()
            .enumerate()
            .map(|(index, second)| {
                format!("2016-09-02T00:00:{second:02}.000Z,{index}.123,-2e-3, 4.0\r\n")
            })
            .collect::<String>()
            .into_bytes()
    }

    #[test]
    fn overlapping_reversals_quarantine_all_occurrences_and_preserve_other_bytes() -> Result<()> {
        let original = body(&[0, 2, 4, 2, 4, 3, 5, 8, 7, 9]);
        let transformed = transform(&original, "2016-09-02")?;
        assert_eq!(transformed.rows, 10);
        assert_eq!(transformed.intervals.len(), 2);
        assert_eq!(
            transformed
                .replacements
                .iter()
                .map(|row| row.raw_ordinal)
                .collect::<Vec<_>>(),
            [2, 3, 4, 5, 6, 8, 9]
        );
        let mut reconstructed = transformed.bytes.clone();
        for replacement in transformed.replacements.iter().rev() {
            assert_eq!(
                &reconstructed[replacement.derived_byte_offset
                    ..replacement.derived_byte_offset + replacement.replacement_byte_length],
                MARKER
            );
            reconstructed.splice(
                replacement.derived_byte_offset
                    ..replacement.derived_byte_offset + replacement.replacement_byte_length,
                replacement.original_timestamp.bytes(),
            );
            assert_eq!(
                &original[replacement.original_byte_offset
                    ..replacement.original_byte_offset + replacement.original_byte_length],
                replacement.original_timestamp.as_bytes()
            );
        }
        assert_eq!(reconstructed, original);
        Ok(())
    }

    #[test]
    fn monotonic_and_equal_timestamps_are_byte_identical() -> Result<()> {
        let original = body(&[0, 0, 1, 2]);
        let transformed = transform(&original, "2016-09-02")?;
        assert_eq!(transformed.bytes, original);
        assert!(transformed.replacements.is_empty());
        let without_newline = original.strip_suffix(b"\r\n").unwrap();
        assert_eq!(
            transform(without_newline, "2016-09-02")?.bytes,
            without_newline
        );
        Ok(())
    }

    #[test]
    fn exhaustive_short_sequences_match_pairwise_inversion_oracle() -> Result<()> {
        for encoded in 0_u32..4096 {
            let seconds: Vec<_> = (0..6).map(|index| (encoded >> (2 * index)) & 3).collect();
            let transformed = transform(&body(&seconds), "2016-09-02")?;
            let expected: Vec<_> = seconds
                .iter()
                .enumerate()
                .filter_map(|(ordinal, &timestamp)| {
                    let covered = seconds.iter().enumerate().any(|(earlier_index, &earlier)| {
                        seconds[earlier_index + 1..].iter().any(|&later| {
                            later < earlier && timestamp >= later && timestamp <= earlier
                        })
                    });
                    covered.then_some(ordinal + 1)
                })
                .collect();
            assert_eq!(
                transformed
                    .replacements
                    .iter()
                    .map(|row| row.raw_ordinal)
                    .collect::<Vec<_>>(),
                expected,
                "sequence {seconds:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn exclusive_output_refuses_replacement() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "staples-intake-amendment-exclusive-{}",
            std::process::id()
        ));
        write_new(&path, b"retained")?;
        let result = write_new(&path, b"replacement");
        let observed = fs::read(&path)?;
        fs::remove_file(path)?;
        assert!(result.is_err());
        assert_eq!(observed, b"retained");
        Ok(())
    }

    #[test]
    fn frozen_protocol_rejects_changed_bytes_and_identity() -> Result<()> {
        let bytes = fs::read(
            repo_root::resolve!()
                .join("data/output/audit/external-crossing-intake-amendment/protocol.toml"),
        )?;
        let mut protocol: toml::Value = toml::from_str(std::str::from_utf8(&bytes)?)?;
        validate_protocol(&bytes, &protocol)?;
        let mut changed = bytes.clone();
        changed.push(b'\n');
        assert!(validate_protocol(&changed, &protocol).is_err());
        protocol["amendment_id"] = toml::Value::String("different".to_owned());
        assert!(validate_protocol(&bytes, &protocol).is_err());
        Ok(())
    }

    #[test]
    fn time_scalar_and_vector_preflight_reject_unsupported_encodings() -> Result<()> {
        for literal in ["2016-09-02T23:59:60Z", "2016-09-02T23:59:60.500Z"] {
            assert!(DateTime::parse_from_rfc3339(literal).is_ok());
            assert!(transform(format!("{literal},1,2,3\n").as_bytes(), "2016-09-02").is_err());
        }
        assert!(transform(b"2500-01-01T00:00:00Z,1,2,3\n", "2500-01-01").is_err());
        for vector in ["NaN", "inf", "-inf", "1e999"] {
            assert!(
                transform(
                    format!("2016-09-02T00:00:00Z,{vector},2,3\n").as_bytes(),
                    "2016-09-02"
                )
                .is_err()
            );
        }
        let fill = b"2016-09-02T00:00:00.000000001Z,-1e30,2,3\n2016-09-02T00:00:00Z,1,2,3\n";
        let transformed = transform(fill, "2016-09-02")?;
        assert_eq!(transformed.replacements.len(), 2);
        assert_eq!(
            transformed.intervals[0].end - transformed.intervals[0].start,
            1
        );
        Ok(())
    }

    #[test]
    fn inverse_proof_rejects_vector_and_range_tampering() -> Result<()> {
        let original = body(&[0, 2, 1, 3]);
        let mut transformed = transform(&original, "2016-09-02")?;
        transformed.bytes[transformed.replacements[0].derived_byte_offset + MARKER.len() + 1] =
            b'9';
        assert!(verify_inverse(&original, &transformed.bytes, &transformed.replacements).is_err());
        let mut transformed = transform(&original, "2016-09-02")?;
        transformed.replacements[0].original_byte_offset += 1;
        assert!(verify_inverse(&original, &transformed.bytes, &transformed.replacements).is_err());
        Ok(())
    }

    #[test]
    fn malformed_timestamps_and_unknown_layouts_block() {
        for input in [
            "",
            "bad,1,2,3\n",
            "\"2016-09-02T00:00:00Z\",1,2,3\n",
            "2016-09-02T00:00:00Z,1,2\n",
            "2016-09-02T00:00:00Z,1,2,3\n\n",
            "2016-09-03T00:00:00Z,1,2,3\n",
            "2016-09-02T00:00:00+01:00,1,2,3\n",
        ] {
            assert!(
                transform(input.as_bytes(), "2016-09-02").is_err(),
                "accepted {input:?}"
            );
        }
    }

    fn parent() -> Value {
        let bytes = body(&[0, 1]);
        let attempt = json!({"raw_path":"/source/2016-09-02/attempt-1.body","sha256":digest(&bytes),"bytes":bytes.len(),"http_status":200,"curl_exit":0});
        json!({"planned_dates":["2016-09-02"],"planned_count":1,"accounted_count":1,"complete_accounting":true,"probe":"d","catalog_sha256":"catalog","protocol_sha256":"protocol","plan":{"planned_dates":["2016-09-02"],"planned_count":1,"probe":"d","catalog_sha256":"catalog","protocol_sha256":"protocol"},"results":[{"date":"2016-09-02","status":"failed","attempts":[attempt]}]})
    }

    #[test]
    fn parent_identity_rejects_empty_conflicting_and_tampered_inputs() -> Result<()> {
        let original = parent();
        assert_eq!(validate_parent(&original, 1)?, ["2016-09-02"]);
        verify_source(&body(&[0, 1]), &original["results"][0])?;
        assert!(verify_source(&body(&[1, 0]), &original["results"][0]).is_err());
        for mutation in 0..5 {
            let mut altered = original.clone();
            match mutation {
                0 => altered["planned_dates"] = json!([]),
                1 => altered["results"][0]["attempts"] = json!([]),
                2 => altered["results"][0]["date"] = json!("2016-09-03"),
                3 => altered["plan"]["catalog_sha256"] = json!("different"),
                _ => {
                    let mut conflict = altered["results"][0]["attempts"][0].clone();
                    conflict["sha256"] = json!("0".repeat(64));
                    altered["results"][0]["attempts"]
                        .as_array_mut()
                        .unwrap()
                        .push(conflict);
                }
            }
            assert!(validate_parent(&altered, 1).is_err());
        }
        Ok(())
    }
}
