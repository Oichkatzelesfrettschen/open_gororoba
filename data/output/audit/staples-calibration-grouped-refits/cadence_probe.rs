//! Audit chronological order after the benchmark exporter's finite-row filter.
use chrono::{DateTime, Utc};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{error::Error, path::PathBuf};

fn timestamp(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .or_else(|_| DateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%:z"))
        .map(|time| time.with_timezone(&Utc))
        .ok()
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(arguments.get(1).ok_or("input root required")?);
    let map = root.join("data/output/benchmark_scores.files.csv");
    let mut sidecar = csv::Reader::from_path(map)?;
    let mut files = Vec::new();
    let mut total_zero = 0u64;
    let mut total_negative = 0u64;
    let mut total_truncated = 0u64;
    let mut total_identical_records = 0u64;
    let mut total_finite = 0u64;
    for entry in sidecar.records() {
        let entry = entry?;
        let id: u16 = entry[0].parse()?;
        let path = &entry[1];
        let input_sha256: String = Sha256::digest(std::fs::read(root.join(path))?)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let mut input = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(root.join(path))?;
        let mut previous: Option<(DateTime<Utc>, [f64; 3], csv::StringRecord, usize)> = None;
        let mut finite_rows = 0u64;
        let mut skipped = 0u64;
        let mut zero = 0u64;
        let mut negative = 0u64;
        let mut truncated = 0u64;
        let mut identical_records = 0u64;
        let mut equal_time_equal_vectors = 0u64;
        let mut equal_time_different_vectors = 0u64;
        let mut minimum = i64::MAX;
        let mut maximum = i64::MIN;
        let mut minimum_positive = i64::MAX;
        let mut maximum_witness = serde_json::Value::Null;
        let mut witnesses = Vec::new();
        let mut category_witnesses = [0u8; 3];
        for (raw_index, record) in input.records().enumerate() {
            let record = record?;
            let time = record.get(0).and_then(timestamp);
            let vector: Option<Vec<f64>> = (1..4)
                .map(|index| record.get(index)?.parse::<f64>().ok())
                .collect();
            let (Some(time), Some(vector)) = (time, vector) else {
                skipped += 1;
                continue;
            };
            if !vector.iter().all(|value| value.is_finite()) {
                skipped += 1;
                continue;
            }
            let vector = [vector[0], vector[1], vector[2]];
            if let Some((prior_time, prior_vector, prior_record, prior_index)) = &previous {
                let duration = time - *prior_time;
                let nanos = duration
                    .num_nanoseconds()
                    .ok_or("cadence exceeds i64 nanoseconds")?;
                let millis = duration.num_milliseconds();
                minimum = minimum.min(nanos);
                if nanos > 0 {
                    minimum_positive = minimum_positive.min(nanos);
                }
                if nanos > maximum {
                    maximum_witness = json!({"previous_raw_record_index":prior_index,"raw_record_index":raw_index,"previous_timestamp":prior_time.to_rfc3339(),"timestamp":time.to_rfc3339(),"delta_nanoseconds":nanos});
                }
                maximum = maximum.max(nanos);
                let category = if nanos == 0 {
                    zero += 1;
                    if vector == *prior_vector {
                        equal_time_equal_vectors += 1;
                    } else {
                        equal_time_different_vectors += 1;
                    }
                    if record == *prior_record {
                        identical_records += 1;
                    }
                    Some(0)
                } else if nanos < 0 {
                    negative += 1;
                    Some(1)
                } else if millis == 0 {
                    truncated += 1;
                    Some(2)
                } else {
                    None
                };
                if let Some(category) = category
                    && category_witnesses[category] < 3
                {
                    category_witnesses[category] += 1;
                    witnesses.push(json!({"category":(["equal_timestamp","backward_timestamp","positive_submillisecond_interval"][category]),"previous_raw_record_index":prior_index,"raw_record_index":raw_index,"finite_record_index":finite_rows,"previous_timestamp":prior_time.to_rfc3339(),"timestamp":time.to_rfc3339(),"delta_nanoseconds":nanos,"delta_milliseconds":millis,"previous_vector":prior_vector,"vector":vector,"identical_full_csv_record":record==*prior_record}));
                }
            }
            finite_rows += 1;
            previous = Some((time, vector, record, raw_index));
        }
        total_zero += zero;
        total_negative += negative;
        total_truncated += truncated;
        total_identical_records += identical_records;
        total_finite += finite_rows;
        files.push(json!({"id":id,"path":path,"input_sha256":input_sha256,"finite_rows":finite_rows,"discarded_raw_rows":skipped,"minimum_delta_nanoseconds":minimum,"minimum_positive_delta_nanoseconds":minimum_positive,"maximum_delta_nanoseconds":maximum,"maximum_interval_witness":maximum_witness,"zero_intervals":zero,"negative_intervals":negative,"positive_submillisecond_intervals":truncated,"identical_full_csv_record_pairs":identical_records,"equal_timestamp_equal_vector_pairs":equal_time_equal_vectors,"equal_timestamp_different_vector_pairs":equal_time_different_vectors,"witnesses":witnesses}));
        eprintln!("file{id}: zero={zero}, negative={negative}, submillisecond={truncated}");
    }
    let report = json!({"schema_version":1,"filter":"Exporter timestamp parser and finite Bx/By/Bz filter, preserving input record order","index_convention":"Zero-based CSV record indexes; finite record index counts retained rows","files":files,"file_count":files.len(),"finite_rows":total_finite,"zero_intervals":total_zero,"negative_intervals":total_negative,"positive_submillisecond_intervals":total_truncated,"identical_full_csv_record_pairs":total_identical_records});
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    Ok(())
}
