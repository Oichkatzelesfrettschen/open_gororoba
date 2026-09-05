//! Compare native CDF source record order to the retained HAPI response.
//! The decoder admits only the observed Network-encoded, gzip-compressed
//! f64 timestamp and three-component f32 vector variables. Index coverage and
//! decompressed byte counts establish the record boundary before comparison.
use anyhow::{Context, Result, bail, ensure};
use cdf::{
    cdf::Cdf,
    record::{
        vxr::{VariableIndexRecord, VariableIndexRecordChild},
        zvdr::ZVariableDescriptorRecord,
    },
    types::CdfType,
};
use flate2::read::GzDecoder;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::Path};

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn variable<'a>(cdf: &'a Cdf, name: &str) -> Result<&'a ZVariableDescriptorRecord> {
    cdf.cdr
        .gdr
        .zvdr_vec
        .iter()
        .find(|item| item.name.to_string() == name)
        .context("missing CDF variable")
}
fn walk(
    index: &VariableIndexRecord,
    components: usize,
    width: usize,
    out: &mut BTreeMap<usize, Vec<f64>>,
) -> Result<()> {
    for (ordinal, child) in index.children.iter().enumerate() {
        let Some(child) = child else { continue };
        match child {
            VariableIndexRecordChild::VXR(inner) => walk(inner, components, width, out)?,
            VariableIndexRecordChild::CVVR(block) => {
                let first =
                    usize::try_from(**index.first_vec[ordinal].as_ref().context("first record")?)?;
                let last =
                    usize::try_from(**index.last_vec[ordinal].as_ref().context("last record")?)?;
                ensure!(last >= first, "inverted source record bounds");
                ensure!(block.data.starts_with(&[0x1f, 0x8b]), "expected gzip CVVR");
                let mut bytes = Vec::new();
                GzDecoder::new(block.data.as_slice()).read_to_end(&mut bytes)?;
                let records = last - first + 1;
                ensure!(
                    bytes.len() == records * components * width,
                    "CVVR byte count disagrees with source index and type"
                );
                for (offset, record) in bytes.chunks_exact(components * width).enumerate() {
                    let values = record
                        .chunks_exact(width)
                        .map(|value| match width {
                            4 => f64::from(f32::from_be_bytes(value.try_into().unwrap())),
                            8 => f64::from_be_bytes(value.try_into().unwrap()),
                            _ => unreachable!(),
                        })
                        .collect::<Vec<_>>();
                    ensure!(
                        out.insert(first + offset, values).is_none(),
                        "overlapping CDF record indices"
                    );
                }
            }
            VariableIndexRecordChild::VVR(_) => {
                bail!("typed audit expects compressed value blocks for selected variables")
            }
        }
    }
    Ok(())
}
fn rows(
    variable: &ZVariableDescriptorRecord,
    components: usize,
    width: usize,
) -> Result<Vec<Vec<f64>>> {
    ensure!(
        *variable.sparse_records == 0,
        "selected vector/time must use dense records"
    );
    let mut indexed = BTreeMap::new();
    for index in &variable.vxr_vec {
        walk(index, components, width, &mut indexed)?;
    }
    let expected = usize::try_from(*variable.max_record)? + 1;
    ensure!(
        indexed.len() == expected && indexed.keys().copied().eq(0..expected),
        "incomplete CDF source index coverage"
    );
    Ok(indexed.into_values().collect())
}
fn metadata(cdf: &Cdf) -> Vec<Value> {
    cdf.cdr.gdr.adr_vec.iter().filter(|attribute|["Data_version","Logical_file_id","Generation_date","MODS","TEXT","UNITS","DEPEND_0","DEPEND_TIME","VIRTUAL","FUNCTION","FUNCT","COMPONENT_0","COMPONENT_1","DEPEND_EPOCH0"].contains(&attribute.name.to_string().as_str())).map(|attribute|json!({"name":attribute.name.to_string(),"global_values":attribute.agredr_vec.iter().map(|entry|format!("{:?}",entry.value)).collect::<Vec<_>>(),"fgs_entries":attribute.azedr_vec.iter().filter(|entry|[0,7,13,14,15].contains(&*entry.num)).map(|entry|json!({"variable_id":*entry.num,"value_debug":format!("{:?}",entry.value)})).collect::<Vec<_>>()})).collect()
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: external-cdf-order-audit CDF HAPI_JSON_OR_CSV OUTPUT_JSON"
    );
    let native_bytes = fs::read(&args[0])?;
    let hapi_bytes = fs::read(&args[1])?;
    let cdf = Cdf::read_cdf_file(Path::new(&args[0]))?;
    ensure!(
        !cdf.is_compressed && format!("{:?}", cdf.cdr.encoding) == "Network",
        "unexpected file encoding"
    );
    let time_variable = variable(&cdf, "thd_fgs_time")?;
    let vector_variable = variable(&cdf, "thd_fgs_gse")?;
    ensure!(
        *time_variable.data_type == 45 && *vector_variable.data_type == 44,
        "unexpected numeric types"
    );
    ensure!(
        time_variable.size_z_dims.is_empty()
            && vector_variable
                .size_z_dims
                .iter()
                .map(|value| **value)
                .eq([3]),
        "unexpected variable dimensionality"
    );
    let times = rows(time_variable, 1, 8)?;
    let vectors = rows(vector_variable, 3, 4)?;
    ensure!(times.len() == vectors.len(), "time/vector record mismatch");
    let epoch0 = variable(&cdf, "thd_fgs_epoch0")?;
    let epoch_value = epoch0
        .vxr_vec
        .iter()
        .flat_map(|index| index.children.iter())
        .filter_map(|child| match child {
            Some(VariableIndexRecordChild::VVR(block)) => Some(block),
            _ => None,
        })
        .flat_map(|block| block.records.iter())
        .flat_map(|record| record.data.iter())
        .next()
        .context("epoch0 value")?;
    let CdfType::Epoch(epoch_value) = epoch_value else {
        bail!("epoch0 must be CDF_EPOCH")
    };
    let epoch0_ms = f64::from(epoch_value.clone());
    ensure!(
        epoch0_ms == 62167219200000.0,
        "source epoch0 differs from CDF_EPOCH Unix origin"
    );
    let mut hapi = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(hapi_bytes.as_slice());
    let hapi_rows = hapi.records().collect::<std::result::Result<Vec<_>, _>>()?;
    let mut native_backwards = Vec::new();
    let mut native_by_millisecond = BTreeMap::<i64, Vec<usize>>::new();
    for (index, time) in times.iter().enumerate() {
        ensure!(time[0].is_finite(), "native timestamp must be finite");
        native_by_millisecond
            .entry((time[0] * 1000.0).floor() as i64)
            .or_default()
            .push(index);
        if index > 0 && time[0] < times[index - 1][0] {
            native_backwards.push(json!({"source_record_index":index,"previous_unix_seconds":times[index-1][0],"unix_seconds":time[0],"delta_seconds":time[0]-times[index-1][0]}));
        }
    }
    let mut hapi_backwards = Vec::new();
    let mut witnesses = Vec::new();
    let mut absent_timestamps = Vec::new();
    let mut vector_mismatches = Vec::new();
    let mut matched_native_indices = std::collections::BTreeSet::new();
    let mut matched_hapi_rows = 0usize;
    let mut previous_hapi_time = None;
    for (index, record) in hapi_rows.iter().enumerate() {
        ensure!(record.len() == 4, "HAPI row width");
        let utc = chrono::DateTime::parse_from_rfc3339(&record[0])?;
        let milliseconds = utc.timestamp_millis();
        let parsed = record
            .iter()
            .skip(1)
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let candidates = native_by_millisecond
            .range((milliseconds - 1)..=(milliseconds + 1))
            .flat_map(|(_, indices)| indices.iter().copied())
            .filter(|candidate| {
                (times[*candidate][0] * 1000.0 - milliseconds as f64).abs() <= 1.001
            })
            .collect::<Vec<_>>();
        if !candidates.is_empty() {
            let matches = candidates
                .iter()
                .copied()
                .filter(|candidate| {
                    parsed
                        .iter()
                        .zip(&vectors[*candidate])
                        .all(|(parsed, native)| {
                            (parsed - native).abs() <= native.abs().max(1e-20) * 5.1e-7
                        })
                })
                .collect::<Vec<_>>();
            if matches.is_empty() {
                vector_mismatches.push(json!({"hapi_csv_row":index+1,"timestamp":record[0],"hapi_gse":parsed,"native_candidates":candidates.iter().map(|candidate|json!({"source_record_index":candidate,"native_gse":vectors[*candidate]})).collect::<Vec<_>>()}));
            } else {
                matched_hapi_rows += 1;
                matched_native_indices.extend(matches);
            }
        } else {
            absent_timestamps
                .push(json!({"hapi_csv_row":index+1,"timestamp":record[0],"hapi_gse":parsed}));
        }
        if previous_hapi_time.is_some_and(|previous| milliseconds < previous) {
            hapi_backwards.push(json!({"hapi_csv_row":index+1,"previous_timestamp":hapi_rows[index-1][0],"timestamp":record[0],"delta_milliseconds":milliseconds-previous_hapi_time.unwrap()}));
            for (witness_index, witness_row) in hapi_rows
                .iter()
                .enumerate()
                .take((index + 4).min(hapi_rows.len()))
                .skip(index.saturating_sub(3))
            {
                let witness_ms =
                    chrono::DateTime::parse_from_rfc3339(&witness_row[0])?.timestamp_millis();
                witnesses.push(json!({"hapi_csv_row":witness_index+1,"hapi_timestamp":hapi_rows[witness_index][0],"hapi_gse_text":hapi_rows[witness_index].iter().skip(1).collect::<Vec<_>>(),"native_candidates":native_by_millisecond.get(&witness_ms).into_iter().flatten().map(|candidate|json!({"source_record_index":candidate,"native_unix_seconds":times[*candidate][0],"native_gse":vectors[*candidate]})).collect::<Vec<_>>()}));
            }
        }
        previous_hapi_time = Some(milliseconds);
    }
    let mut result = json!({"schema_version":1,"native_path":args[0],"native_sha256":digest(&native_bytes),"hapi_path":args[1],"hapi_sha256":digest(&hapi_bytes),"native_version":format!("{:?}",cdf.cdr.cdf_version),"native_encoding":"Network(big-endian)","native_time_variable":"thd_fgs_time","native_vector_variable":"thd_fgs_gse","native_epoch0_cdf_milliseconds":epoch0_ms,"native_rows":times.len(),"hapi_rows":hapi_rows.len(),"matched_hapi_timestamp_vector_rows":matched_hapi_rows,"matched_native_record_count":matched_native_indices.len(),"hapi_absent_timestamp_count":absent_timestamps.len(),"hapi_vector_mismatch_count":vector_mismatches.len(),"join_rule":"Match timestamps within 1.001 milliseconds to admit source-to-CDF_EPOCH floating-point conversion and millisecond text serialization; compare all 3 source f32 components at relative tolerance 5.1e-7 for HAPI text rounding. Source and HAPI sequence order remain unchanged; timestamp lookup is diagnostic only.","native_backward_steps":native_backwards,"hapi_backward_steps":hapi_backwards,"hapi_absent_timestamps":absent_timestamps,"hapi_vector_mismatches":vector_mismatches,"witnesses":witnesses,"metadata":metadata(&cdf),"non_claim":"A different source row count or unmatched values prevents certifying HAPI/native identity. The observations do not identify an upstream instrument acquisition cause or every processing/calibration version.","intake_condition":"A provider-corrected monotonic source with independently retained identity and provenance, followed by a separately declared intake/protocol decision. Sorting, deduplication, and silent day omission remain prohibited by the sealed campaign."});
    result["native_unmatched_records"] = json!(times.iter().enumerate().filter(|(index,_)| !matched_native_indices.contains(index)).map(|(index,time)|json!({"source_record_index":index,"native_unix_seconds":time[0],"timestamp":chrono::DateTime::from_timestamp_millis((time[0]*1000.0).floor() as i64).map(|time|time.to_rfc3339()),"native_gse":vectors[index]})).collect::<Vec<_>>());
    result["native_first_timestamp"] = json!(
        chrono::DateTime::from_timestamp_millis(
            (times.first().context("first native timestamp")?[0] * 1000.0).floor() as i64
        )
        .context("first timestamp range")?
        .to_rfc3339()
    );
    result["native_last_timestamp"] = json!(
        chrono::DateTime::from_timestamp_millis(
            (times.last().context("last native timestamp")?[0] * 1000.0).floor() as i64
        )
        .context("last timestamp range")?
        .to_rfc3339()
    );
    fs::write(&args[2], serde_json::to_vec_pretty(&result)?)?;
    println!(
        "native_rows={} native_backward_steps={}",
        times.len(),
        result["native_backward_steps"].as_array().unwrap().len()
    );
    Ok(())
}
