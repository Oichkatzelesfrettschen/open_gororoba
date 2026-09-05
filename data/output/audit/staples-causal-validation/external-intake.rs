//! Freeze the catalog-selected external date population and retain exact HAPI bytes.
//! Transport success, field semantics, timestamp admission, and label reconciliation
//! produce separate evidence; the intake never constructs scientific predictions.
use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, Datelike, NaiveDate, SecondsFormat, Utc};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

const CATALOG_HASH: &str = "807b79b73b1266981d24c20627d8f85bca585475d07260d4162277a22ce660b5";
const INFO_URL: &str = "https://cdaweb.gsfc.nasa.gov/hapi/info?id=THD_L2_FGM%400";
const FILL: f64 = -1.0e30;

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}
fn timestamp(text: &str) -> Result<DateTime<Utc>> {
    let normalized = text.trim_matches('"').replacen(' ', "T", 1);
    Ok(DateTime::parse_from_rfc3339(&normalized)?.with_timezone(&Utc))
}

fn catalog_rows(bytes: &[u8]) -> Result<Vec<Value>> {
    let text = std::str::from_utf8(bytes)?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') || line.starts_with("TIMESTAMP") {
            continue;
        }
        let columns: Vec<_> = line.split('\t').collect();
        ensure!(
            columns.len() == 8,
            "catalog line {} has {} columns",
            index + 1,
            columns.len()
        );
        let time = timestamp(columns[0])?;
        let coordinates = columns[1..7]
            .iter()
            .map(|cell| cell.parse::<f64>())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            coordinates.iter().all(|value| value.is_finite()),
            "nonfinite catalog location"
        );
        ensure!(
            ["a", "b", "c", "d", "e"].contains(&columns[7]),
            "unknown catalog probe"
        );
        rows.push(json!({"timestamp":time.to_rfc3339_opts(SecondsFormat::Secs,true),"date":time.date_naive().to_string(),"year":time.year(),"probe":columns[7],"gse":&coordinates[..3],"gsm":&coordinates[3..],"line":index+1}));
    }
    Ok(rows)
}

fn reconcile(rows: &[Value], derived_path: &Path, out: &Path) -> Result<()> {
    let raw: BTreeMap<_, _> = rows
        .iter()
        .filter(|row| row["probe"] == "a")
        .map(|row| (row["timestamp"].as_str().unwrap().to_owned(), row.clone()))
        .collect();
    let bytes = fs::read(derived_path)?;
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    ensure!(
        reader
            .headers()?
            .iter()
            .eq(["TIMESTAMP", "R", "year", "doy"]),
        "unexpected derived A catalog columns"
    );
    let mut derived = BTreeMap::new();
    let mut mismatched_fields = Vec::new();
    for record in reader.records() {
        let record = record?;
        let time = timestamp(&record[0])?;
        let key = time.to_rfc3339_opts(SecondsFormat::Secs, true);
        ensure!(
            !derived.contains_key(&key),
            "duplicate derived timestamp {key}"
        );
        let radius: f64 = record[1].parse()?;
        let year: i32 = record[2].parse()?;
        let doy: u32 = record[3].parse()?;
        ensure!(
            year == time.year() && doy == time.ordinal(),
            "derived year/day mismatch at {key}"
        );
        if let Some(source) = raw.get(&key) {
            let location = source["gse"].as_array().unwrap();
            let expected_radius = location
                .iter()
                .map(|value| value.as_f64().unwrap().powi(2))
                .sum::<f64>()
                .sqrt();
            if (radius - expected_radius).abs() > 1e-9 {
                mismatched_fields.push(
                    json!({"timestamp":key,"derived_radius":radius,"gse_norm":expected_radius}),
                );
            }
        }
        derived.insert(key, json!({"timestamp":time.to_rfc3339_opts(SecondsFormat::Secs,true),"radius":radius,"year":year,"doy":doy}));
    }
    let omitted: Vec<_> = raw
        .iter()
        .filter(|(key, _)| !derived.contains_key(*key))
        .map(|(_, row)| row.clone())
        .collect();
    let added: Vec<_> = derived
        .iter()
        .filter(|(key, _)| !raw.contains_key(*key))
        .map(|(_, row)| row.clone())
        .collect();
    let mut omissions_by_date = BTreeMap::<String, usize>::new();
    for row in &omitted {
        *omissions_by_date
            .entry(row["date"].as_str().unwrap().to_owned())
            .or_default() += 1;
    }
    write_json(
        &out.join("catalog-a-set-reconciliation.json"),
        &json!({"schema_version":1,"catalog_sha256":CATALOG_HASH,"derived_path":derived_path,"derived_sha256":digest(&bytes),"raw_a_count":raw.len(),"derived_a_count":derived.len(),"omitted_count":omitted.len(),"added_count":added.len(),"omitted_rows":omitted,"added_rows":added,"omissions_by_date":omissions_by_date,"radius_mismatches":mismatched_fields,"column_semantics":{"TIMESTAMP":"UTC crossing timestamp","R":"Euclidean norm of raw GSE coordinates, Earth radii","year":"UTC calendar year","doy":"UTC ordinal day"},"selection_rule":"Set difference is measured; historical filtering rationale requires generator/provenance evidence and is not inferred from omission alone"}),
    )
}

fn request(url: &str, directory: &Path, stem: &str) -> Result<Value> {
    fs::create_dir_all(directory)?;
    let body = directory.join(format!("{stem}.body"));
    let headers = directory.join(format!("{stem}.headers"));
    ensure!(
        !body.exists() && !headers.exists(),
        "refusing to overwrite retained GET {stem}"
    );
    let start = now();
    let result = Command::new("curl")
        .args([
            "--location",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--silent",
            "--show-error",
            "--max-time",
            "45",
            "--connect-timeout",
            "15",
            "--dump-header",
        ])
        .arg(&headers)
        .arg("--output")
        .arg(&body)
        .args(["--write-out", "%{json}", url])
        .output()?;
    let end = now();
    let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
    fs::write(directory.join(format!("{stem}.stderr")), &stderr)?;
    let transport: Value = serde_json::from_slice(&result.stdout).context("curl transport JSON")?;
    let bytes = fs::read(&body).unwrap_or_default();
    let record = json!({"requested_url":url,"final_url":transport["url_effective"],"http_status":transport["http_code"],"curl_exit":result.status.code(),"started_at":start,"finished_at":end,"elapsed_seconds":transport["time_total"],"bytes":bytes.len(),"sha256":digest(&bytes),"raw_path":body,"headers_path":headers,"stderr":stderr});
    write_json(&directory.join(format!("{stem}.transport.json")), &record)?;
    Ok(record)
}
fn transport_ok(record: &Value) -> bool {
    record["curl_exit"] == 0 && record["http_status"] == 200
}

fn admit_csv(bytes: &[u8], date: NaiveDate) -> Result<Value> {
    ensure!(!bytes.is_empty(), "empty HAPI response");
    ensure!(
        !std::str::from_utf8(bytes)?.trim_start().starts_with('{'),
        "HAPI JSON status response instead of CSV"
    );
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(bytes);
    let mut rows = 0usize;
    let mut finite_rows = 0usize;
    let mut fill_rows = 0usize;
    let mut nonfinite_rows = 0usize;
    let mut duplicate_times = 0usize;
    let mut long_gaps = 0usize;
    let mut first = None;
    let mut last: Option<DateTime<Utc>> = None;
    let mut histogram = BTreeMap::<i64, usize>::new();
    let mut supplied_header = false;
    for (index, record) in reader.records().enumerate() {
        let record = record?;
        ensure!(
            record.len() == 4,
            "CSV row {} has {} columns",
            index + 1,
            record.len()
        );
        if index == 0 && record[0].eq_ignore_ascii_case("Time") {
            ensure!(
                record
                    .iter()
                    .skip(1)
                    .all(|value| value.starts_with("thd_fgs_gse")),
                "unexpected HAPI vector header"
            );
            supplied_header = true;
            continue;
        }
        let time = timestamp(&record[0])?;
        ensure!(
            time.date_naive() == date,
            "timestamp outside requested half-open UTC day: {time}"
        );
        if let Some(previous) = last {
            let delta = (time - previous)
                .num_microseconds()
                .context("cadence range")?;
            ensure!(delta >= 0, "backward timestamp at CSV row {}", index + 1);
            if delta == 0 {
                duplicate_times += 1;
            }
            if delta > 30_000_000 {
                long_gaps += 1;
            }
            *histogram.entry(delta).or_default() += 1;
        }
        first.get_or_insert(time);
        last = Some(time);
        rows += 1;
        let vector = record
            .iter()
            .skip(1)
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if vector.contains(&FILL) {
            fill_rows += 1;
        } else if vector.iter().all(|value| value.is_finite()) {
            finite_rows += 1;
        } else {
            nonfinite_rows += 1;
        }
    }
    ensure!(rows > 0 && finite_rows > 0, "no finite nonfill vectors");
    Ok(
        json!({"rows":rows,"finite_nonfill_rows":finite_rows,"fill_rows":fill_rows,"nonfinite_rows":nonfinite_rows,"duplicate_timestamps":duplicate_times,"gaps_above_30_seconds":long_gaps,"cadence_microseconds_histogram":histogram,"first_timestamp":first.map(|time|time.to_rfc3339()),"last_timestamp":last.map(|time|time.to_rfc3339()),"server_supplied_header":supplied_header,"columns":["Time","thd_fgs_gse[0]","thd_fgs_gse[1]","thd_fgs_gse[2]"],"coverage_scope":"Returned samples belong to requested UTC day; partial-day coverage and gaps are measured, not filled"}),
    )
}
fn fetch_day(date: &str, cache: &Path) -> Result<Value> {
    let parsed = NaiveDate::parse_from_str(date, "%Y-%m-%d")?;
    let next = parsed.succ_opt().context("date successor")?;
    let url = format!(
        "https://cdaweb.gsfc.nasa.gov/hapi/data?id=THD_L2_FGM%400&parameters=Time,thd_fgs_gse&time.min={date}T00:00:00Z&time.max={next}T00:00:00Z&format=csv"
    );
    let directory = cache.join(date);
    let mut attempts = Vec::new();
    for attempt in 1..=3 {
        let mut record = request(&url, &directory, &format!("attempt-{attempt}"))?;
        let admission = if transport_ok(&record) {
            admit_csv(
                &fs::read(record["raw_path"].as_str().context("raw path")?)?,
                parsed,
            )
        } else {
            Err(anyhow::anyhow!("HTTP/curl transport failure"))
        };
        match admission {
            Ok(coverage) => {
                record["admission"] = json!("admitted");
                attempts.push(record.clone());
                return Ok(
                    json!({"date":date,"status":"admitted","raw_path":record["raw_path"],"sha256":record["sha256"],"bytes":record["bytes"],"requested_url":url,"final_url":record["final_url"],"coverage":coverage,"attempts":attempts}),
                );
            }
            Err(error) => {
                record["admission"] = json!("rejected");
                record["admission_error"] = json!(format!("{error:#}"));
                attempts.push(record);
            }
        }
    }
    Ok(json!({"date":date,"status":"failed","raw_path":"","requested_url":url,"attempts":attempts}))
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 4,
        "usage: external-intake WORKTREE PRIMARY_CATALOG DERIVED_A_CATALOG plan|fetch"
    );
    let root = PathBuf::from(&args[0]).canonicalize()?;
    let catalog = PathBuf::from(&args[1]);
    let derived = PathBuf::from(&args[2]);
    let mode = args[3].to_str().context("mode encoding")?;
    ensure!(
        ["plan", "fetch"].contains(&mode),
        "mode must be plan or fetch"
    );
    let out = root.join("data/output/audit/staples-causal-validation");
    let cache = root.join(".cache/staples-external-intake");
    fs::create_dir_all(&cache)?;
    let bytes = fs::read(&catalog)?;
    ensure!(digest(&bytes) == CATALOG_HASH, "catalog identity mismatch");
    let rows = catalog_rows(&bytes)?;
    let dates: BTreeSet<String> = rows
        .iter()
        .filter(|row| row["probe"] == "d" && [2015, 2016].contains(&row["year"].as_i64().unwrap()))
        .map(|row| row["date"].as_str().unwrap().to_owned())
        .collect();
    ensure!(!dates.is_empty(), "empty external date population");
    let dates: Vec<_> = dates.into_iter().collect();
    let protocol_hash = digest(&fs::read(out.join("protocol.toml"))?);
    let plan_path = out.join("external-intake-plan.json");
    let plan = if plan_path.exists() {
        let existing: Value = serde_json::from_slice(&fs::read(&plan_path)?)?;
        ensure!(
            existing["planned_dates"] == json!(dates)
                && existing["catalog_sha256"] == CATALOG_HASH
                && existing["protocol_sha256"] == protocol_hash,
            "frozen plan mismatch"
        );
        existing
    } else {
        let plan = json!({"schema_version":1,"probe":"d","years":[2015,2016],"created_at":now(),"catalog_path":catalog,"catalog_sha256":CATALOG_HASH,"protocol_sha256":protocol_hash,"planned_dates":dates,"planned_count":dates.len(),"selection":"Every unique probe-D crossing UTC date in2015 and2016; vector retrieval follows date freeze","maximum_concurrent_downloads":4,"maximum_attempts_per_url":3,"timeout_seconds":45,"source_sha256":digest(&fs::read(out.join("external-intake.rs"))?),"executable_sha256":digest(&fs::read(std::env::current_exe()?)?)});
        write_json(&plan_path, &plan)?;
        plan
    };
    reconcile(&rows, &derived, &out)?;
    println!("frozen_dates={} mode={mode}", dates.len());
    if mode == "plan" {
        return Ok(());
    }
    let info_record_path = out.join("external-intake-info.json");
    let info_record = if info_record_path.exists() {
        serde_json::from_slice::<Value>(&fs::read(&info_record_path)?)?
    } else {
        let record = request(INFO_URL, &cache, "info")?;
        write_json(&info_record_path, &record)?;
        record
    };
    ensure!(transport_ok(&info_record), "HAPI info transport rejected");
    let info_bytes = fs::read(info_record["raw_path"].as_str().context("info path")?)?;
    ensure!(
        digest(&info_bytes) == info_record["sha256"],
        "info identity changed"
    );
    let info: Value = serde_json::from_slice(&info_bytes)?;
    ensure!(info["status"]["code"] == 1200, "HAPI info status rejected");
    let parameters = info["parameters"]
        .as_array()
        .context("HAPI parameter definitions")?;
    let time = parameters
        .iter()
        .find(|row| row["name"] == "Time")
        .context("Time metadata")?;
    let field = parameters
        .iter()
        .find(|row| row["name"] == "thd_fgs_gse")
        .context("FGS metadata")?;
    ensure!(
        time["type"] == "isotime" && time["units"] == "UTC",
        "time semantics changed"
    );
    ensure!(
        field["type"] == "double"
            && field["size"] == json!([3])
            && field["units"] == "nT GSE (All Qs)"
            && field["fill"] == "-1.0E30",
        "field semantics changed"
    );
    let results_dir = out.join("external-intake-results");
    fs::create_dir_all(&results_dir)?;
    let mut results = Vec::new();
    for batch in dates.chunks(4) {
        let returned = std::thread::scope(|scope| {
            let handles:Vec<_>=batch.iter().map(|date|{
                let cache=&cache;let results_dir=&results_dir;
                scope.spawn(move || -> Result<Value> {
                    let path=results_dir.join(format!("{date}.json"));
                    if path.exists() {
                        let value:Value=serde_json::from_slice(&fs::read(&path)?)?;
                        ensure!(value["date"]==*date,"result date mismatch");
                        if value["status"]=="admitted" {ensure!(digest(&fs::read(value["raw_path"].as_str().context("retained path")?)?)==value["sha256"],"retained bytes changed");}
                        return Ok(value);
                    }
                    let value=fetch_day(date,cache).unwrap_or_else(|error|json!({"date":date,"status":"failed","raw_path":"","execution_error":format!("{error:#}")}));
                    write_json(&path,&value)?;Ok(value)
                })
            }).collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("download worker panicked"))?
                })
                .collect::<Result<Vec<_>>>()
        })?;
        results.extend(returned);
        let admitted = results
            .iter()
            .filter(|row| row["status"] == "admitted")
            .count();
        write_json(
            &out.join("external-intake-manifest.json"),
            &json!({"schema_version":1,"probe":"d","years":[2015,2016],"catalog_sha256":CATALOG_HASH,"protocol_sha256":protocol_hash,"planned_dates":dates,"planned_count":dates.len(),"accounted_count":results.len(),"admitted_count":admitted,"failed_count":results.len()-admitted,"complete_accounting":results.len()==dates.len(),"plan_sha256":digest(&fs::read(&plan_path)?),"plan":plan,"info_record":info_record,"field_semantics":field,"results":results,"updated_at":now()}),
        )?;
        println!(
            "accounted={}/{} admitted={admitted}",
            results.len(),
            dates.len()
        );
    }
    let mut map = csv::Writer::from_path(out.join("external-file-map.csv"))?;
    map.write_record(["file_id", "path"])?;
    for (index, row) in results.iter().enumerate() {
        map.write_record([
            index.to_string(),
            row["raw_path"].as_str().unwrap_or("").to_owned(),
        ])?;
    }
    map.flush()?;
    if results.iter().any(|row| row["status"] != "admitted") {
        bail!("complete accounting retained; some planned dates failed admission");
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_preserves_duplicate_gap_and_fill_observations() -> Result<()> {
        let bytes=b"2015-01-01T00:00:00.000Z,1,2,3\n2015-01-01T00:00:00.000Z,1,2,3\n2015-01-01T00:01:00.000Z,-1e30,2,3\n";
        let result = admit_csv(bytes, NaiveDate::from_ymd_opt(2015, 1, 1).unwrap())?;
        assert_eq!(result["duplicate_timestamps"], 1);
        assert_eq!(result["gaps_above_30_seconds"], 1);
        assert_eq!(result["fill_rows"], 1);
        Ok(())
    }
    #[test]
    fn admission_rejects_status_documents_wrong_dates_and_backwards_times() {
        let date = NaiveDate::from_ymd_opt(2015, 1, 1).unwrap();
        for bytes in [
            b"{\"status\":{\"code\":1201}}".as_slice(),
            b"2015-01-02T00:00:00Z,1,2,3\n",
            b"2015-01-01T00:00:01Z,1,2,3\n2015-01-01T00:00:00Z,1,2,3\n",
        ] {
            assert!(admit_csv(bytes, date).is_err());
        }
    }
}
