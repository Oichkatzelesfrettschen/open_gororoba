use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    process::Command,
};
use toml::Value;
fn baseline(path: &str) -> Vec<u8> {
    let out = Command::new("git")
        .args([
            "show",
            &format!("91467ad14eeeade63177e15f45acc592564b7869:{path}"),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    out.stdout
}
fn value(bytes: &[u8]) -> Value {
    toml::from_str(std::str::from_utf8(bytes).unwrap()).unwrap()
}
fn txt<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key].as_str().unwrap()
}
fn main() {
    let master = "registry/artifact_source_of_truth.toml";
    let before = value(&baseline(master));
    let after = value(&fs::read(master).unwrap());
    let originals = before["artifact"].as_array().unwrap();
    let artifacts = after["artifact"].as_array().unwrap();
    assert_eq!(originals.len(), artifacts.len());
    let mut changed = BTreeMap::new();
    let mut by_lane = BTreeMap::<String, usize>::new();
    let mut blank_hash = 0;
    let mut expected = before.clone();
    for (index, (old, new)) in originals.iter().zip(artifacts).enumerate() {
        assert_eq!(old["id"], new["id"]);
        assert_eq!(old["key"], new["key"]);
        if old != new {
            assert_eq!(txt(old, "status"), "remotely_materializable");
            assert_eq!(txt(new, "status"), "citation_only_no_link");
            let mut corrected = old.clone();
            corrected["status"] = Value::String("citation_only_no_link".into());
            assert_eq!(&corrected, new);
            for field in [
                "all_links",
                "working_mirrors",
                "working_pdf_mirrors",
                "nonworking_mirrors",
                "unverified_mirrors",
                "downloaded_paths",
            ] {
                assert!(old[field].as_array().unwrap().is_empty());
            }
            for field in [
                "canonical_functional_url",
                "canonical_download_path",
                "retrieval_command",
            ] {
                assert_eq!(txt(old, field), "");
            }
            assert_eq!(old["minimum_requirement_met"].as_bool(), Some(false));
            assert!(
                changed
                    .insert(txt(old, "key").to_owned(), txt(old, "id").to_owned())
                    .is_none()
            );
            *by_lane.entry(txt(old, "lane").into()).or_default() += 1;
            if txt(old, "sha256").is_empty() {
                blank_hash += 1;
            }
            expected["artifact"].as_array_mut().unwrap()[index] = corrected;
        }
    }
    assert_eq!(changed.len(), 533);
    expected["artifact_source_of_truth"]["remotely_materializable_count"] = Value::Integer(40);
    expected["artifact_source_of_truth"]["citation_only_no_link_count"] = Value::Integer(2843);
    assert_eq!(expected, after);
    let report_path = "reports/artifact_source_of_truth_reconciliation_2026_02_15.toml";
    let mut report = value(&baseline(report_path));
    let mut report_changed = 0;
    for row in report["missing_minimum_requirement"]
        .as_array_mut()
        .unwrap()
    {
        if changed.contains_key(txt(row, "key")) {
            assert_eq!(txt(row, "status"), "remotely_materializable");
            row["status"] = Value::String("citation_only_no_link".into());
            report_changed += 1;
        }
    }
    report["report"]["remotely_materializable_count"] = Value::Integer(40);
    report["report"]["citation_only_no_link_count"] = Value::Integer(2843);
    assert_eq!(report, value(&fs::read(report_path).unwrap()));
    assert_eq!(report_changed, 533);
    let by_id: BTreeMap<_, _> = artifacts.iter().map(|row| (txt(row, "id"), row)).collect();
    assert_eq!(by_id.len(), artifacts.len());
    let mut observed = BTreeSet::new();
    let mut lane_members = BTreeMap::new();
    for lane in [
        "datasets",
        "papers_pdf",
        "slides_artifacts",
        "web_references",
    ] {
        let inventory = value(&fs::read(format!("registry/source_lanes/{lane}.toml")).unwrap());
        let members = inventory["artifact_ref"].as_array().unwrap();
        assert_eq!(
            members.len() as i64,
            inventory["lane"]["artifact_count"].as_integer().unwrap()
        );
        let mut statuses = BTreeMap::<String, i64>::new();
        let mut missing = 0;
        for row in members {
            let id = txt(row, "id");
            assert!(observed.insert(id.to_owned()));
            let owner = by_id[id];
            assert_eq!(txt(owner, "lane"), lane);
            for (field, derived) in row.as_table().unwrap() {
                let original = owner.get(field).unwrap();
                let projected = match original {
                    Value::String(value) => Value::String(value.trim().to_owned()),
                    Value::Array(values) => Value::Array(
                        values
                            .iter()
                            .map(|value| value.as_str().unwrap().trim())
                            .filter(|value| !value.is_empty())
                            .map(|value| Value::String(value.to_owned()))
                            .collect(),
                    ),
                    other => other.clone(),
                };
                assert_eq!(
                    &projected, derived,
                    "lane {lane} artifact {id} field {field}"
                );
            }
            *statuses.entry(txt(row, "status").into()).or_default() += 1;
            if row["minimum_requirement_met"].as_bool() == Some(false) {
                missing += 1;
            }
        }
        for status in [
            "downloaded",
            "remotely_materializable",
            "downloadable",
            "blocked",
            "citation_only_no_link",
            "unverified",
        ] {
            assert_eq!(
                inventory["lane"][format!("{status}_count")]
                    .as_integer()
                    .unwrap(),
                *statuses.get(status).unwrap_or(&0)
            );
        }
        assert_eq!(
            inventory["lane"]["missing_minimum_requirement_count"]
                .as_integer()
                .unwrap(),
            missing
        );
        lane_members.insert(lane, members.len());
    }
    assert_eq!(observed, by_id.keys().map(|id| (*id).to_owned()).collect());
    let canonical = "registry/canonical/control_plane.sqlite3";
    assert_eq!(baseline(canonical), fs::read(canonical).unwrap());
    let spec = value(
        &fs::read("data/output/audit/artifact-remote-materializability/repair-spec.toml").unwrap(),
    );
    let witnesses: BTreeMap<_, _> = spec["witness"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| (txt(row, "key").to_owned(), txt(row, "id").to_owned()))
        .collect();
    assert_eq!(witnesses, changed);
    println!(
        "baseline_commit = {:?}",
        String::from_utf8(
            Command::new("git")
                .args(["rev-parse", "91467ad14eeeade63177e15f45acc592564b7869"])
                .output()
                .unwrap()
                .stdout
        )
        .unwrap()
        .trim()
    );
    println!(
        "artifact_count = {}\nchanged_rows = {}\nchanged_fields_per_row = [\"status\"]\nchanged_header_fields = [\"remotely_materializable_count\", \"citation_only_no_link_count\"]\nother_master_fields_identical = true\nother_reconciliation_fields_identical = true\nreconciliation_changed_rows = {}\ncanonical_database_bytes_identical = true\nexact_spec_witness_set = true\nlane_unique_members = {}\nlane_all_projected_fields_match_master_under_documented_whitespace_normalization = true\nlane_status_counts_match = true\nempty_hashes_retained = {}\nnonempty_hashes_retained = {}",
        artifacts.len(),
        changed.len(),
        report_changed,
        observed.len(),
        blank_hash,
        changed.len() - blank_hash
    );
    println!("\n[changed_rows_by_lane]");
    for (lane, count) in by_lane {
        println!("{lane} = {count}");
    }
    println!("\n[lane_members]");
    for (lane, count) in lane_members {
        println!("{lane} = {count}");
    }
}
