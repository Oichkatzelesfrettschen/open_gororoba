// SPDX-License-Identifier: GPL-2.0-or-later

//! Sealed, offline correction of linkless remote-materializability assertions.

use super::{
    LANE_ORDER, RowCountReport, ShrinkPolicy, StagedWriteSet, classify_lane, render_infrastructure,
    render_lane, render_source_infrastructure_report,
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
use toml::Value;

const MASTER: &str = "registry/artifact_source_of_truth.toml";
const REPORT: &str = "reports/artifact_source_of_truth_reconciliation_2026_02_15.toml";
const INFRASTRUCTURE: &str = "registry/source_infrastructure.toml";
const INFRASTRUCTURE_REPORT: &str = "reports/source_infrastructure_reconciliation_2026_02_15.toml";
const REMOTE: &str = "remotely_materializable";
const CITATION: &str = "citation_only_no_link";
const EXPECTED_COUNT: usize = 533;

#[derive(Debug)]
pub struct InventoryRepairSummary {
    pub repaired_count: usize,
    pub artifact_count: usize,
    pub already_applied: bool,
    pub row_counts: Vec<RowCountReport>,
}

#[derive(Debug)]
struct Spec {
    master_sha256: String,
    report_sha256: String,
    repaired_at: String,
    witnesses: BTreeMap<String, String>,
}

fn digest(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn text<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string {field}"))
}

fn rows<'a>(value: &'a Value, field: &str) -> Result<&'a Vec<Value>> {
    value
        .get(field)
        .and_then(Value::as_array)
        .with_context(|| format!("missing rows {field}"))
}

fn empty_array(value: &Value, field: &str) -> bool {
    value
        .get(field)
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
}

fn linkless_preconditions(row: &Value) -> bool {
    [
        "all_links",
        "working_mirrors",
        "working_pdf_mirrors",
        "nonworking_mirrors",
        "unverified_mirrors",
        "downloaded_paths",
    ]
    .iter()
    .all(|field| empty_array(row, field))
        && [
            "canonical_functional_url",
            "canonical_download_path",
            "retrieval_command",
            "manual_intervention_reason",
        ]
        .iter()
        .all(|field| row.get(*field).and_then(Value::as_str) == Some(""))
        && row.get("minimum_requirement_met").and_then(Value::as_bool) == Some(false)
        && row
            .get("manual_intervention_required")
            .and_then(Value::as_bool)
            == Some(false)
        && row
            .get("key")
            .and_then(Value::as_str)
            .is_some_and(|key| key.starts_with("title:"))
}

fn find_witnesses(master: &Value) -> Result<BTreeMap<String, String>> {
    let mut witnesses = BTreeMap::new();
    for row in rows(master, "artifact")? {
        if text(row, "status")? == REMOTE && linkless_preconditions(row) {
            ensure!(
                witnesses
                    .insert(text(row, "key")?.to_owned(), text(row, "id")?.to_owned())
                    .is_none(),
                "duplicate witness key"
            );
        }
    }
    Ok(witnesses)
}

impl Spec {
    fn parse(source: &str) -> Result<Self> {
        let value: Value = toml::from_str(source)?;
        let table = value.as_table().context("repair spec must be a table")?;
        ensure!(
            table.keys().all(|key| [
                "schema_version",
                "master_sha256",
                "report_sha256",
                "repaired_at",
                "witness"
            ]
            .contains(&key.as_str())),
            "unknown repair spec field"
        );
        ensure!(
            value.get("schema_version").and_then(Value::as_integer) == Some(1),
            "unsupported repair spec version"
        );
        let mut witnesses = BTreeMap::new();
        let mut ids = BTreeSet::new();
        for row in rows(&value, "witness")? {
            ensure!(
                row.as_table()
                    .context("witness must be a table")?
                    .keys()
                    .all(|key| ["id", "key"].contains(&key.as_str())),
                "unknown witness field"
            );
            let key = text(row, "key")?;
            let id = text(row, "id")?;
            ensure!(
                !id.is_empty() && key.starts_with("title:"),
                "invalid witness identity"
            );
            ensure!(
                ids.insert(id.to_owned())
                    && witnesses.insert(key.to_owned(), id.to_owned()).is_none(),
                "duplicate witness identity"
            );
        }
        let repaired_at = text(&value, "repaired_at")?.to_owned();
        chrono::NaiveDate::parse_from_str(&repaired_at, "%Y-%m-%d")
            .context("repaired_at must be a calendar date")?;
        let spec = Self {
            master_sha256: text(&value, "master_sha256")?.to_owned(),
            report_sha256: text(&value, "report_sha256")?.to_owned(),
            repaired_at,
            witnesses,
        };
        for hash in [&spec.master_sha256, &spec.report_sha256] {
            ensure!(
                hash.len() == 64
                    && hash
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
                "invalid SHA256 seal"
            );
        }
        Ok(spec)
    }
}

/// Capture exact identities and original bytes before the bounded correction.
/// Existing specifications are accepted only when byte-identical.
pub fn write_linkless_materializability_spec(
    repo_root: &Path,
    spec_path: &Path,
    repaired_at: &str,
) -> Result<usize> {
    let master = fs::read_to_string(repo_root.join(MASTER))?;
    let report = fs::read_to_string(repo_root.join(REPORT))?;
    let value: Value = toml::from_str(&master)?;
    let witnesses = find_witnesses(&value)?;
    ensure!(
        witnesses.len() == EXPECTED_COUNT,
        "expected {EXPECTED_COUNT} linkless rows, found {}",
        witnesses.len()
    );
    let mut table = toml::map::Map::new();
    table.insert("schema_version".into(), Value::Integer(1));
    table.insert("master_sha256".into(), Value::String(digest(&master)));
    table.insert("report_sha256".into(), Value::String(digest(&report)));
    table.insert("repaired_at".into(), Value::String(repaired_at.to_owned()));
    table.insert(
        "witness".into(),
        Value::Array(
            witnesses
                .iter()
                .map(|(key, id)| {
                    let mut row = toml::map::Map::new();
                    row.insert("id".into(), Value::String(id.clone()));
                    row.insert("key".into(), Value::String(key.clone()));
                    Value::Table(row)
                })
                .collect(),
        ),
    );
    let rendered = toml::to_string_pretty(&Value::Table(table))?;
    Spec::parse(&rendered)?;
    let destination = repo_root.join(spec_path);
    if destination.exists() {
        ensure!(
            fs::read_to_string(&destination)? == rendered,
            "existing repair specification differs"
        );
    } else {
        let mut staged = StagedWriteSet::new();
        staged.stage(&destination, rendered, "[[witness]]");
        staged.commit(&ShrinkPolicy::default())?;
    }
    Ok(witnesses.len())
}

fn replace_line(block: &str, previous: &str, replacement: &str) -> Result<String> {
    ensure!(
        block.lines().filter(|line| *line == previous).count() == 1,
        "expected one exact line {previous:?}"
    );
    Ok(block
        .split_inclusive('\n')
        .map(|line| {
            if line.strip_suffix('\n').unwrap_or(line) == previous {
                format!(
                    "{replacement}{}",
                    if line.ends_with('\n') { "\n" } else { "" }
                )
            } else {
                line.to_owned()
            }
        })
        .collect())
}

fn transform(
    source: &str,
    marker: &str,
    header: &str,
    spec: &Spec,
    reverse: bool,
) -> Result<String> {
    let parsed: Value = toml::from_str(source)?;
    let from = if reverse { CITATION } else { REMOTE };
    let to = if reverse { REMOTE } else { CITATION };
    let header_value = parsed.get(header).context("inventory header missing")?;
    let remote = header_value
        .get("remotely_materializable_count")
        .and_then(Value::as_integer)
        .context("remote count missing")?;
    let citation = header_value
        .get("citation_only_no_link_count")
        .and_then(Value::as_integer)
        .context("citation count missing")?;
    let delta = spec.witnesses.len() as i64 * if reverse { -1 } else { 1 };
    ensure!(
        remote - delta >= 0 && citation + delta >= 0,
        "negative derived header count"
    );
    let delimiter = format!("[[{marker}]]\n");
    let mut blocks = source.split(&delimiter);
    let leading = blocks.next().context("inventory prefix missing")?;
    let leading = replace_line(
        leading,
        &format!("remotely_materializable_count = {remote}"),
        &format!("remotely_materializable_count = {}", remote - delta),
    )?;
    let mut result = replace_line(
        &leading,
        &format!("citation_only_no_link_count = {citation}"),
        &format!("citation_only_no_link_count = {}", citation + delta),
    )?;
    let mut observed = BTreeSet::new();
    for block in blocks {
        let row: Value = toml::from_str(block).context("parse inventory row block")?;
        result.push_str(&delimiter);
        let key = text(&row, "key")?;
        if let Some(id) = spec.witnesses.get(key) {
            ensure!(
                observed.insert(key.to_owned()),
                "duplicate repair witness row {key}"
            );
            ensure!(
                text(&row, "status")? == from,
                "unexpected repair status for {key}"
            );
            if marker == "artifact" {
                ensure!(
                    text(&row, "id")? == id && linkless_preconditions(&row),
                    "repair witness preconditions changed for {key}"
                );
            } else {
                ensure!(
                    ["all_links", "nonworking_mirrors", "unverified_mirrors"]
                        .iter()
                        .all(|field| empty_array(&row, field)),
                    "report witness carries mirror evidence: {key}"
                );
            }
            result.push_str(&replace_line(
                block,
                &format!("status = \"{from}\""),
                &format!("status = \"{to}\""),
            )?);
        } else {
            result.push_str(block);
        }
    }
    ensure!(
        observed == spec.witnesses.keys().cloned().collect(),
        "repair witness set differs from inventory rows"
    );
    let mut expected = parsed;
    for row in expected
        .get_mut(marker)
        .and_then(Value::as_array_mut)
        .context("inventory rows missing")?
    {
        if spec.witnesses.contains_key(text(row, "key")?) {
            row.as_table_mut()
                .context("inventory row table missing")?
                .insert("status".into(), Value::String(to.to_owned()));
        }
    }
    let counts = expected
        .get_mut(header)
        .and_then(Value::as_table_mut)
        .context("inventory header missing")?;
    counts.insert(
        "remotely_materializable_count".into(),
        Value::Integer(remote - delta),
    );
    counts.insert(
        "citation_only_no_link_count".into(),
        Value::Integer(citation + delta),
    );
    ensure!(
        toml::from_str::<Value>(&result)? == expected,
        "repair changes fields outside status and two header counts"
    );
    Ok(result)
}

fn prepare(master: &str, report: &str, spec: &Spec) -> Result<(String, String, bool)> {
    ensure!(!spec.witnesses.is_empty(), "empty repair witness set");
    if digest(master) == spec.master_sha256 && digest(report) == spec.report_sha256 {
        let value: Value = toml::from_str(master)?;
        ensure!(
            find_witnesses(&value)? == spec.witnesses,
            "sealed inventory has a different repair frontier"
        );
        return Ok((
            transform(master, "artifact", "artifact_source_of_truth", spec, false)?,
            transform(report, "missing_minimum_requirement", "report", spec, false)?,
            false,
        ));
    }
    let original_master = transform(master, "artifact", "artifact_source_of_truth", spec, true)?;
    let original_report = transform(report, "missing_minimum_requirement", "report", spec, true)?;
    ensure!(
        digest(&original_master) == spec.master_sha256
            && digest(&original_report) == spec.report_sha256,
        "inventory differs from sealed pre-state or exact corrected post-state"
    );
    Ok((master.to_owned(), report.to_owned(), true))
}

/// Correct only the sealed linkless frontier and regenerate its projections.
/// The operation neither opens nor imports the canonical SQLite database.
pub fn repair_linkless_materializability(
    repo_root: &Path,
    spec_path: &Path,
    audit_path: &Path,
) -> Result<InventoryRepairSummary> {
    let spec = Spec::parse(&fs::read_to_string(repo_root.join(spec_path))?)?;
    ensure!(
        spec.witnesses.len() == EXPECTED_COUNT,
        "repair requires exactly {EXPECTED_COUNT} witnesses"
    );
    let master = fs::read_to_string(repo_root.join(MASTER))?;
    let report = fs::read_to_string(repo_root.join(REPORT))?;
    let (master, report, already_applied) = prepare(&master, &report, &spec)?;
    let value: Value = toml::from_str(&master)?;
    let artifacts = rows(&value, "artifact")?;
    let mut lane_rows: BTreeMap<String, Vec<toml::map::Map<String, Value>>> = LANE_ORDER
        .iter()
        .map(|lane| ((*lane).to_owned(), Vec::new()))
        .collect();
    for row in artifacts {
        let table = row.as_table().context("artifact must be a table")?;
        let lane = classify_lane(table).0;
        lane_rows
            .get_mut(&lane)
            .context("unknown artifact lane")?
            .push(table.clone());
    }
    let mut staged = StagedWriteSet::new();
    staged.stage(&repo_root.join(MASTER), master.clone(), "[[artifact]]");
    staged.stage(
        &repo_root.join(REPORT),
        report.clone(),
        "[[missing_minimum_requirement]]",
    );
    let mut lane_files = BTreeMap::new();
    let mut lane_counts = BTreeMap::new();
    for (lane, mut lane_artifacts) in lane_rows {
        lane_artifacts.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
        let path = format!("registry/source_lanes/{lane}.toml");
        staged.stage(
            &repo_root.join(&path),
            render_lane(&lane, &lane_artifacts, &spec.repaired_at),
            "[[artifact_ref]]",
        );
        lane_counts.insert(lane.clone(), lane_artifacts.len());
        lane_files.insert(lane, path);
    }
    staged.stage(
        &repo_root.join(INFRASTRUCTURE),
        render_infrastructure(
            MASTER,
            &lane_files,
            &lane_counts,
            artifacts.len(),
            &spec.repaired_at,
        ),
        "[[lane]]",
    );
    staged.stage(
        &repo_root.join(INFRASTRUCTURE_REPORT),
        render_source_infrastructure_report(&lane_counts, artifacts.len(), &spec.repaired_at),
        "[[lane_summary]]",
    );
    let audit_path = repo_root.join(audit_path);
    ensure!(
        audit_path
            .starts_with(repo_root.join("data/output/audit/artifact-remote-materializability"))
            && audit_path
                .components()
                .all(|part| !matches!(part, std::path::Component::ParentDir)),
        "audit output must remain inside the repair evidence directory"
    );
    ensure!(
        audit_path != repo_root.join(spec_path),
        "audit output must differ from repair specification"
    );
    let audit = format!(
        "schema_version = 1\nrepaired_count = {}\nartifact_count = {}\nmaster_before_sha256 = {:?}\nmaster_after_sha256 = {:?}\nreport_before_sha256 = {:?}\nreport_after_sha256 = {:?}\npreserved_fields = \"Every row field except selected status; all header fields except remote and citation counts\"\ncanonical_database_action = \"unopened\"\n",
        spec.witnesses.len(),
        artifacts.len(),
        spec.master_sha256,
        digest(&master),
        spec.report_sha256,
        digest(&report)
    );
    staged.stage(&audit_path, audit, "[[witness]]");
    let row_counts = staged.commit(&ShrinkPolicy::default())?;
    Ok(InventoryRepairSummary {
        repaired_count: spec.witnesses.len(),
        artifact_count: artifacts.len(),
        already_applied,
        row_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (String, String, Spec) {
        let master = "[artifact_source_of_truth]\nremotely_materializable_count = 1\ncitation_only_no_link_count = 0\n\n[[artifact]]\nid = \"ASOT-1\"\nkey = \"title:local\"\nstatus = \"remotely_materializable\"\nall_links = []\nworking_mirrors = []\nworking_pdf_mirrors = []\nnonworking_mirrors = []\nunverified_mirrors = []\ndownloaded_paths = []\ncanonical_functional_url = \"\"\ncanonical_download_path = \"\"\nretrieval_command = \"\"\nmanual_intervention_reason = \"\"\nminimum_requirement_met = false\nmanual_intervention_required = false\nhost_only_path_count = 2\nsha256 = \"retained hash\"\nunknown_field = \"retain me\"\n".to_owned();
        let report = "[report]\nremotely_materializable_count = 1\ncitation_only_no_link_count = 0\n\n[[missing_minimum_requirement]]\nkey = \"title:local\"\nstatus = \"remotely_materializable\"\nall_links = []\nnonworking_mirrors = []\nunverified_mirrors = []\n".to_owned();
        let spec = Spec {
            master_sha256: digest(&master),
            report_sha256: digest(&report),
            repaired_at: "2026-09-04".into(),
            witnesses: BTreeMap::from([("title:local".into(), "ASOT-1".into())]),
        };
        (master, report, spec)
    }

    #[test]
    fn correction_preserves_unknown_fields_and_replays_byte_identically() -> Result<()> {
        let (master, report, spec) = fixture();
        let (corrected, corrected_report, applied) = prepare(&master, &report, &spec)?;
        assert!(!applied);
        assert!(corrected.contains(
            "host_only_path_count = 2\nsha256 = \"retained hash\"\nunknown_field = \"retain me\""
        ));
        let (replayed, replayed_report, applied) = prepare(&corrected, &corrected_report, &spec)?;
        assert!(applied);
        assert_eq!(replayed, corrected);
        assert_eq!(replayed_report, corrected_report);
        Ok(())
    }

    #[test]
    fn sealed_repair_rejects_metadata_drift_and_mixed_application() -> Result<()> {
        let (master, report, spec) = fixture();
        let (corrected, corrected_report, _) = prepare(&master, &report, &spec)?;
        assert!(
            prepare(
                &corrected.replace("retain me", "changed"),
                &corrected_report,
                &spec
            )
            .is_err()
        );
        assert!(prepare(&corrected, &report, &spec).is_err());
        assert!(prepare(&master.replace("retained hash", "changed"), &report, &spec).is_err());
        Ok(())
    }

    #[test]
    fn sealed_repair_rejects_partial_witnesses_and_mirror_evidence() {
        let (master, report, mut spec) = fixture();
        spec.witnesses
            .insert("title:absent".into(), "ASOT-2".into());
        assert!(prepare(&master, &report, &spec).is_err());
        let (master, report, mut spec) = fixture();
        let master = master.replace("all_links = []", "all_links = [\"https://example.org\"]");
        spec.master_sha256 = digest(&master);
        assert!(prepare(&master, &report, &spec).is_err());
    }
}
