use anyhow::{Context, Result, bail};
use chrono::Utc;
use clap::Parser;
use gororoba_cli_data::project_api_contract::{
    AcquisitionJournalRow, ProjectApiContext, ProjectApiCrosswalkBinding, ProjectApiCrosswalkFile,
    load_acquisition_journal_rows, load_project_api_context, load_project_api_crosswalk,
    project_relative_path, split_journal_multi_value,
};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "cd-cache-reconcile",
    about = "Reconcile the CayleyDickson cache/project-api derived ledgers from the append-only acquisition journal"
)]
struct Cli {
    #[arg(long)]
    project_api_root: PathBuf,
}

#[derive(Debug, Clone)]
struct RowLedgerRow {
    source_scope: String,
    row_id: String,
    year: String,
    century_bucket: String,
    authors: String,
    title: String,
    family_or_kind: String,
    relationship_or_location: String,
    target_location: String,
    source_status: String,
    availability_bucket: String,
    evidence_url: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let project_api = load_project_api_context(&cli.project_api_root)?;
    reconcile_project_api(&project_api)
}

fn reconcile_project_api(project_api: &ProjectApiContext) -> Result<()> {
    let crosswalk = load_project_api_crosswalk(&project_api.crosswalk_path)?;
    let journal_rows = load_acquisition_journal_rows(&project_api.acquisition_journal_path)?;
    let mut ledger_rows = load_row_ledger(&project_api.row_ledger_path)?;
    let latest_successes = latest_contract_successes(&journal_rows);

    for row in &mut ledger_rows {
        let row_ref = format!("{}:{}", row.source_scope, row.row_id);
        if let Some(journal_row) = latest_successes.get(&row_ref) {
            apply_journal_success_to_row(row, journal_row);
        }
    }

    write_row_ledger(&project_api.row_ledger_path, &ledger_rows)?;
    write_century_bucket_summary(&project_api.century_bucket_summary_path, &ledger_rows)?;
    write_availability_summary(
        &project_api.availability_summary_path,
        &ledger_rows,
        &crosswalk,
    )?;
    reconcile_inventory(
        &project_api.inventory_path,
        &ledger_rows,
        &crosswalk,
        &journal_rows,
    )?;

    println!(
        "project_api_reconciled={} journal_rows={} derived_files=4",
        project_api.project_api_dir.display(),
        journal_rows.len()
    );
    println!(
        "row_ledger={}",
        project_relative_path(&project_api.repo_root, &project_api.row_ledger_path)
    );
    println!(
        "century_bucket_summary={}",
        project_relative_path(
            &project_api.repo_root,
            &project_api.century_bucket_summary_path
        )
    );
    println!(
        "availability_summary={}",
        project_relative_path(
            &project_api.repo_root,
            &project_api.availability_summary_path
        )
    );
    println!(
        "inventory={}",
        project_relative_path(&project_api.repo_root, &project_api.inventory_path)
    );
    Ok(())
}

fn load_row_ledger(path: &Path) -> Result<Vec<RowLedgerRow>> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut lines = raw.lines();
    let Some(header) = lines.next() else {
        return Ok(Vec::new());
    };
    if header.trim() != RowLedgerRow::header() {
        bail!("unexpected row ledger header in {}", path.display());
    }
    let mut rows = Vec::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        rows.push(RowLedgerRow::from_tsv_line(line)?);
    }
    Ok(rows)
}

fn write_row_ledger(path: &Path, rows: &[RowLedgerRow]) -> Result<()> {
    let mut body = String::new();
    body.push_str(RowLedgerRow::header());
    body.push('\n');
    for row in rows {
        body.push_str(&row.to_tsv_line());
        body.push('\n');
    }
    fs::write(path, body).with_context(|| format!("write {}", path.display()))
}

fn latest_contract_successes(
    journal_rows: &[AcquisitionJournalRow],
) -> HashMap<String, AcquisitionJournalRow> {
    let mut latest: HashMap<String, AcquisitionJournalRow> = HashMap::new();
    for row in journal_rows
        .iter()
        .filter(|row| journal_row_updates_contract(row))
    {
        for row_ref in split_journal_multi_value(&row.row_ledger_refs) {
            match latest.get(&row_ref) {
                Some(existing) if existing.at_utc >= row.at_utc => {}
                _ => {
                    latest.insert(row_ref, row.clone());
                }
            }
        }
    }
    latest
}

fn journal_row_updates_contract(row: &AcquisitionJournalRow) -> bool {
    !row.project_artifact_rel.is_empty()
        && (matches!(row.outcome.as_str(), "downloaded" | "skipped_existing")
            || (row.action == "record" && row.status == "downloaded"))
}

fn apply_journal_success_to_row(row: &mut RowLedgerRow, journal_row: &AcquisitionJournalRow) {
    if row.source_scope == "candidate" {
        row.source_status = format!("downloaded_{}", journal_date_token(&journal_row.at_utc));
        row.availability_bucket = "downloaded_and_free_online".to_string();
        if row.evidence_url.is_empty() && !journal_row.effective_url.is_empty() {
            row.evidence_url = journal_row.effective_url.clone();
        }
    } else if row.source_scope == "chronology" {
        row.source_status = "[ON DISK]".to_string();
        row.availability_bucket = "downloaded".to_string();
        if row.evidence_url.is_empty() && !journal_row.effective_url.is_empty() {
            row.evidence_url = journal_row.effective_url.clone();
        }
    }
}

fn journal_date_token(at_utc: &str) -> String {
    at_utc
        .get(0..10)
        .unwrap_or("unknown-unknown-unknown")
        .replace('-', "_")
}

fn write_century_bucket_summary(path: &Path, rows: &[RowLedgerRow]) -> Result<()> {
    let mut buckets: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for row in rows {
        let counts = buckets.entry(row.century_bucket.clone()).or_default();
        *counts.entry("total_rows".to_string()).or_default() += 1;
        if row.source_scope == "chronology" {
            *counts.entry("chronology_rows".to_string()).or_default() += 1;
        } else if row.source_scope == "candidate" {
            *counts.entry("candidate_rows".to_string()).or_default() += 1;
        }
        *counts.entry(row.availability_bucket.clone()).or_default() += 1;
    }

    let ordered = [
        "1600-1699",
        "1700-1799",
        "1800-1899",
        "1900-1999",
        "2000-2026",
        "unknown",
    ];
    let mut body = String::from(
        "century_bucket\ttotal_rows\tchronology_rows\tcandidate_rows\tdownloaded\tformalized_no_local_pdf\tdownloaded_and_free_online\talready_on_disk_and_free_online\tfree_online_browser_pdf_lane_unfetched\tmetadata_open_access_path_not_yet_pinned\tgated_holder_workflow\tgated_article_delivery\n",
    );
    for bucket in ordered {
        if let Some(counts) = buckets.get(bucket) {
            let line = [
                bucket.to_string(),
                counts
                    .get("total_rows")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                counts
                    .get("chronology_rows")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                counts
                    .get("candidate_rows")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                counts
                    .get("downloaded")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                counts
                    .get("formalized_no_local_pdf")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                counts
                    .get("downloaded_and_free_online")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                counts
                    .get("already_on_disk_and_free_online")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                counts
                    .get("free_online_browser_pdf_lane_unfetched")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                counts
                    .get("metadata_open_access_path_not_yet_pinned")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                counts
                    .get("gated_holder_workflow")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                counts
                    .get("gated_article_delivery")
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
            ]
            .join("\t");
            body.push_str(&line);
            body.push('\n');
        }
    }
    fs::write(path, body).with_context(|| format!("write {}", path.display()))
}

fn write_availability_summary(
    path: &Path,
    rows: &[RowLedgerRow],
    crosswalk: &ProjectApiCrosswalkFile,
) -> Result<()> {
    let chronology_rows = rows
        .iter()
        .filter(|row| row.source_scope == "chronology")
        .count();
    let candidate_rows = rows
        .iter()
        .filter(|row| row.source_scope == "candidate")
        .count();
    let on_disk_or_formalized = rows
        .iter()
        .filter(|row| {
            row.source_scope == "chronology"
                && matches!(
                    row.availability_bucket.as_str(),
                    "downloaded" | "formalized_no_local_pdf"
                )
        })
        .count();
    let exact_missing_rows = rows
        .iter()
        .filter(|row| {
            row.source_scope == "chronology"
                && matches!(
                    row.availability_bucket.as_str(),
                    "gated_holder_workflow" | "gated_article_delivery"
                )
        })
        .count();
    let unresolved_exact_ids = unresolved_exact_gap_ids(rows, crosswalk);
    let downloaded_and_free_online = count_candidate_bucket(rows, "downloaded_and_free_online");
    let already_on_disk_and_free_online =
        count_candidate_bucket(rows, "already_on_disk_and_free_online");
    let free_online_browser_pdf_lane_unfetched =
        count_candidate_bucket(rows, "free_online_browser_pdf_lane_unfetched");
    let metadata_open_access_path_not_yet_pinned =
        count_candidate_bucket(rows, "metadata_open_access_path_not_yet_pinned");
    let downloaded_with_explicit_date = rows
        .iter()
        .filter(|row| {
            row.source_scope == "candidate" && row.source_status.starts_with("downloaded_")
        })
        .count();
    let downloaded_2026_03_26 = rows
        .iter()
        .filter(|row| {
            row.source_scope == "candidate" && row.source_status == "downloaded_2026_03_26"
        })
        .count();

    let body = format!(
        concat!(
            "schema_version = 1\n",
            "project_id = \"cayley-dickson\"\n",
            "last_updated = \"{}\"\n\n",
            "[current_corpus]\n",
            "tracked_rows = {}\n",
            "on_disk_or_formalized = {}\n",
            "exact_missing_rows = {}\n",
            "mislabeled_rows = 0\n\n",
            "[exact_gap_status]\n",
            "free_online_exact_originals = 0\n",
            "holder_workflow_or_gated = {}\n",
            "rows = {}\n\n",
            "[century_expansion_status]\n",
            "candidate_rows = {}\n",
            "downloaded_and_free_online = {}\n",
            "already_on_disk_and_free_online = {}\n",
            "free_online_browser_pdf_lane_unfetched = {}\n",
            "metadata_open_access_path_not_yet_pinned = {}\n",
            "downloaded_with_explicit_date = {}\n",
            "downloaded_2026_03_26 = {}\n\n",
            "[ledger]\n",
            "row_count = {}\n",
            "chronology_rows = {}\n",
            "candidate_rows = {}\n",
            "ledger_path = \"metadata/project_api/row_availability_ledger.tsv\"\n",
            "century_bucket_summary_path = \"metadata/project_api/century_bucket_summary.tsv\"\n\n",
            "[notes]\n",
            "interpretation = \"The row ledger now mixes normalized chronology rows with strict-century expansion candidates so downloaded, free-online, browser-challenge, and gated lanes are explicit in one machine-readable table.\"\n",
            "warning = \"Most already-downloaded chronology rows have not yet had their present-day external access state revalidated; downloaded remains the authoritative local-state label for those rows.\"\n",
            "overlap = \"Some candidate rows are stricter bibliographic anchors for chronology topics already represented on disk, especially Euler E445 and the de Marrais 2002 terminology anchor.\"\n",
            "journal_rule = \"Only journal rows with a project_artifact_rel inside the external repo are allowed to upgrade derived download state.\"\n"
        ),
        utc_today(),
        chronology_rows,
        on_disk_or_formalized,
        exact_missing_rows,
        unresolved_exact_ids.len(),
        toml_array(&unresolved_exact_ids),
        candidate_rows,
        downloaded_and_free_online,
        already_on_disk_and_free_online,
        free_online_browser_pdf_lane_unfetched,
        metadata_open_access_path_not_yet_pinned,
        downloaded_with_explicit_date,
        downloaded_2026_03_26,
        rows.len(),
        chronology_rows,
        candidate_rows
    );
    fs::write(path, body).with_context(|| format!("write {}", path.display()))
}

fn unresolved_exact_gap_ids(
    rows: &[RowLedgerRow],
    crosswalk: &ProjectApiCrosswalkFile,
) -> Vec<String> {
    let row_lookup = row_lookup_map(rows);
    let mut ids = Vec::new();
    for binding in &crosswalk.binding {
        let Some(inventory_blocker_id) = &binding.inventory_blocker_id else {
            continue;
        };
        if binding.chronology_row_id.is_some() && binding_is_unresolved(binding, &row_lookup) {
            ids.push(inventory_blocker_id.clone());
        }
    }
    ids
}

fn count_candidate_bucket(rows: &[RowLedgerRow], bucket: &str) -> usize {
    rows.iter()
        .filter(|row| row.source_scope == "candidate" && row.availability_bucket == bucket)
        .count()
}

fn row_lookup_map(rows: &[RowLedgerRow]) -> HashMap<String, RowLedgerRow> {
    rows.iter()
        .cloned()
        .map(|row| (format!("{}:{}", row.source_scope, row.row_id), row))
        .collect()
}

fn binding_is_unresolved(
    binding: &ProjectApiCrosswalkBinding,
    row_lookup: &HashMap<String, RowLedgerRow>,
) -> bool {
    binding.row_ledger_refs.iter().any(|row_ref| {
        row_lookup
            .get(row_ref)
            .map(|row| !row_is_resolved(row))
            .unwrap_or(true)
    })
}

fn row_is_resolved(row: &RowLedgerRow) -> bool {
    matches!(
        row.availability_bucket.as_str(),
        "downloaded"
            | "formalized_no_local_pdf"
            | "downloaded_and_free_online"
            | "already_on_disk_and_free_online"
    )
}

fn reconcile_inventory(
    path: &Path,
    rows: &[RowLedgerRow],
    crosswalk: &ProjectApiCrosswalkFile,
    journal_rows: &[AcquisitionJournalRow],
) -> Result<()> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut inventory: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    let chronology_rows = rows
        .iter()
        .filter(|row| row.source_scope == "chronology")
        .count();
    let candidate_rows = rows
        .iter()
        .filter(|row| row.source_scope == "candidate")
        .count();
    let on_disk_or_formalized = rows
        .iter()
        .filter(|row| {
            row.source_scope == "chronology"
                && matches!(
                    row.availability_bucket.as_str(),
                    "downloaded" | "formalized_no_local_pdf"
                )
        })
        .count();
    let exact_missing_rows = rows
        .iter()
        .filter(|row| {
            row.source_scope == "chronology"
                && matches!(
                    row.availability_bucket.as_str(),
                    "gated_holder_workflow" | "gated_article_delivery"
                )
        })
        .count();
    let row_lookup = row_lookup_map(rows);
    let active_blockers = derive_active_retrieval_blockers(crosswalk, &row_lookup);

    inventory["audit_checkpoint_date"] = serde_json::json!(utc_today());
    inventory["summary"]["tracked_rows"] = serde_json::json!(chronology_rows);
    inventory["summary"]["on_disk_or_formalized"] = serde_json::json!(on_disk_or_formalized);
    inventory["summary"]["missing_or_audit"] = serde_json::json!(exact_missing_rows);
    inventory["summary"]["candidate_rows"] = serde_json::json!(candidate_rows);
    inventory["summary"]["row_ledger_rows"] = serde_json::json!(rows.len());

    inventory["availability_audit"]["chronology_downloaded_rows"] = serde_json::json!(
        rows.iter()
            .filter(
                |row| row.source_scope == "chronology" && row.availability_bucket == "downloaded"
            )
            .count()
    );
    inventory["availability_audit"]["chronology_formalized_without_local_pdf_rows"] = serde_json::json!(
        rows.iter()
            .filter(|row| {
                row.source_scope == "chronology"
                    && row.availability_bucket == "formalized_no_local_pdf"
            })
            .count()
    );
    inventory["availability_audit"]["chronology_exact_gated_rows"] =
        serde_json::json!(exact_missing_rows);
    inventory["availability_audit"]["downloaded_and_free_online_candidates"] =
        serde_json::json!(count_candidate_bucket(rows, "downloaded_and_free_online"));
    inventory["availability_audit"]["already_on_disk_and_free_online_candidates"] = serde_json::json!(
        count_candidate_bucket(rows, "already_on_disk_and_free_online")
    );
    inventory["availability_audit"]["free_online_browser_pdf_lane_unfetched_candidates"] = serde_json::json!(
        count_candidate_bucket(rows, "free_online_browser_pdf_lane_unfetched")
    );
    inventory["availability_audit"]["metadata_open_access_path_not_yet_pinned_candidates"] = serde_json::json!(
        count_candidate_bucket(rows, "metadata_open_access_path_not_yet_pinned")
    );
    inventory["active_retrieval_blockers"] = serde_json::json!(active_blockers);

    if let Some(exact_hard_gaps) = inventory["exact_hard_gaps"].as_array_mut() {
        for gap in exact_hard_gaps {
            let Some(id) = gap["id"].as_str() else {
                continue;
            };
            let recovered = crosswalk
                .binding
                .iter()
                .find(|binding| binding.inventory_blocker_id.as_deref() == Some(id))
                .map(|binding| !binding_is_unresolved(binding, &row_lookup))
                .unwrap_or(false);
            if recovered {
                gap["status"] = serde_json::json!("recovered");
                gap["download_state"] = serde_json::json!("exact_original_on_disk");
            }
        }
    }

    inventory["recent_downloads"] = serde_json::Value::Array(reconciled_recent_downloads(
        inventory["recent_downloads"].as_array(),
        rows,
        journal_rows,
    ));

    let formatted = serde_json::to_string_pretty(&inventory)?;
    fs::write(path, format!("{formatted}\n")).with_context(|| format!("write {}", path.display()))
}

fn derive_active_retrieval_blockers(
    crosswalk: &ProjectApiCrosswalkFile,
    row_lookup: &HashMap<String, RowLedgerRow>,
) -> Vec<String> {
    let mut blockers = Vec::new();
    for binding in &crosswalk.binding {
        let Some(inventory_blocker_id) = &binding.inventory_blocker_id else {
            continue;
        };
        if binding_is_unresolved(binding, row_lookup) {
            blockers.push(inventory_blocker_id.clone());
        }
    }
    blockers
}

fn reconciled_recent_downloads(
    existing: Option<&Vec<serde_json::Value>>,
    rows: &[RowLedgerRow],
    journal_rows: &[AcquisitionJournalRow],
) -> Vec<serde_json::Value> {
    let row_lookup = row_lookup_map(rows);
    let mut derived = Vec::new();
    let mut seen = HashSet::new();
    let mut successes = journal_rows
        .iter()
        .filter(|row| journal_row_updates_contract(row))
        .cloned()
        .collect::<Vec<_>>();
    successes.sort_by(|left, right| right.at_utc.cmp(&left.at_utc));
    for row in successes {
        let id = if !row.inventory_blocker_id.is_empty() {
            row.inventory_blocker_id.clone()
        } else if !row.candidate_id.is_empty() {
            row.candidate_id.clone()
        } else if !row.crosswalk_id.is_empty() {
            row.crosswalk_id.clone()
        } else {
            row.session_id.clone()
        };
        if !seen.insert(id.clone()) {
            continue;
        }
        let year = split_journal_multi_value(&row.row_ledger_refs)
            .first()
            .and_then(|row_ref| row_lookup.get(row_ref))
            .and_then(|ledger_row| ledger_row.year.parse::<i64>().ok())
            .unwrap_or_default();
        derived.push(serde_json::json!({
            "id": id,
            "year": year,
            "location": row.project_artifact_rel,
        }));
    }
    if let Some(existing) = existing {
        for entry in existing {
            let Some(id) = entry["id"].as_str() else {
                continue;
            };
            if seen.insert(id.to_string()) {
                derived.push(entry.clone());
            }
        }
    }
    derived
}

fn toml_array(values: &[String]) -> String {
    let items = values
        .iter()
        .map(|value| format!("\"{}\"", value.replace('\"', "\\\"")))
        .collect::<Vec<_>>();
    format!("[{}]", items.join(", "))
}

fn utc_today() -> String {
    Utc::now().format("%Y-%m-%d").to_string()
}

fn escape_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n'], " ")
}

impl RowLedgerRow {
    fn header() -> &'static str {
        "source_scope\trow_id\tyear\tcentury_bucket\tauthors\ttitle\tfamily_or_kind\trelationship_or_location\ttarget_location\tsource_status\tavailability_bucket\tevidence_url"
    }

    fn from_tsv_line(line: &str) -> Result<Self> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 12 {
            bail!("row ledger expected 12 fields, found {}", fields.len());
        }
        Ok(Self {
            source_scope: fields[0].to_string(),
            row_id: fields[1].to_string(),
            year: fields[2].to_string(),
            century_bucket: fields[3].to_string(),
            authors: fields[4].to_string(),
            title: fields[5].to_string(),
            family_or_kind: fields[6].to_string(),
            relationship_or_location: fields[7].to_string(),
            target_location: fields[8].to_string(),
            source_status: fields[9].to_string(),
            availability_bucket: fields[10].to_string(),
            evidence_url: fields[11].to_string(),
        })
    }

    fn to_tsv_line(&self) -> String {
        [
            &self.source_scope,
            &self.row_id,
            &self.year,
            &self.century_bucket,
            &self.authors,
            &self.title,
            &self.family_or_kind,
            &self.relationship_or_location,
            &self.target_location,
            &self.source_status,
            &self.availability_bucket,
            &self.evidence_url,
        ]
        .iter()
        .map(|value| escape_tsv_field(value))
        .collect::<Vec<_>>()
        .join("\t")
    }
}
