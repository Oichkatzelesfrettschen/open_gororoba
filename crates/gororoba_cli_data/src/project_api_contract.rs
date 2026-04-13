use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectApiContract {
    pub schema_version: Option<u32>,
    pub project_id: Option<String>,
    pub entrypoints: ProjectApiEntrypoints,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectApiEntrypoints {
    pub inventory: String,
    pub eras: String,
    pub gaps: String,
    pub nomenclature: String,
    pub candidate_expansion: String,
    pub crosswalk: String,
    pub acquisition_journal: String,
    pub availability_summary: String,
    pub row_availability_ledger: String,
    pub century_bucket_summary: String,
    pub search_queue: String,
    pub acquisition_stack: String,
}

#[derive(Debug, Clone)]
pub struct ProjectApiContext {
    pub repo_root: PathBuf,
    pub project_api_dir: PathBuf,
    pub project_toml_path: PathBuf,
    pub inventory_path: PathBuf,
    pub eras_path: PathBuf,
    pub gaps_path: PathBuf,
    pub nomenclature_path: PathBuf,
    pub candidate_expansion_path: PathBuf,
    pub crosswalk_path: PathBuf,
    pub acquisition_journal_path: PathBuf,
    pub availability_summary_path: PathBuf,
    pub row_ledger_path: PathBuf,
    pub century_bucket_summary_path: PathBuf,
    pub search_queue_path: PathBuf,
    pub acquisition_stack_path: PathBuf,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectApiCrosswalkFile {
    pub schema_version: Option<u32>,
    pub project_id: Option<String>,
    pub last_updated: Option<String>,
    pub binding: Vec<ProjectApiCrosswalkBinding>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectApiCrosswalkBinding {
    pub id: String,
    pub search_target_id: String,
    pub kind: String,
    pub candidate_id: Option<String>,
    pub chronology_row_id: Option<String>,
    pub inventory_blocker_id: Option<String>,
    pub row_ledger_refs: Vec<String>,
    pub gap_ids: Vec<String>,
    pub era_ids: Vec<String>,
    pub nomenclature_ids: Vec<String>,
    pub notes: String,
}

#[derive(Debug, Clone, Default)]
pub struct AcquisitionJournalRow {
    pub at_utc: String,
    pub action: String,
    pub session_id: String,
    pub parent_session_id: String,
    pub source_id: String,
    pub search_target_id: String,
    pub status: String,
    pub outcome: String,
    pub url: String,
    pub effective_url: String,
    pub artifact_rel: String,
    pub project_artifact_rel: String,
    pub bytes: String,
    pub sha256: String,
    pub http_code: String,
    pub request_capsule_rel: String,
    pub browser_trace_rel: String,
    pub cookie_jar_rel: String,
    pub storage_state_rel: String,
    pub profile_bundle_rel: String,
    pub transport_substrate: String,
    pub requested_backend: String,
    pub crosswalk_id: String,
    pub candidate_id: String,
    pub chronology_row_id: String,
    pub inventory_blocker_id: String,
    pub row_ledger_refs: String,
    pub gap_ids: String,
    pub era_ids: String,
    pub nomenclature_ids: String,
    pub note: String,
}

impl AcquisitionJournalRow {
    pub fn header() -> &'static str {
        "at_utc\taction\tsession_id\tparent_session_id\tsource_id\tsearch_target_id\tstatus\toutcome\turl\teffective_url\tartifact_rel\tproject_artifact_rel\tbytes\tsha256\thttp_code\trequest_capsule_rel\tbrowser_trace_rel\tcookie_jar_rel\tstorage_state_rel\tprofile_bundle_rel\ttransport_substrate\trequested_backend\tcrosswalk_id\tcandidate_id\tchronology_row_id\tinventory_blocker_id\trow_ledger_refs\tgap_ids\tera_ids\tnomenclature_ids\tnote"
    }

    pub fn to_tsv_line(&self) -> String {
        [
            &self.at_utc,
            &self.action,
            &self.session_id,
            &self.parent_session_id,
            &self.source_id,
            &self.search_target_id,
            &self.status,
            &self.outcome,
            &self.url,
            &self.effective_url,
            &self.artifact_rel,
            &self.project_artifact_rel,
            &self.bytes,
            &self.sha256,
            &self.http_code,
            &self.request_capsule_rel,
            &self.browser_trace_rel,
            &self.cookie_jar_rel,
            &self.storage_state_rel,
            &self.profile_bundle_rel,
            &self.transport_substrate,
            &self.requested_backend,
            &self.crosswalk_id,
            &self.candidate_id,
            &self.chronology_row_id,
            &self.inventory_blocker_id,
            &self.row_ledger_refs,
            &self.gap_ids,
            &self.era_ids,
            &self.nomenclature_ids,
            &self.note,
        ]
        .iter()
        .map(|value| escape_tsv_field(value))
        .collect::<Vec<_>>()
        .join("\t")
    }

    pub fn from_tsv_line(line: &str) -> Result<Self> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 31 {
            bail!(
                "acquisition journal row expected 31 fields, found {}",
                fields.len()
            );
        }
        Ok(Self {
            at_utc: fields[0].to_string(),
            action: fields[1].to_string(),
            session_id: fields[2].to_string(),
            parent_session_id: fields[3].to_string(),
            source_id: fields[4].to_string(),
            search_target_id: fields[5].to_string(),
            status: fields[6].to_string(),
            outcome: fields[7].to_string(),
            url: fields[8].to_string(),
            effective_url: fields[9].to_string(),
            artifact_rel: fields[10].to_string(),
            project_artifact_rel: fields[11].to_string(),
            bytes: fields[12].to_string(),
            sha256: fields[13].to_string(),
            http_code: fields[14].to_string(),
            request_capsule_rel: fields[15].to_string(),
            browser_trace_rel: fields[16].to_string(),
            cookie_jar_rel: fields[17].to_string(),
            storage_state_rel: fields[18].to_string(),
            profile_bundle_rel: fields[19].to_string(),
            transport_substrate: fields[20].to_string(),
            requested_backend: fields[21].to_string(),
            crosswalk_id: fields[22].to_string(),
            candidate_id: fields[23].to_string(),
            chronology_row_id: fields[24].to_string(),
            inventory_blocker_id: fields[25].to_string(),
            row_ledger_refs: fields[26].to_string(),
            gap_ids: fields[27].to_string(),
            era_ids: fields[28].to_string(),
            nomenclature_ids: fields[29].to_string(),
            note: fields[30].to_string(),
        })
    }
}

pub fn load_project_api_context(input: &Path) -> Result<ProjectApiContext> {
    let (repo_root, project_api_dir, project_toml_path) = resolve_project_api_paths(input)?;
    let raw = fs::read_to_string(&project_toml_path)
        .with_context(|| format!("read {}", project_toml_path.display()))?;
    let contract: ProjectApiContract =
        toml::from_str(&raw).with_context(|| format!("parse {}", project_toml_path.display()))?;
    Ok(ProjectApiContext {
        inventory_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.inventory,
            "inventory",
            &project_toml_path,
        )?,
        eras_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.eras,
            "eras",
            &project_toml_path,
        )?,
        gaps_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.gaps,
            "gaps",
            &project_toml_path,
        )?,
        nomenclature_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.nomenclature,
            "nomenclature",
            &project_toml_path,
        )?,
        candidate_expansion_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.candidate_expansion,
            "candidate_expansion",
            &project_toml_path,
        )?,
        crosswalk_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.crosswalk,
            "crosswalk",
            &project_toml_path,
        )?,
        acquisition_journal_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.acquisition_journal,
            "acquisition_journal",
            &project_toml_path,
        )?,
        availability_summary_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.availability_summary,
            "availability_summary",
            &project_toml_path,
        )?,
        row_ledger_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.row_availability_ledger,
            "row_availability_ledger",
            &project_toml_path,
        )?,
        century_bucket_summary_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.century_bucket_summary,
            "century_bucket_summary",
            &project_toml_path,
        )?,
        search_queue_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.search_queue,
            "search_queue",
            &project_toml_path,
        )?,
        acquisition_stack_path: repo_path_required(
            &repo_root,
            &contract.entrypoints.acquisition_stack,
            "acquisition_stack",
            &project_toml_path,
        )?,
        repo_root,
        project_api_dir,
        project_toml_path,
    })
}

pub fn load_project_api_crosswalk(path: &Path) -> Result<ProjectApiCrosswalkFile> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

pub fn resolve_crosswalk_binding<'a>(
    crosswalk: &'a ProjectApiCrosswalkFile,
    search_target_id: Option<&str>,
    source_id: Option<&str>,
) -> Option<&'a ProjectApiCrosswalkBinding> {
    let search_target_id = search_target_id.filter(|value| !value.is_empty());
    let source_id = source_id.filter(|value| !value.is_empty());
    crosswalk.binding.iter().find(|binding| {
        search_target_id == Some(binding.search_target_id.as_str())
            || source_id == Some(binding.search_target_id.as_str())
            || source_id == Some(binding.id.as_str())
            || source_id == binding.candidate_id.as_deref()
            || source_id == binding.inventory_blocker_id.as_deref()
    })
}

pub fn append_acquisition_journal_rows(path: &Path, rows: &[AcquisitionJournalRow]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut body = if path.exists() && fs::metadata(path)?.len() > 0 {
        fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?
    } else {
        format!("{}\n", AcquisitionJournalRow::header())
    };
    if !body.ends_with('\n') {
        body.push('\n');
    }
    for row in rows {
        body.push_str(&row.to_tsv_line());
        body.push('\n');
    }
    fs::write(path, body).with_context(|| format!("write {}", path.display()))
}

pub fn load_acquisition_journal_rows(path: &Path) -> Result<Vec<AcquisitionJournalRow>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut lines = raw.lines();
    let Some(header) = lines.next() else {
        return Ok(Vec::new());
    };
    if header.trim().is_empty() {
        return Ok(Vec::new());
    }
    if header.trim() != AcquisitionJournalRow::header() {
        bail!(
            "unexpected acquisition journal header in {}",
            path.display()
        );
    }
    let mut rows = Vec::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        rows.push(AcquisitionJournalRow::from_tsv_line(line)?);
    }
    Ok(rows)
}

pub fn project_relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(normalize_path_string)
        .unwrap_or_default()
}

pub fn project_relative_path_from_display(
    local_repo_root: &Path,
    project_repo_root: &Path,
    display_path: &str,
) -> String {
    if display_path.is_empty() {
        return String::new();
    }
    let absolute = repo_path(local_repo_root, Path::new(display_path));
    project_relative_path(project_repo_root, &absolute)
}

pub fn journal_multi_value(values: &[String]) -> String {
    values.join(";")
}

pub fn split_journal_multi_value(value: &str) -> Vec<String> {
    value
        .split(';')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect()
}

pub fn repo_path(repo_root: &Path, input: &Path) -> PathBuf {
    if input.is_absolute() {
        input.to_path_buf()
    } else {
        repo_root.join(input)
    }
}

fn resolve_project_api_paths(input: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let direct_project_toml = input.join("project.toml");
    if direct_project_toml.exists()
        && input.file_name().and_then(|name| name.to_str()) == Some("project_api")
    {
        let metadata_dir = input
            .parent()
            .context("project_api directory must have metadata parent")?;
        let repo_root = metadata_dir
            .parent()
            .context("metadata directory must have repo root parent")?;
        return Ok((
            repo_root.to_path_buf(),
            input.to_path_buf(),
            direct_project_toml,
        ));
    }

    let repo_root = input.to_path_buf();
    let project_api_dir = repo_root.join("metadata/project_api");
    let project_toml_path = project_api_dir.join("project.toml");
    if !project_toml_path.exists() {
        bail!(
            "project API contract not found at {}",
            project_toml_path.display()
        );
    }
    Ok((repo_root, project_api_dir, project_toml_path))
}

fn repo_path_required(
    repo_root: &Path,
    raw: &str,
    field: &str,
    project_toml_path: &Path,
) -> Result<PathBuf> {
    if raw.trim().is_empty() {
        bail!(
            "project API entrypoint '{}' missing in {}",
            field,
            project_toml_path.display()
        );
    }
    Ok(repo_path(repo_root, Path::new(raw)))
}

fn normalize_path_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn escape_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_journal_row() -> AcquisitionJournalRow {
        AcquisitionJournalRow {
            at_utc: "2026-03-26T12:00:00Z".to_string(),
            action: "record".to_string(),
            session_id: "session-a".to_string(),
            parent_session_id: "session-root".to_string(),
            source_id: "source-a".to_string(),
            search_target_id: "target-a".to_string(),
            status: "downloaded".to_string(),
            outcome: "recorded_downloaded".to_string(),
            url: "https://example.test/a.pdf".to_string(),
            effective_url: "https://cdn.example.test/a.pdf".to_string(),
            artifact_rel: "reports/a.pdf".to_string(),
            project_artifact_rel: "tier/a.pdf".to_string(),
            bytes: "123".to_string(),
            sha256: "abc".to_string(),
            http_code: "200".to_string(),
            request_capsule_rel: "reports/a.json".to_string(),
            browser_trace_rel: "reports/a.har".to_string(),
            cookie_jar_rel: "reports/a.cookies".to_string(),
            storage_state_rel: "reports/a.state.json".to_string(),
            profile_bundle_rel: "reports/a.bundle.toml".to_string(),
            transport_substrate: "download_stack".to_string(),
            requested_backend: "reqwest".to_string(),
            crosswalk_id: "binding-a".to_string(),
            candidate_id: "candidate-a".to_string(),
            chronology_row_id: "C1".to_string(),
            inventory_blocker_id: "gap-a".to_string(),
            row_ledger_refs: "candidate:candidate-a".to_string(),
            gap_ids: "gap-a".to_string(),
            era_ids: "century_1900_1999".to_string(),
            nomenclature_ids: "tower_name_normalization".to_string(),
            note: "ok".to_string(),
        }
    }

    #[test]
    fn acquisition_journal_round_trip_preserves_fields() {
        let row = sample_journal_row();
        let parsed = AcquisitionJournalRow::from_tsv_line(&row.to_tsv_line()).unwrap();
        assert_eq!(parsed.session_id, row.session_id);
        assert_eq!(parsed.project_artifact_rel, row.project_artifact_rel);
        assert_eq!(parsed.row_ledger_refs, row.row_ledger_refs);
    }

    #[test]
    fn tsv_escaping_strips_tabs_and_newlines() {
        let mut row = sample_journal_row();
        row.note = "has\ttab\nand\rnewline".to_string();
        let line = row.to_tsv_line();
        assert!(!line.contains('\t') || line.matches('\t').count() == 30);
        let parsed = AcquisitionJournalRow::from_tsv_line(&line).unwrap();
        assert_eq!(parsed.note, "has tab and newline");
    }

    #[test]
    fn from_tsv_line_rejects_wrong_field_count() {
        let bad = "a\tb\tc";
        let result = AcquisitionJournalRow::from_tsv_line(bad);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("31"), "error should mention expected 31 fields: {msg}");
    }

    #[test]
    fn header_has_31_columns() {
        let header = AcquisitionJournalRow::header();
        let count = header.split('\t').count();
        assert_eq!(count, 31, "header must have exactly 31 tab-separated columns");
    }

    #[test]
    fn contract_toml_round_trip() {
        let contract = ProjectApiContract {
            schema_version: Some(2),
            project_id: Some("test-project".to_string()),
            entrypoints: ProjectApiEntrypoints {
                inventory: "data/inventory.toml".to_string(),
                eras: "data/eras.toml".to_string(),
                gaps: "data/gaps.toml".to_string(),
                nomenclature: "data/nomenclature.toml".to_string(),
                candidate_expansion: "data/candidates.toml".to_string(),
                crosswalk: "data/crosswalk.toml".to_string(),
                acquisition_journal: "data/journal.tsv".to_string(),
                availability_summary: "data/availability.toml".to_string(),
                row_availability_ledger: "data/row_ledger.toml".to_string(),
                century_bucket_summary: "data/centuries.toml".to_string(),
                search_queue: "data/search_queue.toml".to_string(),
                acquisition_stack: "data/stack.toml".to_string(),
            },
        };
        let serialized = toml::to_string_pretty(&contract).expect("serialize");
        let parsed: ProjectApiContract = toml::from_str(&serialized).expect("deserialize");
        assert_eq!(parsed.schema_version, Some(2));
        assert_eq!(parsed.project_id.as_deref(), Some("test-project"));
        assert_eq!(parsed.entrypoints.inventory, "data/inventory.toml");
        assert_eq!(parsed.entrypoints.acquisition_stack, "data/stack.toml");
    }

    #[test]
    fn crosswalk_resolution_by_search_target_id() {
        let crosswalk = ProjectApiCrosswalkFile {
            binding: vec![ProjectApiCrosswalkBinding {
                id: "binding-1".to_string(),
                search_target_id: "tgt-alpha".to_string(),
                ..ProjectApiCrosswalkBinding::default()
            }],
            ..ProjectApiCrosswalkFile::default()
        };
        let found = resolve_crosswalk_binding(&crosswalk, Some("tgt-alpha"), None);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "binding-1");
    }

    #[test]
    fn crosswalk_resolution_by_source_id_fallback() {
        let crosswalk = ProjectApiCrosswalkFile {
            binding: vec![ProjectApiCrosswalkBinding {
                id: "binding-2".to_string(),
                search_target_id: "tgt-beta".to_string(),
                candidate_id: Some("candidate-x".to_string()),
                ..ProjectApiCrosswalkBinding::default()
            }],
            ..ProjectApiCrosswalkFile::default()
        };
        let found = resolve_crosswalk_binding(&crosswalk, None, Some("candidate-x"));
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "binding-2");
    }

    #[test]
    fn crosswalk_resolution_returns_none_on_miss() {
        let crosswalk = ProjectApiCrosswalkFile::default();
        assert!(resolve_crosswalk_binding(&crosswalk, Some("missing"), None).is_none());
    }

    #[test]
    fn journal_multi_value_round_trip() {
        let values = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let encoded = journal_multi_value(&values);
        assert_eq!(encoded, "a;b;c");
        let decoded = split_journal_multi_value(&encoded);
        assert_eq!(decoded, values);
    }

    #[test]
    fn split_journal_multi_value_handles_empty_segments() {
        let decoded = split_journal_multi_value("a;;b; ;c");
        assert_eq!(decoded, vec!["a", "b", "c"]);
    }

    #[test]
    fn repo_path_absolute_passthrough() {
        let root = Path::new("/repo");
        let abs = Path::new("/other/file.txt");
        assert_eq!(repo_path(root, abs), PathBuf::from("/other/file.txt"));
    }

    #[test]
    fn repo_path_relative_joined() {
        let root = Path::new("/repo");
        let rel = Path::new("data/file.txt");
        assert_eq!(repo_path(root, rel), PathBuf::from("/repo/data/file.txt"));
    }

    #[test]
    fn journal_file_round_trip_via_tempfile() {
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        let path = tmp.path();

        let rows = vec![sample_journal_row()];
        append_acquisition_journal_rows(path, &rows).expect("append");

        let loaded = load_acquisition_journal_rows(path).expect("load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].session_id, "session-a");

        // Appending again should yield 2 rows total.
        append_acquisition_journal_rows(path, &rows).expect("append again");
        let loaded2 = load_acquisition_journal_rows(path).expect("load2");
        assert_eq!(loaded2.len(), 2);
    }
}
