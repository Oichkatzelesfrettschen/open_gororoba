use anyhow::{Context, Result, anyhow, bail};
use camino::Utf8PathBuf;
use chrono::Utc;
use provenance_core::{
    ArtifactQueryResult, ArtifactRecord, ArtifactStatus, BinaryRecord, ClaimRecord,
    ControlPlaneCounts, DoctorReport, DocumentQueryResult, DocumentRecord, DownloadAttemptRecord,
    DownloadCampaignQueryResult, DownloadCampaignRecord, DownloadJobRecord,
    DownloadLedgerProjectionRow, DownloadQueryResult, ExperimentRecord,
    ExternalSourceContractRecord, ExternalSourceContractsMeta, ExternalSourceDossierRecord,
    ExternalSourceDossiersMeta, IndexStats, InsightRecord,
    LiteratureNoveltySimilarPaperRecord, LiteratureVerificationQueryResult,
    LiteratureVerificationResultRecord, LiteratureVerificationRunRecord, MirrorKind,
    MirrorObservationRecord, PantheonSeedSummary, TheoremRecord,
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use toml::Value;

// Canonical claim/insight status taxonomies, theorem-ID allowlist,
// control-plane TOML/SQLite paths, and the rusqlite migration registry
// live in the `migrations` submodule. Items are pub(crate) and brought
// back into lib.rs scope via plain use statements.
mod migrations;
use migrations::{
    CANONICAL_CLAIM_STATUSES, CANONICAL_INSIGHT_STATUSES, JUSTIFIED_UNLINKED_THEOREM_IDS,
    migrations,
};

pub struct ProvenanceStore {
    conn: Connection,
}

#[derive(Clone, Copy)]
pub struct ExternalSourceContractPatch<'a> {
    pub path_glob: Option<&'a str>,
    pub canonical_url: Option<&'a str>,
    pub mirror_urls: Option<&'a [String]>,
    pub access_class: Option<&'a str>,
    pub status: Option<&'a str>,
    pub retrieval_method: Option<&'a str>,
    pub attempt_deadline_utc: Option<&'a str>,
    pub resolution_deadline_utc: Option<&'a str>,
    pub blocker_note: Option<&'a str>,
    pub evidence_refs: Option<&'a [String]>,
    pub manual_manifest_refs: Option<&'a [String]>,
    pub blocked_action_plan: Option<&'a [String]>,
    pub scientific_validator_refs: Option<&'a [String]>,
}

struct ControlPlaneCompatOutputs {
    claims: String,
    insights: String,
    experiments: String,
    binaries: String,
    theorems: String,
    theorems_mirror: String,
}

struct ExternalSourcesCompatOutputs {
    source_contracts: String,
    dossiers_registry: String,
    docs: Vec<(Utf8PathBuf, String)>,
}

#[derive(Clone, Debug)]
struct ProofInventoryEntry {
    stem: String,
    path: Utf8PathBuf,
}

#[derive(Clone, Debug, Default)]
struct ProofInventory {
    project_raw: String,
    verified_entries: Vec<ProofInventoryEntry>,
    verified_by_claim_id: BTreeMap<String, Vec<ProofInventoryEntry>>,
}

// Public row, table, and revision types for the ProvenanceStore control
// plane (17 types: ControlPlaneCompatKind, PlanningCompatTable,
// RoadmapItem, ActionItem, RequirementsMeta, RequirementModuleItem,
// RequirementCoverageGapItem, RoadmapCompatRow, ActionCompatRow,
// RequirementsMetaCompatRow, RequirementModuleCompatRow,
// RequirementCoverageGapCompatRow, ResearchNarrativeRow,
// NotebookSessionRow, ManifestRow, NotebookSessionSummary,
// CompatExportPaths, EntityFieldTarget, StatusNoteRevision) live in
// the `types` submodule.
pub mod types;
pub use types::{
    ActionCompatRow, ActionItem, CompatExportPaths, ControlPlaneCompatKind, EntityFieldTarget,
    ManifestRow, NotebookSessionRow, NotebookSessionSummary, PlanningCompatTable,
    RequirementCoverageGapCompatRow, RequirementCoverageGapItem, RequirementModuleCompatRow,
    RequirementModuleItem, RequirementsMeta, RequirementsMetaCompatRow, ResearchNarrativeRow,
    RoadmapCompatRow, RoadmapItem, StatusNoteRevision,
};

/// Hex-encoded SHA-256 of `s`. Used by the status_note mutators to
/// populate the prev/new_value_sha256 columns in *_revisions tables.
fn sha256_hex(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    let bytes = hasher.finalize();
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes.iter() {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

/// Sentinel inserted into *_revisions.application_id so future triggers
/// can distinguish CLI-driven mutations from raw SQL pokes. The hex
/// digits "go ro" (`0x676f_726f`) -- a fingerprint of the gororoba CLI.
const CLI_APPLICATION_ID: i64 = 0x676f_726f;

impl ProvenanceStore {
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create db directory {}", parent.display()))?;
        }
        let mut conn = Connection::open(db_path)
            .with_context(|| format!("open sqlite database {}", db_path.display()))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        migrations().to_latest(&mut conn)?;
        Ok(Self { conn })
    }

    /// Execute a parameterized SQL statement with typed params.
    pub fn conn_exec<I, T>(&self, sql: &str, params: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        T: rusqlite::types::ToSql,
    {
        let p = rusqlite::params_from_iter(params);
        self.conn
            .execute(sql, p)
            .with_context(|| "execute SQL statement")?;
        Ok(())
    }

    pub fn reindex_from_registries(
        &mut self,
        repo_root: &Path,
        artifact_registry_path: &Path,
        knowledge_sources_path: &Path,
        lane_dir: &Path,
    ) -> Result<IndexStats> {
        let artifacts = load_artifacts(artifact_registry_path)?;
        let artifact_id_set = artifacts
            .iter()
            .map(|artifact| artifact.id.clone())
            .collect::<std::collections::HashSet<_>>();
        let documents = load_documents(knowledge_sources_path)?;
        let lanes = load_lane_assignments(lane_dir)?;
        let mirror_observations = build_mirror_observations(artifact_registry_path)?;
        let indexed_at = Utc::now().to_rfc3339();

        let tx = self.conn.transaction()?;
        clear_tables(&tx)?;

        for path in [artifact_registry_path, knowledge_sources_path] {
            write_fingerprint(&tx, repo_root, path, &indexed_at)?;
        }
        for lane in [
            "datasets.toml",
            "papers_pdf.toml",
            "slides_artifacts.toml",
            "web_references.toml",
        ] {
            let path = lane_dir.join(lane);
            if path.exists() {
                write_fingerprint(&tx, repo_root, &path, &indexed_at)?;
            }
        }

        for document in &documents {
            tx.execute(
                "INSERT INTO documents (
                    id, path, title, kind, authoring_mode, generated, status,
                    toml_backing, sha256, size_bytes, line_count
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    document.id,
                    document.path.as_str(),
                    document.title,
                    document.kind,
                    document.authoring_mode,
                    i64::from(document.generated),
                    document.status,
                    document.toml_backing.as_ref().map(|v| v.as_str()),
                    document.sha256.as_deref(),
                    document.size_bytes,
                    document.line_count
                ],
            )?;
            if let Some(backing) = &document.toml_backing {
                tx.execute(
                    "INSERT INTO record_sources (entity_kind, entity_id, source_ref) VALUES ('document', ?1, ?2)",
                    params![document.id, backing.as_str()],
                )?;
            }
            tx.execute(
                "INSERT INTO document_search (document_id, path, title, kind) VALUES (?1, ?2, ?3, ?4)",
                params![document.id, document.path.as_str(), document.title, document.kind],
            )?;
        }

        for artifact in &artifacts {
            tx.execute(
                "INSERT INTO artifacts (
                    id, key, title, citation, status,
                    minimum_requirement_met, canonical_functional_url, canonical_download_path
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    artifact.id,
                    artifact.key,
                    artifact.title,
                    artifact.citation,
                    artifact.status.as_str(),
                    i64::from(artifact.minimum_requirement_met),
                    artifact.canonical_functional_url.as_deref(),
                    artifact
                        .canonical_download_path
                        .as_ref()
                        .map(|v| v.as_str())
                ],
            )?;
            for source_ref in &artifact.source_refs {
                tx.execute(
                    "INSERT INTO record_sources (entity_kind, entity_id, source_ref) VALUES ('artifact', ?1, ?2)
                     ON CONFLICT(entity_kind, entity_id, source_ref) DO NOTHING",
                    params![artifact.id, source_ref],
                )?;
            }
            for doi in &artifact.doi_list {
                tx.execute(
                    "INSERT INTO citations (artifact_id, citation_text, doi, canonical_url) VALUES (?1, ?2, ?3, ?4)",
                    params![
                        artifact.id,
                        artifact.citation,
                        doi,
                        artifact.canonical_functional_url.as_deref()
                    ],
                )?;
            }
            for url in &artifact.all_links {
                let host = host_for_url(url);
                tx.execute(
                    "INSERT INTO links (url, host) VALUES (?1, ?2)
                     ON CONFLICT(url) DO UPDATE SET host=excluded.host",
                    params![url, host],
                )?;
                tx.execute(
                    "INSERT INTO artifact_links (artifact_id, url, relation) VALUES (?1, ?2, 'all_links')
                     ON CONFLICT(artifact_id, url, relation) DO NOTHING",
                    params![artifact.id, url],
                )?;
            }
            for path in &artifact.downloaded_paths {
                tx.execute(
                    "INSERT INTO artifact_paths (artifact_id, path, relation) VALUES (?1, ?2, 'downloaded')
                     ON CONFLICT(artifact_id, path, relation) DO NOTHING",
                    params![artifact.id, path.as_str()],
                )?;
            }
        }

        for observation in &mirror_observations {
            tx.execute(
                "INSERT OR IGNORE INTO links (url, host) VALUES (?1, ?2)",
                params![observation.url, host_for_url(&observation.url)],
            )?;
            tx.execute(
                "INSERT INTO mirror_observations (artifact_id, url, mirror_kind) VALUES (?1, ?2, ?3)
                 ON CONFLICT(artifact_id, url, mirror_kind) DO NOTHING",
                params![
                    observation.artifact_id,
                    observation.url,
                    observation.mirror_kind.as_str()
                ],
            )?;
        }

        let mut inserted_lane_count = 0usize;
        for lane in &lanes {
            if !artifact_id_set.contains(&lane.artifact_id) {
                continue;
            }
            tx.execute(
                "INSERT INTO lane_assignments (artifact_id, lane_name) VALUES (?1, ?2)
                 ON CONFLICT(artifact_id, lane_name) DO NOTHING",
                params![lane.artifact_id, lane.lane_name],
            )?;
            inserted_lane_count += 1;
        }

        let details = serde_json::json!({
            "artifact_registry": to_repo_rel(repo_root, artifact_registry_path),
            "knowledge_sources": to_repo_rel(repo_root, knowledge_sources_path),
            "lane_dir": to_repo_rel(repo_root, lane_dir),
        });
        tx.execute(
            "INSERT INTO export_runs (action, created_at, artifact_count, document_count, details_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                "index",
                indexed_at,
                artifacts.len() as i64,
                documents.len() as i64,
                details.to_string()
            ],
        )?;

        tx.commit()?;

        Ok(IndexStats {
            indexed_at,
            artifact_count: artifacts.len(),
            document_count: documents.len(),
            lane_assignment_count: inserted_lane_count,
            mirror_observation_count: mirror_observations.len(),
            claim_count: 0,
            insight_count: 0,
            experiment_count: 0,
            binary_count: 0,
            theorem_count: 0,
        })
    }

    pub fn reindex_control_plane_from_registries(
        &mut self,
        repo_root: &Path,
        claims_path: &Path,
        insights_path: &Path,
        experiments_path: &Path,
        binaries_path: &Path,
        proofs_project_path: &Path,
    ) -> Result<IndexStats> {
        let indexed_at = Utc::now().to_rfc3339();
        let claims_text = load_toml_text(claims_path)?;
        let insights_text = load_toml_text(insights_path)?;
        let experiments_text = load_toml_text(experiments_path)?;
        let binaries_text = load_toml_text(binaries_path)?;
        let proof_inventory = load_proof_inventory(proofs_project_path)?;
        let mut claims = load_claims_from_registry(&claims_text)?;
        normalize_claims_against_proof_inventory(repo_root, &mut claims, &proof_inventory)?;
        let insights = load_insights_from_registry(&insights_text)?;
        let experiments = load_experiments_from_registry(&experiments_text)?;
        let experiments_meta_toml =
            load_registry_table_toml(&experiments_text, "experiments")?.unwrap_or_default();
        let registry_binaries = load_binaries_from_registry(&binaries_text)?;
        let binaries = merge_workspace_binaries(repo_root, &registry_binaries)?;
        let theorems = load_theorems_from_inventory(repo_root, &proof_inventory, &claims)?;

        let tx = self.conn.transaction()?;
        clear_control_plane_tables(&tx)?;
        for (kind, path, body) in [
            ("claims", claims_path, claims_text.as_str()),
            ("insights", insights_path, insights_text.as_str()),
            ("experiments", experiments_path, experiments_text.as_str()),
            ("binaries", binaries_path, binaries_text.as_str()),
        ] {
            write_registry_snapshot(&tx, repo_root, kind, path, body, &indexed_at)?;
        }
        write_registry_snapshot(
            &tx,
            repo_root,
            "rocq_project",
            proofs_project_path,
            &proof_inventory.project_raw,
            &indexed_at,
        )?;

        for claim in &claims {
            tx.execute(
                "INSERT INTO claims(id, statement, status, where_stated, last_verified, formal_proof, status_note, compat_toml_text)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(id) DO UPDATE SET
                    statement=excluded.statement,
                    status=excluded.status,
                    where_stated=excluded.where_stated,
                    last_verified=excluded.last_verified,
                    formal_proof=excluded.formal_proof,
                    status_note=excluded.status_note,
                    compat_toml_text=excluded.compat_toml_text",
                params![
                    claim.id,
                    claim.statement,
                    claim.status,
                    claim.where_stated,
                    claim.last_verified,
                    claim.formal_proof,
                    claim.status_note,
                    claim.compat_toml_text
                ],
            )?;
        }
        for insight in &insights {
            tx.execute(
                "INSERT INTO insights(id, title, status, claim_refs_json, compat_toml_text)
                 VALUES(?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    claim_refs_json=excluded.claim_refs_json,
                    compat_toml_text=excluded.compat_toml_text",
                params![
                    insight.id,
                    insight.title,
                    insight.status,
                    serde_json::to_string(&insight.claim_refs)?,
                    insight.compat_toml_text
                ],
            )?;
        }
        for experiment in &experiments {
            tx.execute(
                "INSERT INTO experiments_cp(id, title, status, binary_name, claim_refs_json, compat_toml_text)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    binary_name=excluded.binary_name,
                    claim_refs_json=excluded.claim_refs_json,
                    compat_toml_text=excluded.compat_toml_text",
                params![
                    experiment.id,
                    experiment.title,
                    experiment.status,
                    experiment.binary,
                    serde_json::to_string(&experiment.claim_refs)?,
                    experiment.compat_toml_text
                ],
            )?;
        }
        tx.execute(
            "INSERT INTO control_plane_meta(kind, compat_toml_text)
             VALUES(?1, ?2)
             ON CONFLICT(kind) DO UPDATE SET compat_toml_text=excluded.compat_toml_text",
            params!["experiments", experiments_meta_toml],
        )?;
        for binary in &binaries {
            tx.execute(
                "INSERT INTO binaries_cp(name, crate_name, description, experiment_id, source)
                 VALUES(?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(name) DO UPDATE SET
                    crate_name=excluded.crate_name,
                    description=excluded.description,
                    experiment_id=excluded.experiment_id,
                    source=excluded.source",
                params![
                    binary.name,
                    binary.crate_name,
                    binary.description,
                    binary.experiment,
                    binary.source
                ],
            )?;
        }
        for theorem in &theorems {
            tx.execute(
                "INSERT INTO theorems(id, title, proof_path, status, linked_claim_ids_json, source)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    proof_path=excluded.proof_path,
                    status=excluded.status,
                    linked_claim_ids_json=excluded.linked_claim_ids_json,
                    source=excluded.source",
                params![
                    theorem.id,
                    theorem.title,
                    theorem.proof_path.as_str(),
                    theorem.status,
                    serde_json::to_string(&theorem.linked_claim_ids)?,
                    theorem.source
                ],
            )?;
        }
        tx.execute(
            "INSERT INTO control_plane_runs(action, created_at, details_json)
             VALUES(?1, ?2, ?3)",
            params![
                "index_control_plane",
                indexed_at,
                serde_json::json!({
                    "claims": to_repo_rel(repo_root, claims_path),
                    "insights": to_repo_rel(repo_root, insights_path),
                    "experiments": to_repo_rel(repo_root, experiments_path),
                    "binaries": to_repo_rel(repo_root, binaries_path),
                    "rocq_project": to_repo_rel(repo_root, proofs_project_path),
                })
                .to_string()
            ],
        )?;
        tx.commit()?;

        Ok(IndexStats {
            indexed_at,
            artifact_count: 0,
            document_count: 0,
            lane_assignment_count: 0,
            mirror_observation_count: 0,
            claim_count: claims.len(),
            insight_count: insights.len(),
            experiment_count: experiments.len(),
            binary_count: binaries.len(),
            theorem_count: theorems.len(),
        })
    }

    pub fn control_plane_counts(&self) -> Result<ControlPlaneCounts> {
        let claim_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM claims")?;
        let insight_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM insights")?;
        let experiment_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM experiments_cp")?;
        let complete_experiment_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM experiments_cp WHERE lower(status) IN ('completed','complete','done')",
        )?;
        let binary_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM binaries_cp")?;
        let theorem_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM theorems")?;
        let theorem_linked_claims = self
            .list_theorems()?
            .into_iter()
            .flat_map(|theorem| theorem.linked_claim_ids.into_iter())
            .collect::<BTreeSet<_>>();
        let kernel_checked_claim_count = self
            .list_claims()?
            .into_iter()
            .filter(|claim| {
                claim
                    .formal_proof
                    .as_deref()
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false)
                    || theorem_linked_claims.contains(&claim.id)
            })
            .count();
        let proof_file_count = self
            .registry_snapshot("rocq_project")?
            .map(|raw| {
                raw.lines()
                    .filter(|line| {
                        let trimmed = line.trim();
                        trimmed.ends_with(".v")
                            && (trimmed.starts_with("verified/")
                                || trimmed.starts_with("theories/"))
                    })
                    .count()
            })
            .unwrap_or(theorem_count);
        Ok(ControlPlaneCounts {
            claim_count,
            insight_count,
            experiment_count,
            complete_experiment_count,
            binary_count,
            theorem_count,
            kernel_checked_claim_count,
            proof_file_count,
        })
    }

    pub fn export_control_plane_compat(
        &mut self,
        repo_root: &Path,
        paths: CompatExportPaths<'_>,
    ) -> Result<()> {
        self.backfill_control_plane_compat_from_snapshots()?;
        let outputs = self.render_control_plane_compat_outputs()?;
        write_text(paths.claims, &outputs.claims)?;
        write_text(paths.insights, &outputs.insights)?;
        write_text(paths.experiments, &outputs.experiments)?;
        write_text(paths.binaries, &outputs.binaries)?;
        write_text(paths.theorems, &outputs.theorems)?;
        write_text(paths.theorems_mirror, &outputs.theorems_mirror)?;

        self.record_control_plane_run(
            "export_control_plane",
            &serde_json::json!({
                "claims": to_repo_rel(repo_root, paths.claims),
                "insights": to_repo_rel(repo_root, paths.insights),
                "experiments": to_repo_rel(repo_root, paths.experiments),
                "binaries": to_repo_rel(repo_root, paths.binaries),
                "theorems": to_repo_rel(repo_root, paths.theorems),
                "theorems_mirror": to_repo_rel(repo_root, paths.theorems_mirror),
            })
            .to_string(),
        )?;
        Ok(())
    }

    pub fn replace_control_plane_experiments_from_registry_text(
        &mut self,
        repo_root: &Path,
        source_path: &Path,
        raw: &str,
    ) -> Result<usize> {
        let indexed_at = Utc::now().to_rfc3339();
        let experiments = load_experiments_from_registry(raw)?;
        let experiments_meta_toml =
            load_registry_table_toml(raw, "experiments")?.unwrap_or_default();
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM experiments_cp", [])?;
        tx.execute(
            "DELETE FROM control_plane_meta WHERE kind = 'experiments'",
            [],
        )?;
        write_registry_snapshot(&tx, repo_root, "experiments", source_path, raw, &indexed_at)?;
        for experiment in &experiments {
            tx.execute(
                "INSERT INTO experiments_cp(id, title, status, binary_name, claim_refs_json, compat_toml_text)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    experiment.id,
                    experiment.title,
                    experiment.status,
                    experiment.binary,
                    serde_json::to_string(&experiment.claim_refs)?,
                    experiment.compat_toml_text
                ],
            )?;
        }
        tx.execute(
            "INSERT INTO control_plane_meta(kind, compat_toml_text)
             VALUES(?1, ?2)",
            params!["experiments", experiments_meta_toml],
        )?;
        tx.commit()?;
        self.record_control_plane_run(
            "replace_control_plane_experiments_from_registry_text",
            &serde_json::json!({
                "source_path": to_repo_rel(repo_root, source_path),
                "experiment_count": experiments.len(),
            })
            .to_string(),
        )?;
        Ok(experiments.len())
    }

    pub fn control_plane_compat_text(&mut self, kind: ControlPlaneCompatKind) -> Result<String> {
        self.backfill_control_plane_compat_from_snapshots()?;
        let outputs = self.render_control_plane_compat_outputs()?;
        let text = match kind {
            ControlPlaneCompatKind::Claims => outputs.claims,
            ControlPlaneCompatKind::Insights => outputs.insights,
            ControlPlaneCompatKind::Experiments => outputs.experiments,
            ControlPlaneCompatKind::Binaries => outputs.binaries,
            ControlPlaneCompatKind::Theorems => outputs.theorems,
            ControlPlaneCompatKind::TheoremsMirror => outputs.theorems_mirror,
        };
        Ok(text)
    }

    pub fn verify_control_plane_compat_exports(
        &mut self,
        repo_root: &Path,
        paths: CompatExportPaths<'_>,
    ) -> Result<()> {
        self.backfill_control_plane_compat_from_snapshots()?;
        let outputs = self.render_control_plane_compat_outputs()?;
        let checks = [
            (paths.claims, outputs.claims.as_str()),
            (paths.insights, outputs.insights.as_str()),
            (paths.experiments, outputs.experiments.as_str()),
            (paths.binaries, outputs.binaries.as_str()),
            (paths.theorems, outputs.theorems.as_str()),
            (paths.theorems_mirror, outputs.theorems_mirror.as_str()),
        ];
        let mut failures = Vec::new();
        for (path, expected) in checks {
            if !path.exists() {
                failures.push(format!("missing compatibility export {}", path.display()));
                continue;
            }
            let actual = load_text(path)?;
            if actual != format!("{expected}\n") {
                failures.push(format!(
                    "stale compatibility export {} relative to {}",
                    path.display(),
                    repo_root.display()
                ));
            }
        }
        if !failures.is_empty() {
            bail!(
                "control-plane compatibility exports failed:\n- {}",
                failures.join("\n- ")
            );
        }
        Ok(())
    }

    pub fn verify_control_plane_invariants(&self, repo_root: &Path) -> Result<()> {
        let mut failures = Vec::new();
        let counts = self.control_plane_counts()?;
        if counts.claim_count == 0 {
            failures.push("control-plane database has zero claims".to_string());
        }
        if counts.theorem_count == 0 {
            failures.push("control-plane database has zero theorems".to_string());
        }

        for claim in self.list_claims()? {
            if claim.status.trim().is_empty() {
                failures.push(format!("{} has empty status", claim.id));
            }
            if let Some(proof_path) = claim.formal_proof.as_deref()
                && !proof_path.trim().is_empty()
                && !repo_root.join(proof_path).exists()
            {
                failures.push(format!(
                    "{} formal_proof path missing on disk: {}",
                    claim.id, proof_path
                ));
            }
        }

        for theorem in self.list_theorems()? {
            if !repo_root.join(theorem.proof_path.as_str()).exists() {
                failures.push(format!(
                    "{} proof path missing on disk: {}",
                    theorem.id, theorem.proof_path
                ));
            }
            if theorem.linked_claim_ids.is_empty()
                && !JUSTIFIED_UNLINKED_THEOREM_IDS.contains(&theorem.id.as_str())
            {
                failures.push(format!(
                    "{} has no linked claims and is not in the justified unlinked theorem allowlist",
                    theorem.id
                ));
            }
            for claim_id in theorem.linked_claim_ids {
                let claim_exists = self.conn.query_row(
                    "SELECT COUNT(*) FROM claims WHERE id = ?1",
                    params![claim_id],
                    |row| row.get::<_, i64>(0),
                )? > 0;
                if !claim_exists {
                    failures.push(format!("{} links missing claim {}", theorem.id, claim_id));
                }
            }
        }

        if !failures.is_empty() {
            bail!(
                "control-plane invariants failed:\n- {}",
                failures.join("\n- ")
            );
        }
        Ok(())
    }

    pub fn reindex_external_sources_from_compat(
        &mut self,
        repo_root: &Path,
        source_contracts_path: &Path,
        dossiers_registry_path: &Path,
    ) -> Result<(usize, usize)> {
        let indexed_at = Utc::now().to_rfc3339();
        let source_contracts_text = load_toml_text(source_contracts_path)?;
        let dossiers_text = load_toml_text(dossiers_registry_path)?;
        let (contracts_meta, contracts) =
            load_external_source_contracts_from_registry(&source_contracts_text)?;
        let (dossiers_meta, dossiers) =
            load_external_source_dossiers_from_registry(&dossiers_text)?;

        let tx = self.conn.transaction()?;
        clear_external_source_tables(&tx)?;
        write_registry_snapshot(
            &tx,
            repo_root,
            "external_source_contracts",
            source_contracts_path,
            &source_contracts_text,
            &indexed_at,
        )?;
        write_registry_snapshot(
            &tx,
            repo_root,
            "external_source_dossiers",
            dossiers_registry_path,
            &dossiers_text,
            &indexed_at,
        )?;
        tx.execute(
            "INSERT INTO external_source_contracts_meta(kind, updated, authoritative, policy_version)
             VALUES('source_contracts', ?1, ?2, ?3)
             ON CONFLICT(kind) DO UPDATE SET
                updated=excluded.updated,
                authoritative=excluded.authoritative,
                policy_version=excluded.policy_version",
            params![
                contracts_meta.updated,
                i64::from(contracts_meta.authoritative),
                contracts_meta.policy_version
            ],
        )?;
        for contract in &contracts {
            tx.execute(
                "INSERT INTO external_source_contracts(
                    id, path_glob, canonical_url, access_class, status, retrieval_method,
                    attempt_deadline_utc, resolution_deadline_utc, blocker_note
                ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    contract.id,
                    contract.path_glob,
                    contract.canonical_url,
                    contract.access_class,
                    contract.status,
                    contract.retrieval_method,
                    contract.attempt_deadline_utc,
                    contract.resolution_deadline_utc,
                    contract.blocker_note,
                ],
            )?;
            insert_ranked_values(
                &tx,
                "external_source_contract_values",
                "contract_id",
                &contract.id,
                "mirror_url",
                &contract.mirror_urls,
            )?;
            insert_ranked_values(
                &tx,
                "external_source_contract_values",
                "contract_id",
                &contract.id,
                "evidence_ref",
                &contract.evidence_refs,
            )?;
            insert_ranked_values(
                &tx,
                "external_source_contract_values",
                "contract_id",
                &contract.id,
                "manual_manifest_ref",
                &contract.manual_manifest_refs,
            )?;
            insert_ranked_values(
                &tx,
                "external_source_contract_values",
                "contract_id",
                &contract.id,
                "blocked_action_plan",
                &contract.blocked_action_plan,
            )?;
            insert_ranked_values(
                &tx,
                "external_source_contract_values",
                "contract_id",
                &contract.id,
                "scientific_validator_ref",
                &contract.scientific_validator_refs,
            )?;
        }

        tx.execute(
            "INSERT INTO external_source_dossiers_meta(kind, updated, authoritative, source_markdown_glob, document_count)
             VALUES('source_dossiers', ?1, ?2, ?3, ?4)
             ON CONFLICT(kind) DO UPDATE SET
                updated=excluded.updated,
                authoritative=excluded.authoritative,
                source_markdown_glob=excluded.source_markdown_glob,
                document_count=excluded.document_count",
            params![
                dossiers_meta.updated,
                i64::from(dossiers_meta.authoritative),
                dossiers_meta.source_markdown_glob,
                dossiers_meta.document_count as i64
            ],
        )?;
        for dossier in &dossiers {
            tx.execute(
                "INSERT INTO external_source_dossiers(
                    id, source_markdown, slug, title, status_token, content_kind,
                    authority_level, verification_level, operational_role,
                    source_lineage_summary, has_full_transcript, line_count, notes, body_markdown
                ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                params![
                    dossier.id,
                    dossier.source_markdown,
                    dossier.slug,
                    dossier.title,
                    dossier.status_token,
                    dossier.content_kind,
                    dossier.authority_level,
                    dossier.verification_level,
                    dossier.operational_role,
                    dossier.source_lineage_summary,
                    i64::from(dossier.has_full_transcript),
                    dossier.line_count as i64,
                    dossier.notes,
                    dossier.body_markdown
                ],
            )?;
            insert_ranked_values(
                &tx,
                "external_source_dossier_values",
                "dossier_id",
                &dossier.id,
                "truth_surface",
                &dossier.truth_surfaces,
            )?;
            insert_ranked_values(
                &tx,
                "external_source_dossier_values",
                "dossier_id",
                &dossier.id,
                "artifact_contract_path",
                &dossier.artifact_contract_paths,
            )?;
            insert_ranked_values(
                &tx,
                "external_source_dossier_values",
                "dossier_id",
                &dossier.id,
                "claim_ref",
                &dossier.claim_refs,
            )?;
            insert_ranked_values(
                &tx,
                "external_source_dossier_values",
                "dossier_id",
                &dossier.id,
                "url_ref",
                &dossier.url_refs,
            )?;
            insert_ranked_values(
                &tx,
                "external_source_dossier_values",
                "dossier_id",
                &dossier.id,
                "path_ref",
                &dossier.path_refs,
            )?;
        }
        tx.commit()?;

        self.record_control_plane_run(
            "index_external_sources",
            &serde_json::json!({
                "source_contracts": to_repo_rel(repo_root, source_contracts_path),
                "dossiers_registry": to_repo_rel(repo_root, dossiers_registry_path),
                "source_contract_count": contracts.len(),
                "dossier_count": dossiers.len(),
            })
            .to_string(),
        )?;
        Ok((contracts.len(), dossiers.len()))
    }

    pub fn export_external_sources_compat(
        &mut self,
        repo_root: &Path,
        source_contracts_path: &Path,
        dossiers_registry_path: &Path,
    ) -> Result<()> {
        let outputs = self.render_external_sources_compat_outputs()?;
        write_text(source_contracts_path, &outputs.source_contracts)?;
        write_text(dossiers_registry_path, &outputs.dossiers_registry)?;
        for (path, body) in &outputs.docs {
            write_text(&repo_root.join(path.as_str()), body)?;
        }
        self.record_control_plane_run(
            "export_external_sources",
            &serde_json::json!({
                "source_contracts": to_repo_rel(repo_root, source_contracts_path),
                "dossiers_registry": to_repo_rel(repo_root, dossiers_registry_path),
                "doc_count": outputs.docs.len(),
            })
            .to_string(),
        )?;
        Ok(())
    }

    pub fn verify_external_sources_compat_exports(
        &mut self,
        repo_root: &Path,
        source_contracts_path: &Path,
        dossiers_registry_path: &Path,
    ) -> Result<()> {
        let outputs = self.render_external_sources_compat_outputs()?;
        let mut failures = Vec::new();
        for (path, expected) in [
            (source_contracts_path, outputs.source_contracts.as_str()),
            (dossiers_registry_path, outputs.dossiers_registry.as_str()),
        ] {
            if !path.exists() {
                failures.push(format!("missing compatibility export {}", path.display()));
                continue;
            }
            let actual = load_text(path)?;
            if actual != format!("{expected}\n") {
                failures.push(format!("stale compatibility export {}", path.display()));
            }
        }
        for (path, expected) in outputs.docs {
            let full = repo_root.join(path.as_str());
            if !full.exists() {
                failures.push(format!("missing generated dossier {}", full.display()));
                continue;
            }
            let actual = load_text(&full)?;
            if actual != format!("{expected}\n") {
                failures.push(format!("stale dossier export {}", full.display()));
            }
        }
        if !failures.is_empty() {
            bail!(
                "external-source compatibility exports failed:\n- {}",
                failures.join("\n- ")
            );
        }
        Ok(())
    }

    pub fn verify_external_source_invariants(&self, repo_root: &Path) -> Result<()> {
        let mut failures = Vec::new();
        let contract_count =
            scalar_count(&self.conn, "SELECT COUNT(*) FROM external_source_contracts")?;
        let dossier_count =
            scalar_count(&self.conn, "SELECT COUNT(*) FROM external_source_dossiers")?;
        if contract_count == 0 {
            failures.push("external-source database has zero source contracts".to_string());
        }
        if dossier_count == 0 {
            failures.push("external-source database has zero dossiers".to_string());
        }
        let meta_doc_count = self.conn.query_row(
            "SELECT document_count FROM external_source_dossiers_meta WHERE kind = 'source_dossiers'",
            [],
            |row| row.get::<_, i64>(0),
        ).optional()?.unwrap_or_default();
        if meta_doc_count != dossier_count as i64 {
            failures.push(format!(
                "dossier meta document_count mismatch: meta={} db={}",
                meta_doc_count, dossier_count
            ));
        }
        for dossier in self.list_external_source_dossiers()? {
            if dossier.source_markdown.trim().is_empty() {
                failures.push(format!("{} has empty source_markdown", dossier.id));
            } else {
                let path = repo_root.join(&dossier.source_markdown);
                if let Some(parent) = path.parent()
                    && !parent.exists()
                {
                    failures.push(format!(
                        "{} source_markdown parent missing on disk: {}",
                        dossier.id,
                        parent.display()
                    ));
                }
            }
        }
        if !failures.is_empty() {
            bail!(
                "external-source invariants failed:\n- {}",
                failures.join("\n- ")
            );
        }
        Ok(())
    }

    pub fn update_external_source_contract(
        &self,
        id: &str,
        patch: ExternalSourceContractPatch<'_>,
    ) -> Result<bool> {
        let exists = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM external_source_contracts WHERE id = ?1)",
            [id],
            |row| row.get::<_, i64>(0),
        )?;
        if exists == 0 {
            return Ok(false);
        }

        if let Some(value) = patch.path_glob {
            self.conn.execute(
                "UPDATE external_source_contracts SET path_glob = ?2 WHERE id = ?1",
                params![id, value],
            )?;
        }
        if let Some(value) = patch.canonical_url {
            self.conn.execute(
                "UPDATE external_source_contracts SET canonical_url = ?2 WHERE id = ?1",
                params![id, value],
            )?;
        }
        if let Some(value) = patch.access_class {
            self.conn.execute(
                "UPDATE external_source_contracts SET access_class = ?2 WHERE id = ?1",
                params![id, value],
            )?;
        }
        if let Some(value) = patch.status {
            self.conn.execute(
                "UPDATE external_source_contracts SET status = ?2 WHERE id = ?1",
                params![id, value],
            )?;
        }
        if let Some(value) = patch.retrieval_method {
            self.conn.execute(
                "UPDATE external_source_contracts SET retrieval_method = ?2 WHERE id = ?1",
                params![id, value],
            )?;
        }
        if let Some(value) = patch.attempt_deadline_utc {
            self.conn.execute(
                "UPDATE external_source_contracts SET attempt_deadline_utc = ?2 WHERE id = ?1",
                params![id, value],
            )?;
        }
        if let Some(value) = patch.resolution_deadline_utc {
            self.conn.execute(
                "UPDATE external_source_contracts SET resolution_deadline_utc = ?2 WHERE id = ?1",
                params![id, value],
            )?;
        }
        if let Some(value) = patch.blocker_note {
            self.conn.execute(
                "UPDATE external_source_contracts SET blocker_note = ?2 WHERE id = ?1",
                params![id, value],
            )?;
        }
        if let Some(values) = patch.mirror_urls {
            replace_ranked_values(
                &self.conn,
                "external_source_contract_values",
                "contract_id",
                id,
                "mirror_url",
                values,
            )?;
        }
        if let Some(values) = patch.evidence_refs {
            replace_ranked_values(
                &self.conn,
                "external_source_contract_values",
                "contract_id",
                id,
                "evidence_ref",
                values,
            )?;
        }
        if let Some(values) = patch.manual_manifest_refs {
            replace_ranked_values(
                &self.conn,
                "external_source_contract_values",
                "contract_id",
                id,
                "manual_manifest_ref",
                values,
            )?;
        }
        if let Some(values) = patch.blocked_action_plan {
            replace_ranked_values(
                &self.conn,
                "external_source_contract_values",
                "contract_id",
                id,
                "blocked_action_plan",
                values,
            )?;
        }
        if let Some(values) = patch.scientific_validator_refs {
            replace_ranked_values(
                &self.conn,
                "external_source_contract_values",
                "contract_id",
                id,
                "scientific_validator_ref",
                values,
            )?;
        }

        self.conn.execute(
            "UPDATE external_source_contracts_meta
             SET updated = ?1, authoritative = 1
             WHERE kind = 'source_contracts'",
            [Utc::now().to_rfc3339()],
        )?;
        Ok(true)
    }

    pub fn upsert_external_source_contract(
        &self,
        id: &str,
        patch: ExternalSourceContractPatch<'_>,
    ) -> Result<bool> {
        if self.update_external_source_contract(id, patch)? {
            return Ok(false);
        }
        let path_glob = patch.path_glob.ok_or_else(|| {
            anyhow!("path_glob is required when creating external source contract {id}")
        })?;
        let canonical_url = patch.canonical_url.ok_or_else(|| {
            anyhow!("canonical_url is required when creating external source contract {id}")
        })?;
        let access_class = patch.access_class.ok_or_else(|| {
            anyhow!("access_class is required when creating external source contract {id}")
        })?;
        let status = patch.status.ok_or_else(|| {
            anyhow!("status is required when creating external source contract {id}")
        })?;
        let retrieval_method = patch.retrieval_method.ok_or_else(|| {
            anyhow!("retrieval_method is required when creating external source contract {id}")
        })?;
        let attempt_deadline_utc = patch.attempt_deadline_utc.ok_or_else(|| {
            anyhow!("attempt_deadline_utc is required when creating external source contract {id}")
        })?;
        let resolution_deadline_utc = patch.resolution_deadline_utc.ok_or_else(|| {
            anyhow!(
                "resolution_deadline_utc is required when creating external source contract {id}"
            )
        })?;
        let blocker_note = patch.blocker_note.unwrap_or("");

        self.conn.execute(
            "INSERT INTO external_source_contracts(
                id, path_glob, canonical_url, access_class, status, retrieval_method,
                attempt_deadline_utc, resolution_deadline_utc, blocker_note
            ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                id,
                path_glob,
                canonical_url,
                access_class,
                status,
                retrieval_method,
                attempt_deadline_utc,
                resolution_deadline_utc,
                blocker_note,
            ],
        )?;
        if let Some(values) = patch.mirror_urls {
            insert_ranked_values(
                &self.conn,
                "external_source_contract_values",
                "contract_id",
                id,
                "mirror_url",
                values,
            )?;
        }
        if let Some(values) = patch.evidence_refs {
            insert_ranked_values(
                &self.conn,
                "external_source_contract_values",
                "contract_id",
                id,
                "evidence_ref",
                values,
            )?;
        }
        if let Some(values) = patch.manual_manifest_refs {
            insert_ranked_values(
                &self.conn,
                "external_source_contract_values",
                "contract_id",
                id,
                "manual_manifest_ref",
                values,
            )?;
        }
        if let Some(values) = patch.blocked_action_plan {
            insert_ranked_values(
                &self.conn,
                "external_source_contract_values",
                "contract_id",
                id,
                "blocked_action_plan",
                values,
            )?;
        }
        if let Some(values) = patch.scientific_validator_refs {
            insert_ranked_values(
                &self.conn,
                "external_source_contract_values",
                "contract_id",
                id,
                "scientific_validator_ref",
                values,
            )?;
        }
        self.conn.execute(
            "UPDATE external_source_contracts_meta
             SET updated = ?1, authoritative = 1
             WHERE kind = 'source_contracts'",
            [Utc::now().to_rfc3339()],
        )?;
        Ok(true)
    }

    fn render_control_plane_compat_outputs(&self) -> Result<ControlPlaneCompatOutputs> {
        let theorem_rows = self.list_theorems()?;
        let experiments_meta = self
            .control_plane_meta_toml("experiments")?
            .unwrap_or_default();
        Ok(ControlPlaneCompatOutputs {
            claims: render_claims_registry(&self.list_claims()?),
            insights: render_insights_registry(&self.list_insights()?),
            experiments: render_experiments_registry(&experiments_meta, &self.list_experiments()?),
            binaries: render_binaries_registry(&self.list_binaries()?),
            theorems: render_theorem_markdown(
                "SQLite canonical database (compatibility export)",
                &theorem_rows,
            ),
            theorems_mirror: render_theorem_markdown(
                "registry/canonical/control_plane.sqlite3",
                &theorem_rows,
            ),
        })
    }

    fn render_external_sources_compat_outputs(&self) -> Result<ExternalSourcesCompatOutputs> {
        let contracts_meta = self.external_source_contracts_meta()?;
        let contracts = self.list_external_source_contracts()?;
        let dossiers_meta = self.external_source_dossiers_meta()?;
        let dossiers = self.list_external_source_dossiers()?;
        let docs = dossiers
            .iter()
            .map(|dossier| {
                (
                    Utf8PathBuf::from(dossier.source_markdown.clone()),
                    render_external_source_dossier_markdown(dossier),
                )
            })
            .collect();
        Ok(ExternalSourcesCompatOutputs {
            source_contracts: render_external_source_contracts_registry(
                &contracts_meta,
                &contracts,
            ),
            dossiers_registry: render_external_source_dossiers_registry(&dossiers_meta, &dossiers),
            docs,
        })
    }

    pub fn list_claims(&self) -> Result<Vec<ClaimRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, statement, status, where_stated, last_verified, formal_proof, status_note, compat_toml_text
             FROM claims ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ClaimRecord {
                id: row.get(0)?,
                statement: row.get(1)?,
                status: row.get(2)?,
                where_stated: row.get(3)?,
                last_verified: row.get(4)?,
                formal_proof: row.get(5)?,
                status_note: row.get(6)?,
                compat_toml_text: row.get(7)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_insights(&self) -> Result<Vec<InsightRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, status, claim_refs_json, status_note, compat_toml_text
             FROM insights ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let claim_refs_json: String = row.get(3)?;
            Ok(InsightRecord {
                id: row.get(0)?,
                title: row.get(1)?,
                status: row.get(2)?,
                claim_refs: serde_json::from_str(&claim_refs_json).unwrap_or_default(),
                status_note: row.get(4)?,
                compat_toml_text: row.get(5)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_experiments(&self) -> Result<Vec<ExperimentRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, status, binary_name, claim_refs_json, status_note, compat_toml_text
             FROM experiments_cp ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let claim_refs_json: String = row.get(4)?;
            Ok(ExperimentRecord {
                id: row.get(0)?,
                title: row.get(1)?,
                status: row.get(2)?,
                binary: row.get(3)?,
                claim_refs: serde_json::from_str(&claim_refs_json).unwrap_or_default(),
                status_note: row.get(5)?,
                compat_toml_text: row.get(6)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_theorems(&self) -> Result<Vec<TheoremRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, proof_path, status, linked_claim_ids_json, source
             FROM theorems ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let links: String = row.get(4)?;
            Ok(TheoremRecord {
                id: row.get(0)?,
                title: row.get(1)?,
                proof_path: Utf8PathBuf::from(row.get::<_, String>(2)?),
                status: row.get(3)?,
                linked_claim_ids: serde_json::from_str(&links).unwrap_or_default(),
                source: row.get(5)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_binaries(&self) -> Result<Vec<BinaryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT name, crate_name, description, experiment_id, source
             FROM binaries_cp ORDER BY name",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(BinaryRecord {
                name: row.get(0)?,
                crate_name: row.get(1)?,
                description: row.get(2)?,
                experiment: row.get(3)?,
                source: row.get(4)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_external_source_contracts(&self) -> Result<Vec<ExternalSourceContractRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path_glob, canonical_url, access_class, status, retrieval_method,
                    attempt_deadline_utc, resolution_deadline_utc, blocker_note
             FROM external_source_contracts
             ORDER BY id",
        )?;
        let base_rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut out = Vec::with_capacity(base_rows.len());
        for (
            id,
            path_glob,
            canonical_url,
            access_class,
            status,
            retrieval_method,
            attempt_deadline_utc,
            resolution_deadline_utc,
            blocker_note,
        ) in base_rows
        {
            out.push(ExternalSourceContractRecord {
                id: id.clone(),
                path_glob,
                canonical_url,
                mirror_urls: load_ranked_values(
                    &self.conn,
                    "external_source_contract_values",
                    "contract_id",
                    &id,
                    "mirror_url",
                )?,
                access_class,
                status,
                retrieval_method,
                attempt_deadline_utc,
                resolution_deadline_utc,
                blocker_note,
                evidence_refs: load_ranked_values(
                    &self.conn,
                    "external_source_contract_values",
                    "contract_id",
                    &id,
                    "evidence_ref",
                )?,
                manual_manifest_refs: load_ranked_values(
                    &self.conn,
                    "external_source_contract_values",
                    "contract_id",
                    &id,
                    "manual_manifest_ref",
                )?,
                blocked_action_plan: load_ranked_values(
                    &self.conn,
                    "external_source_contract_values",
                    "contract_id",
                    &id,
                    "blocked_action_plan",
                )?,
                scientific_validator_refs: load_ranked_values(
                    &self.conn,
                    "external_source_contract_values",
                    "contract_id",
                    &id,
                    "scientific_validator_ref",
                )?,
            });
        }
        Ok(out)
    }

    pub fn list_external_source_dossiers(&self) -> Result<Vec<ExternalSourceDossierRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_markdown, slug, title, status_token, content_kind,
                    authority_level, verification_level, operational_role,
                    source_lineage_summary, has_full_transcript, line_count, notes, body_markdown
             FROM external_source_dossiers
             ORDER BY id",
        )?;
        let base_rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                    row.get::<_, String>(12)?,
                    row.get::<_, String>(13)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut out = Vec::with_capacity(base_rows.len());
        for (
            id,
            source_markdown,
            slug,
            title,
            status_token,
            content_kind,
            authority_level,
            verification_level,
            operational_role,
            source_lineage_summary,
            has_full_transcript,
            line_count,
            notes,
            body_markdown,
        ) in base_rows
        {
            out.push(ExternalSourceDossierRecord {
                id: id.clone(),
                source_markdown,
                slug,
                title,
                status_token,
                content_kind,
                authority_level,
                verification_level,
                operational_role,
                source_lineage_summary,
                truth_surfaces: load_ranked_values(
                    &self.conn,
                    "external_source_dossier_values",
                    "dossier_id",
                    &id,
                    "truth_surface",
                )?,
                artifact_contract_paths: load_ranked_values(
                    &self.conn,
                    "external_source_dossier_values",
                    "dossier_id",
                    &id,
                    "artifact_contract_path",
                )?,
                has_full_transcript: has_full_transcript != 0,
                claim_refs: load_ranked_values(
                    &self.conn,
                    "external_source_dossier_values",
                    "dossier_id",
                    &id,
                    "claim_ref",
                )?,
                url_refs: load_ranked_values(
                    &self.conn,
                    "external_source_dossier_values",
                    "dossier_id",
                    &id,
                    "url_ref",
                )?,
                path_refs: load_ranked_values(
                    &self.conn,
                    "external_source_dossier_values",
                    "dossier_id",
                    &id,
                    "path_ref",
                )?,
                line_count: line_count as usize,
                notes,
                body_markdown,
            });
        }
        Ok(out)
    }

    pub fn registry_snapshot(&self, kind: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT content_text FROM registry_snapshots WHERE registry_kind = ?1",
                params![kind],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn record_registry_snapshot(
        &mut self,
        repo_root: &Path,
        kind: &str,
        source_path: &Path,
        raw: &str,
    ) -> Result<()> {
        let indexed_at = Utc::now().to_rfc3339();
        let tx = self.conn.transaction()?;
        write_registry_snapshot(&tx, repo_root, kind, source_path, raw, &indexed_at)?;
        tx.commit()?;
        Ok(())
    }

    fn control_plane_meta_toml(&self, kind: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT compat_toml_text FROM control_plane_meta WHERE kind = ?1",
                params![kind],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    fn external_source_contracts_meta(&self) -> Result<ExternalSourceContractsMeta> {
        self.conn
            .query_row(
                "SELECT updated, authoritative, policy_version
                 FROM external_source_contracts_meta
                 WHERE kind = 'source_contracts'",
                [],
                |row| {
                    Ok(ExternalSourceContractsMeta {
                        updated: row.get(0)?,
                        authoritative: row.get::<_, i64>(1)? != 0,
                        policy_version: row.get(2)?,
                    })
                },
            )
            .optional()
            .map(|row| row.unwrap_or_default())
            .map_err(Into::into)
    }

    fn external_source_dossiers_meta(&self) -> Result<ExternalSourceDossiersMeta> {
        self.conn
            .query_row(
                "SELECT updated, authoritative, source_markdown_glob, document_count
                 FROM external_source_dossiers_meta
                 WHERE kind = 'source_dossiers'",
                [],
                |row| {
                    Ok(ExternalSourceDossiersMeta {
                        updated: row.get(0)?,
                        authoritative: row.get::<_, i64>(1)? != 0,
                        source_markdown_glob: row.get(2)?,
                        document_count: row.get::<_, i64>(3)? as usize,
                    })
                },
            )
            .optional()
            .map(|row| row.unwrap_or_default())
            .map_err(Into::into)
    }

    fn record_control_plane_run(&mut self, action: &str, details_json: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO control_plane_runs(action, created_at, details_json)
             VALUES(?1, ?2, ?3)",
            params![action, Utc::now().to_rfc3339(), details_json],
        )?;
        Ok(())
    }

    fn backfill_control_plane_compat_from_snapshots(&mut self) -> Result<()> {
        self.backfill_claim_compat_from_snapshot()?;
        self.backfill_insight_compat_from_snapshot()?;
        self.backfill_experiment_compat_from_snapshot()?;
        Ok(())
    }

    fn backfill_claim_compat_from_snapshot(&mut self) -> Result<()> {
        let missing = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM claims WHERE compat_toml_text = ''",
        )?;
        if missing == 0 {
            return Ok(());
        }
        let Some(snapshot) = self.registry_snapshot("claims")? else {
            return Ok(());
        };
        let rows = load_claims_from_registry(&snapshot)?;
        let tx = self.conn.transaction()?;
        for row in rows {
            tx.execute(
                "UPDATE claims SET compat_toml_text = ?2 WHERE id = ?1 AND compat_toml_text = ''",
                params![row.id, row.compat_toml_text],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    fn backfill_insight_compat_from_snapshot(&mut self) -> Result<()> {
        let missing = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM insights WHERE compat_toml_text = ''",
        )?;
        if missing == 0 {
            return Ok(());
        }
        let Some(snapshot) = self.registry_snapshot("insights")? else {
            return Ok(());
        };
        let rows = load_insights_from_registry(&snapshot)?;
        let tx = self.conn.transaction()?;
        for row in rows {
            tx.execute(
                "UPDATE insights SET compat_toml_text = ?2 WHERE id = ?1 AND compat_toml_text = ''",
                params![row.id, row.compat_toml_text],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    fn backfill_experiment_compat_from_snapshot(&mut self) -> Result<()> {
        let missing_rows = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM experiments_cp WHERE compat_toml_text = ''",
        )?;
        let missing_meta = self
            .control_plane_meta_toml("experiments")?
            .unwrap_or_default()
            .trim()
            .is_empty();
        if missing_rows == 0 && !missing_meta {
            return Ok(());
        }
        let Some(snapshot) = self.registry_snapshot("experiments")? else {
            return Ok(());
        };
        let rows = load_experiments_from_registry(&snapshot)?;
        let meta = load_registry_table_toml(&snapshot, "experiments")?.unwrap_or_default();
        let tx = self.conn.transaction()?;
        for row in rows {
            tx.execute(
                "UPDATE experiments_cp SET compat_toml_text = ?2 WHERE id = ?1 AND compat_toml_text = ''",
                params![row.id, row.compat_toml_text],
            )?;
        }
        if missing_meta {
            tx.execute(
                "INSERT INTO control_plane_meta(kind, compat_toml_text)
                 VALUES(?1, ?2)
                 ON CONFLICT(kind) DO UPDATE SET compat_toml_text = excluded.compat_toml_text",
                params!["experiments", meta],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn record_export_run(
        &mut self,
        action: &str,
        artifact_count: usize,
        document_count: usize,
        details_json: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO export_runs (action, created_at, artifact_count, document_count, details_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                action,
                Utc::now().to_rfc3339(),
                artifact_count as i64,
                document_count as i64,
                details_json
            ],
        )?;
        Ok(())
    }

    pub fn artifact_by_needle(&self, needle: &str) -> Result<Option<ArtifactQueryResult>> {
        let row = self
            .conn
            .query_row(
                "SELECT id, key, title, citation, status, minimum_requirement_met,
                        canonical_functional_url, canonical_download_path
                 FROM artifacts
                 WHERE id = ?1 OR key = ?1
                    OR lower(title) LIKE '%' || lower(?1) || '%'
                 ORDER BY CASE WHEN id = ?1 OR key = ?1 THEN 0 ELSE 1 END, id
                 LIMIT 1",
                params![needle],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, i64>(5)?,
                        row.get::<_, Option<String>>(6)?,
                        row.get::<_, Option<String>>(7)?,
                    ))
                },
            )
            .optional()?;

        let Some((
            id,
            key,
            title,
            citation,
            status_raw,
            minimum_requirement_met,
            canonical_functional_url,
            canonical_download_path,
        )) = row
        else {
            return Ok(None);
        };
        let artifact = ArtifactRecord {
            id: id.clone(),
            key,
            title,
            citation,
            status: ArtifactStatus::parse(&status_raw)
                .with_context(|| format!("invalid artifact status {status_raw}"))?,
            minimum_requirement_met: minimum_requirement_met != 0,
            canonical_functional_url,
            canonical_download_path: canonical_download_path.map(Utf8PathBuf::from),
            source_refs: load_record_sources(&self.conn, "artifact", &id)?,
            all_links: load_string_vec(
                &self.conn,
                "SELECT url FROM artifact_links WHERE artifact_id = ?1 ORDER BY url",
                &id,
            )?,
            downloaded_paths: load_string_vec(
                &self.conn,
                "SELECT path FROM artifact_paths WHERE artifact_id = ?1 AND relation = 'downloaded' ORDER BY path",
                &id,
            )?
            .into_iter()
            .map(Utf8PathBuf::from)
            .collect(),
            doi_list: load_string_vec(
                &self.conn,
                "SELECT doi FROM citations WHERE artifact_id = ?1 AND doi IS NOT NULL ORDER BY doi",
                &id,
            )?,
            notes: Vec::new(),
        };

        let lanes = load_string_vec(
            &self.conn,
            "SELECT lane_name FROM lane_assignments WHERE artifact_id = ?1 ORDER BY lane_name",
            &artifact.id,
        )?;
        let mirror_observations = self.load_mirrors(&artifact.id)?;
        Ok(Some(ArtifactQueryResult {
            artifact,
            lanes,
            mirror_observations,
        }))
    }

    pub fn document_by_needle(&self, needle: &str) -> Result<Option<DocumentQueryResult>> {
        let document = self
            .conn
            .query_row(
                "SELECT id, path, title, kind, authoring_mode, generated, status,
                        toml_backing, sha256, size_bytes, line_count
                 FROM documents
                 WHERE id = ?1 OR path = ?1
                    OR lower(title) LIKE '%' || lower(?1) || '%'
                 ORDER BY CASE WHEN id = ?1 OR path = ?1 THEN 0 ELSE 1 END, id
                 LIMIT 1",
                params![needle],
                |row| {
                    Ok(DocumentRecord {
                        id: row.get(0)?,
                        path: Utf8PathBuf::from(row.get::<_, String>(1)?),
                        title: row.get(2)?,
                        kind: row.get(3)?,
                        authoring_mode: row.get(4)?,
                        generated: row.get::<_, i64>(5)? != 0,
                        status: row.get(6)?,
                        toml_backing: row.get::<_, Option<String>>(7)?.map(Utf8PathBuf::from),
                        sha256: row.get(8)?,
                        size_bytes: row.get(9)?,
                        line_count: row.get(10)?,
                    })
                },
            )
            .optional()?;

        let Some(document) = document else {
            return Ok(None);
        };
        let source_refs = load_record_sources(&self.conn, "document", &document.id)?;
        Ok(Some(DocumentQueryResult {
            document,
            source_refs,
        }))
    }

    pub fn recent_download_jobs(&self, limit: usize) -> Result<Vec<DownloadQueryResult>> {
        self.query_download_jobs(limit, None, None, None, None)
    }

    pub fn query_download_jobs(
        &self,
        limit: usize,
        needle: Option<&str>,
        host: Option<&str>,
        status: Option<&str>,
        backend: Option<&str>,
    ) -> Result<Vec<DownloadQueryResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, requested_url, transfer_kind, requested_backend, route_scheme, route_host,
                    route_backends_json, note, status, final_url, output_path, created_at
             FROM download_jobs
             WHERE (?1 IS NULL OR requested_url LIKE '%' || ?1 || '%')
               AND (?2 IS NULL OR route_host = ?2)
               AND (?3 IS NULL OR status = ?3)
               AND (?4 IS NULL OR requested_backend = ?4
                    OR EXISTS (
                        SELECT 1 FROM download_attempts a
                        WHERE a.job_id = download_jobs.id AND a.backend = ?4
                    ))
             ORDER BY id DESC
             LIMIT ?5",
        )?;
        let mut rows = stmt.query(params![needle, host, status, backend, limit as i64,])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            let job_id = row.get::<_, i64>(0)?;
            let route_backends_json = row.get::<_, String>(6)?;
            let attempts = self.download_attempts_for_job(job_id)?;
            results.push(DownloadQueryResult {
                job: DownloadJobRecord {
                    id: Some(job_id),
                    requested_url: row.get(1)?,
                    transfer_kind: row.get(2)?,
                    requested_backend: row.get(3)?,
                    route_scheme: row.get(4)?,
                    route_host: row.get(5)?,
                    route_backends: serde_json::from_str(&route_backends_json).unwrap_or_default(),
                    note: row.get(7)?,
                    status: row.get(8)?,
                    final_url: row.get(9)?,
                    output_path: row.get(10)?,
                    created_at: row.get(11)?,
                },
                attempts,
            });
        }
        Ok(results)
    }

    pub fn download_attempts_for_job(&self, job_id: i64) -> Result<Vec<DownloadAttemptRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, backend, succeeded, failure_class, http_code, content_type, bytes, sha256,
                    is_pdf, final_url, note, error_message, recorded_at
             FROM download_attempts
             WHERE job_id = ?1
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![job_id], |row| {
            Ok(DownloadAttemptRecord {
                id: Some(row.get(0)?),
                job_id: Some(job_id),
                backend: row.get(1)?,
                succeeded: row.get::<_, i64>(2)? != 0,
                failure_class: row.get(3)?,
                http_code: row.get(4)?,
                content_type: row.get(5)?,
                bytes: row.get(6)?,
                sha256: row.get(7)?,
                is_pdf: row.get::<_, i64>(8)? != 0,
                final_url: row.get(9)?,
                note: row.get(10)?,
                error_message: row.get(11)?,
                recorded_at: row.get(12)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn recent_download_campaigns(
        &self,
        limit: usize,
    ) -> Result<Vec<DownloadCampaignQueryResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.name, c.command_kind, c.input_path, c.out_ledger_path, c.dest_dir, c.note, c.created_at,
                    COUNT(j.id) AS job_count,
                    SUM(CASE WHEN j.status = 'succeeded' THEN 1 ELSE 0 END) AS success_count,
                    SUM(CASE WHEN j.status = 'failed' THEN 1 ELSE 0 END) AS failure_count
             FROM download_campaigns c
             LEFT JOIN download_campaign_jobs cj ON cj.campaign_id = c.id
             LEFT JOIN download_jobs j ON j.id = cj.job_id
             GROUP BY c.id, c.name, c.command_kind, c.input_path, c.out_ledger_path, c.dest_dir, c.note, c.created_at
             ORDER BY c.id DESC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(DownloadCampaignQueryResult {
                campaign: DownloadCampaignRecord {
                    id: Some(row.get(0)?),
                    name: row.get(1)?,
                    command_kind: row.get(2)?,
                    input_path: row.get(3)?,
                    out_ledger_path: row.get(4)?,
                    dest_dir: row.get(5)?,
                    note: row.get(6)?,
                    created_at: row.get(7)?,
                },
                job_count: row.get::<_, i64>(8)?.max(0) as usize,
                success_count: row.get::<_, Option<i64>>(9)?.unwrap_or(0).max(0) as usize,
                failure_count: row.get::<_, Option<i64>>(10)?.unwrap_or(0).max(0) as usize,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn project_download_history_rows(
        &self,
        limit: usize,
        needle: Option<&str>,
        host: Option<&str>,
        status: Option<&str>,
        backend: Option<&str>,
    ) -> Result<Vec<DownloadLedgerProjectionRow>> {
        let jobs = self.query_download_jobs(limit, needle, host, status, backend)?;
        let mut rows = Vec::new();
        for result in jobs {
            let job_id = result.job.id.unwrap_or_default();
            for attempt in result.attempts {
                let attempt_id = attempt.id.unwrap_or_default();
                let id = format!("job_{job_id:06}_attempt_{attempt_id:06}");
                let note = match attempt.error_message.as_deref() {
                    Some(error) if !error.is_empty() => {
                        format!("{}; error={error}", attempt.note)
                    }
                    _ => attempt.note.clone(),
                };
                let note = match attempt.failure_class.as_deref() {
                    Some(failure_class) if !failure_class.is_empty() => {
                        format!("{note}; failure_class={failure_class}")
                    }
                    _ => note,
                };
                rows.push(DownloadLedgerProjectionRow {
                    id,
                    url: attempt
                        .final_url
                        .clone()
                        .or_else(|| result.job.final_url.clone())
                        .unwrap_or_else(|| result.job.requested_url.clone()),
                    http_code: attempt
                        .http_code
                        .map(|value| value.to_string())
                        .unwrap_or_default(),
                    content_type: attempt.content_type.clone().unwrap_or_default(),
                    bytes: attempt.bytes.max(0) as u64,
                    sha256: attempt.sha256.clone().unwrap_or_default(),
                    is_pdf: if attempt.is_pdf { "yes" } else { "no" }.to_string(),
                    note,
                    status: result.job.status.clone(),
                });
            }
        }
        Ok(rows)
    }

    pub fn doctor_report(&self) -> Result<DoctorReport> {
        let artifact_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM artifacts")?;
        let document_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM documents")?;
        let missing_minimum_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM artifacts WHERE minimum_requirement_met = 0",
        )?;
        let blocked_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM artifacts WHERE status = 'blocked'",
        )?;
        let unverified_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM artifacts WHERE status = 'unverified'",
        )?;
        let citation_only_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM artifacts WHERE status = 'citation_only_no_link'",
        )?;
        let missing_lane_assignment_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM artifacts a
             WHERE NOT EXISTS (
                 SELECT 1 FROM lane_assignments l WHERE l.artifact_id = a.id
             )",
        )?;
        let documents_without_backing_count = scalar_count(
            &self.conn,
            "SELECT COUNT(*) FROM documents WHERE toml_backing IS NULL OR toml_backing = ''",
        )?;
        let download_job_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM download_jobs")?;
        let download_attempt_count =
            scalar_count(&self.conn, "SELECT COUNT(*) FROM download_attempts")?;
        let top_failed_download_hosts = query_count_summaries(
            &self.conn,
            "SELECT route_host, COUNT(*) AS count
             FROM download_jobs
             WHERE status = 'failed' AND route_host IS NOT NULL AND route_host != ''
             GROUP BY route_host
             ORDER BY count DESC, route_host ASC
             LIMIT 5",
        )?;
        let top_active_download_hosts = query_count_summaries(
            &self.conn,
            "SELECT route_host, COUNT(*) AS count
             FROM download_jobs
             WHERE route_host IS NOT NULL AND route_host != ''
             GROUP BY route_host
             ORDER BY count DESC, route_host ASC
             LIMIT 5",
        )?;
        let backend_health = query_backend_health(&self.conn)?;
        let last_indexed_at = self
            .conn
            .query_row(
                "SELECT created_at FROM export_runs WHERE action = 'index' ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let last_exported_at = self
            .conn
            .query_row(
                "SELECT created_at FROM export_runs WHERE action = 'export' ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(DoctorReport {
            generated_at: Utc::now().to_rfc3339(),
            artifact_count,
            document_count,
            missing_minimum_count,
            blocked_count,
            unverified_count,
            citation_only_count,
            missing_lane_assignment_count,
            documents_without_backing_count,
            download_job_count,
            download_attempt_count,
            top_failed_download_hosts,
            top_active_download_hosts,
            backend_health,
            last_indexed_at,
            last_exported_at,
        })
    }

    pub fn record_download_result(
        &mut self,
        job: &DownloadJobRecord,
        attempt: &DownloadAttemptRecord,
    ) -> Result<i64> {
        self.record_download_trace(job, std::slice::from_ref(attempt))
    }

    pub fn record_download_trace(
        &mut self,
        job: &DownloadJobRecord,
        attempts: &[DownloadAttemptRecord],
    ) -> Result<i64> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO download_jobs (
                requested_url, transfer_kind, requested_backend, route_scheme, route_host,
                route_backends_json, note, status, final_url, output_path, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                job.requested_url,
                job.transfer_kind,
                job.requested_backend,
                job.route_scheme,
                job.route_host,
                serde_json::to_string(&job.route_backends)?,
                job.note,
                job.status,
                job.final_url,
                job.output_path,
                job.created_at,
            ],
        )?;
        let job_id = tx.last_insert_rowid();
        for attempt in attempts {
            tx.execute(
                "INSERT INTO download_attempts (
                    job_id, backend, succeeded, http_code, content_type, bytes, sha256, is_pdf,
                    final_url, note, error_message, recorded_at, failure_class
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    job_id,
                    attempt.backend,
                    if attempt.succeeded { 1_i64 } else { 0_i64 },
                    attempt.http_code,
                    attempt.content_type,
                    attempt.bytes,
                    attempt.sha256,
                    if attempt.is_pdf { 1_i64 } else { 0_i64 },
                    attempt.final_url,
                    attempt.note,
                    attempt.error_message,
                    attempt.recorded_at,
                    attempt.failure_class,
                ],
            )?;
        }
        tx.commit()?;
        Ok(job_id)
    }

    pub fn create_download_campaign(&mut self, campaign: &DownloadCampaignRecord) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO download_campaigns (
                name, command_kind, input_path, out_ledger_path, dest_dir, note, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                campaign.name,
                campaign.command_kind,
                campaign.input_path,
                campaign.out_ledger_path,
                campaign.dest_dir,
                campaign.note,
                campaign.created_at,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn link_download_job_to_campaign(&mut self, campaign_id: i64, job_id: i64) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO download_campaign_jobs (campaign_id, job_id) VALUES (?1, ?2)",
            params![campaign_id, job_id],
        )?;
        Ok(())
    }

    pub fn record_literature_verification_run(
        &mut self,
        run: &LiteratureVerificationRunRecord,
        results: &[LiteratureVerificationResultRecord],
        similar_papers: &[LiteratureNoveltySimilarPaperRecord],
    ) -> Result<i64> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO literature_verification_runs (
                input_path, topic, hypotheses_path, domains_json, search_queries_json,
                total_entries, verified_count, suspicious_count, hallucinated_count, skipped_count,
                integrity_score, novelty_score, novelty_assessment, recommendation,
                search_coverage, total_papers_retrieved, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                run.input_path,
                run.topic,
                run.hypotheses_path,
                serde_json::to_string(&run.domains)?,
                serde_json::to_string(&run.search_queries)?,
                run.total_entries as i64,
                run.verified_count as i64,
                run.suspicious_count as i64,
                run.hallucinated_count as i64,
                run.skipped_count as i64,
                run.integrity_score,
                run.novelty_score,
                run.novelty_assessment,
                run.recommendation,
                run.search_coverage,
                run.total_papers_retrieved.map(|value| value as i64),
                run.created_at,
            ],
        )?;
        let run_id = tx.last_insert_rowid();

        for result in results {
            tx.execute(
                "INSERT INTO literature_verification_results (
                    run_id, cite_key, title, status, confidence, method, details, doi, arxiv_id,
                    matched_paper_title, matched_paper_source, matched_paper_year,
                    matched_paper_url, relevance_score
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                params![
                    run_id,
                    result.cite_key,
                    result.title,
                    result.status,
                    result.confidence,
                    result.method,
                    result.details,
                    result.doi,
                    result.arxiv_id,
                    result.matched_paper_title,
                    result.matched_paper_source,
                    result.matched_paper_year,
                    result.matched_paper_url,
                    result.relevance_score,
                ],
            )?;
        }

        for similar in similar_papers {
            tx.execute(
                "INSERT INTO literature_novelty_similar_papers (
                    run_id, title, paper_id, year, venue, citation_count, similarity, url, cite_key
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    run_id,
                    similar.title,
                    similar.paper_id,
                    similar.year,
                    similar.venue,
                    similar.citation_count,
                    similar.similarity,
                    similar.url,
                    similar.cite_key,
                ],
            )?;
        }

        tx.commit()?;
        Ok(run_id)
    }

    pub fn recent_literature_verification_runs(
        &self,
        limit: usize,
    ) -> Result<Vec<LiteratureVerificationQueryResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT
                id, input_path, topic, hypotheses_path, domains_json, search_queries_json,
                total_entries, verified_count, suspicious_count, hallucinated_count, skipped_count,
                integrity_score, novelty_score, novelty_assessment, recommendation,
                search_coverage, total_papers_retrieved, created_at
             FROM literature_verification_runs
             ORDER BY id DESC
             LIMIT ?1",
        )?;
        let run_rows = stmt.query_map(params![limit as i64], |row| {
            let domains_json: String = row.get(4)?;
            let search_queries_json: String = row.get(5)?;
            Ok(LiteratureVerificationRunRecord {
                id: row.get(0)?,
                input_path: row.get(1)?,
                topic: row.get(2)?,
                hypotheses_path: row.get(3)?,
                domains: serde_json::from_str(&domains_json).unwrap_or_default(),
                search_queries: serde_json::from_str(&search_queries_json).unwrap_or_default(),
                total_entries: row.get::<_, i64>(6)? as usize,
                verified_count: row.get::<_, i64>(7)? as usize,
                suspicious_count: row.get::<_, i64>(8)? as usize,
                hallucinated_count: row.get::<_, i64>(9)? as usize,
                skipped_count: row.get::<_, i64>(10)? as usize,
                integrity_score: row.get(11)?,
                novelty_score: row.get(12)?,
                novelty_assessment: row.get(13)?,
                recommendation: row.get(14)?,
                search_coverage: row.get(15)?,
                total_papers_retrieved: row.get::<_, Option<i64>>(16)?.map(|value| value as usize),
                created_at: row.get(17)?,
            })
        })?;

        let mut runs = Vec::new();
        for run in run_rows {
            let run = run?;
            let run_id = run.id.unwrap_or_default();
            runs.push(LiteratureVerificationQueryResult {
                results: self.literature_verification_results_for_run(run_id)?,
                similar_papers: self.literature_novelty_similar_papers_for_run(run_id)?,
                run,
            });
        }
        Ok(runs)
    }

    pub fn literature_verification_results_for_run(
        &self,
        run_id: i64,
    ) -> Result<Vec<LiteratureVerificationResultRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT
                id, run_id, cite_key, title, status, confidence, method, details, doi, arxiv_id,
                matched_paper_title, matched_paper_source, matched_paper_year,
                matched_paper_url, relevance_score
             FROM literature_verification_results
             WHERE run_id = ?1
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![run_id], |row| {
            Ok(LiteratureVerificationResultRecord {
                id: row.get(0)?,
                run_id: row.get(1)?,
                cite_key: row.get(2)?,
                title: row.get(3)?,
                status: row.get(4)?,
                confidence: row.get(5)?,
                method: row.get(6)?,
                details: row.get(7)?,
                doi: row.get(8)?,
                arxiv_id: row.get(9)?,
                matched_paper_title: row.get(10)?,
                matched_paper_source: row.get(11)?,
                matched_paper_year: row.get(12)?,
                matched_paper_url: row.get(13)?,
                relevance_score: row.get(14)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn literature_novelty_similar_papers_for_run(
        &self,
        run_id: i64,
    ) -> Result<Vec<LiteratureNoveltySimilarPaperRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT
                id, run_id, title, paper_id, year, venue, citation_count, similarity, url, cite_key
             FROM literature_novelty_similar_papers
             WHERE run_id = ?1
             ORDER BY similarity DESC, citation_count DESC, id ASC",
        )?;
        let rows = stmt.query_map(params![run_id], |row| {
            Ok(LiteratureNoveltySimilarPaperRecord {
                id: row.get(0)?,
                run_id: row.get(1)?,
                title: row.get(2)?,
                paper_id: row.get(3)?,
                year: row.get(4)?,
                venue: row.get(5)?,
                citation_count: row.get(6)?,
                similarity: row.get(7)?,
                url: row.get(8)?,
                cite_key: row.get(9)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn verify_invariants(&self, repo_root: &Path) -> Result<()> {
        let mut failures = Vec::new();
        let invalid_status_count = self.conn.query_row(
            "SELECT COUNT(*) FROM artifacts
             WHERE status NOT IN ('downloaded','downloadable','blocked','citation_only_no_link','unverified')",
            [],
            |row| row.get::<_, i64>(0),
        )? as usize;
        if invalid_status_count != 0 {
            failures.push(format!(
                "found {invalid_status_count} artifacts with invalid status"
            ));
        }
        let missing_lane_assignment_count = self.conn.query_row(
            "SELECT COUNT(*) FROM artifacts a
             WHERE NOT EXISTS (SELECT 1 FROM lane_assignments l WHERE l.artifact_id = a.id)",
            [],
            |row| row.get::<_, i64>(0),
        )? as usize;
        if missing_lane_assignment_count != 0 {
            failures.push(format!(
                "{missing_lane_assignment_count} artifacts missing lane assignments"
            ));
        }
        let mut stmt = self
            .conn
            .prepare("SELECT canonical_download_path FROM artifacts WHERE canonical_download_path IS NOT NULL AND canonical_download_path != ''")?;
        let paths = stmt.query_map([], |row| row.get::<_, String>(0))?;
        for path in paths {
            let path = path?;
            if !repo_root.join(&path).exists() {
                failures.push(format!("missing canonical download path on disk: {path}"));
            }
        }
        if !failures.is_empty() {
            bail!(
                "provenance store invariants failed:\n- {}",
                failures.join("\n- ")
            );
        }
        Ok(())
    }

    pub fn recovery_candidates(&self, limit: usize) -> Result<Vec<ArtifactRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, key, title, citation, status, minimum_requirement_met,
                    canonical_functional_url, canonical_download_path
             FROM artifacts
             WHERE minimum_requirement_met = 0
             ORDER BY status, id
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, Option<String>>(7)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (
                id,
                key,
                title,
                citation,
                status_raw,
                minimum_requirement_met,
                canonical_functional_url,
                canonical_download_path,
            ) = row?;
            out.push(ArtifactRecord {
                id,
                key,
                title,
                citation,
                status: ArtifactStatus::parse(&status_raw)
                    .with_context(|| format!("invalid artifact status {status_raw}"))?,
                minimum_requirement_met: minimum_requirement_met != 0,
                canonical_functional_url,
                canonical_download_path: canonical_download_path.map(Utf8PathBuf::from),
                source_refs: Vec::new(),
                all_links: Vec::new(),
                downloaded_paths: Vec::new(),
                doi_list: Vec::new(),
                notes: Vec::new(),
            });
        }
        Ok(out)
    }

    pub fn seed_pantheon_physicsforge_migration(
        &mut self,
        findings_path: &Path,
        overflow_path: &Path,
    ) -> Result<PantheonSeedSummary> {
        let findings_raw = load_toml_value(findings_path)?;
        let overflow_raw = load_toml_value(overflow_path)?;

        let findings_meta = findings_raw
            .get("migration_findings")
            .and_then(Value::as_table)
            .context("migration_findings table missing")?;
        let findings = findings_raw
            .get("finding")
            .and_then(Value::as_array)
            .context("finding table missing")?;
        let risks = findings_raw
            .get("risk")
            .and_then(Value::as_array)
            .context("risk table missing")?;

        let overflow_meta = overflow_raw
            .get("overflow_tracker")
            .and_then(Value::as_table)
            .context("overflow_tracker table missing")?;
        let overflow_rows = overflow_raw
            .get("overflow_task")
            .and_then(Value::as_array)
            .context("overflow_task table missing")?;

        let max_active_overflow = overflow_meta
            .get("max_active_tasks")
            .and_then(Value::as_integer)
            .unwrap_or(5)
            .max(0) as usize;
        let active_statuses = string_array_field(overflow_meta, "active_statuses");
        let active_count = overflow_rows
            .iter()
            .filter(|row| {
                row.as_table()
                    .and_then(|table| table.get("status"))
                    .and_then(Value::as_str)
                    .map(|status| {
                        active_statuses
                            .iter()
                            .any(|allowed| allowed == status.trim())
                    })
                    .unwrap_or(false)
            })
            .count();
        if active_count > max_active_overflow {
            bail!(
                "overflow tracker violates max active policy before sqlite seed: active={} max_active={}",
                active_count,
                max_active_overflow
            );
        }

        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS migration_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS migration_findings (
                finding_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                owner TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                updated TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS unresolved_risks (
                risk_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                mitigation TEXT NOT NULL,
                owner TEXT NOT NULL,
                eta TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                updated TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS overflow_tasks (
                overflow_id TEXT PRIMARY KEY,
                source_task_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                status TEXT NOT NULL,
                owner TEXT NOT NULL,
                eta TEXT NOT NULL,
                rationale TEXT NOT NULL,
                deferral_rationale TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                updated TEXT NOT NULL
            );
            ",
        )?;

        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO migration_meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![
                "source_findings_toml",
                findings_path.to_string_lossy().to_string()
            ],
        )?;
        tx.execute(
            "INSERT INTO migration_meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![
                "source_overflow_toml",
                overflow_path.to_string_lossy().to_string()
            ],
        )?;
        tx.execute(
            "INSERT INTO migration_meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params!["max_active_overflow", max_active_overflow.to_string()],
        )?;

        let findings_updated = findings_meta
            .get("updated")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        for row in findings {
            let table = row.as_table().context("finding row must be a table")?;
            tx.execute(
                "
                INSERT INTO migration_findings(
                    finding_id, task_id, phase, severity, status, summary, owner, evidence_refs, updated
                ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                ON CONFLICT(finding_id) DO UPDATE SET
                    task_id=excluded.task_id,
                    phase=excluded.phase,
                    severity=excluded.severity,
                    status=excluded.status,
                    summary=excluded.summary,
                    owner=excluded.owner,
                    evidence_refs=excluded.evidence_refs,
                    updated=excluded.updated
                ",
                params![
                    string_field(table, "id"),
                    string_field(table, "task_id"),
                    optional_integer_field(table, "phase").unwrap_or(0),
                    string_field(table, "severity"),
                    string_field(table, "status"),
                    string_field(table, "summary"),
                    string_field(table, "owner"),
                    join_refs(&string_array_field(table, "evidence_refs")),
                    findings_updated,
                ],
            )?;
        }

        for row in risks {
            let table = row.as_table().context("risk row must be a table")?;
            tx.execute(
                "
                INSERT INTO unresolved_risks(
                    risk_id, task_id, phase, risk_level, status, summary, mitigation, owner, eta, evidence_refs, updated
                ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                ON CONFLICT(risk_id) DO UPDATE SET
                    task_id=excluded.task_id,
                    phase=excluded.phase,
                    risk_level=excluded.risk_level,
                    status=excluded.status,
                    summary=excluded.summary,
                    mitigation=excluded.mitigation,
                    owner=excluded.owner,
                    eta=excluded.eta,
                    evidence_refs=excluded.evidence_refs,
                    updated=excluded.updated
                ",
                params![
                    string_field(table, "id"),
                    string_field(table, "task_id"),
                    optional_integer_field(table, "phase").unwrap_or(0),
                    string_field(table, "risk_level"),
                    string_field(table, "status"),
                    string_field(table, "summary"),
                    string_field(table, "mitigation"),
                    string_field(table, "owner"),
                    string_field(table, "eta"),
                    join_refs(&string_array_field(table, "evidence_refs")),
                    findings_updated,
                ],
            )?;
        }

        let overflow_updated = overflow_meta
            .get("updated")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        for row in overflow_rows {
            let table = row.as_table().context("overflow row must be a table")?;
            tx.execute(
                "
                INSERT INTO overflow_tasks(
                    overflow_id, source_task_id, phase, status, owner, eta, rationale, deferral_rationale, evidence_refs, updated
                ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                ON CONFLICT(overflow_id) DO UPDATE SET
                    source_task_id=excluded.source_task_id,
                    phase=excluded.phase,
                    status=excluded.status,
                    owner=excluded.owner,
                    eta=excluded.eta,
                    rationale=excluded.rationale,
                    deferral_rationale=excluded.deferral_rationale,
                    evidence_refs=excluded.evidence_refs,
                    updated=excluded.updated
                ",
                params![
                    string_field(table, "id"),
                    string_field(table, "source_task_id"),
                    optional_integer_field(table, "phase").unwrap_or(0),
                    string_field(table, "status"),
                    string_field(table, "owner"),
                    string_field(table, "eta"),
                    string_field(table, "rationale"),
                    string_field(table, "deferral_rationale"),
                    join_refs(&string_array_field(table, "evidence_refs")),
                    overflow_updated,
                ],
            )?;
        }
        tx.commit()?;

        let findings_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM migration_findings")?;
        let risk_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM unresolved_risks")?;
        let overflow_task_count = scalar_count(&self.conn, "SELECT COUNT(*) FROM overflow_tasks")?;
        let db_path = self
            .conn
            .pragma_query_value(None, "database_list", |row| row.get::<_, String>(2))
            .unwrap_or_else(|_| String::new());
        Ok(PantheonSeedSummary {
            db_path,
            findings_count,
            risk_count,
            overflow_task_count,
            max_active_overflow,
        })
    }

    fn load_mirrors(&self, artifact_id: &str) -> Result<Vec<MirrorObservationRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT artifact_id, url, mirror_kind
             FROM mirror_observations
             WHERE artifact_id = ?1
             ORDER BY mirror_kind, url",
        )?;
        let rows = stmt.query_map(params![artifact_id], |row| {
            let mirror_kind = match row.get::<_, String>(2)?.as_str() {
                "working" => MirrorKind::Working,
                "working_pdf" => MirrorKind::WorkingPdf,
                "nonworking" => MirrorKind::Nonworking,
                _ => MirrorKind::Unverified,
            };
            Ok(MirrorObservationRecord {
                artifact_id: row.get(0)?,
                url: row.get(1)?,
                mirror_kind,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    // ── Knowledge & planning layer ──────────────────────────────────

    /// Return row counts for all tables known to the source-of-truth manifest.
    pub fn source_of_truth_stats(&self) -> Result<Vec<(String, String, i64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT table_name, category, legacy_toml_path, migration_status
             FROM source_of_truth_manifest ORDER BY category, table_name",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (table, cat, legacy, status) = row?;
            let count: i64 = self
                .table_row_count(&table)
                .with_context(|| format!("failed to get row count for table `{table}`"))?;
            out.push((table, cat, count, format!("{legacy}|{status}")));
        }
        Ok(out)
    }

    /// Return the full source-of-truth manifest as structured rows.
    pub fn source_of_truth_manifest(&self) -> Result<Vec<ManifestRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT table_name, category, authoritative, legacy_toml_path, description, migration_status
             FROM source_of_truth_manifest ORDER BY category, table_name",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ManifestRow {
                table_name: row.get(0)?,
                category: row.get(1)?,
                authoritative: row.get(2)?,
                legacy_toml_path: row.get(3)?,
                description: row.get(4)?,
                migration_status: row.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// List all user-visible tables, their row counts, and column info.
    pub fn schema_summary(&self) -> Result<Vec<(String, i64, Vec<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT name FROM sqlite_master
             WHERE type='table' AND name NOT LIKE 'sqlite_%'
               AND name NOT LIKE '%_config'
               AND name NOT LIKE '__rusqlite%'
             ORDER BY name",
        )?;
        let names: Vec<String> = stmt
            .query_map([], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        let mut out = Vec::new();
        for name in &names {
            let count: i64 = self
                .conn
                .query_row(&format!("SELECT count(*) FROM [{name}]"), [], |r| r.get(0))
                .with_context(|| format!("Failed to count rows in table '{name}'"))?;
            let mut cols_stmt = self.conn.prepare(&format!("PRAGMA table_info([{name}])"))?;
            let cols: Vec<String> = cols_stmt
                .query_map([], |r| {
                    let col_name: String = r.get(1)?;
                    let col_type: String = r.get(2)?;
                    Ok(format!("{col_name} {col_type}"))
                })?
                .filter_map(|r| r.ok())
                .collect();
            out.push((name.clone(), count, cols));
        }
        Ok(out)
    }

    /// Insert or replace a roadmap item.
    pub fn upsert_roadmap_item(&self, item: &RoadmapItem<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO roadmap_items
             (id, name, priority, status, status_token, description, sprint,
              dependencies_json, acceptance_criteria_json, primary_outputs_json,
              evidence_refs_json, lacunae_json, claims_json, insight, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14, datetime('now'))",
            params![
                item.id,
                item.name,
                item.priority,
                item.status,
                item.status_token,
                item.description,
                item.sprint,
                item.dependencies_json,
                item.acceptance_criteria_json,
                item.primary_outputs_json,
                item.evidence_refs_json,
                item.lacunae_json,
                item.claims_json,
                item.insight,
            ],
        )?;
        Ok(())
    }

    pub fn delete_roadmap_item(&self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM roadmap_items WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Insert or replace a todo item.
    pub fn upsert_todo_item(&self, item: &ActionItem<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO todo_items
             (id, area, title, description, priority, status, status_token,
              dependencies_json, acceptance_criteria_json, evidence_refs_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10, datetime('now'))",
            params![
                item.id,
                item.area,
                item.title,
                item.description,
                item.priority,
                item.status,
                item.status_token,
                item.dependencies_json,
                item.acceptance_criteria_json,
                item.evidence_refs_json,
            ],
        )?;
        Ok(())
    }

    pub fn delete_todo_item(&self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM todo_items WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Read-only accessor for the current status_note on a claim row.
    /// Returns Ok(None) if the row exists but the column is NULL,
    /// Err if the row does not exist.
    pub fn claim_status_note(&self, id: &str) -> Result<Option<String>> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT status_note FROM claims WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| anyhow::anyhow!("claim {} not found in canonical DB: {}", id, e))?;
        Ok(row)
    }

    /// Update the status_note on a claim row inside a BEGIN IMMEDIATE
    /// transaction, append a row to claim_revisions, and return the
    /// audit record. The compat-export TOML must be regenerated
    /// afterwards via `make registry-export-markdown`.
    ///
    /// Idempotent: if the new note equals the current note, the
    /// function still records a `touch` revision so the actor + reason
    /// are preserved, but does not change the underlying row.
    pub fn claim_update_status_note(
        &mut self,
        id: &str,
        new_note: &str,
        actor: &str,
        reason: Option<&str>,
    ) -> Result<StatusNoteRevision> {
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let prev_note: Option<String> = tx
            .query_row(
                "SELECT status_note FROM claims WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "claim {} not found in canonical DB (or read failed): {}",
                    id,
                    e
                )
            })?;
        let prev_value_sha256 = prev_note.as_deref().map(sha256_hex);
        let new_value_sha256 = sha256_hex(new_note);
        let operation = if prev_note.as_deref() == Some(new_note) {
            "touch"
        } else {
            tx.execute(
                "UPDATE claims SET status_note = ?2 WHERE id = ?1",
                params![id, new_note],
            )?;
            "update"
        };
        tx.execute(
            "INSERT INTO claim_revisions
             (claim_id, field_name, prev_value_sha256, new_value_sha256,
              actor, reason, operation, application_id)
             VALUES (?1, 'status_note', ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                id,
                prev_value_sha256,
                new_value_sha256,
                actor,
                reason,
                operation,
                CLI_APPLICATION_ID
            ],
        )?;
        let revision_id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(StatusNoteRevision {
            entity_id: id.to_string(),
            field_name: "status_note".to_string(),
            prev_value_sha256,
            new_value_sha256,
            actor: actor.to_string(),
            reason: reason.map(str::to_string),
            revision_id,
        })
    }

    /// Read-only accessor for an insight's status_note column (added in
    /// migration 0016).
    pub fn insight_status_note(&self, id: &str) -> Result<Option<String>> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT status_note FROM insights WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| anyhow::anyhow!("insight {} not found in canonical DB: {}", id, e))?;
        Ok(row)
    }

    /// Update the status_note on an insight row. Mirrors
    /// claim_update_status_note end-to-end.
    pub fn insight_update_status_note(
        &mut self,
        id: &str,
        new_note: &str,
        actor: &str,
        reason: Option<&str>,
    ) -> Result<StatusNoteRevision> {
        self.entity_update_status_note(
            id,
            new_note,
            actor,
            reason,
            EntityFieldTarget {
                table: "insights",
                revisions_table: "insight_revisions",
                fk_col: "insight_id",
                field: "status_note",
            },
        )
    }

    /// Read-only accessor for an experiment's status_note column.
    pub fn experiment_status_note(&self, id: &str) -> Result<Option<String>> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT status_note FROM experiments_cp WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| anyhow::anyhow!("experiment {} not found in canonical DB: {}", id, e))?;
        Ok(row)
    }

    /// Update the status_note on an experiment row. Mirrors
    /// claim_update_status_note end-to-end.
    pub fn experiment_update_status_note(
        &mut self,
        id: &str,
        new_note: &str,
        actor: &str,
        reason: Option<&str>,
    ) -> Result<StatusNoteRevision> {
        self.entity_update_status_note(
            id,
            new_note,
            actor,
            reason,
            EntityFieldTarget {
                table: "experiments_cp",
                revisions_table: "experiment_revisions",
                fk_col: "experiment_id",
                field: "status_note",
            },
        )
    }

    /// Generic helper for status_note updates across claims, insights,
    /// experiments_cp. Caller passes the table, the revisions table, and
    /// the fk column name. All three call sites use this; it is the only
    /// place SQL is constructed for the entity-level update.
    fn entity_update_status_note(
        &mut self,
        id: &str,
        new_note: &str,
        actor: &str,
        reason: Option<&str>,
        target: EntityFieldTarget<'_>,
    ) -> Result<StatusNoteRevision> {
        self.entity_update_field(
            id,
            new_note,
            actor,
            reason,
            EntityFieldTarget {
                table: target.table,
                revisions_table: target.revisions_table,
                fk_col: target.fk_col,
                field: "status_note",
            },
        )
    }

    /// Generic per-column updater used by status_note and formal_proof
    /// mutators. Wraps a single BEGIN IMMEDIATE transaction that reads
    /// the prior value, hashes prev/new, conditionally writes, and
    /// appends a revisions audit row. `target.field` must be a trusted
    /// identifier from the call site (never user input).
    pub fn entity_update_field(
        &mut self,
        id: &str,
        new_value: &str,
        actor: &str,
        reason: Option<&str>,
        target: EntityFieldTarget<'_>,
    ) -> Result<StatusNoteRevision> {
        let EntityFieldTarget {
            table,
            revisions_table,
            fk_col,
            field,
        } = target;
        let select_sql = format!("SELECT {field} FROM {table} WHERE id = ?1");
        let update_sql = format!("UPDATE {table} SET {field} = ?2 WHERE id = ?1");
        let insert_sql = format!(
            "INSERT INTO {revisions_table}
             ({fk_col}, field_name, prev_value_sha256, new_value_sha256,
              actor, reason, operation, application_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"
        );
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let prev: Option<String> = tx
            .query_row(&select_sql, params![id], |row| row.get(0))
            .map_err(|e| anyhow::anyhow!("{} {} not found in canonical DB: {}", table, id, e))?;
        let prev_value_sha256 = prev.as_deref().map(sha256_hex);
        let new_value_sha256 = sha256_hex(new_value);
        let operation = if prev.as_deref() == Some(new_value) {
            "touch"
        } else {
            tx.execute(&update_sql, params![id, new_value])?;
            "update"
        };
        tx.execute(
            &insert_sql,
            params![
                id,
                field,
                prev_value_sha256,
                new_value_sha256,
                actor,
                reason,
                operation,
                CLI_APPLICATION_ID
            ],
        )?;
        let revision_id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(StatusNoteRevision {
            entity_id: id.to_string(),
            field_name: field.to_string(),
            prev_value_sha256,
            new_value_sha256,
            actor: actor.to_string(),
            reason: reason.map(str::to_string),
            revision_id,
        })
    }

    /// Read-only accessor for the current formal_proof on a claim row.
    pub fn claim_formal_proof(&self, id: &str) -> Result<Option<String>> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT formal_proof FROM claims WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| anyhow::anyhow!("claim {} not found in canonical DB: {}", id, e))?;
        Ok(row)
    }

    /// Update the formal_proof on a claim row. Mirrors
    /// claim_update_status_note end-to-end via entity_update_field.
    pub fn claim_update_formal_proof(
        &mut self,
        id: &str,
        new_value: &str,
        actor: &str,
        reason: Option<&str>,
    ) -> Result<StatusNoteRevision> {
        self.entity_update_field(
            id,
            new_value,
            actor,
            reason,
            EntityFieldTarget {
                table: "claims",
                revisions_table: "claim_revisions",
                fk_col: "claim_id",
                field: "formal_proof",
            },
        )
    }

    /// Insert or replace a next-action item.
    pub fn upsert_next_action(&self, item: &ActionItem<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO next_action_items
             (id, area, title, description, priority, status, status_token,
              dependencies_json, acceptance_criteria_json, evidence_refs_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10, datetime('now'))",
            params![
                item.id,
                item.area,
                item.title,
                item.description,
                item.priority,
                item.status,
                item.status_token,
                item.dependencies_json,
                item.acceptance_criteria_json,
                item.evidence_refs_json,
            ],
        )?;
        Ok(())
    }

    pub fn delete_next_action(&self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM next_action_items WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn upsert_requirements_meta(&self, meta: &RequirementsMeta<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT INTO requirements_registry_meta
             (kind, authoritative, status, status_token, updated, python_recommended,
              python_allowed, primary_markdown, status_allowlist_json,
              runtime_stack_allowlist_json, required_module_fields_json,
              required_gap_fields_json, updated_at)
             VALUES ('requirements', ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
             ON CONFLICT(kind) DO UPDATE SET
                 authoritative               = excluded.authoritative,
                 status                      = excluded.status,
                 status_token                = excluded.status_token,
                 updated                     = excluded.updated,
                 python_recommended          = excluded.python_recommended,
                 python_allowed              = excluded.python_allowed,
                 primary_markdown            = excluded.primary_markdown,
                 status_allowlist_json       = excluded.status_allowlist_json,
                 runtime_stack_allowlist_json = excluded.runtime_stack_allowlist_json,
                 required_module_fields_json = excluded.required_module_fields_json,
                 required_gap_fields_json    = excluded.required_gap_fields_json,
                 updated_at                  = excluded.updated_at",
            params![
                if meta.authoritative { 1 } else { 0 },
                meta.status,
                meta.status_token,
                meta.updated,
                meta.python_recommended,
                meta.python_allowed,
                meta.primary_markdown,
                meta.status_allowlist_json,
                meta.runtime_stack_allowlist_json,
                meta.required_module_fields_json,
                meta.required_gap_fields_json,
            ],
        )?;
        Ok(())
    }

    pub fn upsert_requirement_module(&self, item: &RequirementModuleItem<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO requirements_modules
             (id, name, markdown, status, status_token, runtime_stack,
              requires_modules_json, install_targets_json, verify_targets_json,
              acceptance_criteria_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10, datetime('now'))",
            params![
                item.id,
                item.name,
                item.markdown,
                item.status,
                item.status_token,
                item.runtime_stack,
                item.requires_modules_json,
                item.install_targets_json,
                item.verify_targets_json,
                item.acceptance_criteria_json,
            ],
        )?;
        Ok(())
    }

    pub fn delete_requirement_module(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM requirements_modules WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    pub fn upsert_requirement_coverage_gap(
        &self,
        item: &RequirementCoverageGapItem<'_>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO requirements_coverage_gaps
             (id, area, status, status_token, description, proposed_resolution,
              related_module_ids_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7, datetime('now'))",
            params![
                item.id,
                item.area,
                item.status,
                item.status_token,
                item.description,
                item.proposed_resolution,
                item.related_module_ids_json,
            ],
        )?;
        Ok(())
    }

    pub fn delete_requirement_coverage_gap(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM requirements_coverage_gaps WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Insert or replace a research narrative.
    pub fn upsert_research_narrative(&self, row: &ResearchNarrativeRow<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT INTO research_narratives
             (id, source_markdown, domain, slug, title, status_token, content_kind,
              verification_level, claim_refs_json, url_refs_json, path_refs_json,
              body_markdown, line_count)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)
             ON CONFLICT(id) DO UPDATE SET
                 source_markdown      = excluded.source_markdown,
                 domain               = excluded.domain,
                 slug                 = excluded.slug,
                 title                = excluded.title,
                 status_token         = excluded.status_token,
                 content_kind         = excluded.content_kind,
                 verification_level   = excluded.verification_level,
                 claim_refs_json      = excluded.claim_refs_json,
                 url_refs_json        = excluded.url_refs_json,
                 path_refs_json       = excluded.path_refs_json,
                 body_markdown        = excluded.body_markdown,
                 line_count           = excluded.line_count",
            params![
                row.id,
                row.source_markdown,
                row.domain,
                row.slug,
                row.title,
                row.status_token,
                row.content_kind,
                row.verification_level,
                row.claim_refs_json,
                row.url_refs_json,
                row.path_refs_json,
                row.body_markdown,
                row.line_count,
            ],
        )?;
        Ok(())
    }

    /// Count rows in a given table.
    pub fn table_row_count(&self, table: &str) -> Result<i64> {
        // Validate the table name exists in sqlite_master to prevent injection.
        let exists: bool = self.conn.query_row(
            "SELECT count(*) > 0 FROM sqlite_master WHERE type='table' AND name = ?1",
            params![table],
            |r| r.get(0),
        )?;
        if !exists {
            bail!("table '{}' does not exist", table);
        }
        let count = self
            .conn
            .query_row(&format!("SELECT count(*) FROM [{table}]"), [], |r| r.get(0))?;
        Ok(count)
    }

    /// Full-text search across research narratives.
    pub fn search_narratives(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, String, f64)>> {
        search_narratives_on_conn(&self.conn, query, limit)
    }

    /// Shared helper: query four TEXT columns from a table with optional status filter.
    fn list_four_col_table(
        &self,
        table: &str,
        cols: &str,
        status_filter: Option<&str>,
    ) -> Result<Vec<(String, String, String, String)>> {
        let mut out = Vec::new();
        if let Some(s) = status_filter {
            let sql = format!("SELECT {cols} FROM [{table}] WHERE status = ?1 ORDER BY id");
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(params![s], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?;
            for r in rows {
                out.push(r?);
            }
        } else {
            let sql = format!("SELECT {cols} FROM [{table}] ORDER BY id");
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?;
            for r in rows {
                out.push(r?);
            }
        }
        Ok(out)
    }

    /// List roadmap items, optionally filtered by status.
    pub fn list_roadmap_items(
        &self,
        status_filter: Option<&str>,
    ) -> Result<Vec<(String, String, String, String)>> {
        self.list_four_col_table("roadmap_items", "id, name, priority, status", status_filter)
    }

    /// List todo items, optionally filtered by status.
    pub fn list_todo_items(
        &self,
        status_filter: Option<&str>,
    ) -> Result<Vec<(String, String, String, String)>> {
        self.list_four_col_table("todo_items", "id, title, priority, status", status_filter)
    }

    /// List next-action items, optionally filtered by status.
    pub fn list_next_actions(
        &self,
        status_filter: Option<&str>,
    ) -> Result<Vec<(String, String, String, String)>> {
        self.list_four_col_table(
            "next_action_items",
            "id, title, priority, status",
            status_filter,
        )
    }

    pub fn render_planning_compat_toml(&self, table: PlanningCompatTable) -> Result<String> {
        match table {
            PlanningCompatTable::Roadmap => self.render_roadmap_compat_toml(),
            PlanningCompatTable::Todo => self.render_todo_compat_toml(),
            PlanningCompatTable::NextActions => self.render_next_actions_compat_toml(),
        }
    }

    pub fn render_requirements_compat_toml(&self) -> Result<String> {
        let snapshot = self.parse_registry_snapshot("requirements")?;
        let requirements_table = snapshot
            .as_ref()
            .and_then(|value| compat_root_table(value, "requirements"));
        let schema_table = compat_child_table(requirements_table, "schema");
        let meta = self.requirements_meta_row()?;
        let modules = self.requirements_module_rows()?;
        let gaps = self.requirements_coverage_gap_rows()?;

        let authoritative = meta
            .as_ref()
            .map(|row| row.authoritative)
            .unwrap_or_else(|| compat_table_bool(requirements_table, "authoritative", true));
        let status = meta
            .as_ref()
            .map(|row| row.status.clone())
            .unwrap_or_else(|| compat_table_string(requirements_table, "status", "active"));
        let status_token = meta
            .as_ref()
            .map(|row| row.status_token.clone())
            .unwrap_or_else(|| compat_table_string(requirements_table, "status_token", "ACTIVE"));
        let updated = meta
            .as_ref()
            .map(|row| row.updated.clone())
            .unwrap_or_else(|| compat_table_string(requirements_table, "updated", "2026-02-10"));
        let python_recommended = meta
            .as_ref()
            .map(|row| row.python_recommended.clone())
            .unwrap_or_else(|| {
                compat_table_string(requirements_table, "python_recommended", "3.11-3.12")
            });
        let python_allowed = meta
            .as_ref()
            .map(|row| row.python_allowed.clone())
            .unwrap_or_else(|| {
                compat_table_string(
                    requirements_table,
                    "python_allowed",
                    "3.13+ (with optional extras caveats)",
                )
            });
        let primary_markdown = meta
            .as_ref()
            .map(|row| row.primary_markdown.clone())
            .unwrap_or_else(|| {
                compat_table_string(
                    requirements_table,
                    "primary_markdown",
                    "docs/REQUIREMENTS.md",
                )
            });
        let status_allowlist = meta
            .as_ref()
            .map(|row| compat_json_string_array(&row.status_allowlist_json))
            .transpose()?
            .unwrap_or_else(|| {
                compat_table_array(
                    requirements_table,
                    "status_allowlist",
                    &["active", "deprecated", "planned", "blocked"],
                )
            });
        let runtime_stack_allowlist = meta
            .as_ref()
            .map(|row| compat_json_string_array(&row.runtime_stack_allowlist_json))
            .transpose()?
            .unwrap_or_else(|| {
                compat_table_array(
                    requirements_table,
                    "runtime_stack_allowlist",
                    &[
                        "mixed",
                        "rust",
                        "python",
                        "docker_python",
                        "rocq",
                        "latex",
                        "cpp",
                    ],
                )
            });
        let required_module_fields = meta
            .as_ref()
            .map(|row| compat_json_string_array(&row.required_module_fields_json))
            .transpose()?
            .unwrap_or_else(|| {
                compat_table_array(
                    schema_table,
                    "required_module_fields",
                    &[
                        "id",
                        "name",
                        "status",
                        "status_token",
                        "runtime_stack",
                        "requires_modules",
                        "install_targets",
                        "verify_targets",
                        "acceptance_criteria",
                    ],
                )
            });
        let required_gap_fields = meta
            .as_ref()
            .map(|row| compat_json_string_array(&row.required_gap_fields_json))
            .transpose()?
            .unwrap_or_else(|| {
                compat_table_array(
                    schema_table,
                    "required_gap_fields",
                    &[
                        "id",
                        "area",
                        "status",
                        "status_token",
                        "description",
                        "proposed_resolution",
                        "related_module_ids",
                    ],
                )
            });

        let mut lines = vec![
            "# GENERATED VIEW: DO NOT EDIT.".to_string(),
            "# Update via `gororoba-db requirements ...` against `registry/canonical/control_plane.sqlite3`.".to_string(),
            "# Requirements registry (SQLite compatibility export from canonical control_plane.sqlite3).".to_string(),
            "# Generated by `gororoba-db build` / `gororoba-db export-requirements`.".to_string(),
            String::new(),
            "[requirements]".to_string(),
            format!(
                "authoritative = {}",
                if authoritative { "true" } else { "false" }
            ),
            format!("status = {}", compat_toml_quote(&status)),
            format!("status_token = {}", compat_toml_quote(&status_token)),
            format!("updated = {}", compat_toml_quote(&updated)),
            format!(
                "python_recommended = {}",
                compat_toml_quote(&python_recommended)
            ),
            format!("python_allowed = {}", compat_toml_quote(&python_allowed)),
            format!(
                "primary_markdown = {}",
                compat_toml_quote(&primary_markdown)
            ),
            format!("module_count = {}", modules.len()),
            format!("coverage_gap_count = {}", gaps.len()),
            format!(
                "status_allowlist = {}",
                compat_toml_string_array(&status_allowlist)
            ),
            format!(
                "runtime_stack_allowlist = {}",
                compat_toml_string_array(&runtime_stack_allowlist)
            ),
            String::new(),
            "[requirements.schema]".to_string(),
            format!(
                "required_module_fields = {}",
                compat_toml_string_array(&required_module_fields)
            ),
            format!(
                "required_gap_fields = {}",
                compat_toml_string_array(&required_gap_fields)
            ),
            String::new(),
        ];

        for row in modules {
            lines.push("[[module]]".to_string());
            lines.push(format!("id = {}", compat_toml_quote(&row.id)));
            lines.push(format!("name = {}", compat_toml_quote(&row.name)));
            lines.push(format!("markdown = {}", compat_toml_quote(&row.markdown)));
            lines.push(format!("status = {}", compat_toml_quote(&row.status)));
            lines.push(format!(
                "status_token = {}",
                compat_toml_quote(&row.status_token)
            ));
            lines.push(format!(
                "runtime_stack = {}",
                compat_toml_quote(&row.runtime_stack)
            ));
            lines.push(format!(
                "requires_modules = {}",
                compat_toml_string_array(&compat_json_string_array(&row.requires_modules_json)?)
            ));
            lines.push(format!(
                "install_targets = {}",
                compat_toml_string_array(&compat_json_string_array(&row.install_targets_json)?)
            ));
            lines.push(format!(
                "verify_targets = {}",
                compat_toml_string_array(&compat_json_string_array(&row.verify_targets_json)?)
            ));
            lines.push(format!(
                "acceptance_criteria = {}",
                compat_toml_string_array(&compat_json_string_array(&row.acceptance_criteria_json)?)
            ));
            lines.push(String::new());
        }

        for row in gaps {
            lines.push("[[coverage_gap]]".to_string());
            lines.push(format!("id = {}", compat_toml_quote(&row.id)));
            lines.push(format!("area = {}", compat_toml_quote(&row.area)));
            lines.push(format!("status = {}", compat_toml_quote(&row.status)));
            lines.push(format!(
                "status_token = {}",
                compat_toml_quote(&row.status_token)
            ));
            lines.push(format!(
                "description = {}",
                compat_toml_quote(&row.description)
            ));
            lines.push(format!(
                "proposed_resolution = {}",
                compat_toml_quote(&row.proposed_resolution)
            ));
            lines.push(format!(
                "related_module_ids = {}",
                compat_toml_string_array(&compat_json_string_array(&row.related_module_ids_json)?)
            ));
            lines.push(String::new());
        }

        trim_trailing_blank_lines(&mut lines);
        Ok(lines.join("\n"))
    }

    pub fn verify_planning_requirements_compat_exports(
        &self,
        repo_root: &Path,
        roadmap_path: &Path,
        todo_path: &Path,
        next_actions_path: &Path,
        requirements_path: &Path,
    ) -> Result<()> {
        let checks = [
            (
                roadmap_path,
                self.render_planning_compat_toml(PlanningCompatTable::Roadmap)?,
            ),
            (
                todo_path,
                self.render_planning_compat_toml(PlanningCompatTable::Todo)?,
            ),
            (
                next_actions_path,
                self.render_planning_compat_toml(PlanningCompatTable::NextActions)?,
            ),
            (requirements_path, self.render_requirements_compat_toml()?),
        ];
        let mut failures = Vec::new();
        for (path, expected) in checks {
            if !path.exists() {
                failures.push(format!("missing compatibility export {}", path.display()));
                continue;
            }
            let actual =
                fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
            let expected = format!("{expected}\n");
            if actual != expected {
                let rel = path
                    .strip_prefix(repo_root)
                    .unwrap_or(path)
                    .display()
                    .to_string();
                failures.push(format!("{rel}: stale DB-backed compatibility export"));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            bail!(failures.join("\n"))
        }
    }

    fn render_roadmap_compat_toml(&self) -> Result<String> {
        let snapshot = self.parse_registry_snapshot("roadmap")?;
        let roadmap_table = snapshot
            .as_ref()
            .and_then(|value| compat_root_table(value, "roadmap"));
        let schema_table = compat_child_table(roadmap_table, "schema");
        let sections_table = compat_child_table(roadmap_table, "sections");
        let rows = self.planning_roadmap_rows()?;
        let status_allowlist = compat_table_array(
            roadmap_table,
            "status_allowlist",
            &[
                "planned",
                "active",
                "in_progress",
                "done",
                "paused",
                "blocked",
            ],
        );
        let priority_allowlist = compat_table_array(
            roadmap_table,
            "priority_allowlist",
            &["high", "medium", "low"],
        );
        let required_fields = compat_table_array(
            schema_table,
            "required_fields",
            &[
                "id",
                "name",
                "priority",
                "status",
                "status_token",
                "description",
                "dependencies",
                "acceptance_criteria",
            ],
        );

        let mut lines = vec![
            "# GENERATED VIEW: DO NOT EDIT.".to_string(),
            "# Update via `gororoba-db planning ...` against `registry/canonical/control_plane.sqlite3`.".to_string(),
            "# Operational roadmap registry (SQLite compatibility export from canonical control_plane.sqlite3).".to_string(),
            "# Generated by `gororoba-db build` / `gororoba-db export-planning --table roadmap`.".to_string(),
            String::new(),
            "[roadmap]".to_string(),
            format!(
                "source_markdown = {}",
                compat_toml_quote(&compat_table_string(
                    roadmap_table,
                    "source_markdown",
                    "docs/ROADMAP.md",
                ))
            ),
            format!(
                "consolidated_date = {}",
                compat_toml_quote(&compat_table_string(
                    roadmap_table,
                    "consolidated_date",
                    "2026-02-10",
                ))
            ),
            format!(
                "supersedes = {}",
                compat_toml_string_array(&compat_table_array(roadmap_table, "supersedes", &[]))
            ),
            format!(
                "companion_docs = {}",
                compat_toml_string_array(&compat_table_array(
                    roadmap_table,
                    "companion_docs",
                    &[]
                ))
            ),
            format!(
                "status = {}",
                compat_toml_quote(&compat_table_string(roadmap_table, "status", "active"))
            ),
            format!(
                "status_token = {}",
                compat_toml_quote(&compat_table_string(
                    roadmap_table,
                    "status_token",
                    "ACTIVE",
                ))
            ),
            format!(
                "authoritative = {}",
                if compat_table_bool(roadmap_table, "authoritative", true) {
                    "true"
                } else {
                    "false"
                }
            ),
            format!("workstream_count = {}", rows.len()),
            format!(
                "status_allowlist = {}",
                compat_toml_string_array(&status_allowlist)
            ),
            format!(
                "priority_allowlist = {}",
                compat_toml_string_array(&priority_allowlist)
            ),
            String::new(),
            "[roadmap.schema]".to_string(),
            format!(
                "required_fields = {}",
                compat_toml_string_array(&required_fields)
            ),
            format!(
                "dependency_id_pattern = {}",
                compat_toml_quote(&compat_table_string(
                    schema_table,
                    "dependency_id_pattern",
                    "WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*",
                ))
            ),
            String::new(),
            "[roadmap.sections]".to_string(),
            format!(
                "architectural_evolution = {}",
                compat_toml_quote(&compat_table_string(
                    sections_table,
                    "architectural_evolution",
                    "## 1. Architectural Evolution",
                ))
            ),
            format!(
                "crate_ecosystem = {}",
                compat_toml_quote(&compat_table_string(
                    sections_table,
                    "crate_ecosystem",
                    "## 2. Crate Ecosystem",
                ))
            ),
            format!(
                "documentation_registry = {}",
                compat_toml_quote(&compat_table_string(
                    sections_table,
                    "documentation_registry",
                    "### 1.5 Documentation Registry Evolution (Sprint 13)",
                ))
            ),
            format!(
                "long_term_vision = {}",
                compat_toml_quote(&compat_table_string(
                    sections_table,
                    "long_term_vision",
                    "## 8. Long-Term Vision",
                ))
            ),
            format!(
                "remaining_workstreams = {}",
                compat_toml_quote(&compat_table_string(
                    sections_table,
                    "remaining_workstreams",
                    "## 7. Remaining Workstreams from ULTRA_ROADMAP.md",
                ))
            ),
            String::new(),
        ];

        for row in rows {
            lines.push("[[workstream]]".to_string());
            lines.push(format!("id = {}", compat_toml_quote(&row.id)));
            lines.push(format!("name = {}", compat_toml_quote(&row.name)));
            lines.push(format!("priority = {}", compat_toml_quote(&row.priority)));
            lines.push(format!("status = {}", compat_toml_quote(&row.status)));
            lines.push(format!(
                "status_token = {}",
                compat_toml_quote(&row.status_token)
            ));
            lines.push(format!(
                "description = {}",
                compat_toml_quote(&row.description)
            ));
            if !row.sprint.trim().is_empty() {
                lines.push(format!("sprint = {}", compat_toml_quote(&row.sprint)));
            }
            lines.push(format!(
                "primary_outputs = {}",
                compat_toml_string_array(&compat_json_string_array(&row.primary_outputs_json)?)
            ));
            let claims = compat_json_string_array(&row.claims_json)?;
            if !claims.is_empty() {
                lines.push(format!("claims = {}", compat_toml_string_array(&claims)));
            }
            if !row.insight.trim().is_empty() {
                lines.push(format!("insight = {}", compat_toml_quote(&row.insight)));
            }
            lines.push(format!(
                "dependencies = {}",
                compat_toml_string_array(&compat_json_string_array(&row.dependencies_json)?)
            ));
            lines.push(format!(
                "acceptance_criteria = {}",
                compat_toml_string_array(&compat_json_string_array(&row.acceptance_criteria_json)?)
            ));
            lines.push(format!(
                "evidence_refs = {}",
                compat_toml_string_array(&compat_json_string_array(&row.evidence_refs_json)?)
            ));
            let lacunae = compat_json_string_array(&row.lacunae_json)?;
            if !lacunae.is_empty() {
                lines.push(format!("lacunae = {}", compat_toml_string_array(&lacunae)));
            }
            lines.push(String::new());
        }

        trim_trailing_blank_lines(&mut lines);
        Ok(lines.join("\n"))
    }

    fn render_todo_compat_toml(&self) -> Result<String> {
        let snapshot = self.parse_registry_snapshot("todo")?;
        let todo_table = snapshot
            .as_ref()
            .and_then(|value| compat_root_table(value, "todo"));
        let schema_table = compat_child_table(todo_table, "schema");
        let rows = self.planning_todo_rows()?;
        let status_allowlist = compat_table_array(
            todo_table,
            "status_allowlist",
            &["open", "in_progress", "done", "blocked", "deferred"],
        );
        let priority_allowlist =
            compat_table_array(todo_table, "priority_allowlist", &["high", "medium", "low"]);
        let required_fields = compat_table_array(
            schema_table,
            "required_fields",
            &[
                "id",
                "area",
                "title",
                "description",
                "priority",
                "status",
                "status_token",
                "dependencies",
                "acceptance_criteria",
            ],
        );

        let mut lines = vec![
            "# GENERATED VIEW: DO NOT EDIT.".to_string(),
            "# Update via `gororoba-db planning ...` against `registry/canonical/control_plane.sqlite3`.".to_string(),
            "# To-Do Registry (SQLite compatibility export from canonical control_plane.sqlite3).".to_string(),
            "# Generated by `gororoba-db build` / `gororoba-db export-planning --table todo`.".to_string(),
            String::new(),
            "[todo]".to_string(),
            format!(
                "updated = {}",
                compat_toml_quote(&compat_table_string(todo_table, "updated", "2026-02-10"))
            ),
            format!(
                "status = {}",
                compat_toml_quote(&compat_table_string(todo_table, "status", "active"))
            ),
            format!(
                "status_token = {}",
                compat_toml_quote(&compat_table_string(todo_table, "status_token", "ACTIVE"))
            ),
            format!("item_count = {}", rows.len()),
            format!(
                "status_allowlist = {}",
                compat_toml_string_array(&status_allowlist)
            ),
            format!(
                "priority_allowlist = {}",
                compat_toml_string_array(&priority_allowlist)
            ),
            String::new(),
            "[todo.schema]".to_string(),
            format!(
                "required_fields = {}",
                compat_toml_string_array(&required_fields)
            ),
            format!(
                "dependency_id_pattern = {}",
                compat_toml_quote(&compat_table_string(
                    schema_table,
                    "dependency_id_pattern",
                    "WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*",
                ))
            ),
            String::new(),
        ];
        for row in rows {
            lines.push("[[item]]".to_string());
            lines.push(format!("id = {}", compat_toml_quote(&row.id)));
            lines.push(format!("area = {}", compat_toml_quote(&row.area)));
            lines.push(format!("title = {}", compat_toml_quote(&row.title)));
            lines.push(format!(
                "description = {}",
                compat_toml_quote(&row.description)
            ));
            lines.push(format!("priority = {}", compat_toml_quote(&row.priority)));
            lines.push(format!("status = {}", compat_toml_quote(&row.status)));
            lines.push(format!(
                "status_token = {}",
                compat_toml_quote(&row.status_token)
            ));
            lines.push(format!(
                "dependencies = {}",
                compat_toml_string_array(&compat_json_string_array(&row.dependencies_json)?)
            ));
            lines.push(format!(
                "acceptance_criteria = {}",
                compat_toml_string_array(&compat_json_string_array(&row.acceptance_criteria_json)?)
            ));
            lines.push(format!(
                "evidence_refs = {}",
                compat_toml_string_array(&compat_json_string_array(&row.evidence_refs_json)?)
            ));
            lines.push(String::new());
        }
        trim_trailing_blank_lines(&mut lines);
        Ok(lines.join("\n"))
    }

    fn render_next_actions_compat_toml(&self) -> Result<String> {
        let snapshot = self.parse_registry_snapshot("next_actions")?;
        let meta_table = snapshot
            .as_ref()
            .and_then(|value| compat_root_table(value, "meta"));
        let next_actions_table = snapshot
            .as_ref()
            .and_then(|value| compat_root_table(value, "next_actions"));
        let schema_table = compat_child_table(next_actions_table, "schema");
        let rows = self.planning_next_action_rows()?;
        let status_allowlist = compat_table_array(
            meta_table,
            "status_allowlist",
            &["todo", "in_progress", "done", "blocked", "deferred"],
        );
        let priority_allowlist =
            compat_table_array(meta_table, "priority_allowlist", &["high", "medium", "low"]);
        let required_fields = compat_table_array(
            schema_table,
            "required_fields",
            &[
                "id",
                "area",
                "title",
                "description",
                "priority",
                "status",
                "status_token",
                "dependencies",
                "acceptance_criteria",
            ],
        );
        let mut lines = vec![
            "# GENERATED VIEW: DO NOT EDIT.".to_string(),
            "# Update via `gororoba-db planning ...` against `registry/canonical/control_plane.sqlite3`.".to_string(),
            "# Next Actions Registry (SQLite compatibility export from canonical control_plane.sqlite3).".to_string(),
            "# Generated by `gororoba-db build` / `gororoba-db export-planning --table next-actions`.".to_string(),
            String::new(),
            "[meta]".to_string(),
            format!(
                "updated = {}",
                compat_toml_quote(&compat_table_string(meta_table, "updated", "2026-02-10"))
            ),
            format!(
                "status = {}",
                compat_toml_quote(&compat_table_string(meta_table, "status", "active"))
            ),
            format!(
                "status_token = {}",
                compat_toml_quote(&compat_table_string(meta_table, "status_token", "ACTIVE"))
            ),
            format!("action_count = {}", rows.len()),
            format!(
                "status_allowlist = {}",
                compat_toml_string_array(&status_allowlist)
            ),
            format!(
                "priority_allowlist = {}",
                compat_toml_string_array(&priority_allowlist)
            ),
            String::new(),
            "[next_actions.schema]".to_string(),
            format!(
                "required_fields = {}",
                compat_toml_string_array(&required_fields)
            ),
            format!(
                "dependency_id_pattern = {}",
                compat_toml_quote(&compat_table_string(
                    schema_table,
                    "dependency_id_pattern",
                    "WS-*|T-*|NA-*|C-*|I-*|E-*|REQ-*",
                ))
            ),
            String::new(),
        ];
        for row in rows {
            lines.push("[[action]]".to_string());
            lines.push(format!("id = {}", compat_toml_quote(&row.id)));
            lines.push(format!("area = {}", compat_toml_quote(&row.area)));
            lines.push(format!("title = {}", compat_toml_quote(&row.title)));
            lines.push(format!(
                "description = {}",
                compat_toml_quote(&row.description)
            ));
            lines.push(format!("priority = {}", compat_toml_quote(&row.priority)));
            lines.push(format!("status = {}", compat_toml_quote(&row.status)));
            lines.push(format!(
                "status_token = {}",
                compat_toml_quote(&row.status_token)
            ));
            lines.push(format!(
                "dependencies = {}",
                compat_toml_string_array(&compat_json_string_array(&row.dependencies_json)?)
            ));
            lines.push(format!(
                "acceptance_criteria = {}",
                compat_toml_string_array(&compat_json_string_array(&row.acceptance_criteria_json)?)
            ));
            lines.push(format!(
                "evidence_refs = {}",
                compat_toml_string_array(&compat_json_string_array(&row.evidence_refs_json)?)
            ));
            lines.push(String::new());
        }
        trim_trailing_blank_lines(&mut lines);
        Ok(lines.join("\n"))
    }

    fn parse_registry_snapshot(&self, kind: &str) -> Result<Option<Value>> {
        self.registry_snapshot(kind)?
            .map(|raw| {
                toml::from_str::<Value>(&raw)
                    .with_context(|| format!("parse {kind} registry snapshot"))
            })
            .transpose()
    }

    pub fn planning_roadmap_rows(&self) -> Result<Vec<RoadmapCompatRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, priority, status, status_token, description, sprint,
                    dependencies_json, acceptance_criteria_json, primary_outputs_json,
                    evidence_refs_json, lacunae_json, claims_json, insight
             FROM roadmap_items
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(RoadmapCompatRow {
                id: row.get(0)?,
                name: row.get(1)?,
                priority: row.get(2)?,
                status: row.get(3)?,
                status_token: row.get(4)?,
                description: row.get(5)?,
                sprint: row.get(6)?,
                dependencies_json: row.get(7)?,
                acceptance_criteria_json: row.get(8)?,
                primary_outputs_json: row.get(9)?,
                evidence_refs_json: row.get(10)?,
                lacunae_json: row.get(11)?,
                claims_json: row.get(12)?,
                insight: row.get(13)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn planning_todo_rows(&self) -> Result<Vec<ActionCompatRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, area, title, description, priority, status, status_token,
                    dependencies_json, acceptance_criteria_json, evidence_refs_json
             FROM todo_items
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ActionCompatRow {
                id: row.get(0)?,
                area: row.get(1)?,
                title: row.get(2)?,
                description: row.get(3)?,
                priority: row.get(4)?,
                status: row.get(5)?,
                status_token: row.get(6)?,
                dependencies_json: row.get(7)?,
                acceptance_criteria_json: row.get(8)?,
                evidence_refs_json: row.get(9)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn planning_next_action_rows(&self) -> Result<Vec<ActionCompatRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, area, title, description, priority, status, status_token,
                    dependencies_json, acceptance_criteria_json, evidence_refs_json
             FROM next_action_items
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ActionCompatRow {
                id: row.get(0)?,
                area: row.get(1)?,
                title: row.get(2)?,
                description: row.get(3)?,
                priority: row.get(4)?,
                status: row.get(5)?,
                status_token: row.get(6)?,
                dependencies_json: row.get(7)?,
                acceptance_criteria_json: row.get(8)?,
                evidence_refs_json: row.get(9)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn requirements_meta_row(&self) -> Result<Option<RequirementsMetaCompatRow>> {
        self.conn
            .query_row(
                "SELECT authoritative, status, status_token, updated, python_recommended,
                        python_allowed, primary_markdown, status_allowlist_json,
                        runtime_stack_allowlist_json, required_module_fields_json,
                        required_gap_fields_json
                 FROM requirements_registry_meta
                 WHERE kind = 'requirements'",
                [],
                |row| {
                    Ok(RequirementsMetaCompatRow {
                        authoritative: row.get::<_, i64>(0)? != 0,
                        status: row.get(1)?,
                        status_token: row.get(2)?,
                        updated: row.get(3)?,
                        python_recommended: row.get(4)?,
                        python_allowed: row.get(5)?,
                        primary_markdown: row.get(6)?,
                        status_allowlist_json: row.get(7)?,
                        runtime_stack_allowlist_json: row.get(8)?,
                        required_module_fields_json: row.get(9)?,
                        required_gap_fields_json: row.get(10)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn requirements_module_rows(&self) -> Result<Vec<RequirementModuleCompatRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, markdown, status, status_token, runtime_stack,
                    requires_modules_json, install_targets_json, verify_targets_json,
                    acceptance_criteria_json
             FROM requirements_modules
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(RequirementModuleCompatRow {
                id: row.get(0)?,
                name: row.get(1)?,
                markdown: row.get(2)?,
                status: row.get(3)?,
                status_token: row.get(4)?,
                runtime_stack: row.get(5)?,
                requires_modules_json: row.get(6)?,
                install_targets_json: row.get(7)?,
                verify_targets_json: row.get(8)?,
                acceptance_criteria_json: row.get(9)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn requirements_coverage_gap_rows(&self) -> Result<Vec<RequirementCoverageGapCompatRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, area, status, status_token, description, proposed_resolution,
                    related_module_ids_json
             FROM requirements_coverage_gaps
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(RequirementCoverageGapCompatRow {
                id: row.get(0)?,
                area: row.get(1)?,
                status: row.get(2)?,
                status_token: row.get(3)?,
                description: row.get(4)?,
                proposed_resolution: row.get(5)?,
                related_module_ids_json: row.get(6)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Insert or replace a notebook session.
    pub fn upsert_notebook_session(&self, row: &NotebookSessionRow<'_>) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO notebook_sessions
             (id, title, description, kernel, status, cell_count, cells_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7, datetime('now'))",
            params![
                row.id,
                row.title,
                row.description,
                row.kernel,
                row.status,
                row.cell_count,
                row.cells_json
            ],
        )?;
        Ok(())
    }

    /// List notebook sessions.
    pub fn list_notebook_sessions(&self) -> Result<Vec<NotebookSessionSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, kernel, status, cell_count FROM notebook_sessions ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(NotebookSessionSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                kernel: row.get(2)?,
                status: row.get(3)?,
                cell_count: row.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // ── Three-layer registry: build methods ─────────────────────────

    /// Create a fresh derived database at the given path.
    /// Deletes any existing file, runs all migrations, and sets WAL mode.
    pub fn build_fresh(db_path: &Path) -> Result<Self> {
        if db_path.exists() {
            fs::remove_file(db_path)
                .with_context(|| format!("remove stale DB {}", db_path.display()))?;
        }
        let store = Self::open(db_path)?;
        store
            .conn
            .pragma_update(None, "journal_mode", "WAL")
            .with_context(|| "set WAL journal mode")?;
        Ok(store)
    }

    /// Record build metadata (builder version, timestamp, source count).
    pub fn record_build_metadata(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO build_metadata (key, value, updated_at)
             VALUES (?1, ?2, datetime('now'))",
            params![key, value],
        )?;
        Ok(())
    }

    /// Ingest bibliography entries from bibliography.toml.
    pub fn ingest_bibliography(&self, toml_text: &str) -> Result<u64> {
        let val: Value = toml::from_str(toml_text)?;
        let mut count = 0u64;
        if let Some(entries) = val.get("entry").and_then(|v| v.as_array()) {
            for entry in entries {
                let id = entry.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if id.is_empty() {
                    continue;
                }
                let title = entry.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let authors = entry
                    .get("authors")
                    .and_then(|v| v.as_str())
                    .or_else(|| entry.get("author").and_then(|v| v.as_str()))
                    .unwrap_or("");
                let year = entry.get("year").and_then(|v| v.as_str()).unwrap_or("");
                let doi = entry.get("doi").and_then(|v| v.as_str()).unwrap_or("");
                let url = entry.get("url").and_then(|v| v.as_str()).unwrap_or("");
                let bib_type = entry
                    .get("bibtex_type")
                    .and_then(|v| v.as_str())
                    .or_else(|| entry.get("type").and_then(|v| v.as_str()))
                    .unwrap_or("");
                let tags = toml_array_to_json_string(entry, "tags");

                self.conn.execute(
                    "INSERT OR REPLACE INTO bibliography
                     (id, title, authors, year, doi, url, bibtex_type, tags_json)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                    params![id, title, authors, year, doi, url, bib_type, tags],
                )?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// Ingest evidence edges from claims_evidence_edges.toml.
    pub fn ingest_evidence_edges(&self, toml_text: &str) -> Result<u64> {
        let val: Value = toml::from_str(toml_text)?;
        let mut count = 0u64;
        if let Some(edges) = val.get("edge").and_then(|v| v.as_array()) {
            for edge in edges {
                let source = edge.get("source").and_then(|v| v.as_str()).unwrap_or("");
                let target = edge.get("target").and_then(|v| v.as_str()).unwrap_or("");
                if source.is_empty() || target.is_empty() {
                    continue;
                }
                let edge_type = edge
                    .get("edge_type")
                    .or_else(|| edge.get("type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("supports");
                let weight = edge.get("weight").and_then(|v| v.as_float()).unwrap_or(1.0);
                let notes = edge.get("notes").and_then(|v| v.as_str()).unwrap_or("");

                self.conn.execute(
                    "INSERT INTO evidence_edges (source_id, target_id, edge_type, weight, notes)
                     VALUES (?1,?2,?3,?4,?5)",
                    params![source, target, edge_type, weight, notes],
                )?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// Ingest lacunae from lacunae.toml.
    pub fn ingest_lacunae(&self, toml_text: &str) -> Result<u64> {
        let val: Value = toml::from_str(toml_text)?;
        let mut count = 0u64;
        if let Some(items) = val.get("lacuna").and_then(|v| v.as_array()) {
            for item in items {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if id.is_empty() {
                    continue;
                }
                let title = item.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let status = item
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("open");
                let domain = item.get("domain").and_then(|v| v.as_str()).unwrap_or("");
                let description = item
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let claim_refs = toml_array_to_json_string(item, "claim_refs");

                self.conn.execute(
                    "INSERT OR REPLACE INTO lacunae
                     (id, title, status, domain, description, claim_refs_json)
                     VALUES (?1,?2,?3,?4,?5,?6)",
                    params![id, title, status, domain, description, claim_refs],
                )?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// Build crossref join tables by parsing claim_refs from insights and experiments.
    pub fn build_crossrefs(&self) -> Result<(u64, u64)> {
        // claim <-> experiment refs
        let mut ce_count = 0u64;
        {
            let mut stmt = self.conn.prepare(
                "SELECT id, claim_refs_json FROM experiments_cp WHERE claim_refs_json != '[]'",
            )?;
            let rows: Vec<(String, String)> = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .filter_map(|r| r.ok())
                .collect();
            for (exp_id, refs_json) in &rows {
                if let Ok(refs) = serde_json::from_str::<Vec<String>>(refs_json) {
                    for claim_id in &refs {
                        self.conn.execute(
                            "INSERT OR IGNORE INTO claim_experiment_refs (claim_id, experiment_id)
                             VALUES (?1, ?2)",
                            params![claim_id, exp_id],
                        )?;
                        ce_count += 1;
                    }
                }
            }
        }

        // claim <-> insight refs
        let mut ci_count = 0u64;
        {
            let mut stmt = self.conn.prepare(
                "SELECT id, claim_refs_json FROM insights WHERE claim_refs_json != '[]'",
            )?;
            let rows: Vec<(String, String)> = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .filter_map(|r| r.ok())
                .collect();
            for (insight_id, refs_json) in &rows {
                if let Ok(refs) = serde_json::from_str::<Vec<String>>(refs_json) {
                    for claim_id in &refs {
                        self.conn.execute(
                            "INSERT OR IGNORE INTO claim_insight_refs (claim_id, insight_id)
                             VALUES (?1, ?2)",
                            params![claim_id, insight_id],
                        )?;
                        ci_count += 1;
                    }
                }
            }
        }

        Ok((ce_count, ci_count))
    }

    /// Search claims via FTS5.
    pub fn search_claims(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, String, String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.statement, c.status, bm25(claims_fts) as rank
             FROM claims_fts fts
             JOIN claims c ON fts.rowid = c.rowid
             WHERE claims_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![query, limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Search insights via FTS5.
    pub fn search_insights(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, String, String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT i.id, i.title, i.status, bm25(insights_fts) as rank
             FROM insights_fts fts
             JOIN insights i ON fts.rowid = i.rowid
             WHERE insights_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![query, limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Search bibliography via FTS5.
    pub fn search_bibliography(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, String, String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT b.id, b.title, b.authors, bm25(bibliography_fts) as rank
             FROM bibliography_fts fts
             JOIN bibliography b ON fts.rowid = b.rowid
             WHERE bibliography_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![query, limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Get a single claim by ID.
    pub fn claim_by_id(&self, id: &str) -> Result<Option<ClaimRecord>> {
        self.conn
            .query_row(
                "SELECT id, statement, status, where_stated, last_verified, formal_proof, status_note, compat_toml_text
                 FROM claims WHERE id = ?1",
                params![id],
                |row| {
                    Ok(ClaimRecord {
                        id: row.get(0)?,
                        statement: row.get(1)?,
                        status: row.get(2)?,
                        where_stated: row.get(3)?,
                        last_verified: row.get(4)?,
                        formal_proof: row.get(5)?,
                        status_note: row.get(6)?,
                        compat_toml_text: row.get::<_, String>(7).unwrap_or_default(),
                    })
                },
            )
            .optional()
            .map_err(|e| anyhow!(e))
    }

    /// List claims with optional status filter.
    pub fn list_claims_filtered(
        &self,
        status: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ClaimRecord>> {
        let mut out = Vec::new();
        if let Some(s) = status {
            let mut stmt = self.conn.prepare(
                "SELECT id, statement, status, where_stated, last_verified, formal_proof, status_note, compat_toml_text
                 FROM claims WHERE status = ?1 ORDER BY id LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![s, limit as i64], |row| {
                Ok(ClaimRecord {
                    id: row.get(0)?,
                    statement: row.get(1)?,
                    status: row.get(2)?,
                    where_stated: row.get(3)?,
                    last_verified: row.get(4)?,
                    formal_proof: row.get(5)?,
                    status_note: row.get(6)?,
                    compat_toml_text: row.get::<_, String>(7).unwrap_or_default(),
                })
            })?;
            for r in rows {
                out.push(r?);
            }
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT id, statement, status, where_stated, last_verified, formal_proof, status_note, compat_toml_text
                 FROM claims ORDER BY id LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit as i64], |row| {
                Ok(ClaimRecord {
                    id: row.get(0)?,
                    statement: row.get(1)?,
                    status: row.get(2)?,
                    where_stated: row.get(3)?,
                    last_verified: row.get(4)?,
                    formal_proof: row.get(5)?,
                    status_note: row.get(6)?,
                    compat_toml_text: row.get::<_, String>(7).unwrap_or_default(),
                })
            })?;
            for r in rows {
                out.push(r?);
            }
        }
        Ok(out)
    }

    /// List experiments with optional status filter.
    pub fn list_experiments_filtered(
        &self,
        status: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ExperimentRecord>> {
        let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<ExperimentRecord> {
            Ok(ExperimentRecord {
                id: row.get(0)?,
                title: row.get(1)?,
                status: row.get(2)?,
                binary: row.get(3)?,
                claim_refs: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                status_note: row.get(5)?,
                compat_toml_text: row.get::<_, String>(6).unwrap_or_default(),
            })
        };
        let mut out = Vec::new();
        if let Some(s) = status {
            let mut stmt = self.conn.prepare(
                "SELECT id, title, status, binary_name, claim_refs_json, status_note, compat_toml_text
                 FROM experiments_cp WHERE status = ?1 ORDER BY id LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![s, limit as i64], map_row)?;
            for r in rows {
                out.push(r?);
            }
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT id, title, status, binary_name, claim_refs_json, status_note, compat_toml_text
                 FROM experiments_cp ORDER BY id LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit as i64], map_row)?;
            for r in rows {
                out.push(r?);
            }
        }
        Ok(out)
    }

    /// Find claims that have no linked experiments or insights.
    pub fn unlinked_claims(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.statement FROM claims c
             WHERE c.id NOT IN (SELECT claim_id FROM claim_experiment_refs)
               AND c.id NOT IN (SELECT claim_id FROM claim_insight_refs)
             ORDER BY c.id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Find dangling crossrefs (references to non-existent claims).
    pub fn dangling_crossrefs(&self) -> Result<Vec<(String, String, String)>> {
        let mut out = Vec::new();
        // Experiment refs to non-existent claims
        {
            let mut stmt = self.conn.prepare(
                "SELECT cer.experiment_id, cer.claim_id, 'experiment'
                 FROM claim_experiment_refs cer
                 WHERE cer.claim_id NOT IN (SELECT id FROM claims)",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?;
            for r in rows {
                out.push(r?);
            }
        }
        // Insight refs to non-existent claims
        {
            let mut stmt = self.conn.prepare(
                "SELECT cir.insight_id, cir.claim_id, 'insight'
                 FROM claim_insight_refs cir
                 WHERE cir.claim_id NOT IN (SELECT id FROM claims)",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?;
            for r in rows {
                out.push(r?);
            }
        }
        Ok(out)
    }
}

/// Internal helper to run narrative FTS queries against an arbitrary connection.
///
/// Exposed as `pub(crate)` so tests can exercise the FTS wiring with an
/// in-memory database without needing to construct a full store instance.
pub(crate) fn search_narratives_on_conn(
    conn: &rusqlite::Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<(String, String, f64)>> {
    let mut stmt = conn.prepare(
        "SELECT rn.id, rn.title, bm25(research_narrative_search) as rank
         FROM research_narrative_search fts
         JOIN research_narratives rn ON fts.rowid = rn.rowid
         WHERE research_narrative_search MATCH ?1
         ORDER BY rank
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![query, limit as i64], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)?,
        ))
    })?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

// SQL table-clearing and ranked-value upsert/load helpers (clear_tables,
// clear_control_plane_tables, clear_external_source_tables,
// insert_ranked_values, replace_ranked_values, load_ranked_values) live
// in the `table_ops` submodule. Items are pub(crate).
mod table_ops;
use table_ops::{
    clear_control_plane_tables, clear_external_source_tables, clear_tables, insert_ranked_values,
    load_ranked_values, replace_ranked_values,
};

// SQL fingerprint/snapshot writers, count/backend-health/string-vec
// loaders, and the repo-relative path helper (write_fingerprint,
// write_registry_snapshot, scalar_count, query_count_summaries,
// query_backend_health, load_string_vec, load_record_sources,
// to_repo_rel) live in the `sql_helpers` submodule. Items are
// pub(crate).
mod sql_helpers;
use sql_helpers::{
    load_record_sources, load_string_vec, query_backend_health, query_count_summaries,
    scalar_count, to_repo_rel, write_fingerprint, write_registry_snapshot,
};

// Registry TOML loaders for the four primary ingest record families
// (load_artifacts -> Vec<ArtifactRecord>, load_documents ->
// Vec<DocumentRecord>, load_lane_assignments -> Vec<LaneAssignment>,
// build_mirror_observations -> Vec<MirrorObservationRecord>) live in
// the `loaders` submodule. Items are pub(crate).
mod loaders;
use loaders::{build_mirror_observations, load_artifacts, load_documents, load_lane_assignments};

// Pure TOML/JSON utility helpers for the ingest and compat-export paths
// (load_toml_value, load_toml_text, load_text, load_registry_table_toml,
// render_toml_table, compat_toml_quote, compat_json_string_array,
// compat_toml_string_array, compat_root_table, compat_child_table,
// compat_table_string, compat_table_bool, compat_table_array,
// trim_trailing_blank_lines, string_field, optional_string_field,
// string_array_field, bool_field, optional_integer_field, host_for_url,
// join_refs, toml_array_to_json_string) live in the `toml_helpers`
// submodule. Items are pub(crate).
mod toml_helpers;
use toml_helpers::{
    bool_field, compat_child_table, compat_json_string_array, compat_root_table,
    compat_table_array, compat_table_bool, compat_table_string, compat_toml_quote,
    compat_toml_string_array, host_for_url, join_refs, load_registry_table_toml, load_text,
    load_toml_text, load_toml_value, optional_integer_field, optional_string_field,
    render_toml_table, string_array_field, string_field, toml_array_to_json_string,
    trim_trailing_blank_lines,
};

fn load_external_source_contracts_from_registry(
    raw: &str,
) -> Result<(
    ExternalSourceContractsMeta,
    Vec<ExternalSourceContractRecord>,
)> {
    let value: Value = toml::from_str(raw).context("parse external source contracts registry")?;
    let meta_table = value
        .get("external_sources")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let meta = ExternalSourceContractsMeta {
        updated: string_field(&meta_table, "updated"),
        authoritative: bool_field(&meta_table, "authoritative"),
        policy_version: string_field(&meta_table, "policy_version"),
    };
    let rows = value
        .get("source")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| row.as_table().cloned())
        .map(|table| ExternalSourceContractRecord {
            id: string_field(&table, "id"),
            path_glob: string_field(&table, "path_glob"),
            canonical_url: string_field(&table, "canonical_url"),
            mirror_urls: string_array_field(&table, "mirror_urls"),
            access_class: string_field(&table, "access_class"),
            status: string_field(&table, "status"),
            retrieval_method: string_field(&table, "retrieval_method"),
            attempt_deadline_utc: string_field(&table, "attempt_deadline_utc"),
            resolution_deadline_utc: string_field(&table, "resolution_deadline_utc"),
            blocker_note: string_field(&table, "blocker_note"),
            evidence_refs: string_array_field(&table, "evidence_refs"),
            manual_manifest_refs: string_array_field(&table, "manual_manifest_refs"),
            blocked_action_plan: string_array_field(&table, "blocked_action_plan"),
            scientific_validator_refs: string_array_field(&table, "scientific_validator_refs"),
        })
        .collect::<Vec<_>>();
    Ok((meta, rows))
}

fn load_external_source_dossiers_from_registry(
    raw: &str,
) -> Result<(ExternalSourceDossiersMeta, Vec<ExternalSourceDossierRecord>)> {
    let value: Value = toml::from_str(raw).context("parse external source dossiers registry")?;
    let meta_table = value
        .get("external_sources")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let rows = value
        .get("document")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| row.as_table().cloned())
        .map(|table| {
            let id = string_field(&table, "id");
            let slug = string_field(&table, "slug");
            let source_markdown = optional_string_field(&table, "source_markdown")
                .unwrap_or_else(|| default_external_source_markdown_path(&id, &slug));
            ExternalSourceDossierRecord {
                id,
                source_markdown,
                slug,
                title: string_field(&table, "title"),
                status_token: string_field(&table, "status_token"),
                content_kind: string_field(&table, "content_kind"),
                authority_level: string_field(&table, "authority_level"),
                verification_level: string_field(&table, "verification_level"),
                operational_role: string_field(&table, "operational_role"),
                source_lineage_summary: string_field(&table, "source_lineage_summary"),
                truth_surfaces: string_array_field(&table, "truth_surfaces"),
                artifact_contract_paths: string_array_field(&table, "artifact_contract_paths"),
                has_full_transcript: bool_field(&table, "has_full_transcript"),
                claim_refs: string_array_field(&table, "claim_refs"),
                url_refs: string_array_field(&table, "url_refs"),
                path_refs: string_array_field(&table, "path_refs"),
                line_count: optional_integer_field(&table, "line_count").unwrap_or_default()
                    as usize,
                notes: string_field(&table, "notes"),
                body_markdown: string_field(&table, "body_markdown"),
            }
        })
        .collect::<Vec<_>>();
    let meta = ExternalSourceDossiersMeta {
        updated: string_field(&meta_table, "updated"),
        authoritative: bool_field(&meta_table, "authoritative"),
        source_markdown_glob: string_field(&meta_table, "source_markdown_glob"),
        document_count: optional_integer_field(&meta_table, "document_count")
            .unwrap_or(rows.len() as i64) as usize,
    };
    Ok((meta, rows))
}

fn render_external_source_contracts_registry(
    meta: &ExternalSourceContractsMeta,
    rows: &[ExternalSourceContractRecord],
) -> String {
    let mut lines = external_sources_compat_toml_export_header("source_contracts");
    lines.push(String::new());
    lines.push("[external_sources]".to_string());
    lines.push(format!("updated = {:?}", meta.updated));
    lines.push(format!("authoritative = {}", bool_toml(meta.authoritative)));
    lines.push(format!("policy_version = {:?}", meta.policy_version));
    lines.push(String::new());
    for row in rows {
        lines.push("[[source]]".to_string());
        lines.push(format!("id = {:?}", row.id));
        lines.push(format!("path_glob = {:?}", row.path_glob));
        lines.push(format!("canonical_url = {:?}", row.canonical_url));
        render_string_array_lines(&mut lines, "mirror_urls", &row.mirror_urls);
        lines.push(format!("access_class = {:?}", row.access_class));
        lines.push(format!("status = {:?}", row.status));
        lines.push(format!("retrieval_method = {:?}", row.retrieval_method));
        lines.push(format!(
            "attempt_deadline_utc = {:?}",
            row.attempt_deadline_utc
        ));
        lines.push(format!(
            "resolution_deadline_utc = {:?}",
            row.resolution_deadline_utc
        ));
        lines.push(format!("blocker_note = {:?}", row.blocker_note));
        render_string_array_lines(&mut lines, "evidence_refs", &row.evidence_refs);
        render_string_array_lines(
            &mut lines,
            "manual_manifest_refs",
            &row.manual_manifest_refs,
        );
        render_string_array_lines(
            &mut lines,
            "scientific_validator_refs",
            &row.scientific_validator_refs,
        );
        render_string_array_lines(&mut lines, "blocked_action_plan", &row.blocked_action_plan);
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_external_source_dossiers_registry(
    meta: &ExternalSourceDossiersMeta,
    rows: &[ExternalSourceDossierRecord],
) -> String {
    let mut lines = external_sources_compat_toml_export_header("source_dossiers");
    lines.push(String::new());
    lines.push("[external_sources]".to_string());
    lines.push(format!("updated = {:?}", meta.updated));
    lines.push(format!("authoritative = {}", bool_toml(meta.authoritative)));
    lines.push(format!(
        "source_markdown_glob = {:?}",
        meta.source_markdown_glob
    ));
    lines.push(format!("document_count = {}", rows.len()));
    lines.push(String::new());
    for row in rows {
        lines.push("[[document]]".to_string());
        lines.push(format!("id = {:?}", row.id));
        lines.push(format!("source_markdown = {:?}", row.source_markdown));
        lines.push(format!("slug = {:?}", row.slug));
        lines.push(format!("title = {:?}", row.title));
        lines.push(format!("status_token = {:?}", row.status_token));
        lines.push(format!("content_kind = {:?}", row.content_kind));
        lines.push(format!("authority_level = {:?}", row.authority_level));
        lines.push(format!("verification_level = {:?}", row.verification_level));
        lines.push(format!("operational_role = {:?}", row.operational_role));
        lines.push(format!(
            "source_lineage_summary = {:?}",
            row.source_lineage_summary
        ));
        render_string_array_lines(&mut lines, "truth_surfaces", &row.truth_surfaces);
        render_string_array_lines(
            &mut lines,
            "artifact_contract_paths",
            &row.artifact_contract_paths,
        );
        lines.push(format!(
            "has_full_transcript = {}",
            bool_toml(row.has_full_transcript)
        ));
        render_string_array_lines(&mut lines, "claim_refs", &row.claim_refs);
        render_string_array_lines(&mut lines, "url_refs", &row.url_refs);
        render_string_array_lines(&mut lines, "path_refs", &row.path_refs);
        lines.push(format!("line_count = {}", row.line_count));
        lines.push(format!("notes = {:?}", row.notes));
        lines.push(format!(
            "body_markdown = {}",
            render_toml_multiline(&row.body_markdown)
        ));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_external_source_dossier_markdown(row: &ExternalSourceDossierRecord) -> String {
    let mut lines = external_sources_markdown_export_header(&row.id);
    lines.push(row.body_markdown.trim_end().to_string());
    lines.join("\n")
}

fn render_string_array_lines(lines: &mut Vec<String>, key: &str, values: &[String]) {
    if values.is_empty() {
        lines.push(format!("{key} = []"));
        return;
    }
    lines.push(format!("{key} = ["));
    for value in values {
        lines.push(format!("  {:?},", value));
    }
    lines.push("]".to_string());
}

fn render_toml_multiline(body: &str) -> String {
    let sanitized = body.replace("'''", "'''\"\"\"'''");
    format!("'''\n{}\n'''", sanitized.trim_end())
}

fn default_external_source_markdown_path(id: &str, slug: &str) -> String {
    let stem = if !slug.trim().is_empty() {
        slug.trim().to_ascii_uppercase()
    } else if !id.trim().is_empty() {
        id.trim().to_ascii_uppercase()
    } else {
        "UNNAMED_EXTERNAL_SOURCE".to_string()
    };
    format!("docs/external_sources/{stem}.md")
}

fn collect_rows<T>(
    rows: rusqlite::MappedRows<'_, impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>>,
) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn load_claims_from_registry(raw: &str) -> Result<Vec<ClaimRecord>> {
    let value: Value = toml::from_str(raw).context("parse claims registry")?;
    let claims = value
        .get("claim")
        .and_then(Value::as_array)
        .context("claim array missing")?;
    let mut out = Vec::new();
    for claim in claims {
        let table = claim.as_table().context("claim row must be table")?;
        let mut record = ClaimRecord {
            id: string_field(table, "id"),
            statement: string_field(table, "statement"),
            status: string_field(table, "status"),
            where_stated: string_field(table, "where_stated"),
            last_verified: string_field(table, "last_verified"),
            formal_proof: optional_string_field(table, "formal_proof"),
            status_note: optional_string_field(table, "status_note"),
            compat_toml_text: String::new(),
        };
        normalize_claim_record(&mut record)?;
        out.push(record);
    }
    Ok(out)
}

fn load_insights_from_registry(raw: &str) -> Result<Vec<InsightRecord>> {
    let value: Value = toml::from_str(raw).context("parse insights registry")?;
    let insights = value
        .get("insight")
        .and_then(Value::as_array)
        .context("insight array missing")?;
    let mut out = Vec::new();
    for insight in insights {
        let table = insight.as_table().context("insight row must be table")?;
        let title = optional_string_field(table, "title")
            .or_else(|| optional_string_field(table, "insight"))
            .unwrap_or_else(|| string_field(table, "id"));
        let raw_status = optional_string_field(table, "status");
        let status = raw_status
            .as_deref()
            .map(normalize_insight_status)
            .unwrap_or("unknown")
            .to_string();
        let mut claim_refs = string_array_field(table, "claims");
        claim_refs.extend(string_array_field(table, "related_claims"));
        claim_refs.sort();
        claim_refs.dedup();
        out.push(InsightRecord {
            id: string_field(table, "id"),
            title,
            status,
            claim_refs,
            status_note: optional_string_field(table, "status_note"),
            compat_toml_text: render_normalized_insight_compat_toml(table, raw_status.as_deref())?,
        });
    }
    Ok(out)
}

fn load_experiments_from_registry(raw: &str) -> Result<Vec<ExperimentRecord>> {
    let value: Value = toml::from_str(raw).context("parse experiments registry")?;
    let experiments = value
        .get("experiment")
        .and_then(Value::as_array)
        .context("experiment array missing")?;
    let mut out = Vec::new();
    for experiment in experiments {
        let table = experiment
            .as_table()
            .context("experiment row must be table")?;
        let title =
            optional_string_field(table, "title").unwrap_or_else(|| string_field(table, "id"));
        let status = optional_string_field(table, "status")
            .or_else(|| optional_string_field(table, "status_token"))
            .unwrap_or_else(|| "unknown".to_string());
        let mut claim_refs = string_array_field(table, "claims");
        claim_refs.extend(string_array_field(table, "claim_refs"));
        claim_refs.sort();
        claim_refs.dedup();
        out.push(ExperimentRecord {
            id: string_field(table, "id"),
            title,
            status,
            binary: optional_string_field(table, "binary"),
            claim_refs,
            status_note: optional_string_field(table, "status_note"),
            compat_toml_text: render_toml_table(table)?,
        });
    }
    Ok(out)
}

fn load_binaries_from_registry(raw: &str) -> Result<Vec<BinaryRecord>> {
    let value: Value = toml::from_str(raw).context("parse binaries registry")?;
    let binaries = value
        .get("binary")
        .and_then(Value::as_array)
        .context("binary array missing")?;
    let mut out = Vec::new();
    for binary in binaries {
        let table = binary.as_table().context("binary row must be table")?;
        out.push(BinaryRecord {
            name: string_field(table, "name"),
            crate_name: string_field(table, "crate"),
            description: string_field(table, "description"),
            experiment: optional_string_field(table, "experiment"),
            source: "registry".to_string(),
        });
    }
    Ok(out)
}

#[derive(Debug, Default, Deserialize)]
struct WorkspaceManifest {
    #[serde(default)]
    workspace: Option<WorkspaceSection>,
    #[serde(default)]
    package: Option<PackageSection>,
    #[serde(default, rename = "bin")]
    bins: Vec<CargoBinEntry>,
}

#[derive(Debug, Default, Deserialize)]
struct WorkspaceSection {
    #[serde(default)]
    members: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
struct PackageSection {
    #[serde(default)]
    name: String,
}

#[derive(Debug, Default, Deserialize)]
struct CargoBinEntry {
    #[serde(default)]
    name: String,
}

fn merge_workspace_binaries(
    repo_root: &Path,
    registry_binaries: &[BinaryRecord],
) -> Result<Vec<BinaryRecord>> {
    let workspace_bins = load_workspace_binary_records(repo_root)?;
    let mut merged = BTreeMap::new();

    for binary in workspace_bins {
        merged.insert(binary.name.clone(), binary);
    }

    for binary in registry_binaries {
        let Some(entry) = merged.get_mut(&binary.name) else {
            continue;
        };
        if !binary.crate_name.trim().is_empty() {
            entry.crate_name = binary.crate_name.clone();
        }
        if !binary.description.trim().is_empty() {
            entry.description = binary.description.clone();
        }
        if binary.experiment.is_some() {
            entry.experiment = binary.experiment.clone();
        }
        entry.source = if entry.source == "workspace_manifest" {
            "registry+workspace_manifest".to_string()
        } else {
            binary.source.clone()
        };
    }

    Ok(merged.into_values().collect())
}

fn load_workspace_binary_records(repo_root: &Path) -> Result<Vec<BinaryRecord>> {
    match load_workspace_binary_records_via_cargo_metadata(repo_root) {
        Ok(records) => return Ok(records),
        Err(err) => {
            eprintln!(
                "WARNING: cargo metadata binary inventory failed (falling back to manifest walk): {err}"
            );
        }
    }
    load_workspace_binary_records_from_manifests(repo_root)
}

fn load_workspace_binary_records_via_cargo_metadata(repo_root: &Path) -> Result<Vec<BinaryRecord>> {
    #[derive(Deserialize)]
    struct MetadataTarget {
        name: String,
        #[serde(default)]
        kind: Vec<String>,
    }

    #[derive(Deserialize)]
    struct MetadataPackage {
        name: String,
        #[serde(default)]
        targets: Vec<MetadataTarget>,
    }

    #[derive(Deserialize)]
    struct MetadataRoot {
        #[serde(default)]
        packages: Vec<MetadataPackage>,
    }

    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("run cargo metadata from {}", repo_root.display()))?;
    if !output.status.success() {
        bail!(
            "cargo metadata failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let metadata: MetadataRoot = serde_json::from_slice(&output.stdout)
        .context("parse cargo metadata JSON for workspace binaries")?;
    let mut out = BTreeMap::new();
    for package in metadata.packages {
        for target in package.targets {
            if !target.kind.iter().any(|kind| kind == "bin") {
                continue;
            }
            let bin_name = target.name.trim();
            if bin_name.is_empty() {
                continue;
            }
            out.entry(bin_name.to_string()).or_insert(BinaryRecord {
                name: bin_name.to_string(),
                crate_name: package.name.clone(),
                description: format!(
                    "Workspace binary discovered from cargo metadata in crate {}; consult crate source for authoritative behavior.",
                    package.name
                ),
                experiment: None,
                source: "cargo_metadata".to_string(),
            });
        }
    }
    Ok(out.into_values().collect())
}

fn load_workspace_binary_records_from_manifests(repo_root: &Path) -> Result<Vec<BinaryRecord>> {
    let root_manifest_path = repo_root.join("Cargo.toml");
    let root_manifest: WorkspaceManifest = toml::from_str(&load_toml_text(&root_manifest_path)?)
        .with_context(|| format!("parse {}", root_manifest_path.display()))?;
    let mut out = BTreeMap::new();
    let mut seen = BTreeSet::new();

    for member in root_manifest
        .workspace
        .as_ref()
        .map(|workspace| workspace.members.as_slice())
        .unwrap_or(&[])
    {
        let member_manifest_path = member_manifest_path(repo_root, member);
        if !member_manifest_path.exists() {
            bail!(
                "workspace member manifest missing for {}: {}",
                member,
                member_manifest_path.display()
            );
        }
        if !seen.insert(member_manifest_path.clone()) {
            continue;
        }
        let member_manifest: WorkspaceManifest =
            toml::from_str(&load_toml_text(&member_manifest_path)?)
                .with_context(|| format!("parse {}", member_manifest_path.display()))?;
        let crate_name = member_manifest
            .package
            .as_ref()
            .map(|package| package.name.trim().to_string())
            .filter(|value| !value.is_empty())
            .with_context(|| {
                format!("missing package.name in {}", member_manifest_path.display())
            })?;

        for bin in member_manifest.bins {
            let bin_name = bin.name.trim();
            if bin_name.is_empty() {
                continue;
            }
            out.entry(bin_name.to_string()).or_insert(BinaryRecord {
                name: bin_name.to_string(),
                crate_name: crate_name.clone(),
                description: format!(
                    "Workspace binary discovered from {}; consult crate source for authoritative behavior.",
                    to_repo_rel(repo_root, &member_manifest_path)
                ),
                experiment: None,
                source: "workspace_manifest".to_string(),
            });
        }
    }

    Ok(out.into_values().collect())
}

fn member_manifest_path(repo_root: &Path, member: &str) -> PathBuf {
    let member_path = repo_root.join(member);
    if member_path
        .file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("Cargo.toml"))
        .unwrap_or(false)
    {
        member_path
    } else {
        member_path.join("Cargo.toml")
    }
}

fn load_proof_inventory(proofs_project_path: &Path) -> Result<ProofInventory> {
    let raw = load_text(proofs_project_path)?;
    let mut inventory = ProofInventory {
        project_raw: raw.clone(),
        ..ProofInventory::default()
    };
    for line in raw.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("verified/") || !trimmed.ends_with(".v") {
            continue;
        }
        let path = Utf8PathBuf::from(format!("proofs/{trimmed}"));
        let stem = Path::new(trimmed)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or(trimmed)
            .to_string();
        let entry = ProofInventoryEntry {
            stem: stem.clone(),
            path,
        };
        if let Some(claim_id) = normalized_claim_id_from_theorem_stem(&stem) {
            inventory
                .verified_by_claim_id
                .entry(claim_id)
                .or_default()
                .push(entry.clone());
        }
        inventory.verified_entries.push(entry);
    }
    Ok(inventory)
}

fn load_theorems_from_inventory(
    repo_root: &Path,
    proof_inventory: &ProofInventory,
    claims: &[ClaimRecord],
) -> Result<Vec<TheoremRecord>> {
    let mut out = Vec::new();
    for entry in &proof_inventory.verified_entries {
        let linked_claim_ids = link_claims_for_proof(&entry.path, &entry.stem, claims);
        let normalized_claim_id = normalized_claim_id_from_theorem_stem(&entry.stem);
        let title = claims
            .iter()
            .find(|claim| {
                claim.id == entry.stem || normalized_claim_id.as_deref() == Some(&claim.id)
            })
            .map(|claim| claim.statement.clone())
            .unwrap_or_else(|| entry.stem.replace('_', " "));
        if !repo_root.join(entry.path.as_str()).exists() {
            continue;
        }
        out.push(TheoremRecord {
            id: entry.stem.clone(),
            title,
            proof_path: entry.path.clone(),
            status: "kernel_checked".to_string(),
            linked_claim_ids,
            source: "_RocqProject".to_string(),
        });
    }
    Ok(out)
}

fn normalize_claims_against_proof_inventory(
    repo_root: &Path,
    claims: &mut [ClaimRecord],
    proof_inventory: &ProofInventory,
) -> Result<()> {
    for claim in claims {
        normalize_claim_record(claim)?;
        let canonical_formal_proof =
            canonical_formal_proof_for_claim(repo_root, claim, proof_inventory);
        if claim.formal_proof != canonical_formal_proof {
            claim.formal_proof = canonical_formal_proof;
            claim.compat_toml_text = render_normalized_claim_compat_toml(claim)?;
        }
    }
    Ok(())
}

fn normalize_claim_record(claim: &mut ClaimRecord) -> Result<()> {
    let (canonical_status, legacy_status_note) = normalize_claim_status(&claim.status);
    claim.status = canonical_status;
    claim.status_note = merge_status_note(claim.status_note.take(), legacy_status_note);
    claim.compat_toml_text = render_normalized_claim_compat_toml(claim)?;
    Ok(())
}

fn normalize_claim_status(raw: &str) -> (String, Option<String>) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return (String::new(), None);
    }
    if let Some(canonical) = match_case_insensitive(trimmed, CANONICAL_CLAIM_STATUSES) {
        return (canonical.to_string(), None);
    }
    if let Some(paren_idx) = trimmed.find('(') {
        let base = trimmed[..paren_idx].trim();
        if let Some(canonical) = match_case_insensitive(base, CANONICAL_CLAIM_STATUSES) {
            return (canonical.to_string(), Some(trimmed.to_string()));
        }
    }
    match trimmed {
        "Open" | "Pending" | "Active" | "Proposed" | "Deferred" | "Speculative" => {
            ("Provisional".to_string(), Some(trimmed.to_string()))
        }
        "Conjecture" => ("Theoretical".to_string(), Some(trimmed.to_string())),
        "Falsified" => ("Refuted".to_string(), Some(trimmed.to_string())),
        "Closed/Verified" => ("Verified".to_string(), Some(trimmed.to_string())),
        "Closed/Falsified" => ("Closed/Refuted".to_string(), Some(trimmed.to_string())),
        "Closed/Methodology-Mismatch" => (
            "Closed/Methodology-Insufficient".to_string(),
            Some(trimmed.to_string()),
        ),
        _ => (trimmed.to_string(), None),
    }
}

fn normalize_insight_status(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(canonical) = match_case_insensitive(trimmed, CANONICAL_INSIGHT_STATUSES) {
        return canonical;
    }
    match trimmed {
        "Active" | "Proposed" | "Speculative" => "open",
        "Verified" => "verified",
        "Superseded" => "superseded",
        "Partial" => "partial",
        _ => trimmed,
    }
}

fn match_case_insensitive<'a>(raw: &str, allowed: &'a [&'a str]) -> Option<&'a str> {
    allowed
        .iter()
        .copied()
        .find(|candidate| candidate.eq_ignore_ascii_case(raw))
}

fn merge_status_note(existing: Option<String>, legacy_status: Option<String>) -> Option<String> {
    match (existing, legacy_status) {
        (existing, None) => existing,
        (None, Some(raw)) => Some(format!("Legacy status token: {raw}")),
        (Some(existing), Some(raw)) => {
            let legacy_note = format!("Legacy status token: {raw}");
            if existing.contains(&legacy_note) || existing.contains(&raw) {
                Some(existing)
            } else {
                Some(format!("{existing} | {legacy_note}"))
            }
        }
    }
}

fn canonical_formal_proof_for_claim(
    repo_root: &Path,
    claim: &ClaimRecord,
    proof_inventory: &ProofInventory,
) -> Option<String> {
    if let Some(formal_proof) = claim.formal_proof.as_deref()
        && !formal_proof.trim().is_empty()
        && repo_root.join(formal_proof).exists()
    {
        return Some(formal_proof.trim().to_string());
    }

    let mut referenced_paths = extract_proof_paths(&claim.where_stated);
    if let Some(status_note) = &claim.status_note {
        referenced_paths.extend(extract_proof_paths(status_note));
    }
    if let Some(formal_proof) = claim.formal_proof.as_deref() {
        referenced_paths.extend(extract_proof_paths(formal_proof));
    }
    referenced_paths.retain(|path| repo_root.join(path).exists());
    referenced_paths.sort();
    referenced_paths.dedup();

    if let Some(primary_verified) =
        preferred_primary_verified_proof_for_claim(claim, proof_inventory)
    {
        return Some(primary_verified);
    }

    if referenced_paths.len() == 1 {
        return referenced_paths.into_iter().next();
    }

    None
}

fn preferred_primary_verified_proof_for_claim(
    claim: &ClaimRecord,
    proof_inventory: &ProofInventory,
) -> Option<String> {
    let claim_prefix = claim.id.strip_prefix("C-").unwrap_or(&claim.id);
    let mut candidates = proof_inventory
        .verified_by_claim_id
        .get(&claim.id)
        .cloned()
        .unwrap_or_default();
    candidates.sort_by(|lhs, rhs| {
        proof_entry_priority(claim_prefix, lhs)
            .cmp(&proof_entry_priority(claim_prefix, rhs))
            .reverse()
            .then_with(|| lhs.path.as_str().cmp(rhs.path.as_str()))
    });
    candidates
        .first()
        .map(|entry| entry.path.as_str().to_string())
}

fn proof_entry_priority(claim_prefix: &str, entry: &ProofInventoryEntry) -> (u8, usize) {
    let suffix = entry.stem.strip_prefix('C').unwrap_or(&entry.stem);
    let suffix = suffix.strip_prefix(claim_prefix).unwrap_or(suffix);
    let primary_rank = if suffix.starts_with('_') || suffix.is_empty() {
        2
    } else {
        1
    };
    (primary_rank, usize::MAX - entry.stem.len())
}

fn render_normalized_claim_compat_toml(row: &ClaimRecord) -> Result<String> {
    let mut table = if row.compat_toml_text.trim().is_empty() {
        let mut table = toml::map::Map::new();
        table.insert("id".to_string(), Value::String(row.id.clone()));
        table.insert(
            "statement".to_string(),
            Value::String(row.statement.clone()),
        );
        table.insert("status".to_string(), Value::String(row.status.clone()));
        table.insert(
            "where_stated".to_string(),
            Value::String(row.where_stated.clone()),
        );
        table.insert(
            "last_verified".to_string(),
            Value::String(row.last_verified.clone()),
        );
        if let Some(status_note) = &row.status_note {
            table.insert(
                "status_note".to_string(),
                Value::String(status_note.clone()),
            );
        }
        table
    } else {
        toml::from_str::<toml::map::Map<String, Value>>(&row.compat_toml_text)
            .context("parse normalized claim compat row")?
    };
    table.insert("status".to_string(), Value::String(row.status.clone()));
    match &row.status_note {
        Some(status_note) if !status_note.trim().is_empty() => {
            table.insert(
                "status_note".to_string(),
                Value::String(status_note.clone()),
            );
        }
        _ => {
            table.remove("status_note");
        }
    }
    match &row.formal_proof {
        Some(formal_proof) if !formal_proof.trim().is_empty() => {
            table.insert(
                "formal_proof".to_string(),
                Value::String(formal_proof.clone()),
            );
        }
        _ => {
            table.remove("formal_proof");
        }
    }
    render_toml_table(&table)
}

fn render_normalized_insight_compat_toml(
    table: &toml::map::Map<String, Value>,
    raw_status: Option<&str>,
) -> Result<String> {
    let mut table = table.clone();
    match raw_status.map(normalize_insight_status) {
        Some(status) if !status.trim().is_empty() => {
            table.insert("status".to_string(), Value::String(status.to_string()));
        }
        _ => {
            table.remove("status");
        }
    }
    render_toml_table(&table)
}

fn extract_proof_paths(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = text;
    while let Some(start) = cursor.find("proofs/") {
        let candidate = &cursor[start..];
        let mut end = 0usize;
        for ch in candidate.chars() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '/' | '_' | '-' | '.') {
                end += ch.len_utf8();
            } else {
                break;
            }
        }
        let path = candidate[..end].trim_matches('`').trim_matches('"').trim();
        if path.ends_with(".v") {
            out.push(path.to_string());
        }
        cursor = &candidate[end..];
    }
    out
}

fn link_claims_for_proof(
    proof_path: &Utf8PathBuf,
    stem: &str,
    claims: &[ClaimRecord],
) -> Vec<String> {
    let proof_path_str = proof_path.as_str();
    let normalized_claim_id = normalized_claim_id_from_theorem_stem(stem);
    let mut out = Vec::new();
    for claim in claims {
        let matches = claim.id == stem
            || normalized_claim_id.as_deref() == Some(&claim.id)
            || claim
                .formal_proof
                .as_deref()
                .map(|path| path.trim() == proof_path_str)
                .unwrap_or(false)
            || claim.where_stated.contains(proof_path_str)
            || claim
                .status_note
                .as_deref()
                .map(|note| note.contains(proof_path_str))
                .unwrap_or(false);
        if matches {
            out.push(claim.id.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

fn normalized_claim_id_from_theorem_stem(stem: &str) -> Option<String> {
    let suffix = stem.strip_prefix('C')?;
    let digits = suffix
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    Some(format!("C-{digits}"))
}

fn render_theorem_markdown(source_label: &str, theorems: &[TheoremRecord]) -> String {
    let mut lines = compat_markdown_export_header(source_label);
    lines.extend([
        "# Theorems".to_string(),
        String::new(),
        format!(
            "This file is generated from the canonical SQLite control plane and currently indexes {} Rocq proof files.",
            theorems.len()
        ),
        String::new(),
        "| Theorem | Proof File | Status | Linked Claims |".to_string(),
        "|---|---|---|---|".to_string(),
    ]);
    for theorem in theorems {
        let claims = if theorem.linked_claim_ids.is_empty() {
            "-".to_string()
        } else {
            theorem.linked_claim_ids.join(", ")
        };
        lines.push(format!(
            "| `{}` | `{}` | {} | {} |",
            theorem.id, theorem.proof_path, theorem.status, claims
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}

// Registry compat-export rendering helpers (render_claims_registry,
// render_insights_registry, render_experiments_registry,
// rebuild_experiments_header_toml, experiment_row_table/flag/has_seed,
// render_array_of_tables_registry, render_claim_row, render_insight_row,
// render_experiment_row, splice_compat_toml_overrides,
// render_binaries_registry, compat_*_export_header,
// external_sources_*_export_header, bool_toml, write_text) live in the
// `compat_render` submodule. All items are pub(crate) and brought back
// into lib.rs scope via plain use statements.
mod compat_render;
use compat_render::{
    bool_toml, compat_markdown_export_header, external_sources_compat_toml_export_header,
    external_sources_markdown_export_header, render_binaries_registry, render_claims_registry,
    render_experiments_registry, render_insights_registry, write_text,
};

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{Connection, params};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn search_narratives_fts_works_with_in_memory_db() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute(
            "CREATE TABLE research_narratives (
                 id    TEXT PRIMARY KEY,
                 title TEXT NOT NULL,
                 body  TEXT NOT NULL
             )",
            [],
        )
        .expect("create research_narratives table");
        conn.execute(
            "CREATE VIRTUAL TABLE research_narrative_search
             USING fts5(title, body, content='research_narratives', content_rowid='rowid')",
            [],
        )
        .expect("create research_narrative_search FTS table");
        conn.execute(
            "INSERT INTO research_narratives (id, title, body) VALUES (?1, ?2, ?3)",
            params![
                "n1",
                "SQLite narrative",
                "This narrative talks about sqlite and databases."
            ],
        )
        .expect("insert narrative n1");
        conn.execute(
            "INSERT INTO research_narratives (id, title, body) VALUES (?1, ?2, ?3)",
            params![
                "n2",
                "Unrelated narrative",
                "This one is about something else entirely."
            ],
        )
        .expect("insert narrative n2");
        conn.execute(
            "INSERT INTO research_narrative_search (rowid, title, body)
             SELECT rowid, title, body FROM research_narratives",
            [],
        )
        .expect("populate FTS index");
        let results = search_narratives_on_conn(&conn, "sqlite", 10)
            .expect("search_narratives_on_conn failed");
        assert_eq!(results.len(), 1, "expected exactly one FTS match");
        assert_eq!(results[0].0, "n1");
        assert_eq!(results[0].1, "SQLite narrative");
    }

    struct TestWorkspace {
        root: PathBuf,
        claims: PathBuf,
        insights: PathBuf,
        experiments: PathBuf,
        binaries: PathBuf,
        rocq_project: PathBuf,
        db: PathBuf,
    }

    impl Drop for TestWorkspace {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn control_plane_compat_text_renders_from_db_backed_rows() -> Result<()> {
        let fixture = make_test_workspace("compat_text")?;
        let mut store = ProvenanceStore::open(&fixture.db)?;
        store.reindex_control_plane_from_registries(
            &fixture.root,
            &fixture.claims,
            &fixture.insights,
            &fixture.experiments,
            &fixture.binaries,
            &fixture.rocq_project,
        )?;

        let claims = store.control_plane_compat_text(ControlPlaneCompatKind::Claims)?;
        let experiments = store.control_plane_compat_text(ControlPlaneCompatKind::Experiments)?;
        let binaries = store.control_plane_compat_text(ControlPlaneCompatKind::Binaries)?;

        assert!(claims.contains("id = \"C-001\""));
        assert!(experiments.contains("experiment_count = 1"));
        assert!(experiments.contains("id = \"E-001\""));
        assert!(binaries.contains("name = \"mini-bin\""));
        Ok(())
    }

    #[test]
    fn replace_control_plane_experiments_from_registry_text_replaces_rows() -> Result<()> {
        let fixture = make_test_workspace("replace_experiments")?;
        let mut store = ProvenanceStore::open(&fixture.db)?;
        store.reindex_control_plane_from_registries(
            &fixture.root,
            &fixture.claims,
            &fixture.insights,
            &fixture.experiments,
            &fixture.binaries,
            &fixture.rocq_project,
        )?;

        let replacement = r#"
[experiments]
authoritative = true
status_allowlist = ["active", "planned", "blocked", "deprecated"]

[[experiment]]
id = "E-002"
title = "Replacement experiment"
status = "active"
binary = "mini-bin"
claim_refs = ["C-001"]
"#;

        let replaced = store.replace_control_plane_experiments_from_registry_text(
            &fixture.root,
            &fixture.experiments,
            replacement,
        )?;
        assert_eq!(replaced, 1);

        let rendered = store.control_plane_compat_text(ControlPlaneCompatKind::Experiments)?;
        assert!(rendered.contains("id = \"E-002\""));
        assert!(!rendered.contains("id = \"E-001\""));
        assert!(rendered.contains("experiment_count = 1"));
        Ok(())
    }

    fn make_test_workspace(label: &str) -> Result<TestWorkspace> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gororoba_provenance_store_{label}_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root)?;

        write_text(
            &root.join("Cargo.toml"),
            r#"[workspace]
resolver = "3"
members = ["crates/test_cli"]
"#,
        )?;
        write_text(
            &root.join("crates/test_cli/Cargo.toml"),
            r#"[package]
name = "test_cli"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "mini-bin"
path = "src/main.rs"

[[bench]]
name = "mini-bench"
path = "benches/mini_bench.rs"
"#,
        )?;
        write_text(
            &root.join("crates/test_cli/src/main.rs"),
            "fn main() { println!(\"mini-bin\"); }",
        )?;
        write_text(
            &root.join("crates/test_cli/benches/mini_bench.rs"),
            "fn main() {}",
        )?;

        let claims = root.join("registry/claims.toml");
        let insights = root.join("registry/insights.toml");
        let experiments = root.join("registry/experiments.toml");
        let binaries = root.join("registry/binaries.toml");
        let rocq_project = root.join("proofs/_RocqProject");
        let proof_file = root.join("proofs/verified/C001_Test.v");
        let db = root.join("registry/canonical/control_plane.sqlite3");

        write_text(
            &claims,
            r#"[[claim]]
id = "C-001"
statement = "Mini claim"
status = "Verified"
where_stated = "`crates/test_cli/src/main.rs`"
last_verified = "2026-03-13"
formal_proof = "proofs/verified/C001_Test.v"
status_note = "Mini proof"
"#,
        )?;
        write_text(
            &insights,
            r#"[[insight]]
id = "I-001"
title = "Mini insight"
status = "verified"
claims = ["C-001"]
"#,
        )?;
        write_text(
            &experiments,
            r#"[experiments]
authoritative = true
status_allowlist = ["active", "planned", "blocked", "deprecated"]

[[experiment]]
id = "E-001"
title = "Mini experiment"
status = "active"
binary = "mini-bin"
claim_refs = ["C-001"]
deterministic = true
"#,
        )?;
        write_text(
            &binaries,
            r#"[[binary]]
name = "mini-bin"
crate = "test_cli"
description = "Mini binary"
experiment = "E-001"
"#,
        )?;
        write_text(&rocq_project, "verified/C001_Test.v")?;
        write_text(&proof_file, "(* mini proof placeholder *)")?;

        Ok(TestWorkspace {
            root,
            claims,
            insights,
            experiments,
            binaries,
            rocq_project,
            db,
        })
    }
}
