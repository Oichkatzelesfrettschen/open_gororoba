use anyhow::{Context, Result, anyhow, bail};
use blake3::Hasher;
use camino::Utf8PathBuf;
use chrono::Utc;
use provenance_core::{
    ArtifactQueryResult, ArtifactRecord, ArtifactStatus, BackendHealthSummary, BinaryRecord,
    ClaimRecord, ControlPlaneCounts, CountSummary, DoctorReport, DocumentQueryResult,
    DocumentRecord, DownloadAttemptRecord, DownloadCampaignQueryResult, DownloadCampaignRecord,
    DownloadJobRecord, DownloadLedgerProjectionRow, DownloadQueryResult, ExperimentRecord,
    ExternalSourceContractRecord, ExternalSourceContractsMeta, ExternalSourceDossierRecord,
    ExternalSourceDossiersMeta, IndexStats, InsightRecord, LaneAssignment, MirrorKind,
    MirrorObservationRecord, PantheonSeedSummary, TheoremRecord,
};
use rusqlite::{Connection, OptionalExtension, params};
use rusqlite_migration::{M, Migrations};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use toml::Value;

const CANONICAL_CLAIM_STATUSES: &[&str] = &[
    "Verified",
    "Established",
    "Refuted",
    "Partial",
    "Provisional",
    "Theoretical",
    "Inconclusive",
    "Superseded",
    "Closed/Negative-Result",
    "Closed/Obstructed",
    "Closed/Research-Program",
    "Closed/Toy",
    "Closed/Analogy",
    "Closed/Source-Insufficient",
    "Closed/Methodology-Insufficient",
    "Closed/Refuted",
];

const CANONICAL_INSIGHT_STATUSES: &[&str] = &[
    "verified",
    "open",
    "superseded",
    "cross-validation-complete",
    "partial",
];

const JUSTIFIED_UNLINKED_THEOREM_IDS: &[&str] = &[
    "C1007_CDPropertyLoss",
    "C958_ZDGraphTopology",
    "C958b_ZDAdjacencyAnalytical",
    "C959_CHSHClassicalBound",
    "C993_CarlsonBranchFree",
    "C999_PathionEntropyBound",
    "C_ConjugateInvolution",
    "C_NormConjugate",
    "C_OctConjInvolution",
    "C_OverImbalancedSign",
    "C_QIBoundNegative",
    "C_QITauScaling",
    "C_SedConjInvolution",
    "C_TraceTracefreeVanishes",
    "C_WECImpliesNEC",
    "C_WarpEnergyNonpositive",
];

const CONTROL_PLANE_DB_PATH: &str = "registry/canonical/control_plane.sqlite3";
const CONTROL_PLANE_EXPORT_COMMAND: &str =
    "cargo run -p gororoba_cli_data --bin provenance -- export-control-plane";
const EXTERNAL_SOURCES_EXPORT_COMMAND: &str =
    "cargo run -p gororoba_cli_data --bin provenance -- export-external-sources";

fn migrations() -> Migrations<'static> {
    Migrations::new(vec![
        M::up(include_str!(
            "../../../db/migrations/0001_provenance_index.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0002_control_plane.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0003_binaries_crate_source.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0004_control_plane_compat_text.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0005_download_jobs.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0006_download_attempt_outcomes.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0007_download_campaigns.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0008_download_attempt_failure_class.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0009_external_sources.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0010_knowledge_and_planning.sql"
        )),
    ])
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlPlaneCompatKind {
    Claims,
    Insights,
    Experiments,
    Binaries,
    Theorems,
    TheoremsMirror,
}

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

    /// Execute a parameterized SQL statement with string slice params.
    pub fn conn_exec(&self, sql: &str, params: &[&str]) -> Result<()> {
        let p: Vec<&dyn rusqlite::types::ToSql> = params
            .iter()
            .map(|s| s as &dyn rusqlite::types::ToSql)
            .collect();
        self.conn
            .execute(sql, p.as_slice())
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

    #[allow(clippy::too_many_arguments)]
    pub fn export_control_plane_compat(
        &mut self,
        repo_root: &Path,
        claims_path: &Path,
        insights_path: &Path,
        experiments_path: &Path,
        binaries_path: &Path,
        theorems_path: &Path,
        theorems_mirror_path: &Path,
    ) -> Result<()> {
        self.backfill_control_plane_compat_from_snapshots()?;
        let outputs = self.render_control_plane_compat_outputs()?;
        write_text(claims_path, &outputs.claims)?;
        write_text(insights_path, &outputs.insights)?;
        write_text(experiments_path, &outputs.experiments)?;
        write_text(binaries_path, &outputs.binaries)?;
        write_text(theorems_path, &outputs.theorems)?;
        write_text(theorems_mirror_path, &outputs.theorems_mirror)?;

        self.record_control_plane_run(
            "export_control_plane",
            &serde_json::json!({
                "claims": to_repo_rel(repo_root, claims_path),
                "insights": to_repo_rel(repo_root, insights_path),
                "experiments": to_repo_rel(repo_root, experiments_path),
                "binaries": to_repo_rel(repo_root, binaries_path),
                "theorems": to_repo_rel(repo_root, theorems_path),
                "theorems_mirror": to_repo_rel(repo_root, theorems_mirror_path),
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

    #[allow(clippy::too_many_arguments)]
    pub fn verify_control_plane_compat_exports(
        &mut self,
        repo_root: &Path,
        claims_path: &Path,
        insights_path: &Path,
        experiments_path: &Path,
        binaries_path: &Path,
        theorems_path: &Path,
        theorems_mirror_path: &Path,
    ) -> Result<()> {
        self.backfill_control_plane_compat_from_snapshots()?;
        let outputs = self.render_control_plane_compat_outputs()?;
        let checks = [
            (claims_path, outputs.claims.as_str()),
            (insights_path, outputs.insights.as_str()),
            (experiments_path, outputs.experiments.as_str()),
            (binaries_path, outputs.binaries.as_str()),
            (theorems_path, outputs.theorems.as_str()),
            (theorems_mirror_path, outputs.theorems_mirror.as_str()),
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
            "SELECT id, title, status, claim_refs_json, compat_toml_text
             FROM insights ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let claim_refs_json: String = row.get(3)?;
            Ok(InsightRecord {
                id: row.get(0)?,
                title: row.get(1)?,
                status: row.get(2)?,
                claim_refs: serde_json::from_str(&claim_refs_json).unwrap_or_default(),
                compat_toml_text: row.get(4)?,
            })
        })?;
        collect_rows(rows)
    }

    pub fn list_experiments(&self) -> Result<Vec<ExperimentRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, status, binary_name, claim_refs_json, compat_toml_text
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
                compat_toml_text: row.get(5)?,
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
                .conn
                .query_row(&format!("SELECT count(*) FROM [{table}]"), [], |r| {
                    r.get(0)
                })
                .unwrap_or(0);
            out.push((table, cat, count, format!("{legacy}|{status}")));
        }
        Ok(out)
    }

    /// Return the full source-of-truth manifest as structured rows.
    pub fn source_of_truth_manifest(
        &self,
    ) -> Result<Vec<(String, String, bool, String, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT table_name, category, authoritative, legacy_toml_path, description, migration_status
             FROM source_of_truth_manifest ORDER BY category, table_name",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, bool>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
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
                .unwrap_or(0);
            let mut cols_stmt =
                self.conn.prepare(&format!("PRAGMA table_info([{name}])"))?;
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
    pub fn upsert_roadmap_item(
        &self,
        id: &str,
        name: &str,
        priority: &str,
        status: &str,
        status_token: &str,
        description: &str,
        sprint: &str,
        dependencies_json: &str,
        acceptance_criteria_json: &str,
        primary_outputs_json: &str,
        evidence_refs_json: &str,
        lacunae_json: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO roadmap_items
             (id, name, priority, status, status_token, description, sprint,
              dependencies_json, acceptance_criteria_json, primary_outputs_json,
              evidence_refs_json, lacunae_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12, datetime('now'))",
            params![
                id,
                name,
                priority,
                status,
                status_token,
                description,
                sprint,
                dependencies_json,
                acceptance_criteria_json,
                primary_outputs_json,
                evidence_refs_json,
                lacunae_json,
            ],
        )?;
        Ok(())
    }

    /// Insert or replace a todo item.
    pub fn upsert_todo_item(
        &self,
        id: &str,
        area: &str,
        title: &str,
        description: &str,
        priority: &str,
        status: &str,
        status_token: &str,
        dependencies_json: &str,
        acceptance_criteria_json: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO todo_items
             (id, area, title, description, priority, status, status_token,
              dependencies_json, acceptance_criteria_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9, datetime('now'))",
            params![
                id,
                area,
                title,
                description,
                priority,
                status,
                status_token,
                dependencies_json,
                acceptance_criteria_json,
            ],
        )?;
        Ok(())
    }

    /// Insert or replace a next-action item.
    pub fn upsert_next_action(
        &self,
        id: &str,
        area: &str,
        title: &str,
        description: &str,
        priority: &str,
        status: &str,
        status_token: &str,
        dependencies_json: &str,
        acceptance_criteria_json: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO next_action_items
             (id, area, title, description, priority, status, status_token,
              dependencies_json, acceptance_criteria_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9, datetime('now'))",
            params![
                id,
                area,
                title,
                description,
                priority,
                status,
                status_token,
                dependencies_json,
                acceptance_criteria_json,
            ],
        )?;
        Ok(())
    }

    /// Insert or replace a research narrative.
    pub fn upsert_research_narrative(
        &self,
        id: &str,
        source_markdown: &str,
        domain: &str,
        slug: &str,
        title: &str,
        status_token: &str,
        content_kind: &str,
        verification_level: &str,
        claim_refs_json: &str,
        url_refs_json: &str,
        path_refs_json: &str,
        body_markdown: &str,
        line_count: i64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO research_narratives
             (id, source_markdown, domain, slug, title, status_token, content_kind,
              verification_level, claim_refs_json, url_refs_json, path_refs_json,
              body_markdown, line_count)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            params![
                id,
                source_markdown,
                domain,
                slug,
                title,
                status_token,
                content_kind,
                verification_level,
                claim_refs_json,
                url_refs_json,
                path_refs_json,
                body_markdown,
                line_count,
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
            .query_row(&format!("SELECT count(*) FROM [{table}]"), [], |r| {
                r.get(0)
            })?;
        Ok(count)
    }

    /// Full-text search across research narratives.
    pub fn search_narratives(&self, query: &str, limit: usize) -> Result<Vec<(String, String, f64)>> {
        let mut stmt = self.conn.prepare(
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

    /// List roadmap items, optionally filtered by status.
    pub fn list_roadmap_items(&self, status_filter: Option<&str>) -> Result<Vec<(String, String, String, String)>> {
        self.list_four_col_table("roadmap_items", "id, name, priority, status", status_filter)
    }

    /// List todo items, optionally filtered by status.
    pub fn list_todo_items(&self, status_filter: Option<&str>) -> Result<Vec<(String, String, String, String)>> {
        self.list_four_col_table("todo_items", "id, title, priority, status", status_filter)
    }

    /// List next-action items, optionally filtered by status.
    pub fn list_next_actions(&self, status_filter: Option<&str>) -> Result<Vec<(String, String, String, String)>> {
        self.list_four_col_table("next_action_items", "id, title, priority, status", status_filter)
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
            for r in rows { out.push(r?); }
        } else {
            let sql = format!("SELECT {cols} FROM [{table}] ORDER BY id");
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?;
            for r in rows { out.push(r?); }
        }
        Ok(out)
    }

    /// Insert or replace a notebook session.
    pub fn upsert_notebook_session(
        &self,
        id: &str,
        title: &str,
        description: &str,
        kernel: &str,
        status: &str,
        cell_count: i64,
        cells_json: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO notebook_sessions
             (id, title, description, kernel, status, cell_count, cells_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7, datetime('now'))",
            params![id, title, description, kernel, status, cell_count, cells_json],
        )?;
        Ok(())
    }

    /// List notebook sessions.
    pub fn list_notebook_sessions(&self) -> Result<Vec<(String, String, String, String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, kernel, status, cell_count FROM notebook_sessions ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }
}

fn clear_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        INSERT INTO document_search(document_search) VALUES('delete-all');
        DELETE FROM record_sources;
        DELETE FROM citations;
        DELETE FROM mirror_observations;
        DELETE FROM artifact_links;
        DELETE FROM artifact_paths;
        DELETE FROM lane_assignments;
        DELETE FROM links;
        DELETE FROM artifacts;
        DELETE FROM documents;
        DELETE FROM ingest_fingerprints;
        ",
    )?;
    Ok(())
}

fn clear_control_plane_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        DELETE FROM registry_snapshots;
        DELETE FROM claims;
        DELETE FROM insights;
        DELETE FROM experiments_cp;
        DELETE FROM binaries_cp;
        DELETE FROM theorems;
        DELETE FROM control_plane_meta;
        ",
    )?;
    Ok(())
}

fn clear_external_source_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        DELETE FROM external_source_contract_values;
        DELETE FROM external_source_contracts;
        DELETE FROM external_source_contracts_meta;
        DELETE FROM external_source_dossier_values;
        DELETE FROM external_source_dossiers;
        DELETE FROM external_source_dossiers_meta;
        ",
    )?;
    Ok(())
}

fn insert_ranked_values(
    conn: &Connection,
    table: &str,
    owner_column: &str,
    owner_id: &str,
    relation: &str,
    values: &[String],
) -> Result<()> {
    let sql =
        format!("INSERT INTO {table}({owner_column}, relation, ord, value) VALUES(?1, ?2, ?3, ?4)");
    for (ord, value) in values.iter().enumerate() {
        conn.execute(&sql, params![owner_id, relation, ord as i64, value])?;
    }
    Ok(())
}

fn replace_ranked_values(
    conn: &Connection,
    table: &str,
    owner_column: &str,
    owner_id: &str,
    relation: &str,
    values: &[String],
) -> Result<()> {
    let delete_sql = format!("DELETE FROM {table} WHERE {owner_column} = ?1 AND relation = ?2");
    conn.execute(&delete_sql, params![owner_id, relation])?;
    insert_ranked_values(conn, table, owner_column, owner_id, relation, values)
}

fn load_ranked_values(
    conn: &Connection,
    table: &str,
    owner_column: &str,
    owner_id: &str,
    relation: &str,
) -> Result<Vec<String>> {
    let sql = format!(
        "SELECT value FROM {table} WHERE {owner_column} = ?1 AND relation = ?2 ORDER BY ord"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params![owner_id, relation], |row| row.get::<_, String>(0))?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn write_fingerprint(
    conn: &Connection,
    repo_root: &Path,
    path: &Path,
    indexed_at: &str,
) -> Result<()> {
    let raw = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Hasher::new();
    hasher.update(&raw);
    let rel = to_repo_rel(repo_root, path);
    conn.execute(
        "INSERT INTO ingest_fingerprints (path, blake3_hex, size_bytes, indexed_at)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(path) DO UPDATE SET
            blake3_hex=excluded.blake3_hex,
            size_bytes=excluded.size_bytes,
            indexed_at=excluded.indexed_at",
        params![
            rel,
            hasher.finalize().to_hex().to_string(),
            raw.len() as i64,
            indexed_at
        ],
    )?;
    Ok(())
}

fn write_registry_snapshot(
    conn: &Connection,
    repo_root: &Path,
    kind: &str,
    path: &Path,
    body: &str,
    indexed_at: &str,
) -> Result<()> {
    let mut hasher = Hasher::new();
    hasher.update(body.as_bytes());
    conn.execute(
        "INSERT INTO registry_snapshots(registry_kind, source_path, content_text, blake3_hex, indexed_at)
         VALUES(?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(registry_kind) DO UPDATE SET
            source_path=excluded.source_path,
            content_text=excluded.content_text,
            blake3_hex=excluded.blake3_hex,
            indexed_at=excluded.indexed_at",
        params![
            kind,
            to_repo_rel(repo_root, path),
            body,
            hasher.finalize().to_hex().to_string(),
            indexed_at
        ],
    )?;
    Ok(())
}

fn scalar_count(conn: &Connection, sql: &str) -> Result<usize> {
    Ok(conn.query_row(sql, [], |row| row.get::<_, i64>(0))? as usize)
}

fn query_count_summaries(conn: &Connection, sql: &str) -> Result<Vec<CountSummary>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(CountSummary {
            key: row.get(0)?,
            count: row.get::<_, i64>(1)? as usize,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn query_backend_health(conn: &Connection) -> Result<Vec<BackendHealthSummary>> {
    let mut stmt = conn.prepare(
        "SELECT backend,
                SUM(CASE WHEN succeeded != 0 THEN 1 ELSE 0 END) AS success_count,
                SUM(CASE WHEN succeeded = 0 THEN 1 ELSE 0 END) AS failure_count,
                SUM(bytes) AS total_bytes
         FROM download_attempts
         GROUP BY backend
         ORDER BY (success_count + failure_count) DESC, backend ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(BackendHealthSummary {
            backend: row.get(0)?,
            success_count: row.get::<_, i64>(1)? as usize,
            failure_count: row.get::<_, i64>(2)? as usize,
            total_bytes: row.get::<_, i64>(3)?.max(0) as u64,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn load_string_vec(conn: &Connection, sql: &str, id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(params![id], |row| row.get::<_, String>(0))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn load_record_sources(
    conn: &Connection,
    entity_kind: &str,
    entity_id: &str,
) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT source_ref FROM record_sources
         WHERE entity_kind = ?1 AND entity_id = ?2
         ORDER BY source_ref",
    )?;
    let rows = stmt.query_map(params![entity_kind, entity_id], |row| {
        row.get::<_, String>(0)
    })?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn to_repo_rel(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn load_artifacts(path: &Path) -> Result<Vec<ArtifactRecord>> {
    let value = load_toml_value(path)?;
    let artifacts = value
        .get("artifact")
        .and_then(Value::as_array)
        .context("artifact table missing")?;
    let mut out = Vec::new();
    for artifact in artifacts {
        let table = artifact
            .as_table()
            .context("artifact row must be a table")?;
        let status_raw = table
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unverified");
        let status = ArtifactStatus::parse(status_raw)
            .with_context(|| format!("invalid artifact status {status_raw}"))?;
        out.push(ArtifactRecord {
            id: string_field(table, "id"),
            key: string_field(table, "key"),
            title: string_field(table, "title"),
            citation: string_field(table, "citation"),
            status,
            minimum_requirement_met: bool_field(table, "minimum_requirement_met"),
            canonical_functional_url: optional_string_field(table, "canonical_functional_url"),
            canonical_download_path: optional_string_field(table, "canonical_download_path")
                .map(Utf8PathBuf::from),
            source_refs: string_array_field(table, "source_refs"),
            all_links: string_array_field(table, "all_links"),
            downloaded_paths: string_array_field(table, "downloaded_paths")
                .into_iter()
                .map(Utf8PathBuf::from)
                .collect(),
            doi_list: string_array_field(table, "doi_list"),
            notes: string_array_field(table, "notes"),
        });
    }
    Ok(out)
}

fn load_documents(path: &Path) -> Result<Vec<DocumentRecord>> {
    let value = load_toml_value(path)?;
    let documents = value
        .get("document")
        .and_then(Value::as_array)
        .context("document table missing")?;
    let mut out = Vec::new();
    for document in documents {
        let table = document
            .as_table()
            .context("document row must be a table")?;
        out.push(DocumentRecord {
            id: string_field(table, "id"),
            path: Utf8PathBuf::from(string_field(table, "path")),
            title: string_field(table, "title"),
            kind: string_field(table, "kind"),
            authoring_mode: string_field(table, "authoring_mode"),
            generated: bool_field(table, "generated"),
            status: string_field(table, "status"),
            toml_backing: optional_string_field(table, "toml_backing").map(Utf8PathBuf::from),
            sha256: optional_string_field(table, "sha256"),
            size_bytes: optional_integer_field(table, "size_bytes"),
            line_count: optional_integer_field(table, "line_count"),
        });
    }
    Ok(out)
}

fn load_lane_assignments(lane_dir: &Path) -> Result<Vec<LaneAssignment>> {
    let mut out = Vec::new();
    for lane_name in [
        "datasets",
        "slides_artifacts",
        "papers_pdf",
        "web_references",
    ] {
        let path = lane_dir.join(format!("{lane_name}.toml"));
        if !path.exists() {
            continue;
        }
        let value = load_toml_value(&path)?;
        let refs = value
            .get("artifact_ref")
            .and_then(Value::as_array)
            .context("artifact_ref table missing")?;
        for artifact_ref in refs {
            let table = artifact_ref
                .as_table()
                .context("artifact_ref row must be a table")?;
            out.push(LaneAssignment {
                artifact_id: string_field(table, "id"),
                lane_name: lane_name.to_string(),
            });
        }
    }
    Ok(out)
}

fn build_mirror_observations(path: &Path) -> Result<Vec<MirrorObservationRecord>> {
    let value = load_toml_value(path)?;
    let artifacts = value
        .get("artifact")
        .and_then(Value::as_array)
        .context("artifact table missing")?;
    let mut out = Vec::new();
    for artifact in artifacts {
        let table = artifact
            .as_table()
            .context("artifact row must be a table")?;
        let artifact_id = string_field(table, "id");
        for (field, kind) in [
            ("working_mirrors", MirrorKind::Working),
            ("working_pdf_mirrors", MirrorKind::WorkingPdf),
            ("nonworking_mirrors", MirrorKind::Nonworking),
            ("unverified_mirrors", MirrorKind::Unverified),
        ] {
            for url in string_array_field(table, field) {
                out.push(MirrorObservationRecord {
                    artifact_id: artifact_id.clone(),
                    url,
                    mirror_kind: kind.clone(),
                });
            }
        }
    }
    Ok(out)
}

fn load_toml_value(path: &Path) -> Result<Value> {
    let raw = load_toml_text(path)?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

fn load_toml_text(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
}

fn load_text(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
}

fn load_registry_table_toml(raw: &str, key: &str) -> Result<Option<String>> {
    let value: Value = toml::from_str(raw).with_context(|| format!("parse {key} registry"))?;
    let Some(table) = value.get(key).and_then(Value::as_table) else {
        return Ok(None);
    };
    render_toml_table(table).map(Some)
}

fn render_toml_table(table: &toml::map::Map<String, Value>) -> Result<String> {
    toml::to_string(table).context("serialize TOML table")
}

fn string_field(table: &toml::map::Map<String, Value>, key: &str) -> String {
    table
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn optional_string_field(table: &toml::map::Map<String, Value>, key: &str) -> Option<String> {
    let value = table.get(key).and_then(Value::as_str).unwrap_or("").trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn string_array_field(table: &toml::map::Map<String, Value>, key: &str) -> Vec<String> {
    table
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn bool_field(table: &toml::map::Map<String, Value>, key: &str) -> bool {
    table.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn optional_integer_field(table: &toml::map::Map<String, Value>, key: &str) -> Option<i64> {
    table.get(key).and_then(Value::as_integer)
}

fn host_for_url(url: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(|host| host.to_string()))
}

fn join_refs(values: &[String]) -> String {
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(" | ")
}

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
                    "Workspace binary discovered from cargo metadata in crate {}; registry metadata pending.",
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
                    "Workspace binary discovered from {}; registry metadata pending.",
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

fn render_claims_registry(claims: &[ClaimRecord]) -> String {
    render_array_of_tables_registry("claims", "claim", claims.iter().map(render_claim_row))
}

fn render_insights_registry(insights: &[InsightRecord]) -> String {
    render_array_of_tables_registry(
        "insights",
        "insight",
        insights.iter().map(render_insight_row),
    )
}

fn render_experiments_registry(header_toml: &str, experiments: &[ExperimentRecord]) -> String {
    let mut lines = compat_toml_export_header("experiments");
    let header = rebuild_experiments_header_toml(header_toml, experiments);
    if !header.trim().is_empty() {
        lines.push("[experiments]".to_string());
        lines.push(header);
        lines.push(String::new());
    }
    for row in experiments {
        lines.push("[[experiment]]".to_string());
        lines.push(render_experiment_row(row));
        lines.push(String::new());
    }
    lines.join("\n")
}

fn rebuild_experiments_header_toml(header_toml: &str, experiments: &[ExperimentRecord]) -> String {
    let mut table = header_toml
        .trim()
        .parse::<Value>()
        .ok()
        .and_then(|value| value.as_table().cloned())
        .unwrap_or_default();
    table.insert("authoritative".to_string(), Value::Boolean(true));
    table.insert(
        "experiment_count".to_string(),
        Value::Integer(experiments.len() as i64),
    );
    table.insert(
        "deterministic_count".to_string(),
        Value::Integer(
            experiments
                .iter()
                .filter(|row| experiment_row_flag(row, "deterministic").unwrap_or(false))
                .count() as i64,
        ),
    );
    table.insert(
        "gpu_count".to_string(),
        Value::Integer(
            experiments
                .iter()
                .filter(|row| experiment_row_flag(row, "gpu").unwrap_or(false))
                .count() as i64,
        ),
    );
    table.insert(
        "seeded_count".to_string(),
        Value::Integer(
            experiments
                .iter()
                .filter(|row| experiment_row_has_seed(row))
                .count() as i64,
        ),
    );
    toml::to_string(&table)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn experiment_row_table(row: &ExperimentRecord) -> Option<toml::value::Table> {
    row.compat_toml_text
        .trim()
        .parse::<Value>()
        .ok()
        .and_then(|value| value.as_table().cloned())
}

fn experiment_row_flag(row: &ExperimentRecord, key: &str) -> Option<bool> {
    experiment_row_table(row).and_then(|table| table.get(key).and_then(Value::as_bool))
}

fn experiment_row_has_seed(row: &ExperimentRecord) -> bool {
    experiment_row_table(row)
        .and_then(|table| table.get("seed").cloned())
        .is_some()
}

fn render_array_of_tables_registry(
    kind: &str,
    array_key: &str,
    rows: impl IntoIterator<Item = String>,
) -> String {
    let mut lines = compat_toml_export_header(kind);
    for row in rows {
        lines.push(format!("[[{array_key}]]"));
        lines.push(row.trim().to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_claim_row(row: &ClaimRecord) -> String {
    if row.compat_toml_text.trim().is_empty() {
        let mut lines = vec![
            format!("id = {:?}", row.id),
            format!("statement = {:?}", row.statement),
            format!("status = {:?}", row.status),
            format!("where_stated = {:?}", row.where_stated),
            format!("last_verified = {:?}", row.last_verified),
        ];
        if let Some(formal_proof) = &row.formal_proof {
            lines.push(format!("formal_proof = {:?}", formal_proof));
        }
        if let Some(status_note) = &row.status_note {
            lines.push(format!("status_note = {:?}", status_note));
        }
        lines.join("\n")
    } else {
        row.compat_toml_text.trim().to_string()
    }
}

fn render_insight_row(row: &InsightRecord) -> String {
    if row.compat_toml_text.trim().is_empty() {
        let mut lines = vec![
            format!("id = {:?}", row.id),
            format!("title = {:?}", row.title),
            format!("status = {:?}", row.status),
        ];
        if !row.claim_refs.is_empty() {
            lines.push(format!("claims = {:?}", row.claim_refs));
        }
        lines.join("\n")
    } else {
        row.compat_toml_text.trim().to_string()
    }
}

fn render_experiment_row(row: &ExperimentRecord) -> String {
    if row.compat_toml_text.trim().is_empty() {
        let mut lines = vec![
            format!("id = {:?}", row.id),
            format!("title = {:?}", row.title),
            format!("status = {:?}", row.status),
        ];
        if let Some(binary) = &row.binary {
            lines.push(format!("binary = {:?}", binary));
        }
        if !row.claim_refs.is_empty() {
            lines.push(format!("claim_refs = {:?}", row.claim_refs));
        }
        lines.join("\n")
    } else {
        row.compat_toml_text.trim().to_string()
    }
}

fn render_binaries_registry(binaries: &[BinaryRecord]) -> String {
    let mut lines = compat_toml_export_header("binaries");
    lines.extend([
        "# CLI binaries registry -- generated from the canonical SQLite control plane.".to_string(),
        String::new(),
    ]);
    for binary in binaries {
        lines.push("[[binary]]".to_string());
        lines.push(format!("name = {:?}", binary.name));
        lines.push(format!("crate = {:?}", binary.crate_name));
        lines.push(format!("description = {:?}", binary.description));
        if let Some(experiment) = &binary.experiment {
            lines.push(format!("experiment = {:?}", experiment));
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

fn compat_toml_export_header(kind: &str) -> Vec<String> {
    vec![
        "# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.".to_string(),
        format!("# Canonical write path: {CONTROL_PLANE_DB_PATH}"),
        format!("# Regenerate with: {CONTROL_PLANE_EXPORT_COMMAND}"),
        format!("# Compatibility export lane: {kind}"),
        String::new(),
    ]
}

fn compat_markdown_export_header(source_label: &str) -> Vec<String> {
    vec![
        "<!-- AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT. -->".to_string(),
        format!("<!-- Source of truth: {CONTROL_PLANE_DB_PATH} -->"),
        format!("<!-- Canonical write path: {CONTROL_PLANE_DB_PATH} -->"),
        format!("<!-- Source label: {source_label} -->"),
        format!("<!-- Regenerate with: {CONTROL_PLANE_EXPORT_COMMAND} -->"),
        String::new(),
    ]
}

fn external_sources_compat_toml_export_header(kind: &str) -> Vec<String> {
    vec![
        "# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.".to_string(),
        format!("# Canonical write path: {CONTROL_PLANE_DB_PATH}"),
        format!("# Regenerate with: {EXTERNAL_SOURCES_EXPORT_COMMAND}"),
        format!("# Compatibility export lane: {kind}"),
    ]
}

fn external_sources_markdown_export_header(source_label: &str) -> Vec<String> {
    vec![
        "<!-- AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT. -->".to_string(),
        "<!-- Source of truth: registry/external_sources.toml -->".to_string(),
        format!("<!-- Canonical write path: {CONTROL_PLANE_DB_PATH} -->"),
        format!("<!-- Source label: {source_label} -->"),
        format!("<!-- Regenerate with: {EXTERNAL_SOURCES_EXPORT_COMMAND} -->"),
        String::new(),
    ]
}

fn bool_toml(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn write_text(path: &Path, body: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create parent directory {}", parent.display()))?;
    }
    fs::write(path, format!("{body}\n")).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

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
