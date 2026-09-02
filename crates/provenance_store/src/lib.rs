use anyhow::{Context, Result, anyhow, bail};
use camino::Utf8PathBuf;
use chrono::Utc;
use provenance_core::{
    ArtifactQueryResult, ArtifactRecord, ArtifactStatus, ClaimRecord,
    ControlPlaneCounts, DoctorReport, DocumentQueryResult, DocumentRecord, DownloadAttemptRecord,
    DownloadCampaignQueryResult, DownloadCampaignRecord, DownloadJobRecord,
    DownloadLedgerProjectionRow, DownloadQueryResult, ExperimentRecord,
    ExternalSourceContractRecord, ExternalSourceContractsMeta, ExternalSourceDossierRecord,
    ExternalSourceDossiersMeta, IndexStats, InsightRecord, MirrorKind, MirrorObservationRecord,
    TheoremRecord,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
#[cfg(test)]
use std::path::PathBuf;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path},
};
use toml::Value;

// Canonical claim/insight status taxonomies, theorem-ID allowlist,
// control-plane TOML/SQLite paths, and the rusqlite migration registry
// live in the `migrations` submodule. Items are pub(crate) and brought
// back into lib.rs scope via plain use statements.
mod migrations;
use migrations::migrations;

// Compatibility-TOML renderers for the three planning lanes
// (render_roadmap_compat_toml, render_todo_compat_toml,
// render_next_actions_compat_toml) live in the `planning_render`
// submodule via a second impl ProvenanceStore block. Each method
// is pub(crate) so the parent's `render_planning_compat_toml`
// dispatcher can call it as a method on self.
mod planning_render;

// Literature-verification run persistence + recent-runs query +
// per-run results / similar-papers fetchers live in the
// `literature_verification` submodule via a second impl
// ProvenanceStore block. All four methods are pub.
mod literature_verification;

// One-shot Pantheon/PhysicsForge migration seeder
// (seed_pantheon_physicsforge_migration) lives in the
// `pantheon_seed` submodule via a second impl ProvenanceStore block.
mod pantheon_seed;

// SQLite row-fetchers for the planning + requirements compat
// renderers (planning_roadmap_rows, planning_todo_rows,
// planning_next_action_rows, requirements_meta_row,
// requirements_module_rows, requirements_coverage_gap_rows) live
// in the `planning_rows` submodule via a second impl
// ProvenanceStore block. All six methods are pub.
mod planning_rows;

// Mutator methods for the requirements registry
// (upsert_requirements_meta, upsert_requirement_module,
// delete_requirement_module, upsert_requirement_coverage_gap,
// delete_requirement_coverage_gap) live in the `requirements_mut`
// submodule via a second impl ProvenanceStore block.
mod requirements_mut;

mod theorem_identity;
pub use theorem_identity::{
    TheoremBindingSpec, TheoremClaimMapping, TheoremIdentityBindResult, TheoremIdentitySpec,
    bind_theorem_identities, parse_theorem_identity_spec,
};
use theorem_identity::{
    default_stable_theorem_id, is_declared_legacy_alias, validate_theorem_identities,
};

mod claim_transitions;
pub use claim_transitions::{
    AllocatedSuccessor, ClaimRelationView, ClaimTransitionApplyResult, ClaimTransitionCompatPaths,
    ClaimTransitionEventView, ClaimTransitionPlan, ClaimTransitionRequest, ClaimTransitionSpec,
    SuccessorClaimSpec,
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

pub struct LocalArtifactRegistration<'a> {
    pub id: &'a str,
    pub key: &'a str,
    pub title: &'a str,
    pub citation: &'a str,
    pub paths: &'a [String],
    pub lane_name: &'a str,
    pub source_refs: &'a [String],
    pub actor: Option<&'a str>,
    pub reason: Option<&'a str>,
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
pub(crate) struct ProofInventoryEntry {
    pub(crate) stem: String,
    pub(crate) path: Utf8PathBuf,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ProofInventory {
    pub(crate) project_raw: String,
    pub(crate) verified_entries: Vec<ProofInventoryEntry>,
    pub(crate) verified_by_claim_id: BTreeMap<String, Vec<ProofInventoryEntry>>,
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
// `list_binaries` and `binaries_sync` both traffic in BinaryRecord, so callers
// outside this crate need the type to build a sync request.
pub use provenance_core::BinaryRecord;
pub mod types;
pub use types::{
    ActionCompatRow, ActionItem, ActionItemWithEvidence, BinariesSyncSummary, CompatExportPaths,
    ControlPlaneCompatKind, EntityFieldTarget, ExecutionTargetRetarget,
    ExecutionTargetRetargetSummary, ManifestRow, NotebookSessionRow, NotebookSessionSummary,
    PlanningCompatTable, RequirementCoverageGapCompatRow, RequirementCoverageGapItem,
    RequirementModuleCompatRow, RequirementModuleItem, RequirementsMeta, RequirementsMetaCompatRow,
    ResearchNarrativeRow, RoadmapCompatRow, RoadmapItem, RoadmapItemWithLinks,
    RegistryImportPaths, SourcePathRetarget, SourcePathRetargetSummary, StatusNoteRevision,
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

/// Columns an execution-target rename reaches. Reproduction commands live in
/// the experiment compat blob; a claim cites its producing lane in both its
/// status note and its own compat blob.
const RETARGET_FIELDS: &[EntityFieldTarget<'static>] = &[
    EntityFieldTarget {
        table: "experiments_cp",
        revisions_table: "experiment_revisions",
        fk_col: "experiment_id",
        field: "compat_toml_text",
    },
    EntityFieldTarget {
        table: "claims",
        revisions_table: "claim_revisions",
        fk_col: "claim_id",
        field: "status_note",
    },
    EntityFieldTarget {
        table: "claims",
        revisions_table: "claim_revisions",
        fk_col: "claim_id",
        field: "where_stated",
    },
    EntityFieldTarget {
        table: "claims",
        revisions_table: "claim_revisions",
        fk_col: "claim_id",
        field: "compat_toml_text",
    },
];

/// True when `text[at..at + len]` is a whole execution-target token rather than
/// a piece of a longer name. Target names share prefixes -- `heliosphere-r16-
/// ablation` sits beside `heliosphere-r16-ablation-sparse` -- so a plain
/// substring replace would corrupt the longer name while rewriting the shorter.
fn is_token_boundary(text: &str, at: usize, len: usize) -> bool {
    let tail = |c: char| c.is_alphanumeric() || c == '-' || c == '_';
    let before = text[..at].chars().next_back().is_none_or(|c| !tail(c));
    let after = text[at + len..].chars().next().is_none_or(|c| !tail(c));
    before && after
}

/// Replace whole-token occurrences of `from` with `to`.
///
/// A rejected match advances the cursor by one character rather than past the
/// whole candidate, so an overlapping match starting inside it is still found.
fn replace_token(text: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    while let Some(offset) = text[cursor..].find(from) {
        let at = cursor + offset;
        if is_token_boundary(text, at, from.len()) {
            out.push_str(&text[cursor..at]);
            out.push_str(to);
            cursor = at + from.len();
        } else {
            let step = text[at..].chars().next().map_or(1, char::len_utf8);
            out.push_str(&text[cursor..at + step]);
            cursor = at + step;
        }
    }
    out.push_str(&text[cursor..]);
    out
}

/// Rewrite one execution target inside a reproduction command or citation.
///
/// `cargo run --bin <target> -- <args>` already carries the `--` that separates
/// cargo's arguments from the program's, so a target that gains a subcommand
/// consumes that separator instead of emitting a second one: `--bin a-b -- --x`
/// becomes `--bin a -- b --x`, never `--bin a -- b -- --x`. Outside a `--bin`
/// argument the target is a plain name and is substituted verbatim.
fn rewrite_execution_target(text: &str, from: &str, to: &str) -> String {
    let (to_bin, to_sub) = to.split_once(' ').unwrap_or((to, ""));
    let invocation = if to_sub.is_empty() {
        format!("--bin {to_bin}")
    } else {
        format!("--bin {to_bin} -- {to_sub}")
    };
    let out = replace_token(text, &format!("--bin {from} --"), &invocation);
    let out = replace_token(&out, &format!("--bin {from}"), &invocation);
    replace_token(&out, from, to)
}

/// Sentinel inserted into *_revisions.application_id so future triggers
/// can distinguish CLI-driven mutations from raw SQL pokes. The hex
/// digits "go ro" (`0x676f_726f`) -- a fingerprint of the gororoba CLI.
const CLI_APPLICATION_ID: i64 = 0x676f_726f;

impl ProvenanceStore {
    pub fn bind_theorem_identities(
        &mut self,
        repo_root: &Path,
        spec: &TheoremIdentitySpec,
        raw_spec: &str,
    ) -> Result<TheoremIdentityBindResult> {
        theorem_identity::bind_theorem_identities(&mut self.conn, repo_root, spec, raw_spec)
    }

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

    /// Open an already-migrated canonical database without running migrations.
    ///
    /// Read-only transition planning must not create directories, apply schema
    /// changes, or acquire a write handle. The caller receives a clear error
    /// when the database is absent or has not reached the expected schema.
    pub fn open_read_only(db_path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("open read-only sqlite database {}", db_path.display()))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
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
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                ON CONFLICT(id) DO UPDATE SET
                    key = excluded.key,
                    title = excluded.title,
                    citation = excluded.citation,
                    status = excluded.status,
                    minimum_requirement_met = excluded.minimum_requirement_met,
                    canonical_functional_url = excluded.canonical_functional_url,
                    canonical_download_path = excluded.canonical_download_path",
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
                    "INSERT INTO citations (artifact_id, citation_text, doi, canonical_url)
                     SELECT ?1, ?2, ?3, ?4
                     WHERE NOT EXISTS (
                         SELECT 1 FROM citations
                         WHERE artifact_id = ?1
                           AND citation_text = ?2
                           AND doi IS ?3
                           AND canonical_url IS ?4
                     )",
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

    /// Import the compatibility mirrors into the canonical store.
    ///
    /// This is a bootstrap path, not a refresh path. `registry/*.toml` omit
    /// columns the canonical store owns -- `insights` carries no `status_note`
    /// key -- and `clear_control_plane_tables` deletes the insight rows outright,
    /// so importing over a populated database resets those columns to NULL.
    /// `options` decides: `ReimportOptions::bootstrap` refuses on a populated
    /// database, `ReimportOptions::destructive` backs it up and writes a semantic
    /// diff first. Either way the canonical-only values captured before the
    /// tables are cleared are reapplied wherever the mirror row leaves the column
    /// empty.
    pub fn reindex_control_plane_from_registries(
        &mut self,
        repo_root: &Path,
        paths: RegistryImportPaths<'_>,
        options: ReimportOptions<'_>,
    ) -> Result<IndexStats> {
        let RegistryImportPaths {
            claims: claims_path,
            insights: insights_path,
            experiments: experiments_path,
            binaries: binaries_path,
            rocq_project: proofs_project_path,
        } = paths;
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

        // Every column the importer touches has to be either written from the
        // mirror or carried across from SQLite. A schema column in neither set
        // would be nulled silently, so the run stops before the first DELETE.
        assert_column_mapping_total(&self.conn)?;
        let population = measure_population(&self.conn)?;
        if population.is_populated() && !options.allow_destructive_reimport {
            bail!("{}", refusal_message(options.db_path, &population));
        }
        if population.is_populated() {
            let db_path = options.db_path.context(
                "destructive re-import needs ReimportOptions::db_path to write the backup",
            )?;
            let backup = backup_database(&self.conn, db_path)?;
            let backup_display = backup.display().to_string();
            let lanes = vec![
                DiffLane {
                    table: "claims",
                    select_sql: "SELECT id, status_note FROM claims",
                    incoming: claims
                        .iter()
                        .map(|row| (row.id.clone(), row.status_note.clone()))
                        .collect(),
                },
                DiffLane {
                    table: "insights",
                    select_sql: "SELECT id, status_note FROM insights",
                    incoming: insights
                        .iter()
                        .map(|row| (row.id.clone(), row.status_note.clone()))
                        .collect(),
                },
                DiffLane {
                    table: "experiments_cp",
                    select_sql: "SELECT id, status_note FROM experiments_cp",
                    incoming: experiments
                        .iter()
                        .map(|row| (row.id.clone(), row.status_note.clone()))
                        .collect(),
                },
                DiffLane {
                    table: "binaries_cp",
                    select_sql: "SELECT name, description FROM binaries_cp",
                    incoming: binaries
                        .iter()
                        .map(|row| {
                            (
                                row.name.clone(),
                                Some(row.description.clone()).filter(|d| !d.is_empty()),
                            )
                        })
                        .collect(),
                },
            ];
            let diff = build_diff(&self.conn, &backup_display, lanes)?;
            let diff_path = diff_path_for_backup(&backup);
            fs::write(&diff_path, diff.to_toml())
                .with_context(|| format!("write {}", diff_path.display()))?;
            print!("{}", diff.to_summary());
            println!("  diff written to {}", diff_path.display());
        }
        // The insight rows are deleted outright below, so a COALESCE upsert
        // cannot see their prior values. Read them out first, reapply after.
        let preserved = capture_preserved_values(&self.conn)?;

        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO claim_status_write_context (
                 id, mode, transition_event_id, source_claim_id, proposed_status
             ) VALUES (1, 'registry_reindex', NULL, NULL, NULL)
             ON CONFLICT(id) DO UPDATE SET
                 mode = excluded.mode,
                 transition_event_id = excluded.transition_event_id,
                 source_claim_id = excluded.source_claim_id,
                 proposed_status = excluded.proposed_status",
            [],
        )?;
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
                    status = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM claim_transition_events
                            WHERE source_claim_id = claims.id
                        ) THEN claims.status
                        ELSE excluded.status
                    END,
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
                "INSERT INTO insights(id, title, status, claim_refs_json, status_note, compat_toml_text)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    claim_refs_json=excluded.claim_refs_json,
                    status_note=excluded.status_note,
                    compat_toml_text=excluded.compat_toml_text",
                params![
                    insight.id,
                    insight.title,
                    insight.status,
                    serde_json::to_string(&insight.claim_refs)?,
                    insight.status_note,
                    insight.compat_toml_text
                ],
            )?;
        }
        for experiment in &experiments {
            tx.execute(
                "INSERT INTO experiments_cp(id, title, status, binary_name, claim_refs_json, status_note, compat_toml_text)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    binary_name=excluded.binary_name,
                    claim_refs_json=excluded.claim_refs_json,
                    status_note=excluded.status_note,
                    compat_toml_text=excluded.compat_toml_text",
                params![
                    experiment.id,
                    experiment.title,
                    experiment.status,
                    experiment.binary,
                    serde_json::to_string(&experiment.claim_refs)?,
                    experiment.status_note,
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
            let identity = tx
                .query_row(
                    "SELECT stable_id, identity_kind FROM theorem_identities
                     WHERE legacy_name = ?1",
                    params![theorem.legacy_name],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .optional()?;
            let (stable_id, identity_kind) = if let Some(identity) = identity {
                identity
            } else {
                let stable_id = default_stable_theorem_id(&theorem.legacy_name);
                tx.execute(
                    "INSERT INTO theorem_identities (
                         stable_id, legacy_name, proof_path, identity_kind, source
                     ) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        stable_id,
                        theorem.legacy_name,
                        theorem.proof_path.as_str(),
                        theorem.identity_kind,
                        theorem.source
                    ],
                )?;
                (stable_id, theorem.identity_kind.clone())
            };
            if theorem.linked_claim_ids.is_empty()
                && is_declared_legacy_alias(&theorem.legacy_name)
                && identity_kind != "legacy_alias"
            {
                tx.execute(
                    "UPDATE theorem_identities SET identity_kind = 'legacy_alias'
                     WHERE stable_id = ?1",
                    params![stable_id],
                )?;
            } else if theorem.linked_claim_ids.is_empty() && identity_kind == "explicit_link" {
                tx.execute(
                    "UPDATE theorem_identities SET identity_kind = 'unresolved'
                     WHERE stable_id = ?1",
                    params![stable_id],
                )?;
            } else if !theorem.linked_claim_ids.is_empty() && identity_kind == "unresolved" {
                tx.execute(
                    "UPDATE theorem_identities SET identity_kind = 'explicit_link'
                     WHERE stable_id = ?1",
                    params![stable_id],
                )?;
            }
            for claim_id in &theorem.linked_claim_ids {
                tx.execute(
                    "INSERT OR IGNORE INTO theorem_claim_links (
                         theorem_stable_id, claim_id, relation_kind
                     ) VALUES (?1, ?2, 'formal_proposition')",
                    params![stable_id, claim_id],
                )?;
            }
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
                    theorem.legacy_name,
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
        restore_preserved_values(&tx, &preserved)?;
        tx.execute("DELETE FROM claim_status_write_context WHERE id = 1", [])?;
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

    pub fn export_control_plane_compat_paths(
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
        let transition_events = repo_root.join("registry/claim_transitions.toml");
        let transition_relations = repo_root.join("registry/claim_relations.toml");
        self.export_claim_transition_compat_paths(
            repo_root,
            ClaimTransitionCompatPaths {
                events: &transition_events,
                relations: &transition_relations,
            },
        )?;

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

    // Separate path arguments mirror the CLI/export surface; CompatExportPaths
    // keeps the implementation typed after the public wrapper boundary.
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
        self.export_control_plane_compat_paths(
            repo_root,
            CompatExportPaths {
                claims: claims_path,
                insights: insights_path,
                experiments: experiments_path,
                binaries: binaries_path,
                theorems: theorems_path,
                theorems_mirror: theorems_mirror_path,
            },
        )
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
        let incoming_experiment_ids = experiments
            .iter()
            .map(|experiment| experiment.id.as_str())
            .collect::<BTreeSet<_>>();
        let protected_experiment_ids = {
            let mut statement = tx.prepare(
                "SELECT experiment_id FROM claim_transition_experiments
                 UNION
                 SELECT experiment_id FROM experiment_revisions",
            )?;
            statement
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<BTreeSet<_>, _>>()?
        };
        let missing_protected_ids = protected_experiment_ids
            .iter()
            .filter(|id| !incoming_experiment_ids.contains(id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if !missing_protected_ids.is_empty() {
            bail!(
                "experiment registry omits protected canonical experiments: {}",
                missing_protected_ids.join(", ")
            );
        }

        // The execution-planning build reaches this path with rendered registry
        // text, which is the same mirror-to-canonical direction the
        // index-control-plane guard covers. Capture the canonical-only column
        // values first so the delete-and-reinsert below cannot null them.
        let preserved = capture_preserved_values(&tx)?;

        // Rebuild the derived claim-to-experiment join before replacing the
        // experiment rows. Transition evidence and revision history retain
        // their direct foreign keys and require their referenced experiments
        // to remain present in the incoming registry.
        tx.execute("DELETE FROM claim_experiment_refs", [])?;
        tx.execute(
            "DELETE FROM experiments_cp
             WHERE id NOT IN (
                 SELECT experiment_id FROM claim_transition_experiments
                 UNION
                 SELECT experiment_id FROM experiment_revisions
             )",
            [],
        )?;
        tx.execute(
            "DELETE FROM control_plane_meta WHERE kind = 'experiments'",
            [],
        )?;
        write_registry_snapshot(&tx, repo_root, "experiments", source_path, raw, &indexed_at)?;
        for experiment in &experiments {
            tx.execute(
                "INSERT INTO experiments_cp(id, title, status, binary_name, claim_refs_json, status_note, compat_toml_text)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    status = excluded.status,
                    binary_name = excluded.binary_name,
                    claim_refs_json = excluded.claim_refs_json,
                    status_note = excluded.status_note,
                    compat_toml_text = excluded.compat_toml_text",
                params![
                    experiment.id,
                    experiment.title,
                    experiment.status,
                    experiment.binary,
                    serde_json::to_string(&experiment.claim_refs)?,
                    experiment.status_note,
                    experiment.compat_toml_text
                ],
            )?;
            for claim_id in &experiment.claim_refs {
                tx.execute(
                    "INSERT INTO claim_experiment_refs (claim_id, experiment_id)
                     VALUES (?1, ?2)
                     ON CONFLICT(claim_id, experiment_id) DO NOTHING",
                    params![claim_id, experiment.id],
                )?;
            }
        }
        tx.execute(
            "INSERT INTO control_plane_meta(kind, compat_toml_text)
             VALUES(?1, ?2)",
            params!["experiments", experiments_meta_toml],
        )?;
        restore_preserved_values(&tx, &preserved)?;
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

    pub fn verify_control_plane_compat_exports_paths(
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
        let transition_events_path = repo_root.join("registry/claim_transitions.toml");
        let transition_relations_path = repo_root.join("registry/claim_relations.toml");
        let transition_checks =
            if scalar_count(&self.conn, "SELECT COUNT(*) FROM claim_transition_events")? != 0
                || transition_events_path.exists()
                || transition_relations_path.exists()
            {
                let (transition_events, transition_relations) =
                    self.claim_transition_compat_texts()?;
                Some((transition_events, transition_relations))
            } else {
                None
            };
        let mut failures = Vec::new();
        for (path, expected) in checks {
            if !path.exists() {
                failures.push(format!("missing compatibility export {}", path.display()));
                continue;
            }
            let actual = load_text(path)?;
            if actual != compat_render::normalized_export_text(expected) {
                failures.push(format!(
                    "stale compatibility export {} relative to {}",
                    path.display(),
                    repo_root.display()
                ));
            }
        }
        if let Some((transition_events, transition_relations)) = transition_checks {
            for (path, expected) in [
                (transition_events_path.as_path(), transition_events.as_str()),
                (
                    transition_relations_path.as_path(),
                    transition_relations.as_str(),
                ),
            ] {
                if !path.exists() {
                    failures.push(format!("missing compatibility export {}", path.display()));
                    continue;
                }
                let actual = load_text(path)?;
                if actual != compat_render::normalized_export_text(expected) {
                    failures.push(format!(
                        "stale compatibility export {} relative to {}",
                        path.display(),
                        repo_root.display()
                    ));
                }
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

    // Separate path arguments mirror the CLI/verify surface; CompatExportPaths
    // keeps the implementation typed after the public wrapper boundary.
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
        self.verify_control_plane_compat_exports_paths(
            repo_root,
            CompatExportPaths {
                claims: claims_path,
                insights: insights_path,
                experiments: experiments_path,
                binaries: binaries_path,
                theorems: theorems_path,
                theorems_mirror: theorems_mirror_path,
            },
        )
    }

    pub fn verify_control_plane_invariants(&self, repo_root: &Path) -> Result<()> {
        let mut failures = Vec::new();
        if let Err(error) = self.verify_claim_transition_invariants() {
            failures.push(error.to_string());
        }
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

        if let Err(error) = validate_theorem_identities(&self.conn, repo_root) {
            failures.push(error.to_string());
        }

        for theorem in self.list_theorems()? {
            if !repo_root.join(theorem.proof_path.as_str()).exists() {
                failures.push(format!(
                    "{} proof path missing on disk: {}",
                    theorem.id, theorem.proof_path
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
            if actual != compat_render::normalized_export_text(expected) {
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
            if actual != compat_render::normalized_export_text(&expected) {
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
            insights: render_insights_registry(&self.list_insights_for_compat()?),
            experiments: render_experiments_registry(
                &experiments_meta,
                &self.list_experiments_for_compat()?,
            ),
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

    fn list_insights_for_compat(&self) -> Result<Vec<types::InsightCompatRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, status, claim_refs_json, status_note, compat_toml_text
             FROM insights ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let claim_refs_json: String = row.get(3)?;
            Ok(types::InsightCompatRecord {
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

    fn list_experiments_for_compat(&self) -> Result<Vec<types::ExperimentCompatRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, status, binary_name, claim_refs_json, status_note, compat_toml_text
             FROM experiments_cp ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let claim_refs_json: String = row.get(4)?;
            Ok(types::ExperimentCompatRecord {
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
            "SELECT COALESCE(theorem_identities.stable_id, 'THM-LEGACY-' || theorems.id),
                    theorems.id, theorems.title, theorems.proof_path, theorems.status,
                    COALESCE(theorem_identities.identity_kind, 'unresolved'),
                    theorems.linked_claim_ids_json, theorems.source
             FROM theorems
             LEFT JOIN theorem_identities
               ON theorem_identities.legacy_name = theorems.id
             ORDER BY theorem_identities.stable_id, theorems.id",
        )?;
        let rows = stmt.query_map([], |row| {
            let links: String = row.get(6)?;
            Ok(TheoremRecord {
                id: row.get(0)?,
                legacy_name: row.get(1)?,
                title: row.get(2)?,
                proof_path: Utf8PathBuf::from(row.get::<_, String>(3)?),
                status: row.get(4)?,
                identity_kind: row.get(5)?,
                linked_claim_ids: serde_json::from_str(&links).unwrap_or_default(),
                source: row.get(7)?,
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

    pub fn register_local_artifact(
        &mut self,
        repo_root: &Path,
        registration: &LocalArtifactRegistration<'_>,
    ) -> Result<usize> {
        let LocalArtifactRegistration {
            id,
            key,
            title,
            citation,
            paths,
            lane_name,
            source_refs,
            actor,
            reason,
        } = *registration;
        for (field_name, value) in [
            ("id", id),
            ("key", key),
            ("title", title),
            ("citation", citation),
            ("lane", lane_name),
        ] {
            if value.trim().is_empty() {
                bail!("local artifact {field_name} must not be empty");
            }
            if !value.is_ascii() {
                bail!("local artifact {field_name} must contain ASCII only");
            }
        }
        if !matches!(
            lane_name,
            "datasets" | "papers_pdf" | "slides_artifacts" | "web_references"
        ) {
            bail!("unsupported artifact lane {lane_name}");
        }
        if paths.is_empty() {
            bail!("local artifact registration requires at least one path");
        }

        let mut relative_paths = BTreeSet::new();
        for raw_path in paths {
            if raw_path.trim().is_empty() || !raw_path.is_ascii() {
                bail!("artifact paths must be non-empty ASCII strings");
            }
            let relative_path = Path::new(raw_path);
            if relative_path.is_absolute()
                || relative_path.components().any(|component| {
                    matches!(
                        component,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
            {
                bail!("artifact path must be repository-relative: {raw_path}");
            }
            let full_path = repo_root.join(relative_path);
            let metadata = fs::symlink_metadata(&full_path)
                .with_context(|| format!("inspect artifact path {}", full_path.display()))?;
            if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
                bail!("artifact path is not a regular file: {raw_path}");
            }
            relative_paths.insert(raw_path.clone());
        }
        if relative_paths.is_empty() {
            bail!("local artifact registration requires distinct paths");
        }
        for source_ref in source_refs {
            if source_ref.trim().is_empty() || !source_ref.is_ascii() {
                bail!("artifact source references must be non-empty ASCII strings");
            }
        }
        if let Some(actor) = actor
            && (actor.trim().is_empty() || !actor.is_ascii())
        {
            bail!("artifact actor must be a non-empty ASCII string");
        }
        if let Some(reason) = reason
            && (reason.trim().is_empty() || !reason.is_ascii())
        {
            bail!("artifact reason must be a non-empty ASCII string");
        }

        let relative_paths = relative_paths.into_iter().collect::<Vec<_>>();
        let canonical_path = relative_paths
            .first()
            .expect("non-empty artifact paths after validation");
        let tx = self.conn.transaction()?;
        let conflicting_id = tx
            .query_row(
                "SELECT id FROM artifacts WHERE key = ?1 AND id <> ?2",
                params![key, id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        if let Some(conflicting_id) = conflicting_id {
            bail!("artifact key {key} already belongs to {conflicting_id}");
        }

        tx.execute(
            "INSERT INTO artifacts (
                id, key, title, citation, status,
                minimum_requirement_met, canonical_functional_url, canonical_download_path
            ) VALUES (?1, ?2, ?3, ?4, 'downloaded', 1, NULL, ?5)
            ON CONFLICT(id) DO UPDATE SET
                key = excluded.key,
                title = excluded.title,
                citation = excluded.citation,
                status = excluded.status,
                minimum_requirement_met = excluded.minimum_requirement_met,
                canonical_functional_url = excluded.canonical_functional_url,
                canonical_download_path = excluded.canonical_download_path",
            params![id, key, title, citation, canonical_path],
        )?;
        tx.execute(
            "DELETE FROM artifact_paths WHERE artifact_id = ?1 AND relation = 'downloaded'",
            params![id],
        )?;
        tx.execute(
            "DELETE FROM lane_assignments WHERE artifact_id = ?1",
            params![id],
        )?;
        tx.execute(
            "DELETE FROM record_sources WHERE entity_kind = 'artifact' AND entity_id = ?1",
            params![id],
        )?;
        for path in &relative_paths {
            tx.execute(
                "INSERT INTO artifact_paths (artifact_id, path, relation) VALUES (?1, ?2, 'downloaded')",
                params![id, path],
            )?;
        }
        tx.execute(
            "INSERT INTO lane_assignments (artifact_id, lane_name) VALUES (?1, ?2)",
            params![id, lane_name],
        )?;
        for source_ref in source_refs {
            tx.execute(
                "INSERT INTO record_sources (entity_kind, entity_id, source_ref) VALUES ('artifact', ?1, ?2)",
                params![id, source_ref],
            )?;
        }
        let details = serde_json::json!({
            "artifact_id": id,
            "key": key,
            "paths": &relative_paths,
            "lane": lane_name,
            "source_refs": source_refs,
            "actor": actor,
            "reason": reason,
        });
        tx.execute(
            "INSERT INTO export_runs (
                action, created_at, artifact_count, document_count, details_json
            ) VALUES ('register-local-artifact', ?1, 1, 0, ?2)",
            params![Utc::now().to_rfc3339(), details.to_string()],
        )?;
        tx.commit()?;
        Ok(relative_paths.len())
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
        self.upsert_roadmap_item_columns(item, "[]", "")
    }

    /// Insert or replace a roadmap item including claim and insight links.
    pub fn upsert_roadmap_item_with_links(&self, item: &RoadmapItemWithLinks<'_>) -> Result<()> {
        let base = RoadmapItem {
            id: item.id,
            name: item.name,
            priority: item.priority,
            status: item.status,
            status_token: item.status_token,
            description: item.description,
            sprint: item.sprint,
            dependencies_json: item.dependencies_json,
            acceptance_criteria_json: item.acceptance_criteria_json,
            primary_outputs_json: item.primary_outputs_json,
            evidence_refs_json: item.evidence_refs_json,
            lacunae_json: item.lacunae_json,
        };
        self.upsert_roadmap_item_columns(&base, item.claims_json, item.insight)
    }

    fn upsert_roadmap_item_columns(
        &self,
        item: &RoadmapItem<'_>,
        claims_json: &str,
        insight: &str,
    ) -> Result<()> {
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
                claims_json,
                insight,
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
        self.upsert_action_item_in_table("todo_items", item, "[]")
    }

    /// Insert or replace a todo item including evidence references.
    pub fn upsert_todo_item_with_evidence(&self, item: &ActionItemWithEvidence<'_>) -> Result<()> {
        let base = ActionItem {
            id: item.id,
            area: item.area,
            title: item.title,
            description: item.description,
            priority: item.priority,
            status: item.status,
            status_token: item.status_token,
            dependencies_json: item.dependencies_json,
            acceptance_criteria_json: item.acceptance_criteria_json,
        };
        self.upsert_action_item_in_table("todo_items", &base, item.evidence_refs_json)
    }

    fn upsert_action_item_in_table(
        &self,
        table_name: &str,
        item: &ActionItem<'_>,
        evidence_refs_json: &str,
    ) -> Result<()> {
        self.conn.execute(
            &format!(
                "INSERT OR REPLACE INTO {table_name}
             (id, area, title, description, priority, status, status_token,
              dependencies_json, acceptance_criteria_json, evidence_refs_json, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10, datetime('now'))"
            ),
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
                evidence_refs_json,
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

    /// Rewrite every reference to a moved source file across the canonical
    /// control plane.
    ///
    /// A claim's evidence citation and an external-source artifact contract
    /// both name a repository path, and `governance-verify
    /// external-source-operational-contracts` fails closed when a declared
    /// `artifact_contract_path` no longer exists. Collapsing a binary cluster
    /// moves those files, so the two move together here.
    ///
    /// Matching is token-bounded on the same rule as the execution-target
    /// rename, which protects a prefix: rewriting `src/bin/solar_wind_ic.rs`
    /// leaves `src/bin/solar_wind_ic_helpers.rs` alone.
    pub fn retarget_source_path(
        &mut self,
        request: SourcePathRetarget<'_>,
    ) -> Result<SourcePathRetargetSummary> {
        let mut summary = SourcePathRetargetSummary::default();
        for target in RETARGET_FIELDS {
            let select = format!(
                "SELECT id, {field} FROM {table} WHERE {field} LIKE ?1",
                field = target.field,
                table = target.table,
            );
            let pending: Vec<(String, String)> = {
                let mut stmt = self.conn.prepare(&select)?;
                let rows = stmt.query_map(params![format!("%{}%", request.from)], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?;
                collect_rows(rows)?
            };
            for (id, text) in pending {
                let rewritten = replace_token(&text, request.from, request.to);
                if rewritten == text {
                    continue;
                }
                let revision = self.entity_update_field(
                    &id,
                    &rewritten,
                    request.actor,
                    request.reason,
                    *target,
                )?;
                summary.revisions.push(revision);
            }
        }

        let tx = self.conn.transaction()?;
        {
            let pending: Vec<(String, String, i64, String)> = {
                let mut stmt = tx.prepare(
                    "SELECT dossier_id, relation, ord, value \
                     FROM external_source_dossier_values WHERE value LIKE ?1",
                )?;
                let rows = stmt.query_map(params![format!("%{}%", request.from)], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })?;
                collect_rows(rows)?
            };
            for (dossier_id, relation, ord, value) in pending {
                let rewritten = replace_token(&value, request.from, request.to);
                if rewritten == value {
                    continue;
                }
                tx.execute(
                    "UPDATE external_source_dossier_values SET value = ?1 \
                     WHERE dossier_id = ?2 AND relation = ?3 AND ord = ?4",
                    params![rewritten, dossier_id, relation, ord],
                )?;
                summary.contract_paths_updated += 1;
            }
        }
        tx.commit()?;

        Ok(summary)
    }

    /// Rewrite every reference to a renamed execution target across the
    /// canonical control plane.
    ///
    /// A reproduction command reaches the registry in two syntactic positions:
    /// as the argument of `cargo run --bin <target>`, and as a bare name in an
    /// experiment `binary` field or a claim's `Binary:` citation. Both move
    /// together here, because a rename that reaches one and not the other
    /// leaves a claim citing evidence no command reproduces.
    ///
    /// Each touched column goes through `entity_update_field`, so the rename
    /// appends a revision row per record naming the actor and the prev/new
    /// content hashes.
    pub fn retarget_execution_target(
        &mut self,
        request: ExecutionTargetRetarget<'_>,
    ) -> Result<ExecutionTargetRetargetSummary> {
        let mut summary = ExecutionTargetRetargetSummary::default();
        for target in RETARGET_FIELDS {
            let select = format!(
                "SELECT id, {field} FROM {table} WHERE {field} LIKE ?1",
                field = target.field,
                table = target.table,
            );
            let pending: Vec<(String, String)> = {
                let mut stmt = self.conn.prepare(&select)?;
                let rows = stmt.query_map(params![format!("%{}%", request.from)], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?;
                collect_rows(rows)?
            };
            for (id, text) in pending {
                let rewritten = rewrite_execution_target(&text, request.from, request.to);
                if rewritten == text {
                    continue;
                }
                let revision = self.entity_update_field(
                    &id,
                    &rewritten,
                    request.actor,
                    request.reason,
                    *target,
                )?;
                summary.revisions.push(revision);
            }
        }
        Ok(summary)
    }

    /// Rows a retarget would rewrite, as (table, field, id). Reads only, so an
    /// operator can size a rename before committing to it.
    pub fn preview_execution_target_retarget(
        &self,
        from: &str,
        to: &str,
    ) -> Result<Vec<(String, String, String)>> {
        let mut pending = Vec::new();
        for target in RETARGET_FIELDS {
            let select = format!(
                "SELECT id, {field} FROM {table} WHERE {field} LIKE ?1",
                field = target.field,
                table = target.table,
            );
            let mut stmt = self.conn.prepare(&select)?;
            let rows = stmt.query_map(params![format!("%{}%", from)], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            for (id, text) in collect_rows(rows)? {
                if rewrite_execution_target(&text, from, to) != text {
                    pending.push((target.table.to_string(), target.field.to_string(), id));
                }
            }
        }
        Ok(pending)
    }

    /// Names `binaries_sync` would add and remove, without writing.
    pub fn preview_binaries_sync(
        &self,
        declared: &[BinaryRecord],
    ) -> Result<(Vec<String>, Vec<String>)> {
        let existing: BTreeSet<String> = self
            .list_binaries()?
            .into_iter()
            .map(|record| record.name)
            .collect();
        let declared_names: BTreeSet<String> =
            declared.iter().map(|record| record.name.clone()).collect();
        Ok((
            declared_names.difference(&existing).cloned().collect(),
            existing.difference(&declared_names).cloned().collect(),
        ))
    }

    /// Reconcile `binaries_cp` against the binary targets cargo declares.
    ///
    /// Rows carry curated descriptions, so a name already present keeps its
    /// row untouched; only absent names are inserted and stale names deleted.
    /// This is the discovery half of the registry, and it drifts whenever a
    /// `[[bin]]` is added, removed, or folded into a subcommand tree.
    pub fn binaries_sync(&mut self, declared: &[BinaryRecord]) -> Result<BinariesSyncSummary> {
        let existing: BTreeSet<String> = self
            .list_binaries()?
            .into_iter()
            .map(|record| record.name)
            .collect();
        let declared_names: BTreeSet<String> =
            declared.iter().map(|record| record.name.clone()).collect();

        let mut summary = BinariesSyncSummary {
            retained: existing.intersection(&declared_names).count(),
            ..Default::default()
        };

        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        for name in existing.difference(&declared_names) {
            tx.execute("DELETE FROM binaries_cp WHERE name = ?1", params![name])?;
            summary.removed.push(name.clone());
        }
        for record in declared {
            if existing.contains(&record.name) {
                continue;
            }
            tx.execute(
                "INSERT INTO binaries_cp(name, crate_name, description, experiment_id, source)
                 VALUES(?1, ?2, ?3, ?4, ?5)",
                params![
                    record.name,
                    record.crate_name,
                    record.description,
                    record.experiment,
                    record.source
                ],
            )?;
            summary.added.push(record.name.clone());
        }
        tx.commit()?;
        Ok(summary)
    }

    /// Insert or replace a next-action item.
    pub fn upsert_next_action(&self, item: &ActionItem<'_>) -> Result<()> {
        self.upsert_action_item_in_table("next_action_items", item, "[]")
    }

    /// Insert or replace a next-action item including evidence references.
    pub fn upsert_next_action_with_evidence(
        &self,
        item: &ActionItemWithEvidence<'_>,
    ) -> Result<()> {
        let base = ActionItem {
            id: item.id,
            area: item.area,
            title: item.title,
            description: item.description,
            priority: item.priority,
            status: item.status,
            status_token: item.status_token,
            dependencies_json: item.dependencies_json,
            acceptance_criteria_json: item.acceptance_criteria_json,
        };
        self.upsert_action_item_in_table("next_action_items", &base, item.evidence_refs_json)
    }

    pub fn delete_next_action(&self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM next_action_items WHERE id = ?1", params![id])?;
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
            let expected = compat_render::normalized_export_text(&expected);
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

    fn parse_registry_snapshot(&self, kind: &str) -> Result<Option<Value>> {
        self.registry_snapshot(kind)?
            .map(|raw| {
                toml::from_str::<Value>(&raw)
                    .with_context(|| format!("parse {kind} registry snapshot"))
            })
            .transpose()
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
                compat_toml_text: row.get::<_, String>(5).unwrap_or_default(),
            })
        };
        let mut out = Vec::new();
        if let Some(s) = status {
            let mut stmt = self.conn.prepare(
                "SELECT id, title, status, binary_name, claim_refs_json, compat_toml_text
                 FROM experiments_cp WHERE status = ?1 ORDER BY id LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![s, limit as i64], map_row)?;
            for r in rows {
                out.push(r?);
            }
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT id, title, status, binary_name, claim_refs_json, compat_toml_text
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
    compat_child_table, compat_json_string_array, compat_root_table, compat_table_array,
    compat_table_bool, compat_table_string, compat_toml_quote, compat_toml_string_array,
    host_for_url, load_registry_table_toml, load_text, load_toml_text, toml_array_to_json_string,
    trim_trailing_blank_lines,
};

// External-source contracts and dossiers (loaders +
// compat-TOML/Markdown renderers) live in the `external_sources`
// submodule. Items are pub(crate).
mod external_sources;
use external_sources::{
    load_external_source_contracts_from_registry, load_external_source_dossiers_from_registry,
    render_external_source_contracts_registry, render_external_source_dossier_markdown,
    render_external_source_dossiers_registry,
};

fn collect_rows<T>(
    rows: rusqlite::MappedRows<'_, impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>>,
) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

// Registry TOML loaders for claim, insight, and experiment records
// (load_claims_from_registry, load_insights_from_registry,
// load_experiments_from_registry) live in the `registry_loaders`
// submodule. Items are pub(crate).
mod registry_loaders;
use registry_loaders::{
    load_claims_from_registry, load_experiments_from_registry, load_insights_from_registry,
};

// Workspace binary inventory (load_binaries_from_registry + cargo
// metadata discovery + manifest-walk fallback + merge) lives in the
// `binaries_loader` submodule. Items are pub(crate); helpers like
// member_manifest_path stay private to the submodule.
mod binaries_loader;
use binaries_loader::{load_binaries_from_registry, merge_workspace_binaries};

// Status-token normalization helpers (normalize_claim_record,
// normalize_claim_status, normalize_insight_status,
// match_case_insensitive, merge_status_note) live in the
// `status_normalize` submodule. Items are pub(crate).
mod status_normalize;

// Claim <-> proof correlation and per-row compat-export rendering
// (canonical_formal_proof_for_claim, proof_entry_priority,
// render_normalized_claim_compat_toml,
// render_normalized_insight_compat_toml, extract_proof_paths,
// link_claims_for_proof, normalized_claim_id_from_theorem_stem) live
// in the `claim_proofs` submodule. Items are pub(crate).
mod claim_proofs;
pub use claim_proofs::is_formal_proof_disposition;
use claim_proofs::render_normalized_claim_compat_toml;

// Rocq proof inventory and theorem-table rendering (load_proof_inventory,
// load_theorems_from_inventory, normalize_claims_against_proof_inventory,
// render_theorem_markdown) live in the `proof_inventory` submodule.
// Items are pub(crate).
mod proof_inventory;
use proof_inventory::{
    load_proof_inventory, load_theorems_from_inventory, normalize_claims_against_proof_inventory,
    render_theorem_markdown,
};

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
    render_binaries_registry, render_claims_registry, render_experiments_registry,
    render_insights_registry, write_text,
};

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{Connection, params};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn retarget_consumes_the_existing_argument_separator() {
        let text = "cargo run --release --bin heliosphere-fa-attribution -- --mission cluster";
        assert_eq!(
            rewrite_execution_target(
                text,
                "heliosphere-fa-attribution",
                "heliosphere fa-attribution"
            ),
            "cargo run --release --bin heliosphere -- fa-attribution --mission cluster"
        );
    }

    #[test]
    fn retarget_supplies_a_separator_when_the_command_has_no_arguments() {
        let text = "cargo run --bin heliosphere-zd-audit";
        assert_eq!(
            rewrite_execution_target(text, "heliosphere-zd-audit", "heliosphere zd-audit"),
            "cargo run --bin heliosphere -- zd-audit"
        );
    }

    #[test]
    fn retarget_rewrites_bare_citations_and_binary_fields() {
        let text = "binary = \"heliosphere-mms-multiday\"\nBinary: heliosphere-mms-multiday.";
        assert_eq!(
            rewrite_execution_target(text, "heliosphere-mms-multiday", "heliosphere mms-multiday"),
            "binary = \"heliosphere mms-multiday\"\nBinary: heliosphere mms-multiday."
        );
    }

    // A shorter target name is a prefix of a longer one, so a plain substring
    // replace would rewrite the longer name's head and leave a corrupt tail.
    #[test]
    fn retarget_leaves_a_longer_target_sharing_the_prefix_intact() {
        let text = "run heliosphere-r16-ablation then heliosphere-r16-ablation-sparse";
        assert_eq!(
            rewrite_execution_target(
                text,
                "heliosphere-r16-ablation",
                "heliosphere r16-ablation"
            ),
            "run heliosphere r16-ablation then heliosphere-r16-ablation-sparse"
        );
    }

    #[test]
    fn retarget_is_a_no_op_when_the_target_is_absent() {
        let text = "cargo run --bin themis-staples-score-export";
        assert_eq!(
            rewrite_execution_target(text, "heliosphere-norm-dump", "heliosphere norm-dump"),
            text
        );
    }

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
            RegistryImportPaths {
                claims: &fixture.claims,
                insights: &fixture.insights,
                experiments: &fixture.experiments,
                binaries: &fixture.binaries,
                rocq_project: &fixture.rocq_project,
            },
            ReimportOptions::destructive(&fixture.db),
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

    /// A control-plane reindex rewrites five snapshot kinds and leaves the
    /// others alone. The roadmap snapshot feeds `render_roadmap_compat_toml`;
    /// deleting it turned every `supersedes` and `companion_docs` array into
    /// an empty export on the next planning run.
    #[test]
    fn control_plane_reindex_preserves_unrelated_registry_snapshots() -> Result<()> {
        let fixture = make_test_workspace("snapshot_preserve")?;
        let mut store = ProvenanceStore::open(&fixture.db)?;
        let roadmap_path = fixture.root.join("registry/roadmap.toml");
        let roadmap_text = "[roadmap]\nsupersedes = [\"plans/old.toml\"]\n";
        write_text(&roadmap_path, roadmap_text)?;
        store.record_registry_snapshot(&fixture.root, "roadmap", &roadmap_path, roadmap_text)?;

        store.reindex_control_plane_from_registries(
            &fixture.root,
            RegistryImportPaths {
                claims: &fixture.claims,
                insights: &fixture.insights,
                experiments: &fixture.experiments,
                binaries: &fixture.binaries,
                rocq_project: &fixture.rocq_project,
            },
            ReimportOptions::destructive(&fixture.db),
        )?;

        assert_eq!(
            store.registry_snapshot("roadmap")?.as_deref(),
            Some(roadmap_text),
            "the roadmap snapshot must survive a control-plane reindex"
        );
        for kind in table_ops::CONTROL_PLANE_SNAPSHOT_KINDS {
            assert!(
                store.registry_snapshot(kind)?.is_some(),
                "{kind} snapshot is rewritten by the reindex"
            );
        }
        Ok(())
    }

    /// `na_empirical:<rationale>` is a reviewer decision, not a path. It
    /// must come back byte-identical through update, export, reindex and a
    /// second export, or the next backfill relinks a proof by numeric prefix.
    #[test]
    fn formal_proof_disposition_round_trips_through_export_and_reindex() -> Result<()> {
        let fixture = make_test_workspace("disposition_round_trip")?;
        let mut store = ProvenanceStore::open(&fixture.db)?;
        store.reindex_control_plane_from_registries(
            &fixture.root,
            RegistryImportPaths {
                claims: &fixture.claims,
                insights: &fixture.insights,
                experiments: &fixture.experiments,
                binaries: &fixture.binaries,
                rocq_project: &fixture.rocq_project,
            },
            ReimportOptions::destructive(&fixture.db),
        )?;
        let disposition = "na_empirical:ROC-AUC on THEMIS-A minutes, no theorem to prove";
        store.claim_update_formal_proof("C-001", disposition, "test", Some("review"))?;
        assert_eq!(store.claim_formal_proof("C-001")?.as_deref(), Some(disposition));

        let first_export = store.control_plane_compat_text(ControlPlaneCompatKind::Claims)?;
        assert!(
            first_export.contains(&format!("formal_proof = \"{disposition}\"")),
            "export must carry the disposition:\n{first_export}"
        );
        write_text(&fixture.claims, &first_export)?;

        store.reindex_control_plane_from_registries(
            &fixture.root,
            RegistryImportPaths {
                claims: &fixture.claims,
                insights: &fixture.insights,
                experiments: &fixture.experiments,
                binaries: &fixture.binaries,
                rocq_project: &fixture.rocq_project,
            },
            ReimportOptions::destructive(&fixture.db),
        )?;
        assert_eq!(
            store.claim_formal_proof("C-001")?.as_deref(),
            Some(disposition),
            "reindex must keep the disposition rather than resolving it to NULL"
        );
        let second_export = store.control_plane_compat_text(ControlPlaneCompatKind::Claims)?;
        assert_eq!(first_export, second_export);

        // A proof path that exists still resolves; one that does not is dropped as before.
        store.claim_update_formal_proof("C-001", "proofs/verified/C001_Test.v", "test", None)?;
        write_text(
            &fixture.claims,
            &store.control_plane_compat_text(ControlPlaneCompatKind::Claims)?,
        )?;
        store.reindex_control_plane_from_registries(
            &fixture.root,
            RegistryImportPaths {
                claims: &fixture.claims,
                insights: &fixture.insights,
                experiments: &fixture.experiments,
                binaries: &fixture.binaries,
                rocq_project: &fixture.rocq_project,
            },
            ReimportOptions::destructive(&fixture.db),
        )?;
        assert_eq!(
            store.claim_formal_proof("C-001")?.as_deref(),
            Some("proofs/verified/C001_Test.v")
        );
        Ok(())
    }

    #[test]
    fn formal_proof_disposition_tokens_are_recognized() {
        for value in [
            "na_empirical",
            "na_empirical:rationale with spaces",
            "na_observational:ACE MAG",
            "na_methodology:simulation",
            "pending",
            "pending:reviewed_pending",
            "external:arXiv:1234.5678",
        ] {
            assert!(is_formal_proof_disposition(value), "{value}");
        }
        for value in [
            "",
            "proofs/verified/C001_Test.v",
            "na_empirically",
            "pendingx",
            "externalarXiv",
        ] {
            assert!(!is_formal_proof_disposition(value), "{value}");
        }
    }

    #[test]
    fn replace_control_plane_experiments_from_registry_text_replaces_rows() -> Result<()> {
        let fixture = make_test_workspace("replace_experiments")?;
        let mut store = ProvenanceStore::open(&fixture.db)?;
        store.reindex_control_plane_from_registries(
            &fixture.root,
            RegistryImportPaths {
                claims: &fixture.claims,
                insights: &fixture.insights,
                experiments: &fixture.experiments,
                binaries: &fixture.binaries,
                rocq_project: &fixture.rocq_project,
            },
            ReimportOptions::destructive(&fixture.db),
        )?;
        store.build_crossrefs()?;

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

    #[test]
    fn replace_preserves_transition_experiments_and_rebuilds_claim_refs() -> Result<()> {
        let fixture = make_test_workspace("replace_transition_experiments")?;
        let mut store = ProvenanceStore::open(&fixture.db)?;
        store.reindex_control_plane_from_registries(
            &fixture.root,
            RegistryImportPaths {
                claims: &fixture.claims,
                insights: &fixture.insights,
                experiments: &fixture.experiments,
                binaries: &fixture.binaries,
                rocq_project: &fixture.rocq_project,
            },
            ReimportOptions::destructive(&fixture.db),
        )?;
        store.build_crossrefs()?;
        store.conn.execute(
            "INSERT INTO claim_transition_events (
                 transition_key, source_claim_id, expected_prior_status,
                 experiment_verdict, proposed_claim_status, exercised_falsifier,
                 rationale, actor, reason, transition_ts_utc,
                 transition_spec_sha256, expected_source_state_sha256,
                 expected_claim_id_max
             ) VALUES (
                 'replace-transition-test', 'C-001', 'Verified', 'Inconclusive',
                 'Provisional', 'test falsifier', 'test rationale', 'test actor',
                 'test reason', '2026-08-04T00:00:00Z', 'spec-hash',
                 'source-hash', 1
             )",
            [],
        )?;
        store.conn.execute(
            "INSERT INTO claim_transition_experiments (transition_event_id, experiment_id)
             VALUES (1, 'E-001')",
            [],
        )?;

        let replacement = r#"
[experiments]
authoritative = true
status_allowlist = ["active", "planned", "blocked", "deprecated"]

[[experiment]]
id = "E-001"
title = "Retained transition experiment"
status = "active"
binary = "mini-bin"
claim_refs = ["C-001"]

[[experiment]]
id = "E-002"
title = "New experiment"
status = "active"
binary = "mini-bin"
claim_refs = ["C-001"]
"#;

        store.replace_control_plane_experiments_from_registry_text(
            &fixture.root,
            &fixture.experiments,
            replacement,
        )?;

        let rendered = store.control_plane_compat_text(ControlPlaneCompatKind::Experiments)?;
        assert!(rendered.contains("id = \"E-001\""));
        assert!(rendered.contains("Retained transition experiment"));
        assert!(rendered.contains("id = \"E-002\""));
        let reference_count: i64 = store.conn.query_row(
            "SELECT COUNT(*) FROM claim_experiment_refs WHERE claim_id = 'C-001'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(reference_count, 2);
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

    /// Fixture with the canonical-only values the compatibility mirrors omit:
    /// an insight status_note (I-212 has no compatibility key at all), a claim
    /// formal_proof, and one claim transition event.
    fn seed_canonical_only_values(fixture: &TestWorkspace) -> Result<ProvenanceStore> {
        write_text(
            &fixture.insights,
            r#"[[insight]]
id = "I-212"
title = "Matched receptive field reverses the associator ranking"
status = "verified"
claims = ["C-001"]
"#,
        )?;
        let mut store = ProvenanceStore::open(&fixture.db)?;
        store.reindex_control_plane_from_registries(
            &fixture.root,
            RegistryImportPaths {
                claims: &fixture.claims,
                insights: &fixture.insights,
                experiments: &fixture.experiments,
                binaries: &fixture.binaries,
                rocq_project: &fixture.rocq_project,
            },
            ReimportOptions::bootstrap(),
        )?;
        store.insight_update_status_note("I-212", SEEDED_NOTE, "test", Some("seed"))?;
        store.claim_update_formal_proof("C-001", SEEDED_PROOF, "test", Some("seed"))?;
        store.conn.execute(
            "INSERT INTO claim_transition_events (
                 transition_key, source_claim_id, expected_prior_status,
                 experiment_verdict, proposed_claim_status, exercised_falsifier,
                 rationale, actor, reason, transition_ts_utc,
                 transition_spec_sha256, expected_source_state_sha256,
                 expected_claim_id_max
             ) VALUES (
                 'canonical-authority-test', 'C-001', 'Verified', 'Inconclusive',
                 'Provisional', 'test falsifier', 'test rationale', 'test actor',
                 'test reason', '2026-09-01T00:00:00Z', 'spec-hash',
                 'source-hash', 1
             )",
            [],
        )?;
        Ok(store)
    }

    const SEEDED_NOTE: &str = "seeded canonical note that no compatibility TOML carries";
    const SEEDED_PROOF: &str = "na_empirical:canonical-only disposition";
    const PERMUTED_NOTE: &str = "note written during the permutation";

    fn export_to(store: &mut ProvenanceStore, fixture: &TestWorkspace) -> Result<()> {
        let theorems = fixture.root.join("docs/THEOREMS.md");
        let mirror = fixture.root.join("docs/generated/THEOREMS_REGISTRY_MIRROR.md");
        store.export_control_plane_compat_paths(
            &fixture.root,
            CompatExportPaths {
                claims: &fixture.claims,
                insights: &fixture.insights,
                experiments: &fixture.experiments,
                binaries: &fixture.binaries,
                theorems: &theorems,
                theorems_mirror: &mirror,
            },
        )
    }

    /// Run the three canonical-authority operations in one ordering and report
    /// the state that must not depend on the ordering.
    fn run_ordering(order: [char; 3]) -> Result<(String, i64, i64, String)> {
        let label = format!("order_{}{}{}", order[0], order[1], order[2]);
        let fixture = make_test_workspace(&label)?;
        let mut store = seed_canonical_only_values(&fixture)?;
        for op in order {
            match op {
                'i' => {
                    store.reindex_control_plane_from_registries(
                        &fixture.root,
                        RegistryImportPaths {
                            claims: &fixture.claims,
                            insights: &fixture.insights,
                            experiments: &fixture.experiments,
                            binaries: &fixture.binaries,
                            rocq_project: &fixture.rocq_project,
                        },
                        ReimportOptions::destructive(&fixture.db),
                    )?;
                }
                'e' => export_to(&mut store, &fixture)?,
                'n' => {
                    store.insight_update_status_note(
                        "I-212",
                        PERMUTED_NOTE,
                        "test",
                        Some("permutation"),
                    )?;
                }
                other => panic!("unknown operation {other}"),
            }
        }
        let note = store
            .insight_status_note("I-212")?
            .expect("I-212 keeps its status note");
        let revisions: i64 = store.conn.query_row(
            "SELECT COUNT(*) FROM insight_revisions WHERE insight_id = 'I-212'",
            [],
            |row| row.get(0),
        )?;
        let events = store.list_claim_transition_events()?.len() as i64;
        // Export once more so the compared mirror text is produced from the same
        // final state in every ordering.
        export_to(&mut store, &fixture)?;
        // Compare the mirror by parsed content, not by bytes: splicing a live
        // column into a cached compat row appends the key while a mirror-sourced
        // row carries it in sorted position, so key order tracks provenance.
        let mirror = {
            let parsed: Value = toml::from_str(&fs::read_to_string(&fixture.insights)?)?;
            let mut normalized = parsed;
            if let Some(rows) = normalized.get_mut("insight").and_then(Value::as_array_mut) {
                for row in rows.iter_mut() {
                    if let Some(table) = row.as_table_mut() {
                        let sorted: toml::map::Map<String, Value> =
                            table.clone().into_iter().collect();
                        *table = sorted;
                    }
                }
            }
            toml::to_string(&normalized)?
        };
        // claims.toml carries formal_proof, so the mirror is authoritative for it
        // and an import that precedes any export legitimately restores the
        // inventory disposition. What must hold in every ordering is that the
        // column is never nulled; where an export ran first, the seeded value
        // round-trips exactly.
        let proof = store.claim_formal_proof("C-001")?;
        assert!(
            proof.as_deref().is_some_and(|value| !value.is_empty()),
            "{label}: claim formal_proof was nulled"
        );
        if order.iter().position(|op| *op == 'e') < order.iter().position(|op| *op == 'i') {
            assert_eq!(
                proof.as_deref(),
                Some(SEEDED_PROOF),
                "{label}: exported disposition failed to round-trip"
            );
        }
        Ok((note, revisions, events, mirror))
    }

    #[test]
    fn canonical_values_survive_every_operation_ordering() -> Result<()> {
        use std::collections::BTreeMap;
        let orderings = [
            ['i', 'e', 'n'],
            ['i', 'n', 'e'],
            ['e', 'i', 'n'],
            ['e', 'n', 'i'],
            ['n', 'i', 'e'],
            ['n', 'e', 'i'],
        ];
        // The mirror wins only when it carries its own value. An export before the
        // note edit puts the seeded note into insights.toml, so a re-import that
        // runs last restores it; that overwrite is recorded in the diff. Every
        // other ordering leaves the mirror silent on status_note, and the
        // canonical value survives.
        let mut mirrors: BTreeMap<String, String> = BTreeMap::new();
        let mut baseline_revisions: Option<i64> = None;
        for order in orderings {
            let observed = run_ordering(order)?;
            let expected_note = if order == ['e', 'n', 'i'] {
                SEEDED_NOTE
            } else {
                PERMUTED_NOTE
            };
            assert_eq!(
                observed.0, expected_note,
                "ordering {order:?} lost the insight status note"
            );
            assert_eq!(
                observed.2, 1,
                "ordering {order:?} changed the transition event count"
            );
            assert!(
                observed.1 >= 2,
                "ordering {order:?} lost insight_revisions history: {}",
                observed.1
            );
            match mirrors.get(&observed.0) {
                None => {
                    mirrors.insert(observed.0.clone(), observed.3.clone());
                }
                Some(first) => assert_eq!(
                    &observed.3, first,
                    "ordering {order:?} changed the exported mirror at an identical final state"
                ),
            }
            match baseline_revisions {
                None => baseline_revisions = Some(observed.1),
                Some(first) => {
                    assert_eq!(observed.1, first, "ordering {order:?} changed revisions");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn index_control_plane_refuses_a_populated_database() -> Result<()> {
        let fixture = make_test_workspace("refuse_populated")?;
        let mut store = seed_canonical_only_values(&fixture)?;
        let err = store
            .reindex_control_plane_from_registries(
                &fixture.root,
                RegistryImportPaths {
                    claims: &fixture.claims,
                    insights: &fixture.insights,
                    experiments: &fixture.experiments,
                    binaries: &fixture.binaries,
                    rocq_project: &fixture.rocq_project,
                },
                ReimportOptions::bootstrap(),
            )
            .expect_err("bootstrap import must refuse a populated database");
        let text = err.to_string();
        assert!(text.contains("refusing to re-import"), "{text}");
        assert!(text.contains("--allow-destructive-reimport"), "{text}");
        assert!(text.contains("claim transition events"), "{text}");
        Ok(())
    }

    #[test]
    fn destructive_reimport_writes_a_backup_and_a_diff() -> Result<()> {
        let fixture = make_test_workspace("destructive_backup")?;
        let mut store = seed_canonical_only_values(&fixture)?;
        store.reindex_control_plane_from_registries(
            &fixture.root,
            RegistryImportPaths {
                claims: &fixture.claims,
                insights: &fixture.insights,
                experiments: &fixture.experiments,
                binaries: &fixture.binaries,
                rocq_project: &fixture.rocq_project,
            },
            ReimportOptions::destructive(&fixture.db),
        )?;
        let backups = fixture
            .db
            .parent()
            .expect("db has a parent")
            .join("backups");
        let mut sqlite_backups = Vec::new();
        let mut diffs = Vec::new();
        for entry in fs::read_dir(&backups)? {
            let path = entry?.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            if name.ends_with(".diff.toml") {
                diffs.push(path);
            } else if name.ends_with(".sqlite3") {
                sqlite_backups.push(path);
            }
        }
        assert_eq!(sqlite_backups.len(), 1, "one backup database");
        assert_eq!(diffs.len(), 1, "one diff beside it");
        let backup_store = ProvenanceStore::open(&sqlite_backups[0])?;
        assert_eq!(
            backup_store.insight_status_note("I-212")?.as_deref(),
            Some(SEEDED_NOTE),
            "the backup holds the pre-import note"
        );
        let diff_text = fs::read_to_string(&diffs[0])?;
        let diff_value: Value = toml::from_str(&diff_text)?;
        assert!(diff_value.get("meta").is_some(), "{diff_text}");
        let tables = diff_value
            .get("table")
            .and_then(Value::as_array)
            .expect("diff lists tables");
        assert_eq!(tables.len(), 4, "{diff_text}");
        let insights_table = tables
            .iter()
            .find(|t| t.get("name").and_then(Value::as_str) == Some("insights"))
            .expect("insights lane present");
        let nulled = insights_table
            .get("nulled_field_ids")
            .and_then(Value::as_array)
            .expect("nulled_field_ids array");
        assert!(
            nulled.iter().any(|v| v.as_str() == Some("I-212")),
            "the diff names the insight whose status_note the mirror omits: {diff_text}"
        );
        Ok(())
    }
}

mod reimport_guard;
pub use reimport_guard::{
    ControlPlanePopulation, ReimportDiff, ReimportOptions, TableDiff, diff_path_for_backup,
    refusal_message,
};
use reimport_guard::{
    DiffLane, assert_column_mapping_total, backup_database, build_diff, capture_preserved_values,
    measure_population, restore_preserved_values,
};
