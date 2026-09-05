//! Typed empirical evidence contracts and transactional revision history.
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use anyhow::{Context, Result, bail, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ProvenanceStore;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLayer {
    SourceProposition,
    ImplementationConformance,
    PhenomenologicalMapping,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MapKind {
    Computational,
    Interpretive,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DepthStatus {
    Declared,
    NotAssessed,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InterveningMap {
    pub name: String,
    pub branch: String,
    pub description: String,
    pub kind: MapKind,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FittedParameterBranch {
    pub branch: String,
    pub count: u32,
    pub names: Vec<String>,
    pub training_boundary: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DecisiveExperiment {
    pub experiment_ids: Vec<String>,
    pub protocol_artifact: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_sha256: Option<String>,
    pub description: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FalsifierOutcomes {
    pub verification_outcomes: Vec<String>,
    pub revision_outcomes: Vec<String>,
    pub abandonment_outcomes: Vec<String>,
    pub inconclusive_outcomes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClaimEvidenceSpec {
    pub claim_id: String,
    pub evidence_layer: EvidenceLayer,
    pub depth_status: DepthStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lift_depth: Option<u32>,
    pub depth_rationale: String,
    pub intervening_maps: Vec<InterveningMap>,
    pub fitted_parameters: Vec<FittedParameterBranch>,
    pub fixed_hyperparameters: Vec<String>,
    pub decisive_experiment: DecisiveExperiment,
    pub what_would_verify_refute: FalsifierOutcomes,
}

fn require_text(value: &str, name: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "{name} must contain text");
    Ok(())
}

fn unique_text(values: &[String], name: &str, allow_empty: bool) -> Result<()> {
    ensure!(
        allow_empty || !values.is_empty(),
        "{name} must contain outcomes or references"
    );
    let mut seen = BTreeSet::new();
    for value in values {
        require_text(value, name)?;
        ensure!(seen.insert(value.trim()), "duplicate {name}: {value}");
    }
    Ok(())
}

impl ClaimEvidenceSpec {
    pub fn validate(&self) -> Result<()> {
        require_text(&self.claim_id, "claim_id")?;
        require_text(&self.depth_rationale, "depth_rationale")?;
        let mut map_names = BTreeSet::new();
        let mut depths = BTreeMap::<&str, u32>::new();
        for mapping in &self.intervening_maps {
            require_text(&mapping.name, "map name")?;
            require_text(&mapping.branch, "map branch")?;
            require_text(&mapping.description, "map description")?;
            ensure!(
                map_names.insert((mapping.branch.trim(), mapping.name.trim())),
                "duplicate map in branch"
            );
            if mapping.kind == MapKind::Interpretive {
                *depths.entry(mapping.branch.trim()).or_default() += 1;
            }
        }
        match self.depth_status {
            DepthStatus::Declared => ensure!(
                self.lift_depth == Some(depths.values().copied().max().unwrap_or(0)),
                "lift_depth must equal the maximum interpretive-map count per branch"
            ),
            DepthStatus::NotAssessed => ensure!(
                self.lift_depth.is_none(),
                "not_assessed depth requires omitted lift_depth"
            ),
        }
        let mut branches = BTreeSet::new();
        ensure!(
            !self.fitted_parameters.is_empty(),
            "declare fitted_parameters even for zero fitted parameters"
        );
        for fitted in &self.fitted_parameters {
            require_text(&fitted.branch, "fitted branch")?;
            require_text(&fitted.training_boundary, "training boundary")?;
            ensure!(
                branches.insert(fitted.branch.trim()),
                "duplicate fitted branch"
            );
            unique_text(&fitted.names, "parameter names", true)?;
            ensure!(
                fitted.names.len() == fitted.count as usize,
                "parameter count must equal names length"
            );
        }
        unique_text(&self.fixed_hyperparameters, "fixed hyperparameters", true)?;
        unique_text(
            &self.decisive_experiment.experiment_ids,
            "experiment_ids",
            false,
        )?;
        require_text(
            &self.decisive_experiment.description,
            "experiment description",
        )?;
        let protocol = std::path::Path::new(&self.decisive_experiment.protocol_artifact);
        ensure!(
            !protocol.as_os_str().is_empty()
                && protocol
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_))),
            "protocol_artifact must be a repository-relative path"
        );
        if let Some(digest) = &self.decisive_experiment.protocol_sha256 {
            ensure!(
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                "protocol_sha256 must be lowercase SHA256"
            );
        }
        let outcomes = &self.what_would_verify_refute;
        unique_text(
            &outcomes.verification_outcomes,
            "verification_outcomes",
            false,
        )?;
        unique_text(&outcomes.revision_outcomes, "revision_outcomes", false)?;
        unique_text(
            &outcomes.abandonment_outcomes,
            "abandonment_outcomes",
            false,
        )?;
        unique_text(
            &outcomes.inconclusive_outcomes,
            "inconclusive_outcomes",
            false,
        )?;
        let mut classified = BTreeSet::new();
        for outcome in outcomes
            .verification_outcomes
            .iter()
            .chain(&outcomes.revision_outcomes)
            .chain(&outcomes.abandonment_outcomes)
            .chain(&outcomes.inconclusive_outcomes)
        {
            ensure!(
                classified.insert(outcome.trim()),
                "outcome appears in multiple verdict categories"
            );
        }
        Ok(())
    }
}

impl ProvenanceStore {
    /// Return legacy claim IDs whose retained protocol is admitted without a
    /// declaration-bound digest. Sealed declarations also verify exact bytes.
    pub fn verify_claim_evidence_artifacts(&self, repo_root: &Path) -> Result<Vec<String>> {
        let mut statement = self
            .conn
            .prepare("SELECT claim_id,spec_json FROM claim_evidence ORDER BY claim_id")?;
        let mut rows = statement.query([])?;
        let mut legacy_unsealed = Vec::new();
        while let Some(row) = rows.next()? {
            let claim_id: String = row.get(0)?;
            let spec: ClaimEvidenceSpec = serde_json::from_str(&row.get::<_, String>(1)?)?;
            ensure!(
                spec.claim_id == claim_id,
                "claim evidence identity differs for {claim_id}"
            );
            spec.validate()?;
            admit_protocol(repo_root, &spec)
                .with_context(|| format!("verify protocol for {claim_id}"))?;
            if spec.decisive_experiment.protocol_sha256.is_none() {
                legacy_unsealed.push(claim_id);
            }
        }
        Ok(legacy_unsealed)
    }

    pub fn parse_claim_evidence_spec(text: &str) -> Result<ClaimEvidenceSpec> {
        let spec: ClaimEvidenceSpec =
            toml::from_str(text).context("parse claim evidence specification")?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn claim_evidence(&self, claim_id: &str) -> Result<Option<ClaimEvidenceSpec>> {
        let value: Option<String> = self
            .conn
            .query_row(
                "SELECT spec_json FROM claim_evidence WHERE claim_id=?1",
                [claim_id],
                |row| row.get(0),
            )
            .optional()?;
        value
            .map(|text| serde_json::from_str(&text).context("decode canonical claim evidence"))
            .transpose()
    }

    pub fn set_claim_evidence(
        &mut self,
        repo_root: &Path,
        spec: &ClaimEvidenceSpec,
        actor: &str,
        reason: &str,
    ) -> Result<i64> {
        spec.validate()?;
        require_text(actor, "actor")?;
        require_text(reason, "reason")?;
        let mut sealed = spec.clone();
        sealed.decisive_experiment.protocol_sha256 = Some(admit_protocol(repo_root, spec)?);
        let spec = &sealed;
        let next = serde_json::to_string(spec)?;
        let transaction = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let exists: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM claims WHERE id=?1)",
            [&spec.claim_id],
            |row| row.get(0),
        )?;
        ensure!(exists, "unknown claim {}", spec.claim_id);
        for experiment in &spec.decisive_experiment.experiment_ids {
            let exists: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM experiments_cp WHERE id=?1)",
                [experiment],
                |row| row.get(0),
            )?;
            ensure!(exists, "unknown experiment {experiment}");
        }
        let previous: Option<String> = transaction
            .query_row(
                "SELECT spec_json FROM claim_evidence WHERE claim_id=?1",
                [&spec.claim_id],
                |row| row.get(0),
            )
            .optional()?;
        transaction.execute("INSERT INTO claim_evidence (claim_id,spec_json) VALUES (?1,?2) ON CONFLICT(claim_id) DO UPDATE SET spec_json=excluded.spec_json", params![spec.claim_id, next])?;
        transaction.execute("INSERT INTO claim_evidence_revisions (claim_id,previous_spec_json,new_spec_json,actor,reason) VALUES (?1,?2,?3,?4,?5)", params![spec.claim_id, previous, next, actor, reason])?;
        let revision = transaction.last_insert_rowid();
        for experiment_id in &spec.decisive_experiment.experiment_ids {
            transaction.execute(
                "INSERT INTO claim_evidence_revision_experiments (revision_id,experiment_id) VALUES (?1,?2)",
                params![revision, experiment_id],
            )?;
        }
        admit_protocol(repo_root, spec).context("recheck protocol before claim evidence commit")?;
        transaction.commit()?;
        Ok(revision)
    }

    pub(crate) fn overlay_claim_evidence(&self, rendered: String) -> Result<String> {
        let mut document: toml_edit::DocumentMut = rendered.parse()?;
        let mut expected: toml::Value = toml::from_str(&rendered)?;
        if let Some(claims) = document
            .get_mut("claim")
            .and_then(toml_edit::Item::as_array_of_tables_mut)
        {
            for (index, claim) in claims.iter_mut().enumerate() {
                let Some(id) = claim.get("id").and_then(toml_edit::Item::as_str) else {
                    continue;
                };
                if let Some(spec) = self.claim_evidence(id)? {
                    let serialized = toml::to_string(&spec)?;
                    let fields: toml_edit::DocumentMut = serialized.parse()?;
                    let semantic_fields: toml::Value = toml::from_str(&serialized)?;
                    let expected_claim = expected["claim"][index]
                        .as_table_mut()
                        .context("rendered claim must be a table")?;
                    for (key, value) in fields.iter() {
                        if key != "claim_id" {
                            let mut transplanted = value.clone();
                            clear_transplanted_table_positions(&mut transplanted);
                            claim.insert(key, transplanted);
                            expected_claim.insert(key.to_owned(), semantic_fields[key].clone());
                        }
                    }
                    if spec.lift_depth.is_none() {
                        claim.remove("lift_depth");
                        expected_claim.remove("lift_depth");
                    }
                    claim.insert(
                        "lift_depth_convention",
                        toml_edit::value("maximum_interpretive_maps_per_branch"),
                    );
                    expected_claim.insert(
                        "lift_depth_convention".into(),
                        toml::Value::String("maximum_interpretive_maps_per_branch".into()),
                    );
                    expected_claim.insert(
                        "intervening_map_count".into(),
                        toml::Value::Integer(spec.intervening_maps.len() as i64),
                    );
                    expected_claim.insert(
                        "computational_stage_count".into(),
                        toml::Value::Integer(
                            spec.intervening_maps
                                .iter()
                                .filter(|mapping| mapping.kind == MapKind::Computational)
                                .count() as i64,
                        ),
                    );
                    claim.insert(
                        "intervening_map_count",
                        toml_edit::value(spec.intervening_maps.len() as i64),
                    );
                    claim.insert(
                        "computational_stage_count",
                        toml_edit::value(
                            spec.intervening_maps
                                .iter()
                                .filter(|mapping| mapping.kind == MapKind::Computational)
                                .count() as i64,
                        ),
                    );
                }
            }
        }
        checked_semantic_render(&document, &expected)
    }
}

fn admit_protocol(repo_root: &Path, spec: &ClaimEvidenceSpec) -> Result<String> {
    let relative = &spec.decisive_experiment.protocol_artifact;
    let expected = if let Some(expected) = &spec.decisive_experiment.protocol_sha256 {
        expected.clone()
    } else {
        let mut full = repo_root.canonicalize()?;
        for component in Path::new(relative).components() {
            ensure!(
                matches!(component, std::path::Component::Normal(_)),
                "protocol path must be repository-relative"
            );
            full.push(component);
            let metadata = fs::symlink_metadata(&full)
                .with_context(|| format!("inspect protocol {relative}"))?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "protocol path contains a symlink: {relative}"
            );
        }
        ensure!(
            fs::metadata(&full)?.is_file(),
            "protocol artifact must be a regular file: {relative}"
        );
        Sha256::digest(fs::read(full)?)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    };
    let bytes = crate::artifact_paths::verified_file_bytes(repo_root, relative, &expected)?;
    ensure!(
        !bytes.is_empty(),
        "protocol artifact must contain retained content: {relative}"
    );
    Ok(expected)
}

/// Parsed table positions belong to the source document. Clearing every nested
/// position makes transplanted tables follow their owning claim during emission.
fn clear_transplanted_table_positions(item: &mut toml_edit::Item) {
    match item {
        toml_edit::Item::Table(table) => {
            table.set_position(None);
            for (_, child) in table.iter_mut() {
                clear_transplanted_table_positions(child);
            }
        }
        toml_edit::Item::ArrayOfTables(tables) => {
            for table in tables.iter_mut() {
                table.set_position(None);
                for (_, child) in table.iter_mut() {
                    clear_transplanted_table_positions(child);
                }
            }
        }
        toml_edit::Item::None | toml_edit::Item::Value(_) => {}
    }
}

fn checked_semantic_render(
    document: &toml_edit::DocumentMut,
    expected: &toml::Value,
) -> Result<String> {
    let rendered = document.to_string();
    let observed: toml::Value =
        toml::from_str(&rendered).context("typed evidence export produced invalid TOML")?;
    ensure!(
        observed == *expected,
        "typed evidence export changed claim associations or unrelated document values"
    );
    Ok(rendered)
}

pub(crate) fn refuse_claim_evidence_history_loss(connection: &Connection) -> Result<()> {
    let exists: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='claim_evidence_revisions')", [], |row| row.get(0))?;
    if exists {
        let count: i64 =
            connection.query_row("SELECT count(*) FROM claim_evidence_revisions", [], |row| {
                row.get(0)
            })?;
        if count > 0 {
            bail!(
                "refusing destructive import: {count} claim evidence revisions require preserving the canonical database"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::PathBuf, process::Command};

    const PROTOCOL: &[u8] = b"Frozen protocol with explicit holdouts and outcomes.\n";

    struct ProtocolRepo {
        root: PathBuf,
    }
    impl Drop for ProtocolRepo {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
    fn git(root: &Path, arguments: &[&str]) -> Result<()> {
        let mut command = Command::new("git");
        for (name, _) in std::env::vars_os() {
            if name.as_encoded_bytes().starts_with(b"GIT_") {
                command.env_remove(name);
            }
        }
        let output = command.arg("-C").arg(root).args(arguments).output()?;
        ensure!(
            output.status.success(),
            "Git fixture failure: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    }

    fn fixture() -> Result<(ProvenanceStore, ProtocolRepo)> {
        let root = std::env::temp_dir().join(format!(
            "gororoba-claim-protocol-{}-{}",
            std::process::id(),
            chrono::Utc::now()
                .timestamp_nanos_opt()
                .context("clock range")?
        ));
        fs::create_dir_all(root.join("docs"))?;
        let protocol_repo = ProtocolRepo { root };
        fs::write(protocol_repo.root.join("docs/protocol.toml"), PROTOCOL)?;
        git(&protocol_repo.root, &["init", "--quiet"])?;
        git(&protocol_repo.root, &["add", "docs/protocol.toml"])?;
        let store = ProvenanceStore::open(std::path::Path::new(":memory:"))?;
        store.conn.execute("INSERT INTO claims (id,statement,status,where_stated,last_verified) VALUES ('C-1','bounded empirical result','Provisional','','')", [])?;
        store.conn.execute("INSERT INTO experiments_cp (id,title,status,claim_refs_json) VALUES ('E-1','held-out comparison','planned','[]')", [])?;
        Ok((store, protocol_repo))
    }

    fn spec() -> ClaimEvidenceSpec {
        ClaimEvidenceSpec {
            claim_id: "C-1".into(),
            evidence_layer: EvidenceLayer::PhenomenologicalMapping,
            depth_status: DepthStatus::NotAssessed,
            lift_depth: None,
            depth_rationale:
                "Computational preprocessing leaves physical interpretation unassessed".into(),
            intervening_maps: vec![InterveningMap {
                name: "training scaling".into(),
                branch: "detector".into(),
                description: "Fit scaling only on training epochs".into(),
                kind: MapKind::Computational,
            }],
            fitted_parameters: vec![FittedParameterBranch {
                branch: "detector".into(),
                count: 1,
                names: vec!["scale".into()],
                training_boundary: "training epochs only".into(),
            }],
            fixed_hyperparameters: vec!["width=6".into()],
            decisive_experiment: DecisiveExperiment {
                experiment_ids: vec!["E-1".into()],
                protocol_artifact: "docs/protocol.toml".into(),
                protocol_sha256: Some(
                    Sha256::digest(PROTOCOL)
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect(),
                ),
                description: "Frozen external-epoch comparison".into(),
            },
            what_would_verify_refute: FalsifierOutcomes {
                verification_outcomes: vec!["interval lower bound exceeds useful threshold".into()],
                revision_outcomes: vec!["matched controls reproduce increment".into()],
                abandonment_outcomes: vec![
                    "interval upper bound falls below useful threshold".into(),
                ],
                inconclusive_outcomes: vec!["interval straddles threshold".into()],
            },
        }
    }

    #[test]
    fn protocol_admission_seals_bytes_and_rejects_invalid_paths_without_mutation() -> Result<()> {
        let (mut store, protocol_repo) = fixture()?;
        let mut declaration = spec();
        declaration.decisive_experiment.protocol_sha256 = None;
        store.set_claim_evidence(
            &protocol_repo.root,
            &declaration,
            "reviewer",
            "seal retained protocol",
        )?;
        let sealed = store.claim_evidence("C-1")?.unwrap();
        assert_eq!(
            sealed.decisive_experiment.protocol_sha256,
            spec().decisive_experiment.protocol_sha256
        );
        assert!(
            store
                .verify_claim_evidence_artifacts(&protocol_repo.root)?
                .is_empty()
        );
        fs::write(protocol_repo.root.join("docs/untracked.toml"), PROTOCOL)?;
        fs::write(protocol_repo.root.join("docs/empty.toml"), b"")?;
        git(&protocol_repo.root, &["add", "docs/empty.toml"])?;
        let mut invalid_paths = vec![
            "docs/missing.toml".to_owned(),
            "docs".into(),
            "../outside.toml".into(),
            protocol_repo
                .root
                .join("docs/protocol.toml")
                .to_string_lossy()
                .into_owned(),
            "docs/untracked.toml".into(),
            "docs/empty.toml".into(),
        ];
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(
                "protocol.toml",
                protocol_repo.root.join("docs/linked.toml"),
            )?;
            std::os::unix::fs::symlink("docs", protocol_repo.root.join("linked-directory"))?;
            invalid_paths.extend([
                "docs/linked.toml".into(),
                "linked-directory/protocol.toml".into(),
            ]);
        }
        for path in invalid_paths {
            let mut invalid = declaration.clone();
            invalid.decisive_experiment.protocol_artifact = path.clone();
            assert!(
                store
                    .set_claim_evidence(
                        &protocol_repo.root,
                        &invalid,
                        "reviewer",
                        "reject invalid admission"
                    )
                    .is_err(),
                "admitted {path}"
            );
            assert_eq!(store.claim_evidence("C-1")?, Some(sealed.clone()));
            let revisions: i64 = store.conn.query_row(
                "SELECT count(*) FROM claim_evidence_revisions",
                [],
                |row| row.get(0),
            )?;
            assert_eq!(revisions, 1, "revision written for {path}");
        }
        let mut incorrect_digest = declaration;
        incorrect_digest.decisive_experiment.protocol_sha256 = Some("0".repeat(64));
        assert!(
            store
                .set_claim_evidence(
                    &protocol_repo.root,
                    &incorrect_digest,
                    "reviewer",
                    "reject wrong bytes"
                )
                .is_err()
        );
        assert_eq!(store.claim_evidence("C-1")?, Some(sealed));
        Ok(())
    }

    #[test]
    fn protocol_invariants_recheck_retention_and_distinguish_legacy_unsealed_contracts()
    -> Result<()> {
        let (mut store, protocol_repo) = fixture()?;
        store.set_claim_evidence(&protocol_repo.root, &spec(), "reviewer", "seal protocol")?;
        let path = protocol_repo.root.join("docs/protocol.toml");
        fs::write(&path, b"changed protocol")?;
        assert!(
            store
                .verify_claim_evidence_artifacts(&protocol_repo.root)
                .is_err()
        );
        assert!(
            store
                .verify_control_plane_invariants(&protocol_repo.root)
                .unwrap_err()
                .to_string()
                .contains("SHA256 mismatch")
        );
        fs::remove_file(&path)?;
        assert!(
            store
                .verify_claim_evidence_artifacts(&protocol_repo.root)
                .is_err()
        );
        fs::create_dir(&path)?;
        assert!(
            store
                .verify_claim_evidence_artifacts(&protocol_repo.root)
                .is_err()
        );
        fs::remove_dir(&path)?;
        fs::write(&path, PROTOCOL)?;
        assert!(
            store
                .verify_claim_evidence_artifacts(&protocol_repo.root)?
                .is_empty()
        );
        let mut legacy = spec();
        legacy.decisive_experiment.protocol_sha256 = None;
        store.conn.execute(
            "UPDATE claim_evidence SET spec_json=?1 WHERE claim_id='C-1'",
            [serde_json::to_string(&legacy)?],
        )?;
        assert_eq!(
            store.verify_claim_evidence_artifacts(&protocol_repo.root)?,
            vec!["C-1"]
        );
        fs::write(&path, b"legacy content lacks a declaration-bound digest")?;
        assert_eq!(
            store.verify_claim_evidence_artifacts(&protocol_repo.root)?,
            vec!["C-1"]
        );
        fs::remove_file(&path)?;
        assert!(
            store
                .verify_claim_evidence_artifacts(&protocol_repo.root)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn evidence_overlay_preserves_early_middle_and_late_claim_ownership() -> Result<()> {
        for selected in [vec![1], vec![4], vec![7], vec![1, 4, 7]] {
            let (mut store, protocol_repo) = fixture()?;
            let mut rendered = "[metadata]\nmarker = 'keep metadata'\n".to_owned();
            for index in 1..=7 {
                if index > 1 {
                    store.conn.execute("INSERT INTO claims (id,statement,status,where_stated,last_verified) VALUES (?1,'bounded result','Provisional','','')", [format!("C-{index}")])?;
                }
                rendered.push_str(&format!("\n[[claim]]\nid = 'C-{index}'\nstatement = 'original statement {index}'\n[claim.legacy]\nmarker = 'original nested table {index}'\n[[claim.legacy.samples]]\nvalue = {index}\n"));
            }
            rendered.push_str("\n[trailer]\nmarker = 'keep trailer'\n");
            let original: toml::Value = toml::from_str(&rendered)?;
            for index in &selected {
                let mut declaration = spec();
                declaration.claim_id = format!("C-{index}");
                declaration.depth_rationale = format!("Physical depth unassessed for C-{index}");
                declaration.decisive_experiment.description =
                    format!("Decisive experiment for C-{index}");
                declaration.what_would_verify_refute.revision_outcomes =
                    vec![format!("Revise C-{index} on matched-control reproduction")];
                store.set_claim_evidence(
                    &protocol_repo.root,
                    &declaration,
                    "reviewer",
                    "bind nested tables to claim ID",
                )?;
            }
            let output = store.overlay_claim_evidence(rendered)?;
            let observed: toml::Value = toml::from_str(&output)?;
            assert_eq!(observed["metadata"], original["metadata"]);
            assert_eq!(observed["trailer"], original["trailer"]);
            assert_eq!(observed["claim"].as_array().unwrap().len(), 7);
            for index in 1..=7 {
                let actual = &observed["claim"][index - 1];
                let previous = &original["claim"][index - 1];
                assert_eq!(actual["id"], previous["id"]);
                if selected.contains(&index) {
                    let declaration = store.claim_evidence(&format!("C-{index}"))?.unwrap();
                    let expected: toml::Value = toml::from_str(&toml::to_string(&declaration)?)?;
                    for (key, value) in expected.as_table().unwrap() {
                        if key != "claim_id" {
                            assert_eq!(actual[key], *value, "C-{index} field {key}");
                        }
                    }
                    for (key, value) in previous.as_table().unwrap() {
                        assert_eq!(actual[key], *value, "original C-{index} field {key}");
                    }
                } else {
                    assert_eq!(actual, previous, "untouched C-{index}");
                }
            }
            assert_eq!(store.overlay_claim_evidence(output.clone())?, output);
        }
        Ok(())
    }

    #[test]
    fn semantic_export_guard_rejects_transplanted_positions_and_unrelated_changes() -> Result<()> {
        let source =
            "[[claim]]\nid = 'C-early'\n[[claim]]\nid = 'C-middle'\n[[claim]]\nid = 'C-late'\n";
        let mut document: toml_edit::DocumentMut = source.parse()?;
        let fields: toml_edit::DocumentMut =
            "[decisive_experiment]\ndescription = 'belongs to late claim'\n".parse()?;
        document["claim"]
            .as_array_of_tables_mut()
            .unwrap()
            .get_mut(2)
            .unwrap()
            .insert("decisive_experiment", fields["decisive_experiment"].clone());
        let mut expected: toml::Value = toml::from_str(source)?;
        expected["claim"][2].as_table_mut().unwrap().insert(
            "decisive_experiment".into(),
            toml::from_str::<toml::Value>("description = 'belongs to late claim'")?,
        );
        assert!(checked_semantic_render(&document, &expected).is_err());
        clear_transplanted_table_positions(
            &mut document["claim"]
                .as_array_of_tables_mut()
                .unwrap()
                .get_mut(2)
                .unwrap()["decisive_experiment"],
        );
        checked_semantic_render(&document, &expected)?;
        document["claim"]
            .as_array_of_tables_mut()
            .unwrap()
            .get_mut(0)
            .unwrap()
            .insert("id", toml_edit::value("C-altered"));
        assert!(checked_semantic_render(&document, &expected).is_err());
        Ok(())
    }

    #[test]
    fn evidence_roundtrip_exports_named_fields_and_complete_history() -> Result<()> {
        let (mut store, protocol_repo) = fixture()?;
        let original = spec();
        let parsed = ProvenanceStore::parse_claim_evidence_spec(&toml::to_string(&original)?)?;
        assert_eq!(parsed, original);
        store.set_claim_evidence(&protocol_repo.root, &parsed, "reviewer", "preregister")?;
        let mut updated = original.clone();
        updated.depth_status = DepthStatus::Declared;
        updated.lift_depth = Some(0);
        updated.depth_rationale =
            "The declared estimand is predictive performance without an interpretive map".into();
        store.set_claim_evidence(
            &protocol_repo.root,
            &updated,
            "reviewer",
            "specify estimand",
        )?;
        assert_eq!(store.claim_evidence("C-1")?, Some(updated.clone()));
        let (previous, next): (String, String) = store.conn.query_row("SELECT previous_spec_json,new_spec_json FROM claim_evidence_revisions ORDER BY id DESC LIMIT 1", [], |row| Ok((row.get(0)?, row.get(1)?)))?;
        assert_eq!(
            serde_json::from_str::<ClaimEvidenceSpec>(&previous)?,
            original
        );
        assert_eq!(serde_json::from_str::<ClaimEvidenceSpec>(&next)?, updated);
        let rendered = store.render_control_plane_compat_outputs()?.claims;
        let document: toml::Value = toml::from_str(&rendered)?;
        let claim = &document["claim"][0];
        assert_eq!(claim["lift_depth"].as_integer(), Some(0));
        assert_eq!(claim["computational_stage_count"].as_integer(), Some(1));
        assert!(claim.get("what_would_verify_refute").is_some());
        assert_eq!(store.overlay_claim_evidence(rendered.clone())?, rendered);
        store.set_claim_evidence(
            &protocol_repo.root,
            &original,
            "reviewer",
            "retain unassessed interpretation",
        )?;
        let unassessed: toml::Value = toml::from_str(&store.overlay_claim_evidence(rendered)?)?;
        assert!(unassessed["claim"][0].get("lift_depth").is_none());
        assert_eq!(
            unassessed["claim"][0]["depth_status"].as_str(),
            Some("not_assessed")
        );
        assert!(
            store
                .conn
                .execute("DELETE FROM claim_evidence_revisions", [])
                .is_err()
        );
        assert!(
            store
                .conn
                .execute("UPDATE claim_evidence_revisions SET reason='rewrite'", [])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn interpretive_depth_uses_normalized_branch_identity() -> Result<()> {
        let mut declaration = spec();
        declaration.depth_status = DepthStatus::Declared;
        declaration.lift_depth = Some(1);
        declaration.intervening_maps = vec![
            InterveningMap {
                name: "embedding".into(),
                branch: "detector".into(),
                description: "Embed the observable".into(),
                kind: MapKind::Interpretive,
            },
            InterveningMap {
                name: "physical reading".into(),
                branch: " detector \t".into(),
                description: "Assign a physical interpretation".into(),
                kind: MapKind::Interpretive,
            },
        ];
        assert!(declaration.validate().is_err());
        declaration.lift_depth = Some(2);
        declaration.validate()?;
        declaration.intervening_maps[1].branch = "independent branch".into();
        declaration.lift_depth = Some(1);
        declaration.validate()?;
        Ok(())
    }

    #[test]
    fn evidence_rejects_unknown_fields_duplicates_depth_and_references() -> Result<()> {
        let (mut store, protocol_repo) = fixture()?;
        let original = spec();
        let text = format!("unknown_field = true\n{}", toml::to_string(&original)?);
        assert!(ProvenanceStore::parse_claim_evidence_spec(&text).is_err());
        let mut malformed = original.clone();
        malformed.lift_depth = Some(0);
        assert!(malformed.validate().is_err());
        malformed = original.clone();
        malformed.fitted_parameters[0].count = 2;
        assert!(malformed.validate().is_err());
        malformed = original.clone();
        malformed
            .intervening_maps
            .push(malformed.intervening_maps[0].clone());
        assert!(malformed.validate().is_err());
        malformed = original.clone();
        malformed.decisive_experiment.experiment_ids = vec!["E-missing".into()];
        assert!(
            store
                .set_claim_evidence(&protocol_repo.root, &malformed, "reviewer", "reject")
                .is_err()
        );
        malformed = original.clone();
        malformed.claim_id = "C-missing".into();
        assert!(
            store
                .set_claim_evidence(&protocol_repo.root, &malformed, "reviewer", "reject")
                .is_err()
        );
        assert!(
            store
                .set_claim_evidence(&protocol_repo.root, &original, " ", "reject")
                .is_err()
        );
        assert_eq!(store.claim_evidence("C-1")?, None);
        Ok(())
    }

    #[test]
    fn evidence_revision_failure_rolls_back_payload_and_import_guard_preserves_history()
    -> Result<()> {
        let (mut store, protocol_repo) = fixture()?;
        let original = spec();
        store.set_claim_evidence(&protocol_repo.root, &original, "reviewer", "initial")?;
        store.conn.execute_batch("CREATE TRIGGER reject_evidence_revision BEFORE INSERT ON claim_evidence_revisions BEGIN SELECT RAISE(ABORT,'injected history failure'); END;")?;
        let mut updated = original.clone();
        updated.depth_rationale = "changed rationale".into();
        assert!(
            store
                .set_claim_evidence(&protocol_repo.root, &updated, "reviewer", "update")
                .is_err()
        );
        assert_eq!(store.claim_evidence("C-1")?, Some(original));
        let count: i64 =
            store
                .conn
                .query_row("SELECT count(*) FROM claim_evidence_revisions", [], |row| {
                    row.get(0)
                })?;
        assert_eq!(count, 1);
        assert!(refuse_claim_evidence_history_loss(&store.conn).is_err());
        assert!(crate::table_ops::clear_control_plane_tables(&store.conn).is_err());
        assert!(store.claim_evidence("C-1")?.is_some());
        Ok(())
    }

    #[test]
    fn evidence_retains_historical_experiment_references_across_lane_replacement() -> Result<()> {
        let (mut store, protocol_repo) = fixture()?;
        store.conn.execute("INSERT INTO experiments_cp (id,title,status,claim_refs_json) VALUES ('E-2','successor comparison','planned','[]')", [])?;
        let mut declaration = spec();
        store.set_claim_evidence(
            &protocol_repo.root,
            &declaration,
            "reviewer",
            "initial experiment",
        )?;
        declaration.decisive_experiment.experiment_ids = vec!["E-2".into()];
        store.set_claim_evidence(
            &protocol_repo.root,
            &declaration,
            "reviewer",
            "successor experiment",
        )?;
        let replacement = "[[experiment]]\nid='E-2'\ntitle='successor comparison'\nstatus='planned'\nclaim_refs=[]\n";
        let root = std::path::Path::new(".");
        let source = root.join("registry/experiments.toml");
        let error = store
            .replace_control_plane_experiments_from_registry_text(root, &source, replacement)
            .expect_err("historical experiment must remain present");
        assert!(error.to_string().contains("E-1"));
        let complete = format!(
            "{replacement}\n[[experiment]]\nid='E-1'\ntitle='held-out comparison'\nstatus='planned'\nclaim_refs=[]\n"
        );
        assert_eq!(
            store.replace_control_plane_experiments_from_registry_text(root, &source, &complete)?,
            2
        );
        assert!(
            store
                .conn
                .execute("DELETE FROM experiments_cp WHERE id='E-1'", [])
                .is_err()
        );
        assert_eq!(store.claim_evidence("C-1")?, Some(declaration));
        Ok(())
    }

    #[test]
    fn evidence_file_rebuild_refusal_preserves_bytes() -> Result<()> {
        let (mut store, protocol_repo) = fixture()?;
        store.set_claim_evidence(
            &protocol_repo.root,
            &spec(),
            "reviewer",
            "retain canonical history",
        )?;
        let filename = format!(
            "gororoba-claim-evidence-{}-{}.sqlite3",
            std::process::id(),
            chrono::Utc::now()
                .timestamp_nanos_opt()
                .context("clock range")?
        );
        let path = std::env::temp_dir().join(filename);
        store.conn.execute(
            "VACUUM INTO ?1",
            [path.to_str().context("temporary path encoding")?],
        )?;
        let before = std::fs::read(&path)?;
        let result = ProvenanceStore::build_fresh(&path);
        let after = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        assert!(result.is_err());
        assert_eq!(before, after);
        Ok(())
    }
}
