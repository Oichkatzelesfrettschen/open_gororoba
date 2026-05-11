//! Registry TOML loaders for the four primary ingest record families.
//!
//! Each loader parses one of the canonical registry TOMLs into a
//! Vec<...> of typed records consumed by the ProvenanceStore mutators:
//!
//! - `load_artifacts(path)`: reads the `[[artifact]]` array, validates
//!   the `status` token through ArtifactStatus::parse, and collects
//!   path/string/optional fields into ArtifactRecord rows.
//! - `load_documents(path)`: reads the `[[document]]` array (or returns
//!   empty if the knowledge_documents.raw_capture_count is 0) into
//!   DocumentRecord rows.
//! - `load_lane_assignments(lane_dir)`: iterates the four lane TOMLs
//!   (datasets, slides_artifacts, papers_pdf, web_references) and
//!   builds (artifact_id, lane_name) pairs.
//! - `build_mirror_observations(path)`: cross-product over the four
//!   mirror columns (working_mirrors, working_pdf_mirrors,
//!   nonworking_mirrors, unverified_mirrors) into MirrorObservationRecord.

use std::path::Path;

use anyhow::{Context, Result, bail};
use camino::Utf8PathBuf;
use provenance_core::{
    ArtifactRecord, ArtifactStatus, DocumentRecord, LaneAssignment, MirrorKind,
    MirrorObservationRecord,
};
use toml::Value;

use super::toml_helpers::{
    bool_field, load_toml_value, optional_integer_field, optional_string_field, string_array_field,
    string_field,
};

pub(crate) fn load_artifacts(path: &Path) -> Result<Vec<ArtifactRecord>> {
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

pub(crate) fn load_documents(path: &Path) -> Result<Vec<DocumentRecord>> {
    let value = load_toml_value(path)?;
    let Some(documents) = value.get("document").and_then(Value::as_array) else {
        if let Some(meta) = value.get("knowledge_documents").and_then(Value::as_table) {
            let raw_capture_count = meta
                .get("raw_capture_count")
                .and_then(Value::as_integer)
                .unwrap_or(0);
            if raw_capture_count == 0 {
                return Ok(Vec::new());
            }
        }
        bail!("document table missing");
    };
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

pub(crate) fn load_lane_assignments(lane_dir: &Path) -> Result<Vec<LaneAssignment>> {
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

pub(crate) fn build_mirror_observations(path: &Path) -> Result<Vec<MirrorObservationRecord>> {
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
