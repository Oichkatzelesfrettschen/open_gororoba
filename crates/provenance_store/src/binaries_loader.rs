//! Workspace binary inventory: registry parsing, cargo-metadata
//! discovery, manifest-walk fallback, and merge.
//!
//! Three discovery paths feed the inventory:
//! 1. `load_binaries_from_registry` reads the canonical
//!    `registry/binaries.toml` and returns each `[[binary]]` row.
//! 2. `load_workspace_binary_records_via_cargo_metadata` runs
//!    `cargo metadata --no-deps --format-version 1` and extracts
//!    `kind=["bin"]` targets per package.
//! 3. `load_workspace_binary_records_from_manifests` parses the
//!    root `Cargo.toml`, walks the workspace members, and reads
//!    each member manifest's `[[bin]]` entries directly.
//!
//! The cargo-metadata path is preferred (matches resolver semantics
//! and handles target.cfg-gated bins). The manifest walk is a
//! deterministic fallback for offline or vendored snapshots where
//! cargo metadata is unavailable.
//!
//! `merge_workspace_binaries` then promotes any registry-only
//! description/experiment/crate_name annotations onto the discovered
//! workspace entries.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail};
use provenance_core::BinaryRecord;
use serde::Deserialize;
use toml::Value;

use super::{
    sql_helpers::to_repo_rel,
    toml_helpers::{load_toml_text, optional_string_field, string_field},
};

pub(crate) fn load_binaries_from_registry(raw: &str) -> Result<Vec<BinaryRecord>> {
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

pub(crate) fn merge_workspace_binaries(
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
