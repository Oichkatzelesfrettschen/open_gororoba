use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Parser, Debug)]
#[command(
    name = "crate-inventory",
    about = "Validate the governed Rust crate catalog against cargo metadata"
)]
struct Cli {
    /// Catalog path relative to the repo root.
    #[arg(long, default_value = "registry/rust_crate_catalog.toml")]
    inventory: PathBuf,
    /// Fail on drift instead of only printing a summary.
    #[arg(long)]
    check: bool,
}

#[derive(Debug, Deserialize)]
struct Catalog {
    schema: String,
    meta: CatalogMeta,
    #[serde(rename = "workspace_crate", default)]
    workspace_crates: Vec<WorkspaceCrate>,
    #[serde(rename = "candidate_crate", default)]
    candidate_crates: Vec<CandidateCrate>,
}

#[derive(Debug, Deserialize)]
struct CatalogMeta {
    repo: String,
    source_of_truth: String,
}

#[derive(Debug, Deserialize)]
struct WorkspaceCrate {
    name: String,
    manifest_path: String,
    target_types: Vec<String>,
    domain: String,
    backend_relevance: String,
    status: String,
}

#[derive(Debug, Deserialize)]
struct CandidateCrate {
    name: String,
    status: String,
    landing_zone: String,
    docs_url: String,
    source_url: String,
}

#[derive(Debug, Deserialize)]
struct MetadataRoot {
    packages: Vec<MetadataPackage>,
}

#[derive(Debug, Deserialize)]
struct MetadataPackage {
    name: String,
    manifest_path: String,
    targets: Vec<MetadataTarget>,
}

#[derive(Debug, Deserialize)]
struct MetadataTarget {
    kind: Vec<String>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate must be nested under repo/crates")
        .to_path_buf()
}

fn load_catalog(path: &Path) -> Result<Catalog> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("read crate catalog {}", path.display()))?;
    toml::from_str(&content).with_context(|| format!("parse crate catalog {}", path.display()))
}

fn load_metadata(root: &Path) -> Result<MetadataRoot> {
    let output = Command::new("cargo")
        .arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--no-deps")
        .arg("--manifest-path")
        .arg(root.join("Cargo.toml"))
        .current_dir(root)
        .output()
        .context("run cargo metadata")?;
    if !output.status.success() {
        bail!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    serde_json::from_slice(&output.stdout).context("parse cargo metadata JSON")
}

fn normalize_targets(pkg: &MetadataPackage) -> Vec<String> {
    let mut unique = BTreeSet::new();
    for target in &pkg.targets {
        if let Some(kind) = target.kind.first() {
            unique.insert(kind.clone());
        }
    }
    unique.into_iter().collect()
}

fn validate_agents_reference(root: &Path, inventory_rel: &str) -> Result<()> {
    let agents = fs::read_to_string(root.join("agents.toml")).context("read agents.toml")?;
    if !agents.contains(inventory_rel) {
        bail!("agents.toml does not reference {}", inventory_rel);
    }
    Ok(())
}

fn validate_catalog(catalog: &Catalog, metadata: &MetadataRoot) -> Result<()> {
    if catalog.schema != "gororoba.rust.crate_catalog.v1" {
        bail!("unexpected schema {}", catalog.schema);
    }
    if catalog.meta.repo != "open_gororoba" {
        bail!("unexpected repo {}", catalog.meta.repo);
    }

    let mut inventory_by_name = BTreeMap::new();
    for row in &catalog.workspace_crates {
        if inventory_by_name.insert(row.name.as_str(), row).is_some() {
            bail!("duplicate workspace crate entry `{}`", row.name);
        }
        if row.status != "active" {
            bail!("workspace crate `{}` must be status=active", row.name);
        }
        if row.domain.trim().is_empty() || row.backend_relevance.trim().is_empty() {
            bail!(
                "workspace crate `{}` must declare domain and backend_relevance",
                row.name
            );
        }
    }

    if catalog.workspace_crates.len() != metadata.packages.len() {
        bail!(
            "workspace crate count mismatch: catalog={} metadata={}",
            catalog.workspace_crates.len(),
            metadata.packages.len()
        );
    }

    for pkg in &metadata.packages {
        let row = inventory_by_name
            .get(pkg.name.as_str())
            .ok_or_else(|| anyhow!("missing workspace crate entry `{}`", pkg.name))?;
        if row.manifest_path != strip_repo_prefix(&pkg.manifest_path) {
            bail!(
                "manifest path mismatch for `{}`: catalog={} metadata={}",
                pkg.name,
                row.manifest_path,
                strip_repo_prefix(&pkg.manifest_path)
            );
        }
        let actual_targets = normalize_targets(pkg);
        if row.target_types != actual_targets {
            bail!(
                "target type mismatch for `{}`: catalog={:?} metadata={:?}",
                pkg.name,
                row.target_types,
                actual_targets
            );
        }
    }

    for row in &catalog.candidate_crates {
        if row.docs_url.trim().is_empty() || row.source_url.trim().is_empty() {
            bail!(
                "candidate crate `{}` must declare docs_url and source_url",
                row.name
            );
        }
        if row.status.trim().is_empty() || row.landing_zone.trim().is_empty() {
            bail!(
                "candidate crate `{}` must declare status and landing_zone",
                row.name
            );
        }
    }

    Ok(())
}

fn strip_repo_prefix(path: &str) -> String {
    let root = repo_root();
    let mut prefix = root.to_string_lossy().into_owned();
    if !prefix.ends_with('/') {
        prefix.push('/');
    }
    path.strip_prefix(&prefix).unwrap_or(path).to_string()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = repo_root();
    let inventory_path = root.join(&cli.inventory);
    let catalog = load_catalog(&inventory_path)?;
    let metadata = load_metadata(&root)?;

    let result = validate_catalog(&catalog, &metadata)
        .and_then(|_| validate_agents_reference(&root, &cli.inventory.to_string_lossy()));

    println!("crate-inventory:");
    println!("  schema={}", catalog.schema);
    println!("  repo={}", catalog.meta.repo);
    println!("  source_of_truth={}", catalog.meta.source_of_truth);
    println!("  workspace_crates={}", catalog.workspace_crates.len());
    println!("  candidate_crates={}", catalog.candidate_crates.len());
    println!("  metadata_packages={}", metadata.packages.len());

    match result {
        Ok(()) => {
            println!("  status=ok");
            Ok(())
        }
        Err(err) if cli.check => Err(err),
        Err(err) => {
            println!("  status=drift");
            println!("  detail={err}");
            Ok(())
        }
    }
}
