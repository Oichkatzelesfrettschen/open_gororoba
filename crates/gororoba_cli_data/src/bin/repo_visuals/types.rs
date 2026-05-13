//! Type definitions for the `repo-visuals` binary.
//! Cli args + data-frame structs for the various renderers.
//! Fields are `pub(crate)` so the bin root can construct them and
//! read field values across the module boundary.

use clap::Parser;
use plotters::style::RGBColor;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "repo-visuals",
    about = "Build deterministic dark-mode repo overview visuals and guides"
)]
pub(crate) struct Cli {
    #[arg(long, default_value = "Cargo.toml")]
    pub(crate) manifest_path: PathBuf,

    #[arg(long, default_value = "registry/project.toml")]
    pub(crate) project: PathBuf,

    #[arg(long, default_value = "registry/entrypoint_docs.toml")]
    pub(crate) entrypoint_docs: PathBuf,

    #[arg(long, default_value = "data/artifacts/images")]
    pub(crate) image_dir: PathBuf,

    #[arg(long, default_value = "docs/book/src/assets")]
    pub(crate) book_asset_dir: PathBuf,

    #[arg(long, default_value = "data/csv")]
    pub(crate) csv_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MetadataRoot {
    pub(crate) packages: Vec<MetadataPackage>,
    pub(crate) workspace_members: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MetadataPackage {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) dependencies: Vec<MetadataDependency>,
    pub(crate) targets: Vec<MetadataTarget>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MetadataDependency {
    pub(crate) name: String,
    pub(crate) path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MetadataTarget {
    pub(crate) kind: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ProjectRegistry {
    pub(crate) project: ProjectBlock,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ProjectBlock {
    pub(crate) version: String,
    pub(crate) test_count: usize,
    pub(crate) claim_count: usize,
    pub(crate) insight_count: usize,
    pub(crate) experiment_count: usize,
    pub(crate) complete_experiment_count: usize,
    pub(crate) paper_count: usize,
    pub(crate) binary_count: usize,
    pub(crate) kernel_checked_claims: usize,
    pub(crate) proof_files: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EntrypointDocsRegistry {
    #[serde(default)]
    pub(crate) document: Vec<EntrypointDoc>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EntrypointDoc {
    pub(crate) path: String,
    pub(crate) body_markdown: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum FamilyKind {
    Algebra,
    Physics,
    Data,
    Interface,
}

#[derive(Clone, Debug)]
pub(crate) struct WorkspaceCrate {
    pub(crate) name: String,
    pub(crate) bin_targets: usize,
    pub(crate) family: FamilyKind,
    pub(crate) internal_deps: Vec<String>,
    pub(crate) inbound_count: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct FamilySummary {
    pub(crate) kind: FamilyKind,
    pub(crate) label: &'static str,
    pub(crate) description: &'static str,
    pub(crate) accent: RGBColor,
    pub(crate) crates: Vec<WorkspaceCrate>,
}

#[derive(Clone, Debug)]
pub(crate) struct ScopeMetric {
    pub(crate) label: &'static str,
    pub(crate) value: String,
    pub(crate) note: String,
    pub(crate) accent: RGBColor,
}

#[derive(Clone, Debug)]
pub(crate) struct RepoEdge {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) from_family: FamilyKind,
    pub(crate) to_family: FamilyKind,
}

#[derive(Clone, Debug)]
pub(crate) struct OperatorRow {
    pub(crate) surface: &'static str,
    pub(crate) canonical: &'static str,
    pub(crate) command: &'static str,
    pub(crate) outputs: &'static str,
    pub(crate) touches: [bool; 7],
    pub(crate) accent: RGBColor,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RectBox {
    pub(crate) x0: i32,
    pub(crate) y0: i32,
    pub(crate) x1: i32,
    pub(crate) y1: i32,
}

#[derive(Clone, Debug)]
pub(crate) struct RepoSurface {
    pub(crate) title: &'static str,
    pub(crate) summary: String,
    #[allow(dead_code)]
    pub(crate) detail: String,
    pub(crate) point: (f64, f64),
    pub(crate) rect: RectBox,
    pub(crate) accent: RGBColor,
}

#[derive(Clone, Debug)]
pub(crate) struct GraphNode {
    pub(crate) family: FamilyKind,
    pub(crate) name: String,
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) weight: f64,
    pub(crate) outbound: usize,
    pub(crate) inbound: usize,
    pub(crate) bin_targets: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct E183LieJordanRow {
    pub(crate) algebra: String,
    pub(crate) snr: f64,
    pub(crate) max_power: f64,
    pub(crate) max_k: f64,
    pub(crate) k_list: String,
    pub(crate) power_list: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct E183MassPhaseRow {
    pub(crate) mode: usize,
    pub(crate) k: f64,
    pub(crate) bin_index: usize,
    pub(crate) log_m200_median: f64,
    pub(crate) power: f64,
    pub(crate) phase: f64,
    pub(crate) mode_snr: f64,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub(crate) struct E183MassPhaseSummaryRow {
    pub(crate) mode: usize,
    pub(crate) k: f64,
    pub(crate) spearman_rho: f64,
    pub(crate) rayleigh_r: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct E183CrossAlgebraRow {
    pub(crate) pair: String,
    pub(crate) rho_avg: f64,
    pub(crate) excess: f64,
    pub(crate) fisher_z: f64,
}

#[allow(dead_code, non_snake_case)]
#[derive(Debug, Deserialize)]
pub(crate) struct GravastarRadialRow {
    #[serde(rename = "M_target")]
    pub(crate) m_target: f64,
    pub(crate) core_compactness: f64,
    #[serde(rename = "R2")]
    pub(crate) r2: f64,
    pub(crate) dM_drho_c: f64,
    #[serde(deserialize_with = "super::deserialize_boolish")]
    pub(crate) harrison_wheeler_stable: bool,
}

#[allow(non_snake_case)]
#[derive(Debug, Deserialize)]
pub(crate) struct GravastarLigoRow {
    #[serde(rename = "M_target")]
    pub(crate) m_target: f64,
    pub(crate) core_compactness: f64,
    #[serde(rename = "R2")]
    pub(crate) r2: f64,
    pub(crate) compactness_2M_R2: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GenesisGravastarRow {
    pub(crate) gamma: f64,
    #[serde(rename = "R1")]
    pub(crate) r1: f64,
    #[serde(rename = "R2")]
    pub(crate) r2: f64,
    #[serde(rename = "M_total")]
    pub(crate) m_total: f64,
    #[serde(deserialize_with = "super::deserialize_boolish")]
    pub(crate) is_stable: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SedenionMassRow {
    #[serde(rename = "Mode_n")]
    pub(crate) mode_n: usize,
    #[serde(rename = "Predicted_Mass")]
    pub(crate) predicted_mass: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PathionCouplingRow {
    pub(crate) coupling: f64,
    pub(crate) final_energy: f64,
    pub(crate) absorbed: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PathionSinkRow {
    pub(crate) step: usize,
    pub(crate) energy_no_sink: f64,
    pub(crate) energy_with_sink: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SedenionFieldMetricRow {
    pub(crate) step: usize,
    pub(crate) mean_associator: f64,
    pub(crate) mean_energy: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ZeroDivisorEdgeRow {
    pub(crate) source: usize,
    pub(crate) target: usize,
    pub(crate) label_s: String,
    pub(crate) label_t: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ZdNode {
    pub(crate) id: usize,
    pub(crate) label: String,
    pub(crate) degree: usize,
    pub(crate) x: f64,
    pub(crate) y: f64,
}
