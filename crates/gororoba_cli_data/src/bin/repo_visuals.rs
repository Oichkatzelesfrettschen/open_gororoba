use anyhow::{Context, Result, bail};
use clap::Parser;
use csv::Writer;
use plotters::{
    coord::{Shift, types::RangedCoordf64},
    prelude::*,
};
use serde::{Deserialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, BTreeSet},
    f64::consts::TAU,
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use walkdir::WalkDir;

const WIDTH: u32 = 3160;
const HEIGHT: u32 = 2820;
const BACKGROUND: RGBColor = RGBColor(11, 15, 21);
const PANEL: RGBColor = RGBColor(21, 28, 37);
const PANEL_ALT: RGBColor = RGBColor(27, 34, 45);
const GRID: RGBColor = RGBColor(55, 65, 81);
const TEXT: RGBColor = RGBColor(241, 245, 249);
const MUTED: RGBColor = RGBColor(148, 163, 184);
const CYAN: RGBColor = RGBColor(56, 189, 248);
const AMBER: RGBColor = RGBColor(251, 191, 36);
const EMERALD: RGBColor = RGBColor(52, 211, 153);
const MAGENTA: RGBColor = RGBColor(244, 114, 182);

const SCOPE_DASHBOARD_FILE: &str = "repo_scope_dashboard_3160x2820.png";
const FAMILY_MAP_FILE: &str = "repo_crate_family_map_3160x2820.png";
const OPERATOR_MATRIX_FILE: &str = "repo_operator_matrix_3160x2820.png";
const E183_PHASE_PLATE_FILE: &str = "science_e183_phase_plate_3160x2820.png";
const GRAVASTAR_PLATE_FILE: &str = "science_gravastar_stability_plate_3160x2820.png";
const ALGEBRA_PLATE_FILE: &str = "science_pathion_zero_divisor_interaction_graph_3160x2820.png";

#[derive(Parser, Debug)]
#[command(
    name = "repo-visuals",
    about = "Build deterministic dark-mode repo overview visuals and guides"
)]
struct Cli {
    #[arg(long, default_value = "Cargo.toml")]
    manifest_path: PathBuf,

    #[arg(long, default_value = "registry/project.toml")]
    project: PathBuf,

    #[arg(long, default_value = "registry/entrypoint_docs.toml")]
    entrypoint_docs: PathBuf,

    #[arg(long, default_value = "data/artifacts/images")]
    image_dir: PathBuf,

    #[arg(long, default_value = "docs/book/src/assets")]
    book_asset_dir: PathBuf,

    #[arg(long, default_value = "data/csv")]
    csv_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct MetadataRoot {
    packages: Vec<MetadataPackage>,
    workspace_members: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MetadataPackage {
    id: String,
    name: String,
    dependencies: Vec<MetadataDependency>,
    targets: Vec<MetadataTarget>,
}

#[derive(Debug, Deserialize)]
struct MetadataDependency {
    name: String,
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MetadataTarget {
    kind: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ProjectRegistry {
    project: ProjectBlock,
}

#[derive(Debug, Deserialize)]
struct ProjectBlock {
    version: String,
    test_count: usize,
    claim_count: usize,
    insight_count: usize,
    experiment_count: usize,
    complete_experiment_count: usize,
    paper_count: usize,
    binary_count: usize,
    kernel_checked_claims: usize,
    proof_files: usize,
}

#[derive(Debug, Deserialize)]
struct EntrypointDocsRegistry {
    #[serde(default)]
    document: Vec<EntrypointDoc>,
}

#[derive(Debug, Deserialize)]
struct EntrypointDoc {
    path: String,
    body_markdown: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum FamilyKind {
    Algebra,
    Physics,
    Data,
    Interface,
}

#[derive(Clone, Debug)]
struct WorkspaceCrate {
    name: String,
    bin_targets: usize,
    family: FamilyKind,
    internal_deps: Vec<String>,
    inbound_count: usize,
}

#[derive(Clone, Debug)]
struct FamilySummary {
    kind: FamilyKind,
    label: &'static str,
    description: &'static str,
    accent: RGBColor,
    crates: Vec<WorkspaceCrate>,
}

#[derive(Clone, Debug)]
struct ScopeMetric {
    label: &'static str,
    value: String,
    note: String,
    accent: RGBColor,
}

#[derive(Clone, Debug)]
struct RepoEdge {
    from: String,
    to: String,
    from_family: FamilyKind,
    to_family: FamilyKind,
}

#[derive(Clone, Debug)]
struct OperatorRow {
    surface: &'static str,
    canonical: &'static str,
    command: &'static str,
    outputs: &'static str,
    touches: [bool; 7],
    accent: RGBColor,
}

#[derive(Clone, Copy, Debug)]
struct RectBox {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

#[derive(Clone, Debug)]
struct RepoSurface {
    title: &'static str,
    summary: String,
    #[allow(dead_code)]
    detail: String,
    point: (f64, f64),
    rect: RectBox,
    accent: RGBColor,
}

#[derive(Clone, Debug)]
struct GraphNode {
    family: FamilyKind,
    name: String,
    x: f64,
    y: f64,
    weight: f64,
    outbound: usize,
    inbound: usize,
    bin_targets: usize,
}

#[derive(Debug, Deserialize)]
struct E183LieJordanRow {
    algebra: String,
    snr: f64,
    max_power: f64,
    max_k: f64,
    k_list: String,
    power_list: String,
}

#[derive(Debug, Deserialize)]
struct E183MassPhaseRow {
    mode: usize,
    k: f64,
    bin_index: usize,
    log_m200_median: f64,
    power: f64,
    phase: f64,
    mode_snr: f64,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct E183MassPhaseSummaryRow {
    mode: usize,
    k: f64,
    spearman_rho: f64,
    rayleigh_r: f64,
}

#[derive(Debug, Deserialize)]
struct E183CrossAlgebraRow {
    pair: String,
    rho_avg: f64,
    excess: f64,
    fisher_z: f64,
}

#[allow(dead_code, non_snake_case)]
#[derive(Debug, Deserialize)]
struct GravastarRadialRow {
    #[serde(rename = "M_target")]
    m_target: f64,
    core_compactness: f64,
    #[serde(rename = "R2")]
    r2: f64,
    dM_drho_c: f64,
    #[serde(deserialize_with = "deserialize_boolish")]
    harrison_wheeler_stable: bool,
}

#[allow(non_snake_case)]
#[derive(Debug, Deserialize)]
struct GravastarLigoRow {
    #[serde(rename = "M_target")]
    m_target: f64,
    core_compactness: f64,
    #[serde(rename = "R2")]
    r2: f64,
    compactness_2M_R2: f64,
}

#[derive(Debug, Deserialize)]
struct GenesisGravastarRow {
    gamma: f64,
    #[serde(rename = "R1")]
    r1: f64,
    #[serde(rename = "R2")]
    r2: f64,
    #[serde(rename = "M_total")]
    m_total: f64,
    #[serde(deserialize_with = "deserialize_boolish")]
    is_stable: bool,
}

#[derive(Debug, Deserialize)]
struct SedenionMassRow {
    #[serde(rename = "Mode_n")]
    mode_n: usize,
    #[serde(rename = "Predicted_Mass")]
    predicted_mass: f64,
}

#[derive(Debug, Deserialize)]
struct PathionCouplingRow {
    coupling: f64,
    final_energy: f64,
    absorbed: f64,
}

#[derive(Debug, Deserialize)]
struct PathionSinkRow {
    step: usize,
    energy_no_sink: f64,
    energy_with_sink: f64,
}

#[derive(Debug, Deserialize)]
struct SedenionFieldMetricRow {
    step: usize,
    mean_associator: f64,
    mean_energy: f64,
}

#[derive(Debug, Deserialize)]
struct ZeroDivisorEdgeRow {
    source: usize,
    target: usize,
    label_s: String,
    label_t: String,
}

#[derive(Clone, Debug)]
struct ZdNode {
    id: usize,
    label: String,
    degree: usize,
    x: f64,
    y: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = repo_root();

    let manifest_path = absolutize(&root, &cli.manifest_path);
    let project_path = absolutize(&root, &cli.project);
    let entrypoint_docs_path = absolutize(&root, &cli.entrypoint_docs);
    let image_dir = absolutize(&root, &cli.image_dir);
    let book_asset_dir = absolutize(&root, &cli.book_asset_dir);
    let csv_dir = absolutize(&root, &cli.csv_dir);

    fs::create_dir_all(&image_dir)
        .with_context(|| format!("create image dir {}", image_dir.display()))?;
    fs::create_dir_all(&book_asset_dir)
        .with_context(|| format!("create book asset dir {}", book_asset_dir.display()))?;
    fs::create_dir_all(&csv_dir)
        .with_context(|| format!("create csv dir {}", csv_dir.display()))?;

    let metadata = load_metadata(&root, &manifest_path)?;
    let project = load_project(&project_path)?;
    let entrypoint_docs = load_entrypoint_docs(&entrypoint_docs_path)?;
    let readme_body = readme_body(&entrypoint_docs)?;
    let workspace_crates = collect_workspace_crates(&metadata);
    let family_summaries = build_family_summaries(&workspace_crates);
    let repo_edges = build_repo_edges(&workspace_crates);
    let repo_surfaces = build_repo_surfaces(&root, &project)?;
    let scope_metrics = build_scope_metrics(&workspace_crates, &project, repo_edges.len());
    let operating_layers = extract_numbered_after(
        readme_body,
        "The repository currently combines four major operating layers:",
    );
    let where_to_look_first = extract_section_list(readme_body, "Where to look first");
    let stable_today = extract_section_list(readme_body, "What is stable today");
    let exploratory = extract_section_list(readme_body, "What is still exploratory");
    let priorities = extract_section_list(readme_body, "Current repo-level priorities");
    let operator_rows = build_operator_rows();

    write_scope_summary_csv(&csv_dir.join("repo_scope_summary.csv"), &scope_metrics)?;
    write_family_csv(
        &csv_dir.join("repo_crate_families.csv"),
        &family_summaries,
        workspace_crates.len(),
    )?;
    write_operator_matrix_csv(
        &csv_dir.join("repo_operator_matrix.csv"),
        &operator_rows,
        &project,
    )?;

    let scope_path = image_dir.join(SCOPE_DASHBOARD_FILE);
    let family_path = image_dir.join(FAMILY_MAP_FILE);
    let matrix_path = image_dir.join(OPERATOR_MATRIX_FILE);
    let e183_path = image_dir.join(E183_PHASE_PLATE_FILE);
    let gravastar_path = image_dir.join(GRAVASTAR_PLATE_FILE);
    let algebra_path = image_dir.join(ALGEBRA_PLATE_FILE);

    render_scope_dashboard(
        &scope_path,
        ScopeDashboardInputs {
            project: &project,
            metrics: &scope_metrics,
            families: &family_summaries,
            edges: &repo_edges,
            surfaces: &repo_surfaces,
            layers: &operating_layers,
            look_first: &where_to_look_first,
            stable_today: &stable_today,
            exploratory: &exploratory,
            priorities: &priorities,
        },
    )?;
    render_family_map(&family_path, &project, &family_summaries, &repo_edges)?;
    render_operator_matrix(&matrix_path, &project, &where_to_look_first, &operator_rows)?;
    render_e183_phase_plate_v2(&e183_path, &root, &project)?;
    render_gravastar_plate_v2(&gravastar_path, &root, &project)?;
    render_algebra_plate_v2(&algebra_path, &root, &project)?;

    mirror_to_book_assets(&scope_path, &book_asset_dir.join(SCOPE_DASHBOARD_FILE))?;
    mirror_to_book_assets(&family_path, &book_asset_dir.join(FAMILY_MAP_FILE))?;
    mirror_to_book_assets(&matrix_path, &book_asset_dir.join(OPERATOR_MATRIX_FILE))?;
    mirror_to_book_assets(&e183_path, &book_asset_dir.join(E183_PHASE_PLATE_FILE))?;
    mirror_to_book_assets(&gravastar_path, &book_asset_dir.join(GRAVASTAR_PLATE_FILE))?;
    mirror_to_book_assets(&algebra_path, &book_asset_dir.join(ALGEBRA_PLATE_FILE))?;

    println!("WROTE {}", scope_path.display());
    println!("WROTE {}", family_path.display());
    println!("WROTE {}", matrix_path.display());
    println!("WROTE {}", e183_path.display());
    println!("WROTE {}", gravastar_path.display());
    println!("WROTE {}", algebra_path.display());
    println!("WROTE {}", csv_dir.join("repo_scope_summary.csv").display());
    println!(
        "WROTE {}",
        csv_dir.join("repo_crate_families.csv").display()
    );
    println!(
        "WROTE {}",
        csv_dir.join("repo_operator_matrix.csv").display()
    );
    println!(
        "WROTE {}",
        book_asset_dir.join(SCOPE_DASHBOARD_FILE).display()
    );
    println!("WROTE {}", book_asset_dir.join(FAMILY_MAP_FILE).display());
    println!(
        "WROTE {}",
        book_asset_dir.join(OPERATOR_MATRIX_FILE).display()
    );
    println!(
        "WROTE {}",
        book_asset_dir.join(E183_PHASE_PLATE_FILE).display()
    );
    println!(
        "WROTE {}",
        book_asset_dir.join(GRAVASTAR_PLATE_FILE).display()
    );
    println!(
        "WROTE {}",
        book_asset_dir.join(ALGEBRA_PLATE_FILE).display()
    );
    Ok(())
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate must be nested under repo/crates")
        .to_path_buf()
}

fn absolutize(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn load_metadata(root: &Path, manifest_path: &Path) -> Result<MetadataRoot> {
    let output = Command::new("cargo")
        .arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--no-deps")
        .arg("--manifest-path")
        .arg(manifest_path)
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

fn load_project(path: &Path) -> Result<ProjectBlock> {
    let content =
        fs::read_to_string(path).with_context(|| format!("read project {}", path.display()))?;
    let registry: ProjectRegistry =
        toml::from_str(&content).with_context(|| format!("parse {}", path.display()))?;
    Ok(registry.project)
}

fn load_entrypoint_docs(path: &Path) -> Result<EntrypointDocsRegistry> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("read entrypoint docs {}", path.display()))?;
    toml::from_str(&content).with_context(|| format!("parse {}", path.display()))
}

fn readme_body(registry: &EntrypointDocsRegistry) -> Result<&str> {
    registry
        .document
        .iter()
        .find(|doc| doc.path == "README.md")
        .map(|doc| doc.body_markdown.as_str())
        .context("find README.md body in registry/entrypoint_docs.toml")
}

fn collect_workspace_crates(metadata: &MetadataRoot) -> Vec<WorkspaceCrate> {
    let members: BTreeSet<&str> = metadata
        .workspace_members
        .iter()
        .map(String::as_str)
        .collect();
    let workspace_names = metadata
        .packages
        .iter()
        .filter(|pkg| members.contains(pkg.id.as_str()))
        .map(|pkg| pkg.name.as_str())
        .collect::<BTreeSet<_>>();

    let mut crates = metadata
        .packages
        .iter()
        .filter(|pkg| members.contains(pkg.id.as_str()))
        .map(|pkg| {
            let mut internal_deps = pkg
                .dependencies
                .iter()
                .filter(|dep| dep.path.is_some() && workspace_names.contains(dep.name.as_str()))
                .map(|dep| dep.name.clone())
                .collect::<Vec<_>>();
            internal_deps.sort();
            internal_deps.dedup();

            WorkspaceCrate {
                name: pkg.name.clone(),
                bin_targets: pkg
                    .targets
                    .iter()
                    .filter(|target| target.kind.iter().any(|kind| kind == "bin"))
                    .count(),
                family: classify_family(&pkg.name),
                internal_deps,
                inbound_count: 0,
            }
        })
        .collect::<Vec<_>>();

    let mut inbound = BTreeMap::<String, usize>::new();
    for krate in &crates {
        for dep in &krate.internal_deps {
            *inbound.entry(dep.clone()).or_default() += 1;
        }
    }
    for krate in &mut crates {
        krate.inbound_count = *inbound.get(&krate.name).unwrap_or(&0);
    }
    crates.sort_by(|a, b| a.name.cmp(&b.name));
    crates
}

fn classify_family(name: &str) -> FamilyKind {
    if name.starts_with("gororoba_cli") || name == "gororoba_py" || name == "xtask" {
        FamilyKind::Interface
    } else if matches!(
        name,
        "provenance_core"
            | "provenance_ops"
            | "provenance_store"
            | "data_core"
            | "docpipe"
            | "scrolls_core"
            | "gororoba_contracts"
            | "gororoba_pipeline"
            | "cosmic_scheduler"
    ) {
        FamilyKind::Data
    } else if name.contains("algebra")
        || name.starts_with("cd_")
        || matches!(
            name,
            "sign_imbalance"
                | "verified_core"
                | "pathion_ellip"
                | "lattice_filtration"
                | "neural_homotopy"
                | "spin_tomography_core"
                | "cd_spin_bridge"
        )
    {
        FamilyKind::Algebra
    } else {
        FamilyKind::Physics
    }
}

fn family_label(kind: FamilyKind) -> &'static str {
    match kind {
        FamilyKind::Algebra => "Algebra / topology",
        FamilyKind::Physics => "Physics / simulation",
        FamilyKind::Data => "Data / provenance / control",
        FamilyKind::Interface => "CLI / interface surfaces",
    }
}

fn family_accent(kind: FamilyKind) -> RGBColor {
    match kind {
        FamilyKind::Algebra => CYAN,
        FamilyKind::Physics => AMBER,
        FamilyKind::Data => EMERALD,
        FamilyKind::Interface => MAGENTA,
    }
}

fn total_degree(krate: &WorkspaceCrate) -> usize {
    krate.internal_deps.len() + krate.inbound_count
}

fn build_repo_edges(crates: &[WorkspaceCrate]) -> Vec<RepoEdge> {
    let families = crates
        .iter()
        .map(|krate| (krate.name.as_str(), krate.family))
        .collect::<BTreeMap<_, _>>();
    let mut edges = Vec::new();
    for krate in crates {
        for dep in &krate.internal_deps {
            if let Some(&to_family) = families.get(dep.as_str()) {
                edges.push(RepoEdge {
                    from: krate.name.clone(),
                    to: dep.clone(),
                    from_family: krate.family,
                    to_family,
                });
            }
        }
    }
    edges.sort_by(|a, b| a.from.cmp(&b.from).then_with(|| a.to.cmp(&b.to)));
    edges
}

fn build_family_summaries(crates: &[WorkspaceCrate]) -> Vec<FamilySummary> {
    let mut grouped: BTreeMap<FamilyKind, Vec<WorkspaceCrate>> = BTreeMap::new();
    for krate in crates {
        grouped.entry(krate.family).or_default().push(krate.clone());
    }
    [
        (
            FamilyKind::Algebra,
            "Algebra / topology",
            "Recursive structure, proofs, motifs, filtration, and zero-divisor analysis.",
            CYAN,
        ),
        (
            FamilyKind::Physics,
            "Physics / simulation",
            "Cosmology, GR, materials, optics, transport, GPU backends, and runtime engines.",
            AMBER,
        ),
        (
            FamilyKind::Data,
            "Data / provenance / control",
            "Registry, provenance, extraction, contracts, pipelines, and canonical indexing.",
            EMERALD,
        ),
        (
            FamilyKind::Interface,
            "CLI / interface surfaces",
            "Operator entrypoints, orchestration binaries, Python bridge, and workspace tasks.",
            MAGENTA,
        ),
    ]
    .into_iter()
    .map(|(kind, label, description, accent)| {
        let mut crates = grouped.remove(&kind).unwrap_or_default();
        crates.sort_by(|a, b| {
            total_degree(b)
                .cmp(&total_degree(a))
                .then_with(|| b.bin_targets.cmp(&a.bin_targets))
                .then_with(|| a.name.cmp(&b.name))
        });
        FamilySummary {
            kind,
            label,
            description,
            accent,
            crates,
        }
    })
    .collect()
}

fn build_scope_metrics(
    crates: &[WorkspaceCrate],
    project: &ProjectBlock,
    internal_edge_count: usize,
) -> Vec<ScopeMetric> {
    let cargo_bin_targets = crates.iter().map(|krate| krate.bin_targets).sum::<usize>();
    let heaviest_hub = crates
        .iter()
        .max_by_key(|krate| total_degree(krate))
        .map(|krate| format!("{} ({})", krate.name, total_degree(krate)))
        .unwrap_or_else(|| "none".to_string());

    vec![
        ScopeMetric {
            label: "Workspace packages",
            value: crates.len().to_string(),
            note: "cargo metadata workspace members".to_string(),
            accent: CYAN,
        },
        ScopeMetric {
            label: "Cargo bin targets",
            value: cargo_bin_targets.to_string(),
            note: "direct executable targets in the workspace".to_string(),
            accent: MAGENTA,
        },
        ScopeMetric {
            label: "Internal couplings",
            value: internal_edge_count.to_string(),
            note: format!("workspace path-dependency edges | hub {heaviest_hub}"),
            accent: EMERALD,
        },
        ScopeMetric {
            label: "Tests",
            value: project.test_count.to_string(),
            note: "tracked in registry/project.toml".to_string(),
            accent: AMBER,
        },
        ScopeMetric {
            label: "Claims",
            value: project.claim_count.to_string(),
            note: "evidence-facing canonical claim rows".to_string(),
            accent: CYAN,
        },
        ScopeMetric {
            label: "Experiments",
            value: project.experiment_count.to_string(),
            note: format!("{} completed", project.complete_experiment_count),
            accent: EMERALD,
        },
        ScopeMetric {
            label: "Papers",
            value: project.paper_count.to_string(),
            note: "manifested research corpus".to_string(),
            accent: AMBER,
        },
        ScopeMetric {
            label: "Proof files",
            value: project.proof_files.to_string(),
            note: format!("{} kernel-checked claims", project.kernel_checked_claims),
            accent: MAGENTA,
        },
    ]
}

fn build_operator_rows() -> Vec<OperatorRow> {
    vec![
        OperatorRow {
            surface: "Workspace membership / target census",
            canonical: "Cargo.toml + cargo metadata",
            command: "cargo metadata --format-version 1 --no-deps",
            outputs: "Workspace package inventory, internal path deps, and executable target counts.",
            touches: [true, true, false, false, false, false, false],
            accent: CYAN,
        },
        OperatorRow {
            surface: "Dependency hot spots before edits",
            canonical: "Path dependencies across crates/*",
            command: "cargo tree --workspace --depth 2",
            outputs: "Fast read on heavy hubs and cross-family couplings before touching a subsystem.",
            touches: [true, true, false, false, false, false, false],
            accent: CYAN,
        },
        OperatorRow {
            surface: "Rust API and crate-level docs",
            canonical: "crates/* + rustdoc output",
            command: "cargo doc --workspace --no-deps",
            outputs: "Browse module contracts and exported surfaces without guessing from filenames alone.",
            touches: [false, true, false, false, true, false, false],
            accent: MAGENTA,
        },
        OperatorRow {
            surface: "Root scope / policy / README landing",
            canonical: "AGENTS.md + agents.toml + registry/entrypoint_docs.toml",
            command: "cargo run -p gororoba_cli_data --bin registry-emit -- entrypoint-docs-legacy",
            outputs: "Regenerates README-facing entrypoints from canonical TOML sections instead of hand editing mirrors.",
            touches: [false, false, true, true, false, false, true],
            accent: MAGENTA,
        },
        OperatorRow {
            surface: "Book architecture and repo guides",
            canonical: "registry/book_docs.toml",
            command: "cargo run -p gororoba_cli_data --bin registry-emit -- book-docs-legacy",
            outputs: "Re-emits mdBook source pages from the canonical registry lane for consistent documentation.",
            touches: [false, false, true, false, true, false, true],
            accent: EMERALD,
        },
        OperatorRow {
            surface: "Claims / experiments / control plane",
            canonical: "registry/canonical/control_plane.sqlite3",
            command: "cargo run -p gororoba_cli_data --bin registry-check",
            outputs: "Validates the evidence-facing control plane that backs counts, claims, and experiment state.",
            touches: [false, false, true, false, false, false, false],
            accent: AMBER,
        },
        OperatorRow {
            surface: "Repo-native visuals and CSV summaries",
            canonical: "crates/gororoba_cli_data/src/bin/repo_visuals.rs",
            command: "cargo run -p gororoba_cli_data --bin repo-visuals",
            outputs: "Builds repo maps, operator heatmaps, CSV companions, and copied book assets under stable names.",
            touches: [false, true, true, false, true, true, false],
            accent: EMERALD,
        },
        OperatorRow {
            surface: "Full artifact lane",
            canonical: "Makefile artifacts target",
            command: "make artifacts",
            outputs: "Runs the broader deterministic artifact rebuild once the scoped lane looks correct locally.",
            touches: [true, true, true, false, true, true, false],
            accent: AMBER,
        },
    ]
}

fn build_repo_surfaces(root: &Path, project: &ProjectBlock) -> Result<Vec<RepoSurface>> {
    let external_sources = count_immediate_dirs(&root.join("data/external"))?;
    let result_lanes = count_immediate_dirs(&root.join("data/results"))?;
    let artifact_images = count_files_with_ext(&root.join("data/artifacts/images"), "png")?;
    let csv_tables = count_files_with_ext(&root.join("data/csv"), "csv")?;
    let experiment_workdirs = count_immediate_dirs(&root.join("experiments"))?;

    Ok(vec![
        RepoSurface {
            title: "External datasets",
            summary: format!("{external_sources} source lanes under data/external"),
            detail: "Imported surveys, probes, and measurement corpora feed the evidence field."
                .to_string(),
            point: (-0.28, 0.38),
            rect: RectBox {
                x0: 78,
                y0: 334,
                x1: 454,
                y1: 434,
            },
            accent: EMERALD,
        },
        RepoSurface {
            title: "Claims + proofs",
            summary: format!(
                "{} claims | {} proof files",
                project.claim_count, project.proof_files
            ),
            detail:
                "Evidence-facing statements and kernel-checked proofs stay near the algebraic core."
                    .to_string(),
            point: (-0.56, -0.10),
            rect: RectBox {
                x0: 88,
                y0: 1450,
                x1: 474,
                y1: 1550,
            },
            accent: CYAN,
        },
        RepoSurface {
            title: "Experiment field",
            summary: format!(
                "{} registered | {} complete | {} workdirs",
                project.experiment_count, project.complete_experiment_count, experiment_workdirs
            ),
            detail:
                "Registry state and active workdirs anchor reproducible experiment definitions."
                    .to_string(),
            point: (0.04, -0.06),
            rect: RectBox {
                x0: 1248,
                y0: 246,
                x1: 1688,
                y1: 346,
            },
            accent: AMBER,
        },
        RepoSurface {
            title: "Result lanes",
            summary: format!("{result_lanes} directories under data/results"),
            detail:
                "Named result sets and sweeps sit on the physics-facing side of the workspace field."
                    .to_string(),
            point: (0.54, -0.04),
            rect: RectBox {
                x0: 2596,
                y0: 516,
                x1: 3074,
                y1: 616,
            },
            accent: AMBER,
        },
        RepoSurface {
            title: "Artifact images",
            summary: format!("{artifact_images} png artifacts in data/artifacts/images"),
            detail:
                "Rendered figures, scientific plates, and mirrored book assets leave the CLI surface here."
                    .to_string(),
            point: (0.56, 0.34),
            rect: RectBox {
                x0: 2542,
                y0: 2068,
                x1: 3074,
                y1: 2168,
            },
            accent: MAGENTA,
        },
        RepoSurface {
            title: "CSV evidence tables",
            summary: format!("{csv_tables} csv companions in data/csv"),
            detail:
                "Tabular summaries, sweeps, and plot companions support the visualization lanes."
                    .to_string(),
            point: (-0.06, 0.56),
            rect: RectBox {
                x0: 690,
                y0: 2424,
                x1: 1188,
                y1: 2524,
            },
            accent: EMERALD,
        },
    ])
}

fn count_immediate_dirs(path: &Path) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let count = fs::read_dir(path)
        .with_context(|| format!("read {}", path.display()))?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_type().ok())
        .filter(|file_type| file_type.is_dir())
        .count();
    Ok(count)
}

fn count_files_with_ext(path: &Path, ext: &str) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let count = WalkDir::new(path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| value.eq_ignore_ascii_case(ext))
                .unwrap_or(false)
        })
        .count();
    Ok(count)
}

fn extract_section_list(body: &str, heading: &str) -> Vec<String> {
    let target = format!("## {heading}");
    let mut capture = false;
    let mut out = Vec::new();
    let mut active_index = None;

    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("## ") {
            if trimmed == target {
                capture = true;
                active_index = None;
                continue;
            }
            if capture {
                break;
            }
        }
        if !capture {
            continue;
        }
        if trimmed.is_empty() {
            active_index = None;
            continue;
        }
        if let Some(item) = trimmed.strip_prefix("- ") {
            out.push(item.trim().to_string());
            active_index = Some(out.len() - 1);
            continue;
        }
        if let Some(item) = strip_number_prefix(trimmed) {
            out.push(item.to_string());
            active_index = Some(out.len() - 1);
            continue;
        }
        if let Some(idx) = active_index {
            out[idx].push(' ');
            out[idx].push_str(trimmed);
        }
    }
    out
}

fn extract_numbered_after(body: &str, anchor: &str) -> Vec<String> {
    let mut waiting = false;
    let mut out = Vec::new();
    let mut active_index = None;

    for line in body.lines() {
        let trimmed = line.trim();
        if !waiting {
            if trimmed == anchor {
                waiting = true;
            }
            continue;
        }
        if trimmed.is_empty() {
            if !out.is_empty() && active_index.is_none() {
                break;
            }
            active_index = None;
            continue;
        }
        if let Some(item) = strip_number_prefix(trimmed) {
            out.push(item.to_string());
            active_index = Some(out.len() - 1);
            continue;
        }
        if let Some(idx) = active_index {
            out[idx].push(' ');
            out[idx].push_str(trimmed);
            continue;
        }
        if !out.is_empty() {
            break;
        }
    }
    out
}

fn strip_number_prefix(text: &str) -> Option<&str> {
    let mut seen_digit = false;
    let mut dot_index = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_ascii_digit() {
            seen_digit = true;
            continue;
        }
        if seen_digit && ch == '.' {
            dot_index = Some(idx);
            break;
        }
        return None;
    }
    let idx = dot_index?;
    text.get(idx + 1..)
        .map(str::trim_start)
        .filter(|rest| !rest.is_empty())
}

fn write_scope_summary_csv(path: &Path, metrics: &[ScopeMetric]) -> Result<()> {
    let mut writer =
        Writer::from_path(path).with_context(|| format!("create {}", path.display()))?;
    writer.write_record(["metric", "value", "note"])?;
    for metric in metrics {
        writer.write_record([metric.label, metric.value.as_str(), metric.note.as_str()])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_family_csv(path: &Path, families: &[FamilySummary], total_crates: usize) -> Result<()> {
    let mut writer =
        Writer::from_path(path).with_context(|| format!("create {}", path.display()))?;
    writer.write_record([
        "family",
        "crate_count",
        "bin_target_count",
        "coverage_fraction",
        "crates",
    ])?;
    for family in families {
        let crate_count = family.crates.len();
        let bin_targets = family
            .crates
            .iter()
            .map(|krate| krate.bin_targets)
            .sum::<usize>();
        let coverage_fraction = if total_crates == 0 {
            "0.0000".to_string()
        } else {
            format!("{:.4}", crate_count as f64 / total_crates as f64)
        };
        let crate_names = family
            .crates
            .iter()
            .map(|krate| krate.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        writer.write_record([
            family.label,
            &crate_count.to_string(),
            &bin_targets.to_string(),
            &coverage_fraction,
            &crate_names,
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_operator_matrix_csv(
    path: &Path,
    rows: &[OperatorRow],
    project: &ProjectBlock,
) -> Result<()> {
    let mut writer =
        Writer::from_path(path).with_context(|| format!("create {}", path.display()))?;
    writer.write_record([
        "surface",
        "canonical",
        "first_command",
        "outputs",
        "repo_version",
    ])?;
    for row in rows {
        writer.write_record([
            row.surface,
            row.canonical,
            row.command,
            row.outputs,
            project.version.as_str(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

/// Bundle of scope-dashboard render inputs. Decomposes into structural
/// data (metrics, families, edges, surfaces) and four annotated string
/// lists (layers, look_first, stable_today, exploratory, priorities) used
/// in the legend / sidebars.
struct ScopeDashboardInputs<'a> {
    project: &'a ProjectBlock,
    metrics: &'a [ScopeMetric],
    families: &'a [FamilySummary],
    edges: &'a [RepoEdge],
    surfaces: &'a [RepoSurface],
    layers: &'a [String],
    look_first: &'a [String],
    stable_today: &'a [String],
    exploratory: &'a [String],
    priorities: &'a [String],
}

fn render_scope_dashboard(path: &Path, inputs: ScopeDashboardInputs<'_>) -> Result<()> {
    let project = inputs.project;
    let metrics = inputs.metrics;
    let families = inputs.families;
    let edges = inputs.edges;
    let surfaces = inputs.surfaces;
    let layers = inputs.layers;
    let look_first = inputs.look_first;
    let stable_today = inputs.stable_today;
    let exploratory = inputs.exploratory;
    let priorities = inputs.priorities;
    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_err)?;
    draw_background_field(&root, CYAN, AMBER)?;
    let nodes = fit_graph_nodes(&build_graph_nodes(families), (0.02, 0.08), (0.78, 0.66));
    let field_rect = RectBox {
        x0: 26,
        y0: 142,
        x1: 3134,
        y1: 2790,
    };
    draw_repo_manifold(&root, field_rect, &nodes, edges, 12)?;

    root.draw(&Text::new(
        "OPEN_GOROROBA: SCIENTIFIC WORKSPACE MANIFOLD",
        (84, 86),
        ("sans-serif", 52).into_font().color(&TEXT),
    ))
    .map_err(plot_err)?;
    root.draw(&Text::new(
        format!(
            "Cargo-metadata coupling field with dataset, result, experiment, and evidence annotations | version {}",
            project.version
        ),
        (84, 132),
        ("sans-serif", 22).into_font().color(&MUTED),
    ))
    .map_err(plot_err)?;

    let badge_specs = [
        ("Workspace packages", "packages"),
        ("Internal couplings", "couplings"),
        ("Experiments", "experiments"),
        ("Claims", "claims"),
        ("Proof files", "proofs"),
    ];
    let badge_y = 92;
    let mut badge_x = 1760;
    for (label, suffix) in badge_specs {
        let Some(metric) = metrics.iter().find(|metric| metric.label == label) else {
            continue;
        };
        let badge_text = format!("{} {}", metric.value, suffix);
        draw_badge(
            &root,
            badge_x,
            badge_y,
            &badge_text,
            metric.accent.mix(0.92),
        )?;
        badge_x += badge_text.len() as i32 * 11 + 56;
    }

    for surface in surfaces {
        draw_surface_callout(&root, field_rect, surface)?;
    }

    let guide_line = format!(
        "Read first: {} | {} | {}",
        short_text(
            look_first
                .first()
                .map(String::as_str)
                .unwrap_or("Cargo.toml workspace root"),
            34
        ),
        short_text(
            look_first
                .get(1)
                .map(String::as_str)
                .unwrap_or("AGENTS.md and agents.toml"),
            34
        ),
        short_text(
            look_first
                .get(2)
                .map(String::as_str)
                .unwrap_or("registry/project.toml"),
            34
        )
    );
    let stable_line = format!(
        "Stable: {}",
        short_text(
            stable_today
                .first()
                .map(String::as_str)
                .unwrap_or("The workspace is Rust-first and Cargo-first."),
            112
        )
    );
    let frontier_line = format!(
        "Frontier: {} | Priority: {}",
        short_text(
            exploratory
                .first()
                .map(String::as_str)
                .unwrap_or("Exploratory lanes remain benchmark-backed prototypes."),
            66
        ),
        short_text(
            priorities
                .first()
                .map(String::as_str)
                .unwrap_or("Clarify root entrypoints and scope."),
            66
        )
    );
    let cognition_line = format!(
        "Accessibility: {} operating layers compressed into one field; position, envelope geometry, labels, and luminance resolve before hue.",
        layers.len()
    );
    root.draw(&PathElement::new(
        vec![(84, 2524), (1520, 2524)],
        ShapeStyle::from(&CYAN.mix(0.28)).stroke_width(2),
    ))
    .map_err(plot_err)?;
    root.draw(&PathElement::new(
        vec![(1780, 2524), (3074, 2524)],
        ShapeStyle::from(&MAGENTA.mix(0.24)).stroke_width(2),
    ))
    .map_err(plot_err)?;
    draw_wrapped_lines(&root, 84, 2558, 18, &wrap_text(&guide_line, 92), &MUTED)?;
    draw_wrapped_lines(&root, 84, 2620, 18, &wrap_text(&stable_line, 92), &MUTED)?;
    draw_wrapped_lines(
        &root,
        1780,
        2558,
        18,
        &wrap_text(&frontier_line, 86),
        &MUTED,
    )?;
    draw_wrapped_lines(
        &root,
        1780,
        2620,
        18,
        &wrap_text(&cognition_line, 86),
        &MUTED,
    )?;

    root.present().map_err(plot_err)?;
    Ok(())
}

fn render_family_map(
    path: &Path,
    project: &ProjectBlock,
    families: &[FamilySummary],
    edges: &[RepoEdge],
) -> Result<()> {
    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_err)?;
    draw_background_field(&root, EMERALD, AMBER)?;
    let nodes = build_graph_nodes(families);
    let field_rect = RectBox {
        x0: 60,
        y0: 220,
        x1: 3080,
        y1: 2500,
    };
    draw_repo_manifold(&root, field_rect, &nodes, edges, 16)?;

    let total_crates = families
        .iter()
        .map(|family| family.crates.len())
        .sum::<usize>();
    let total_bins = families
        .iter()
        .flat_map(|family| family.crates.iter())
        .map(|krate| krate.bin_targets)
        .sum::<usize>();
    let cross_edges = edges
        .iter()
        .filter(|edge| edge.from_family != edge.to_family)
        .count();
    let hub = families
        .iter()
        .flat_map(|family| family.crates.iter())
        .max_by_key(|krate| total_degree(krate))
        .map(|krate| format!("{} ({})", krate.name, total_degree(krate)))
        .unwrap_or_else(|| "none".to_string());

    draw_title(
        &root,
        "OPEN_GOROROBA: WORKSPACE CRATE ATLAS",
        &format!(
            "{} crates | {} internal couplings | {} cross-family couplings | hub {} | {} bin targets | {} claims",
            total_crates,
            edges.len(),
            cross_edges,
            hub,
            total_bins,
            project.claim_count
        ),
    )?;

    let family_boxes = grid_positions(GridLayout {
        left: 80,
        top: 2540,
        gap_x: 24,
        gap_y: 0,
        cols: 4,
        rows: 1,
        cell_w: 732,
        cell_h: 200,
    });
    for (family, &(x0, y0, x1, y1)) in families.iter().zip(family_boxes.iter()) {
        let intra_edges = edges
            .iter()
            .filter(|edge| edge.from_family == family.kind && edge.to_family == family.kind)
            .count();
        let cross_edges = edges
            .iter()
            .filter(|edge| edge.from_family == family.kind && edge.to_family != family.kind)
            .count();
        let hub = family
            .crates
            .iter()
            .max_by_key(|krate| total_degree(krate))
            .map(|krate| format!("hub {} ({})", krate.name, total_degree(krate)))
            .unwrap_or_else(|| "hub none".to_string());
        let lines = vec![
            format!(
                "{} crates | {} bin | {} intra | {} cross",
                family.crates.len(),
                family
                    .crates
                    .iter()
                    .map(|krate| krate.bin_targets)
                    .sum::<usize>(),
                intra_edges,
                cross_edges
            ),
            hub,
            family.description.to_string(),
        ];
        draw_text_panel(
            &root,
            RectBox { x0, y0, x1, y1 },
            family.label,
            &lines,
            family.accent,
        )?;
    }

    root.draw(&Text::new(
        "field = weighted workspace coupling density | labels = highest-degree hubs from cargo metadata path dependencies",
        (80, 2785),
        ("sans-serif", 20).into_font().color(&MUTED),
    ))
    .map_err(plot_err)?;

    root.present().map_err(plot_err)?;
    Ok(())
}

fn render_operator_matrix(
    path: &Path,
    project: &ProjectBlock,
    look_first: &[String],
    rows: &[OperatorRow],
) -> Result<()> {
    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_err)?;
    draw_background_field(&root, MAGENTA, CYAN)?;

    draw_title(
        &root,
        "OPEN_GOROROBA: OPERATOR MATRIX",
        &format!(
            "Task-to-surface heatmap for repo navigation | {} tests | {} insights | {} papers | {} registered binaries",
            project.test_count, project.insight_count, project.paper_count, project.binary_count
        ),
    )?;

    let top_panels = [
        (
            RectBox {
                x0: 80,
                y0: 250,
                x1: 1020,
                y1: 690,
            },
            "Authoritative paths",
            vec![
                "Cargo.toml: workspace root and shared dependency policy.".to_string(),
                "AGENTS.md + agents.toml: contribution and automation rules.".to_string(),
                "registry/project.toml: project counts, summaries, and sprint state.".to_string(),
                "registry/entrypoint_docs.toml + registry/book_docs.toml: canonical docs source.".to_string(),
            ],
            CYAN,
        ),
        (
            RectBox {
                x0: 1110,
                y0: 250,
                x1: 2050,
                y1: 690,
            },
            "Where to look first",
            look_first.to_vec(),
            EMERALD,
        ),
        (
            RectBox {
                x0: 2140,
                y0: 250,
                x1: 3080,
                y1: 690,
            },
            "Emitted, not authored",
            vec![
                "README.md and docs/book/src/*.md are emitted from canonical registry TOMLs.".to_string(),
                "docs/generated/* mirrors are browsing exports, not source of truth.".to_string(),
                "data/artifacts/images/* and docs/book/src/assets/* are generated artifact surfaces.".to_string(),
                "Patch the Rust bins or registry sources, then re-emit the mirrors.".to_string(),
            ],
            MAGENTA,
        ),
    ];

    for (rect, title, rows, accent) in top_panels {
        draw_list_panel(&root, rect, title, &rows, accent)?;
    }

    draw_badge(&root, 80, 735, "Operator surfaces", CYAN.mix(0.85))?;
    draw_operator_heatmap(
        &root,
        RectBox {
            x0: 80,
            y0: 790,
            x1: 3080,
            y1: 2700,
        },
        rows,
    )?;

    root.present().map_err(plot_err)?;
    Ok(())
}

fn read_csv_rows<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let mut reader = csv::ReaderBuilder::new()
        .from_path(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        rows.push(row.with_context(|| format!("parse {}", path.display()))?);
    }
    Ok(rows)
}

fn deserialize_boolish<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        other => Err(serde::de::Error::custom(format!(
            "unsupported bool literal '{other}'"
        ))),
    }
}

fn draw_panel_frame(area: &DrawingArea<BitMapBackend<'_>, Shift>, accent: RGBColor) -> Result<()> {
    let (w, h) = area.dim_in_pixel();
    area.draw(&Rectangle::new(
        [(0, 0), (w as i32 - 1, h as i32 - 1)],
        ShapeStyle::from(&PANEL.mix(0.92)).filled(),
    ))
    .map_err(plot_err)?;
    area.draw(&Rectangle::new(
        [(1, 1), (w as i32 - 2, h as i32 - 2)],
        ShapeStyle::from(&GRID.mix(0.28)).stroke_width(1),
    ))
    .map_err(plot_err)?;
    area.draw(&PathElement::new(
        vec![(0, 0), (w as i32 - 1, 0)],
        ShapeStyle::from(&accent.mix(0.82)).stroke_width(3),
    ))
    .map_err(plot_err)?;
    Ok(())
}

fn configure_science_mesh<'a, DB: DrawingBackend>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x_desc: &str,
    y_desc: &str,
) -> Result<()> {
    chart
        .configure_mesh()
        .label_style(("sans-serif", 20).into_font().color(&TEXT))
        .axis_style(ShapeStyle::from(&TEXT).stroke_width(2))
        .light_line_style(ShapeStyle::from(&GRID.mix(0.28)).stroke_width(1))
        .bold_line_style(ShapeStyle::from(&GRID.mix(0.52)).stroke_width(1))
        .x_desc(x_desc)
        .y_desc(y_desc)
        .draw()
        .map_err(plot_err)
}

fn finite_bounds(values: &[f64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        if value.is_finite() {
            min = min.min(*value);
            max = max.max(*value);
        }
    }
    if !min.is_finite() || !max.is_finite() {
        (0.0, 1.0)
    } else {
        (min, max)
    }
}

fn padded_range(min: f64, max: f64, frac: f64) -> (f64, f64) {
    let span = (max - min).abs().max(1e-9);
    (min - span * frac, max + span * frac)
}

fn parse_semicolon_series(text: &str) -> Result<Vec<f64>> {
    text.split(';')
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .with_context(|| format!("parse float list entry '{part}'"))
        })
        .collect()
}

fn phase_power_color(power: f64, max_power: f64) -> RGBColor {
    let t = if max_power <= 0.0 {
        0.0
    } else {
        (power / max_power).clamp(0.0, 1.0)
    };
    let r = (16.0 + 82.0 * t + 26.0 * t * t) as u8;
    let g = (22.0 + 132.0 * t + 70.0 * t * t) as u8;
    let b = (38.0 + 170.0 * t + 20.0 * (1.0 - t)) as u8;
    RGBColor(r, g, b)
}

fn corr_color(value: f64, min: f64, max: f64) -> RGBColor {
    let t = if (max - min).abs() < 1e-12 {
        0.5
    } else {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    };
    let r = (54.0 + 180.0 * t) as u8;
    let g = (94.0 + 120.0 * (1.0 - (t - 0.5).abs() * 1.7).clamp(0.0, 1.0)) as u8;
    let b = (180.0 + 50.0 * (1.0 - t)) as u8;
    RGBColor(r, g, b)
}

fn science_palette(idx: usize) -> RGBColor {
    match idx % 5 {
        0 => CYAN,
        1 => AMBER,
        2 => EMERALD,
        3 => MAGENTA,
        _ => RGBColor(245, 248, 252),
    }
}

#[allow(dead_code)]
fn render_e183_phase_plate(path: &Path, repo_root: &Path, project: &ProjectBlock) -> Result<()> {
    let lie_rows = read_csv_rows::<E183LieJordanRow>(
        &repo_root.join("data/results/e183/lie_jordan_full.csv"),
    )?;
    let phase_rows = read_csv_rows::<E183MassPhaseRow>(
        &repo_root.join("data/results/e183/mass_binned_phase.csv"),
    )?;
    let summary_rows = read_csv_rows::<E183MassPhaseSummaryRow>(
        &repo_root.join("data/results/e183/mass_binned_phase.summary.csv"),
    )?;
    let corr_rows = read_csv_rows::<E183CrossAlgebraRow>(
        &repo_root.join("data/results/e183/cross_algebra_correlation.csv"),
    )?;

    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_err)?;
    draw_background_field(&root, CYAN, MAGENTA)?;
    draw_title(
        &root,
        "SCIENCE PLATE: E-183 MANGA PHASE FIELD",
        &format!(
            "Mass-binned phase structure, cross-algebra coherence, and spectral combs from data/results/e183 | project {}",
            project.version
        ),
    )?;
    let panels = root.margin(240, 70, 70, 70).split_evenly((2, 2));
    draw_panel_frame(&panels[0], CYAN)?;
    draw_panel_frame(&panels[1], AMBER)?;
    draw_panel_frame(&panels[2], EMERALD)?;
    draw_panel_frame(&panels[3], MAGENTA)?;

    {
        let max_bin = phase_rows
            .iter()
            .map(|row| row.bin_index)
            .max()
            .unwrap_or(0) as f64
            + 1.0;
        let max_mode = phase_rows.iter().map(|row| row.mode).max().unwrap_or(1) as f64;
        let max_power = phase_rows
            .iter()
            .map(|row| row.power)
            .fold(0.0_f64, f64::max);
        let mut mass_labels = BTreeMap::new();
        for row in &phase_rows {
            mass_labels
                .entry(row.bin_index as i32)
                .or_insert_with(|| format!("{:.2}", row.log_m200_median));
        }
        let max_snr = phase_rows
            .iter()
            .map(|row| row.mode_snr)
            .fold(0.0_f64, f64::max);
        let mut chart = ChartBuilder::on(&panels[0])
            .margin(24)
            .caption(
                "Mass-bin phase field: power in luminance, phase in segment angle",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(56)
            .y_label_area_size(64)
            .build_cartesian_2d(-0.5_f64..(max_bin - 0.5), 0.5_f64..(max_mode + 0.5))
            .map_err(plot_err)?;
        chart
            .configure_mesh()
            .label_style(("sans-serif", 18).into_font().color(&TEXT))
            .axis_style(ShapeStyle::from(&TEXT).stroke_width(2))
            .light_line_style(ShapeStyle::from(&GRID.mix(0.25)).stroke_width(1))
            .bold_line_style(ShapeStyle::from(&GRID.mix(0.50)).stroke_width(1))
            .x_desc("halo-mass bin median log10(M200)")
            .y_desc("algebraic mode")
            .x_labels(max_bin as usize)
            .y_labels(max_mode as usize)
            .x_label_formatter(&|value: &f64| {
                mass_labels
                    .get(&(value.round() as i32))
                    .cloned()
                    .unwrap_or_default()
            })
            .draw()
            .map_err(plot_err)?;
        chart
            .draw_series(phase_rows.iter().map(|row| {
                let x0 = row.bin_index as f64 - 0.45;
                let x1 = row.bin_index as f64 + 0.45;
                let y0 = row.mode as f64 - 0.45;
                let y1 = row.mode as f64 + 0.45;
                Rectangle::new(
                    [(x0, y0), (x1, y1)],
                    ShapeStyle::from(&phase_power_color(row.power, max_power)).filled(),
                )
            }))
            .map_err(plot_err)?;
        chart
            .draw_series(phase_rows.iter().map(|row| {
                let len = 0.18 + 0.18 * (row.mode_snr / max_snr.max(1e-9));
                let dx = len * row.phase.cos();
                let dy = len * row.phase.sin() * 0.85;
                PathElement::new(
                    vec![
                        (row.bin_index as f64 - dx, row.mode as f64 - dy),
                        (row.bin_index as f64 + dx, row.mode as f64 + dy),
                    ],
                    ShapeStyle::from(&RGBColor(248, 250, 252).mix(0.92)).stroke_width(2),
                )
            }))
            .map_err(plot_err)?;
    }

    {
        let mut spectra = Vec::new();
        for row in &lie_rows {
            let ks = parse_semicolon_series(&row.k_list)?;
            let powers = parse_semicolon_series(&row.power_list)?;
            spectra.push((row, ks.into_iter().zip(powers).collect::<Vec<_>>()));
        }
        let x_values = spectra
            .iter()
            .flat_map(|(_, series)| series.iter().map(|(x, _)| *x))
            .collect::<Vec<_>>();
        let y_values = spectra
            .iter()
            .flat_map(|(_, series)| series.iter().map(|(_, y)| *y))
            .collect::<Vec<_>>();
        let (x_min, x_max) = finite_bounds(&x_values);
        let (y_min, y_max) = finite_bounds(&y_values);
        let (y_lo, y_hi) = padded_range(y_min.min(0.0), y_max, 0.12);
        let mut chart = ChartBuilder::on(&panels[1])
            .margin(24)
            .caption(
                "Lie / Jordan spectral combs",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(56)
            .y_label_area_size(70)
            .build_cartesian_2d(x_min..x_max, y_lo..y_hi)
            .map_err(plot_err)?;
        configure_science_mesh(&mut chart, "wavenumber k", "power")?;
        for (idx, (row, series)) in spectra.iter().enumerate() {
            let color = science_palette(idx);
            chart
                .draw_series(LineSeries::new(
                    series.iter().copied(),
                    ShapeStyle::from(&color).stroke_width(4),
                ))
                .map_err(plot_err)?
                .label(format!(
                    "{} | SNR {:.3} | max k {:.2}",
                    short_text(&row.algebra, 20),
                    row.snr,
                    row.max_k
                ))
                .legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 28, y)],
                        ShapeStyle::from(&color).stroke_width(4),
                    )
                });
            chart
                .draw_series(series.iter().map(|point| {
                    Circle::new(*point, 4, ShapeStyle::from(&color.mix(0.95)).filled())
                }))
                .map_err(plot_err)?;
            let label_text = format!("{:.2e}", row.max_power);
            if let Some(last) = series.last() {
                chart
                    .draw_series(std::iter::once(Text::new(
                        label_text,
                        *last,
                        ("sans-serif", 16).into_font().color(&color),
                    )))
                    .map_err(plot_err)?;
            }
        }
        chart
            .configure_series_labels()
            .background_style(ShapeStyle::from(&PANEL.mix(0.92)).filled())
            .border_style(ShapeStyle::from(&GRID.mix(0.35)).stroke_width(1))
            .label_font(("sans-serif", 16).into_font().color(&TEXT))
            .draw()
            .map_err(plot_err)?;
    }

    {
        let x_values = summary_rows.iter().map(|row| row.k).collect::<Vec<_>>();
        let rho_values = summary_rows
            .iter()
            .flat_map(|row| [row.spearman_rho, row.rayleigh_r])
            .collect::<Vec<_>>();
        let (x_min, x_max) = finite_bounds(&x_values);
        let (y_min, y_max) = finite_bounds(&rho_values);
        let (y_lo, y_hi) = padded_range(y_min, y_max, 0.18);
        let mut chart = ChartBuilder::on(&panels[2])
            .margin(24)
            .caption(
                "Mode summary: mass ordering vs phase coherence",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(56)
            .y_label_area_size(72)
            .build_cartesian_2d(x_min..x_max, y_lo..y_hi)
            .map_err(plot_err)?;
        configure_science_mesh(&mut chart, "mode wavenumber k", "summary statistic")?;
        let rho_series = summary_rows
            .iter()
            .map(|row| (row.k, row.spearman_rho))
            .collect::<Vec<_>>();
        let rayleigh_series = summary_rows
            .iter()
            .map(|row| (row.k, row.rayleigh_r))
            .collect::<Vec<_>>();
        chart
            .draw_series(LineSeries::new(
                rho_series.iter().copied(),
                ShapeStyle::from(&CYAN).stroke_width(4),
            ))
            .map_err(plot_err)?
            .label("Spearman rho")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 26, y)],
                    ShapeStyle::from(&CYAN).stroke_width(4),
                )
            });
        chart
            .draw_series(LineSeries::new(
                rayleigh_series.iter().copied(),
                ShapeStyle::from(&EMERALD).stroke_width(4),
            ))
            .map_err(plot_err)?
            .label("Rayleigh R")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 26, y)],
                    ShapeStyle::from(&EMERALD).stroke_width(4),
                )
            });
        chart
            .draw_series(
                rho_series.iter().map(|point| {
                    Circle::new(*point, 5, ShapeStyle::from(&CYAN.mix(0.95)).filled())
                }),
            )
            .map_err(plot_err)?;
        chart
            .draw_series(rayleigh_series.iter().map(|point| {
                TriangleMarker::new(*point, 8, ShapeStyle::from(&EMERALD.mix(0.95)).filled())
            }))
            .map_err(plot_err)?;
        chart
            .configure_series_labels()
            .background_style(ShapeStyle::from(&PANEL.mix(0.92)).filled())
            .border_style(ShapeStyle::from(&GRID.mix(0.35)).stroke_width(1))
            .label_font(("sans-serif", 18).into_font().color(&TEXT))
            .draw()
            .map_err(plot_err)?;
    }

    {
        let labels = ["CD-ZD", "G2", "J3(O)", "sl(2)"];
        let mut matrix = BTreeMap::new();
        for row in &corr_rows {
            let parts = row.pair.split('-').collect::<Vec<_>>();
            if parts.len() == 2 {
                matrix.insert((parts[0].to_string(), parts[1].to_string()), row.rho_avg);
                matrix.insert((parts[1].to_string(), parts[0].to_string()), row.rho_avg);
            }
        }
        let corr_values = corr_rows.iter().map(|row| row.rho_avg).collect::<Vec<_>>();
        let (corr_min, corr_max) = finite_bounds(&corr_values);
        let mut chart = ChartBuilder::on(&panels[3])
            .margin(30)
            .caption(
                "Cross-algebra coherence matrix",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(70)
            .y_label_area_size(90)
            .build_cartesian_2d(0.0..4.0, 0.0..4.0)
            .map_err(plot_err)?;
        chart
            .configure_mesh()
            .disable_mesh()
            .label_style(("sans-serif", 18).into_font().color(&TEXT))
            .x_labels(4)
            .y_labels(4)
            .x_desc("target algebra")
            .y_desc("source algebra")
            .x_label_formatter(&|value: &f64| {
                labels
                    .get(value.floor().clamp(0.0, 3.0) as usize)
                    .copied()
                    .unwrap_or("")
                    .to_string()
            })
            .y_label_formatter(&|value: &f64| {
                labels
                    .get(value.floor().clamp(0.0, 3.0) as usize)
                    .copied()
                    .unwrap_or("")
                    .to_string()
            })
            .draw()
            .map_err(plot_err)?;
        for (y_idx, source) in labels.iter().enumerate() {
            for (x_idx, target) in labels.iter().enumerate() {
                let value = if x_idx == y_idx {
                    1.0
                } else {
                    *matrix
                        .get(&(source.to_string(), target.to_string()))
                        .unwrap_or(&0.0)
                };
                let rect = Rectangle::new(
                    [
                        (x_idx as f64, y_idx as f64),
                        (x_idx as f64 + 1.0, y_idx as f64 + 1.0),
                    ],
                    ShapeStyle::from(&corr_color(value, corr_min.min(0.0), corr_max.max(1.0)))
                        .filled(),
                );
                chart.draw_series(std::iter::once(rect)).map_err(plot_err)?;
                chart
                    .draw_series(std::iter::once(Text::new(
                        format!("{value:.3}"),
                        (x_idx as f64 + 0.5, y_idx as f64 + 0.52),
                        ("sans-serif", 18).into_font().color(&TEXT),
                    )))
                    .map_err(plot_err)?;
            }
        }
        let strongest = corr_rows
            .iter()
            .max_by(|a, b| {
                a.excess
                    .partial_cmp(&b.excess)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|row| {
                format!(
                    "strongest excess {} ({:.3}), fisher z {:.1}",
                    row.pair, row.excess, row.fisher_z
                )
            })
            .unwrap_or_else(|| "no cross-algebra rows".to_string());
        panels[3]
            .draw(&Text::new(
                strongest,
                (34, panels[3].dim_in_pixel().1 as i32 - 28),
                ("sans-serif", 18).into_font().color(&MUTED),
            ))
            .map_err(plot_err)?;
    }

    root.present().map_err(plot_err)?;
    Ok(())
}

#[allow(dead_code)]
fn render_gravastar_plate(path: &Path, repo_root: &Path, project: &ProjectBlock) -> Result<()> {
    let radial = read_csv_rows::<GravastarRadialRow>(
        &repo_root.join("data/csv/gravastar_radial_stability.csv"),
    )?;
    let ligo = read_csv_rows::<GravastarLigoRow>(
        &repo_root.join("data/csv/gravastar_ligo_mass_sweep.csv"),
    )?;
    let genesis = read_csv_rows::<GenesisGravastarRow>(
        &repo_root.join("data/csv/genesis_gravastar_bridge.csv"),
    )?;

    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_err)?;
    draw_background_field(&root, AMBER, CYAN)?;
    draw_title(
        &root,
        "SCIENCE PLATE: GRAVASTAR STABILITY FIELD",
        &format!(
            "Compactness, radius, and bridge-state structure from gravastar radial + LIGO sweeps | project {}",
            project.version
        ),
    )?;
    let panels = root.margin(240, 70, 70, 70).split_evenly((2, 2));
    draw_panel_frame(&panels[0], AMBER)?;
    draw_panel_frame(&panels[1], CYAN)?;
    draw_panel_frame(&panels[2], MAGENTA)?;
    draw_panel_frame(&panels[3], EMERALD)?;

    {
        let masses = ligo.iter().map(|row| row.m_target).collect::<Vec<_>>();
        let comps = ligo
            .iter()
            .map(|row| row.core_compactness)
            .collect::<Vec<_>>();
        let compactness = ligo
            .iter()
            .map(|row| row.compactness_2M_R2)
            .collect::<Vec<_>>();
        let stable_lookup = radial
            .iter()
            .map(|row| {
                (
                    (
                        row.m_target as i64,
                        (row.core_compactness * 10.0).round() as i64,
                    ),
                    row.harrison_wheeler_stable,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let (x_min, x_max) = finite_bounds(&masses);
        let (y_min, y_max) = finite_bounds(&comps);
        let (c_min, c_max) = finite_bounds(&compactness);
        let mut chart = ChartBuilder::on(&panels[0])
            .margin(24)
            .caption(
                "Outer compactness field across target mass and core compactness",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(60)
            .y_label_area_size(76)
            .build_cartesian_2d((x_min - 1.0)..(x_max + 1.0), (y_min - 0.03)..(y_max + 0.03))
            .map_err(plot_err)?;
        configure_science_mesh(&mut chart, "target mass", "core compactness")?;
        chart
            .draw_series(ligo.iter().map(|row| {
                let dx = 2.35;
                let dy = 0.055;
                Rectangle::new(
                    [
                        (row.m_target - dx, row.core_compactness - dy),
                        (row.m_target + dx, row.core_compactness + dy),
                    ],
                    ShapeStyle::from(&corr_color(row.compactness_2M_R2, c_min, c_max)).filled(),
                )
            }))
            .map_err(plot_err)?;
        chart
            .draw_series(ligo.iter().map(|row| {
                let key = (
                    row.m_target.round() as i64,
                    (row.core_compactness * 10.0).round() as i64,
                );
                let stable = *stable_lookup.get(&key).unwrap_or(&false);
                let marker = if stable {
                    CYAN
                } else {
                    RGBColor(248, 250, 252)
                };
                Circle::new(
                    (row.m_target, row.core_compactness),
                    if stable { 5 } else { 3 },
                    ShapeStyle::from(&marker.mix(if stable { 0.95 } else { 0.55 })).filled(),
                )
            }))
            .map_err(plot_err)?;
    }

    {
        let mass_values = ligo.iter().map(|row| row.m_target).collect::<Vec<_>>();
        let radius_values = ligo.iter().map(|row| row.r2).collect::<Vec<_>>();
        let (x_min, x_max) = finite_bounds(&mass_values);
        let (y_min, y_max) = finite_bounds(&radius_values);
        let (y_lo, y_hi) = padded_range(y_min, y_max, 0.08);
        let comp_levels = [0.5, 0.6, 0.7, 0.8, 0.9];
        let mut chart = ChartBuilder::on(&panels[1])
            .margin(24)
            .caption(
                "Outer shell radius R2 by compactness family",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(60)
            .y_label_area_size(76)
            .build_cartesian_2d(x_min..x_max, y_lo..y_hi)
            .map_err(plot_err)?;
        configure_science_mesh(&mut chart, "target mass", "R2")?;
        for (idx, comp) in comp_levels.iter().enumerate() {
            let color = science_palette(idx);
            let series = ligo
                .iter()
                .filter(|row| (row.core_compactness - *comp).abs() < 1e-6)
                .map(|row| (row.m_target, row.r2))
                .collect::<Vec<_>>();
            chart
                .draw_series(LineSeries::new(
                    series.iter().copied(),
                    ShapeStyle::from(&color).stroke_width(4),
                ))
                .map_err(plot_err)?
                .label(format!("c = {:.1}", comp))
                .legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 26, y)],
                        ShapeStyle::from(&color).stroke_width(4),
                    )
                });
        }
        chart
            .configure_series_labels()
            .background_style(ShapeStyle::from(&PANEL.mix(0.92)).filled())
            .border_style(ShapeStyle::from(&GRID.mix(0.35)).stroke_width(1))
            .label_font(("sans-serif", 16).into_font().color(&TEXT))
            .draw()
            .map_err(plot_err)?;
    }

    {
        let mut selected_masses = radial.iter().map(|row| row.m_target).collect::<Vec<_>>();
        selected_masses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        selected_masses.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let picks = selected_masses
            .iter()
            .enumerate()
            .filter_map(|(idx, mass)| {
                let last = selected_masses.len().saturating_sub(1);
                (idx == 0 || idx == last / 3 || idx == 2 * last / 3 || idx == last).then_some(*mass)
            })
            .collect::<Vec<_>>();
        let comp_values = radial
            .iter()
            .map(|row| row.core_compactness)
            .collect::<Vec<_>>();
        let dmdrho = radial
            .iter()
            .map(|row| row.dM_drho_c.abs().max(1e-9).log10())
            .collect::<Vec<_>>();
        let (x_min, x_max) = finite_bounds(&comp_values);
        let (y_min, y_max) = finite_bounds(&dmdrho);
        let (y_lo, y_hi) = padded_range(y_min, y_max, 0.10);
        let mut chart = ChartBuilder::on(&panels[2])
            .margin(24)
            .caption(
                "Radial derivative magnitude log10 |dM/drho_c|",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(60)
            .y_label_area_size(86)
            .build_cartesian_2d(x_min..x_max, y_lo..y_hi)
            .map_err(plot_err)?;
        configure_science_mesh(&mut chart, "core compactness", "log10 |dM/drho_c|")?;
        for (idx, mass) in picks.iter().enumerate() {
            let color = science_palette(idx);
            let series = radial
                .iter()
                .filter(|row| (row.m_target - *mass).abs() < 1e-9)
                .map(|row| (row.core_compactness, row.dM_drho_c.abs().max(1e-9).log10()))
                .collect::<Vec<_>>();
            chart
                .draw_series(LineSeries::new(
                    series.iter().copied(),
                    ShapeStyle::from(&color).stroke_width(4),
                ))
                .map_err(plot_err)?
                .label(format!("M = {:.0}", mass))
                .legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 26, y)],
                        ShapeStyle::from(&color).stroke_width(4),
                    )
                });
        }
        chart
            .configure_series_labels()
            .background_style(ShapeStyle::from(&PANEL.mix(0.92)).filled())
            .border_style(ShapeStyle::from(&GRID.mix(0.35)).stroke_width(1))
            .label_font(("sans-serif", 16).into_font().color(&TEXT))
            .draw()
            .map_err(plot_err)?;
    }

    {
        let stable_rows = genesis
            .iter()
            .filter(|row| row.is_stable)
            .collect::<Vec<_>>();
        let x_values = stable_rows.iter().map(|row| row.gamma).collect::<Vec<_>>();
        let y_values = stable_rows
            .iter()
            .map(|row| (row.r2 / row.r1.max(1e-12)).log10())
            .collect::<Vec<_>>();
        let mass_values = stable_rows
            .iter()
            .map(|row| row.m_total.abs().max(1e-18).log10())
            .collect::<Vec<_>>();
        let (x_min, x_max) = finite_bounds(&x_values);
        let (y_min, y_max) = finite_bounds(&y_values);
        let (m_min, m_max) = finite_bounds(&mass_values);
        let mut chart = ChartBuilder::on(&panels[3])
            .margin(24)
            .caption(
                "Genesis bridge branch: gamma vs log10(R2/R1)",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(60)
            .y_label_area_size(84)
            .build_cartesian_2d(
                (x_min - 0.05)..(x_max + 0.05),
                (y_min - 0.06)..(y_max + 0.06),
            )
            .map_err(plot_err)?;
        configure_science_mesh(&mut chart, "gamma", "log10(R2/R1)")?;
        chart
            .draw_series(stable_rows.iter().map(|row| {
                let mass = row.m_total.abs().max(1e-18).log10();
                let color = corr_color(mass, m_min, m_max);
                Circle::new(
                    (row.gamma, (row.r2 / row.r1.max(1e-12)).log10()),
                    4,
                    ShapeStyle::from(&color.mix(0.92)).filled(),
                )
            }))
            .map_err(plot_err)?;
    }

    root.present().map_err(plot_err)?;
    Ok(())
}

#[allow(dead_code)]
fn render_algebra_plate(path: &Path, repo_root: &Path, project: &ProjectBlock) -> Result<()> {
    let mass_rows =
        read_csv_rows::<SedenionMassRow>(&repo_root.join("data/csv/sedenion_mass_spectrum.csv"))?;
    let coupling_rows = read_csv_rows::<PathionCouplingRow>(
        &repo_root.join("data/csv/pathion_coupling_sweep.csv"),
    )?;
    let sink_rows =
        read_csv_rows::<PathionSinkRow>(&repo_root.join("data/csv/pathion_sink_compare.csv"))?;
    let field_rows = read_csv_rows::<SedenionFieldMetricRow>(
        &repo_root.join("data/csv/sedenion_field_metrics_3D.csv"),
    )?;

    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_err)?;
    draw_background_field(&root, MAGENTA, EMERALD)?;
    draw_title(
        &root,
        "SCIENCE PLATE: ALGEBRAIC RESONANCE AND DAMPING",
        &format!(
            "Mass ladder, sink suppression, coupling uptake, and sedenion field decay from generated CSV lanes | project {}",
            project.version
        ),
    )?;
    let panels = root.margin(240, 70, 70, 70).split_evenly((2, 2));
    draw_panel_frame(&panels[0], MAGENTA)?;
    draw_panel_frame(&panels[1], AMBER)?;
    draw_panel_frame(&panels[2], CYAN)?;
    draw_panel_frame(&panels[3], EMERALD)?;

    {
        let x_values = mass_rows
            .iter()
            .map(|row| row.mode_n as f64)
            .collect::<Vec<_>>();
        let y_values = mass_rows
            .iter()
            .map(|row| row.predicted_mass)
            .collect::<Vec<_>>();
        let (x_min, x_max) = finite_bounds(&x_values);
        let (y_min, y_max) = finite_bounds(&y_values);
        let (_, y_hi) = padded_range(y_min, y_max, 0.08);
        let mut chart = ChartBuilder::on(&panels[0])
            .margin(24)
            .caption(
                "Sedenion mass ladder",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(56)
            .y_label_area_size(80)
            .build_cartesian_2d(x_min..x_max, 0.0..y_hi)
            .map_err(plot_err)?;
        configure_science_mesh(&mut chart, "mode n", "predicted mass")?;
        let series = mass_rows
            .iter()
            .map(|row| (row.mode_n as f64, row.predicted_mass))
            .collect::<Vec<_>>();
        chart
            .draw_series(LineSeries::new(
                series.iter().copied(),
                ShapeStyle::from(&MAGENTA).stroke_width(4),
            ))
            .map_err(plot_err)?;
        chart
            .draw_series(
                series.iter().map(|point| {
                    Circle::new(*point, 5, ShapeStyle::from(&MAGENTA.mix(0.95)).filled())
                }),
            )
            .map_err(plot_err)?;
    }

    {
        let x_values = coupling_rows
            .iter()
            .map(|row| row.coupling.log10())
            .collect::<Vec<_>>();
        let y_values = coupling_rows
            .iter()
            .flat_map(|row| [row.final_energy, row.absorbed])
            .collect::<Vec<_>>();
        let (x_min, x_max) = finite_bounds(&x_values);
        let (y_min, y_max) = finite_bounds(&y_values);
        let (y_lo, y_hi) = padded_range(y_min.min(0.0), y_max, 0.08);
        let mut chart = ChartBuilder::on(&panels[1])
            .margin(24)
            .caption(
                "Pathion coupling uptake",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(64)
            .y_label_area_size(80)
            .build_cartesian_2d(x_min..x_max, y_lo..y_hi)
            .map_err(plot_err)?;
        configure_science_mesh(&mut chart, "log10 coupling", "energy / absorbed")?;
        let final_series = coupling_rows
            .iter()
            .map(|row| (row.coupling.log10(), row.final_energy))
            .collect::<Vec<_>>();
        let absorbed_series = coupling_rows
            .iter()
            .map(|row| (row.coupling.log10(), row.absorbed))
            .collect::<Vec<_>>();
        chart
            .draw_series(LineSeries::new(
                final_series.iter().copied(),
                ShapeStyle::from(&AMBER).stroke_width(4),
            ))
            .map_err(plot_err)?
            .label("final energy")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 26, y)],
                    ShapeStyle::from(&AMBER).stroke_width(4),
                )
            });
        chart
            .draw_series(LineSeries::new(
                absorbed_series.iter().copied(),
                ShapeStyle::from(&CYAN).stroke_width(4),
            ))
            .map_err(plot_err)?
            .label("absorbed")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 26, y)],
                    ShapeStyle::from(&CYAN).stroke_width(4),
                )
            });
        chart
            .draw_series(absorbed_series.iter().map(|point| {
                TriangleMarker::new(*point, 8, ShapeStyle::from(&CYAN.mix(0.95)).filled())
            }))
            .map_err(plot_err)?;
        chart
            .configure_series_labels()
            .background_style(ShapeStyle::from(&PANEL.mix(0.92)).filled())
            .border_style(ShapeStyle::from(&GRID.mix(0.35)).stroke_width(1))
            .label_font(("sans-serif", 16).into_font().color(&TEXT))
            .draw()
            .map_err(plot_err)?;
    }

    {
        let x_values = sink_rows
            .iter()
            .map(|row| row.step as f64)
            .collect::<Vec<_>>();
        let y_values = sink_rows
            .iter()
            .flat_map(|row| [row.energy_no_sink, row.energy_with_sink])
            .collect::<Vec<_>>();
        let (x_min, x_max) = finite_bounds(&x_values);
        let (y_min, y_max) = finite_bounds(&y_values);
        let (y_lo, y_hi) = padded_range(y_min, y_max, 0.05);
        let mut chart = ChartBuilder::on(&panels[2])
            .margin(24)
            .caption(
                "Sink suppression trajectory",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(60)
            .y_label_area_size(84)
            .build_cartesian_2d(x_min..x_max, y_lo..y_hi)
            .map_err(plot_err)?;
        configure_science_mesh(&mut chart, "step", "energy")?;
        let no_sink = sink_rows
            .iter()
            .map(|row| (row.step as f64, row.energy_no_sink))
            .collect::<Vec<_>>();
        let with_sink = sink_rows
            .iter()
            .map(|row| (row.step as f64, row.energy_with_sink))
            .collect::<Vec<_>>();
        chart
            .draw_series(LineSeries::new(
                no_sink.iter().copied(),
                ShapeStyle::from(&MAGENTA).stroke_width(4),
            ))
            .map_err(plot_err)?
            .label("no sink")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 26, y)],
                    ShapeStyle::from(&MAGENTA).stroke_width(4),
                )
            });
        chart
            .draw_series(LineSeries::new(
                with_sink.iter().copied(),
                ShapeStyle::from(&EMERALD).stroke_width(4),
            ))
            .map_err(plot_err)?
            .label("with sink")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 26, y)],
                    ShapeStyle::from(&EMERALD).stroke_width(4),
                )
            });
        chart
            .configure_series_labels()
            .background_style(ShapeStyle::from(&PANEL.mix(0.92)).filled())
            .border_style(ShapeStyle::from(&GRID.mix(0.35)).stroke_width(1))
            .label_font(("sans-serif", 16).into_font().color(&TEXT))
            .draw()
            .map_err(plot_err)?;
    }

    {
        let x_values = field_rows
            .iter()
            .map(|row| row.step as f64)
            .collect::<Vec<_>>();
        let assoc0 = field_rows
            .first()
            .map(|row| row.mean_associator.max(1e-12))
            .unwrap_or(1.0);
        let energy0 = field_rows
            .first()
            .map(|row| row.mean_energy.max(1e-12))
            .unwrap_or(1.0);
        let assoc_series = field_rows
            .iter()
            .map(|row| (row.step as f64, row.mean_associator / assoc0))
            .collect::<Vec<_>>();
        let energy_series = field_rows
            .iter()
            .map(|row| (row.step as f64, row.mean_energy / energy0))
            .collect::<Vec<_>>();
        let (x_min, x_max) = finite_bounds(&x_values);
        let mut chart = ChartBuilder::on(&panels[3])
            .margin(24)
            .caption(
                "Normalized 3D field decay",
                ("sans-serif", 26).into_font().color(&TEXT),
            )
            .x_label_area_size(60)
            .y_label_area_size(84)
            .build_cartesian_2d(x_min..x_max, 0.0..1.08)
            .map_err(plot_err)?;
        configure_science_mesh(&mut chart, "step", "value / initial")?;
        chart
            .draw_series(LineSeries::new(
                assoc_series.iter().copied(),
                ShapeStyle::from(&CYAN).stroke_width(4),
            ))
            .map_err(plot_err)?
            .label("mean associator")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 26, y)],
                    ShapeStyle::from(&CYAN).stroke_width(4),
                )
            });
        chart
            .draw_series(LineSeries::new(
                energy_series.iter().copied(),
                ShapeStyle::from(&EMERALD).stroke_width(4),
            ))
            .map_err(plot_err)?
            .label("mean energy")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 26, y)],
                    ShapeStyle::from(&EMERALD).stroke_width(4),
                )
            });
        chart
            .configure_series_labels()
            .background_style(ShapeStyle::from(&PANEL.mix(0.92)).filled())
            .border_style(ShapeStyle::from(&GRID.mix(0.35)).stroke_width(1))
            .label_font(("sans-serif", 16).into_font().color(&TEXT))
            .draw()
            .map_err(plot_err)?;
    }

    root.present().map_err(plot_err)?;
    Ok(())
}

fn sample_phase_manifold(
    x: f64,
    y: f64,
    rows: &[E183MassPhaseRow],
    sx: f64,
    sy: f64,
) -> (f64, f64, f64) {
    let mut total = 0.0;
    let mut signal = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for row in rows {
        let dx = (x - row.log_m200_median) / sx;
        let dy = (y - row.k) / sy;
        let kernel = (-0.5 * (dx * dx + dy * dy)).exp();
        total += row.power * kernel;
        let coherent = row.power * row.mode_snr.max(0.18) * kernel;
        signal += coherent;
        vx += coherent * row.phase.cos();
        vy += coherent * row.phase.sin();
    }
    let coherence = (vx * vx + vy * vy).sqrt() / signal.max(1e-9);
    (total, coherence.clamp(0.0, 1.0), vy.atan2(vx))
}

fn phase_manifold_color(intensity: f64, coherence: f64) -> RGBColor {
    let cool = lerp_color(
        RGBColor(6, 22, 74),
        RGBColor(0, 176, 255),
        intensity.powf(1.08),
    );
    let warm = lerp_color(
        RGBColor(18, 92, 138),
        RGBColor(240, 246, 176),
        (intensity * coherence).powf(0.90),
    );
    let lifted = lerp_color(cool, warm, (0.12 + 0.52 * coherence).clamp(0.0, 0.82));
    lerp_color(
        lifted,
        RGBColor(248, 252, 255),
        (intensity * coherence * 0.12).clamp(0.0, 0.12),
    )
}

fn collapse_sheet_color(severity_t: f64, radius_t: f64) -> RGBColor {
    let cool = lerp_color(
        RGBColor(16, 24, 62),
        RGBColor(66, 110, 225),
        radius_t.powf(0.72),
    );
    let hot = lerp_color(
        RGBColor(92, 18, 104),
        RGBColor(255, 198, 92),
        severity_t.powf(0.88),
    );
    let mixed = lerp_color(cool, hot, (0.28 + 0.56 * severity_t).clamp(0.0, 0.92));
    lerp_color(
        mixed,
        RGBColor(252, 248, 240),
        (severity_t * 0.18).clamp(0.0, 0.18),
    )
}

fn network_density_color(pathion_t: f64, core_t: f64) -> RGBColor {
    let outer = lerp_color(
        RGBColor(10, 28, 74),
        RGBColor(0, 214, 255),
        pathion_t.powf(0.80),
    );
    let inner = lerp_color(
        RGBColor(66, 0, 120),
        RGBColor(255, 120, 212),
        core_t.powf(0.76),
    );
    let mix_t = (core_t / (pathion_t + core_t + 1e-9))
        .clamp(0.0, 1.0)
        .powf(0.78);
    let mixed = lerp_color(outer, inner, mix_t);
    lerp_color(
        mixed,
        RGBColor(250, 252, 255),
        ((pathion_t + core_t) * 0.16).clamp(0.0, 0.22),
    )
}

fn contour_hit(value: f64, levels: &[f64], epsilon: f64) -> bool {
    levels.iter().any(|level| (value - *level).abs() <= epsilon)
}

fn remap_unit(value: f64, min: f64, max: f64) -> f64 {
    if (max - min).abs() < 1e-12 {
        0.5
    } else {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }
}

fn lerp_color(a: RGBColor, b: RGBColor, t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    let mix = |lhs: u8, rhs: u8| -> u8 {
        (lhs as f64 + (rhs as f64 - lhs as f64) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    RGBColor(mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
}

fn ellipse_points(
    center: (f64, f64),
    rx: f64,
    ry: f64,
    theta: f64,
    steps: usize,
) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(steps + 1);
    for idx in 0..=steps {
        let angle = idx as f64 / steps.max(1) as f64 * TAU;
        let ex = rx * angle.cos();
        let ey = ry * angle.sin();
        let x = center.0 + ex * theta.cos() - ey * theta.sin();
        let y = center.1 + ex * theta.sin() + ey * theta.cos();
        points.push((x, y));
    }
    points
}

fn draw_note(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    x: i32,
    y: i32,
    width: i32,
    title: &str,
    lines: &[String],
    accent: RGBColor,
) -> Result<()> {
    let wrapped = lines
        .iter()
        .flat_map(|line| wrap_text(line, available_chars(width, 18)))
        .collect::<Vec<_>>();
    let height = 54 + wrapped.len() as i32 * 27 + 18;
    area.draw(&Rectangle::new(
        [(x - 16, y - 12), (x + width + 12, y + height)],
        ShapeStyle::from(&BACKGROUND.mix(0.26)).filled(),
    ))
    .map_err(plot_err)?;
    area.draw(&PathElement::new(
        vec![(x, y), (x + width, y)],
        ShapeStyle::from(&accent.mix(0.92)).stroke_width(3),
    ))
    .map_err(plot_err)?;
    area.draw(&Text::new(
        title,
        (x, y + 30),
        ("sans-serif", 24).into_font().color(&TEXT),
    ))
    .map_err(plot_err)?;
    draw_wrapped_lines(area, x, y + 58, 18, &wrapped, &MUTED)?;
    Ok(())
}

fn build_zd_graph(
    rows: &[ZeroDivisorEdgeRow],
    prefix: &str,
    iterations: usize,
) -> (Vec<ZdNode>, Vec<(usize, usize)>) {
    let mut labels = BTreeMap::<usize, String>::new();
    let mut degree = BTreeMap::<usize, usize>::new();
    let mut raw_edges = Vec::with_capacity(rows.len());
    for row in rows {
        labels
            .entry(row.source)
            .or_insert_with(|| row.label_s.clone());
        labels
            .entry(row.target)
            .or_insert_with(|| row.label_t.clone());
        *degree.entry(row.source).or_default() += 1;
        *degree.entry(row.target).or_default() += 1;
        raw_edges.push((row.source, row.target));
    }
    let ids = labels.keys().copied().collect::<Vec<_>>();
    let max_degree = degree.values().copied().max().unwrap_or(1) as f64;
    let mut index_by_id = BTreeMap::new();
    let mut nodes = Vec::with_capacity(ids.len());
    for (idx, id) in ids.iter().enumerate() {
        index_by_id.insert(*id, idx);
        let label = labels.get(id).cloned().unwrap_or_else(|| format!("z{id}"));
        let basis = parse_basis_indices(&label);
        let mean_basis = if basis.is_empty() {
            0.0
        } else {
            basis.iter().copied().map(f64::from).sum::<f64>() / basis.len() as f64
        };
        let basis_gap = if basis.len() >= 2 {
            (basis[1] - basis[0]).abs() as f64
        } else {
            0.0
        };
        let hash_angle = (stable_unit(&format!("{prefix}:{label}:angle")) + 1.0) * 0.5;
        let hash_radius = (stable_unit(&format!("{prefix}:{label}:radius")) + 1.0) * 0.5;
        let angle = TAU * (0.34 * (mean_basis / 32.0) + 0.66 * hash_angle);
        let sign_bias = if label.contains('-') { -0.03 } else { 0.03 };
        let radius = 0.18
            + 0.52 * hash_radius.powf(0.84)
            + 0.12 * (basis_gap / 30.0).powf(0.68)
            + 0.10 * ((degree[id] as f64) / max_degree.max(1.0)).powf(0.48)
            + sign_bias;
        nodes.push(ZdNode {
            id: *id,
            label,
            degree: degree[id],
            x: radius * angle.cos(),
            y: radius * angle.sin() * 0.86,
        });
    }
    let edges = raw_edges
        .into_iter()
        .filter_map(|(src, dst)| Some((*index_by_id.get(&src)?, *index_by_id.get(&dst)?)))
        .collect::<Vec<_>>();
    force_relax_zd_nodes(&mut nodes, &edges, iterations);
    (nodes, edges)
}

fn read_zd_edge_rows(path: &Path) -> Result<Vec<ZeroDivisorEdgeRow>> {
    let body = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut rows = Vec::new();
    for (idx, line) in body.lines().enumerate() {
        if idx == 0 || line.trim().is_empty() {
            continue;
        }
        let mut head = line.splitn(3, ',');
        let source = head
            .next()
            .with_context(|| format!("missing source in {}", path.display()))?
            .trim()
            .parse::<usize>()
            .with_context(|| format!("parse source on line {} in {}", idx + 1, path.display()))?;
        let target = head
            .next()
            .with_context(|| format!("missing target in {}", path.display()))?
            .trim()
            .parse::<usize>()
            .with_context(|| format!("parse target on line {} in {}", idx + 1, path.display()))?;
        let rest = head.next().with_context(|| {
            format!(
                "missing label payload on line {} in {}",
                idx + 1,
                path.display()
            )
        })?;
        let mut depth = 0_i32;
        let mut split_at = None;
        for (char_idx, ch) in rest.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => depth = (depth - 1).max(0),
                ',' if depth == 0 => {
                    split_at = Some(char_idx);
                    break;
                }
                _ => {}
            }
        }
        let split_at = split_at.with_context(|| {
            format!(
                "could not split label pair on line {} in {}",
                idx + 1,
                path.display()
            )
        })?;
        let label_s = rest[..split_at].trim().to_string();
        let label_t = rest[split_at + 1..].trim().to_string();
        rows.push(ZeroDivisorEdgeRow {
            source,
            target,
            label_s,
            label_t,
        });
    }
    Ok(rows)
}

fn force_relax_zd_nodes(nodes: &mut [ZdNode], edges: &[(usize, usize)], iterations: usize) {
    if nodes.len() < 2 {
        return;
    }
    for iter in 0..iterations {
        let mut disp = vec![(0.0_f64, 0.0_f64); nodes.len()];
        let temp = 0.18 * (1.0 - iter as f64 / iterations.max(1) as f64).powf(1.10) + 0.007;
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let dx = nodes[i].x - nodes[j].x;
                let dy = nodes[i].y - nodes[j].y;
                let d2 = dx * dx + dy * dy + 1e-6;
                let dist = d2.sqrt();
                let force = 0.0044 / d2;
                let fx = dx / dist * force;
                let fy = dy / dist * force;
                disp[i].0 += fx;
                disp[i].1 += fy;
                disp[j].0 -= fx;
                disp[j].1 -= fy;
            }
        }
        for &(a, b) in edges {
            let dx = nodes[b].x - nodes[a].x;
            let dy = nodes[b].y - nodes[a].y;
            let dist = (dx * dx + dy * dy + 1e-6).sqrt();
            let target = 0.042 + ((nodes[a].degree + nodes[b].degree) as f64).sqrt() * 0.0018;
            let force = 0.044 * (dist - target);
            let fx = dx / dist * force;
            let fy = dy / dist * force;
            disp[a].0 += fx;
            disp[a].1 += fy;
            disp[b].0 -= fx;
            disp[b].1 -= fy;
        }
        for (node, (dx, dy)) in nodes.iter_mut().zip(disp) {
            node.x = (node.x + dx.clamp(-temp, temp)).clamp(-1.35, 1.35);
            node.y = (node.y + dy.clamp(-temp, temp)).clamp(-1.20, 1.20);
        }
    }
}

fn normalize_zd_nodes(nodes: &mut [ZdNode], half_span: (f64, f64), offset: (f64, f64)) {
    if nodes.is_empty() {
        return;
    }
    let (min_x, max_x) = nodes
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), node| {
            (lo.min(node.x), hi.max(node.x))
        });
    let (min_y, max_y) = nodes
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), node| {
            (lo.min(node.y), hi.max(node.y))
        });
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;
    let span_x = (max_x - min_x).max(1e-6);
    let span_y = (max_y - min_y).max(1e-6);
    let scale = ((half_span.0 * 2.0) / span_x).min((half_span.1 * 2.0) / span_y);
    for node in nodes {
        node.x = offset.0 + (node.x - center_x) * scale;
        node.y = offset.1 + (node.y - center_y) * scale;
    }
}

fn swirl_zd_nodes(nodes: &mut [ZdNode], twist: f64, squash: f64) {
    for node in nodes {
        let radius = (node.x * node.x + node.y * node.y).sqrt();
        let angle = node.y.atan2(node.x) + twist * (0.30 + 0.70 * radius);
        let warped = radius.powf(0.94);
        node.x = warped * angle.cos();
        node.y = warped * angle.sin() * squash;
    }
}

fn parse_basis_indices(label: &str) -> Vec<i32> {
    let mut values = Vec::new();
    let mut current = String::new();
    for ch in label.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            values.push(current.parse::<i32>().unwrap_or(0));
            current.clear();
        }
    }
    if !current.is_empty() {
        values.push(current.parse::<i32>().unwrap_or(0));
    }
    values
}

fn render_e183_phase_plate_v2(path: &Path, repo_root: &Path, project: &ProjectBlock) -> Result<()> {
    let lie_rows = read_csv_rows::<E183LieJordanRow>(
        &repo_root.join("data/results/e183/lie_jordan_full.csv"),
    )?;
    let phase_rows = read_csv_rows::<E183MassPhaseRow>(
        &repo_root.join("data/results/e183/mass_binned_phase.csv"),
    )?;
    let summary_rows = read_csv_rows::<E183MassPhaseSummaryRow>(
        &repo_root.join("data/results/e183/mass_binned_phase.summary.csv"),
    )?;
    let corr_rows = read_csv_rows::<E183CrossAlgebraRow>(
        &repo_root.join("data/results/e183/cross_algebra_correlation.csv"),
    )?;

    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_err)?;
    draw_background_field(&root, CYAN, MAGENTA)?;
    draw_title(
        &root,
        "SCIENCE PLATE: E-183 MASS-PHASE MANIFOLD",
        &format!(
            "Mass-binned phase structure from data/results/e183, rendered as a continuous field with phase texture and coherence overlays | project {}",
            project.version
        ),
    )?;

    let x_values = phase_rows
        .iter()
        .filter(|row| row.power > 1e-12)
        .map(|row| row.log_m200_median)
        .collect::<Vec<_>>();
    let y_values = phase_rows.iter().map(|row| row.k).collect::<Vec<_>>();
    let (x_min, x_max) = finite_bounds(&x_values);
    let (y_min, y_max) = finite_bounds(&y_values);
    let x_lo = x_min - 0.18;
    let x_hi = x_max + 0.18;
    let y_lo = y_min - 0.48;
    let y_hi = y_max + 0.48;
    let field = root.margin(240, 110, 110, 90);
    let mut chart = ChartBuilder::on(&field)
        .margin(10)
        .x_label_area_size(72)
        .y_label_area_size(88)
        .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .label_style(("sans-serif", 20).into_font().color(&TEXT))
        .axis_style(ShapeStyle::from(&TEXT).stroke_width(2))
        .light_line_style(ShapeStyle::from(&GRID.mix(0.16)).stroke_width(1))
        .bold_line_style(ShapeStyle::from(&GRID.mix(0.28)).stroke_width(1))
        .x_desc("halo-mass median log10(M200)")
        .y_desc("harmonic wavenumber k")
        .x_labels(7)
        .y_labels(7)
        .draw()
        .map_err(plot_err)?;

    let nx = 240;
    let ny = 180;
    let dx = (x_hi - x_lo) / nx as f64;
    let dy = (y_hi - y_lo) / ny as f64;
    let sx = 0.11;
    let sy = 0.29;
    let mut field_samples = Vec::with_capacity(nx * ny);
    let mut max_total = 0.0_f64;
    for iy in 0..ny {
        let yc = y_lo + (iy as f64 + 0.5) * dy;
        for ix in 0..nx {
            let xc = x_lo + (ix as f64 + 0.5) * dx;
            let (total, coherence, angle) = sample_phase_manifold(xc, yc, &phase_rows, sx, sy);
            max_total = max_total.max(total);
            field_samples.push((xc, yc, total, coherence, angle));
        }
    }

    chart
        .draw_series(
            field_samples
                .iter()
                .filter_map(|(xc, yc, total, coherence, _)| {
                    let intensity = if max_total <= 0.0 {
                        0.0
                    } else {
                        total / max_total
                    };
                    (intensity > 0.028).then(|| {
                        Rectangle::new(
                            [
                                (*xc - dx * 0.52, *yc - dy * 0.52),
                                (*xc + dx * 0.52, *yc + dy * 0.52),
                            ],
                            ShapeStyle::from(&phase_manifold_color(intensity, *coherence)).filled(),
                        )
                    })
                }),
        )
        .map_err(plot_err)?;

    let contour_levels = [0.14, 0.27, 0.41, 0.58, 0.74];
    chart
        .draw_series(
            field_samples
                .iter()
                .filter_map(|(xc, yc, total, coherence, _)| {
                    let intensity = if max_total <= 0.0 {
                        0.0
                    } else {
                        total / max_total
                    };
                    (intensity > 0.12 && contour_hit(intensity, &contour_levels, 0.012)).then(
                        || {
                            Rectangle::new(
                                [
                                    (*xc - dx * 0.52, *yc - dy * 0.52),
                                    (*xc + dx * 0.52, *yc + dy * 0.52),
                                ],
                                ShapeStyle::from(
                                    &RGBColor(246, 250, 255)
                                        .mix((0.04 + 0.12 * coherence).clamp(0.04, 0.18)),
                                ),
                            )
                        },
                    )
                }),
        )
        .map_err(plot_err)?;

    let max_power = phase_rows
        .iter()
        .map(|row| row.power)
        .fold(0.0_f64, f64::max);
    let max_snr = phase_rows
        .iter()
        .map(|row| row.mode_snr)
        .fold(0.0_f64, f64::max);
    let mut peaks = phase_rows.iter().collect::<Vec<_>>();
    peaks.sort_by(|a, b| {
        b.power
            .partial_cmp(&a.power)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for row in peaks
        .iter()
        .copied()
        .filter(|row| row.power > max_power * 0.13)
    {
        let t = (row.power / max_power.max(1e-9)).clamp(0.0, 1.0);
        for (scale, alpha, width) in [(1.0, 0.16, 1), (1.55, 0.11, 1), (2.15, 0.08, 1)] {
            chart
                .draw_series(std::iter::once(PathElement::new(
                    ellipse_points(
                        (row.log_m200_median, row.k),
                        (0.055 + 0.085 * t) * scale,
                        (0.22 + 0.36 * t) * scale,
                        row.phase * 0.22,
                        72,
                    ),
                    ShapeStyle::from(&RGBColor(248, 251, 255).mix(alpha)).stroke_width(width),
                )))
                .map_err(plot_err)?;
        }
    }

    chart
        .draw_series(phase_rows.iter().filter(|row| row.power > 0.0).map(|row| {
            let t = (row.mode_snr / max_snr.max(1e-9)).clamp(0.0, 1.0);
            Circle::new(
                (row.log_m200_median, row.k),
                (3.0 + 4.0 * t).round() as i32,
                ShapeStyle::from(&RGBColor(250, 251, 255).mix(0.55 + 0.35 * t)).filled(),
            )
        }))
        .map_err(plot_err)?;

    let glyph_cols = 28;
    let glyph_rows = 18;
    for gy in 0..glyph_rows {
        let y = y_lo + (gy as f64 + 0.5) * (y_hi - y_lo) / glyph_rows as f64;
        for gx in 0..glyph_cols {
            let x = x_lo + (gx as f64 + 0.5) * (x_hi - x_lo) / glyph_cols as f64;
            let (total, coherence, angle) = sample_phase_manifold(x, y, &phase_rows, sx, sy);
            let intensity = if max_total <= 0.0 {
                0.0
            } else {
                total / max_total
            };
            if intensity < 0.20 || coherence < 0.12 {
                continue;
            }
            let glyph_len = (0.045 + 0.10 * coherence).min(0.18);
            let dx_glyph = glyph_len * angle.cos();
            let dy_glyph = glyph_len * angle.sin();
            chart
                .draw_series(std::iter::once(PathElement::new(
                    vec![(x - dx_glyph, y - dy_glyph), (x + dx_glyph, y + dy_glyph)],
                    ShapeStyle::from(&RGBColor(248, 250, 252).mix(0.42 + 0.46 * coherence))
                        .stroke_width(if intensity > 0.42 { 2 } else { 1 }),
                )))
                .map_err(plot_err)?;
        }
    }

    for row in peaks.iter().copied().take(4) {
        chart
            .draw_series(std::iter::once(Text::new(
                format!("M={:.2}, k={:.2}", row.log_m200_median, row.k),
                (row.log_m200_median + 0.05, row.k + 0.18),
                ("sans-serif", 18).into_font().color(&TEXT),
            )))
            .map_err(plot_err)?;
    }

    let strongest_corr = corr_rows
        .iter()
        .max_by(|a, b| {
            a.excess
                .partial_cmp(&b.excess)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|row| format!("cross-algebra excess | {} -> {:.3}", row.pair, row.excess))
        .unwrap_or_else(|| "cross-algebra excess | none".to_string());
    let strongest_rayleigh = summary_rows
        .iter()
        .max_by(|a, b| {
            a.rayleigh_r
                .partial_cmp(&b.rayleigh_r)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|row| {
            format!(
                "phase coherence crest | k={:.2}, R={:.3}",
                row.k, row.rayleigh_r
            )
        })
        .unwrap_or_else(|| "phase coherence crest | none".to_string());
    root.draw(&PathElement::new(
        vec![(2330, 290), (3010, 290)],
        ShapeStyle::from(&CYAN.mix(0.88)).stroke_width(3),
    ))
    .map_err(plot_err)?;
    root.draw(&Text::new(
        "Readout",
        (2330, 324),
        ("sans-serif", 24).into_font().color(&TEXT),
    ))
    .map_err(plot_err)?;
    draw_wrapped_lines(
        &root,
        2330,
        354,
        18,
        &[
            format!(
                "peak | log10(M200) {:.2}, k {:.2}, SNR {:.2}",
                peaks.first().map(|row| row.log_m200_median).unwrap_or(0.0),
                peaks.first().map(|row| row.k).unwrap_or(0.0),
                peaks.first().map(|row| row.mode_snr).unwrap_or(0.0),
            ),
            strongest_rayleigh,
            strongest_corr,
            format!("spectral comb families | {}", lie_rows.len()),
        ],
        &MUTED,
    )?;

    root.present().map_err(plot_err)?;
    Ok(())
}

fn render_gravastar_plate_v2(path: &Path, repo_root: &Path, project: &ProjectBlock) -> Result<()> {
    let radial = read_csv_rows::<GravastarRadialRow>(
        &repo_root.join("data/csv/gravastar_radial_stability.csv"),
    )?;
    let ligo = read_csv_rows::<GravastarLigoRow>(
        &repo_root.join("data/csv/gravastar_ligo_mass_sweep.csv"),
    )?;
    let genesis = read_csv_rows::<GenesisGravastarRow>(
        &repo_root.join("data/csv/genesis_gravastar_bridge.csv"),
    )?;

    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_err)?;
    draw_background_field(&root, AMBER, CYAN)?;
    draw_title(
        &root,
        "SCIENCE PLATE: GRAVASTAR COLLAPSE / BRIDGE LANDSCAPE",
        &format!(
            "Radial collapse severity from gravastar sweeps, with stable genesis bridge branches rendered as a secondary plume | project {}",
            project.version
        ),
    )?;

    let mass_values = radial.iter().map(|row| row.m_target).collect::<Vec<_>>();
    let comp_values = radial
        .iter()
        .map(|row| row.core_compactness)
        .collect::<Vec<_>>();
    let severity_values = radial
        .iter()
        .map(|row| row.dM_drho_c.abs().max(1e-9).log10())
        .collect::<Vec<_>>();
    let radius_lookup = ligo
        .iter()
        .map(|row| {
            (
                (
                    (row.m_target * 100.0).round() as i64,
                    (row.core_compactness * 1000.0).round() as i64,
                ),
                row.r2.max(1e-9).log10(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let radius_values = ligo
        .iter()
        .map(|row| row.r2.max(1e-9).log10())
        .collect::<Vec<_>>();
    let (x_min, x_max) = finite_bounds(&mass_values);
    let (y_min, y_max) = finite_bounds(&comp_values);
    let (severity_min, severity_max) = finite_bounds(&severity_values);
    let (radius_min, radius_max) = finite_bounds(&radius_values);
    let field = root.margin(240, 110, 520, 90);
    let mut chart = ChartBuilder::on(&field)
        .margin(8)
        .x_label_area_size(72)
        .y_label_area_size(88)
        .build_cartesian_2d(
            (x_min - 2.0)..(x_max + 2.0),
            (y_min - 0.035)..(y_max + 0.035),
        )
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .label_style(("sans-serif", 20).into_font().color(&TEXT))
        .axis_style(ShapeStyle::from(&TEXT).stroke_width(2))
        .light_line_style(ShapeStyle::from(&GRID.mix(0.16)).stroke_width(1))
        .bold_line_style(ShapeStyle::from(&GRID.mix(0.28)).stroke_width(1))
        .x_desc("target mass")
        .y_desc("core compactness")
        .x_labels(8)
        .y_labels(6)
        .draw()
        .map_err(plot_err)?;

    let nx = 220;
    let ny = 160;
    let x_lo = x_min - 2.0;
    let x_hi = x_max + 2.0;
    let y_lo = y_min - 0.035;
    let y_hi = y_max + 0.035;
    let dx = (x_hi - x_lo) / nx as f64;
    let dy = (y_hi - y_lo) / ny as f64;
    let sx = 7.5;
    let sy = 0.055;
    let mut field_samples = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        let yc = y_lo + (iy as f64 + 0.5) * dy;
        for ix in 0..nx {
            let xc = x_lo + (ix as f64 + 0.5) * dx;
            let mut weight_total = 0.0;
            let mut severity_sum = 0.0;
            let mut radius_sum = 0.0;
            for row in &radial {
                let dx_n = (xc - row.m_target) / sx;
                let dy_n = (yc - row.core_compactness) / sy;
                let kernel = (-0.5 * (dx_n * dx_n + dy_n * dy_n)).exp();
                let severity = row.dM_drho_c.abs().max(1e-9).log10();
                let radius_log = *radius_lookup
                    .get(&(
                        (row.m_target * 100.0).round() as i64,
                        (row.core_compactness * 1000.0).round() as i64,
                    ))
                    .unwrap_or(&0.0);
                weight_total += kernel;
                severity_sum += severity * kernel;
                radius_sum += radius_log * kernel;
            }
            field_samples.push((
                xc,
                yc,
                severity_sum / weight_total.max(1e-9),
                radius_sum / weight_total.max(1e-9),
            ));
        }
    }

    chart
        .draw_series(field_samples.iter().map(|(xc, yc, severity, radius_log)| {
            Rectangle::new(
                [
                    (*xc - dx * 0.52, *yc - dy * 0.52),
                    (*xc + dx * 0.52, *yc + dy * 0.52),
                ],
                ShapeStyle::from(&collapse_sheet_color(
                    remap_unit(*severity, severity_min, severity_max),
                    remap_unit(*radius_log, radius_min, radius_max),
                ))
                .filled(),
            )
        }))
        .map_err(plot_err)?;

    let contour_levels = [0.12, 0.28, 0.46, 0.64, 0.82];
    chart
        .draw_series(field_samples.iter().filter_map(|(xc, yc, severity, _)| {
            let t = remap_unit(*severity, severity_min, severity_max);
            contour_hit(t, &contour_levels, 0.015).then(|| {
                Rectangle::new(
                    [
                        (*xc - dx * 0.52, *yc - dy * 0.52),
                        (*xc + dx * 0.52, *yc + dy * 0.52),
                    ],
                    ShapeStyle::from(&RGBColor(246, 250, 255).mix(0.08)),
                )
            })
        }))
        .map_err(plot_err)?;

    chart
        .draw_series(radial.iter().map(|row| {
            let severity = row.dM_drho_c.abs().max(1e-9).log10();
            let t = remap_unit(severity, severity_min, severity_max);
            Circle::new(
                (row.m_target, row.core_compactness),
                (3.0 + 3.0 * t).round() as i32,
                ShapeStyle::from(&RGBColor(248, 251, 255).mix(0.40 + 0.30 * t)),
            )
        }))
        .map_err(plot_err)?;

    if let Some(row) = radial.iter().max_by(|a, b| {
        a.dM_drho_c
            .abs()
            .partial_cmp(&b.dM_drho_c.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        chart
            .draw_series(std::iter::once(Text::new(
                format!(
                    "collapse wall | M {:.0}, c {:.1}",
                    row.m_target, row.core_compactness
                ),
                (row.m_target - 18.0, row.core_compactness - 0.018),
                ("sans-serif", 19).into_font().color(&TEXT),
            )))
            .map_err(plot_err)?;
    }

    let stable_genesis = genesis
        .iter()
        .filter(|row| row.is_stable)
        .collect::<Vec<_>>();
    let ratio_logs = stable_genesis
        .iter()
        .map(|row| (row.r2 / row.r1.max(1e-12)).max(1e-12).log10())
        .collect::<Vec<_>>();
    let mass_logs = stable_genesis
        .iter()
        .map(|row| row.m_total.abs().max(1e-18).log10())
        .collect::<Vec<_>>();
    let (ratio_min, ratio_max) = finite_bounds(&ratio_logs);
    let (mass_log_min, mass_log_max) = finite_bounds(&mass_logs);
    let strip = RectBox {
        x0: 2670,
        y0: 430,
        x1: 3040,
        y1: 2460,
    };
    let strip_step = 8;
    for y in (strip.y0..strip.y1).step_by(strip_step as usize) {
        for x in (strip.x0..strip.x1).step_by(strip_step as usize) {
            let gamma = 1.5 + (x - strip.x0) as f64 / (strip.x1 - strip.x0) as f64;
            let ratio = ratio_min
                + (1.0 - (y - strip.y0) as f64 / (strip.y1 - strip.y0) as f64)
                    * (ratio_max - ratio_min);
            let mut total = 0.0;
            let mut mass_mix = 0.0;
            for row in &stable_genesis {
                let row_ratio = (row.r2 / row.r1.max(1e-12)).max(1e-12).log10();
                let dx_n = (gamma - row.gamma) / 0.11;
                let dy_n = (ratio - row_ratio) / ((ratio_max - ratio_min).abs().max(1e-6) * 0.08);
                let kernel = (-0.5 * (dx_n * dx_n + dy_n * dy_n)).exp();
                let mass_log = row.m_total.abs().max(1e-18).log10();
                total += kernel;
                mass_mix += kernel * remap_unit(mass_log, mass_log_min, mass_log_max);
            }
            if total < 0.03 {
                continue;
            }
            let t = (1.0 - (-0.22 * total).exp()).clamp(0.0, 1.0);
            let mass_t = mass_mix / total.max(1e-9);
            let color = lerp_color(
                lerp_color(RGBColor(20, 40, 88), RGBColor(66, 217, 255), t.powf(0.88)),
                RGBColor(255, 240, 180),
                (0.15 + 0.50 * mass_t).clamp(0.0, 0.72),
            );
            root.draw(&Rectangle::new(
                [
                    (x, y),
                    (
                        (x + strip_step).min(strip.x1),
                        (y + strip_step).min(strip.y1),
                    ),
                ],
                ShapeStyle::from(&color.mix(0.86)).filled(),
            ))
            .map_err(plot_err)?;
        }
    }
    for gamma in [1.5, 2.0, 2.5] {
        let x = strip.x0 + ((gamma - 1.5) * (strip.x1 - strip.x0) as f64).round() as i32;
        root.draw(&PathElement::new(
            vec![(x, strip.y0), (x, strip.y1)],
            ShapeStyle::from(&GRID.mix(0.24)).stroke_width(1),
        ))
        .map_err(plot_err)?;
        root.draw(&Text::new(
            format!("gamma {:.1}", gamma),
            (x - 34, strip.y0 - 18),
            ("sans-serif", 18).into_font().color(&TEXT),
        ))
        .map_err(plot_err)?;
    }
    for row in &stable_genesis {
        let ratio = (row.r2 / row.r1.max(1e-12)).max(1e-12).log10();
        let px = strip.x0 + ((row.gamma - 1.5) * (strip.x1 - strip.x0) as f64).round() as i32;
        let py = strip.y1
            - (remap_unit(ratio, ratio_min, ratio_max) * (strip.y1 - strip.y0) as f64).round()
                as i32;
        let mass_t = remap_unit(
            row.m_total.abs().max(1e-18).log10(),
            mass_log_min,
            mass_log_max,
        );
        let color = lerp_color(RGBColor(164, 245, 255), RGBColor(255, 228, 154), mass_t);
        root.draw(&Circle::new(
            (px, py),
            (2.0 + 3.0 * mass_t).round() as i32,
            ShapeStyle::from(&color.mix(0.84)).filled(),
        ))
        .map_err(plot_err)?;
    }

    let unstable_count = radial
        .iter()
        .filter(|row| !row.harrison_wheeler_stable)
        .count();
    draw_note(
        &root,
        192,
        2205,
        1050,
        "Radial Sheet",
        &[
            format!(
                "{} / {} radial sweep samples violate the Harrison-Wheeler criterion",
                unstable_count,
                radial.len()
            ),
            "luminance tracks collapse severity log10 |dM/drho_c|; cooler structure preserves shell-radius context from the LIGO sweep".to_string(),
        ],
        AMBER,
    )?;
    draw_note(
        &root,
        2600,
        300,
        380,
        "Stable Bridge Branches",
        &[
            format!(
                "{} genesis microbranches remain stable",
                stable_genesis.len()
            ),
            "gamma = 1.5 supports the widest shells; gamma = 2.5 contracts toward R2/R1 ~ 1"
                .to_string(),
        ],
        CYAN,
    )?;

    root.present().map_err(plot_err)?;
    Ok(())
}

fn render_algebra_plate_v2(path: &Path, repo_root: &Path, project: &ProjectBlock) -> Result<()> {
    let mass_rows =
        read_csv_rows::<SedenionMassRow>(&repo_root.join("data/csv/sedenion_mass_spectrum.csv"))?;
    let coupling_rows = read_csv_rows::<PathionCouplingRow>(
        &repo_root.join("data/csv/pathion_coupling_sweep.csv"),
    )?;
    let sink_rows =
        read_csv_rows::<PathionSinkRow>(&repo_root.join("data/csv/pathion_sink_compare.csv"))?;
    let field_rows = read_csv_rows::<SedenionFieldMetricRow>(
        &repo_root.join("data/csv/sedenion_field_metrics_3D.csv"),
    )?;
    let pathion_rows = read_zd_edge_rows(&repo_root.join("data/csv/pathion_zd_edges.csv"))?;
    let sedenion_rows = read_zd_edge_rows(&repo_root.join("data/csv/sedenion_zd_edges.csv"))?;

    let root = BitMapBackend::new(path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_err)?;
    draw_background_field(&root, MAGENTA, EMERALD)?;
    draw_title(
        &root,
        "SCIENCE PLATE: PATHION ZERO-DIVISOR INTERACTION GRAPH",
        &format!(
            "Zero-divisor interaction graph from pathion and sedenion edge tables, with mass spectrum, coupling response, damping trajectory, and field relaxation summarized as annotations | project {}",
            project.version
        ),
    )?;

    let field_rect = RectBox {
        x0: 120,
        y0: 240,
        x1: 3040,
        y1: 2660,
    };
    let (mut pathion_nodes, pathion_edges) = build_zd_graph(&pathion_rows, "pathion", 84);
    let (mut sedenion_nodes, sedenion_edges) = build_zd_graph(&sedenion_rows, "sedenion", 108);
    normalize_zd_nodes(&mut pathion_nodes, (0.84, 0.78), (0.0, 0.02));
    normalize_zd_nodes(&mut sedenion_nodes, (0.26, 0.22), (0.03, -0.01));
    swirl_zd_nodes(&mut pathion_nodes, 0.28, 0.94);
    swirl_zd_nodes(&mut sedenion_nodes, 0.72, 0.90);

    let step = 12;
    let mut density_samples = Vec::new();
    let mut max_pathion_density = 0.0_f64;
    let mut max_core_density = 0.0_f64;
    for y in (field_rect.y0..field_rect.y1).step_by(step as usize) {
        for x in (field_rect.x0..field_rect.x1).step_by(step as usize) {
            let (nx, ny) = unproject_norm(field_rect, x, y);
            let mut pathion_density = 0.0;
            let mut core_density = 0.0;
            for node in &pathion_nodes {
                let dx = nx - node.x;
                let dy = ny - node.y;
                let d2 = dx * dx + dy * dy;
                let sigma = 0.024 + (node.degree as f64).sqrt() * 0.0018;
                pathion_density += node.degree as f64 * (-d2 / (2.0 * sigma * sigma)).exp();
            }
            for node in &sedenion_nodes {
                let dx = nx - node.x;
                let dy = ny - node.y;
                let d2 = dx * dx + dy * dy;
                let sigma = 0.022 + (node.degree as f64).sqrt() * 0.0019;
                core_density += 1.35 * node.degree as f64 * (-d2 / (2.0 * sigma * sigma)).exp();
            }
            max_pathion_density = max_pathion_density.max(pathion_density);
            max_core_density = max_core_density.max(core_density);
            density_samples.push((x, y, pathion_density, core_density));
        }
    }

    for (x, y, pathion_density, core_density) in &density_samples {
        let pathion_t = if max_pathion_density <= 0.0 {
            0.0
        } else {
            pathion_density / max_pathion_density
        };
        let core_t = if max_core_density <= 0.0 {
            0.0
        } else {
            core_density / max_core_density
        };
        let intensity = pathion_t + 1.2 * core_t;
        if intensity < 0.014 {
            continue;
        }
        root.draw(&Rectangle::new(
            [
                (*x, *y),
                (
                    ((*x + step).min(field_rect.x1)),
                    ((*y + step).min(field_rect.y1)),
                ),
            ],
            ShapeStyle::from(&network_density_color(pathion_t, core_t).mix(0.84)).filled(),
        ))
        .map_err(plot_err)?;
    }

    for (x, y, pathion_density, core_density) in &density_samples {
        let t = ((pathion_density / max_pathion_density.max(1e-9))
            + 1.15 * (core_density / max_core_density.max(1e-9)))
            / 2.15;
        if contour_hit(t, &[0.16, 0.30, 0.47, 0.64, 0.80], 0.016) {
            root.draw(&Rectangle::new(
                [
                    (*x, *y),
                    (
                        ((*x + step).min(field_rect.x1)),
                        ((*y + step).min(field_rect.y1)),
                    ),
                ],
                ShapeStyle::from(&RGBColor(246, 250, 255).mix(0.08)),
            ))
            .map_err(plot_err)?;
        }
    }

    let pathion_points = pathion_nodes
        .iter()
        .map(|node| (project_norm(field_rect, node.x, node.y), node))
        .collect::<Vec<_>>();
    let sedenion_points = sedenion_nodes
        .iter()
        .map(|node| (project_norm(field_rect, node.x, node.y), node))
        .collect::<Vec<_>>();
    for &(a, b) in &pathion_edges {
        let pa = project_norm(field_rect, pathion_nodes[a].x, pathion_nodes[a].y);
        let pb = project_norm(field_rect, pathion_nodes[b].x, pathion_nodes[b].y);
        let parity = ((pathion_nodes[a].id ^ pathion_nodes[b].id) & 1) as f64;
        let accent = if parity > 0.5 {
            RGBColor(103, 214, 255)
        } else {
            RGBColor(246, 113, 206)
        };
        root.draw(&PathElement::new(
            vec![pa, pb],
            ShapeStyle::from(&accent.mix(0.09)).stroke_width(1),
        ))
        .map_err(plot_err)?;
    }
    for &(a, b) in &sedenion_edges {
        let pa = project_norm(field_rect, sedenion_nodes[a].x, sedenion_nodes[a].y);
        let pb = project_norm(field_rect, sedenion_nodes[b].x, sedenion_nodes[b].y);
        root.draw(&PathElement::new(
            vec![pa, pb],
            ShapeStyle::from(&RGBColor(255, 219, 155).mix(0.22)).stroke_width(2),
        ))
        .map_err(plot_err)?;
    }

    let mut pathion_hubs = pathion_nodes.iter().collect::<Vec<_>>();
    pathion_hubs.sort_by_key(|node| std::cmp::Reverse(node.degree));
    for hub in pathion_hubs.iter().take(12).copied() {
        let center = project_norm(field_rect, hub.x, hub.y);
        for (scale, alpha, width) in [(1.0, 0.14, 1), (1.55, 0.10, 1), (2.10, 0.07, 1)] {
            let rx = 26.0 + scale * 10.0 + hub.degree as f64 * 0.55;
            let ry = 18.0 + scale * 8.0 + hub.degree as f64 * 0.42;
            let pts = (0..=60)
                .map(|idx| {
                    let angle = idx as f64 / 60.0 * TAU;
                    (
                        (center.0 as f64 + rx * angle.cos()).round() as i32,
                        (center.1 as f64 + ry * angle.sin()).round() as i32,
                    )
                })
                .collect::<Vec<_>>();
            root.draw(&PathElement::new(
                pts,
                ShapeStyle::from(&RGBColor(240, 248, 255).mix(alpha)).stroke_width(width),
            ))
            .map_err(plot_err)?;
        }
    }

    for (point, node) in &pathion_points {
        root.draw(&Circle::new(
            *point,
            (2.0 + (node.degree as f64).sqrt() * 0.5).round() as i32,
            ShapeStyle::from(&RGBColor(235, 244, 255).mix(0.78)).filled(),
        ))
        .map_err(plot_err)?;
    }
    for (point, node) in &sedenion_points {
        root.draw(&Circle::new(
            *point,
            (3.0 + (node.degree as f64).sqrt() * 0.65).round() as i32,
            ShapeStyle::from(&RGBColor(255, 231, 168).mix(0.88)).filled(),
        ))
        .map_err(plot_err)?;
    }

    for node in pathion_hubs.iter().take(6).copied() {
        let point = project_norm(field_rect, node.x, node.y);
        root.draw(&Text::new(
            short_text(&node.label, 12),
            (point.0 + 10, point.1 - 10),
            ("sans-serif", 18).into_font().color(&TEXT),
        ))
        .map_err(plot_err)?;
    }
    let mut core_hubs = sedenion_nodes.iter().collect::<Vec<_>>();
    core_hubs.sort_by_key(|node| std::cmp::Reverse(node.degree));
    for node in core_hubs.iter().take(4).copied() {
        let point = project_norm(field_rect, node.x, node.y);
        root.draw(&Text::new(
            short_text(&node.label, 12),
            (point.0 + 10, point.1 - 10),
            ("sans-serif", 18).into_font().color(&TEXT),
        ))
        .map_err(plot_err)?;
    }
    root.draw(&Text::new(
        "sedenion zero-divisor subgraph",
        (1515, 1365),
        ("sans-serif", 30).into_font().color(&AMBER),
    ))
    .map_err(plot_err)?;

    let coupling_cross = coupling_rows
        .iter()
        .find(|row| row.absorbed > row.final_energy)
        .map(|row| row.coupling.log10())
        .unwrap_or(0.0);
    let sink_terminal_delta = sink_rows
        .last()
        .map(|row| row.energy_no_sink - row.energy_with_sink)
        .unwrap_or(0.0);
    let field_tail = field_rows
        .last()
        .map(|row| {
            row.mean_energy
                / field_rows
                    .first()
                    .map(|base| base.mean_energy)
                    .unwrap_or(1.0)
        })
        .unwrap_or(0.0);
    let mass_peak_15 = mass_rows
        .iter()
        .find(|row| row.mode_n == 15)
        .map(|row| row.predicted_mass)
        .unwrap_or(0.0);
    let mass_peak_25 = mass_rows
        .iter()
        .find(|row| row.mode_n == 25)
        .map(|row| row.predicted_mass)
        .unwrap_or(0.0);
    draw_note(
        &root,
        154,
        276,
        760,
        "Network Summary",
        &[
            format!(
                "pathion graph | {} vertices, {} edges",
                pathion_nodes.len(),
                pathion_edges.len()
            ),
            format!(
                "sedenion subgraph | {} vertices, {} edges",
                sedenion_nodes.len(),
                sedenion_edges.len()
            ),
        ],
        CYAN,
    )?;
    draw_note(
        &root,
        156,
        2240,
        910,
        "Mass Spectrum / Coupling Response",
        &[
            format!(
                "mass spectrum anchors | n=15 -> {:.1} Mo, n=25 -> {:.1} Mo",
                mass_peak_15, mass_peak_25
            ),
            format!(
                "absorbed energy exceeds final energy near log10 coupling {:.2}",
                coupling_cross
            ),
        ],
        MAGENTA,
    )?;
    draw_note(
        &root,
        2140,
        2240,
        860,
        "Damping / Field Relaxation",
        &[
            format!("terminal damping offset | {:.2}", sink_terminal_delta),
            format!(
                "late 3D field energy remains at {:.3} of the initial value",
                field_tail
            ),
        ],
        EMERALD,
    )?;

    root.present().map_err(plot_err)?;
    Ok(())
}

fn mirror_to_book_assets(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create asset parent {}", parent.display()))?;
    }
    fs::copy(src, dst).with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
    Ok(())
}

fn short_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let clipped = text
        .char_indices()
        .take_while(|(idx, _)| *idx < max_chars.saturating_sub(3))
        .map(|(_, ch)| ch)
        .collect::<String>();
    format!("{clipped}...")
}

fn draw_background_field(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    primary: RGBColor,
    secondary: RGBColor,
) -> Result<()> {
    for x in (60..WIDTH as i32).step_by(180) {
        area.draw(&PathElement::new(
            vec![(x, 0), (x, HEIGHT as i32)],
            ShapeStyle::from(&GRID.mix(0.08)).stroke_width(1),
        ))
        .map_err(plot_err)?;
    }
    for y in (40..HEIGHT as i32).step_by(180) {
        area.draw(&PathElement::new(
            vec![(0, y), (WIDTH as i32, y)],
            ShapeStyle::from(&GRID.mix(0.08)).stroke_width(1),
        ))
        .map_err(plot_err)?;
    }
    for radius in [520, 860, 1220, 1680] {
        area.draw(&Circle::new(
            (WIDTH as i32 - 340, 220),
            radius,
            ShapeStyle::from(&primary.mix(0.05)).stroke_width(1),
        ))
        .map_err(plot_err)?;
        area.draw(&Circle::new(
            (180, HEIGHT as i32 - 260),
            radius,
            ShapeStyle::from(&secondary.mix(0.05)).stroke_width(1),
        ))
        .map_err(plot_err)?;
    }
    for idx in 0..9 {
        let y = 220 + idx * 260;
        area.draw(&PathElement::new(
            vec![(0, y), (WIDTH as i32, y - 180)],
            ShapeStyle::from(&primary.mix(0.03)).stroke_width(1),
        ))
        .map_err(plot_err)?;
    }
    Ok(())
}

fn draw_title(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    title: &str,
    subtitle: &str,
) -> Result<()> {
    area.draw(&Text::new(
        title,
        (80, 86),
        ("sans-serif", 58).into_font().color(&TEXT),
    ))
    .map_err(plot_err)?;
    area.draw(&Text::new(
        subtitle,
        (80, 150),
        ("sans-serif", 24).into_font().color(&MUTED),
    ))
    .map_err(plot_err)?;
    area.draw(&PathElement::new(
        vec![(80, 188), (3080, 188)],
        ShapeStyle::from(&GRID.mix(0.7)).stroke_width(2),
    ))
    .map_err(plot_err)?;
    Ok(())
}

fn connector_anchor(rect: RectBox, point: (i32, i32)) -> (i32, i32) {
    let candidates = [
        ((rect.x0 + rect.x1) / 2, rect.y0),
        ((rect.x0 + rect.x1) / 2, rect.y1),
        (rect.x0, (rect.y0 + rect.y1) / 2),
        (rect.x1, (rect.y0 + rect.y1) / 2),
    ];
    candidates
        .into_iter()
        .min_by_key(|candidate| {
            let dx = candidate.0 - point.0;
            let dy = candidate.1 - point.1;
            dx * dx + dy * dy
        })
        .unwrap_or((rect.x0, rect.y0))
}

fn draw_surface_callout(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    field_rect: RectBox,
    surface: &RepoSurface,
) -> Result<()> {
    let point = project_norm(field_rect, surface.point.0, surface.point.1);
    let anchor = connector_anchor(surface.rect, point);
    area.draw(&Rectangle::new(
        [
            (surface.rect.x0, surface.rect.y0),
            (surface.rect.x1, surface.rect.y1),
        ],
        ShapeStyle::from(&PANEL.mix(0.82)).filled(),
    ))
    .map_err(plot_err)?;
    area.draw(&Rectangle::new(
        [
            (surface.rect.x0 + 1, surface.rect.y0 + 1),
            (surface.rect.x1 - 1, surface.rect.y1 - 1),
        ],
        ShapeStyle::from(&surface.accent.mix(0.18)).stroke_width(1),
    ))
    .map_err(plot_err)?;
    area.draw(&PathElement::new(
        vec![
            (surface.rect.x0, surface.rect.y0),
            (surface.rect.x1, surface.rect.y0),
        ],
        ShapeStyle::from(&surface.accent.mix(0.82)).stroke_width(3),
    ))
    .map_err(plot_err)?;
    area.draw(&PathElement::new(
        vec![anchor, point],
        ShapeStyle::from(&surface.accent.mix(0.30)).stroke_width(2),
    ))
    .map_err(plot_err)?;
    area.draw(&Circle::new(
        point,
        4,
        ShapeStyle::from(&RGBColor(248, 250, 252)).filled(),
    ))
    .map_err(plot_err)?;
    area.draw(&Text::new(
        surface.title,
        (surface.rect.x0 + 14, surface.rect.y0 + 24),
        ("sans-serif", 20).into_font().color(&TEXT),
    ))
    .map_err(plot_err)?;
    let summary = wrap_text(
        &surface.summary,
        available_chars(surface.rect.x1 - surface.rect.x0 - 28, 15),
    );
    let clipped = summary.into_iter().take(2).collect::<Vec<_>>();
    draw_wrapped_lines(
        area,
        surface.rect.x0 + 14,
        surface.rect.y0 + 48,
        15,
        &clipped,
        &MUTED,
    )?;
    Ok(())
}

fn draw_text_panel(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    rect: RectBox,
    title: &str,
    lines: &[String],
    accent: RGBColor,
) -> Result<()> {
    draw_card_background(area, rect, accent)?;
    area.draw(&Text::new(
        title,
        (rect.x0 + 24, rect.y0 + 40),
        ("sans-serif", 30).into_font().color(&TEXT),
    ))
    .map_err(plot_err)?;
    let mut cursor_y = rect.y0 + 84;
    for line in lines {
        let wrapped = wrap_text(line, available_chars(rect.x1 - rect.x0 - 48, 24));
        draw_wrapped_lines(area, rect.x0 + 24, cursor_y, 24, &wrapped, &MUTED)?;
        cursor_y += wrapped.len() as i32 * 32 + 12;
    }
    Ok(())
}

fn draw_list_panel(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    rect: RectBox,
    title: &str,
    rows: &[String],
    accent: RGBColor,
) -> Result<()> {
    draw_card_background(area, rect, accent)?;
    area.draw(&Text::new(
        title,
        (rect.x0 + 24, rect.y0 + 40),
        ("sans-serif", 30).into_font().color(&TEXT),
    ))
    .map_err(plot_err)?;
    let mut cursor_y = rect.y0 + 92;
    for row in rows {
        let wrapped = wrap_text(row, available_chars(rect.x1 - rect.x0 - 88, 23));
        area.draw(&Circle::new(
            (rect.x0 + 30, cursor_y - 8),
            5,
            ShapeStyle::from(&accent).filled(),
        ))
        .map_err(plot_err)?;
        draw_wrapped_lines(area, rect.x0 + 48, cursor_y - 18, 23, &wrapped, &MUTED)?;
        cursor_y += wrapped.len() as i32 * 31 + 20;
    }
    Ok(())
}

fn build_graph_nodes(families: &[FamilySummary]) -> Vec<GraphNode> {
    let mut nodes = Vec::new();
    for family in families {
        let (cx, cy, phase) = family_anchor(family.kind);
        let count = family.crates.len().max(1) as f64;
        for (idx, krate) in family.crates.iter().enumerate() {
            let orbit = idx as f64 / count;
            let angle = phase + orbit * TAU + stable_unit(&krate.name) * 0.42;
            let radius = 0.08
                + 0.22 * orbit.sqrt()
                + krate.bin_targets.min(4) as f64 * 0.008
                + total_degree(krate) as f64 * 0.0025;
            let jitter_x = stable_unit(&format!("{}:x", krate.name)) * 0.016;
            let jitter_y = stable_unit(&format!("{}:y", krate.name)) * 0.020;
            let x = (cx + radius * angle.cos() + jitter_x).clamp(-0.92, 0.92);
            let y =
                (cy + radius * angle.sin() * 0.82 + jitter_y - orbit * 0.015).clamp(-0.92, 0.92);
            let weight = 1.0 + total_degree(krate) as f64 * 0.18 + krate.bin_targets as f64 * 0.24;
            nodes.push(GraphNode {
                family: family.kind,
                name: krate.name.clone(),
                x,
                y,
                weight,
                outbound: krate.internal_deps.len(),
                inbound: krate.inbound_count,
                bin_targets: krate.bin_targets,
            });
        }
    }
    nodes
}

fn fit_graph_nodes(
    nodes: &[GraphNode],
    target_center: (f64, f64),
    target_half_span: (f64, f64),
) -> Vec<GraphNode> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let (min_x, max_x) = nodes
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), node| {
            (lo.min(node.x), hi.max(node.x))
        });
    let (min_y, max_y) = nodes
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), node| {
            (lo.min(node.y), hi.max(node.y))
        });
    let source_center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
    let source_span_x = (max_x - min_x).max(1e-6);
    let source_span_y = (max_y - min_y).max(1e-6);
    let scale_x = (target_half_span.0 * 2.0) / source_span_x;
    let scale_y = (target_half_span.1 * 2.0) / source_span_y;
    let scale = scale_x.min(scale_y);

    nodes
        .iter()
        .cloned()
        .map(|mut node| {
            node.x = (target_center.0 + (node.x - source_center.0) * scale).clamp(-0.94, 0.94);
            node.y = (target_center.1 + (node.y - source_center.1) * scale).clamp(-0.92, 0.92);
            node
        })
        .collect()
}

fn family_anchor(kind: FamilyKind) -> (f64, f64, f64) {
    match kind {
        FamilyKind::Algebra => (-0.28, -0.20, 2.8),
        FamilyKind::Physics => (0.34, -0.10, 0.15),
        FamilyKind::Data => (-0.08, 0.28, 1.6),
        FamilyKind::Interface => (0.36, 0.28, -1.2),
    }
}

fn stable_unit(text: &str) -> f64 {
    let mut hash = 1469598103934665603_u64;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    (hash % 10_000) as f64 / 4999.5 - 1.0
}

fn project_norm(rect: RectBox, x: f64, y: f64) -> (i32, i32) {
    let px = rect.x0 as f64 + (x + 1.0) * 0.5 * (rect.x1 - rect.x0) as f64;
    let py = rect.y0 as f64 + (y + 1.0) * 0.5 * (rect.y1 - rect.y0) as f64;
    (px.round() as i32, py.round() as i32)
}

fn unproject_norm(rect: RectBox, x: i32, y: i32) -> (f64, f64) {
    let nx = ((x - rect.x0) as f64 / (rect.x1 - rect.x0) as f64) * 2.0 - 1.0;
    let ny = ((y - rect.y0) as f64 / (rect.y1 - rect.y0) as f64) * 2.0 - 1.0;
    (nx, ny)
}

#[allow(dead_code)]
fn scale_from_center(point: (i32, i32), center: (i32, i32), scale: f64) -> (i32, i32) {
    (
        (center.0 as f64 + (point.0 - center.0) as f64 * scale).round() as i32,
        (center.1 as f64 + (point.1 - center.1) as f64 * scale).round() as i32,
    )
}

fn draw_repo_manifold(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    rect: RectBox,
    nodes: &[GraphNode],
    edges: &[RepoEdge],
    label_count: usize,
) -> Result<()> {
    let mut hubs = nodes.iter().collect::<Vec<_>>();
    hubs.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let highlighted_hubs = hubs.iter().take(label_count).copied().collect::<Vec<_>>();

    let step = 8;
    for y in (rect.y0..rect.y1).step_by(step) {
        for x in (rect.x0..rect.x1).step_by(step) {
            let (nx, ny) = unproject_norm(rect, x, y);
            let mut total = 0.0;
            let mut mix = [0.0, 0.0, 0.0];
            for node in nodes {
                let dx = nx - node.x;
                let dy = ny - node.y;
                let d2 = dx * dx + dy * dy;
                let sigma = 0.036 + node.weight * 0.006;
                let contrib = node.weight * (-d2 / (2.0 * sigma * sigma)).exp();
                total += contrib;
                let accent = family_accent(node.family);
                mix[0] += contrib * accent.0 as f64;
                mix[1] += contrib * accent.1 as f64;
                mix[2] += contrib * accent.2 as f64;
            }
            let intensity = 1.0 - (-0.14 * total).exp();
            if intensity < 0.025 {
                continue;
            }
            let inv = 1.0 / total.max(1e-6);
            let family_mix = [mix[0] * inv, mix[1] * inv, mix[2] * inv];
            let color = repo_field_color(intensity, family_mix);
            area.draw(&Rectangle::new(
                [
                    (x, y),
                    (
                        (x + step as i32).min(rect.x1),
                        (y + step as i32).min(rect.y1),
                    ),
                ],
                ShapeStyle::from(&color).filled(),
            ))
            .map_err(plot_err)?;
        }
    }

    let center_lookup = nodes
        .iter()
        .map(|node| (node.name.as_str(), project_norm(rect, node.x, node.y)))
        .collect::<BTreeMap<_, _>>();

    for family in [
        FamilyKind::Algebra,
        FamilyKind::Physics,
        FamilyKind::Data,
        FamilyKind::Interface,
    ] {
        let (cx, cy, _) = family_anchor(family);
        let center = project_norm(rect, cx, cy);
        let mut family_nodes = nodes
            .iter()
            .filter(|node| node.family == family)
            .collect::<Vec<_>>();
        family_nodes.sort_by(|a, b| {
            let aa = (a.y - cy).atan2(a.x - cx);
            let bb = (b.y - cy).atan2(b.x - cx);
            aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
        });
        if family_nodes.len() >= 3 {
            let count = family_nodes.len() as f64;
            let mean_x = family_nodes.iter().map(|node| node.x).sum::<f64>() / count;
            let mean_y = family_nodes.iter().map(|node| node.y).sum::<f64>() / count;
            let var_x = family_nodes
                .iter()
                .map(|node| (node.x - mean_x).powi(2))
                .sum::<f64>()
                / count;
            let var_y = family_nodes
                .iter()
                .map(|node| (node.y - mean_y).powi(2))
                .sum::<f64>()
                / count;
            let cov_xy = family_nodes
                .iter()
                .map(|node| (node.x - mean_x) * (node.y - mean_y))
                .sum::<f64>()
                / count;
            let theta = 0.5 * (2.0 * cov_xy).atan2(var_x - var_y + 1e-9);
            let sigma_x = var_x.sqrt().max(0.08);
            let sigma_y = var_y.sqrt().max(0.07);
            for (scale, alpha, width) in [
                (1.75, 0.14, 1),
                (2.25, 0.20, 1),
                (2.75, 0.28, 2),
                (3.25, 0.12, 1),
            ] {
                let mut contour = Vec::new();
                for step_idx in 0..72 {
                    let angle = step_idx as f64 / 72.0 * TAU;
                    let ex = scale * sigma_x * angle.cos();
                    let ey = scale * sigma_y * angle.sin();
                    let rx = ex * theta.cos() - ey * theta.sin();
                    let ry = ex * theta.sin() + ey * theta.cos();
                    contour.push(project_norm(rect, mean_x + rx, mean_y + ry));
                }
                contour.push(contour[0]);
                area.draw(&PathElement::new(
                    contour,
                    ShapeStyle::from(&family_accent(family).mix(alpha)).stroke_width(width),
                ))
                .map_err(plot_err)?;
            }
        }
        area.draw(&Text::new(
            format!("{} | {} crates", family_label(family), family_nodes.len()),
            (center.0 - 70, center.1 - 12),
            ("sans-serif", 24)
                .into_font()
                .color(&family_accent(family).mix(0.96)),
        ))
        .map_err(plot_err)?;
    }

    for edge in edges {
        let Some(&from) = center_lookup.get(edge.from.as_str()) else {
            continue;
        };
        let Some(&to) = center_lookup.get(edge.to.as_str()) else {
            continue;
        };
        let alpha = if edge.from_family == edge.to_family {
            0.16
        } else {
            0.08
        };
        let width = if edge.from_family == edge.to_family {
            2
        } else {
            1
        };
        area.draw(&PathElement::new(
            vec![from, to],
            ShapeStyle::from(&family_accent(edge.from_family).mix(alpha)).stroke_width(width),
        ))
        .map_err(plot_err)?;
    }

    for node in nodes {
        let center = project_norm(rect, node.x, node.y);
        area.draw(&Circle::new(
            center,
            2 + node.bin_targets.min(2) as i32,
            ShapeStyle::from(&family_accent(node.family).mix(0.84)).filled(),
        ))
        .map_err(plot_err)?;
    }

    for hub in highlighted_hubs {
        let center = project_norm(rect, hub.x, hub.y);
        area.draw(&Circle::new(
            center,
            18 + hub.bin_targets.min(3) as i32 * 4,
            ShapeStyle::from(&family_accent(hub.family).mix(0.08)).filled(),
        ))
        .map_err(plot_err)?;
        for radius in [10, 18, 28] {
            area.draw(&Circle::new(
                center,
                radius,
                ShapeStyle::from(&family_accent(hub.family).mix(0.18)).stroke_width(1),
            ))
            .map_err(plot_err)?;
        }
        for radius in [14, 24, 38] {
            area.draw(&Circle::new(
                center,
                radius,
                ShapeStyle::from(&RGBColor(245, 248, 255).mix(0.08)).stroke_width(1),
            ))
            .map_err(plot_err)?;
        }
        area.draw(&Circle::new(
            center,
            5 + hub.bin_targets.min(3) as i32,
            ShapeStyle::from(&RGBColor(252, 248, 255)).filled(),
        ))
        .map_err(plot_err)?;
        let label_dx = if center.0 < (rect.x0 + rect.x1) / 2 {
            20
        } else {
            -240
        };
        let label_dy = (stable_unit(&hub.name) * 24.0).round() as i32;
        let label_anchor = (center.0 + label_dx, center.1 + label_dy);
        area.draw(&PathElement::new(
            vec![center, label_anchor],
            ShapeStyle::from(&family_accent(hub.family).mix(0.28)).stroke_width(1),
        ))
        .map_err(plot_err)?;
        area.draw(&Text::new(
            format!("{} [{} out / {} in]", hub.name, hub.outbound, hub.inbound),
            label_anchor,
            ("sans-serif", 20).into_font().color(&TEXT),
        ))
        .map_err(plot_err)?;
    }

    Ok(())
}

fn repo_field_color(intensity: f64, mix: [f64; 3]) -> RGBColor {
    let bg = [
        BACKGROUND.0 as f64,
        BACKGROUND.1 as f64,
        BACKGROUND.2 as f64,
    ];
    let body = intensity.clamp(0.0, 1.0);
    let cyan_lift = body.powf(0.9);
    let white = body.powf(2.3) * 12.0;
    let mut r = bg[0] + 10.0 * body + mix[0] * 0.24 * body + 16.0 * cyan_lift;
    let mut g = bg[1] + 18.0 * body + mix[1] * 0.42 * body + 26.0 * cyan_lift;
    let mut b = bg[2] + 84.0 * body + mix[2] * 0.22 * body + 118.0 * body;
    r += white;
    g += white;
    b += white;
    RGBColor(
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

fn draw_badge(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    x: i32,
    y: i32,
    text: &str,
    fill: RGBAColor,
) -> Result<()> {
    let width = text.len() as i32 * 11 + 34;
    area.draw(&Rectangle::new(
        [(x, y), (x + width, y + 34)],
        ShapeStyle::from(&fill.mix(0.22)).filled(),
    ))
    .map_err(plot_err)?;
    area.draw(&PathElement::new(
        vec![(x, y + 34), (x + width, y + 34)],
        ShapeStyle::from(&fill).stroke_width(2),
    ))
    .map_err(plot_err)?;
    area.draw(&Text::new(
        text,
        (x + 14, y + 22),
        ("sans-serif", 17).into_font().color(&TEXT),
    ))
    .map_err(plot_err)?;
    Ok(())
}

fn draw_operator_heatmap(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    rect: RectBox,
    rows: &[OperatorRow],
) -> Result<()> {
    draw_card_background(area, rect, CYAN)?;
    let label_w = 420;
    let heat_w = 980;
    let command_w = 700;
    let output_w = rect.x1 - rect.x0 - label_w - heat_w - command_w;
    let heat_cols = [
        "Cargo",
        "Crates",
        "Registry",
        "README",
        "Book",
        "Artifacts",
        "Mirrors",
    ];
    let header_h = 96;
    let cell_w = heat_w / heat_cols.len() as i32;

    area.draw(&Text::new(
        "Task-surface coupling matrix",
        (rect.x0 + 22, rect.y0 + 42),
        ("sans-serif", 30).into_font().color(&TEXT),
    ))
    .map_err(plot_err)?;

    let top = rect.y0 + header_h;
    let row_h = ((rect.y1 - top) / rows.len() as i32).max(180);

    let headers = [
        ("Operator task", rect.x0, rect.x0 + label_w),
        (
            "Source coupling",
            rect.x0 + label_w,
            rect.x0 + label_w + heat_w,
        ),
        (
            "First command",
            rect.x0 + label_w + heat_w,
            rect.x0 + label_w + heat_w + command_w,
        ),
        (
            "Outputs / why",
            rect.x0 + label_w + heat_w + command_w,
            rect.x1,
        ),
    ];
    for (title, x0, x1) in headers {
        area.draw(&Rectangle::new(
            [(x0, rect.y0), (x1, rect.y0 + header_h)],
            ShapeStyle::from(&PANEL_ALT).filled(),
        ))
        .map_err(plot_err)?;
        area.draw(&Text::new(
            title,
            (x0 + 18, rect.y0 + 58),
            ("sans-serif", 24).into_font().color(&TEXT),
        ))
        .map_err(plot_err)?;
    }

    for (idx, column) in heat_cols.iter().enumerate() {
        let x = rect.x0 + label_w + idx as i32 * cell_w;
        area.draw(&Text::new(
            *column,
            (x + 10, rect.y0 + 88),
            ("sans-serif", 18).into_font().color(&MUTED),
        ))
        .map_err(plot_err)?;
    }

    for divider in [
        rect.x0 + label_w,
        rect.x0 + label_w + heat_w,
        rect.x0 + label_w + heat_w + command_w,
    ] {
        area.draw(&PathElement::new(
            vec![(divider, rect.y0), (divider, rect.y1)],
            ShapeStyle::from(&GRID.mix(0.8)).stroke_width(2),
        ))
        .map_err(plot_err)?;
    }
    area.draw(&PathElement::new(
        vec![(rect.x0, rect.y0 + header_h), (rect.x1, rect.y0 + header_h)],
        ShapeStyle::from(&GRID.mix(0.8)).stroke_width(2),
    ))
    .map_err(plot_err)?;

    for row_idx in 0..=rows.len() {
        let y = top + row_idx as i32 * row_h;
        area.draw(&PathElement::new(
            vec![(rect.x0, y), (rect.x1, y)],
            ShapeStyle::from(&GRID.mix(0.55)).stroke_width(1),
        ))
        .map_err(plot_err)?;
    }
    for col_idx in 0..=heat_cols.len() {
        let x = rect.x0 + label_w + col_idx as i32 * cell_w;
        area.draw(&PathElement::new(
            vec![(x, rect.y0), (x, rect.y1)],
            ShapeStyle::from(&GRID.mix(0.45)).stroke_width(1),
        ))
        .map_err(plot_err)?;
    }

    for (idx, row) in rows.iter().enumerate() {
        let y0 = top + idx as i32 * row_h;
        let y1 = (y0 + row_h).min(rect.y1);
        if idx % 2 == 0 {
            area.draw(&Rectangle::new(
                [(rect.x0, y0), (rect.x1, y1)],
                ShapeStyle::from(&PANEL.mix(0.93)).filled(),
            ))
            .map_err(plot_err)?;
        }

        let label_lines = wrap_text(row.surface, available_chars(label_w - 36, 22));
        draw_wrapped_lines(area, rect.x0 + 18, y0 + 18, 22, &label_lines, &TEXT)?;

        for (col_idx, touched) in row.touches.iter().enumerate() {
            if !touched {
                continue;
            }
            let cx0 = rect.x0 + label_w + col_idx as i32 * cell_w + 6;
            let cx1 = cx0 + cell_w - 12;
            area.draw(&Rectangle::new(
                [(cx0, y0 + 12), (cx1, y1 - 12)],
                ShapeStyle::from(&row.accent.mix(0.32)).filled(),
            ))
            .map_err(plot_err)?;
            area.draw(&Rectangle::new(
                [(cx0, y0 + 12), (cx1, y1 - 12)],
                ShapeStyle::from(&row.accent.mix(0.65)).stroke_width(1),
            ))
            .map_err(plot_err)?;
        }

        let command_lines = wrap_text(row.command, available_chars(command_w - 36, 21));
        draw_wrapped_lines(
            area,
            rect.x0 + label_w + heat_w + 18,
            y0 + 18,
            21,
            &command_lines,
            &TEXT,
        )?;

        let output_lines = wrap_text(row.outputs, available_chars(output_w - 36, 20));
        draw_wrapped_lines(
            area,
            rect.x0 + label_w + heat_w + command_w + 18,
            y0 + 18,
            20,
            &output_lines,
            &MUTED,
        )?;
    }

    Ok(())
}

fn draw_card_background(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    rect: RectBox,
    accent: RGBColor,
) -> Result<()> {
    area.draw(&Rectangle::new(
        [(rect.x0, rect.y0), (rect.x1, rect.y1)],
        ShapeStyle::from(&PANEL).filled(),
    ))
    .map_err(plot_err)?;
    area.draw(&Rectangle::new(
        [(rect.x0 + 1, rect.y0 + 1), (rect.x1 - 1, rect.y1 - 1)],
        ShapeStyle::from(&GRID.mix(0.32)).stroke_width(1),
    ))
    .map_err(plot_err)?;
    area.draw(&PathElement::new(
        vec![(rect.x0, rect.y0), (rect.x1, rect.y0)],
        ShapeStyle::from(&accent.mix(0.85)).stroke_width(5),
    ))
    .map_err(plot_err)?;
    Ok(())
}

fn draw_wrapped_lines(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    x: i32,
    y: i32,
    font_size: i32,
    lines: &[String],
    color: &RGBColor,
) -> Result<()> {
    for (idx, line) in lines.iter().enumerate() {
        area.draw(&Text::new(
            line.as_str(),
            (x, y + idx as i32 * (font_size + 9)),
            ("sans-serif", font_size).into_font().color(color),
        ))
        .map_err(plot_err)?;
    }
    Ok(())
}

fn wrap_text(text: &str, max_chars: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return vec![String::new()];
    }
    let limit = max_chars.max(12);
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate_len = if current.is_empty() {
            word.len()
        } else {
            current.len() + 1 + word.len()
        };
        if candidate_len > limit && !current.is_empty() {
            lines.push(current);
            current = word.to_string();
        } else if current.is_empty() {
            current = word.to_string();
        } else {
            current.push(' ');
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn available_chars(width: i32, font_size: i32) -> usize {
    ((width.max(120) as f64) / (font_size.max(10) as f64 * 0.62)).floor() as usize
}

/// 2-D grid layout parameters for `grid_positions`. Six dimensional fields
/// (origin + cell-size + gap) plus two shape fields (cols + rows); bundling
/// keeps the helper under clippy::too_many_arguments and makes the caller
/// site self-documenting at the call site.
struct GridLayout {
    left: i32,
    top: i32,
    gap_x: i32,
    gap_y: i32,
    cols: usize,
    rows: usize,
    cell_w: i32,
    cell_h: i32,
}

fn grid_positions(layout: GridLayout) -> Vec<(i32, i32, i32, i32)> {
    let GridLayout {
        left,
        top,
        gap_x,
        gap_y,
        cols,
        rows,
        cell_w,
        cell_h,
    } = layout;
    let mut out = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let x0 = left + col as i32 * (cell_w + gap_x);
            let y0 = top + row as i32 * (cell_h + gap_y);
            out.push((x0, y0, x0 + cell_w, y0 + cell_h));
        }
    }
    out
}

fn plot_err<E: std::fmt::Display>(err: E) -> anyhow::Error {
    anyhow::anyhow!(err.to_string())
}
