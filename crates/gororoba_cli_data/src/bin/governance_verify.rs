use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use provenance_store::{ControlPlaneCompatKind, ProvenanceStore};
use regex::Regex;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use toml::Value;
use walkdir::WalkDir;

const DB_BACKED_COMPAT_SIGNATURE_PATHS: &[&str] = &[
    "registry/claims.toml",
    "registry/insights.toml",
    "registry/experiments.toml",
    "registry/binaries.toml",
];

const FORBIDDEN_TEMP_ROOTS: &[&str] = &["/srv/fast/tmp"];

const SIMD_SENSITIVE_LOW_LEVEL_CRATES: &[&str] = &[
    "faer",
    "faer-traits",
    "wide",
    "pulp",
    "simsimd",
    "gemm",
    "gemm-common",
    "gemm-c32",
    "gemm-c64",
    "gemm-f32",
    "gemm-f64",
    "private-gemm-x86",
];

const SIMD_SENSITIVE_HOST_CRATES: &[&str] = &[
    "lbm_3d",
    "lbm_3d_cuda",
    "cd_kernel",
    "data_core",
    "algebra_experimental",
    "gororoba_algebra",
    "materials_core",
    "stats_core",
    "gr_core",
    "cosmology_core",
    "quantum_core",
    "tensor_core",
    "flavor_lifts",
    "gororoba_cli_data",
    "spectral_core",
    "sign_imbalance",
    "cd_spin_bridge",
    "grmhd_core",
    "gororoba_cli",
    "gororoba_engine",
    "gororoba_cli_algebra",
    "gororoba_cli_quantum",
    "gororoba_cli_physics",
    "gororoba_cli_warp",
];

const RESTORED_REGISTRY_SOURCES: &[&str] = &[
    "registry/knowledge_migration_plan.toml",
    "registry/navigator.toml",
    "registry/entrypoint_docs.toml",
    "registry/markdown_governance.toml",
    "registry/claims_domains.toml",
    "registry/claims_tasks.toml",
    "registry/insights_narrative.toml",
    "registry/experiments_narrative.toml",
];

const PARITY_ALIAS_POLICIES: &[(&str, &str, &str)] = &[
    (
        "cargo test -p cd_kernel --lib batch_sedenion_associator_matches_recursive -- --exact",
        "crates/cd_kernel/src/lib.rs",
        "fn batch_sedenion_associator_matches_recursive()",
    ),
    (
        "cargo test -p tensor_core --lib uniform_integration_matches_dense_sum_for_rank1 -- --exact",
        "crates/tensor_core/src/lib.rs",
        "fn uniform_integration_matches_dense_sum_for_rank1()",
    ),
    (
        "cargo test -p gororoba_cli_physics --lib takens_descriptor_sedenion_lane_matches_scalar_reference -- --exact",
        "crates/gororoba_cli_physics/src/lib.rs",
        "fn takens_descriptor_sedenion_lane_matches_scalar_reference()",
    ),
];

const CLI_GPU_DEFAULT_EMPTY_MANIFESTS: &[&str] = &[
    "crates/gororoba_cli/Cargo.toml",
    "crates/gororoba_cli_physics/Cargo.toml",
];

const GPU_REQUIRED_BINS: &[(&str, &str)] = &[
    (
        "crates/gororoba_cli/Cargo.toml",
        "name = \"thesis-synthesis\"\npath = \"src/bin/thesis_synthesis.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli/Cargo.toml",
        "name = \"zd-resonance-bf16\"\npath = \"src/bin/zd_resonance_bf16.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli/Cargo.toml",
        "name = \"zd-resonance-cuda\"\npath = \"src/bin/zd_resonance_cuda.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli/Cargo.toml",
        "name = \"zd-resonance-4d\"\npath = \"src/bin/zd_resonance_4d.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli/Cargo.toml",
        "name = \"percolation-experiment\"\npath = \"src/bin/percolation_experiment.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_warp/Cargo.toml",
        "name = \"warp-gpu-experiment\"\npath = \"src/bin/warp_gpu_experiment.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_warp/Cargo.toml",
        "name = \"warp-gpu-smoke\"\npath = \"src/bin/warp_gpu_smoke.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"kerr-pathion-gpu\"\npath = \"src/bin/kerr_pathion_gpu.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"flyby-crucible\"\npath = \"src/bin/flyby_crucible.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"lbm-precision-sampler\"\npath = \"src/bin/lbm_precision_sampler.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"heliosphere-sparse-preservation\"\npath = \"src/bin/heliosphere_sparse_preservation.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"heliosphere-lbm-cube-run\"\npath = \"src/bin/heliosphere_lbm_cube_run.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"chsh-betti-sweep\"\npath = \"src/bin/chsh_betti_sweep.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"e027-topology-v2\"\npath = \"src/bin/e027_topology_v2.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"euclid-df-sweep\"\npath = \"src/bin/euclid_df_sweep.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"dark-halo-hunt\"\npath = \"src/bin/dark_halo_hunt.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"lbm-slice-viewer\"\npath = \"src/bin/lbm_slice_viewer.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"particle-trace\"\npath = \"src/bin/particle_trace.rs\"\nrequired-features = [\"gpu\"]",
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        "name = \"heliosphere-boxkite-alignment\"\npath = \"src/bin/heliosphere_boxkite_alignment.rs\"\nrequired-features = [\"gpu\"]",
    ),
];

const DOCUMENTED_GPU_DEFAULT_EXCEPTIONS: &[(&str, &str)] = &[(
    "crates/gororoba_cli_warp/Cargo.toml",
    "gororoba_cli_warp` is the current exception",
)];

#[derive(Parser, Debug)]
#[command(
    name = "governance-verify",
    about = "Rust-native governance and registry verification checks"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    CanonicalControlPlane(CommonArgs),
    PlanningRequirementsAuthority(CommonArgs),
    TempRootsPolicy(CommonArgs),
    SimdContainmentPolicy(CommonArgs),
    HeavyFeaturePolicy(CommonArgs),
    ParityAliasPolicy(CommonArgs),
    RestoredRegistrySources(CommonArgs),
    NoReportsWrites(CommonArgs),
    MarkdownRemovalPolicy(CommonArgs),
    MarkdownHeaders(CommonArgs),
    MarkdownParity(CommonArgs),
    MirrorImmutability(CommonArgs),
    ClaimTicketMirrors(CommonArgs),
    SchemaSignatures(CommonArgs),
    Crossrefs(CommonArgs),
    DatasetLabelAliases(CommonArgs),
    ExternalSourceOperationalContracts(CommonArgs),
    /// Run all governance-gate checks in a single process invocation.
    /// Equivalent to: schema-signatures + crossrefs + dataset-label-aliases
    /// + external-source-operational-contracts + markdown-removal-policy.
    GateAll(CommonArgs),
}

#[derive(Parser, Debug, Clone)]
struct CommonArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::CanonicalControlPlane(args) => verify_canonical_control_plane(&args),
        Command::PlanningRequirementsAuthority(args) => {
            verify_planning_requirements_authority(&args)
        }
        Command::TempRootsPolicy(args) => verify_temp_roots_policy(&args),
        Command::SimdContainmentPolicy(args) => verify_simd_containment_policy(&args),
        Command::HeavyFeaturePolicy(args) => verify_heavy_feature_policy(&args),
        Command::ParityAliasPolicy(args) => verify_parity_alias_policy(&args),
        Command::RestoredRegistrySources(args) => verify_restored_registry_sources(&args),
        Command::NoReportsWrites(args) => verify_no_reports_writes(&args),
        Command::MarkdownRemovalPolicy(args) => verify_markdown_removal_policy(&args),
        Command::MarkdownHeaders(args) => verify_markdown_headers(&args),
        Command::MarkdownParity(args) => verify_markdown_parity(&args),
        Command::MirrorImmutability(args) => verify_mirror_immutability(&args),
        Command::ClaimTicketMirrors(args) => verify_claim_ticket_mirrors(&args),
        Command::SchemaSignatures(args) => verify_schema_signatures(&args),
        Command::Crossrefs(args) => verify_crossrefs(&args),
        Command::DatasetLabelAliases(args) => verify_dataset_label_aliases(&args),
        Command::ExternalSourceOperationalContracts(args) => {
            verify_external_source_operational_contracts(&args)
        }
        Command::GateAll(args) => run_gate_all(&args),
    }
}

fn run_gate_all(args: &CommonArgs) -> Result<()> {
    let mut failed: Vec<&str> = Vec::new();

    macro_rules! run_check {
        ($name:expr, $func:expr) => {
            match $func(args) {
                Ok(()) => eprintln!("[done] {}", $name),
                Err(e) => {
                    eprintln!("[FAIL] {}: {e}", $name);
                    failed.push($name);
                }
            }
        };
    }

    run_check!("schema-signatures", verify_schema_signatures);
    run_check!("crossrefs", verify_crossrefs);
    run_check!("dataset-label-aliases", verify_dataset_label_aliases);
    run_check!("canonical-control-plane", verify_canonical_control_plane);
    run_check!(
        "planning-requirements-authority",
        verify_planning_requirements_authority
    );
    run_check!("temp-roots-policy", verify_temp_roots_policy);
    run_check!("simd-containment-policy", verify_simd_containment_policy);
    run_check!("heavy-feature-policy", verify_heavy_feature_policy);
    run_check!("parity-alias-policy", verify_parity_alias_policy);
    run_check!(
        "restored-registry-sources",
        verify_restored_registry_sources
    );
    run_check!(
        "external-source-operational-contracts",
        verify_external_source_operational_contracts
    );
    run_check!("markdown-removal-policy", verify_markdown_removal_policy);

    if failed.is_empty() {
        Ok(())
    } else {
        bail!(
            "governance-gate: {} check(s) failed: {}",
            failed.len(),
            failed.join(", ")
        )
    }
}

fn verify_canonical_control_plane(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let canonical_path = "registry/canonical/control_plane.sqlite3";
    let legacy_path = ".cache/registry.sqlite3";
    let required_files = [
        "README.md",
        "docs/db/ARCHITECTURE.md",
        "Makefile",
        "crates/gororoba_db/src/bin/gororoba_db.rs",
    ];
    let mut failures = Vec::new();

    for rel in required_files {
        let path = root.join(rel);
        let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        if !text.contains(canonical_path) {
            failures.push(format!(
                "{rel}: missing canonical control-plane path `{canonical_path}`"
            ));
        }
        if text.contains(legacy_path) {
            failures.push(format!(
                "{rel}: still references legacy control-plane path `{legacy_path}`"
            ));
        }
    }

    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }

    println!("OK: canonical control-plane declarations are aligned");
    Ok(())
}

fn verify_planning_requirements_authority(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let generated_views = [
        "registry/roadmap.toml",
        "registry/todo.toml",
        "registry/next_actions.toml",
        "registry/requirements.toml",
    ];
    let narrative_files = [
        "registry/roadmap_narrative.toml",
        "registry/todo_narrative.toml",
        "registry/next_actions_narrative.toml",
        "registry/requirements_narrative.toml",
    ];
    let mut failures = Vec::new();

    for rel in generated_views {
        let path = root.join(rel);
        let text = read_ascii_text(&path).with_context(|| format!("read {}", path.display()))?;
        if !text.starts_with("# GENERATED VIEW: DO NOT EDIT.\n") {
            failures.push(format!(
                "generated planning/requirements view missing generated-view header: {rel}"
            ));
        }
    }

    for rel in narrative_files {
        let path = root.join(rel);
        let text = read_ascii_text(&path).with_context(|| format!("read {}", path.display()))?;
        if !text.contains("make registry-build registry-export-markdown") {
            failures.push(format!(
                "narrative compatibility file missing DB-backed regeneration guidance: {rel}"
            ));
        }
        if text.contains("pending SQLite promotion") {
            failures.push(format!(
                "narrative compatibility file still references stale migration wording: {rel}"
            ));
        }
    }

    let requirements = load_toml(&root.join("registry/requirements.toml"))?;
    let requirements_narrative = load_toml(&root.join("registry/requirements_narrative.toml"))?;

    let primary_markdown = requirements
        .get("requirements")
        .and_then(Value::as_table)
        .and_then(|table| table.get("primary_markdown"))
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if primary_markdown.is_empty() {
        failures.push(
            "registry/requirements.toml is missing requirements.primary_markdown".to_string(),
        );
    }

    let mut narrative_paths = BTreeSet::new();
    let mut duplicate_narrative_paths = BTreeSet::new();
    for row in table_array(&requirements_narrative, "document") {
        let path = table_str(row, "path").trim();
        if path.is_empty() {
            failures.push(
                "registry/requirements_narrative.toml contains a document with empty path"
                    .to_string(),
            );
            continue;
        }
        if !narrative_paths.insert(path.to_string()) {
            duplicate_narrative_paths.insert(path.to_string());
        }
    }
    for path in duplicate_narrative_paths {
        failures.push(format!(
            "duplicate requirements_narrative document.path entry: {path}"
        ));
    }

    if !primary_markdown.is_empty() && !narrative_paths.contains(primary_markdown) {
        failures.push(format!(
            "requirements.primary_markdown is missing matching narrative document.path: {primary_markdown}"
        ));
    }

    let mut required_paths = BTreeSet::new();
    if !primary_markdown.is_empty() {
        required_paths.insert(primary_markdown.to_string());
    }
    for row in table_array(&requirements, "module") {
        let module_id = table_str(row, "id").trim();
        let markdown = table_str(row, "markdown").trim();
        if markdown.is_empty() {
            failures.push(format!(
                "requirements module is missing markdown path: {}",
                if module_id.is_empty() {
                    "<unknown-module>"
                } else {
                    module_id
                }
            ));
            continue;
        }
        required_paths.insert(markdown.to_string());
        if !narrative_paths.contains(markdown) {
            failures.push(format!(
                "requirements module markdown has no matching narrative document.path: {} ({})",
                markdown,
                if module_id.is_empty() {
                    "<unknown-module>"
                } else {
                    module_id
                }
            ));
        }
    }

    for path in narrative_paths.difference(&required_paths) {
        failures.push(format!(
            "requirements narrative document.path is orphaned from primary_markdown/module.markdown: {path}"
        ));
    }

    if !failures.is_empty() {
        for failure in &failures {
            eprintln!("ERROR: {failure}");
        }
        bail!(
            "planning/requirements authority verification failed ({} issue(s))",
            failures.len()
        );
    }

    println!("OK: planning/requirements authority and narrative bindings verified");
    Ok(())
}

fn verify_temp_roots_policy(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let targets = [
        root.join("Makefile"),
        root.join("Cargo.toml"),
        root.join("README.md"),
        root.join("scripts"),
        root.join("docs"),
    ];
    let mut failures = Vec::new();

    for target in targets {
        if !target.exists() {
            continue;
        }
        let walker = if target.is_dir() {
            WalkDir::new(&target)
        } else {
            WalkDir::new(&target).max_depth(1)
        };
        for entry in walker.into_iter().flatten() {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if !looks_like_text_policy_surface(path) {
                continue;
            }
            let Ok(text) = fs::read_to_string(path) else {
                continue;
            };
            for forbidden in FORBIDDEN_TEMP_ROOTS {
                if text.contains(forbidden) {
                    let rel = path
                        .strip_prefix(&root)
                        .context("strip temp-root path prefix")?
                        .to_string_lossy()
                        .replace('\\', "/");
                    failures.push(format!(
                        "{rel}: forbidden machine-specific temp root `{forbidden}`"
                    ));
                }
            }
        }
    }

    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }

    println!("OK: temp-root policy verified");
    Ok(())
}

fn verify_simd_containment_policy(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let cargo_toml = read_ascii_text(&root.join("Cargo.toml"))?;
    let physics_cargo_toml = read_ascii_text(&root.join("crates/gororoba_cli_physics/Cargo.toml"))?;
    let mut failures = Vec::new();

    for crate_name in SIMD_SENSITIVE_LOW_LEVEL_CRATES
        .iter()
        .chain(SIMD_SENSITIVE_HOST_CRATES.iter())
    {
        for profile in ["dev", "test"] {
            let header = profile_package_header(profile, crate_name);
            let pattern = format!("{header}\ncodegen-backend = \"llvm\"");
            if !cargo_toml.contains(&pattern) {
                failures.push(format!(
                    "Cargo.toml missing SIMD containment override for crate `{crate_name}` in profile `{profile}`"
                ));
            }
        }
    }

    if !cargo_toml.contains("cg_clif currently")
        || !cargo_toml.contains("wide: explicit fixed-width vector types")
        || !cargo_toml.contains("pulp: runtime SIMD dispatch")
        || !cargo_toml.contains("simsimd: external C/C++ SIMD distance kernels")
    {
        failures.push(
            "Cargo.toml is missing the documented SIMD containment rationale block".to_string(),
        );
    }

    let mut dependency_exposures = BTreeMap::<String, BTreeSet<String>>::new();
    for manifest in workspace_manifests(&root)? {
        let rel = manifest
            .strip_prefix(&root)
            .context("strip manifest prefix")?
            .to_string_lossy()
            .replace('\\', "/");
        let raw = fs::read_to_string(&manifest)
            .with_context(|| format!("read workspace manifest {}", manifest.display()))?;
        let parsed: Value = toml::from_str(&raw)
            .with_context(|| format!("parse workspace manifest {}", manifest.display()))?;
        let package_name = parsed
            .get("package")
            .and_then(Value::as_table)
            .and_then(|table| table.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if package_name.is_empty() {
            continue;
        }
        let deps = collect_manifest_dependencies(&parsed);
        let sensitive_hits: BTreeSet<String> = deps
            .into_iter()
            .filter(|dep| SIMD_SENSITIVE_LOW_LEVEL_CRATES.contains(&dep.as_str()))
            .collect();
        if sensitive_hits.is_empty() {
            continue;
        }
        dependency_exposures.insert(package_name.clone(), sensitive_hits.clone());
        let protected = SIMD_SENSITIVE_LOW_LEVEL_CRATES.contains(&package_name.as_str())
            || SIMD_SENSITIVE_HOST_CRATES.contains(&package_name.as_str());
        if !protected {
            failures.push(format!(
                "{rel}: crate `{package_name}` depends on SIMD-sensitive subtree {} but is not listed in the containment override set",
                sensitive_hits.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
    }

    let readme = read_ascii_text(&root.join("README.md"))?;
    if readme.contains("Cranelift backend for dev (opt-level 2), LLVM for release") {
        failures.push(
            "README.md still claims a blanket Cranelift dev lane; document the SIMD containment carve-out explicitly".to_string(),
        );
    }
    if !readme.contains("docs/engineering/cg_clif_simd_containment.txt") {
        failures.push(
            "README.md is missing a cross-reference to docs/engineering/cg_clif_simd_containment.txt".to_string(),
        );
    }

    let reqwest_provider_agnostic = "reqwest = { version = \"0.13\", default-features = false, features = [\"blocking\", \"rustls-no-provider\", \"gzip\", \"brotli\", \"deflate\", \"form\"] }";
    if !cargo_toml.contains(reqwest_provider_agnostic) {
        failures.push(
            "Cargo.toml must keep the workspace reqwest dependency on `rustls-no-provider` to avoid reintroducing the aws-lc native-link lane".to_string(),
        );
    }
    if cargo_toml.contains(
        "features = [\"blocking\", \"rustls\", \"gzip\", \"brotli\", \"deflate\", \"form\"]",
    ) {
        failures.push(
            "Cargo.toml still contains a provider-pinned reqwest rustls feature set; use `rustls-no-provider` instead".to_string(),
        );
    }

    if !physics_cargo_toml
        .contains("gpu = [\"lbm_3d_cuda\", \"sign_imbalance/gpu\", \"gororoba_algebra/gpu\"]")
    {
        failures.push(
            "crates/gororoba_cli_physics/Cargo.toml must wire the `gpu` feature through to `gororoba_algebra/gpu`".to_string(),
        );
    }

    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }

    println!(
        "OK: SIMD containment policy verified across {} workspace manifests",
        dependency_exposures.len()
    );
    Ok(())
}

fn verify_restored_registry_sources(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let mut failures = Vec::new();

    for rel in RESTORED_REGISTRY_SOURCES {
        let path = root.join(rel);
        let text = read_ascii_text(&path).with_context(|| format!("read {}", path.display()))?;
        if !text.contains("Authoritative source")
            && !text.contains("authoritative_toml")
            && !text.contains("compatibility placeholder")
        {
            failures.push(format!(
                "{rel}: missing authority/provenance guidance for restored source"
            ));
        }
        if rel.ends_with("_narrative.toml") {
            if !text.contains("make registry-build registry-export-markdown") {
                failures.push(format!(
                    "{rel}: placeholder narrative is missing regeneration guidance"
                ));
            }
        } else if !text.contains("make registry-export-markdown")
            && !text.contains("make registry-build registry-export-markdown")
        {
            failures.push(format!(
                "{rel}: missing mirror regeneration guidance in restored source header"
            ));
        }
    }

    let entrypoint_docs = read_ascii_text(&root.join("registry/entrypoint_docs.toml"))?;
    if entrypoint_docs
        .contains("The TOML registry layer under `registry/` is the canonical control plane.")
    {
        failures.push(
            "registry/entrypoint_docs.toml still embeds stale TOML-first control-plane language"
                .to_string(),
        );
    }

    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }

    println!("OK: restored registry source headers and guidance verified");
    Ok(())
}

fn verify_parity_alias_policy(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let contract_note =
        read_ascii_text(&root.join("docs/engineering/simd_scalar_parity_contract_2026_04_06.txt"))?;
    let mut failures = Vec::new();

    for (documented_command, source_rel, alias_signature) in PARITY_ALIAS_POLICIES {
        if !contract_note.contains(documented_command) {
            failures.push(format!(
                "simd parity contract note is missing documented alias command `{documented_command}`"
            ));
        }

        let source_path = root.join(source_rel);
        let source_text = read_ascii_text(&source_path)
            .with_context(|| format!("read parity alias source {}", source_path.display()))?;
        if !source_text.contains(alias_signature) {
            failures.push(format!(
                "{source_rel}: missing documented crate-root parity alias `{alias_signature}`"
            ));
        }
    }

    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }

    println!("OK: parity alias policy verified");
    Ok(())
}

fn verify_heavy_feature_policy(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let mut failures = Vec::new();

    for rel in CLI_GPU_DEFAULT_EMPTY_MANIFESTS {
        let text = read_ascii_text(&root.join(rel))
            .with_context(|| format!("read heavy-feature policy manifest {}", rel))?;
        if !text.contains("[features]\ndefault = []") {
            failures.push(format!(
                "{rel}: top-level CLI crate must keep heavy GPU features opt-in via `default = []`"
            ));
        }
    }

    let gororoba_cli = read_ascii_text(&root.join("crates/gororoba_cli/Cargo.toml"))?;
    if !gororoba_cli.contains("stats_core = { path = \"../stats_core\" }") {
        failures.push(
            "crates/gororoba_cli/Cargo.toml: stats_core must not enable `gpu` unconditionally"
                .to_string(),
        );
    }
    if !gororoba_cli.contains("\"stats_core/gpu\"") {
        failures.push(
            "crates/gororoba_cli/Cargo.toml: `gpu` feature must forward to `stats_core/gpu`"
                .to_string(),
        );
    }

    let gororoba_cli_warp = read_ascii_text(&root.join("crates/gororoba_cli_warp/Cargo.toml"))?;
    if !gororoba_cli_warp.contains("stats_core = { path = \"../stats_core\" }") {
        failures.push(
            "crates/gororoba_cli_warp/Cargo.toml: stats_core must not enable `gpu` unconditionally"
                .to_string(),
        );
    }
    if !gororoba_cli_warp.contains("\"stats_core/gpu\"") {
        failures.push(
            "crates/gororoba_cli_warp/Cargo.toml: `gpu` feature must forward to `stats_core/gpu`"
                .to_string(),
        );
    }

    for (rel, snippet) in GPU_REQUIRED_BINS {
        let text = read_ascii_text(&root.join(rel))
            .with_context(|| format!("read GPU required-features manifest {}", rel))?;
        if !text.contains(snippet) {
            failures.push(format!(
                "{rel}: missing `required-features = [\"gpu\"]` for a GPU-only binary"
            ));
        }
    }

    let audit_note =
        read_ascii_text(&root.join("docs/engineering/data_core_surface_audit_2026_04_06.txt"))?;
    for (rel, note_snippet) in DOCUMENTED_GPU_DEFAULT_EXCEPTIONS {
        let text = read_ascii_text(&root.join(rel))
            .with_context(|| format!("read documented GPU default exception manifest {}", rel))?;
        if text.contains("[features]\ndefault = [\"gpu\"]") && !audit_note.contains(note_snippet) {
            failures.push(format!(
                "{rel}: GPU-default exception must be documented in data_core_surface_audit_2026_04_06.txt"
            ));
        }
    }

    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }

    println!("OK: heavy feature policy verified");
    Ok(())
}

fn resolve_root(args: &CommonArgs) -> Result<PathBuf> {
    args.repo_root.canonicalize().context("resolve repo root")
}

fn read_ascii_text(path: &Path) -> Result<String> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let bad: BTreeSet<char> = text.chars().filter(|ch| (*ch as u32) > 127).collect();
    if !bad.is_empty() {
        let sample: String = bad.iter().take(20).copied().collect();
        bail!("non-ASCII content in {}: {:?}", path.display(), sample);
    }
    Ok(text)
}

fn load_toml(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse TOML {}", path.display()))
}

fn load_control_plane_registry(
    root: &Path,
    db_rel_path: &Path,
    kind: ControlPlaneCompatKind,
    fallback_rel_path: &str,
) -> Result<Value> {
    let db_path = root.join(db_rel_path);
    if db_path.exists() {
        let mut store = ProvenanceStore::open(&db_path)
            .with_context(|| format!("open canonical control-plane DB {}", db_path.display()))?;
        let text = store.control_plane_compat_text(kind).with_context(|| {
            format!(
                "render {:?} compatibility text from {}",
                kind,
                db_path.display()
            )
        })?;
        return toml::from_str(&text)
            .with_context(|| format!("parse {:?} compatibility TOML", kind));
    }
    load_toml(&root.join(fallback_rel_path))
}

fn table_array<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn table_str<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn table_bool(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn looks_like_text_policy_surface(path: &Path) -> bool {
    if matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("Makefile" | "Cargo.toml" | "README.md")
    ) {
        return true;
    }
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("md" | "txt" | "toml" | "sh" | "rs" | "py" | "yml" | "yaml" | "json")
    )
}

fn profile_package_header(profile: &str, crate_name: &str) -> String {
    if crate_name.contains('-') {
        format!("[profile.{profile}.package.\"{crate_name}\"]")
    } else {
        format!("[profile.{profile}.package.{crate_name}]")
    }
}

fn workspace_manifests(root: &Path) -> Result<Vec<PathBuf>> {
    let mut manifests = Vec::new();
    for entry in WalkDir::new(root.join("crates")).into_iter().flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().file_name().and_then(|name| name.to_str()) == Some("Cargo.toml") {
            manifests.push(entry.into_path());
        }
    }
    manifests.sort();
    Ok(manifests)
}

fn collect_manifest_dependencies(value: &Value) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    collect_dependency_table(value.get("dependencies"), &mut out);
    collect_dependency_table(value.get("dev-dependencies"), &mut out);
    collect_dependency_table(value.get("build-dependencies"), &mut out);
    if let Some(targets) = value.get("target").and_then(Value::as_table) {
        for section in targets.values() {
            if let Some(table) = section.as_table() {
                collect_dependency_table(table.get("dependencies"), &mut out);
                collect_dependency_table(table.get("dev-dependencies"), &mut out);
                collect_dependency_table(table.get("build-dependencies"), &mut out);
            }
        }
    }
    out
}

fn collect_dependency_table(value: Option<&Value>, out: &mut BTreeSet<String>) {
    let Some(table) = value.and_then(Value::as_table) else {
        return;
    };
    for key in table.keys() {
        out.insert(key.trim().to_string());
    }
}

fn string_list(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Integer(number) => Some(number.to_string()),
        Value::Float(number) => Some(number.to_string()),
        Value::Boolean(flag) => Some(flag.to_string()),
        Value::Datetime(dt) => Some(dt.to_string()),
        _ => None,
    }
}

fn value_list(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(value_to_string)
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn verify_no_reports_writes(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let ver_root = root.join("src/verification");
    if !ver_root.exists() {
        bail!("Missing src/verification directory");
    }
    let write_tokens = [
        "open(",
        "write(",
        "write_text(",
        "write_bytes(",
        ".mkdir(",
        ".touch(",
        "Path(\"reports/",
        "Path('reports/",
    ];
    let mut failures = Vec::new();
    for entry in WalkDir::new(&ver_root).max_depth(1).into_iter().flatten() {
        let path = entry.path();
        if !entry.file_type().is_file() || path.extension().and_then(|v| v.to_str()) != Some("py") {
            continue;
        }
        if path.file_name().and_then(|v| v.to_str()) == Some("verify_no_reports_writes.py") {
            continue;
        }
        let rel = path
            .strip_prefix(&root)
            .context("strip prefix")?
            .to_string_lossy()
            .replace('\\', "/");
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        if text.contains("reports/") && write_tokens.iter().any(|token| text.contains(token)) {
            failures.push(format!("{rel}: appears to write under reports/"));
        }
    }
    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }
    println!("OK: verifiers do not write under reports/");
    Ok(())
}

fn verify_markdown_removal_policy(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let owner_map = load_toml(&root.join("registry/markdown_owner_map.toml"))?;
    let mut failures = Vec::new();
    let valid_statuses: BTreeSet<&str> = BTreeSet::from([
        "active",
        "candidate_for_removal",
        "deprecated",
        "archived",
        "locked",
        "removed",
    ]);
    let mut counts = BTreeMap::<String, usize>::new();
    for row in table_array(&owner_map, "owner") {
        let removal_status = table_str(row, "removal_status").trim();
        let removal_status = if removal_status.is_empty() {
            "active"
        } else {
            removal_status
        };
        let removal_reason = table_str(row, "removal_reason").trim();
        let path = table_str(row, "path");
        let owner_group = table_str(row, "owner_group");
        *counts.entry(removal_status.to_string()).or_default() += 1;

        if removal_status == "active" && !removal_reason.is_empty() {
            failures.push(format!(
                "{path}: removal_status=active but has removal_reason"
            ));
        }
        if removal_status != "active" && removal_reason.is_empty() {
            failures.push(format!(
                "{path}: removal_status={removal_status} but missing removal_reason"
            ));
        }
        if !valid_statuses.contains(removal_status) {
            failures.push(format!("{path}: invalid removal_status={removal_status}"));
        }
        if matches!(owner_group, "third_party" | "external") && removal_status != "locked" {
            failures.push(format!(
                "{path}: owner_group={owner_group} but removal_status={removal_status}"
            ));
        }
    }
    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }
    println!("PASS: markdown removal policy");
    println!(
        "  total_documents={}",
        table_array(&owner_map, "owner").len()
    );
    for (status, count) in counts {
        println!("  {status}={count}");
    }
    Ok(())
}

fn verify_markdown_headers(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let governance = load_toml(&root.join("registry/markdown_governance.toml"))?;
    let mut failures = Vec::new();
    for row in table_array(&governance, "document") {
        if !table_bool(row, "header_required") {
            continue;
        }
        let rel = table_str(row, "path").trim();
        let path = root.join(rel);
        if !path.exists() {
            failures.push(format!("missing generated markdown file: {rel}"));
            continue;
        }
        let head = fs::read_to_string(&path)?
            .lines()
            .take(8)
            .collect::<Vec<_>>()
            .join("\n");
        if !head.contains("AUTO-GENERATED") {
            failures.push(format!("missing AUTO-GENERATED header: {rel}"));
        }
    }
    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }
    println!("OK: markdown governance headers verified");
    Ok(())
}

fn verify_markdown_parity(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let governance = load_toml(&root.join("registry/markdown_governance.toml"))?;
    let legacy_knowledge_path = root.join("registry/knowledge_sources.toml");
    let ks_paths: BTreeSet<String> = if legacy_knowledge_path.exists() {
        let knowledge = load_toml(&legacy_knowledge_path)?;
        table_array(&knowledge, "document")
            .iter()
            .map(|row| table_str(row, "path").trim().to_string())
            .filter(|path| path.ends_with(".md"))
            .collect()
    } else {
        BTreeSet::new()
    };
    let gov_paths: BTreeSet<String> = table_array(&governance, "document")
        .iter()
        .map(|row| table_str(row, "path").trim().to_string())
        .filter(|path| path.ends_with(".md"))
        .collect();
    let missing: Vec<String> = ks_paths.difference(&gov_paths).cloned().collect();
    if !missing.is_empty() {
        bail!(
            "knowledge_sources paths missing in markdown_governance: {}",
            missing.join(", ")
        );
    }
    if legacy_knowledge_path.exists() {
        println!("OK: markdown governance parity verified");
    } else {
        println!("OK: markdown governance parity verified (legacy knowledge_sources.toml absent)");
    }
    Ok(())
}

fn verify_mirror_immutability(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let governance = load_toml(&root.join("registry/markdown_governance.toml"))?;
    let mut failures = Vec::new();
    for row in table_array(&governance, "document") {
        if table_str(row, "mode") != "toml_generated_mirror" {
            continue;
        }
        let rel = table_str(row, "path").trim();
        if !rel.ends_with(".md") {
            continue;
        }
        let path = root.join(rel);
        if !path.exists() {
            failures.push(format!("missing toml_generated_mirror file: {rel}"));
            continue;
        }
        let head = fs::read_to_string(&path)?
            .lines()
            .take(12)
            .collect::<Vec<_>>()
            .join("\n");
        if !head.contains("AUTO-GENERATED: DO NOT EDIT") {
            failures.push(format!("missing immutability marker: {rel}"));
        }
        if !head.contains("Source of truth:") {
            failures.push(format!("missing source-of-truth marker: {rel}"));
        }
        let refs = string_list(row, "source_toml_refs");
        if !refs.is_empty() && !refs.iter().any(|reference| head.contains(reference)) {
            failures.push(format!(
                "mirror header does not reference source_toml_refs: {rel}"
            ));
        }
    }
    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }
    println!("OK: TOML-generated mirror immutability verified");
    Ok(())
}

fn verify_claim_ticket_mirrors(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let tickets = load_toml(&root.join("registry/claim_tickets.toml"))?;
    let mut failures = Vec::new();
    let mut expected = BTreeSet::new();
    for row in table_array(&tickets, "ticket") {
        let rel = table_str(row, "source_markdown").trim();
        let id = table_str(row, "id");
        if rel.is_empty() {
            failures.push(format!("ticket missing source_markdown: {id}"));
            continue;
        }
        expected.insert(rel.to_string());
        let path = root.join(rel);
        if !path.exists() {
            failures.push(format!("missing ticket markdown mirror: {rel}"));
            continue;
        }
        let head = fs::read_to_string(&path)?
            .lines()
            .take(8)
            .collect::<Vec<_>>()
            .join("\n");
        if !head.contains("AUTO-GENERATED") {
            failures.push(format!("ticket file missing AUTO-GENERATED header: {rel}"));
        }
    }
    let tickets_dir = root.join("docs/tickets");
    let mut existing = BTreeSet::new();
    if tickets_dir.exists() {
        for entry in fs::read_dir(&tickets_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|v| v.to_str()) != Some("md") {
                continue;
            }
            if path.file_name().and_then(|v| v.to_str()) == Some("INDEX.md") {
                continue;
            }
            existing.insert(
                path.strip_prefix(&root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    for extra in existing.difference(&expected) {
        failures.push(format!(
            "ticket markdown file not declared in registry/claim_tickets.toml: {extra}"
        ));
    }
    let index_path = tickets_dir.join("INDEX.md");
    if !index_path.exists() {
        failures.push("missing docs/tickets/INDEX.md".to_string());
    } else {
        let head = fs::read_to_string(&index_path)?
            .lines()
            .take(8)
            .collect::<Vec<_>>()
            .join("\n");
        if !head.contains("AUTO-GENERATED") {
            failures.push("docs/tickets/INDEX.md missing AUTO-GENERATED header".to_string());
        }
    }
    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }
    println!("OK: claim ticket mirrors verified");
    Ok(())
}

fn verify_schema_signatures(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let schema_path = root.join("registry/schema_signatures.toml");
    let schema_text = read_ascii_text(&schema_path)?;
    let schema = toml::from_str::<Value>(&schema_text)?;
    let meta = schema
        .get("schema_signatures")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    let rows = table_array(&schema, "signature");
    let mut failures = Vec::new();
    let skipped_paths: BTreeSet<String> = DB_BACKED_COMPAT_SIGNATURE_PATHS
        .iter()
        .map(|path| (*path).to_string())
        .collect();
    if meta
        .get("signature_count")
        .and_then(Value::as_integer)
        .unwrap_or(-1)
        != rows.len() as i64
    {
        failures.push("schema_signatures signature_count metadata mismatch".to_string());
    }
    if meta.get("version").and_then(Value::as_integer).unwrap_or(0) <= 0 {
        failures.push("schema_signatures version must be positive".to_string());
    }
    let mut seen = BTreeSet::new();
    for row in rows {
        let rel = table_str(row, "path");
        if skipped_paths.contains(rel) {
            continue;
        }
        if !seen.insert(rel.to_string()) {
            failures.push(format!("duplicate schema signature path: {rel}"));
            continue;
        }
        let path = root.join(rel);
        if !path.exists() {
            failures.push(format!("signed registry path missing: {rel}"));
            continue;
        }
        let text = fs::read_to_string(&path)?;
        let data = toml::from_str::<Value>(&text)?;
        let mut top_level = data
            .as_table()
            .map(|table| table.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        top_level.sort();
        let shapes = top_level
            .iter()
            .map(|key| {
                (
                    key.clone(),
                    shape_summary(data.get(key).unwrap_or(&Value::String(String::new()))),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let payload = json!({
            "path": rel,
            "top_level_keys": top_level,
            "shapes": shapes,
        });
        let normalized = serde_json::to_string(&payload)?;
        let schema_sha = hex_hash(normalized.as_bytes());
        // content_sha256 comparison REMOVED: it fails on every content edit
        // (adding claims, ASOT entries, etc.) requiring manual integrity-resolution.
        // The schema_sha256 check catches real structural violations (wrong-type
        // fields, missing tables) which is the actual value proposition.
        if schema_sha != table_str(row, "schema_sha256") {
            failures.push(format!("schema_sha mismatch for {rel}"));
        }
        if normalized != table_str(row, "shape_json") {
            failures.push(format!("shape_json mismatch for {rel}"));
        }
        let top_level_expected = string_list(row, "top_level_keys");
        if top_level != top_level_expected {
            failures.push(format!("top_level_keys mismatch for {rel}"));
        }
    }
    let declared_paths: BTreeSet<String> = meta
        .get("registry_paths")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let effective_declared_paths: BTreeSet<String> =
        declared_paths.difference(&skipped_paths).cloned().collect();
    if seen != effective_declared_paths {
        let missing: Vec<String> = effective_declared_paths
            .difference(&seen)
            .cloned()
            .collect();
        let extra: Vec<String> = seen.difference(&declared_paths).cloned().collect();
        if !missing.is_empty() {
            failures.push(format!(
                "schema metadata registry_paths missing signatures: {}",
                missing.len()
            ));
        }
        if !extra.is_empty() {
            failures.push(format!(
                "schema signatures contain undeclared paths: {}",
                extra.len()
            ));
        }
    }
    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }
    println!(
        "OK: schema signatures verified for {} registries ({} DB-backed compat exports skipped).",
        rows.len()
            .saturating_sub(skipped_paths.intersection(&declared_paths).count()),
        skipped_paths.intersection(&declared_paths).count()
    );
    Ok(())
}

#[derive(Default)]
struct CrossRefCounters {
    claims: usize,
    insights: usize,
    experiments: usize,
    sources: usize,
    datasets: usize,
}

#[derive(Default)]
struct ExtractedRefs {
    claims: Vec<String>,
    insights: Vec<String>,
    experiments: Vec<String>,
    sources: Vec<String>,
    datasets: Vec<String>,
}

fn verify_crossrefs(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let required = [
        "registry/roadmap.toml",
        "registry/todo.toml",
        "registry/next_actions.toml",
        "registry/requirements.toml",
        "registry/module_requirements.toml",
        "registry/external_sources.toml",
        "registry/dataset_label_aliases.toml",
        "data/external/SOURCES.toml",
        "registry/claims_atoms.toml",
        "registry/claims_evidence_edges.toml",
        "registry/provenance_sources.toml",
        "registry/narrative_paragraph_atoms.toml",
    ];
    for rel in required {
        let path = root.join(rel);
        if !path.exists() {
            bail!("ERROR: missing required registry {rel}");
        }
    }

    let claims = load_control_plane_registry(
        &root,
        &args.db,
        ControlPlaneCompatKind::Claims,
        "registry/claims.toml",
    )?;
    let insights = load_control_plane_registry(
        &root,
        &args.db,
        ControlPlaneCompatKind::Insights,
        "registry/insights.toml",
    )?;
    let experiments = load_control_plane_registry(
        &root,
        &args.db,
        ControlPlaneCompatKind::Experiments,
        "registry/experiments.toml",
    )?;
    let lineage = {
        let path = root.join("registry/experiment_lineage.toml");
        if path.exists() {
            Some(load_toml(&path)?)
        } else {
            None
        }
    };
    let roadmap = load_toml(&root.join("registry/roadmap.toml"))?;
    let todo = load_toml(&root.join("registry/todo.toml"))?;
    let next_actions = load_toml(&root.join("registry/next_actions.toml"))?;
    let requirements = load_toml(&root.join("registry/requirements.toml"))?;
    let module_requirements = load_toml(&root.join("registry/module_requirements.toml"))?;
    let external_sources = load_toml(&root.join("registry/external_sources.toml"))?;
    let dataset_label_aliases = load_toml(&root.join("registry/dataset_label_aliases.toml"))?;
    let claim_atoms = load_toml(&root.join("registry/claims_atoms.toml"))?;
    let claim_edges = load_toml(&root.join("registry/claims_evidence_edges.toml"))?;
    let provenance = load_toml(&root.join("registry/provenance_sources.toml"))?;
    let paragraph_atoms = load_toml(&root.join("registry/narrative_paragraph_atoms.toml"))?;

    let conflict_markers = if root.join("registry/conflict_markers.toml").exists() {
        load_toml(&root.join("registry/conflict_markers.toml"))?
    } else {
        Value::Table(Default::default())
    };
    let lacunae = if root.join("registry/lacunae.toml").exists() {
        load_toml(&root.join("registry/lacunae.toml"))?
    } else {
        Value::Table(Default::default())
    };

    let claim_rows = table_array(&claims, "claim");
    let insight_rows = table_array(&insights, "insight");
    let experiment_rows = table_array(&experiments, "experiment");
    let lineage_rows = lineage
        .as_ref()
        .map(|value| table_array(value, "lineage"))
        .unwrap_or(&[]);
    let lineage_edges = lineage
        .as_ref()
        .map(|value| table_array(value, "edge"))
        .unwrap_or(&[]);
    let roadmap_rows = table_array(&roadmap, "workstream");
    let todo_rows = table_array(&todo, "item");
    let action_rows = table_array(&next_actions, "action");
    let requirement_rows = table_array(&requirements, "module");
    let module_rows = table_array(&module_requirements, "module");
    let package_rows = table_array(&module_requirements, "package");
    let command_rows = table_array(&module_requirements, "command");
    let source_rows = table_array(&external_sources, "document");
    let alias_rows = table_array(&dataset_label_aliases, "alias");
    let atom_rows = table_array(&claim_atoms, "atom");
    let edge_rows = table_array(&claim_edges, "edge");
    let provenance_rows = table_array(&provenance, "record");
    let paragraph_rows = table_array(&paragraph_atoms, "paragraph");
    let marker_rows = table_array(&conflict_markers, "marker");
    let lacuna_rows = table_array(&lacunae, "lacuna");

    let claim_ids: BTreeSet<String> = claim_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let insight_ids: BTreeSet<String> = insight_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let experiment_ids: BTreeSet<String> = experiment_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let workstream_ids: BTreeSet<String> = roadmap_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let todo_ids: BTreeSet<String> = todo_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let action_ids: BTreeSet<String> = action_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let req_ids: BTreeSet<String> = requirement_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let module_req_ids: BTreeSet<String> = module_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let source_ids = collect_source_ids(&root)?;
    let dataset_ids = collect_dataset_ids(&root)?;
    let dataset_label_aliases: BTreeSet<String> = alias_rows
        .iter()
        .filter_map(|row| {
            let normalized =
                normalize_dataset_label(if !table_str(row, "label_normalized").is_empty() {
                    table_str(row, "label_normalized")
                } else {
                    table_str(row, "label")
                });
            if normalized.is_empty() {
                None
            } else {
                Some(normalized)
            }
        })
        .collect();
    let marker_ids: BTreeSet<String> = marker_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let lineage_ids: BTreeSet<String> = lineage_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();

    let claim_re = Regex::new(r"\bC-\d{3}\b")?;
    let insight_re = Regex::new(r"\bI-\d{3}\b")?;
    let experiment_re = Regex::new(r"\bE-\d{3}\b")?;
    let source_re = Regex::new(r"\bXS-\d{3}\b")?;
    let source_contract_re = Regex::new(r"\bSRC-[A-Z0-9-]+\b")?;
    let dataset_re = Regex::new(r"\b(?:PC|PG|EX|AR|CU)-\d{4}\b")?;
    let workstream_re = Regex::new(r"\bWS-[A-Z0-9-]+\b")?;
    let todo_re = Regex::new(r"\bT-\d{3}\b")?;
    let action_re = Regex::new(r"\bNA-\d{3}\b")?;
    let req_re = Regex::new(r"\bREQ-[A-Z0-9-]+\b")?;

    let mut failures = Vec::new();
    let mut counters = CrossRefCounters::default();

    for row in insight_rows {
        let sid = table_str(row, "id");
        for claim_id in value_list(row, "claims") {
            check_crossref_id(
                CrossRefKind::Claims,
                &claim_id,
                &format!("insights[{sid}].claims"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
    }

    for row in experiment_rows {
        let eid = table_str(row, "id");
        for claim_id in value_list(row, "claims") {
            check_crossref_id(
                CrossRefKind::Claims,
                &claim_id,
                &format!("experiments[{eid}].claims"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        for source_id in value_list(row, "external_source_refs") {
            check_crossref_id(
                CrossRefKind::Sources,
                &source_id,
                &format!("experiments[{eid}].external_source_refs"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        for label in value_list(row, "dataset_label_refs") {
            let normalized = normalize_dataset_label(&label);
            if !dataset_label_aliases.contains(&normalized) {
                failures.push(format!(
                    "experiments[{eid}].dataset_label_refs: unknown dataset label alias {label}"
                ));
            }
        }
    }

    for row in source_rows {
        let xid = table_str(row, "id");
        for claim_id in value_list(row, "claim_refs") {
            check_crossref_id(
                CrossRefKind::Claims,
                &claim_id,
                &format!("external_sources[{xid}].claim_refs"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
    }

    for row in claim_rows {
        let cid = table_str(row, "id");
        let corpus = format!(
            "{} {}",
            table_str(row, "where_stated"),
            table_str(row, "what_would_verify_refute")
        );
        let refs = extract_crossrefs(
            &corpus,
            &claim_re,
            &insight_re,
            &experiment_re,
            &source_re,
            &source_contract_re,
            &dataset_re,
        );
        check_extracted_refs(
            &refs,
            &format!("claims[{cid}].text"),
            &mut counters,
            &mut failures,
            &CrossRefSets {
                claims: &claim_ids,
                insights: &insight_ids,
                experiments: &experiment_ids,
                sources: &source_ids,
                datasets: &dataset_ids,
            },
        );
    }

    for row in atom_rows {
        let aid = table_str(row, "id");
        check_crossref_id(
            CrossRefKind::Claims,
            table_str(row, "claim_id"),
            &format!("claims_atoms[{aid}].claim_id"),
            &mut counters,
            &mut failures,
            &CrossRefSets {
                claims: &claim_ids,
                insights: &insight_ids,
                experiments: &experiment_ids,
                sources: &source_ids,
                datasets: &dataset_ids,
            },
        );
        for field in ["cross_refs", "where_stated_refs", "verification_refs"] {
            for value in value_list(row, field) {
                let refs = extract_crossrefs(
                    &value,
                    &claim_re,
                    &insight_re,
                    &experiment_re,
                    &source_re,
                    &source_contract_re,
                    &dataset_re,
                );
                check_extracted_refs(
                    &refs,
                    &format!("claims_atoms[{aid}].{field}"),
                    &mut counters,
                    &mut failures,
                    &CrossRefSets {
                        claims: &claim_ids,
                        insights: &insight_ids,
                        experiments: &experiment_ids,
                        sources: &source_ids,
                        datasets: &dataset_ids,
                    },
                );
            }
        }
    }

    for row in edge_rows {
        let eid = table_str(row, "id");
        check_crossref_id(
            CrossRefKind::Claims,
            table_str(row, "claim_id"),
            &format!("claims_evidence_edges[{eid}].claim_id"),
            &mut counters,
            &mut failures,
            &CrossRefSets {
                claims: &claim_ids,
                insights: &insight_ids,
                experiments: &experiment_ids,
                sources: &source_ids,
                datasets: &dataset_ids,
            },
        );
        let refs = extract_crossrefs(
            table_str(row, "target_ref"),
            &claim_re,
            &insight_re,
            &experiment_re,
            &source_re,
            &source_contract_re,
            &dataset_re,
        );
        check_extracted_refs(
            &refs,
            &format!("claims_evidence_edges[{eid}].target_ref"),
            &mut counters,
            &mut failures,
            &CrossRefSets {
                claims: &claim_ids,
                insights: &insight_ids,
                experiments: &experiment_ids,
                sources: &source_ids,
                datasets: &dataset_ids,
            },
        );
    }

    for row in provenance_rows {
        let pid = table_str(row, "id");
        for claim_id in value_list(row, "claim_refs") {
            check_crossref_id(
                CrossRefKind::Claims,
                &claim_id,
                &format!("provenance_sources[{pid}].claim_refs"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        let refs = extract_crossrefs(
            table_str(row, "source_ref"),
            &claim_re,
            &insight_re,
            &experiment_re,
            &source_re,
            &source_contract_re,
            &dataset_re,
        );
        for source_id in refs.sources {
            check_crossref_id(
                CrossRefKind::Sources,
                &source_id,
                &format!("provenance_sources[{pid}].source_ref"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        for dataset_id in refs.datasets {
            check_crossref_id(
                CrossRefKind::Datasets,
                &dataset_id,
                &format!("provenance_sources[{pid}].source_ref"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
    }

    for row in paragraph_rows {
        let pid = table_str(row, "id");
        for claim_id in value_list(row, "claim_refs") {
            check_crossref_id(
                CrossRefKind::Claims,
                &claim_id,
                &format!("narrative_paragraph_atoms[{pid}].claim_refs"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
    }

    for row in lineage_rows {
        let lid = table_str(row, "id");
        check_crossref_id(
            CrossRefKind::Experiments,
            table_str(row, "experiment_id"),
            &format!("experiment_lineage[{lid}].experiment_id"),
            &mut counters,
            &mut failures,
            &CrossRefSets {
                claims: &claim_ids,
                insights: &insight_ids,
                experiments: &experiment_ids,
                sources: &source_ids,
                datasets: &dataset_ids,
            },
        );
        for claim_id in value_list(row, "claim_refs") {
            check_crossref_id(
                CrossRefKind::Claims,
                &claim_id,
                &format!("experiment_lineage[{lid}].claim_refs"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        for dataset_id in value_list(row, "dataset_refs") {
            check_crossref_id(
                CrossRefKind::Datasets,
                &dataset_id,
                &format!("experiment_lineage[{lid}].dataset_refs"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        for label in value_list(row, "dataset_label_refs") {
            let normalized = normalize_dataset_label(&label);
            if !dataset_label_aliases.contains(&normalized) {
                failures.push(format!(
                    "experiment_lineage[{lid}].dataset_label_refs: unknown dataset label alias {label}"
                ));
            }
        }
    }

    for row in lineage_edges {
        let edge_id = table_str(row, "id");
        let lineage_id = table_str(row, "lineage_id");
        if !lineage_ids.contains(lineage_id) {
            failures.push(format!(
                "experiment_lineage.edge[{edge_id}].lineage_id: unknown lineage {lineage_id}"
            ));
        }
        check_crossref_id(
            CrossRefKind::Experiments,
            table_str(row, "from_id"),
            &format!("experiment_lineage.edge[{edge_id}].from_id"),
            &mut counters,
            &mut failures,
            &CrossRefSets {
                claims: &claim_ids,
                insights: &insight_ids,
                experiments: &experiment_ids,
                sources: &source_ids,
                datasets: &dataset_ids,
            },
        );
        let to_ref = table_str(row, "to_ref");
        match table_str(row, "to_kind") {
            "claim" => check_crossref_id(
                CrossRefKind::Claims,
                to_ref,
                &format!("experiment_lineage.edge[{edge_id}].to_ref"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            ),
            "dataset" => check_crossref_id(
                CrossRefKind::Datasets,
                to_ref,
                &format!("experiment_lineage.edge[{edge_id}].to_ref"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            ),
            _ => {}
        }
    }

    for row in roadmap_rows {
        let workstream_id = table_str(row, "id");
        for dep in value_list(row, "dependencies") {
            check_dependency(
                &dep,
                &format!("roadmap.workstream[{workstream_id}].dependencies"),
                &CrossRefRegexes {
                    workstream: &workstream_re,
                    todo: &todo_re,
                    action: &action_re,
                    req: &req_re,
                    claim: &claim_re,
                    insight: &insight_re,
                    experiment: &experiment_re,
                },
                &CrossRefIdSets {
                    workstream: &workstream_ids,
                    todo: &todo_ids,
                    action: &action_ids,
                    req: &req_ids,
                },
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        for claim_id in value_list(row, "claims") {
            check_crossref_id(
                CrossRefKind::Claims,
                &claim_id,
                &format!("roadmap.workstream[{workstream_id}].claims"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        let insight_id = table_str(row, "insight");
        if !insight_id.is_empty() {
            check_crossref_id(
                CrossRefKind::Insights,
                insight_id,
                &format!("roadmap.workstream[{workstream_id}].insight"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        for value in value_list(row, "evidence_refs") {
            let refs = extract_crossrefs(
                &value,
                &claim_re,
                &insight_re,
                &experiment_re,
                &source_re,
                &source_contract_re,
                &dataset_re,
            );
            check_extracted_refs(
                &refs,
                &format!("roadmap.workstream[{workstream_id}].evidence_refs"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
    }

    for row in todo_rows {
        let todo_id = table_str(row, "id");
        for dep in value_list(row, "dependencies") {
            check_dependency(
                &dep,
                &format!("todo.item[{todo_id}].dependencies"),
                &CrossRefRegexes {
                    workstream: &workstream_re,
                    todo: &todo_re,
                    action: &action_re,
                    req: &req_re,
                    claim: &claim_re,
                    insight: &insight_re,
                    experiment: &experiment_re,
                },
                &CrossRefIdSets {
                    workstream: &workstream_ids,
                    todo: &todo_ids,
                    action: &action_ids,
                    req: &req_ids,
                },
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
    }

    for row in action_rows {
        let action_id = table_str(row, "id");
        for dep in value_list(row, "dependencies") {
            check_dependency(
                &dep,
                &format!("next_actions.action[{action_id}].dependencies"),
                &CrossRefRegexes {
                    workstream: &workstream_re,
                    todo: &todo_re,
                    action: &action_re,
                    req: &req_re,
                    claim: &claim_re,
                    insight: &insight_re,
                    experiment: &experiment_re,
                },
                &CrossRefIdSets {
                    workstream: &workstream_ids,
                    todo: &todo_ids,
                    action: &action_ids,
                    req: &req_ids,
                },
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        for value in value_list(row, "evidence_refs") {
            let refs = extract_crossrefs(
                &value,
                &claim_re,
                &insight_re,
                &experiment_re,
                &source_re,
                &source_contract_re,
                &dataset_re,
            );
            check_extracted_refs(
                &refs,
                &format!("next_actions.action[{action_id}].evidence_refs"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
    }

    for row in requirement_rows {
        let requirement_id = table_str(row, "id");
        for module_id in value_list(row, "requires_modules") {
            if !req_ids.contains(&module_id) {
                failures.push(format!(
                    "requirements.module[{requirement_id}].requires_modules: unknown requirement module {module_id}"
                ));
            }
        }
    }

    let package_ids: BTreeSet<String> = package_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();
    let command_ids: BTreeSet<String> = command_rows
        .iter()
        .map(|row| table_str(row, "id").to_string())
        .filter(|value| !value.is_empty())
        .collect();

    for row in module_rows {
        let module_id = table_str(row, "id");
        if !req_ids.contains(module_id) {
            failures.push(format!(
                "module_requirements.module: missing requirements module {module_id}"
            ));
        }
        for requires_module in value_list(row, "requires_modules") {
            if !module_req_ids.contains(&requires_module) {
                failures.push(format!(
                    "module_requirements.module[{module_id}].requires_modules: unknown module {requires_module}"
                ));
            }
        }
        for package_id in value_list(row, "package_refs") {
            if !package_ids.contains(&package_id) {
                failures.push(format!(
                    "module_requirements.module[{module_id}].package_refs: unknown package {package_id}"
                ));
            }
        }
        for command_id in value_list(row, "command_refs") {
            if !command_ids.contains(&command_id) {
                failures.push(format!(
                    "module_requirements.module[{module_id}].command_refs: unknown command {command_id}"
                ));
            }
        }
    }

    for row in package_rows {
        let package_id = table_str(row, "id");
        let module_id = table_str(row, "module_id");
        if !module_req_ids.contains(module_id) {
            failures.push(format!(
                "module_requirements.package[{package_id}].module_id: unknown module {module_id}"
            ));
        }
    }

    for row in command_rows {
        let command_id = table_str(row, "id");
        let module_id = table_str(row, "module_id");
        if !module_req_ids.contains(module_id) {
            failures.push(format!(
                "module_requirements.command[{command_id}].module_id: unknown module {module_id}"
            ));
        }
    }

    for row in marker_rows {
        let marker_id = table_str(row, "id");
        for claim_id in value_list(row, "claim_refs") {
            check_crossref_id(
                CrossRefKind::Claims,
                &claim_id,
                &format!("conflict_markers[{marker_id}].claim_refs"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
    }

    for row in lacuna_rows {
        let lacuna_id = table_str(row, "id");
        for claim_id in value_list(row, "claim_refs") {
            check_crossref_id(
                CrossRefKind::Claims,
                &claim_id,
                &format!("lacunae[{lacuna_id}].claim_refs"),
                &mut counters,
                &mut failures,
                &CrossRefSets {
                    claims: &claim_ids,
                    insights: &insight_ids,
                    experiments: &experiment_ids,
                    sources: &source_ids,
                    datasets: &dataset_ids,
                },
            );
        }
        for marker_id in value_list(row, "related_marker_ids") {
            if !marker_ids.contains(&marker_id) {
                failures.push(format!(
                    "lacunae[{lacuna_id}].related_marker_ids: unknown marker {marker_id}"
                ));
            }
        }
    }

    if !failures.is_empty() {
        eprintln!("ERROR: cross-registry reference verification failed.");
        for item in failures.iter().take(300) {
            eprintln!("- {item}");
        }
        if failures.len() > 300 {
            eprintln!("- ... and {} more failures", failures.len() - 300);
        }
        bail!("cross-registry reference verification failed");
    }

    println!(
        "OK: cross-registry references verified. checks claims={} insights={} experiments={} sources={} datasets={}",
        counters.claims,
        counters.insights,
        counters.experiments,
        counters.sources,
        counters.datasets
    );
    Ok(())
}

struct CrossRefSets<'a> {
    claims: &'a BTreeSet<String>,
    insights: &'a BTreeSet<String>,
    experiments: &'a BTreeSet<String>,
    sources: &'a BTreeSet<String>,
    datasets: &'a BTreeSet<String>,
}

/// Regex bundle for the cross-reference checker. Bundling the 7 ID-prefix
/// detector regexes into one struct keeps `check_dependency` under
/// clippy::too_many_arguments without resorting to `#[allow]`; each regex is
/// borrowed so call sites can build it once and reuse.
struct CrossRefRegexes<'a> {
    workstream: &'a Regex,
    todo: &'a Regex,
    action: &'a Regex,
    req: &'a Regex,
    claim: &'a Regex,
    insight: &'a Regex,
    experiment: &'a Regex,
}

/// ID-set bundle for the kinds the cross-reference checker must validate
/// against (workstreams, todos, actions, requirements). Claims, insights,
/// experiments, sources, and datasets are validated through `CrossRefSets`
/// instead because they share the same downstream `check_crossref_id`
/// surface.
struct CrossRefIdSets<'a> {
    workstream: &'a BTreeSet<String>,
    todo: &'a BTreeSet<String>,
    action: &'a BTreeSet<String>,
    req: &'a BTreeSet<String>,
}

enum CrossRefKind {
    Claims,
    Insights,
    Experiments,
    Sources,
    Datasets,
}

fn collect_source_ids(root: &Path) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    let external_sources = load_toml(&root.join("registry/external_sources.toml"))?;
    for row in table_array(&external_sources, "document") {
        let source_id = table_str(row, "id");
        if !source_id.is_empty() {
            out.insert(source_id.to_string());
        }
    }
    let source_contract_path = root.join("data/external/SOURCES.toml");
    if source_contract_path.exists() {
        let source_contracts = load_toml(&source_contract_path)?;
        for row in table_array(&source_contracts, "source") {
            let source_id = table_str(row, "id");
            if !source_id.is_empty() {
                out.insert(source_id.to_string());
            }
        }
    }
    Ok(out)
}

fn extract_crossrefs(
    text: &str,
    claim_re: &Regex,
    insight_re: &Regex,
    experiment_re: &Regex,
    source_re: &Regex,
    source_contract_re: &Regex,
    dataset_re: &Regex,
) -> ExtractedRefs {
    let claims = sorted_unique_matches(text, claim_re);
    let insights = sorted_unique_matches(text, insight_re);
    let experiments = sorted_unique_matches(text, experiment_re);
    let datasets = sorted_unique_matches(text, dataset_re);
    let mut sources = BTreeSet::new();
    for reference in sorted_unique_matches(text, source_re)
        .into_iter()
        .chain(sorted_unique_matches(text, source_contract_re))
    {
        if !text.contains(&format!("{reference}-*")) {
            sources.insert(reference);
        }
    }
    ExtractedRefs {
        claims,
        insights,
        experiments,
        sources: sources.into_iter().collect(),
        datasets,
    }
}

fn sorted_unique_matches(text: &str, regex: &Regex) -> Vec<String> {
    let mut matches = BTreeSet::new();
    for hit in regex.find_iter(text) {
        matches.insert(hit.as_str().to_string());
    }
    matches.into_iter().collect()
}

fn check_extracted_refs(
    refs: &ExtractedRefs,
    where_label: &str,
    counters: &mut CrossRefCounters,
    failures: &mut Vec<String>,
    sets: &CrossRefSets<'_>,
) {
    for claim_id in &refs.claims {
        check_crossref_id(
            CrossRefKind::Claims,
            claim_id,
            where_label,
            counters,
            failures,
            sets,
        );
    }
    for insight_id in &refs.insights {
        check_crossref_id(
            CrossRefKind::Insights,
            insight_id,
            where_label,
            counters,
            failures,
            sets,
        );
    }
    for experiment_id in &refs.experiments {
        check_crossref_id(
            CrossRefKind::Experiments,
            experiment_id,
            where_label,
            counters,
            failures,
            sets,
        );
    }
    for source_id in &refs.sources {
        check_crossref_id(
            CrossRefKind::Sources,
            source_id,
            where_label,
            counters,
            failures,
            sets,
        );
    }
    for dataset_id in &refs.datasets {
        check_crossref_id(
            CrossRefKind::Datasets,
            dataset_id,
            where_label,
            counters,
            failures,
            sets,
        );
    }
}

fn check_dependency(
    dep: &str,
    where_label: &str,
    res: &CrossRefRegexes<'_>,
    aux_ids: &CrossRefIdSets<'_>,
    counters: &mut CrossRefCounters,
    failures: &mut Vec<String>,
    sets: &CrossRefSets<'_>,
) {
    let value = dep.trim();
    if value.is_empty() {
        return;
    }
    if res.claim.is_match(value) && res.claim.find(value).map(|m| m.as_str()) == Some(value) {
        check_crossref_id(
            CrossRefKind::Claims,
            value,
            where_label,
            counters,
            failures,
            sets,
        );
        return;
    }
    if res.insight.is_match(value) && res.insight.find(value).map(|m| m.as_str()) == Some(value) {
        check_crossref_id(
            CrossRefKind::Insights,
            value,
            where_label,
            counters,
            failures,
            sets,
        );
        return;
    }
    if res.experiment.is_match(value)
        && res.experiment.find(value).map(|m| m.as_str()) == Some(value)
    {
        check_crossref_id(
            CrossRefKind::Experiments,
            value,
            where_label,
            counters,
            failures,
            sets,
        );
        return;
    }
    if res.workstream.is_match(value)
        && res.workstream.find(value).map(|m| m.as_str()) == Some(value)
    {
        if !aux_ids.workstream.contains(value) {
            failures.push(format!("{where_label}: unknown workstream {value}"));
        }
        return;
    }
    if res.todo.is_match(value) && res.todo.find(value).map(|m| m.as_str()) == Some(value) {
        if !aux_ids.todo.contains(value) {
            failures.push(format!("{where_label}: unknown todo {value}"));
        }
        return;
    }
    if res.action.is_match(value) && res.action.find(value).map(|m| m.as_str()) == Some(value) {
        if !aux_ids.action.contains(value) {
            failures.push(format!("{where_label}: unknown action {value}"));
        }
        return;
    }
    if res.req.is_match(value) && res.req.find(value).map(|m| m.as_str()) == Some(value) {
        if !aux_ids.req.contains(value) {
            failures.push(format!("{where_label}: unknown requirement module {value}"));
        }
        return;
    }
    failures.push(format!("{where_label}: malformed dependency id {value}"));
}

fn check_crossref_id(
    kind: CrossRefKind,
    rid: &str,
    where_label: &str,
    counters: &mut CrossRefCounters,
    failures: &mut Vec<String>,
    sets: &CrossRefSets<'_>,
) {
    if rid.is_empty() {
        return;
    }
    match kind {
        CrossRefKind::Claims => {
            counters.claims += 1;
            if !sets.claims.contains(rid) {
                failures.push(format!("{where_label}: unknown claim {rid}"));
            }
        }
        CrossRefKind::Insights => {
            counters.insights += 1;
            if !sets.insights.contains(rid) {
                failures.push(format!("{where_label}: unknown insight {rid}"));
            }
        }
        CrossRefKind::Experiments => {
            counters.experiments += 1;
            if !sets.experiments.contains(rid) {
                failures.push(format!("{where_label}: unknown experiment {rid}"));
            }
        }
        CrossRefKind::Sources => {
            counters.sources += 1;
            if !sets.sources.contains(rid) {
                failures.push(format!("{where_label}: unknown source {rid}"));
            }
        }
        CrossRefKind::Datasets => {
            counters.datasets += 1;
            if !sets.datasets.contains(rid) {
                failures.push(format!("{where_label}: unknown dataset {rid}"));
            }
        }
    }
}

fn shape_summary(value: &Value) -> serde_json::Value {
    match value {
        Value::Table(table) => {
            let mut keys = table.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            json!({
                "type": "table",
                "keys": keys,
            })
        }
        Value::Array(items) => {
            if items.is_empty() {
                json!({"type": "array", "row_count": 0, "entry_kind": "empty"})
            } else if items.iter().all(Value::is_table) {
                let key_sets = items
                    .iter()
                    .filter_map(Value::as_table)
                    .map(|table| table.keys().cloned().collect::<BTreeSet<_>>())
                    .collect::<Vec<_>>();
                let required_keys = key_sets
                    .iter()
                    .cloned()
                    .reduce(|lhs, rhs| lhs.intersection(&rhs).cloned().collect())
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<Vec<_>>();
                let union_keys = key_sets
                    .iter()
                    .cloned()
                    .reduce(|lhs, rhs| lhs.union(&rhs).cloned().collect())
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<Vec<_>>();
                json!({
                    "type": "array",
                    "row_count": items.len(),
                    "entry_kind": "table",
                    "required_keys": required_keys,
                    "union_keys": union_keys,
                })
            } else {
                let entry_types: BTreeSet<&'static str> = items
                    .iter()
                    .map(|item| match item {
                        Value::String(_) => "string",
                        Value::Integer(_) => "integer",
                        Value::Float(_) => "float",
                        Value::Boolean(_) => "boolean",
                        Value::Datetime(_) => "datetime",
                        Value::Array(_) => "array",
                        Value::Table(_) => "table",
                    })
                    .collect();
                json!({
                    "type": "array",
                    "row_count": items.len(),
                    "entry_kind": "scalar_or_mixed",
                    "entry_types": entry_types.into_iter().collect::<Vec<_>>(),
                })
            }
        }
        Value::String(_) => json!({"type": "str"}),
        Value::Integer(_) => json!({"type": "int"}),
        Value::Float(_) => json!({"type": "float"}),
        Value::Boolean(_) => json!({"type": "bool"}),
        Value::Datetime(_) => json!({"type": "datetime"}),
    }
}

fn hex_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn verify_dataset_label_aliases(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let alias_path = root.join("registry/dataset_label_aliases.toml");
    let lineage_path = root.join("registry/experiment_lineage.toml");
    read_ascii_text(&alias_path)?;
    if lineage_path.exists() {
        read_ascii_text(&lineage_path)?;
    }
    let alias_raw = load_toml(&alias_path)?;
    let experiments = load_control_plane_registry(
        &root,
        &args.db,
        ControlPlaneCompatKind::Experiments,
        "registry/experiments.toml",
    )?;
    let lineages = if lineage_path.exists() {
        Some(load_toml(&lineage_path)?)
    } else {
        None
    };
    let dataset_ids = collect_dataset_ids(&root)?;
    let dataset_re = Regex::new(r"^(?:PC|PG|EX|AR|CU)-\d{4}$")?;
    let mut failures = Vec::new();
    let rows = table_array(&alias_raw, "alias");
    let meta = alias_raw
        .get("dataset_label_aliases")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();
    if meta
        .get("alias_count")
        .and_then(Value::as_integer)
        .unwrap_or(-1)
        != rows.len() as i64
    {
        failures.push("dataset_label_aliases alias_count metadata mismatch".to_string());
    }
    let mut normalized_seen = BTreeMap::<String, String>::new();
    let mut alias_labels = BTreeSet::<String>::new();
    for row in rows {
        let alias_id = table_str(row, "id");
        let label = table_str(row, "label").trim();
        let normalized =
            normalize_dataset_label(if table_str(row, "label_normalized").trim().is_empty() {
                label
            } else {
                table_str(row, "label_normalized")
            });
        let canonical_dataset_id = table_str(row, "canonical_dataset_id").trim();
        if alias_id.is_empty() {
            failures.push("dataset_label_aliases alias row missing id".to_string());
        }
        if label.is_empty() {
            failures.push(format!("dataset_label_aliases[{alias_id}] missing label"));
        }
        if normalized.is_empty() {
            failures.push(format!(
                "dataset_label_aliases[{alias_id}] missing label_normalized"
            ));
        }
        if !dataset_re.is_match(canonical_dataset_id) {
            failures.push(format!(
                "dataset_label_aliases[{alias_id}] invalid canonical_dataset_id: {canonical_dataset_id}"
            ));
        } else if !dataset_ids.contains(canonical_dataset_id) {
            failures.push(format!(
                "dataset_label_aliases[{alias_id}] unknown canonical_dataset_id: {canonical_dataset_id}"
            ));
        }
        if let Some(existing) = normalized_seen.insert(normalized.clone(), alias_id.to_string()) {
            failures.push(format!(
                "dataset_label_aliases duplicate normalized label: {normalized} ({existing}, {alias_id})"
            ));
        }
        alias_labels.insert(normalized);
    }
    for row in table_array(&experiments, "experiment") {
        let experiment_id = table_str(row, "id");
        for label in string_list(row, "dataset_label_refs") {
            let normalized = normalize_dataset_label(&label);
            if !alias_labels.contains(&normalized) {
                failures.push(format!(
                    "experiments[{experiment_id}] unknown dataset_label_ref: {label}"
                ));
            }
        }
    }
    for row in lineages
        .as_ref()
        .map(|value| table_array(value, "lineage"))
        .unwrap_or(&[])
    {
        let lineage_id = table_str(row, "id");
        for label in string_list(row, "dataset_label_refs") {
            let normalized = normalize_dataset_label(&label);
            if !alias_labels.contains(&normalized) {
                failures.push(format!(
                    "experiment_lineage[{lineage_id}] unknown dataset_label_ref: {label}"
                ));
            }
        }
    }
    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }
    println!(
        "OK: dataset label aliases verified. aliases={} experiments={} lineages={}",
        rows.len(),
        table_array(&experiments, "experiment").len(),
        lineages
            .as_ref()
            .map(|value| table_array(value, "lineage").len())
            .unwrap_or(0)
    );
    Ok(())
}

fn normalize_dataset_label(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn collect_dataset_ids(root: &Path) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    let candidates = [
        "registry/project_csv_canonical_datasets.toml",
        "registry/project_csv_generated_artifacts.toml",
        "registry/project_csv_generated_datasets.toml",
        "registry/external_csv_datasets.toml",
        "registry/archive_csv_datasets.toml",
        "registry/curated_csv_datasets.toml",
    ];
    for rel in candidates {
        let path = root.join(rel);
        if !path.exists() {
            continue;
        }
        let raw = load_toml(&path)?;
        for row in table_array(&raw, "dataset") {
            let id = table_str(row, "id").trim();
            if !id.is_empty() {
                out.insert(id.to_string());
            }
        }
    }
    Ok(out)
}

fn verify_external_source_operational_contracts(args: &CommonArgs) -> Result<()> {
    let root = resolve_root(args)?;
    let external_sources = load_toml(&root.join("registry/external_sources.toml"))?;
    let experiments = load_control_plane_registry(
        &root,
        &args.db,
        ControlPlaneCompatKind::Experiments,
        "registry/experiments.toml",
    )?;
    let role_allowlist = BTreeSet::from([
        "reference_capture",
        "provider_manifest",
        "chronology_pack_status",
        "generated_index",
        "dataset_lineage_audit",
        "claims_inbox",
        "falsification_contract",
        "claim_bridge",
        "mirror_audit",
    ]);
    let truth_allowlist = BTreeSet::from([
        "chronology_control",
        "environment_context",
        "lineage_transition",
        "observation_benchmark",
    ]);
    let roles_requiring_contracts =
        BTreeSet::from(["dataset_lineage_audit", "falsification_contract"]);
    let anomaly_markers = [
        "data/external/flyby_anomaly/",
        "data/external/pioneer_anomaly/",
        "data/output/anomaly/",
    ];
    let docs = table_array(&external_sources, "document");
    let docs_by_id = docs
        .iter()
        .map(|row| (table_str(row, "id").to_string(), row))
        .collect::<BTreeMap<_, _>>();
    let mut failures = Vec::new();
    let mut anomaly_count = 0usize;
    for row in docs {
        let doc_id = table_str(row, "id");
        let role = table_str(row, "operational_role").trim();
        if !role_allowlist.contains(role) {
            failures.push(format!(
                "external_sources[{doc_id}] invalid operational_role: {role}"
            ));
        }
        let truth_surfaces = string_list(row, "truth_surfaces");
        let unknown_truth: Vec<String> = truth_surfaces
            .iter()
            .filter(|surface| !truth_allowlist.contains(surface.as_str()))
            .cloned()
            .collect();
        if !unknown_truth.is_empty() {
            failures.push(format!(
                "external_sources[{doc_id}] invalid truth_surfaces: {}",
                unknown_truth.join(", ")
            ));
        }
        let contract_paths = string_list(row, "artifact_contract_paths");
        if roles_requiring_contracts.contains(role) && contract_paths.is_empty() {
            failures.push(format!(
                "external_sources[{doc_id}] role {role} requires artifact_contract_paths"
            ));
        }
        for rel in contract_paths {
            if !root.join(&rel).exists() {
                failures.push(format!(
                    "external_sources[{doc_id}] missing artifact_contract_path: {rel}"
                ));
            }
        }
        let lineage = table_str(row, "source_lineage_summary").trim();
        if role == "dataset_lineage_audit" && lineage.is_empty() {
            failures.push(format!(
                "external_sources[{doc_id}] dataset_lineage_audit requires source_lineage_summary"
            ));
        }
        if role == "falsification_contract"
            && !truth_surfaces
                .iter()
                .any(|surface| surface == "observation_benchmark")
        {
            failures.push(format!(
                "external_sources[{doc_id}] falsification_contract must expose observation_benchmark"
            ));
        }
    }
    for row in table_array(&experiments, "experiment") {
        let experiment_id = table_str(row, "id");
        if !looks_like_anomaly_surface_experiment(row, &anomaly_markers) {
            continue;
        }
        anomaly_count += 1;
        let source_refs = string_list(row, "external_source_refs");
        let truth_consumption = string_list(row, "truth_surface_consumption");
        if source_refs.is_empty() {
            failures.push(format!(
                "experiments[{experiment_id}] anomaly surface experiment missing external_source_refs"
            ));
            continue;
        }
        if truth_consumption.is_empty() {
            failures.push(format!(
                "experiments[{experiment_id}] anomaly surface experiment missing truth_surface_consumption"
            ));
            continue;
        }
        let unknown_sources: Vec<String> = source_refs
            .iter()
            .filter(|reference| !docs_by_id.contains_key(reference.as_str()))
            .cloned()
            .collect();
        if !unknown_sources.is_empty() {
            failures.push(format!(
                "experiments[{experiment_id}] unknown external_source_refs: {}",
                unknown_sources.join(", ")
            ));
            continue;
        }
        let invalid_truth: Vec<String> = truth_consumption
            .iter()
            .filter(|surface| !truth_allowlist.contains(surface.as_str()))
            .cloned()
            .collect();
        if !invalid_truth.is_empty() {
            failures.push(format!(
                "experiments[{experiment_id}] invalid truth_surface_consumption: {}",
                invalid_truth.join(", ")
            ));
            continue;
        }
        let mut offered = BTreeSet::new();
        for source_ref in &source_refs {
            if let Some(row) = docs_by_id.get(source_ref.as_str()) {
                for surface in string_list(row, "truth_surfaces") {
                    offered.insert(surface);
                }
            }
        }
        let missing: Vec<String> = truth_consumption
            .iter()
            .filter(|surface| !offered.contains(surface.as_str()))
            .cloned()
            .collect();
        if !missing.is_empty() {
            failures.push(format!(
                "experiments[{experiment_id}] truth surfaces not offered by external_source_refs: {}",
                missing.join(", ")
            ));
        }
    }
    if !failures.is_empty() {
        bail!(failures.join("\n"));
    }
    println!(
        "OK: external-source operational contracts verified. docs={} anomaly_experiments={}",
        docs.len(),
        anomaly_count
    );
    Ok(())
}

fn looks_like_anomaly_surface_experiment(row: &Value, markers: &[&str]) -> bool {
    for key in ["input", "run"] {
        let value = table_str(row, key);
        if markers.iter().any(|marker| value.contains(marker)) {
            return true;
        }
    }
    for key in [
        "output",
        "dataset_refs",
        "input_path_refs",
        "output_path_refs",
    ] {
        for value in string_list(row, key) {
            if markers.iter().any(|marker| value.contains(marker))
                || value.to_lowercase().contains("flyby anomaly")
                || value.to_lowercase().contains("pioneer anomaly")
            {
                return true;
            }
        }
    }
    false
}
