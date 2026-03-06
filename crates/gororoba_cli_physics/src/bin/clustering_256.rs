//! clustering-256: Re-evaluate C-659 clustering claim at 256^3 grid resolution.
//!
//! Uses Besag-Clifford permutation testing to determine whether ZD-imbalance
//! clustering in a 256^3 LBM grid is statistically significant. Outputs TOML
//! evidence file suitable for the claims registry.
//!
//! Usage:
//!   clustering-256                                  # Default 256^3, 1000 perms
//!   clustering-256 --grid 128 --max-perms 500       # Quick test
//!   clustering-256 --output data/evidence/test.toml  # Custom output
//!   clustering-256 --dry-run                         # VRAM budget only (no GPU)

use clap::Parser;
use lbm_vulkan::besag_clifford_vulkan::{
    BcResult, BesagCliffordVulkanConfig, BesagCliffordVulkanEngine, should_stop_early, vram_budget,
};
use std::io::Write;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(name = "clustering-256")]
#[command(about = "Besag-Clifford permutation test for ZD-imbalance clustering at 256^3")]
struct Args {
    /// Grid dimension (N for NxNxN).
    #[arg(long, default_value = "256")]
    grid: u32,

    /// Number of LBM steps per permutation.
    #[arg(long, default_value = "1000")]
    steps: u32,

    /// Maximum number of permutations.
    #[arg(long, default_value = "1000")]
    max_perms: u32,

    /// Batch size (permutations per dispatch).
    #[arg(long, default_value = "1")]
    batch_size: u32,

    /// Number of subregions per dimension.
    #[arg(long, default_value = "8")]
    n_sub: u32,

    /// Significance level for the permutation test.
    #[arg(long, default_value = "0.05")]
    alpha: f64,

    /// RNG seed.
    #[arg(long, default_value = "42")]
    seed: u32,

    /// Output path for TOML evidence file.
    #[arg(long, default_value = "data/evidence/clustering_256.toml")]
    output: String,

    /// Dry-run mode (compute VRAM budget, skip GPU dispatch).
    #[arg(long)]
    dry_run: bool,
}

fn main() {
    let args = Args::parse();

    let config = BesagCliffordVulkanConfig {
        grid_dim: (args.grid, args.grid, args.grid),
        lbm_steps: args.steps,
        max_perms: args.max_perms,
        batch_size: args.batch_size,
        n_sub: args.n_sub,
        alpha: args.alpha,
        seed: args.seed,
        ..Default::default()
    };

    eprintln!("=== Clustering-256 Besag-Clifford Test ===");
    eprintln!(
        "    Grid: {}^3 = {} cells",
        args.grid,
        (args.grid as u64).pow(3)
    );
    eprintln!("    Max permutations: {}", args.max_perms);
    eprintln!("    Subregions: {}^3 = {}", args.n_sub, args.n_sub.pow(3));
    eprintln!("    Alpha: {}", args.alpha);
    eprintln!();

    let (bc_bytes, total_bytes) = vram_budget(&config);
    eprintln!(
        "    VRAM budget: BC={} MB, Total={} MB",
        bc_bytes / (1024 * 1024),
        total_bytes / (1024 * 1024),
    );

    if args.dry_run {
        eprintln!("    [DRY RUN] Skipping GPU dispatch.");
        eprintln!();

        // Generate synthetic result for testing
        let result = BcResult {
            n_perms: 0,
            n_extreme: 0,
            p_value: 1.0,
            early_stopped: false,
            observed_stat: 0.0,
        };
        write_evidence(&args.output, &config, &result, "dry_run");
        return;
    }

    // Live GPU dispatch
    eprintln!("    Initializing Vulkan context...");
    let ctx = match lbm_vulkan::VulkanContext::new(false) {
        Ok(c) => {
            eprintln!(
                "      GPU: {} ({} MB VRAM)",
                c.caps.device_name, c.caps.vram_mb
            );
            c
        }
        Err(e) => {
            eprintln!("    ERROR: Vulkan init failed: {e}");
            eprintln!("    Falling back to dry-run mode.");
            let result = BcResult {
                n_perms: 0,
                n_extreme: 0,
                p_value: 1.0,
                early_stopped: false,
                observed_stat: 0.0,
            };
            write_evidence(&args.output, &config, &result, "no_gpu");
            return;
        }
    };

    let mut bc_engine = match BesagCliffordVulkanEngine::new(config.clone(), Some(ctx.clone())) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("    ERROR: {e}");
            std::process::exit(1);
        }
    };

    // Create LBM engine with (1,1) screen dim for headless compute
    let mut lbm = match lbm_vulkan::compute::GororobaEngine::new(
        &ctx,
        config.grid_dim,
        (1, 1),
        lbm_vulkan::Precision::FP32,
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("    ERROR: LBM engine init failed: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = bc_engine.init_pipelines(&lbm) {
        eprintln!("    ERROR: Pipeline init failed: {e}");
        std::process::exit(1);
    }
    eprintln!("    Pipelines initialized.");

    eprintln!("    Running permutation test...");
    match bc_engine.run_permutation_test(&ctx, &mut lbm) {
        Ok(result) => {
            eprintln!();
            eprintln!("    === Results ===");
            eprintln!("    Permutations: {}", result.n_perms);
            eprintln!("    Extreme count: {}", result.n_extreme);
            eprintln!("    p-value: {:.6}", result.p_value);
            eprintln!("    Early stopped: {}", result.early_stopped);
            eprintln!("    Observed stat: {:.6}", result.observed_stat);
            eprintln!();

            let status = if result.early_stopped {
                "early_stopped"
            } else {
                "complete"
            };
            write_evidence(&args.output, &config, &result, status);
        }
        Err(e) => {
            eprintln!("    ERROR: Permutation test failed: {e}");
            std::process::exit(1);
        }
    }

    // Suppress unused import warning
    let _ = should_stop_early(0, 0, 0.05);
}

fn write_evidence(path: &str, config: &BesagCliffordVulkanConfig, result: &BcResult, status: &str) {
    // Ensure parent directory exists
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let verdict = if result.n_perms == 0 {
        "pending"
    } else if result.p_value < config.alpha {
        "verified"
    } else {
        "falsified"
    };

    let toml = format!(
        r#"# Besag-Clifford permutation test for C-659 clustering claim
# Generated by clustering-256 binary

[experiment]
id = "E-088"
name = "clustering-256"
claim = "C-659"
status = "{status}"

[config]
grid_dim = {grid}
lbm_steps = {steps}
max_perms = {max_perms}
n_sub = {n_sub}
alpha = {alpha}
seed = {seed}

[result]
n_perms = {n_perms}
n_extreme = {n_extreme}
p_value = {p_value:.6}
early_stopped = {early_stopped}
observed_stat = {observed_stat:.6}
verdict = "{verdict}"
"#,
        status = status,
        grid = config.grid_dim.0,
        steps = config.lbm_steps,
        max_perms = config.max_perms,
        n_sub = config.n_sub,
        alpha = config.alpha,
        seed = config.seed,
        n_perms = result.n_perms,
        n_extreme = result.n_extreme,
        p_value = result.p_value,
        early_stopped = result.early_stopped,
        observed_stat = result.observed_stat,
        verdict = verdict,
    );

    match std::fs::File::create(path) {
        Ok(mut f) => {
            f.write_all(toml.as_bytes()).unwrap();
            eprintln!("    Evidence written to {path}");
        }
        Err(e) => eprintln!("    ERROR: Cannot write {path}: {e}"),
    }
}
