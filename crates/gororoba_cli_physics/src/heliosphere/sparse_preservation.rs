use anyhow::Result;
use chrono::Utc;
use clap::Args;
use data_core::{SparseExecutionPlan, SparseHardwareEnvelope, estimate_sparse_execution_plan};
use crate::heliosphere_eval::{
    SparseMaskSummary, load_heliosphere_rows, summarize_sparse_policies,
};
use gororoba_gpu_bridge::{HardwareCaps, probe_simd};
use lbm_3d_cuda::{probe_cuda_available, probe_cuda_device_props};
use serde::Serialize;
use std::{fs, path::PathBuf};
use verified_core::topology::HardwareTopology;

#[derive(Args, Debug)]
pub struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value_t = 24)]
    horizon_hours: i64,

    #[arg(long, default_value_t = 1024)]
    grid: usize,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct PlanSummary {
    mask_name: String,
    active_fraction: f64,
    sparse_bf16_aa_projected_gib: f64,
    execution_mode: String,
    temporal_tile_count_hint: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    cube_csv: String,
    horizon_hours: i64,
    grid: usize,
    policies: Vec<SparseMaskSummary>,
    execution_plans: Vec<PlanSummary>,
    notes: Vec<String>,
}

pub fn run(cli: Cli) -> Result<()> {
    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "heliosphere_sparse_policy_{}.toml",
            Utc::now().date_naive()
        ))
    });
    let rows = load_heliosphere_rows(&cli.cube_csv)?;
    let cache_root = cli.repo_root.join("data/external");
    let policies = summarize_sparse_policies(&rows, &cache_root, cli.horizon_hours, cli.grid)?;
    let execution_plans = policies
        .iter()
        .map(|policy| summarize_plan(&policy.name, cli.grid, policy.active_fraction))
        .collect::<Vec<_>>();
    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.display().to_string(),
        horizon_hours: cli.horizon_hours,
        grid: cli.grid,
        policies,
        execution_plans,
        notes: vec![
            "Sparse policy now compares the robust baseline against supervised budgeted policies without dropping source rows."
                .to_string(),
            "The invariant-only budget policy is the primary challenger; the hybrid algebra policy only matters if it wins under the same budget."
                .to_string(),
            "Execution planning uses the current host hardware envelope and keeps managed memory as a fallback, not the primary path."
                .to_string(),
        ],
    };
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)?;
    println!("grid = {}", report.grid);
    println!("out = {}", out.display());
    Ok(())
}

fn summarize_plan(mask_name: &str, grid: usize, active_fraction: f64) -> PlanSummary {
    let topo = HardwareTopology::current();
    let cuda_props = probe_cuda_device_props();
    let simd = probe_simd();
    let caps = HardwareCaps {
        cuda_available: probe_cuda_available(),
        cuda_ada_available: cuda_props.map(|value| value.is_ada()).unwrap_or(false),
        cuda_compute_major: cuda_props.map(|value| value.major).unwrap_or(0),
        cuda_compute_minor: cuda_props.map(|value| value.minor).unwrap_or(0),
        cuda_l2_bytes: cuda_props.map(|value| value.l2_bytes).unwrap_or(0),
        cuda_shared_mem_per_block: cuda_props
            .map(|value| value.shared_mem_per_block)
            .unwrap_or(0),
        cuda_bf16_native: cuda_props.map(|value| value.bf16_native).unwrap_or(false),
        cuda_sparse_tile_preferred: cuda_props
            .map(|value| value.sparse_tile_preferred)
            .unwrap_or(false),
        vulkan_available: false,
        simd,
    };
    let plan: SparseExecutionPlan = estimate_sparse_execution_plan(
        grid,
        active_fraction,
        None,
        SparseHardwareEnvelope {
            cuda_vram_budget_bytes: cuda_props.map(|value| value.total_global_mem_bytes),
            cuda_l2_bytes: cuda_props.map(|value| value.l2_bytes),
            cuda_shared_mem_per_block: cuda_props.map(|value| value.shared_mem_per_block),
            cuda_managed_memory: cuda_props.map(|value| value.managed_memory),
            cuda_concurrent_managed_access: cuda_props.map(|value| value.concurrent_managed_access),
            cpu_l3_safe_working_set_bytes: Some(topo.l3_safe_working_set_bytes),
            prefer_sparse_tile: caps.cuda_sparse_tile_preferred,
        },
    );
    PlanSummary {
        mask_name: mask_name.to_string(),
        active_fraction,
        sparse_bf16_aa_projected_gib: plan.memory.sparse_bf16_aa_projected_gib,
        execution_mode: format!("{:?}", plan.mode),
        temporal_tile_count_hint: plan.recommended_temporal_tiles,
    }
}
