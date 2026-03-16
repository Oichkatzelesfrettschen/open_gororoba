use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use csv::ReaderBuilder;
use data_core::{HeliosphereFeatureRow, SparseMemoryPlan, estimate_sparse_memory_plan};
use gororoba_gpu_bridge::{HardwareCaps, detect_best_backend, probe_simd};
use lbm_3d::solver::LbmSolver3D;
use lbm_3d_cuda::{LbmSolver3DCuda, Precision, probe_cuda_available, probe_cuda_device_props};
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf, str::FromStr, time::Instant};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-lbm-cube-run",
    about = "Benchmark dense LBM and sparse memory plans from heliosphere feature cubes"
)]
struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long)]
    window: Option<String>,

    #[arg(long, default_value_t = 128)]
    grid: usize,

    #[arg(long, default_value_t = 64)]
    steps: usize,

    #[arg(long, default_value = "raw")]
    activity_mode: String,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivityMode {
    Raw,
    EventMask,
}

impl FromStr for ActivityMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "raw" => Ok(Self::Raw),
            "event-mask" => Ok(Self::EventMask),
            other => Err(format!(
                "unsupported activity mode '{other}'; expected raw or event-mask"
            )),
        }
    }
}

#[derive(Debug, Serialize)]
struct DenseBenchmarkResult {
    backend: String,
    grid: usize,
    steps: usize,
    elapsed_seconds: f64,
    mlups: f64,
    rho_init: f64,
    ux_init: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    generated_at_utc: String,
    cube_csv: String,
    selected_window: Option<String>,
    selected_rows: usize,
    activity_mode: String,
    active_fraction_from_cube: f64,
    activity_rows: usize,
    occupancy_rows: usize,
    occupancy_excluded_rows: usize,
    hardware: HardwareSummary,
    dense_cpu: DenseBenchmarkResult,
    dense_gpu: Option<DenseBenchmarkResult>,
    memory_plans: Vec<SparseMemoryPlan>,
}

#[derive(Debug, Serialize)]
struct HardwareSummary {
    best_backend: String,
    simd: String,
    cuda_available: bool,
    cuda_compute_capability: Option<String>,
    cuda_l2_bytes: Option<usize>,
    cuda_shared_mem_per_block: Option<usize>,
    cuda_bf16_native: Option<bool>,
    cuda_sparse_tile_preferred: Option<bool>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let activity_mode = ActivityMode::from_str(&cli.activity_mode).map_err(anyhow::Error::msg)?;
    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "heliosphere_lbm_cube_run_{}.toml",
            Utc::now().date_naive()
        ))
    });
    let rows = load_rows(&cli.cube_csv)?;
    let selected: Vec<HeliosphereFeatureRow> = rows
        .into_iter()
        .filter(|row| {
            cli.window
                .as_ref()
                .map(|value| row.window_name == *value)
                .unwrap_or(true)
        })
        .collect();
    let stats = benchmark_stats(&selected, activity_mode);

    let cpu = run_cpu_dense(cli.grid, cli.steps, stats.rho_init, stats.ux_init);
    let gpu = if probe_cuda_available() {
        run_gpu_dense(cli.grid, cli.steps, stats.rho_init, stats.ux_init).ok()
    } else {
        None
    };

    let simd = probe_simd();
    let cuda_props = probe_cuda_device_props();
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

    let memory_plans = [128usize, 256, 512, 1024]
        .into_iter()
        .map(|grid| estimate_sparse_memory_plan(grid, stats.active_fraction))
        .collect::<Vec<_>>();

    let report = Report {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.display().to_string(),
        selected_window: cli.window.clone(),
        selected_rows: selected.len(),
        activity_mode: cli.activity_mode.clone(),
        active_fraction_from_cube: stats.active_fraction,
        activity_rows: stats.activity_rows,
        occupancy_rows: stats.occupancy_rows,
        occupancy_excluded_rows: stats.occupancy_excluded_rows,
        hardware: HardwareSummary {
            best_backend: format!("{:?}", detect_best_backend(&caps, cli.grid.pow(3))),
            simd: simd.to_string(),
            cuda_available: caps.cuda_available,
            cuda_compute_capability: cuda_props
                .map(|value| format!("{}.{}", value.major, value.minor)),
            cuda_l2_bytes: cuda_props.map(|value| value.l2_bytes),
            cuda_shared_mem_per_block: cuda_props.map(|value| value.shared_mem_per_block),
            cuda_bf16_native: cuda_props.map(|value| value.bf16_native),
            cuda_sparse_tile_preferred: cuda_props.map(|value| value.sparse_tile_preferred),
        },
        dense_cpu: cpu,
        dense_gpu: gpu,
        memory_plans,
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, toml::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", out.display()))?;
    println!("selected_rows = {}", report.selected_rows);
    println!("active_fraction = {:.5}", report.active_fraction_from_cube);
    println!("out = {}", out.display());
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct Stats {
    rho_init: f64,
    ux_init: f64,
    active_fraction: f64,
    activity_rows: usize,
    occupancy_rows: usize,
    occupancy_excluded_rows: usize,
}

fn load_rows(path: &PathBuf) -> Result<Vec<HeliosphereFeatureRow>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("open {}", path.display()))?;
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        rows.push(row.with_context(|| format!("deserialize {}", path.display()))?);
    }
    Ok(rows)
}

fn benchmark_stats(rows: &[HeliosphereFeatureRow], activity_mode: ActivityMode) -> Stats {
    let density = mean(rows.iter().map(|row| row.density_cm3));
    let speed = mean(rows.iter().map(|row| row.speed_kms));
    let (active_fraction, activity_rows, occupancy_rows, occupancy_excluded_rows) =
        match activity_mode {
            ActivityMode::Raw => {
                let energies: Vec<f64> = rows.iter().map(|row| row.signal_energy()).collect();
                let energy_mean = mean(energies.iter().copied());
                let energy_std = stddev(&energies, energy_mean);
                let active = energies
                    .iter()
                    .filter(|value| value.is_finite() && **value > energy_mean + 0.5 * energy_std)
                    .count();
                let fraction = if rows.is_empty() {
                    0.05
                } else {
                    (active as f64 / rows.len() as f64).clamp(0.02, 0.25)
                };
                (fraction, active, active, 0usize)
            }
            ActivityMode::EventMask => event_mask_sparse_stats(rows),
        };

    Stats {
        rho_init: if density.is_finite() {
            (1.0 + density / 50.0).clamp(0.8, 1.5)
        } else {
            1.0
        },
        ux_init: if speed.is_finite() {
            (speed / 10_000.0).clamp(0.002, 0.08)
        } else {
            0.01
        },
        active_fraction,
        activity_rows,
        occupancy_rows,
        occupancy_excluded_rows,
    }
}

fn event_mask_sparse_stats(rows: &[HeliosphereFeatureRow]) -> (f64, usize, usize, usize) {
    if rows.is_empty() {
        return (0.05, 0, 0, 0);
    }
    let mut groups: BTreeMap<(String, String, String), usize> = BTreeMap::new();
    for row in rows {
        *groups
            .entry((
                row.window_name.clone(),
                row.mission.clone(),
                row.product.clone(),
            ))
            .or_insert(0) += 1;
    }

    let activity_rows = rows.iter().filter(|row| row.event_active()).count();
    let occupancy_rows = rows
        .iter()
        .filter(|row| {
            let group_len = *groups
                .get(&(
                    row.window_name.clone(),
                    row.mission.clone(),
                    row.product.clone(),
                ))
                .unwrap_or(&1);
            row.event_active() && counts_for_sparse_occupancy(row, group_len)
        })
        .count();
    let occupancy_excluded_rows = activity_rows.saturating_sub(occupancy_rows);
    let active_fraction = (occupancy_rows as f64 / rows.len() as f64).clamp(0.0, 1.0);
    (
        active_fraction,
        activity_rows,
        occupancy_rows,
        occupancy_excluded_rows,
    )
}

fn counts_for_sparse_occupancy(row: &HeliosphereFeatureRow, group_len: usize) -> bool {
    if group_len == 1 {
        let product = row.product.to_ascii_lowercase();
        if product.contains("summary") {
            return false;
        }
        let dynamic_present = row
            .dynamic_signal_channels()
            .iter()
            .any(|value| value.is_finite() && value.abs() > 0.0);
        if !dynamic_present && (row.map_flux_mean.is_finite() || row.map_flux_std.is_finite()) {
            return false;
        }
    }
    true
}

fn run_cpu_dense(grid: usize, steps: usize, rho_init: f64, ux_init: f64) -> DenseBenchmarkResult {
    let mut solver = LbmSolver3D::new(grid, grid, grid, 0.8);
    solver.initialize_uniform(rho_init, [ux_init, 0.0, 0.0]);
    let start = Instant::now();
    solver.evolve(steps);
    let elapsed = start.elapsed().as_secs_f64();
    DenseBenchmarkResult {
        backend: "cpu_dense".to_string(),
        grid,
        steps,
        elapsed_seconds: elapsed,
        mlups: (grid.pow(3) as f64 * steps as f64) / elapsed / 1.0e6,
        rho_init,
        ux_init,
    }
}

fn run_gpu_dense(
    grid: usize,
    steps: usize,
    rho_init: f64,
    ux_init: f64,
) -> Result<DenseBenchmarkResult> {
    let mut solver = LbmSolver3DCuda::new(grid, grid, grid, 0.8, Precision::FP32)
        .context("init gpu dense solver")?;
    solver
        .initialize_uniform(rho_init as f32, [ux_init as f32, 0.0, 0.0])
        .context("gpu initialize_uniform")?;
    let start = Instant::now();
    solver.evolve(steps).context("gpu evolve")?;
    let elapsed = start.elapsed().as_secs_f64();
    Ok(DenseBenchmarkResult {
        backend: "gpu_dense_fp32".to_string(),
        grid,
        steps,
        elapsed_seconds: elapsed,
        mlups: (grid.pow(3) as f64 * steps as f64) / elapsed / 1.0e6,
        rho_init,
        ux_init,
    })
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let finite: Vec<f64> = values.filter(|value| value.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.len() < 2 || !mean.is_finite() {
        return 0.0;
    }
    let var = finite
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / finite.len() as f64;
    var.sqrt()
}
