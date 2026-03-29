//! GRMHD torus -> CD associator pipeline.
//!
//! Initializes a Fishbone-Moncrief torus around a Schwarzschild black hole,
//! takes Euler steps to evolve MHD turbulence, and computes the CD associator
//! on the B-field time series at each step.
//!
//! This is the controlled-simulation counterpart to the nubhlight HDF5 analysis.

use anyhow::Result;
use clap::Parser;
use grmhd_core::{
    eos::GammaLaw,
    evolve,
    flux::{self, MetricCache},
    grid::Grid,
    metric::KerrMetric,
    prims,
    torus::FMTorus,
};
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "grmhd-torus-cd")]
struct Cli {
    /// Radial resolution.
    #[arg(long, default_value_t = 32)]
    n1: usize,

    /// Theta resolution.
    #[arg(long, default_value_t = 16)]
    n2: usize,

    /// Phi resolution (1 = 2D, >1 = 3D with periodic phi).
    #[arg(long, default_value_t = 1)]
    n3: usize,

    /// Number of evolution steps.
    #[arg(long, default_value_t = 10)]
    n_steps: usize,

    /// CFL number.
    #[arg(long, default_value_t = 0.4)]
    cfl: f64,

    /// CD embedding dimension.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Output JSON.
    #[arg(long, default_value = "data/output/heliosphere/ablations/grmhd_torus_cd.json")]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct StepResult {
    step: usize,
    dt: f64,
    max_rho: f64,
    max_bmag: f64,
    cd_mean: f64,
}

#[derive(Debug, Serialize)]
struct TorusCdOutput {
    n1: usize,
    n2: usize,
    n_steps: usize,
    embedding_dim: usize,
    steps: Vec<StepResult>,
}

fn extract_midplane_bfield(prims_grid: &prims::PrimGrid, grid: &Grid) -> Vec<[f64; 3]> {
    let j_mid = grid.ng + grid.n2 / 2;
    let k = grid.ng;
    let mut bfield = Vec::with_capacity(grid.n1);
    for i in grid.ng..grid.ng + grid.n1 {
        let idx = grid.idx(i, j_mid, k);
        let p = prims_grid.get(idx);
        bfield.push([p[prims::B1], p[prims::B2], p[prims::B3]]);
    }
    bfield
}

fn cd_on_bfield(bfield: &[[f64; 3]], dim: usize) -> f64 {
    if bfield.len() < dim / 3 + 3 {
        return 0.0;
    }
    let channels = 3;
    let steps = dim / channels;
    let mut embedded = Vec::new();
    for start in 0..bfield.len().saturating_sub(steps) {
        let mut v = vec![0.0f64; dim];
        for s in 0..steps {
            if start + s < bfield.len() {
                v[s * channels] = bfield[start + s][0];
                v[s * channels + 1] = bfield[start + s][1];
                v[s * channels + 2] = bfield[start + s][2];
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in v.iter_mut() { *x /= norm; }
        }
        embedded.push(v);
    }
    if embedded.len() < 3 {
        return 0.0;
    }
    // Use nearest power-of-2 dim for CD kernel
    let cd_dim = dim.next_power_of_two();
    let padded: Vec<Vec<f64>> = embedded.into_iter().map(|mut v| {
        v.resize(cd_dim, 0.0);
        v
    }).collect();
    let norms = cd_kernel::batch_sliding_associator_norms_dispatch(&padded, cd_dim, "f64");
    if norms.is_empty() { 0.0 } else { norms.iter().sum::<f64>() / norms.len() as f64 }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== GRMHD Torus -> CD Pipeline ===");
    println!("  grid: {}x{}x{}, steps: {}, dim: {}", cli.n1, cli.n2, cli.n3, cli.n_steps, cli.embedding_dim);

    let metric = KerrMetric::schwarzschild();
    let grid = Grid::new(cli.n1, cli.n2, cli.n3, 2.5, 40.0, metric);
    let eos = GammaLaw::harm_default();
    let mc = MetricCache::new(&grid);

    // Initialize FM torus with weak B-field
    let torus = FMTorus::schwarzschild(6.5, 12.0);
    let mut prim_grid = torus.initialize(&grid);
    torus.add_magnetic_loop(&grid, &mut prim_grid, 100.0);

    // Initialize constrained transport (staggered B-field)
    let mut staggered_b = grmhd_core::ct::StaggeredB::new(&grid);
    staggered_b.init_from_cell_centered(&prim_grid, &grid);

    let mut steps = Vec::new();

    // Step 0: initial state
    let bf = extract_midplane_bfield(&prim_grid, &grid);
    let max_rho = (0..grid.n_total()).map(|i| prim_grid.get(i)[prims::RHO]).fold(0.0f64, f64::max);
    let max_bmag = bf.iter().map(|b| (b[0]*b[0] + b[1]*b[1] + b[2]*b[2]).sqrt()).fold(0.0f64, f64::max);
    let cd_mean = cd_on_bfield(&bf, cli.embedding_dim);
    println!("  step 0: rho_max={:.4}, B_max={:.6}, CD={:.6}", max_rho, max_bmag, cd_mean);
    steps.push(StepResult { step: 0, dt: 0.0, max_rho, max_bmag, cd_mean });

    // Evolution loop (full 3D Euler steps with B-field evolution)
    for step in 1..=cli.n_steps {
        let max_signal = flux::max_signal_speed(&prim_grid, &grid, &mc, &eos);
        let dt = evolve::cfl_dt(&grid, max_signal, cli.cfl);

        // Full 3D Euler step with constrained transport for B-field
        flux::euler_step_3d(&mut prim_grid, &grid, &mc, &eos, dt, Some(&mut staggered_b));

        // CD analysis
        let bf = extract_midplane_bfield(&prim_grid, &grid);
        let max_rho = (0..grid.n_total()).map(|i| prim_grid.get(i)[prims::RHO]).fold(0.0f64, f64::max);
        let max_bmag = bf.iter().map(|b| (b[0]*b[0] + b[1]*b[1] + b[2]*b[2]).sqrt()).fold(0.0f64, f64::max);
        let cd_mean = cd_on_bfield(&bf, cli.embedding_dim);

        if step % 2 == 0 || step == cli.n_steps {
            println!("  step {}: dt={:.2e}, rho_max={:.4}, B_max={:.6}, CD={:.6}",
                step, dt, max_rho, max_bmag, cd_mean);
        }
        steps.push(StepResult { step, dt, max_rho, max_bmag, cd_mean });
    }

    let output = TorusCdOutput {
        n1: cli.n1, n2: cli.n2, n_steps: cli.n_steps,
        embedding_dim: cli.embedding_dim,
        steps,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("  Wrote {}", cli.out_json.display());
    Ok(())
}
