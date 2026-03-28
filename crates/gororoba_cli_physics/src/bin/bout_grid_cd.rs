//! BOUT++ tokamak grid CD analysis: read NetCDF-3 equilibrium grid, extract
//! multiple field variable radial profiles, build multi-dimensional Takens
//! embedding (16D/32D/64D), and compute CD associator as function of radius.
//!
//! Uses pure-Rust netcdf3 crate -- no C dependencies.

use anyhow::{bail, Result};
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "bout-grid-cd")]
struct Cli {
    /// Path to BOUT++ grid file (NetCDF-3 classic .nc).
    #[arg(long)]
    grid: PathBuf,

    /// Embedding dimension (16, 32, or 64).
    #[arg(long, default_value_t = 16)]
    embedding_dim: usize,

    /// Variables to use (comma-separated). Default depends on embedding_dim.
    #[arg(long)]
    vars: Option<String>,

    /// Fixed poloidal index (default: midplane ny/2).
    #[arg(long)]
    y_idx: Option<usize>,

    /// Output JSON path.
    #[arg(long, default_value = "data/output/heliosphere/ablations/bout_grid_cd.json")]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct RadialResult {
    x_idx: usize,
    psi_norm: f64,
    mean_associator: f64,
    max_associator: f64,
}

#[derive(Debug, Serialize)]
struct GridCdOutput {
    grid_file: String,
    embedding_dim: usize,
    variables: Vec<String>,
    nx: usize,
    ny: usize,
    n_radial_windows: usize,
    results: Vec<RadialResult>,
    pedestal_a: f64,
    core_a: f64,
    edge_a: f64,
    interpretation: String,
}

fn read_var_f64(reader: &mut netcdf3::FileReader, name: &str) -> Result<Vec<f64>> {
    match reader.read_var(name) {
        Ok(data) => Ok(match data {
            netcdf3::DataVector::F64(v) => v,
            netcdf3::DataVector::F32(v) => v.into_iter().map(|x| x as f64).collect(),
            netcdf3::DataVector::I32(v) => v.into_iter().map(|x| x as f64).collect(),
            netcdf3::DataVector::I16(v) => v.into_iter().map(|x| x as f64).collect(),
            netcdf3::DataVector::U8(v) => v.into_iter().map(|x| x as f64).collect(),
            netcdf3::DataVector::I8(v) => v.into_iter().map(|x| x as f64).collect(),
        }),
        Err(e) => bail!("Cannot read variable '{}': {:?}", name, e),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== BOUT++ Grid CD Analysis (netcdf3 pure Rust) ===");
    println!("  Grid: {}", cli.grid.display());
    println!("  Embedding dim: {}", cli.embedding_dim);

    let mut reader = netcdf3::FileReader::open(&cli.grid)
        .map_err(|e| anyhow::anyhow!("Failed to open NC3: {:?}", e))?;

    let ds = reader.data_set();
    let nx_val = ds.dim_size("x").unwrap_or(0);
    let ny_val = ds.dim_size("y").unwrap_or(0);
    println!("  Grid: {}x{}", nx_val, ny_val);

    // Select variables based on embedding dimension
    let default_vars = match cli.embedding_dim {
        d if d <= 16 => "Rxy,Zxy,Bpxy,Btxy,psixy,Bxy",
        d if d <= 32 => "Rxy,Zxy,Bpxy,Btxy,psixy,Bxy,pressure,Ni0,Te0,Ti0,bxcvx,bxcvy",
        _ => "Rxy,Zxy,Bpxy,Btxy,psixy,Bxy,pressure,Ni0,Te0,Ti0,bxcvx,bxcvy,bxcvz,Jpar0,dx,dy,hthe,sinty,ShiftTorsion,pol_angle,zShift,d2x",
    };
    let var_str = cli.vars.as_deref().unwrap_or(default_vars);
    let var_names: Vec<&str> = var_str.split(',').collect();
    let n_channels = var_names.len();

    println!("  Variables ({}): {:?}", n_channels, var_names);

    let steps_per_window = cli.embedding_dim / n_channels;
    let _remainder = cli.embedding_dim - steps_per_window * n_channels;

    // Read all variables as 2D arrays (nx, ny)
    let y_idx = cli.y_idx.unwrap_or(ny_val / 2);
    println!("  Poloidal index: y={}", y_idx);

    let mut profiles: Vec<Vec<f64>> = Vec::new();
    for &name in &var_names {
        let data = read_var_f64(&mut reader, name)?;
        // Extract radial profile at fixed y
        let profile: Vec<f64> = if data.len() == nx_val * ny_val {
            (0..nx_val).map(|x| data[x * ny_val + y_idx]).collect()
        } else if data.len() == nx_val {
            data
        } else if data.len() == 1 {
            vec![data[0]; nx_val]
        } else {
            println!("  Warning: {} has {} elements, expected {}x{}", name, data.len(), nx_val, ny_val);
            vec![0.0; nx_val]
        };
        profiles.push(profile);
    }

    // Read psixy for normalization
    let psi_data = read_var_f64(&mut reader, "psixy")?;
    let psi_profile: Vec<f64> = (0..nx_val).map(|x| psi_data[x * ny_val + y_idx]).collect();
    let psi_min = psi_profile.iter().cloned().fold(f64::INFINITY, f64::min);
    let psi_max = psi_profile.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let psi_range = (psi_max - psi_min).max(1e-15);

    // Build embeddings at each radial position (sliding window)
    let n_windows = if nx_val > steps_per_window { nx_val - steps_per_window } else { 0 };
    println!("  Radial windows: {}", n_windows);

    let mut results = Vec::new();

    for w in 0..n_windows {
        let mut embedded: Vec<Vec<f64>> = Vec::new();

        // Build embedding vectors from a local radial neighborhood
        let local_size = steps_per_window.min(5); // small local neighborhood
        for s in 0..local_size {
            let x = w + s;
            let mut v = Vec::with_capacity(cli.embedding_dim);
            for ch in &profiles {
                v.push(ch[x]);
            }
            // Pad with shifted values for higher dimension
            if v.len() < cli.embedding_dim {
                for ch in &profiles {
                    if v.len() >= cli.embedding_dim {
                        break;
                    }
                    let x2 = (x + 1).min(nx_val - 1);
                    v.push(ch[x2]);
                }
            }
            while v.len() < cli.embedding_dim {
                v.push(0.0);
            }
            v.truncate(cli.embedding_dim);

            // Direction normalization
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
            embedded.push(v);
        }

        if embedded.len() < 3 {
            continue;
        }

        let norms =
            cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);

        let (mean_a, max_a) = if norms.is_empty() {
            (0.0, 0.0)
        } else {
            let m = norms.iter().sum::<f64>() / norms.len() as f64;
            let mx = norms.iter().cloned().fold(0.0f64, f64::max);
            (m, mx)
        };

        let center_x = w + steps_per_window / 2;
        let psi_norm = (psi_profile[center_x] - psi_min) / psi_range;

        results.push(RadialResult {
            x_idx: center_x,
            psi_norm,
            mean_associator: mean_a,
            max_associator: max_a,
        });
    }

    // Classify regions: core (psi < 0.3), pedestal (0.7-0.95), edge/SOL (> 0.95)
    let core_a = results
        .iter()
        .filter(|r| r.psi_norm < 0.3)
        .map(|r| r.mean_associator)
        .sum::<f64>()
        / results.iter().filter(|r| r.psi_norm < 0.3).count().max(1) as f64;
    let pedestal_a = results
        .iter()
        .filter(|r| r.psi_norm > 0.7 && r.psi_norm < 0.95)
        .map(|r| r.mean_associator)
        .sum::<f64>()
        / results
            .iter()
            .filter(|r| r.psi_norm > 0.7 && r.psi_norm < 0.95)
            .count()
            .max(1) as f64;
    let edge_a = results
        .iter()
        .filter(|r| r.psi_norm > 0.95)
        .map(|r| r.mean_associator)
        .sum::<f64>()
        / results.iter().filter(|r| r.psi_norm > 0.95).count().max(1) as f64;

    // Print key results
    for r in &results {
        if (r.x_idx % 10 == 0) || r.x_idx == results.len() - 1 {
            println!(
                "  x={:3}, psi_n={:.3}: A_mean={:.6}",
                r.x_idx, r.psi_norm, r.mean_associator
            );
        }
    }

    let interp = format!(
        "{}D grid CD: {} vars, {}x{} grid. Core(psi<0.3) A={:.6}, Pedestal(0.7-0.95) A={:.6}, Edge(>0.95) A={:.6}. Ratio pedestal/core={:.2}x.",
        cli.embedding_dim, n_channels, nx_val, ny_val,
        core_a, pedestal_a, edge_a,
        if core_a > 1e-15 { pedestal_a / core_a } else { 0.0 }
    );
    println!("\n  {}", interp);

    let output = GridCdOutput {
        grid_file: cli.grid.display().to_string(),
        embedding_dim: cli.embedding_dim,
        variables: var_names.iter().map(|s| s.to_string()).collect(),
        nx: nx_val,
        ny: ny_val,
        n_radial_windows: results.len(),
        results,
        pedestal_a,
        core_a,
        edge_a,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}
