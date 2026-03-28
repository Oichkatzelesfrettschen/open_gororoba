//! BOUT++ CD analysis: read NetCDF4 output from BOUT++ simulations, extract
//! evolved field variables (n, phi, vort), build 32D Takens embeddings from
//! spatial profiles, and compute the Cayley-Dickson associator at each timestep.
//!
//! BOUT++ NetCDF4 files are HDF5 under the hood. Variables have shape
//! (t, x, y, z) where y=1 for 2D slab geometry.

use anyhow::{bail, Context, Result};
use clap::Parser;
use hdf5::File as H5File;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "bout-cd-analysis")]
struct Cli {
    /// Path to BOUT++ NetCDF4 dump file (BOUT.dmp.0.nc).
    #[arg(long)]
    input: PathBuf,

    /// Field variables to use as embedding channels (comma-separated).
    /// Default: "n,phi" (density + electrostatic potential).
    #[arg(long, default_value = "n,phi")]
    fields: String,

    /// Embedding dimension for CD associator.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Spatial axis to sweep for embedding: "x" or "z".
    #[arg(long, default_value = "x")]
    sweep_axis: String,

    /// Fixed index on the non-swept axis (default: midpoint).
    #[arg(long)]
    fixed_idx: Option<usize>,

    /// Normalization: "direction" (unit vectors) or "raw".
    #[arg(long, default_value = "direction")]
    normalization: String,

    /// Output JSON path.
    #[arg(long, default_value = "data/output/heliosphere/ablations/bout_cd_analysis.json")]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct TimestepResult {
    time: f64,
    step: usize,
    mean_associator: f64,
    max_associator: f64,
    std_associator: f64,
    n_windows: usize,
    field_rms: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct BoutCdOutput {
    input_file: String,
    fields: Vec<String>,
    embedding_dim: usize,
    sweep_axis: String,
    normalization: String,
    n_timesteps: usize,
    nx: usize,
    nz: usize,
    results: Vec<TimestepResult>,
    interpretation: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== BOUT++ CD Analysis ===");
    println!("  Input: {}", cli.input.display());

    let field_names: Vec<&str> = cli.fields.split(',').collect();
    let n_channels = field_names.len();
    let steps_per_window = cli.embedding_dim / n_channels;

    if steps_per_window * n_channels != cli.embedding_dim {
        println!(
            "  Note: embedding_dim {} not divisible by {} channels, padding {} dims with zeros",
            cli.embedding_dim,
            n_channels,
            cli.embedding_dim - steps_per_window * n_channels
        );
    }
    let remainder = cli.embedding_dim - steps_per_window * n_channels;

    let file = H5File::open(&cli.input)
        .with_context(|| format!("Failed to open {}", cli.input.display()))?;

    // Read time array
    let t_ds = file.dataset("t_array")?;
    let times: Vec<f64> = t_ds.read_raw()?;
    let n_times = times.len();

    // Read grid dimensions from first field
    let first_ds = file.dataset(field_names[0])
        .with_context(|| format!("Field '{}' not found in file", field_names[0]))?;
    let shape = first_ds.shape();
    if shape.len() != 4 {
        bail!("Expected 4D variable (t,x,y,z), got {}D", shape.len());
    }
    let (nt, nx, _ny, nz) = (shape[0], shape[1], shape[2], shape[3]);
    assert_eq!(nt, n_times, "Time dimension mismatch");

    println!("  Grid: {}x{}x{}, {} timesteps, {} channels {:?}",
        nx, shape[2], nz, n_times, n_channels, field_names);

    // Read all field data upfront (fits in memory for BOUT++ output)
    let mut field_data: Vec<Vec<f64>> = Vec::new();
    for &name in &field_names {
        let ds = file.dataset(name)
            .with_context(|| format!("Field '{}' not found", name))?;
        let data: Vec<f64> = ds.read_raw()?;
        field_data.push(data);
    }

    let sweep_len = match cli.sweep_axis.as_str() {
        "x" => nx,
        "z" => nz,
        _ => bail!("sweep_axis must be 'x' or 'z'"),
    };

    let fixed_max = match cli.sweep_axis.as_str() {
        "x" => nz,
        "z" => nx,
        _ => unreachable!(),
    };
    let fixed_idx = cli.fixed_idx.unwrap_or(fixed_max / 2);

    let n_windows = if sweep_len > steps_per_window {
        sweep_len - steps_per_window
    } else {
        bail!("Sweep axis length {} < steps_per_window {}", sweep_len, steps_per_window);
    };

    println!("  Sweep: {} axis, fixed_idx={}, windows={}", cli.sweep_axis, fixed_idx, n_windows);

    let mut results = Vec::with_capacity(n_times);

    for ti in 0..n_times {
        // Extract spatial profiles for each channel at this timestep
        let mut channels: Vec<Vec<f64>> = Vec::with_capacity(n_channels);
        let mut field_rms_vec = Vec::with_capacity(n_channels);

        for (ci, data) in field_data.iter().enumerate() {
            let mut profile = Vec::with_capacity(sweep_len);

            for si in 0..sweep_len {
                // Index into 4D array: data[ti * nx*ny*nz + xi * ny*nz + yi * nz + zi]
                // ny = shape[2] (usually 1 for 2D)
                let ny = shape[2];
                let idx = match cli.sweep_axis.as_str() {
                    "x" => ti * nx * ny * nz + si * ny * nz + 0 * nz + fixed_idx,
                    "z" => ti * nx * ny * nz + fixed_idx * ny * nz + 0 * nz + si,
                    _ => unreachable!(),
                };
                if idx < data.len() {
                    profile.push(data[idx]);
                } else {
                    profile.push(0.0);
                }
            }

            let rms = if profile.is_empty() {
                0.0
            } else {
                let mean = profile.iter().sum::<f64>() / profile.len() as f64;
                (profile.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / profile.len() as f64)
                    .sqrt()
            };
            field_rms_vec.push(rms);

            let _ = ci; // suppress unused warning
            channels.push(profile);
        }

        // Build Takens embedding: sliding window over the sweep axis
        let mut embedded: Vec<Vec<f64>> = Vec::with_capacity(n_windows);
        for w in 0..n_windows {
            let mut v = Vec::with_capacity(cli.embedding_dim);
            for s in 0..steps_per_window {
                for ch in &channels {
                    v.push(ch[w + s]);
                }
            }
            for _ in 0..remainder {
                v.push(0.0);
            }
            embedded.push(v);
        }

        // Normalize
        if cli.normalization == "direction" {
            for v in &mut embedded {
                let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-15 {
                    for x in v.iter_mut() {
                        *x /= norm;
                    }
                }
            }
        }

        // Compute CD associator norms (f32 dispatch for high dims)
        let precision = if cli.embedding_dim >= 128 { "f32" } else { "f64" };
        let norms = cd_kernel::batch_sliding_associator_norms_dispatch(
            &embedded, cli.embedding_dim, precision,
        );

        let (mean_a, max_a, std_a) = if norms.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let n_f = norms.len() as f64;
            let m = norms.iter().sum::<f64>() / n_f;
            let mx = norms.iter().cloned().fold(0.0f64, f64::max);
            let var = norms.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n_f;
            (m, mx, var.sqrt())
        };

        if ti % 10 == 0 || ti == n_times - 1 {
            println!(
                "  t={:8.1}: A_mean={:.6}, A_max={:.6}, A_std={:.6}, rms={:.4e}",
                times[ti], mean_a, max_a, std_a,
                field_rms_vec.first().copied().unwrap_or(0.0)
            );
        }

        results.push(TimestepResult {
            time: times[ti],
            step: ti,
            mean_associator: mean_a,
            max_associator: max_a,
            std_associator: std_a,
            n_windows: norms.len(),
            field_rms: field_rms_vec,
        });
    }

    // Summarize
    let initial_a = results.first().map(|r| r.mean_associator).unwrap_or(0.0);
    let final_a = results.last().map(|r| r.mean_associator).unwrap_or(0.0);
    let peak_a = results.iter().map(|r| r.mean_associator).fold(0.0f64, f64::max);
    let peak_time = results.iter()
        .max_by(|a, b| a.mean_associator.partial_cmp(&b.mean_associator).unwrap())
        .map(|r| r.time)
        .unwrap_or(0.0);

    let interp = format!(
        "BOUT++ {} channel(s) {:?}, {} timesteps. A(t=0)={:.6}, A(peak at t={:.1})={:.6}, A(final)={:.6}. Ratio peak/initial={:.1}x.",
        n_channels, field_names, n_times,
        initial_a, peak_time, peak_a, final_a,
        if initial_a > 1e-15 { peak_a / initial_a } else { f64::INFINITY }
    );
    println!("\n  {}", interp);

    let output = BoutCdOutput {
        input_file: cli.input.display().to_string(),
        fields: field_names.iter().map(|s| s.to_string()).collect(),
        embedding_dim: cli.embedding_dim,
        sweep_axis: cli.sweep_axis,
        normalization: cli.normalization,
        n_timesteps: n_times,
        nx,
        nz,
        results,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}
