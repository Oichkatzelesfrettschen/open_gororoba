//! GRMHD MRI CD analysis: extract B-field from nubhlight HDF5 dumps, compute
//! 32D Cayley-Dickson associator through MRI onset and saturation.
//!
//! Reads primitive variable array P[N1,N2,N3,8] where indices 5,6,7 are B1,B2,B3.
//! Extracts midplane (theta=pi/2) slice, builds sliding Takens embedding windows,
//! and computes associator norms at each dump time.

use anyhow::{Context, Result};
use clap::Parser;
use hdf5::File as H5File;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "grmhd-mri-cd")]
struct Cli {
    /// Directory containing nubhlight dump_*.h5 files.
    #[arg(long, default_value = "/var/tmp/nubhlight_builds/lowres_serial/dumps")]
    dump_dir: PathBuf,

    /// Embedding dimension for CD associator.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Theta index for midplane slice (N2/2 for equatorial).
    #[arg(long)]
    theta_idx: Option<usize>,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/grmhd_mri_cd.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct DumpResult {
    dump_id: usize,
    time: f64,
    n1: usize,
    n2: usize,
    n3: usize,
    midplane_cells: usize,
    mean_bmag: f64,
    max_bmag: f64,
    mean_associator: f64,
    max_associator: f64,
    n_embedding_windows: usize,
}

#[derive(Debug, Serialize)]
struct GrmhdOutput {
    n_dumps: usize,
    embedding_dim: usize,
    results: Vec<DumpResult>,
    mri_onset_dump: Option<usize>,
    mri_onset_time: Option<f64>,
    saturation_mean_a: f64,
    interpretation: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== GRMHD MRI CD Analysis ===");

    // Discover dump files
    let mut dump_paths: Vec<(usize, PathBuf)> = fs::read_dir(&cli.dump_dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("dump_") && name.ends_with(".h5") {
                let id: usize = name
                    .trim_start_matches("dump_")
                    .trim_end_matches(".h5")
                    .parse()
                    .ok()?;
                Some((id, e.path()))
            } else {
                None
            }
        })
        .collect();
    dump_paths.sort_by_key(|(id, _)| *id);

    println!("  Found {} dumps in {}", dump_paths.len(), cli.dump_dir.display());

    let channels = 3usize; // B1, B2, B3
    let steps = cli.embedding_dim / channels; // windows per embedding
    let remainder = cli.embedding_dim % channels;

    let mut results = Vec::new();

    for (dump_id, path) in &dump_paths {
        let file = H5File::open(path)
            .with_context(|| format!("Failed to open {}", path.display()))?;

        // Read simulation time
        let t_ds = file.dataset("t")?;
        let t_arr: Vec<f64> = t_ds.read_raw()?;
        let time = t_arr.first().copied().unwrap_or(0.0);

        // Read grid dimensions
        let n1: usize = {
            let ds = file.dataset("N1tot")?;
            let v: Vec<i32> = ds.read_raw()?;
            v[0] as usize
        };
        let n2: usize = {
            let ds = file.dataset("N2tot")?;
            let v: Vec<i32> = ds.read_raw()?;
            v[0] as usize
        };
        let n3: usize = {
            let ds = file.dataset("N3tot")?;
            let v: Vec<i32> = ds.read_raw()?;
            v[0] as usize
        };

        // Read P array: shape (N1, N2, N3, 8), dtype f32
        let p_ds = file.dataset("P")?;
        let p_flat: Vec<f32> = p_ds.read_raw()?;

        let theta_idx = cli.theta_idx.unwrap_or(n2 / 2);

        // Extract midplane B-field: iterate over (r, phi) at fixed theta
        // P layout: P[i1 * n2*n3*8 + i2 * n3*8 + i3 * 8 + var]
        let mut b1_slice = Vec::with_capacity(n1 * n3);
        let mut b2_slice = Vec::with_capacity(n1 * n3);
        let mut b3_slice = Vec::with_capacity(n1 * n3);

        for i1 in 0..n1 {
            for i3 in 0..n3 {
                let base = i1 * n2 * n3 * 8 + theta_idx * n3 * 8 + i3 * 8;
                b1_slice.push(p_flat[base + 5] as f64);
                b2_slice.push(p_flat[base + 6] as f64);
                b3_slice.push(p_flat[base + 7] as f64);
            }
        }

        let midplane_cells = b1_slice.len();

        // Compute |B| statistics
        let bmag: Vec<f64> = b1_slice
            .iter()
            .zip(b2_slice.iter())
            .zip(b3_slice.iter())
            .map(|((&b1, &b2), &b3)| (b1 * b1 + b2 * b2 + b3 * b3).sqrt())
            .collect();

        let mean_bmag = bmag.iter().sum::<f64>() / bmag.len().max(1) as f64;
        let max_bmag = bmag.iter().cloned().fold(0.0f64, f64::max);

        // Build Takens embedding: sliding window over the midplane cells
        // Each window = [B1(i), B2(i), B3(i), B1(i+1), B2(i+1), B3(i+1), ...]
        // for `steps` consecutive cells, yielding `embedding_dim`-dimensional vectors
        let n_windows = if midplane_cells > steps { midplane_cells - steps } else { 0 };

        if n_windows < 3 {
            println!("  dump {:05}: t={:.1}, too few cells for embedding", dump_id, time);
            continue;
        }

        let mut embedded: Vec<Vec<f64>> = Vec::with_capacity(n_windows);
        for w in 0..n_windows {
            let mut v = Vec::with_capacity(cli.embedding_dim);
            for s in 0..steps {
                let idx = w + s;
                v.push(b1_slice[idx]);
                v.push(b2_slice[idx]);
                v.push(b3_slice[idx]);
            }
            // Pad remainder dimensions with zeros if embedding_dim not divisible by 3
            for _ in 0..remainder {
                v.push(0.0);
            }
            embedded.push(v);
        }

        // Normalize each embedding vector by its L2 norm (direction-only)
        for v in &mut embedded {
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
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

        println!(
            "  dump {:05}: t={:6.1}, |B|_mean={:.4e}, |B|_max={:.4e}, A_mean={:.6}, A_max={:.6}, windows={}",
            dump_id, time, mean_bmag, max_bmag, mean_a, max_a, norms.len()
        );

        results.push(DumpResult {
            dump_id: *dump_id,
            time,
            n1,
            n2,
            n3,
            midplane_cells,
            mean_bmag,
            max_bmag,
            mean_associator: mean_a,
            max_associator: max_a,
            n_embedding_windows: norms.len(),
        });
    }

    // Detect MRI onset: first dump where mean_associator > 2x the initial value
    let initial_a = results.first().map(|r| r.mean_associator).unwrap_or(0.0);
    let mri_onset = results
        .iter()
        .find(|r| r.mean_associator > initial_a * 2.0 && r.dump_id > 0);

    let mri_onset_dump = mri_onset.map(|r| r.dump_id);
    let mri_onset_time = mri_onset.map(|r| r.time);

    // Saturation phase: last 5 dumps
    let sat_count = results.len().min(5);
    let saturation_mean_a = if sat_count > 0 {
        results[results.len() - sat_count..]
            .iter()
            .map(|r| r.mean_associator)
            .sum::<f64>()
            / sat_count as f64
    } else {
        0.0
    };

    let interp = format!(
        "Analyzed {} GRMHD dumps (t=0..{:.0}). Initial A={:.6}, saturation A={:.6} (ratio={:.2}x). {}",
        results.len(),
        results.last().map(|r| r.time).unwrap_or(0.0),
        initial_a,
        saturation_mean_a,
        if initial_a > 1e-15 { saturation_mean_a / initial_a } else { 0.0 },
        if let Some(t) = mri_onset_time {
            format!("MRI onset detected at t={:.1} (A doubles from initial). B-field topology transition tracked by CD associator.", t)
        } else {
            "No clear MRI onset detected (A did not double from initial).".to_string()
        }
    );

    println!("\n  {}", interp);

    let output = GrmhdOutput {
        n_dumps: results.len(),
        embedding_dim: cli.embedding_dim,
        results,
        mri_onset_dump,
        mri_onset_time,
        saturation_mean_a,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}
