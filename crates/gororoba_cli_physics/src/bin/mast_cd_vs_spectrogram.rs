//! MAST CD vs Spectrogram comparison: compute both the 32D CD associator
//! AND the FFT power spectrum on real MAST Mirnov data, then compare
//! what each diagnostic reveals about discharge evolution.
//!
//! This is the key comparison: CD associator (novel) vs spectral analysis
//! (established) on the SAME experimental tokamak data.

use anyhow::{Context, Result};
use clap::Parser;
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Serialize;
use std::{fs, path::PathBuf, sync::Arc};
use zarrs::array::Array;
use zarrs_filesystem::FilesystemStore;

#[derive(Parser)]
#[command(name = "mast-cd-vs-spectrogram")]
struct Cli {
    /// Zarr directory for MAST shot ama diagnostic.
    #[arg(long)]
    zarr_dir: PathBuf,

    /// Embedding dimension for CD.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// FFT window size.
    #[arg(long, default_value_t = 4096)]
    fft_size: usize,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/mast_cd_vs_spectrogram.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct WindowResult {
    center_time: f64,
    cd_mean: f64,
    cd_max: f64,
    spectral_peak_freq_hz: f64,
    spectral_peak_power: f64,
    spectral_total_power: f64,
    spectral_bandwidth_hz: f64,
}

#[derive(Debug, Serialize)]
struct ComparisonOutput {
    n_samples: usize,
    n_windows: usize,
    embedding_dim: usize,
    fft_size: usize,
    results: Vec<WindowResult>,
    cd_spectral_correlation: f64,
    interpretation: String,
}

fn read_zarr_f32(store: &Arc<FilesystemStore>, name: &str) -> Result<Vec<f32>> {
    let array = Array::open(store.clone(), &format!("/{name}"))
        .with_context(|| format!("Failed to open '{name}'"))?;
    let shape = array.shape().to_vec();
    let subset = zarrs::array::ArraySubset::new_with_shape(shape);
    let flat: Vec<f32> = array
        .retrieve_array_subset::<Vec<f32>>(&subset)
        .with_context(|| format!("Failed to read '{name}'"))?;
    Ok(flat)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== MAST CD vs Spectrogram Comparison ===");

    let store = Arc::new(
        FilesystemStore::new(&cli.zarr_dir)
            .map_err(|e| anyhow::anyhow!("Store: {:?}", e))?,
    );

    let time_f32 = read_zarr_f32(&store, "time")?;
    let sig_n2_f32 = read_zarr_f32(&store, "n=2_signal")?;
    let sig_nodd_f32 = read_zarr_f32(&store, "n=odd_signal")?;

    let n = time_f32.len();
    let dt = if n > 1 {
        (time_f32[1] - time_f32[0]) as f64
    } else {
        1.0 / 500000.0
    };
    let sample_rate = 1.0 / dt;

    println!("  Loaded {} samples, dt={:.2e}s, fs={:.0} Hz", n, dt, sample_rate);

    let time: Vec<f64> = time_f32.iter().map(|&x| x as f64).collect();
    let sig_n2: Vec<f64> = sig_n2_f32.iter().map(|&x| x as f64).collect();
    let sig_nodd: Vec<f64> = sig_nodd_f32.iter().map(|&x| x as f64).collect();

    // Process in windows for both CD and FFT
    let window_size = 50000usize;
    let window_stride = 25000usize;
    let n_windows = if n > window_size { (n - window_size) / window_stride + 1 } else { 1 };

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(cli.fft_size);

    let cd_stride = 50;
    let cd_channels = 2;
    let cd_steps = cli.embedding_dim / cd_channels;

    let mut results = Vec::with_capacity(n_windows);
    let mut cd_vals = Vec::new();
    let mut spectral_vals = Vec::new();

    for w in 0..n_windows {
        let start = w * window_stride;
        let end = (start + window_size).min(n);
        let center_time = time[start + (end - start) / 2];

        // --- CD Analysis ---
        let local_n = end - start;
        let n_embed = if local_n > cd_steps * cd_stride {
            (local_n - cd_steps * cd_stride) / cd_stride
        } else {
            0
        };

        let mut embedded: Vec<Vec<f64>> = Vec::with_capacity(n_embed);
        for e in 0..n_embed {
            let base = start + e * cd_stride;
            let mut v = Vec::with_capacity(cli.embedding_dim);
            for s in 0..cd_steps {
                let idx = base + s * cd_stride;
                if idx < n {
                    v.push(sig_n2[idx]);
                    v.push(sig_nodd[idx]);
                }
            }
            while v.len() < cli.embedding_dim {
                v.push(0.0);
            }
            v.truncate(cli.embedding_dim);

            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
            embedded.push(v);
        }

        let (cd_mean, cd_max) = if embedded.len() >= 3 {
            let norms = cd_kernel::batch_sliding_associator_norms_parallel(
                &embedded,
                cli.embedding_dim,
            );
            let m = norms.iter().sum::<f64>() / norms.len().max(1) as f64;
            let mx = norms.iter().cloned().fold(0.0f64, f64::max);
            (m, mx)
        } else {
            (0.0, 0.0)
        };

        // --- FFT Spectral Analysis ---
        // Use n=2 signal for FFT (standard Mirnov spectral analysis)
        let fft_start = start;
        let fft_end = (fft_start + cli.fft_size).min(end);
        let fft_n = fft_end - fft_start;

        let mut buffer: Vec<Complex<f64>> = (0..cli.fft_size)
            .map(|i| {
                if i < fft_n {
                    Complex::new(sig_n2[fft_start + i], 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                }
            })
            .collect();

        fft.process(&mut buffer);

        // Power spectrum (first half = positive frequencies)
        let n_freq = cli.fft_size / 2;
        let df = sample_rate / cli.fft_size as f64;
        let power: Vec<f64> = buffer[..n_freq]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) / cli.fft_size as f64)
            .collect();

        let total_power: f64 = power.iter().sum();
        let peak_idx = power
            .iter()
            .enumerate()
            .skip(1) // skip DC
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);
        let peak_freq = peak_idx as f64 * df;
        let peak_power = power[peak_idx];

        // Spectral bandwidth (width at half-maximum)
        let half_max = peak_power / 2.0;
        let mut bw_count = 0;
        for &p in &power {
            if p > half_max {
                bw_count += 1;
            }
        }
        let bandwidth = bw_count as f64 * df;

        if w % 5 == 0 || w == n_windows - 1 {
            println!(
                "  w{:2}/{}: t={:.3}s CD={:.4} FFT_peak={:.0}Hz pwr={:.2e} bw={:.0}Hz",
                w + 1, n_windows, center_time, cd_mean, peak_freq, peak_power, bandwidth
            );
        }

        cd_vals.push(cd_mean);
        spectral_vals.push(total_power);

        results.push(WindowResult {
            center_time,
            cd_mean,
            cd_max,
            spectral_peak_freq_hz: peak_freq,
            spectral_peak_power: peak_power,
            spectral_total_power: total_power,
            spectral_bandwidth_hz: bandwidth,
        });
    }

    // Compute Pearson correlation between CD and spectral total power
    let n_w = cd_vals.len() as f64;
    let cd_mean_all = cd_vals.iter().sum::<f64>() / n_w;
    let sp_mean_all = spectral_vals.iter().sum::<f64>() / n_w;
    let mut cov = 0.0;
    let mut var_cd = 0.0;
    let mut var_sp = 0.0;
    for i in 0..cd_vals.len() {
        let dc = cd_vals[i] - cd_mean_all;
        let ds = spectral_vals[i] - sp_mean_all;
        cov += dc * ds;
        var_cd += dc * dc;
        var_sp += ds * ds;
    }
    let correlation = if var_cd > 1e-30 && var_sp > 1e-30 {
        cov / (var_cd * var_sp).sqrt()
    } else {
        0.0
    };

    let interp = format!(
        "CD vs Spectrogram on {} windows. CD-spectral_power correlation: r={:.4}. {}",
        results.len(),
        correlation,
        if correlation.abs() > 0.7 {
            "Strong correlation: CD tracks spectral power changes."
        } else if correlation.abs() > 0.3 {
            "Moderate correlation: CD captures some spectral features but adds independent info."
        } else {
            "Weak correlation: CD measures DIFFERENT information than spectral analysis."
        }
    );
    println!("\n  {}", interp);

    let output = ComparisonOutput {
        n_samples: n,
        n_windows: results.len(),
        embedding_dim: cli.embedding_dim,
        fft_size: cli.fft_size,
        results,
        cd_spectral_correlation: correlation,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}
