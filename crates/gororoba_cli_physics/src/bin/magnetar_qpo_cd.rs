//! Magnetar QPO CD analysis: generate synthetic SGR 1806-20 giant flare
//! quasi-periodic oscillation time series matching published frequencies,
//! then compute 32D CD associator through the flare tail.
//!
//! Published QPO frequencies (Israel et al. 2005, Watts & Strohmayer 2006):
//! 18, 26, 30, 92, 150, 625, 1840 Hz
//! These are interpreted as crustal torsional oscillation modes of the
//! neutron star, modulating the X-ray flux during the giant flare tail.

use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::{f64::consts::PI, fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "magnetar-qpo-cd")]
struct Cli {
    /// Sample rate in Hz.
    #[arg(long, default_value_t = 8192.0)]
    sample_rate: f64,

    /// Duration of synthetic flare tail in seconds.
    #[arg(long, default_value_t = 300.0)]
    duration: f64,

    /// Embedding dimension.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/magnetar_qpo_cd.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct PhaseResult {
    phase: String,
    time_start: f64,
    time_end: f64,
    mean_a: f64,
    max_a: f64,
    n_active_modes: usize,
}

#[derive(Debug, Serialize)]
struct MagnetarOutput {
    sample_rate: f64,
    duration: f64,
    total_samples: usize,
    embedding_dim: usize,
    qpo_frequencies_hz: Vec<f64>,
    phases: Vec<PhaseResult>,
    interpretation: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== Magnetar QPO CD Analysis (synthetic SGR 1806-20) ===");

    let dt = 1.0 / cli.sample_rate;
    let n_samples = (cli.duration * cli.sample_rate) as usize;

    // Published QPO frequencies and their approximate onset times in the flare tail
    // (from Israel et al. 2005, Watts & Strohmayer 2006)
    let qpo_modes: Vec<(f64, f64, f64, f64)> = vec![
        // (frequency_hz, onset_s, decay_time_s, amplitude)
        (18.0, 0.0, 200.0, 1.0),     // fundamental crustal mode, present throughout
        (26.0, 10.0, 150.0, 0.6),    // l=2 overtone
        (30.0, 10.0, 120.0, 0.5),    // l=2 overtone
        (92.0, 20.0, 100.0, 0.8),    // l=7 or magnetic mode
        (150.0, 30.0, 80.0, 0.4),    // higher overtone
        (625.0, 50.0, 40.0, 0.3),    // high-frequency crustal mode
        (1840.0, 100.0, 20.0, 0.15), // highest observed, brief
    ];

    let freqs: Vec<f64> = qpo_modes.iter().map(|(f, _, _, _)| *f).collect();
    println!("  QPO frequencies: {:?} Hz", freqs);
    println!("  Duration: {}s, sample rate: {} Hz, samples: {}", cli.duration, cli.sample_rate, n_samples);

    // Generate synthetic X-ray flux time series
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut flux = vec![0.0f64; n_samples];

    for (i, flux_val) in flux.iter_mut().enumerate() {
        let t = i as f64 * dt;

        // Base flux: exponential decay (giant flare tail)
        let base = (-t / 300.0).exp();

        // Add QPO modes with time-dependent amplitudes
        let mut signal = base;
        for &(freq, onset, decay, amp) in &qpo_modes {
            if t > onset {
                let mode_amp = amp * (-(t - onset) / decay).exp();
                // Add phase jitter for realism
                let phase = 2.0 * PI * freq * t + rng.gen_range(0.0..1.0) * 0.1;
                signal += mode_amp * phase.sin();
            }
        }

        // Add Poisson-like noise (proportional to sqrt of flux)
        let noise = rng.gen_range(0.0..1.0) * 0.05 * base.sqrt();
        *flux_val = (signal + noise).max(0.0);
    }

    println!("  Generated synthetic flare tail with {} QPO modes", qpo_modes.len());

    // Analyze in phases: initial spike, QPO onset, multi-mode, decay, quiet
    let phases: Vec<(&str, f64, f64, usize)> = vec![
        ("initial_spike", 0.0, 10.0, 1),
        ("qpo_onset", 10.0, 30.0, 3),
        ("multi_mode", 30.0, 80.0, 6),
        ("high_freq", 80.0, 120.0, 7),
        ("decay", 120.0, 200.0, 4),
        ("quiet_tail", 200.0, 300.0, 2),
    ];

    let steps = cli.embedding_dim;
    let stride = 10;
    let mut results = Vec::new();

    for &(name, t_start, t_end, n_modes) in &phases {
        let i_start = (t_start * cli.sample_rate) as usize;
        let i_end = ((t_end * cli.sample_rate) as usize).min(n_samples);
        let local_n = i_end - i_start;

        if local_n < steps * stride + 3 {
            println!("  {}: too few samples, skipping", name);
            continue;
        }

        let n_windows = (local_n - steps * stride) / stride;
        let mut embedded: Vec<Vec<f64>> = Vec::with_capacity(n_windows);

        for w in 0..n_windows {
            let mut v = Vec::with_capacity(cli.embedding_dim);
            for s in 0..steps {
                let idx = i_start + w * stride + s * stride;
                if idx < n_samples {
                    v.push(flux[idx]);
                }
            }
            while v.len() < cli.embedding_dim {
                v.push(0.0);
            }
            v.truncate(cli.embedding_dim);

            // Direction normalize
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
            cd_kernel::batch_sliding_associator_norms_dispatch(&embedded, cli.embedding_dim, if cli.embedding_dim >= 128 { "f32" } else { "f64" });

        let mean_a = if norms.is_empty() {
            0.0
        } else {
            norms.iter().sum::<f64>() / norms.len() as f64
        };
        let max_a = norms.iter().cloned().fold(0.0f64, f64::max);

        println!(
            "  {}: t={:.0}-{:.0}s, modes={}, A_mean={:.6}, A_max={:.6}, windows={}",
            name, t_start, t_end, n_modes, mean_a, max_a, norms.len()
        );

        results.push(PhaseResult {
            phase: name.to_string(),
            time_start: t_start,
            time_end: t_end,
            mean_a,
            max_a,
            n_active_modes: n_modes,
        });
    }

    // Interpret
    let max_phase = results
        .iter()
        .max_by(|a, b| a.mean_a.partial_cmp(&b.mean_a).unwrap());
    let min_phase = results
        .iter()
        .min_by(|a, b| a.mean_a.partial_cmp(&b.mean_a).unwrap());

    let interp = format!(
        "Synthetic SGR 1806-20: {} QPO modes (18-1840 Hz). Peak A in '{}' phase (A={:.6}, {} modes active). Min A in '{}' phase (A={:.6}). Ratio={:.2}x.",
        qpo_modes.len(),
        max_phase.map(|p| p.phase.as_str()).unwrap_or("?"),
        max_phase.map(|p| p.mean_a).unwrap_or(0.0),
        max_phase.map(|p| p.n_active_modes).unwrap_or(0),
        min_phase.map(|p| p.phase.as_str()).unwrap_or("?"),
        min_phase.map(|p| p.mean_a).unwrap_or(0.0),
        if min_phase.map(|p| p.mean_a).unwrap_or(0.0) > 1e-15 {
            max_phase.map(|p| p.mean_a).unwrap_or(0.0) / min_phase.map(|p| p.mean_a).unwrap_or(1.0)
        } else { 0.0 }
    );
    println!("\n  {}", interp);

    let output = MagnetarOutput {
        sample_rate: cli.sample_rate,
        duration: cli.duration,
        total_samples: n_samples,
        embedding_dim: cli.embedding_dim,
        qpo_frequencies_hz: freqs,
        phases: results,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}
