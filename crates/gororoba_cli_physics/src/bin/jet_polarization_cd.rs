//! Relativistic jet polarization CD analysis: synthetic M87-like EVPA
//! (Electric Vector Position Angle) time series through jet knot ejection.
//!
//! The magnetic field structure in AGN jets transitions between:
//! 1. Helical (ordered): low CD associator (coherent phase geometry)
//! 2. Shock-compressed (tangled): high CD associator (disordered)
//! 3. Faraday-rotated: intermediate CD (systematic rotation)
//!
//! Stokes parameters: I (total intensity), Q, U (linear polarization), V (circular)
//! EVPA = 0.5 * atan2(U, Q)

use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::{f64::consts::PI, fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "jet-polarization-cd")]
struct Cli {
    /// Number of time samples.
    #[arg(long, default_value_t = 10000)]
    n_samples: usize,

    /// Embedding dimension.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Output JSON path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/jet_polarization_cd.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct PhaseResult {
    phase: String,
    mean_a: f64,
    max_a: f64,
    mean_pol_frac: f64,
}

#[derive(Debug, Serialize)]
struct JetOutput {
    n_samples: usize,
    embedding_dim: usize,
    phases: Vec<PhaseResult>,
    interpretation: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== Relativistic Jet Polarization CD Analysis ===");

    use rand::RngExt;
    let mut rng = rand::rng();

    // Generate synthetic Stokes parameters for 4 jet phases
    // Using M87-like parameters (EVPA ~-20 deg, pol fraction ~1-15%)
    type StokesGen = Box<dyn Fn(usize, &mut rand::rngs::ThreadRng) -> (f64, f64, f64, f64)>;
    let phases: Vec<(&str, StokesGen)> = vec![
        // Phase 1: Quiescent helical field (ordered, low variability)
        (
            "helical_quiescent",
            Box::new(|i, rng: &mut rand::rngs::ThreadRng| {
                let t = i as f64 / 1000.0;
                let evpa = -20.0_f64.to_radians() + 0.05 * (2.0 * PI * 0.1 * t).sin();
                let pol_frac = 0.03 + 0.005 * rng.random_range(0.0..1.0);
                let i_flux = 1.0 + 0.02 * rng.random_range(0.0..1.0);
                let q = i_flux * pol_frac * (2.0 * evpa).cos();
                let u = i_flux * pol_frac * (2.0 * evpa).sin();
                (i_flux, q, u, 0.001 * rng.random_range(-1.0..1.0))
            }),
        ),
        // Phase 2: Knot ejection (shock, EVPA rotation, high variability)
        (
            "shock_ejection",
            Box::new(|i, rng: &mut rand::rngs::ThreadRng| {
                let t = i as f64 / 1000.0;
                // EVPA rotates rapidly during shock passage
                let evpa =
                    -20.0_f64.to_radians() + PI * t / 5.0 + 0.3 * rng.random_range(-1.0..1.0);
                let pol_frac = 0.08 + 0.06 * rng.random_range(0.0..1.0); // higher, variable
                let i_flux =
                    2.0 + 0.5 * (2.0 * PI * 0.5 * t).sin() + 0.1 * rng.random_range(0.0..1.0);
                let q = i_flux * pol_frac * (2.0 * evpa).cos();
                let u = i_flux * pol_frac * (2.0 * evpa).sin();
                (i_flux, q, u, 0.01 * rng.random_range(-1.0..1.0))
            }),
        ),
        // Phase 3: Faraday rotation screen (systematic EVPA gradient)
        (
            "faraday_screen",
            Box::new(|i, rng: &mut rand::rngs::ThreadRng| {
                let t = i as f64 / 1000.0;
                // Linear EVPA rotation from Faraday effect (RM ~ 10^5 rad/m^2 at mm wavelengths)
                let evpa = -20.0_f64.to_radians() + 0.5 * t + 0.02 * rng.random_range(-1.0..1.0);
                let pol_frac = 0.05 + 0.01 * rng.random_range(0.0..1.0);
                let i_flux = 1.5 + 0.05 * rng.random_range(0.0..1.0);
                let q = i_flux * pol_frac * (2.0 * evpa).cos();
                let u = i_flux * pol_frac * (2.0 * evpa).sin();
                (i_flux, q, u, 0.005 * rng.random_range(-1.0..1.0))
            }),
        ),
        // Phase 4: Turbulent wake (tangled field, depolarized)
        (
            "turbulent_wake",
            Box::new(|i, rng: &mut rand::rngs::ThreadRng| {
                let _t = i as f64 / 1000.0;
                let evpa = rng.random_range(0.0..PI); // random EVPA
                let pol_frac = 0.01 + 0.005 * rng.random_range(0.0..1.0); // very low
                let i_flux = 0.8 + 0.3 * rng.random_range(0.0..1.0);
                let q = i_flux * pol_frac * (2.0 * evpa).cos();
                let u = i_flux * pol_frac * (2.0 * evpa).sin();
                let v = 0.02 * rng.random_range(-1.0..1.0); // circular polarization from turbulence
                (i_flux, q, u, v)
            }),
        ),
    ];

    let samples_per_phase = cli.n_samples / phases.len();
    let stride = 5;
    let steps = cli.embedding_dim / 4; // 4 Stokes channels
    let mut results = Vec::new();

    for (name, generator) in &phases {
        // Generate Stokes time series for this phase
        let mut stokes_i = Vec::with_capacity(samples_per_phase);
        let mut stokes_q = Vec::with_capacity(samples_per_phase);
        let mut stokes_u = Vec::with_capacity(samples_per_phase);
        let mut stokes_v = Vec::with_capacity(samples_per_phase);

        for i in 0..samples_per_phase {
            let (si, sq, su, sv) = generator(i, &mut rng);
            stokes_i.push(si);
            stokes_q.push(sq);
            stokes_u.push(su);
            stokes_v.push(sv);
        }

        let mean_pol = stokes_i
            .iter()
            .zip(stokes_q.iter())
            .zip(stokes_u.iter())
            .map(|((&si, &sq), &su)| (sq * sq + su * su).sqrt() / si.max(1e-15))
            .sum::<f64>()
            / samples_per_phase as f64;

        // Build 32D Takens embedding (4 Stokes channels x 8 time steps)
        let n_windows = if samples_per_phase > steps * stride {
            (samples_per_phase - steps * stride) / stride
        } else {
            0
        };

        let mut embedded: Vec<Vec<f64>> = Vec::with_capacity(n_windows);
        for w in 0..n_windows {
            let mut v = Vec::with_capacity(cli.embedding_dim);
            for s in 0..steps {
                let idx = w * stride + s * stride;
                if idx < samples_per_phase {
                    v.push(stokes_i[idx]);
                    v.push(stokes_q[idx]);
                    v.push(stokes_u[idx]);
                    v.push(stokes_v[idx]);
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

        if embedded.len() < 3 {
            continue;
        }

        let norms = cd_kernel::batch_sliding_associator_norms_dispatch(
            &embedded,
            cli.embedding_dim,
            if cli.embedding_dim >= 128 {
                "f32"
            } else {
                "f64"
            },
        );

        let mean_a = norms.iter().sum::<f64>() / norms.len().max(1) as f64;
        let max_a = norms.iter().cloned().fold(0.0f64, f64::max);

        println!(
            "  {}: A_mean={:.6}, A_max={:.6}, pol_frac={:.4}, windows={}",
            name,
            mean_a,
            max_a,
            mean_pol,
            norms.len()
        );

        results.push(PhaseResult {
            phase: name.to_string(),
            mean_a,
            max_a,
            mean_pol_frac: mean_pol,
        });
    }

    let interp = format!(
        "Synthetic M87-like jet: 4 phases. Helical A={:.4}, Shock A={:.4}, Faraday A={:.4}, Turbulent A={:.4}. Shock/Helical ratio={:.1}x.",
        results.first().map(|r| r.mean_a).unwrap_or(0.0),
        results.get(1).map(|r| r.mean_a).unwrap_or(0.0),
        results.get(2).map(|r| r.mean_a).unwrap_or(0.0),
        results.get(3).map(|r| r.mean_a).unwrap_or(0.0),
        if results.first().map(|r| r.mean_a).unwrap_or(0.0) > 1e-15 {
            results.get(1).map(|r| r.mean_a).unwrap_or(0.0)
                / results.first().map(|r| r.mean_a).unwrap_or(1.0)
        } else {
            0.0
        }
    );
    println!("\n  {}", interp);

    let output = JetOutput {
        n_samples: cli.n_samples,
        embedding_dim: cli.embedding_dim,
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
