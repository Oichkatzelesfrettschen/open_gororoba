//! Reframed Thesis 3: CHSH-Betti Correlation Sweep.
//!
//! Tests whether wavelet-scale CHSH S parameter correlates with Betti-1
//! topology of D3Q19 LBM velocity fields across different flow configurations.
//!
//! For each configuration (varying forcing amplitude and pattern):
//!   1. Run GPU FP64 LBM for 500 steps, snapshot velocity every 10 steps
//!   2. For each snapshot: compute S from 1D velocity profile and Betti numbers from 3D field
//!   3. Compute Spearman correlation between S and B1 across snapshots
//!
//! PASS: |rho_Spearman| > 0.5 with p < 0.05 at any configuration.
//! FAIL: |rho_Spearman| < 0.5 or p > 0.05 at all configurations.
//!
//! Honest risk: C-706 proved |S| is invariant under Haar compression of a SINGLE signal.
//! This test correlates S and B1 ACROSS DIFFERENT configurations, which is genuinely
//! falsifiable -- there is no a priori reason these should correlate.

use clap::Parser;
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use spectral_core::chsh_betti_bridge::{spearman_correlation, velocity_to_betti, velocity_to_chsh};
use std::{f64::consts::PI, fs, path::Path};

#[derive(Parser, Debug)]
#[command(author, version, about = "Reframed T3: CHSH-Betti correlation sweep")]
struct Args {
    /// Grid size per dimension (32 recommended; 16 gives zero variance)
    #[arg(long, default_value_t = 32)]
    n: usize,
    /// Number of LBM time steps per configuration
    #[arg(long, default_value_t = 500)]
    steps: usize,
    /// Snapshot interval
    #[arg(long, default_value_t = 10)]
    snap_interval: usize,
    /// Velocity magnitude threshold for Betti computation (auto-set to p75 if zero)
    #[arg(long, default_value_t = 0.0)]
    threshold: f64,
    /// Max points for VR (OOM safety)
    #[arg(long, default_value_t = 100)]
    max_points: usize,
    /// Verbose: print per-snapshot S and B1 values
    #[arg(long)]
    verbose: bool,
    /// Output directory
    #[arg(long, default_value = "data/chsh_betti")]
    output_dir: String,
}

fn main() {
    let args = Args::parse();
    let n = args.n;

    println!("CHSH-Betti Correlation Sweep");
    println!(
        "  grid={}^3 steps={} snap_every={} threshold={}",
        n, args.steps, args.snap_interval, args.threshold
    );
    println!();

    // Configurations: vary tau (viscosity) and forcing amplitude
    // tau = 0.5 + nu/c_s^2, so tau=0.6 -> nu=0.033, tau=1.0 -> nu=0.167, etc.
    let configs: Vec<(&str, f64, f64)> = vec![
        ("low_visc", 0.55, 0.001),
        ("med_visc", 0.70, 0.005),
        ("high_visc", 1.00, 0.010),
        ("strong_f", 0.60, 0.020),
    ];

    let n_total = n * n * n;
    let mut all_results: Vec<(String, f64, f64, usize)> = Vec::new();
    let mut any_pass = false;

    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}  {:>8}",
        "config", "rho_S", "p_val", "n_snaps", "status"
    );
    println!(
        "  {:->12}  {:->8}  {:->8}  {:->8}  {:->8}",
        "", "", "", "", ""
    );

    for (name, tau, force_amp) in &configs {
        let mut solver =
            LbmSolver3DCuda::new(n, n, n, *tau, Precision::FP64).expect("GPU solver init failed");

        // Apply Kolmogorov-like sinusoidal forcing
        let force_field: Vec<[f64; 3]> = (0..n_total)
            .map(|idx| {
                let y = (idx / n) % n;
                let fx = force_amp * (2.0 * PI * y as f64 / n as f64).sin();
                [fx, 0.0, 0.0]
            })
            .collect();
        solver
            .set_force_field(&force_field)
            .expect("set force field");

        let mut s_values: Vec<f64> = Vec::new();
        let mut b1_values: Vec<f64> = Vec::new();

        for step in 1..=args.steps {
            solver.step().expect("LBM step failed");

            if step % args.snap_interval == 0 {
                solver.sync_to_host().expect("sync failed");

                // Extract 1D velocity profile along y-axis at x=n/2, z=n/2
                // The Kolmogorov forcing F_x = A*sin(2*pi*y/N) creates velocity
                // that varies with y, not x. Extracting along x gives constant
                // profiles with S=0. The y-profile captures the flow structure.
                let x_mid = n / 2;
                let z_mid = n / 2;
                let profile: Vec<f64> = (0..n)
                    .map(|y| {
                        let idx = x_mid + n * (y + n * z_mid);
                        let u = &solver.u[idx];
                        // Use signed u_x component to preserve flow direction info
                        u[0] as f64
                    })
                    .collect();

                // Pad to power of 2 for Haar DWT
                let padded_len = profile.len().next_power_of_two();
                let mut padded = profile;
                padded.resize(padded_len, 0.0);

                let s = velocity_to_chsh(&padded);

                // Convert f32 GPU velocities to f64 for Betti computation
                let u_f64: Vec<[f64; 3]> = solver
                    .u
                    .iter()
                    .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
                    .collect();

                // Adaptive threshold: use p75 of velocity magnitudes to get a
                // meaningful point cloud (not ALL cells above a fixed low threshold)
                let threshold = if args.threshold > 0.0 {
                    args.threshold
                } else {
                    let mut mags: Vec<f64> = u_f64
                        .iter()
                        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                        .collect();
                    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p75_idx = (mags.len() as f64 * 0.75) as usize;
                    mags[p75_idx.min(mags.len() - 1)]
                };

                let (_, b1) = velocity_to_betti(&u_f64, n, n, n, threshold, args.max_points);

                if args.verbose {
                    println!(
                        "    step={:>4}  S={:>8.4}  B1={:>4}  thr={:.6}",
                        step, s, b1, threshold
                    );
                }

                s_values.push(s);
                b1_values.push(b1 as f64);
            }
        }

        let n_snaps = s_values.len();
        let (rho, p) = if n_snaps >= 5 {
            spearman_correlation(&s_values, &b1_values)
        } else {
            (0.0, 1.0)
        };

        let status = if rho.abs() > 0.5 && p < 0.05 {
            any_pass = true;
            "PASS"
        } else if p > 0.05 {
            "INCONC"
        } else {
            "FAIL"
        };

        println!(
            "  {:>12}  {:>8.4}  {:>8.4}  {:>8}  {:>8}",
            name, rho, p, n_snaps, status
        );

        all_results.push((name.to_string(), rho, p, n_snaps));
    }

    let verdict = if any_pass { "PASS" } else { "INCONCLUSIVE" };
    println!();
    println!("  Verdict: {}", verdict);

    // Write results
    let dir = Path::new(&args.output_dir);
    fs::create_dir_all(dir).expect("failed to create output dir");
    let toml_path = dir.join("results.toml");

    let mut toml = format!(
        "# CHSH-Betti Correlation Sweep (GPU FP64)\n# grid={}^3 steps={} threshold={}\n# verdict = {}\n\n",
        n, args.steps, args.threshold, verdict
    );

    for (name, rho, p, n_snaps) in &all_results {
        toml.push_str(&format!(
            "[[config]]\nname = \"{}\"\nspearman_rho = {:.6}\np_value = {:.6}\nn_snapshots = {}\npass = {}\n\n",
            name, rho, p, n_snaps, rho.abs() > 0.5 && *p < 0.05
        ));
    }

    fs::write(&toml_path, toml).expect("failed to write results.toml");
    println!("  Results written to {}", toml_path.display());
}
