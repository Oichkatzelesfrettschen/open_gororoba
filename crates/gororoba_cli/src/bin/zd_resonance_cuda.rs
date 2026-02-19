// CUDA BF16 Zero-Divisor Resonance Experiments (Sprint 49B)
//
// Multi-subcommand binary for systematic ZD resonance detection at 128^3/256^3.
//
// WHY: CPU experiments at 32^3 (Sprint 49A) were resolution-limited. Real-data
// cross-catalog analysis shows Ghost (phi^{-1/2}) at SNR>4 in L-band radio
// catalogs. GPU experiments at 128^3+ test whether LBM + Sedenion ZD modulation
// can reproduce this spectral signature.
//
// OBSERVABLE: Kinetic energy KE = sum(rho * |u|^2) / N is the primary time-series
// observable. Unlike mean density (which is a conservation law in LBM with periodic
// BCs), KE evolves non-trivially under forcing + viscous dissipation + ZD modulation.
// Secondary: density variance var(rho) and spatial power spectrum of final rho field.
//
// ALL experiments use Kolmogorov sinusoidal forcing F_x = A * sin(2*pi*k*z/N) so
// that the flow develops non-trivial structure. Without forcing, the flow decays
// monotonically to rest and there is nothing to analyze spectrally.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use spectral_core::ghost_spectral::{
    check_ghost, compute_power_spectrum, find_peaks, noise_floor, peak_fwhm, peak_snr,
    ALIASED_GHOST_FREQ,
};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use vacuum_frustration::bridge::{FrustrationViscosityBridge, SedenionField, ViscosityCouplingModel};

/// CUDA BF16 ZD Resonance Experiments
#[derive(Parser, Debug)]
#[command(author, version, about = "CUDA BF16 Zero-Divisor Resonance Experiments")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Tau sweep WITH Sedenion ZD modulation
    Sweep {
        /// Grid resolution (N for NxNxN)
        #[arg(long, default_value_t = 128)]
        res: usize,
        /// Number of steps per tau value
        #[arg(long, default_value_t = 2000)]
        steps: usize,
        /// Use FP32 instead of BF16
        #[arg(long)]
        fp32: bool,
        /// Output directory
        #[arg(long, default_value = "data/csv")]
        out_dir: PathBuf,
    },
    /// Control sweep WITHOUT ZD modulation (constant tau)
    Control {
        #[arg(long, default_value_t = 128)]
        res: usize,
        #[arg(long, default_value_t = 2000)]
        steps: usize,
        #[arg(long)]
        fp32: bool,
        #[arg(long, default_value = "data/csv")]
        out_dir: PathBuf,
    },
    /// Fix tau=0.55, sweep Kubo coupling strength
    CouplingSweep {
        #[arg(long, default_value_t = 128)]
        res: usize,
        #[arg(long, default_value_t = 2000)]
        steps: usize,
        /// Use FP32 instead of BF16
        #[arg(long)]
        fp32: bool,
        #[arg(long, default_value = "data/csv")]
        out_dir: PathBuf,
    },
    /// Reynolds number sweep: vary tau and forcing amplitude
    ReynoldsSweep {
        #[arg(long, default_value_t = 128)]
        res: usize,
        #[arg(long, default_value_t = 2000)]
        steps: usize,
        /// Use FP32 instead of BF16
        #[arg(long)]
        fp32: bool,
        #[arg(long, default_value = "data/csv")]
        out_dir: PathBuf,
    },
    /// Read sweep CSVs and emit verdict for C-774/C-775
    Analyze {
        #[arg(long, default_value = "data/csv")]
        data_dir: PathBuf,
    },
}

// Tau values for the primary sweep
const TAU_VALUES: [f64; 10] = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00];

// Coupling strengths for the coupling sweep (lambda in Kubo model)
const COUPLING_VALUES: [f64; 8] = [0.0, 0.1, 0.2, 0.3, 0.375, 0.5, 0.75, 1.0];

// Reynolds sweep: tau values (low tau = high Re)
const RE_TAU_VALUES: [f64; 7] = [0.501, 0.505, 0.51, 0.52, 0.55, 0.60, 0.70];

// Forcing amplitudes for Reynolds sweep
const FORCE_AMPS: [f64; 4] = [0.001, 0.01, 0.1, 0.5];

// Default forcing amplitude for sweep/control/coupling (drives non-trivial flow)
const DEFAULT_FORCE_AMP: f64 = 0.001;

// Sampling interval: sync every N steps (balance resolution vs PCIe cost)
const SAMPLE_INTERVAL: usize = 20;

/// Generate sinusoidal Sedenion field (same pattern as zd_resonance_bf16.rs)
fn generate_sedenion_field(nx: usize, ny: usize, nz: usize, wavelength: f64) -> SedenionField {
    let mut field = SedenionField::uniform(nx, ny, nz);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let s = field.get_mut(x, y, z);
                let px = x as f64 / nx as f64 - 0.5;
                let py = y as f64 / ny as f64 - 0.5;
                let pz = z as f64 / nz as f64 - 0.5;
                for (i, si) in s.iter_mut().enumerate() {
                    let phase = i as f64 * std::f64::consts::FRAC_PI_8;
                    let kx = phase.sin();
                    let ky = phase.cos();
                    let kz = (phase * 1.3).sin();
                    let dot = px * kx + py * ky + pz * kz;
                    *si = (dot * wavelength).sin();
                }
            }
        }
    }
    field
}

/// Compute viscosity field from Sedenion frustration + coupling model.
fn compute_viscosity_field(
    nx: usize,
    ny: usize,
    nz: usize,
    model: &ViscosityCouplingModel,
    wavelength: f64,
) -> Vec<f64> {
    let bridge = FrustrationViscosityBridge::new(16);
    let field = generate_sedenion_field(nx, ny, nz, wavelength);
    let frustration = field.local_frustration_density_par(16);
    let mut viscosity = bridge.frustration_to_viscosity_model(&frustration, model);
    for nu in &mut viscosity {
        *nu = nu.clamp(0.001, 0.5);
    }
    viscosity
}

/// Apply Kolmogorov sinusoidal forcing: F_x = A * sin(2*pi*k*z/N)
fn apply_kolmogorov_forcing(
    solver: &mut LbmSolver3DCuda,
    nx: usize,
    force_amp: f64,
) -> Result<()> {
    let n_cells = nx * nx * nx;
    let mut force_field = vec![[0.0_f64; 3]; n_cells];
    let k_force = 2.0;
    for z in 0..nx {
        let fz =
            force_amp * (2.0 * std::f64::consts::PI * k_force * z as f64 / nx as f64).sin();
        for y in 0..nx {
            for x in 0..nx {
                let idx = x + nx * (y + nx * z);
                force_field[idx][0] = fz;
            }
        }
    }
    solver
        .set_force_field(&force_field)
        .context("Failed to upload force field")?;
    Ok(())
}

/// Compute physical observables from host-side rho[] and u[] arrays.
fn compute_observables(rho: &[f32], u: &[[f32; 3]]) -> (f64, f64, f64) {
    let n = rho.len() as f64;
    let mut ke_sum = 0.0_f64;
    let mut rho_sum = 0.0_f64;
    let mut rho_sq_sum = 0.0_f64;

    for (i, &r) in rho.iter().enumerate() {
        let r64 = r as f64;
        rho_sum += r64;
        rho_sq_sum += r64 * r64;
        let ux = u[i][0] as f64;
        let uy = u[i][1] as f64;
        let uz = u[i][2] as f64;
        ke_sum += r64 * (ux * ux + uy * uy + uz * uz);
    }

    let mean_rho = rho_sum / n;
    let var_rho = rho_sq_sum / n - mean_rho * mean_rho;
    let ke = ke_sum / n;

    (ke, var_rho, mean_rho)
}

/// Compute isotropic (shell-averaged) spatial power spectrum of rho field.
/// Returns (k_bins, power) where k ranges from 1 to nx/2.
fn spatial_power_spectrum(rho: &[f32], nx: usize) -> (Vec<f64>, Vec<f64>) {
    let n = nx;
    let half = n / 2;

    // Shell-averaged: accumulate |rho_hat(k)|^2 in radial bins
    let mut shell_power = vec![0.0_f64; half + 1];
    let mut shell_count = vec![0_u64; half + 1];

    // 1D FFTs along each axis, then combine (cheaper than full 3D FFT for shell averaging)
    // For a proper diagnostic, compute the 3D power spectrum via sliced 1D FFTs
    // along z-axis for each (x,y), then shell-average in k-space.
    //
    // Simplified approach: compute power spectrum of rho along z-axis at each (x,y),
    // accumulate into radial bins. This captures the z-structure from Kolmogorov forcing.
    for y in 0..n {
        for x in 0..n {
            let mut z_slice = Vec::with_capacity(n);
            for z in 0..n {
                let idx = x + n * (y + n * z);
                z_slice.push(rho[idx] as f64);
            }
            let (freqs, power) = compute_power_spectrum(&z_slice);
            for (i, (&f, &p)) in freqs.iter().zip(power.iter()).enumerate() {
                if i == 0 {
                    continue; // skip DC
                }
                let k_bin = (f * n as f64).round() as usize;
                if k_bin > 0 && k_bin <= half {
                    shell_power[k_bin] += p;
                    shell_count[k_bin] += 1;
                }
            }
        }
    }

    let mut k_bins = Vec::new();
    let mut power = Vec::new();
    for k in 1..=half {
        if shell_count[k] > 0 {
            k_bins.push(k as f64 / n as f64);
            power.push(shell_power[k] / shell_count[k] as f64);
        }
    }

    (k_bins, power)
}

/// Run a single experiment: init solver, apply viscosity + forcing, run steps.
/// Returns ExperimentData with multiple time-series observables.
fn run_experiment(
    nx: usize,
    tau: f64,
    steps: usize,
    precision: Precision,
    viscosity: Option<&[f64]>,
    force_amp: f64,
    label: &str,
) -> Result<ExperimentData> {
    let mut solver =
        LbmSolver3DCuda::new(nx, nx, nx, tau, precision).context("Failed to init CUDA solver")?;

    if let Some(visc) = viscosity {
        solver
            .set_viscosity_field(visc)
            .context("Failed to upload viscosity")?;
    }

    apply_kolmogorov_forcing(&mut solver, nx, force_amp)?;

    // Small perturbation to break symmetry
    solver.initialize_uniform(1.0, [0.01, 0.0, 0.0])?;

    let n_samples = steps / SAMPLE_INTERVAL + 1;
    let mut ke_trace = Vec::with_capacity(n_samples);
    let mut var_trace = Vec::with_capacity(n_samples);
    let mut rho_mean_trace = Vec::with_capacity(n_samples);
    let start = Instant::now();

    for step in 0..steps {
        solver.step()?;

        if step % SAMPLE_INTERVAL == 0 {
            solver.sync_to_host()?;
            let (ke, var_rho, mean_rho) = compute_observables(&solver.rho, &solver.u);
            ke_trace.push(ke);
            var_trace.push(var_rho);
            rho_mean_trace.push(mean_rho);
        }

        if step % 500 == 0 {
            let ke_str = if let Some(&last_ke) = ke_trace.last() {
                format!(" KE={last_ke:.6e}")
            } else {
                String::new()
            };
            print!(
                "\r  [{label}] Step {step}/{steps} ({:.1}%){ke_str}",
                step as f64 / steps as f64 * 100.0
            );
            std::io::stdout().flush()?;
        }
    }

    // Final sync for spatial analysis
    solver.sync_to_host()?;
    let (final_ke, final_var, final_rho) = compute_observables(&solver.rho, &solver.u);

    println!(
        "\r  [{label}] Complete in {:.2?} | KE={final_ke:.6e} var={final_var:.6e} rho={final_rho:.6}",
        start.elapsed()
    );

    // Spatial power spectrum of final rho field
    let (spatial_k, spatial_power) = spatial_power_spectrum(&solver.rho, nx);

    Ok(ExperimentData {
        ke_trace,
        var_trace,
        rho_mean_trace,
        spatial_k,
        spatial_power,
    })
}

struct ExperimentData {
    ke_trace: Vec<f64>,
    var_trace: Vec<f64>,
    #[allow(dead_code)] // retained for diagnostics / future CSV dump
    rho_mean_trace: Vec<f64>,
    spatial_k: Vec<f64>,
    spatial_power: Vec<f64>,
}

/// Analyze time series: compute FFT, find ghost peak, return row data.
/// Analyzes KINETIC ENERGY trace (primary) and density variance (secondary).
fn analyze_traces(data: &ExperimentData) -> TraceResult {
    // Primary: KE trace
    let ke_result = analyze_single_trace(&data.ke_trace, "KE");

    // Secondary: variance trace
    let var_result = analyze_single_trace(&data.var_trace, "var");

    // Spatial: check for ghost in spatial power spectrum
    let spatial_ghost = if data.spatial_k.len() >= 10 {
        let peaks = find_peaks(&data.spatial_k, &data.spatial_power, 10);
        let ghost = check_ghost(&peaks);
        if let Some(gp) = ghost {
            let nf = noise_floor(&data.spatial_power);
            let snr = peak_snr(gp, nf);
            Some((gp.freq, snr))
        } else {
            None
        }
    } else {
        None
    };

    TraceResult {
        ke_ghost_detected: ke_result.ghost_detected,
        ke_ghost_freq: ke_result.ghost_freq,
        ke_ghost_snr: ke_result.ghost_snr,
        ke_ghost_fwhm: ke_result.ghost_fwhm,
        ke_dominant_freq: ke_result.dominant_freq,
        ke_dominant_power: ke_result.dominant_power,
        var_ghost_detected: var_result.ghost_detected,
        var_ghost_freq: var_result.ghost_freq,
        var_ghost_snr: var_result.ghost_snr,
        spatial_ghost_detected: spatial_ghost.is_some(),
        spatial_ghost_freq: spatial_ghost.map_or(f64::NAN, |(f, _)| f),
        spatial_ghost_snr: spatial_ghost.map_or(0.0, |(_, s)| s),
        final_ke: *data.ke_trace.last().unwrap_or(&0.0),
        final_var: *data.var_trace.last().unwrap_or(&0.0),
    }
}

struct SingleTraceResult {
    ghost_detected: bool,
    ghost_freq: f64,
    ghost_snr: f64,
    ghost_fwhm: f64,
    dominant_freq: f64,
    dominant_power: f64,
}

fn analyze_single_trace(trace: &[f64], _name: &str) -> SingleTraceResult {
    if trace.len() < 32 {
        return SingleTraceResult {
            ghost_detected: false,
            ghost_freq: f64::NAN,
            ghost_snr: 0.0,
            ghost_fwhm: f64::NAN,
            dominant_freq: f64::NAN,
            dominant_power: f64::NAN,
        };
    }
    let (freqs, power) = compute_power_spectrum(trace);
    let peaks = find_peaks(&freqs, &power, 10);
    let ghost = check_ghost(&peaks);
    let nf = noise_floor(&power);

    if let Some(gp) = ghost {
        let snr = peak_snr(gp, nf);
        let fwhm = peak_fwhm(&freqs, &power, gp.freq);
        SingleTraceResult {
            ghost_detected: true,
            ghost_freq: gp.freq,
            ghost_snr: snr,
            ghost_fwhm: fwhm.unwrap_or(f64::NAN),
            dominant_freq: peaks.first().map_or(f64::NAN, |p| p.freq),
            dominant_power: peaks.first().map_or(f64::NAN, |p| p.power),
        }
    } else {
        SingleTraceResult {
            ghost_detected: false,
            ghost_freq: f64::NAN,
            ghost_snr: 0.0,
            ghost_fwhm: f64::NAN,
            dominant_freq: peaks.first().map_or(f64::NAN, |p| p.freq),
            dominant_power: peaks.first().map_or(f64::NAN, |p| p.power),
        }
    }
}

#[derive(Debug, Clone)]
struct TraceResult {
    // KE temporal spectrum
    ke_ghost_detected: bool,
    ke_ghost_freq: f64,
    ke_ghost_snr: f64,
    ke_ghost_fwhm: f64,
    ke_dominant_freq: f64,
    ke_dominant_power: f64,
    // Density variance temporal spectrum
    var_ghost_detected: bool,
    var_ghost_freq: f64,
    var_ghost_snr: f64,
    // Spatial power spectrum
    spatial_ghost_detected: bool,
    spatial_ghost_freq: f64,
    spatial_ghost_snr: f64,
    // Final-step values
    final_ke: f64,
    final_var: f64,
}

fn csv_header() -> [&'static str; 16] {
    [
        "tau",
        "ke_ghost_detected",
        "ke_ghost_freq",
        "ke_ghost_snr",
        "ke_ghost_fwhm",
        "ke_dominant_freq",
        "ke_dominant_power",
        "var_ghost_detected",
        "var_ghost_freq",
        "var_ghost_snr",
        "spatial_ghost_detected",
        "spatial_ghost_freq",
        "spatial_ghost_snr",
        "final_ke",
        "final_var",
        "label",
    ]
}

fn result_to_record(tau_or_param: &str, result: &TraceResult, label: &str) -> Vec<String> {
    vec![
        tau_or_param.to_string(),
        format!("{}", result.ke_ghost_detected),
        format!("{:.6}", result.ke_ghost_freq),
        format!("{:.4}", result.ke_ghost_snr),
        format!("{:.6}", result.ke_ghost_fwhm),
        format!("{:.6}", result.ke_dominant_freq),
        format!("{:.6e}", result.ke_dominant_power),
        format!("{}", result.var_ghost_detected),
        format!("{:.6}", result.var_ghost_freq),
        format!("{:.4}", result.var_ghost_snr),
        format!("{}", result.spatial_ghost_detected),
        format!("{:.6}", result.spatial_ghost_freq),
        format!("{:.4}", result.spatial_ghost_snr),
        format!("{:.6e}", result.final_ke),
        format!("{:.6e}", result.final_var),
        label.to_string(),
    ]
}

fn run_sweep(res: usize, steps: usize, fp32: bool, out_dir: &std::path::Path) -> Result<()> {
    let precision = if fp32 { Precision::FP32 } else { Precision::BF16 };
    let prec_str = if fp32 { "fp32" } else { "bf16" };
    println!("=== ZD Resonance Tau Sweep ({prec_str}, {res}^3, {steps} steps) ===");
    println!("Observable: kinetic energy + density variance + spatial spectrum");
    println!("Forcing: Kolmogorov A={DEFAULT_FORCE_AMP}, k=2");

    std::fs::create_dir_all(out_dir)?;

    let csv_path = out_dir.join(format!("zd_resonance_cuda_{res}_{prec_str}.csv"));
    let mut wtr = csv::Writer::from_path(&csv_path)?;
    wtr.write_record(csv_header())?;

    let wavelength = 10.0;
    // Exponential model: nu = nu_base * exp(-lambda * frustration)
    // With lambda=3.0: nu ranges from nu_base (f=0) to nu_base*exp(-3)=0.05*nu_base
    // Per-tau: nu_base = (tau - 0.5) / 3.0, so each tau produces a distinct viscosity field.
    // This ensures the sweep actually tests different flow regimes.

    for &tau in &TAU_VALUES {
        let nu_base = (tau - 0.5) / 3.0;
        let model = ViscosityCouplingModel::Exponential {
            nu_base,
            lambda: 3.0,
        };

        print!("  Viscosity field for tau={tau:.2} (nu_base={nu_base:.6})...");
        std::io::stdout().flush()?;
        let visc_start = Instant::now();
        let viscosity = compute_viscosity_field(res, res, res, &model, wavelength);
        let nu_min = viscosity.iter().cloned().fold(f64::INFINITY, f64::min);
        let nu_max = viscosity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            " {:.2?} | nu=[{nu_min:.6}, {nu_max:.6}]",
            visc_start.elapsed()
        );

        let label = format!("tau={tau:.2}");
        let data = run_experiment(
            res,
            tau,
            steps,
            precision,
            Some(&viscosity),
            DEFAULT_FORCE_AMP,
            &label,
        )?;
        let result = analyze_traces(&data);

        let record = result_to_record(&format!("{tau:.3}"), &result, &label);
        wtr.write_record(&record)?;
        wtr.flush()?;

        // Report all three channels
        let ke_status = if result.ke_ghost_detected {
            format!(
                "KE: GHOST f={:.4} SNR={:.2}",
                result.ke_ghost_freq, result.ke_ghost_snr
            )
        } else {
            format!("KE: no ghost (dom={:.4})", result.ke_dominant_freq)
        };
        let spatial_status = if result.spatial_ghost_detected {
            format!(
                "Spatial: GHOST k={:.4} SNR={:.2}",
                result.spatial_ghost_freq, result.spatial_ghost_snr
            )
        } else {
            "Spatial: no ghost".to_string()
        };
        println!("  tau={tau:.2}: {ke_status} | {spatial_status}");
    }

    println!("Results saved to {}", csv_path.display());
    Ok(())
}

fn run_control(res: usize, steps: usize, fp32: bool, out_dir: &std::path::Path) -> Result<()> {
    let precision = if fp32 { Precision::FP32 } else { Precision::BF16 };
    let prec_str = if fp32 { "fp32" } else { "bf16" };
    println!("=== ZD Resonance Control (NO ZD modulation, {prec_str}, {res}^3, {steps} steps) ===");
    println!("Observable: kinetic energy + density variance + spatial spectrum");
    println!("Forcing: Kolmogorov A={DEFAULT_FORCE_AMP}, k=2 (same as sweep)");

    std::fs::create_dir_all(out_dir)?;

    let csv_path = out_dir.join(format!("zd_resonance_cuda_control_{res}.csv"));
    let mut wtr = csv::Writer::from_path(&csv_path)?;
    wtr.write_record(csv_header())?;

    for &tau in &TAU_VALUES {
        let label = format!("ctrl tau={tau:.2}");
        // No viscosity field = constant tau (null hypothesis)
        let data = run_experiment(res, tau, steps, precision, None, DEFAULT_FORCE_AMP, &label)?;
        let result = analyze_traces(&data);

        let record = result_to_record(&format!("{tau:.3}"), &result, &label);
        wtr.write_record(&record)?;
        wtr.flush()?;

        let ke_status = if result.ke_ghost_detected {
            format!(
                "WARNING: KE GHOST in control at f={:.4} SNR={:.2}",
                result.ke_ghost_freq, result.ke_ghost_snr
            )
        } else {
            "KE: clean (no ghost)".to_string()
        };
        let spatial_status = if result.spatial_ghost_detected {
            format!(
                "WARNING: Spatial GHOST at k={:.4} SNR={:.2}",
                result.spatial_ghost_freq, result.spatial_ghost_snr
            )
        } else {
            "Spatial: clean".to_string()
        };
        println!("  tau={tau:.2}: {ke_status} | {spatial_status}");
    }

    println!("Control results saved to {}", csv_path.display());
    Ok(())
}

fn run_coupling_sweep(
    res: usize,
    steps: usize,
    fp32: bool,
    out_dir: &std::path::Path,
) -> Result<()> {
    let precision = if fp32 { Precision::FP32 } else { Precision::BF16 };
    let prec_str = if fp32 { "fp32" } else { "bf16" };
    println!("=== ZD Resonance Coupling Sweep ({prec_str}, tau=0.55, {res}^3, {steps} steps) ===");
    std::fs::create_dir_all(out_dir)?;

    let csv_path = out_dir.join(format!("zd_resonance_cuda_coupling_{res}_{prec_str}.csv"));
    let mut wtr = csv::Writer::from_path(&csv_path)?;
    wtr.write_record(csv_header())?;

    let tau = 0.55;
    let wavelength = 10.0;
    let bridge = FrustrationViscosityBridge::new(16);
    let field = generate_sedenion_field(res, res, res, wavelength);
    let frustration = field.local_frustration_density_par(16);

    for &coupling in &COUPLING_VALUES {
        let nu_base = (tau - 0.5) / 3.0;
        let model = if coupling == 0.0 {
            ViscosityCouplingModel::Constant { nu_base }
        } else {
            ViscosityCouplingModel::Exponential {
                nu_base,
                lambda: coupling * 10.0,
            }
        };
        let mut viscosity = bridge.frustration_to_viscosity_model(&frustration, &model);
        for nu in &mut viscosity {
            *nu = nu.clamp(0.001, 0.5);
        }

        let label = format!("lambda={coupling:.3}");
        let data = run_experiment(
            res,
            tau,
            steps,
            precision,
            Some(&viscosity),
            DEFAULT_FORCE_AMP,
            &label,
        )?;
        let result = analyze_traces(&data);

        let record = result_to_record(&format!("{coupling:.3}"), &result, &label);
        wtr.write_record(&record)?;
        wtr.flush()?;

        let status = if result.ke_ghost_detected {
            format!(
                "KE: GHOST f={:.4} SNR={:.2}",
                result.ke_ghost_freq, result.ke_ghost_snr
            )
        } else {
            format!(
                "KE: no ghost (KE={:.4e} var={:.4e})",
                result.final_ke, result.final_var
            )
        };
        println!("  lambda={coupling:.3}: {status}");
    }

    println!("Coupling sweep saved to {}", csv_path.display());
    Ok(())
}

fn run_reynolds_sweep(
    res: usize,
    steps: usize,
    fp32: bool,
    out_dir: &std::path::Path,
) -> Result<()> {
    let precision = if fp32 { Precision::FP32 } else { Precision::BF16 };
    let prec_str = if fp32 { "fp32" } else { "bf16" };
    println!("=== ZD Resonance Reynolds Sweep ({prec_str}, {res}^3, {steps} steps) ===");
    println!("Motivated by real-data finding: Re_eff=0.16 (laminar) insufficient for Ghost onset.");
    std::fs::create_dir_all(out_dir)?;

    let csv_path = out_dir.join(format!("zd_resonance_cuda_reynolds_{res}_{prec_str}.csv"));
    let mut wtr = csv::Writer::from_path(&csv_path)?;
    wtr.write_record([
        "tau",
        "force_amp",
        "re_eff",
        "ke_ghost_detected",
        "ke_ghost_freq",
        "ke_ghost_snr",
        "spatial_ghost_detected",
        "spatial_ghost_freq",
        "spatial_ghost_snr",
        "final_ke",
        "final_var",
        "ke_dominant_freq",
    ])?;

    let wavelength = 10.0;

    for &tau in &RE_TAU_VALUES {
        let nu = (tau - 0.5) / 3.0;
        let model = ViscosityCouplingModel::Exponential {
            nu_base: nu,
            lambda: 3.0,
        };
        let viscosity = compute_viscosity_field(res, res, res, &model, wavelength);

        for &force_amp in &FORCE_AMPS {
            // Re = U*L/nu, U ~ F*L/nu for Stokes, L = N/(2*pi*k)
            let u_est = force_amp / nu.max(1e-10);
            let re_eff = u_est * res as f64 / nu.max(1e-10);

            let label = format!("tau={tau:.3} F={force_amp:.3}");
            let data = run_experiment(
                res,
                tau,
                steps,
                precision,
                Some(&viscosity),
                force_amp,
                &label,
            )?;
            let result = analyze_traces(&data);

            wtr.write_record(&[
                format!("{tau:.4}"),
                format!("{force_amp:.4}"),
                format!("{re_eff:.2}"),
                format!("{}", result.ke_ghost_detected),
                format!("{:.6}", result.ke_ghost_freq),
                format!("{:.4}", result.ke_ghost_snr),
                format!("{}", result.spatial_ghost_detected),
                format!("{:.6}", result.spatial_ghost_freq),
                format!("{:.4}", result.spatial_ghost_snr),
                format!("{:.6e}", result.final_ke),
                format!("{:.6e}", result.final_var),
                format!("{:.6}", result.ke_dominant_freq),
            ])?;
            wtr.flush()?;
        }
    }

    println!("Reynolds sweep saved to {}", csv_path.display());
    Ok(())
}

fn run_analyze(data_dir: &std::path::Path) -> Result<()> {
    println!("=== ZD Resonance Analysis ===");

    let sweep_path = data_dir.join("zd_resonance_cuda_128_bf16.csv");
    let control_path = data_dir.join("zd_resonance_cuda_control_128.csv");

    let mut sweep_ke_ghosts: Vec<(f64, f64)> = Vec::new(); // (freq, snr)
    let mut sweep_spatial_ghosts: Vec<(f64, f64)> = Vec::new();
    let mut control_ke_ghosts: Vec<(f64, f64)> = Vec::new();
    let mut control_spatial_ghosts: Vec<(f64, f64)> = Vec::new();

    if sweep_path.exists() {
        println!("\nSweep results ({}):", sweep_path.display());
        let mut rdr = csv::Reader::from_path(&sweep_path)?;
        for result in rdr.records() {
            let record = result?;
            let tau: f64 = record[0].parse().unwrap_or(0.0);
            let ke_det: bool = record[1].parse().unwrap_or(false);
            let ke_freq: f64 = record[2].parse().unwrap_or(f64::NAN);
            let ke_snr: f64 = record[3].parse().unwrap_or(0.0);
            let sp_det: bool = record[10].parse().unwrap_or(false);
            let sp_freq: f64 = record[11].parse().unwrap_or(f64::NAN);
            let sp_snr: f64 = record[12].parse().unwrap_or(0.0);
            let final_ke: f64 = record[13].parse().unwrap_or(0.0);

            if ke_det {
                sweep_ke_ghosts.push((ke_freq, ke_snr));
            }
            if sp_det {
                sweep_spatial_ghosts.push((sp_freq, sp_snr));
            }

            let ke_str = if ke_det {
                format!("KE ghost f={ke_freq:.4} SNR={ke_snr:.2}")
            } else {
                "KE: no ghost".to_string()
            };
            let sp_str = if sp_det {
                format!("Spatial ghost k={sp_freq:.4} SNR={sp_snr:.2}")
            } else {
                "Spatial: no ghost".to_string()
            };
            println!("  tau={tau:.2}: {ke_str} | {sp_str} | KE={final_ke:.4e}");
        }
    } else {
        println!("WARNING: Sweep CSV not found at {}", sweep_path.display());
    }

    if control_path.exists() {
        println!("\nControl results ({}):", control_path.display());
        let mut rdr = csv::Reader::from_path(&control_path)?;
        for result in rdr.records() {
            let record = result?;
            let tau: f64 = record[0].parse().unwrap_or(0.0);
            let ke_det: bool = record[1].parse().unwrap_or(false);
            let ke_freq: f64 = record[2].parse().unwrap_or(f64::NAN);
            let ke_snr: f64 = record[3].parse().unwrap_or(0.0);
            let sp_det: bool = record[10].parse().unwrap_or(false);
            let sp_freq: f64 = record[11].parse().unwrap_or(f64::NAN);
            let sp_snr: f64 = record[12].parse().unwrap_or(0.0);

            if ke_det {
                control_ke_ghosts.push((ke_freq, ke_snr));
            }
            if sp_det {
                control_spatial_ghosts.push((sp_freq, sp_snr));
            }

            if ke_det || sp_det {
                println!(
                    "  tau={tau:.2}: ARTIFACT KE f={ke_freq:.4} SNR={ke_snr:.2} | Spatial f={sp_freq:.4} SNR={sp_snr:.2}"
                );
            }
        }
    } else {
        println!("WARNING: Control CSV not found at {}", control_path.display());
    }

    // Verdict
    println!("\n=== VERDICTS ===");

    let sweep_has_ke = !sweep_ke_ghosts.is_empty();
    let sweep_has_spatial = !sweep_spatial_ghosts.is_empty();
    let control_has_ke = !control_ke_ghosts.is_empty();
    let control_has_spatial = !control_spatial_ghosts.is_empty();

    // C-774: ZD Resonance Frequency Locking
    if (sweep_has_ke || sweep_has_spatial) && !control_has_ke && !control_has_spatial {
        let ghosts = if sweep_has_ke {
            &sweep_ke_ghosts
        } else {
            &sweep_spatial_ghosts
        };
        let n = ghosts.len() as f64;
        let mean_freq = ghosts.iter().map(|(f, _)| f).sum::<f64>() / n;
        let freq_std = (ghosts
            .iter()
            .map(|(f, _)| (f - mean_freq).powi(2))
            .sum::<f64>()
            / (n - 1.0).max(1.0))
        .sqrt();
        let mean_snr = ghosts.iter().map(|(_, s)| s).sum::<f64>() / n;

        println!("Ghost frequency: mean={mean_freq:.6} +/- {freq_std:.6}");
        println!(
            "Distance from theoretical {ALIASED_GHOST_FREQ:.6}: {:.6}",
            (mean_freq - ALIASED_GHOST_FREQ).abs()
        );
        println!("Mean SNR: {mean_snr:.2}");

        if freq_std < 0.01 && mean_snr > 3.0 {
            println!("C-774: CONFIRMED (tau-independent ghost, absent in control)");
            println!("C-775: CONFIRMED (production ghost detection at SNR>{mean_snr:.1})");
        } else if freq_std >= 0.01 {
            println!("C-774: FALSIFIED (ghost frequency varies with tau)");
            println!("C-775: INCONCLUSIVE");
        } else {
            println!("C-774: INCONCLUSIVE (ghost detected but SNR < 3)");
            println!("C-775: INCONCLUSIVE");
        }
    } else if control_has_ke || control_has_spatial {
        println!("C-774: FALSIFIED (ghost present in control -- lattice/forcing artifact)");
        println!("C-775: FALSIFIED (not specific to ZD modulation)");
    } else if !sweep_has_ke && !sweep_has_spatial {
        println!("C-774: NULL (no ghost detected in sweep or spatial spectrum)");
        println!("C-775: NULL (no ghost detected in any channel)");
    }

    // Reynolds sweep
    let reynolds_path = data_dir.join("zd_resonance_cuda_reynolds_128.csv");
    if reynolds_path.exists() {
        println!("\nReynolds sweep results:");
        let mut rdr = csv::Reader::from_path(&reynolds_path)?;
        let mut onset_re = f64::NAN;
        for result in rdr.records() {
            let record = result?;
            let tau: f64 = record[0].parse().unwrap_or(0.0);
            let force: f64 = record[1].parse().unwrap_or(0.0);
            let re_eff: f64 = record[2].parse().unwrap_or(0.0);
            let ke_det: bool = record[3].parse().unwrap_or(false);
            let ke_snr: f64 = record[5].parse().unwrap_or(0.0);
            let final_ke: f64 = record[9].parse().unwrap_or(0.0);

            if ke_det && ke_snr > 3.0 && onset_re.is_nan() {
                onset_re = re_eff;
                println!(
                    "  Ghost ONSET: tau={tau:.3} F={force:.3} Re={re_eff:.1} SNR={ke_snr:.2} KE={final_ke:.4e}"
                );
            }
        }
        if onset_re.is_nan() {
            println!("  No ghost onset found with SNR>3 across entire Reynolds range.");
        } else {
            println!("  Onset Re_eff = {onset_re:.1}");
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Sweep {
            res,
            steps,
            fp32,
            ref out_dir,
        } => run_sweep(res, steps, fp32, out_dir),
        Commands::Control {
            res,
            steps,
            fp32,
            ref out_dir,
        } => run_control(res, steps, fp32, out_dir),
        Commands::CouplingSweep {
            res,
            steps,
            fp32,
            ref out_dir,
        } => run_coupling_sweep(res, steps, fp32, out_dir),
        Commands::ReynoldsSweep {
            res,
            steps,
            fp32,
            ref out_dir,
        } => run_reynolds_sweep(res, steps, fp32, out_dir),
        Commands::Analyze { ref data_dir } => run_analyze(data_dir),
    }
}
