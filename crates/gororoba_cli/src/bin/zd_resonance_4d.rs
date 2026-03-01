// 4D ZD Resonance Experiments (Sprint 49B)
//
// WHY: The 16D sedenion algebra has 16 components, and a 4D spatial embedding
// (16 = 4^2) may reveal coupling symmetries that project out in 3D.
// This binary uses the existing 4D batch kernel (D3Q19 x nw independent slices)
// to sweep the 4th dimension as a Sedenion field parameter axis.
//
// The 4th dimension (w) parametrizes different Sedenion field configurations,
// allowing us to test whether ghost signatures correlate across field realizations.
// This is NOT a true D4Q33 lattice -- streaming is periodic in x/y/z only.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use spectral_core::ghost_spectral::{
    ALIASED_GHOST_FREQ, check_ghost, compute_power_spectrum, find_peaks, noise_floor, peak_snr,
};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use sign_imbalance::bridge::{
    ImbalanceViscosityBridge, SedenionField, ViscosityCouplingModel,
};

/// 4D ZD Resonance Experiments (batch 3D worlds along w-axis)
#[derive(Parser, Debug)]
#[command(author, version, about = "4D ZD Resonance via D3Q19 batch kernel")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 32^3 x nw sweep (1.05M cells at nw=32, ~90 MB GPU)
    Sweep32 {
        /// Number of w-slices (4th dimension)
        #[arg(long, default_value_t = 32)]
        nw: usize,
        /// Steps per configuration
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value = "data/csv")]
        out_dir: PathBuf,
    },
    /// 64^3 x nw sweep (16.8M cells at nw=64, ~1.4 GB GPU)
    Sweep64 {
        #[arg(long, default_value_t = 16)]
        nw: usize,
        #[arg(long, default_value_t = 500)]
        steps: usize,
        #[arg(long, default_value = "data/csv")]
        out_dir: PathBuf,
    },
    /// Control: constant tau, no ZD modulation
    Control {
        #[arg(long, default_value_t = 32)]
        res: usize,
        #[arg(long, default_value_t = 32)]
        nw: usize,
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value = "data/csv")]
        out_dir: PathBuf,
    },
}

/// Generate Sedenion field with w-dependent phase rotation.
/// Each w-slice sees a different phase offset, probing the full 16D parameter space.
fn generate_4d_viscosity(nx: usize, nw: usize, tau_base: f64, wavelength: f64) -> Vec<f64> {
    let nz_total = nx * nw; // Flattened z dimension = nz_sub * nw
    let n_cells = nx * nx * nz_total;
    let bridge = ImbalanceViscosityBridge::new(16);
    let nu_base = (tau_base - 0.5) / 3.0;
    let model = ViscosityCouplingModel::Exponential {
        nu_base,
        lambda: 3.0,
    };

    let mut viscosity = vec![0.0_f64; n_cells];

    // Process w-slices independently to limit memory
    for w in 0..nw {
        let mut field = SedenionField::uniform(nx, nx, nx);
        let phase_offset = w as f64 * std::f64::consts::TAU / nw as f64;

        for z in 0..nx {
            for y in 0..nx {
                for x in 0..nx {
                    let s = field.get_mut(x, y, z);
                    let px = x as f64 / nx as f64 - 0.5;
                    let py = y as f64 / nx as f64 - 0.5;
                    let pz = z as f64 / nx as f64 - 0.5;
                    for (i, si) in s.iter_mut().enumerate() {
                        let phase = phase_offset + i as f64 * std::f64::consts::FRAC_PI_8;
                        let kx = phase.sin();
                        let ky = phase.cos();
                        let kz = (phase * 1.3).sin();
                        let dot = px * kx + py * ky + pz * kz;
                        *si = (dot * wavelength).sin();
                    }
                }
            }
        }

        let imbalance = field.local_imbalance_density_par(16);
        let visc_slice = bridge.imbalance_to_viscosity_model(&imbalance, &model);

        // Copy into flattened 4D array: idx = x + nx*(y + nx*(z_sub + nx*w))
        // where z_sub is the z within the w-slice
        let vol_3d = nx * nx * nx;
        #[allow(clippy::needless_range_loop)] // idx_3d computes 3D->4D spatial remap
        for idx_3d in 0..vol_3d {
            // In the 4D layout: z_global = z_sub + nx * w
            let x_local = idx_3d % nx;
            let y_local = (idx_3d / nx) % nx;
            let z_local = idx_3d / (nx * nx);
            let z_global = z_local + nx * w;
            let idx_4d = x_local + nx * (y_local + nx * z_global);
            viscosity[idx_4d] = visc_slice[idx_3d].clamp(0.001, 0.5);
        }
    }

    viscosity
}

/// Compute kinetic energy density from host-side arrays.
fn compute_ke(rho: &[f32], u: &[[f32; 3]]) -> f64 {
    let n = rho.len() as f64;
    let mut ke_sum = 0.0_f64;
    for (i, &r) in rho.iter().enumerate() {
        let r64 = r as f64;
        let ux = u[i][0] as f64;
        let uy = u[i][1] as f64;
        let uz = u[i][2] as f64;
        ke_sum += r64 * (ux * ux + uy * uy + uz * uz);
    }
    ke_sum / n
}

fn run_4d_sweep(res: usize, nw: usize, steps: usize, out_dir: &std::path::Path) -> Result<()> {
    let nz_total = res * nw;
    let n_cells = res * res * nz_total;
    println!("=== 4D ZD Resonance Sweep ({res}^3 x {nw} = {n_cells} cells) ===");

    std::fs::create_dir_all(out_dir)?;

    let tau_base = 0.60;
    let wavelength = 10.0;

    println!("Computing 4D viscosity field...");
    let visc_start = Instant::now();
    let viscosity = generate_4d_viscosity(res, nw, tau_base, wavelength);
    println!("Viscosity computed in {:.2?}", visc_start.elapsed());

    // Create solver with flattened 4D grid (nx, ny, nz_total)
    let mut solver = LbmSolver3DCuda::new(res, res, nz_total, tau_base, Precision::BF16)
        .context("Failed to init CUDA solver")?;
    solver
        .set_viscosity_field(&viscosity)
        .context("Failed to upload viscosity")?;
    solver.initialize_uniform(1.0, [0.01, 0.0, 0.0])?;

    // Apply Kolmogorov forcing: F_x = A * sin(2*pi*k*z/nz_total) across full grid
    let force_amp = 0.001;
    let k_force = 2.0;
    let n_cells_total = res * res * nz_total;
    let mut force_field = vec![[0.0_f64; 3]; n_cells_total];
    for z in 0..nz_total {
        let fz =
            force_amp * (2.0 * std::f64::consts::PI * k_force * z as f64 / nz_total as f64).sin();
        for y in 0..res {
            for x in 0..res {
                let idx = x + res * (y + res * z);
                force_field[idx][0] = fz;
            }
        }
    }
    solver
        .set_force_field(&force_field)
        .context("Failed to upload force field")?;

    // Run simulation using 4D batch kernel -- track KE (not conserved rho_mean)
    let sample_interval = 20;
    let mut ke_trace = Vec::with_capacity(steps / sample_interval + 1);
    let start = Instant::now();

    for step in 0..steps {
        solver.step_4d(nw)?;
        if step % sample_interval == 0 {
            solver.sync_to_host()?;
            let ke = compute_ke(&solver.rho, &solver.u);
            ke_trace.push(ke);
        }
        if step % 200 == 0 {
            let ke_str = if let Some(&last) = ke_trace.last() {
                format!(" KE={last:.6e}")
            } else {
                String::new()
            };
            print!(
                "\rStep {step}/{steps} ({:.1}%){ke_str}",
                step as f64 / steps as f64 * 100.0
            );
            std::io::stdout().flush()?;
        }
    }
    println!("\rComplete in {:.2?}                ", start.elapsed());

    // Analyze KE temporal spectrum
    let csv_path = out_dir.join(format!("zd_resonance_4d_{res}x{nw}.csv"));
    let mut wtr = csv::Writer::from_path(&csv_path)?;
    wtr.write_record(["step", "ke"])?;
    for (i, &ke) in ke_trace.iter().enumerate() {
        wtr.write_record(&[(i * sample_interval).to_string(), ke.to_string()])?;
    }
    wtr.flush()?;

    if ke_trace.len() >= 32 {
        let (freqs, power) = compute_power_spectrum(&ke_trace);
        let peaks = find_peaks(&freqs, &power, 10);
        let ghost = check_ghost(&peaks);

        if let Some(gp) = ghost {
            let nf = noise_floor(&power);
            let snr = peak_snr(gp, nf);
            println!(
                "4D Ghost DETECTED: freq={:.4}, SNR={:.2}, delta={:.4}",
                gp.freq,
                snr,
                (gp.freq - ALIASED_GHOST_FREQ).abs()
            );
        } else {
            println!("4D Ghost: NOT DETECTED");
            if let Some(p) = peaks.first() {
                println!("  Dominant peak: freq={:.4}, power={:.6}", p.freq, p.power);
            }
        }
    }

    println!("Results saved to {}", csv_path.display());
    Ok(())
}

fn run_4d_control(res: usize, nw: usize, steps: usize, out_dir: &std::path::Path) -> Result<()> {
    let nz_total = res * nw;
    let n_cells = res * res * nz_total;
    println!("=== 4D Control (NO ZD modulation, {res}^3 x {nw} = {n_cells} cells) ===");

    std::fs::create_dir_all(out_dir)?;

    let tau_base = 0.60;
    let mut solver = LbmSolver3DCuda::new(res, res, nz_total, tau_base, Precision::BF16)
        .context("Failed to init CUDA solver")?;
    solver.initialize_uniform(1.0, [0.01, 0.0, 0.0])?;

    // Same forcing as sweep for fair comparison
    let force_amp = 0.001;
    let k_force = 2.0;
    let n_cells_total = res * res * nz_total;
    let mut force_field = vec![[0.0_f64; 3]; n_cells_total];
    for z in 0..nz_total {
        let fz =
            force_amp * (2.0 * std::f64::consts::PI * k_force * z as f64 / nz_total as f64).sin();
        for y in 0..res {
            for x in 0..res {
                let idx = x + res * (y + res * z);
                force_field[idx][0] = fz;
            }
        }
    }
    solver
        .set_force_field(&force_field)
        .context("Failed to upload force field")?;

    let sample_interval = 20;
    let mut ke_trace = Vec::with_capacity(steps / sample_interval + 1);
    let start = Instant::now();

    for step in 0..steps {
        solver.step_4d(nw)?;
        if step % sample_interval == 0 {
            solver.sync_to_host()?;
            let ke = compute_ke(&solver.rho, &solver.u);
            ke_trace.push(ke);
        }
        if step % 200 == 0 {
            let ke_str = if let Some(&last) = ke_trace.last() {
                format!(" KE={last:.6e}")
            } else {
                String::new()
            };
            print!(
                "\rStep {step}/{steps} ({:.1}%){ke_str}",
                step as f64 / steps as f64 * 100.0
            );
            std::io::stdout().flush()?;
        }
    }
    println!("\rComplete in {:.2?}                ", start.elapsed());

    let csv_path = out_dir.join(format!("zd_resonance_4d_control_{res}x{nw}.csv"));
    let mut wtr = csv::Writer::from_path(&csv_path)?;
    wtr.write_record(["step", "ke"])?;
    for (i, &ke) in ke_trace.iter().enumerate() {
        wtr.write_record(&[(i * sample_interval).to_string(), ke.to_string()])?;
    }
    wtr.flush()?;

    if ke_trace.len() >= 32 {
        let (freqs, power) = compute_power_spectrum(&ke_trace);
        let peaks = find_peaks(&freqs, &power, 5);
        let ghost = check_ghost(&peaks);
        if let Some(gp) = ghost {
            let nf = noise_floor(&power);
            let snr = peak_snr(gp, nf);
            println!("4D Control: WARNING GHOST at {:.4} SNR={:.2}", gp.freq, snr);
        } else {
            println!("4D Control: clean (no ghost)");
        }
    }

    println!("Control saved to {}", csv_path.display());
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Sweep32 {
            nw,
            steps,
            ref out_dir,
        } => run_4d_sweep(32, nw, steps, out_dir),
        Commands::Sweep64 {
            nw,
            steps,
            ref out_dir,
        } => run_4d_sweep(64, nw, steps, out_dir),
        Commands::Control {
            res,
            nw,
            steps,
            ref out_dir,
        } => run_4d_control(res, nw, steps, out_dir),
    }
}
