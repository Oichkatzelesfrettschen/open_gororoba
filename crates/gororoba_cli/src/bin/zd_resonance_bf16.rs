use anyhow::{Context, Result};
use clap::Parser;
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use sign_imbalance::bridge::{ImbalanceViscosityBridge, SedenionField, ViscosityCouplingModel};
use spectral_core::ghost_spectral::{
    GHOST_FREQ, check_ghost, compute_power_spectrum, find_peaks, peak_snr,
};
use std::{path::PathBuf, time::Instant};

/// Zero-Divisor Resonance Sweep (BF16)
///
/// High-performance LBM simulation with Sedenion ZD-modulated viscosity.
/// Detects ghost frequencies (phi^-1/2) in the density evolution trace.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Grid size (N for NxNxN cube)
    #[arg(short, long, default_value_t = 128)]
    size: usize,

    /// Number of simulation steps
    #[arg(short, long, default_value_t = 10000)]
    steps: usize,

    /// Relaxation time base (tau_base)
    #[arg(long, default_value_t = 0.6)]
    tau_base: f64,

    /// Relaxation time amplitude (tau_amp)
    /// Strength of ZD modulation. 0.0 = control run.
    #[arg(long, default_value_t = 0.01)]
    tau_amp: f64,

    /// ZD pattern wavelength (lambda)
    /// Controls spatial frequency of sedenion field.
    #[arg(long, default_value_t = 10.0)]
    lambda: f64,

    /// Output directory
    #[arg(short, long, default_value = "data/zd_resonance")]
    out_dir: PathBuf,

    /// Coupling model: exponential, linear, power_law, sigmoid, kubo
    #[arg(long, default_value = "exponential")]
    model: String,
}

fn generate_sedenion_field(nx: usize, ny: usize, nz: usize, lambda: f64) -> SedenionField {
    let mut field = SedenionField::uniform(nx, ny, nz);
    let _inv_scale = lambda / nx as f64; // Scale factor

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let s = field.get_mut(x, y, z);

                // Normalized coordinates [-0.5, 0.5]
                let px = x as f64 / nx as f64 - 0.5;
                let py = y as f64 / ny as f64 - 0.5;
                let pz = z as f64 / nz as f64 - 0.5;

                let t = 0.0;

                for (i, si) in s.iter_mut().enumerate() {
                    let phase = t + i as f64 * std::f64::consts::FRAC_PI_8; // 2pi/16 = pi/8

                    let kx = phase.sin();
                    let ky = phase.cos();
                    let kz = (phase * 1.3).sin();

                    let dot = px * kx + py * ky + pz * kz;
                    *si = (dot * lambda).sin();
                }
            }
        }
    }
    field
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("=== ZD Resonance Sweep (BF16) ===");
    println!("Grid: {}x{}x{}", args.size, args.size, args.size);
    println!("Steps: {}", args.steps);
    println!("Tau: base={}, amp={}", args.tau_base, args.tau_amp);
    println!("Lambda: {}", args.lambda);
    println!("Model: {}", args.model);

    std::fs::create_dir_all(&args.out_dir)?;

    // 1. Generate Sedenion Field (CPU)
    println!("Generating Sedenion field...");
    let start_gen = Instant::now();
    let field = generate_sedenion_field(args.size, args.size, args.size, args.lambda);
    println!("Generation took {:.2?}", start_gen.elapsed());

    // 2. Compute Imbalance & Viscosity (CPU)
    println!("Computing local imbalance & viscosity...");
    let start_visc = Instant::now();
    let bridge = ImbalanceViscosityBridge::new(16);

    let nu_base = (args.tau_base - 0.5) / 3.0;

    let model = match args.model.as_str() {
        "exponential" => ViscosityCouplingModel::Exponential {
            nu_base,
            lambda: 2.0,
        },
        "linear" => ViscosityCouplingModel::Linear {
            nu_base,
            alpha: 1.0,
        },
        "power_law" => ViscosityCouplingModel::PowerLaw { nu_base, n: 1.5 },
        "sigmoid" => ViscosityCouplingModel::Sigmoid {
            nu_low: nu_base * 0.5,
            nu_high: nu_base * 1.5,
            k: 50.0,
            f_crit: 0.38,
        },
        "kubo" => ViscosityCouplingModel::kubo_default(nu_base),
        _ => ViscosityCouplingModel::Exponential {
            nu_base,
            lambda: 2.0,
        },
    };

    let imbalance = field.local_imbalance_density_par(16);
    let mut viscosity = bridge.imbalance_to_viscosity_model(&imbalance, &model);

    for nu in &mut viscosity {
        *nu = nu.clamp(0.001, 0.5); // tau range [0.503, 2.0]
    }

    println!("Viscosity field computed in {:.2?}", start_visc.elapsed());

    // 3. Initialize Solver (GPU)
    println!("Initializing BF16 Solver...");
    let mut solver = LbmSolver3DCuda::new(
        args.size,
        args.size,
        args.size,
        args.tau_base,
        Precision::BF16,
    )
    .context("Failed to init CUDA solver")?;

    solver
        .set_viscosity_field(&viscosity)
        .context("Failed to upload viscosity")?;

    let perturbation = 0.01;
    solver.initialize_uniform(1.0, [perturbation as f32, 0.0, 0.0])?;

    // 4. Run Simulation
    println!("Starting simulation...");
    let mut rho_trace = Vec::with_capacity(args.steps);
    let start_sim = Instant::now();

    for step in 0..args.steps {
        solver.step()?;

        if step % 10 == 0 {
            let rho_mean = solver.calculate_mean_density()?;
            rho_trace.push(rho_mean as f64);
        }

        if step % 1000 == 0 {
            use std::io::Write;
            print!(
                "\rStep {}/{} ({:.1}%)",
                step,
                args.steps,
                step as f64 / args.steps as f64 * 100.0
            );
            std::io::stdout().flush()?;
        }
    }
    println!("\nSimulation complete in {:.2?}", start_sim.elapsed());

    // 5. Analysis (Ghost Frequency)
    println!("Analyzing density trace for ghost signatures...");
    let (freqs, power) = compute_power_spectrum(&rho_trace);
    let peaks = find_peaks(&freqs, &power, 5);

    let ghost = check_ghost(&peaks);
    let verdict = if let Some(p) = ghost {
        let noise = spectral_core::ghost_spectral::noise_floor(&power);
        let snr = peak_snr(p, noise);
        format!("DETECTED (freq={:.4}, SNR={:.2})", p.freq, snr)
    } else {
        "NOT DETECTED".to_string()
    };

    println!("Ghost Verdict: {}", verdict);
    println!("Target Ghost Freq: {:.6}", GHOST_FREQ);
    if let Some(p) = peaks.first() {
        println!("Dominant Peak: {:.6}", p.freq);
    }

    // 6. Save Data
    let trace_path = args.out_dir.join(format!("rho_trace_{}.csv", args.size));
    let mut wtr = csv::Writer::from_path(&trace_path)?;
    wtr.write_record(["step", "rho_mean"])?;
    for (i, &rho) in rho_trace.iter().enumerate() {
        wtr.write_record(&[(i * 10).to_string(), rho.to_string()])?;
    }
    wtr.flush()?;
    println!("Trace saved to {}", trace_path.display());

    Ok(())
}
