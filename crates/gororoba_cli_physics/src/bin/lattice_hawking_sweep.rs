//! Reframed Thesis 1: Lattice-Hawking Spectrum Sweep.
//!
//! Tests whether the D3Q19 LBM lattice spacing acts as a UV cutoff on
//! analog Hawking radiation.  Sweeps grid resolution and compares:
//!   - Ideal Planck/Bose-Einstein spectrum
//!   - Lattice-cutoff spectrum (sinc^2 window from D3Q19 dispersion)
//!   - Viscosity-cutoff spectrum (exponential damping at omega_visc)
//!
//! For each resolution N:
//!   1. Set up a 1D radial inflow profile (analog black hole)
//!   2. Find the acoustic horizon where |v| = c_s
//!   3. Compute the acoustic surface gravity kappa and Hawking temperature T_H
//!   4. Compute spectra and chi-squared deviations
//!   5. Extract effective temperatures from low-frequency fits
//!
//! PASS: Effective T from viscosity-cutoff spectrum remains bounded (does NOT
//!       converge to ideal T_H as N -> inf), proving viscosity limits the
//!       analog Hawking temperature.
//! FAIL: Effective T converges to ideal T_H, meaning the lattice cutoff
//!       disappears in the continuum limit.

use clap::Parser;
use gr_core::{
    acoustic_metric::{
        LBM_CS, acoustic_hawking_temperature, acoustic_horizon_1d, acoustic_surface_gravity,
        radial_inflow_profile,
    },
    lattice_hawking::{
        compute_spectra, compute_spectra_variable_nu, effective_temperature, spectral_chi_squared,
        viscosity_cutoff,
    },
};
use sign_imbalance::bridge::{ImbalanceViscosityBridge, SedenionField, ViscosityCouplingModel};
use std::{fs, path::Path};

#[derive(Parser, Debug)]
#[command(author, version, about = "Reframed T1: Lattice-Hawking spectrum sweep")]
struct Args {
    /// Grid resolutions to sweep (comma-separated)
    #[arg(long, default_value = "16,32,64,128")]
    grids: String,
    /// Kinematic viscosity in lattice units
    #[arg(long, default_value_t = 0.01)]
    nu: f64,
    /// Number of spectral bins
    #[arg(long, default_value_t = 200)]
    n_omega: usize,
    /// Inflow velocity amplitude
    #[arg(long, default_value_t = 0.3)]
    v0: f64,
    /// Reference radius for inflow profile
    #[arg(long, default_value_t = 20.0)]
    r0: f64,
    /// Power-law exponent (1=2D, 2=3D)
    #[arg(long, default_value_t = 1.0)]
    alpha: f64,
    /// Output directory
    #[arg(long, default_value = "data/lattice_hawking")]
    output_dir: String,
    /// Sedenion viscosity coupling model (constant, exponential, kubo_response)
    #[arg(long, default_value = "constant")]
    coupling_model: String,
    /// Coupling strength (lambda)
    #[arg(long, default_value_t = 2.0)]
    coupling_strength: f64,
    /// Sedenion noise perturbation amplitude
    #[arg(long, default_value_t = 0.0)]
    sedenion_noise: f64,
}

fn main() {
    let args = Args::parse();

    let grids: Vec<usize> = args
        .grids
        .split(',')
        .map(|s| s.trim().parse().expect("invalid grid size"))
        .collect();

    println!("Lattice-Hawking Spectrum Sweep");
    println!(
        "  grids={:?} nu={} n_omega={}",
        grids, args.nu, args.n_omega
    );
    println!("  v0={} r0={} alpha={}", args.v0, args.r0, args.alpha);
    println!(
        "  coupling={} lambda={} noise={}",
        args.coupling_model, args.coupling_strength, args.sedenion_noise
    );
    println!();

    // Header
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "N",
        "dx",
        "kappa",
        "T_H_ideal",
        "T_eff_lat",
        "T_eff_visc",
        "chi2_lat",
        "chi2_visc",
        "nu_horiz"
    );
    println!(
        "  {:->5}  {:->10}  {:->10}  {:->10}  {:->10}  {:->10}  {:->10}  {:->10}  {:->10}",
        "", "", "", "", "", "", "", "", ""
    );

    let mut results: Vec<SweepRow> = Vec::new();

    let bridge = ImbalanceViscosityBridge::new(16);

    for &n in &grids {
        let dx = 1.0 / n as f64;
        let dx_ref = 1.0 / 16.0;
        let scale_factor = dx_ref / dx;

        // Generate radial inflow profile (1D analog black hole)
        let (velocity_mag, grid_dx) =
            radial_inflow_profile(n, args.v0, args.r0, args.alpha, LBM_CS);

        // Find acoustic horizon
        let horizons = acoustic_horizon_1d(&velocity_mag, LBM_CS);
        if horizons.is_empty() {
            println!("  {:>5}  -- no acoustic horizon found, skipping --", n);
            continue;
        }

        // Use first horizon crossing
        let h_pos = horizons[0];
        let h_pos_idx = h_pos.round() as usize;
        let kappa = acoustic_surface_gravity(&velocity_mag, grid_dx, h_pos);
        let t_h = acoustic_hawking_temperature(kappa);

        if t_h < 1e-15 {
            println!("  {:>5}  -- T_H ~ 0, skipping --", n);
            continue;
        }

        // Generate Sedenion Field and local viscosity
        let mut field = SedenionField::uniform(n, 1, 1);
        let scaled_noise = args.sedenion_noise; // No scaling

        if args.sedenion_noise > 0.0 {
            for x in 0..n {
                let s = field.get_mut(x, 0, 0);
                let phase = 2.0 * std::f64::consts::PI * (x as f64) / (n as f64);
                s[1] = scaled_noise * phase.sin();
                s[3] = scaled_noise * (2.0 * phase).cos();
                s[5] = scaled_noise * (0.5 * phase).sin();
            }
        }

        let imbalance = field.local_imbalance_density(16);

        // Scale the effective lambda for Kubo/Exponential models
        let scaled_coupling = match args.coupling_model.as_str() {
            "exponential" => ViscosityCouplingModel::Exponential { 
                nu_base: args.nu * scale_factor.powi(2), 
                lambda: args.coupling_strength * scale_factor.powi(2)
            },
            "linear" => ViscosityCouplingModel::Linear { 
                nu_base: args.nu * scale_factor.powi(2), 
                alpha: args.coupling_strength
            },
            "power_law" => ViscosityCouplingModel::PowerLaw {
                nu_base: args.nu * scale_factor.powi(2),
                n: 2.0
            },
            "kubo_response" => {
                let mut kubo = ViscosityCouplingModel::kubo_default(args.nu * scale_factor.powi(2));
                if let ViscosityCouplingModel::KuboResponse { ref mut f_cd, .. } = kubo {
                    *f_cd *= 1.0 / scale_factor;
                }
                kubo
            },
            _ => ViscosityCouplingModel::Constant { nu_base: args.nu * scale_factor.powi(2) },
        };

        let nu_array = bridge.imbalance_to_viscosity_model(&imbalance, &scaled_coupling);
        let nu_horizon = nu_array[h_pos_idx];
        // Compute spectra at this resolution
        let (omegas, ideal, lattice, viscous) = if args.coupling_model == "constant" {
            compute_spectra(t_h, n, args.nu, args.n_omega)
        } else {
            compute_spectra_variable_nu(
                t_h,
                n,
                dx,
                &nu_array,
                &velocity_mag,
                h_pos_idx,
                args.n_omega,
            )
        };

        // Chi-squared deviations from ideal
        let chi2_lat = spectral_chi_squared(&ideal, &lattice, 1e-10);
        let chi2_visc = spectral_chi_squared(&ideal, &viscous, 1e-10);

        // Effective temperatures from low-frequency fits
        let omega_visc = viscosity_cutoff(nu_horizon, dx);
        let n_low = omegas.iter().filter(|&&w| w < omega_visc).count().max(5);
        let t_eff_lat = effective_temperature(&omegas[..n_low], &lattice[..n_low]);
        let t_eff_visc = effective_temperature(&omegas[..n_low], &viscous[..n_low]);

        println!(
            "  {:>5}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.4}  {:>10.4}  {:>10.6}",
            n, dx, kappa, t_h, t_eff_lat, t_eff_visc, chi2_lat, chi2_visc, nu_horizon
        );

        results.push(SweepRow {
            n,
            dx,
            kappa,
            t_h_ideal: t_h,
            t_eff_lattice: t_eff_lat,
            t_eff_viscous: t_eff_visc,
            chi2_lattice: chi2_lat,
            chi2_viscous: chi2_visc,
            omega_visc,
            nu_horizon,
        });
    }

    if results.is_empty() {
        println!("\n  ERROR: No valid configurations found.");
        return;
    }

    // Verdict: check if T_eff_viscous is bounded away from T_H_ideal as N grows
    let first = &results[0];
    let last = &results[results.len() - 1];

    let rel_dev_last = if last.t_h_ideal > 1e-15 {
        (last.t_eff_viscous - last.t_h_ideal).abs() / last.t_h_ideal
    } else {
        0.0
    };

    let rel_dev_first = if first.t_h_ideal > 1e-15 {
        (first.t_eff_viscous - first.t_h_ideal).abs() / first.t_h_ideal
    } else {
        0.0
    };

    let verdict = if rel_dev_last > 0.05 {
        "PASS (viscosity bounds analog Hawking temperature)"
    } else if (rel_dev_last - rel_dev_first).abs() < 0.01 {
        "INCONCLUSIVE (deviation stable but small)"
    } else {
        "FAIL (T_eff_visc converges to T_H_ideal)"
    };

    println!();
    println!("  omega_visc(finest) = {:.4}", last.omega_visc);
    println!("  nu_horizon(finest) = {:.6}", last.nu_horizon);
    println!(
        "  |T_visc - T_ideal|/T_ideal: coarsest={:.4}, finest={:.4}",
        rel_dev_first, rel_dev_last
    );
    println!("  Verdict: {}", verdict);

    let dir = Path::new(&args.output_dir);
    fs::create_dir_all(dir).expect("failed to create output dir");
    let toml_path = dir.join("results.toml");

    let mut toml = format!(
        "# Lattice-Hawking Spectrum Sweep\n# nu={} n_omega={} coupling={} lambda={} noise={} verdict=\"{}\"\n\n",
        args.nu,
        args.n_omega,
        args.coupling_model,
        args.coupling_strength,
        args.sedenion_noise,
        verdict
    );

    for row in &results {
        toml.push_str(&format!(
            "[[sweep]]\nN = {}\ndx = {:.8}\nkappa = {:.8}\nT_H_ideal = {:.8}\n\
             T_eff_lattice = {:.8}\nT_eff_viscous = {:.8}\n\
             chi2_lattice = {:.6}\nchi2_viscous = {:.6}\nomega_visc = {:.4}\nnu_horizon = {:.6}\n\n",
            row.n,
            row.dx,
            row.kappa,
            row.t_h_ideal,
            row.t_eff_lattice,
            row.t_eff_viscous,
            row.chi2_lattice,
            row.chi2_viscous,
            row.omega_visc,
            row.nu_horizon
        ));
    }

    fs::write(&toml_path, toml).expect("failed to write results.toml");
    println!("  Results written to {}", toml_path.display());
}

struct SweepRow {
    n: usize,
    dx: f64,
    kappa: f64,
    t_h_ideal: f64,
    t_eff_lattice: f64,
    t_eff_viscous: f64,
    chi2_lattice: f64,
    chi2_viscous: f64,
    omega_visc: f64,
    nu_horizon: f64,
}
