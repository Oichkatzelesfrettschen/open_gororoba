//! Harmonic Halo Rotation Curve Generator
//!
//! Computes v_circ(r) for an NFW halo with and without harmonic halo
//! modulation from sedenion box-kite topology. Outputs CSV for validation
//! and visualization.
//!
//! # Usage
//! ```text
//! cargo run --bin harmonic-halo -- rotation-curve \
//!   --m200 1e12 --alpha-zd 0.05 --csv rotation_curve.csv
//! ```

use clap::Args;
use cosmology_core::{
    harmonic_halos::{HarmonicHaloConfig, v_circ_nfw, v_circ_with_halos},
    nfw_utils::nfw_params_from_mass,
};
use std::io::Write;

#[derive(Args)]
pub struct Cli {
    /// Halo virial mass M_200 in solar masses.
    #[arg(long, default_value = "1e12")]
    m200: f64,

    /// NFW concentration (0 = use Dutton-Maccio CMR).
    #[arg(long, default_value = "0")]
    c200: f64,

    /// Redshift.
    #[arg(long, default_value = "0.0")]
    z: f64,

    /// Cayley-Dickson dimension (16, 32, 64) for algebraic corrections.
    #[arg(long, default_value = "16")]
    cd_dim: usize,

    /// ZD forcing strength (0.0 = disabled).
    #[arg(long, default_value = "0.0")]
    alpha_zd: f64,

    /// Number of box-kite harmonic modes (1..=7).
    #[arg(long, default_value = "7")]
    n_modes: usize,

    /// Minimum radius in kpc.
    #[arg(long, default_value = "0.1")]
    r_min: f64,

    /// Maximum radius in kpc.
    #[arg(long, default_value = "100.0")]
    r_max: f64,

    /// Number of radial sample points.
    #[arg(long, default_value = "500")]
    n_points: usize,

    /// Output CSV path.
    #[arg(long, default_value = "harmonic_halo_rotation_curve.csv")]
    csv: String,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    let nfw = nfw_params_from_mass(cli.m200, cli.z);
    let c200 = if cli.c200 > 0.0 { cli.c200 } else { nfw.c200 };
    let r_s = nfw.r200_kpc / c200;

    eprintln!("=== Harmonic Halo Rotation Curve ===");
    eprintln!(
        "M_200 = {:.3e} Msun, c_200 = {:.2}, r_s = {:.2} kpc",
        cli.m200, c200, r_s
    );
    eprintln!("r_200 = {:.2} kpc, z = {:.4}", nfw.r200_kpc, cli.z);
    eprintln!(
        "alpha_zd = {}, n_modes = {}, cd_dim = {}",
        cli.alpha_zd, cli.n_modes, cli.cd_dim
    );

    let config = HarmonicHaloConfig::new_cd(cli.alpha_zd, cli.n_modes, r_s, cli.cd_dim);

    let mut file = std::fs::File::create(&cli.csv)?;
    writeln!(
        file,
        "r_kpc,v_circ_nfw_km_s,v_circ_halo_km_s,modulation_factor,delta_v_percent"
    )?;

    let log_r_min = cli.r_min.ln();
    let log_r_max = cli.r_max.ln();

    for i in 0..cli.n_points {
        let frac = i as f64 / (cli.n_points - 1).max(1) as f64;
        let r = (log_r_min + frac * (log_r_max - log_r_min)).exp();

        let v_nfw = v_circ_nfw(r, cli.m200, cli.z);
        let v_halo = v_circ_with_halos(r, cli.m200, c200, cli.z, &config);
        let modulation = cosmology_core::harmonic_halos::harmonic_halo_modulation(r, &config);
        let delta_v_pct = if v_nfw > 0.0 {
            (v_halo - v_nfw) / v_nfw * 100.0
        } else {
            0.0
        };

        writeln!(
            file,
            "{:.6},{:.6},{:.6},{:.8},{:.6}",
            r, v_nfw, v_halo, modulation, delta_v_pct
        )?;
    }

    eprintln!("Wrote {} points to {}", cli.n_points, cli.csv);

    // Print summary statistics
    let v_peak_nfw = (0..cli.n_points)
        .map(|i| {
            let frac = i as f64 / (cli.n_points - 1).max(1) as f64;
            let r = (log_r_min + frac * (log_r_max - log_r_min)).exp();
            v_circ_nfw(r, cli.m200, cli.z)
        })
        .fold(0.0_f64, f64::max);

    eprintln!("v_peak(NFW) = {:.2} km/s", v_peak_nfw);

    if cli.alpha_zd > 0.0 {
        let max_delta: f64 = (0..cli.n_points)
            .map(|i| {
                let frac = i as f64 / (cli.n_points - 1).max(1) as f64;
                let r = (log_r_min + frac * (log_r_max - log_r_min)).exp();
                let v_nfw = v_circ_nfw(r, cli.m200, cli.z);
                let v_halo = v_circ_with_halos(r, cli.m200, c200, cli.z, &config);
                if v_nfw > 0.0 {
                    ((v_halo - v_nfw) / v_nfw * 100.0).abs()
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, f64::max);

        eprintln!("max |delta_v| = {:.4}%", max_delta);
    }

    Ok(())
}
