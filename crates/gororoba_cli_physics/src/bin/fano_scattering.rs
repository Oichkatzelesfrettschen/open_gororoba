//! fano-scattering: Ruan & Fan (2009) TCMT Fano resonance reproduction.
//!
//! Generates CSV data for all 5 figures from the paper:
//! - Figures 1-2: Analytical Fano lineshape (lossless/lossy)
//! - Figure 4: Single-channel MDM validation (Mie vs TCMT)
//! - Figure 5: Multi-channel MDM geometry
//!
//! Usage:
//!   fano-scattering --figure all --output-dir data/fano_scattering/
//!   fano-scattering --figure fig1 --output-dir data/fano_scattering/

use clap::{Parser, ValueEnum};
use optics_core::{
    CrossSections, FanoChannel, FanoDrudeParams, extract_fano_params,
    fano_cross_sections_normalized, mie_mdm_sweep, multi_channel_cross_sections,
    normalized_fano_c_sct, ruan_fan_mdm_fig4, ruan_fan_mdm_fig5,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Figure {
    /// Lossless analytical Fano (gamma_0=0)
    Fig1,
    /// Lossy analytical Fano (gamma_0=gamma)
    Fig2,
    /// Single-channel MDM validation
    Fig4,
    /// Multi-channel MDM geometry
    Fig5,
    /// All figures
    All,
}

#[derive(Parser)]
#[command(name = "fano-scattering")]
#[command(about = "Ruan & Fan (2009) TCMT Fano resonance reproduction")]
struct Args {
    /// Figure to generate
    #[arg(short, long, value_enum, default_value = "all")]
    figure: Figure,

    /// Output directory
    #[arg(short, long, default_value = "data/fano_scattering")]
    output_dir: PathBuf,
}

fn main() {
    let args = Args::parse();

    // Create output directory
    if let Err(e) = fs::create_dir_all(&args.output_dir) {
        eprintln!("Failed to create output directory: {}", e);
        std::process::exit(1);
    }

    match args.figure {
        Figure::Fig1 => generate_fig1(&args.output_dir),
        Figure::Fig2 => generate_fig2(&args.output_dir),
        Figure::Fig4 => generate_fig4(&args.output_dir),
        Figure::Fig5 => generate_fig5(&args.output_dir),
        Figure::All => {
            eprintln!("Generating all figures...");
            generate_fig1(&args.output_dir);
            generate_fig2(&args.output_dir);
            generate_fig4(&args.output_dir);
            generate_fig5(&args.output_dir);
            eprintln!("All figures generated successfully.");
        }
    }
}

/// Figure 1: Lossless analytical Fano
///
/// x = (omega - omega_0) / gamma in range [-10, 10]
/// phi = 0, pi/2, pi
/// gamma_0 = 0 (lossless)
fn generate_fig1(out_dir: &Path) {
    eprintln!("Generating Figure 1 (lossless analytical Fano)...");

    let n_points = 500;
    let x_min = -10.0;
    let x_max = 10.0;
    let phi_values = [0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI];

    let mut csv_lines = vec!["x,c_sct_phi0,c_sct_phi_pi2,c_sct_phi_pi".to_string()];

    for i in 0..n_points {
        let x = x_min + (x_max - x_min) * (i as f64) / ((n_points - 1) as f64);

        let c_sct_0 = normalized_fano_c_sct(x, phi_values[0], 0.0);
        let c_sct_pi2 = normalized_fano_c_sct(x, phi_values[1], 0.0);
        let c_sct_pi = normalized_fano_c_sct(x, phi_values[2], 0.0);

        csv_lines.push(format!("{},{},{},{}", x, c_sct_0, c_sct_pi2, c_sct_pi));
    }

    let file_path = out_dir.join("fig1_lossless_fano.csv");
    if let Err(e) = fs::write(&file_path, csv_lines.join("\n")) {
        eprintln!("Failed to write Figure 1 CSV: {}", e);
    } else {
        eprintln!("  -> {}", file_path.display());
    }
}

/// Figure 2: Lossy analytical Fano
///
/// x = (omega - omega_0) / gamma in range [-10, 10]
/// phi = 0, pi/2, pi
/// gamma_0 = gamma (equal radiative and non-radiative damping)
fn generate_fig2(out_dir: &Path) {
    eprintln!("Generating Figure 2 (lossy analytical Fano)...");

    let n_points = 500;
    let x_min = -10.0;
    let x_max = 10.0;
    let phi_values = [0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI];

    let mut csv_lines = vec!["x,c_sct_phi0,c_sct_phi_pi2,c_sct_phi_pi".to_string()];

    for i in 0..n_points {
        let x = x_min + (x_max - x_min) * (i as f64) / ((n_points - 1) as f64);

        // gamma_0 / gamma = 1.0 (equal damping)
        let c_sct_0 = normalized_fano_c_sct(x, phi_values[0], 1.0);
        let c_sct_pi2 = normalized_fano_c_sct(x, phi_values[1], 1.0);
        let c_sct_pi = normalized_fano_c_sct(x, phi_values[2], 1.0);

        csv_lines.push(format!("{},{},{},{}", x, c_sct_0, c_sct_pi2, c_sct_pi));
    }

    let file_path = out_dir.join("fig2_lossy_fano.csv");
    if let Err(e) = fs::write(&file_path, csv_lines.join("\n")) {
        eprintln!("Failed to write Figure 2 CSV: {}", e);
    } else {
        eprintln!("  -> {}", file_path.display());
    }
}

/// Figure 4: Single-channel MDM validation
///
/// Geometry from Ruan & Fan Fig. 4:
/// - rho1 = 0.285*lp (metal core)
/// - rho2 = lp (metal-dielectric interface)
/// - rho3 = 1.5*lp (outer boundary)
/// - Drude metal: omega_p, gamma_d
///
/// Generates two CSV files:
/// - fig4a_lossless.csv: lossless (gamma_d=0)
/// - fig4bcd_lossy.csv: lossy case with C_ext and C_abs
fn generate_fig4(out_dir: &Path) {
    eprintln!("Generating Figure 4 (single-channel MDM validation)...");

    // Ruan & Fan parameters for Fig. 4 (in natural units, lp = 2*pi*c/omega_p)
    let drude = FanoDrudeParams {
        omega_p: 1.0,
        gamma_d: 0.0, // Will be updated for lossy case
    };

    let n_points = 200;
    let omega_min = 0.1 * drude.omega_p;
    let omega_max = 0.2 * drude.omega_p;

    let omegas: Vec<f64> = (0..n_points)
        .map(|i| omega_min + (omega_max - omega_min) * (i as f64) / ((n_points - 1) as f64))
        .collect();

    // LOSSLESS CASE (gamma_d = 0)
    eprintln!("  Generating lossless sweep (l=0)...");
    let geom_ll = ruan_fan_mdm_fig4(&drude);

    // Do a Mie frequency sweep with Drude updates at each frequency
    let results_ll = mie_mdm_sweep(&geom_ll, 1, &drude, &omegas, 0);

    // Extract Fano channel parameters from the Mie sweep for l=0
    let fano_ch_ll = extract_fano_params(&omegas, &results_ll, 0);

    let mut csv_lines_ll = vec!["omega_norm,c_sct_mie,c_sct_tcmt".to_string()];

    for (i, &omega) in omegas.iter().enumerate() {
        let omega_norm = omega / drude.omega_p;

        let c_sct_mie = if !results_ll[i].channels.is_empty() {
            results_ll[i].channels[0].s_l.norm_sqr()
        } else {
            0.0
        };

        // TCMT cross-section from extracted Fano parameters
        let c_sct_tcmt = match &fano_ch_ll {
            Some(ch) => fano_cross_sections_normalized(ch, omega).c_sct,
            None => c_sct_mie, // fallback if extraction failed
        };

        csv_lines_ll.push(format!("{},{},{}", omega_norm, c_sct_mie, c_sct_tcmt));
    }

    let file_path_ll = out_dir.join("fig4a_lossless.csv");
    if let Err(e) = fs::write(&file_path_ll, csv_lines_ll.join("\n")) {
        eprintln!("Failed to write Figure 4a CSV: {}", e);
    } else {
        eprintln!("  -> {}", file_path_ll.display());
    }

    // LOSSY CASE (gamma_d = 0.001*omega_p)
    eprintln!("  Generating lossy sweep (l=0)...");
    let drude_lossy = FanoDrudeParams {
        omega_p: 1.0,
        gamma_d: 0.001,
    };
    let geom_lossy = ruan_fan_mdm_fig4(&drude_lossy);

    let results_lossy = mie_mdm_sweep(&geom_lossy, 1, &drude_lossy, &omegas, 0);
    let fano_ch_lossy = extract_fano_params(&omegas, &results_lossy, 0);

    let mut csv_lines_lossy = vec![
        "omega_norm,c_sct_mie,c_sct_tcmt,c_ext_mie,c_ext_tcmt,c_abs_mie,c_abs_tcmt".to_string(),
    ];

    for (i, &omega) in omegas.iter().enumerate() {
        let omega_norm = omega / drude_lossy.omega_p;

        let c_sct_mie = if !results_lossy[i].channels.is_empty() {
            results_lossy[i].channels[0].s_l.norm_sqr()
        } else {
            0.0
        };
        let c_ext_mie = results_lossy[i].cross_sections.c_ext;
        let c_abs_mie = results_lossy[i].cross_sections.c_abs;

        let (c_sct_tcmt, c_ext_tcmt, c_abs_tcmt) = match &fano_ch_lossy {
            Some(ch) => {
                let cs = fano_cross_sections_normalized(ch, omega);
                (cs.c_sct, cs.c_ext, cs.c_abs)
            }
            None => (c_sct_mie, c_ext_mie, c_abs_mie),
        };

        csv_lines_lossy.push(format!(
            "{},{},{},{},{},{},{}",
            omega_norm, c_sct_mie, c_sct_tcmt, c_ext_mie, c_ext_tcmt, c_abs_mie, c_abs_tcmt
        ));
    }

    let file_path_lossy = out_dir.join("fig4bcd_lossy.csv");
    if let Err(e) = fs::write(&file_path_lossy, csv_lines_lossy.join("\n")) {
        eprintln!("Failed to write Figure 4bcd CSV: {}", e);
    } else {
        eprintln!("  -> {}", file_path_lossy.display());
    }
}

/// Figure 5: Multi-channel MDM geometry
///
/// Geometry from Ruan & Fan Fig. 5:
/// - rho1 = 0.36*lp (metal core)
/// - rho2 = 0.73*lp (metal-dielectric interface)
/// - rho3 = lp (outer boundary)
/// - Drude with gamma_d = 0.001*omega_p
/// - Frequency range: 0.22 to 0.233 times omega_p
///
/// Generates CSV with total and per-channel contributions
fn generate_fig5(out_dir: &Path) {
    eprintln!("Generating Figure 5 (multi-channel MDM)...");

    let drude = FanoDrudeParams {
        omega_p: 1.0,
        gamma_d: 0.001,
    };

    let n_points = 200;
    let omega_min = 0.22 * drude.omega_p;
    let omega_max = 0.233 * drude.omega_p;
    let l_max = 2;

    let omegas: Vec<f64> = (0..n_points)
        .map(|i| omega_min + (omega_max - omega_min) * (i as f64) / ((n_points - 1) as f64))
        .collect();

    let geom = ruan_fan_mdm_fig5(&drude);

    // Mie frequency sweep with Drude updates
    let results = mie_mdm_sweep(&geom, 1, &drude, &omegas, l_max);

    // Extract Fano channel parameters for each angular momentum channel
    let mut fano_channels: Vec<FanoChannel> = Vec::new();
    for l in -l_max..=l_max {
        if let Some(ch) = extract_fano_params(&omegas, &results, l) {
            fano_channels.push(ch);
        }
    }

    let mut csv_header =
        "omega_norm,c_sct_total_mie,c_sct_total_tcmt,c_abs_total_mie,c_abs_total_tcmt".to_string();
    for l in -l_max..=l_max {
        csv_header.push_str(&format!(",c_sct_l{}_mie", l));
    }

    let mut csv_lines = vec![csv_header];

    for (i, &omega) in omegas.iter().enumerate() {
        let omega_norm = omega / drude.omega_p;

        let c_sct_total_mie = results[i].cross_sections.c_sct;
        let c_abs_total_mie = results[i].cross_sections.c_abs;

        // TCMT total cross-sections from extracted multi-channel parameters
        let tcmt_cs = if !fano_channels.is_empty() {
            multi_channel_cross_sections(&fano_channels, omega)
        } else {
            CrossSections {
                c_sct: c_sct_total_mie,
                c_abs: c_abs_total_mie,
                c_ext: c_sct_total_mie + c_abs_total_mie,
            }
        };

        let mut line = format!(
            "{},{},{},{},{}",
            omega_norm, c_sct_total_mie, tcmt_cs.c_sct, c_abs_total_mie, tcmt_cs.c_abs
        );

        // Per-channel contributions (l=0, +/-1, +/-2)
        for l in -l_max..=l_max {
            let c_sct_l = results[i]
                .channels
                .iter()
                .find(|ch| ch.l == l)
                .map(|ch| ch.s_l.norm_sqr())
                .unwrap_or(0.0);
            line.push_str(&format!(",{}", c_sct_l));
        }

        csv_lines.push(line);
    }

    let file_path = out_dir.join("fig5_multi.csv");
    if let Err(e) = fs::write(&file_path, csv_lines.join("\n")) {
        eprintln!("Failed to write Figure 5 CSV: {}", e);
    } else {
        eprintln!("  -> {}", file_path.display());
    }
}
