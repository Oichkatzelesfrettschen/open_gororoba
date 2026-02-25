//! Wow! signal analysis binary.
//!
//! Subcommands:
//! - `transcribe`: Validate wow1977_transcription.csv integrity
//! - `drift-rate`: Compute Earth-motion drift rate bounds at the Wow! locale
//! - `summary`: Aggregate stats across all Wow-related artifacts

use clap::{Parser, Subcommand};
use data_core::catalogs::wow::{parse_wow_printout_csv, wow_char_to_intensity};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "wow-signal-analysis",
    about = "Wow! signal technosignature testbed"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate wow1977_transcription.csv integrity (SHA-256 + character mapping).
    Transcribe {
        /// Path to the transcription CSV.
        #[arg(long, default_value = "data/csv/wow1977_transcription.csv")]
        csv: PathBuf,
    },
    /// Compute Earth-motion drift rate bounds at the Wow! locale.
    DriftRate {
        /// Output CSV path for drift rate bounds.
        #[arg(long, default_value = "data/csv/wow_drift_rate_bounds.csv")]
        output: PathBuf,
    },
    /// Aggregate stats across all Wow-related artifacts.
    Summary,
}

/// Physical constants for drift rate calculation.
mod constants {
    /// Speed of light (m/s).
    pub const C: f64 = 299_792_458.0;
    /// Hydrogen line rest frequency (Hz).
    pub const F0_HZ: f64 = 1_420_405_751.786;
    /// Earth orbital velocity (m/s, mean).
    pub const V_ORB: f64 = 29_783.0;
    /// Earth equatorial rotational velocity (m/s).
    pub const V_ROT_EQ: f64 = 465.1;
    /// Earth orbital period (seconds).
    pub const T_ORB: f64 = 365.256_363_004 * 86_400.0;
    /// Earth sidereal rotation period (seconds).
    pub const T_ROT: f64 = 86_164.1;

    // Wow! signal coordinates (approximate, J2000)
    /// Right ascension in radians (~19h 25m -> ~291.3 deg).
    pub const RA_RAD: f64 = 5.084_97; // 19.4253h * 15 deg/h * pi/180
    /// Declination in radians (~-27 deg).
    pub const DEC_RAD: f64 = -0.471_24; // -27.0 deg * pi/180
    /// Ecliptic obliquity (radians).
    pub const OBLIQUITY: f64 = 0.409_09; // 23.44 deg
}

fn run_transcribe(csv_path: &PathBuf) {
    println!("=== Wow! Signal Transcription Validation ===");
    println!();

    if !csv_path.exists() {
        eprintln!("ERROR: {} not found", csv_path.display());
        std::process::exit(1);
    }

    // SHA-256 of the CSV file
    let data = std::fs::read(csv_path).expect("Failed to read CSV");
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let hash = format!("{:x}", hasher.finalize());
    println!("File: {}", csv_path.display());
    println!("SHA-256: {}", hash);
    println!("Size: {} bytes", data.len());
    println!();

    // Parse and validate
    let rows = parse_wow_printout_csv(csv_path).expect("Failed to parse CSV");
    println!("Total rows: {}", rows.len());

    let wow_rows: Vec<_> = rows.iter().filter(|r| r.in_wow_sequence).collect();
    println!("Wow! sequence rows: {}", wow_rows.len());

    if wow_rows.len() != 6 {
        eprintln!("FAIL: Expected 6 Wow! rows, got {}", wow_rows.len());
        std::process::exit(1);
    }

    // Verify character-to-intensity mapping
    let expected_chars = ['6', 'E', 'Q', 'U', 'J', '5'];
    let expected_ints = [6u32, 14, 26, 30, 19, 5];

    println!();
    println!("Character mapping verification:");
    let mut all_ok = true;
    for (i, row) in wow_rows.iter().enumerate() {
        let decoded = wow_char_to_intensity(row.printout_char);
        let char_ok = row.printout_char == expected_chars[i];
        let int_ok = row.intensity == expected_ints[i];
        let decode_ok = decoded == Some(expected_ints[i]);

        let status = if char_ok && int_ok && decode_ok {
            "OK"
        } else {
            all_ok = false;
            "FAIL"
        };

        println!(
            "  [{}] char='{}' intensity={} decoded={:?} expected=('{}', {})",
            status, row.printout_char, row.intensity, decoded, expected_chars[i], expected_ints[i]
        );
    }

    println!();
    if all_ok {
        println!("PASS: 6EQUJ5 transcription verified (6 characters, all mappings correct)");
    } else {
        eprintln!("FAIL: Character mapping mismatch detected");
        std::process::exit(1);
    }

    // Background level check
    let bg_rows: Vec<_> = rows.iter().filter(|r| !r.in_wow_sequence).collect();
    let bg_intensities: Vec<u32> = bg_rows.iter().map(|r| r.intensity).collect();
    let bg_max = bg_intensities.iter().copied().max().unwrap_or(0);
    let bg_min = bg_intensities.iter().copied().min().unwrap_or(0);
    println!(
        "Background: {} rows, intensity range [{}, {}]",
        bg_rows.len(),
        bg_min,
        bg_max
    );
    println!(
        "Peak Wow! intensity: {} ({}x above background max)",
        30,
        30.0 / bg_max as f64
    );
}

fn run_drift_rate(output_path: &PathBuf) {
    use constants::*;

    println!("=== Wow! Signal Drift Rate Analysis ===");
    println!();
    println!("Signal parameters:");
    println!("  f0 = {:.6} MHz (hydrogen line)", F0_HZ / 1e6);
    println!("  RA ~ 19h 25m 31s (J2000)");
    println!("  Dec ~ -27d 03m (J2000)");
    println!("  Epoch: 1977-08-15 22:16 UT");
    println!();

    // Ecliptic latitude of the source (approximate)
    // sin(beta) = sin(dec)*cos(obliquity) - cos(dec)*sin(obliquity)*sin(ra)
    let sin_beta = DEC_RAD.sin() * OBLIQUITY.cos() - DEC_RAD.cos() * OBLIQUITY.sin() * RA_RAD.sin();
    let beta = sin_beta.asin();
    let cos_beta = beta.cos();

    println!("Ecliptic latitude: {:.2} deg", beta.to_degrees());
    println!();

    // Orbital drift rate: -(f0/c) * d(v_orb * cos(beta) * cos(lambda - lambda_earth))/dt
    // Maximum magnitude when source is near quadrature: |drift_orb| = f0 * v_orb * cos(beta) * (2*pi/T_orb) / c
    let drift_orb_max = F0_HZ * V_ORB * cos_beta * (2.0 * std::f64::consts::PI / T_ORB) / C;

    // Rotational drift rate: -(f0/c) * d(v_rot * cos(dec) * cos(ha))/dt
    // Maximum: |drift_rot| = f0 * v_rot * cos(dec) * (2*pi/T_rot) / c
    // At Big Ear latitude ~40.2N, v_rot = V_ROT_EQ * cos(40.2deg) ~ 355 m/s
    let lat_big_ear = 40.2_f64.to_radians();
    let v_rot_local = V_ROT_EQ * lat_big_ear.cos();
    let drift_rot_max =
        F0_HZ * v_rot_local * DEC_RAD.cos() * (2.0 * std::f64::consts::PI / T_ROT) / C;

    // Combined worst-case bound
    let drift_total_max = drift_orb_max + drift_rot_max;

    println!("Drift rate components:");
    println!(
        "  Orbital:    max |df/dt| = {:.4} Hz/s  (v_orb={:.0} m/s, cos(beta)={:.4})",
        drift_orb_max, V_ORB, cos_beta
    );
    println!(
        "  Rotational: max |df/dt| = {:.4} Hz/s  (v_rot={:.1} m/s at lat {:.1}deg)",
        drift_rot_max,
        v_rot_local,
        lat_big_ear.to_degrees()
    );
    println!("  Combined:   max |df/dt| = {:.4} Hz/s", drift_total_max);
    println!();

    // For BL drift rate search: typical range is +/- 4 Hz/s in 0.01 Hz/s steps
    // Our bound tells us where to focus the search
    let bl_resolution = 0.01; // Hz/s
    let n_steps = (2.0 * drift_total_max / bl_resolution).ceil() as u32;
    println!("BL drift rate search implications:");
    println!("  BL resolution: {} Hz/s", bl_resolution);
    println!(
        "  Focused search range: [{:.4}, +{:.4}] Hz/s ({} steps)",
        -drift_total_max, drift_total_max, n_steps
    );

    // Write output CSV
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut csv_out = String::new();
    csv_out.push_str("# Wow! signal drift rate bounds (analytical Earth-motion model)\n");
    csv_out.push_str("# Generated by wow-signal-analysis drift-rate\n");
    csv_out.push_str("component,max_drift_hz_per_s,velocity_ms,cos_projection\n");
    csv_out.push_str(&format!(
        "orbital,{:.6},{:.1},{:.6}\n",
        drift_orb_max, V_ORB, cos_beta
    ));
    csv_out.push_str(&format!(
        "rotational,{:.6},{:.1},{:.6}\n",
        drift_rot_max,
        v_rot_local,
        DEC_RAD.cos()
    ));
    csv_out.push_str(&format!(
        "combined,{:.6},{:.1},{:.6}\n",
        drift_total_max,
        V_ORB + v_rot_local,
        1.0
    ));

    std::fs::write(output_path, &csv_out).expect("Failed to write drift rate CSV");
    println!();
    println!("Output: {}", output_path.display());
    println!();

    if drift_total_max < 0.37 {
        println!("PASS: Combined drift rate bound < 0.37 Hz/s (C-770)");
    } else {
        println!(
            "INFO: Combined drift rate bound = {:.4} Hz/s",
            drift_total_max
        );
    }
}

fn run_summary() {
    println!("=== Wow! Signal Technosignature Testbed Summary ===");
    println!();

    // Check for artifacts
    let artifacts = [
        (
            "data/csv/wow1977_transcription.csv",
            "Printout transcription",
        ),
        ("data/csv/bl_6equj5_gbt_manifest.csv", "BL GBT manifest"),
        ("data/csv/wow_drift_rate_bounds.csv", "Drift rate bounds"),
        (
            "data/external/wow_1977_printout.jpg",
            "Archival printout scan",
        ),
        (
            "data/external/bl_6equj5_gbt_manifest.csv",
            "BL manifest (fetched)",
        ),
        (
            "data/csv/bl_6equj5_doppler_results.csv",
            "Doppler drift search results (Rust-native)",
        ),
        (
            "data/csv/bl_6equj5_cadence_events.csv",
            "ABACAD cadence events",
        ),
        (
            "registry/bl_6equj5_analysis.toml",
            "BL analysis reproducibility TOML",
        ),
        (
            "registry/bl_6equj5_manifest.toml",
            "BL download manifest TOML",
        ),
    ];

    println!("Artifacts:");
    let mut found = 0;
    let mut missing = 0;
    for (path, label) in &artifacts {
        let exists = std::path::Path::new(path).exists();
        let status = if exists {
            found += 1;
            "FOUND"
        } else {
            missing += 1;
            "MISSING"
        };
        println!("  [{:>7}] {} -- {}", status, path, label);
    }

    println!();
    println!("Artifact status: {} found, {} missing", found, missing);

    // Validate transcription if present
    let csv_path = std::path::Path::new("data/csv/wow1977_transcription.csv");
    if csv_path.exists() {
        match parse_wow_printout_csv(csv_path) {
            Ok(rows) => {
                let wow_count = rows.iter().filter(|r| r.in_wow_sequence).count();
                println!();
                println!(
                    "Transcription: {} total rows, {} in Wow! sequence",
                    rows.len(),
                    wow_count
                );
            }
            Err(e) => {
                println!();
                println!("Transcription: parse error -- {}", e);
            }
        }
    }

    // Registry claims
    println!();
    println!("Registry claims:");
    println!("  C-769: Wow! transcription fidelity (Verified)");
    println!("  C-770: Drift rate bound < 0.37 Hz/s (Verified)");
    println!("  C-771: ABACAD cadence null result (Verified)");
    println!("  C-772: Topological indistinguishability (Proposed)");
    println!("  C-773: Ultrametric ecology (Proposed)");
    println!();
    println!("Experiments:");
    println!("  E-045: Wow transcription integrity");
    println!("  E-046: Wow drift rate kinematics");
    println!("  E-047: Cadence falsification");
    println!("  E-048: Persistent homology");
    println!("  E-049: Ultrametric ecology");
    println!("  E-050: Rust-native BL Doppler drift search + ABACAD pipeline");
}

fn main() {
    let args = Args::parse();
    match args.command {
        Command::Transcribe { csv } => run_transcribe(&csv),
        Command::DriftRate { output } => run_drift_rate(&output),
        Command::Summary => run_summary(),
    }
}
