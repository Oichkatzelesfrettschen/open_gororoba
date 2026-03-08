//! Ephemeris-Driven Holographic Entropy Trap Scan
//!
//! Lifts planetary positions into sedenion (16D) algebra, computes box-kite
//! alignment spectrum at each time step, and detects temporal patterns in
//! zero-divisor subspace projections.
//!
//! Uses JPL DE440 ephemeris if available, otherwise falls back to simplified
//! Keplerian orbits for CI/testing.
//!
//! This is a *structural/exploratory* tool. C-010 (Holographic Entropy Trap)
//! was previously falsified; new claims are computational, NOT physical.
//!
//! # Usage
//! ```text
//! cargo run --bin entropy-trap-scan -- --years 10 --output scan.csv
//! ```

use algebra_analysis::{
    boxkite_alignment::{AlignmentScanPoint, alignment_spectrum},
    boxkites::find_box_kites,
    sedenion_lifting::{compute_triple_associator, lift_positions_to_sedenion},
};
use clap::Parser;
use std::path::PathBuf;

/// NAIF body IDs for inner solar system planets.
const PLANET_IDS: [i32; 4] = [1, 2, 3, 4]; // Mercury, Venus, Earth-Moon, Mars
const PLANET_NAMES: [&str; 4] = ["Mercury", "Venus", "Earth", "Mars"];

/// J2000.0 in Julian Ephemeris Date.
const J2000_JED: f64 = 2451545.0;

#[derive(Parser)]
#[command(
    name = "entropy-trap-scan",
    about = "Sedenion box-kite alignment scan over planetary ephemeris"
)]
struct Args {
    /// Number of years to scan (default 100).
    #[arg(long, default_value_t = 100.0)]
    years: f64,

    /// Step size in days (default 1.0).
    #[arg(long, default_value_t = 1.0)]
    step_days: f64,

    /// Output CSV file path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Path to DE440 .bsp file.
    #[arg(long, default_value = "data/external/de440s.bsp")]
    ephemeris: PathBuf,
}

/// Simplified Keplerian orbit for fallback (circular approximation).
struct KeplerianPlanet {
    semi_major_au: f64,
    period_days: f64,
    phase_rad: f64,
    inclination_rad: f64,
}

/// Fallback: approximate inner planet positions with circular Keplerian orbits.
fn keplerian_planets() -> [KeplerianPlanet; 4] {
    [
        KeplerianPlanet {
            semi_major_au: 0.387,
            period_days: 87.97,
            phase_rad: 0.0,
            inclination_rad: 7.0_f64.to_radians(),
        },
        KeplerianPlanet {
            semi_major_au: 0.723,
            period_days: 224.7,
            phase_rad: 1.2,
            inclination_rad: 3.4_f64.to_radians(),
        },
        KeplerianPlanet {
            semi_major_au: 1.000,
            period_days: 365.25,
            phase_rad: 2.5,
            inclination_rad: 0.0,
        },
        KeplerianPlanet {
            semi_major_au: 1.524,
            period_days: 687.0,
            phase_rad: 4.1,
            inclination_rad: 1.85_f64.to_radians(),
        },
    ]
}

fn keplerian_position(planet: &KeplerianPlanet, days_from_j2000: f64) -> [f64; 3] {
    let mean_anomaly =
        planet.phase_rad + 2.0 * std::f64::consts::PI * days_from_j2000 / planet.period_days;
    let r = planet.semi_major_au;
    let x = r * mean_anomaly.cos();
    let y = r * mean_anomaly.sin() * planet.inclination_rad.cos();
    let z = r * mean_anomaly.sin() * planet.inclination_rad.sin();
    [x, y, z]
}

/// Enum to abstract over ephemeris sources.
enum EphemerisSource {
    Spk(data_core::spice::spk::SpkReader),
    Keplerian([KeplerianPlanet; 4]),
}

impl EphemerisSource {
    fn get_positions(&self, jed: f64) -> [[f64; 3]; 4] {
        match self {
            EphemerisSource::Spk(spk) => {
                let mut positions = [[0.0; 3]; 4];
                for (i, &id) in PLANET_IDS.iter().enumerate() {
                    if let Some(state) = spk.compute_state(0, id, jed) {
                        // Convert from km to AU for normalization stability
                        let au = 1.496e8; // km per AU
                        positions[i] = [
                            state.position[0] / au,
                            state.position[1] / au,
                            state.position[2] / au,
                        ];
                    }
                }
                positions
            }
            EphemerisSource::Keplerian(planets) => {
                let days = jed - J2000_JED;
                let mut positions = [[0.0; 3]; 4];
                for (i, planet) in planets.iter().enumerate() {
                    positions[i] = keplerian_position(planet, days);
                }
                positions
            }
        }
    }

    fn label(&self) -> &str {
        match self {
            EphemerisSource::Spk(_) => "DE440",
            EphemerisSource::Keplerian(_) => "Keplerian",
        }
    }
}

fn main() {
    let args = Args::parse();

    // Try to load DE440 ephemeris
    let source = if args.ephemeris.exists() {
        match data_core::spice::daf::DafReader::open(&args.ephemeris) {
            Ok(daf) => match data_core::spice::spk::SpkReader::new(daf) {
                Ok(spk) => {
                    eprintln!("Loaded ephemeris: {}", args.ephemeris.display());
                    EphemerisSource::Spk(spk)
                }
                Err(e) => {
                    eprintln!(
                        "WARNING: Failed to parse SPK: {}. Using Keplerian fallback.",
                        e
                    );
                    EphemerisSource::Keplerian(keplerian_planets())
                }
            },
            Err(e) => {
                eprintln!(
                    "WARNING: Failed to open DAF: {}. Using Keplerian fallback.",
                    e
                );
                EphemerisSource::Keplerian(keplerian_planets())
            }
        }
    } else {
        eprintln!(
            "WARNING: Ephemeris file not found: {}. Using Keplerian fallback.",
            args.ephemeris.display()
        );
        EphemerisSource::Keplerian(keplerian_planets())
    };

    eprintln!("=== Entropy Trap Scan ===");
    eprintln!("Source: {}", source.label());
    eprintln!(
        "Duration: {:.1} years, step: {:.1} days",
        args.years, args.step_days
    );
    eprintln!("Planets: {}", PLANET_NAMES.join(", "));

    // Precompute box-kites (only once -- they're structural constants of the sedenion algebra)
    let boxkites = find_box_kites(16, 1e-10);
    eprintln!("Box-kites: {} (expected 7)", boxkites.len());

    let total_days = args.years * 365.25;
    let n_steps = (total_days / args.step_days) as usize;
    let jed_start = J2000_JED;

    // CSV header
    let header = format!(
        "jed,days_from_j2000,{},dominant_bk,assoc_norm",
        (0..boxkites.len())
            .map(|i| format!("bk{}", i))
            .collect::<Vec<_>>()
            .join(",")
    );

    let mut csv_lines = vec![header.clone()];
    println!("{}", header);

    let mut scan_points: Vec<AlignmentScanPoint> = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let days = step as f64 * args.step_days;
        let jed = jed_start + days;

        let positions = source.get_positions(jed);

        // Lift to sedenion
        let lift = lift_positions_to_sedenion(&positions);

        // Compute alignment spectrum
        let spectrum = alignment_spectrum(&lift.element, &boxkites);

        // Compute associator norm: use two time-offset lifts as the triple (a, b, a)
        // For single-step analysis, use the lift against a reference (e1 unit)
        let mut ref_element = vec![0.0; 16];
        ref_element[1] = 1.0;
        let (_, assoc_norm) = compute_triple_associator(&lift.element, &ref_element, &lift.element);

        let weights_str = spectrum
            .weights
            .iter()
            .map(|w| format!("{:.6}", w))
            .collect::<Vec<_>>()
            .join(",");

        let line = format!(
            "{:.1},{:.1},{},{},{:.8}",
            jed, days, weights_str, spectrum.dominant_bk, assoc_norm
        );
        println!("{}", line);
        csv_lines.push(line);

        scan_points.push(AlignmentScanPoint {
            jed,
            weights: spectrum.weights,
            dominant_bk: spectrum.dominant_bk,
            assoc_norm,
        });

        if step % 10000 == 0 && step > 0 {
            eprintln!(
                "  ... step {}/{} ({:.1} years)",
                step,
                n_steps,
                days / 365.25
            );
        }
    }

    // Write CSV if output path specified
    if let Some(path) = &args.output {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, csv_lines.join("\n") + "\n").expect("Failed to write CSV");
        eprintln!("Wrote: {}", path.display());
    }

    // Summary statistics
    eprintln!("\n=== Summary ===");
    eprintln!("Total steps: {}", scan_points.len());

    // Dominant box-kite distribution
    let mut bk_counts = vec![0usize; boxkites.len()];
    for point in &scan_points {
        bk_counts[point.dominant_bk] += 1;
    }
    eprintln!("Dominant BK distribution:");
    for (i, count) in bk_counts.iter().enumerate() {
        let pct = 100.0 * *count as f64 / scan_points.len() as f64;
        eprintln!("  BK{}: {} ({:.1}%)", i, count, pct);
    }

    // Associator norm statistics
    let assoc_norms: Vec<f64> = scan_points.iter().map(|p| p.assoc_norm).collect();
    let mean_assoc: f64 = assoc_norms.iter().sum::<f64>() / assoc_norms.len() as f64;
    let max_assoc = assoc_norms.iter().cloned().fold(0.0f64, f64::max);
    let min_assoc = assoc_norms.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!(
        "Associator norm: mean={:.6}, min={:.6}, max={:.6}",
        mean_assoc, min_assoc, max_assoc
    );

    eprintln!("\n=== Verdict ===");
    eprintln!(
        "Structural scan complete. {} data points generated.",
        scan_points.len()
    );
    eprintln!("NOTE: This is exploratory analysis, not a physical prediction.");
}
