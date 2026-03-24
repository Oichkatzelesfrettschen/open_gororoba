use std::{fs::File, io::Write, path::Path};

/// Simulates the ZD Harmonic Forcing on Dark Matter Halos
/// Based on the breakthrough from the Unified Monograph:
///
/// "Hypothesis 4.1 (ZD Harmonic Forcing). If dark matter density is governed by
/// Cayley-Dickson algebraic structure at dimension D, then the rotation velocity
/// exhibits harmonic modulations corresponding to the ZD graph spectrum."
///
/// For Sedenions (D=16), the ZD graph has 7 distinct box-kite components,
/// inducing 7 fundamental spatial harmonics.
/// For Pathions (D=32), the Flat Band Fraction jumps to 4/7 (The Pathion Anomaly),
/// altering the harmonic density and coupling strength.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌌 Initializing ZD Harmonic Forcing Profile Generator...");
    println!("   Computing Halo Velocity Modulations from Sedenion & Pathion ZD Spectra.");

    let out_dir = Path::new("data/results");
    std::fs::create_dir_all(out_dir)?;

    let out_path = out_dir.join("zd_harmonic_forcing_halo_profiles.csv");
    let mut file = File::create(&out_path)?;
    writeln!(
        file,
        "radius_x,v_nfw,v_zd_sedenion,v_zd_pathion,v_total_16d,v_total_32d"
    )?;

    // NFW Halo Parameters
    let v_max = 200.0; // km/s
    let r_s = 10.0; // scale radius in kpc

    // ZD Forcing Parameters
    let alpha_zd = 15.0; // km/s amplitude of the ZD modulation

    // Sedenion (16D) ZD Harmonic Modes
    // 7 box-kites -> 7 harmonics
    let n_modes_16d = 7;
    let base_k_16d = 2.0 * std::f64::consts::PI / (n_modes_16d as f64);

    // Pathion (32D) ZD Harmonic Modes
    // Pathions have a completely different ZD graph structure.
    // While 16D has FBF = 1/2, 32D has FBF = 4/7 (Anomaly).
    // The number of discrete components (modes) in Pathions scales as 15
    // (from the 15 Fano planes in the PG(3,2) geometry).
    let n_modes_32d = 15;
    let base_k_32d = 2.0 * std::f64::consts::PI / (n_modes_32d as f64);

    let steps = 1000;
    let max_radius = 50.0; // kpc

    for i in 1..=steps {
        let r = (i as f64) / (steps as f64) * max_radius;
        let x = r / r_s; // dimensionless radius

        // Base NFW Velocity Profile (Simplified)
        // V^2(r) ~ (ln(1+x) - x/(1+x)) / x
        let enclosure = (1.0 + x).ln() - (x / (1.0 + x));
        let v_nfw = v_max * (enclosure / x).sqrt();

        // 16D Sedenion Forcing (Grows with radius, peaks at outer halo ~3-5 r/r_s)
        let mut v_zd_sedenion = 0.0;
        let spatial_envelope = if x > 1.0 { (x - 1.0).ln() } else { 0.0 }; // Grows in outer halo

        for n in 1..=n_modes_16d {
            let k_n = base_k_16d * (n as f64);
            // Coupling decreases slightly for higher modes, damped by the 1/2 Flat Band Fraction
            let c_n = 1.0 / (n as f64) * 0.5;
            v_zd_sedenion += c_n * (k_n * x).sin();
        }
        v_zd_sedenion *= alpha_zd * spatial_envelope;

        // 32D Pathion Forcing
        let mut v_zd_pathion = 0.0;
        for n in 1..=n_modes_32d {
            let k_n = base_k_32d * (n as f64);
            // Stronger coupling due to the Pathion Anomaly (4/7 FBF)
            let c_n = 1.0 / (n as f64) * (4.0 / 7.0);
            v_zd_pathion += c_n * (k_n * x).sin();
        }
        v_zd_pathion *= alpha_zd * spatial_envelope;

        let v_total_16d = v_nfw + v_zd_sedenion;
        let v_total_32d = v_nfw + v_zd_pathion;

        writeln!(
            file,
            "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            x, v_nfw, v_zd_sedenion, v_zd_pathion, v_total_16d, v_total_32d
        )?;
    }

    println!(
        "✅ ZD Harmonic Forcing Profiles generated at {}",
        out_path.display()
    );
    println!("   - 16D Forcing utilizes the 7-mode Box-Kite spectrum (FBF = 1/2).");
    println!(
        "   - 32D Forcing utilizes the 15-mode PG(3,2) spectrum with the (FBF = 4/7) Pathion Anomaly amplification."
    );

    Ok(())
}
