//! Trimmed stacking robustness test for per-galaxy DFT coefficients.
//!
//! Tests whether the extreme kurtosis (~284) in per-galaxy Fourier power
//! means the stacked null result is driven by outlier galaxies.
//!
//! Computes the DFT mean at multiple trimming levels (0%, 5%, 10%, 20%, 50%)
//! and checks whether SNR changes as outliers are removed.
//!
//! If the null is robust: trimmed and full stacks give similar SNR.
//! If outliers drive the null: trimmed stack shows different spectral properties.
//!
//! Also computes the median-based SNR (maximum robustness) as a baseline.
//!
//! Reference: Contrarian hypothesis D4.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "trimmed-stacking-robustness")]
#[command(about = "Trimmed mean and median DFT for outlier robustness test")]
struct Cli {
    /// Per-galaxy DFT CSV.
    #[arg(long)]
    galaxies_csv: PathBuf,
}

fn trimmed_mean_complex(
    values: &mut [(f64, f64)],
    trim_frac: f64,
) -> (f64, f64) {
    let n = values.len();
    let trim_count = (n as f64 * trim_frac) as usize;
    let start = trim_count;
    let end = n - trim_count;
    if start >= end {
        return (0.0, 0.0);
    }

    // Sort by power (re^2 + im^2) and trim extremes.
    values.sort_by(|a, b| {
        let pa = a.0 * a.0 + a.1 * a.1;
        let pb = b.0 * b.0 + b.1 * b.1;
        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let trimmed = &values[start..end];
    let n_t = trimmed.len() as f64;
    let re_mean = trimmed.iter().map(|(r, _)| r).sum::<f64>() / n_t;
    let im_mean = trimmed.iter().map(|(_, i)| i).sum::<f64>() / n_t;
    (re_mean, im_mean)
}

fn median_complex(values: &mut [(f64, f64)]) -> (f64, f64) {
    // Compute component-wise median (not the geometric median).
    let n = values.len();
    let mid = n / 2;

    // Median of real parts.
    values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let re_med = if n % 2 == 0 {
        (values[mid - 1].0 + values[mid].0) / 2.0
    } else {
        values[mid].0
    };

    // Median of imaginary parts.
    values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let im_med = if n % 2 == 0 {
        (values[mid - 1].1 + values[mid].1) / 2.0
    } else {
        values[mid].1
    };

    (re_med, im_med)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("=== D4: Trimmed Stacking Robustness Test ===");
    eprintln!();

    // Load per-galaxy DFT coefficients.
    let mut rdr = csv::Reader::from_path(&cli.galaxies_csv)?;

    struct GalaxyDFT {
        cd: (f64, f64),
        g2: (f64, f64),
        j3o: (f64, f64),
        sl2: (f64, f64),
    }

    let mut galaxies: Vec<GalaxyDFT> = Vec::new();

    for result in rdr.records() {
        let rec = result?;
        galaxies.push(GalaxyDFT {
            cd: (
                rec.get(1).unwrap_or("0").parse().unwrap_or(0.0),
                rec.get(2).unwrap_or("0").parse().unwrap_or(0.0),
            ),
            g2: (
                rec.get(3).unwrap_or("0").parse().unwrap_or(0.0),
                rec.get(4).unwrap_or("0").parse().unwrap_or(0.0),
            ),
            j3o: (
                rec.get(5).unwrap_or("0").parse().unwrap_or(0.0),
                rec.get(6).unwrap_or("0").parse().unwrap_or(0.0),
            ),
            sl2: (
                rec.get(7).unwrap_or("0").parse().unwrap_or(0.0),
                rec.get(8).unwrap_or("0").parse().unwrap_or(0.0),
            ),
        });
    }

    let n_gal = galaxies.len();
    eprintln!("Loaded {} galaxies", n_gal);
    eprintln!();

    let trim_levels = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50];
    let algebra_names = ["CD-ZD", "G2", "J3O", "sl2"];

    println!("algebra,trim_frac,n_used,mean_re,mean_im,power,snr_equiv,phase");

    for (alg_idx, alg_name) in algebra_names.iter().enumerate() {
        eprintln!("--- {} ---", alg_name);

        // Extract complex coefficients for this algebra.
        let coeffs: Vec<(f64, f64)> = galaxies.iter().map(|g| match alg_idx {
            0 => g.cd,
            1 => g.g2,
            2 => g.j3o,
            3 => g.sl2,
            _ => (0.0, 0.0),
        }).collect();

        // Full (untrimmed) mean for reference.
        let full_re = coeffs.iter().map(|(r, _)| r).sum::<f64>() / n_gal as f64;
        let full_im = coeffs.iter().map(|(_, i)| i).sum::<f64>() / n_gal as f64;
        let full_power = full_re * full_re + full_im * full_im;

        // RMS of individual powers for SNR baseline.
        for &trim_frac in &trim_levels {
            let mut c = coeffs.clone();

            let (mean_re, mean_im) = if trim_frac == 0.0 {
                (full_re, full_im)
            } else {
                trimmed_mean_complex(&mut c, trim_frac)
            };

            let power = mean_re * mean_re + mean_im * mean_im;
            let phase = mean_im.atan2(mean_re);
            let n_used = n_gal - 2 * ((n_gal as f64 * trim_frac) as usize);

            // SNR relative to full-stack baseline.
            let snr = if full_power > 1e-30 { power.sqrt() / full_power.sqrt() } else { 0.0 };

            println!("{},{:.2},{},{:.8e},{:.8e},{:.8e},{:.6},{:.4}",
                alg_name, trim_frac, n_used, mean_re, mean_im, power, snr, phase);

            eprintln!("  trim={:.0}% (n={}): power={:.6e}, SNR_ratio={:.4}, phase={:.4}",
                trim_frac * 100.0, n_used, power, snr, phase);
        }

        // Median (maximum robustness).
        let mut c = coeffs.clone();
        let (med_re, med_im) = median_complex(&mut c);
        let med_power = med_re * med_re + med_im * med_im;
        let med_snr = if full_power > 1e-30 { med_power.sqrt() / full_power.sqrt() } else { 0.0 };
        let med_phase = med_im.atan2(med_re);

        println!("{},median,{},{:.8e},{:.8e},{:.8e},{:.6},{:.4}",
            alg_name, n_gal, med_re, med_im, med_power, med_snr, med_phase);
        eprintln!("  MEDIAN: power={:.6e}, SNR_ratio={:.4}, phase={:.4}",
            med_power, med_snr, med_phase);
        eprintln!();
    }

    // Cross-algebra summary.
    eprintln!("=== Verdict ===");

    // Check if trimming changes the SNR significantly.
    for (alg_idx, alg_name) in algebra_names.iter().enumerate() {
        let coeffs: Vec<(f64, f64)> = galaxies.iter().map(|g| match alg_idx {
            0 => g.cd,
            1 => g.g2,
            2 => g.j3o,
            3 => g.sl2,
            _ => (0.0, 0.0),
        }).collect();

        let full_re = coeffs.iter().map(|(r, _)| r).sum::<f64>() / n_gal as f64;
        let full_im = coeffs.iter().map(|(_, i)| i).sum::<f64>() / n_gal as f64;
        let full_power = full_re * full_re + full_im * full_im;

        let mut c20 = coeffs.clone();
        let (t20_re, t20_im) = trimmed_mean_complex(&mut c20, 0.20);
        let t20_power = t20_re * t20_re + t20_im * t20_im;
        let ratio_20 = if full_power > 1e-30 { t20_power / full_power } else { 0.0 };

        eprintln!("  {}: 20%-trimmed/full power ratio = {:.4}", alg_name, ratio_20);

        if (ratio_20 - 1.0).abs() > 0.3 {
            eprintln!("    INSTABILITY: >30% change with 20% trimming -- outliers drive the result!");
        } else {
            eprintln!("    ROBUST: <30% change -- null result is not outlier-driven.");
        }
    }

    Ok(())
}
