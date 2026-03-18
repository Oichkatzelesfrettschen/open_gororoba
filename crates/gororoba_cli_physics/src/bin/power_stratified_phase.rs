//! Power-stratified phase coherence test.
//!
//! Exploits the extreme kurtosis (~284) in per-galaxy Fourier power to test
//! whether high-power galaxies (the ~5% driving the tails) show phase
//! coherence (R > 0.3) while low-power galaxies do not.
//!
//! If ZD forcing is mass- or halo-dependent, it would preferentially affect
//! a subset of galaxies, creating a signal in the power tails that is
//! washed out by mean stacking.
//!
//! Algorithm:
//!   1. Sort 6995 galaxies by aggregate CD-ZD power = re^2 + im^2
//!   2. Split into 5 quintiles (1399 galaxies each)
//!   3. For each quintile, compute Rayleigh R on the phase distribution
//!   4. ZD prediction: top quintile R > 0.3 (coherent high-power galaxies)
//!   5. Null prediction: all quintiles R ~ 1/sqrt(1399) = 0.027
//!
//! Also tests each algebra separately and all four combined.
//!
//! Reference: Hypothesis P3 from practical ML engineer plan.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "power-stratified-phase")]
#[command(about = "Phase coherence test stratified by per-galaxy Fourier power quintile")]
struct Cli {
    /// Per-galaxy DFT CSV (columns: plateifu, cd_re, cd_im, g2_re, g2_im, ...).
    #[arg(long)]
    galaxies_csv: PathBuf,

    /// Number of strata (default 5 = quintiles).
    #[arg(long, default_value_t = 5)]
    n_strata: usize,
}

struct Galaxy {
    cd_re: f64,
    cd_im: f64,
    g2_re: f64,
    g2_im: f64,
    j3o_re: f64,
    j3o_im: f64,
    sl2_re: f64,
    sl2_im: f64,
}

impl Galaxy {
    fn power(&self, algebra: &str) -> f64 {
        match algebra {
            "CD-ZD" => self.cd_re * self.cd_re + self.cd_im * self.cd_im,
            "G2" => self.g2_re * self.g2_re + self.g2_im * self.g2_im,
            "J3O" => self.j3o_re * self.j3o_re + self.j3o_im * self.j3o_im,
            "sl2" => self.sl2_re * self.sl2_re + self.sl2_im * self.sl2_im,
            _ => 0.0,
        }
    }

    fn phase(&self, algebra: &str) -> f64 {
        match algebra {
            "CD-ZD" => self.cd_im.atan2(self.cd_re),
            "G2" => self.g2_im.atan2(self.g2_re),
            "J3O" => self.j3o_im.atan2(self.j3o_re),
            "sl2" => self.sl2_im.atan2(self.sl2_re),
            _ => 0.0,
        }
    }
}

/// Rayleigh R statistic: mean resultant length of circular data.
/// R ~ 1/sqrt(N) for uniform distribution, R -> 1 for concentrated.
fn rayleigh_r(phases: &[f64]) -> f64 {
    let n = phases.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
    let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();
    ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
}

/// Rayleigh p-value: P(R > r | H0: uniform) ~ exp(-n * R^2) for large n.
fn rayleigh_p(r: f64, n: usize) -> f64 {
    (-(n as f64) * r * r).exp()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("=== P3: Power-Stratified Phase Coherence ===");
    eprintln!();

    // Load per-galaxy DFT coefficients.
    let mut rdr = csv::Reader::from_path(&cli.galaxies_csv)?;
    let mut galaxies: Vec<Galaxy> = Vec::new();

    for result in rdr.records() {
        let rec = result?;
        galaxies.push(Galaxy {
            cd_re: rec.get(1).unwrap_or("0").parse().unwrap_or(0.0),
            cd_im: rec.get(2).unwrap_or("0").parse().unwrap_or(0.0),
            g2_re: rec.get(3).unwrap_or("0").parse().unwrap_or(0.0),
            g2_im: rec.get(4).unwrap_or("0").parse().unwrap_or(0.0),
            j3o_re: rec.get(5).unwrap_or("0").parse().unwrap_or(0.0),
            j3o_im: rec.get(6).unwrap_or("0").parse().unwrap_or(0.0),
            sl2_re: rec.get(7).unwrap_or("0").parse().unwrap_or(0.0),
            sl2_im: rec.get(8).unwrap_or("0").parse().unwrap_or(0.0),
        });
    }

    let n_gal = galaxies.len();
    let n_per_stratum = n_gal / cli.n_strata;
    eprintln!("Loaded {} galaxies, {} strata of ~{} each",
        n_gal, cli.n_strata, n_per_stratum);
    eprintln!("Expected R under H0 ~ 1/sqrt({}) = {:.4}",
        n_per_stratum, 1.0 / (n_per_stratum as f64).sqrt());
    eprintln!();

    println!("algebra,stratum,n_galaxies,power_min,power_max,power_median,rayleigh_r,rayleigh_p,mean_phase");

    let algebras = ["CD-ZD", "G2", "J3O", "sl2"];

    for algebra in &algebras {
        eprintln!("--- {} ---", algebra);

        // Sort galaxies by power in this algebra.
        let mut indices: Vec<usize> = (0..n_gal).collect();
        indices.sort_by(|&a, &b| {
            galaxies[a].power(algebra)
                .partial_cmp(&galaxies[b].power(algebra))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for s in 0..cli.n_strata {
            let start = s * n_per_stratum;
            let end = if s == cli.n_strata - 1 { n_gal } else { (s + 1) * n_per_stratum };
            let stratum_indices = &indices[start..end];
            let n_s = stratum_indices.len();

            let powers: Vec<f64> = stratum_indices.iter()
                .map(|&i| galaxies[i].power(algebra))
                .collect();
            let phases: Vec<f64> = stratum_indices.iter()
                .map(|&i| galaxies[i].phase(algebra))
                .collect();

            let r = rayleigh_r(&phases);
            let p = rayleigh_p(r, n_s);

            let power_min = powers.first().copied().unwrap_or(0.0);
            let power_max = powers.last().copied().unwrap_or(0.0);
            let power_med = powers[powers.len() / 2];

            // Mean phase (circular mean).
            let mean_cos: f64 = phases.iter().map(|p| p.cos()).sum::<f64>() / n_s as f64;
            let mean_sin: f64 = phases.iter().map(|p| p.sin()).sum::<f64>() / n_s as f64;
            let mean_phase = mean_sin.atan2(mean_cos);

            let label = match s {
                0 => "Q1 (lowest)",
                s if s == cli.n_strata - 1 => "Q5 (highest)",
                _ => "middle",
            };

            println!("{},{},{},{:.6e},{:.6e},{:.6e},{:.6},{:.6e},{:.4}",
                algebra, s + 1, n_s, power_min, power_max, power_med, r, p, mean_phase);

            eprintln!("  {} (n={}): R={:.4}, p={:.4e}, power=[{:.4e}, {:.4e}]",
                label, n_s, r, p, power_min, power_max);
        }
        eprintln!();
    }

    // Summary: compare top quintile R across algebras.
    eprintln!("=== Cross-algebra top-quintile comparison ===");
    let mut top_r_values: Vec<(&str, f64)> = Vec::new();

    for algebra in &algebras {
        let mut indices: Vec<usize> = (0..n_gal).collect();
        indices.sort_by(|&a, &b| {
            galaxies[a].power(algebra)
                .partial_cmp(&galaxies[b].power(algebra))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let start = (cli.n_strata - 1) * n_per_stratum;
        let top_phases: Vec<f64> = indices[start..]
            .iter()
            .map(|&i| galaxies[i].phase(algebra))
            .collect();
        let r = rayleigh_r(&top_phases);
        top_r_values.push((algebra, r));
        eprintln!("  {}: top-quintile R = {:.4}", algebra, r);
    }

    eprintln!();
    eprintln!("=== Verdict ===");
    let max_top_r = top_r_values.iter().map(|(_, r)| *r).fold(0.0_f64, f64::max);
    let expected_r = 1.0 / (n_per_stratum as f64).sqrt();

    if max_top_r > 0.3 {
        eprintln!("  Max top-quintile R = {:.4} > 0.3: strong phase coherence in outliers!", max_top_r);
        eprintln!("  Possible ZD signal concentrated in high-power galaxy subset.");
    } else if max_top_r > 3.0 * expected_r {
        eprintln!("  Max top-quintile R = {:.4} > 3x expected ({:.4}): marginal excess.",
            max_top_r, expected_r);
        eprintln!("  Weakly suggestive but below 0.3 threshold.");
    } else {
        eprintln!("  Max top-quintile R = {:.4}, consistent with noise (expected {:.4}).",
            max_top_r, expected_r);
        eprintln!("  FALSIFIED: no phase coherence even in high-power outlier galaxies.");
    }

    Ok(())
}
