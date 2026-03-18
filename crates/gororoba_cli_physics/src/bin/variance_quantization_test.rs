//! Variance quantization test on per-galaxy Fourier power distributions.
//!
//! Tests whether the distribution of per-galaxy Fourier power at CD-ZD
//! wavenumbers is multimodal (Hartigan dip test), indicating heteroscedastic
//! ZD forcing, versus unimodal at control wavenumbers.
//!
//! Uses the existing per-galaxy DFT CSV (cross_algebra_correlation.galaxies.csv)
//! which provides a single complex coefficient per algebra per galaxy.
//! Computes power = re^2 + im^2 for each galaxy and applies the dip test
//! to the resulting distribution.
//!
//! Also computes the coefficient of variation (CV = std/mean) of per-galaxy
//! power for each algebra and tests whether CD-ZD CV exceeds control CV.
//!
//! Reference: Hypothesis H2 from novel hypothesis plan.

use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use stats_core::dip::hartigan_dip_test;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "variance-quantization-test")]
#[command(about = "Dip test on per-galaxy Fourier power for variance quantization")]
struct Cli {
    /// Per-galaxy DFT CSV (columns: plateifu, cd_re, cd_im, g2_re, g2_im, ...).
    #[arg(long)]
    galaxies_csv: PathBuf,

    /// Number of permutations for dip test.
    #[arg(long, default_value_t = 5000)]
    n_perm: usize,

    /// RNG seed for reproducibility.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

struct AlgebraPower {
    label: &'static str,
    powers: Vec<f64>,
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if mean.abs() < 1e-30 {
        return 0.0;
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    var.sqrt() / mean
}

fn excess_kurtosis(values: &[f64]) -> f64 {
    if values.len() < 4 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    if var < 1e-30 {
        return 0.0;
    }
    let m4 = values.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    m4 / (var * var) - 3.0
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("=== H2: Variance Quantization Test ===");
    eprintln!();

    // Load per-galaxy DFT coefficients.
    let mut rdr = csv::Reader::from_path(&cli.galaxies_csv)?;
    let headers = rdr.headers()?.clone();
    eprintln!("Columns: {:?}", headers.iter().collect::<Vec<_>>());

    let mut algebras: Vec<AlgebraPower> = vec![
        AlgebraPower {
            label: "CD-ZD (sedenion)",
            powers: Vec::new(),
        },
        AlgebraPower {
            label: "G2 Aut(O)",
            powers: Vec::new(),
        },
        AlgebraPower {
            label: "Albert J3(O)",
            powers: Vec::new(),
        },
        AlgebraPower {
            label: "sl(2) partner",
            powers: Vec::new(),
        },
    ];

    // Column indices: cd_re=1, cd_im=2, g2_re=3, g2_im=4, j3o_re=5, j3o_im=6, sl2_re=7, sl2_im=8.
    for result in rdr.records() {
        let rec = result?;
        for (alg_idx, re_col) in [1_usize, 3, 5, 7].iter().enumerate() {
            let re: f64 = rec.get(*re_col).unwrap_or("0").parse().unwrap_or(0.0);
            let im: f64 = rec.get(re_col + 1).unwrap_or("0").parse().unwrap_or(0.0);
            let power = re * re + im * im;
            algebras[alg_idx].powers.push(power);
        }
    }

    let n_gal = algebras[0].powers.len();
    eprintln!("Loaded {} galaxies with 4 algebra powers", n_gal);
    eprintln!("Dip test: {} permutations, seed {}", cli.n_perm, cli.seed);
    eprintln!();

    // Run dip test on each algebra.
    println!("algebra,n_galaxies,dip_statistic,dip_p_value,cv,excess_kurtosis");

    let mut rng = ChaCha8Rng::seed_from_u64(cli.seed);

    for alg in &algebras {
        let dip_result = hartigan_dip_test(&alg.powers, cli.n_perm, &mut rng);
        let cv = coefficient_of_variation(&alg.powers);
        let kurt = excess_kurtosis(&alg.powers);

        println!(
            "{},{},{:.8e},{:.6},{:.6},{:.4}",
            alg.label,
            alg.powers.len(),
            dip_result.dip_statistic,
            dip_result.p_value,
            cv,
            kurt
        );

        eprintln!(
            "  {}: D={:.6e}, p={:.4}, CV={:.4}, kurtosis={:.4}",
            alg.label, dip_result.dip_statistic, dip_result.p_value, cv, kurt
        );
    }

    eprintln!();

    // Cross-algebra comparison: CD-ZD vs average of other three.
    let cd_cv = coefficient_of_variation(&algebras[0].powers);
    let other_cv_mean = (1..4)
        .map(|i| coefficient_of_variation(&algebras[i].powers))
        .sum::<f64>()
        / 3.0;
    let cv_ratio = if other_cv_mean > 1e-30 {
        cd_cv / other_cv_mean
    } else {
        f64::NAN
    };

    eprintln!("--- Cross-algebra CV comparison ---");
    eprintln!("  CD-ZD CV = {:.4}", cd_cv);
    eprintln!("  Mean other CV = {:.4}", other_cv_mean);
    eprintln!("  CV ratio (CD / others) = {:.4}", cv_ratio);

    eprintln!();
    eprintln!("=== Verdict ===");

    // Check dip test results for CD-ZD.
    let cd_dip = hartigan_dip_test(&algebras[0].powers, cli.n_perm, &mut rng);
    if cd_dip.p_value > 0.10 {
        eprintln!(
            "  CD-ZD dip p = {:.4} > 0.10: no multimodality detected.",
            cd_dip.p_value
        );
        eprintln!("  FALSIFICATION: unimodal power distribution -> REJECT variance quantization.");
    } else if cd_dip.p_value < 0.05 {
        eprintln!(
            "  CD-ZD dip p = {:.4} < 0.05: multimodality detected!",
            cd_dip.p_value
        );
        eprintln!("  Check if other algebras also show multimodality (shared baryonic origin)");
        eprintln!("  or if CD-ZD is uniquely multimodal (possible ZD structure).");
    } else {
        eprintln!(
            "  CD-ZD dip p = {:.4}: borderline, inconclusive.",
            cd_dip.p_value
        );
    }

    if cv_ratio > 1.5 {
        eprintln!(
            "  CV ratio = {:.4} > 1.5: CD-ZD has excess variance (consistent with ZD).",
            cv_ratio
        );
    } else {
        eprintln!(
            "  CV ratio = {:.4} <= 1.5: no excess CD-ZD variance.",
            cv_ratio
        );
    }

    Ok(())
}
