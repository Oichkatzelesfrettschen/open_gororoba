use anyhow::Result;
use clap::Parser;
use rand::prelude::*;
use rand_distr::{ChiSquared, Distribution, Exp, LogNormal, Normal, Pareto, Uniform};
use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const GHOST_FREQ: f64 = 1.0 - 0.786_151_377_757_423; // Aliased ghost frequency

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate synthetic datasets for ghost spectral audit")]
struct Args {
    #[arg(short, long, default_value = "data/csv/ghost_audit")]
    output_dir: PathBuf,

    #[arg(short, long, default_value_t = 2048)]
    n: usize,

    #[arg(short, long, default_value_t = 42)]
    seed: u64,
}

fn write_csv(path: &Path, values: &[f64], column: &str) -> Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "index,{}", column)?;
    for (i, v) in values.iter().enumerate() {
        writeln!(file, "{},{:.12e}", i, v)?;
    }
    println!("  Wrote {} values to {}", values.len(), path.display());
    Ok(())
}

fn generate_all(output_dir: &Path, n: usize, seed: u64) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let mut rng = StdRng::seed_from_u64(seed);

    // ---- NULL CONTROLS (unsorted) ----
    println!("\n--- Null Controls (unsorted) ---");

    let normal = Normal::new(0.0, 1.0).unwrap();
    let log_normal = LogNormal::new(0.0, 1.0).unwrap();
    let pareto = Pareto::new(1.0, 2.5).unwrap();
    let uniform = Uniform::new(0.0, 100.0);

    let noise_gauss: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    write_csv(&output_dir.join("null_gaussian.csv"), &noise_gauss, "value")?;

    let noise_lognorm: Vec<f64> = (0..n).map(|_| log_normal.sample(&mut rng)).collect();
    write_csv(&output_dir.join("null_lognormal.csv"), &noise_lognorm, "value")?;

    let noise_pareto: Vec<f64> = (0..n).map(|_| (pareto.sample(&mut rng) + 1.0) * 10.0).collect();
    write_csv(&output_dir.join("null_pareto.csv"), &noise_pareto, "value")?;

    let noise_uniform: Vec<f64> = (0..n).map(|_| uniform.sample(&mut rng)).collect();
    write_csv(&output_dir.join("null_uniform.csv"), &noise_uniform, "value")?;

    // ---- SORTED DISTRIBUTION TESTS ----
    println!("\n--- Sorted Distributions ---");

    let sorted_normal_dist = Normal::new(50.0, 15.0).unwrap();
    let mut sorted_gauss: Vec<f64> = (0..n).map(|_| sorted_normal_dist.sample(&mut rng)).collect();
    sorted_gauss.sort_by(|a, b| a.partial_cmp(b).unwrap());
    write_csv(&output_dir.join("sorted_gaussian.csv"), &sorted_gauss, "value")?;

    let sorted_lognorm_dist = LogNormal::new(5.0, 1.2).unwrap();
    let mut sorted_lognorm: Vec<f64> = (0..n).map(|_| sorted_lognorm_dist.sample(&mut rng)).collect();
    sorted_lognorm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    write_csv(&output_dir.join("sorted_lognormal.csv"), &sorted_lognorm, "value")?;

    let sorted_pareto_dist = Pareto::new(1.0, 2.0).unwrap();
    let mut sorted_pareto: Vec<f64> = (0..n).map(|_| (sorted_pareto_dist.sample(&mut rng) + 1.0) * 100.0).collect();
    sorted_pareto.sort_by(|a, b| a.partial_cmp(b).unwrap());
    write_csv(&output_dir.join("sorted_pareto.csv"), &sorted_pareto, "value")?;

    let exp_dist = Exp::new(1.0 / 100.0).unwrap();
    let mut sorted_exp: Vec<f64> = (0..n).map(|_| exp_dist.sample(&mut rng)).collect();
    sorted_exp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    write_csv(&output_dir.join("sorted_exponential.csv"), &sorted_exp, "value")?;

    let chi2_dist = ChiSquared::new(4.0).unwrap();
    let mut sorted_chi2: Vec<f64> = (0..n).map(|_| chi2_dist.sample(&mut rng) * 10.0).collect();
    sorted_chi2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    write_csv(&output_dir.join("sorted_chisq.csv"), &sorted_chi2, "value")?;

    // ---- SIGNAL INJECTION TESTS ----
    println!("\n--- Signal Injections ---");

    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    
    let signal_ghost: Vec<f64> = t.iter().map(|&ti| 5.0 * (2.0 * PI * GHOST_FREQ * ti).sin() + normal.sample(&mut rng)).collect();
    write_csv(&output_dir.join("signal_ghost_strong.csv"), &signal_ghost, "value")?;

    let signal_ghost_weak: Vec<f64> = t.iter().map(|&ti| 1.5 * (2.0 * PI * GHOST_FREQ * ti).sin() + normal.sample(&mut rng)).collect();
    write_csv(&output_dir.join("signal_ghost_weak.csv"), &signal_ghost_weak, "value")?;

    let wrong_freq = 0.35;
    let signal_wrong: Vec<f64> = t.iter().map(|&ti| 5.0 * (2.0 * PI * wrong_freq * ti).sin() + normal.sample(&mut rng)).collect();
    write_csv(&output_dir.join("signal_wrong_freq.csv"), &signal_wrong, "value")?;

    // Mixed: sine at ghost freq embedded in red noise
    let mut red_noise = vec![0.0; n];
    red_noise[0] = normal.sample(&mut rng);
    for i in 1..n {
        red_noise[i] = 0.7 * red_noise[i - 1] + normal.sample(&mut rng);
    }
    let signal_red: Vec<f64> = t.iter().zip(red_noise.iter()).map(|(&ti, &rn)| 3.0 * (2.0 * PI * GHOST_FREQ * ti).sin() + rn).collect();
    write_csv(&output_dir.join("signal_ghost_red_noise.csv"), &signal_red, "value")?;

    // ---- MOCK CATALOG DATA ----
    println!("\n--- Mock Catalogs ---");

    let n_frb = 500;
    let frb_dist = LogNormal::new(400.0_f64.ln(), 0.8).unwrap();
    let mut mock_frb_dm: Vec<f64> = (0..n_frb).map(|_| frb_dist.sample(&mut rng)).collect();
    mock_frb_dm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    write_csv(&output_dir.join("mock_frb_dm.csv"), &mock_frb_dm, "value")?;

    let n_psr = 4000;
    let psr_dist = LogNormal::new(0.5_f64.ln(), 1.5).unwrap();
    let mut mock_psr: Vec<f64> = (0..n_psr).map(|_| psr_dist.sample(&mut rng)).collect();
    mock_psr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    write_csv(&output_dir.join("mock_pulsar_period.csv"), &mock_psr, "value")?;

    let n_sn = 1700;
    let sn_dist = Normal::new(38.0, 3.0).unwrap();
    let mut mock_sn: Vec<f64> = (0..n_sn).map(|_| sn_dist.sample(&mut rng)).collect();
    mock_sn.sort_by(|a, b| a.partial_cmp(b).unwrap());
    write_csv(&output_dir.join("mock_sn_mu.csv"), &mock_sn, "value")?;

    let n_gaia = 5000;
    let gaia_dist = Exp::new(1.0 / 2.0).unwrap();
    let mut mock_gaia: Vec<f64> = (0..n_gaia).map(|_| gaia_dist.sample(&mut rng)).collect();
    mock_gaia.sort_by(|a, b| a.partial_cmp(b).unwrap());
    write_csv(&output_dir.join("mock_gaia_parallax.csv"), &mock_gaia, "value")?;

    println!("\n==================================================");
    println!("Generated 17 datasets in {}", output_dir.display());

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    generate_all(&args.output_dir, args.n, args.seed)?;
    Ok(())
}
