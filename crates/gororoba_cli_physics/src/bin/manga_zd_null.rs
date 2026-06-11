//! MaNGA IFU Rotation Curve Null Result Experiment (C-010).
//!
//! Investigates synthetic galaxy rotation curves for Zero-Divisor (ZD) signals.
//! Implements NFW and Baryonic baseline fits, harmonic stacking, and
//! algebra-specific DFT mode analysis.
//!
//! Migrated from main.py.

use anyhow::Result;
use clap::Parser;
use cosmology_core::{
    manga_analysis::{
        run_baryonic_baseline, run_harmonic_stacking, run_multi_algebra_dft, run_nfw_baseline,
    },
    manga_sim::{MangaSimParams, generate_synthetic_manga},
};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "manga-zd-null")]
#[command(about = "MaNGA IFU Rotation Curve Null Result Experiment")]
struct Args {
    /// Number of galaxies
    #[arg(short, long, default_value_t = 1000)]
    n_galaxies: usize,

    /// Number of bootstrap iterations
    #[arg(short, long, default_value_t = 100)]
    n_bootstrap: usize,

    /// Random seed
    #[arg(short, long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();

    println!("--- MaNGA IFU Null Experiment [RUST-FIRST] ---");

    // 1. Setup & Data Generation
    let mut sim_params = MangaSimParams::default();
    sim_params.n_galaxies = args.n_galaxies;
    sim_params.seed = args.seed;

    println!("Generating synthetic MaNGA data (N={})...", args.n_galaxies);
    let data = generate_synthetic_manga(&sim_params);

    // 2. Baselines
    let nfw_res = run_nfw_baseline(&data);
    println!("NFW Baseline SNR: {:.4}", nfw_res.snr);

    let baryonic_res = run_baryonic_baseline(&data);
    println!("Baryonic Baseline SNR: {:.4}", baryonic_res.snr);

    // 3. Test Conditions
    let harmonic_res = run_harmonic_stacking(&data, args.n_bootstrap);
    println!("Harmonic Halo Stack SNR: {:.4}", harmonic_res.snr);

    let dft_res = run_multi_algebra_dft(&data, args.n_bootstrap);
    println!("Multi-Algebra DFT SNR: {:.4}", dft_res.snr);

    println!("\n--- FINAL SUMMARY ---");
    println!("Total Time: {:.2?}", start.elapsed());

    Ok(())
}
