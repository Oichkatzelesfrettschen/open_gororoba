use anyhow::{Context, Result};
use clap::Parser;
use gororoba_cli_data::nanograv_timing::load_release;
use algebra_experimental::higher_cd::HigherAvt;
use std::path::PathBuf;
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(
    name = "nanograv-vacuum-symmetry",
    about = "Detects point group symmetry in the 1024D DekaVoudon vacuum frustration field"
)]
struct Args {
    #[arg(
        long,
        default_value = "data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0"
    )]
    root: PathBuf,

    #[arg(long, default_value_t = 100_000)]
    n_samples: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let _release = load_release(&args.root).context("failed to load timing release")?;
    
    println!("Generating 1024D DekaVoudon AVT ({} samples)...", args.n_samples);
    let avt_wrapper = HigherAvt::sampled(1024, args.n_samples, 42);
    let avt = &avt_wrapper.avt;

    // 1. Sample the sphere and map frustration
    let n_points = 2000;
    let mut field = HashMap::new();
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;

    for k in 0..n_points {
        let theta = std::f64::consts::TAU * k as f64 / golden_ratio;
        let phi = (1.0 - 2.0 * (k as f64 + 0.5) / n_points as f64).acos();
        
        let x = phi.sin() * theta.cos();
        let y = phi.sin() * theta.sin();
        let z = phi.cos();

        let ra = y.atan2(x);
        let idx = (((ra + std::f64::consts::PI) / std::f64::consts::TAU * 1024.0) as usize) % 1024;
        
        let frustration = avt.violations.iter().filter(|&&(a, b, c, _, _)| {
            a == idx || b == idx || c == idx
        }).count();

        // Use quantized key for lookup
        let key = ( (x*10.0).round() as i32, (y*10.0).round() as i32, (z*10.0).round() as i32 );
        field.insert(key, frustration as f64);
    }

    println!("Detecting Point Group Symmetry...");

    // 2. Test Inversion Symmetry (Ci)
    let mut inversion_error = 0.0;
    let mut count = 0;
    for (&(ix, iy, iz), &f1) in &field {
        if let Some(&f2) = field.get(&(-ix, -iy, -iz)) {
            inversion_error += (f1 - f2).abs();
            count += 1;
        }
    }
    let inversion_score = if count > 0 { inversion_error / count as f64 } else { 1.0 };

    // 3. Test 90-degree Z-rotation (C4)
    let mut c4_error = 0.0;
    count = 0;
    for (&(ix, iy, iz), &f1) in &field {
        if let Some(&f2) = field.get(&(-iy, ix, iz)) {
            c4_error += (f1 - f2).abs();
            count += 1;
        }
    }
    let c4_score = if count > 0 { c4_error / count as f64 } else { 1.0 };

    println!("\nVACUUM SYMMETRY SCORES (Lower is more symmetric):");
    println!("--------------------------------------------------");
    println!("Inversion (Ci):    {:.4}", inversion_score);
    println!("4-fold Z (C4):     {:.4}", c4_score);

    if inversion_score < 0.1 && c4_score < 0.1 {
        println!("\n>>> RESULT: HIGH SYMMETRY DETECTED (Cubic-like).");
    } else if inversion_score < 0.1 {
        println!("\n>>> RESULT: CENTROSYMMETRIC VACUUM DETECTED.");
    } else {
        println!("\n>>> RESULT: ANISOTROPIC CHIRAL VACUUM.");
    }

    Ok(())
}
