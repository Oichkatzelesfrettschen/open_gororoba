use anyhow::Result;
use cd_kernel::cayley_dickson::cd_multiply;
use clap::Parser;
use faer::Mat;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::{fs::File, io::Write};

#[derive(Parser, Debug)]
#[command(author, version, about = "Spectral Flow of Sedenion Field")]
struct Args {
    #[arg(short, long, default_value_t = 6)]
    l: usize,

    #[arg(short, long, default_value_t = 30)]
    steps: usize,

    #[arg(short, long, default_value_t = 0.01)]
    dt: f64,

    #[arg(short, long, default_value = "data/csv/spectral_flow.csv")]
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let l = args.l;
    let n_points = l * l * l;
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Sedenion field: [point_idx][component_idx]
    // Point-major for easier iteration over points, then component-major for SVD.
    let mut phi = vec![vec![0.0_f64; 16]; n_points];
    for p in &mut phi {
        for val in p.iter_mut() {
            *val = rng.r#gen::<f64>() * 0.1;
        }
    }

    let mut csv_file = File::create(&args.output)?;
    let mut header = "step".to_string();
    for i in 0..16 {
        header.push_str(&format!(",mode_{}", i));
    }
    writeln!(csv_file, "{}", header)?;

    for t in 0..args.steps {
        println!("Step {}/{}...", t, args.steps);

        // 1. Evolve: phi += dt * (phi * phi)
        for p in &mut phi {
            let phi_sq = cd_multiply(p, p);
            for (i, val) in p.iter_mut().enumerate() {
                *val += args.dt * phi_sq[i];
            }
        }

        // 2. Compute Spectrum
        // Construct (16, n_points) matrix for SVD
        let mut mat = Mat::<f64>::zeros(16, n_points);
        #[allow(clippy::needless_range_loop)]
        for i in 0..16 {
            for j in 0..n_points {
                mat.write(i, j, phi[j][i]);
            }
        }

        // SVD
        let svd = mat.svd();
        let s = svd.s_diagonal();

        // 3. Save
        let mut row = format!("{}", t);
        for i in 0..16 {
            let val = s.read(i);
            row.push_str(&format!(",{}", val));
        }
        writeln!(csv_file, "{}", row)?;
    }

    println!("Spectral Flow Complete. Saved to {}", args.output);

    Ok(())
}
