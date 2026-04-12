use anyhow::Result;
use clap::Parser;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use spectral_core::spectral_flow::{cd_evolve_step, sedenion_field_svd};
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

    // Sedenion field: [point_idx][component_idx], 16 components per site.
    let mut phi = vec![vec![0.0_f64; 16]; n_points];
    for p in &mut phi {
        for val in p.iter_mut() {
            *val = rng.random::<f64>() * 0.1;
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

        // Evolve via Cayley-Dickson product: phi += dt * (phi * phi)
        cd_evolve_step(&mut phi, args.dt);

        // Singular values of the unfolded (16 x n_points) field matrix.
        let s = sedenion_field_svd(&phi);

        let mut row = format!("{}", t);
        for val in &s {
            row.push_str(&format!(",{}", val));
        }
        writeln!(csv_file, "{}", row)?;
    }

    println!("Spectral Flow Complete. Saved to {}", args.output);

    Ok(())
}
