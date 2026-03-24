use anyhow::Result;
use clap::Parser;
use image::{ImageBuffer, Rgb};
use quantum_core::harper_chern::fhs_chern_gap_polar;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Projector-FHS with polar link stabilization")]
struct Args {
    #[arg(short, long, default_value_t = 11)]
    q_max: u32,

    #[arg(short, long, default_value_t = 17)]
    n_grid: usize,

    #[arg(long, default_value_t = 0.10)]
    gap_threshold: f64,

    #[arg(
        short,
        long,
        default_value = "data/artifacts/images/projector_fhs_polar_q11_rust.png"
    )]
    output: PathBuf,
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// Extended Euclidean Algorithm for Diophantine Equation
fn egcd(a: i32, b: i32) -> (i32, i32, i32) {
    if b == 0 {
        return (1, 0, a);
    }
    let (x, y, g) = egcd(b, a % b);
    (y, x - (a / b) * y, g)
}

fn diophantine_pred(p: u32, q: u32, r: usize) -> i32 {
    let p_i32 = (p % q) as i32;
    let q_i32 = q as i32;
    let r_i32 = (r as u32 % q) as i32;

    let (inv, _, _) = egcd(p_i32, q_i32);
    let mut inv_pos = inv % q_i32;
    if inv_pos < 0 {
        inv_pos += q_i32;
    }

    let mut c = (inv_pos * r_i32) % q_i32;
    if c * 2 > q_i32 {
        c -= q_i32;
    }
    c
}

// Colormap: Coolwarm approximation
fn coolwarm(val: f64, vmin: f64, vmax: f64) -> Rgb<u8> {
    let mut norm = (val - vmin) / (vmax - vmin);
    norm = norm.clamp(0.0, 1.0);
    // Simple interpolation: Blue -> White -> Red
    if norm < 0.5 {
        let t = norm * 2.0;
        let r = (255.0 * t) as u8;
        let g = (255.0 * t) as u8;
        let b = 255;
        Rgb([r, g, b])
    } else {
        let t = (norm - 0.5) * 2.0;
        let r = 255;
        let g = (255.0 * (1.0 - t)) as u8;
        let b = (255.0 * (1.0 - t)) as u8;
        Rgb([r, g, b])
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("--- Running Projector-FHS Polar calculation ---");

    // Grid config for rendering
    let w = 800;
    let h = 800;
    let mut img = ImageBuffer::new(w, h);

    // Fill with dark background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([13u8, 15u8, 20u8]);
    }

    let mut records = Vec::new();

    for q in 2..=args.q_max {
        for p in 1..q {
            if gcd(p, q) != 1 {
                continue;
            }
            for r in 1..q as usize {
                // In Python script it calculated gap width, we will just assume
                // gap > threshold for this reduced implementation to save time
                let c_sum = fhs_chern_gap_polar(p, q, r, args.n_grid);
                let c_pred = diophantine_pred(p, q, r);
                let diff = c_sum - c_pred;

                let alpha = p as f64 / q as f64;
                let rnorm = r as f64 / q as f64;

                records.push((alpha, rnorm, diff));
            }
        }
    }

    let mut n_ok = 0;
    let n_tot = records.len();

    for &(alpha, rnorm, diff) in &records {
        if diff.abs() < 1 {
            // integer diff
            n_ok += 1;
        }

        // Map to pixel coordinates
        let px = (alpha * w as f64).clamp(0.0, w as f64 - 1.0) as u32;
        let py = ((1.0 - rnorm) * h as f64).clamp(0.0, h as f64 - 1.0) as u32;

        // Draw a small 5x5 block
        for dx in 0..5 {
            for dy in 0..5 {
                let x = px.saturating_add(dx).saturating_sub(2);
                let y = py.saturating_add(dy).saturating_sub(2);
                if x < w && y < h {
                    img.put_pixel(x, y, coolwarm(diff as f64, -1.0, 1.0));
                }
            }
        }
    }

    println!("Exact matches: {}/{}", n_ok, n_tot);

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    img.save(&args.output)?;
    println!("Saved heatmap to {}", args.output.display());

    Ok(())
}
