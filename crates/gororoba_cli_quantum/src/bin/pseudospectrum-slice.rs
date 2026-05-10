use anyhow::Result;
use clap::Parser;
use image::{ImageBuffer, Rgb};
use quantum_core::pseudospectrum::fractional_laplacian_pseudospectrum;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Pseudospectrum slice")]
struct Args {
    #[arg(
        short,
        long,
        default_value = "data/artifacts/images/pseudospectrum_slice_rust_3160x2820.png"
    )]
    output: PathBuf,
}

/// Map a normalized [0, 1] value to a blue-white-red (coolwarm) RGB triplet.
fn coolwarm(val: f64, vmin: f64, vmax: f64) -> Rgb<u8> {
    let mut norm = (val - vmin) / (vmax - vmin);
    norm = norm.clamp(0.0, 1.0);
    if norm < 0.5 {
        let t = norm * 2.0;
        Rgb([(255.0 * t) as u8, (255.0 * t) as u8, 255])
    } else {
        let t = (norm - 0.5) * 2.0;
        Rgb([255, (255.0 * (1.0 - t)) as u8, (255.0 * (1.0 - t)) as u8])
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Running Pseudospectrum Slice...");

    // Operator: A = 1.0 * L^1.0 + 0.7 * 0.5 * L^0.7 + 0.7^2 * 0.3 * L^0.4
    // powers_coeffs encodes (power, base_coeff); lambda = 0.7 scales term k by lambda^k.
    let result = fractional_laplacian_pseudospectrum(
        80,
        &[(1.0, 1.0), (0.7, 0.5), (0.4, 0.3)],
        0.7,
        (0.0, 40.0, 20),
        (-12.0, 12.0, 20),
    );

    let n_re = result.re_grid.len();
    let n_im = result.im_grid.len();

    let min_val = result
        .log_smin
        .iter()
        .flatten()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_val = result
        .log_smin
        .iter()
        .flatten()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Upscale: each grid cell becomes a cell_size x cell_size block of pixels.
    let cell_size = 40_usize;
    let img_w = (n_re * cell_size) as u32;
    let img_h = (n_im * cell_size) as u32;
    let mut img = ImageBuffer::new(img_w, img_h);

    for (i, row) in result.log_smin.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let color = coolwarm(v, min_val, max_val);
            // Flip Y so origin is bottom-left (imshow convention).
            let screen_y = n_im - 1 - i;
            for dy in 0..cell_size {
                for dx in 0..cell_size {
                    let px = (j * cell_size + dx) as u32;
                    let py = (screen_y * cell_size + dy) as u32;
                    img.put_pixel(px, py, color);
                }
            }
        }
    }

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    img.save(&args.output)?;
    println!("Saved: {}", args.output.display());

    Ok(())
}
