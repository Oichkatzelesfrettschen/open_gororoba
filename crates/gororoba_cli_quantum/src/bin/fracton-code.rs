use anyhow::Result;
use clap::Parser;
use image::{ImageBuffer, Rgb};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Fracton Stabilizer Code Visualization (Rule-90)"
)]
struct Args {
    #[arg(short, long, default_value_t = 65)]
    l: usize,

    #[arg(
        short,
        long,
        default_value = "data/artifacts/images/fracton_stabilizer_mask.png"
    )]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("--- Generating Fracton Stabilizer Code Visualization ---");

    let l = args.l;
    let mut grid = vec![vec![0u8; l]; l];

    // Seed Rule-90
    grid[0][l / 2] = 1;

    for r in 1..l {
        for c in 0..l {
            let left_idx = if c == 0 { l - 1 } else { c - 1 };
            let right_idx = if c == l - 1 { 0 } else { c + 1 };

            let left = grid[r - 1][left_idx];
            let right = grid[r - 1][right_idx];

            grid[r][c] = left ^ right;
        }
    }

    // Visualization: magma-like colors
    // 0 -> #0d0f14 (dark background)
    // 1 -> #fcfdbf (bright magma foreground)
    let bg_color = Rgb([13u8, 15u8, 20u8]);
    let fg_color = Rgb([252u8, 253u8, 191u8]);

    // Upscale for visibility (each cell is 4x4 pixels)
    let cell_size = 4;
    let img_size = (l * cell_size) as u32;
    let mut img = ImageBuffer::new(img_size, img_size);

    // PNG raster: outer (r, c) selects a grid cell, inner (dy, dx) fills
    // each cell at cell_size resolution; both grid and pixel offsets
    // need raw indices for the 4-deep nested loop.
    #[allow(clippy::needless_range_loop)]
    for r in 0..l {
        for c in 0..l {
            let color = if grid[r][c] == 1 { fg_color } else { bg_color };
            for dy in 0..cell_size {
                for dx in 0..cell_size {
                    let px = (c * cell_size + dx) as u32;
                    let py = (r * cell_size + dy) as u32;
                    img.put_pixel(px, py, color);
                }
            }
        }
    }

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    img.save(&args.output)?;
    println!("Saved Fracton Mask to {}.", args.output.display());

    Ok(())
}
