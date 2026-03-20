use anyhow::Result;
use clap::Parser;
use image::{ImageBuffer, Rgb};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fractal CSS Expanded Stabilizer Search")]
struct Args {
    #[arg(short, long, default_value = "data/artifacts/images/css_fractal_candidate_logical_expanded_rust.png")]
    output: PathBuf,
}

// GF(2) rank via Gaussian elimination
fn gf2_rank(matrix: &mut [Vec<u8>]) -> usize {
    if matrix.is_empty() {
        return 0;
    }
    let m = matrix.len();
    let n = matrix[0].len();
    let mut r = 0;

    for c in 0..n {
        let mut piv = None;
        #[allow(clippy::needless_range_loop)]
        for i in r..m {
            if matrix[i][c] == 1 {
                piv = Some(i);
                break;
            }
        }
        
        if let Some(pivot_row) = piv {
            if pivot_row != r {
                matrix.swap(r, pivot_row);
            }
            #[allow(clippy::needless_range_loop)]
            for i in 0..m {
                if i != r && matrix[i][c] == 1 {
                    for col in 0..n {
                        matrix[i][col] ^= matrix[r][col];
                    }
                }
            }
            r += 1;
            if r == m {
                break;
            }
        }
    }
    r
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Computing GF(2) Ranks for Fractal Code...");

    let l = 33;
    let n_qubits = l * l;
    let mut s = vec![vec![0u8; l]; l];

    // Seed Rule-90
    s[0][l / 2] = 1;
    for r in 1..l {
        for c in 0..l {
            let left_idx = if c == 0 { l - 1 } else { c - 1 };
            let right_idx = if c == l - 1 { 0 } else { c + 1 };
            let left = s[r - 1][left_idx];
            let right = s[r - 1][right_idx];
            s[r][c] = left ^ right;
        }
    }

    let mut x_parity = vec![vec![0u8; l]; l];
    let mut z_parity = vec![vec![0u8; l]; l];
    for r in 0..l {
        for c in 0..l {
            if (r + c) % 2 == 0 {
                x_parity[r][c] = 1;
            } else {
                z_parity[r][c] = 1;
            }
        }
    }

    let translate = |mask: &[Vec<u8>], tx: usize, ty: usize| -> Vec<Vec<u8>> {
        let mut out = vec![vec![0u8; l]; l];
        #[allow(clippy::needless_range_loop)]
        for r in 0..l {
            for c in 0..l {
                let src_r = (r + l - ty % l) % l;
                let src_c = (c + l - tx % l) % l;
                out[r][c] = mask[src_r][src_c];
            }
        }
        out
    };

    let x_shifts = vec![(0,0),(2,0),(0,2),(2,2),(4,2),(2,4),(6,0),(0,6),(6,4),(4,6),(8,2),(2,8)];
    let z_shifts = vec![(2,0),(0,2),(4,0),(0,4),(6,2),(2,6),(4,4),(8,0),(0,8),(6,6),(8,4),(4,8)];

    let flat = |mask: &[Vec<u8>]| -> Vec<u8> {
        let mut out = Vec::with_capacity(n_qubits);
        #[allow(clippy::needless_range_loop)]
        for r in 0..l {
            for c in 0..l {
                out.push(mask[r][c] % 2);
            }
        }
        out
    };

    let mut hx_orig = Vec::new();
    for &(tx, ty) in &x_shifts {
        let translated = translate(&s, tx, ty);
        let mut mask = vec![vec![0u8; l]; l];
        for r in 0..l {
            for c in 0..l {
                mask[r][c] = translated[r][c] & x_parity[r][c];
            }
        }
        hx_orig.push(flat(&mask));
    }

    let mut hz = Vec::new();
    for &(tx, ty) in &z_shifts {
        let translated = translate(&s, tx, ty);
        let mut mask = vec![vec![0u8; l]; l];
        for r in 0..l {
            for c in 0..l {
                mask[r][c] = translated[r][c] & z_parity[r][c];
            }
        }
        hz.push(flat(&mask));
    }

    let mut hx_working = hx_orig.clone();
    let mut hz_working = hz.clone();
    let rank_hx = gf2_rank(&mut hx_working);
    let rank_hz = gf2_rank(&mut hz_working);
    let k = n_qubits - rank_hx - rank_hz;

    println!("L={}, n={}, rank(HX)={}, rank(HZ)={}, k={}", l, n_qubits, rank_hx, rank_hz, k);

    let mut best: Option<(usize, usize, usize, Vec<Vec<u8>>)> = None;

    for tx in (0..l).step_by(2) {
        for ty in (0..l).step_by(2) {
            let translated = translate(&s, tx, ty);
            let mut xmask = vec![vec![0u8; l]; l];
            for r in 0..l {
                for c in 0..l {
                    xmask[r][c] = translated[r][c] & x_parity[r][c];
                }
            }
            let xrow = flat(&xmask);

            // must commute with all Z stabilizers
            let mut commutes = true;
            for zrow in &hz {
                let mut dot = 0;
                for i in 0..n_qubits {
                    dot ^= xrow[i] & zrow[i];
                }
                if dot != 0 {
                    commutes = false;
                    break;
                }
            }
            if !commutes { continue; }

            // must be independent of H_X
            let mut hx_ext = hx_orig.clone();
            hx_ext.push(xrow.clone());
            if gf2_rank(&mut hx_ext) == rank_hx {
                continue;
            }

            let weight: usize = xrow.iter().map(|&x| x as usize).sum();
            if best.is_none() || weight < best.as_ref().unwrap().0 {
                best = Some((weight, tx, ty, xmask));
            }
        }
    }

    if let Some((w_best, tx_best, ty_best, xlog)) = best {
        println!("Found X-Logical: w={}, shift=({},{})", w_best, tx_best, ty_best);

        let cell_size = 20;
        let img_size = (l * cell_size) as u32;
        let mut img = ImageBuffer::new(img_size, img_size);

        let bg_color = Rgb([13u8, 15u8, 20u8]);
        let blue = Rgb([97u8, 166u8, 245u8]);

        #[allow(clippy::needless_range_loop)]
        for r in 0..l {
            for c in 0..l {
                let color = if xlog[r][c] == 1 { blue } else { bg_color };
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
        println!("Saved: {}", args.output.display());
    } else {
        println!("No candidate X-logical found in this restricted family.");
    }

    Ok(())
}
