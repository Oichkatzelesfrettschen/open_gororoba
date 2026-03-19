use anyhow::Result;
use clap::Parser;
use faer::{Mat, Side, complex_native::c64};
use image::{ImageBuffer, Rgb};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Pseudospectrum slice")]
struct Args {
    #[arg(short, long, default_value = "data/artifacts/images/pseudospectrum_slice_rust_3160x2820.png")]
    output: PathBuf,
}

fn coolwarm(val: f64, vmin: f64, vmax: f64) -> Rgb<u8> {
    let mut norm = (val - vmin) / (vmax - vmin);
    norm = norm.clamp(0.0, 1.0);
    // Interpolation: Blue -> White -> Red
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
    println!("Running Pseudospectrum Slice...");

    let n = 80;
    let h = 1.0 / (n as f64 + 1.0);
    let mut l_mat = Mat::<f64>::zeros(n, n);
    
    // Build Dirichlet Laplacian
    for i in 0..n {
        l_mat.write(i, i, 2.0 / (h * h));
        if i > 0 {
            l_mat.write(i, i - 1, -1.0 / (h * h));
        }
        if i < n - 1 {
            l_mat.write(i, i + 1, -1.0 / (h * h));
        }
    }

    let evd = l_mat.selfadjoint_eigendecomposition(Side::Lower);
    let s = evd.s();
    let u = evd.u();

    let mut w_vals = Vec::with_capacity(n);
    for i in 0..n {
        w_vals.push(s.column_vector().read(i).max(0.0)); // Ensure non-negative
    }

    let lpow = |power: f64| -> Mat<f64> {
        let mut p_mat = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    // V * diag(w^s) * V^T
                    sum += u.read(i, k) * w_vals[k].powf(power) * u.read(j, k);
                }
                p_mat.write(i, j, sum);
            }
        }
        p_mat
    };

    let s0 = 1.0;
    let s1 = 0.7;
    let s2 = 0.4;
    let c0 = 1.0;
    let c1 = 0.5;
    let c2 = 0.3;
    let lam = 0.7;

    let lp0 = lpow(s0);
    let lp1 = lpow(s1);
    let lp2 = lpow(s2);

    let mut l_lam = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let val = c0 * lp0.read(i, j)
                    + lam * c1 * lp1.read(i, j)
                    + (lam * lam) * c2 * lp2.read(i, j);
            l_lam.write(i, j, val);
        }
    }

    let n_re = 20;
    let n_im = 20;
    
    let re_min = 0.0;
    let re_max = 40.0;
    let im_min = -12.0;
    let im_max = 12.0;

    let mut smin = vec![vec![0.0; n_re]; n_im];

    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for i in 0..n_im {
        let im_z = im_min + (im_max - im_min) * (i as f64) / (n_im as f64 - 1.0);
        for j in 0..n_re {
            let re_z = re_min + (re_max - re_min) * (j as f64) / (n_re as f64 - 1.0);
            
            let mut a = Mat::<c64>::zeros(n, n);
            for r in 0..n {
                for c in 0..n {
                    if r == c {
                        a.write(r, c, c64::new(re_z - l_lam.read(r, c), im_z));
                    } else {
                        a.write(r, c, c64::new(-l_lam.read(r, c), 0.0));
                    }
                }
            }

            let svd = a.svd();
            let sing_vals = svd.s_diagonal();
            
            let mut s_min_val = f64::INFINITY;
            for k in 0..n {
                let sv = sing_vals.read(k).re; // SVD of c64 matrix yields real singular values
                if sv < s_min_val {
                    s_min_val = sv;
                }
            }
            
            let log_smin = (s_min_val + 1e-14).log10();
            smin[i][j] = log_smin;
            
            if log_smin < min_val { min_val = log_smin; }
            if log_smin > max_val { max_val = log_smin; }
        }
    }

    // Upscale to a larger image
    let cell_size = 40;
    let img_w = (n_re * cell_size) as u32;
    let img_h = (n_im * cell_size) as u32;
    let mut img = ImageBuffer::new(img_w, img_h);

    for i in 0..n_im {
        for j in 0..n_re {
            let color = coolwarm(smin[i][j], min_val, max_val);
            // Flip Y axis so origin is bottom-left as in imshow(origin='lower')
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
