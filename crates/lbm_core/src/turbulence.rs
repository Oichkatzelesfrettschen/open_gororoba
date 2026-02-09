//! Turbulence Statistics and Spectral Analysis.
//!
//! Provides tools for analyzing flow fields in spectral space,
//! extracting triad interactions, and computing hypergraph metrics.

use ndarray::{Array2, Axis};
use rustfft::{num_complex::Complex, FftPlanner};

/// Triad interaction in spectral space (k, p, q).
#[derive(Debug, Clone)]
pub struct SpectralTriad {
    pub k: [usize; 2],
    pub p: [usize; 2],
    pub q: [usize; 2],
    pub energy_transfer: f64,
}

/// Compute 2D power spectrum of a scalar field.
pub fn power_spectrum(field: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let (nx, ny) = field.dim();
    let mut planner = FftPlanner::new();
    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);

    // Create complex buffer
    // Rows first
    let mut transformed = Array2::<Complex<f64>>::zeros((nx, ny));

    for (i, row) in field.axis_iter(Axis(0)).enumerate() {
        let mut buffer: Vec<Complex<f64>> = row.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft_y.process(&mut buffer);
        for (j, val) in buffer.iter().enumerate() {
            transformed[[i, j]] = *val;
        }
    }

    // Then columns
    for j in 0..ny {
        let mut col_buffer: Vec<Complex<f64>> = (0..nx).map(|i| transformed[[i, j]]).collect();
        fft_x.process(&mut col_buffer);
        for (i, val) in col_buffer.iter().enumerate() {
            transformed[[i, j]] = *val;
        }
    }

    // Binning by wavenumber k = sqrt(kx^2 + ky^2)
    let n_bins = nx / 2;
    let mut bins = vec![0.0; n_bins];
    let mut counts = vec![0.0; n_bins];

    for x in 0..nx {
        for y in 0..ny {
            // Shifted frequencies
            let kx = if x <= nx / 2 {
                x as f64
            } else {
                (x as f64) - (nx as f64)
            };
            let ky = if y <= ny / 2 {
                y as f64
            } else {
                (y as f64) - (ny as f64)
            };
            let k = (kx * kx + ky * ky).sqrt();
            let bin_idx = k as usize;

            if bin_idx < n_bins {
                bins[bin_idx] += transformed[[x, y]].norm_sqr();
                counts[bin_idx] += 1.0;
            }
        }
    }

    let k_axis: Vec<f64> = (0..n_bins).map(|i| i as f64).collect();
    let power: Vec<f64> = bins
        .iter()
        .zip(counts.iter())
        .map(|(b, c)| if *c > 0.0 { *b / *c } else { 0.0 })
        .collect();

    (k_axis, power)
}

/// Extract dominant triads from flow field.
///
/// This function identifies wavevector triplets (k,p,q) with significant
/// energy transfer T(k,p,q).
///
/// NOTE: Full N^2 convolution is expensive. This implementation uses a
/// randomized sampling approach to find triads above threshold.
pub fn extract_dominant_triads(
    u: &Array2<f64>,
    _v: &Array2<f64>,
    threshold: f64,
) -> Vec<SpectralTriad> {
    let (nx, ny) = u.dim();
    let mut planner = FftPlanner::new();
    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);

    // FFT of u -> U_k
    let mut uk_field = Array2::<Complex<f64>>::zeros((nx, ny));

    // 2D FFT logic (simplified copy from above)
    for (i, row) in u.axis_iter(Axis(0)).enumerate() {
        let mut buffer: Vec<Complex<f64>> = row.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft_y.process(&mut buffer);
        for (j, val) in buffer.iter().enumerate() {
            uk_field[[i, j]] = *val;
        }
    }
    for j in 0..ny {
        let mut col_buffer: Vec<Complex<f64>> = (0..nx).map(|i| uk_field[[i, j]]).collect();
        fft_x.process(&mut col_buffer);
        for (i, val) in col_buffer.iter().enumerate() {
            uk_field[[i, j]] = *val;
        }
    }

    // Sampling triads k+p+q = 0
    let mut triads = Vec::new();
    let n_samples = 1000; // Limit samples for performance

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..n_samples {
        let kx = rng.gen_range(0..nx);
        let ky = rng.gen_range(0..ny);

        let px = rng.gen_range(0..nx);
        let py = rng.gen_range(0..ny);

        // q = -(k+p) mod N
        let qx = (nx - (kx + px) % nx) % nx;
        let qy = (ny - (ky + py) % ny) % ny;

        let val_k = uk_field[[kx, ky]];
        let val_p = uk_field[[px, py]];
        let val_q = uk_field[[qx, qy]];

        let energy = calculate_triad_energy_transfer(val_k, val_p, val_q);

        if energy > threshold {
            triads.push(SpectralTriad {
                k: [kx, ky],
                p: [px, py],
                q: [qx, qy],
                energy_transfer: energy,
            });
        }
    }

    triads
}

/// Calculate energy transfer for a specific triad.
///
/// T(k,p,q) = -Im[ u_k* \cdot (u_p \cdot \nabla) u_q ]
///
/// Simplified proxy: T ~ |uk||up||uq|
pub fn calculate_triad_energy_transfer(
    uk: Complex<f64>,
    up: Complex<f64>,
    uq: Complex<f64>,
) -> f64 {
    uk.norm() * up.norm() * uq.norm()
}

/// Compute clustering coefficient of the triad interaction graph.
pub fn triad_clustering_coefficient(triads: &[SpectralTriad]) -> f64 {
    if triads.is_empty() {
        return 0.0;
    }
    // Simple heuristic for now: pure count / max possible
    let n = triads.len() as f64;
    (n / (n + 100.0)).min(1.0)
}
