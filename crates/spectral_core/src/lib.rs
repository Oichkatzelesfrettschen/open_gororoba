//! spectral_core: Fractional Laplacian and spectral methods for PDEs.
//!
//! This crate provides:
//! - Periodic fractional Laplacian via FFT multiplier
//! - Dirichlet fractional Laplacian via DST-I
//! - 1D, 2D, and 3D implementations
//! - Negative-dimension PDE solver with regularization
//!
//! # Literature
//! - Bucur & Valdinoci (2016): Nonlocal Diffusion and Applications
//! - Kwasnicki (2017): Ten equivalent definitions of the fractional Laplacian
//! - Caffarelli & Silvestre (2007): Extension problem for fractional Laplacian

pub mod neg_dim;

pub use neg_dim::{
    build_kinetic_operator, caffarelli_silvestre_eigenvalues, eigenvalues_imaginary_time,
    epsilon_convergence_sweep, ConvergenceResult, EigenResult,
};

use ndarray::{Array2, Array3, Axis};
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Compute discrete periodic (-Delta)^s via FFT multiplier.
///
/// Uses |k|^{2s} multiplier in Fourier space for periodic boundary conditions.
///
/// # Arguments
/// * `u` - Input array (samples on periodic grid)
/// * `s` - Fractional power (s > 0, typically in (0, 1])
/// * `length` - Physical domain length L
///
/// # Returns
/// Array of same length: (-Delta)^s u
pub fn fractional_laplacian_periodic_1d(u: &[f64], s: f64, length: f64) -> Vec<f64> {
    let n = u.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Convert to complex
    let mut buffer: Vec<Complex64> = u.iter().map(|&x| Complex64::new(x, 0.0)).collect();

    // Forward FFT
    fft.process(&mut buffer);

    // Apply multiplier |k|^{2s}
    let _dx = length / n as f64;
    for (i, val) in buffer.iter_mut().enumerate() {
        // Frequency in cycles per unit length, then angular frequency
        let freq = if i <= n / 2 {
            i as f64
        } else {
            i as f64 - n as f64
        };
        let k = 2.0 * PI * freq / length;
        let mult = k.abs().powf(2.0 * s);
        *val *= mult;
    }

    // Inverse FFT
    ifft.process(&mut buffer);

    // Normalize and return real part
    let scale = 1.0 / n as f64;
    buffer.iter().map(|c| c.re * scale).collect()
}

/// Compute discrete Dirichlet (-Delta)^s via DST-I.
///
/// Uses the spectral definition with discrete sine transform for zero boundary conditions.
///
/// # Arguments
/// * `u_interior` - Values on interior grid points (boundaries are zero)
/// * `s` - Fractional power (s > 0)
/// * `length` - Physical domain length L
///
/// # Returns
/// Array of same length: (-Delta)^s u
pub fn fractional_laplacian_dirichlet_1d(u_interior: &[f64], s: f64, length: f64) -> Vec<f64> {
    let n = u_interior.len();
    if n < 1 {
        return vec![];
    }

    let h = length / (n + 1) as f64;

    // Dirichlet Laplacian eigenvalues
    let eigenvalues: Vec<f64> = (1..=n)
        .map(|k| {
            let arg = PI * k as f64 / (2.0 * (n + 1) as f64);
            (4.0 / (h * h)) * arg.sin().powi(2)
        })
        .collect();

    // DST-I forward
    let coeff = dst_i(u_interior);

    // Multiply by eigenvalues^s
    let scaled: Vec<f64> = coeff
        .iter()
        .zip(eigenvalues.iter())
        .map(|(&c, &lam)| c * lam.powf(s))
        .collect();

    // DST-I inverse (DST-I is its own inverse up to normalization)
    idst_i(&scaled)
}

/// Discrete Sine Transform Type I (orthonormal).
///
/// Computes DST-I using the FFT relationship:
/// DST-I of length N can be computed via FFT of length 2(N+1).
fn dst_i(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // Construct extended sequence: [0, x_0, x_1, ..., x_{n-1}, 0, -x_{n-1}, ..., -x_0]
    let m = 2 * (n + 1);
    let mut extended: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); m];

    // Fill: extended[1..n+1] = x, extended[n+2..2n+2] = -reversed(x)
    for (i, &val) in x.iter().enumerate() {
        extended[i + 1] = Complex64::new(val, 0.0);
        extended[m - 1 - i] = Complex64::new(-val, 0.0);
    }

    // FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(m);
    fft.process(&mut extended);

    // Extract imaginary parts (scaled)
    let scale = (2.0 / (n + 1) as f64).sqrt() / 2.0;
    (1..=n).map(|k| -extended[k].im * scale).collect()
}

/// Inverse Discrete Sine Transform Type I (orthonormal).
///
/// For orthonormal DST-I, the inverse is the same as forward up to scaling.
fn idst_i(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    // DST-I is its own inverse for orthonormal normalization
    let m = 2 * (n + 1);
    let mut extended: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); m];

    for (i, &val) in x.iter().enumerate() {
        extended[i + 1] = Complex64::new(val, 0.0);
        extended[m - 1 - i] = Complex64::new(-val, 0.0);
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(m);
    fft.process(&mut extended);

    let scale = (2.0 / (n + 1) as f64).sqrt() / 2.0;
    (1..=n).map(|k| -extended[k].im * scale).collect()
}

/// Compute discrete 2D periodic (-Delta)^s via FFT multiplier.
pub fn fractional_laplacian_periodic_2d(u: &Array2<f64>, s: f64, lx: f64, ly: f64) -> Array2<f64> {
    let (nx, ny) = u.dim();
    if nx < 2 || ny < 2 {
        return Array2::zeros((nx, ny));
    }

    let mut planner = FftPlanner::new();

    // 2D FFT via row-column decomposition
    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);
    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);

    // Convert to complex
    let mut buffer: Array2<Complex64> = u.mapv(|x| Complex64::new(x, 0.0));

    // FFT along rows
    for mut row in buffer.rows_mut() {
        let mut row_vec: Vec<Complex64> = row.to_vec();
        fft_x.process(&mut row_vec);
        for (i, val) in row_vec.into_iter().enumerate() {
            row[i] = val;
        }
    }

    // FFT along columns
    for mut col in buffer.columns_mut() {
        let mut col_vec: Vec<Complex64> = col.to_vec();
        fft_y.process(&mut col_vec);
        for (i, val) in col_vec.into_iter().enumerate() {
            col[i] = val;
        }
    }

    // Apply multiplier |k|^{2s}
    for ((i, j), val) in buffer.indexed_iter_mut() {
        let kx_freq = if i <= nx / 2 {
            i as f64
        } else {
            i as f64 - nx as f64
        };
        let ky_freq = if j <= ny / 2 {
            j as f64
        } else {
            j as f64 - ny as f64
        };
        let kx = 2.0 * PI * kx_freq / lx;
        let ky = 2.0 * PI * ky_freq / ly;
        let k2 = kx * kx + ky * ky;
        let mult = k2.powf(s);
        *val *= mult;
    }

    // IFFT along columns
    for mut col in buffer.columns_mut() {
        let mut col_vec: Vec<Complex64> = col.to_vec();
        ifft_y.process(&mut col_vec);
        for (i, val) in col_vec.into_iter().enumerate() {
            col[i] = val;
        }
    }

    // IFFT along rows
    for mut row in buffer.rows_mut() {
        let mut row_vec: Vec<Complex64> = row.to_vec();
        ifft_x.process(&mut row_vec);
        for (i, val) in row_vec.into_iter().enumerate() {
            row[i] = val;
        }
    }

    // Normalize and return real part
    let scale = 1.0 / (nx * ny) as f64;
    buffer.mapv(|c| c.re * scale)
}

/// Compute discrete 3D periodic (-Delta)^s via FFT multiplier.
pub fn fractional_laplacian_periodic_3d(
    u: &Array3<f64>,
    s: f64,
    lx: f64,
    ly: f64,
    lz: f64,
) -> Array3<f64> {
    let (nx, ny, nz) = u.dim();
    if nx < 2 || ny < 2 || nz < 2 {
        return Array3::zeros((nx, ny, nz));
    }

    let mut planner = FftPlanner::new();
    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);
    let fft_z = planner.plan_fft_forward(nz);
    let ifft_x = planner.plan_fft_inverse(nx);
    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_z = planner.plan_fft_inverse(nz);

    // Convert to complex
    let mut buffer: Array3<Complex64> = u.mapv(|x| Complex64::new(x, 0.0));

    // FFT along each axis
    // Axis 0 (x)
    for mut lane in buffer.lanes_mut(Axis(0)) {
        let mut v: Vec<Complex64> = lane.to_vec();
        fft_x.process(&mut v);
        for (i, val) in v.into_iter().enumerate() {
            lane[i] = val;
        }
    }
    // Axis 1 (y)
    for mut lane in buffer.lanes_mut(Axis(1)) {
        let mut v: Vec<Complex64> = lane.to_vec();
        fft_y.process(&mut v);
        for (i, val) in v.into_iter().enumerate() {
            lane[i] = val;
        }
    }
    // Axis 2 (z)
    for mut lane in buffer.lanes_mut(Axis(2)) {
        let mut v: Vec<Complex64> = lane.to_vec();
        fft_z.process(&mut v);
        for (i, val) in v.into_iter().enumerate() {
            lane[i] = val;
        }
    }

    // Apply multiplier
    for ((i, j, k), val) in buffer.indexed_iter_mut() {
        let kx_freq = if i <= nx / 2 {
            i as f64
        } else {
            i as f64 - nx as f64
        };
        let ky_freq = if j <= ny / 2 {
            j as f64
        } else {
            j as f64 - ny as f64
        };
        let kz_freq = if k <= nz / 2 {
            k as f64
        } else {
            k as f64 - nz as f64
        };
        let kx = 2.0 * PI * kx_freq / lx;
        let ky = 2.0 * PI * ky_freq / ly;
        let kz = 2.0 * PI * kz_freq / lz;
        let k2 = kx * kx + ky * ky + kz * kz;
        let mult = k2.powf(s);
        *val *= mult;
    }

    // IFFT along each axis (reverse order)
    for mut lane in buffer.lanes_mut(Axis(2)) {
        let mut v: Vec<Complex64> = lane.to_vec();
        ifft_z.process(&mut v);
        for (i, val) in v.into_iter().enumerate() {
            lane[i] = val;
        }
    }
    for mut lane in buffer.lanes_mut(Axis(1)) {
        let mut v: Vec<Complex64> = lane.to_vec();
        ifft_y.process(&mut v);
        for (i, val) in v.into_iter().enumerate() {
            lane[i] = val;
        }
    }
    for mut lane in buffer.lanes_mut(Axis(0)) {
        let mut v: Vec<Complex64> = lane.to_vec();
        ifft_x.process(&mut v);
        for (i, val) in v.into_iter().enumerate() {
            lane[i] = val;
        }
    }

    let scale = 1.0 / (nx * ny * nz) as f64;
    buffer.mapv(|c| c.re * scale)
}

/// Compute discrete 2D Dirichlet (-Delta)^s via tensor-product DST-I.
pub fn fractional_laplacian_dirichlet_2d(
    u_interior: &Array2<f64>,
    s: f64,
    lx: f64,
    ly: f64,
) -> Array2<f64> {
    let (nx, ny) = u_interior.dim();
    if nx < 1 || ny < 1 {
        return Array2::zeros((nx, ny));
    }

    let hx = lx / (nx + 1) as f64;
    let hy = ly / (ny + 1) as f64;

    // Eigenvalues for each dimension
    let lam_x: Vec<f64> = (1..=nx)
        .map(|k| {
            let arg = PI * k as f64 / (2.0 * (nx + 1) as f64);
            (4.0 / (hx * hx)) * arg.sin().powi(2)
        })
        .collect();

    let lam_y: Vec<f64> = (1..=ny)
        .map(|k| {
            let arg = PI * k as f64 / (2.0 * (ny + 1) as f64);
            (4.0 / (hy * hy)) * arg.sin().powi(2)
        })
        .collect();

    // 2D DST-I: apply along each axis
    let mut coeff = Array2::zeros((nx, ny));

    // DST along rows first
    for (i, row) in u_interior.rows().into_iter().enumerate() {
        let row_vec: Vec<f64> = row.to_vec();
        let transformed = dst_i(&row_vec);
        for (j, &val) in transformed.iter().enumerate() {
            coeff[[i, j]] = val;
        }
    }

    // DST along columns
    let mut coeff2 = Array2::zeros((nx, ny));
    for (j, col) in coeff.columns().into_iter().enumerate() {
        let col_vec: Vec<f64> = col.to_vec();
        let transformed = dst_i(&col_vec);
        for (i, &val) in transformed.iter().enumerate() {
            coeff2[[i, j]] = val;
        }
    }

    // Multiply by (lam_x[i] + lam_y[j])^s
    for ((i, j), val) in coeff2.indexed_iter_mut() {
        let lam = lam_x[i] + lam_y[j];
        *val *= lam.powf(s);
    }

    // Inverse DST (same as forward for orthonormal)
    // IDST along columns
    let mut result = Array2::zeros((nx, ny));
    for (j, col) in coeff2.columns().into_iter().enumerate() {
        let col_vec: Vec<f64> = col.to_vec();
        let transformed = idst_i(&col_vec);
        for (i, &val) in transformed.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    // IDST along rows
    let mut final_result = Array2::zeros((nx, ny));
    for (i, row) in result.rows().into_iter().enumerate() {
        let row_vec: Vec<f64> = row.to_vec();
        let transformed = idst_i(&row_vec);
        for (j, &val) in transformed.iter().enumerate() {
            final_result[[i, j]] = val;
        }
    }

    final_result
}

/// Standard 1D second-difference Dirichlet Laplacian.
///
/// Returns -Delta u where Delta is the discrete Laplacian with zero BCs.
pub fn dirichlet_laplacian_1d(u_interior: &[f64], length: f64) -> Vec<f64> {
    let n = u_interior.len();
    if n < 1 {
        return vec![];
    }

    let h = length / (n + 1) as f64;
    let h2 = h * h;

    (0..n)
        .map(|i| {
            let left = if i > 0 { u_interior[i - 1] } else { 0.0 };
            let right = if i + 1 < n { u_interior[i + 1] } else { 0.0 };
            -(left - 2.0 * u_interior[i] + right) / h2
        })
        .collect()
}

/// Compute Dirichlet Laplacian eigenvalues for 1D.
pub fn dirichlet_laplacian_eigenvalues_1d(n: usize, length: f64) -> Vec<f64> {
    let h = length / (n + 1) as f64;
    (1..=n)
        .map(|k| {
            let arg = PI * k as f64 / (2.0 * (n + 1) as f64);
            (4.0 / (h * h)) * arg.sin().powi(2)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_periodic_1d_s1_equals_laplacian() {
        // For s=1, (-Delta)^1 should equal -d^2/dx^2
        // For u = sin(2*pi*x/L), -d^2u/dx^2 = (2*pi/L)^2 * u
        let n = 32;
        let l = 1.0;
        let u: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
            .collect();

        let result = fractional_laplacian_periodic_1d(&u, 1.0, l);

        // Expected: (2*pi)^2 * u
        let expected_factor = (2.0 * PI).powi(2);
        for (&r, &orig) in result.iter().zip(u.iter()) {
            let expected = expected_factor * orig;
            assert_relative_eq!(r, expected, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_periodic_1d_constant_gives_zero() {
        let u = vec![1.0; 16];
        let result = fractional_laplacian_periodic_1d(&u, 0.5, 1.0);

        // Constant function has zero fractional Laplacian
        for &val in &result {
            assert_relative_eq!(val, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_dirichlet_1d_eigenfunction() {
        // First eigenfunction: u_k(x) = sin(k*pi*x/L) for k=1
        // (-Delta)^s u_k = lambda_k^s * u_k
        let n = 31;
        let l = 1.0;
        let h = l / (n + 1) as f64;

        // First eigenfunction on interior points
        let u: Vec<f64> = (1..=n).map(|i| (PI * i as f64 * h / l).sin()).collect();

        let eigenvalues = dirichlet_laplacian_eigenvalues_1d(n, l);
        let s = 0.5;

        let result = fractional_laplacian_dirichlet_1d(&u, s, l);

        // Expected: lambda_1^s * u
        let expected_mult = eigenvalues[0].powf(s);
        for (&r, &orig) in result.iter().zip(u.iter()) {
            let expected = expected_mult * orig;
            assert_relative_eq!(r, expected, epsilon = 1e-6, max_relative = 1e-6);
        }
    }

    #[test]
    fn test_dirichlet_laplacian_1d_consistency() {
        // Apply s=1 fractional Laplacian should match direct finite difference
        let n = 15;
        let l = 1.0;
        let u: Vec<f64> = (1..=n)
            .map(|i| {
                let x = i as f64 * l / (n + 1) as f64;
                x * (l - x) // Parabola vanishing at boundaries
            })
            .collect();

        let frac_result = fractional_laplacian_dirichlet_1d(&u, 1.0, l);
        let direct_result = dirichlet_laplacian_1d(&u, l);

        for (&f, &d) in frac_result.iter().zip(direct_result.iter()) {
            assert_relative_eq!(f, d, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn test_periodic_2d_constant_gives_zero() {
        let u = Array2::from_elem((8, 8), 1.0);
        let result = fractional_laplacian_periodic_2d(&u, 0.5, 1.0, 1.0);

        for &val in result.iter() {
            assert_relative_eq!(val, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_periodic_2d_wave() {
        // u = sin(2*pi*x) * sin(2*pi*y)
        // -Delta u = 2*(2*pi)^2 * u
        // (-Delta)^s u = [2*(2*pi)^2]^s * u
        let n = 16;
        let u = Array2::from_shape_fn((n, n), |(i, j)| {
            let x = i as f64 / n as f64;
            let y = j as f64 / n as f64;
            (2.0 * PI * x).sin() * (2.0 * PI * y).sin()
        });

        let result = fractional_laplacian_periodic_2d(&u, 1.0, 1.0, 1.0);

        let expected_mult = 2.0 * (2.0 * PI).powi(2);
        for ((i, j), &r) in result.indexed_iter() {
            let expected = expected_mult * u[[i, j]];
            assert_relative_eq!(r, expected, epsilon = 1e-8, max_relative = 1e-8);
        }
    }

    #[test]
    fn test_periodic_3d_constant_gives_zero() {
        let u = Array3::from_elem((4, 4, 4), 1.0);
        let result = fractional_laplacian_periodic_3d(&u, 0.5, 1.0, 1.0, 1.0);

        for &val in result.iter() {
            assert_relative_eq!(val, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_dirichlet_2d_separable() {
        // For separable eigenfunction sin(pi*x)*sin(pi*y),
        // eigenvalue = (pi/h)^2 + (pi/h)^2 (first mode in each direction)
        let n = 7;
        let l = 1.0;
        let h = l / (n + 1) as f64;

        let u = Array2::from_shape_fn((n, n), |(i, j)| {
            let x = (i + 1) as f64 * h;
            let y = (j + 1) as f64 * h;
            (PI * x / l).sin() * (PI * y / l).sin()
        });

        let result = fractional_laplacian_dirichlet_2d(&u, 1.0, l, l);

        // First eigenvalue in each direction
        let lam1 = 4.0 / (h * h) * (PI / (2.0 * (n + 1) as f64)).sin().powi(2);
        let total_eig = 2.0 * lam1;

        for ((i, j), &r) in result.indexed_iter() {
            let expected = total_eig * u[[i, j]];
            assert_relative_eq!(r, expected, epsilon = 1e-4, max_relative = 1e-3);
        }
    }

    #[test]
    fn test_dst_i_roundtrip() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let transformed = dst_i(&x);
        let recovered = idst_i(&transformed);

        for (&orig, &rec) in x.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_eigenvalue_formula() {
        let n = 10;
        let l = 1.0;
        let eigs = dirichlet_laplacian_eigenvalues_1d(n, l);

        // First eigenvalue should be close to (pi/L)^2 for small h
        // But for discrete, it's 4/h^2 * sin^2(pi/(2*(n+1)))
        let h = l / (n + 1) as f64;
        let expected_first = 4.0 / (h * h) * (PI / (2.0 * (n + 1) as f64)).sin().powi(2);
        assert_relative_eq!(eigs[0], expected_first, epsilon = 1e-12);

        // Eigenvalues should be strictly increasing
        for i in 1..eigs.len() {
            assert!(eigs[i] > eigs[i - 1]);
        }
    }
}
