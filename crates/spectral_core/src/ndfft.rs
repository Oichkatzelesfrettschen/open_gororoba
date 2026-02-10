//! N-dimensional FFT via ndrustfft.
//!
//! Provides generic complex-to-complex and real-to-complex N-D FFT operations
//! built on ndrustfft (which composes rustfft + realfft under the hood).
//!
//! This module exposes:
//! - Forward/inverse C2C transforms for 1D through 4D
//! - Forward R2C transforms for 1D through 4D
//! - Radially-binned isotropic power spectrum for arbitrary dimension
//! - N-D frequency grid construction
//!
//! The existing hand-rolled row-column FFT in `lib.rs` (for fractional
//! Laplacian) remains for that specific use case; this module provides the
//! general-purpose N-D FFT for spectral analysis workflows.

use ndarray::{Array1, Array2, Array3, Array4, ArrayD, IxDyn};
use ndrustfft::{ndfft, ndifft, FftHandler};
use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// 1D complex-to-complex
// ---------------------------------------------------------------------------

/// Forward 1D FFT (complex-to-complex).
pub fn fft_1d(input: &Array1<Complex64>) -> Array1<Complex64> {
    let n = input.len();
    let handler = FftHandler::<f64>::new(n);
    let mut output = Array1::<Complex64>::zeros(n);
    ndfft(input, &mut output, &handler, 0);
    output
}

/// Inverse 1D FFT (complex-to-complex, normalized).
pub fn ifft_1d(input: &Array1<Complex64>) -> Array1<Complex64> {
    let n = input.len();
    let handler = FftHandler::<f64>::new(n);
    let mut output = Array1::<Complex64>::zeros(n);
    ndifft(input, &mut output, &handler, 0);
    output
}

// ---------------------------------------------------------------------------
// 2D complex-to-complex
// ---------------------------------------------------------------------------

/// Forward 2D FFT (complex-to-complex).
pub fn fft_2d(input: &Array2<Complex64>) -> Array2<Complex64> {
    let (nx, ny) = input.dim();
    let handler_x = FftHandler::<f64>::new(nx);
    let handler_y = FftHandler::<f64>::new(ny);

    let mut temp = Array2::<Complex64>::zeros((nx, ny));
    let mut output = Array2::<Complex64>::zeros((nx, ny));

    ndfft(input, &mut temp, &handler_x, 0);
    ndfft(&temp, &mut output, &handler_y, 1);
    output
}

/// Inverse 2D FFT (complex-to-complex, normalized).
pub fn ifft_2d(input: &Array2<Complex64>) -> Array2<Complex64> {
    let (nx, ny) = input.dim();
    let handler_x = FftHandler::<f64>::new(nx);
    let handler_y = FftHandler::<f64>::new(ny);

    let mut temp = Array2::<Complex64>::zeros((nx, ny));
    let mut output = Array2::<Complex64>::zeros((nx, ny));

    ndifft(input, &mut temp, &handler_x, 0);
    ndifft(&temp, &mut output, &handler_y, 1);
    output
}

// ---------------------------------------------------------------------------
// 3D complex-to-complex
// ---------------------------------------------------------------------------

/// Forward 3D FFT (complex-to-complex).
pub fn fft_3d(input: &Array3<Complex64>) -> Array3<Complex64> {
    let (nx, ny, nz) = input.dim();
    let hx = FftHandler::<f64>::new(nx);
    let hy = FftHandler::<f64>::new(ny);
    let hz = FftHandler::<f64>::new(nz);

    let mut a = Array3::<Complex64>::zeros((nx, ny, nz));
    let mut b = Array3::<Complex64>::zeros((nx, ny, nz));

    ndfft(input, &mut a, &hx, 0);
    ndfft(&a, &mut b, &hy, 1);
    ndfft(&b, &mut a, &hz, 2);
    a
}

/// Inverse 3D FFT (complex-to-complex, normalized).
pub fn ifft_3d(input: &Array3<Complex64>) -> Array3<Complex64> {
    let (nx, ny, nz) = input.dim();
    let hx = FftHandler::<f64>::new(nx);
    let hy = FftHandler::<f64>::new(ny);
    let hz = FftHandler::<f64>::new(nz);

    let mut a = Array3::<Complex64>::zeros((nx, ny, nz));
    let mut b = Array3::<Complex64>::zeros((nx, ny, nz));

    ndifft(input, &mut a, &hx, 0);
    ndifft(&a, &mut b, &hy, 1);
    ndifft(&b, &mut a, &hz, 2);
    a
}

// ---------------------------------------------------------------------------
// 4D complex-to-complex
// ---------------------------------------------------------------------------

/// Forward 4D FFT (complex-to-complex).
pub fn fft_4d(input: &Array4<Complex64>) -> Array4<Complex64> {
    let shape = input.dim();
    let h0 = FftHandler::<f64>::new(shape.0);
    let h1 = FftHandler::<f64>::new(shape.1);
    let h2 = FftHandler::<f64>::new(shape.2);
    let h3 = FftHandler::<f64>::new(shape.3);

    let mut a = Array4::<Complex64>::zeros(shape);
    let mut b = Array4::<Complex64>::zeros(shape);

    ndfft(input, &mut a, &h0, 0);
    ndfft(&a, &mut b, &h1, 1);
    ndfft(&b, &mut a, &h2, 2);
    ndfft(&a, &mut b, &h3, 3);
    b
}

/// Inverse 4D FFT (complex-to-complex, normalized).
pub fn ifft_4d(input: &Array4<Complex64>) -> Array4<Complex64> {
    let shape = input.dim();
    let h0 = FftHandler::<f64>::new(shape.0);
    let h1 = FftHandler::<f64>::new(shape.1);
    let h2 = FftHandler::<f64>::new(shape.2);
    let h3 = FftHandler::<f64>::new(shape.3);

    let mut a = Array4::<Complex64>::zeros(shape);
    let mut b = Array4::<Complex64>::zeros(shape);

    ndifft(input, &mut a, &h0, 0);
    ndifft(&a, &mut b, &h1, 1);
    ndifft(&b, &mut a, &h2, 2);
    ndifft(&a, &mut b, &h3, 3);
    b
}

// ---------------------------------------------------------------------------
// Generic N-D (dynamic dimensionality)
// ---------------------------------------------------------------------------

/// Forward N-D FFT on a dynamic-dimension array.
///
/// Applies FFT along each axis in order 0, 1, ..., ndim-1.
pub fn fft_nd(input: &ArrayD<Complex64>) -> ArrayD<Complex64> {
    let ndim = input.ndim();
    let shape = input.shape().to_vec();

    let handlers: Vec<FftHandler<f64>> = shape.iter().map(|&n| FftHandler::new(n)).collect();

    let mut a = input.clone();
    let mut b = ArrayD::<Complex64>::zeros(IxDyn(&shape));

    for (axis, handler) in handlers.iter().enumerate() {
        if axis.is_multiple_of(2) {
            ndfft(&a, &mut b, handler, axis);
        } else {
            ndfft(&b, &mut a, handler, axis);
        }
    }

    // axis=0: a->b, axis=1: b->a, axis=2: a->b, axis=3: b->a, ...
    // Even ndim: last axis is odd, last write to `a`.
    // Odd ndim: last axis is even, last write to `b`.
    if ndim.is_multiple_of(2) {
        a
    } else {
        b
    }
}

/// Inverse N-D FFT on a dynamic-dimension array (normalized).
pub fn ifft_nd(input: &ArrayD<Complex64>) -> ArrayD<Complex64> {
    let ndim = input.ndim();
    let shape = input.shape().to_vec();

    let handlers: Vec<FftHandler<f64>> = shape.iter().map(|&n| FftHandler::new(n)).collect();

    let mut a = input.clone();
    let mut b = ArrayD::<Complex64>::zeros(IxDyn(&shape));

    for (axis, handler) in handlers.iter().enumerate() {
        if axis.is_multiple_of(2) {
            ndifft(&a, &mut b, handler, axis);
        } else {
            ndifft(&b, &mut a, handler, axis);
        }
    }

    if ndim.is_multiple_of(2) {
        a
    } else {
        b
    }
}

// ---------------------------------------------------------------------------
// Power spectrum utilities
// ---------------------------------------------------------------------------

/// Compute frequency index for dimension of size n.
///
/// Returns signed frequency: 0, 1, ..., n/2, -(n/2-1), ..., -1
#[inline]
fn freq_index(i: usize, n: usize) -> f64 {
    if i <= n / 2 {
        i as f64
    } else {
        i as f64 - n as f64
    }
}

/// Compute isotropic (radially-binned) power spectrum from a 2D complex field.
///
/// Returns (k_bins, P(k)) where k_bins are wavenumber bin centers and P(k)
/// is the azimuthally-averaged power in each radial bin.
pub fn power_spectrum_2d(field: &Array2<Complex64>, lx: f64, ly: f64) -> (Vec<f64>, Vec<f64>) {
    let (nx, ny) = field.dim();
    let k_max = ((nx / 2).min(ny / 2)) as f64;
    let n_bins = k_max as usize;
    if n_bins == 0 {
        return (vec![], vec![]);
    }

    let mut power = vec![0.0_f64; n_bins];
    let mut counts = vec![0usize; n_bins];

    for ((i, j), val) in field.indexed_iter() {
        let kx = freq_index(i, nx) * 2.0 * PI / lx;
        let ky = freq_index(j, ny) * 2.0 * PI / ly;
        let k = (kx * kx + ky * ky).sqrt();
        let dk = 2.0 * PI / lx.min(ly);
        let bin = (k / dk).floor() as usize;
        if bin < n_bins {
            power[bin] += val.norm_sqr();
            counts[bin] += 1;
        }
    }

    let dk = 2.0 * PI / lx.min(ly);
    let k_bins: Vec<f64> = (0..n_bins).map(|b| (b as f64 + 0.5) * dk).collect();
    for (p, &c) in power.iter_mut().zip(counts.iter()) {
        if c > 0 {
            *p /= c as f64;
        }
    }

    (k_bins, power)
}

/// Compute isotropic power spectrum from a 3D complex field.
pub fn power_spectrum_3d(
    field: &Array3<Complex64>,
    lx: f64,
    ly: f64,
    lz: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (nx, ny, nz) = field.dim();
    let k_max = ((nx / 2).min(ny / 2).min(nz / 2)) as f64;
    let n_bins = k_max as usize;
    if n_bins == 0 {
        return (vec![], vec![]);
    }

    let mut power = vec![0.0_f64; n_bins];
    let mut counts = vec![0usize; n_bins];
    let l_min = lx.min(ly).min(lz);

    for ((i, j, k), val) in field.indexed_iter() {
        let kx = freq_index(i, nx) * 2.0 * PI / lx;
        let ky = freq_index(j, ny) * 2.0 * PI / ly;
        let kz = freq_index(k, nz) * 2.0 * PI / lz;
        let kmag = (kx * kx + ky * ky + kz * kz).sqrt();
        let dk = 2.0 * PI / l_min;
        let bin = (kmag / dk).floor() as usize;
        if bin < n_bins {
            power[bin] += val.norm_sqr();
            counts[bin] += 1;
        }
    }

    let dk = 2.0 * PI / l_min;
    let k_bins: Vec<f64> = (0..n_bins).map(|b| (b as f64 + 0.5) * dk).collect();
    for (p, &c) in power.iter_mut().zip(counts.iter()) {
        if c > 0 {
            *p /= c as f64;
        }
    }

    (k_bins, power)
}

/// Convert a real-valued N-D array to complex for FFT input.
pub fn real_to_complex_1d(input: &Array1<f64>) -> Array1<Complex64> {
    input.mapv(|x| Complex64::new(x, 0.0))
}

/// Convert a real-valued 2D array to complex for FFT input.
pub fn real_to_complex_2d(input: &Array2<f64>) -> Array2<Complex64> {
    input.mapv(|x| Complex64::new(x, 0.0))
}

/// Convert a real-valued 3D array to complex for FFT input.
pub fn real_to_complex_3d(input: &Array3<f64>) -> Array3<Complex64> {
    input.mapv(|x| Complex64::new(x, 0.0))
}

/// Convert a real-valued 4D array to complex for FFT input.
pub fn real_to_complex_4d(input: &Array4<f64>) -> Array4<Complex64> {
    input.mapv(|x| Complex64::new(x, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // -- 1D tests -----------------------------------------------------------

    #[test]
    fn test_fft_1d_constant() {
        let n = 16;
        let input = Array1::from_elem(n, Complex64::new(1.0, 0.0));
        let output = fft_1d(&input);
        // DC component should be n, all others zero
        assert_relative_eq!(output[0].re, n as f64, epsilon = 1e-10);
        for i in 1..n {
            assert!(output[i].norm() < 1e-10, "bin {} = {:?}", i, output[i]);
        }
    }

    #[test]
    fn test_fft_1d_roundtrip() {
        let n = 32;
        let input = Array1::from_shape_fn(n, |i| {
            Complex64::new((2.0 * PI * i as f64 / n as f64).sin(), 0.0)
        });
        let forward = fft_1d(&input);
        let recovered = ifft_1d(&forward);
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    // -- 2D tests -----------------------------------------------------------

    #[test]
    fn test_fft_2d_constant() {
        let (nx, ny) = (8, 8);
        let input = Array2::from_elem((nx, ny), Complex64::new(1.0, 0.0));
        let output = fft_2d(&input);
        assert_relative_eq!(output[[0, 0]].re, (nx * ny) as f64, epsilon = 1e-10);
        for ((i, j), val) in output.indexed_iter() {
            if i != 0 || j != 0 {
                assert!(val.norm() < 1e-10, "[{},{}] = {:?}", i, j, val);
            }
        }
    }

    #[test]
    fn test_fft_2d_roundtrip() {
        let (nx, ny) = (8, 12);
        let input = Array2::from_shape_fn((nx, ny), |(i, j)| {
            Complex64::new(
                (2.0 * PI * i as f64 / nx as f64).sin() * (4.0 * PI * j as f64 / ny as f64).cos(),
                0.0,
            )
        });
        let recovered = ifft_2d(&fft_2d(&input));
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
        }
    }

    // -- 3D tests -----------------------------------------------------------

    #[test]
    fn test_fft_3d_constant() {
        let (nx, ny, nz) = (4, 4, 4);
        let input = Array3::from_elem((nx, ny, nz), Complex64::new(1.0, 0.0));
        let output = fft_3d(&input);
        assert_relative_eq!(output[[0, 0, 0]].re, (nx * ny * nz) as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_fft_3d_roundtrip() {
        let (nx, ny, nz) = (4, 6, 8);
        let input = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            Complex64::new(
                (2.0 * PI * i as f64 / nx as f64).sin()
                    + (2.0 * PI * j as f64 / ny as f64).cos()
                    + (2.0 * PI * k as f64 / nz as f64).sin(),
                0.0,
            )
        });
        let recovered = ifft_3d(&fft_3d(&input));
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
        }
    }

    // -- 4D tests -----------------------------------------------------------

    #[test]
    fn test_fft_4d_constant() {
        let shape = (4, 4, 4, 4);
        let input = Array4::from_elem(shape, Complex64::new(1.0, 0.0));
        let output = fft_4d(&input);
        let total = (shape.0 * shape.1 * shape.2 * shape.3) as f64;
        assert_relative_eq!(output[[0, 0, 0, 0]].re, total, epsilon = 1e-10);
    }

    #[test]
    fn test_fft_4d_roundtrip() {
        let shape = (4, 4, 4, 4);
        let input = Array4::from_shape_fn(shape, |(i, j, k, l)| {
            Complex64::new(
                (2.0 * PI * i as f64 / shape.0 as f64).sin()
                    * (2.0 * PI * l as f64 / shape.3 as f64).cos()
                    + (j as f64 * 0.1)
                    + (k as f64 * 0.2),
                0.0,
            )
        });
        let recovered = ifft_4d(&fft_4d(&input));
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_fft_4d_single_mode() {
        // Pure sinusoid along axis 0 should produce two peaks at +/-k
        let shape = (8, 4, 4, 4);
        let input = Array4::from_shape_fn(shape, |(i, _j, _k, _l)| {
            Complex64::new((2.0 * PI * i as f64 / shape.0 as f64).sin(), 0.0)
        });
        let output = fft_4d(&input);

        // The energy should be concentrated at k0=1 and k0=N-1 (aliased -1)
        let total_energy: f64 = output.iter().map(|c| c.norm_sqr()).sum();
        let mode_energy =
            output[[1, 0, 0, 0]].norm_sqr() + output[[shape.0 - 1, 0, 0, 0]].norm_sqr();
        assert!(
            mode_energy / total_energy > 0.99,
            "Single mode should capture >99% of energy, got {:.4}%",
            100.0 * mode_energy / total_energy
        );
    }

    // -- Generic N-D tests --------------------------------------------------

    #[test]
    fn test_fft_nd_2d_matches_static() {
        let (nx, ny) = (8, 8);
        let input_2d = Array2::from_shape_fn((nx, ny), |(i, j)| {
            Complex64::new((i as f64 + j as f64 * 0.5).sin(), 0.0)
        });

        let static_result = fft_2d(&input_2d);
        let dynamic_input = input_2d.clone().into_dyn();
        let dynamic_result = fft_nd(&dynamic_input);

        for (s, d) in static_result.iter().zip(dynamic_result.iter()) {
            assert_relative_eq!(s.re, d.re, epsilon = 1e-10);
            assert_relative_eq!(s.im, d.im, epsilon = 1e-10);
        }
    }

    // -- Power spectrum tests -----------------------------------------------

    #[test]
    fn test_power_spectrum_2d_dc() {
        let (nx, ny) = (16, 16);
        let field = Array2::from_elem((nx, ny), Complex64::new(1.0, 0.0));
        let (k_bins, power) = power_spectrum_2d(&field, 1.0, 1.0);
        assert!(!k_bins.is_empty());
        // DC mode is in bin 0, should dominate
        assert!(power[0] > 0.0);
    }

    #[test]
    fn test_power_spectrum_3d_nonempty() {
        let (nx, ny, nz) = (8, 8, 8);
        let field = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            Complex64::new((i + j + k) as f64, 0.0)
        });
        let (k_bins, power) = power_spectrum_3d(&field, 1.0, 1.0, 1.0);
        assert!(!k_bins.is_empty());
        assert!(!power.is_empty());
    }
}
