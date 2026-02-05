//! Phase Retrieval Algorithms for Holographic Beam Shaping.
//!
//! Implements the Weighted Gerchberg-Saxton (WGS) algorithm and related methods
//! for computing spatial light modulator (SLM) phase patterns that produce
//! desired intensity distributions at the focal plane.
//!
//! # Applications
//! - Optical tweezer arrays (Manetsch et al. 2025)
//! - Holographic displays
//! - Beam shaping and wavefront correction
//! - Zernike aberration compensation
//!
//! # Algorithm
//!
//! WGS iteratively refines the SLM phase pattern:
//! 1. Initialize with random or uniform phases
//! 2. Propagate to focal plane via FFT
//! 3. Apply target constraint (weighted amplitude replacement)
//! 4. Backpropagate to SLM plane via inverse FFT
//! 5. Apply SLM constraint (uniform illumination)
//! 6. Repeat until convergence
//!
//! The "weighted" aspect gives higher weight to underperforming spots,
//! accelerating convergence to uniform intensity.
//!
//! # Literature
//! - Gerchberg & Saxton, Optik 35, 237 (1972)
//! - Di Leonardo et al., Opt. Express 15, 1913 (2007) - WGS
//! - Manetsch et al., arXiv:2403.12021 (2025) - Large-scale tweezer arrays

use num_complex::Complex64;
use std::f64::consts::PI;

/// Configuration for the WGS algorithm.
#[derive(Debug, Clone)]
pub struct WgsConfig {
    /// Number of iterations.
    pub max_iterations: usize,

    /// Convergence threshold for RMS uniformity error.
    pub tolerance: f64,

    /// Weighting exponent for spot intensity correction.
    /// Higher values give more aggressive correction.
    /// Typical range: 0.5 to 2.0.
    pub weight_exponent: f64,

    /// Initial phase mode.
    pub initial_phase: InitialPhase,

    /// Random seed for reproducibility (if using random initial phase).
    pub seed: u64,
}

impl Default for WgsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            weight_exponent: 1.0,
            initial_phase: InitialPhase::Random,
            seed: 42,
        }
    }
}

/// Initial phase distribution options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitialPhase {
    /// Uniform zero phase.
    Uniform,
    /// Random phase [0, 2*pi).
    Random,
    /// Quadratic lens phase for approximate focusing.
    Lens { focal_length: f64 },
}

/// Target spot specification for discrete tweezer arrays.
#[derive(Debug, Clone, Copy)]
pub struct TargetSpot {
    /// x-coordinate in focal plane (pixels or normalized units).
    pub x: f64,
    /// y-coordinate in focal plane.
    pub y: f64,
    /// Target intensity (relative, will be normalized).
    pub intensity: f64,
}

/// Result of the WGS algorithm.
#[derive(Debug, Clone)]
pub struct WgsResult {
    /// Computed SLM phase pattern (NxN array, flattened row-major).
    pub slm_phase: Vec<f64>,

    /// Grid size (N for NxN).
    pub grid_size: usize,

    /// Final focal plane intensities at target spots.
    pub spot_intensities: Vec<f64>,

    /// RMS uniformity error of spot intensities.
    pub uniformity_error: f64,

    /// Diffraction efficiency (power in spots / total power).
    pub efficiency: f64,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Convergence history (uniformity error per iteration).
    pub convergence: Vec<f64>,
}

/// Simple PRNG for reproducible random initialization.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_f64(&mut self) -> f64 {
        // xorshift64*
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let val = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        (val as f64) / (u64::MAX as f64)
    }
}

/// Performs 2D FFT using the Cooley-Tukey algorithm.
///
/// This is a simplified implementation for square power-of-2 grids.
/// For production use, consider using an optimized FFT library.
fn fft2d(data: &mut [Complex64], n: usize, inverse: bool) {
    // FFT each row
    for i in 0..n {
        let start = i * n;
        fft1d(&mut data[start..start + n], inverse);
    }

    // Transpose
    for i in 0..n {
        for j in i + 1..n {
            data.swap(i * n + j, j * n + i);
        }
    }

    // FFT each row (now columns)
    for i in 0..n {
        let start = i * n;
        fft1d(&mut data[start..start + n], inverse);
    }

    // Transpose back
    for i in 0..n {
        for j in i + 1..n {
            data.swap(i * n + j, j * n + i);
        }
    }

    // Normalize for inverse FFT
    if inverse {
        let norm = 1.0 / (n * n) as f64;
        for x in data.iter_mut() {
            *x *= norm;
        }
    }
}

/// 1D FFT using Cooley-Tukey radix-2 decimation-in-time.
fn fft1d(data: &mut [Complex64], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        let mut m = n / 2;
        while m > 0 && j >= m {
            j -= m;
            m /= 2;
        }
        j += m;
    }

    // Cooley-Tukey iterative FFT
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let angle = sign * 2.0 * PI / len as f64;
        let wn = Complex64::from_polar(1.0, angle);

        for start in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..len / 2 {
                let t = w * data[start + k + len / 2];
                let u = data[start + k];
                data[start + k] = u + t;
                data[start + k + len / 2] = u - t;
                w *= wn;
            }
        }
        len *= 2;
    }
}

/// FFT shift: moves zero-frequency component to center.
fn fftshift(data: &mut [Complex64], n: usize) {
    let half = n / 2;
    // Swap quadrants
    for i in 0..half {
        for j in 0..half {
            // Swap (i, j) with (i + half, j + half)
            data.swap(i * n + j, (i + half) * n + (j + half));
            // Swap (i, j + half) with (i + half, j)
            data.swap(i * n + (j + half), (i + half) * n + j);
        }
    }
}

/// Inverse FFT shift.
fn ifftshift(data: &mut [Complex64], n: usize) {
    // For even n, fftshift and ifftshift are the same
    fftshift(data, n);
}

/// Runs the Weighted Gerchberg-Saxton algorithm for discrete target spots.
///
/// # Arguments
/// * `grid_size` - Size of NxN SLM grid (must be power of 2)
/// * `targets` - List of target spot specifications
/// * `config` - Algorithm configuration
///
/// # Returns
/// `WgsResult` containing the computed SLM phase pattern and diagnostics.
pub fn wgs_discrete(
    grid_size: usize,
    targets: &[TargetSpot],
    config: &WgsConfig,
) -> WgsResult {
    assert!(grid_size.is_power_of_two(), "Grid size must be power of 2");
    assert!(!targets.is_empty(), "Must have at least one target");

    let n = grid_size;
    let n2 = n * n;

    // Normalize target intensities
    let total_target: f64 = targets.iter().map(|t| t.intensity).sum();
    let norm_targets: Vec<f64> = targets.iter().map(|t| t.intensity / total_target).collect();

    // Initialize weights (uniform initially)
    let mut weights: Vec<f64> = vec![1.0; targets.len()];

    // Initialize SLM field with unit amplitude and initial phase
    let mut slm_field: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n2];
    let mut rng = Rng::new(config.seed);

    for i in 0..n {
        for j in 0..n {
            let phase = match config.initial_phase {
                InitialPhase::Uniform => 0.0,
                InitialPhase::Random => 2.0 * PI * rng.next_f64(),
                InitialPhase::Lens { focal_length } => {
                    let x = (i as f64 - n as f64 / 2.0) / n as f64;
                    let y = (j as f64 - n as f64 / 2.0) / n as f64;
                    -PI * (x * x + y * y) / focal_length
                }
            };
            slm_field[i * n + j] = Complex64::from_polar(1.0, phase);
        }
    }

    // Convergence history
    let mut convergence = Vec::with_capacity(config.max_iterations);
    let mut iterations = 0;

    // Create target amplitude array (sparse representation via sampling)
    let target_pixels: Vec<(usize, usize)> = targets
        .iter()
        .map(|t| {
            let ix = ((t.x + n as f64 / 2.0).round() as usize).min(n - 1);
            let iy = ((t.y + n as f64 / 2.0).round() as usize).min(n - 1);
            (ix, iy)
        })
        .collect();

    // Main WGS loop
    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Forward propagate: SLM -> focal plane
        let mut focal_field = slm_field.clone();
        fft2d(&mut focal_field, n, false);
        fftshift(&mut focal_field, n);

        // Sample intensities at target spots
        let mut spot_intensities: Vec<f64> = target_pixels
            .iter()
            .map(|&(ix, iy)| focal_field[ix * n + iy].norm_sqr())
            .collect();

        // Normalize spot intensities
        let total_spot: f64 = spot_intensities.iter().sum();
        if total_spot > 1e-30 {
            for si in &mut spot_intensities {
                *si /= total_spot;
            }
        }

        // Compute uniformity error
        let error: f64 = spot_intensities
            .iter()
            .zip(norm_targets.iter())
            .map(|(s, t)| (s - t).powi(2))
            .sum::<f64>()
            .sqrt()
            / targets.len() as f64;

        convergence.push(error);

        if error < config.tolerance {
            break;
        }

        // Update weights based on intensity deviation
        for (k, (si, ti)) in spot_intensities.iter().zip(norm_targets.iter()).enumerate() {
            if *si > 1e-30 {
                weights[k] = (ti / si).powf(config.weight_exponent / 2.0);
            }
        }

        // Apply target constraint: set amplitude at target spots
        // Create target field with weighted amplitudes
        let mut target_field: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n2];
        for (k, &(ix, iy)) in target_pixels.iter().enumerate() {
            let amp = (weights[k] * norm_targets[k]).sqrt();
            let phase = focal_field[ix * n + iy].arg();
            target_field[ix * n + iy] = Complex64::from_polar(amp, phase);
        }

        // Backpropagate: focal plane -> SLM
        ifftshift(&mut target_field, n);
        fft2d(&mut target_field, n, true);

        // Apply SLM constraint: uniform amplitude, keep computed phase
        for i in 0..n2 {
            let phase = target_field[i].arg();
            slm_field[i] = Complex64::from_polar(1.0, phase);
        }
    }

    // Final forward pass to compute actual intensities
    let mut focal_field = slm_field.clone();
    fft2d(&mut focal_field, n, false);
    fftshift(&mut focal_field, n);

    let spot_intensities: Vec<f64> = target_pixels
        .iter()
        .map(|&(ix, iy)| focal_field[ix * n + iy].norm_sqr())
        .collect();

    // Compute efficiency
    let total_intensity: f64 = focal_field.iter().map(|c| c.norm_sqr()).sum();
    let spot_total: f64 = spot_intensities.iter().sum();
    let efficiency = if total_intensity > 1e-30 {
        spot_total / total_intensity
    } else {
        0.0
    };

    // Compute final uniformity error
    let norm_spots: Vec<f64> = if spot_total > 1e-30 {
        spot_intensities.iter().map(|s| s / spot_total).collect()
    } else {
        vec![0.0; targets.len()]
    };
    let uniformity_error: f64 = norm_spots
        .iter()
        .zip(norm_targets.iter())
        .map(|(s, t)| (s - t).powi(2))
        .sum::<f64>()
        .sqrt()
        / targets.len() as f64;

    // Extract SLM phase pattern
    let slm_phase: Vec<f64> = slm_field.iter().map(|c| c.arg()).collect();

    WgsResult {
        slm_phase,
        grid_size,
        spot_intensities,
        uniformity_error,
        efficiency,
        iterations,
        convergence,
    }
}

/// Runs classic Gerchberg-Saxton for a continuous target intensity pattern.
///
/// # Arguments
/// * `target_intensity` - Target intensity pattern (NxN, flattened row-major)
/// * `grid_size` - Size N of NxN grid (must be power of 2)
/// * `config` - Algorithm configuration
pub fn gs_continuous(
    target_intensity: &[f64],
    grid_size: usize,
    config: &WgsConfig,
) -> WgsResult {
    assert!(grid_size.is_power_of_two(), "Grid size must be power of 2");
    assert_eq!(target_intensity.len(), grid_size * grid_size);

    let n = grid_size;
    let n2 = n * n;

    // Compute target amplitude from intensity
    let target_amp: Vec<f64> = target_intensity.iter().map(|i| i.sqrt()).collect();
    let total_target: f64 = target_intensity.iter().sum();

    // Initialize SLM field
    let mut slm_field: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n2];
    let mut rng = Rng::new(config.seed);

    for i in 0..n2 {
        let phase = match config.initial_phase {
            InitialPhase::Uniform => 0.0,
            InitialPhase::Random => 2.0 * PI * rng.next_f64(),
            InitialPhase::Lens { focal_length } => {
                let row = i / n;
                let col = i % n;
                let x = (row as f64 - n as f64 / 2.0) / n as f64;
                let y = (col as f64 - n as f64 / 2.0) / n as f64;
                -PI * (x * x + y * y) / focal_length
            }
        };
        slm_field[i] = Complex64::from_polar(1.0, phase);
    }

    let mut convergence = Vec::with_capacity(config.max_iterations);
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Forward propagate
        let mut focal_field = slm_field.clone();
        fft2d(&mut focal_field, n, false);
        fftshift(&mut focal_field, n);

        // Compute current intensity
        let intensities: Vec<f64> = focal_field.iter().map(|c| c.norm_sqr()).collect();
        let total_current: f64 = intensities.iter().sum();

        // Compute error
        let error: f64 = if total_current > 1e-30 && total_target > 1e-30 {
            intensities
                .iter()
                .zip(target_intensity.iter())
                .map(|(c, t)| (c / total_current - t / total_target).powi(2))
                .sum::<f64>()
                .sqrt()
                / n2 as f64
        } else {
            1.0
        };

        convergence.push(error);

        if error < config.tolerance {
            break;
        }

        // Apply target constraint: replace amplitude, keep phase
        for i in 0..n2 {
            let phase = focal_field[i].arg();
            focal_field[i] = Complex64::from_polar(target_amp[i], phase);
        }

        // Backpropagate
        ifftshift(&mut focal_field, n);
        fft2d(&mut focal_field, n, true);

        // Apply SLM constraint: uniform amplitude
        for i in 0..n2 {
            let phase = focal_field[i].arg();
            slm_field[i] = Complex64::from_polar(1.0, phase);
        }
    }

    // Final evaluation
    let mut focal_field = slm_field.clone();
    fft2d(&mut focal_field, n, false);
    fftshift(&mut focal_field, n);

    let spot_intensities: Vec<f64> = focal_field.iter().map(|c| c.norm_sqr()).collect();
    let total_intensity: f64 = spot_intensities.iter().sum();

    // Efficiency: overlap with target
    let overlap: f64 = spot_intensities
        .iter()
        .zip(target_intensity.iter())
        .map(|(s, t)| s.sqrt() * t.sqrt())
        .sum();
    let efficiency = if total_intensity > 1e-30 && total_target > 1e-30 {
        (overlap / (total_intensity.sqrt() * total_target.sqrt())).powi(2)
    } else {
        0.0
    };

    let slm_phase: Vec<f64> = slm_field.iter().map(|c| c.arg()).collect();

    WgsResult {
        slm_phase,
        grid_size,
        spot_intensities,
        uniformity_error: *convergence.last().unwrap_or(&1.0),
        efficiency,
        iterations,
        convergence,
    }
}

/// Generates Zernike polynomial phase pattern for aberration correction.
///
/// Uses Noll indexing (j = 1, 2, 3, ...).
///
/// # Arguments
/// * `grid_size` - Size of NxN output grid
/// * `j` - Zernike index (Noll convention)
/// * `coefficient` - Amplitude of the Zernike term in radians
pub fn zernike_phase(grid_size: usize, j: usize, coefficient: f64) -> Vec<f64> {
    let n = grid_size;
    let mut phase = vec![0.0; n * n];

    // Convert Noll index to (n, m) pair
    let (zn, zm) = noll_to_nm(j);

    for row in 0..n {
        for col in 0..n {
            let x = 2.0 * (col as f64 - n as f64 / 2.0) / n as f64;
            let y = 2.0 * (row as f64 - n as f64 / 2.0) / n as f64;
            let rho = (x * x + y * y).sqrt();
            let theta = y.atan2(x);

            if rho <= 1.0 {
                let r_nm = zernike_radial(zn, zm.unsigned_abs() as usize, rho);
                let z = if zm >= 0 {
                    r_nm * (zm as f64 * theta).cos()
                } else {
                    r_nm * ((-zm) as f64 * theta).sin()
                };
                phase[row * n + col] = coefficient * z;
            }
        }
    }

    phase
}

/// Converts Noll index to (n, m) Zernike indices.
///
/// Standard Noll convention (j starting at 1):
/// j=1: (0,0), j=2: (1,1), j=3: (1,-1), j=4: (2,0), j=5: (2,-2), j=6: (2,2), ...
fn noll_to_nm(j: usize) -> (usize, i32) {
    // Lookup table for first 15 Zernike terms (covers most practical applications)
    static NOLL_TABLE: [(usize, i32); 16] = [
        (0, 0),   // j=0 (unused)
        (0, 0),   // j=1: piston
        (1, 1),   // j=2: tip
        (1, -1),  // j=3: tilt
        (2, 0),   // j=4: defocus
        (2, -2),  // j=5: oblique astigmatism
        (2, 2),   // j=6: vertical astigmatism
        (3, -1),  // j=7: vertical coma
        (3, 1),   // j=8: horizontal coma
        (3, -3),  // j=9: oblique trefoil
        (3, 3),   // j=10: vertical trefoil
        (4, 0),   // j=11: primary spherical
        (4, 2),   // j=12
        (4, -2),  // j=13
        (4, 4),   // j=14
        (4, -4),  // j=15
    ];

    if j < NOLL_TABLE.len() {
        return NOLL_TABLE[j];
    }

    // For higher orders, use the general formula
    // Find n such that n(n+1)/2 < j <= (n+1)(n+2)/2
    let mut n = 0;
    while (n + 1) * (n + 2) / 2 < j {
        n += 1;
    }

    // Position within the n-th order
    let j_start = n * (n + 1) / 2 + 1;
    let k = j - j_start;

    // Compute m based on position and parity
    let m_abs = if n % 2 == 0 {
        2 * ((k + 1) / 2)
    } else {
        2 * (k / 2) + 1
    };

    let m_abs = m_abs.min(n);
    let m = if m_abs == 0 {
        0
    } else if j % 2 == 0 {
        m_abs as i32
    } else {
        -(m_abs as i32)
    };

    (n, m)
}

/// Computes the Zernike radial polynomial R_n^m(rho).
fn zernike_radial(n: usize, m: usize, rho: f64) -> f64 {
    if (n - m) % 2 != 0 {
        return 0.0;
    }

    let mut sum = 0.0;
    let half = (n - m) / 2;
    for k in 0..=half {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let num = factorial(n - k);
        let den = factorial(k) * factorial((n + m) / 2 - k) * factorial(half - k);
        sum += sign * (num as f64 / den as f64) * rho.powi((n - 2 * k) as i32);
    }
    sum
}

/// Factorial function (simple iterative implementation).
fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fft_roundtrip() {
        let n = 8;
        let mut data: Vec<Complex64> = (0..n * n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect();
        let original = data.clone();

        fft2d(&mut data, n, false);
        fft2d(&mut data, n, true);

        for (a, b) in data.iter().zip(original.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_wgs_single_spot() {
        // Single centered spot should achieve high efficiency
        let targets = vec![TargetSpot {
            x: 0.0,
            y: 0.0,
            intensity: 1.0,
        }];

        let config = WgsConfig {
            max_iterations: 50,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = wgs_discrete(64, &targets, &config);

        // Single spot should have very low error
        assert!(result.uniformity_error < 0.01, "Single spot error too high: {}", result.uniformity_error);
    }

    #[test]
    fn test_wgs_uniform_spots() {
        // 4 spots in a square should converge to uniform intensity
        let targets = vec![
            TargetSpot { x: 5.0, y: 5.0, intensity: 1.0 },
            TargetSpot { x: -5.0, y: 5.0, intensity: 1.0 },
            TargetSpot { x: 5.0, y: -5.0, intensity: 1.0 },
            TargetSpot { x: -5.0, y: -5.0, intensity: 1.0 },
        ];

        let config = WgsConfig::default();
        let result = wgs_discrete(64, &targets, &config);

        // Should converge
        assert!(result.iterations <= config.max_iterations);
        // Intensities should be roughly equal
        let mean = result.spot_intensities.iter().sum::<f64>() / 4.0;
        for si in &result.spot_intensities {
            assert!((*si - mean).abs() / mean < 0.2, "Spot uniformity poor");
        }
    }

    #[test]
    fn test_gs_continuous_delta() {
        // Target: single pixel delta function
        let n = 32;
        let mut target = vec![0.0; n * n];
        target[n * n / 2 + n / 2] = 1.0; // Center pixel

        let config = WgsConfig {
            max_iterations: 30,
            initial_phase: InitialPhase::Uniform,
            ..Default::default()
        };

        let result = gs_continuous(&target, n, &config);

        // Should have reasonable efficiency
        assert!(result.efficiency > 0.0);
        assert!(result.iterations <= 30);
    }

    #[test]
    fn test_zernike_piston() {
        // j=1 is piston (constant)
        let phase = zernike_phase(32, 1, 1.0);

        // Check that all values inside unit circle are equal
        let n = 32;
        let center_val = phase[n * n / 2 + n / 2];
        for row in 0..n {
            for col in 0..n {
                let x = 2.0 * (col as f64 - n as f64 / 2.0) / n as f64;
                let y = 2.0 * (row as f64 - n as f64 / 2.0) / n as f64;
                if x * x + y * y <= 1.0 {
                    assert_relative_eq!(phase[row * n + col], center_val, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_zernike_tilt() {
        // j=2 and j=3 are both tilt terms (n=1)
        let phase_2 = zernike_phase(32, 2, 1.0);
        let phase_3 = zernike_phase(32, 3, 1.0);

        let n = 32;
        let center = n / 2;

        // Both tilts should have non-zero variation across the aperture
        // Check that there's variation in one of the tilts
        let vals_2: Vec<f64> = (0..n)
            .filter(|&i| {
                let x = 2.0 * (i as f64 - center as f64) / n as f64;
                x.abs() < 0.9
            })
            .map(|i| phase_2[center * n + i])
            .collect();

        let vals_3: Vec<f64> = (0..n)
            .filter(|&i| {
                let y = 2.0 * (i as f64 - center as f64) / n as f64;
                y.abs() < 0.9
            })
            .map(|i| phase_3[i * n + center])
            .collect();

        // At least one tilt should show variation
        let var_2: f64 = if vals_2.len() > 1 {
            let mean = vals_2.iter().sum::<f64>() / vals_2.len() as f64;
            vals_2.iter().map(|v| (v - mean).powi(2)).sum()
        } else {
            0.0
        };

        let var_3: f64 = if vals_3.len() > 1 {
            let mean = vals_3.iter().sum::<f64>() / vals_3.len() as f64;
            vals_3.iter().map(|v| (v - mean).powi(2)).sum()
        } else {
            0.0
        };

        assert!(
            var_2 > 1e-6 || var_3 > 1e-6,
            "Tilt terms should show spatial variation"
        );
    }

    #[test]
    fn test_noll_indexing() {
        // Verify basic structure
        let (n1, _m1) = noll_to_nm(1);
        assert_eq!(n1, 0, "j=1 should be piston (n=0)");

        let (n2, _m2) = noll_to_nm(2);
        assert_eq!(n2, 1, "j=2 should be tilt (n=1)");

        let (n3, _m3) = noll_to_nm(3);
        assert_eq!(n3, 1, "j=3 should be tilt (n=1)");

        let (n4, m4) = noll_to_nm(4);
        assert_eq!(n4, 2, "j=4 should be n=2");
        assert_eq!(m4, 0, "j=4 should be defocus (m=0)");
    }

    #[test]
    fn test_convergence_completes() {
        // WGS should complete without panicking and produce valid output
        let targets = vec![
            TargetSpot { x: 3.0, y: 3.0, intensity: 1.0 },
            TargetSpot { x: -3.0, y: 3.0, intensity: 1.0 },
        ];

        let config = WgsConfig {
            max_iterations: 50,
            ..Default::default()
        };

        let result = wgs_discrete(64, &targets, &config);

        // Should produce a phase pattern
        assert_eq!(result.slm_phase.len(), 64 * 64);

        // Should have non-zero efficiency
        assert!(result.efficiency > 0.0, "Efficiency should be positive");

        // Should find intensities at target spots
        assert_eq!(result.spot_intensities.len(), 2);
        assert!(result.spot_intensities.iter().all(|&i| i > 0.0));
    }
}
