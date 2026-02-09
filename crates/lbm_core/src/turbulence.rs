//! Turbulence Statistics and Spectral Analysis.
//!
//! Provides tools for analyzing 2D flow fields in spectral space:
//! - 2D FFT via row-column decomposition (rustfft)
//! - Radially-binned isotropic power spectrum P(k)
//! - Triad extraction: wavevector triplets (k, p, q) with k + p + q = 0
//! - Energy transfer T(k|p,q) between spectral modes
//! - Clustering coefficient of the triad interaction graph
//!
//! # References
//! - Kraichnan, "Inertial ranges in 2D turbulence", Phys. Fluids 10 (1967)
//! - Boffetta & Ecke, "Two-dimensional turbulence", Ann. Rev. Fluid Mech. 44 (2012)

use ndarray::{Array2, Axis};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::{HashMap, HashSet};

/// Triad interaction in spectral space (k, p, q) with k + p + q = 0.
#[derive(Debug, Clone)]
pub struct SpectralTriad {
    /// Wavevector indices of mode k (centered: can be negative)
    pub k: [i32; 2],
    /// Wavevector indices of mode p
    pub p: [i32; 2],
    /// Wavevector indices of mode q = -(k + p)
    pub q: [i32; 2],
    /// Energy transfer T(k|p,q) for this triad
    pub energy_transfer: f64,
}

/// Result of 2D FFT: complex amplitudes on the (kx, ky) grid.
#[derive(Debug, Clone)]
pub struct SpectralField {
    /// Complex Fourier coefficients, row-major (nx, ny)
    pub coeffs: Array2<Complex<f64>>,
    pub nx: usize,
    pub ny: usize,
}

impl SpectralField {
    /// Access coefficient at wavevector indices (kx, ky).
    ///
    /// Indices wrap modulo (nx, ny) following standard FFT convention.
    pub fn get(&self, kx: i32, ky: i32) -> Complex<f64> {
        let ix = kx.rem_euclid(self.nx as i32) as usize;
        let iy = ky.rem_euclid(self.ny as i32) as usize;
        self.coeffs[[ix, iy]]
    }
}

/// Compute 2D FFT of a real scalar field.
///
/// Uses row-column decomposition: 1D FFT on each row (axis 1),
/// then 1D FFT on each column (axis 0) of the result.
pub fn fft2d(field: &Array2<f64>) -> SpectralField {
    let (nx, ny) = field.dim();
    let mut planner = FftPlanner::new();

    let mut transformed = Array2::<Complex<f64>>::zeros((nx, ny));

    // Row-wise FFT (along axis 1)
    let fft_row = planner.plan_fft_forward(ny);
    for (i, row) in field.axis_iter(Axis(0)).enumerate() {
        let mut buffer: Vec<Complex<f64>> = row.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft_row.process(&mut buffer);
        for (j, val) in buffer.iter().enumerate() {
            transformed[[i, j]] = *val;
        }
    }

    // Column-wise FFT (along axis 0)
    let fft_col = planner.plan_fft_forward(nx);
    for j in 0..ny {
        let mut col_buf: Vec<Complex<f64>> = (0..nx).map(|i| transformed[[i, j]]).collect();
        fft_col.process(&mut col_buf);
        for (i, val) in col_buf.iter().enumerate() {
            transformed[[i, j]] = *val;
        }
    }

    SpectralField {
        coeffs: transformed,
        nx,
        ny,
    }
}

/// Compute radially-binned isotropic power spectrum P(k).
///
/// Returns (k_bins, power) where k_bins[i] is the wavenumber magnitude
/// and power[i] is the average |F(kx,ky)|^2 / N^2 in that radial shell.
/// The DC component (k=0) is excluded.
pub fn power_spectrum(field: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let (nx, ny) = field.dim();
    let spec = fft2d(field);
    let n_bins = nx.max(ny) / 2;
    let norm_sq = (nx * ny) as f64 * (nx * ny) as f64;

    let mut bin_sum = vec![0.0; n_bins + 1];
    let mut bin_count = vec![0usize; n_bins + 1];

    for ikx in 0..nx {
        for iky in 0..ny {
            // Map FFT indices to centered wavevectors
            let kx = if ikx <= nx / 2 {
                ikx as f64
            } else {
                ikx as f64 - nx as f64
            };
            let ky = if iky <= ny / 2 {
                iky as f64
            } else {
                iky as f64 - ny as f64
            };

            let k_mag = (kx * kx + ky * ky).sqrt();
            let bin = k_mag.round() as usize;

            if bin > 0 && bin <= n_bins {
                bin_sum[bin] += spec.coeffs[[ikx, iky]].norm_sqr() / norm_sq;
                bin_count[bin] += 1;
            }
        }
    }

    let mut k_bins = Vec::new();
    let mut power = Vec::new();
    for bin in 1..=n_bins {
        if bin_count[bin] > 0 {
            k_bins.push(bin as f64);
            power.push(bin_sum[bin] / bin_count[bin] as f64);
        }
    }

    (k_bins, power)
}

/// Extract dominant triads from 2D velocity field components.
///
/// A triad (k, p, q) satisfies wavevector closure k + p + q = 0.
/// We select triads where the product |u(k)|*|u(p)|*|u(q)| exceeds
/// the given threshold, indicating significant nonlinear interaction.
///
/// The search is restricted to wavenumbers |kx| <= nx/4, |ky| <= ny/4
/// for efficiency (the most energetic modes in turbulence).
pub fn extract_dominant_triads(
    u: &Array2<f64>,
    v: &Array2<f64>,
    threshold: f64,
) -> Vec<SpectralTriad> {
    let (nx, ny) = u.dim();
    let u_hat = fft2d(u);
    let v_hat = fft2d(v);
    let norm = (nx * ny) as f64;

    // Combined velocity amplitude at each wavevector
    let amplitude = |kx: i32, ky: i32| -> f64 {
        let au = u_hat.get(kx, ky).norm();
        let av = v_hat.get(kx, ky).norm();
        (au * au + av * av).sqrt() / norm
    };

    let kx_max = (nx / 4) as i32;
    let ky_max = (ny / 4) as i32;

    let mut triads = Vec::new();

    for kx1 in -kx_max..=kx_max {
        for ky1 in -ky_max..=ky_max {
            if kx1 == 0 && ky1 == 0 {
                continue;
            }
            let a_k = amplitude(kx1, ky1);

            for kx2 in -kx_max..=kx_max {
                for ky2 in -ky_max..=ky_max {
                    if kx2 == 0 && ky2 == 0 {
                        continue;
                    }
                    if kx1 == kx2 && ky1 == ky2 {
                        continue;
                    }

                    let qx = -(kx1 + kx2);
                    let qy = -(ky1 + ky2);

                    if qx.abs() > kx_max || qy.abs() > ky_max {
                        continue;
                    }
                    if qx == 0 && qy == 0 {
                        continue;
                    }

                    // Canonical ordering to emit each unordered triple once
                    let mut triple = [(kx1, ky1), (kx2, ky2), (qx, qy)];
                    triple.sort();
                    if (kx1, ky1) != triple[0] || (kx2, ky2) != triple[1] {
                        continue;
                    }

                    let a_p = amplitude(kx2, ky2);
                    let a_q = amplitude(qx, qy);
                    let product = a_k * a_p * a_q;

                    if product > threshold {
                        let t = calculate_triad_energy_transfer(
                            u_hat.get(kx1, ky1),
                            u_hat.get(kx2, ky2),
                            u_hat.get(qx, qy),
                            [kx2 as f64, ky2 as f64],
                        );

                        triads.push(SpectralTriad {
                            k: [kx1, ky1],
                            p: [kx2, ky2],
                            q: [qx, qy],
                            energy_transfer: t / (norm * norm * norm),
                        });
                    }
                }
            }
        }
    }

    triads
}

/// Calculate energy transfer for a specific triad.
///
/// T(k|p,q) = -Im[ conj(u_hat(k)) * (p . u_hat(p)) * u_hat(q) ]
///
/// This is a simplified scalar form of the nonlinear transfer term.
/// The vector p modulates the advection (p . u_hat(p)) representing
/// the gradient operator in Fourier space.
pub fn calculate_triad_energy_transfer(
    uk: Complex<f64>,
    up: Complex<f64>,
    uq: Complex<f64>,
    p_vec: [f64; 2],
) -> f64 {
    // (p . u_hat(p)): scalar product of wavevector p with velocity mode
    let p_dot_up = Complex::new(p_vec[0] + p_vec[1], 0.0) * up;
    // T = -Im[ conj(uk) * (p.up) * uq ]
    let triple = uk.conj() * p_dot_up * uq;
    -triple.im
}

/// Compute clustering coefficient of the triad interaction graph.
///
/// Projects triads onto a simple graph where nodes are wavevectors
/// and edges connect wavevectors in the same triad. Returns the
/// global clustering coefficient C = triangles / connected_triplets.
pub fn triad_clustering_coefficient(triads: &[SpectralTriad]) -> f64 {
    if triads.is_empty() {
        return 0.0;
    }

    let mut adj: HashMap<(i32, i32), HashSet<(i32, i32)>> = HashMap::new();
    let mut vertices: HashSet<(i32, i32)> = HashSet::new();

    for t in triads {
        let k = (t.k[0], t.k[1]);
        let p = (t.p[0], t.p[1]);
        let q = (t.q[0], t.q[1]);

        for &(a, b) in &[(k, p), (p, q), (q, k)] {
            adj.entry(a).or_default().insert(b);
            adj.entry(b).or_default().insert(a);
            vertices.insert(a);
            vertices.insert(b);
        }
    }

    let mut triangles = 0usize;
    let mut triplets = 0usize;

    for v in &vertices {
        if let Some(neighbors) = adj.get(v) {
            let deg = neighbors.len();
            if deg < 2 {
                continue;
            }
            triplets += deg * (deg - 1) / 2;

            let nbrs: Vec<_> = neighbors.iter().collect();
            for i in 0..deg {
                for j in (i + 1)..deg {
                    if adj.get(nbrs[i]).is_some_and(|n| n.contains(nbrs[j])) {
                        triangles += 1;
                    }
                }
            }
        }
    }

    if triplets == 0 {
        0.0
    } else {
        (triangles as f64) / (triplets as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_fft2d_constant_field() {
        // Constant field -> only DC component nonzero
        let field = Array2::from_elem((8, 8), 3.0);
        let spec = fft2d(&field);
        let dc = spec.get(0, 0);
        // DC = 3.0 * 64 = 192
        assert!((dc.re - 192.0).abs() < 1e-10, "DC = {}", dc.re);
        assert!(dc.im.abs() < 1e-10);
        // Non-DC modes should be zero
        assert!(spec.get(1, 0).norm() < 1e-10);
        assert!(spec.get(0, 1).norm() < 1e-10);
    }

    #[test]
    fn test_fft2d_sinusoid_x() {
        // sin(2*pi*x/N) should peak at kx=+/-1, ky=0
        let n = 16;
        let mut field = Array2::zeros((n, n));
        for x in 0..n {
            let val = (2.0 * std::f64::consts::PI * x as f64 / n as f64).sin();
            for y in 0..n {
                field[[x, y]] = val;
            }
        }
        let spec = fft2d(&field);

        let peak = spec.get(1, 0).norm();
        assert!(peak > 1.0, "Peak at (1,0) should be large: {}", peak);

        let noise = spec.get(2, 0).norm();
        assert!(noise < 1e-10, "Mode (2,0) should be ~0: {}", noise);
    }

    #[test]
    fn test_power_spectrum_nonempty() {
        let field = Array2::from_elem((16, 16), 1.0);
        let (k_bins, power) = power_spectrum(&field);
        // Constant field has no power at k>0
        // (all power is in DC which is excluded)
        assert!(!k_bins.is_empty());
        for &p in &power {
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn test_power_spectrum_single_mode() {
        let n = 32;
        let mut field = Array2::zeros((n, n));
        for x in 0..n {
            let val = (2.0 * std::f64::consts::PI * x as f64 / n as f64).sin();
            for y in 0..n {
                field[[x, y]] = val;
            }
        }
        let (k_bins, power) = power_spectrum(&field);

        // k=1 bin should have the most power
        if let Some(k1_idx) = k_bins.iter().position(|&k| (k - 1.0).abs() < 0.5) {
            let p1 = power[k1_idx];
            for (i, &p) in power.iter().enumerate() {
                if i != k1_idx && p1 > 0.0 {
                    assert!(
                        p < p1 * 0.01,
                        "Power at k={} ({:.6}) should be << power at k=1 ({:.6})",
                        k_bins[i],
                        p,
                        p1
                    );
                }
            }
        }
    }

    #[test]
    fn test_triad_closure() {
        // For any extracted triad, k + p + q = 0
        let n = 16;
        let mut u = Array2::zeros((n, n));
        let mut v = Array2::zeros((n, n));
        for x in 0..n {
            for y in 0..n {
                let fx = x as f64 / n as f64;
                let fy = y as f64 / n as f64;
                u[[x, y]] = (2.0 * std::f64::consts::PI * fx).sin()
                    + 0.5 * (4.0 * std::f64::consts::PI * fy).sin();
                v[[x, y]] = (2.0 * std::f64::consts::PI * fy).cos();
            }
        }

        let triads = extract_dominant_triads(&u, &v, 0.0);
        for t in &triads {
            assert_eq!(t.k[0] + t.p[0] + t.q[0], 0, "kx closure violated");
            assert_eq!(t.k[1] + t.p[1] + t.q[1], 0, "ky closure violated");
        }
    }

    #[test]
    fn test_energy_transfer_finite() {
        let uk = Complex::new(1.0, 0.5);
        let up = Complex::new(0.3, -0.2);
        let uq = Complex::new(-0.4, 0.1);
        let p = [1.0, 2.0];
        let t = calculate_triad_energy_transfer(uk, up, uq, p);
        assert!(t.is_finite());
    }

    #[test]
    fn test_clustering_empty() {
        assert_eq!(triad_clustering_coefficient(&[]), 0.0);
    }

    #[test]
    fn test_clustering_single_triad() {
        let triads = vec![SpectralTriad {
            k: [1, 0],
            p: [0, 1],
            q: [-1, -1],
            energy_transfer: 0.1,
        }];
        // Single triad forms a triangle in the projected graph -> C = 1.0
        let c = triad_clustering_coefficient(&triads);
        assert!((c - 1.0).abs() < 1e-14, "Expected 1.0, got {}", c);
    }

    #[test]
    fn test_spectral_field_wraparound() {
        let field = Array2::from_elem((8, 8), 0.0);
        let spec = fft2d(&field);
        // Negative indices should wrap: get(-1, 0) == get(7, 0) for nx=8
        assert_eq!(spec.get(-1, 0), spec.get(7, 0));
        assert_eq!(spec.get(0, -1), spec.get(0, 7));
    }
}
