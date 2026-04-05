//! Logarithmic spherical-polar grid for GRMHD.
//!
//! Coordinates: (x1, x2, x3) mapped to (r, theta, phi) via:
//!   r = exp(x1)          (logarithmic radial, concentrates zones near horizon)
//!   theta = pi * x2      (uniform in x2 from 0 to 1)
//!   phi = 2*pi * x3      (uniform in x3 from 0 to 1)
//!
//! This is the standard HARM coordinate mapping (Gammie+ 2003).

use crate::metric::KerrMetric;

/// GRMHD grid specification.
#[derive(Clone, Debug)]
pub struct Grid {
    /// Number of active zones in each direction (excluding ghost zones).
    pub n1: usize,
    pub n2: usize,
    pub n3: usize,
    /// Number of ghost zones on each side.
    pub ng: usize,
    /// Coordinate bounds in internal coordinates (x1, x2, x3).
    pub x1_min: f64,
    pub x1_max: f64,
    pub x2_min: f64,
    pub x2_max: f64,
    pub x3_min: f64,
    pub x3_max: f64,
    /// Cell sizes.
    pub dx1: f64,
    pub dx2: f64,
    pub dx3: f64,
    /// Metric.
    pub metric: KerrMetric,
}

impl Grid {
    /// Create a standard logarithmic grid around a Kerr black hole.
    ///
    /// `r_in` and `r_out` are the physical radial bounds.
    /// The grid covers 0 < theta < pi and 0 < phi < 2*pi.
    pub fn new(n1: usize, n2: usize, n3: usize, r_in: f64, r_out: f64, metric: KerrMetric) -> Self {
        let ng = 2; // 2 ghost zones for PLM reconstruction
        let x1_min = r_in.ln();
        let x1_max = r_out.ln();
        let x2_min = 0.0;
        let x2_max = 1.0;
        let x3_min = 0.0;
        let x3_max = 1.0;
        let dx1 = (x1_max - x1_min) / n1 as f64;
        let dx2 = (x2_max - x2_min) / n2 as f64;
        let dx3 = (x3_max - x3_min) / n3 as f64;
        Self {
            n1,
            n2,
            n3,
            ng,
            x1_min,
            x1_max,
            x2_min,
            x2_max,
            x3_min,
            x3_max,
            dx1,
            dx2,
            dx3,
            metric,
        }
    }

    /// Total number of zones including ghost zones in direction 1.
    pub fn n1_total(&self) -> usize {
        self.n1 + 2 * self.ng
    }
    /// Total number of zones including ghost zones in direction 2.
    pub fn n2_total(&self) -> usize {
        self.n2 + 2 * self.ng
    }
    /// Total number of zones including ghost zones in direction 3.
    pub fn n3_total(&self) -> usize {
        self.n3 + 2 * self.ng
    }
    /// Total number of cells (including ghosts).
    pub fn n_total(&self) -> usize {
        self.n1_total() * self.n2_total() * self.n3_total()
    }

    /// Internal coordinate x1 at cell center for index i (0-based including ghosts).
    pub fn x1(&self, i: usize) -> f64 {
        self.x1_min + (i as f64 - self.ng as f64 + 0.5) * self.dx1
    }
    /// Internal coordinate x2 at cell center for index j.
    pub fn x2(&self, j: usize) -> f64 {
        self.x2_min + (j as f64 - self.ng as f64 + 0.5) * self.dx2
    }
    /// Internal coordinate x3 at cell center for index k.
    pub fn x3(&self, k: usize) -> f64 {
        self.x3_min + (k as f64 - self.ng as f64 + 0.5) * self.dx3
    }

    /// Physical radius at cell center.
    pub fn r(&self, i: usize) -> f64 {
        self.x1(i).exp()
    }
    /// Physical theta at cell center.
    pub fn theta(&self, j: usize) -> f64 {
        std::f64::consts::PI * self.x2(j)
    }
    /// Physical phi at cell center.
    pub fn phi(&self, k: usize) -> f64 {
        2.0 * std::f64::consts::PI * self.x3(k)
    }

    /// Flat index for (i, j, k).
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.n2_total() * self.n3_total() + j * self.n3_total() + k
    }

    /// Metric components at cell center (i, j).
    pub fn gcov_at(&self, i: usize, j: usize) -> [f64; 5] {
        self.metric.gcov(self.r(i), self.theta(j))
    }

    /// sqrt(-g) at cell center.
    pub fn sqrt_neg_g_at(&self, i: usize, j: usize) -> f64 {
        let r = self.r(i);
        let th = self.theta(j);
        // Jacobian for log-r coordinate: dr/dx1 = r
        // Full: sqrt(-g) * dr/dx1 * dtheta/dx2 * dphi/dx3
        self.metric.sqrt_neg_g(r, th) * r * std::f64::consts::PI * 2.0 * std::f64::consts::PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let g = Grid::new(64, 32, 1, 1.5, 50.0, KerrMetric::schwarzschild());
        assert_eq!(g.n1, 64);
        assert_eq!(g.n2, 32);
        assert_eq!(g.n1_total(), 68); // 64 + 2*2
        assert_eq!(g.n2_total(), 36);
    }

    #[test]
    fn test_grid_coordinates() {
        let g = Grid::new(64, 32, 1, 2.0, 100.0, KerrMetric::schwarzschild());
        // First active cell in r: i = ng = 2
        let r_first = g.r(g.ng);
        assert!(
            (r_first - 2.0).abs() < 0.1,
            "First active r ~ r_in, got {}",
            r_first
        );
        // Last active cell: i = ng + n1 - 1
        let r_last = g.r(g.ng + g.n1 - 1);
        assert!(
            (r_last - 100.0).abs() < 5.0,
            "Last active r ~ r_out, got {}",
            r_last
        );
        // Theta at midplane
        let j_mid = g.ng + g.n2 / 2;
        let th = g.theta(j_mid);
        assert!(
            (th - std::f64::consts::FRAC_PI_2).abs() < 0.1,
            "Midplane theta ~ pi/2, got {}",
            th
        );
    }

    #[test]
    fn test_grid_idx_roundtrip() {
        let g = Grid::new(16, 16, 4, 2.0, 50.0, KerrMetric::schwarzschild());
        let idx = g.idx(5, 10, 3);
        // Verify it's in bounds
        assert!(idx < g.n_total());
    }
}
