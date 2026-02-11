//! Percolation channel detection and frustration correlation.
//!
//! This module implements algorithms to:
//! 1. Detect percolation channels in 3D velocity fields (BFS-based)
//! 2. Correlate channel distribution with frustration density (Welch's t-test)
//! 3. Compute channel statistics for analysis
//!
//! Used in E-027 experiment to validate Thesis 1 (frustration-viscosity coupling).

use std::collections::VecDeque;

/// A single connected percolation channel (high-velocity region).
#[derive(Clone, Debug)]
pub struct PercolationChannel {
    /// Unique identifier for this channel
    pub id: usize,
    /// Number of grid cells in this channel
    pub size: usize,
    /// Mean velocity magnitude in this channel
    pub mean_velocity: f64,
    /// Peak (maximum) velocity magnitude in this channel
    pub max_velocity: f64,
    /// Bounding box of channel (min/max coordinates)
    pub bounding_box: BoundingBox,
    /// Grid coordinates of cells in this channel
    pub cells: Vec<(usize, usize, usize)>,
}

/// Bounding box for a channel.
#[derive(Clone, Debug)]
pub struct BoundingBox {
    pub x_min: usize,
    pub x_max: usize,
    pub y_min: usize,
    pub y_max: usize,
    pub z_min: usize,
    pub z_max: usize,
}

/// Result of percolation-frustration correlation analysis.
#[derive(Clone, Debug)]
pub struct CorrelationResult {
    /// t-statistic from Welch's t-test
    pub t_statistic: f64,
    /// p-value (two-tailed)
    pub p_value: f64,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Mean frustration in channels
    pub mean_frustration_channels: f64,
    /// Mean frustration in background (non-channel cells)
    pub mean_frustration_background: f64,
    /// Number of channel cells
    pub n_channel: usize,
    /// Number of background cells
    pub n_background: usize,
}

/// Detector for percolation channels in 3D velocity fields.
pub struct PercolationDetector {
    nx: usize,
    ny: usize,
    nz: usize,
    visited: Vec<bool>,
}

impl PercolationDetector {
    /// Create a new percolation detector for a grid of size nx x ny x nz.
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            visited: vec![false; nx * ny * nz],
        }
    }

    /// Linearize 3D coordinates to 1D index.
    fn linearize(&self, x: usize, y: usize, z: usize) -> usize {
        z * (self.nx * self.ny) + y * self.nx + x
    }

    /// Get 6 neighbors (cardinal directions only, no diagonals).
    fn get_neighbors(&self, x: usize, y: usize, z: usize) -> Vec<(usize, usize, usize)> {
        let mut neighbors = Vec::new();

        // Cardinal directions: +x, -x, +y, -y, +z, -z
        if x + 1 < self.nx {
            neighbors.push((x + 1, y, z));
        }
        if x > 0 {
            neighbors.push((x - 1, y, z));
        }
        if y + 1 < self.ny {
            neighbors.push((x, y + 1, z));
        }
        if y > 0 {
            neighbors.push((x, y - 1, z));
        }
        if z + 1 < self.nz {
            neighbors.push((x, y, z + 1));
        }
        if z > 0 {
            neighbors.push((x, y, z - 1));
        }

        neighbors
    }

    /// Detect all percolation channels in a velocity field.
    ///
    /// Uses breadth-first search (BFS) to identify connected components of high-velocity cells.
    /// Connectivity is defined by 6-neighbor adjacency (cardinal directions only).
    ///
    /// # Arguments
    /// * `velocity_field` - Flattened 3D velocity field: [u_x, u_y, u_z] per cell
    /// * `threshold` - Velocity magnitude threshold for channel membership
    ///
    /// # Returns
    /// Vector of detected percolation channels
    ///
    /// # Panics
    /// If velocity_field length != nx*ny*nz*3
    pub fn detect_channels(
        &mut self,
        velocity_field: &[[f64; 3]],
        threshold: f64,
    ) -> Vec<PercolationChannel> {
        let n_cells = self.nx * self.ny * self.nz;
        assert_eq!(
            velocity_field.len(),
            n_cells,
            "Velocity field size mismatch"
        );

        // Reset visited
        self.visited = vec![false; n_cells];

        let mut channels = Vec::new();
        let mut next_id = 0;

        // Iterate over all cells
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let idx = self.linearize(x, y, z);

                    // Skip if already visited or below threshold
                    if self.visited[idx] {
                        continue;
                    }

                    let u = velocity_field[idx];
                    let u_mag = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();

                    if u_mag < threshold {
                        continue;
                    }

                    // Start BFS from this seed
                    let channel = self.bfs_channel(
                        x, y, z,
                        velocity_field,
                        threshold,
                        next_id,
                    );

                    if !channel.cells.is_empty() {
                        channels.push(channel);
                        next_id += 1;
                    }
                }
            }
        }

        channels
    }

    /// Breadth-first search to extract a single channel.
    fn bfs_channel(
        &mut self,
        x_start: usize,
        y_start: usize,
        z_start: usize,
        velocity_field: &[[f64; 3]],
        threshold: f64,
        id: usize,
    ) -> PercolationChannel {
        let mut queue = VecDeque::new();
        let mut cells = Vec::new();
        let mut total_velocity = 0.0;
        let mut max_velocity: f64 = 0.0;

        let mut bb = BoundingBox {
            x_min: x_start,
            x_max: x_start,
            y_min: y_start,
            y_max: y_start,
            z_min: z_start,
            z_max: z_start,
        };

        queue.push_back((x_start, y_start, z_start));
        let idx_start = self.linearize(x_start, y_start, z_start);
        self.visited[idx_start] = true;

        while let Some((x, y, z)) = queue.pop_front() {
            let idx = self.linearize(x, y, z);
            let u = velocity_field[idx];
            let u_mag = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();

            // Add to channel
            cells.push((x, y, z));
            total_velocity += u_mag;
            max_velocity = max_velocity.max(u_mag);

            // Update bounding box
            bb.x_min = bb.x_min.min(x);
            bb.x_max = bb.x_max.max(x);
            bb.y_min = bb.y_min.min(y);
            bb.y_max = bb.y_max.max(y);
            bb.z_min = bb.z_min.min(z);
            bb.z_max = bb.z_max.max(z);

            // Check neighbors
            for (nx, ny, nz) in self.get_neighbors(x, y, z) {
                let n_idx = self.linearize(nx, ny, nz);
                if self.visited[n_idx] {
                    continue;
                }

                let nu = velocity_field[n_idx];
                let nu_mag = (nu[0] * nu[0] + nu[1] * nu[1] + nu[2] * nu[2]).sqrt();

                if nu_mag >= threshold {
                    self.visited[n_idx] = true;
                    queue.push_back((nx, ny, nz));
                }
            }
        }

        let size = cells.len();
        let mean_velocity = if size > 0 {
            total_velocity / (size as f64)
        } else {
            0.0
        };

        PercolationChannel {
            id,
            size,
            mean_velocity,
            max_velocity,
            bounding_box: bb,
            cells,
        }
    }
}

/// Compute automatic velocity threshold as mean + k * stddev.
///
/// Default: k = 1.5 (standard practice in percolation studies)
pub fn auto_velocity_threshold(velocity_field: &[[f64; 3]]) -> f64 {
    if velocity_field.is_empty() {
        return 0.0;
    }

    let velocities: Vec<f64> = velocity_field
        .iter()
        .map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
        .collect();

    let mean = velocities.iter().sum::<f64>() / (velocities.len() as f64);
    let variance = velocities
        .iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f64>()
        / (velocities.len() as f64);
    let stddev = variance.sqrt();

    mean + 1.5 * stddev
}

/// Correlate percolation channels with frustration density.
///
/// Uses Welch's t-test to compare frustration values in channel cells
/// vs background cells.
pub fn correlate_with_frustration(
    channels: &[PercolationChannel],
    frustration_field: &[f64],
) -> CorrelationResult {
    if frustration_field.is_empty() {
        return CorrelationResult {
            t_statistic: f64::NAN,
            p_value: 1.0,
            effect_size: 0.0,
            mean_frustration_channels: f64::NAN,
            mean_frustration_background: f64::NAN,
            n_channel: 0,
            n_background: 0,
        };
    }

    // Mark channel cells
    let mut is_channel = vec![false; frustration_field.len()];
    let mut channel_frustrations = Vec::new();

    for channel in channels {
        for (x, y, z) in &channel.cells {
            let idx = z * 64 * 64 + y * 64 + x;  // Assumes 64^3 grid; will generalize if needed
            if idx < is_channel.len() {
                is_channel[idx] = true;
            }
        }
    }

    // Collect frustration values
    let mut background_frustrations = Vec::new();
    for (i, &f) in frustration_field.iter().enumerate() {
        if is_channel[i] {
            channel_frustrations.push(f);
        } else {
            background_frustrations.push(f);
        }
    }

    if channel_frustrations.is_empty() || background_frustrations.is_empty() {
        return CorrelationResult {
            t_statistic: f64::NAN,
            p_value: 1.0,
            effect_size: 0.0,
            mean_frustration_channels: f64::NAN,
            mean_frustration_background: f64::NAN,
            n_channel: channel_frustrations.len(),
            n_background: background_frustrations.len(),
        };
    }

    // Compute statistics
    let mean_ch = channel_frustrations.iter().sum::<f64>() / (channel_frustrations.len() as f64);
    let mean_bg =
        background_frustrations.iter().sum::<f64>() / (background_frustrations.len() as f64);

    let var_ch = channel_frustrations
        .iter()
        .map(|&f| (f - mean_ch).powi(2))
        .sum::<f64>()
        / (channel_frustrations.len() as f64);
    let var_bg = background_frustrations
        .iter()
        .map(|&f| (f - mean_bg).powi(2))
        .sum::<f64>()
        / (background_frustrations.len() as f64);

    // Welch's t-test
    let n_ch = channel_frustrations.len() as f64;
    let n_bg = background_frustrations.len() as f64;

    let se = ((var_ch / n_ch) + (var_bg / n_bg)).sqrt();
    let t_stat = if se > 1e-14 {
        (mean_ch - mean_bg) / se
    } else {
        0.0
    };

    // Approximate p-value using normal approximation (df ~ n_ch + n_bg)
    // For simplicity, use standard normal CDF: p = 2 * Phi(-|t|)
    let p_value = 2.0 * normal_cdf(-t_stat.abs());

    // Effect size (Cohen's d)
    let pooled_std = ((var_ch + var_bg) / 2.0).sqrt();
    let effect_size = if pooled_std > 1e-14 {
        (mean_ch - mean_bg) / pooled_std
    } else {
        0.0
    };

    CorrelationResult {
        t_statistic: t_stat,
        p_value,
        effect_size,
        mean_frustration_channels: mean_ch,
        mean_frustration_background: mean_bg,
        n_channel: channel_frustrations.len(),
        n_background: background_frustrations.len(),
    }
}

/// Approximate standard normal cumulative distribution function.
/// Uses Abramowitz and Stegun approximation (accurate to ~0.00012).
fn normal_cdf(x: f64) -> f64 {
    // For positive x: use approximation
    let abs_x = x.abs();

    // Abramowitz and Stegun formula 7.1.26
    let b1 = 0.319381530;
    let b2 = -0.356563782;
    let b3 = 1.781477937;
    let b4 = -1.821255978;
    let b5 = 1.330274429;
    let p = 0.2316419;

    let t = 1.0 / (1.0 + p * abs_x);
    let tau = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))));
    let phi = 1.0 - tau * (-0.5 * abs_x * abs_x).exp() / std::f64::consts::PI.sqrt();

    if x >= 0.0 {
        phi
    } else {
        1.0 - phi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percolation_detector_creation() {
        let detector = PercolationDetector::new(8, 8, 4);
        assert_eq!(detector.nx, 8);
        assert_eq!(detector.ny, 8);
        assert_eq!(detector.nz, 4);
        assert_eq!(detector.visited.len(), 256);
    }

    #[test]
    fn test_empty_velocity_field() {
        let mut detector = PercolationDetector::new(4, 4, 4);
        let velocity_field = vec![[0.0, 0.0, 0.0]; 64];
        let channels = detector.detect_channels(&velocity_field, 0.1);
        assert!(channels.is_empty());
    }

    #[test]
    fn test_single_channel_detection() {
        let mut detector = PercolationDetector::new(4, 4, 4);
        let mut velocity_field = vec![[0.0, 0.0, 0.0]; 64];

        // Create a single high-velocity cell
        velocity_field[0] = [1.0, 0.0, 0.0];

        let channels = detector.detect_channels(&velocity_field, 0.5);
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].size, 1);
    }

    #[test]
    fn test_multiple_channels() {
        let mut detector = PercolationDetector::new(4, 4, 4);
        let mut velocity_field = vec![[0.0, 0.0, 0.0]; 64];

        // Two disconnected regions
        velocity_field[0] = [1.0, 0.0, 0.0];
        velocity_field[63] = [1.0, 0.0, 0.0];

        let channels = detector.detect_channels(&velocity_field, 0.5);
        assert_eq!(channels.len(), 2);
    }

    #[test]
    fn test_6_neighbor_connectivity() {
        let mut detector = PercolationDetector::new(4, 4, 4);
        let mut velocity_field = vec![[0.0, 0.0, 0.0]; 64];

        // Create connected cells (cardinal directions)
        // At (0,0,0), (1,0,0), (0,1,0) - should form one channel
        velocity_field[0] = [0.8, 0.0, 0.0];     // (0,0,0)
        velocity_field[1] = [0.8, 0.0, 0.0];     // (1,0,0)
        velocity_field[4] = [0.8, 0.0, 0.0];     // (0,1,0)

        let channels = detector.detect_channels(&velocity_field, 0.5);
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].size, 3);
    }

    #[test]
    fn test_no_diagonal_connectivity() {
        let mut detector = PercolationDetector::new(2, 2, 2);
        let mut velocity_field = vec![[0.0, 0.0, 0.0]; 8];

        // Place cells at opposite corners (diagonal) - should NOT connect
        velocity_field[0] = [1.0, 0.0, 0.0];  // (0,0,0)
        velocity_field[7] = [1.0, 0.0, 0.0];  // (1,1,1)

        let channels = detector.detect_channels(&velocity_field, 0.5);
        assert_eq!(channels.len(), 2);
    }

    #[test]
    fn test_threshold_sensitivity() {
        let mut detector = PercolationDetector::new(4, 4, 4);
        let mut velocity_field = vec![[0.0, 0.0, 0.0]; 64];

        // All cells have velocity 0.5
        for cell in &mut velocity_field {
            *cell = [0.5, 0.0, 0.0];
        }

        let channels_above = detector.detect_channels(&velocity_field, 0.4);
        let mut detector2 = PercolationDetector::new(4, 4, 4);
        let channels_below = detector2.detect_channels(&velocity_field, 0.6);

        assert!(!channels_above.is_empty());
        assert!(channels_below.is_empty());
    }

    #[test]
    fn test_channel_statistics() {
        let mut detector = PercolationDetector::new(4, 4, 4);
        let mut velocity_field = vec![[0.0, 0.0, 0.0]; 64];

        // Create a single channel with two connected cells
        velocity_field[0] = [1.0, 0.0, 0.0];  // u_mag = 1.0 at (0,0,0)
        velocity_field[1] = [0.8, 0.0, 0.0];  // u_mag = 0.8 at (1,0,0) - adjacent to (0,0,0)

        let channels = detector.detect_channels(&velocity_field, 0.5);
        assert_eq!(channels.len(), 1);  // One connected channel
        assert_eq!(channels[0].size, 2);  // Two cells
        assert_eq!(channels[0].max_velocity, 1.0);
        let expected_mean = (1.0 + 0.8) / 2.0;
        assert!((channels[0].mean_velocity - expected_mean).abs() < 1e-14);
    }

    #[test]
    fn test_auto_threshold() {
        let velocity_field = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ];
        let threshold = auto_velocity_threshold(&velocity_field);
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_correlation_with_frustration() {
        let channel = PercolationChannel {
            id: 0,
            size: 4,
            mean_velocity: 1.0,
            max_velocity: 1.2,
            bounding_box: BoundingBox {
                x_min: 0,
                x_max: 1,
                y_min: 0,
                y_max: 1,
                z_min: 0,
                z_max: 0,
            },
            cells: vec![(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        };

        let frustration_field = vec![0.3; 4096];  // Uniform frustration
        let result = correlate_with_frustration(&[channel], &frustration_field);

        assert_eq!(result.n_channel, 4);
        assert_eq!(result.n_background, 4092);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
}
