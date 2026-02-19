//! Zero-Divisor Census: Topological filtration of vacuum collapse events.
//!
//! This module implements a "Zero-Divisor Walk" simulation where particles
//! traverse a continuous Sedenion field interpolated from the discrete
//! APT vacuum state.
//!
//! # Theory
//!
//! In the Sedenion algebra (S), zero divisors are dense. A "Collapse Event"
//! is defined as a point (x,y,z) where the local field value V(x,y,z)
//! forms a zero-divisor pair with the particle's internal state P.
//!
//! Metric: rho = |P * V| / (|P| * |V|)
//! If rho < threshold, the particle "collapses" (its state is annihilated).
//! The set of all collapse points forms the "Singular Manifold" of the vacuum.

use algebra_core::{cd_multiply, cd_norm_sq};
use vacuum_frustration::AptSedenionField;

/// Represents a continuous Sedenion field interpolated from discrete grid data.
pub struct ContinuousVacuum {
    grid_size: usize,
    /// Flat buffer of 16D vectors (16 * grid_size^3 floats)
    data: Vec<f64>,
}

impl ContinuousVacuum {
    /// Create a continuous field from an APT discrete field.
    /// Uses one-hot encoding of basis indices.
    pub fn from_apt(apt: &AptSedenionField) -> Self {
        let onehot_vecs = apt.to_onehot_16d();
        let mut data = Vec::with_capacity(onehot_vecs.len() * 16);
        for v in onehot_vecs {
            data.extend(v);
        }
        
        // Infer grid size from total cells (assuming cube)
        let total_cells = apt.n_cells();
        let grid_size = (total_cells as f64).cbrt().round() as usize;

        Self { grid_size, data }
    }

    /// Trilinear interpolation of the Sedenion field at (x, y, z).
    /// Coordinates are in grid space [0, grid_size).
    /// Handles periodic boundary conditions.
    pub fn sample(&self, x: f64, y: f64, z: f64) -> Vec<f64> {
        let size = self.grid_size as f64;
        
        // Wrap coordinates
        let x = x.rem_euclid(size);
        let y = y.rem_euclid(size);
        let z = z.rem_euclid(size);

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let z0 = z.floor() as usize;

        let x1 = (x0 + 1) % self.grid_size;
        let y1 = (y0 + 1) % self.grid_size;
        let z1 = (z0 + 1) % self.grid_size;

        let u = x - x.floor();
        let v = y - y.floor();
        let w = z - z.floor();

        // Helper to get 16D vector at grid index
        let get_vec = |ix, iy, iz| {
            let idx = (ix + iy * self.grid_size + iz * self.grid_size * self.grid_size) * 16;
            &self.data[idx..idx + 16]
        };

        let c000 = get_vec(x0, y0, z0);
        let c100 = get_vec(x1, y0, z0);
        let c010 = get_vec(x0, y1, z0);
        let c110 = get_vec(x1, y1, z0);
        let c001 = get_vec(x0, y0, z1);
        let c101 = get_vec(x1, y0, z1);
        let c011 = get_vec(x0, y1, z1);
        let c111 = get_vec(x1, y1, z1);

        // Interpolate each component
        let mut result = vec![0.0; 16];
        for i in 0..16 {
            let i00 = c000[i] * (1.0 - u) + c100[i] * u;
            let i10 = c010[i] * (1.0 - u) + c110[i] * u;
            let i01 = c001[i] * (1.0 - u) + c101[i] * u;
            let i11 = c011[i] * (1.0 - u) + c111[i] * u;

            let i0 = i00 * (1.0 - v) + i10 * v;
            let i1 = i01 * (1.0 - v) + i11 * v;

            result[i] = i0 * (1.0 - w) + i1 * w;
        }

        result
    }
}

/// A particle probing the algebraic vacuum.
pub struct ZeroDivisorWalker {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub state: Vec<f64>, // 16D internal state
}

impl ZeroDivisorWalker {
    pub fn new(position: [f64; 3], velocity: [f64; 3], state: Vec<f64>) -> Self {
        Self { position, velocity, state }
    }

    /// Step the walker and check for collapse.
    /// Returns Some(collapse_metric) if collapse is imminent (metric < threshold).
    pub fn step(&mut self, vacuum: &ContinuousVacuum, dt: f64) -> Option<f64> {
        // Update position
        for i in 0..3 {
            self.position[i] += self.velocity[i] * dt;
        }

        // Sample vacuum at new position
        let field_val = vacuum.sample(self.position[0], self.position[1], self.position[2]);

        // Interact: Check Zero Divisor Metric
        // metric = |state * field| / (|state| * |field|)
        // Ideally should be 1.0. If 0.0, we hit a zero divisor.
        
        let prod = cd_multiply(&self.state, &field_val);
        
        // We assume 16D
        let norm_sq_state = cd_norm_sq(&self.state);
        let norm_sq_field = cd_norm_sq(&field_val);
        let norm_sq_prod = cd_norm_sq(&prod);

        if norm_sq_state < 1e-9 || norm_sq_field < 1e-9 {
            // Trivial zero product, not interesting
            return None;
        }

        let metric = (norm_sq_prod / (norm_sq_state * norm_sq_field)).sqrt();

        // Evolve state: Rotation/Precession by field
        // Simple update: S_new = normalize(S + alpha * (S * V))
        // This is a "measurement" update.
        let alpha = 0.1 * dt;
        for (s, p) in self.state.iter_mut().zip(prod.iter()) {
            *s += alpha * p;
        }
        // Renormalize to keep stable
        let new_norm = cd_norm_sq(&self.state).sqrt();
        if new_norm > 1e-9 {
            for x in &mut self.state {
                *x /= new_norm;
            }
        }

        Some(metric)
    }
}

/// Simulation result containing the locations of zero-divisor collapses.
pub struct CollapseManifold {
    pub points: Vec<[f64; 3]>,
    pub min_metric: f64,
}

/// Run a census of zero-divisor collapse points.
pub fn simulate_collapse_manifold(
    vacuum: &ContinuousVacuum,
    n_particles: usize,
    steps: usize,
    threshold: f64,
) -> CollapseManifold {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut walkers = Vec::with_capacity(n_particles);
    let size = vacuum.grid_size as f64;

    // Initialize walkers with random positions and random Sedenion states
    for _ in 0..n_particles {
        let pos = [
            rng.r#gen::<f64>() * size,
            rng.r#gen::<f64>() * size,
            rng.r#gen::<f64>() * size,
        ];
        let vel = [
            (rng.r#gen::<f64>() - 0.5) * 0.1,
            (rng.r#gen::<f64>() - 0.5) * 0.1,
            (rng.r#gen::<f64>() - 0.5) * 0.1,
        ];
        let mut state = vec![0.0; 16];
        for x in &mut state {
            *x = rng.r#gen::<f64>() - 0.5;
        }
        // Normalize state
        let norm = cd_norm_sq(&state).sqrt();
        for x in &mut state {
            *x /= norm;
        }

        walkers.push(ZeroDivisorWalker::new(pos, vel, state));
    }

    let mut collapse_points = Vec::new();
    let mut global_min_metric = 1.0;

    for _ in 0..steps {
        for walker in &mut walkers {
            if let Some(metric) = walker.step(vacuum, 1.0) {
                if metric < global_min_metric {
                    global_min_metric = metric;
                }
                if metric < threshold {
                    // Record collapse
                    collapse_points.push(walker.position);
                    // Respawn walker to continue sampling
                    walker.position = [
                         rng.r#gen::<f64>() * size,
                         rng.r#gen::<f64>() * size,
                         rng.r#gen::<f64>() * size,
                    ];
                }
            }
        }
    }

    CollapseManifold {
        points: collapse_points,
        min_metric: global_min_metric,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vacuum_frustration::AptSedenionField;

    #[test]
    fn test_continuous_vacuum_creation() {
        let apt = AptSedenionField::new(4, 42); // 4x4x4 grid
        let vacuum = ContinuousVacuum::from_apt(&apt);
        
        // 4^3 * 16 = 64 * 16 = 1024
        assert_eq!(vacuum.data.len(), 1024);
        assert_eq!(vacuum.grid_size, 4);
    }

    #[test]
    fn test_vacuum_sampling_bounds() {
        let apt = AptSedenionField::new(4, 42);
        let vacuum = ContinuousVacuum::from_apt(&apt);

        // Sample at grid point
        let s1 = vacuum.sample(0.0, 0.0, 0.0);
        assert_eq!(s1.len(), 16);
        // Norm should be approx 1.0 (interpolation of unit vectors)
        // Note: linear interpolation of unit vectors is not necessarily unit norm, 
        // but close if vectors are aligned. If they are orthogonal, norm decreases.
        // E.g. (e1 + e2)/2 has norm 1/sqrt(2).
        
        // Sample off-grid
        let s2 = vacuum.sample(0.5, 0.5, 0.5);
        assert_eq!(s2.len(), 16);
    }

    #[test]
    fn test_collapse_simulation_runs() {
        let apt = AptSedenionField::new(4, 12345);
        let vacuum = ContinuousVacuum::from_apt(&apt);

        // Run simulation with high threshold to force some "collapses" 
        // (technically just proximity to zero divisor)
        let result = simulate_collapse_manifold(&vacuum, 10, 50, 0.5);
        
        // We expect it to run. We don't guarantee collapses with random seed,
        // but min_metric should be <= 1.0.
        assert!(result.min_metric <= 1.0000001);
        
        println!("Min metric found: {}", result.min_metric);
        println!("Collapse points: {}", result.points.len());
    }
}
