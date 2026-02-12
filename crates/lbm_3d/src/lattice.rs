//! D3Q19 Lattice Boltzmann lattice definition.
//!
//! The D3Q19 lattice is a standard choice for 3D LBM simulations.
//! It includes:
//! - 1 rest particle (velocity = 0)
//! - 6 axis-aligned particles (c_s^2 = 1/3)
//! - 12 diagonal particles (c_s^2 = 1/3)
//!
//! Total: 19 discrete velocity directions.

/// D3Q19 lattice with 19 discrete velocity directions.
#[derive(Clone, Debug)]
pub struct D3Q19Lattice {
    /// Discrete velocities as [vx, vy, vz] triplets (19 directions)
    pub velocities: [[i32; 3]; 19],
    /// Weights for equilibrium distribution
    pub weights: [f64; 19],
    /// Lattice speed of sound squared (c_s^2 = 1/3 for D3Q19)
    pub cs_sq: f64,
}

impl D3Q19Lattice {
    /// Create a D3Q19 lattice with standard D3Q19 parameters.
    ///
    /// Velocity ordering:
    /// - Index 0: Rest particle (0,0,0)
    /// - Indices 1-6: Axis-aligned (+/-1,0,0), (0,+/-1,0), (0,0,+/-1)
    /// - Indices 7-18: Face diagonals (+/-1,+/-1,0), (+/-1,0,+/-1), (0,+/-1,+/-1)
    pub fn new() -> Self {
        Self {
            velocities: D3Q19Lattice::velocities(),
            weights: D3Q19Lattice::weights(),
            cs_sq: 1.0 / 3.0,
        }
    }

    /// Get the 19 velocity directions for D3Q19 lattice.
    fn velocities() -> [[i32; 3]; 19] {
        [
            // Index 0: rest
            [0, 0, 0],
            // Indices 1-6: axis-aligned (weight = 1/18)
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            // Indices 7-18: face diagonals (weight = 1/36)
            [1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [1, 0, 1],
            [-1, 0, -1],
            [1, 0, -1],
            [-1, 0, 1],
            [0, 1, 1],
            [0, -1, -1],
            [0, 1, -1],
            [0, -1, 1],
        ]
    }

    /// Get the 19 weights for equilibrium distribution.
    ///
    /// Standard D3Q19 weights:
    /// - w_0 = 1/3 (rest)
    /// - w_1..w_6 = 1/18 (axis-aligned)
    /// - w_7..w_18 = 1/36 (face diagonals)
    fn weights() -> [f64; 19] {
        [
            1.0 / 3.0,  // rest
            1.0 / 18.0, // axis-aligned (6 directions)
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 36.0, // face diagonals (12 directions)
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ]
    }

    /// Get weight for a given velocity index.
    pub fn weight(&self, i: usize) -> f64 {
        self.weights[i]
    }

    /// Get velocity vector for a given direction.
    pub fn velocity(&self, i: usize) -> [i32; 3] {
        self.velocities[i]
    }

    /// Compute the opposite direction index (bounce-back).
    ///
    /// For each velocity direction c_i, the opposite is -c_i.
    pub fn opposite_direction(&self, i: usize) -> usize {
        let [vx, vy, vz] = self.velocities[i];
        let opposite = [-vx, -vy, -vz];

        // Find index of opposite velocity
        for (idx, &vel) in self.velocities.iter().enumerate() {
            if vel == opposite {
                return idx;
            }
        }

        panic!("No opposite direction found for index {}", i);
    }

    /// Total number of directions.
    pub fn num_directions(&self) -> usize {
        19
    }

    /// Verify lattice invariants.
    ///
    /// Returns true if:
    /// - Weights sum to 1.0
    /// - cs_sq = 1/3
    /// - Velocities are symmetric
    pub fn verify_invariants(&self) -> bool {
        // Check weight sum
        let weight_sum: f64 = self.weights.iter().sum();
        if (weight_sum - 1.0).abs() > 1e-14 {
            return false;
        }

        // Check cs_sq
        if (self.cs_sq - 1.0 / 3.0).abs() > 1e-14 {
            return false;
        }

        // Check velocity symmetry (opposite directions exist)
        for i in 0..19 {
            let _opposite = self.opposite_direction(i);
            // If no panic, opposite exists
        }

        true
    }

    /// Compute equilibrium distribution.
    ///
    /// f_i^eq(rho, u) = rho * w_i * [1 + (c_i*u)/c_s^2 + (c_i*u)^2/(2c_s^4) - u^2/(2c_s^2)]
    ///
    /// # Arguments
    /// * `rho` - Local density
    /// * `u` - Local velocity [ux, uy, uz]
    /// * `i` - Velocity index
    pub fn equilibrium(&self, rho: f64, u: [f64; 3], i: usize) -> f64 {
        let w = self.weight(i);
        let c = self.velocity(i);

        // c_i * u
        let cu = (c[0] as f64) * u[0] + (c[1] as f64) * u[1] + (c[2] as f64) * u[2];

        // u^2
        let u_sq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];

        // f_i^eq = rho * w_i * [1 + (cu)/cs^2 + (cu)^2/(2*cs^4) - u^2/(2*cs^2)]
        let cs2_inv = 1.0 / self.cs_sq;
        let cs4_inv = cs2_inv * cs2_inv;

        w * rho * (1.0 + cu * cs2_inv + cu * cu * cs4_inv / 2.0 - u_sq * cs2_inv / 2.0)
    }
}

impl Default for D3Q19Lattice {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d3q19_creation() {
        let lattice = D3Q19Lattice::new();
        assert_eq!(lattice.num_directions(), 19);
        assert_eq!(lattice.velocities.len(), 19);
        assert_eq!(lattice.weights.len(), 19);
    }

    #[test]
    fn test_weight_normalization() {
        let lattice = D3Q19Lattice::new();
        let weight_sum: f64 = lattice.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-14, "Weights must sum to 1.0");
    }

    #[test]
    fn test_cs_squared() {
        let lattice = D3Q19Lattice::new();
        let expected_cs_sq = 1.0 / 3.0;
        assert!(
            (lattice.cs_sq - expected_cs_sq).abs() < 1e-14,
            "cs_sq must be 1/3"
        );
    }

    #[test]
    fn test_velocity_symmetry() {
        let lattice = D3Q19Lattice::new();
        // For each direction, opposite must exist
        for i in 0..19 {
            let opposite_idx = lattice.opposite_direction(i);
            let [vx, vy, vz] = lattice.velocity(i);
            let [ox, oy, oz] = lattice.velocity(opposite_idx);
            assert_eq!(vx, -ox);
            assert_eq!(vy, -oy);
            assert_eq!(vz, -oz);
        }
    }

    #[test]
    fn test_equilibrium_zero_velocity() {
        let lattice = D3Q19Lattice::new();
        let rho = 1.0;
        let u = [0.0, 0.0, 0.0];

        // At zero velocity, f_eq = rho * w_i
        for i in 0..19 {
            let f_eq = lattice.equilibrium(rho, u, i);
            let expected = rho * lattice.weight(i);
            assert!((f_eq - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_equilibrium_normalization() {
        let lattice = D3Q19Lattice::new();
        let rho = 2.0;
        let u = [0.1, 0.05, 0.02];

        // Sum of equilibrium should equal rho
        let sum_eq: f64 = (0..19).map(|i| lattice.equilibrium(rho, u, i)).sum();
        assert!((sum_eq - rho).abs() < 1e-12);
    }

    #[test]
    fn test_lattice_invariants() {
        let lattice = D3Q19Lattice::new();
        assert!(lattice.verify_invariants());
    }

    #[test]
    fn test_velocity_directions_count() {
        let lattice = D3Q19Lattice::new();
        assert_eq!(lattice.velocities.len(), 19);
        // Should have 1 rest + 6 axis-aligned + 12 face-diagonals
        assert_eq!(lattice.velocities[0], [0, 0, 0]);
    }

    #[test]
    fn test_weight_distribution() {
        let lattice = D3Q19Lattice::new();
        let w_rest = lattice.weight(0);
        let w_axis = lattice.weight(1);
        let w_diag = lattice.weight(7);

        assert!((w_rest - 1.0 / 3.0).abs() < 1e-14);
        assert!((w_axis - 1.0 / 18.0).abs() < 1e-14);
        assert!((w_diag - 1.0 / 36.0).abs() < 1e-14);
    }
}
