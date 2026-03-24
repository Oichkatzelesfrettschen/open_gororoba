//! N-dimensional Entropy Growth PDE simulation.
//!
//! Implements:
//!   dS/dt = D * Lap(S) - gamma * S - alpha * S^3 + J
//! where S is the entropy field, D is diffusion (mapped from algebraic twist),
//! gamma is linear decay, and alpha is non-linear saturation.

/// Parameters for Entropy PDE.
pub struct EntropyParams {
    pub diffusion: f64,
    pub gamma: f64,
    pub alpha: f64,
    pub source: f64,
    pub dt: f64,
}

impl Default for EntropyParams {
    fn default() -> Self {
        Self {
            diffusion: 0.25,
            gamma: 0.01,
            alpha: 0.1,
            source: 0.05,
            dt: 0.1,
        }
    }
}

/// 3D Entropy field simulation on a periodic grid.
pub struct EntropyField3D {
    pub s: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub params: EntropyParams,
}

impl EntropyField3D {
    pub fn new(nx: usize, ny: usize, nz: usize, params: EntropyParams) -> Self {
        let mut s = vec![0.0; nx * ny * nz];
        // Seed center
        s[(nx / 2) * ny * nz + (ny / 2) * nz + (nz / 2)] = 0.1;
        Self {
            s,
            nx,
            ny,
            nz,
            params,
        }
    }

    pub fn step(&mut self) {
        let n_total = self.nx * self.ny * self.nz;
        let mut ds = vec![0.0; n_total];
        let p = &self.params;

        for x in 0..self.nx {
            for y in 0..self.ny {
                for z in 0..self.nz {
                    let idx = x * self.ny * self.nz + y * self.nz + z;

                    let idx_px = ((x + 1) % self.nx) * self.ny * self.nz + y * self.nz + z;
                    let idx_mx =
                        ((x + self.nx - 1) % self.nx) * self.ny * self.nz + y * self.nz + z;
                    let idx_py = x * self.ny * self.nz + ((y + 1) % self.ny) * self.nz + z;
                    let idx_my =
                        x * self.ny * self.nz + ((y + self.ny - 1) % self.ny) * self.nz + z;
                    let idx_pz = x * self.ny * self.nz + y * self.nz + ((z + 1) % self.nz);
                    let idx_mz =
                        x * self.ny * self.nz + y * self.nz + ((z + self.nz - 1) % self.nz);

                    let lap = self.s[idx_px]
                        + self.s[idx_mx]
                        + self.s[idx_py]
                        + self.s[idx_my]
                        + self.s[idx_pz]
                        + self.s[idx_mz]
                        - 6.0 * self.s[idx];

                    ds[idx] =
                        p.diffusion * lap - p.gamma * self.s[idx] - p.alpha * self.s[idx].powi(3)
                            + p.source;
                }
            }
        }

        for (s, delta) in self.s.iter_mut().zip(ds.iter()).take(n_total) {
            *s = (*s + delta * p.dt).max(0.0);
        }
    }

    pub fn mean_entropy(&self) -> f64 {
        self.s.iter().sum::<f64>() / self.s.len() as f64
    }
}
