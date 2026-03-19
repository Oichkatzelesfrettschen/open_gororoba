//! Genesis Simulation: 2D Vacuum Instability and Soliton Formation (C-010).
//!
//! Implements a 2D spectral evolution of a complex field under the 
//! "Negative Dimension" anti-diffusion operator:
//!   i * dpsi/dt = -(-Delta)^alpha * psi + V(|psi|^2) * psi
//! where alpha = -1.5.
//!
//! Migrated from src/genesis_simulation.py.

use ndarray::{Array2, Zip};
use num_complex::Complex64;
use crate::ndfft::{fft_2d, ifft_2d, freq_index};
use std::f64::consts::PI;

/// Parameters for the Genesis simulation.
pub struct GenesisParams {
    /// Grid resolution (NxN)
    pub n: usize,
    /// Domain size (L x L)
    pub l: f64,
    /// Sedenion fractional exponent (typically -1.5)
    pub alpha: f64,
    /// Time step
    pub dt: f64,
    /// Non-linear coupling strength
    pub coupling: f64,
}

impl Default for GenesisParams {
    fn default() -> Self {
        Self {
            n: 256,
            l: 100.0,
            alpha: -1.5,
            dt: 0.01,
            coupling: 10.0,
        }
    }
}

/// A 2D Genesis simulation state.
pub struct GenesisSim {
    pub psi: Array2<Complex64>,
    pub params: GenesisParams,
    /// Kinetic operator in k-space
    pub t_k: Array2<Complex64>,
}

impl GenesisSim {
    /// Initialize a new simulation with random vacuum noise.
    pub fn new(params: GenesisParams) -> Self {
        let n = params.n;
        let mut psi = Array2::<Complex64>::zeros((n, n));
        let _dx = params.l / n as f64;

        // Seed with random noise (stub for proper K^-0.75 scaling)
        // In a real run, we'd apply the spectral filter here.
        for val in psi.iter_mut() {
            *val = Complex64::new(rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5);
        }

        // Precompute kinetic operator T(k) = -(|k| + eps)^(2*alpha)
        let mut t_k = Array2::<Complex64>::zeros((n, n));
        for ((i, j), val) in t_k.indexed_iter_mut() {
            let kx = freq_index(i, n) * 2.0 * PI / params.l;
            let ky = freq_index(j, n) * 2.0 * PI / params.l;
            let k_mag = (kx * kx + ky * ky).sqrt();
            let tk_val = -(k_mag + 0.1).powf(2.0 * params.alpha);
            *val = Complex64::new(tk_val, 0.0);
        }

        Self { psi, params, t_k }
    }

    /// Perform one time step of the Genesis evolution.
    pub fn step(&mut self) {
        // 1. Kinetic evolution in k-space (Split-step Fourier)
        let mut psi_k = fft_2d(&self.psi);
        let dt = self.params.dt;
        
        Zip::from(&mut psi_k).and(&self.t_k).for_each(|pk, &tk| {
            // exp(-i * T_k * dt)
            let phase = -tk.re * dt;
            let prop = Complex64::new(phase.cos(), phase.sin());
            *pk *= prop;
        });
        
        self.psi = ifft_2d(&psi_k);

        // 2. Non-linear interaction in real space
        // V = -coupling * |psi|^2
        let coupling = self.params.coupling;
        for val in self.psi.iter_mut() {
            let density = val.norm_sqr();
            let phase = -coupling * density * dt;
            let prop = Complex64::new(phase.cos(), phase.sin());
            *val *= prop;
        }

        // 3. Renormalize to maintain unitary vacuum
        let norm_sq: f64 = self.psi.iter().map(|c| c.norm_sqr()).sum();
        let norm = (norm_sq / (self.params.n * self.params.n) as f64).sqrt();
        if norm > 1e-12 {
            for val in self.psi.iter_mut() {
                *val /= norm;
            }
        }
    }

    /// Identify "Gravastars" (local density peaks above threshold).
    pub fn count_gravastars(&self, threshold: f64) -> usize {
        let mut count = 0;
        let n = self.params.n;
        let density = self.psi.mapv(|c| c.norm_sqr());
        
        for i in 1..n-1 {
            for j in 1..n-1 {
                let d = density[[i, j]];
                if d > threshold {
                    // Check if local maximum
                    if d >= density[[i-1, j]] && d >= density[[i+1, j]] &&
                       d >= density[[i, j-1]] && d >= density[[i, j+1]] {
                        count += 1;
                    }
                }
            }
        }
        count
    }
}
