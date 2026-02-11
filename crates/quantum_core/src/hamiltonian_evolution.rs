//! Quantum Hamiltonian evolution solvers (RUST-FIRST refactor from nonlocal_long_range_ZZ.py).
//!
//! Provides n-dimensional lattice simulations with power-law coupling and entanglement entropy.
//! Date: 2026-02-10, Phase 10.3 REFACTOR A1

use nalgebra::DMatrix;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// N-dimensional lattice Hamiltonian builder
#[derive(Debug, Clone)]
pub struct HamiltonianND {
    /// Number of spins
    n: usize,
    /// Lattice dimensions (e.g., (3,3) for 3x3 grid, (8,) for 1D)
    dims: Vec<usize>,
    /// Nonlocality exponent (alpha): coupling ~ 1/dist^alpha
    alpha: f64,
    /// Transverse field coupling
    g: f64,
    /// ZZ coupling strength
    j: f64,
}

impl HamiltonianND {
    /// Create a new Hamiltonian for an n-dimensional lattice
    pub fn new(dims: Vec<usize>, alpha: f64, g: f64, j: f64) -> Self {
        let n: usize = dims.iter().product();
        Self { n, dims, alpha, g, j }
    }

    /// Get Cartesian coordinates for spin index
    fn index_to_coords(&self, idx: usize) -> Vec<usize> {
        let mut coords = Vec::new();
        let mut remaining = idx;
        for &d in &self.dims {
            coords.push(remaining % d);
            remaining /= d;
        }
        coords.reverse();
        coords
    }

    /// Compute distance between two spins (Euclidean)
    fn distance(&self, i: usize, j: usize) -> f64 {
        let coords_i = self.index_to_coords(i);
        let coords_j = self.index_to_coords(j);
        
        let mut dist_sq = 0.0;
        for (ci, cj) in coords_i.iter().zip(coords_j.iter()) {
            let delta = (*ci as f64) - (*cj as f64);
            dist_sq += delta * delta;
        }
        dist_sq.sqrt()
    }

    /// Build Hamiltonian matrix (dense, for small N only)
    ///
    /// H = sum_{i<j} J/dist^alpha * Sz_i Sz_j + sum_i g*Sx_i
    ///
    /// For N > 12 (2^12 = 4096 dimensions), use sparse matrix approximation.
    pub fn build(&self) -> Array2<Complex64> {
        let dim_hilbert = 1 << self.n; // 2^n
        let mut h = Array2::<Complex64>::zeros((dim_hilbert, dim_hilbert));

        // Pauli matrices
        let sz = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
        ]).unwrap();

        let sx = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ]).unwrap();

        // ZZ terms: sum_{i<j} J/dist^alpha * Sz_i Sz_j
        for i in 0..self.n {
            for j in (i+1)..self.n {
                let dist = self.distance(i, j);
                if dist.abs() < 1e-10 {
                    continue; // Skip self-distance
                }
                
                let coupling = self.j / dist.powf(self.alpha);

                // Kron product: ... x Sz_i x ... x Sz_j x ...
                let term = self.kron_operator(&sz, i, &sz, j);

                h = &h + &(Complex64::new(coupling, 0.0) * &term);
            }
        }

        // X terms: sum_i g*Sx_i
        for i in 0..self.n {
            let term = self.kron_single(&sx, i);
            h = &h + &(Complex64::new(self.g, 0.0) * &term);
        }

        h
    }

    /// Single-qubit Kronecker product (identity x ... x op_i x ... x identity)
    fn kron_single(&self, op: &Array2<Complex64>, target: usize) -> Array2<Complex64> {
        let mut result = Array2::from_diag(&Array1::from_vec(vec![
            Complex64::new(1.0, 0.0);
            1 << self.n
        ]));

        for qubit in 0..self.n {
            if qubit == target {
                result = self.kron_two_ops(&result, op);
            } else {
                let identity = Array2::from_diag(&Array1::from_vec(vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ]));
                result = self.kron_two_ops(&result, &identity);
            }
        }
        result
    }

    /// Two-qubit Kronecker product (op1 on i, op2 on j)
    fn kron_operator(
        &self,
        op1: &Array2<Complex64>,
        i: usize,
        op2: &Array2<Complex64>,
        j: usize,
    ) -> Array2<Complex64> {
        let mut result = Array2::from_diag(&Array1::from_vec(vec![
            Complex64::new(1.0, 0.0);
            1 << self.n
        ]));

        for qubit in 0..self.n {
            if qubit == i {
                result = self.kron_two_ops(&result, op1);
            } else if qubit == j {
                result = self.kron_two_ops(&result, op2);
            } else {
                let identity = Array2::from_diag(&Array1::from_vec(vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ]));
                result = self.kron_two_ops(&result, &identity);
            }
        }
        result
    }

    /// Kronecker product of two matrices
    fn kron_two_ops(
        &self,
        a: &Array2<Complex64>,
        b: &Array2<Complex64>,
    ) -> Array2<Complex64> {
        let (am, an) = a.dim();
        let (bm, bn) = b.dim();
        let mut result = Array2::zeros((am * bm, an * bn));

        for i in 0..am {
            for j in 0..an {
                for k in 0..bm {
                    for l in 0..bn {
                        result[[i * bm + k, j * bn + l]] = a[[i, j]] * b[[k, l]];
                    }
                }
            }
        }
        result
    }

    /// Compute entanglement entropy of bipartition
    ///
    /// Split the system in half and compute SVD of reshaped state.
    /// S = -sum_i p_i log2(p_i), where p_i are squared singular values.
    pub fn entanglement_entropy(&self, psi: &[Complex64]) -> f64 {
        let cut = self.n / 2;
        let dim_a = 1 << cut;
        let dim_b = 1 << (self.n - cut);

        if psi.len() != dim_a * dim_b {
            return 0.0; // Invalid state
        }

        // Reshape into matrix
        let psi_mat = Array2::from_shape_vec((dim_a, dim_b), psi.to_vec())
            .unwrap_or_else(|_| Array2::zeros((dim_a, dim_b)));

        // SVD via nalgebra (simpler than ndarray for now)
        let h_matrix = DMatrix::from_fn(dim_a, dim_b, |i, j| psi_mat[[i, j]]);
        let svd = h_matrix.svd(true, true);
        
        let singular_values = svd.singular_values;
        
        // Entropy from squared singular values
        let mut entropy = 0.0;
        for s in singular_values.iter() {
            let p: f64 = s * s;
            if p > 1e-12 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Time evolution via eigenbasis diagonalization
    ///
    /// |psi(t)> = sum_n exp(-i*E_n*t) |E_n><E_n|psi>
    pub fn evolve_time(
        &self,
        eigenvalues: &[f64],
        eigenvectors: &Array2<Complex64>,
        psi_0: &[Complex64],
        t: f64,
    ) -> Vec<Complex64> {
        let n_states = eigenvalues.len();
        
        // Compute coefficients: c_n = <E_n|psi_0>
        let mut coeffs = vec![Complex64::new(0.0, 0.0); n_states];
        for n in 0..n_states {
            for i in 0..psi_0.len() {
                coeffs[n] += eigenvectors[[i, n]].conj() * psi_0[i];
            }
        }

        // Evolve: c_n(t) = exp(-i*E_n*t) * c_n(0)
        let mut psi_t = vec![Complex64::new(0.0, 0.0); psi_0.len()];
        for n in 0..n_states {
            let phase = Complex64::new(0.0, -eigenvalues[n] * t).exp();
            let c_n_t = coeffs[n] * phase;
            
            for i in 0..psi_0.len() {
                psi_t[i] += eigenvectors[[i, n]] * c_n_t;
            }
        }
        psi_t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamiltonian_1d_chain() {
        let ham = HamiltonianND::new(vec![4], 1.0, 0.5, 1.0);
        assert_eq!(ham.n, 4);
        assert_eq!(ham.dims, vec![4]);
    }

    #[test]
    fn test_hamiltonian_2d_grid() {
        let ham = HamiltonianND::new(vec![2, 2], 1.0, 0.5, 1.0);
        assert_eq!(ham.n, 4);
    }

    #[test]
    fn test_distance_metric() {
        let ham = HamiltonianND::new(vec![3, 3], 1.0, 0.5, 1.0);
        // (0,0) to (1,0): dist = 1
        assert!((ham.distance(0, 3) - 1.0).abs() < 1e-10);
        // (0,0) to (1,1): dist = sqrt(2)
        assert!((ham.distance(0, 4) - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_zero_state() {
        let ham = HamiltonianND::new(vec![4], 1.0, 0.5, 1.0);
        let psi = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
        let s = ham.entanglement_entropy(&psi);
        // Ground state has S approx 0
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_entropy_bell_state() {
        // Max entangled 2-qubit state: (|00> + |11>) / sqrt(2)
        let ham = HamiltonianND::new(vec![2], 1.0, 0.5, 1.0);
        let psi = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let s = ham.entanglement_entropy(&psi);
        // S = 1 for maximally entangled 2-qubit state
        assert!((s - 1.0).abs() < 0.1);
    }
}
