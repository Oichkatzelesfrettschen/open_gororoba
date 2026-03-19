//! Quantum measurement formalism: Kraus operators, Lindblad master equation, 
//! and Wiseman-Milburn stochastic master equation.

use nalgebra::{Complex, SMatrix};
use rand_distr::{Normal, Distribution};
use rand::thread_rng;

pub type DensityMatrix1Q = SMatrix<Complex<f64>, 2, 2>;
pub type Matrix2x2 = SMatrix<Complex<f64>, 2, 2>;

/// Represents a set of Kraus operators describing a quantum channel
pub struct KrausChannel {
    pub operators: Vec<Matrix2x2>,
}

impl KrausChannel {
    pub fn apply(&self, rho: &DensityMatrix1Q) -> DensityMatrix1Q {
        let mut new_rho = DensityMatrix1Q::zeros();
        for k in &self.operators {
            let k_adj = k.map(|c| c.conj()).transpose();
            new_rho += k * rho * k_adj;
        }
        new_rho
    }
}

/// Computes the Lindblad dissipator D[L](rho) = L * rho * L^dagger - 0.5 * {L^dagger * L, rho}
pub fn lindblad_dissipator(l: &Matrix2x2, rho: &DensityMatrix1Q) -> DensityMatrix1Q {
    let l_adj = l.map(|c| c.conj()).transpose();
    let l_rho_l_adj = l * rho * l_adj;
    let l_adj_l = l_adj * l;
    let anticomm = l_adj_l * rho + rho * l_adj_l;
    
    // D[L](rho)
    l_rho_l_adj - anticomm.map(|c| c * 0.5)
}

/// Applies a small stroboscopic evolution step dt using Lindblad master equation
pub fn lindblad_evolve(
    rho: &DensityMatrix1Q,
    h: &Matrix2x2,
    jump_ops: &[(f64, Matrix2x2)],
    dt: f64,
) -> DensityMatrix1Q {
    let i = Complex::new(0.0, 1.0);
    // Commutator [H, rho]
    let comm = h * rho - rho * h;
    let mut drho = comm.map(|c| -i * c);
    
    for (gamma, l) in jump_ops {
        let d = lindblad_dissipator(l, rho);
        drho += d.map(|c| c * *gamma);
    }
    
    rho + drho.map(|c| c * dt)
}

/// Wiseman-Milburn stochastic master equation step for continuous homodyne detection
/// d rho = dt(-i[H, rho] + D[c]rho) + sqrt(eta) dW H[c]rho
pub fn wiseman_milburn_step(
    rho: &DensityMatrix1Q,
    h: &Matrix2x2,
    c_op: &Matrix2x2,
    eta: f64,
    dt: f64,
) -> DensityMatrix1Q {
    // Deterministic part
    let i = Complex::new(0.0, 1.0);
    let comm = h * rho - rho * h;
    let det_drho = comm.map(|c| -i * c) + lindblad_dissipator(c_op, rho);
    
    // Stochastic part (H_meas[c] rho = c rho + rho c^dag - Tr(c rho + rho c^dag) rho)
    let c_adj = c_op.map(|c| c.conj()).transpose();
    let c_rho_rho_c_adj = c_op * rho + rho * c_adj;
    let expected_val = c_rho_rho_c_adj.trace().re; // Should be real
    
    let meas_drho = c_rho_rho_c_adj - rho.map(|x| x * expected_val);
    
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, dt.sqrt()).unwrap();
    let dw = normal.sample(&mut rng);
    
    let stoch_drho = meas_drho.map(|x| x * (eta.sqrt() * dw));
    
    rho + det_drho.map(|x| x * dt) + stoch_drho
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lindblad_dissipator() {
        let sigma_minus = Matrix2x2::new(
            Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
        );
        let rho = Matrix2x2::new(
            Complex::new(0.5, 0.0), Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new(0.5, 0.0),
        );
        let d = lindblad_dissipator(&sigma_minus, &rho);
        // Spontaneous emission: population moves from excited (|1><1|) to ground (|0><0|)
        // rho_11 goes from 0.5 to 0.0 (rate -0.5), rho_00 goes from 0.5 to 1.0 (rate +0.5)
        assert!((d[(0, 0)].re - 0.5).abs() < 1e-6);
        assert!((d[(1, 1)].re - -0.5).abs() < 1e-6);
    }
}
