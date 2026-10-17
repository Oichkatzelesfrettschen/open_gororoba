//! Quantum measurement formalism: Kraus operators, Lindblad master equation,
//! and Wiseman-Milburn stochastic master equation.

use nalgebra::{Complex, SMatrix};
use rand_distr::{Distribution, Normal};

pub type DensityMatrix1Q = SMatrix<Complex<f64>, 2, 2>;
pub type Matrix2x2 = SMatrix<Complex<f64>, 2, 2>;

/// Represents a set of Kraus operators describing a quantum channel
pub struct KrausChannel {
    pub operators: Vec<Matrix2x2>,
}

impl KrausChannel {
    /// Average a zero-mean classical Gaussian phase with the supplied variance.
    ///
    /// Populations stay fixed and off-diagonal entries acquire exp(-variance/2).
    /// The Kraus pair is sqrt((1+factor)/2) I and sqrt((1-factor)/2) Z.
    /// Population relaxation and its T1 contribution require a separate channel.
    /// The classical phase model supplies neither a quantum-vacuum derivation
    /// nor a unitary bath realization. Large finite variance may round factor to zero.
    pub fn gaussian_phase_noise(phase_variance: f64) -> Option<Self> {
        if !phase_variance.is_finite() || phase_variance < 0.0 {
            return None;
        }
        let factor = (-phase_variance / 2.0).exp();
        let identity_amplitude = ((1.0 + factor) / 2.0).sqrt();
        let phase_amplitude = if phase_variance < f64::MIN_POSITIVE {
            // Taking sqrt before division preserves subnormal variances. The
            // relative Taylor correction is O(variance), below f64 roundoff.
            phase_variance.sqrt() / 2.0
        } else {
            // exp_m1 preserves the phase probability when exp rounds to one.
            (-0.5 * (-phase_variance / 2.0).exp_m1()).sqrt()
        };
        let identity = Matrix2x2::identity().map(|entry| entry * identity_amplitude);
        let phase_flip = Matrix2x2::new(
            Complex::new(phase_amplitude, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-phase_amplitude, 0.0),
        );
        Some(Self {
            operators: vec![identity, phase_flip],
        })
    }

    pub fn apply(&self, rho: &DensityMatrix1Q) -> DensityMatrix1Q {
        let mut new_rho = DensityMatrix1Q::zeros();
        for k in &self.operators {
            let k_adj = k.map(|c| c.conj()).transpose();
            new_rho += k * rho * k_adj;
        }
        new_rho
    }
}

/// Computes the Lindblad dissipator `D[L](rho) = L * rho * L^dagger - 0.5 * {L^dagger * L, rho}`.
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
/// `d rho = dt(-i[H, rho] + D[c]rho) + sqrt(eta) dW H[c]rho`
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

    let mut rng = rand::rng();
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
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        );
        let rho = Matrix2x2::new(
            Complex::new(0.5, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.5, 0.0),
        );
        let d = lindblad_dissipator(&sigma_minus, &rho);
        // Spontaneous emission: population moves from excited (|1><1|) to ground (|0><0|)
        // rho_11 goes from 0.5 to 0.0 (rate -0.5), rho_00 goes from 0.5 to 1.0 (rate +0.5)
        assert!((d[(0, 0)].re - 0.5).abs() < 1e-6);
        assert!((d[(1, 1)].re - -0.5).abs() < 1e-6);
    }

    #[test]
    fn gaussian_phase_kraus_completeness_and_complex_coherence() {
        let variance = 1.4;
        let channel = KrausChannel::gaussian_phase_noise(variance).unwrap();
        let completeness = channel
            .operators
            .iter()
            .fold(Matrix2x2::zeros(), |sum, operator| {
                sum + operator.adjoint() * operator
            });
        assert!((completeness - Matrix2x2::identity()).norm() < 1e-14);
        let coherence = Complex::new(0.2, 0.1);
        let state = Matrix2x2::new(
            Complex::new(0.7, 0.0),
            coherence,
            coherence.conj(),
            Complex::new(0.3, 0.0),
        );
        let result = channel.apply(&state);
        assert!((result[(0, 0)] - state[(0, 0)]).norm() < 1e-14);
        assert!((result[(1, 1)] - state[(1, 1)]).norm() < 1e-14);
        assert!((result[(0, 1)] - coherence * (-variance / 2.0).exp()).norm() < 1e-14);
        assert!((result[(1, 0)] - result[(0, 1)].conj()).norm() < 1e-14);
    }

    #[test]
    fn gaussian_phase_identity_full_average_and_invalid_variances() {
        let state = Matrix2x2::from_element(Complex::new(0.5, 0.0));
        assert!(
            (KrausChannel::gaussian_phase_noise(0.0)
                .unwrap()
                .apply(&state)
                - state)
                .norm()
                < 1e-14
        );
        let averaged = KrausChannel::gaussian_phase_noise(f64::MAX)
            .unwrap()
            .apply(&state);
        assert!(averaged[(0, 1)].norm() < 1e-14);
        assert!((averaged[(0, 0)].re - 0.5).abs() < 1e-14);
        assert!((averaged[(1, 1)].re - 0.5).abs() < 1e-14);
        for variance in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(KrausChannel::gaussian_phase_noise(variance).is_none());
        }
    }

    #[test]
    fn gaussian_phase_normalized_choi_matches_bell_weights_and_is_psd() {
        for variance in [0.0, 1.4, f64::MAX] {
            let channel = KrausChannel::gaussian_phase_noise(variance).unwrap();
            let mut choi = SMatrix::<Complex<f64>, 4, 4>::zeros();
            for input_row in 0..2 {
                for input_column in 0..2 {
                    let mut basis = Matrix2x2::zeros();
                    basis[(input_row, input_column)] = Complex::new(1.0, 0.0);
                    let output = channel.apply(&basis);
                    for output_row in 0..2 {
                        for output_column in 0..2 {
                            choi[(2 * input_row + output_row, 2 * input_column + output_column)] =
                                output[(output_row, output_column)] * 0.5;
                        }
                    }
                }
            }
            let factor = (-variance / 2.0).exp();
            let mut expected = SMatrix::<Complex<f64>, 4, 4>::zeros();
            expected[(0, 0)] = Complex::new(0.5, 0.0);
            expected[(3, 3)] = Complex::new(0.5, 0.0);
            expected[(0, 3)] = Complex::new(factor / 2.0, 0.0);
            expected[(3, 0)] = Complex::new(factor / 2.0, 0.0);
            assert!((choi - expected).norm() < 1e-14);
            assert!((choi.trace() - Complex::new(1.0, 0.0)).norm() < 1e-14);
            assert!((choi - choi.adjoint()).norm() < 1e-14);
            let spectrum = nalgebra::SymmetricEigen::new(choi);
            assert!(
                spectrum
                    .eigenvalues
                    .iter()
                    .all(|eigenvalue| *eigenvalue >= -1e-14)
            );
        }
    }

    #[test]
    fn gaussian_phase_amplitude_preserves_tiny_positive_variance() {
        for variance in [1e-20_f64, f64::from_bits(1)] {
            let channel = KrausChannel::gaussian_phase_noise(variance).unwrap();
            let expected = variance.sqrt() / 2.0;
            let amplitude = channel.operators[1][(0, 0)].re;
            assert!(
                amplitude > 0.0,
                "phase amplitude vanished at variance={variance:e}"
            );
            assert!((amplitude / expected - 1.0).abs() < 4.0 * f64::EPSILON);
            assert_eq!(channel.operators[1][(1, 1)].re, -amplitude);
            let completeness = channel
                .operators
                .iter()
                .fold(Matrix2x2::zeros(), |sum, operator| {
                    sum + operator.adjoint() * operator
                });
            assert!((completeness - Matrix2x2::identity()).norm() < 1e-14);
        }
    }
}
