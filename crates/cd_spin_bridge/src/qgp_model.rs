use algebra_core::kron2;
use nalgebra::{Matrix4, Vector3};
use num_complex::Complex64;
use spin_tomography_core::TwoQubitState;

/// Represents the physical state of Quark-Gluon Plasma.
#[derive(Debug, Clone)]
pub struct QGPState {
    /// Temperature in GeV
    pub temperature: f64,
    /// Fluid vorticity (angular velocity)
    pub vorticity: Vector3<f64>,
    /// Energy density
    pub energy_density: f64,
}

/// Bridge between QGP physical parameters and Cayley-Dickson imbalance metrics.
pub struct QGPImbalanceBridge {
    /// Scaling for vorticity to algebraic twist
    pub k_vorticity: f64,
    /// Scaling for temperature to attractor proximity
    pub k_temp: f64,
}

impl Default for QGPImbalanceBridge {
    fn default() -> Self {
        Self {
            k_vorticity: 0.1,
            k_temp: 1.0,
        }
    }
}

impl QGPImbalanceBridge {
    /// Maps QGP state to CD imbalance density and associator norm proxies.
    ///
    /// Hypothesis:
    /// - High temperature drives imbalance towards the 3/8 attractor (chiral restoration).
    /// - Vorticity induces a preferred axis in the spin-space, modeled as a biased channel.
    pub fn predict_imbalance(&self, state: &QGPState) -> (f64, f64) {
        // As temperature increases, imbalance density F converges to 3/8
        // We model the residual |F - 3/8| as decaying with temperature
        let f_residual = 0.1 * (-self.k_temp * state.temperature).exp();
        let imbalance = 0.375 + f_residual;

        // Associator norm reflects energy density
        let associator = 0.5 * state.energy_density;

        (imbalance, associator)
    }

    /// Applies a QGP-driven decoherence channel to an initial spin state.
    /// This includes both isotropic depolarizing (from temperature)
    /// and biased alignment (from vorticity).
    pub fn apply_qgp_decoherence(&self, state: &TwoQubitState, qgp: &QGPState) -> TwoQubitState {
        // Isotropic part (noise)
        let gamma_iso = 0.5 * qgp.temperature * qgp.energy_density;
        let p_iso = 1.0 - (-gamma_iso).exp();

        // Biased part (vorticity alignment)
        // Alignment polarization P ~ (h_bar * omega) / (2 k_B T)
        let p_vort = if qgp.temperature > 0.0 {
            (qgp.vorticity.norm() * 0.5) / qgp.temperature
        } else {
            0.0
        };

        let mut rho = state.rho;

        // 1. Isotropic depolarizing
        let eye = Matrix4::<Complex64>::identity();
        rho = rho * Complex64::from(1.0 - p_iso) + eye * Complex64::from(0.25 * p_iso);

        // 2. Vorticity-induced polarization bias
        // For a single particle, rho -> 1/2 (I + P.sigma)
        // For a pair, we model it as a shift in the Bloch vector a or b.
        // Or directly modifying the density matrix.

        // Simplification: add a term proportional to omega . sigma x I + I x omega . sigma
        let (d1, d2, d3) = algebra_core::physics::clifford::pauli_matrices();
        let to_m2 =
            |m: algebra_core::physics::clifford::GammaMatrix| -> nalgebra::Matrix2<Complex64> {
                nalgebra::Matrix2::from_iterator(m.into_iter().cloned())
            };
        let sigmas = [to_m2(d1), to_m2(d2), to_m2(d3)];
        let eye2 = nalgebra::Matrix2::<Complex64>::identity();

        let omega_hat = if qgp.vorticity.norm() > 1e-12 {
            qgp.vorticity.normalize()
        } else {
            Vector3::zeros()
        };

        let mut bias_term = Matrix4::<Complex64>::zeros();
        for i in 0..3 {
            let s_i = sigmas[i];
            let comp = Complex64::from(omega_hat[i] * p_vort);

            // System 1 bias
            bias_term += kron2(&s_i, &eye2) * comp;
            // System 2 bias
            bias_term += kron2(&eye2, &s_i) * comp;
        }

        // Add bias and re-normalize trace to 1
        rho += bias_term;
        let tr = rho.trace();
        rho /= tr;

        TwoQubitState::new(rho)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    fn default_qgp_state() -> QGPState {
        QGPState {
            temperature: 0.2,      // 200 MeV in GeV
            vorticity: Vector3::new(0.0, 0.0, 0.01),
            energy_density: 1.0,
        }
    }

    fn identity_state() -> TwoQubitState {
        // Maximally mixed state I/4
        TwoQubitState::new(Matrix4::<Complex64>::identity() * Complex64::from(0.25))
    }

    fn bell_state() -> TwoQubitState {
        let a = Vector3::zeros();
        let b = Vector3::zeros();
        let mut t = Matrix3::<f64>::zeros();
        t[(0, 0)] = 1.0;
        t[(1, 1)] = -1.0;
        t[(2, 2)] = 1.0;
        TwoQubitState::from_ab_t(&a, &b, &t)
    }

    #[test]
    fn test_predict_imbalance_converges_to_attractor() {
        let bridge = QGPImbalanceBridge::default();
        // High temperature should drive imbalance towards 3/8
        let hot = QGPState {
            temperature: 100.0,
            vorticity: Vector3::zeros(),
            energy_density: 1.0,
        };
        let (f, _a) = bridge.predict_imbalance(&hot);
        assert!((f - 0.375).abs() < 1e-6,
            "High-T imbalance should converge to 3/8: got {f}");
    }

    #[test]
    fn test_predict_imbalance_low_temp() {
        let bridge = QGPImbalanceBridge::default();
        let cold = QGPState {
            temperature: 0.01,
            vorticity: Vector3::zeros(),
            energy_density: 0.5,
        };
        let (f, a) = bridge.predict_imbalance(&cold);
        // Low T: residual is 0.1 * exp(-0.01) ~ 0.099, so F ~ 0.474
        assert!(f > 0.375, "Low-T imbalance should exceed imbalance attractor");
        assert!((a - 0.25).abs() < 1e-14, "Associator = 0.5 * energy_density");
    }

    #[test]
    fn test_qgp_decoherence_trace_preservation() {
        let bridge = QGPImbalanceBridge::default();
        let qgp = default_qgp_state();
        let state = bell_state();
        let out = bridge.apply_qgp_decoherence(&state, &qgp);
        let tr = out.rho.trace().re;
        assert!((tr - 1.0).abs() < 1e-10,
            "QGP decoherence must preserve trace: got {tr}");
    }

    #[test]
    fn test_qgp_decoherence_zero_vorticity() {
        let bridge = QGPImbalanceBridge::default();
        let qgp = QGPState {
            temperature: 0.2,
            vorticity: Vector3::zeros(),
            energy_density: 1.0,
        };
        let state = identity_state();
        let out = bridge.apply_qgp_decoherence(&state, &qgp);
        // Zero vorticity: output should still have trace 1 and be Hermitian
        let tr = out.rho.trace().re;
        assert!((tr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kron_identity() {
        let eye = nalgebra::Matrix2::<Complex64>::identity();
        let result = kron2(&eye, &eye);
        let expected = Matrix4::<Complex64>::identity();
        for i in 0..4 {
            for j in 0..4 {
                let diff = (result[(i, j)] - expected[(i, j)]).norm();
                assert!(diff < 1e-14, "kron2(I,I) should be I_4: [{i},{j}] diff={diff}");
            }
        }
    }

    #[test]
    fn test_default_bridge() {
        let bridge = QGPImbalanceBridge::default();
        assert_eq!(bridge.k_vorticity, 0.1);
        assert_eq!(bridge.k_temp, 1.0);
    }
}
