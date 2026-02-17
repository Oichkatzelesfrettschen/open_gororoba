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

/// Bridge between QGP physical parameters and Cayley-Dickson frustration metrics.
pub struct QGPFrustrationBridge {
    /// Scaling for vorticity to algebraic twist
    pub k_vorticity: f64,
    /// Scaling for temperature to attractor proximity
    pub k_temp: f64,
}

impl Default for QGPFrustrationBridge {
    fn default() -> Self {
        Self {
            k_vorticity: 0.1,
            k_temp: 1.0,
        }
    }
}

impl QGPFrustrationBridge {
    /// Maps QGP state to CD frustration density and associator norm proxies.
    ///
    /// Hypothesis:
    /// - High temperature drives frustration towards the 3/8 attractor (chiral restoration).
    /// - Vorticity induces a preferred axis in the spin-space, modeled as a biased channel.
    pub fn predict_frustration(&self, state: &QGPState) -> (f64, f64) {
        // As temperature increases, frustration density F converges to 3/8
        // We model the residual |F - 3/8| as decaying with temperature
        let f_residual = 0.1 * (-self.k_temp * state.temperature).exp();
        let frustration = 0.375 + f_residual;

        // Associator norm reflects energy density
        let associator = 0.5 * state.energy_density;

        (frustration, associator)
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
            bias_term += kron(&s_i, &eye2) * comp;
            // System 2 bias
            bias_term += kron(&eye2, &s_i) * comp;
        }

        // Add bias and re-normalize trace to 1
        rho += bias_term;
        let tr = rho.trace();
        rho /= tr;

        TwoQubitState::new(rho)
    }
}

// Helper for Kronecker product (duplicate from state.rs, should be moved to a shared util)
fn kron(a: &nalgebra::Matrix2<Complex64>, b: &nalgebra::Matrix2<Complex64>) -> Matrix4<Complex64> {
    let mut m = Matrix4::zeros();
    for i in 0..2 {
        for j in 0..2 {
            let val_a = a[(i, j)];
            for k in 0..2 {
                for l in 0..2 {
                    let val_b = b[(k, l)];
                    m[(2 * i + k, 2 * j + l)] = val_a * val_b;
                }
            }
        }
    }
    m
}
