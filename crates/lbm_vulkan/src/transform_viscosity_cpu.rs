//! CPU reference for the Besag-Clifford `transform_viscosity` kernel.
//!
//! WHY: Pointwise transform of an imbalance field `phi` into LBM tau-buffer
//! values, used by the permutation-test pipeline in
//! `lbm_vulkan/shaders/besag_clifford.wgsl::transform_viscosity`. The CPU
//! reference is the parity oracle for the Vulkan and cubecl paths.
//!
//! WHAT: For each cell `i`:
//!
//!   nu_i  = nu_base * exp(lambda * (phi_i - phi_mean))
//!   tau_i = 3 * nu_i + 0.5
//!
//! HOW: Pure-Rust f32 evaluation, single-threaded for determinism. The
//! returned vector matches the WGSL kernel's element-wise output exactly
//! (modulo the standard float-evaluation order convention).

/// Inputs to the viscosity transform.
#[derive(Debug, Clone)]
pub struct ViscosityTransformInputs {
    /// Per-cell imbalance field (typically from an LBM rho-shuffle output).
    pub phi: Vec<f32>,
    /// Base kinematic viscosity (`nu_base`) before modulation.
    pub nu_base: f32,
    /// Coupling strength of the imbalance-to-viscosity transform.
    pub lambda: f32,
    /// Mean of `phi` used as the linearization point.
    pub phi_mean: f32,
}

/// Pointwise viscosity transform CPU reference.
/// Returns the per-cell `tau = 3 * nu_base * exp(lambda * (phi - phi_mean)) + 0.5`.
pub fn transform_viscosity_cpu(inputs: &ViscosityTransformInputs) -> Vec<f32> {
    inputs
        .phi
        .iter()
        .map(|&phi_i| {
            let nu_i = inputs.nu_base * (inputs.lambda * (phi_i - inputs.phi_mean)).exp();
            3.0 * nu_i + 0.5
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At lambda=0, every cell collapses to `tau = 3*nu_base + 0.5`.
    #[test]
    fn lambda_zero_gives_constant_tau() {
        let inputs = ViscosityTransformInputs {
            phi: vec![0.1, 0.5, 1.0, -0.3],
            nu_base: 0.01,
            lambda: 0.0,
            phi_mean: 0.4,
        };
        let tau = transform_viscosity_cpu(&inputs);
        let expected = 3.0 * 0.01 + 0.5;
        for &t in &tau {
            assert!((t - expected).abs() < 1e-6, "tau {} != {}", t, expected);
        }
    }

    /// At phi == phi_mean, exp(0)=1 so tau = 3*nu_base + 0.5 regardless of lambda.
    #[test]
    fn phi_equals_mean_gives_baseline_tau() {
        let inputs = ViscosityTransformInputs {
            phi: vec![0.4],
            nu_base: 0.005,
            lambda: 2.5,
            phi_mean: 0.4,
        };
        let tau = transform_viscosity_cpu(&inputs);
        let expected = 3.0 * 0.005 + 0.5;
        assert!((tau[0] - expected).abs() < 1e-6);
    }

    /// Positive lambda makes phi > mean cells more viscous.
    #[test]
    fn positive_lambda_increases_tau_for_above_mean_cells() {
        let inputs = ViscosityTransformInputs {
            phi: vec![0.2, 0.4, 0.6],
            nu_base: 0.01,
            lambda: 1.0,
            phi_mean: 0.4,
        };
        let tau = transform_viscosity_cpu(&inputs);
        assert!(tau[0] < tau[1]);
        assert!(tau[1] < tau[2]);
    }

    /// Empty input gives empty output (degenerate case).
    #[test]
    fn empty_phi_gives_empty_tau() {
        let inputs = ViscosityTransformInputs {
            phi: vec![],
            nu_base: 0.01,
            lambda: 1.0,
            phi_mean: 0.0,
        };
        let tau = transform_viscosity_cpu(&inputs);
        assert!(tau.is_empty());
    }
}
