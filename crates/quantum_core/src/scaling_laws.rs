//! Quantum scaling laws and telecom network correlation limits.
//!
//! Implements models for quantum error propagation, multi-node network scaling,
//! and resource efficiency limits derived from synchronization overhead analyses.

/// Computes the dimensional scaling of error propagation.
///
/// P_error(d) = 1 - (1 - p_0)^(d - 1)
/// where d is the system dimension and p_0 is the base error rate.
pub fn quantum_error_propagation(d: usize, p0: f64) -> f64 {
    if d == 0 {
        return 0.0;
    }
    1.0 - (1.0 - p0).powi((d - 1) as i32)
}

/// Calculate network topology scaling factor alpha(n).
///
/// alpha(n) = n^p * sum_k (c_k * phi_k(G))
/// where phi_k(G) are graph-theoretic invariants and c_k are weights.
pub fn network_topology_scaling(n: usize, p: f64, invariants: &[f64], weights: &[f64]) -> f64 {
    let n_pow = (n as f64).powf(p);
    let mut sum = 0.0;
    for (inv, w) in invariants.iter().zip(weights.iter()) {
        sum += inv * w;
    }
    n_pow * sum
}

/// Fidelity scaling law with explicit loss mechanisms.
///
/// F(d) = F_0 * exp(- sum(lambda_i(d)))
/// where lambda_i represents different loss channels (spectral crosstalk,
/// two-photon absorption, free-carrier effects, etc.).
pub fn fidelity_scaling(f0: f64, loss_channels: &[f64]) -> f64 {
    let sum_lambda: f64 = loss_channels.iter().sum();
    f0 * (-sum_lambda).exp()
}

/// Resource efficiency factor eta_resource(d, n).
///
/// Captures energy overhead, memory requirements, and classical processing limits.
/// eta = exp(-gamma * E(d, n))
pub fn resource_efficiency(energy_cost: f64, gamma: f64) -> f64 {
    (-gamma * energy_cost).exp()
}

/// Overall Net Advantage metric for high-dimensional quantum synchronization.
///
/// Net Advantage = (log2(d) / N_corr(d, n)) * F(d) * eta_resource(d, n)
pub fn net_advantage(d: usize, n_corr: f64, fidelity: f64, eta_resource: f64) -> f64 {
    if n_corr <= 0.0 || d == 0 {
        return 0.0;
    }
    let log_d = (d as f64).log2();
    (log_d / n_corr) * fidelity * eta_resource
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_prop() {
        let err = quantum_error_propagation(3, 0.1);
        // 1 - (0.9)^2 = 1 - 0.81 = 0.19
        assert!((err - 0.19).abs() < 1e-6);
    }

    #[test]
    fn test_fidelity() {
        let f = fidelity_scaling(1.0, &[0.1, 0.2]);
        // exp(-0.3)
        assert!((f - (-0.3_f64).exp()).abs() < 1e-6);
    }
}
