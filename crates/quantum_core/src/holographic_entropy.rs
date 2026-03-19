//! Holographic Entropy Scaling in MERA-like Tensor Networks.
//!
//! Investigates entanglement entropy S(L) as a function of subsystem size L
//! for standard CFT (AdS/CFT) and modified Sedenion-inspired bulk geometries
//! with non-associative "wormhole" connectivity.
//!
//! Migrated from src/holo_tensor_net.py.

/// Result of a holographic entropy scaling analysis.
#[derive(Debug, Clone)]
pub struct EntropyScaling {
    pub subsystem_size: f64,
    pub entropy_cft: f64,
    pub entropy_sedenion: f64,
}

/// Compute entropy scaling for a range of subsystem sizes.
pub fn compute_holographic_scaling(max_size: usize, steps: usize) -> Vec<EntropyScaling> {
    let mut results = Vec::with_capacity(steps);
    
    // Subsystem sizes L from 2 to max_size/2, log-spaced
    let start = (2.0_f64).ln();
    let end = (max_size as f64 / 2.0).ln();
    let delta = (end - start) / (steps - 1) as f64;

    for i in 0..steps {
        let l = (start + i as f64 * delta).exp();
        
        // 1. Standard Holography (CFT): S ~ (c/3) * log2(L)
        // Using central charge c = 1 for simplicity.
        let s_cft = (1.0 / 3.0) * l.log2();

        // 2. Sedenion Bulk (Non-Associative): S ~ log(L) + L^beta
        // Non-associativity introduces non-local connectivity ("wormholes"),
        // leading to super-logarithmic growth.
        // beta = 0.5 based on the "Mass Ladder" hypothesis (M ~ n^1.5).
        let s_sedenion = 0.5 * l.log2() + 0.05 * l.powf(0.5);

        results.push(EntropyScaling {
            subsystem_size: l,
            entropy_cft: s_cft,
            entropy_sedenion: s_sedenion,
        });
    }

    results
}

/// Estimate the central charge `c` from an entropy-length dataset.
///
/// Assumes S = (c/3) * log2(L).
pub fn estimate_central_charge(data: &[EntropyScaling]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let sum_x: f64 = data.iter().map(|d| d.subsystem_size.log2()).sum();
    let sum_y: f64 = data.iter().map(|d| d.entropy_cft).sum();
    let sum_xx: f64 = data.iter().map(|d| d.subsystem_size.log2().powi(2)).sum();
    let sum_xy: f64 = data.iter().map(|d| d.subsystem_size.log2() * d.entropy_cft).sum();
    
    let n = data.len() as f64;
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    slope * 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_monotone() {
        let scaling = compute_holographic_scaling(1024, 10);
        for i in 1..scaling.len() {
            assert!(scaling[i].entropy_cft > scaling[i-1].entropy_cft);
            assert!(scaling[i].entropy_sedenion > scaling[i-1].entropy_sedenion);
        }
    }

    #[test]
    fn test_central_charge_recovery() {
        // Generate pure log data with c=1.0
        let mut data = Vec::new();
        for i in 1..10 {
            let l = (i as f64) * 10.0;
            data.push(EntropyScaling {
                subsystem_size: l,
                entropy_cft: (1.0 / 3.0) * l.log2(),
                entropy_sedenion: 0.0,
            });
        }
        let c = estimate_central_charge(&data);
        assert!((c - 1.0).abs() < 1e-10);
    }
}
