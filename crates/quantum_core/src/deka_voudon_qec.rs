use algebra_experimental::higher_cd::HigherAvt;
use std::collections::HashSet;

/// A generalized holographic stabilizer for the 1024D DekaVoudon manifold.
/// 
/// Unlike the 6-node Sedenion box-kite, the DekaVoudon stabilizer is a 
/// high-dimensional "Kite-Chain Midden" consisting of recursive ZD paths.
#[derive(Debug, Clone)]
pub struct DekaVoudonStabilizer {
    pub level: usize,
    pub nodes: Vec<usize>,
    pub parity_paths: Vec<Vec<usize>>,
}

impl DekaVoudonStabilizer {
    /// Construct a new stabilizer from a recursive Kite-Chain.
    pub fn new(level: usize, initial_nodes: &[usize], avt: &HigherAvt) -> Self {
        let mut nodes = initial_nodes.to_vec();
        let mut parity_paths = Vec::new();
        
        // Recursively build the parity-check matrix using ZD paths
        // At level n, the stabilizer checks correlations across 2^n units.
        for level_idx in 0..level {
            let mut new_path = Vec::new();
            for &u in &nodes {
                // Find associated zero-divisors in the DekaVoudon manifold
                for &(i, j, _k, m, _sign) in &avt.violations {
                    if i == u || j == u {
                        new_path.push(m);
                        if new_path.len() > 2_usize.pow(level_idx as u32) { break; }
                    }
                }
            }
            nodes.extend(new_path.iter().cloned());
            parity_paths.push(new_path);
        }
        
        Self { level, nodes, parity_paths }
    }

    /// Detect syndromes (non-associative noise) across the holographic layer.
    pub fn detect_syndrome(&self, active_violations: &HashSet<usize>) -> f64 {
        let mut triggered = 0;
        for path in &self.parity_paths {
            for &node in path {
                if active_violations.contains(&node) {
                    triggered += 1;
                }
            }
        }
        
        // The syndrome strength is the normalized violation density
        if self.nodes.is_empty() { 0.0 } else { triggered as f64 / self.nodes.len() as f64 }
    }
}

/// The Macroscopic Stabilizer Code representing the Vacuum Parity-Check Matrix.
pub struct HolographicVacuumCode {
    pub stabilizers: Vec<DekaVoudonStabilizer>,
    pub dimension: usize,
}

impl HolographicVacuumCode {
    pub fn new(dim: usize, avt: &HigherAvt) -> Self {
        let mut stabilizers = Vec::new();
        // Seed the code with the 7 primary box-kites, then expand
        for i in 1..=7 {
            stabilizers.push(DekaVoudonStabilizer::new(4, &[i, i+8, i+16], avt));
        }
        
        Self { stabilizers, dimension: dim }
    }
    
    /// Returns the global vacuum stability metric.
    pub fn evaluate_stability(&self, active_violations: &HashSet<usize>) -> f64 {
        let sum: f64 = self.stabilizers.iter()
            .map(|s| s.detect_syndrome(active_violations))
            .sum();
        
        1.0 - (sum / self.stabilizers.len() as f64)
    }
}
