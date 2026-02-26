//! Topological frustration layer.

use crate::traits::TopologyLayer;

/// Sliding-window frustration estimator from sign streams.
#[derive(Debug, Clone, Copy)]
pub struct SlidingTriadTopology {
    pub window: usize,
}

impl Default for SlidingTriadTopology {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl TopologyLayer for SlidingTriadTopology {
    fn frustration_density(&self, signs: &[i32]) -> Vec<f64> {
        if signs.is_empty() {
            return Vec::new();
        }
        let w = self.window.max(3);
        let mut out = Vec::with_capacity(signs.len());
        for i in 0..signs.len() {
            let start = i.saturating_sub(w - 1);
            let slice = &signs[start..=i];
            let mut unbalanced = 0usize;
            let mut total = 0usize;
            for a in 0..slice.len() {
                for b in (a + 1)..slice.len() {
                    for c in (b + 1)..slice.len() {
                        total += 1;
                        let prod = slice[a] * slice[b] * slice[c];
                        if prod < 0 {
                            unbalanced += 1;
                        }
                    }
                }
            }
            let f = if total == 0 {
                0.375
            } else {
                unbalanced as f64 / total as f64
            };
            out.push(f);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::TopologyLayer;

    #[test]
    fn test_topology_emits_density_series() {
        let topo = SlidingTriadTopology::default();
        let f = topo.frustration_density(&[1, -1, 1, -1, 1]);
        assert_eq!(f.len(), 5);
        assert!(f.iter().all(|x| (0.0..=1.0).contains(x)));
    }
}
