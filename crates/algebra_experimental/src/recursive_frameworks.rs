//! Recursive Frameworks for Self-Similar Algebraic Patterns.
//!
//! Models Pattern 4 from the Unified Synthesis: Recursive Self-Similarity.
//! Similar patterns (like property loss and doubling) repeat across scales.

/// A recursive property metric that scales with the Cayley-Dickson level.
pub struct RecursivePropertyMetric {
    pub level: usize, // CD level (0=R, 1=C, 2=H, 3=O, 4=S, ...)
}

impl RecursivePropertyMetric {
    pub fn new(level: usize) -> Self {
        Self { level }
    }

    /// Dimensionality of the level: 2^level.
    pub fn dimension(&self) -> usize {
        1 << self.level
    }

    /// Measures the "fractal complexity" of the algebra level.
    /// This is a heuristic metric representing the recursive doubling density.
    pub fn fractal_complexity(&self) -> f64 {
        if self.level == 0 {
            1.0
        } else {
            // Complexity grows as log2(dim) * doubling_factor
            (self.level as f64) * 2.0_f64.powi(self.level as i32 - 3).max(1.0)
        }
    }

    /// Heuristic measure of "symmetry retention" which decreases recursively.
    pub fn symmetry_retention(&self) -> f64 {
        if self.level <= 2 {
            1.0 // R, C, H have high symmetry retention
        } else {
            // Recursive decay factor
            0.5_f64.powi(self.level as i32 - 2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_metrics() {
        let octonions = RecursivePropertyMetric::new(3);
        assert_eq!(octonions.dimension(), 8);
        assert!(octonions.symmetry_retention() > 0.4);

        let sedenions = RecursivePropertyMetric::new(4);
        assert_eq!(sedenions.dimension(), 16);
        assert!(sedenions.symmetry_retention() < octonions.symmetry_retention());
    }
}
