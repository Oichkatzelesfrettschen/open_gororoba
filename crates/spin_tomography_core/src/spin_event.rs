use nalgebra::Vector3;

/// A single spin analyzer event, consisting of two analyzer directions
/// and their corresponding weak decay analyzing powers.
#[derive(Debug, Clone)]
pub struct SpinEvent {
    /// Analyzer direction for particle 1 (e.g., Lambda) in its rest frame
    pub n1: Vector3<f64>,
    /// Analyzer direction for particle 2 (e.g., Anti-Lambda) in its rest frame
    pub n2: Vector3<f64>,
    /// Analyzing power for particle 1
    pub alpha1: f64,
    /// Analyzing power for particle 2
    pub alpha2: f64,
    /// Optional weight for acceptance correction (default 1.0)
    pub weight: f64,
}

impl SpinEvent {
    pub fn new(n1: Vector3<f64>, n2: Vector3<f64>, alpha1: f64, alpha2: f64) -> Self {
        Self {
            n1,
            n2,
            alpha1,
            alpha2,
            weight: 1.0,
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_and_defaults() {
        let n1 = Vector3::new(0.0, 0.0, 1.0);
        let n2 = Vector3::new(1.0, 0.0, 0.0);
        let ev = SpinEvent::new(n1, n2, 0.642, 0.732);
        assert_eq!(ev.alpha1, 0.642);
        assert_eq!(ev.alpha2, 0.732);
        assert_eq!(ev.weight, 1.0);
    }

    #[test]
    fn test_with_weight() {
        let n1 = Vector3::z();
        let n2 = Vector3::x();
        let ev = SpinEvent::new(n1, n2, 1.0, 1.0).with_weight(2.5);
        assert_eq!(ev.weight, 2.5);
    }
}
