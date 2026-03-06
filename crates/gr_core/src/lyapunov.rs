//! Lyapunov exponent computation via Benettin's algorithm (1980).
//!
//! Computes the maximal Lyapunov exponent (MLE) for a dynamical system
//! by tracking the divergence of a nearby trajectory and periodically
//! renormalizing the separation vector.

/// State of the Lyapunov exponent computation.
pub struct LyapunovState {
    /// Accumulated sum of log(stretch) values.
    log_sum: f64,
    /// Number of renormalization steps performed.
    n_steps: usize,
    /// Current estimate of lambda_max.
    pub lambda_max: f64,
}

impl LyapunovState {
    pub fn new() -> Self {
        Self {
            log_sum: 0.0,
            n_steps: 0,
            lambda_max: 0.0,
        }
    }

    /// Record a renormalization event.
    ///
    /// `stretch` is the ratio |delta(t)| / |delta(0)| of the separation
    /// vector length before renormalization. `dt_renorm` is the time
    /// between renormalizations.
    pub fn record(&mut self, stretch: f64, dt_renorm: f64) {
        if stretch > 0.0 && dt_renorm > 0.0 {
            self.log_sum += stretch.ln();
            self.n_steps += 1;
            self.lambda_max = self.log_sum / (self.n_steps as f64 * dt_renorm);
        }
    }

    /// Current estimate of the maximal Lyapunov exponent.
    pub fn estimate(&self) -> f64 {
        self.lambda_max
    }

    /// Number of renormalization steps so far.
    pub fn steps(&self) -> usize {
        self.n_steps
    }
}

impl Default for LyapunovState {
    fn default() -> Self {
        Self::new()
    }
}

/// Classification of dynamical regime from Lyapunov exponent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicalRegime {
    /// lambda_max <= 0: trajectories converge or are marginally stable.
    Regular,
    /// 0 < lambda_max < threshold: weak sensitivity to initial conditions.
    WeaklyChaotic,
    /// lambda_max >= threshold: strong chaos.
    StronglyChaotic,
}

impl std::fmt::Display for DynamicalRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Regular => write!(f, "Regular"),
            Self::WeaklyChaotic => write!(f, "WeaklyChaotic"),
            Self::StronglyChaotic => write!(f, "StronglyChaotic"),
        }
    }
}

/// Classify the dynamical regime from a Lyapunov exponent.
///
/// `threshold` is the boundary between weak and strong chaos
/// (typically ~0.01 in natural units for orbital dynamics).
pub fn classify_regime(lambda_max: f64, threshold: f64) -> DynamicalRegime {
    if lambda_max <= 0.0 {
        DynamicalRegime::Regular
    } else if lambda_max < threshold {
        DynamicalRegime::WeaklyChaotic
    } else {
        DynamicalRegime::StronglyChaotic
    }
}

/// Compute the Kaplan-Yorke dimension from a sorted Lyapunov spectrum.
///
/// D_KY = j + sum_{i=1}^{j} lambda_i / |lambda_{j+1}|
/// where j is the largest index such that sum_{i=1}^{j} lambda_i >= 0.
pub fn kaplan_yorke_dimension(spectrum: &[f64]) -> f64 {
    let mut sorted = spectrum.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    for (j, &lam) in sorted.iter().enumerate() {
        cumsum += lam;
        if cumsum < 0.0 {
            if lam.abs() > 1e-15 {
                return j as f64 + (cumsum - lam) / lam.abs();
            }
            return j as f64;
        }
    }
    sorted.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lyapunov_basic() {
        let mut state = LyapunovState::new();
        // Constant stretch of e per unit time -> lambda = 1
        for _ in 0..100 {
            state.record(std::f64::consts::E, 1.0);
        }
        assert!((state.estimate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn classify_regimes() {
        assert_eq!(classify_regime(-0.1, 0.01), DynamicalRegime::Regular);
        assert_eq!(classify_regime(0.0, 0.01), DynamicalRegime::Regular);
        assert_eq!(classify_regime(0.005, 0.01), DynamicalRegime::WeaklyChaotic);
        assert_eq!(classify_regime(0.1, 0.01), DynamicalRegime::StronglyChaotic);
    }

    #[test]
    fn kaplan_yorke_ordered() {
        // Lorenz attractor-like spectrum: [0.9, 0, -14.57]
        let spectrum = [0.9, 0.0, -14.57];
        let d_ky = kaplan_yorke_dimension(&spectrum);
        // D_KY = 2 + 0.9 / 14.57 ~ 2.062
        assert!((d_ky - 2.062).abs() < 0.01);
    }
}
