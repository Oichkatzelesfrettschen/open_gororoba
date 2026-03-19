//! Multi-particle cumulant formulations for extracting flow harmonics ($v_n$).
//!
//! Provides mathematically rigorous implementations of 2- and 4-particle cumulants
//! $c_n\{2\}$ and $c_n\{4\}$, and their relation to anisotropic flow coefficients
//! $v_n\{2\}$ and $v_n\{4\}$. Used for analyzing nuclear deformation (e.g., Ne-Ne).

/// Computes the 2-particle cumulant $c_n\{2\}$ and $v_n\{2\}$.
pub struct TwoParticleCumulant {
    pub n: usize,
    pub c_n_2: f64,
}

impl TwoParticleCumulant {
    /// Create a new `TwoParticleCumulant` given the evaluated $c_n\{2\}$ value.
    /// $c_n\{2\} = \langle\!\langle e^{in(\phi_1-\phi_2)}\rangle\!\rangle$
    pub fn new(n: usize, c_n_2: f64) -> Self {
        Self { n, c_n_2 }
    }

    /// Extract flow harmonic $v_n\{2\} = \sqrt{c_n\{2\}}$.
    /// Returns None if $c_n\{2\} < 0$ (e.g. non-flow dominated or statistical fluctuation).
    pub fn v_n_2(&self) -> Option<f64> {
        if self.c_n_2 >= 0.0 {
            Some(self.c_n_2.sqrt())
        } else {
            None
        }
    }
}

/// Computes the 4-particle cumulant $c_n\{4\}$ and $v_n\{4\}$.
pub struct FourParticleCumulant {
    pub n: usize,
    pub c_n_4: f64,
}

impl FourParticleCumulant {
    /// Create a new `FourParticleCumulant` given the raw 4-particle correlator
    /// and the 2-particle cumulant.
    /// $c_n\{4\} = \langle\!\langle e^{in(\phi_1+\phi_2-\phi_3-\phi_4)}\rangle\!\rangle - 2\,c_n\{2\}^2$
    pub fn from_correlators(n: usize, corr_4: f64, c_n_2: f64) -> Self {
        let c_n_4 = corr_4 - 2.0 * c_n_2 * c_n_2;
        Self { n, c_n_4 }
    }

    /// Extract flow harmonic $v_n\{4\} = (-c_n\{4\})^{1/4}$.
    /// Returns None if $c_n\{4\} > 0$ (where the 4-particle extraction breaks down).
    pub fn v_n_4(&self) -> Option<f64> {
        if self.c_n_4 <= 0.0 {
            Some((-self.c_n_4).powf(0.25))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_2() {
        let c2 = TwoParticleCumulant::new(2, 0.04);
        assert!((c2.v_n_2().unwrap() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_v2_4() {
        let c4 = FourParticleCumulant { n: 2, c_n_4: -0.0016 };
        assert!((c4.v_n_4().unwrap() - 0.2).abs() < 1e-6);
    }
}
