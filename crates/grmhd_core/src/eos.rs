//! Ideal gas equation of state (gamma-law).
//!
//! p = (gamma - 1) * u_internal
//! where u_internal is internal energy density.
//!
//! Adiabatic index gamma = 5/3 for non-relativistic, 4/3 for ultra-relativistic.

/// Gamma-law equation of state.
#[derive(Clone, Debug)]
pub struct GammaLaw {
    pub gamma: f64,
}

impl GammaLaw {
    pub fn new(gamma: f64) -> Self {
        assert!(gamma > 1.0 && gamma <= 2.0, "gamma must be in (1, 2], got {}", gamma);
        Self { gamma }
    }

    /// Standard non-relativistic monatomic gas.
    pub fn ideal_mono() -> Self { Self::new(5.0 / 3.0) }

    /// Ultra-relativistic gas (radiation-dominated).
    pub fn relativistic() -> Self { Self::new(4.0 / 3.0) }

    /// Standard GRMHD choice (Gammie+ 2003 default).
    pub fn harm_default() -> Self { Self::new(13.0 / 9.0) }

    /// Pressure from internal energy density.
    #[inline]
    pub fn pressure(&self, u_int: f64) -> f64 {
        (self.gamma - 1.0) * u_int
    }

    /// Sound speed squared: cs^2 = gamma * p / (rho + u + p) = gamma * (gamma-1) * u / (rho + gamma * u)
    #[inline]
    pub fn cs2(&self, rho: f64, u_int: f64) -> f64 {
        let p = self.pressure(u_int);
        let h_rho = rho + u_int + p; // enthalpy density
        if h_rho > 1e-30 {
            self.gamma * p / h_rho
        } else {
            0.0
        }
    }

    /// Specific enthalpy: h = 1 + u/rho + p/rho = 1 + gamma * u / (rho * (gamma - 1)) ... wait
    /// Actually h = (rho + u + p) / rho for relativistic fluid
    #[inline]
    pub fn enthalpy(&self, rho: f64, u_int: f64) -> f64 {
        let p = self.pressure(u_int);
        (rho + u_int + p) / rho.max(1e-30)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure() {
        let eos = GammaLaw::ideal_mono();
        // p = (5/3 - 1) * u = 2/3 * u
        assert!((eos.pressure(3.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sound_speed_nonrelativistic() {
        let eos = GammaLaw::ideal_mono();
        // In non-relativistic limit: cs^2 ~ gamma * p / rho
        let rho = 1.0;
        let u = 0.01; // cold gas
        let cs2 = eos.cs2(rho, u);
        let p = eos.pressure(u);
        let cs2_nr = eos.gamma * p / rho;
        assert!((cs2 - cs2_nr).abs() / cs2_nr < 0.02, "Non-relativistic cs2: {} vs {}", cs2, cs2_nr);
    }
}
