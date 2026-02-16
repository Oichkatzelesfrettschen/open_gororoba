use vacuum_frustration::VACUUM_ATTRACTOR;

/// Maps CD field observables (frustration density, associator norms) to quantum channel parameters.
#[derive(Debug, Clone)]
pub struct DecoherenceMap {
    /// Scaling for frustration term
    pub c_f: f64,
    /// Scaling for associator term
    pub c_a: f64,
    /// Base gamma
    pub gamma_0: f64,
    /// Temperature for lambda computation (if used)
    pub temperature: f64,
}

impl Default for DecoherenceMap {
    fn default() -> Self {
        Self {
            c_f: 10.0,
            c_a: 1.0,
            gamma_0: 0.0,
            temperature: 1.0,
        }
    }
}

impl DecoherenceMap {
    pub fn new(c_f: f64, c_a: f64, gamma_0: f64, temperature: f64) -> Self {
        Self { c_f, c_a, gamma_0, temperature }
    }

    /// Computes gamma from local frustration density and associator norm.
    /// gamma = gamma_0 + c_f * (F - 3/8)^2 + c_a * A
    pub fn gamma_from_frustration(&self, frustration: f64, associator_norm: f64) -> f64 {
        let f_diff = frustration - VACUUM_ATTRACTOR;
        // Using squared difference as deviation from vacuum attractor implies higher energy cost/decoherence?
        // Or maybe just linear?
        // The plan suggests: gamma = gamma_0 + c_F * (F - 3/8)^2 + c_A * A
        
        let term_f = self.c_f * f_diff * f_diff;
        let term_a = self.c_a * associator_norm;
        
        self.gamma_0 + term_f + term_a
    }

    /// Computes gamma from lambda (coupling strength).
    /// gamma = kappa * lambda
    pub fn gamma_from_lambda(&self, lambda: f64, kappa: f64) -> f64 {
        kappa * lambda
    }
}
