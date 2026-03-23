//! Cosmological Mass Ladder Predictor
//!
//! Uses Cayley-Dickson tower doubling ratios to predict dark matter and axion 
//! mass thresholds without relying on continuous standard model parameters.

/// **Mass Scale Predictor**
/// Predicts the next logical mass step in a cosmological hierarchy
/// by assuming mass scales inversely with the algebraic dimension of the
/// symmetry group broken at that scale (e.g., 4, 8, 16, 32).
pub fn predict_next_mass_scale(current_mass_ev: f64, current_cd_dim: usize, target_cd_dim: usize) -> f64 {
    // A toy heuristic: Mass drops logarithmically as dimensions double
    let ratio = (target_cd_dim as f64) / (current_cd_dim as f64);
    current_mass_ev / ratio.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mass_predictor() {
        let electron_mass = 511_000.0; // eV
        // Predict from 8D (Octonions) to 16D (Sedenions)
        let neutrino_scale = predict_next_mass_scale(electron_mass, 8, 16);
        assert!(neutrino_scale < electron_mass);
    }
}
