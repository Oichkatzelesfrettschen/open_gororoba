//! TCMT Fano resonance scattering (Rocq-verified: C-913..C-914).
//!
//! Temporal coupled-mode theory for single-resonance scattering.
//! Kernel-checked in `proofs/theories/FanoResonance.v`.

/// Squared transmission coefficient |t|^2 for TCMT single resonance.
///
/// T = (delta^2 + gamma_i^2) / (delta^2 + (gamma_e + gamma_i)^2)
///
/// - `delta`: detuning (omega - omega_0)
/// - `gamma_e`: external coupling rate (>= 0)
/// - `gamma_i`: intrinsic loss rate (>= 0)
pub fn tcmt_transmission_coeff(delta: f64, gamma_e: f64, gamma_i: f64) -> f64 {
    let num = delta.powi(2) + gamma_i.powi(2);
    let den = delta.powi(2) + (gamma_e + gamma_i).powi(2);
    if den == 0.0 {
        1.0 // both rates zero and on-resonance: trivially transparent
    } else {
        num / den
    }
}

/// Lossless reflection coefficient.
///
/// R = gamma_e^2 / (delta^2 + gamma_e^2) for gamma_i = 0.
pub fn tcmt_reflection_coeff(delta: f64, gamma_e: f64) -> f64 {
    let den = delta.powi(2) + gamma_e.powi(2);
    if den == 0.0 {
        0.0
    } else {
        gamma_e.powi(2) / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn antiresonance() {
        // At resonance (delta=0) with no loss: T = 0
        let t = tcmt_transmission_coeff(0.0, 1.0, 0.0);
        assert!(t.abs() < TOL, "T at resonance should be 0, got {t}");
    }

    #[test]
    fn off_resonance() {
        // Far off resonance: T -> 1
        let t = tcmt_transmission_coeff(1e6, 1.0, 0.0);
        assert!(
            (t - 1.0).abs() < 1e-6,
            "T far off resonance should be ~1, got {t}"
        );
    }

    #[test]
    fn unitarity_lossless() {
        // T + R = 1 for gamma_i = 0
        for &delta in &[0.0, 0.5, 1.0, 2.0, 10.0] {
            let gamma_e = 1.5;
            let t = tcmt_transmission_coeff(delta, gamma_e, 0.0);
            let r = tcmt_reflection_coeff(delta, gamma_e);
            assert!(
                (t + r - 1.0).abs() < TOL,
                "T+R should be 1 at delta={delta}: T={t}, R={r}"
            );
        }
    }
}
