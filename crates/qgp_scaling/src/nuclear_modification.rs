//! Nuclear modification factor ($R_{AA}$) computations.
//!
//! Provides the canonical observable for hard probe suppression (jet quenching)
//! in heavy-ion collisions, normalizing AA yields by pp baseline cross sections
//! and the optical Glauber overlap function $T_{AA}$.

/// Represents a measurement of the Nuclear Modification Factor $R_{AA}$ at a given $p_T$.
pub struct RaaMeasurement {
    pub pt: f64,
    pub raa: f64,
    pub stat_err: f64,
    pub sys_err: f64,
}

/// Computes $R_{AA}$ from yields and cross sections.
///
/// # Formula
/// $R_{AA}(p_T) = \frac{1}{\langle T_{AA} \rangle} \frac{dN_{AA}/dp_T}{d\sigma_{pp}/dp_T}$
pub fn compute_raa(yield_aa: f64, cross_section_pp: f64, t_aa: f64) -> f64 {
    if t_aa <= 0.0 || cross_section_pp <= 0.0 {
        return 0.0;
    }
    yield_aa / (t_aa * cross_section_pp)
}

/// Calculates the statistical uncertainty of $R_{AA}$ using standard error propagation.
pub fn compute_raa_stat_err(
    yield_aa: f64,
    err_yield_aa: f64,
    cross_section_pp: f64,
    err_cross_section_pp: f64,
    t_aa: f64,
) -> f64 {
    let raa = compute_raa(yield_aa, cross_section_pp, t_aa);
    if raa == 0.0 {
        return 0.0;
    }

    let rel_err_aa = err_yield_aa / yield_aa;
    let rel_err_pp = err_cross_section_pp / cross_section_pp;

    raa * (rel_err_aa * rel_err_aa + rel_err_pp * rel_err_pp).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_raa() {
        let y_aa = 0.005;
        let sig_pp = 0.1;
        let t_aa = 0.05; // 1/mb equivalent mapping
        let raa = compute_raa(y_aa, sig_pp, t_aa);
        assert!((raa - 1.0).abs() < 1e-6); // No suppression
    }
}
