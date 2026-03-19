//! QCD Critical Point (QCP) and Beam Energy Scan (BES) observables.
//!
//! Implements net-baryon fluctuation cumulants ($C_n$) used to search for the 
//! critical end point in the QCD phase diagram. Relates experimental net-proton 
//! distributions to theoretical susceptibilities.

/// Net-proton cumulant ratio $C_4/C_2$ (kurtosis * variance / variance).
/// Used as a signature for the critical point.
pub struct CumulantRatio {
    pub order_num: usize,
    pub order_den: usize,
    pub value: f64,
}

impl CumulantRatio {
    /// Net-proton $C_4/C_2$ ratio.
    pub fn c4_c2(c4: f64, c2: f64) -> Self {
        Self {
            order_num: 4,
            order_den: 2,
            value: if c2 != 0.0 { c4 / c2 } else { 0.0 },
        }
    }

    /// Skewness ratio $C_3/C_2$.
    pub fn c3_c2(c3: f64, c2: f64) -> Self {
        Self {
            order_num: 3,
            order_den: 2,
            value: if c2 != 0.0 { c3 / c2 } else { 0.0 },
        }
    }
}

/// Skewness $S\sigma = C_3/C_2$ and kurtosis $\kappa\sigma^2 = C_4/C_2$ 
/// relative to Hadron Resonance Gas (HRG) baseline.
pub fn calculate_deviation_from_baseline(
    measured: f64,
    baseline: f64,
) -> f64 {
    (measured - baseline) / baseline
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c4_c2_ratio() {
        let ratio = CumulantRatio::c4_c2(1.2, 1.0);
        assert!((ratio.value - 1.2).abs() < 1e-6);
    }
}
