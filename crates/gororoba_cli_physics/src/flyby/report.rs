//! Flyby result formatting and delta-V comparison reporting.

/// Result of a single flyby simulation.
#[derive(Debug, Clone)]
pub struct FlybyResult {
    pub name: String,
    /// Outbound velocity magnitude without anomalous force (km/s).
    pub v_out_control: f64,
    /// Outbound velocity magnitude with anomalous force (km/s).
    pub v_out_anomalous: f64,
    /// Observed anomalous delta-V (mm/s).
    pub observed_dv_mm_s: f64,
}

impl FlybyResult {
    /// Predicted anomalous delta-V in mm/s.
    pub fn predicted_dv_mm_s(&self) -> f64 {
        (self.v_out_anomalous - self.v_out_control) * 1e6
    }

    /// Ratio of predicted to observed delta-V.
    pub fn ratio(&self) -> f64 {
        if self.observed_dv_mm_s.abs() < 1e-10 {
            return f64::NAN;
        }
        self.predicted_dv_mm_s() / self.observed_dv_mm_s
    }

    /// Sign match: predicted and observed have the same sign.
    pub fn sign_match(&self) -> bool {
        let pred = self.predicted_dv_mm_s();
        (pred > 0.0 && self.observed_dv_mm_s > 0.0)
            || (pred < 0.0 && self.observed_dv_mm_s < 0.0)
            || (pred.abs() < 1e-10 && self.observed_dv_mm_s.abs() < 1e-10)
    }
}

/// Print a formatted comparison table of flyby results.
pub fn print_comparison_table(results: &[FlybyResult]) {
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>6}",
        "Spacecraft", "Obs (mm/s)", "Pred (mm/s)", "Ratio", "Sign"
    );
    println!("{}", "-".repeat(70));
    for r in results {
        let sign = if r.sign_match() { "OK" } else { "FAIL" };
        println!(
            "{:<28} {:>10.2} {:>10.2} {:>10.3} {:>6}",
            r.name,
            r.observed_dv_mm_s,
            r.predicted_dv_mm_s(),
            r.ratio(),
            sign,
        );
    }
}
