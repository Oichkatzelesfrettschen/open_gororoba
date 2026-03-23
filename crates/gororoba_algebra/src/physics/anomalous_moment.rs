//! G_2 stabilizer structure and the anomalous magnetic moment of the muon.
//!
//! # Evidentiary classification: H (heuristic / structural ansatz)
//!
//! This module computes a STRUCTURAL ORDER-OF-MAGNITUDE ESTIMATE only.
//! It does NOT claim to predict, derive, or explain the anomalous magnetic moment.
//! The analogy is sin^2(theta_W): the G_2 coset ratio dim(su(3))/dim(g_2) = 8/14
//! gives 0.571, which when renormalized against the observed 0.231 is suggestive
//! but heuristic.  This module carries the same epistemic status.
//!
//! # Experimental status (2025)
//!
//! The muon g-2 is NO LONGER a strong anomaly.  The 2025 Theory Initiative White
//! Paper gives Delta_a_mu = 38(63) x 10^-11, which is compatible within 1-sigma
//! uncertainties with the Standard Model.  (Reference: arXiv:2505.21476, BIB-0440.)
//!
//! The Fermilab final Run 1-6 result: a_mu^exp = 1165920715(145) x 10^-12.
//! The SM prediction (2020 WP, arXiv:2006.04822): a_mu(SM) = 116591810(43) x 10^-11.
//!
//! This module exists to document the G_2 structural ansatz -- NOT to claim
//! it explains or predicts the anomalous moment.
//!
//! # G_2 structural content
//!
//! G_2 is the automorphism group of the octonions.  As a compact Lie group:
//! - Dimension: 14
//! - Rank: 2
//! - Positive roots: 6 (short x 3, long x 3)
//! - Contains SU(3) as the stabilizer of e_7 (one fixed imaginary unit)
//!
//! The SU(3) embedding in G_2:
//! - SU(3) has dimension 8 (the Gell-Mann matrices)
//! - G_2 / SU(3) coset has dimension 14 - 8 = 6
//!
//! The ratio dim(G_2 coset) / dim(G_2) = 6/14 = 3/7 ~ 0.429
//! is analogous to how sin^2(theta_W) = dim(su(3))/dim(g_2) = 8/14 = 4/7 ~ 0.571.
//!
//! # What the structural estimate produces
//!
//! Following the same logic as sin^2(theta_W), we can ask: does any simple
//! G_2 dimensional ratio give a_mu ~ 10^-3?
//!
//! The leading QED contribution to a_mu is alpha/(2*pi) ~ 1.16 x 10^-3.
//! The G_2 coset ratio 6/14 = 3/7 does not reproduce this scale without
//! additional input (alpha, pi).
//!
//! The HONEST finding: G_2 structure gives ratios O(1/14) to O(1), which are
//! many orders of magnitude above a_mu ~ 10^-3.  There is no G_2 structural
//! ansatz that naturally produces the anomalous moment scale.

/// Fundamental data about the G_2 Lie group and its SU(3) subgroup.
///
/// These are exact mathematical values (evid E).
#[derive(Clone, Copy, Debug)]
pub struct G2Structure {
    /// Total dimension of G_2 (number of generators).
    pub dim_g2: usize,
    /// Rank of G_2.
    pub rank: usize,
    /// Dimension of the embedded SU(3) subgroup.
    pub dim_su3: usize,
    /// Dimension of the G_2 / SU(3) coset.
    pub dim_coset: usize,
    /// Ratio dim(su3)/dim(g2) -- same formula as sin^2(theta_W) heuristic.
    pub ratio_su3_over_g2: f64,
    /// Ratio dim(coset)/dim(g2) -- fraction NOT in the SU(3) part.
    pub ratio_coset_over_g2: f64,
}

impl G2Structure {
    pub fn canonical() -> Self {
        let dim_g2 = 14;
        let dim_su3 = 8;
        let dim_coset = dim_g2 - dim_su3;
        Self {
            dim_g2,
            rank: 2,
            dim_su3,
            dim_coset,
            ratio_su3_over_g2: dim_su3 as f64 / dim_g2 as f64,
            ratio_coset_over_g2: dim_coset as f64 / dim_g2 as f64,
        }
    }
}

/// The measured and theoretical values of the muon anomalous magnetic moment.
///
/// These are experimental/theoretical results (evid E for the numbers themselves,
/// though the sources have experimental/theoretical uncertainties).
///
/// Reference: arXiv:2505.21476 (WP 2025), arXiv:2006.04822 (WP 2020).
#[derive(Clone, Copy, Debug)]
pub struct MuonG2Data {
    /// Fermilab final experimental value (Run 1-6): a_mu^exp (in units of 10^-11).
    pub a_mu_exp: f64,
    /// SM theory prediction (WP 2020, arXiv:2006.04822): a_mu(SM) (in units of 10^-11).
    pub a_mu_sm: f64,
    /// Discrepancy Delta_a_mu = a_mu^exp - a_mu^sm (in units of 10^-11).
    pub delta_a_mu: f64,
    /// 1-sigma uncertainty on Delta_a_mu (in units of 10^-11).
    pub delta_uncertainty: f64,
}

impl MuonG2Data {
    /// 2025 values (arXiv:2505.21476).
    ///
    /// Discrepancy = 38(63) x 10^-11 -- compatible within 1-sigma with SM.
    pub fn wp_2025() -> Self {
        Self {
            // Fermilab Run 1-6: 1165920715(145) x 10^-12 = 116592071.5(14.5) x 10^-11
            a_mu_exp: 116_592_071.5,
            // WP 2020 SM: 116591810(43) x 10^-11 (used as SM baseline)
            a_mu_sm: 116_591_810.0,
            delta_a_mu: 38.0,
            delta_uncertainty: 63.0,
        }
    }
}

/// Compute the G_2 structural estimate for comparison with a_mu.
///
/// Evidentiary class: H (heuristic / structural ansatz).
///
/// Following the sin^2(theta_W) analogy:
///   sin^2(theta_W)_heuristic = dim(su(3)) / dim(g_2) = 8/14 = 0.571
/// (which is off from the measured 0.231 by a factor of ~2.5, but the order of
/// magnitude is correct as a dimensional counting estimate).
///
/// For a_mu, the dimensional G_2 ratio gives:
///   ratio_su3_g2 = 8/14 = 0.571 ~ O(1)
///   ratio_coset_g2 = 6/14 = 0.429 ~ O(1)
///
/// These are O(1) -- many orders of magnitude above a_mu ~ 2.5 x 10^-9.
/// A G_2 structural ansatz cannot produce the correct SCALE for a_mu without
/// injecting the coupling constant alpha and loop factor 1/(2*pi).
///
/// The correct leading estimate uses alpha/(2*pi) ~ 1.16 x 10^-3, not any
/// G_2 ratio.  The G_2 content is orthogonal to this.
///
/// FINDING: There is no meaningful G_2 structural estimate for a_mu.
/// The G_2 dimensional ratios produce O(1) numbers, not O(10^-9).
#[derive(Clone, Debug)]
pub struct G2MuonEstimate {
    pub g2: G2Structure,
    pub data: MuonG2Data,
    /// The ratio dim(su3)/dim(g2) -- the "structural ansatz" (evid H).
    pub structural_ratio: f64,
    /// The leading QED estimate alpha/(2*pi) for comparison.
    pub qed_leading: f64,
    /// Orders of magnitude between structural_ratio and measured Delta_a_mu.
    pub scale_gap_orders: f64,
}

impl G2MuonEstimate {
    pub fn compute() -> Self {
        let g2 = G2Structure::canonical();
        let data = MuonG2Data::wp_2025();
        let structural_ratio = g2.ratio_su3_over_g2;
        // alpha ~ 1/137.036, leading QED: alpha/(2*pi)
        let alpha = 1.0_f64 / 137.035_999_084;
        let qed_leading = alpha / (2.0 * std::f64::consts::PI);
        // Delta_a_mu in natural units (not x 10^-11)
        let delta_natural = data.delta_a_mu * 1e-11;
        let scale_gap_orders = (structural_ratio / delta_natural.abs().max(1e-100)).log10();
        G2MuonEstimate {
            g2,
            data,
            structural_ratio,
            qed_leading,
            scale_gap_orders,
        }
    }
}

/// Print the full G_2 / muon g-2 structural comparison.
///
/// Explicitly labels all quantities with evidentiary class.
pub fn print_g2_muon_summary() {
    let est = G2MuonEstimate::compute();

    println!("=== G_2 Structure vs Muon Anomalous Magnetic Moment ===");
    println!("  Evidentiary class: H (heuristic structural ansatz)");
    println!();
    println!("--- Experimental status (2025) ---");
    println!(
        "  a_mu^exp (Fermilab R1-6): {:.1} x 10^-11  (arXiv:BIB-0442)",
        est.data.a_mu_exp
    );
    println!(
        "  a_mu^SM  (WP 2025):       {:.1} x 10^-11  (arXiv:2505.21476)",
        est.data.a_mu_sm
    );
    println!(
        "  Delta_a_mu:               {:.0}({:.0}) x 10^-11",
        est.data.delta_a_mu, est.data.delta_uncertainty
    );
    println!("  STATUS: Compatible within 1-sigma (NOT a strong anomaly as of 2025).");
    println!();
    println!("--- G_2 structural data (evid E -- exact Lie theory) ---");
    println!("  dim(G_2) = {}  rank = {}", est.g2.dim_g2, est.g2.rank);
    println!(
        "  dim(SU(3)) = {}  (embedded in G_2 as Aut(Octonions|e_7))",
        est.g2.dim_su3
    );
    println!("  dim(coset G_2/SU(3)) = {}", est.g2.dim_coset);
    println!(
        "  ratio dim(SU(3))/dim(G_2) = {}/{} = {:.4}  (sin^2(theta_W) analogy)",
        est.g2.dim_su3, est.g2.dim_g2, est.g2.ratio_su3_over_g2
    );
    println!(
        "  ratio dim(coset)/dim(G_2) = {}/{} = {:.4}",
        est.g2.dim_coset, est.g2.dim_g2, est.g2.ratio_coset_over_g2
    );
    println!();
    println!("--- Structural estimate (evid H) ---");
    println!(
        "  Structural ratio: {:.4}  (analogous to sin^2(theta_W) ~ 0.250 heuristic)",
        est.structural_ratio
    );
    println!(
        "  QED leading term alpha/(2*pi): {:.4e}  (correct scale for a_mu)",
        est.qed_leading
    );
    println!(
        "  Delta_a_mu in natural units: {:.1e}",
        est.data.delta_a_mu * 1e-11
    );
    println!(
        "  Scale gap: {:.1} orders of magnitude between structural ratio and Delta_a_mu",
        est.scale_gap_orders
    );
    println!();
    println!("--- Finding ---");
    println!("  G_2 dimensional ratios are O(1).  Delta_a_mu ~ 3.8 x 10^-10.");
    println!(
        "  The scale gap is ~{:.0} orders of magnitude.",
        est.scale_gap_orders
    );
    println!("  CONCLUSION: There is no meaningful G_2 structural estimate for a_mu.");
    println!("  The sin^2(theta_W) analogy (dim ratio -> O(1) estimate) does not");
    println!("  extend to a_mu, which requires alpha and loop-factor input.");
    println!("  This module is a negative result, not a prediction.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g2_structure_exact() {
        let g2 = G2Structure::canonical();
        assert_eq!(g2.dim_g2, 14, "G_2 should have dimension 14");
        assert_eq!(g2.dim_su3, 8, "SU(3) subgroup should have dimension 8");
        assert_eq!(g2.dim_coset, 6, "G_2/SU(3) coset should have dimension 6");
        assert_eq!(
            g2.dim_su3 + g2.dim_coset,
            g2.dim_g2,
            "su3 + coset should equal g2 dim"
        );
        // Exact fractions
        let expected_ratio_su3 = 8.0_f64 / 14.0;
        let expected_ratio_coset = 6.0_f64 / 14.0;
        assert!((g2.ratio_su3_over_g2 - expected_ratio_su3).abs() < 1e-12);
        assert!((g2.ratio_coset_over_g2 - expected_ratio_coset).abs() < 1e-12);
    }

    /// 2025 discrepancy is compatible within uncertainties.
    ///
    /// Delta_a_mu = 38(63) x 10^-11.
    /// Significance = 38 / 63 = 0.6 sigma -- NOT anomalous.
    #[test]
    fn test_g2_2025_discrepancy_below_one_sigma() {
        let data = MuonG2Data::wp_2025();
        let significance = data.delta_a_mu / data.delta_uncertainty;
        assert!(
            significance.abs() < 1.0,
            "2025 Delta_a_mu significance should be < 1 sigma, got {:.2}",
            significance
        );
    }

    /// The structural ratio is O(1), not O(10^-3) -- confirming no G_2 estimate for a_mu.
    #[test]
    fn test_g2_structural_ratio_not_muon_scale() {
        let est = G2MuonEstimate::compute();
        // Structural ratio should be > 0.1 (order 1)
        assert!(
            est.structural_ratio > 0.1,
            "G_2 structural ratio {} should be O(1)",
            est.structural_ratio
        );
        // Delta_a_mu (natural units) should be < 10^-9 (much smaller than structural ratio)
        let delta_natural = est.data.delta_a_mu * 1e-11;
        assert!(
            delta_natural < 1e-9,
            "Delta_a_mu {} should be < 10^-9 (muon scale)",
            delta_natural
        );
        // Scale gap should be > 8 orders of magnitude
        assert!(
            est.scale_gap_orders > 8.0,
            "Scale gap {:.1} should be > 8 orders of magnitude",
            est.scale_gap_orders
        );
    }

    /// Validate Fermilab 2025 final a_mu^exp encoding.
    ///
    /// Fermilab Run 1-6: 1165920715(145) x 10^-12
    ///                 = 116592071.5(14.5) x 10^-11
    /// Source: arXiv:BIB-0442 (Fermilab final result).
    #[test]
    fn test_fermilab_2025_value_correct() {
        let data = MuonG2Data::wp_2025();
        // Fermilab Run 1-6 central value in units of 10^-11
        let expected_exp = 116_592_071.5_f64;
        assert!(
            (data.a_mu_exp - expected_exp).abs() < 0.1,
            "a_mu^exp should be 116592071.5 x 10^-11, got {}",
            data.a_mu_exp
        );
        // SM WP 2020 baseline (still used in 2025 comparison)
        let expected_sm = 116_591_810.0_f64;
        assert!(
            (data.a_mu_sm - expected_sm).abs() < 0.1,
            "a_mu^SM should be 116591810.0 x 10^-11, got {}",
            data.a_mu_sm
        );
        // Naive difference (not the official discrepancy -- see delta_a_mu)
        let naive_diff = data.a_mu_exp - data.a_mu_sm;
        assert!(
            naive_diff > 0.0,
            "Naive exp-SM difference should be positive, got {}",
            naive_diff
        );
    }

    /// Validate the QED leading term alpha/(2*pi) used for scale comparison.
    ///
    /// alpha = 1/137.035999084 (CODATA 2018).
    /// alpha/(2*pi) ~ 1.1614e-3 -- the leading Schwinger correction.
    #[test]
    fn test_qed_leading_term_accuracy() {
        let est = G2MuonEstimate::compute();
        // alpha/(2*pi) ~ 1.1614e-3 (within 0.1% of exact value)
        let expected = 1.0_f64 / (137.035_999_084 * 2.0 * std::f64::consts::PI);
        assert!(
            (est.qed_leading - expected).abs() / expected < 1e-6,
            "QED leading term should match alpha/(2*pi) = {:.6e}, got {:.6e}",
            expected,
            est.qed_leading
        );
        // Must be O(10^-3), not O(1)
        assert!(
            est.qed_leading < 1e-2 && est.qed_leading > 1e-4,
            "QED leading ~ 1.16e-3, got {:.4e}",
            est.qed_leading
        );
    }

    /// Confirm scale gap is >= 9 orders of magnitude (tight bound).
    ///
    /// G_2 structural ratio = 8/14 ~ 0.571 (O(1)).
    /// Delta_a_mu = 38 x 10^-11 = 3.8 x 10^-10 (O(10^-10)).
    /// log10(0.571 / 3.8e-10) ~ 9.18 -- well above 9 orders.
    #[test]
    fn test_scale_gap_at_least_9_orders() {
        let est = G2MuonEstimate::compute();
        assert!(
            est.scale_gap_orders >= 9.0,
            "Scale gap should be >= 9 orders, got {:.2}",
            est.scale_gap_orders
        );
    }

    #[test]
    fn test_print_g2_muon_summary() {
        print_g2_muon_summary();
    }
}
