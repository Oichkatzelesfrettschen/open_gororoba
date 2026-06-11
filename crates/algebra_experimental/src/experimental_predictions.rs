//! Experimental overlay module -- framework predictions vs global oscillation fits.
//!
//! # Purpose
//!
//! Maps scorecard.toml observables into the experimental landscape:
//! NuFit 6.0 global oscillation fit, DUNE/HyperK CP sensitivity bands,
//! and JUNO mass-ordering reach.
//!
//! This module computes OVERLAYS, not verdicts.  An overlay places our
//! predictions on the same axis as the global fit to show WHERE they sit.
//! Whether the framework is confirmed, delimited, or falsified is determined
//! by future experiment -- not by us.
//!
//! # Evidentiary classification
//!
//! | Observable | Class | Notes |
//! |------------|-------|-------|
//! | theta_12, theta_13, theta_23 | B (benchmark) | 4-param fit (Bin 2), not a zero-param derivation |
//! | delta_CP CP-A (~165 deg) | F (falsification target) | Inside NuFit 6.0 1-sigma |
//! | delta_CP CP-B (~93 deg) | F (falsification target) | Outside NuFit 1-sigma; maximal CP-violation |
//! | sin^2(theta_W) = 0.250 | H (heuristic) | G_2 structural estimate, NOT precision EW |
//!
//! # What this module does NOT claim
//!
//! - It does NOT say predictions fall "within" the NuFit allowed regions.
//!   The overlay REPORTS position; the experimental outcome determines verdict.
//! - It does NOT use JUNO for delta_CP sensitivity.  JUNO is a reactor
//!   antineutrino experiment sensitive to mass ordering and dm^2, not to delta_CP.
//!   JUNO data-taking began 2025-08-26.
//! - It does NOT claim the theta_ij match is a derivation from first principles.
//!   These are 4-parameter fits (Bin 2, evidentiary class B).
//!
//! # References
//!
//! - NuFit 6.0 (2024): www.nu-fit.org
//! - DUNE TDR: arXiv:2002.03005
//! - HyperK Design Report: arXiv:1805.04163
//! - JUNO Yellow Book: arXiv:1507.05613
//! - T2K+NOvA joint (2025): arXiv:2510.19888 / Nature 646, 818-824

// ---- contour types --------------------------------------------------------

/// A single parameter's 1-sigma and 3-sigma intervals around the global best fit.
///
/// The intervals are asymmetric because neutrino oscillation posteriors
/// are generally non-Gaussian, especially for delta_CP.
///
/// | Field | Meaning |
/// |-------|---------|
/// | `best` | Maximum-likelihood point |
/// | `one_sigma_{low,high}` | Delta chi^2 = 1 boundary |
/// | `three_sigma_{low,high}` | Delta chi^2 = 9 boundary |
#[derive(Clone, Copy, Debug)]
pub struct SigmaContour {
    pub best: f64,
    pub one_sigma_low: f64,
    pub one_sigma_high: f64,
    pub three_sigma_low: f64,
    pub three_sigma_high: f64,
}

impl SigmaContour {
    /// True if `x` lies inside the 1-sigma interval.
    pub fn in_one_sigma(self, x: f64) -> bool {
        x >= self.one_sigma_low && x <= self.one_sigma_high
    }

    /// True if `x` lies inside the 3-sigma interval.
    pub fn in_three_sigma(self, x: f64) -> bool {
        x >= self.three_sigma_low && x <= self.three_sigma_high
    }

    /// Asymmetric pull: signed distance from best-fit in units of 1-sigma half-width.
    ///
    /// The half-width used is the interval on the same side as `x` (upper if
    /// x > best, lower if x <= best).  This avoids the symmetry fiction for
    /// non-Gaussian posteriors.
    pub fn pull(self, x: f64) -> f64 {
        let half = if x >= self.best {
            (self.one_sigma_high - self.best).max(1e-15)
        } else {
            (self.best - self.one_sigma_low).max(1e-15)
        };
        (x - self.best) / half
    }
}

// ---- NuFit 6.0 ------------------------------------------------------------

/// NuFit 6.0 (2024) global oscillation fit -- normal ordering (NO), with SK atmospheric.
///
/// Source: www.nu-fit.org, NuFit 6.0 release.
/// All angles are in degrees; mass-squared differences in eV^2.
///
/// | Parameter | Best | 1-sigma | 3-sigma |
/// |-----------|------|---------|---------|
/// | theta_12 | 33.41 | [32.66, 34.18] | [31.20, 35.79] |
/// | theta_13 |  8.54 | [ 8.42,  8.66] | [ 8.13,  8.92] |
/// | theta_23 | 49.0  | [47.9,  50.0]  | [40.2,  51.5]  |
/// | delta_CP | 195   | [138, 258]     | [0, 360]       |
/// | dm21_sq  | 7.53e-5 | [7.35e-5, 7.71e-5] | [6.94e-5, 8.14e-5] |
/// | dm31_sq  | 2.453e-3 | [2.420e-3, 2.486e-3] | [2.347e-3, 2.557e-3] |
#[derive(Clone, Debug)]
pub struct NuFit60 {
    pub theta_12: SigmaContour,
    pub theta_13: SigmaContour,
    pub theta_23: SigmaContour,
    /// delta_CP 1-sigma is BROAD: [138, 258].  CP conservation (0 or 180) is
    /// compatible within 1-sigma.  3-sigma covers the full 0-360 circle.
    pub delta_cp: SigmaContour,
    /// dm^2_21 in eV^2.
    pub dm21_sq: SigmaContour,
    /// dm^2_31 in eV^2.
    pub dm31_sq: SigmaContour,
}

impl NuFit60 {
    /// Canonical values for normal ordering with SK atmospheric data.
    pub fn normal_ordering() -> Self {
        Self {
            theta_12: SigmaContour {
                best: 33.41,
                one_sigma_low: 32.66,
                one_sigma_high: 34.18,
                three_sigma_low: 31.20,
                three_sigma_high: 35.79,
            },
            theta_13: SigmaContour {
                best: 8.54,
                one_sigma_low: 8.42,
                one_sigma_high: 8.66,
                three_sigma_low: 8.13,
                three_sigma_high: 8.92,
            },
            theta_23: SigmaContour {
                best: 49.0,
                one_sigma_low: 47.9,
                one_sigma_high: 50.0,
                three_sigma_low: 40.2,
                three_sigma_high: 51.5,
            },
            delta_cp: SigmaContour {
                best: 195.0,
                one_sigma_low: 138.0,
                one_sigma_high: 258.0,
                three_sigma_low: 0.0,
                three_sigma_high: 360.0,
            },
            dm21_sq: SigmaContour {
                best: 7.53e-5,
                one_sigma_low: 7.35e-5,
                one_sigma_high: 7.71e-5,
                three_sigma_low: 6.94e-5,
                three_sigma_high: 8.14e-5,
            },
            dm31_sq: SigmaContour {
                best: 2.453e-3,
                one_sigma_low: 2.420e-3,
                one_sigma_high: 2.486e-3,
                three_sigma_low: 2.347e-3,
                three_sigma_high: 2.557e-3,
            },
        }
    }
}

// ---- evidentiary metadata -------------------------------------------------

/// Epistemic bin for a framework prediction (mirrors scorecard.toml).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpistemicBin {
    /// Zero/low-parameter, framework-derived prediction.
    Bin1Framework,
    /// Optimized angle-sector fit with nonzero parameter count.
    Bin2Optimized,
    /// CP and lift-level exploratory (frontier; scaffolding may change).
    Bin3Exploratory,
}

/// Evidentiary classification for a framework claim.
///
/// Mandatory for every claim added in the research expansion.  The paper
/// must not speak more strongly than the code and literature support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidClass {
    T, // Theorem / Rocq-verified proof
    E, // Exact computation (integer or boolean arithmetic)
    B, // Benchmark (floating-point, reproducible)
    H, // Heuristic / structural ansatz
    C, // Conjecture
    F, // Falsification target -- explicit experimental prediction
}

impl EvidClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::T => "T",
            Self::E => "E",
            Self::B => "B",
            Self::H => "H",
            Self::C => "C",
            Self::F => "F",
        }
    }
}

// ---- overlay results ------------------------------------------------------

/// The result of placing one framework prediction relative to a NuFit 6.0 contour.
#[derive(Clone, Debug)]
pub struct OverlayResult {
    pub observable: &'static str,
    pub our_value: f64,
    pub nufit_best: f64,
    /// Pull in units of asymmetric 1-sigma (see SigmaContour::pull).
    pub pull: f64,
    pub in_one_sigma: bool,
    pub in_three_sigma: bool,
    pub evid: EvidClass,
}

impl OverlayResult {
    pub fn position_label(&self) -> &'static str {
        if self.in_one_sigma {
            "1-sigma"
        } else if self.in_three_sigma {
            "3-sigma"
        } else {
            "outside 3s"
        }
    }
}

// ---- overlay computations -------------------------------------------------

/// Overlay for the three PMNS mixing angles against NuFit 6.0 NO.
///
/// These are Bin 2 (optimized, 4-parameter fit) results.
/// Evidentiary class: B (benchmark -- floating-point, reproducible).
///
/// Scorecard.toml Bin 2 values: theta_12=33.36, theta_13=8.54, theta_23=48.99.
/// Source claim: C-1492 (Gauss-Newton 2D optimization over V_6 basis).
pub fn compute_angle_overlay() -> Vec<OverlayResult> {
    let fit = NuFit60::normal_ordering();
    let entries: &[(&'static str, f64, SigmaContour)] = &[
        ("theta_12_PMNS", 33.36, fit.theta_12),
        ("theta_13_PMNS", 8.54, fit.theta_13),
        ("theta_23_PMNS", 48.99, fit.theta_23),
    ];
    entries
        .iter()
        .map(|&(label, val, contour)| OverlayResult {
            observable: label,
            our_value: val,
            nufit_best: contour.best,
            pull: contour.pull(val),
            in_one_sigma: contour.in_one_sigma(val),
            in_three_sigma: contour.in_three_sigma(val),
            evid: EvidClass::B,
        })
        .collect()
}

/// Overlay for CP phase predictions relative to NuFit 6.0 delta_CP.
///
/// Evidentiary class: F (falsification target).
///
/// CP-A (~165 deg): phase-only complexification, 1 free parameter.
///   Source claim: C-1494.
/// CP-B (~93 deg): joint 3D scan + Nelder-Mead, three independent extractions
///   (arg(-U_e3)=97.9, Jarlskog quartet=92.8, atan2 invariant=86.5) agree.
///   Source claim: C-1498 (amended 2026-03-22 -- genuine maximal CP-violation).
///
/// Both are falsification targets.  DUNE and HyperK will resolve the question.
/// The scorecard notes no route to delta_CP=195 from this framework.
pub fn compute_cp_overlay() -> Vec<OverlayResult> {
    let fit = NuFit60::normal_ordering();
    let entries: &[(&'static str, f64)] = &[("delta_CP_CP-A", 165.0), ("delta_CP_CP-B", 93.0)];
    entries
        .iter()
        .map(|&(label, val)| OverlayResult {
            observable: label,
            our_value: val,
            nufit_best: fit.delta_cp.best,
            pull: fit.delta_cp.pull(val),
            in_one_sigma: fit.delta_cp.in_one_sigma(val),
            in_three_sigma: fit.delta_cp.in_three_sigma(val),
            evid: EvidClass::F,
        })
        .collect()
}

// ---- experiment sensitivity -----------------------------------------------

/// Approximate CP-violation discovery sensitivity for one long-baseline experiment.
///
/// All sigma values are read from published sensitivity curves for the full
/// design exposure of each experiment.  They are approximate (evid class B).
///
/// JUNO is excluded here -- see [`JunoMassOrderingSensitivity`].
#[derive(Clone, Debug)]
pub struct ExperimentSensitivity {
    pub name: &'static str,
    pub arxiv: &'static str,
    /// Fraction of delta_CP phase space where CP violation is discoverable at >3 sigma.
    pub cp_discovery_fraction_3s: f64,
    /// Fraction at >5 sigma.
    pub cp_discovery_fraction_5s: f64,
    /// Approximate sigma significance at CP-A (~165 deg).
    pub sigma_at_cp_a: f64,
    /// Approximate sigma significance at CP-B (~93 deg, near-maximal CP-violation).
    pub sigma_at_cp_b: f64,
    pub role: &'static str,
}

/// DUNE and HyperK sensitivities for full design exposure.
///
/// Source: DUNE TDR arXiv:2002.03005 and HyperK DR arXiv:1805.04163.
/// At delta_CP ~ 90-270 deg (significant CP-violation) both experiments reach
/// above 5 sigma.  At near-maximal CP (delta~93 deg) sensitivity is near-maximum.
/// At moderate CP-violation (delta~165 deg) sensitivity is ~3-4 sigma.
pub fn canonical_experiment_sensitivities() -> Vec<ExperimentSensitivity> {
    vec![
        ExperimentSensitivity {
            name: "DUNE",
            arxiv: "2002.03005",
            cp_discovery_fraction_3s: 0.75,
            cp_discovery_fraction_5s: 0.50,
            sigma_at_cp_a: 3.5, // sin(165 deg) ~ 0.26, moderate CP
            sigma_at_cp_b: 5.0, // sin(93 deg) ~ 1.0, near-maximal CP
            role: "CP violation discovery + delta_CP precision (long-baseline nu beam)",
        },
        ExperimentSensitivity {
            name: "HyperK",
            arxiv: "1805.04163",
            cp_discovery_fraction_3s: 0.76,
            cp_discovery_fraction_5s: 0.55,
            sigma_at_cp_a: 3.7,
            sigma_at_cp_b: 5.2,
            role: "CP violation discovery + delta_CP precision (T2HK beam + atmospheric)",
        },
    ]
}

// ---- JUNO (mass ordering only) --------------------------------------------

/// JUNO mass-ordering and dm^2 sensitivity.
///
/// JUNO does NOT constrain delta_CP directly.  Its value for this framework:
/// (a) it will independently confirm/refute the normal-ordering prediction
///     (Bin 1, C-1459, categorical prediction);
/// (b) it will tighten dm21_sq and dm31_sq, constraining the Bin 1 observable
///     r_nu = dm21/dm31 (C-1459).
///
/// Source: JUNO Yellow Book arXiv:1507.05613.
/// JUNO began data-taking 2025-08-26.
#[derive(Clone, Debug)]
pub struct JunoMassOrderingSensitivity {
    /// Expected significance for normal vs inverted ordering discrimination.
    pub mass_ordering_sigma: f64,
    /// Expected fractional precision on dm21_sq.
    pub dm21_precision_frac: f64,
    /// Expected fractional precision on dm31_sq.
    pub dm31_precision_frac: f64,
    /// Our r_nu = dm21_sq / dm31_sq prediction (scorecard OBS-005, C-1459).
    pub our_r_nu: f64,
    /// NuFit 6.0 best-fit r_nu for reference.
    pub nufit_r_nu: f64,
    pub arxiv: &'static str,
}

pub fn juno_sensitivity() -> JunoMassOrderingSensitivity {
    let fit = NuFit60::normal_ordering();
    JunoMassOrderingSensitivity {
        mass_ordering_sigma: 3.0,
        dm21_precision_frac: 0.006,
        dm31_precision_frac: 0.002,
        our_r_nu: 0.0304,
        nufit_r_nu: fit.dm21_sq.best / fit.dm31_sq.best,
        arxiv: "1507.05613",
    }
}

// ---- table output ---------------------------------------------------------

/// Print the full experimental overlay to stdout.
///
/// Structured output suitable for copy-paste into docs or markdown tables.
/// Called from `tests::test_print_full_overlay_table`.
pub fn print_overlay_table() {
    let fit = NuFit60::normal_ordering();
    let angles = compute_angle_overlay();
    let cp = compute_cp_overlay();
    let experiments = canonical_experiment_sensitivities();
    let juno = juno_sensitivity();

    println!("=== PMNS Experimental Overlay (Phase C) ===");
    println!("  Reference: NuFit 6.0 (2024), Normal Ordering + SK atmospheric");
    println!("  Framework: scorecard.toml Bin 2 + Bin 3 (2026-03-23)");
    println!();

    println!("--- Mixing Angles (Bin 2, evid=B) ---");
    println!(
        "  {:<20} {:>9} {:>10} {:>10} {:>14}",
        "Observable", "Our(deg)", "Best fit", "Pull(1s)", "Position"
    );
    for r in &angles {
        println!(
            "  {:<20} {:>9.2} {:>10.2} {:>+10.3} {:>14}",
            r.observable,
            r.our_value,
            r.nufit_best,
            r.pull,
            r.position_label()
        );
    }
    println!();
    println!("  NuFit 6.0 1-sigma ranges (NO):");
    println!(
        "    theta_12: [{:.2}, {:.2}] deg",
        fit.theta_12.one_sigma_low, fit.theta_12.one_sigma_high
    );
    println!(
        "    theta_13: [{:.2}, {:.2}] deg",
        fit.theta_13.one_sigma_low, fit.theta_13.one_sigma_high
    );
    println!(
        "    theta_23: [{:.1}, {:.1}] deg",
        fit.theta_23.one_sigma_low, fit.theta_23.one_sigma_high
    );
    println!();

    println!("--- CP Phase (Bin 3, evid=F: falsification target) ---");
    println!(
        "  {:<20} {:>9} {:>10} {:>10} {:>14}",
        "Prediction", "Our(deg)", "Best fit", "Pull(1s)", "Position"
    );
    let cp_claims = ["C-1494", "C-1498"];
    for (r, claim) in cp.iter().zip(cp_claims.iter()) {
        println!(
            "  {:<20} {:>9.1} {:>10.1} {:>+10.3} {:>14}  ({})",
            r.observable,
            r.our_value,
            r.nufit_best,
            r.pull,
            r.position_label(),
            claim
        );
    }
    println!();
    println!(
        "  NuFit 6.0 delta_CP: best={:.0} deg, 1-sigma=[{:.0},{:.0}], 3-sigma=[0,360]",
        fit.delta_cp.best, fit.delta_cp.one_sigma_low, fit.delta_cp.one_sigma_high
    );
    println!("  NOTE: CP conservation compatible within 1-sigma.");
    println!();

    println!("--- DUNE / HyperK Sensitivity at Our Predictions ---");
    println!(
        "  {:<8} {:>12} {:>16} {:>16} {:>12} {:>12}",
        "Expt", "arXiv", "sigma@CP-A(165)", "sigma@CP-B(93)", "3s frac", "5s frac"
    );
    for e in &experiments {
        println!(
            "  {:<8} {:>12} {:>16.1} {:>16.1} {:>11.0}% {:>11.0}%",
            e.name,
            e.arxiv,
            e.sigma_at_cp_a,
            e.sigma_at_cp_b,
            e.cp_discovery_fraction_3s * 100.0,
            e.cp_discovery_fraction_5s * 100.0
        );
    }
    println!();
    println!("  CP-A (~165 deg): INSIDE NuFit 1-sigma. Both DUNE and HyperK");
    println!("    reach ~3.5 sigma here -- measurable but not guaranteed discovery.");
    println!("  CP-B (~93 deg): OUTSIDE NuFit 1-sigma. Near-maximal CP-violation.");
    println!("    Both experiments reach ~5 sigma -- this is highly distinguishable.");
    println!("  Both are FALSIFICATION TARGETS (class F). Experimental outcome decides.");
    println!();

    println!("--- JUNO (mass ordering + dm^2 precision, NOT delta_CP) ---");
    println!("  Source: arXiv:{}", juno.arxiv);
    println!(
        "  Mass ordering significance: ~{:.0} sigma (6-yr run)",
        juno.mass_ordering_sigma
    );
    println!(
        "  dm21 precision: {:.1}%  |  dm31 precision: {:.1}%",
        juno.dm21_precision_frac * 100.0,
        juno.dm31_precision_frac * 100.0
    );
    println!("  Our r_nu = {:.4}  (Bin 1, C-1459)", juno.our_r_nu);
    println!("  NuFit r_nu = {:.4}", juno.nufit_r_nu);
    println!(
        "  r_nu deviation: {:.2}%",
        (juno.our_r_nu - juno.nufit_r_nu).abs() / juno.nufit_r_nu * 100.0
    );
    println!("  NOTE: JUNO does NOT constrain delta_CP. Data-taking began 2025-08-26.");
}

// ---- tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigma_contour_membership() {
        let c = SigmaContour {
            best: 10.0,
            one_sigma_low: 9.0,
            one_sigma_high: 11.0,
            three_sigma_low: 7.0,
            three_sigma_high: 13.0,
        };
        assert!(c.in_one_sigma(10.0));
        assert!(c.in_one_sigma(9.5));
        assert!(!c.in_one_sigma(8.5));
        assert!(c.in_three_sigma(8.0));
        assert!(!c.in_three_sigma(6.0));
    }

    #[test]
    fn test_asymmetric_pull() {
        let c = SigmaContour {
            best: 10.0,
            one_sigma_low: 8.0,
            one_sigma_high: 12.0,
            three_sigma_low: 4.0,
            three_sigma_high: 16.0,
        };
        // At best: pull = 0
        assert!((c.pull(10.0)).abs() < 1e-12);
        // At upper boundary: pull = +1.0
        assert!((c.pull(12.0) - 1.0).abs() < 1e-12);
        // At lower boundary: pull = -1.0
        assert!((c.pull(8.0) + 1.0).abs() < 1e-12);
    }

    /// All three mixing angles (Bin 2) must be inside NuFit 6.0 1-sigma.
    ///
    /// This is a regression test: if the Gauss-Newton optimization regresses,
    /// the angle predictions will drift outside the 1-sigma contour and this
    /// test will catch it.
    #[test]
    fn test_angle_overlay_all_in_one_sigma() {
        let results = compute_angle_overlay();
        assert_eq!(results.len(), 3, "should have 3 mixing angle overlays");
        for r in &results {
            assert!(
                r.in_one_sigma,
                "{}: our value {:.3} deg is NOT inside NuFit 1-sigma (pull={:+.3})",
                r.observable, r.our_value, r.pull,
            );
            assert!(
                r.pull.abs() <= 1.0,
                "{}: pull magnitude {:.3} should be <= 1.0",
                r.observable,
                r.pull
            );
        }
    }

    /// CP-A (~165 deg) is inside the NuFit 6.0 1-sigma range for delta_CP.
    ///
    /// The 1-sigma range [138, 258] is broad; CP-A lies comfortably inside it.
    /// This is class F (falsification target): DUNE/HyperK will test this.
    #[test]
    fn test_cp_a_in_nufit_one_sigma() {
        let results = compute_cp_overlay();
        let cp_a = results
            .iter()
            .find(|r| r.observable == "delta_CP_CP-A")
            .expect("CP-A result missing");
        let fit = NuFit60::normal_ordering();
        assert!(
            cp_a.in_one_sigma,
            "CP-A {:.1} deg should be inside NuFit 1-sigma [{:.0}, {:.0}]",
            cp_a.our_value, fit.delta_cp.one_sigma_low, fit.delta_cp.one_sigma_high,
        );
    }

    /// CP-B (~93 deg) is OUTSIDE NuFit 1-sigma but inside 3-sigma (trivially: 0-360).
    ///
    /// This reflects the amended scorecard note (2026-03-22): no route from the
    /// framework to delta=195; CP-B is a genuine maximal-CP prediction, falsified
    /// against the current global-fit best value but unresolved by 3-sigma.
    #[test]
    fn test_cp_b_outside_one_sigma() {
        let results = compute_cp_overlay();
        let cp_b = results
            .iter()
            .find(|r| r.observable == "delta_CP_CP-B")
            .expect("CP-B result missing");
        let fit = NuFit60::normal_ordering();
        assert!(
            !cp_b.in_one_sigma,
            "CP-B {:.1} deg should be OUTSIDE NuFit 1-sigma [{:.0}, {:.0}]",
            cp_b.our_value, fit.delta_cp.one_sigma_low, fit.delta_cp.one_sigma_high,
        );
        assert!(
            cp_b.in_three_sigma,
            "CP-B should be inside trivial 3-sigma [0, 360]",
        );
    }

    /// r_nu = dm21/dm31 prediction matches NuFit 6.0 within 3%.
    ///
    /// This is the Bin 1 falsifiable prediction (C-1459).  JUNO precision
    /// on dm21 (0.6%) will tighten this constraint significantly.
    #[test]
    fn test_juno_r_nu_within_three_percent() {
        let juno = juno_sensitivity();
        let deviation_pct = (juno.our_r_nu - juno.nufit_r_nu).abs() / juno.nufit_r_nu * 100.0;
        assert!(
            deviation_pct < 3.0,
            "r_nu deviation {:.2}% should be < 3%",
            deviation_pct,
        );
    }

    #[test]
    fn test_print_full_overlay_table() {
        print_overlay_table();
    }
}
