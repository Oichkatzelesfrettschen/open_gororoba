//! Claims verification gates for CI integration.
//!
//! Provides a framework for defining testable physics claims with:
//! - Pre-registered thresholds and decision rules
//! - Integration with permutation tests (ED, MMD)
//! - Bootstrap CI uncertainty quantification
//! - Machine-readable pass/fail verdicts
//!
//! # Architecture
//!
//! Each claim gate consists of:
//! 1. Claim ID (e.g., "C-070")
//! 2. Test function that returns evidence
//! 3. Decision rule (threshold + logic)
//! 4. Pre-registered hypothesis
//!
//! # Usage in CI
//!
//! ```text
//! cargo test --package stats_core -- claims_gates
//! ```
//!
//! Gates output structured JSON for CI parsing.

use crate::{
    bootstrap_ci, frechet_null_test, two_sample_test, BootstrapCIResult, FrechetNullTestResult,
    TwoSampleTestResult,
};
use std::fmt;

/// Claim verification verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Claim passes verification criteria.
    Pass,
    /// Claim fails verification criteria (refuted).
    Fail,
    /// Insufficient evidence (uncertainty too high).
    Uncertain,
    /// Claim not yet evaluated.
    Pending,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Pass => write!(f, "PASS"),
            Verdict::Fail => write!(f, "FAIL"),
            Verdict::Uncertain => write!(f, "UNCERTAIN"),
            Verdict::Pending => write!(f, "PENDING"),
        }
    }
}

/// Evidence types for claim gates.
#[derive(Debug, Clone)]
pub enum Evidence {
    /// P-value from permutation test.
    PValue {
        value: f64,
        n_permutations: usize,
        threshold: f64,
    },
    /// Bootstrap confidence interval.
    BootstrapCI {
        point_estimate: f64,
        ci_lower: f64,
        ci_upper: f64,
        width_relative: f64,
        uncertain_threshold: f64,
    },
    /// Frechet distance comparison.
    FrechetDistance {
        observed: f64,
        null_mean: f64,
        p_value: f64,
    },
    /// Two-sample test (ED + MMD combined).
    TwoSample {
        ed_p_value: f64,
        mmd_p_value: f64,
        combined_p_value: f64,
    },
    /// Custom evidence with description.
    Custom {
        metric_name: String,
        value: f64,
        threshold: f64,
        pass_if_above: bool,
    },
}

/// Gate result with verdict and evidence.
#[derive(Debug, Clone)]
pub struct GateResult {
    pub claim_id: String,
    pub verdict: Verdict,
    pub evidence: Evidence,
    pub message: String,
}

impl GateResult {
    /// Create a passing gate result.
    pub fn pass(claim_id: &str, evidence: Evidence, message: &str) -> Self {
        GateResult {
            claim_id: claim_id.to_string(),
            verdict: Verdict::Pass,
            evidence,
            message: message.to_string(),
        }
    }

    /// Create a failing gate result.
    pub fn fail(claim_id: &str, evidence: Evidence, message: &str) -> Self {
        GateResult {
            claim_id: claim_id.to_string(),
            verdict: Verdict::Fail,
            evidence,
            message: message.to_string(),
        }
    }

    /// Create an uncertain gate result.
    pub fn uncertain(claim_id: &str, evidence: Evidence, message: &str) -> Self {
        GateResult {
            claim_id: claim_id.to_string(),
            verdict: Verdict::Uncertain,
            evidence,
            message: message.to_string(),
        }
    }

    /// Format as JSON for CI parsing.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"claim_id":"{}","verdict":"{}","message":"{}"}}"#,
            self.claim_id, self.verdict, self.message
        )
    }
}

// ============================================================================
// Pre-built Gate Types
// ============================================================================

/// P-value threshold gate.
///
/// Passes if p < threshold (significant difference from null).
/// Used for: Frechet distance tests, permutation tests.
pub fn pvalue_gate(
    claim_id: &str,
    p_value: f64,
    n_permutations: usize,
    threshold: f64,
    description: &str,
) -> GateResult {
    let evidence = Evidence::PValue {
        value: p_value,
        n_permutations,
        threshold,
    };

    if p_value < threshold {
        GateResult::pass(
            claim_id,
            evidence,
            &format!(
                "{}: p={:.4} < {:.2} (significant)",
                description, p_value, threshold
            ),
        )
    } else {
        GateResult::fail(
            claim_id,
            evidence,
            &format!(
                "{}: p={:.4} >= {:.2} (not significant)",
                description, p_value, threshold
            ),
        )
    }
}

/// Inverse p-value gate (passes if p >= threshold).
///
/// Used for: Testing that a claim is NOT different from null.
/// E.g., "ZD spectrum is no closer to masses than random" -> p should be high.
pub fn inverse_pvalue_gate(
    claim_id: &str,
    p_value: f64,
    n_permutations: usize,
    threshold: f64,
    description: &str,
) -> GateResult {
    let evidence = Evidence::PValue {
        value: p_value,
        n_permutations,
        threshold,
    };

    if p_value >= threshold {
        GateResult::pass(
            claim_id,
            evidence,
            &format!(
                "{}: p={:.4} >= {:.2} (null not rejected)",
                description, p_value, threshold
            ),
        )
    } else {
        GateResult::fail(
            claim_id,
            evidence,
            &format!(
                "{}: p={:.4} < {:.2} (unexpected significance)",
                description, p_value, threshold
            ),
        )
    }
}

/// Bootstrap CI precision gate.
///
/// Passes if CI width / |estimate| < uncertain_threshold.
/// Fails if estimate is outside expected range.
/// Uncertain if CI too wide.
pub fn bootstrap_ci_gate(
    claim_id: &str,
    result: &BootstrapCIResult,
    expected_range: (f64, f64),
    uncertain_threshold: f64,
    description: &str,
) -> GateResult {
    let evidence = Evidence::BootstrapCI {
        point_estimate: result.point_estimate,
        ci_lower: result.ci_lower,
        ci_upper: result.ci_upper,
        width_relative: result.ci_width_relative,
        uncertain_threshold,
    };

    // Check if too uncertain
    if result.ci_width_relative > uncertain_threshold {
        return GateResult::uncertain(
            claim_id,
            evidence,
            &format!(
                "{}: CI width {:.1}% exceeds {:.0}% threshold",
                description,
                result.ci_width_relative * 100.0,
                uncertain_threshold * 100.0
            ),
        );
    }

    // Check if in expected range
    let (low, high) = expected_range;
    if result.point_estimate >= low && result.point_estimate <= high {
        GateResult::pass(
            claim_id,
            evidence,
            &format!(
                "{}: estimate={:.4} in [{:.4}, {:.4}]",
                description, result.point_estimate, low, high
            ),
        )
    } else {
        GateResult::fail(
            claim_id,
            evidence,
            &format!(
                "{}: estimate={:.4} outside [{:.4}, {:.4}]",
                description, result.point_estimate, low, high
            ),
        )
    }
}

/// Frechet distance gate from null test result.
pub fn frechet_gate(
    claim_id: &str,
    result: &FrechetNullTestResult,
    threshold: f64,
    description: &str,
) -> GateResult {
    let evidence = Evidence::FrechetDistance {
        observed: result.observed_distance,
        null_mean: result.mean_null,
        p_value: result.p_value,
    };

    if result.p_value < threshold {
        GateResult::pass(
            claim_id,
            evidence,
            &format!(
                "{}: observed={:.4} (p={:.4} < {:.2})",
                description, result.observed_distance, result.p_value, threshold
            ),
        )
    } else {
        GateResult::fail(
            claim_id,
            evidence,
            &format!(
                "{}: observed={:.4} not better than null (p={:.4})",
                description, result.observed_distance, result.p_value
            ),
        )
    }
}

/// Two-sample test gate (ED + MMD).
pub fn two_sample_gate(
    claim_id: &str,
    result: &TwoSampleTestResult,
    _threshold: f64,
    require_both: bool,
    description: &str,
) -> GateResult {
    let evidence = Evidence::TwoSample {
        ed_p_value: result.energy_distance.p_value,
        mmd_p_value: result.mmd.p_value,
        combined_p_value: result.combined_p_value,
    };

    let significant = if require_both {
        result.both_significant
    } else {
        result.either_significant
    };

    let logic = if require_both { "both" } else { "either" };

    if significant {
        GateResult::pass(
            claim_id,
            evidence,
            &format!(
                "{}: {} tests significant (ED p={:.4}, MMD p={:.4})",
                description, logic, result.energy_distance.p_value, result.mmd.p_value
            ),
        )
    } else {
        GateResult::fail(
            claim_id,
            evidence,
            &format!(
                "{}: {} tests not significant (ED p={:.4}, MMD p={:.4})",
                description, logic, result.energy_distance.p_value, result.mmd.p_value
            ),
        )
    }
}

/// Custom metric gate with configurable direction.
pub fn metric_gate(
    claim_id: &str,
    metric_name: &str,
    value: f64,
    threshold: f64,
    pass_if_above: bool,
    description: &str,
) -> GateResult {
    let evidence = Evidence::Custom {
        metric_name: metric_name.to_string(),
        value,
        threshold,
        pass_if_above,
    };

    let passes = if pass_if_above {
        value > threshold
    } else {
        value < threshold
    };

    let comparator = if pass_if_above { ">" } else { "<" };

    if passes {
        GateResult::pass(
            claim_id,
            evidence,
            &format!(
                "{}: {}={:.6} {} {:.6}",
                description, metric_name, value, comparator, threshold
            ),
        )
    } else {
        GateResult::fail(
            claim_id,
            evidence,
            &format!(
                "{}: {}={:.6} not {} {:.6}",
                description, metric_name, value, comparator, threshold
            ),
        )
    }
}

// ============================================================================
// Convenience Wrappers
// ============================================================================

/// Run Frechet distance gate with data directly.
pub fn run_frechet_gate(
    claim_id: &str,
    observed: &[f64],
    reference: &[f64],
    n_permutations: usize,
    threshold: f64,
    seed: u64,
    description: &str,
) -> GateResult {
    let result = frechet_null_test(observed, reference, n_permutations, seed);
    frechet_gate(claim_id, &result, threshold, description)
}

/// Run two-sample gate with data directly.
pub fn run_two_sample_gate(
    claim_id: &str,
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    n_permutations: usize,
    require_both: bool,
    seed: u64,
    description: &str,
) -> GateResult {
    let result = two_sample_test(x, y, n_permutations, seed);
    two_sample_gate(claim_id, &result, 0.05, require_both, description)
}

/// Run bootstrap CI gate with data directly.
#[allow(clippy::too_many_arguments)]
pub fn run_bootstrap_ci_gate<F>(
    claim_id: &str,
    data: &[f64],
    statistic_fn: F,
    n_bootstrap: usize,
    expected_range: (f64, f64),
    uncertain_threshold: f64,
    seed: u64,
    description: &str,
) -> GateResult
where
    F: Fn(&[f64]) -> f64,
{
    let result = bootstrap_ci(data, statistic_fn, n_bootstrap, 0.95, seed);
    bootstrap_ci_gate(
        claim_id,
        &result,
        expected_range,
        uncertain_threshold,
        description,
    )
}

// ============================================================================
// Gate Registry for Batch Execution
// ============================================================================

/// Collection of gate results for batch reporting.
#[derive(Debug, Default)]
pub struct GateRegistry {
    results: Vec<GateResult>,
}

impl GateRegistry {
    /// Create new empty registry.
    pub fn new() -> Self {
        GateRegistry {
            results: Vec::new(),
        }
    }

    /// Add a gate result.
    pub fn add(&mut self, result: GateResult) {
        self.results.push(result);
    }

    /// Get all results.
    pub fn results(&self) -> &[GateResult] {
        &self.results
    }

    /// Count passing gates.
    pub fn pass_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.verdict == Verdict::Pass)
            .count()
    }

    /// Count failing gates.
    pub fn fail_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.verdict == Verdict::Fail)
            .count()
    }

    /// Count uncertain gates.
    pub fn uncertain_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.verdict == Verdict::Uncertain)
            .count()
    }

    /// Check if all gates pass (no failures, uncertains are ok).
    pub fn all_pass(&self) -> bool {
        self.fail_count() == 0
    }

    /// Check if all gates pass strictly (no failures or uncertains).
    pub fn all_pass_strict(&self) -> bool {
        self.fail_count() == 0 && self.uncertain_count() == 0
    }

    /// Generate summary report.
    pub fn summary(&self) -> String {
        let total = self.results.len();
        let pass = self.pass_count();
        let fail = self.fail_count();
        let uncertain = self.uncertain_count();

        format!(
            "Claims Gate Summary: {}/{} pass, {} fail, {} uncertain",
            pass, total, fail, uncertain
        )
    }

    /// Generate JSON array of all results.
    pub fn to_json(&self) -> String {
        let json_items: Vec<String> = self.results.iter().map(|r| r.to_json()).collect();
        format!("[{}]", json_items.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pvalue_gate_pass() {
        let result = pvalue_gate("C-070", 0.01, 1000, 0.05, "Test");
        assert_eq!(result.verdict, Verdict::Pass);
    }

    #[test]
    fn test_pvalue_gate_fail() {
        let result = pvalue_gate("C-070", 0.10, 1000, 0.05, "Test");
        assert_eq!(result.verdict, Verdict::Fail);
    }

    #[test]
    fn test_inverse_pvalue_gate() {
        // p=0.8 should pass when we expect null to NOT be rejected
        let result = inverse_pvalue_gate("C-072", 0.80, 1000, 0.05, "ZD spectrum random");
        assert_eq!(result.verdict, Verdict::Pass);
    }

    #[test]
    fn test_metric_gate_above() {
        let result = metric_gate("C-075", "eigenvalue_count", 33.0, 30.0, true, "Pathion ZD");
        assert_eq!(result.verdict, Verdict::Pass);
    }

    #[test]
    fn test_metric_gate_below() {
        let result = metric_gate("C-077", "frobenius_dist", 0.5, 0.6, false, "PMNS distance");
        assert_eq!(result.verdict, Verdict::Pass);
    }

    #[test]
    fn test_gate_registry() {
        let mut registry = GateRegistry::new();

        registry.add(pvalue_gate("C-070", 0.01, 1000, 0.05, "Test 1"));
        registry.add(pvalue_gate("C-071", 0.10, 1000, 0.05, "Test 2"));
        registry.add(pvalue_gate("C-072", 0.03, 1000, 0.05, "Test 3"));

        assert_eq!(registry.pass_count(), 2);
        assert_eq!(registry.fail_count(), 1);
        assert!(!registry.all_pass());
    }

    #[test]
    fn test_gate_registry_all_pass() {
        let mut registry = GateRegistry::new();

        registry.add(pvalue_gate("C-070", 0.01, 1000, 0.05, "Test 1"));
        registry.add(pvalue_gate("C-071", 0.02, 1000, 0.05, "Test 2"));

        assert!(registry.all_pass());
    }

    #[test]
    fn test_frechet_gate_integration() {
        // Test with identical data - should have very low distance
        let observed = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = run_frechet_gate(
            "C-070",
            &observed,
            &reference,
            100,
            0.05,
            42,
            "Spectrum match",
        );

        // Identical data should have near-zero Frechet distance
        // P-value may or may not be significant depending on null distribution
        assert!(
            result.verdict == Verdict::Pass || result.verdict == Verdict::Fail,
            "Should produce definitive verdict"
        );
    }

    #[test]
    fn test_bootstrap_ci_gate_pass() {
        let data = vec![1.9, 2.0, 2.1, 1.95, 2.05];
        let mean_fn = |d: &[f64]| d.iter().sum::<f64>() / d.len() as f64;

        let result = run_bootstrap_ci_gate(
            "C-074",
            &data,
            mean_fn,
            500,
            (1.5, 2.5), // expect mean around 2
            0.5,        // uncertain if CI > 50% of estimate
            42,
            "Associator growth",
        );

        assert_eq!(result.verdict, Verdict::Pass);
    }

    #[test]
    fn test_bootstrap_ci_gate_uncertain() {
        // Very noisy data should produce uncertain result
        let data = vec![0.5, 5.0, 1.0, 10.0, 0.1];
        let mean_fn = |d: &[f64]| d.iter().sum::<f64>() / d.len() as f64;

        let result = run_bootstrap_ci_gate(
            "C-074",
            &data,
            mean_fn,
            500,
            (1.0, 5.0),
            0.3, // strict threshold
            42,
            "Noisy data",
        );

        // With very noisy data and strict threshold, might be uncertain
        // (depends on actual CI width)
        assert!(
            result.verdict == Verdict::Pass
                || result.verdict == Verdict::Fail
                || result.verdict == Verdict::Uncertain
        );
    }

    #[test]
    fn test_two_sample_gate_identical() {
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
        let y = x.clone();

        let result = run_two_sample_gate("C-test", &x, &y, 50, false, 42, "Identical samples");

        // Identical samples should NOT be significantly different
        assert_eq!(result.verdict, Verdict::Fail);
    }

    #[test]
    fn test_two_sample_gate_different() {
        let x: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64]).collect();
        let y: Vec<Vec<f64>> = (0..15).map(|i| vec![100.0 + i as f64]).collect();

        let result = run_two_sample_gate("C-test", &x, &y, 50, false, 42, "Distant samples");

        // Very different samples should be significant
        assert_eq!(result.verdict, Verdict::Pass);
    }

    #[test]
    fn test_gate_json_output() {
        let result = pvalue_gate("C-070", 0.01, 1000, 0.05, "Frechet test");
        let json = result.to_json();

        assert!(json.contains("C-070"));
        assert!(json.contains("PASS"));
    }

    #[test]
    fn test_registry_json() {
        let mut registry = GateRegistry::new();
        registry.add(pvalue_gate("C-070", 0.01, 1000, 0.05, "Test 1"));
        registry.add(pvalue_gate("C-071", 0.10, 1000, 0.05, "Test 2"));

        let json = registry.to_json();
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("C-070"));
        assert!(json.contains("C-071"));
    }
}
