//! Competing model BIC comparison for QGP R_AA predictions.
//!
//! Compares the Arleo-Falmagne scaling ansatz (2 params: beta, K) against:
//! - CUJET3.0 (Xu, Liao, Gyulassy, JHEP02(2016)169): ~5 free parameters
//! - Fractional Langevin (arXiv:2401.03757, PRD 109 114004): ~3 free parameters
//!
//! BIC = chi2 + k * ln(n) where k = number of free parameters, n = data points.
//! Lower BIC is preferred. Delta-BIC > 10 is "very strong" evidence (Kass & Raftery).

/// Piecewise-linear interpolation.
///
/// Given sorted (xs, ys) with xs strictly increasing, returns the interpolated
/// value at `x_query`. Returns `None` if `x_query` is outside the range of `xs`
/// (no extrapolation -- this is a deliberate guard against index misalignment).
pub fn linear_interp(xs: &[f64], ys: &[f64], x_query: f64) -> Option<f64> {
    assert_eq!(xs.len(), ys.len(), "xs and ys must have equal length");
    if xs.len() < 2 {
        return None;
    }
    if x_query < xs[0] || x_query > xs[xs.len() - 1] {
        return None;
    }
    // Binary search for the interval
    let idx = match xs.binary_search_by(|x| x.partial_cmp(&x_query).unwrap()) {
        Ok(i) => return Some(ys[i]), // exact match
        Err(i) => i,
    };
    if idx == 0 || idx >= xs.len() {
        return None;
    }
    let x0 = xs[idx - 1];
    let x1 = xs[idx];
    let y0 = ys[idx - 1];
    let y1 = ys[idx];
    let t = (x_query - x0) / (x1 - x0);
    Some(y0 + t * (y1 - y0))
}

/// Collision and observable identity required before a physical comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferencePopulation {
    pub collision_system: &'static str,
    pub energy_gev: u32,
    pub species: &'static str,
    pub centrality_percent: (u8, u8),
}

/// Admit physical model ranking only for the same observable population.
/// Covariance provenance and fitted parameter counts require additional checks.
pub fn require_matched_population(
    measured: &ReferencePopulation,
    predicted: &ReferencePopulation,
) -> Result<(), String> {
    for population in [measured, predicted] {
        if population.collision_system.trim().is_empty()
            || population.species.trim().is_empty()
            || population.energy_gev == 0
            || population.centrality_percent.0 >= population.centrality_percent.1
            || population.centrality_percent.1 > 100
        {
            return Err("reference admission failed: invalid population identity".to_string());
        }
    }
    let mut mismatches = Vec::new();
    if measured.collision_system != predicted.collision_system {
        mismatches.push("collision system");
    }
    if measured.energy_gev != predicted.energy_gev {
        mismatches.push("collision energy");
    }
    if measured.species != predicted.species {
        mismatches.push("particle species");
    }
    if measured.centrality_percent != predicted.centrality_percent {
        mismatches.push("centrality");
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "reference admission failed: {}",
            mismatches.join(", ")
        ))
    }
}

/// Check the declared populations used by the retained three-curve comparison.
pub fn retained_bic_population_admission() -> Result<(), String> {
    let measured = ReferencePopulation {
        collision_system: "Pb-Pb",
        energy_gev: 5020,
        species: "charged hadron",
        centrality_percent: (0, 5),
    };
    let competitors = [
        (
            "CUJET3.0",
            ReferencePopulation {
                energy_gev: 2760,
                ..measured.clone()
            },
        ),
        (
            "fractional Langevin",
            ReferencePopulation {
                species: "D meson",
                centrality_percent: (0, 10),
                ..measured.clone()
            },
        ),
    ];
    let failures: Vec<_> = competitors
        .iter()
        .filter_map(|(name, population)| {
            require_matched_population(&measured, population)
                .err()
                .map(|error| format!("{name}: {error}"))
        })
        .collect();
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

/// A digitized theoretical R_AA prediction curve.
#[derive(Debug, Clone)]
pub struct TheoreticalRaaCurve {
    /// Model name.
    pub name: &'static str,
    /// Number of free parameters in the model.
    pub n_params: usize,
    /// pT values (GeV), sorted ascending.
    pub pt: Vec<f64>,
    /// R_AA central values at each pT.
    pub raa: Vec<f64>,
}

impl TheoreticalRaaCurve {
    /// Interpolate the theoretical R_AA at a given pT.
    pub fn raa_at(&self, pt_query: f64) -> Option<f64> {
        linear_interp(&self.pt, &self.raa, pt_query)
    }

    /// The pT range covered by this curve.
    pub fn pt_range(&self) -> (f64, f64) {
        (self.pt[0], self.pt[self.pt.len() - 1])
    }
}

/// Measured R_AA data point for BIC comparison.
#[derive(Debug, Clone)]
pub struct MeasuredRaaPoint {
    /// pT bin center (GeV).
    pub pt: f64,
    /// Measured R_AA.
    pub raa: f64,
    /// Total uncertainty (stat + syst in quadrature).
    pub total_err: f64,
}

/// Result of a BIC comparison between models.
#[derive(Debug, Clone)]
pub struct BicResult {
    /// Model name.
    pub model_name: &'static str,
    /// Number of free parameters.
    pub n_params: usize,
    /// Chi2 against measured data.
    pub chi2: f64,
    /// Number of data points used.
    pub n_points: usize,
    /// BIC = chi2 + k * ln(n).
    pub bic: f64,
    /// Number of measured points excluded due to extrapolation guard.
    pub n_excluded: usize,
}

/// Compute chi2 and BIC for a theoretical curve against measured data.
///
/// Only uses measured points where the theoretical curve covers the pT range.
/// Points outside the theoretical curve's pT range are excluded and counted
/// in `n_excluded`.
pub fn compute_bic(theory: &TheoreticalRaaCurve, data: &[MeasuredRaaPoint]) -> BicResult {
    let (pt_lo, pt_hi) = theory.pt_range();
    let mut chi2 = 0.0;
    let mut n_used = 0usize;
    let mut n_excluded = 0usize;

    for d in data {
        if d.pt < pt_lo || d.pt > pt_hi {
            n_excluded += 1;
            continue;
        }
        if let Some(raa_theory) = theory.raa_at(d.pt) {
            let diff = d.raa - raa_theory;
            chi2 += (diff * diff) / (d.total_err * d.total_err);
            n_used += 1;
        } else {
            n_excluded += 1;
        }
    }

    let bic = if n_used > 0 {
        chi2 + theory.n_params as f64 * (n_used as f64).ln()
    } else {
        f64::INFINITY
    };

    BicResult {
        model_name: theory.name,
        n_params: theory.n_params,
        chi2,
        n_points: n_used,
        bic,
        n_excluded,
    }
}

/// Compare multiple models via BIC, returning results sorted by BIC (best first).
pub fn compare_models(
    theories: &[TheoreticalRaaCurve],
    data: &[MeasuredRaaPoint],
) -> Vec<BicResult> {
    let mut results: Vec<BicResult> = theories.iter().map(|t| compute_bic(t, data)).collect();
    results.sort_by(|a, b| a.bic.partial_cmp(&b.bic).unwrap());
    results
}

// ============================================================================
// Digitized R_AA predictions from published figures
// ============================================================================

/// CUJET3.0 R_AA prediction for Pb-Pb 2.76 TeV, 0-5% centrality.
///
/// Digitized from Xu, Liao, Gyulassy, JHEP02(2016)169, Fig. 4 (left panel).
/// CUJET3.0 combines perturbative QCD radiative + collisional energy loss
/// with a non-perturbative magnetic monopole component near Tc.
/// Free parameters: alpha_s(max), mu_M/T, c_M, f_E, f_M (~5 effective).
pub fn cujet3_pbpb_2760_0_5() -> TheoreticalRaaCurve {
    // Digitized from JHEP02(2016)169 Fig. 4 (left), Pb-Pb 2.76 TeV, 0-5%
    // pT range: 5-100 GeV
    TheoreticalRaaCurve {
        name: "CUJET3.0 Pb-Pb 2.76 TeV 0-5%",
        n_params: 5,
        pt: vec![
            5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0,
        ],
        raa: vec![
            0.13, 0.16, 0.21, 0.29, 0.34, 0.40, 0.44, 0.47, 0.50, 0.54, 0.57,
        ],
    }
}

/// CUJET3.0 R_AA prediction for Pb-Pb 2.76 TeV, 10-20% centrality.
///
/// Digitized from Xu, Liao, Gyulassy, JHEP02(2016)169, Fig. 4 (right panel).
pub fn cujet3_pbpb_2760_10_20() -> TheoreticalRaaCurve {
    TheoreticalRaaCurve {
        name: "CUJET3.0 Pb-Pb 2.76 TeV 10-20%",
        n_params: 5,
        pt: vec![
            5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0,
        ],
        raa: vec![
            0.18, 0.22, 0.28, 0.35, 0.40, 0.47, 0.51, 0.54, 0.56, 0.60, 0.62,
        ],
    }
}

/// CUJET3.0 R_AA prediction for Pb-Pb 2.76 TeV, 30-40% centrality.
///
/// Digitized from Xu, Liao, Gyulassy, JHEP02(2016)169, Fig. 5.
pub fn cujet3_pbpb_2760_30_40() -> TheoreticalRaaCurve {
    TheoreticalRaaCurve {
        name: "CUJET3.0 Pb-Pb 2.76 TeV 30-40%",
        n_params: 5,
        pt: vec![
            5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0,
        ],
        raa: vec![
            0.30, 0.35, 0.41, 0.49, 0.53, 0.58, 0.62, 0.65, 0.67, 0.70, 0.72,
        ],
    }
}

/// Fractional Langevin R_AA prediction for Pb-Pb 5.02 TeV, 0-10% centrality.
///
/// Digitized from Feal, Salgado, arXiv:2401.03757 (PRD 109, 114004), Fig. 3.
/// Fractional Langevin model: heavy quark anomalous diffusion with fractional
/// index alpha (superdiffusion). The R_AA is suppressed due to enhanced
/// energy loss from non-Markovian memory effects.
/// Free parameters: alpha (fractional index), D_s (spatial diffusion), T_c (~3 effective).
pub fn langevin_pbpb_5020_0_10() -> TheoreticalRaaCurve {
    // Digitized from arXiv:2401.03757 Fig. 3, Pb-Pb 5.02 TeV, 0-10%
    // This is for D mesons (heavy quarks), not light hadrons.
    // pT range: 2-40 GeV
    TheoreticalRaaCurve {
        name: "Frac. Langevin Pb-Pb 5.02 TeV 0-10%",
        n_params: 3,
        pt: vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0,
        ],
        raa: vec![
            0.60, 0.38, 0.28, 0.22, 0.20, 0.21, 0.24, 0.28, 0.33, 0.40, 0.46, 0.50, 0.54, 0.57,
        ],
    }
}

/// Fractional Langevin R_AA prediction for Pb-Pb 5.02 TeV, 30-50% centrality.
///
/// Digitized from arXiv:2401.03757, Fig. 3 (peripheral bin).
pub fn langevin_pbpb_5020_30_50() -> TheoreticalRaaCurve {
    TheoreticalRaaCurve {
        name: "Frac. Langevin Pb-Pb 5.02 TeV 30-50%",
        n_params: 3,
        pt: vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0,
        ],
        raa: vec![
            0.72, 0.55, 0.46, 0.42, 0.41, 0.43, 0.47, 0.50, 0.55, 0.62, 0.67, 0.71, 0.74, 0.76,
        ],
    }
}

/// Generate the same declared shift ansatz fitted by `epsilon_fit::extract_epsilon`.
/// Sharing `quenching::r_aa_model` keeps fitted and scored objectives consistent.
/// The comparison counts one fitted epsilon with fixed spectral index; source
/// conformance and matched-population likelihood admission remain separate.
pub fn arleo_falmagne_raa(
    epsilon_bar: f64,
    n_spectral: f64,
    pt_points: &[f64],
) -> TheoreticalRaaCurve {
    let raa: Vec<f64> = pt_points
        .iter()
        .map(|&pt| crate::quenching::r_aa_model(pt, epsilon_bar, n_spectral))
        .collect();
    TheoreticalRaaCurve {
        name: "Arleo-Falmagne",
        n_params: 1,
        pt: pt_points.to_vec(),
        raa,
    }
}

/// Collect all available CUJET3.0 curves.
pub fn all_cujet3_curves() -> Vec<TheoreticalRaaCurve> {
    vec![
        cujet3_pbpb_2760_0_5(),
        cujet3_pbpb_2760_10_20(),
        cujet3_pbpb_2760_30_40(),
    ]
}

/// Collect all available fractional Langevin curves.
pub fn all_langevin_curves() -> Vec<TheoreticalRaaCurve> {
    vec![langevin_pbpb_5020_0_10(), langevin_pbpb_5020_30_50()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interp_basic() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [10.0, 20.0, 30.0, 40.0, 50.0];

        // Exact points
        assert!((linear_interp(&xs, &ys, 1.0).unwrap() - 10.0).abs() < 1e-12);
        assert!((linear_interp(&xs, &ys, 3.0).unwrap() - 30.0).abs() < 1e-12);

        // Midpoints
        assert!((linear_interp(&xs, &ys, 1.5).unwrap() - 15.0).abs() < 1e-12);
        assert!((linear_interp(&xs, &ys, 2.7).unwrap() - 27.0).abs() < 1e-12);

        // Out of range -> None
        assert!(linear_interp(&xs, &ys, 0.5).is_none());
        assert!(linear_interp(&xs, &ys, 5.5).is_none());
    }

    #[test]
    fn test_linear_interp_nonuniform() {
        let xs = [1.0, 3.0, 10.0];
        let ys = [0.0, 4.0, 18.0];

        // Between 1 and 3: slope = 4/2 = 2
        assert!((linear_interp(&xs, &ys, 2.0).unwrap() - 2.0).abs() < 1e-12);

        // Between 3 and 10: slope = 14/7 = 2
        assert!((linear_interp(&xs, &ys, 6.5).unwrap() - 11.0).abs() < 1e-12);
    }

    #[test]
    fn test_bic_computation() {
        // Synthetic: perfect match (chi2 = 0)
        let theory = TheoreticalRaaCurve {
            name: "perfect",
            n_params: 2,
            pt: vec![5.0, 10.0, 20.0, 40.0],
            raa: vec![0.2, 0.3, 0.5, 0.7],
        };
        let data: Vec<MeasuredRaaPoint> = theory
            .pt
            .iter()
            .zip(theory.raa.iter())
            .map(|(&pt, &raa)| MeasuredRaaPoint {
                pt,
                raa,
                total_err: 0.01,
            })
            .collect();

        let result = compute_bic(&theory, &data);
        assert!(result.chi2 < 1e-20, "chi2 should be ~0 for perfect match");
        assert_eq!(result.n_points, 4);
        assert_eq!(result.n_excluded, 0);
        // BIC = 0 + 2 * ln(4) = 2.77
        assert!((result.bic - 2.0 * (4.0_f64).ln()).abs() < 1e-6);
    }

    #[test]
    fn test_bic_excludes_out_of_range() {
        let theory = TheoreticalRaaCurve {
            name: "narrow",
            n_params: 1,
            pt: vec![10.0, 20.0],
            raa: vec![0.3, 0.5],
        };
        let data = vec![
            MeasuredRaaPoint {
                pt: 5.0,
                raa: 0.2,
                total_err: 0.01,
            }, // below range
            MeasuredRaaPoint {
                pt: 15.0,
                raa: 0.4,
                total_err: 0.01,
            }, // in range
            MeasuredRaaPoint {
                pt: 30.0,
                raa: 0.6,
                total_err: 0.01,
            }, // above range
        ];

        let result = compute_bic(&theory, &data);
        assert_eq!(result.n_points, 1);
        assert_eq!(result.n_excluded, 2);
    }

    #[test]
    fn test_cujet3_curves_valid() {
        let curves = all_cujet3_curves();
        assert_eq!(curves.len(), 3);
        for c in &curves {
            assert_eq!(c.pt.len(), c.raa.len());
            assert!(c.pt.len() >= 10);
            assert_eq!(c.n_params, 5);
            // R_AA should be in (0, 1)
            for &r in &c.raa {
                assert!(r > 0.0 && r < 1.0, "R_AA out of range: {}", r);
            }
        }
    }

    #[test]
    fn test_langevin_curves_valid() {
        let curves = all_langevin_curves();
        assert_eq!(curves.len(), 2);
        for c in &curves {
            assert_eq!(c.pt.len(), c.raa.len());
            assert!(c.pt.len() >= 10);
            assert_eq!(c.n_params, 3);
            for &r in &c.raa {
                assert!(r > 0.0 && r < 1.0, "R_AA out of range: {}", r);
            }
        }
    }

    #[test]
    fn test_arleo_falmagne_raa_model() {
        let eps = 3.0;
        let n = 6.0;
        let pts = vec![5.0, 10.0, 20.0, 50.0];
        let curve = arleo_falmagne_raa(eps, n, &pts);

        assert_eq!(curve.n_params, 1);
        for (i, &pt) in pts.iter().enumerate() {
            let expected = ((pt - eps) / pt).powf(n - 1.0);
            assert!(
                (curve.raa[i] - expected).abs() < 1e-12,
                "Mismatch at pT={}: {} vs {}",
                pt,
                curve.raa[i],
                expected
            );
        }
    }

    #[test]
    fn test_model_comparison_ordering() {
        // Model with lower BIC should come first
        let good_model = TheoreticalRaaCurve {
            name: "good",
            n_params: 2,
            pt: vec![5.0, 10.0, 20.0],
            raa: vec![0.2, 0.3, 0.5],
        };
        let bad_model = TheoreticalRaaCurve {
            name: "bad",
            n_params: 5,
            pt: vec![5.0, 10.0, 20.0],
            raa: vec![0.5, 0.5, 0.5], // poor fit
        };

        let data = vec![
            MeasuredRaaPoint {
                pt: 5.0,
                raa: 0.2,
                total_err: 0.01,
            },
            MeasuredRaaPoint {
                pt: 10.0,
                raa: 0.3,
                total_err: 0.01,
            },
            MeasuredRaaPoint {
                pt: 20.0,
                raa: 0.5,
                total_err: 0.01,
            },
        ];

        let results = compare_models(&[good_model, bad_model], &data);
        assert_eq!(results[0].model_name, "good");
        assert!(results[0].bic < results[1].bic);
    }
}
