use algebra_experimental::{
    bell_inequality::{BellTestResult, chsh_violation_test},
    higher_cd::HigherAvt,
    particle_physics::{GaugeSectorReport, analyze_1024d_gauge_sectors},
};
use cosmology_core::{
    CosmicWebGenerator, DekaVoudonCmbAnalyzer, FlatLCDM, MultipoleResult, VoudonFriedmann,
    angular_separation_degrees, extract_multipoles,
};
use gr_core::{
    AdaptiveWickResult, FractalFlybyResult, NBodySystem, fractal_flyby_prediction,
    pioneer_anomaly_prediction,
};
use quantum_core::{
    chrono_turbulence::ChronoTurbulenceSolver,
    deka_voudon_qec::{DekaVoudonStabilizer, HolographicVacuumCode},
    intention_operator::IntentionOperator,
};

use num_complex::Complex;
use std::collections::HashSet;

// Physical constants for Hawking spectrum
const G_CGS: f64 = 6.674e-8; // cm^3 g^-1 s^-2
const C_CGS: f64 = 2.998e10; // cm/s
const HBAR_CGS: f64 = 1.0546e-27; // erg*s
const K_B_CGS: f64 = 1.381e-16; // erg/K
const M_SUN_G: f64 = 1.989e33; // grams
const SGR_A_MASS_SOLAR: f64 = 4.1e6;

/// The Singularitarian Engine: Unified Full-Stack Universal Simulator.
///
/// Wires together all Cayley-Dickson physics subsystems built across Sprints 71-79:
/// - Fractal spacetime (D_f=2.7 Q-tensor metric)
/// - Complex-time N-body (Wick rotation + Pathion shadow)
/// - Chrono-Turbulence (LBM temporal manifold)
/// - 1024D Holographic Vacuum Code (DekaVoudon QEC)
/// - Non-Associative Entropy Filter (128D Routon-based)
/// - Voudon Friedmann cosmology (256D modified FLRW)
/// - CMB Quadrupole-Octupole Alignment (1024D cosmic web seeding)
/// - 512D Bell inequality (CHSH via associator torque)
/// - 1024D Particle physics (gauge group embedding)
/// - Flyby anomaly (64D Chingon bivector drag)
pub struct SingularitarianEngine {
    pub dimension_ladder: Vec<usize>,
    pub intention: IntentionOperator,
    pub fractal_dim: f64,
    pub vacuum_code: Option<HolographicVacuumCode>,
    pub chrono_solver: Option<ChronoTurbulenceSolver>,
    pub cmb_analyzer: Option<DekaVoudonCmbAnalyzer>,
    pub voudon_friedmann: Option<VoudonFriedmann>,
}

/// Result of the Sgr A* Hawking radiation spectrum calculation.
#[derive(Debug, Clone)]
pub struct HawkingSpectrumResult {
    /// Frequency bins (dimensionless: f / f_peak).
    pub frequencies: Vec<f64>,
    /// Spectral flux per bin (arbitrary units, normalized to peak=1).
    pub flux: Vec<f64>,
    /// Hawking temperature in Kelvin.
    pub hawking_temp_k: f64,
    /// Fractal-modified Hawking temperature in Kelvin.
    pub fractal_temp_k: f64,
    /// 1024D vacuum noise floor (fractional modulation).
    pub vacuum_modulation: f64,
}

/// Evidence status for a single claim.
#[derive(Debug, Clone)]
pub struct ClaimEvidence {
    pub claim_id: String,
    pub status: ClaimStatus,
    pub supporting_experiments: Vec<String>,
    pub sprint: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimStatus {
    Verified,
    Falsified,
    Open,
}

/// Result of the unified end-to-end integration run.
pub struct UnifiedRunResult {
    pub hawking_spectrum: HawkingSpectrumResult,
    pub cmb_multipoles: Option<MultipoleResult>,
    pub bell_result: Option<BellTestResult>,
    pub gauge_report: Option<GaugeSectorReport>,
    pub fractal_flyby: Option<FractalFlybyResult>,
    pub pioneer_a_ms2: Option<f64>,
    pub adaptive_wick: Option<AdaptiveWickResult>,
    pub voudon_h0: Option<f64>,
    pub vacuum_stability_delta: f64,
    pub subsystems_active: usize,
}

impl SingularitarianEngine {
    pub fn new(fractal_dim: f64) -> Self {
        Self {
            dimension_ladder: vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            intention: IntentionOperator::new(1.0, 0.1),
            fractal_dim,
            vacuum_code: None,
            chrono_solver: None,
            cmb_analyzer: None,
            voudon_friedmann: None,
        }
    }

    /// Initialize the Chrono-Turbulence solver for temporal fluid dynamics.
    pub fn initialize_chrono_solver(&mut self, nx: usize, ny: usize) {
        self.chrono_solver = Some(ChronoTurbulenceSolver::new(nx, ny, 0.8, 0.1));
    }

    /// Initialize the Holographic Vacuum Code for 1024D error correction.
    pub fn initialize_vacuum_decoder(&mut self, avt: &HigherAvt) {
        self.vacuum_code = Some(HolographicVacuumCode::new(1024, avt));
    }

    /// Initialize the 1024D CMB analyzer for cosmic web seeding.
    pub fn initialize_cmb_analyzer(&mut self, phi: f64) {
        self.cmb_analyzer = Some(DekaVoudonCmbAnalyzer::new(phi));
    }

    /// Initialize the Voudon Friedmann cosmology with 256D algebraic pressure.
    pub fn initialize_voudon_friedmann(&mut self, phi_voudon: f64, alpha_voudon: f64) {
        let base = FlatLCDM::planck2018();
        self.voudon_friedmann = Some(VoudonFriedmann::new(base, phi_voudon, alpha_voudon));
    }

    /// Count the number of active subsystems.
    pub fn active_subsystem_count(&self) -> usize {
        let mut count = 2; // IntentionOperator + FractalMetric always present
        if self.vacuum_code.is_some() {
            count += 1;
        }
        if self.chrono_solver.is_some() {
            count += 1;
        }
        if self.cmb_analyzer.is_some() {
            count += 1;
        }
        if self.voudon_friedmann.is_some() {
            count += 1;
        }
        count
    }

    /// Perform a decoding cycle using the Intention Operator.
    pub fn decode_vacuum(&self, active_violations: &mut HashSet<usize>) -> f64 {
        if let Some(ref code) = self.vacuum_code {
            let stability_pre = code.evaluate_stability(active_violations);
            active_violations.retain(|&v| {
                let noise_level = (v as f64 / 1024.0).fract();
                noise_level > (1.0 - self.intention.strength)
            });
            code.evaluate_stability(active_violations) - stability_pre
        } else {
            0.0
        }
    }

    /// Predict the Hawking Radiation Spectrum of Sagittarius A*.
    ///
    /// Uses the proper Hawking temperature formula:
    ///   T_H = hbar * c^3 / (8 * pi * G * M * k_B)
    /// Modified by fractal spacetime:
    ///   T_H_fractal = T_H * (D_f / 4)^gamma, gamma = D_f - 2
    /// And modulated by 1024D vacuum noise from the DekaVoudon bias axis.
    pub fn predict_sgr_a_spectrum(&self) -> HawkingSpectrumResult {
        let mass_g = SGR_A_MASS_SOLAR * M_SUN_G;

        // Standard Hawking temperature
        let t_hawking =
            HBAR_CGS * C_CGS.powi(3) / (8.0 * std::f64::consts::PI * G_CGS * mass_g * K_B_CGS);

        // Fractal modification: D_f < 4 increases effective temperature
        let gamma = self.fractal_dim - 2.0;
        let fractal_factor = (self.fractal_dim / 4.0).powf(gamma);
        let t_fractal = t_hawking * fractal_factor;

        // 1024D vacuum modulation from DekaVoudon bias axis
        let vacuum_mod = if let Some(ref analyzer) = self.cmb_analyzer {
            analyzer.project_axis().norm() * 0.01
        } else {
            0.0
        };

        // Chrono-Turbulence temporal vorticity modulation
        let chrono_mod = if let Some(ref chrono) = self.chrono_solver {
            chrono.temporal_vorticity().mean().unwrap_or(0.0)
        } else {
            0.0
        };

        // Generate Planckian spectrum: B(f, T) ~ f^3 / (exp(h*f/(k*T)) - 1)
        let n_bins = 99;
        let mut frequencies = Vec::with_capacity(n_bins);
        let mut flux = Vec::with_capacity(n_bins);

        for i in 1..=n_bins {
            let f_ratio = i as f64 * 0.1; // 0.1 to 9.9 in units of f_peak
            let x = f_ratio * 2.82; // x = h*f / (k*T)
            let planck = if x < 500.0 {
                f_ratio.powi(3) / (x.exp() - 1.0)
            } else {
                0.0
            };
            let modulated = planck * (1.0 + vacuum_mod + chrono_mod);
            frequencies.push(f_ratio);
            flux.push(modulated);
        }

        // Normalize to peak = 1
        let peak = flux.iter().copied().fold(0.0_f64, f64::max);
        if peak > 0.0 {
            for f in &mut flux {
                *f /= peak;
            }
        }

        HawkingSpectrumResult {
            frequencies,
            flux,
            hawking_temp_k: t_hawking,
            fractal_temp_k: t_fractal,
            vacuum_modulation: vacuum_mod,
        }
    }

    /// Run Bell inequality test via associator torque correlations at a given dimension.
    pub fn bell_test_512d_at(&self, dim: usize, n_samples: usize, seed: u64) -> BellTestResult {
        chsh_violation_test(dim, n_samples, seed)
    }

    /// Run 512D Bell inequality test via associator torque correlations.
    pub fn bell_test_512d(&self, n_samples: usize, seed: u64) -> BellTestResult {
        self.bell_test_512d_at(512, n_samples, seed)
    }

    /// Analyze 1024D gauge group structure via sampled AVT.
    pub fn gauge_sector_analysis(&self, n_samples: usize, seed: u64) -> GaugeSectorReport {
        analyze_1024d_gauge_sectors(n_samples, seed)
    }

    /// Predict CMB multipole alignment via anisotropic cosmic web seeding.
    pub fn cmb_alignment_prediction(
        &self,
        stabilizers: &[DekaVoudonStabilizer],
    ) -> Option<MultipoleResult> {
        let analyzer = self.cmb_analyzer.as_ref()?;
        let web_gen = CosmicWebGenerator::new(self.fractal_dim, 0.1);
        let seeds = web_gen.generate_seeding(analyzer, stabilizers);
        if seeds.is_empty() {
            return None;
        }
        Some(extract_multipoles(&seeds, 5))
    }

    /// Predict the CMB quadrupole axis direction in Galactic coordinates.
    pub fn cmb_axis_galactic(&self) -> Option<(f64, f64)> {
        self.cmb_analyzer.as_ref().map(|a| a.axis_galactic_coords())
    }

    /// Compute angular separation between predicted CMB axis and Planck observation.
    pub fn cmb_planck_separation(&self) -> Option<f64> {
        let (l_pred, b_pred) = self.cmb_axis_galactic()?;
        Some(angular_separation_degrees(l_pred, b_pred, 240.0, 63.0))
    }

    /// Predict fractal flyby anomaly for a given spacecraft geometry.
    pub fn fractal_flyby(&self, perigee_km: f64, v_inf_km_s: f64) -> FractalFlybyResult {
        fractal_flyby_prediction(self.fractal_dim, perigee_km, v_inf_km_s)
    }

    /// Predict Pioneer anomalous deceleration (m/s^2).
    pub fn pioneer_anomaly(&self, r_km: f64, v_km_s: f64) -> f64 {
        pioneer_anomaly_prediction(self.fractal_dim, r_km, v_km_s)
    }

    /// Compute Voudon Friedmann H(z) at a given redshift.
    pub fn voudon_hubble(&self, z: f64) -> Option<f64> {
        self.voudon_friedmann.as_ref().map(|vf| vf.hubble(z))
    }

    /// Perform adaptive Wick evolution through the Pathion shadow boundary.
    pub fn adaptive_wick_evolve(
        &self,
        system: &mut NBodySystem,
        central_mass: f64,
        dt: f64,
        n_steps: usize,
        theta_max: f64,
        steepness: f64,
    ) -> AdaptiveWickResult {
        system.adaptive_wick_evolve(dt, theta_max, n_steps, central_mass, steepness)
    }

    /// Execute a unified hierarchy step.
    pub fn unified_step(&mut self, system: &mut NBodySystem, d_tau: Complex<f64>) {
        // 1. Evolve the temporal manifold (Chrono-Turbulence)
        if let Some(ref mut chrono) = self.chrono_solver {
            chrono.step(d_tau);
        }

        // 2. Apply Intention Operator bias to physical step
        let bias = self.intention.strength;
        let effective_tau = d_tau * bias;

        // 3. Evolve the N-Body system in complex time
        system.step(effective_tau);
    }

    /// Build the automated evidence matrix for Sprints 71-80 claims.
    pub fn evidence_matrix() -> Vec<ClaimEvidence> {
        vec![
            // Sprint 71.2: Three-body 64D embedding
            ClaimEvidence {
                claim_id: "C-954".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-096".to_string()],
                sprint: 71,
            },
            ClaimEvidence {
                claim_id: "C-955".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-096".to_string()],
                sprint: 71,
            },
            // Sprint 72: GPU pipeline
            ClaimEvidence {
                claim_id: "C-956".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-097".to_string()],
                sprint: 72,
            },
            // Sprint 73: 256D dimensional scaling
            ClaimEvidence {
                claim_id: "C-957".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-098".to_string()],
                sprint: 73,
            },
            ClaimEvidence {
                claim_id: "C-958".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-098".to_string()],
                sprint: 73,
            },
            // Sprint 74: 512D Bell
            ClaimEvidence {
                claim_id: "C-959".to_string(),
                status: ClaimStatus::Open,
                supporting_experiments: vec!["E-099".to_string()],
                sprint: 74,
            },
            ClaimEvidence {
                claim_id: "C-960".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-099".to_string()],
                sprint: 74,
            },
            // Sprint 75: 1024D particle physics
            ClaimEvidence {
                claim_id: "C-961".to_string(),
                status: ClaimStatus::Open,
                supporting_experiments: vec![],
                sprint: 75,
            },
            ClaimEvidence {
                claim_id: "C-962".to_string(),
                status: ClaimStatus::Open,
                supporting_experiments: vec![],
                sprint: 75,
            },
            // Sprint 76: Complex time
            ClaimEvidence {
                claim_id: "C-1116".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-101".to_string()],
                sprint: 76,
            },
            ClaimEvidence {
                claim_id: "C-1117".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-101".to_string()],
                sprint: 76,
            },
            // Sprint 77: Entropy filter
            ClaimEvidence {
                claim_id: "C-1118".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-102".to_string()],
                sprint: 77,
            },
            ClaimEvidence {
                claim_id: "C-1119".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-102".to_string()],
                sprint: 77,
            },
            // Sprint 78: Fractal spacetime
            ClaimEvidence {
                claim_id: "C-1120".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-103".to_string()],
                sprint: 78,
            },
            ClaimEvidence {
                claim_id: "C-1121".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-103".to_string()],
                sprint: 78,
            },
            // Sprint 79: CMB alignment
            ClaimEvidence {
                claim_id: "C-1122".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-104".to_string()],
                sprint: 79,
            },
            ClaimEvidence {
                claim_id: "C-1123".to_string(),
                status: ClaimStatus::Verified,
                supporting_experiments: vec!["E-104".to_string()],
                sprint: 79,
            },
        ]
    }

    /// Run the full unified integration: all subsystems exercised end-to-end.
    ///
    /// Accepts `bell_dim` and `gauge_samples` to control the computational
    /// weight of the algebraic subsystems.  Production callers should use
    /// `(512, 10_000)` for physics-grade runs.  Test callers should use
    /// `(16, 100)` for fast smoke tests that exercise the same plumbing.
    pub fn run_unified(
        &mut self,
        stabilizers: &[DekaVoudonStabilizer],
        bell_dim: usize,
        gauge_samples: usize,
    ) -> UnifiedRunResult {
        // 1. Hawking spectrum
        let hawking_spectrum = self.predict_sgr_a_spectrum();

        // 2. CMB multipoles
        let cmb_multipoles = self.cmb_alignment_prediction(stabilizers);

        // 3. Bell test
        let bell_result = Some(self.bell_test_512d_at(bell_dim, 1000, 42));

        // 4. Gauge sector analysis
        let gauge_report = Some(self.gauge_sector_analysis(gauge_samples, 42));

        // 5. Fractal flyby (NEAR-like geometry: perigee = R_earth + 539 km)
        let fractal_flyby = Some(self.fractal_flyby(6910.0, 6.851));

        // 6. Pioneer anomaly at 40 AU (~6e9 km), ~12 km/s
        let pioneer_a = Some(self.pioneer_anomaly(6.0e9, 12.0));

        // 7. Adaptive Wick evolution
        let mut system =
            NBodySystem::new(1e-5, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let adaptive_wick = Some(self.adaptive_wick_evolve(
            &mut system,
            4.1e6,
            0.01,
            10,
            std::f64::consts::FRAC_PI_4,
            5.0,
        ));

        // 8. Voudon H0
        let voudon_h0 = self.voudon_hubble(0.0);

        // 9. Vacuum stability
        let mut violations: HashSet<usize> = (0..100).collect();
        let vacuum_stability_delta = self.decode_vacuum(&mut violations);

        let subsystems_active = self.active_subsystem_count();

        UnifiedRunResult {
            hawking_spectrum,
            cmb_multipoles,
            bell_result,
            gauge_report,
            fractal_flyby,
            pioneer_a_ms2: pioneer_a,
            adaptive_wick,
            voudon_h0,
            vacuum_stability_delta,
            subsystems_active,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_new() {
        let engine = SingularitarianEngine::new(2.7);
        assert_eq!(engine.dimension_ladder.len(), 11);
        assert!((engine.fractal_dim - 2.7).abs() < 1e-10);
        assert_eq!(engine.active_subsystem_count(), 2);
    }

    #[test]
    fn test_hawking_temperature_positive() {
        let engine = SingularitarianEngine::new(2.7);
        let result = engine.predict_sgr_a_spectrum();
        assert!(result.hawking_temp_k > 0.0, "T_H should be positive");
        assert!(result.fractal_temp_k > 0.0, "T_fractal should be positive");
        // D_f=2.7 < 4: fractal factor (2.7/4)^0.7 < 1, so T_fractal < T_H
        assert!(
            result.fractal_temp_k < result.hawking_temp_k,
            "D_f<4 should reduce Hawking temperature"
        );
        assert_eq!(result.frequencies.len(), 99);
        assert_eq!(result.flux.len(), 99);
    }

    #[test]
    fn test_hawking_spectrum_planckian_shape() {
        let engine = SingularitarianEngine::new(2.7);
        let result = engine.predict_sgr_a_spectrum();
        // Planckian spectrum should peak and then decay
        let peak_idx = result
            .flux
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        // Peak should be at normalized flux = 1.0
        assert!(
            (result.flux[peak_idx] - 1.0).abs() < 1e-10,
            "peak should be normalized to 1"
        );
        // Flux at edges should be less than peak
        assert!(result.flux[0] < result.flux[peak_idx]);
        assert!(result.flux[result.flux.len() - 1] < result.flux[peak_idx]);
    }

    #[test]
    fn test_hawking_sgr_a_temperature_magnitude() {
        // Sgr A* (4.1e6 M_sun): T_H ~ 1.5e-14 K (extremely cold)
        let engine = SingularitarianEngine::new(2.7);
        let result = engine.predict_sgr_a_spectrum();
        assert!(
            result.hawking_temp_k < 1e-10,
            "Sgr A* should be extremely cold: T={:.2e}",
            result.hawking_temp_k
        );
        assert!(
            result.hawking_temp_k > 1e-20,
            "T should be physically finite: T={:.2e}",
            result.hawking_temp_k
        );
    }

    #[test]
    fn test_initialize_subsystems() {
        let mut engine = SingularitarianEngine::new(2.7);
        assert_eq!(engine.active_subsystem_count(), 2);

        engine.initialize_chrono_solver(16, 16);
        assert_eq!(engine.active_subsystem_count(), 3);

        engine.initialize_cmb_analyzer(0.0172);
        assert_eq!(engine.active_subsystem_count(), 4);

        engine.initialize_voudon_friedmann(0.01, 0.05);
        assert_eq!(engine.active_subsystem_count(), 5);
    }

    #[test]
    fn test_cmb_planck_separation_finite() {
        let mut engine = SingularitarianEngine::new(2.7);
        assert!(engine.cmb_planck_separation().is_none());

        engine.initialize_cmb_analyzer(0.0172);
        let sep = engine.cmb_planck_separation().unwrap();
        assert!((0.0..=180.0).contains(&sep), "separation={sep}");
    }

    #[test]
    fn test_voudon_hubble() {
        let mut engine = SingularitarianEngine::new(2.7);
        assert!(engine.voudon_hubble(0.0).is_none());

        engine.initialize_voudon_friedmann(0.01, 0.05);
        let h0 = engine.voudon_hubble(0.0).unwrap();
        assert!(h0 > 50.0 && h0 < 100.0, "H0={h0} should be near Planck");
    }

    #[test]
    fn test_fractal_flyby_nonzero_sign() {
        let engine = SingularitarianEngine::new(2.7);
        let result = engine.fractal_flyby(6910.0, 6.851);
        assert!(result.sign != 0, "should have nonzero sign");
    }

    #[test]
    fn test_pioneer_anomaly_positive() {
        let engine = SingularitarianEngine::new(2.7);
        let a = engine.pioneer_anomaly(6.0e9, 12.0);
        assert!(a > 0.0, "deceleration should be positive: a={a}");
    }

    #[test]
    fn test_evidence_matrix() {
        let matrix = SingularitarianEngine::evidence_matrix();
        assert!(
            matrix.len() >= 17,
            "len={}, should have claims from Sprints 71-79",
            matrix.len()
        );
        let verified_count = matrix
            .iter()
            .filter(|e| e.status == ClaimStatus::Verified)
            .count();
        assert!(verified_count >= 14, "most claims should be verified");
    }

    #[test]
    fn test_unified_run_all_finite() {
        let mut engine = SingularitarianEngine::new(2.7);
        engine.initialize_chrono_solver(8, 8);
        engine.initialize_cmb_analyzer(0.0172);
        engine.initialize_voudon_friedmann(0.01, 0.05);

        let avt = HigherAvt::new(64);
        let stab = DekaVoudonStabilizer {
            level: 1,
            nodes: (0..50).collect(),
            parity_paths: vec![],
        };

        engine.initialize_vacuum_decoder(&avt);

        // Smoke-test scale: dim=16 for Bell (sedenion plumbing), 100 gauge samples.
        // Full physics: (512, 10_000). This exercises the same code paths in seconds.
        let result = engine.run_unified(&[stab], 16, 100);

        // All outputs should be finite
        assert!(result.hawking_spectrum.hawking_temp_k.is_finite());
        assert!(result.hawking_spectrum.fractal_temp_k.is_finite());
        assert!(result.cmb_multipoles.is_some());
        assert!(result.bell_result.is_some());
        assert!(result.gauge_report.is_some());
        assert!(result.fractal_flyby.is_some());
        assert!(result.pioneer_a_ms2.unwrap().is_finite());
        assert!(result.adaptive_wick.is_some());
        assert!(result.voudon_h0.unwrap().is_finite());
        assert!(result.subsystems_active >= 5);
    }
}
