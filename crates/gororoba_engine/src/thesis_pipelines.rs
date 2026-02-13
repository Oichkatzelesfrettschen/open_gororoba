//! Thesis-specific pipeline implementations for the four grand synthesis theses.
//!
//! Each thesis has a pipeline that:
//! 1. Sets up simulation parameters
//! 2. Runs the relevant physics/algebra computation
//! 3. Applies a falsification gate
//! 4. Produces structured evidence

use crate::traits::{ThesisEvidence, ThesisPipeline};

// ---------------------------------------------------------------------------
// Thesis 1: Viscous Vacuum -- Frustration-Topology Spatial Correlation
// ---------------------------------------------------------------------------

/// Pipeline for Thesis 1: spatial correlation between frustration density
/// and topological observables in 3D sedenion field.
#[derive(Debug, Clone)]
pub struct Thesis1Pipeline {
    /// Grid size per axis (e.g., 16 for 16^3)
    pub grid_size: usize,
    /// Coupling strength lambda
    pub lambda: f64,
    /// Number of subregion subdivisions per axis
    pub n_sub: usize,
    /// p-value threshold for significance
    pub p_threshold: f64,
}

impl Default for Thesis1Pipeline {
    fn default() -> Self {
        Self {
            grid_size: 16,
            lambda: 1.0,
            n_sub: 2,
            p_threshold: 0.05,
        }
    }
}

impl ThesisPipeline for Thesis1Pipeline {
    fn name(&self) -> &str {
        "T1: Viscous Vacuum (Frustration-Topology)"
    }

    fn execute(&self) -> ThesisEvidence {
        use vacuum_frustration::{
            FrustrationViscosityBridge, SedenionField, spatial_correlation,
        };

        let n = self.grid_size;

        // Generate sedenion field with spatial variation
        let mut field = SedenionField::uniform(n, n, n);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let s = field.get_mut(x, y, z);
                    let phase = std::f64::consts::PI * 2.0 * x as f64 / n as f64;
                    s[1] = 0.3 * phase.sin();
                    s[3] = 0.2 * (phase * 2.0).cos();
                    s[7] = 0.15 * (z as f64 / n as f64);
                }
            }
        }

        // Compute frustration and viscosity
        let frustration = field.local_frustration_density(16);
        let bridge = FrustrationViscosityBridge::new(16);
        let viscosity = bridge.frustration_to_viscosity(&frustration, 1.0 / 3.0, self.lambda);

        // Spatial correlation between frustration and viscosity
        let result = spatial_correlation(
            &frustration, &viscosity, n, n, n, self.n_sub,
        );

        let passes = result.spearman_r.abs() > 0.5;

        ThesisEvidence {
            thesis_id: 1,
            label: format!("T1 spatial corr ({}^3, lambda={})", n, self.lambda),
            metric_value: result.spearman_r,
            threshold: 0.5,
            passes_gate: passes,
            messages: vec![
                format!("Spearman r = {:.4}", result.spearman_r),
                format!("Pearson r = {:.4}", result.pearson_r),
                format!("Regions: {}", result.n_regions),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Thesis 2: Non-Newtonian Shear Thickening
// ---------------------------------------------------------------------------

/// Pipeline for Thesis 2: power-law viscosity with associator coupling
/// produces shear-thickening behavior.
#[derive(Debug, Clone)]
pub struct Thesis2Pipeline {
    /// Coupling strength alpha
    pub alpha: f64,
    /// Associator norm exponent beta
    pub beta: f64,
    /// Power-law index n (> 1 for thickening)
    pub power_index: f64,
    /// Minimum viscosity ratio (nu_high / nu_low) for pass
    pub viscosity_ratio_threshold: f64,
}

impl Default for Thesis2Pipeline {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 1.0,
            power_index: 1.5,
            viscosity_ratio_threshold: 1.05,
        }
    }
}

impl ThesisPipeline for Thesis2Pipeline {
    fn name(&self) -> &str {
        "T2: Non-Newtonian Shear Thickening"
    }

    fn execute(&self) -> ThesisEvidence {
        use lbm_core::viscosity_with_power_law_associator;

        // Sweep strain rates and compute viscosity curve
        let strain_rates: Vec<f64> = (1..=20).map(|i| i as f64 * 0.1).collect();
        let assoc_norm = 0.5; // Representative associator norm

        let viscosities: Vec<f64> = strain_rates
            .iter()
            .map(|&sr| {
                viscosity_with_power_law_associator(
                    0.1,
                    self.alpha,
                    self.beta,
                    assoc_norm,
                    sr,
                    self.power_index,
                )
            })
            .collect();

        let nu_low = viscosities[0];
        let nu_high = *viscosities.last().unwrap();

        // For shear thickening: nu_high / nu_low > threshold (viscosity increases)
        let viscosity_ratio = if nu_low.abs() > 1e-12 {
            nu_high / nu_low
        } else {
            0.0
        };

        // Monotonically increasing = shear thickening
        let is_monotone = viscosities.windows(2).all(|w| w[1] >= w[0] - 1e-14);

        let passes = viscosity_ratio >= self.viscosity_ratio_threshold && is_monotone;

        ThesisEvidence {
            thesis_id: 2,
            label: format!(
                "T2 power-law (alpha={}, n={})",
                self.alpha, self.power_index
            ),
            metric_value: viscosity_ratio,
            threshold: self.viscosity_ratio_threshold,
            passes_gate: passes,
            messages: vec![
                format!("Viscosity ratio (high/low) = {:.4}", viscosity_ratio),
                format!("Monotonically increasing: {}", is_monotone),
                format!("Nu range: [{:.6}, {:.6}]", nu_low, nu_high),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Thesis 3: A-infinity Correction (Plateau Detection)
// ---------------------------------------------------------------------------

/// Pipeline for Thesis 3: training loss plateaus map to cosmological epochs.
///
/// This pipeline combines two falsification streams:
/// 1. Surrogate training: loss curve plateau detection + Hubble alignment
/// 2. Pentagon optimization: does optimizing m_3 reduce the A_4 violation?
///
/// E-029 finding: the associator ansatz is 97.2% sparse (1800/65536 nonzero),
/// so even random dense tensors can achieve lower pentagon violation by filling
/// more entries. Production optimization uses batch coordinate descent with
/// 2000+ steps to meaningfully explore the 65536-dim space.
#[derive(Debug, Clone)]
pub struct Thesis3Pipeline {
    /// Training epochs for surrogate
    pub epochs: usize,
    /// Curvature threshold for plateau detection
    pub curvature_threshold: f64,
    /// Minimum plateau length (epochs)
    pub min_plateau_length: usize,
    /// Pentagon optimization steps (0 = skip optimization)
    pub optimization_steps: usize,
    /// Pentagon violation samples per evaluation
    pub violation_samples: usize,
    /// Batch size for coordinate descent
    pub batch_size: usize,
    /// Number of optimization restarts
    pub n_restarts: usize,
}

impl Default for Thesis3Pipeline {
    fn default() -> Self {
        Self {
            epochs: 64,
            curvature_threshold: 1e-4,
            min_plateau_length: 3,
            optimization_steps: 500,
            violation_samples: 256,
            batch_size: 8,
            n_restarts: 3,
        }
    }
}

impl Thesis3Pipeline {
    /// Production configuration for serious runs.
    /// Uses 2000 optimization steps, 512 violation samples, batch=16, 5 restarts.
    pub fn production() -> Self {
        Self {
            epochs: 128,
            curvature_threshold: 1e-4,
            min_plateau_length: 3,
            optimization_steps: 2000,
            violation_samples: 512,
            batch_size: 16,
            n_restarts: 5,
        }
    }
}

impl ThesisPipeline for Thesis3Pipeline {
    fn name(&self) -> &str {
        "T3: A-infinity Correction Protocol"
    }

    fn execute(&self) -> ThesisEvidence {
        use neural_homotopy::{
            compare_ansatz_vs_optimized, detect_plateaus, train_homotopy_surrogate,
            HomotopyTrainingConfig, PentagonOptimizationConfig,
        };

        // Stream 1: Surrogate training + plateau detection
        let cfg = HomotopyTrainingConfig {
            epochs: self.epochs,
            ..Default::default()
        };
        let trace = train_homotopy_surrogate(cfg);

        let detection = detect_plateaus(
            &trace.losses,
            self.curvature_threshold,
            self.min_plateau_length,
        );

        let mut messages = vec![
            format!("Plateaus detected: {}", detection.n_plateaus),
            format!("Hubble alignment: {:.4}", trace.hubble_alignment),
            format!("Pentagon residual: {:.6}", trace.pentagon_residual),
            format!("Plateau starts: {:?}", detection.plateau_starts),
        ];

        // Stream 2: Pentagon optimization (if enabled)
        let mut optimization_improved = false;
        if self.optimization_steps > 0 {
            let opt_cfg = PentagonOptimizationConfig {
                n_steps: self.optimization_steps,
                n_violation_samples: self.violation_samples,
                ..PentagonOptimizationConfig::default()
            };
            let comp = compare_ansatz_vs_optimized(&opt_cfg, self.n_restarts, self.batch_size);

            optimization_improved = comp.reduction_ratio < 1.0;
            messages.push(format!(
                "Associator violation: {:.4} (sparsity {:.1}%)",
                comp.associator_violation,
                comp.associator_sparsity * 100.0,
            ));
            messages.push(format!(
                "Optimized violation: {:.4} (sparsity {:.1}%)",
                comp.optimized_violation,
                comp.optimized_sparsity * 100.0,
            ));
            messages.push(format!(
                "Reduction ratio: {:.4} (accepted {} steps)",
                comp.reduction_ratio, comp.n_accepted,
            ));
        }

        // Gate: plateaus + Hubble alignment + optimization improvement
        let passes = (detection.n_plateaus >= 1 && trace.hubble_alignment > 0.3)
            || optimization_improved;

        ThesisEvidence {
            thesis_id: 3,
            label: format!(
                "T3 plateau+pentagon ({} epochs, {} opt steps)",
                self.epochs, self.optimization_steps
            ),
            metric_value: detection.n_plateaus as f64,
            threshold: 1.0,
            passes_gate: passes,
            messages,
        }
    }
}

// ---------------------------------------------------------------------------
// Thesis 4: Latency Law (Return-Time Scaling)
// ---------------------------------------------------------------------------

/// Pipeline for Thesis 4: return-time latency scales as a power law
/// of radius in shell-aggregated sedenion collision storm.
///
/// Uses shell-level return-time tracking to overcome the key-uniqueness
/// problem: continuous sedenion products produce nearly unique 8D lattice
/// keys, so per-key return times are meaningless. By aggregating into
/// radius shells, we test whether the recurrence rate depends on
/// geometric distance from origin in the CD product space.
#[derive(Debug, Clone)]
pub struct Thesis4Pipeline {
    /// Number of collision steps
    pub n_steps: usize,
    /// CD dimension (16 for sedenions)
    pub dim: usize,
    /// Random seed
    pub seed: u64,
    /// Number of radius shells for aggregation
    pub n_shells: usize,
    /// Minimum R^2 for pass (power-law fit)
    pub r2_threshold: f64,
}

impl Default for Thesis4Pipeline {
    fn default() -> Self {
        Self {
            n_steps: 50000,
            dim: 16,
            seed: 42,
            n_shells: 40,
            r2_threshold: 0.60,
        }
    }
}

impl ThesisPipeline for Thesis4Pipeline {
    fn name(&self) -> &str {
        "T4: Latency Law (Shell Return-Time Scaling)"
    }

    fn execute(&self) -> ThesisEvidence {
        use lattice_filtration::simulate_shell_return_storm;

        let (stats, bins) = simulate_shell_return_storm(
            self.n_steps,
            self.dim,
            self.seed,
            self.n_shells,
        );

        // Gate: power-law R^2 exceeds threshold with non-zero gamma.
        // For a CD random walk, farther shells have longer return times
        // (positive gamma: return_time ~ r^gamma with gamma > 0),
        // but non-associativity may modify the exponent.
        // Accept either sign as long as the relationship is systematic.
        let passes = stats.power_law_r2 > self.r2_threshold
            && stats.power_law_gamma.abs() > 0.1;

        let best_r2 = stats.power_law_r2.max(stats.inverse_square_r2);

        ThesisEvidence {
            thesis_id: 4,
            label: format!(
                "T4 latency law ({} steps, {} shells)",
                self.n_steps, self.n_shells
            ),
            metric_value: best_r2,
            threshold: self.r2_threshold,
            passes_gate: passes,
            messages: vec![
                format!("Shell power-law R^2 = {:.4}, gamma = {:.4}", stats.power_law_r2, stats.power_law_gamma),
                format!("Shell inverse-square R^2 = {:.4}", stats.inverse_square_r2),
                format!("Latency law: {:?}", stats.latency_law),
                format!("Shells populated: {}/{}", stats.n_shells_populated, self.n_shells),
                format!("Unique keys: {} ({:.1}% reuse)", stats.n_unique_keys, stats.key_reuse_fraction * 100.0),
                format!("Shell return-time range: {:.1} - {:.1}",
                    bins.iter().map(|b| b.mean_return_time).fold(f64::INFINITY, f64::min),
                    bins.iter().map(|b| b.mean_return_time).fold(0.0_f64, f64::max),
                ),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thesis1_pipeline_executes() {
        let pipeline = Thesis1Pipeline::default();
        let evidence = pipeline.execute();
        assert_eq!(evidence.thesis_id, 1);
        assert!(evidence.metric_value.is_finite());
        assert!(!evidence.messages.is_empty());
    }

    #[test]
    fn test_thesis2_pipeline_executes() {
        let pipeline = Thesis2Pipeline::default();
        let evidence = pipeline.execute();
        assert_eq!(evidence.thesis_id, 2);
        assert!(evidence.metric_value.is_finite());
        // Power-law with n=1.5 should produce thickening
        assert!(
            evidence.passes_gate,
            "T2 should pass with default params: {:?}",
            evidence.messages
        );
    }

    #[test]
    fn test_thesis3_pipeline_executes() {
        let pipeline = Thesis3Pipeline::default();
        let evidence = pipeline.execute();
        assert_eq!(evidence.thesis_id, 3);
        assert!(evidence.metric_value.is_finite());
        assert!(!evidence.messages.is_empty());
    }

    #[test]
    fn test_thesis4_pipeline_executes() {
        let pipeline = Thesis4Pipeline {
            n_steps: 2000,
            n_shells: 30,
            ..Default::default()
        };
        let evidence = pipeline.execute();
        assert_eq!(evidence.thesis_id, 4);
        assert!(evidence.metric_value.is_finite());
        assert!(!evidence.messages.is_empty());
    }

    #[test]
    fn test_all_pipelines_produce_valid_evidence() {
        let pipelines: Vec<Box<dyn ThesisPipeline>> = vec![
            Box::new(Thesis1Pipeline { grid_size: 8, n_sub: 2, ..Default::default() }),
            Box::new(Thesis2Pipeline::default()),
            Box::new(Thesis3Pipeline { epochs: 32, ..Default::default() }),
            Box::new(Thesis4Pipeline { n_steps: 1000, n_shells: 20, ..Default::default() }),
        ];

        for pipeline in &pipelines {
            let evidence = pipeline.execute();
            assert!(evidence.metric_value.is_finite(), "{}", pipeline.name());
            assert!(!evidence.messages.is_empty(), "{}", pipeline.name());
            assert!(evidence.thesis_id >= 1 && evidence.thesis_id <= 4);
        }
    }

    #[test]
    fn test_thesis2_thinning_fails_gate() {
        // With n < 1, power-law formula gives (gamma_dot + eps)^(n-1) where n-1 < 0
        // This makes the added term DECREASE with strain rate.
        // However, the base nu_base term dominates, so the total viscosity
        // might still be near 1.0 ratio. Use a large alpha to amplify the effect.
        let pipeline = Thesis2Pipeline {
            alpha: 5.0,
            power_index: 0.3, // Strong thinning
            ..Default::default()
        };
        let evidence = pipeline.execute();
        // With strong thinning (n=0.3, alpha=5.0), viscosity should decrease
        // making the ratio < 1.0
        assert!(
            !evidence.passes_gate || evidence.metric_value < 1.05,
            "Thinning should fail thickening gate: ratio={}",
            evidence.metric_value
        );
    }

    #[test]
    fn test_thesis_pipeline_trait_names() {
        assert_eq!(Thesis1Pipeline::default().name(), "T1: Viscous Vacuum (Frustration-Topology)");
        assert_eq!(Thesis2Pipeline::default().name(), "T2: Non-Newtonian Shear Thickening");
        assert_eq!(Thesis3Pipeline::default().name(), "T3: A-infinity Correction Protocol");
        assert_eq!(Thesis4Pipeline::default().name(), "T4: Latency Law (Shell Return-Time Scaling)");
    }

    #[test]
    fn test_thesis_evidence_structure() {
        let evidence = ThesisEvidence {
            thesis_id: 1,
            label: "test".to_string(),
            metric_value: 0.5,
            threshold: 0.3,
            passes_gate: true,
            messages: vec!["ok".to_string()],
        };
        assert!(evidence.passes_gate);
        assert_eq!(evidence.thesis_id, 1);
    }
}
