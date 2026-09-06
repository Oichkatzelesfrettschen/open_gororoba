//! mrt-stability-audit: Systematic BGK vs MRT stability comparison.
//!
//! Runs 3D LBM at increasing density contrast and reports:
//! - Maximum Mach number reached before crash (NaN in density)
//! - Mass conservation error
//! - Crash threshold (density contrast where solver diverges)
//!
//! Survival requires finite fields, positive density and explicitly supplied
//! mass-error/Mach budgets. Numerical survival does not establish physical validity.

use clap::Parser;
use gororoba_cli_physics::lbm_dispatch::LbmBackend;
use gororoba_cli_physics::lbm_population_diagnostics::inspect_fields;
use lbm_3d::solver::CollisionMode;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_explosion_and_invalid_fields_fail_separate_predicates() {
        let args = Args::parse_from([
            "audit",
            "--max-relative-mass-error",
            "0.000001",
            "--max-mach",
            "0.3",
        ]);
        let explosion = assess(true, 1.0, 1e113, 0.1, &args);
        assert!(explosion.finite_state);
        assert!(!explosion.mass_within_budget);
        assert!(!explosion.passed());
        assert!(!assess(true, -1.0, 0.0, 0.1, &args).positive_density);
        assert!(!assess(false, 1.0, 0.0, 0.1, &args).finite_state);
        assert!(!assess(true, 1.0, 0.0, 1e100, &args).mach_within_budget);
        assert!(assess(true, 1.0, 1e-8, 0.1, &args).passed());
    }
    #[test]
    fn budgets_are_explicit_and_calm_trial_is_measured() {
        assert!(Args::try_parse_from(["audit"]).is_err());
        let args = Args::parse_from([
            "audit",
            "--max-relative-mass-error",
            "0.000001",
            "--max-mach",
            "0.3",
            "--n",
            "4",
            "--steps",
            "2",
        ]);
        let result = run_trial(&args, 1.0, CollisionMode::Bgk);
        assert!(result.survived && result.assessment.passed());
        assert_eq!(result.observed_step, Some(2));
        assert_eq!(result.observation_stage, "post_step_population_moments");
    }
    #[test]
    fn nonfinite_post_step_population_rejects_finite_cached_macros() {
        let args = Args::parse_from([
            "audit",
            "--max-relative-mass-error",
            "0.000001",
            "--max-mach",
            "0.3",
        ]);
        let mut solver = lbm_3d::solver::LbmSolver3D::new(4, 4, 4, 0.8);
        solver.initialize_uniform(1.0, [0.0; 3]);
        assert!(solver.rho.iter().all(|value| value.is_finite()));
        solver.f[lbm_3d::solver::aosoa_idx(63, 18)] = f64::NAN;
        let mut backend = LbmBackend::Avx2(Box::new(solver));
        let observation = inspect_fields(&mut backend).unwrap();
        assert!(!observation.finite);
        assert!(
            !assess(
                observation.finite,
                observation.minimum_density,
                (observation.mass / 64.0 - 1.0).abs(),
                observation.mach,
                &args
            )
            .passed()
        );
    }
}

#[derive(Parser)]
#[command(name = "mrt-stability-audit")]
#[command(about = "Compare BGK vs MRT stability at increasing density contrast")]
struct Args {
    /// Grid size (cubic: n x n x n)
    #[arg(long, default_value = "16")]
    n: usize,

    /// Relaxation time
    #[arg(long, default_value = "0.7")]
    tau: f64,

    /// Number of LBM steps per trial
    #[arg(long, default_value = "200")]
    steps: usize,

    /// Minimum density contrast (rho_peak / rho_background)
    #[arg(long, default_value = "2.0")]
    contrast_min: f64,

    /// Maximum density contrast
    #[arg(long, default_value = "100.0")]
    contrast_max: f64,

    /// Number of contrast levels to test
    #[arg(long, default_value = "10")]
    n_levels: usize,

    /// Output as CSV
    #[arg(long)]
    csv: bool,

    /// Use CUDA GPU solver (requires --features gpu)
    #[arg(long)]
    gpu: bool,

    /// Enable thread-coarsened GPU kernels (1 thread = 2 cells)
    #[arg(long)]
    coarsened: bool,

    /// Enable shared-memory tiled GPU kernels (8x8x4 tile + halo).
    /// Highest priority dispatch: overrides coarsening and pull-streaming.
    #[arg(long)]
    tiling: bool,

    /// Mach check interval: sample max Mach every N steps instead of
    /// every step. Higher values use graph-pair acceleration for bulk
    /// stepping between checks. Default 1 (check every step).
    #[arg(long, default_value = "1")]
    mach_interval: usize,
    /// Investigator-declared relative mass-error acceptance budget.
    #[arg(long)]
    max_relative_mass_error: f64,
    /// Investigator-declared maximum Mach acceptance budget.
    #[arg(long)]
    max_mach: f64,
}

/// Result of one stability trial.
struct TrialResult {
    contrast: f64,
    mode: &'static str,
    survived: bool,
    max_mach: f64,
    mass_err: f64,
    final_step: usize,
    assessment: StabilityAssessment,
    observed_step: Option<usize>,
    observation_stage: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct StabilityAssessment {
    finite_state: bool,
    positive_density: bool,
    mass_within_budget: bool,
    mach_within_budget: bool,
}
impl StabilityAssessment {
    fn passed(self) -> bool {
        self.finite_state
            && self.positive_density
            && self.mass_within_budget
            && self.mach_within_budget
    }
    fn failed() -> Self {
        Self {
            finite_state: false,
            positive_density: false,
            mass_within_budget: false,
            mach_within_budget: false,
        }
    }
}
fn assess(
    finite_fields: bool,
    minimum_density: f64,
    mass_error: f64,
    mach: f64,
    args: &Args,
) -> StabilityAssessment {
    StabilityAssessment {
        finite_state: finite_fields
            && minimum_density.is_finite()
            && mass_error.is_finite()
            && mach.is_finite(),
        positive_density: minimum_density.is_finite() && minimum_density > 0.0,
        mass_within_budget: mass_error.is_finite() && mass_error <= args.max_relative_mass_error,
        mach_within_budget: mach.is_finite() && mach >= 0.0 && mach <= args.max_mach,
    }
}
/// Build Gaussian density perturbation centered on the grid.
fn gaussian_density(n: usize, contrast: f64) -> Vec<f64> {
    let center = n / 2;
    let sigma = (n as f64) / 6.0;
    let n_cells = n * n * n;
    let mut rho = Vec::with_capacity(n_cells);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let dx = ix as f64 - center as f64;
                let dy = iy as f64 - center as f64;
                let dz = iz as f64 - center as f64;
                let r2 = dx * dx + dy * dy + dz * dz;
                rho.push(1.0 + (contrast - 1.0) * (-r2 / (2.0 * sigma * sigma)).exp());
            }
        }
    }
    rho
}

/// Run one stability trial using the unified LbmBackend dispatcher.
fn run_trial(args: &Args, contrast: f64, mode: CollisionMode) -> TrialResult {
    let n = args.n;
    let mut backend = if args.gpu {
        #[cfg(feature = "gpu")]
        {
            LbmBackend::cuda(n, n, n, args.tau, mode).expect("GPU solver init failed")
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = mode;
            unreachable!("--gpu check prevents reaching here without feature");
        }
    } else {
        LbmBackend::cpu(n, n, n, args.tau, mode)
    };

    if args.coarsened {
        backend.set_coarsening(true);
    }
    if args.tiling {
        backend.set_tiling(true);
    }

    let rho = gaussian_density(n, contrast);
    let u_zero = vec![[0.0_f64; 3]; n * n * n];
    backend
        .initialize_custom(&rho, &u_zero)
        .expect("initialize_custom failed");

    let initial_mass = inspect_fields(&mut backend)
        .expect("initial population observation failed")
        .mass;
    let mut max_mach = 0.0_f64;
    let mut final_step = 0;
    let mut last_assessment = StabilityAssessment::failed();
    let mut last_mass_error = f64::NAN;
    let interval = args.mach_interval.max(1);

    // Bulk-step with periodic Mach monitoring.
    // When interval > 1, step_n() uses graph-pair acceleration on GPU,
    // amortizing kernel launch overhead across the interval.
    let mut step = 0;
    while step < args.steps {
        let chunk = interval.min(args.steps - step);

        if chunk > 1 {
            if backend.step_n(chunk).is_err() {
                return TrialResult {
                    contrast,
                    mode: mode_name(mode),
                    survived: false,
                    max_mach,
                    mass_err: f64::NAN,
                    final_step: step,
                    assessment: StabilityAssessment::failed(),
                    observed_step: None,
                    observation_stage: "step_failed",
                };
            }
        } else if backend.step().is_err() {
            return TrialResult {
                contrast,
                mode: mode_name(mode),
                survived: false,
                max_mach,
                mass_err: f64::NAN,
                final_step: step,
                assessment: StabilityAssessment::failed(),
                observed_step: None,
                observation_stage: "step_failed",
            };
        }

        step += chunk;
        final_step = step;

        let observation = match inspect_fields(&mut backend) {
            Ok(observation) => observation,
            Err(_) => {
                return TrialResult {
                    contrast,
                    mode: mode_name(mode),
                    survived: false,
                    max_mach,
                    mass_err: f64::NAN,
                    final_step,
                    assessment: StabilityAssessment::failed(),
                    observed_step: None,
                    observation_stage: "population_readback_failed",
                };
            }
        };
        let ma = observation.mach;
        if ma.is_finite() {
            max_mach = max_mach.max(ma);
        }
        let mass = observation.mass;
        let mass_err = ((mass - initial_mass) / initial_mass).abs();
        let assessment = assess(
            observation.finite,
            observation.minimum_density,
            mass_err,
            ma,
            args,
        );
        if !assessment.passed() {
            return TrialResult {
                contrast,
                mode: mode_name(mode),
                survived: false,
                max_mach,
                mass_err,
                final_step,
                assessment,
                observed_step: Some(step),
                observation_stage: "post_step_population_moments",
            };
        }
        last_assessment = assessment;
        last_mass_error = mass_err;
    }

    TrialResult {
        contrast,
        mode: mode_name(mode),
        survived: last_assessment.passed(),
        max_mach,
        mass_err: last_mass_error,
        final_step,
        assessment: last_assessment,
        observed_step: Some(final_step),
        observation_stage: "post_step_population_moments",
    }
}

fn mode_name(mode: CollisionMode) -> &'static str {
    match mode {
        CollisionMode::Bgk => "BGK",
        CollisionMode::Mrt => "MRT",
    }
}

fn print_result(result: &TrialResult, csv: bool) {
    if csv {
        println!(
            "{:.2},{},{},{:.6},{:.2e},{},{},{},{},{},{},{}",
            result.contrast,
            result.mode,
            result.survived,
            result.max_mach,
            result.mass_err,
            result.final_step,
            result.assessment.finite_state,
            result.assessment.positive_density,
            result.assessment.mass_within_budget,
            result.assessment.mach_within_budget,
            result
                .observed_step
                .map_or_else(|| "unobserved".to_owned(), |step| step.to_string()),
            result.observation_stage
        );
    } else {
        println!(
            "{:>10.2} {:>5} {:>8} {:>10.6} {:>12.2e} {:>10} {:?} observed_step={:?} observation_stage={}",
            result.contrast,
            result.mode,
            if result.survived { "OK" } else { "CRASH" },
            result.max_mach,
            result.mass_err,
            result.final_step,
            result.assessment,
            result.observed_step,
            result.observation_stage
        );
    }
}

fn main() {
    let args = Args::parse();
    assert!(
        args.steps > 0
            && args.max_relative_mass_error.is_finite()
            && args.max_relative_mass_error >= 0.0
            && args.max_mach.is_finite()
            && args.max_mach > 0.0,
        "positive steps and finite declared stability budgets required"
    );
    eprintln!(
        "declared budgets: relative_mass_error={} max_mach={} observation_interval={} steps; post-step population readback and moment-reduction overhead included",
        args.max_relative_mass_error,
        args.max_mach,
        args.mach_interval.max(1)
    );

    #[cfg(not(feature = "gpu"))]
    if args.gpu {
        eprintln!("ERROR: --gpu requires building with --features gpu");
        std::process::exit(1);
    }

    let contrasts: Vec<f64> = if args.n_levels <= 1 {
        vec![args.contrast_min]
    } else {
        (0..args.n_levels)
            .map(|i| {
                let t = i as f64 / (args.n_levels - 1) as f64;
                (args.contrast_min.ln() * (1.0 - t) + args.contrast_max.ln() * t).exp()
            })
            .collect()
    };

    let backend_name = if args.gpu { "GPU" } else { "CPU" };
    let kernel_tag = if args.tiling {
        " (tiled)"
    } else if args.coarsened {
        " (coarsened)"
    } else {
        ""
    };
    if args.csv {
        println!(
            "contrast,mode,survived,max_mach,mass_err,final_step,finite_state,positive_density,mass_within_budget,mach_within_budget,observed_step,observation_stage"
        );
    } else {
        eprintln!(
            "MRT Stability Audit [{backend_name}{kernel_tag}]: {}^3 grid, tau={}, {} steps (Mach interval={}), {} levels [{:.1}..{:.1}]",
            args.n,
            args.tau,
            args.steps,
            args.mach_interval,
            args.n_levels,
            args.contrast_min,
            args.contrast_max
        );
        println!(
            "{:>10} {:>5} {:>8} {:>10} {:>12} {:>10}",
            "contrast", "mode", "survived", "max_mach", "mass_err", "step"
        );
        println!("{}", "-".repeat(60));
    }

    for &contrast in &contrasts {
        for mode in [CollisionMode::Bgk, CollisionMode::Mrt] {
            let result = run_trial(&args, contrast, mode);
            print_result(&result, args.csv);
        }
    }
}
