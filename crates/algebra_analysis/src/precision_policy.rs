//! Deterministic backend selection for the algebra precision ladder.
//!
//! This module turns the current Jacobi backend choice into an explicit policy
//! decision instead of leaving it implicit in `cfg` blocks and comments. The
//! initial policy is intentionally conservative: it encodes backend intent and
//! portability requirements, while leaving room for future size- or
//! benchmark-derived thresholds.

/// Backend lanes currently available for symmetric eigensolvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobiBackend {
    /// Native x87 / FP80 lane.
    X87,
    /// Software double-double fallback.
    DoubleDouble,
    /// Plain f64 oracle lane.
    ReferenceF64,
}

/// Solver family selected for the current symmetric-eigen workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumSolverFamily {
    /// The current full-spectrum Jacobi family.
    FullJacobi,
    /// The structure-aware exact reduction plus Jacobi solve.
    StructuredReducedJacobi,
    /// Partial-spectrum subspace iteration / Rayleigh-Ritz lane.
    PartialSubspaceIteration,
    /// Block Jacobi prototype family.
    BlockJacobi,
}

/// Spectrum objective requested by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumObjective {
    /// Solve for the full spectrum.
    FullSpectrum,
    /// Solve for the k largest-magnitude eigenvalues.
    LargestAbs { k: usize },
    /// Solve for the k smallest-magnitude eigenvalues.
    SmallestAbs { k: usize },
}

impl SpectrumObjective {
    /// Number of requested eigenvalues.
    #[must_use]
    pub const fn requested_count(self) -> usize {
        match self {
            Self::FullSpectrum => usize::MAX,
            Self::LargestAbs { k } | Self::SmallestAbs { k } => k,
        }
    }

    /// Whether this is a partial-spectrum request.
    #[must_use]
    pub const fn is_partial(self) -> bool {
        !matches!(self, Self::FullSpectrum)
    }
}

/// High-level matrix workload class used by solver-family dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixWorkloadClass {
    /// Dense full-spectrum problem with no special structure assumptions.
    FullSpectrumDense,
    /// Only a small extremal subset is required.
    FewExtremal,
    /// Obstruction matrices and close relatives with exploitable structure.
    ObstructionStructured,
}

/// Structural hints that can steer solver-family choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatrixStructureHints {
    /// Matrix is symmetric / self-adjoint.
    pub symmetric: bool,
    /// All entries are nonnegative within tolerance.
    pub nonnegative: bool,
    /// Diagonal is exactly or effectively zero.
    pub zero_diagonal: bool,
    /// Matrix contains isolated exact or near-exact zero modes.
    pub isolated_zero_modes: bool,
    /// Matrix values appear to come from a small quantized ladder.
    pub quantized_value_ladder: bool,
    /// Row classes suggest an equitable-partition style reduction candidate.
    pub equitable_partition_candidate: bool,
}

/// Numerical contract requested by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobiPrecisionGoal {
    /// Default obstruction-spectrum policy backed by the offline threshold table.
    ObstructionSpectrumDefault,
    /// Prefer the ext80 lane when the host can support it.
    StrictExt80,
    /// Prefer a portable software high-precision lane.
    PortableHighPrecision,
    /// Prefer the simplest reference/oracle lane.
    ReferenceCrossCheck,
}

impl JacobiPrecisionGoal {
    /// Bridge to the canonical workspace precision vocabulary
    /// (gororoba_gpu_bridge::StoragePrecision). This maps each goal to
    /// its representative target storage type; the goal enum itself
    /// captures a portability + accuracy policy that StoragePrecision
    /// does not.
    pub fn storage_precision(self) -> gororoba_gpu_bridge::StoragePrecision {
        use gororoba_gpu_bridge::StoragePrecision as SP;
        match self {
            // Default obstruction-spectrum policy uses FP64 as the
            // reference accumulator type.
            Self::ObstructionSpectrumDefault => SP::Fp64,
            // ext80 = x87 80-bit extended; closest StoragePrecision is
            // DdFp128 (the only >64-bit variant).
            Self::StrictExt80 => SP::DdFp128,
            // Portable software high precision -- pick FP64 since no
            // workspace variant captures double-double explicitly here.
            Self::PortableHighPrecision => SP::Fp64,
            // Reference cross-check -- FP64 oracle.
            Self::ReferenceCrossCheck => SP::Fp64,
        }
    }
}

/// Portability constraint for the solver selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobiPortability {
    /// Native accelerated lanes may be used when available.
    NativePreferred,
    /// Keep the choice portable across non-x87 targets.
    PortableOnly,
}

/// Deterministic policy inputs for selecting a symmetric-eigen solver family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpectrumDispatchInput {
    /// Matrix order.
    pub matrix_order: usize,
    /// Requested spectrum objective.
    pub objective: SpectrumObjective,
    /// Workload class.
    pub workload_class: MatrixWorkloadClass,
    /// Structural hints for the matrix family.
    pub structure_hints: MatrixStructureHints,
    /// Numerical contract for this solve.
    pub precision_goal: JacobiPrecisionGoal,
    /// Portability requirement.
    pub portability: JacobiPortability,
    /// Whether the build/host can use the x87 backend.
    pub x87_available: bool,
    /// Whether the software double-double lane is available.
    pub double_double_available: bool,
}

impl SpectrumDispatchInput {
    /// Default obstruction-spectrum full-spectrum solve.
    #[must_use]
    pub fn obstruction_full_spectrum(
        matrix_order: usize,
        structure_hints: MatrixStructureHints,
    ) -> Self {
        Self {
            matrix_order,
            objective: SpectrumObjective::FullSpectrum,
            workload_class: MatrixWorkloadClass::ObstructionStructured,
            structure_hints,
            precision_goal: JacobiPrecisionGoal::ObstructionSpectrumDefault,
            portability: JacobiPortability::NativePreferred,
            x87_available: cfg!(target_arch = "x86_64"),
            double_double_available: true,
        }
    }

    /// Default obstruction-spectrum extremal solve.
    #[must_use]
    pub fn obstruction_extremal(
        matrix_order: usize,
        objective: SpectrumObjective,
        structure_hints: MatrixStructureHints,
    ) -> Self {
        Self {
            matrix_order,
            objective,
            workload_class: MatrixWorkloadClass::FewExtremal,
            structure_hints,
            precision_goal: JacobiPrecisionGoal::ObstructionSpectrumDefault,
            portability: JacobiPortability::NativePreferred,
            x87_available: cfg!(target_arch = "x86_64"),
            double_double_available: true,
        }
    }
}

/// Deterministic policy inputs for selecting a Jacobi backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JacobiDispatchInput {
    /// Matrix order. Kept explicit so future threshold tables can consume it.
    pub matrix_order: usize,
    /// Numerical contract for this solve.
    pub precision_goal: JacobiPrecisionGoal,
    /// Portability requirement for this solve.
    pub portability: JacobiPortability,
    /// Whether the build/host can use the x87 backend.
    pub x87_available: bool,
    /// Whether the software double-double lane is available.
    pub double_double_available: bool,
}

impl JacobiDispatchInput {
    /// Default policy for obstruction-spectrum style solves.
    #[must_use]
    pub fn obstruction_spectrum(matrix_order: usize) -> Self {
        Self {
            matrix_order,
            precision_goal: JacobiPrecisionGoal::ObstructionSpectrumDefault,
            portability: JacobiPortability::NativePreferred,
            x87_available: cfg!(target_arch = "x86_64"),
            double_double_available: true,
        }
    }

    /// Force the portable software lane.
    #[must_use]
    pub fn portable_high_precision(matrix_order: usize) -> Self {
        Self {
            matrix_order,
            precision_goal: JacobiPrecisionGoal::PortableHighPrecision,
            portability: JacobiPortability::PortableOnly,
            x87_available: cfg!(target_arch = "x86_64"),
            double_double_available: true,
        }
    }

    /// Force the reference/oracle lane.
    #[must_use]
    pub fn reference_cross_check(matrix_order: usize) -> Self {
        Self {
            matrix_order,
            precision_goal: JacobiPrecisionGoal::ReferenceCrossCheck,
            portability: JacobiPortability::PortableOnly,
            x87_available: cfg!(target_arch = "x86_64"),
            double_double_available: true,
        }
    }
}

/// Result of the deterministic solver-family decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpectrumDispatchDecision {
    /// Selected solver family.
    pub solver_family: SpectrumSolverFamily,
    /// Chosen backend lane for the numeric core.
    pub backend: JacobiBackend,
    /// Human-readable reason for this deterministic selection.
    pub reason: &'static str,
}

/// Result of the deterministic backend-policy decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JacobiDispatchDecision {
    /// Chosen backend lane.
    pub backend: JacobiBackend,
    /// Human-readable reason for this deterministic selection.
    pub reason: &'static str,
}

/// Choose a symmetric-eigen solver family from explicit policy inputs.
#[must_use]
pub fn choose_spectrum_solver(input: SpectrumDispatchInput) -> SpectrumDispatchDecision {
    if input.objective.is_partial() || input.workload_class == MatrixWorkloadClass::FewExtremal {
        return SpectrumDispatchDecision {
            solver_family: SpectrumSolverFamily::PartialSubspaceIteration,
            backend: JacobiBackend::ReferenceF64,
            reason: "few-extremal workloads use the partial-spectrum subspace-iteration lane",
        };
    }

    let backend = choose_jacobi_backend(JacobiDispatchInput {
        matrix_order: input.matrix_order,
        precision_goal: input.precision_goal,
        portability: input.portability,
        x87_available: input.x87_available,
        double_double_available: input.double_double_available,
    })
    .backend;

    if input.workload_class == MatrixWorkloadClass::ObstructionStructured
        && input.structure_hints.isolated_zero_modes
    {
        return SpectrumDispatchDecision {
            solver_family: SpectrumSolverFamily::StructuredReducedJacobi,
            backend,
            reason: "obstruction-structured workloads with isolated zero modes use exact structured deflation before the Jacobi backend",
        };
    }

    SpectrumDispatchDecision {
        solver_family: SpectrumSolverFamily::FullJacobi,
        backend,
        reason: "default full-spectrum workloads stay on the current Jacobi backend ladder",
    }
}

/// First offline threshold table for the default obstruction-spectrum policy.
///
/// Derived from the solver-shaped sweeps in:
/// - `reports/benchmarks/jacobi_backend_boundary_24_40.csv`
/// - `reports/benchmarks/jacobi_backend_boundary_44_56.csv`
/// - `reports/benchmarks/jacobi_backend_boundary_60_68.csv`
/// - `reports/benchmarks/jacobi_backend_boundary_obstruction_like_52_64.csv`
/// - `reports/benchmarks/jacobi_backend_boundary_obstruction_like_60_68.csv`
/// - `reports/benchmarks/jacobi_backend_boundary_entrywise_64_72_fixed_dd.csv`
///
/// The reliable timing region starts at order 16; below that the runs become
/// timer-noise dominated. After extending the sweep to six spectral families,
/// then adding an entrywise zero-diagonal quantized obstruction family and
/// repairing the double-double Jacobi rotation path, order 64 remains the
/// largest measured size where `reference_f64` stays fastest across every
/// tested family and backend. The current threshold rule is therefore still:
/// keep orders up to 64 on the reference f64 lane, and escalate larger orders
/// to the strict-ext80 fallback ladder.
pub const OBSTRUCTION_SPECTRUM_REFERENCE_MAX_ORDER: usize = 64;

/// Choose a Jacobi backend from explicit policy inputs.
#[must_use]
pub fn choose_jacobi_backend(input: JacobiDispatchInput) -> JacobiDispatchDecision {
    match input.precision_goal {
        JacobiPrecisionGoal::ObstructionSpectrumDefault => {
            if input.matrix_order <= OBSTRUCTION_SPECTRUM_REFERENCE_MAX_ORDER {
                JacobiDispatchDecision {
                    backend: JacobiBackend::ReferenceF64,
                    reason: "offline threshold table keeps obstruction-spectrum orders up to 64 on the reference f64 lane",
                }
            } else {
                choose_strict_ext80_backend(input)
            }
        }
        JacobiPrecisionGoal::ReferenceCrossCheck => JacobiDispatchDecision {
            backend: JacobiBackend::ReferenceF64,
            reason: "reference cross-check explicitly requested",
        },
        JacobiPrecisionGoal::PortableHighPrecision => {
            if input.double_double_available {
                JacobiDispatchDecision {
                    backend: JacobiBackend::DoubleDouble,
                    reason: "portable high-precision policy prefers the software double-double lane",
                }
            } else if input.x87_available && input.portability == JacobiPortability::NativePreferred
            {
                JacobiDispatchDecision {
                    backend: JacobiBackend::X87,
                    reason: "portable software lane unavailable; falling back to the native x87 lane",
                }
            } else {
                JacobiDispatchDecision {
                    backend: JacobiBackend::ReferenceF64,
                    reason: "no high-precision lane is available; falling back to the reference f64 solver",
                }
            }
        }
        JacobiPrecisionGoal::StrictExt80 => choose_strict_ext80_backend(input),
    }
}

fn choose_strict_ext80_backend(input: JacobiDispatchInput) -> JacobiDispatchDecision {
    if input.portability == JacobiPortability::PortableOnly && input.double_double_available {
        JacobiDispatchDecision {
            backend: JacobiBackend::DoubleDouble,
            reason: "portable-only policy disallows the native x87 lane; using double-double",
        }
    } else if input.x87_available {
        JacobiDispatchDecision {
            backend: JacobiBackend::X87,
            reason: "strict ext80 policy prefers the native x87 lane when it is available",
        }
    } else if input.double_double_available {
        JacobiDispatchDecision {
            backend: JacobiBackend::DoubleDouble,
            reason: "strict ext80 requested but x87 is unavailable; using double-double as the deterministic fallback",
        }
    } else {
        JacobiDispatchDecision {
            backend: JacobiBackend::ReferenceF64,
            reason: "no high-precision lane is available; falling back to the reference f64 solver",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        JacobiBackend, JacobiDispatchInput, JacobiPortability, JacobiPrecisionGoal,
        MatrixStructureHints, OBSTRUCTION_SPECTRUM_REFERENCE_MAX_ORDER, SpectrumDispatchInput,
        SpectrumObjective, SpectrumSolverFamily, choose_jacobi_backend, choose_spectrum_solver,
    };

    #[test]
    fn obstruction_default_uses_reference_lane_through_threshold() {
        let decision = choose_jacobi_backend(JacobiDispatchInput::obstruction_spectrum(
            OBSTRUCTION_SPECTRUM_REFERENCE_MAX_ORDER,
        ));

        assert_eq!(decision.backend, JacobiBackend::ReferenceF64);
    }
    #[test]
    fn obstruction_default_escalates_above_threshold() {
        let decision = choose_jacobi_backend(JacobiDispatchInput::obstruction_spectrum(
            OBSTRUCTION_SPECTRUM_REFERENCE_MAX_ORDER + 1,
        ));

        let expected = if cfg!(target_arch = "x86_64") {
            JacobiBackend::X87
        } else {
            JacobiBackend::DoubleDouble
        };

        assert_eq!(decision.backend, expected);
    }

    #[test]
    fn strict_ext80_prefers_x87_when_available() {
        let decision = choose_jacobi_backend(JacobiDispatchInput {
            matrix_order: 64,
            precision_goal: JacobiPrecisionGoal::StrictExt80,
            portability: JacobiPortability::NativePreferred,
            x87_available: true,
            double_double_available: true,
        });

        assert_eq!(decision.backend, JacobiBackend::X87);
    }

    #[test]
    fn portable_policy_prefers_double_double() {
        let decision = choose_jacobi_backend(JacobiDispatchInput::portable_high_precision(64));

        assert_eq!(decision.backend, JacobiBackend::DoubleDouble);
    }

    #[test]
    fn strict_ext80_falls_back_to_double_double_without_x87() {
        let decision = choose_jacobi_backend(JacobiDispatchInput {
            matrix_order: 64,
            precision_goal: JacobiPrecisionGoal::StrictExt80,
            portability: JacobiPortability::NativePreferred,
            x87_available: false,
            double_double_available: true,
        });

        assert_eq!(decision.backend, JacobiBackend::DoubleDouble);
    }

    #[test]
    fn reference_policy_picks_oracle_lane() {
        let decision = choose_jacobi_backend(JacobiDispatchInput::reference_cross_check(32));

        assert_eq!(decision.backend, JacobiBackend::ReferenceF64);
    }

    #[test]
    fn obstruction_structured_dispatch_prefers_exact_reduction_when_zero_modes_exist() {
        let decision = choose_spectrum_solver(SpectrumDispatchInput::obstruction_full_spectrum(
            16,
            MatrixStructureHints {
                symmetric: true,
                nonnegative: true,
                zero_diagonal: true,
                isolated_zero_modes: true,
                quantized_value_ladder: false,
                equitable_partition_candidate: false,
            },
        ));

        assert_eq!(
            decision.solver_family,
            SpectrumSolverFamily::StructuredReducedJacobi
        );
    }

    #[test]
    fn few_extremal_dispatch_prefers_partial_lane() {
        let decision = choose_spectrum_solver(SpectrumDispatchInput::obstruction_extremal(
            32,
            SpectrumObjective::SmallestAbs { k: 2 },
            MatrixStructureHints::default(),
        ));

        assert_eq!(
            decision.solver_family,
            SpectrumSolverFamily::PartialSubspaceIteration
        );
        assert_eq!(decision.backend, JacobiBackend::ReferenceF64);
    }
}
