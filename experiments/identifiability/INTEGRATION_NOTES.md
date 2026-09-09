# Integration notes

The repository already contains mature Casimir-Lifshitz kernels, optical material models, a claim/falsification registry, and generated audit artifacts. The identifiability work should therefore be an inference layer over those existing forward models rather than another independent physics implementation.

## Immediate integration targets

1. Replace the synthetic boundary target with predictions from `quantum_core::casimir::lifshitz` and specimen optical data from `materials_core`.
2. Treat patch electrostatics, distance offset, roughness, thickness, optical-model uncertainty, drift, and thermal terms as explicit nuisance tangent vectors.
3. Compute the whitened nuisance-orthogonal signature and efficient Fisher information over candidate measurement schedules.
4. Adversarially expand the nuisance basis before promoting any claim.
5. Register computational claims and falsifiers through the canonical control-plane workflow; do not make this directory a competing registry.

## New hypotheses

- H1: Reversed multilayer ordering has a nonzero Lifshitz differential signature after realistic Au-cap screening.
- H2: H1 remains identifiable after patch, gap, roughness, thickness, and optical uncertainty are jointly fitted.
- H3: Multi-observable closure (force plus an independently calibrated boundary-conditioned material observable) increases the minimum principal angle to the conventional nuisance manifold.
- H4: Symmetry-projected control schedules can be optimized by efficient Fisher information rather than raw response amplitude.
- H5: The existing claim/falsification registry can use nuisance-orthogonal information as a quantitative promotion gate, turning qualitative falsifiers into executable design criteria.
- H6: The same inference kernel can score non-Casimir forward models (quantum recovery, temporal-order sensing, plasma/positron transport) without merging their physical mechanisms.
- H7: Numerical convergence should include convergence of target/nuisance Jacobians and Fisher geometry, not only convergence of mean observables.

## Proposed adversarial evidence ladder

- L0: forward model executes and satisfies invariants.
- L1: target is detectable against measurement noise.
- L2: target is locally identifiable against the baseline nuisance family.
- L3: target survives physically justified nuisance-family expansion.
- L4: target survives held-out controls/specimens and covariance misspecification tests.
- L5: one calibrated parameter set closes multiple independent observables.
- L6: independent experimental replication.

The level is metadata about evidential robustness, not a probability that the claim is true.

## Architecture consequence

The clean repository boundary is `forward physics -> sensitivity/Jacobian -> identifiability -> claim registry`. Physics crates remain responsible for equations and observables. A small inference crate should own whitening, rank-revealing factorization, principal angles, Schur complements, Fisher metrics, schedule optimization, and adversarial nuisance expansion. Registry/data crates should persist provenance and verdicts. This avoids contaminating domain physics with claim semantics while allowing every differentiable model to use the same falsification machinery.

## Mathematical implementation note

Do not form `P_N = N(N^T N)^-1 N^T` directly in production. Whiten first, use pivoted QR or SVD to obtain a rank-revealing nuisance basis, project the target with that orthonormal basis, and report singular values/effective rank. The very large nuisance condition number observed in the boundary prototype is itself a warning that normal-equation implementations would be numerically fragile.

## Required falsification gates

A physical claim is not promoted from simulation unless its target component survives independently justified nuisance-family expansion, mesh/quadrature refinement, held-out control conditions, and uncertainty propagation. Exact target-shaped nuisance degeneracy must reduce identifiability to zero.

## Implementation note

The compact prototype currently depends on NumPy and is intentionally isolated under `experiments/`. The repository is Rust-first; production integration should move the linear algebra and experiment-design kernel into an existing or dedicated Rust crate rather than adding NumPy as a project runtime dependency.

The remote workstation connector was unavailable during this integration, so repository-native Cargo/pytest/CI execution has not been claimed. The synthetic experiment itself was executed in the chat compute environment; repository code was inspected through the GitHub connection.
