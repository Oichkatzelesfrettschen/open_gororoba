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

## Required falsification gates

A physical claim is not promoted from simulation unless its target component survives independently justified nuisance-family expansion, mesh/quadrature refinement, held-out control conditions, and uncertainty propagation. Exact target-shaped nuisance degeneracy must reduce identifiability to zero.

## Implementation note

The compact prototype currently depends on NumPy and is intentionally isolated under `experiments/`. The repository is Rust-first; production integration should move the linear algebra and experiment-design kernel into an existing or dedicated Rust crate rather than adding NumPy as a project runtime dependency.
