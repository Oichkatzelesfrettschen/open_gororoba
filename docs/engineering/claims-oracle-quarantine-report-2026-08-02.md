---
description: P0 oracle quarantine and claim transition report
last_verified: 2026-08-02
evidence_class: methodology_audit
---

# P0 Oracle Quarantine and Claim Transition Report

## Outcome

The bounded claim audit is complete. The repository now retains executable
negative controls, source contracts, canonical status-note revisions, a typed
transition proposal record, a schema blocker, and post-change validation.

The audit does not promote any of the seven target claim families. It also
does not refute the underlying source propositions merely because their old
implementation validators are invalid.

No tensor Ward implementation, optics repair, SFWM sweep, heliosphere
refactor, LBM change, ultrametric change, GPU workload, or algebra-family
consolidation was started.

## Baseline binding

The audit evidence was generated from the pre-publication `main` checkout at
`954696e5eaf96530207ddcbea1c8f1ef9c259ceb`, with the P0 source changes held as
uncommitted work. Publication was rebased onto `origin/main` at
`743ad2c6`, and the seven canonical status-note mutations were replayed through
`gororoba-db` against that database before export. The live control plane
contains
1448 claims, 183 insights, and 232 experiments. The pre-publication inventory
recorded 494 binaries and 132 theorems; post-rebase control-plane verification
reports 496 binaries, 162 theorems, 156 kernel-checked claims, and 297 proof
files. The historical March campaign report contains 1300 claims and is
retained as a checkpoint, not substituted for the live count.

The host has a healthy NVIDIA GeForce RTX 4070 Ti and proprietary Vulkan
stack. P0 recorded that environment but launched no GPU workload.

The structural baselines remain tool-qualified. PMD CPD used 2057 paths and
42 tokens. The targeted jscpd lane used 536 files, 100 tokens, and 10 lines.
The lizard lane covered 24 files with complexity threshold 15 and length
threshold 1000.

## Negative controls

Eight controls pass as audit controls. Seven expose non-discriminating or
mismatched positive evidence paths; one separates integrator error from model
decay:

| Claim | Verdict | Observed defect |
| --- | --- | --- |
| C-820 | MethodologyInvalid | The on-shell zero factor hides a nonzero Ward integrand. |
| C-822 | MethodologyInvalid | The gravitational gate ignores perturbed tadpole and external-leg contributions. |
| C-865 | MethodologyInvalid | A 0.9 relative discrepancy passes the implementation tolerance 1.0 despite the stated 1e-10 claim. |
| C-866 | MethodologyInvalid | Scalar transmission cannot distinguish complex channel flux 1 from flux 2. |
| C-866 | MethodologyInvalid | Extinction is defined from the same inputs used by the optical-theorem ratio. |
| C-867 | MethodologyInvalid | The base Fano profile is even in detuning, so q cannot create the claimed asymmetry. |
| C-866 | MethodologyInvalid | The named lossless fixture retains gravitational decay. |
| C-866 | SurvivesChallenge | RK4 refinement is distinguishable from exact lossless two-mode evolution. |

The complete controls are retained in
`data/output/audit/2026-08-02/negative_controls.toml`.

## Claim transitions

Experiment verdicts remain separate from repository claim status.

| Claim | Evidence verdict | Proposed evolution | Canonical status after P0 |
| --- | --- | --- | --- |
| C-820 | MethodologyInvalid | Provisional implementation claim; split source gauge identity from tensor conformance. | Verified with explicit non-promotion note |
| C-821 | Inconclusive | Provisional; split scalar tadpole cross-validation from tensor Ward contribution. | Verified with explicit non-promotion note |
| C-822 | MethodologyInvalid | Provisional implementation claim; split source gravitational identity from tensor conformance. | Verified with explicit non-promotion note |
| C-864 | Inconclusive | Provisional phenomenological or source-derived TCMT mapping. | Verified with explicit non-promotion note |
| C-865 | MethodologyInvalid | Provisional source constraint and declared implementation predicate. | Verified with explicit non-promotion note |
| C-866 | MethodologyInvalid | Supersession proposal split into unitarity, reciprocity, time reversal, passive loss, and absorption balance. | Verified with explicit non-promotion note |
| C-867 | Inconclusive | Provisional complex Fano amplitude and weak-field source comparisons. | Verified with explicit non-promotion note |

The status-note mutations were applied through `gororoba-db` and produced
seven append-only `claim_revisions` rows after the publication rebase.
Compatibility claims were then
exported and integrity signatures refreshed. The exact revision hashes are in
`data/output/audit/2026-08-02/claim_transition_results.toml`.

The current CLI has no typed claim-status transition, successor-claim
creation, or transition-event operation. Therefore no successor IDs were
invented and no direct SQLite update was used. The bounded intake condition is
tracked as roadmap item `WS-CLAIM-TRANSITION-001` and documented in
`data/output/audit/2026-08-02/claim_transition_schema_blocker.toml`.

## Source contracts

The source-faithful boundary for Ward identities, scattering semantics, and
TCMT mapping is recorded in
`photon-graviton-ward-tcmt-source-contract-2026-08-02.md`.

The contract distinguishes electromagnetic gauge identities from the
off-shell gravitational lower-point identity, Ruan-Fan radial channels from a
generic multiport scattering matrix, and unitarity from reciprocity, time
reversal, passive loss, and absorption balance.

## Validation

The complete pre-publication command record is in
`data/output/audit/2026-08-02/validation_results.toml`. Publication revalidation
against the rebased source is in
`data/output/audit/2026-08-02/publication_revalidation_results.toml`.

Passing gates include scoped rustfmt, `gr_core` clippy with `-D warnings`,
851 `gr_core` tests, all three targeted optics test lanes, the fast shared
check, registry validation, and control-plane verification. The historical
`release-gate` clippy command is no longer valid after the rebase because the
profile is absent; the current repository-defined plain clippy command passes.

The rebased registry checker passes with no errors or warnings. Claims
verification retains 79 bounded path failures. Generic provenance verification
retains its known missing lane assignments and missing paths. These are not
attributed to the P0 changes.

The standalone manifest records the pre-publication run. Its verifier stops
because the audit outputs are not registered in canonical `artifact_paths`; it
is retained as a blocked manifest, not replayable evidence from the publication
checkout.

Post-change structural measurements are 6771 PMD clusters and 79579 duplicate
lines, 251 jscpd clone blocks and 6094 duplicate lines, and 446 lizard
functions with one CCN 16 warning for `extract_fano_params`. These metrics are
baselines for later modularization, not a reason to perform cleanup in P0.

## P1 handoff

P1 may begin only in a fresh bounded run. It must implement tensor-valued Ward
residuals with explicit conventions and compare them against an independent
dense-index or generated symbolic contraction. It must test a predeclared
kinematic grid, shell modes, loop types, diagram omissions, sign mutations,
and quadrature refinement. Tolerances must be justified by convergence and
conditioning before observing the final residuals.
