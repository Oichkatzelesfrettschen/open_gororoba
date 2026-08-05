---
description: Adversarial review of the signed Son-Chekhova SFWM reconstruction
last_verified: 2026-08-05
evidence_class: bounded-source-reproduction-review
---

# Signed SFWM source reproduction adversarial review

Declare the source-reproduction result admissible only after the signed
mismatch identity, complete Eq. 6 prefactor, complex amplitudes, independent
component fixtures, frozen thickness grid, and unresolved material assumptions
are all visible in the retained outputs.

## Review boundary

The review covers `optics_core::sfwm_source`, the typed reproduction CLI, the
dense-index Rust reference, the independent 80-digit mpmath fixture, the
extraordinary Zelmon branch, and the legacy `sfwm.rs` exclusion boundary.
It does not adjudicate experimental counts, the fused-silica percentage, or
any P2C2 figure fit.

## Adversarial findings

| Question | Finding | Disposition |
| --- | --- | --- |
| Can positive coherence lengths determine the coherent Eq. 6 sign? | No. The implementation fixes SFWM and SHG from printed anchors and derives SPDC as `Delta_k_SFWM - Delta_k_SHG`, yielding a negative SPDC mismatch. | Retain the printed 3.4 micrometre SPDC length as a rounded residual, not as an independent override. |
| Can the old scalar fringe tests validate the new sign? | No. `abs(F)^2` is insensitive to mismatch sign, while the Eq. 6 bracket is not. | Keep old tests as characterization and use complex component controls. |
| Can the omitted Eq. 6 factor cancel in the ratio? | No. The factor contains `n_SH`, `lambda_SH`, and `Delta_k_SHG` and has no counterpart in Eq. 8. | Require the prefactor mutation to fail. |
| Can the near-0.048 result prove the material model? | No. The source susceptibility tuple is applied as a declared source input while the Wang sample identity remains unresolved. | Do not convert numerical agreement into a pristine-LN material claim. |
| Does the extraordinary branch follow directly from a printed source label? | No. It is a geometry and tensor inference from x-cut propagation along +x with z-polarized fields. | Retain ordinary and extraordinary envelopes and keep the branch assumption explicit. |
| Does the 0.1 micrometre grid resolve every fringe extremum? | Not exactly. The SPDC first maximum and period converge toward the analytic values under 0.05 and 0.01 micrometre refinements. | Freeze the final 0.1 micrometre grid and retain refinement errors; do not select a post-hoc grid. |
| Can an undefined direct rate be made finite with a floor? | No. A zero direct amplitude makes the ratio undefined. | Return `None` and test the zero-rate path. |
| Does a standalone fused-silica amplitude establish 25 percent of coincidences? | No. The total denominator, interfaces, collection, filters, and Supplement 1 derivation remain unavailable. | Leave C-838 blocked and do not run a substrate percentage transition. |

## Results that survive the attack

The paper-input path gives:

- signed mismatches `[0.0943421217, 1.0134169850, -0.9190748633]` per micrometre;
- identity defect `0` within f64 representation;
- derived coherence lengths `[33.3, 3.1, 3.4182119205]` micrometres;
- `R_cas/R_dir = 0.04774423559348579` at 10 micrometres.

The extraordinary Sellmeier path gives:

- signed mismatches `[-0.1245797599, -1.0215433250, 0.8969635651]` per micrometre;
- identity defect `0` within f64 representation;
- coherence lengths `[25.2175205384, 3.0753396127, 3.5024752127]` micrometres;
- `R_cas/R_dir = 0.04801092757771039` at 10 micrometres.

The two paths therefore support the source-model direct-dominance boundary
by more than 5x, while they do not support an unqualified claim that the
absolute coherence lengths reproduce the printed source anchors under one
unique material convention.

## Cross-implementation status

The dense-index Rust reference reproduces real and imaginary components at
five thickness fixtures. The pinned mpmath fixture reproduces both source
cases at the same five thicknesses. These are independent implementations of
the equations, not external replications of the paper or its hidden input
data.

The mutation controls detect:

1. positive-only SPDC mismatch;
2. omitted Eq. 6 prefactor;
3. substituted legacy susceptibilities;
4. phase and fringe misuse through the corrected source controls;
5. negative pump-field-squared and negative-thickness inputs;
6. zero direct-rate denominator handling.

## New combined insight

The old result near `0.0878783` was not merely a small parameter drift. The
legacy susceptibility substitution and the omitted Eq. 6 scale moved the
ratio in compensating directions. Applying the missing source scale and source
susceptibilities to the legacy stress calculation gives approximately
`0.0443353`, while the fully signed coherent path gives approximately
`0.0477`. The near match is therefore evidence that the omitted source scale
was load-bearing, not evidence that the old oracle was approximately valid.

The ratio remains near `0.048` when the extraordinary Sellmeier branch changes
the absolute SFWM coherence length from the printed `33.3` to `25.2`
micrometres. This separates two effects: the source rate ratio is relatively
stable under the declared branch change, while the absolute fringe positions
remain a material-convention discriminator. An authoritative temperature,
index, and susceptibility packet can falsify this separation.

## Claim boundary recommendation

Use the canonical transition path to record:

- C-832: source-model direct dominance survives the corrected signed coherent
  calculation, bounded to the declared source tuple;
- C-833: the extraordinary Sellmeier hierarchy survives as a qualitative
  hierarchy, with absolute lengths retained as branch-sensitive outputs;
- C-834: the nominal paper-input ratio is below five percent, while its
  deterministic uncertainty envelope crosses the five-percent boundary;
- C-839: scalar Maker-fringe structure survives under the corrected signed
  representation;
- C-838: remain provisional and blocked because the public packet lacks the
  total-coincidence derivation.

Do not interpret any of these results as an experimental-data reproduction or
as a refutation of Son and Chekhova.

## Final validation attack

The independent dense-index reference initially exceeded the repository's
warnings-as-errors argument-count limit. Grouping its reference inputs into a
typed record fixed the interface without changing any equation or generated
numerical output. The corrected reference passes clippy and all component
tests.

The legacy `make registry-verify-control-plane` lane is not a valid P2C1
scientific gate. Its first command names a removed `verify-origin-audit`
subcommand. Its current-subcommand probe reaches nineteen source paths removed
before the SFWM branch. The canonical `make validate-registry` lane verifies
SQLite authority, compatibility exports, cross-references, governance, and
artifact integrity successfully. The inherited failure is retained as a
separate infrastructure blocker and does not alter a claim status.

Adversarial verdict: the source reproduction survives the declared
implementation and numerical attacks. The material identity, Supplement 1,
absolute-rate normalization, and substrate percentage remain unresolved. No
P2C2 experimental-data conclusion is admissible from this bundle.
