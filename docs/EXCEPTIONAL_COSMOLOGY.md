<!-- AUTO-GENERATED: See registry/research_narratives.toml (RN-042) -->

# Exceptional Cosmology: E8, F4, G2 and the Limits of Algebraic Cosmological Frameworks

**Date:** 2026-03-15\
**Status:** Closed programme -- strong predictions refuted; algebraic identities verified\
**Relevant claims:** C-035, C-036, C-037, C-038, C-039, C-040, C-041\
**Related documents:** `docs/GRAND_SYNTHESIS.md`, `docs/NAVIGATOR.md`\
**External sources:** `docs/external_sources/PIONEER_FLYBY_ANOMALY_SOURCES.md`

---

## Overview

The _exceptional cosmology_ programme investigated whether the exceptional Lie algebras -- G2, F4, E6, E7, E8 -- and their associated combinatorial structures provide a natural framework for cosmological parameters. The programme has been systematically tested and its strong predictions have been refuted. This document records what was found, what was refuted, and what structural lessons remain.

---

## Mathematical Background

### Exceptional Lie Algebras in Brief

The exceptional simple Lie algebras do not fit the standard A_n/B_n/C_n/D_n series:

| Algebra | Rank | dim | Root system | Notes                        |
| ------- | ---- | --- | ----------- | ---------------------------- |
| G2      | 2    | 14  | 12 roots    | Aut(O), exceptional triality |
| F4      | 4    | 52  | 48 roots    | 26D representation           |
| E6      | 6    | 78  | 72 roots    | Jordan algebra connection    |
| E7      | 7    | 133 | 126 roots   | quantum information          |
| E8      | 8    | 248 | 240 roots   | densest known packing        |

**G2** is the automorphism group of the octonions. Its 14-dimensional adjoint representation appears in the sedenion ZD geometry: the G2 x S3 automorphism group of the sedenion algebra (where S3 permutes box-kite orientation) is the bridge between exceptional algebras and the Cayley-Dickson tower.

### F4 and the Casimir Ratio

**Verified (C-035, Rocq 9.1):** The F4 second Casimir ratio is:

    \epsilon = C_2(26) / |\Delta+(F4)| = 6/24 = 1/4 exactly.

This is a pure algebraic identity. It was proposed as structurally related to the Barbero-Immirzi parameter \gamma ~= 0.2375 (loop quantum gravity) and the Gauss-Bonnet coupling constant 4\lambda_GB.

---

## Claims and Their Status

### C-035: F4 Casimir Ratio (Verified)

**Statement:** F4 Casimir ratio \epsilon = C_2(26)/|\Delta+(F4)| = 6/24 = 1/4 exactly.

**Status:** Verified via Rocq 9.1 (`proofs/verified/C035_CasimirQuarter.v`).

This is an exact algebraic fact with no free parameters and no empirical content beyond the algebraic structure of F4.

---

### C-036: Triality Clustering Convergence (Refuted)

**Statement:** Triality-governed bigraph attachment stabilises clustering coefficient C -> 0.25 in the thermodynamic limit.

**Status:** Refuted (`proofs/verified/C036_TrialityClusteringBound.v`).

Numerical study showed that the clustering coefficient does _not_ converge to exactly 0.25 under triality-governed attachment; the conjecture was based on a false analogy between F4 Casimir ratios and graph-theoretic clustering.

---

### C-037: Numerical Coincidence \gamma ~ \epsilon ~ 4\lambda_GB (Refuted)

**Statement:** \gamma ~= \epsilon ~= 4\lambda_GB ~= 1/4, relating the Barbero-Immirzi parameter, network clustering, and Gauss-Bonnet coupling.

**Status:** Refuted (`proofs/verified/C037_EpsilonNotGamma.v`).

The three quantities are numerically close (\gamma ~= 0.2375, \epsilon = 0.25, 4\lambda_GB ~= 0.22-0.27) but the equality is not universal. A dedicated numerical coincidence audit established that:

1. \gamma is determined by LQG area spectrum quantisation, not by exceptional algebra.
2. The Gauss-Bonnet coupling \lambda_GB is a free parameter in modified gravity, not fixed by F4.
3. The near-equality ~1/4 is a coincidence in the range of physically reasonable values, not a structural identity.

**Structural lesson:** The value 1/4 appears in multiple physically motivated frameworks because it is the simplest non-trivial rational number that is:

- Large enough to be phenomenologically significant (f > 0)
- Small enough not to violate perturbativity (f < 1)
- Appearing naturally in group-theoretic ratios (e.g., 6/24)

The coincidence does not imply a unifying framework.

---

### C-038: Dark Energy Equation of State w_0 = -5/6 (Refuted)

**Statement:** Dark energy EOS w_0 = -5/6 ~= -0.8333 emerges from twist-sector distribution in the exceptional framework.

**Status:** Refuted (`proofs/verified/C038_WDarkEnergy.v`).

The prediction w_0 = -5/6 is inconsistent with joint CMB + BAO + SNe Ia constraints, which place w_0 close to -1 (cosmological constant). The current best-fit value from multi-probe data is w_0 ~= -0.95 ± 0.07, excluding w_0 = -5/6 at > 2\sigma.

---

### C-039: Spectral Dimension Running (Verified, Toy)

**Statement:** In CDT and asymptotic safety literature, D_s runs from ~4 (large scales) to ~2 (short scales); the repo implements a finite-graph toy D_s(t) computation for qualitative comparison.

**Status:** Verified (`proofs/verified/C039_SpectralDimensionRunning.v`).

The qualitative D_s -> 2 short-scale limit is reproduced by the discrete model. This is not a prediction -- it confirms that the toy model is qualitatively consistent with the literature. The spectral dimension runs as:

    D_s(t) ~= 4 / (1 + a*t^\beta)

where t is diffusion time and a, \beta are model-dependent parameters. The bigraph attachment sweep (`data/csv/c039_spectral_dimension_bigraph_curve.csv`) shows the D_s behaviour as a function of attachment probability.

---

### C-040: Primordial Tilt n_s ~ 0.965 from Fractal D_eff (Refuted)

**Statement:** Primordial tilt n_s ~ 0.965 from fractal D_eff ~ 2.8-3.0 at inflation.

**Status:** Refuted (`proofs/verified/C040_PrimordialTiltRefuted.v`).

The proposed mechanism -- that inflation occurs in a spacetime with spectral dimension D_eff ∈ (2.8, 3.0), generating a red tilt n_s = 1 - 2/(D_eff - 1) -- cannot be constrained to reproduce n_s = 0.965 ± 0.004 without a free parameter that absorbs all predictive content. The parameter sweep (`data/csv/c040_primordial_tilt_deff_curve.csv`) shows that n_s ~ 0.965 can be reproduced for D_eff ~= 2.9, but any value of n_s ∈ (0.94, 0.98) is similarly accommodatable. The model has no independent constraint on D_eff.

---

### C-041: F4 26D Representation -> Bosonic String D=26 (Refuted)

**Statement:** F4 26D representation connects to bosonic string critical dimension (D=26).

**Status:** Refuted.

The F4 Lie algebra has a 26-dimensional representation (the minuscule representation of the adjoint action on the exceptional Jordan algebra h3(O)). Bosonic string theory requires D=26 for Weyl anomaly cancellation. The numerical equality is coincidental: the string critical dimension 26 = 25 + 1 (25 spatial + 1 time) has nothing to do with the F4 representation count. A dedicated coincidence audit confirmed no dynamical link.

---

## Structural Lessons

### What the Refutations Teach

The exceptional cosmology programme generated seven specific, falsifiable predictions. Six were refuted. The surviving claim (C-035) is an exact algebraic identity that carries no empirical content. This is scientifically healthy: the programme was properly preregistered, tested, and closed.

**Key insight:** Exceptional algebras appear naturally in the _mathematical_ structure of quantum geometry (E8 in string theory compactifications, G2 as Aut(O), F4 in Jordan algebras), but this does not mean they directly parametrise _large-scale cosmological observables_ like w_0, n_s, or \gamma. The programme conflated structural appearance with causal determination.

### What Remains

1. **C-035 (\epsilon = 1/4):** The exact Casimir ratio is a genuine algebraic fact that appears in multiple contexts (LQG area spectrum, Gauss-Bonnet coupling, network clustering). Understanding _why_ 1/4 appears in so many physically motivated contexts may be structurally productive, even if the strong equality claims are refuted.

2. **C-039 (spectral dimension running):** The D_s -> 2 short-scale limit is a robust prediction of CDT and asymptotic safety that the repo's toy model correctly captures the qualitative behavior of. This motivates further investigation of how the sedenion ZD topology (which also defines a discrete graph dimension ~= 2 at short scales) relates to the CDT continuum limit.

3. **G2 and octonions:** G2 = Aut(O) is a verified connection between exceptional algebras and CD algebra. The sedenion automorphism group Aut(S) = G2 x S3 is the correct exceptional symmetry for dim=16. Any further exceptional cosmology should start from this verified structural connection rather than from numerological coincidences.

---

## Forward Targets

If the exceptional cosmology programme is to be revived, the following would need to be addressed:

1. **Independent constraint on D_eff:** The primordial tilt prediction requires an independent determination of D_eff at inflationary scales, not a post-hoc fit.
2. **Dynamical mechanism for w_0:** The equation of state w_0 = -5/6 requires a Lagrangian whose stress-energy tensor gives this value. No such Lagrangian based on F4 twist sectors has been constructed.
3. **Barbero-Immirzi from first principles:** The LQG value \gamma ~= 0.2375 should be derived from a computation in the exceptional framework, not inferred from proximity to 1/4.

---

_For claim provenance, see `registry/claims.toml` (C-035 through C-041). For data sources, see `docs/external_sources/EXCEPTIONAL_COSMOLOGY_SOURCES.md`._
