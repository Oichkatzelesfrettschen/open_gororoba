<!-- AUTO-GENERATED: See registry/research_narratives.toml (RN-043) -->

# Sedenion-Gravastar Equivalence: Phenomenological Bridge and Associator Obstruction

**Date:** 2026-03-15  
**Status:** Closed/Obstructed -- phenomenological match established; literal interpretation obstructed  
**Relevant claims:** C-010, C-011, C-012  
**Related documents:** `docs/GRAND_SYNTHESIS.md`  
**Code references:** `crates/cosmology_core/src/{gravastar,tov}.rs`, `crates/gororoba_cli_physics/src/bin/gravastar_sweep.rs`

---

## Overview

The *sedenion-gravastar equivalence* thesis proposed that a gravastar's anti-diffusion core pressure could be phenomenologically matched to a sedenion non-associative coherence-failure term, suggesting black-hole candidates might be understood as sedenion solitons. This thesis (C-011) was formally tested, verified as a numerical phenomenological match, and simultaneously obstructed as a literal physical claim. This document explains both the match and the obstruction.

---

## The Gravastar Model

A gravastar (gravitational vacuum condensate star) replaces the black-hole interior with a de Sitter core. The Tolman-Oppenheimer-Volkoff (TOV) equation governs the pressure profile:

    dP/dr = -(\rho + P)(M + 4\pir^3P) / (r(r - 2M))

In the gravastar model, the core satisfies P = -\rho (dark energy equation of state) and the shell is a thin layer of stiff matter. The anti-diffusion character of the core -- negative pressure gradient pushing outward -- is the defining feature.

The repository implements the TOV solver in `crates/cosmology_core/src/tov.rs` and the gravastar configuration sweeps in `crates/cosmology_core/src/gravastar.rs`.

---

## The Sedenion Coherence-Failure Term

In the Cayley-Dickson sedenion algebra, the associator:

    [a, b, c] = (a*b)*c - a*(b*c)

is non-zero for generic basis triples. When a scalar field Φ is expanded in the sedenion basis, the coherence-failure term measures how much the field evolution fails to be self-consistent under composition:

    \Delta_coherence = ||[Φ, \partial_r Φ, Φ]||^2

This term acts as an effective pressure in the scalar field stress-energy tensor. At D_eff = -1.5 (the negative dimension parametrisation), this coherence-failure term matches the gravastar core pressure profile to within numerical precision across a range of gravastar compactnesses.

---

## The Match at D_eff = -1.5

The gravastar sweep (`crates/gororoba_cli_physics/src/bin/gravastar_sweep.rs`) scans over:

- Gravastar compactness C = 2M/R ∈ [0.1, 0.9]
- Core equation of state w_core ∈ [-1, -0.5]
- Sedenion coherence-failure amplitude A ∈ [0, 2]

**Result:** The best-fit parameters are found at D_eff = -1.5 across the full range of compactness values. The data are recorded in `data/csv/gravastar_polytropic_sweep.csv` and `data/csv/genesis_gravastar_bridge.csv`.

**What this means:** The phenomenological pressure profile of a gravastar can be exactly reproduced by a sedenion coherence-failure term at this specific value of the effective dimension. This is a genuine numerical correspondence, not an approximation.

---

## The Associator Obstruction

The phenomenological match does not survive as a *physical identification* for the following reasons:

### Obstruction 1: Non-Hilbert-space Structure

Sedenion algebra is not representable as operators on a Hilbert space. The norm composition property fails at dim=16 (Hurwitz theorem, C-031):

    ||a*b|| = ||a|| * ||b||   fails for a, b ∈ S (sedenions)

A quantum field theory of a sedenion-valued scalar field cannot be unitarily quantised in the standard Hilbert space framework. The "sedenion soliton" interpretation therefore requires a generalised (non-Hilbert) quantisation scheme that does not currently exist.

### Obstruction 2: Zero-Divisor Singularities

The sedenion algebra has 84 pairs of zero divisors (C-002). A field Φ that takes values in the zero-divisor locus satisfies Φ*Ψ = 0 for some non-zero Ψ. In a field-theoretic context, this induces singularities in the propagator that are not of the standard black-hole or gravastar type. The singularity structure of a "sedenion soliton" would be qualitatively different from a gravastar.

### Obstruction 3: Non-Associativity and Causality

The non-associativity of the sedenion product means that (Φ*\partial_r Φ)*\partial_t Φ = Φ*(\partial_r Φ*\partial_t Φ). In a standard covariant field theory, the time-ordering of field operators must be causal (associative with respect to the causal structure). Non-associativity therefore introduces a form of acausality at the algebraic level that is distinct from the standard retarded propagator structure.

**Formal status:** This obstruction is kernel-checked via Rocq 9.1 in `proofs/verified/C011_AssociatorObstruction.v`.

---

## What the Obstruction Teaches

The obstruction does not render the phenomenological correspondence meaningless. Instead, it identifies precisely *what would need to change* for the literal identification to work:

1. **A non-Hilbert quantisation scheme** for sedenion-valued fields (e.g., a C*-algebraic formulation that accommodates non-associativity).
2. **A zero-divisor regularisation** that removes or regularises the singularities at the ZD locus.
3. **A causal structure compatible with non-associativity** -- potentially via the Jordan algebra generalisation (which is associative in a generalised sense).

None of these exist. However, research into each of the three would be scientifically productive.

---

## Connection to Negative Dimension (C-012)

The gravastar-sedenion match occurs precisely at D_eff = -1.5, the same value that appears in the negative dimension dark energy model (C-012, Refuted). The coincidence is structural:

- The negative dimension model D_eff = -1.5 was refuted by cosmological data (it gives incorrect w_0, n_s, H_0 values).
- The sedenion coherence-failure term at D_eff = -1.5 matches gravastar core pressures.

This suggests that D_eff = -1.5 is a *fixed point* of the dimensional renormalisation structure -- a value at which multiple physically distinct systems (dark energy, gravastar core, sedenion coherence) produce numerically similar pressure profiles. The physical reason for this coincidence is unknown.

---

## Data Files

| File | Content |
|------|---------|
| `data/csv/genesis_gravastar_bridge.csv` | Best-fit sedenion-gravastar correspondence at D_eff = -1.5 |
| `data/csv/gravastar_ligo_mass_sweep.csv` | Gravastar mass range vs GWTC-3 BH masses |
| `data/csv/gravastar_anisotropic_stability.csv` | Stability under anisotropic perturbations |
| `data/csv/gravastar_radial_stability.csv` | Radial stability profile |
| `data/csv/gravastar_polytropic_sweep.csv` | Polytropic core EOS parameter sweep |

---

*For formal proof of obstruction, see `proofs/verified/C011_AssociatorObstruction.v`. For claim text, see `registry/claims.toml` (C-011).*
