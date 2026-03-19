<!-- AUTO-GENERATED: See registry/research_narratives.toml (RN-041) -->

# Navigator: Conceptual Map of the open_gororoba Research Space

**Date:** 2026-03-15  
**Status:** Active navigation reference  
**Relevant claims:** C-001–C-041, C-1013, C-1045–C-1050, C-1137–C-1142  
**Related documents:** `docs/GRAND_SYNTHESIS.md`, `docs/EXCEPTIONAL_COSMOLOGY.md`

---

## Purpose

This document is a conceptual roadmap for traversing the claims, experiments, and code in this repository. It organises the research space into navigable layers and identifies the principal entry points, open questions, and dependency chains.

---

## Layer 0 — Algebraic Foundations

**Entry point:** `crates/algebra_analysis/`, `proofs/verified/`

The entire research programme rests on Cayley-Dickson algebra. All other layers inherit structure from the CD tower.

| Concept | Key claims | Code |
|---------|-----------|------|
| CD non-associativity onset | C-001, C-002 | `crates/algebra_analysis/src/cd_mul.rs` |
| 42 assessors / 7 box-kites | C-003, C-013 | `src/boxkites.rs`, `src/reggiani.rs` |
| PSL(2,7) symmetry | C-004 | `src/boxkites.rs` |
| Annihilator geometry (S³) | C-014 | `src/annihilator.rs` |
| Hurwitz theorem | C-031 | `src/hurwitz.rs` |
| Quantised gap (dim=16,32) | C-1137, C-1140 | `src/phase_transition.rs` |
| ZD graph structure | C-1141 | `src/phase_transition.rs` |
| XOR scatter-gather duality | C-1142 | `src/gpu/tensor_avt/` |

**Falsification threshold:** Any claim that requires sedenions to be associative or zero-divisor-free is automatically refuted (C-001, C-002 are Rocq-verified and not revisable).

---

## Layer 1 — Materials Science

**Entry point:** `crates/materials_core/`, `crates/quantum_core/`, `crates/optics_core/`

| Concept | Key claims | Code | Status |
|---------|-----------|------|--------|
| Nonlocal metamaterial topology | C-010 | `crates/materials_core/src/nonlocal_metamaterial.rs` | Closed/Negative-Result |
| Pathion metamaterial mapping | C-053 | — | Verified/Toy |
| Tight-binding sedenion basis | — | `crates/materials_core/src/tight_binding.rs` | Active |
| Magnonic crystal modes | — | `crates/quantum_core/src/magnonic_crystal.rs` | Active |
| AFLOW/NOMAD integration | — | `crates/data_core/src/catalogs/` | Active |

**Key lesson:** The 7-clique ZD topology requires explicit non-local coupling bridges. Local lattice models alone cannot reproduce holographic mode selection.

---

## Layer 2 — Particle Physics and Field Theory

**Entry point:** `crates/cosmology_core/`, `crates/qgp_scaling/`, `crates/quantum_core/`

| Concept | Key claims | Code | Status |
|---------|-----------|------|--------|
| Gravastar–sedenion bridge | C-011 | `src/gravastar.rs`, `src/tov.rs` | Closed/Obstructed |
| MERA logarithmic entropy | C-052 | `crates/quantum_core/` | Verified |
| Sedenion eigenspectra vs PDG | — | — | Closed/Analogy |
| Negative dimension dark energy | C-012 | `crates/spectral_core/src/neg_dim.rs` | Refuted |
| R_AA straggling (QGP) | — | `crates/qgp_scaling/src/straggling.rs` | Active |

---

## Layer 3 — Heliospheric Physics

**Entry point:** `crates/data_core/src/catalogs/`, `crates/gr_core/`

| Concept | Key claims | Code | Status |
|---------|-----------|------|--------|
| Parker spiral validation | C-1162 | `crates/data_core/` | Verified |
| GCR modulation potential | C-1171, C-1210 | — | Verified |
| DM null invariance | C-1156 | — | Verified |
| LBM Knudsen boundary | C-1159 | `crates/lbm_core/` | Verified |
| Pioneer merged data | — | `src/catalogs/pioneer.rs` | Active |
| Voyager merged data | — | `src/catalogs/voyager.rs` | Active |

**Data dependency chain:**
```
SPDF/PDS-PPI → data_core parsers → OMNI-merged hourly CSV
    → heliospheric claims (C-1013, C-1045, C-1047, C-1048)
```

---

## Layer 4 — Solar System Anomalies

**Entry point:** `crates/gr_core/src/fractal_metric.rs`, `crates/gr_core/src/nbody_integration.rs`

| Concept | Key claims | Code | Status |
|---------|-----------|------|--------|
| Pioneer fractal metric fit | — | `crates/gr_core/src/fractal_metric.rs` | Provisional |
| Flyby anomaly Chingon coupling | C-952 | — | Provisional |
| JPL DE440 ephemeris | C-953 | `crates/gr_core/src/nbody_integration.rs` | Verified |

**Known magnitude problem:** The D_f ~ 2.7 metric gives the correct sign for the Pioneer anomaly but is ~4 orders of magnitude too large. The thermal-recoil explanation (Turyshev et al. 2012) accounts for the observed magnitude. The fractal hypothesis is *not* presented as an alternative solution, but as a structural curiosity warranting investigation.

---

## Layer 5 — Cosmology

**Entry point:** `crates/cosmology_core/`, `crates/spectral_core/`

| Concept | Key claims | Code | Status |
|---------|-----------|------|--------|
| Spectral dimension running | C-039 | `crates/cosmology_core/src/spectral.rs` | Verified (toy) |
| Exceptional cosmology (E8/F4) | C-035–C-041 | — | See below |
| GWTC-3 BH mass multimodality | C-025 | `src/scripts/gwtc3_*.py` | Active |
| CMB Planck constraints | — | `crates/cosmology_core/` | Active |
| Orthoplex heat kernel | C-931 | — | Verified |
| Dark energy EOS (orthoplex) | C-932 | — | Verified/Toy |

**Exceptional cosmology status:**

- **Refuted strong claims:** w₀ = −5/6 (C-038), n_s from D_eff (C-040), F4 → string D=26 (C-041), γ ~ ε (C-037), triality clustering (C-036)
- **Verified exact identities:** F4 Casimir ε = 1/4 (C-035)
- **Interpretation:** The exceptional algebra is a genuine feature of the mathematical structure at the small-scale (D_s → 2) regime but does not drive large-scale cosmological parameters.

---

## Dependency and Refutation Summary

```
CD algebra (L0)
    ├── ZD graph topology  ──▶ metamaterial design (L1) [negative result: C-010]
    │                      ──▶ Wick friction bridge (L0→L5) [C-1138]
    ├── dimensional tower  ──▶ DFT force map (speculative)
    │                      ──▶ Chingon flyby coupling (L4) [provisional: C-952]
    ├── associator gap=4   ──▶ universal conjecture (gap universality)
    └── defect density     ──▶ fractal floor D_f ~ 2.73 conjecture
heliospheric data (L3)
    ├── Parker spiral      ──▶ B-field scaling verified
    ├── DM contribution    ──▶ null result verified
    └── LBM boundary       ──▶ Knudsen breakdown at 30–50 AU
Pioneer/flyby (L4)
    └── D_f ~ 2.7          ──▶ correct sign, magnitude mismatch
LBM galaxy sims (L5)
    └── D_f = 2.732 ± 0.034 ──▶ near-coincidence with Pioneer metric [open]
```

---

## Quick Reference: Claim Status by Domain

| Domain | Verified | Refuted | Provisional |
|--------|----------|---------|-------------|
| CD algebra | C-001–C-004, C-013–C-014, C-031, C-1137–C-1142 | — | gap universality |
| Materials | C-053 | C-010 (negative result) | 7-bridge design |
| Particle/field | C-052 | C-012, C-011 (obstructed) | |
| Heliosphere | C-1162, C-1156, C-1159, C-1171, C-1210 | — | LBM fractal boundary |
| Solar system | C-953 | — | C-952, Pioneer fit |
| Cosmology | C-035, C-039 | C-036–C-038, C-040, C-041 | D_f floor, GWTC-3 |

---

*For full claim text, see `registry/claims.toml`. For experiment data, see `registry/experiments.toml`.*
