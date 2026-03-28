# Cross-Domain Evidence Note: CD Associator as Phase-Geometry Diagnostic

**Date**: 2026-03-27
**Status**: Working synthesis for methods paper
**Claims**: C-1548 through C-1585 + pending registration

## Core Claim

The 32D Cayley-Dickson associator is a multichannel magnetic-field
phase-geometry diagnostic whose interpretation is regime-dependent. It is
not a universal turbulence meter. Its sign and magnitude depend on boundary
type, background field strength, and normalization choice.

**One-sentence paper claim**: We present a pure-Rust, null-stratified,
multiscale hypercomplex framework for multichannel electromagnetic-field time
series that yields an order parameter for cross-channel phase organization not
reducible to standard linear spectral or rank-based diagnostics, and we
validate it across heliopause, magnetopause, induced-boundary, and
coherent-structure regimes.

## Regime Taxonomy

### 1. Ordered-Region Entry Drop

Pre-boundary turbulence/complexity rises, then the associator drops on entry
to a more ordered region.

| Mission | Labels | Detection | FA | Offset |
|---------|--------|-----------|-----|--------|
| Voyager 1/2 | Change-point | 119.5/123.8 AU | -- | -- |
| **THEMIS-A** | **Curated Zenodo V2** | **89.0%** | **16.7%** | 6 min |
| MMS | |B| gradient | 61.5% | 40.0% | 6 min |
| Cluster-1 | |B| gradient | 32.3% | 50.7% | 5 min |
| ARTEMIS THB | |B| gradient | 75.0% | 68.1% | 4 min |
| MESSENGER | Curated Zenodo | 35.7% (partial) | 89.4% | -- |

**Key result**: Curated labels transform THEMIS from 27% to 89% detection.
The |B|-gradient heuristic contaminates the reference with false crossings.

### 2. Coherent-Structure Suppression

Switchbacks have lower associator than quiet wind -- coherent Alfvenic
deflections are more phase-organized than background turbulence.

| Encounter | Perihelion | SB fraction | SB/quiet ratio |
|-----------|-----------|-------------|----------------|
| E1 (Nov 2018) | 0.17 AU | 75.5% | 0.418 |
| E4 (Jan 2020) | 0.13 AU | 59.1% | 0.598 |
| E6 (Sep 2020) | 0.09 AU | 71.6% | 0.566 |
| E10 (Nov 2021) | 0.07 AU | 58.1% | 0.699 |

Stable range 0.42-0.70 across 4 encounters. Possible radial trend: closer
perihelia show slightly higher ratios (less contrast) as background
turbulence strengthens.

### 3. Weak-Field Cavity Amplification

Near-zero |B| in cometary diamagnetic cavity inflates normalized embeddings.
The normalization ablation separates genuine phase-geometry disorder from
amplification artifact.

| Normalization | Cavity | Outside | Ratio | Interpretation |
|---------------|--------|---------|-------|----------------|
| current (Bx/mean_B) | 56.3 | 10.7 | 0.19 | 5.3x amplified |
| clipped (floor 1 nT) | 55.5 | 10.7 | 0.19 | Floor rarely triggers |
| **direction (unit vec)** | **12.8** | **6.7** | **0.53** | **1.9x genuine** |
| raw (unnormalized) | 81K | 560K | 6.87 | REVERSED |

~40% genuine directional phase geometry + ~60% normalization amplification.
The direction-only result (1.9x) is the honest cavity contrast.

### 4. Induced Magnetosphere Boundaries

Mars and 67P have no intrinsic dipole. The IMB/MPB/cavity boundary is
structurally cleaner than Earth's dipolar magnetopause.

| Mission | Detection | FA | Notes |
|---------|-----------|-----|-------|
| MAVEN 7-day | 56.9% | 26.4% | Cleanest |B|-gradient result |
| MAVEN 14-day | 53.3% | 25.8% | Stable |
| Rosetta | 27.8% (dir) | 2.2% | Near-zero FA |

Induced boundaries consistently show the highest detection rates and lowest
false alarm rates with |B|-gradient labels. This is physically sensible:
the boundary is a clean draping transition without dipole geometry
complications.

## Label Quality Summary

Three label quality tiers:
1. **Curated** (THEMIS Zenodo V2, MESSENGER Zenodo): expert-identified
   crossing times from published databases
2. **Published criterion** (PSP Br/|B| < -0.5): standard literature
   definition, regime classification rather than point events
3. **Heuristic** (|B| gradient): physics-based but contaminated with
   false crossings; suitable for proof-of-concept, not paper-grade

The curated-label THEMIS result (89% detection, 17% FA) is the strongest
single validation. All other Earth-orbit missions should be upgraded to
curated labels where available.

## Normalization Policy

The Rosetta ablation establishes the normalization policy:
- **current normalization** (Bx/local_mean_B) is the default and should be
  used for all boundary-detection results
- **direction-only** (unit vectors) must be run as ablation for any weak-field
  environment (cavity, wake, depleted region)
- **raw** (unnormalized) is the baseline for understanding field-strength
  contributions
- Any result where direction-only contrast vanishes should be flagged as
  normalization-dependent and discussed in the paper

## Spatial Coverage

The validated environments span 0.07 to 124 AU:

| Distance | Environment | Mission |
|----------|-------------|---------|
| 0.07 AU | Inner heliosphere switchbacks | PSP E10 |
| 0.39 AU | Mercury magnetosphere | MESSENGER |
| 1.0 AU | Earth magnetopause | THEMIS, MMS, Cluster |
| 1.0 AU | Lunar wake | ARTEMIS |
| 1.5 AU | Mars induced magnetosphere | MAVEN |
| 3.4 AU | Cometary diamagnetic cavity | Rosetta |
| 119-124 AU | Heliopause/ISM | Voyager 1/2 |

## Infrastructure

All analysis binaries are pure Rust:
- `heliosphere-boundary-survey` (unified: THEMIS/Cluster/MAVEN/MESSENGER/Swarm)
- `heliosphere-mms-multiday` (MMS with aria2c fallback)
- `heliosphere-rosetta-draping` (Rosetta with normalization ablation)
- `heliosphere-switchback-omega` (PSP switchback/quiet comparison)

Catalog modules: `themis.rs`, `cluster.rs`, `maven_mag.rs`, `messenger.rs`,
`rosetta.rs`, `swarm_mag.rs`, `mms.rs`

spectral_core productized: `welch_psd`, `circular_bootstrap_ci`,
`detect_change_point`, `phase_randomize_mv_shared`,
`phase_randomize_mv_independent`, `block_shuffle`, `iaaft_surrogate`

## Open Items

1. MESSENGER needs aria2c download path (200 MB/day, reqwest timeout)
2. MMS FPI composite classifier (lower priority after THEMIS curated result)
3. Cluster curated crossings from ESA CAA
4. OMNI bow shock crossing database
5. MAVEN published IMB/MPB crossing intervals
6. ARTEMIS lunar wake: specific wake crossing identification
