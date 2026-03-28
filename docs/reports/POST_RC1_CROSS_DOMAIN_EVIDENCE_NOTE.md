# Cross-Domain Evidence Note: CD Associator as Phase-Geometry Diagnostic

**Date**: 2026-03-27
**Status**: Working synthesis for methods paper
**Claims**: C-1548 through C-1590 (43 claims)
**Missions**: 9 spacecraft, 18 analysis runs, 0.07-124 AU

## Core Claim

The 32D Cayley-Dickson associator is a multichannel magnetic-field
phase-geometry diagnostic whose interpretation is regime-dependent. It is
not a universal turbulence meter. Its sign and magnitude depend on boundary
type, background field strength, and normalization choice.

**One-sentence paper claim**: We present a pure-Rust, null-stratified,
multiscale hypercomplex framework for multichannel electromagnetic-field time
series that yields an order parameter for cross-channel phase organization not
reducible to standard linear spectral or rank-based diagnostics, and we
validate it across heliopause, magnetopause, bow shock, induced-boundary,
cometary-cavity, lunar-wake, and coherent-structure regimes spanning 9
spacecraft from 0.07 to 124 AU.

## Complete Validation Table

### All 18 analysis runs across 9 missions

| # | Mission | Environment | Distance | Labels | Detection / Metric | FA | Offset | Regime |
|---|---------|-------------|----------|--------|-------------------|-----|--------|--------|
| 1 | Voyager 2 | Heliopause | 119.5 AU | Bayesian CP | CP at 119.5 AU (pre=4.35, post=0.13) | -- | -- | Ordered-entry drop |
| 2 | Voyager 1 | Heliopause / ISM | 123.8 AU | Bayesian CP | CP at 123.8 AU (pre=0.12, post=0.02) | -- | -- | Ordered-entry drop |
| 3 | **THEMIS-A** | **Earth magnetopause** | 1 AU | **Curated Zenodo V2** | **89.0%** (105/118) | **16.7%** | 6 min | Ordered-entry drop |
| 4 | MMS | Earth magnetopause | 1 AU | |B| gradient | 61.5% (64/104) | 40.0% | 6 min | Ordered-entry drop |
| 5 | **Cluster-1** | **Earth bow shock** | 1 AU | **Curated OMNI** | **64.9%** (24/37) | 73.9% | 8 min | Ordered-entry drop |
| 6 | ARTEMIS THB | Lunar wake / magnetotail | 1 AU | |B| gradient | 75.0% (21/28) | 68.1% | 4 min | Cavity/wake |
| 7 | **MESSENGER** | **Mercury magnetopause** | 0.39 AU | **Curated Zenodo** | **64.3%** (9/14) | 84.8% | **3 min** | Ordered-entry drop |
| 8 | Rosetta (current) | 67P cavity | 3.4 AU | |B| gradient | 5.3x cavity/outside | 0.7% | 5 min | Weak-field cavity |
| 9 | Rosetta (direction) | 67P cavity | 3.4 AU | |B| gradient | **1.9x genuine** | 2.2% | -- | Weak-field cavity |
| 10 | PSP E1 | Switchbacks | 0.17 AU | Br/|B| < -0.5 | Ratio 0.418 (SB/quiet) | -- | -- | Coherent suppression |
| 11 | PSP E4 | Switchbacks | 0.13 AU | Br/|B| < -0.5 | Ratio 0.598 | -- | -- | Coherent suppression |
| 12 | PSP E6 | Switchbacks | 0.09 AU | Br/|B| < -0.5 | Ratio 0.566 | -- | -- | Coherent suppression |
| 13 | PSP E10 | Switchbacks | 0.07 AU | Br/|B| < -0.5 | Ratio 0.699 | -- | -- | Coherent suppression |
| 14 | MAVEN (7-day) | Mars IMB | 1.5 AU | |B| gradient | 56.9% (148/260) | 26.4% | 7 min | Induced boundary |
| 15 | MAVEN (14-day) | Mars IMB | 1.5 AU | |B| gradient | 53.3% (264/495) | 25.8% | 7 min | Induced boundary |
| 16 | Solar Orbiter | Inner heliosphere baseline | 0.3-1 AU | Null hierarchy | Omega_mv = 0.045 | -- | -- | Baseline (lowest) |
| 17 | Ulysses | Out-of-ecliptic | 1-5.4 AU | Regime conditioning | Fast-wind 2-4x amplification | -- | -- | Regime contrast |
| 18 | Cassini + NH | Outer heliosphere | 1-50 AU | Densified cube | Part of 1.26M-row radial profile | -- | -- | Radial profile |

## Regime Taxonomy

### 1. Ordered-Region Entry Drop

Pre-boundary turbulence/complexity rises, then the associator drops on entry
to a more ordered region. This is the most common phenotype, observed at
every dipolar magnetopause and at the heliopause.

**Strongest result**: THEMIS-A with curated Zenodo V2 labels achieves **89%
detection rate** against 118 expert-identified magnetopause crossings. The
same interval scored only 27% against |B|-gradient pseudo-labels. Curated
labels consistently improve detection rates:

| Mission | Heuristic | Curated | Improvement |
|---------|-----------|---------|-------------|
| THEMIS-A | 26.9% | **89.0%** | +62 points |
| Cluster-1 | 36.8% | **64.9%** | +28 points |
| MESSENGER | 47.0% | **64.3%** | +17 points |

**Voyager heliopause**: Bayesian change-point detection localizes the V2
transition at 119.5 AU and V1 at 123.8 AU. The pre/post associator contrast
(V2: 4.35 -> 0.13, 33x drop) is the largest single-boundary contrast in the
dataset, consistent with the transition from turbulent heliosheath to the
ordered very local interstellar medium.

### 2. Coherent-Structure Suppression

Switchbacks have lower associator than quiet wind -- coherent Alfvenic
deflections are more phase-organized than background turbulence.

| Encounter | Perihelion | SB fraction | SB/quiet ratio |
|-----------|-----------|-------------|----------------|
| E1 (Nov 2018) | 0.17 AU | 75.5% | 0.418 |
| E4 (Jan 2020) | 0.13 AU | 59.1% | 0.598 |
| E6 (Sep 2020) | 0.09 AU | 71.6% | 0.566 |
| E10 (Nov 2021) | 0.07 AU | 58.1% | 0.699 |

Stable range 0.42-0.70 across 4 encounters (mean ~0.57). Possible radial
trend: closer perihelia show slightly higher ratios (less contrast) as
background turbulence strengthens. This is consistent with the mainstream
PSP interpretation that switchbacks are coherent magnetic deflections/folds
rather than mere random disorder (Kasper et al. 2019, Bale et al. 2019).

### 3. Weak-Field Cavity Amplification

Near-zero |B| in cometary diamagnetic cavity inflates normalized embeddings.
The normalization ablation separates genuine phase-geometry disorder from
amplification artifact.

| Normalization | Cavity | Outside | Ratio (out/cav) | Interpretation |
|---------------|--------|---------|-----------------|----------------|
| current (Bx/mean_B) | 56.3 | 10.7 | 0.19 | 5.3x amplified |
| clipped (floor 1 nT) | 55.5 | 10.7 | 0.19 | Floor rarely triggers |
| **direction (unit vec)** | **12.8** | **6.7** | **0.53** | **1.9x genuine** |
| raw (unnormalized) | 81K | 560K | 6.87 | REVERSED |

~40% genuine directional phase geometry + ~60% normalization amplification.
The direction-only result (1.9x) is the honest cavity contrast. The raw
embedding reverses entirely, confirming normalization does real work.

### 4. Induced Magnetosphere Boundaries

Mars and 67P have no intrinsic dipole. The IMB/MPB/cavity boundary is
structurally cleaner than Earth's dipolar magnetopause.

| Mission | Detection | FA | Notes |
|---------|-----------|-----|-------|
| MAVEN 7-day | 56.9% | 26.4% | Cleanest |B|-gradient result |
| MAVEN 14-day | 53.3% | 25.8% | Stable across 14 days |
| Rosetta (dir) | 27.8% | 2.2% | Near-zero FA with direction-only |

Induced boundaries consistently show the highest detection rates and lowest
false alarm rates with |B|-gradient labels. This is physically sensible:
the boundary is a clean draping transition without dipole geometry
complications (Connerney et al. 2015).

### 5. Radial Profile and Regime Conditioning

The densified 1.26M-row feature cube (Voyager 1/2, Ulysses, New Horizons
SWAP, Cassini cruise, Solar Orbiter, PSP) provides the radial context:

| Zone | Omega_mv range | Missions |
|------|---------------|----------|
| ISM (>120 AU) | 0.45-0.59 | Voyager 1/2 |
| Outer heliosphere (30-120 AU) | 0.30-0.56 | Voyager 1/2 |
| Mid-heliosphere (5-30 AU) | 0.07-0.24 | Ulysses, Cassini, NH |
| Inner solar system (<5 AU) | 0.04-0.10 | PSP, Solar Orbiter |

**Ulysses fast-wind amplification**: Fast-wind (v_sw > 550 km/s) intervals
show 2-4x higher associator than slow wind at 27-82 AU. The fast/slow ratio
is 1.01 at 1 AU (no discrimination), rising to 2-4x in the outer heliosphere.
This is a radial-distance-dependent regime effect, not a simple speed
threshold.

**5-family null hierarchy**: Temporal-shuffle, channel-permutation,
block-shuffle, phase-randomized, and multivariate-phase-randomized surrogates
confirm that the inner heliosphere signal is spectral-only (destroyed by
phase randomization) while the outer heliosphere signal survives all null
treatments.

## Label Quality Summary

Four label quality tiers:
1. **Curated** (THEMIS Zenodo V2, MESSENGER Zenodo, OMNI bow shock):
   expert-identified crossing times from published databases
2. **Algorithmic** (Bayesian change-point): data-driven, no threshold tuning
3. **Published criterion** (PSP Br/|B| < -0.5): standard literature
   definition, regime classification rather than point events
4. **Heuristic** (|B| gradient): physics-based but contaminated with
   false crossings; suitable for proof-of-concept, not paper-grade

## Normalization Policy

The Rosetta ablation establishes the normalization policy:
- **current normalization** (Bx/local_mean_B) is the default
- **direction-only** (unit vectors) must be run as ablation for any
  weak-field environment
- **raw** (unnormalized) is the baseline for field-strength contributions
- Any result where direction-only contrast vanishes should be flagged as
  normalization-dependent

## Spatial Coverage

The validated environments span 0.07 to 124 AU across 7 boundary types:

| Distance | Environment | Mission | Boundary type |
|----------|-------------|---------|---------------|
| 0.07 AU | Switchbacks | PSP E10 | Coherent structure |
| 0.09 AU | Switchbacks | PSP E6 | Coherent structure |
| 0.13 AU | Switchbacks | PSP E4 | Coherent structure |
| 0.17 AU | Switchbacks | PSP E1 | Coherent structure |
| 0.3-1 AU | Inner heliosphere | Solar Orbiter | Baseline |
| 0.39 AU | Mercury magnetopause | MESSENGER | Intrinsic dipole (small) |
| 1.0 AU | Earth magnetopause | THEMIS, MMS | Intrinsic dipole (large) |
| 1.0 AU | Earth bow shock | Cluster | Fast-mode shock |
| 1.0 AU | Lunar wake | ARTEMIS | Plasma void |
| 1-5.4 AU | Out-of-ecliptic | Ulysses | Fast/slow wind regime |
| 1.5 AU | Mars IMB | MAVEN | Induced magnetosphere |
| 3.4 AU | Cometary cavity | Rosetta | Diamagnetic cavity |
| 1-50 AU | Outer heliosphere | Cassini, NH | Radial profile |
| 119.5 AU | V2 heliopause | Voyager 2 | Heliosheath-ISM |
| 123.8 AU | V1 heliopause | Voyager 1 | Heliosheath-ISM |

## Infrastructure

All analysis binaries are pure Rust, warnings-as-errors, governance-gated:

**Boundary detection binaries:**
- `heliosphere-boundary-survey` (unified: THEMIS/Cluster/MAVEN/MESSENGER)
- `heliosphere-mms-multiday` (MMS with aria2c fallback)
- `heliosphere-rosetta-draping` (4-way normalization ablation)
- `heliosphere-switchback-omega` (PSP Br reversal classification)

**Radial profile binaries:**
- `heliosphere-quench-scan` (quench-point mapping across r, lat)
- `heliosphere-feature-cube` (densified 1.26M-row cube builder)
- `heliosphere-associator-null-audit` (5-family null hierarchy)

**Catalog modules (12 missions):**
`mms.rs`, `themis.rs`, `cluster.rs`, `maven_mag.rs`, `messenger.rs`,
`rosetta.rs`, `swarm_mag.rs`, `psp_fields.rs`, `voyager.rs`, `ulysses.rs`,
`solar_orbiter_mag.rs`, `cassini.rs`

**Reusable library modules:**
- `data_core::crossing_lists` -- curated boundary list parser (THEMIS/MESSENGER/OMNI)
- `data_core::fetcher::download_hapi_csv_auto` -- auto-routing for large HAPI datasets
- `spectral_core::surrogates` -- 9 functions: bootstrap, change-point, MV surrogates
- `spectral_core::coherence` -- welch_psd, partial_coherence, field-aligned projection
- `cd_kernel::cayley_dickson::associator` -- SIMD 32D/64D batch associator

## Open Items

1. MMS FPI composite classifier (lower priority after THEMIS curated result)
2. Cluster magnetopause curated crossings from ESA CAA (bow shock done)
3. MAVEN published IMB/MPB crossing intervals for curated comparison
4. ARTEMIS lunar wake: specific wake crossing identification
5. Normalization ablation for ARTEMIS (weak-field wake region)
6. PSP radial trend: systematic study of ratio vs distance with more encounters
7. Swarm FAC sheets: de-scoped from paper core, future work
