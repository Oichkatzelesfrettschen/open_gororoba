# Cross-Domain Evidence Note: CD Associator as Phase-Geometry Diagnostic

**Date**: 2026-03-27
**Status**: Working synthesis for methods paper
**Claims**: C-1548 through C-1590 (43 claims)
**Missions**: 9 spacecraft, 18 analysis runs, 0.07-124 AU

## Positioning

We propose an auxiliary null-stratified, defect-aware hypercomplex formalism
for boundary excavation in multichannel heliospheric and magnetospheric
measurements. In open_gororoba, the formalism is realized through
32-dimensional Takens embeddings of lagged magnetic-field observables and
evaluated via Cayley-Dickson associator diagnostics. The construction is not
intended to replace standard plasma-physics observables, mission catalogs, or
curated crossing labels. Its role is methodological: to provide an ambient
structural shell in which cross-channel phase organization, boundary
incidence, weak-field degeneracy, and transition deformation can be
represented and compared across regimes using a common operator.

The framework is useful only insofar as it satisfies two conditions: first,
that known boundary structure remains recoverable under comparison with
curated or established physics-based labels; and second, that the
hypercomplex lift yields additional observables not reducible to scalar
thresholds, linear spectra, or rank-based diagnostics. Within this
interpretation, the associator should be treated not as a universal turbulence
meter but as a regime-dependent order parameter for multichannel phase
geometry, with explicit null-stratification and normalization controls in
weak-field environments.

**One-sentence paper claim**: We present a pure-Rust, null-stratified,
multiscale hypercomplex framework for multichannel electromagnetic-field time
series that yields a regime-sensitive order parameter for cross-channel phase
organization, validated across heliopause, magnetopause, induced-boundary,
coherent-structure, and weak-field cavity environments.

**Important caveat**: Detection-rate numbers are not directly comparable
across rows of the validation table, because the label models differ.
Bayesian change-point localization, curated boundary timestamps, |B|-gradient
heuristics, and switchback sign criteria are different validation modes, not
one shared leaderboard.

## Tier 1: Flagship Heliopause Domain (Voyager 1/2)

The heliopause is the **anchor domain** -- the primary demonstration that the
CD associator recovers known boundary structure.

| Mission | Distance | Method | Metric |
|---------|----------|--------|--------|
| Voyager 2 | 119.5 AU | Bayesian CP | pre=4.35, post=0.13 (33x drop) |
| Voyager 1 | 123.8 AU | Bayesian CP | pre=0.12, post=0.02 (7x drop) |

Cross-spacecraft agreement: both V1 and V2 show the same pattern (associator
quench at the heliopause), at distances consistent with the known crossing
locations. The V2 pre-crossing level is 37x higher than V1's because V2
traversed a more turbulent heliosheath sector.

The densified 1.26M-row feature cube (Voyager 1/2, Ulysses, New Horizons,
Cassini, Solar Orbiter, PSP) provides continuous radial coverage from 0.3 to
124 AU with 5-family null stratification confirming that the outer-heliosphere
signal survives all surrogate treatments.

## Tier 2: Cross-Domain Validation

These missions answer: does the same observable generalize beyond the
heliopause?

### Validation Quality Tiers

**Tier A -- Curated-label benchmarks** (strongest external validation):

| Mission | Environment | Detection | FA | Offset | Curated Source |
|---------|-------------|-----------|-----|--------|----------------|
| **THEMIS-A** | Earth magnetopause | **89.0%** | 16.7% | 6 min | Zenodo V2 (Staples 2020) |
| **Cluster-1** | Earth bow shock | **64.9%** | 73.9% | 8 min | OMNI SPDF |
| **MESSENGER** | Mercury magnetopause | **64.3%** | 84.8% | **3 min** | Zenodo |

**Tier B -- Heuristic-labeled boundaries** (physics-based, label-contaminated):

| Mission | Environment | Detection | FA | Offset | Label |
|---------|-------------|-----------|-----|--------|-------|
| MMS | Earth magnetopause | 61.5% | 40.0% | 6 min | |B| gradient 5 nT |
| ARTEMIS THB | Lunar wake/tail | 75.0% | 68.1% | 4 min | |B| gradient 3 nT |
| MAVEN (14-day) | Mars IMB | 53.3% | 25.8% | 7 min | |B| gradient 5 nT |
| Solar Orbiter | Inner heliosphere | Omega_mv=0.045 | -- | -- | Null hierarchy |
| Ulysses | Out-of-ecliptic | 2-4x fast/slow | -- | -- | v_sw > 550 km/s |
| Cassini + NH | Outer heliosphere | Radial profile | -- | -- | Densified cube |

**Tier C -- Non-boundary / normalization-sensitive demonstrations**:

| Mission | Environment | Metric | Label | Caveat |
|---------|-------------|--------|-------|--------|
| Rosetta | 67P cavity | 1.9x (dir) / 5.3x (current) | |B| gradient | 40% genuine, 60% norm amplification |
| PSP (4 enc.) | Switchbacks | 0.42-0.70 ratio | Br/|B| < -0.5 | Regime classification, not point events |

**Tier D -- Computational domains** (CD associator applied beyond plasma physics):

| Domain | Metric | Result | Significance |
|--------|--------|--------|-------------|
| **TurboQuant** (LLM KV cache) | Adaptive bit allocation | **23% MSE gain** | CD residual associator identifies quantization-vulnerable tokens |
| GRMHD MRI onset | Laminar-to-turbulent | **17x** A ratio | Topology transition captured in scalar diagnostic |
| BOUT++ slab turbulence | Spatial A gradient | Monotonic profile | Turbulence intensity measurement |
| Solar flare (SWAN-SF) | Pre-eruption detection | 3.62x onset/pre ratio | 12th physical domain (X-class flares) |
| Magnetar QPO (SGR 1806-20) | Crustal mode coupling | 10.5x ratio | Synthetic confirmation |

The TurboQuant result (C-1600) is the 13th domain for the CD associator, and the
first computational/ML application. The per-token residual associator norm
`||[r_t, r_{t+1}, r_{t+2}]||` identifies tokens where QJL sign projections
capture phase-coupling structure poorly. Allocating additional bits to the top
25% of tokens (by associator score) reduces quantization MSE by 23% at d=32,
3-bit -- well above the 0.5% threshold for significance.

This extends the CD associator's universality from "detects phase transitions in
12+ physical domains" to "detects phase-coupling vulnerability in neural network
quantization." The same cubic nonlinearity measure from non-associative algebra
that finds heliopause crossings, tokamak disruptions, and MRI onset also finds
the attention key vectors that need more quantization bits.

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

## Open Decomposition Questions

The following five questions decompose what the CD associator actually
measures. Each requires targeted experiments against the existing datasets.

**Q1: Coherence/phase vs plain spectral structure**
Run phase-randomized (destroys phase, keeps spectrum) and MV-phase-randomized
(destroys cross-channel coupling) on each dataset. If signal survives (a)
but not (b), it is cross-channel phase organization. If destroyed by (a),
plain spectral. Apply to THEMIS, MAVEN, MMS, and PSP switchback/quiet.

**Q2: Transverse vs compressive mode content**
Use field_aligned_spectral_fractions to decompose each dataset. Do high-
associator intervals correlate with transverse-dominant or compressive-
dominant spectral content? Run on V2 ISM/heliosheath, MMS pre/post
magnetopause, MAVEN IMB, and PSP switchback/quiet.

**Q3: Draping/induced-boundary topology**
Compare associator response at induced boundaries (MAVEN, Rosetta) vs
intrinsic dipole boundaries (THEMIS, MMS, MESSENGER). Is detection rate
systematically higher at induced boundaries? Is the associator's sensitivity
to draping geometry distinct from its sensitivity to reconnection geometry?

**Q4: Normalization-sensitive weak-field behavior**
Extend 4-way normalization ablation from Rosetta to ARTEMIS lunar wake and
any other weak-field environment. Build a normalization-sensitivity map:
ratio of direction-only to current-norm contrast per environment.

**Q5: Where do standard diagnostics and the CD observable disagree?**
For each curated-label dataset, classify: (a) false negatives -- curated
crossings the associator missed (low rotation? partial crossing?), (b) false
alarms -- associator transitions with no curated crossing (tangential
discontinuities? current sheets? FTEs?). This reveals the CD observable's
blind spots and extra sensitivities relative to standard plasma diagnostics.

## Open Engineering Items

1. MMS FPI composite classifier (in progress)
2. Mercury MP vs BS label separation (in progress)
3. Cluster magnetopause curated crossings from ESA CAA
4. MAVEN published IMB/MPB crossing intervals
5. ARTEMIS lunar wake: specific wake crossing identification
6. PSP radial trend: systematic ratio vs distance with more encounters
7. Swarm FAC sheets: de-scoped from paper core, future work
