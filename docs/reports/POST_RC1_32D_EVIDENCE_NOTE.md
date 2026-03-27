# Post-RC1 32D Evidence Note

## RC1 Baseline (frozen reference, commit a51405fd)

The RC1 tightening established a 16D sedenion Takens embedding over a
100,943-row feature cube spanning 13 missions. Key RC1 results:
- Quench transition at 132.5-137.5 AU (sparse, mission_diversity=1)
- OMNI at 1 AU below both null families (suppression ratio 0.58-0.65)
- CPU/Vulkan backend parity to 1 ULP
- Claims C-1548 through C-1551

## 32D Infrastructure (commit 5977f5f9)

- Dimension-generic embedding with SIMD fast paths for 16D (sedenion)
  and 32D (pathion, new AVX2+FMA CD-doubling implementation)
- Fixed pre-existing sedenion SIMD bug (cr_al[7] vs ar_conj_cl[7] typo,
  corrupting element 11 of 16D products, 8 of 9 test failures resolved)
- Unified metadata indexing (off-by-2 spatial tagging correction)
- Extended null-audit schema (null_std, null_median, null_p05, null_p95)

## Densified Cube (commit f694446e)

1,153,207 rows (11.4x over RC1):
- Voyager 1: 423,962 rows (1977-2024, 1-166 AU)
- Voyager 2: 389,044 rows (1977-2020, 1-126 AU)
- Ulysses: 175,320 rows (1990-2009, all 3 polar orbits, 1-5.4 AU)
- Provenance QA: 7 duplicate keys / 1,153,151 unique (0.0006%, all Solar Orbiter)

## 32D Quench Map (87 bins, 2.5-167.5 AU)

The densified cube resolves the quench transition in continuous 5 AU steps:
- Inner plateau: mean ~10 (r < 107 AU), no strong radial trend
- Steep descent: 107 -> 122 AU (factor 45x drop over 15 AU)
- ISM floor: mean ~0.02 (r > 127 AU)

The transition aligns with the heliopause (~120 AU), not the termination
shock (~84-94 AU).

## Where 32D Exceeds Harsh Nulls Only

At inner heliosphere distances (1 AU, all near-Earth missions):

| Null Family | Type | Base/Null Ratio | Interpretation |
|-------------|------|-----------------|----------------|
| temporal-shuffle | Harsh | 0.52 | Strong separation |
| channel-permutation | Harsh | 0.84 | Moderate separation |
| block-shuffle (K=10) | Moderate | 0.93 | Marginal |
| phase-randomized | Spectral | 0.99 | **Indistinguishable** |

The inner-heliosphere 32D signal is primarily spectral autocorrelation.
The algebraic structure detected at 1 AU is almost entirely explained
by the B-field power spectrum shape, not by nonlinear phase coupling.

Classification per mission (phase_ratio / block_ratio):
- SPECTRAL_ONLY: Helios 1/2, PSP, STEREO-A, Solar Orbiter
- BEYOND_BLOCK_AUTOCORRELATION: ACE, Cassini, OMNI, Ulysses, Voyager 1, WIND
- BEYOND_SPECTRUM: IMP 8
- GENUINE_NONLINEAR: **Voyager 2 only** (phase=0.929, block=0.809)

## Where 32D Reveals Excess Algebraic Order

**Voyager 2 heliosheath (100-130 AU)** exhibits a specific cross-channel
B-field phase geometry that produces LESS non-associative structure than
independently phase-randomized channels would. Under the 5-family null
suite (including multivariate phase-randomized with independent per-channel
phases), the base signal sits at 0.574-0.581x the multivariate null -- well
below, and stable across all block sizes K=5 through K=100.

This is NOT best described as "excess nonlinearity." The heliosheath does
not have MORE non-associative structure than spectral nulls; it has a
SPECIFIC cross-channel phase coupling that is more algebraically ordered
than random phase relationships would produce. The quench transition at
the heliopause (117-122 AU) marks the boundary where this ordered phase
geometry breaks down and the field becomes algebraically inert.

Within-mission split validation (inner 1-60 AU vs outer 60-126 AU) confirms
the quench transition is stable across disjoint observation epochs.

The quench front aligns with the heliopause (~120 AU), not the termination
shock (~84-94 AU). The ISM floor at >127 AU is real quenching.

## Regime-Conditioned Results

### Fast vs Slow Wind

| r_au | Fast Wind | Slow Wind | Ratio | Interpretation |
|------|-----------|-----------|-------|----------------|
| 2.5 | 11.4 | 11.2 | 1.01 | No discrimination |
| 27.5 | 12.5 | 6.0 | 2.09 | Fast wind amplified |
| 42.5 | 22.0 | 10.0 | 2.19 | Strong |
| 77.5 | 35.8 | 12.5 | 2.87 | Very strong |
| 82.5 | **68.2** | 15.8 | **4.33** | Strongest signal in fleet |

The 32D embedding discriminates fast Alfvenic polar wind from slow
equatorial wind at mid-heliosphere distances (27-82 AU). This is
physically interpretable: fast polar wind has stronger Alfvenic
fluctuations carrying nonlinear phase coupling (McComas et al., 2000).

### Br Polarity

- Inner heliosphere: Br+/Br- = 1.49 at 7.5 AU (Parker spiral asymmetry)
- Outer heliosphere: insufficient polarity data for conditioning

## Invariance Tests

### Leave-Ulysses-Out

- Inner bins shift -11% to -34% (Ulysses provides high-lat structure)
- Outer bins (>97 AU): 0.0% change (Ulysses never reaches there)
- Quench transition is entirely Voyager-driven and Ulysses-independent

### Leave-Voyager-2-Out

- Transition zone (107-127 AU) shifts magnitudes up to +/-40%
- Quench shape SURVIVES: still drops from 9.8 to 0.32 to 0.024
- ISM floor (>132 AU) is invariant (Voyager 1 only)

Verdict: quench transition is structurally robust under leave-one-out.

## Remaining Caveats

1. Inner-heliosphere 32D signal is primarily spectral autocorrelation,
   not genuine nonlinear geometry. Use harsh-null separation language
   carefully; it overstates the physical content.

2. Voyager 2's GENUINE_NONLINEAR classification is mission_diversity=1.
   No independent confirmation exists at the same heliocentric distances.

3. Fast-wind amplification at 27-82 AU is dominated by Ulysses data.
   Leave-Ulysses-out on fast wind leaves only 14K rows with thin coverage.

4. Phase-randomized null preserves linear cross-channel correlations.
   A stronger null (e.g., multivariate phase-randomized with independent
   channel phases) would test whether cross-channel nonlinear coupling
   contributes beyond within-channel spectral shape.

5. Block-shuffle with K=10 is one choice of block size. Sensitivity to
   K (5, 20, 50) has not been tested.

6. 32D remains the preferred working embedding, not the canonical
   physical claim surface. Promotion requires the mixed-embedding test
   (Task G) and further invariance validation.

## Artifact Index

| File | Description |
|------|-------------|
| `quench_scan_densified_32d.csv` | 87-bin quench map on 1.15M-row cube |
| `densified_null_audit_32d_4family.json` | 4-family null audit (10 iterations) |
| `spectral_excess_by_mission_32d.csv` | Per-mission spectral classification |
| `diagnostic_stack_densified_32d.csv` | 87-bin diagnostic stack |
| `quench_scan_densified_32d_fast_wind.csv` | Fast wind (>550 km/s) regime map |
| `quench_scan_densified_32d_slow_wind.csv` | Slow wind regime map |
| `quench_scan_densified_32d_br_pos.csv` | Br-positive polarity map |
| `quench_scan_densified_32d_br_neg.csv` | Br-negative polarity map |
| `invariance_no_voyager2_32d.csv` | Leave-V2-out quench map |
| `invariance_no_ulysses_32d.csv` | Leave-Ulysses-out quench map |
| `invariance_fast_wind_no_ulysses_32d.csv` | Fast wind without Ulysses |
| `densified_provenance_qa.json` | Provenance QA report |
