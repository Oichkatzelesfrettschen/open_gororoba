# Ablation Pre-Registration: JGR CD Magnetopause Paper

**Date:** 2026-04-17
**Plan reference:** Plan P6A.S2 (ablation campaign, tasks 2.0-2.14)
**Status:** PRE-REGISTERED (commit hash and git tag to be recorded below)

This document is committed BEFORE any ablation binary executes.
Results in Section 7 of the paper reference this pre-registration.

---

## Exact Hyperparameters

All ablation binaries share these defaults unless stated otherwise.
Changes from these defaults must be documented as sensitivity analyses.

### Embedding (Phase 0.5 CdEmbeddingParams defaults)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `window_secs` | 480.0 (8 min) | 5-20 min physical crossing timescale; 8 min is center |
| `n_lags` | 8 | `embedding_dim / n_channels = 32 / 4` |
| `lag_secs` | 60.0 (1 min) | THEMIS FGM cadence; tau=1 min baseline from paper |
| `eps_mode` | `Absolute(0.5)` | THEMIS FGS noise floor ~0.5 nT; matches paper baseline |
| `norm_mode` | `None` | Matches paper baseline; amplitude-preserving |
| `coord_mode` | `Raw` | (Bx, By, Bz, |B|-mean)/denom; matches paper baseline |

### Detection

| Parameter | Value |
|-----------|-------|
| `MAD_SCALE_FACTOR` | 1.5 (named constant, fixed prior to evaluation) |
| `crossing_window_minutes` | 10 |
| `bmag_noise_floor` | 0.5 nT |
| `max_bmag` | 100.0 nT |

### Evaluation

| Parameter | Value |
|-----------|-------|
| Dataset | THEMIS-A Staples+2020, 2016-08-29 to 2016-09-04 (E-237) |
| Matching window | +/-10 min (600 s) around catalog event midpoint |
| Bootstrap | 1800 s blocks, N=10000, seed=42 |
| Permutation | N=10000, seed=42 |

---

## Ablation Axes

### Axis A: Fixed window (8 min), vary CD algebra dimension

| Variant | Channels | Lags | Total dim | Notes |
|---------|----------|------|-----------|-------|
| R16 | 2 (Bx, By) | 8 | 16 | Drops Bz and |B| |
| R32 | 4 (Bx, By, Bz, |B|-mean) | 8 | 32 | **Paper baseline** |
| R64 | 8 (4 channels + half-lag offsets) | 8 | 64 | Doubled temporal resolution |

### Axis B: Fixed R32 algebra, vary lag depth

| Variant | Channels | Lags | Window (min) | Notes |
|---------|----------|------|--------------|-------|
| d=4 | 4 | 4 | 4 | Shortest window |
| d=6 | 4 | 6 | 6 | |
| d=8 | 4 | 8 | 8 | **Paper baseline** |
| d=12 | 4 | 12 | 12 | Within physical crossing regime |

### Baselines

| Baseline | Method |
|----------|--------|
| L2-delay | ||v_t - v_{t-lag}||_2 sliding change-point |
| Commutator | ||v_t * v_{t-lag} - v_{t-lag} * v_t||_2 |
| PCA variance | Fraction of variance in first principal component |
| Dense-random trilinear | T_ijk from N(0, 1/32), 100 draws |
| Sparsity-matched random | Same zero-pattern as CD, randomize non-zero from N(0, sigma_cd), 100 draws |

---

## Pre-registered Predictions

These predictions were made before any ablation binary was run.
Deviation from predictions must be reported honestly.

**P1:** CD R32 F1 will EXCEED L2-delay baseline F1 by >= 0.05.
*Rationale:* The CD associator captures nonlinear cross-lag interactions
that the L2-delay norm misses.

**P2:** CD R32 mean F1 will EXCEED the mean of 100 dense-random-trilinear draws
by >= 2 sigma of the dense-random distribution.
*Rationale:* If true, the specific CD coefficient structure contributes
beyond embedding dimension alone.

**P3:** CD R32 F1 will be MONOTONICALLY DECREASING from d=8 to d=4 on Axis B
(shorter windows degrade performance).
*Rationale:* The 8-min window captures the median crossing timescale.
Shorter windows miss multi-minute transitions.

**P4:** CD R16 (2 channels) F1 will be LOWER than CD R32 (4 channels).
*Rationale:* Bz and |B| carry independent boundary information not
captured by Bx and By alone.

**P5:** CD R64 F1 will be WITHIN 0.05 of CD R32 F1 (not significantly better).
*Rationale:* At 1-minute cadence, doubling temporal resolution via half-lag
offsets adds noise without proportional signal gain.

---

## Actual Results (recorded post-execution, 2026-04-17)

All binaries ran against THEMIS-A Staples+2020, 2016-08-29 to 2016-09-04 (7 days).
F1 CI from block-bootstrap (1800 s blocks, N=10000, seed=42).

### Prediction verdicts

| Prediction | Verdict | Details |
|-----------|---------|---------|
| P1: CD R32 > L2-delay by >=0.05 | **FALSIFIED** | L2-delay F1=0.615 > CD R32 F1=0.601 (delta=-0.014) |
| P2: CD R32 > dense-random mean by >=2 sigma | **FALSIFIED** | CD R32 at 1.30 sigma above dense-random mean (need 2.0) |
| P3: Monotonically decreasing F1 from d=8 to d=4 | **FALSIFIED** | d=4 F1=0.625 > d=8 F1=0.601 (shorter window is BETTER) |
| P4: CD R16 F1 < CD R32 F1 | **CONFIRMED** | R16 F1=0.207 << R32 F1=0.601 (Bz and |B| are critical) |
| P5: CD R64 within 0.05 of CD R32 | **FALSIFIED** | R64 F1=0.419, delta=-0.182 from R32 (much worse, not within 0.05) |

Only P4 confirmed. P1, P2, P3, P5 falsified.

### Full results table

| Method | F1 | Notes |
|--------|-----|-------|
| L2-delay baseline | 0.615 | Simple Euclidean distance; exceeds CD R32 |
| d=4 (Axis B, 4-min window) | 0.625 | Shorter window better on this dataset |
| CD R32 (paper baseline) | 0.601 | CI=[0.456,0.667] |
| Commutator ||xy-yx|| | 0.590 | CI=[0.452,0.675]; close to CD |
| Dense-random mean | 0.528 | sigma=0.056; CD R32 at 1.30 sigma above mean |
| Sparsity-matched mean | 0.535 | sigma=0.042; CD R32 at 1.57 sigma above mean; 15960/1M nonzero entries |
| Window Hamming | 0.536 | Boxcar is optimal |
| Window Hann | 0.510 | Boxcar is optimal |
| d=8 (Axis B, paper) | 0.601 | = CD R32 baseline |
| d=16 (Axis B, 16-min) | 0.419 | Longer window degrades performance |
| CD R64 (Axis A) | 0.419 | 8ch x 8 lags; much worse than R32 |
| CD R16 (Axis A) | 0.207 | 2ch (no Bz/|B|); poor |
| PCA variance ratio | 0.127 | CI=[0.076,0.226]; nonlinear structure essential |

### Scientific interpretation

CD R32 falls within the distribution of both random baselines. The specific CD coefficient
structure is not demonstrably superior to random trilinear coefficients with the same
zero-pattern. Nor is the CD zero-pattern demonstrably superior to fully-random dense trilinear.

The positive findings are:
1. PCA (linear) F1=0.127 confirms nonlinear structure is essential.
2. R16 (2ch) F1=0.207 confirms cross-axis field components (Bz, |B|) are critical.
3. Boxcar window is optimal (Hamming/Hann degrade by 0.065-0.091).
4. Commutator F1=0.590 is close to CD R32, suggesting the non-associative 3-body
   structure adds only marginal value over the 2-body anti-commutative structure.

The paper must be reframed from "CD algebra uniquely detects boundaries" to:
"Nonlinear trilinear forms over multi-channel delay embeddings detect boundaries;
the CD structure is one canonical member of this family but is not uniquely optimal."

---

## Commit Hash

**Pre-registration commit hash:** 455d47455c46b623413c601fa2f8ad4c97d8c45f

**Git tag:** `ablation-preregistered-v1`

This tag is referenced in the paper Section 7: "Results in Section 7 were
pre-registered at commit [hash] (tag: ablation-preregistered-v1) prior to
execution of any ablation binary."
