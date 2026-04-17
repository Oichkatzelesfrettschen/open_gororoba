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

## Commit Hash

**Pre-registration commit hash:** 455d47455c46b623413c601fa2f8ad4c97d8c45f

**Git tag:** `ablation-preregistered-v1`

This tag is referenced in the paper Section 7: "Results in Section 7 were
pre-registered at commit [hash] (tag: ablation-preregistered-v1) prior to
execution of any ablation binary."
