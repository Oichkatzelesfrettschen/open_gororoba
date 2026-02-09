<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Dataset Coverage Report

Maps the 30 dataset providers to the 7 scientific pillars and their
backing claims in `CLAIMS_EVIDENCE_MATRIX.md`.

Generated: 2026-02-07

## Coverage Summary

| Pillar | Datasets | Claims backed | Status |
|--------|----------|---------------|--------|
| candle | 3 | 4 | Active |
| gravitational | 2 | 10 | Active |
| electromagnetic | 2 | 1 | Partial |
| survey | 7 | 12 | Active |
| cmb | 3 | 2 | Active |
| solar | 2 | 0 | Infrastructure |
| geophysical | 11 | 0 | Infrastructure |

**12 of 30 datasets** currently back at least one claim.
**18 datasets** are infrastructure (provisioned for future analysis).

## Pillar: candle (Standard Candles / Rulers)

| Dataset | Provider | Claims |
|---------|----------|--------|
| Pantheon+ SH0ES | PantheonProvider | C-038, C-437, C-441 |
| Union3 Legacy SN Ia | Union3Provider | (none yet) |
| DESI DR1 BAO | (hardcoded) | C-057, C-441 |

## Pillar: gravitational (GW Events + PTA)

| Dataset | Provider | Claims |
|---------|----------|--------|
| GWTC-3 Combined | Gwtc3Provider | C-006, C-007, C-025, C-060, C-061, C-070, C-437, C-439, C-440, C-441 |
| NANOGrav 15yr | NanoGrav15yrProvider | C-059, C-070 |

## Pillar: electromagnetic (EM Transients + Imaging)

| Dataset | Provider | Claims |
|---------|----------|--------|
| Fermi GBM Bursts | FermiGbmProvider | C-064, C-437 |
| EHT M87 2018 Bundle | EhtM87Provider | (none yet) |
| EHT SgrA 2022 Bundle | EhtSgrAProvider | (none yet) |

## Pillar: survey (Multi-Object Surveys)

| Dataset | Provider | Claims |
|---------|----------|--------|
| CHIME/FRB Catalog 2 | ChimeCat2Provider | C-043, C-062, C-071, C-080, C-436, C-437, C-438, C-440 |
| ATNF Pulsar Catalogue | AtnfProvider | C-043, C-063, C-437 |
| McGill Magnetar Catalog | McgillProvider | C-043, C-063, C-437 |
| SDSS DR18 QSOs | SdssQsoProvider | C-437 |
| Gaia DR3 Nearby | GaiaDr3Provider | C-437 |
| Hipparcos Legacy Catalog | HipparcosProvider | C-437 |
| TSIS-1 TSI Daily | -- | (see solar pillar) |

## Pillar: cmb (CMB / WMAP)

| Dataset | Provider | Claims |
|---------|----------|--------|
| Planck 2018 Summary | PlanckSummaryProvider | C-040, C-058 |
| WMAP 9yr Chains | Wmap9ChainsProvider | (none yet) |
| Planck 2018 Chains | PlanckChainsProvider | (none yet) |

## Pillar: solar (Solar Irradiance)

| Dataset | Provider | Claims |
|---------|----------|--------|
| TSIS-1 TSI Daily | TsisTsiProvider | (none yet) |
| SORCE TSI Daily | SorceTsiProvider | (none yet) |

## Pillar: geophysical (Gravity + Magnetic Field Models)

| Dataset | Provider | Claims |
|---------|----------|--------|
| IGRF-13 Coefficients | Igrf13Provider | (none yet) |
| WMM 2025 | Wmm2025Provider | (none yet) |
| GRACE GGM05S | GraceGgm05sProvider | (none yet) |
| GRACE-FO Gravity Field | GraceFoProvider | (none yet) |
| GRAIL GRGM1200B | GrailGrgm1200bProvider | (none yet) |
| EGM2008 Static Geoid | Egm2008Provider | (none yet) |
| Swarm L1B Magnetic Sample | SwarmMagAProvider | (none yet) |
| Landsat C2 L2 STAC Metadata | LandsatStacProvider | (none yet) |
| JPL DE440 Ephemeris Kernel | De440Provider | (none yet) |
| JPL DE441 Ephemeris Kernel | De441Provider | (none yet) |
| JPL Horizons Planets | JplEphemerisProvider | (none yet) |

## Key Claim Clusters

**Ultrametric analysis (C-437)**: The largest consumer, using 9 catalogs
(CHIME, ATNF, McGill, GWOSC, Pantheon+, Gaia, SDSS, Fermi, Hipparcos)
across 472 attribute-subset tests. GPU exploration (I-011) found 82/472
significant at BH-FDR<0.05.

**Cosmological fitting (C-441)**: Uses Pantheon+ (1578 SNe) + DESI DR1 BAO
(7 bins, 12 data points) for Lambda-CDM vs bounce model comparison.

**FRB ultrametricity (C-071, C-436, C-438, C-440)**: Uses CHIME Cat 2
(5045 events) for DM-based and multi-attribute ultrametric tests.

**GW mass distribution (C-007)**: Uses GWTC-3 combined catalog (219 events)
for Bayesian mixture modeling of black hole mass spectrum.

## Infrastructure Datasets (No Claims Yet)

These 18 datasets are provisioned for future analysis pipelines:

- **Union3**: Legacy SN Ia likelihood chains (DESI Y3 vintage)
- **EHT M87/SgrA**: Event Horizon Telescope release bundles
- **WMAP/Planck chains**: Full MCMC posterior chains
- **TSIS/SORCE**: Solar irradiance time series
- **All geophysical**: Gravity (GRACE, EGM2008, GRAIL), magnetic (Swarm, IGRF, WMM),
  ephemeris (DE440/441), and Earth observation (Landsat STAC)

Future directions: connect geophysical datasets to optics_core ray-tracing
(atmospheric refraction), materials_core effective-medium models, and
cosmology_core distance-redshift calibration cross-checks.
