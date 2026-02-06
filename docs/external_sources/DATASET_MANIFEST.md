# Dataset Manifest

All 18 datasets managed by `fetch-datasets` CLI. Fetch with:

```
cargo run --bin fetch-datasets -- --list
cargo run --bin fetch-datasets -- --all --skip-existing
```

## Astrophysical Catalogs

| Dataset | Provider | Source | Format | Size | License |
|---------|----------|--------|--------|------|---------|
| CHIME FRB Cat 2 | ChimeCat2Provider | CHIME/FRB Collab (2025), Zenodo | CSV | ~15 MB | CC-BY-4.0 |
| GWTC-3 Confident | Gwtc3Provider | GWTC-3 LIGO/Virgo/KAGRA | CSV | ~2 MB | CC-BY-4.0 |
| ATNF Pulsars | AtnfProvider | Manchester et al., ATNF | CSV | ~5 MB | Open |
| McGill Magnetars | McgillProvider | McGill Pulsar Group | HTML/CSV | ~50 KB | Open |
| Fermi GBM Bursts | FermiGbmProvider | HEASARC fermigbrst | Pipe->CSV | ~10 MB | Open |
| NANOGrav 15yr | NanoGrav15yrProvider | Agazie+ (2023), Zenodo | CSV | ~10 KB | CC-BY-4.0 |
| SDSS DR18 QSOs | SdssQsoProvider | SDSS SkyServer SQL | CSV | ~20 MB | Open |
| Gaia DR3 Nearby | GaiaDr3Provider | ESA Gaia Archive TAP | CSV | ~15 MB | Open |
| TSIS-1 TSI Daily | TsisTsiProvider | LASP LISIRD | CSV | ~500 KB | Open |

## Cosmological Datasets

| Dataset | Provider | Source | Format | Size | License |
|---------|----------|--------|--------|------|---------|
| Pantheon+ SH0ES | PantheonProvider | Scolnic+ (2022), GitHub | DAT | ~200 KB | CC-BY-4.0 |
| Planck 2018 Summary | PlanckSummaryProvider | Planck Legacy Archive | ZIP | ~1 MB | Open |
| WMAP 9yr Chains | Wmap9ChainsProvider | LAMBDA (NASA) | tar.gz | ~100 MB | Open |
| Planck 2018 Chains | PlanckChainsProvider | Planck Legacy Archive | tar.gz | ~9 GB | Open |
| DESI DR1 BAO | (hardcoded) | DESI Collab (2024) | In-code | 0 | CC-BY-4.0 |

## Geophysical Datasets

| Dataset | Provider | Source | Format | Size | License |
|---------|----------|--------|--------|------|---------|
| IGRF-13 | Igrf13Provider | NOAA/NCEI | TXT | ~30 KB | Open |
| WMM 2025 | Wmm2025Provider | NOAA/NCEI | ZIP | ~2 MB | Open |
| GRACE GGM05S | GraceGgm05sProvider | ICGEM GFZ | GFC | ~350 KB | Open |
| GRAIL GRGM1200B | GrailGrgm1200bProvider | ICGEM GFZ | GFC | ~84 MB | Open |
| JPL Horizons Planets | JplEphemerisProvider | JPL SSD API | CSV | ~200 KB | Open |

## Format Parsers

| Format | Module | Description |
|--------|--------|-------------|
| CosmoMC chains | formats::mcmc_chain | .paramnames + whitespace-delimited chain files |
| TAP/ADQL | formats::tap | IVOA Table Access Protocol sync queries |
| ICGEM .gfc | formats::gfc | Spherical harmonic gravity field coefficients |
| Pantheon .dat | formats::pantheon_dat | Whitespace-delimited supernova data |

## References

- Amiri+ (2021), ApJS 257, 59 (CHIME Cat 1)
- CHIME/FRB Collab (2025) (CHIME Cat 2)
- Abbott+ (2023), PRX 13, 041039 (GWTC-3)
- Manchester+ (2005), AJ 129, 1993 (ATNF)
- Olausen & Kaspi (2014), ApJS 212, 6 (McGill)
- Meegan+ (2009), ApJ 702, 791 (Fermi GBM)
- Agazie+ (2023), ApJL 951, L8 (NANOGrav 15yr)
- Almeida+ (2023), ApJS 267, 44 (SDSS DR18)
- Gaia Collab, Vallenari+ (2023), A&A 674, A1 (Gaia DR3)
- Kopp (2021), Sol Phys 296, 133 (TSIS-1)
- Scolnic+ (2022), ApJ 938, 113 (Pantheon+)
- Planck VI (2020), A&A 641, A6 (Planck 2018)
- Hinshaw+ (2013), ApJS 208, 19 (WMAP 9yr)
- DESI Collab (2024), arXiv:2404.03002
- Alken+ (2021), EPS 73, 49 (IGRF-13)
- Ries+ (2016), GFZ Data Services (GGM05S)
- Lemoine+ (2014), JGR 119, 1698 (GRAIL)
- Giorgini+ (1996), Bull. AAS 28, 1158 (JPL Horizons)
