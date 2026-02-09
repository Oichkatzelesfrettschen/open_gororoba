<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/external_sources.toml -->

# C-071: FRB Ultrametric Structure -- Data Provenance

## Claim

C-071: "FRB dispersion measures exhibit p-adic ultrametric structure."

## Outcome

**Refuted** (2026-02-06). No ultrametric signal detected in either CHIME
Catalog 1 (600 events) or Catalog 2 (5045 events) across three DM columns
and eight statistical tests per column, with Bonferroni correction.

## Data Sources

### CHIME/FRB Catalog 1

- **Citation**: Amiri, M. et al. (2021). "The First CHIME/FRB Fast Radio
  Burst Catalog." ApJS 257, 59. arXiv:2106.04352
- **Download URL**: https://storage.googleapis.com/chimefrb-dev.appspot.com/catalog1/chimefrbcat1.csv
- **SHA256**: `affd86573fee8fe94be9d0d85e73cc787fa260b2497ccec5d9d93e1d26610ed5`
- **Events**: 600 rows (536 unique FRBs including repeat bursts)
- **Date retrieved**: 2026-02-06
- **Local path**: `data/external/chime_frb_cat1.csv`

### CHIME/FRB Catalog 2

- **Citation**: CHIME/FRB Collaboration (2025). "The Second CHIME/FRB
  Fast Radio Burst Catalog." arXiv:2601.09399
- **Download URL**: https://storage.googleapis.com/chimefrb-dev.appspot.com/catalog2/chimefrbcat2.csv
- **SHA256**: `5108ada779d279a2547d9f9e73ae25bfdd40d8496d6ba7255ec29c6629057a48`
- **Events**: 5045 rows
- **Date retrieved**: 2026-02-06
- **Local path**: `data/external/chime_frb_cat2.csv`

## DM Columns Used

| Column | Description |
|--------|-------------|
| `bonsai_dm` | Real-time DM from CHIME/FRB's bonsai search pipeline |
| `dm_exc_ne2001` | Extragalactic DM excess (total DM minus NE2001 Milky Way contribution) |
| `dm_exc_ymw16` | Extragalactic DM excess (total DM minus YMW16 Milky Way contribution) |

NE2001 and YMW16 are two independent models of the Milky Way's electron
density distribution. Subtracting the Galactic contribution yields the
extragalactic DM, which is the physically relevant quantity for testing
cosmic ultrametric structure.

## Analysis Pipeline

- **Code**: `crates/stats_core/src/ultrametric.rs` (analysis engine)
- **CLI**: `crates/gororoba_cli/src/bin/frb_ultrametric.rs`
- **Results**: `data/csv/c071_frb_ultrametric.csv` (combined),
  `data/csv/c071_frb_ultrametric_cat1.csv`,
  `data/csv/c071_frb_ultrametric_cat2.csv`

## Test Battery (per DM column)

1. Euclidean ultrametric fraction (5% isosceles tolerance)
2. Graded ultrametric defect (permutation null)
3. P-adic clustering, p=2
4. P-adic clustering, p=3
5. P-adic clustering, p=5
6. P-adic clustering, p=7
7. P-adic clustering, p=11
8. P-adic clustering, p=13

Bonferroni threshold: 0.05 / 8 = 0.00625

## Cross-references

- C-080: Same hypothesis tested in Python with Cat 1 only (536 events).
  Also refuted (ultrametric fraction 19.8% vs null 20.2%). Python code
  since removed; Rust pipeline is the definitive replacement.
