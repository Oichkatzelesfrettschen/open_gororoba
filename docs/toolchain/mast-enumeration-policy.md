# MAST (STScI) Catalog Enumeration Policy (plan P6A.S5.T2)

## Context

`registry/data_servers.toml#mast_api` (HST/JWST REST) and
`#mast_s3_zarr` (FAIR-MAST Zarr mirror) both carry
`catalog_enumeration = "partial"`. We currently cherry-pick HST and
JWST cone-search queries plus FAIR-MAST shot-level Zarr chunks;
there is no full enumeration of missions x instruments x filters
across the MAST catalog.

A complete MAST inventory is more expensive than HEASARC because:

- The archive spans 19+ distinct missions (HST, JWST, TESS, Kepler,
  GALEX, Hubble Legacy Archive, PanSTARRS, etc.).
- Each mission has its own instrument set with distinct query
  parameters and often distinct result schemas.
- The public API is REST JSON (not TAP VOTable), so a universal
  parser must normalize across response formats.

## Decision: scoped discovery, not exhaustive enumeration

MAST enumeration is scoped to the active-experiment surface plus a
curated catalog of observatory-level metadata:

1. **Per-mission metadata snapshot**: one row per mission listing
   (name, full_title, operational_years, instruments, query_base_url,
   last_verified). ~19 rows total. Hand-curated with automated
   drift-check (weekly CI).
2. **Per-instrument inventory**: only for missions with an active
   experiment binding in `registry/datasets.toml`. Initially HST
   (via `hst_fetch.rs`) and JWST (via `jwst_fetch.rs`).
3. **Per-observation query**: on demand only; not pre-walked.

This defers the "complete MAST catalog" goal while ensuring our
registered fetchers stay accurate as MAST evolves.

## Endpoints

| Purpose                      | URL                                                          |
|------------------------------|--------------------------------------------------------------|
| REST base                    | https://mast.stsci.edu/api/v0                                 |
| Missions list                | https://mast.stsci.edu/api/v0.1/Download/file_ids?mission=    |
| FAIR-MAST Zarr (S3)          | https://s3.echo.stfc.ac.uk/mast                               |
| TAP (if needed for enumeration) | https://vao.stsci.edu/CAOMTAP/TapService.aspx             |

Note: STScI also operates a VAO TAP service at the CAOMTAP URL for
richer queries. Currently unused; consider for future HEASARC-style
enumeration if richer introspection is needed.

## Target schema (`registry/mast_catalogs.toml`)

```toml
[mast_catalogs]
updated = "YYYY-MM-DD"
source_urls = [
  "https://mast.stsci.edu/api/v0",
  "https://vao.stsci.edu/CAOMTAP/TapService.aspx",
]

[[mission]]
name = "HST"
title = "Hubble Space Telescope"
operational_years = "1990-present"
instruments = ["WFC3", "ACS", "COS", "STIS", "FGS", "NICMOS"]
query_base_url = "https://mast.stsci.edu/api/v0/invoke?service=Mast.Caom.Cone"
owner_fetcher = "crates/data_core/src/catalogs/hst_fetch.rs"
last_verified = "YYYY-MM-DD"
```

## Refresh procedure

- Weekly: drift-check against MAST announcements (new missions,
  instrument retirements) via the official STScI NEWSLETTER RSS
  feed (alternative: scrape the missions page).
- Ad hoc when a new active dataset row lands with
  `server_ref = "mast_api"`: walker verifies the referenced mission
  + instrument is present in the current snapshot.

## Known failure modes

- **API version drift**: MAST has rolled v0 -> v0.1; endpoint URLs
  change without deprecation warnings. Mitigation: pin exact
  subpath in `query_base_url`; drift-check fails if HTTP 404.
- **Rate-limit surprises**: bulk catalog queries trigger rate
  limits that cone-search queries do not. Keep 250 ms inter-request
  delay and per-mission circuit breakers.
- **S3 egress costs**: FAIR-MAST Zarr is on STFC S3; daily budget
  already capped at 30 GB per `data_servers.toml#mast_s3_zarr`.
  Do NOT enumerate S3 object listing in bulk.

## RCA seed

2026-04-17: Initial ADR; no prior incidents recorded.
