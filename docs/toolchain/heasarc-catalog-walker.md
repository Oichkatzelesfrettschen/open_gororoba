# HEASARC Catalog-of-Catalogs Walker (plan P6A.S5.T1)

## Context

`registry/data_servers.toml` entry `heasarc_w3browse` carried
`catalog_enumeration = "none"` before this ADR. The survey found that
we cherry-pick catalog tables (fermi_gbm, atnf pulsars) without any
mechanism to discover what HEASARC actually exposes. This creates two
hazards:

1. **Dataset blind spots** -- new HEASARC catalog tables (e.g. XMM
   PPS, NICER L3) land without our registry noticing.
2. **Schema drift** -- table columns can change between HEASARC
   revisions; our downstream parsers assume stable layouts.

A proper catalog walker enumerates HEASARC once per run and pins an
auditable snapshot in `registry/heasarc_catalogs.toml`. The governance
gate can then assert that every HEASARC-sourced experiment cites a
table that still exists in the current snapshot.

## Endpoint

HEASARC exposes a TAP 1.0 service at

    https://heasarc.gsfc.nasa.gov/xamin/vo/tap

The `/tables` sub-endpoint returns VOTable XML listing every
catalog table (hundreds of rows; ~1-3 MB response).

Full enumeration URL:

    https://heasarc.gsfc.nasa.gov/xamin/vo/tap/tables

Authoritative reference: HEASARC "Xamin API Users Guide"
(https://heasarc.gsfc.nasa.gov/docs/xamin-api.html).

## Schema (target: registry/heasarc_catalogs.toml)

```toml
[heasarc_catalogs]
updated = "YYYY-MM-DD"
snapshot_source_url = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/tables"
snapshot_sha256 = "..."
entry_count = N

[[catalog]]
name = "fermigbrst"
title = "Fermi GBM Burst Catalog"
description = "..."
row_count = 3500
columns = ["name", "ra", "dec", "trigger_time", ...]
last_verified = "YYYY-MM-DD"
```

## Walker binary (P6A.S5.T1 follow-up)

Located at `crates/data_core/src/catalogs/heasarc_enumerate.rs`.

Responsibilities:

1. HTTP GET `/tables` with the unified `download_stack.rs` backend.
2. Parse VOTable via `crates/data_core/src/formats/votable.rs`
   (feature `fits`). VOTable is IVOA-standard and already supported.
3. Extract rows (table_name, description, column_list).
4. Compute snapshot SHA256 (deterministic serialization).
5. Rewrite `registry/heasarc_catalogs.toml`; diff against prior.

## Refresh cadence

- Monthly at a minimum.
- Manually triggered:  `make heasarc-catalog-refresh`
- CI-gated drift check:  `make heasarc-catalog-verify` reads the
  snapshot, hits `/tables`, verifies SHA256 still matches; fails on
  drift. Run weekly in scheduled CI.

## Governance integration

Plan P6A.S5.T1 wires the snapshot into:

- `ndlb-gate`:  every row in `registry/datasets.toml` whose
  `server_ref = "heasarc_w3browse"` must cite a table that exists
  in `registry/heasarc_catalogs.toml`.
- `data_servers_xref`:  flip `catalog_enumeration` from `partial`
  to `full` in `data_servers.toml#heasarc_w3browse` once a walker
  run produces >=500 rows.

## Known failure modes

- **TAP endpoint 5xx**: intermittent at peak GSFC load. Backoff
  (already handled by `download_stack::RetryClass::DefaultHttp`).
- **VOTable binary serialization**: unlikely here (HEASARC returns
  TABLEDATA per TAP spec), but validate Content-Type before parse.
- **Empty tables list**: a successful response with zero rows has
  been observed during maintenance windows; fail-closed rather than
  overwrite the snapshot with an empty set.

## RCA for past failures (seed)

(No prior incident record; this is the initial ADR. Future incidents
should be appended here with the format: `YYYY-MM-DD: <summary> --
<mitigation>`.)
