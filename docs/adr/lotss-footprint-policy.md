# LoTSS DR3 Footprint Policy (plan P6A.S5.T3)

## Context

`registry/data_servers.toml#lotss_vo` and `#lotss_astrowise` carry
`catalog_enumeration = "partial"`. The LoTSS DR3 VO cone-search
endpoint (`https://vo.astron.nl/hetdex/q`) is tile-based: you query
against a specific on-sky tile, not against the full survey.

A true "full enumeration" of DR3 would require iterating every
HEALPix cell that intersects the LOFAR HBA-Dutch survey footprint
(~5400 square degrees at nside=64), which is ~500-1500 VO queries at
the endpoint's 250 ms rate-limit ceiling: 2-6 minutes per refresh.
That is affordable but expensive enough to warrant a boundary
decision rather than a blanket "enumerate everything" rule.

## Decision

LoTSS DR3 enumeration is SCOPED to the footprints that our active
experiments cover, plus a 0.5-degree buffer. Specifically:

1. **MaNGA overlap footprint**: all DR3 tiles within 0.5 degrees of
   any MaNGA DRPall galaxy position (RA, DEC columns). MaNGA peaks
   at DEC 60-62, where DR3 has best coverage.
2. **Explicit HETDEX field**: the Hobby-Eberly Telescope Dark Energy
   eXperiment field (RA 150-250 deg, DEC 50-60 deg) where LoTSS has
   highest completeness.
3. **Any field referenced in an `active`-status dataset row** with
   `server_ref = "lotss_vo"`.

Out of scope (for now):

- Full 5400 sq deg HBA-Dutch footprint enumeration.
- LoTSS-Low (15-30 MHz) extension.
- Future LOFAR2 data releases.

## Rationale

- **Active experiments are the forcing function.** We do not need a
  full survey snapshot if no analysis touches the additional tiles.
- **Rate-limit budget.** 250 ms per query x 500+ tiles = ~2 min per
  refresh, and the endpoint operators have asked for conservative
  use. A scoped walk stays well under 10% of the recommended
  query budget.
- **Refetch cost.** Bulk FITS from `lotss_astrowise` is the preferred
  path for full-footprint work; VO cone search is for targeted
  cross-match validation against active datasets.

## Refresh procedure

1. Grep `registry/datasets.toml` for rows with `server_ref =
   "lotss_vo"` and status=`active`; extract their RA/DEC refs or
   footprint descriptors.
2. Pull MaNGA DRPall from the cached `data/external/manga/` snapshot
   (38 GB root; see `registry/external_sources.toml` XS-0xx row for
   MaNGA provenance once P6A.S4 maps it).
3. Deduplicate tile coordinates; emit a query manifest.
4. Run the walker (future: `crates/data_core/src/catalogs/lotss_enumerate.rs`).
5. Persist to `registry/lotss_dr3_tiles.toml`.

## Expansion triggers

The scope expands (i.e., this ADR is superseded) if ANY of:

- A new active experiment references a LoTSS tile outside the
  current scope.
- The HBA-Dutch footprint is needed as a whole for a statistical
  null (e.g. noise characterization).
- LoTSS DR4 lands and expands the sky coverage materially.

## Known failure modes

- **Tile query returns >1000 rows**: enforce a hard limit in the
  walker; paginate using `OFFSET`.
- **VO service outage**: rare but multi-day outages have occurred;
  the walker should tolerate 24h staleness before failing
  `data_servers_xref`.
- **MaNGA footprint drift**: if MaNGA DR upgrades, re-extract
  coordinates; do not rely on a cached list.

## Acknowledgment

Per the ASTRON VO service terms of use (linked from
`data_servers.toml#lotss_vo`), every paper using DR3 data must cite
Shimwell et al. 2022 and the ASTRON operational acknowledgment
statement. Future automation should emit a citation stub into each
dataset row's `notes` field.
