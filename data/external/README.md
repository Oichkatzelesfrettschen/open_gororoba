# External Data Lane

This directory is for reproducibly fetched external datasets and source documents.

## Policy

- Files in this lane are treated as cacheable fetch artifacts.
- Every file must be reproducible from a documented source URL and fetch command.
- `data/external/PROVENANCE.local.json` is the machine-readable hash index.
- `registry/canonical/control_plane.sqlite3` is the canonical write-authoritative
  source-control plane for external source contracts and dossiers.
- `data/external/SOURCES.toml` is the generated machine-readable compatibility
  export for source-of-origin contracts (canonical URL, mirrors, access class,
  status, retrieval method, and blocker deadlines).
- Strict governance is defined in `registry/data_governance.toml` (lane `external`).

## Rebuild

1. Fetch data with Rust entrypoints (preferred):
   - `cargo run -p gororoba_cli --bin fetch-datasets -- --output-dir data/external`
2. Refresh hash provenance:
   - `cargo run -p gororoba_cli --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json`
3. Audit source coverage and blocked-policy deadlines:
   - `cargo run -p gororoba_cli --bin external-redownload-audit -- --out reports/external_redownload_audit_YYYY-MM-DD.toml --backend-order wget,curl,fetch`
4. Audit origin coverage:
   - `cargo run -p gororoba_cli --bin data-origin-audit -- --out reports/data_origin_audit_YYYY-MM-DD.toml`
5. Run fail-closed governance + semantic lane validators:
   - `cargo run -p gororoba_cli --bin data-governance-gate -- --enforce-origin true --enforce-semantic true --enforce-blocked-deadlines true`
   - `cargo run -p gororoba_cli --bin data-semantic-validate --`

## JPL Ephemeris (DE440)

Required by `flyby-crucible` three-body integration (Sprint 71+).

- **File**: `de440.bsp` (~120 MB, gitignored)
- **Source**: `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp`
- **SHA256**: `a4ce9bf9b3282becc9f4b2ac3cebe03a2ae7599981aabd7265fd8482fff7c4b5`
- **Coverage**: 1550-2650 CE, sub-meter inner planet accuracy
- **Fetch**: `wget -O data/external/de440.bsp "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"`

The loader (`gororoba_cli_physics::ephemeris_loader`) uses anise 0.9.5 to parse
this file. All state vectors are returned in ICRS/J2000 ECI -- zero frame rotation
required for flyby RA/Dec coordinates.

## Dark Matter Wind / Gravitational Focusing Papers (Sprint 71.3)

Used for ETA_WAKE downwind caustic parameter in flyby crucible.

- **arxiv_2502.04456_earth_dm_wind.pdf**
  - Title: Enhanced Axion-wind near Earth's Surface
  - Source: `https://arxiv.org/pdf/2502.04456`
  - Date: Feb 2025
  - Relevance: Earth's gravitational potential modifies local DM wind velocity and density

- **arxiv_2112.05718_gravitational_focusing.pdf**
  - Title: Gravitational focusing of light bosonic dark matter
  - Source: `https://arxiv.org/pdf/2112.05718`
  - Date: Dec 2021
  - Relevance: Quantifies density enhancement downstream of Earth in DM wind (caustic wake)

- **arxiv_1409.2858_earth_moon_modulation.pdf**
  - Title: Monthly Modulation in Dark Matter Direct-Detection Experiments
  - Source: `https://arxiv.org/pdf/1409.2858`
  - Date: Sep 2014
  - Relevance: Moon's gravitational focusing creates monthly density modulation at Earth

## Compatibility Note

`PROVENANCE.local.json` remains the hash/size/mtime cache, while
`data/external/SOURCES.toml` is now a read-only compatibility export generated
from the canonical SQLite control plane.
