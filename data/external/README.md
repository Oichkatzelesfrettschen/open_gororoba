# External Data Lane

This directory is for reproducibly fetched external datasets and source documents.

## Policy

- Files in this lane are treated as cacheable fetch artifacts.
- Every file must be reproducible from a documented source URL and fetch command.
- `data/external/PROVENANCE.local.json` is the machine-readable hash index.
- `data/external/SOURCES.toml` is the machine-readable source-of-origin contract
  (canonical URL, mirrors, access class, status, retrieval method, and blocker deadlines).
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

## Known Gap

`PROVENANCE.local.json` records hashes/sizes/mtimes while `SOURCES.toml` carries source URLs and blocker workflow state.
