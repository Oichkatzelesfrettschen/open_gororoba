<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# TODO/FIXME Scan (2026-02-07)

Command used (excluding venv/cache/artifact trees):

`rg -n "\\b(TODO|FIXME)\\b" -S --glob '!venv/**' --glob '!.mamba/**' --glob '!data/**' --glob '!curated/**' --glob '!archive/**'`

## High-signal hotspots

- Claims pipeline backlog: `docs/CLAIMS_TASKS.md` (many rows still `TODO`)
- Global tracker: `docs/TODO.md`
- Open ticket stubs:
  - `docs/tickets/TICKET_WARP_PHYSICS_RECONCILIATION.md`
  - `reports/tickets_inventory.md`
- Code-level implementation TODOs:
  - `crates/quantum_core/src/tensor_networks.rs`
  - `crates/quantum_core/src/casimir.rs`

## Notes

- This scan is a navigation aid only. Treat each TODO as a scoped
  hypothesis-to-check task.
- TODO entries in docs are expected and intentionally retained for provenance.
