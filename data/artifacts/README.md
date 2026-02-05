data/artifacts conventions
=========================

This tree stores generated (derived) outputs. It is distinct from:
- data/external/: cached third-party inputs (with provenance).
- data/csv/: small, shareable derived tables (some are also artifacts).

Path convention
---------------

All artifacts should live under:

data/artifacts/<domain>/<name>.<ext>

Examples:
- data/artifacts/images/foo.png
- data/artifacts/manufacturing/bom.csv
- data/artifacts/reports/summary.md

This convention keeps outputs discoverable and enables automated hygiene checks.

Artifact manifest
-----------------

The file data/artifacts/ARTIFACTS_MANIFEST.csv is a small, explicit index of
artifacts that are referenced by canonical docs/verifiers.

Verifier:
- src/verification/verify_artifacts_manifest.py

Provenance sidecars
-------------------

Some artifacts are required to include a provenance sidecar:

<artifact>.<ext>.PROVENANCE.json

Example:
data/artifacts/images/foo.png.PROVENANCE.json

Schema (minimal):
- provenance_version: 1
- artifact_path: "data/artifacts/<domain>/<name>.<ext>"
- generator: path to the generating script (usually under src/scripts/)
- inputs: list of input paths (strings)
- parameters: JSON object (dict) of relevant knobs

Notes:
- Sidecars are intended to be stable and offline-checkable.
- Full reproducibility also depends on pinned externals under data/external/ and
  deterministic generator code paths.

Reorganization planning (safe, phase-gated)
-------------------------------------------

Historical artifacts may exist directly under data/artifacts/ (flat layout).
Do not move these casually. Use the opt-in plan and Phase 0 mover:

- Plan: `make artifacts-reorg-plan` -> reports/artifacts_reorg_plan.md
- Phase 0 dry-run: `make artifacts-phase0-dry`
- Phase 0 apply: `make artifacts-phase0-apply`
- Phase 0 rollback: `make artifacts-phase0-rollback LOG=reports/artifacts_phase0_moves_<ts>.json`

Phase 0 is intentionally conservative: it moves only artifacts that are both
unreferenced (by repo text) and unmanifested (not listed in ARTIFACTS_MANIFEST.csv).

Phase 1 (suggested):
- For artifacts that ARE referenced but not yet in ARTIFACTS_MANIFEST.csv, prefer
  adding them to the manifest first (no moves). This makes references auditable
  before any path changes.
- If the artifact is high-value, add a .PROVENANCE.json sidecar (inputs, generator,
  parameters) before relocating.

Phase 2 (suggested):
- Only after Phase 1 coverage exists, migrate referenced artifacts into
  data/artifacts/<domain>/... and update ALL references (docs, scripts, tests).
- Consider keeping a short "compatibility window" by leaving copies at old paths
  temporarily, but avoid long-lived duplication.
