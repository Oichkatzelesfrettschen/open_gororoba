# GEMINI Workspace Logbook

This file tracks consolidated workspace governance and progress for `lambda_gororoba`.

## Current State
- Workspace root: `/home/eirikr/Github/lambda_gororoba`
- Merge staging root: `/home/eirikr/Github/lambda_gororoba/merge_in`
- Repos are resolved from `merge_in/<repo>` first, with workspace paths as fallback bootstrap:
  - `lambda-research`
  - `lambda-synthesis-experiments`
  - `LambdaLearner`
- Language integration remains multi-language (Python, TypeScript, docs tooling); no pure Rust consolidation requirement.

## Reproducible Operations (in place)
- `scripts/sync_and_audit_workspace.sh` — pulls from origin and writes `logs/workspace_sync_audit_<TS>.md`.
- `scripts/backup_documents_workspace.sh` — creates `~/Documents/lambda_gororoba_backups/<TS>/` plus `backup_manifest.sha256`.
- `scripts/audit_repos.sh` — baseline repository metadata collector.
- `scripts/discover_original_sources.sh` — discovers explicit upstream URLs and emits reproducible source manifests in `logs/source_discovery_*`.

## Consolidation artifacts
- `TODOWRITE.md` — task ledger and ownership model.
- `ROADMAP_SYNTHESIS.md` — phased program.
- `MODULE_REQUIREMENTS_MATRIX.md` — installation and stack mapping by module.
- `docs/ROADMAP_STATUS.md` — live completion indicator.
- `docs/ORIGINAL_SOURCE_MANIFEST.md` — source discovery process, reproducible commands, and retention rules.
- `docs/REPRODUCIBLE_BACKUP_MANIFEST.md` — backup protocol.
- `agents.md` — role and operating conventions.
- `CLAUDE.md` — execution constraints and coordination rules.

## Compliance Notes
- No items were deleted during this pass.
- `~/Documents` is not modified by deletion; backups are additive.
- License posture is harmonized to `GPL-2.0-only` for workspace repo-level licensing artifacts in this consolidation stream.

## Policy
- Treat audit warnings and TODO/FIXME debt as gating items in roadmap progression.
- Every action sequence should emit logs under `logs/`.
- Every claim introduced in planning artifacts should have a source or explicit hypothesis.
