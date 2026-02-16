# CLAUDE Workspace Notes

This folder is the consolidated lambda workspace controller.

- Keep operations non-destructive unless explicitly approved.
- Treat warning findings from audits as hard blockers.
- Preserve all source repositories and avoid deleting from `~/Documents`.

## Operating Rules
1. Pull all repos from origin before any consolidation pass.
2. Use `merge_in/` as the repository intake/staging lane and keep workspace root as source-of-truth.
3. Run `scripts/sync_and_audit_workspace.sh` before major edits.
4. Run `scripts/backup_documents_workspace.sh` before major edits when possible.
5. Update `TODOWRITE.md` whenever task states change.
6. For external artifacts, run `scripts/discover_original_sources.sh` and keep generated `logs/source_discovery_*` files with timestamp.
7. Do not require a Pure Rust implementation path; preserve multi-language workflows as-is.

## Repository Pointers
- `merge_in/lambda-research` -> `/home/eirikr/Github/lambda_gororoba/lambda-research` (or direct clone target)
- `merge_in/lambda-synthesis-experiments` -> `/home/eirikr/Github/lambda_gororoba/lambda-synthesis-experiments` (or direct clone target)
- `merge_in/LambdaLearner` -> `/home/eirikr/Github/lambda_gororoba/LambdaLearner` (or direct clone target)
