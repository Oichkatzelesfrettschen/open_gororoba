//! # TODOWRITE — Lambda Workspace Consolidation Ledger
//!
//! Status legend: [ ] not started, [~] in progress, [x] done
//!
//! ## Scope Statement
//! Consolidate `lambda-research`, `lambda-synthesis-experiments`, and `LambdaLearner` into a single normalized workspace under `/home/eirikr/Github/lambda_gororoba`, with root workspace as source-of-truth and `merge_in/` as the integration intake lane, plus auditable sync, reproducible backups, standardized requirements documentation, and harmonized documentation and claims.
//!
//! ## Tasks (with owner and acceptance)
//!
//! 1. [x] Create workspace-level inventory of repo existence, remotes, and commit state (owner: Codex; done: today).
//! 2. [x] Verify origin pull status for every known lambda repo instance (owner: Codex; done: today).
//! 3. [x] Ensure each target repo is addressable under `lambda_gororoba` and normalized (owner: Codex; done).
//! 4. [x] Create workspace scaffolding directories `docs/`, `scripts/`, `logs/`, `src/` (owner: Codex; done).
//! 5. [x] Create non-destructive documentation backups in `~/Documents/lambda_gororoba_backups/<timestamp>/` (owner: Codex; done).
//! 6. [x] Build automated repo audit script and produce baseline audit logs (owner: Codex; done).
//! 7. [x] Generate baseline TODO/FIXME extraction and per-module counts for source control visibility (owner: Codex; done).
//! 8. [x] Reconcile roadmaps from child repos into `ROADMAP_SYNTHESIS.md` and preserve provenance links (owner: Codex).
//! 9. [x] Build `MODULE_REQUIREMENTS_MATRIX.md` and keep it synced (owner: Codex).
//! 10. [x] Standardize each module’s `requirements.md` structure and cross-link to authoritative sources (owner: Codex; done: 2026-02-13).
//! 11. [x] Audit and harmonize license files to a common `GPL-2.0-only` target (owner: Codex; decision: final harmonization applied across all target repos).
//! 12. [x] Define workspace quality policy where TODO/FIXME, warnings, and unverified claims are tracked as blocking risks (owner: Codex; policy in `docs/QUALITY_POLICY.md`).
//! 13. [x] Draft unified backup + restore playbook with checksums and timestamps (owner: Codex).
//! 14. [x] Add workspace-level sync script: origin status, pull, audit, and log outputs (owner: Codex).
//! 15. [x] Add workspace-level backup script for docs and requirement evidence (owner: Codex).
//! 16. [x] Create roadmap status dashboard in `docs/ROADMAP_STATUS.md` (owner: Codex).
//! 17. [x] Map all TODO/FIXME hotspots to concrete tasks by priority and estimate effort (owner: Codex; latest hotspot scan refreshed: 2026-02-13).
//! 18. [x] Create a workspace integration contract that defines how repos exchange data and artifacts without forcing one implementation language (owner: Codex; implemented via `docs/REPO_INTEGRATION_CONTRACT.md`).
//! 19. [x] Define benchmark protocol: accuracy, runtime, memory, reproducibility metadata schema (owner: Codex; in `docs/BENCHMARK_PROTOCOL.md`).
//! 20. [x] Define claim registry for assertions and falsifiable hypotheses with evidence sources (owner: Codex; in `docs/CLAIM_REGISTER.md`).
//! 21. [x] Update `GEMINI.md` and adjacent governance docs in workspace root with consolidated mission and scope (owner: Codex).
//! 22. [x] Run sync + audit and archive completed tasks after each consolidation pass (owner: Codex; latest pass recorded in `logs/workspace_sync_audit_20260213_155129.md` and `logs/repo_audit_2026-02-13T15:51:39-08:00.md`).
//! 23. [x] Compare origin alignment for each workspace repository and record report (owner: Codex).
//! 24. [x] Consolidate duplicate working copies outside workspace into canonical workspace locations (owner: Codex).
//! 25. [x] Build a lightweight orchestration playbook for coordinating scripted workflows across the three repos (owner: Codex; scripts and evidence logs now coordinate by `scripts/*.sh` outputs).
//! 26. [ ] Verify build/test reproducibility checklist for Python and Node workflows; log command sequence and failures (owner: Codex).
//! 27. [x] Add script outputs to `logs/` as build artifacts with machine-readable naming convention (owner: Codex).
//! 28. [x] Prepare push-ready branch guidance and conflict-resolution strategy for each repo (owner: Codex).
//! 29. [x] Publish conflict-resolution and release checklist in `docs/SEVEN_DAY_EXECUTION_NOTE.md` (owner: Codex; active checklist pending with next-pass completion matrix).
//! 30. [x] Archive source-discovery manifests and older runs using retention policy (owner: Codex).
//! 31. [ ] Archive no-longer-relevant docs in `docs/archive/` with rationale and hash log (owner: Codex).
//! 32. [ ] Resolve top-level false/uncertain claims in existing documentation via evidence checks (owner: Codex).
//! 33. [ ] Establish visual documentation style guide for dark-mode static infographics (owner: Codex).
//! 34. [x] Add reproducible original-source discovery script and manifest for external links, checksums, and retrieval commands (owner: Codex).
//! 35. [x] Add claim review checkpoints and TODO debt triage to each consolidation pass (owner: Codex; no fixed weekly cadence).
//! 36. [x] Publish a harmonized integration contract and ownership matrix for each functional area (owner: Codex; see `docs/REPO_INTEGRATION_CONTRACT.md`).
//!
//! 37. [x] Run deterministic root synthesis pass and publish conflict matrix in `logs/conflict_matrix_*` (owner: Codex; latest pass `logs/conflict_matrix_20260213_155203.tsv`).
//! 38. [x] Adopt `merge_in/` staging model across core workspace scripts with clone-if-missing bootstrap from origin URLs (owner: Codex; implemented in `scripts/*.sh` integration stack).
//! 39. [x] Capture canonical origin URLs for reproducible repository retrieval commands in source discovery workflow (owner: Codex; maintained in `scripts/discover_original_sources.sh` + `docs/ORIGINAL_SOURCE_MANIFEST.md`).
//! 40. [x] Publish `merge_in` operational workflow and evidence map for deterministic root-led integration passes (owner: Codex; see `docs/MERGE_IN_WORKFLOW.md`).
//! 41. [x] Produce file-by-file consolidation decision table for `src/lambda_unified/modules/*` with role, hash, duplicate scope, and action class (owner: Codex; `docs/module_file_consolidation_decisions_latest.tsv`).
//! 42. [x] Publish module consolidation rulebook documenting canonical/merge/archive policy by file class and explicit duplicate candidates (owner: Codex; `docs/MODULE_FILE_CONSOLIDATION_RULES.md`).
//! 43. [x] Execute archive moves for approved duplicate doc candidates with hash-preserving manifest log (owner: Codex; `docs/module_archive_moves_latest.tsv`).
//! 44. [x] Generate root-level canonical docs and bibliography indices from consolidation decisions (owner: Codex; `src/lambda_unified/modules/indices/`).
//! 45. [x] Generate module topic map from canonical doc decisions for navigation and planning (owner: Codex; `src/lambda_unified/modules/indices/MODULE_TOPIC_MAP.md`).
//! 46. [x] Implement and run automated consolidation policy verifier against decision table + archive manifest (owner: Codex; `logs/module_consolidation_verification_20260213_160549.md`).
//! 47. [x] Refresh consolidation backup snapshot in `~/Documents` after archive/index/verification execution (owner: Codex; latest backup path is under `~/Documents/lambda_gororoba_backups/`).
//! 48. [x] Add orchestrated consolidation gate script to run inventory, decisions, indices, and verifiers in one command (owner: Codex; `scripts/run_module_consolidation_gate.sh`).
//! 49. [x] Add repository policy: Python scripts must be multithreaded or explicitly exempted, with automated verification and evidence logs (owner: Codex; `docs/PYTHON_THREADING_POLICY.md`, `scripts/verify_python_multithreading_policy.sh`).
//! 50. [x] Wire optional consolidation gate execution into workspace sync flow for one-command pass orchestration (owner: Codex; `RUN_CONSOLIDATION_GATE=1 bash scripts/sync_and_audit_workspace.sh`).
//! 51. [x] Refactor first high-impact Python IO scripts to real threaded execution and reduce active exemptions (owner: Codex; updated `lambda-research/*/scripts/download_papers.py`, policy now `7` compliant / `44` exempt / `0` non-compliant in `logs/python_multithreading_policy_20260213_162455.md`).
//! 52. [x] Refactor second high-impact Python scripts (`search_papers.py`, `update_metadata.py` pairs) to threaded execution and reduce exemptions (owner: Codex; policy now `11` compliant / `40` exempt / `0` non-compliant in `logs/python_multithreading_policy_20260213_163022.md`).
//! 53. [x] Author first-party synthesis documentation corpus to define unified project identity, interface contracts, reconciliation logic, and novel research program (owner: Codex; `docs/first_party/*`).
//!
//! ## Ownership model
//! - Primary owner: Codex
//! - Reviewer checkpoints: user signoff at end of each phase
//! - Conflict resolution: prioritize origin consistency first, then legal compliance, then research coherence
//!
