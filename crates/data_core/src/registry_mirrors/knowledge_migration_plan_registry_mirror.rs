//! # Knowledge Migration Plan Registry Mirror
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: see authoritative source line below -->
//!
//! Authoritative source: `registry/knowledge_migration_plan.toml`.
//!
//! - Status: `active`
//! - Updated: 2026-02-09
//! - Scope: knowledge sources and operational trackers
//!
//! ## Domains
//!
//! ### KM-001: claims
//!
//! - Strategy: `toml_primary_markdown_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/CLAIMS_EVIDENCE_MATRIX.md`
//!   - `docs/CLAIMS_TASKS.md`
//! - Authoritative TOML:
//!   - `registry/claims.toml`
//!   - `registry/claims_tasks.toml`
//! - Generated mirrors:
//!   - `docs/CLAIMS_EVIDENCE_MATRIX.md`
//!   - `docs/CLAIMS_TASKS.md`
//!   - `docs/generated/CLAIMS_REGISTRY_MIRROR.md`
//!   - `docs/generated/CLAIMS_TASKS_REGISTRY_MIRROR.md`
//! - Notes: Claims registry and executable task ledger are TOML-first; claims matrix and claims tasks markdown are generated mirrors.
//!
//! ### KM-008: claims_domains
//!
//! - Strategy: `toml_primary_generated_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/claims/CLAIMS_DOMAIN_MAP.csv`
//!   - `docs/claims/by_domain/*.md`
//! - Authoritative TOML:
//!   - `registry/claims_domains.toml`
//! - Generated mirrors:
//!   - `docs/generated/CLAIMS_DOMAINS_REGISTRY_MIRROR.md`
//!   - `docs/claims/INDEX.md`
//!   - `docs/claims/CLAIMS_DOMAIN_MAP.csv`
//!   - `docs/claims/by_domain/*.md`
//! - Notes: Domain crosswalk and by-domain markdown/csv mirrors are generated from TOML.
//!
//! ### KM-009: claim_tickets
//!
//! - Strategy: `toml_primary_generated_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/tickets/*.md`
//! - Authoritative TOML:
//!   - `registry/claim_tickets.toml`
//! - Generated mirrors:
//!   - `docs/generated/CLAIM_TICKETS_REGISTRY_MIRROR.md`
//!   - `docs/tickets/*.md`
//!   - `docs/tickets/INDEX.md`
//! - Notes: Claims audit ticket metadata and claim linkage are normalized into TOML and rendered back into docs/tickets mirrors.
//!
//! ### KM-002: insights
//!
//! - Strategy: `toml_primary_narrative_overlay`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/INSIGHTS.md`
//! - Authoritative TOML:
//!   - `registry/insights.toml`
//! - Generated mirrors:
//!   - `docs/generated/INSIGHTS_REGISTRY_MIRROR.md`
//! - Notes: Legacy insights markdown is now generated from TOML registry + narrative overlay.
//!
//! ### KM-003: experiments
//!
//! - Strategy: `toml_primary_narrative_overlay`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md`
//! - Authoritative TOML:
//!   - `registry/experiments.toml`
//! - Generated mirrors:
//!   - `docs/generated/EXPERIMENTS_REGISTRY_MIRROR.md`
//! - Notes: Legacy experiments markdown is now generated from TOML registry + narrative overlay.
//!
//! ### KM-004: operational_trackers
//!
//! - Strategy: `toml_primary_markdown_overlay`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/ROADMAP.md`
//!   - `docs/TODO.md`
//!   - `docs/NEXT_ACTIONS.md`
//! - Authoritative TOML:
//!   - `registry/roadmap.toml`
//!   - `registry/todo.toml`
//!   - `registry/next_actions.toml`
//! - Generated mirrors:
//!   - `docs/generated/ROADMAP_REGISTRY_MIRROR.md`
//!   - `docs/generated/TODO_REGISTRY_MIRROR.md`
//!   - `docs/generated/NEXT_ACTIONS_REGISTRY_MIRROR.md`
//! - Notes: Legacy operational markdown is generated from TOML registries + narrative overlays.
//!
//! ### KM-005: requirements
//!
//! - Strategy: `toml_primary_markdown_overlay`
//! - Status: `complete`
//! - Source markdown:
//!   - `REQUIREMENTS.md`
//!   - `docs/REQUIREMENTS.md`
//!   - `docs/requirements/*.md`
//! - Authoritative TOML:
//!   - `registry/requirements.toml`
//! - Generated mirrors:
//!   - `docs/generated/REQUIREMENTS_REGISTRY_MIRROR.md`
//! - Notes: Requirements markdown set is generated from requirements registry + narrative overlay.
//!
//! ### KM-010: markdown_governance
//!
//! - Strategy: `toml_primary_policy_enforcement`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/generated/MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md`
//! - Authoritative TOML:
//!   - `registry/markdown_governance.toml`
//! - Generated mirrors:
//!   - `docs/generated/MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md`
//! - Notes: Governance registry classifies all markdown lifecycle modes and drives header/parity verifiers.
//!
//! ### KM-011: bibliography
//!
//! - Strategy: `toml_primary_markdown_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/BIBLIOGRAPHY.md`
//! - Authoritative TOML:
//!   - `registry/bibliography.toml`
//! - Generated mirrors:
//!   - `docs/BIBLIOGRAPHY.md`
//!   - `docs/generated/BIBLIOGRAPHY_REGISTRY_MIRROR.md`
//! - Notes: Bibliography is TOML-first with generated markdown mirrors.
//!
//! ### KM-006: generated_artifacts
//!
//! - Strategy: `artifact_only_no_manual_edit`
//! - Status: `active`
//! - Source markdown:
//!   - `data/artifacts/README.md`
//! - Notes: Keep README markdown in-place for repo navigation and artifact conventions.
//!
//! ### KM-007: research_narratives
//!
//! - Strategy: `narrative_primary_raw_capture_backup`
//! - Status: `active`
//! - Source markdown:
//!   - `AGENTS.md`
//!   - `CLAUDE.md`
//!   - `GEMINI.md`
//!   - `README.md`
//!   - `curated/**/*.md`
//!   - `data/csv/README.md`
//! - Notes: Residual manual narratives remain for repository governance and curated inputs that should stay human-authored.
//!
//! ### KM-012: external_sources
//!
//! - Strategy: `toml_primary_generated_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/external_sources/*.md`
//! - Authoritative TOML:
//!   - `registry/external_sources.toml`
//! - Generated mirrors:
//!   - `docs/generated/EXTERNAL_SOURCES_REGISTRY_MIRROR.md`
//!   - `docs/external_sources/*.md`
//!   - `docs/external_sources/INDEX.md`
//! - Notes: External source dossiers are TOML-first; markdown under docs/external_sources is generated from registry/external_sources.toml.
//!
//! ### KM-013: theory_engineering_narratives
//!
//! - Strategy: `toml_primary_generated_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/theory/*.md`
//!   - `docs/engineering/*.md`
//! - Authoritative TOML:
//!   - `registry/research_narratives.toml`
//! - Generated mirrors:
//!   - `docs/generated/RESEARCH_NARRATIVES_REGISTRY_MIRROR.md`
//!   - `docs/theory/*.md`
//!   - `docs/theory/INDEX.md`
//!   - `docs/engineering/*.md`
//!   - `docs/engineering/INDEX.md`
//! - Notes: Theory and engineering narrative dossiers are TOML-first with generated markdown mirrors.
//!
//! ### KM-014: book_docs
//!
//! - Strategy: `toml_primary_generated_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/book/src/**/*.md`
//! - Authoritative TOML:
//!   - `registry/book_docs.toml`
//! - Generated mirrors:
//!   - `docs/generated/BOOK_DOCS_REGISTRY_MIRROR.md`
//!   - `docs/book/src/**/*.md`
//! - Notes: mdBook source pages are TOML-first and rendered back into docs/book/src markdown mirrors.
//!
//! ### KM-015: docs_root_narratives
//!
//! - Strategy: `toml_primary_generated_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/*.md`
//! - Authoritative TOML:
//!   - `registry/docs_root_narratives.toml`
//! - Generated mirrors:
//!   - `docs/generated/DOCS_ROOT_NARRATIVES_REGISTRY_MIRROR.md`
//!   - `docs/*.md`
//! - Notes: Root-level docs narratives are TOML-first and generated back into docs/*.md.
//!
//! ### KM-016: reports_narratives
//!
//! - Strategy: `toml_primary_generated_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `reports/*.md`
//! - Authoritative TOML:
//!   - `registry/reports_narratives.toml`
//! - Generated mirrors:
//!   - `docs/generated/REPORTS_NARRATIVES_REGISTRY_MIRROR.md`
//!   - `reports/*.md`
//! - Notes: Reports narratives are TOML-first and rendered back into reports/*.md mirrors.
//!
//! ### KM-017: docs_convos_narratives
//!
//! - Strategy: `toml_primary_generated_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `docs/convos/*.md`
//! - Authoritative TOML:
//!   - `registry/docs_convos.toml`
//! - Generated mirrors:
//!   - `docs/generated/DOCS_CONVOS_REGISTRY_MIRROR.md`
//!   - `docs/convos/*.md`
//! - Notes: docs/convos extracts are TOML-first and rendered back into docs/convos/*.md mirrors.
//!
//! ### KM-018: data_artifact_narratives
//!
//! - Strategy: `toml_primary_generated_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `data/artifacts/ALGEBRAIC_FOUNDATIONS.md`
//!   - `data/artifacts/BIBLIOGRAPHY.md`
//!   - `data/artifacts/FINAL_REPORT.md`
//!   - `data/artifacts/QUANTUM_REPORT.md`
//!   - `data/artifacts/SIMULATION_REPORT.md`
//!   - `data/artifacts/extracted_equations.md`
//!   - `data/artifacts/reality_check_and_synthesis.md`
//! - Authoritative TOML:
//!   - `registry/data_artifact_narratives.toml`
//! - Generated mirrors:
//!   - `docs/generated/DATA_ARTIFACT_NARRATIVES_REGISTRY_MIRROR.md`
//!   - `data/artifacts/ALGEBRAIC_FOUNDATIONS.md`
//!   - `data/artifacts/BIBLIOGRAPHY.md`
//!   - `data/artifacts/FINAL_REPORT.md`
//!   - `data/artifacts/QUANTUM_REPORT.md`
//!   - `data/artifacts/SIMULATION_REPORT.md`
//!   - `data/artifacts/extracted_equations.md`
//!   - `data/artifacts/reality_check_and_synthesis.md`
//! - Notes: Selected high-information artifact narratives are TOML-first; README remains markdown for repository conventions.
//!
//! ### KM-019: entrypoint_docs
//!
//! - Strategy: `toml_primary_generated_mirror_with_manual_overlay_exceptions`
//! - Status: `complete`
//! - Source markdown:
//!   - `CLAUDE.md`
//!   - `GEMINI.md`
//!   - `README.md`
//!   - `curated/README.md`
//!   - `curated/01_theory_frameworks/README_ROCQ.md`
//!   - `data/csv/README.md`
//!   - `data/artifacts/README.md`
//! - Authoritative TOML:
//!   - `registry/entrypoint_docs.toml`
//! - Generated mirrors:
//!   - `README.md`
//!   - `curated/README.md`
//!   - `curated/01_theory_frameworks/README_ROCQ.md`
//!   - `data/csv/README.md`
//!   - `data/artifacts/README.md`
//!   - `docs/generated/ENTRYPOINT_DOCS_REGISTRY_MIRROR.md`
//! - Notes: Entrypoint markdown is TOML-first except CLAUDE.md and GEMINI.md, which are immutable manual stubs that must not be rewritten by export pipelines. AGENTS.md is governed separately by agents.toml.
//!
//! ### KM-020: navigator
//!
//! - Strategy: `toml_primary_generated_mirror`
//! - Status: `complete`
//! - Source markdown:
//!   - `NAVIGATOR.md`
//! - Authoritative TOML:
//!   - `registry/navigator.toml`
//! - Generated mirrors:
//!   - `NAVIGATOR.md`
//!   - `docs/generated/NAVIGATOR_REGISTRY_MIRROR.md`
//! - Notes: Navigator is TOML-first and rendered as publish output markdown.
//!
//! ### KM-021: legacy_csv_canonical_data
//!
//! - Strategy: `toml_primary_data_canonicalization`
//! - Status: `complete`
//! - Authoritative TOML:
//!   - `registry/legacy_csv_datasets.toml`
//!   - `registry/data/legacy_csv/*.toml`
//! - Notes: Legacy CSV corpora is migrated into canonical TOML datasets with parity verification.
//!
//! ## Phases
//!
//! ### KMP-P1: inventory_and_capture
//!
//! - Status: `complete`
//! - Deliverables:
//!   - registry/knowledge_sources.toml
//!   - registry/knowledge/documents.toml
//!
//! ### KMP-P2: curated_operational_registries
//!
//! - Status: `complete`
//! - Deliverables:
//!   - registry/roadmap.toml
//!   - registry/todo.toml
//!   - registry/next_actions.toml
//!   - registry/requirements.toml
//!
//! ### KMP-P3: toml_to_markdown_mirror_exports
//!
//! - Status: `complete`
//! - Deliverables:
//!   - docs/generated/*_REGISTRY_MIRROR.md
//!
//! ### KMP-P4: claims_support_normalization
//!
//! - Status: `complete`
//! - Deliverables:
//!   - registry/claims_tasks.toml
//!   - registry/claims_domains.toml
//!   - registry/claim_tickets.toml
//!
//! ### KMP-P5: policy_hardening
//!
//! - Status: `active`
//! - Deliverables:
//!   - Generated-header compliance verifier for TOML mirrors
//!   - Governance parity verifier (knowledge_sources vs markdown_governance)
//!   - CI checks for mirror freshness
//!   - Registry-first make target sequencing
//!   - Ticket mirror consistency verifier (registry/claim_tickets.toml <-> docs/tickets/*.md)
//!
//! ### KMP-P6: legacy_tracker_retirement
//!
//! - Status: `pending`
//! - Deliverables:
//!   - Retire remaining manually-edited markdown trackers after TOML parity review
//!
//! ### KMP-P7: markdown_governance_registry
//!
//! - Status: `complete`
//! - Deliverables:
//!   - crates/gororoba_cli_data/src/bin/markdown_registry.rs
//!   - registry/markdown_governance.toml
//!   - src/verification/verify_markdown_governance_headers.py
//!   - src/verification/verify_markdown_governance_parity.py
//!   - docs/generated/MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md
//!
//! ### KMP-P8: external_sources_registry_normalization
//!
//! - Status: `complete`
//! - Deliverables:
//!   - cargo run -p gororoba_cli_data --bin markdown-registry -- normalize-external-sources --bootstrap-from-markdown
//!   - registry/external_sources.toml
//!   - docs/generated/EXTERNAL_SOURCES_REGISTRY_MIRROR.md
//!   - docs/external_sources/*.md
//!   - docs/external_sources/INDEX.md
//!
//! ### KMP-P9: theory_engineering_registry_normalization
//!
//! - Status: `complete`
//! - Deliverables:
//!   - cargo run -p gororoba_cli_data --bin markdown-registry -- promote-research-narratives
//!   - registry/research_narratives.toml
//!   - cargo run -p gororoba_cli_data --bin registry-emit -- research-narratives-mirror
//!   - docs/generated/RESEARCH_NARRATIVES_REGISTRY_MIRROR.md
//!   - cargo run -p gororoba_cli_data --bin registry-emit -- research-narratives-legacy
//!   - docs/theory/*.md
//!   - docs/theory/INDEX.md
//!   - docs/engineering/*.md
//!   - docs/engineering/INDEX.md
//!
//! ### KMP-P10: global_markdown_inventory_and_entrypoint_conversion
//!
//! - Status: `complete`
//! - Deliverables:
//!   - cargo run -p gororoba_cli_data --bin markdown-registry -- build-inventory
//!   - registry/markdown_inventory.toml
//!   - cargo run -p gororoba_cli_data --bin markdown-registry -- normalize-entrypoint-docs --bootstrap-from-markdown
//!   - registry/entrypoint_docs.toml
//!   - docs/generated/ENTRYPOINT_DOCS_REGISTRY_MIRROR.md
//!
//! ### KMP-P10B: book_docs_registry_normalization
//!
//! - Status: `complete`
//! - Deliverables:
//!   - cargo run -p gororoba_cli_data --bin markdown-registry -- normalize-book-docs --bootstrap-from-markdown
//!   - registry/book_docs.toml
//!   - docs/generated/BOOK_DOCS_REGISTRY_MIRROR.md
//!   - docs/book/src/**/*.md
//!
//! ### KMP-P11: docs_root_narratives_registry_normalization
//!
//! - Status: `complete`
//! - Deliverables:
//!   - cargo run -p gororoba_cli_data --bin markdown-registry -- promote-docs-root-narratives
//!   - registry/docs_root_narratives.toml
//!   - cargo run -p gororoba_cli_data --bin registry-emit -- docs-root-narratives-mirror
//!   - docs/generated/DOCS_ROOT_NARRATIVES_REGISTRY_MIRROR.md
//!   - cargo run -p gororoba_cli_data --bin registry-emit -- docs-root-narratives-legacy
//!   - docs/*.md
//!
//! ### KMP-P12: reports_and_docs_convos_registry_normalization
//!
//! - Status: `complete`
//! - Deliverables:
//!   - cargo run -p gororoba_cli_data --bin markdown-registry -- normalize-reports-narratives --bootstrap-from-markdown
//!   - registry/reports_narratives.toml
//!   - docs/generated/REPORTS_NARRATIVES_REGISTRY_MIRROR.md
//!   - reports/*.md
//!   - cargo run -p gororoba_cli_data --bin markdown-registry -- normalize-docs-convos --bootstrap-from-markdown
//!   - registry/docs_convos.toml
//!   - docs/generated/DOCS_CONVOS_REGISTRY_MIRROR.md
//!   - docs/convos/*.md
//!
//! ### KMP-P13: data_artifact_narratives_registry_normalization
//!
//! - Status: `complete`
//! - Deliverables:
//!   - cargo run -p gororoba_cli_data --bin markdown-registry -- normalize-data-artifact-narratives --bootstrap-from-markdown
//!   - registry/data_artifact_narratives.toml
//!   - docs/generated/DATA_ARTIFACT_NARRATIVES_REGISTRY_MIRROR.md
//!   - data/artifacts/ALGEBRAIC_FOUNDATIONS.md
//!   - data/artifacts/BIBLIOGRAPHY.md
//!   - data/artifacts/FINAL_REPORT.md
//!   - data/artifacts/QUANTUM_REPORT.md
//!   - data/artifacts/SIMULATION_REPORT.md
//!   - data/artifacts/extracted_equations.md
//!   - data/artifacts/reality_check_and_synthesis.md
//!
//! ### KMP-P14: legacy_csv_canonical_toml_migration
//!
//! - Status: `complete`
//! - Deliverables:
//!   - crates/gororoba_cli_data/src/bin/csv_canonicalization.rs (inventory)
//!   - registry/csv_inventory.toml
//!   - crates/gororoba_cli_data/src/bin/csv_canonicalization.rs (migrate)
//!   - registry/legacy_csv_datasets.toml
//!   - registry/data/legacy_csv/*.toml
//!   - crates/gororoba_cli_data/src/bin/csv_canonicalization.rs (verify)
//!   - Makefile registry-data targets
//!
//! ### KMP-P15: curated_csv_canonical_toml_migration_wave
//!
//! - Status: `complete`
//! - Deliverables:
//!   - registry/curated_csv_datasets.toml
//!   - registry/data/curated_csv/*.toml
//!   - crates/gororoba_cli_data/src/bin/csv_canonicalization.rs (corpus-parameterized migrate)
//!   - crates/gororoba_cli_data/src/bin/csv_canonicalization.rs (corpus-parameterized verify)
//!   - crates/gororoba_cli_data/src/bin/csv_canonicalization.rs (inventory progress classification)
//!   - crates/gororoba_cli_data/src/bin/csv_canonicalization.rs (migration-scope summary from inventory counts)
//!   - registry/csv_migration_scope.toml wave_2 complete
//!
//! ### KMP-P16: project_external_archive_csv_policy_wave
//!
//! - Status: `complete`
//! - Deliverables:
//!   - registry/project_csv_split_policy.toml
//!   - registry/manifests/project_csv_canonical_manifest.txt
//!   - registry/manifests/project_csv_generated_manifest.txt
//!   - registry/project_csv_canonical_datasets.toml
//!   - registry/project_csv_generated_artifacts.toml
//!   - registry/data/project_csv/canonical/*.toml
//!   - registry/data/project_csv/generated/*.toml
//!   - registry/external_csv_holding.toml
//!   - registry/archive_csv_holding.toml
//!   - registry/manifests/external_csv_holding_manifest.txt
//!   - registry/manifests/archive_csv_holding_manifest.txt
//!   - registry/archive_csv_holding_datasets.toml
//!   - registry/data/archive_csv_holding/*.toml
//!   - crates/scrolls_core
//!   - gororoba_cli binary: scrollify-csv
//!
//! ### KMP-P17: external_csv_scroll_conversion_wave
//!
//! - Status: `pending`
//! - Deliverables:
//!   - registry/external_csv_holding_datasets.toml
//!   - registry/data/external_csv_holding/*.toml
//!   - external source provenance-preserving conversion policy checks
//!
//! ## Policies
//!
//! ### KMPOL-001: bootstrap_vs_operational_authoring
//!
//! - Status: `active`
//! - Statement: Markdown->TOML normalization for claims-support registries is bootstrap-only. Operational authoring is TOML-first with generated markdown mirrors.
//! - Enforcement:
//!   - cargo run -p gororoba_cli_data --bin markdown-registry -- normalize-claims-support --bootstrap-from-markdown
//!   - make registry excludes registry-normalize-claims from default ingest flow
//!   - mirror freshness and governance verifiers run in make registry
//!
//! ### KMPOL-002: untracked_markdown_mirror_publish_mode
//!
//! - Status: `active`
//! - Statement: TOML-generated markdown mirrors are publish artifacts and are intentionally not tracked in git.
//! - Enforcement:
//!   - Git index tracks only manual markdown exceptions and explicit artifact README files
//!   - make registry validates TOML registries without requiring tracked markdown mirrors
//!   - make docs-publish generates and verifies markdown mirrors for documentation output
//!   - .gitignore excludes TOML-generated markdown mirror paths from git tracking
//!
//! ### KMPOL-003: legacy_csv_toml_canonical_policy
//!
//! - Status: `active`
//! - Statement: Legacy CSV under data/csv/legacy must be represented by canonical TOML datasets under registry/data/legacy_csv with parity verification.
//! - Enforcement:
//!   - make registry-data builds csv inventory and migrates legacy CSV to TOML canon
//!   - csv-canonicalization verify enforces full coverage and checksum/row parity
//!   - registry/legacy_csv_datasets.toml is authoritative for legacy CSV canonical map
