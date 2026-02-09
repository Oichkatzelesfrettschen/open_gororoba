# Knowledge Migration Plan Registry Mirror

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: TOML registry files under registry/ -->

Authoritative source: `registry/knowledge_migration_plan.toml`.

- Status: `active`
- Updated: 2026-02-09
- Scope: knowledge sources and operational trackers

## Domains

### KM-001: claims

- Strategy: `toml_primary_markdown_mirror`
- Status: `active`
- Source markdown:
  - `docs/CLAIMS_EVIDENCE_MATRIX.md`
  - `docs/CLAIMS_TASKS.md`
- Authoritative TOML:
  - `registry/claims.toml`
  - `registry/claims_tasks.toml`
- Generated mirrors:
  - `docs/CLAIMS_TASKS.md`
  - `docs/generated/CLAIMS_TASKS_REGISTRY_MIRROR.md`
- Notes: Claims matrix and executable task ledger are TOML-first; CLAIMS_TASKS markdown is generated from TOML.

### KM-008: claims_domains

- Strategy: `toml_primary_generated_mirror`
- Status: `complete`
- Source markdown:
  - `docs/claims/CLAIMS_DOMAIN_MAP.csv`
  - `docs/claims/by_domain/*.md`
- Authoritative TOML:
  - `registry/claims_domains.toml`
- Generated mirrors:
  - `docs/generated/CLAIMS_DOMAINS_REGISTRY_MIRROR.md`
  - `docs/claims/INDEX.md`
  - `docs/claims/CLAIMS_DOMAIN_MAP.csv`
  - `docs/claims/by_domain/*.md`
- Notes: Domain crosswalk and by-domain markdown/csv mirrors are generated from TOML.

### KM-009: claim_tickets

- Strategy: `toml_primary_generated_mirror`
- Status: `complete`
- Source markdown:
  - `docs/tickets/*.md`
- Authoritative TOML:
  - `registry/claim_tickets.toml`
- Generated mirrors:
  - `docs/generated/CLAIM_TICKETS_REGISTRY_MIRROR.md`
  - `docs/tickets/*.md`
  - `docs/tickets/INDEX.md`
- Notes: Claims audit ticket metadata and claim linkage are normalized into TOML and rendered back into docs/tickets mirrors.

### KM-002: insights

- Strategy: `toml_primary_narrative_overlay`
- Status: `complete`
- Source markdown:
  - `docs/INSIGHTS.md`
- Authoritative TOML:
  - `registry/insights.toml`
- Generated mirrors:
  - `docs/generated/INSIGHTS_REGISTRY_MIRROR.md`
- Notes: Legacy insights markdown is now generated from TOML registry + narrative overlay.

### KM-003: experiments

- Strategy: `toml_primary_narrative_overlay`
- Status: `complete`
- Source markdown:
  - `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md`
- Authoritative TOML:
  - `registry/experiments.toml`
- Generated mirrors:
  - `docs/generated/EXPERIMENTS_REGISTRY_MIRROR.md`
- Notes: Legacy experiments markdown is now generated from TOML registry + narrative overlay.

### KM-004: operational_trackers

- Strategy: `toml_primary_markdown_overlay`
- Status: `complete`
- Source markdown:
  - `docs/ROADMAP.md`
  - `docs/TODO.md`
  - `docs/NEXT_ACTIONS.md`
- Authoritative TOML:
  - `registry/roadmap.toml`
  - `registry/todo.toml`
  - `registry/next_actions.toml`
- Generated mirrors:
  - `docs/generated/ROADMAP_REGISTRY_MIRROR.md`
  - `docs/generated/TODO_REGISTRY_MIRROR.md`
  - `docs/generated/NEXT_ACTIONS_REGISTRY_MIRROR.md`
- Notes: Legacy operational markdown is generated from TOML registries + narrative overlays.

### KM-005: requirements

- Strategy: `toml_primary_markdown_overlay`
- Status: `complete`
- Source markdown:
  - `REQUIREMENTS.md`
  - `docs/REQUIREMENTS.md`
  - `docs/requirements/*.md`
- Authoritative TOML:
  - `registry/requirements.toml`
- Generated mirrors:
  - `docs/generated/REQUIREMENTS_REGISTRY_MIRROR.md`
- Notes: Requirements markdown set is generated from requirements registry + narrative overlay.

### KM-010: markdown_governance

- Strategy: `toml_primary_policy_enforcement`
- Status: `complete`
- Source markdown:
  - `docs/generated/MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md`
- Authoritative TOML:
  - `registry/markdown_governance.toml`
- Generated mirrors:
  - `docs/generated/MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md`
- Notes: Governance registry classifies all markdown lifecycle modes and drives header/parity verifiers.

### KM-006: generated_artifacts

- Strategy: `artifact_only_no_manual_edit`
- Status: `active`
- Source markdown:
  - `reports/*.md`
  - `data/artifacts/*.md`
  - `docs/claims/by_domain/*.md`
  - `docs/tickets/*_claims_audit.md`
- Notes: Keep as generated outputs with explicit regeneration commands; do not hand-edit.

### KM-007: research_narratives

- Strategy: `narrative_primary_raw_capture_backup`
- Status: `active`
- Source markdown:
  - `docs/theory/*.md`
  - `docs/external_sources/*.md`
  - `docs/engineering/*.md`
- Notes: Not all narrative content should be normalized into rigid schemas; raw capture in registry/knowledge/docs preserves provenance.

## Phases

### KMP-P1: inventory_and_capture

- Status: `complete`
- Deliverables:
  - registry/knowledge_sources.toml
  - registry/knowledge/documents.toml

### KMP-P2: curated_operational_registries

- Status: `complete`
- Deliverables:
  - registry/roadmap.toml
  - registry/todo.toml
  - registry/next_actions.toml
  - registry/requirements.toml

### KMP-P3: toml_to_markdown_mirror_exports

- Status: `complete`
- Deliverables:
  - docs/generated/*_REGISTRY_MIRROR.md

### KMP-P4: claims_support_normalization

- Status: `complete`
- Deliverables:
  - registry/claims_tasks.toml
  - registry/claims_domains.toml
  - registry/claim_tickets.toml

### KMP-P5: policy_hardening

- Status: `active`
- Deliverables:
  - Generated-header compliance verifier for TOML mirrors
  - Governance parity verifier (knowledge_sources vs markdown_governance)
  - CI checks for mirror freshness
  - Registry-first make target sequencing
  - Ticket mirror consistency verifier (registry/claim_tickets.toml <-> docs/tickets/*.md)

### KMP-P6: legacy_tracker_retirement

- Status: `pending`
- Deliverables:
  - Retire remaining manually-edited markdown trackers after TOML parity review

### KMP-P7: markdown_governance_registry

- Status: `complete`
- Deliverables:
  - src/scripts/analysis/build_markdown_governance_registry.py
  - registry/markdown_governance.toml
  - src/verification/verify_markdown_governance_headers.py
  - src/verification/verify_markdown_governance_parity.py
  - docs/generated/MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md

## Policies

### KMPOL-001: bootstrap_vs_operational_authoring

- Status: `active`
- Statement: Markdown->TOML normalization for claims-support registries is bootstrap-only. Operational authoring is TOML-first with generated markdown mirrors.
- Enforcement:
  - normalize_claims_support_registries.py requires --bootstrap-from-markdown
  - make registry excludes registry-normalize-claims from default ingest flow
  - mirror freshness and governance verifiers run in make registry
