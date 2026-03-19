-- Migration 0010: Knowledge base, planning, and research narrative tables.
-- Moves knowledge atoms, proof skeletons, derivation steps, roadmap items,
-- todo items, next actions, and research narratives into the canonical
-- SQLite source of truth.

-- ── Knowledge base ──────────────────────────────────────────────────

CREATE TABLE equation_atoms (
    id TEXT PRIMARY KEY,
    expression TEXT NOT NULL,
    normalized_expression TEXT NOT NULL DEFAULT '',
    relation_operator TEXT NOT NULL DEFAULT 'implicit',
    equation_kind TEXT NOT NULL DEFAULT '',
    extraction_confidence TEXT NOT NULL DEFAULT 'medium',
    domain_applicability TEXT NOT NULL DEFAULT '',
    source_uid TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',
    section_title TEXT NOT NULL DEFAULT '',
    assumptions_json TEXT NOT NULL DEFAULT '[]',
    parameter_sweep_json TEXT NOT NULL DEFAULT '{}',
    derivation_links_json TEXT NOT NULL DEFAULT '[]',
    depends_on_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE proof_atoms (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    proof_kind TEXT NOT NULL DEFAULT '',
    proof_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft',
    source_uid TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',
    body_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE proof_skeletons (
    id TEXT PRIMARY KEY,
    skeleton_kind TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',
    source_uid TEXT NOT NULL DEFAULT '',
    claim_id TEXT NOT NULL DEFAULT '',
    claim_refs_json TEXT NOT NULL DEFAULT '[]',
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft',
    step_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE derivation_steps (
    id TEXT PRIMARY KEY,
    skeleton_id TEXT NOT NULL DEFAULT '',
    skeleton_kind TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',
    source_uid TEXT NOT NULL DEFAULT '',
    claim_id TEXT NOT NULL DEFAULT '',
    claim_refs_json TEXT NOT NULL DEFAULT '[]',
    step_index INTEGER NOT NULL DEFAULT 0,
    step_kind TEXT NOT NULL DEFAULT 'derivation_step',
    text TEXT NOT NULL DEFAULT '',
    text_sha256 TEXT NOT NULL DEFAULT '',
    equation_refs_json TEXT NOT NULL DEFAULT '[]',
    symbol_refs_json TEXT NOT NULL DEFAULT '[]',
    numeric_constants_json TEXT NOT NULL DEFAULT '[]',
    key_tokens_json TEXT NOT NULL DEFAULT '[]',
    depends_on_step_ids_json TEXT NOT NULL DEFAULT '[]',
    line_start INTEGER NOT NULL DEFAULT 0,
    line_end INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(skeleton_id) REFERENCES proof_skeletons(id) ON DELETE CASCADE
);

CREATE INDEX idx_derivation_steps_skeleton ON derivation_steps(skeleton_id);

-- ── Planning ────────────────────────────────────────────────────────

CREATE TABLE roadmap_items (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'planned',
    status_token TEXT NOT NULL DEFAULT 'PLANNED',
    description TEXT NOT NULL DEFAULT '',
    sprint TEXT NOT NULL DEFAULT '',
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    acceptance_criteria_json TEXT NOT NULL DEFAULT '[]',
    primary_outputs_json TEXT NOT NULL DEFAULT '[]',
    evidence_refs_json TEXT NOT NULL DEFAULT '[]',
    lacunae_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE todo_items (
    id TEXT PRIMARY KEY,
    area TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'open',
    status_token TEXT NOT NULL DEFAULT 'OPEN',
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    acceptance_criteria_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE next_action_items (
    id TEXT PRIMARY KEY,
    area TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'open',
    status_token TEXT NOT NULL DEFAULT 'OPEN',
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    acceptance_criteria_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Research narratives ─────────────────────────────────────────────

CREATE TABLE research_narratives (
    id TEXT PRIMARY KEY,
    source_markdown TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT '',
    slug TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    status_token TEXT NOT NULL DEFAULT 'NARRATIVE',
    content_kind TEXT NOT NULL DEFAULT 'research_note',
    verification_level TEXT NOT NULL DEFAULT '',
    claim_refs_json TEXT NOT NULL DEFAULT '[]',
    url_refs_json TEXT NOT NULL DEFAULT '[]',
    path_refs_json TEXT NOT NULL DEFAULT '[]',
    body_markdown TEXT NOT NULL DEFAULT '',
    line_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE research_narrative_search USING fts5(
    id,
    title,
    body_markdown,
    content='research_narratives',
    content_rowid='rowid'
);

-- ── Notebook sessions (evcxr / Jupyter integration) ─────────────────

CREATE TABLE notebook_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    kernel TEXT NOT NULL DEFAULT 'evcxr',
    status TEXT NOT NULL DEFAULT 'draft',
    cell_count INTEGER NOT NULL DEFAULT 0,
    cells_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Source-of-truth metadata ────────────────────────────────────────

CREATE TABLE source_of_truth_manifest (
    table_name TEXT PRIMARY KEY,
    category TEXT NOT NULL DEFAULT '',
    authoritative INTEGER NOT NULL DEFAULT 1,
    legacy_toml_path TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    migration_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO source_of_truth_manifest (table_name, category, authoritative, legacy_toml_path, description, migration_status) VALUES
    ('claims', 'control_plane', 1, 'registry/claims.toml', 'Formal research claims with verification status', 'migrated'),
    ('insights', 'control_plane', 1, 'registry/insights.toml', 'Cross-domain research insights', 'migrated'),
    ('experiments_cp', 'control_plane', 1, 'registry/experiments.toml', 'Experiment metadata and reproducibility', 'migrated'),
    ('binaries_cp', 'control_plane', 1, 'registry/binaries.toml', 'Binary entrypoint metadata', 'migrated'),
    ('theorems', 'control_plane', 1, '', 'Formal proofs with Rocq verification', 'migrated'),
    ('artifacts', 'provenance', 1, 'registry/artifact_source_of_truth.toml', 'Dataset and resource artifacts', 'migrated'),
    ('documents', 'provenance', 1, '', 'Proof and claim document files', 'migrated'),
    ('external_source_contracts', 'external', 1, 'registry/external_sources.toml', 'External data source contracts', 'migrated'),
    ('download_jobs', 'downloads', 1, '', 'Download task tracking', 'migrated'),
    ('equation_atoms', 'knowledge', 1, 'registry/knowledge/equation_atoms_v3.toml', 'Mathematical equation atoms', 'pending'),
    ('proof_atoms', 'knowledge', 1, 'registry/knowledge/proof_atoms.toml', 'Proof structure atoms', 'pending'),
    ('proof_skeletons', 'knowledge', 1, 'registry/knowledge/proof_skeletons.toml', 'Proof outlines', 'pending'),
    ('derivation_steps', 'knowledge', 1, 'registry/knowledge/derivation_steps.toml', 'Derivation chains', 'pending'),
    ('roadmap_items', 'planning', 1, 'registry/roadmap.toml', 'Operational workstream roadmap', 'pending'),
    ('todo_items', 'planning', 1, 'registry/todo.toml', 'Task tracking', 'pending'),
    ('next_action_items', 'planning', 1, 'registry/next_actions.toml', 'Actionable next steps', 'pending'),
    ('research_narratives', 'narratives', 1, 'registry/research_narratives.toml', 'Research narrative documents', 'pending'),
    ('notebook_sessions', 'notebooks', 1, '', 'Interactive analysis notebook sessions', 'new');
