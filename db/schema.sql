-- GENERATED FILE. DO NOT EDIT.
-- Canonical source: db/migrations/*.sql
-- Regenerate with: cargo run -p xtask -- db-docs

CREATE TABLE artifact_links (
    artifact_id TEXT NOT NULL,
    url TEXT NOT NULL,
    relation TEXT NOT NULL,
    PRIMARY KEY(artifact_id, url, relation),
    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE,
    FOREIGN KEY(url) REFERENCES links(url) ON DELETE CASCADE
);

CREATE TABLE artifact_paths (
    artifact_id TEXT NOT NULL,
    path TEXT NOT NULL,
    relation TEXT NOT NULL,
    PRIMARY KEY(artifact_id, path, relation),
    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
);

CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    citation TEXT NOT NULL,
    status TEXT NOT NULL,
    minimum_requirement_met INTEGER NOT NULL,
    canonical_functional_url TEXT,
    canonical_download_path TEXT
);

CREATE TRIGGER bibliography_fts_ad
AFTER DELETE ON bibliography BEGIN
    DELETE FROM bibliography_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER bibliography_fts_ai
AFTER INSERT ON bibliography BEGIN
    INSERT INTO bibliography_fts(rowid, id, title, authors)
    VALUES (new.rowid, new.id, new.title, new.authors);
END;

CREATE TRIGGER bibliography_fts_au
AFTER UPDATE ON bibliography BEGIN
    DELETE FROM bibliography_fts WHERE rowid = old.rowid;
    INSERT INTO bibliography_fts(rowid, id, title, authors)
    VALUES (new.rowid, new.id, new.title, new.authors);
END;

CREATE TABLE bibliography (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    authors TEXT NOT NULL DEFAULT '',
    year TEXT NOT NULL DEFAULT '',
    doi TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    bibtex_type TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]'
);

CREATE VIRTUAL TABLE bibliography_fts USING fts5(
    id,
    title,
    authors,
    content='bibliography',
    content_rowid='ROWID'
);

CREATE TABLE binaries_cp (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    experiment_id TEXT
, crate_name TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT 'registry');

CREATE TABLE build_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE citations (
    id INTEGER PRIMARY KEY,
    artifact_id TEXT,
    citation_text TEXT NOT NULL,
    doi TEXT,
    canonical_url TEXT,
    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
);

CREATE TABLE claim_experiment_refs (
    claim_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    PRIMARY KEY (claim_id, experiment_id)
);

CREATE INDEX idx_cer_experiment ON claim_experiment_refs(experiment_id);

CREATE TABLE claim_insight_refs (
    claim_id TEXT NOT NULL,
    insight_id TEXT NOT NULL,
    PRIMARY KEY (claim_id, insight_id)
);

CREATE INDEX idx_cir_insight ON claim_insight_refs(insight_id);

CREATE TRIGGER claims_fts_ad
AFTER DELETE ON claims BEGIN
    DELETE FROM claims_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER claims_fts_ai
AFTER INSERT ON claims BEGIN
    INSERT INTO claims_fts(rowid, id, statement, status)
    VALUES (new.rowid, new.id, new.statement, new.status);
END;

CREATE TRIGGER claims_fts_au
AFTER UPDATE ON claims BEGIN
    DELETE FROM claims_fts WHERE rowid = old.rowid;
    INSERT INTO claims_fts(rowid, id, statement, status)
    VALUES (new.rowid, new.id, new.statement, new.status);
END;

CREATE TABLE claims (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    where_stated TEXT NOT NULL,
    last_verified TEXT NOT NULL,
    formal_proof TEXT,
    status_note TEXT
, compat_toml_text TEXT NOT NULL DEFAULT '');

CREATE VIRTUAL TABLE claims_fts USING fts5(
    id,
    statement,
    status,
    content='claims',
    content_rowid='ROWID'
);

CREATE TABLE control_plane_meta (
    kind TEXT PRIMARY KEY,
    compat_toml_text TEXT NOT NULL
);

CREATE TABLE control_plane_runs (
    id INTEGER PRIMARY KEY,
    action TEXT NOT NULL,
    created_at TEXT NOT NULL,
    details_json TEXT NOT NULL
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

CREATE VIRTUAL TABLE document_search USING fts5(document_id, path, title, kind, content='');

CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    authoring_mode TEXT NOT NULL,
    generated INTEGER NOT NULL,
    status TEXT NOT NULL,
    toml_backing TEXT,
    sha256 TEXT,
    size_bytes INTEGER,
    line_count INTEGER
);

CREATE TABLE download_attempts (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL,
    backend TEXT NOT NULL,
    http_code INTEGER,
    content_type TEXT,
    bytes INTEGER NOT NULL,
    sha256 TEXT,
    is_pdf INTEGER NOT NULL,
    final_url TEXT,
    note TEXT NOT NULL,
    recorded_at TEXT NOT NULL, succeeded INTEGER NOT NULL DEFAULT 1, error_message TEXT, failure_class TEXT,
    FOREIGN KEY(job_id) REFERENCES download_jobs(id) ON DELETE CASCADE
);

CREATE INDEX idx_download_attempts_job_id ON download_attempts(job_id);

CREATE TABLE download_campaign_jobs (
    campaign_id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    PRIMARY KEY(campaign_id, job_id),
    FOREIGN KEY(campaign_id) REFERENCES download_campaigns(id) ON DELETE CASCADE,
    FOREIGN KEY(job_id) REFERENCES download_jobs(id) ON DELETE CASCADE
);

CREATE INDEX idx_download_campaign_jobs_job_id ON download_campaign_jobs(job_id);

CREATE TABLE download_campaigns (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    command_kind TEXT NOT NULL,
    input_path TEXT NOT NULL,
    out_ledger_path TEXT,
    dest_dir TEXT,
    note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE download_jobs (
    id INTEGER PRIMARY KEY,
    requested_url TEXT NOT NULL,
    transfer_kind TEXT NOT NULL,
    requested_backend TEXT NOT NULL,
    route_scheme TEXT NOT NULL,
    route_host TEXT,
    route_backends_json TEXT NOT NULL,
    note TEXT,
    status TEXT NOT NULL,
    final_url TEXT,
    output_path TEXT,
    created_at TEXT NOT NULL
);

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

CREATE TABLE evidence_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL DEFAULT 'supports',
    weight REAL NOT NULL DEFAULT 1.0,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX idx_ee_source ON evidence_edges(source_id);

CREATE INDEX idx_ee_target ON evidence_edges(target_id);

CREATE TABLE experiments_cp (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    binary_name TEXT,
    claim_refs_json TEXT NOT NULL
, compat_toml_text TEXT NOT NULL DEFAULT '');

CREATE TABLE export_runs (
    id INTEGER PRIMARY KEY,
    action TEXT NOT NULL,
    created_at TEXT NOT NULL,
    artifact_count INTEGER NOT NULL,
    document_count INTEGER NOT NULL,
    details_json TEXT NOT NULL
);

CREATE TABLE external_source_contract_values (
    contract_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    ord INTEGER NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(contract_id, relation, ord),
    FOREIGN KEY(contract_id) REFERENCES external_source_contracts(id) ON DELETE CASCADE
);

CREATE TABLE external_source_contracts (
    id TEXT PRIMARY KEY,
    path_glob TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    access_class TEXT NOT NULL,
    status TEXT NOT NULL,
    retrieval_method TEXT NOT NULL,
    attempt_deadline_utc TEXT NOT NULL,
    resolution_deadline_utc TEXT NOT NULL,
    blocker_note TEXT NOT NULL
);

CREATE TABLE external_source_contracts_meta (
    kind TEXT PRIMARY KEY,
    updated TEXT NOT NULL,
    authoritative INTEGER NOT NULL,
    policy_version TEXT NOT NULL
);

CREATE TABLE external_source_dossier_values (
    dossier_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    ord INTEGER NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(dossier_id, relation, ord),
    FOREIGN KEY(dossier_id) REFERENCES external_source_dossiers(id) ON DELETE CASCADE
);

CREATE TABLE external_source_dossiers (
    id TEXT PRIMARY KEY,
    source_markdown TEXT NOT NULL,
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    status_token TEXT NOT NULL,
    content_kind TEXT NOT NULL,
    authority_level TEXT NOT NULL,
    verification_level TEXT NOT NULL,
    operational_role TEXT NOT NULL,
    source_lineage_summary TEXT NOT NULL,
    has_full_transcript INTEGER NOT NULL,
    line_count INTEGER NOT NULL,
    notes TEXT NOT NULL,
    body_markdown TEXT NOT NULL
);

CREATE TABLE external_source_dossiers_meta (
    kind TEXT PRIMARY KEY,
    updated TEXT NOT NULL,
    authoritative INTEGER NOT NULL,
    source_markdown_glob TEXT NOT NULL,
    document_count INTEGER NOT NULL
);

CREATE TABLE ingest_fingerprints (
    path TEXT PRIMARY KEY,
    blake3_hex TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE TRIGGER insights_fts_ad
AFTER DELETE ON insights BEGIN
    DELETE FROM insights_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER insights_fts_ai
AFTER INSERT ON insights BEGIN
    INSERT INTO insights_fts(rowid, id, title, status)
    VALUES (new.rowid, new.id, new.title, new.status);
END;

CREATE TRIGGER insights_fts_au
AFTER UPDATE ON insights BEGIN
    DELETE FROM insights_fts WHERE rowid = old.rowid;
    INSERT INTO insights_fts(rowid, id, title, status)
    VALUES (new.rowid, new.id, new.title, new.status);
END;

CREATE TABLE insights (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    claim_refs_json TEXT NOT NULL
, compat_toml_text TEXT NOT NULL DEFAULT '');

CREATE VIRTUAL TABLE insights_fts USING fts5(
    id,
    title,
    status,
    content='insights',
    content_rowid='ROWID'
);

CREATE TABLE lacunae (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    domain TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    claim_refs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE lane_assignments (
    artifact_id TEXT NOT NULL,
    lane_name TEXT NOT NULL,
    PRIMARY KEY(artifact_id, lane_name),
    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
);

CREATE TABLE links (
    url TEXT PRIMARY KEY,
    host TEXT
);

CREATE TABLE literature_novelty_similar_papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    year INTEGER NOT NULL,
    venue TEXT NOT NULL,
    citation_count INTEGER NOT NULL,
    similarity REAL NOT NULL,
    url TEXT NOT NULL,
    cite_key TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES literature_verification_runs(id) ON DELETE CASCADE
);

CREATE INDEX idx_literature_novelty_similar_papers_run_id
    ON literature_novelty_similar_papers(run_id);

CREATE TABLE literature_verification_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    cite_key TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    method TEXT NOT NULL,
    details TEXT NOT NULL,
    doi TEXT,
    arxiv_id TEXT,
    matched_paper_title TEXT,
    matched_paper_source TEXT,
    matched_paper_year INTEGER,
    matched_paper_url TEXT,
    relevance_score REAL,
    FOREIGN KEY(run_id) REFERENCES literature_verification_runs(id) ON DELETE CASCADE
);

CREATE INDEX idx_literature_verification_results_run_id
    ON literature_verification_results(run_id);

CREATE INDEX idx_literature_verification_results_status
    ON literature_verification_results(status);

CREATE TABLE literature_verification_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_path TEXT NOT NULL,
    topic TEXT,
    hypotheses_path TEXT,
    domains_json TEXT NOT NULL DEFAULT '[]',
    search_queries_json TEXT NOT NULL DEFAULT '[]',
    total_entries INTEGER NOT NULL,
    verified_count INTEGER NOT NULL,
    suspicious_count INTEGER NOT NULL,
    hallucinated_count INTEGER NOT NULL,
    skipped_count INTEGER NOT NULL,
    integrity_score REAL NOT NULL,
    novelty_score REAL,
    novelty_assessment TEXT,
    recommendation TEXT,
    search_coverage TEXT,
    total_papers_retrieved INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE mirror_observations (
    artifact_id TEXT NOT NULL,
    url TEXT NOT NULL,
    mirror_kind TEXT NOT NULL,
    PRIMARY KEY(artifact_id, url, mirror_kind),
    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE,
    FOREIGN KEY(url) REFERENCES links(url) ON DELETE CASCADE
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
    evidence_refs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

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

CREATE TABLE record_sources (
    entity_kind TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    PRIMARY KEY(entity_kind, entity_id, source_ref)
);

CREATE TABLE registry_snapshots (
    registry_kind TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    content_text TEXT NOT NULL,
    blake3_hex TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE TABLE requirements_coverage_gaps (
    id TEXT PRIMARY KEY,
    area TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    status_token TEXT NOT NULL DEFAULT 'OPEN',
    description TEXT NOT NULL DEFAULT '',
    proposed_resolution TEXT NOT NULL DEFAULT '',
    related_module_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE requirements_modules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    markdown TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    status_token TEXT NOT NULL DEFAULT 'ACTIVE',
    runtime_stack TEXT NOT NULL DEFAULT 'mixed',
    requires_modules_json TEXT NOT NULL DEFAULT '[]',
    install_targets_json TEXT NOT NULL DEFAULT '[]',
    verify_targets_json TEXT NOT NULL DEFAULT '[]',
    acceptance_criteria_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE requirements_registry_meta (
    kind TEXT PRIMARY KEY,
    authoritative INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    status_token TEXT NOT NULL DEFAULT 'ACTIVE',
    updated TEXT NOT NULL DEFAULT '',
    python_recommended TEXT NOT NULL DEFAULT '',
    python_allowed TEXT NOT NULL DEFAULT '',
    primary_markdown TEXT NOT NULL DEFAULT '',
    status_allowlist_json TEXT NOT NULL DEFAULT '[]',
    runtime_stack_allowlist_json TEXT NOT NULL DEFAULT '[]',
    required_module_fields_json TEXT NOT NULL DEFAULT '[]',
    required_gap_fields_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE research_narrative_search USING fts5(
    id,
    title,
    body_markdown,
    content='research_narratives',
    content_rowid='ROWID'
);

CREATE TRIGGER research_narratives_ad
AFTER DELETE ON research_narratives BEGIN
    DELETE FROM research_narrative_search WHERE rowid = old.rowid;
END;

CREATE TRIGGER research_narratives_ai
AFTER INSERT ON research_narratives BEGIN
    INSERT INTO research_narrative_search(rowid, id, title, body_markdown)
    VALUES (new.rowid, new.id, new.title, new.body_markdown);
END;

CREATE TRIGGER research_narratives_au
AFTER UPDATE ON research_narratives BEGIN
    DELETE FROM research_narrative_search WHERE rowid = old.rowid;
    INSERT INTO research_narrative_search(rowid, id, title, body_markdown)
    VALUES (new.rowid, new.id, new.title, new.body_markdown);
END;

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
, claims_json TEXT NOT NULL DEFAULT '[]', insight TEXT NOT NULL DEFAULT '');

CREATE TABLE source_of_truth_manifest (
    table_name TEXT PRIMARY KEY,
    category TEXT NOT NULL DEFAULT '',
    authoritative INTEGER NOT NULL DEFAULT 1,
    legacy_toml_path TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    migration_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE theorems (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    proof_path TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    linked_claim_ids_json TEXT NOT NULL,
    source TEXT NOT NULL
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
    evidence_refs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
