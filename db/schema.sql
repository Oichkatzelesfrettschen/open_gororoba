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

CREATE TABLE binaries_cp (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    experiment_id TEXT
, crate_name TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT 'registry');

CREATE TABLE citations (
    id INTEGER PRIMARY KEY,
    artifact_id TEXT,
    citation_text TEXT NOT NULL,
    doi TEXT,
    canonical_url TEXT,
    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
);

CREATE TABLE claims (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    where_stated TEXT NOT NULL,
    last_verified TEXT NOT NULL,
    formal_proof TEXT,
    status_note TEXT
, compat_toml_text TEXT NOT NULL DEFAULT '');

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

CREATE TABLE insights (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    claim_refs_json TEXT NOT NULL
, compat_toml_text TEXT NOT NULL DEFAULT '');

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

CREATE TABLE mirror_observations (
    artifact_id TEXT NOT NULL,
    url TEXT NOT NULL,
    mirror_kind TEXT NOT NULL,
    PRIMARY KEY(artifact_id, url, mirror_kind),
    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE,
    FOREIGN KEY(url) REFERENCES links(url) ON DELETE CASCADE
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

CREATE TABLE theorems (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    proof_path TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    linked_claim_ids_json TEXT NOT NULL,
    source TEXT NOT NULL
);
