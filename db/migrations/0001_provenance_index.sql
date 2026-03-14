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
CREATE TABLE record_sources (
    entity_kind TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    PRIMARY KEY(entity_kind, entity_id, source_ref)
);
CREATE TABLE citations (
    id INTEGER PRIMARY KEY,
    artifact_id TEXT,
    citation_text TEXT NOT NULL,
    doi TEXT,
    canonical_url TEXT,
    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
);
CREATE TABLE links (
    url TEXT PRIMARY KEY,
    host TEXT
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
CREATE TABLE mirror_observations (
    artifact_id TEXT NOT NULL,
    url TEXT NOT NULL,
    mirror_kind TEXT NOT NULL,
    PRIMARY KEY(artifact_id, url, mirror_kind),
    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE,
    FOREIGN KEY(url) REFERENCES links(url) ON DELETE CASCADE
);
CREATE TABLE lane_assignments (
    artifact_id TEXT NOT NULL,
    lane_name TEXT NOT NULL,
    PRIMARY KEY(artifact_id, lane_name),
    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
);
CREATE TABLE export_runs (
    id INTEGER PRIMARY KEY,
    action TEXT NOT NULL,
    created_at TEXT NOT NULL,
    artifact_count INTEGER NOT NULL,
    document_count INTEGER NOT NULL,
    details_json TEXT NOT NULL
);
CREATE TABLE ingest_fingerprints (
    path TEXT PRIMARY KEY,
    blake3_hex TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE document_search USING fts5(document_id, path, title, kind, content='');
