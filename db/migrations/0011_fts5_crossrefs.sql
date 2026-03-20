-- Migration 0011: FTS5 full-text search indexes and crossref join tables
-- for the three-layer registry architecture (TOML source -> SQLite build -> CLI query).

-- FTS5 index on claims for full-text search on statement text.
CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
    id,
    statement,
    status,
    content='claims',
    content_rowid='ROWID'
);

CREATE TRIGGER IF NOT EXISTS claims_fts_ai
AFTER INSERT ON claims BEGIN
    INSERT INTO claims_fts(rowid, id, statement, status)
    VALUES (new.rowid, new.id, new.statement, new.status);
END;

CREATE TRIGGER IF NOT EXISTS claims_fts_ad
AFTER DELETE ON claims BEGIN
    DELETE FROM claims_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS claims_fts_au
AFTER UPDATE ON claims BEGIN
    DELETE FROM claims_fts WHERE rowid = old.rowid;
    INSERT INTO claims_fts(rowid, id, statement, status)
    VALUES (new.rowid, new.id, new.statement, new.status);
END;

INSERT INTO claims_fts(claims_fts) VALUES('rebuild');

-- FTS5 index on insights for full-text search.
CREATE VIRTUAL TABLE IF NOT EXISTS insights_fts USING fts5(
    id,
    title,
    status,
    content='insights',
    content_rowid='ROWID'
);

CREATE TRIGGER IF NOT EXISTS insights_fts_ai
AFTER INSERT ON insights BEGIN
    INSERT INTO insights_fts(rowid, id, title, status)
    VALUES (new.rowid, new.id, new.title, new.status);
END;

CREATE TRIGGER IF NOT EXISTS insights_fts_ad
AFTER DELETE ON insights BEGIN
    DELETE FROM insights_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS insights_fts_au
AFTER UPDATE ON insights BEGIN
    DELETE FROM insights_fts WHERE rowid = old.rowid;
    INSERT INTO insights_fts(rowid, id, title, status)
    VALUES (new.rowid, new.id, new.title, new.status);
END;

INSERT INTO insights_fts(insights_fts) VALUES('rebuild');

-- Crossref join tables: claim <-> experiment references.
CREATE TABLE IF NOT EXISTS claim_experiment_refs (
    claim_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    PRIMARY KEY (claim_id, experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_cer_experiment ON claim_experiment_refs(experiment_id);

-- Crossref join tables: claim <-> insight references.
CREATE TABLE IF NOT EXISTS claim_insight_refs (
    claim_id TEXT NOT NULL,
    insight_id TEXT NOT NULL,
    PRIMARY KEY (claim_id, insight_id)
);

CREATE INDEX IF NOT EXISTS idx_cir_insight ON claim_insight_refs(insight_id);

-- Evidence edges table (populated from claims_evidence_edges.toml).
CREATE TABLE IF NOT EXISTS evidence_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL DEFAULT 'supports',
    weight REAL NOT NULL DEFAULT 1.0,
    notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_ee_source ON evidence_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_ee_target ON evidence_edges(target_id);

-- Bibliography table (populated from bibliography.toml).
CREATE TABLE IF NOT EXISTS bibliography (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    authors TEXT NOT NULL DEFAULT '',
    year TEXT NOT NULL DEFAULT '',
    doi TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    bibtex_type TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]'
);

CREATE VIRTUAL TABLE IF NOT EXISTS bibliography_fts USING fts5(
    id,
    title,
    authors,
    content='bibliography',
    content_rowid='ROWID'
);

CREATE TRIGGER IF NOT EXISTS bibliography_fts_ai
AFTER INSERT ON bibliography BEGIN
    INSERT INTO bibliography_fts(rowid, id, title, authors)
    VALUES (new.rowid, new.id, new.title, new.authors);
END;

CREATE TRIGGER IF NOT EXISTS bibliography_fts_ad
AFTER DELETE ON bibliography BEGIN
    DELETE FROM bibliography_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS bibliography_fts_au
AFTER UPDATE ON bibliography BEGIN
    DELETE FROM bibliography_fts WHERE rowid = old.rowid;
    INSERT INTO bibliography_fts(rowid, id, title, authors)
    VALUES (new.rowid, new.id, new.title, new.authors);
END;

INSERT INTO bibliography_fts(bibliography_fts) VALUES('rebuild');

-- Lacunae table (open questions and research gaps).
CREATE TABLE IF NOT EXISTS lacunae (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    domain TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    claim_refs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Build metadata: tracks when the DB was last built and from which sources.
CREATE TABLE IF NOT EXISTS build_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
