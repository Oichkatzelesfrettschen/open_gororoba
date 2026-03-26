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

INSERT INTO source_of_truth_manifest (
    table_name,
    category,
    authoritative,
    legacy_toml_path,
    description,
    migration_status
) VALUES
    (
        'literature_verification_runs',
        'research_ops',
        1,
        '',
        'Literature verification and novelty runs backed by lit_search',
        'new'
    );
