CREATE TABLE artifact_retrieval_observations (
    observation_key TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    artifact_key TEXT NOT NULL,
    original_url TEXT NOT NULL,
    requested_url TEXT NOT NULL,
    final_url TEXT NOT NULL,
    expected_sha256 TEXT NOT NULL,
    expected_bytes INTEGER NOT NULL CHECK(expected_bytes >= 0),
    response_path TEXT,
    observed_sha256 TEXT,
    observed_bytes INTEGER NOT NULL CHECK(observed_bytes >= 0),
    completed INTEGER NOT NULL CHECK(completed IN (0,1)),
    http_status INTEGER NOT NULL,
    digest_matches INTEGER NOT NULL CHECK(digest_matches IN (0,1)),
    canonical_url_corrected INTEGER NOT NULL CHECK(canonical_url_corrected IN (0,1)),
    document_identity TEXT NOT NULL CHECK(document_identity IN ('verified','unresolved')),
    recorded_at TEXT NOT NULL,
    spec_sha256 TEXT NOT NULL,
    report_json TEXT NOT NULL,
    CHECK(canonical_url_corrected = 0 OR (digest_matches = 1 AND completed = 1)),
    CHECK(digest_matches = 0 OR (completed = 1 AND http_status BETWEEN 200 AND 299
        AND expected_sha256 = observed_sha256 AND expected_bytes = observed_bytes
        AND response_path IS NOT NULL))
);
CREATE INDEX artifact_retrieval_by_identity ON artifact_retrieval_observations(artifact_key, digest_matches);
CREATE TRIGGER artifact_retrieval_append_only_update BEFORE UPDATE ON artifact_retrieval_observations
BEGIN SELECT RAISE(ABORT, 'artifact retrieval observations are append-only'); END;
CREATE TRIGGER artifact_retrieval_append_only_delete BEFORE DELETE ON artifact_retrieval_observations
BEGIN SELECT RAISE(ABORT, 'artifact retrieval observations are append-only'); END;
