CREATE TABLE source_observations (
    observation_key TEXT PRIMARY KEY,
    source_key TEXT NOT NULL,
    artifact_id TEXT REFERENCES artifacts(id),
    spec_sha256 TEXT NOT NULL,
    admitted_at TEXT NOT NULL,
    report_json TEXT NOT NULL CHECK(json_valid(report_json))
);
CREATE TRIGGER source_observations_no_update BEFORE UPDATE ON source_observations BEGIN SELECT RAISE(ABORT, 'source observations are append-only'); END;
CREATE TRIGGER source_observations_no_delete BEFORE DELETE ON source_observations BEGIN SELECT RAISE(ABORT, 'source observations are append-only'); END;
