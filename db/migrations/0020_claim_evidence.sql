-- Typed evidence contracts retain complete prior declarations and actor intent.
CREATE TABLE claim_evidence (
    claim_id TEXT PRIMARY KEY REFERENCES claims(id),
    spec_json TEXT NOT NULL CHECK(json_valid(spec_json))
);
CREATE TABLE claim_evidence_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    previous_spec_json TEXT CHECK(previous_spec_json IS NULL OR json_valid(previous_spec_json)),
    new_spec_json TEXT NOT NULL CHECK(json_valid(new_spec_json)),
    actor TEXT NOT NULL CHECK(length(trim(actor)) > 0),
    reason TEXT NOT NULL CHECK(length(trim(reason)) > 0),
    changed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TRIGGER claim_evidence_revisions_no_update BEFORE UPDATE ON claim_evidence_revisions
BEGIN SELECT RAISE(ABORT, 'claim evidence history is append-only'); END;
CREATE TRIGGER claim_evidence_revisions_no_delete BEFORE DELETE ON claim_evidence_revisions
BEGIN SELECT RAISE(ABORT, 'claim evidence history is append-only'); END;
CREATE TABLE claim_evidence_revision_experiments (
    revision_id INTEGER NOT NULL REFERENCES claim_evidence_revisions(id),
    experiment_id TEXT NOT NULL REFERENCES experiments_cp(id),
    PRIMARY KEY (revision_id, experiment_id)
);
CREATE TRIGGER claim_evidence_revision_experiments_no_update BEFORE UPDATE ON claim_evidence_revision_experiments
BEGIN SELECT RAISE(ABORT, 'claim evidence experiment history is append-only'); END;
CREATE TRIGGER claim_evidence_revision_experiments_no_delete BEFORE DELETE ON claim_evidence_revision_experiments
BEGIN SELECT RAISE(ABORT, 'claim evidence experiment history is append-only'); END;
