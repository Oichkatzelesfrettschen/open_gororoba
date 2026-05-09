-- 0015_revisions_audit: append-only audit tables for claim, insight, and
-- experiment status_note mutations.
--
-- Why: registry/canonical/control_plane.sqlite3 is the canonical write
-- target for all registry rows. Without an audit table, edits are
-- replayable only via git log of the regenerated TOML. An explicit
-- revision row per mutation gives us:
--   - prev/new content hashes that detect tampering
--   - actor + reason fields for human accountability
--   - chronological history queryable via `gororoba-db claim history`
--
-- All three tables share the same schema modulo the entity foreign-key
-- column. They are append-only (no UPDATE, no DELETE).

CREATE TABLE claim_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    prev_value_sha256 TEXT,
    new_value_sha256 TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    ts_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    operation TEXT NOT NULL CHECK (operation IN ('update', 'touch', 'create', 'delete')),
    application_id INTEGER,
    FOREIGN KEY (claim_id) REFERENCES claims(id) DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX claim_revisions_by_claim ON claim_revisions(claim_id, ts_utc);
CREATE INDEX claim_revisions_by_actor ON claim_revisions(actor, ts_utc);

CREATE TABLE insight_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    insight_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    prev_value_sha256 TEXT,
    new_value_sha256 TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    ts_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    operation TEXT NOT NULL CHECK (operation IN ('update', 'touch', 'create', 'delete')),
    application_id INTEGER,
    FOREIGN KEY (insight_id) REFERENCES insights(id) DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX insight_revisions_by_insight ON insight_revisions(insight_id, ts_utc);
CREATE INDEX insight_revisions_by_actor ON insight_revisions(actor, ts_utc);

CREATE TABLE experiment_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    prev_value_sha256 TEXT,
    new_value_sha256 TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    ts_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    operation TEXT NOT NULL CHECK (operation IN ('update', 'touch', 'create', 'delete')),
    application_id INTEGER,
    -- experiments_cp is the compat-export table; the FK targets that.
    FOREIGN KEY (experiment_id) REFERENCES experiments_cp(id) DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX experiment_revisions_by_experiment ON experiment_revisions(experiment_id, ts_utc);
CREATE INDEX experiment_revisions_by_actor ON experiment_revisions(actor, ts_utc);
