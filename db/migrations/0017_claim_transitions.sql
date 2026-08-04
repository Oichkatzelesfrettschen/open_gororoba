-- 0017_claim_transitions: typed, append-only claim transition history.
--
-- Experiment verdicts remain separate from canonical claim statuses. The
-- normalized child tables retain evidence, assumptions, successor claims,
-- and typed relations without packing transition history into status_note.

CREATE TABLE claim_transition_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transition_key TEXT NOT NULL UNIQUE,
    source_claim_id TEXT NOT NULL,
    expected_prior_status TEXT NOT NULL,
    experiment_verdict TEXT NOT NULL CHECK (
        experiment_verdict IN (
            'Falsifies',
            'MethodologyInvalid',
            'Inconclusive',
            'SurvivesChallenge',
            'Replicates'
        )
    ),
    proposed_claim_status TEXT NOT NULL,
    exercised_falsifier TEXT NOT NULL,
    rationale TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    transition_ts_utc TEXT NOT NULL,
    transition_spec_sha256 TEXT NOT NULL,
    expected_source_state_sha256 TEXT NOT NULL,
    expected_claim_id_max INTEGER NOT NULL,
    FOREIGN KEY (source_claim_id) REFERENCES claims(id)
);

CREATE INDEX claim_transition_events_by_source
    ON claim_transition_events(source_claim_id, transition_ts_utc);

CREATE TABLE claim_transition_evidence (
    transition_event_id INTEGER NOT NULL,
    artifact_id TEXT NOT NULL,
    PRIMARY KEY (transition_event_id, artifact_id),
    FOREIGN KEY (transition_event_id) REFERENCES claim_transition_events(id),
    FOREIGN KEY (artifact_id) REFERENCES artifacts(id)
);

CREATE TABLE claim_transition_experiments (
    transition_event_id INTEGER NOT NULL,
    experiment_id TEXT NOT NULL,
    PRIMARY KEY (transition_event_id, experiment_id),
    FOREIGN KEY (transition_event_id) REFERENCES claim_transition_events(id),
    FOREIGN KEY (experiment_id) REFERENCES experiments_cp(id)
);

CREATE TABLE claim_transition_assumptions (
    transition_event_id INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    assumption TEXT NOT NULL,
    PRIMARY KEY (transition_event_id, ordinal),
    UNIQUE (transition_event_id, assumption),
    FOREIGN KEY (transition_event_id) REFERENCES claim_transition_events(id)
);

CREATE TABLE claim_transition_successors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transition_event_id INTEGER NOT NULL,
    proposal_key TEXT NOT NULL,
    successor_claim_id TEXT NOT NULL UNIQUE,
    statement TEXT NOT NULL,
    initial_status TEXT NOT NULL,
    source_or_implementation_boundary TEXT NOT NULL,
    required_falsifier TEXT NOT NULL,
    predecessor_relation_kind TEXT NOT NULL CHECK (
        predecessor_relation_kind IN (
            'source_split',
            'implementation_split',
            'narrows',
            'refines',
            'supersedes'
        )
    ),
    FOREIGN KEY (transition_event_id) REFERENCES claim_transition_events(id),
    FOREIGN KEY (successor_claim_id) REFERENCES claims(id),
    UNIQUE (transition_event_id, proposal_key),
    UNIQUE (transition_event_id, statement)
);

CREATE TABLE claim_transition_successor_where_stated (
    successor_id INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    reference TEXT NOT NULL,
    PRIMARY KEY (successor_id, ordinal),
    UNIQUE (successor_id, reference),
    FOREIGN KEY (successor_id) REFERENCES claim_transition_successors(id)
);

CREATE TABLE claim_transition_successor_evidence (
    successor_id INTEGER NOT NULL,
    artifact_id TEXT NOT NULL,
    PRIMARY KEY (successor_id, artifact_id),
    FOREIGN KEY (successor_id) REFERENCES claim_transition_successors(id),
    FOREIGN KEY (artifact_id) REFERENCES artifacts(id)
);

CREATE TABLE claim_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predecessor_claim_id TEXT NOT NULL,
    successor_claim_id TEXT NOT NULL,
    relation_kind TEXT NOT NULL CHECK (
        relation_kind IN (
            'source_split',
            'implementation_split',
            'narrows',
            'refines',
            'supersedes'
        )
    ),
    transition_event_id INTEGER NOT NULL,
    FOREIGN KEY (predecessor_claim_id) REFERENCES claims(id),
    FOREIGN KEY (successor_claim_id) REFERENCES claims(id),
    FOREIGN KEY (transition_event_id) REFERENCES claim_transition_events(id),
    UNIQUE (predecessor_claim_id, successor_claim_id, relation_kind)
);

CREATE INDEX claim_relations_by_predecessor
    ON claim_relations(predecessor_claim_id, relation_kind);

CREATE INDEX claim_relations_by_successor
    ON claim_relations(successor_claim_id, relation_kind);

CREATE TABLE claim_status_write_context (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    mode TEXT NOT NULL CHECK (mode IN ('registry_reindex', 'transition_apply')),
    transition_event_id INTEGER,
    source_claim_id TEXT,
    proposed_status TEXT,
    FOREIGN KEY (transition_event_id) REFERENCES claim_transition_events(id),
    FOREIGN KEY (source_claim_id) REFERENCES claims(id),
    CHECK (
        (mode = 'registry_reindex'
            AND transition_event_id IS NULL
            AND source_claim_id IS NULL
            AND proposed_status IS NULL)
        OR
        (mode = 'transition_apply'
            AND transition_event_id IS NOT NULL
            AND source_claim_id IS NOT NULL
            AND proposed_status IS NOT NULL)
    )
);

CREATE TRIGGER claim_transition_events_append_only_update
BEFORE UPDATE ON claim_transition_events
BEGIN
    SELECT RAISE(ABORT, 'claim transition events are append-only');
END;

CREATE TRIGGER claim_transition_events_append_only_delete
BEFORE DELETE ON claim_transition_events
BEGIN
    SELECT RAISE(ABORT, 'claim transition events are append-only');
END;

CREATE TRIGGER claim_transition_evidence_append_only_update
BEFORE UPDATE ON claim_transition_evidence
BEGIN
    SELECT RAISE(ABORT, 'claim transition evidence is append-only');
END;

CREATE TRIGGER claim_transition_evidence_append_only_delete
BEFORE DELETE ON claim_transition_evidence
BEGIN
    SELECT RAISE(ABORT, 'claim transition evidence is append-only');
END;

CREATE TRIGGER claim_transition_experiments_append_only_update
BEFORE UPDATE ON claim_transition_experiments
BEGIN
    SELECT RAISE(ABORT, 'claim transition experiments are append-only');
END;

CREATE TRIGGER claim_transition_experiments_append_only_delete
BEFORE DELETE ON claim_transition_experiments
BEGIN
    SELECT RAISE(ABORT, 'claim transition experiments are append-only');
END;

CREATE TRIGGER claim_transition_assumptions_append_only_update
BEFORE UPDATE ON claim_transition_assumptions
BEGIN
    SELECT RAISE(ABORT, 'claim transition assumptions are append-only');
END;

CREATE TRIGGER claim_transition_assumptions_append_only_delete
BEFORE DELETE ON claim_transition_assumptions
BEGIN
    SELECT RAISE(ABORT, 'claim transition assumptions are append-only');
END;

CREATE TRIGGER claim_transition_successors_append_only_update
BEFORE UPDATE ON claim_transition_successors
BEGIN
    SELECT RAISE(ABORT, 'claim transition successors are append-only');
END;

CREATE TRIGGER claim_transition_successors_append_only_delete
BEFORE DELETE ON claim_transition_successors
BEGIN
    SELECT RAISE(ABORT, 'claim transition successors are append-only');
END;

CREATE TRIGGER claim_transition_successor_where_stated_append_only_update
BEFORE UPDATE ON claim_transition_successor_where_stated
BEGIN
    SELECT RAISE(ABORT, 'claim transition where-stated references are append-only');
END;

CREATE TRIGGER claim_transition_successor_where_stated_append_only_delete
BEFORE DELETE ON claim_transition_successor_where_stated
BEGIN
    SELECT RAISE(ABORT, 'claim transition where-stated references are append-only');
END;

CREATE TRIGGER claim_transition_successor_evidence_append_only_update
BEFORE UPDATE ON claim_transition_successor_evidence
BEGIN
    SELECT RAISE(ABORT, 'claim transition successor evidence is append-only');
END;

CREATE TRIGGER claim_transition_successor_evidence_append_only_delete
BEFORE DELETE ON claim_transition_successor_evidence
BEGIN
    SELECT RAISE(ABORT, 'claim transition successor evidence is append-only');
END;

CREATE TRIGGER claim_relations_append_only_update
BEFORE UPDATE ON claim_relations
BEGIN
    SELECT RAISE(ABORT, 'claim relations are append-only');
END;

CREATE TRIGGER claim_relations_append_only_delete
BEFORE DELETE ON claim_relations
BEGIN
    SELECT RAISE(ABORT, 'claim relations are append-only');
END;

CREATE TRIGGER claims_status_requires_event
BEFORE UPDATE OF status ON claims
WHEN OLD.status IS NOT NEW.status
    AND NOT EXISTS (
        SELECT 1 FROM claim_status_write_context
        WHERE id = 1 AND mode = 'registry_reindex'
    )
    AND NOT EXISTS (
        SELECT 1
        FROM claim_status_write_context
        WHERE id = 1
          AND mode = 'transition_apply'
          AND source_claim_id = OLD.id
          AND proposed_status = NEW.status
    )
BEGIN
    SELECT RAISE(ABORT, 'claim status changes require a transition event');
END;

INSERT INTO source_of_truth_manifest (
    table_name, category, authoritative, legacy_toml_path, description, migration_status
) VALUES
    (
        'claim_transition_events',
        'control_plane',
        1,
        'registry/claim_transitions.toml',
        'Append-only claim transition events with experiment verdicts and evidence',
        'migrated'
    ),
    (
        'claim_relations',
        'control_plane',
        1,
        'registry/claim_relations.toml',
        'Typed predecessor and successor claim relations',
        'migrated'
    );
