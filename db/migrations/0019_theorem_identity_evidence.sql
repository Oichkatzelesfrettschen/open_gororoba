-- 0019_theorem_identity_evidence: retain explicit artifact references for
-- theorem bindings without placing them in a status note.

CREATE TABLE theorem_identity_evidence (
    theorem_stable_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    PRIMARY KEY (theorem_stable_id, artifact_id),
    FOREIGN KEY (theorem_stable_id) REFERENCES theorem_identities(stable_id),
    FOREIGN KEY (artifact_id) REFERENCES artifacts(id)
);

CREATE INDEX theorem_identity_evidence_by_artifact
    ON theorem_identity_evidence(artifact_id);

INSERT INTO source_of_truth_manifest (
    table_name, category, authoritative, legacy_toml_path,
    description, migration_status
) VALUES (
    'theorem_identity_evidence',
    'control_plane',
    1,
    'docs/THEOREMS.md',
    'Evidence artifact references for explicit theorem bindings',
    'migrated'
);
