-- Restore d517da87:registry/canonical/control_plane.sqlite3 into the named cache
-- before running sqlite3 -json against the canonical database.
ATTACH DATABASE '.cache/canonical-before-protocol-sealing.sqlite3' AS before_sealing;
SELECT 'claims' AS surface,
       (SELECT count(*) FROM before_sealing.claims) AS expected_rows,
       (SELECT count(*) FROM (SELECT * FROM before_sealing.claims EXCEPT SELECT * FROM claims)) AS missing_or_changed_rows
UNION ALL
SELECT 'claim_transition_events',
       (SELECT count(*) FROM before_sealing.claim_transition_events),
       (SELECT count(*) FROM (SELECT * FROM before_sealing.claim_transition_events EXCEPT SELECT * FROM claim_transition_events))
UNION ALL
SELECT 'claim_evidence_revisions',
       (SELECT count(*) FROM before_sealing.claim_evidence_revisions),
       (SELECT count(*) FROM (SELECT * FROM before_sealing.claim_evidence_revisions EXCEPT SELECT * FROM claim_evidence_revisions))
UNION ALL
SELECT 'claim_evidence_revision_experiments',
       (SELECT count(*) FROM before_sealing.claim_evidence_revision_experiments),
       (SELECT count(*) FROM (SELECT * FROM before_sealing.claim_evidence_revision_experiments EXCEPT SELECT * FROM claim_evidence_revision_experiments))
UNION ALL
SELECT 'unrelated_experiments',
       (SELECT count(*) FROM before_sealing.experiments_cp WHERE id!='E-283'),
       (SELECT count(*) FROM (SELECT * FROM before_sealing.experiments_cp WHERE id!='E-283' EXCEPT SELECT * FROM experiments_cp WHERE id!='E-283'))
UNION ALL
SELECT 'contracts_except_protocol_digest',
       (SELECT count(*) FROM before_sealing.claim_evidence),
       (SELECT count(*) FROM (
           SELECT claim_id,json_remove(spec_json,'$.decisive_experiment.protocol_sha256') FROM before_sealing.claim_evidence
           EXCEPT
           SELECT claim_id,json_remove(spec_json,'$.decisive_experiment.protocol_sha256') FROM claim_evidence
       ))
UNION ALL
SELECT 'artifacts',
       (SELECT count(*) FROM before_sealing.artifacts),
       (SELECT count(*) FROM (SELECT * FROM before_sealing.artifacts EXCEPT SELECT * FROM artifacts))
UNION ALL
SELECT 'artifact_paths',
       (SELECT count(*) FROM before_sealing.artifact_paths),
       (SELECT count(*) FROM (SELECT * FROM before_sealing.artifact_paths EXCEPT SELECT * FROM artifact_paths))
UNION ALL
SELECT 'artifact_retrieval_observations',
       (SELECT count(*) FROM before_sealing.artifact_retrieval_observations),
       (SELECT count(*) FROM (SELECT * FROM before_sealing.artifact_retrieval_observations EXCEPT SELECT * FROM artifact_retrieval_observations));
