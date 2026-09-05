-- Reconstruct the immutable baseline with git show 1ad0e782:registry/canonical/control_plane.sqlite3.
-- Save those bytes as .cache/canonical-before-evidence-contracts.sqlite3 before running sqlite3 -json.
ATTACH DATABASE '.cache/canonical-before-evidence-contracts.sqlite3' AS baseline;
SELECT 'historical_claims' AS surface,
       (SELECT count(*) FROM baseline.claims) AS expected_rows,
       (SELECT count(*) FROM (SELECT * FROM baseline.claims EXCEPT SELECT * FROM claims)) AS missing_or_changed_rows
UNION ALL
SELECT 'historical_transitions',
       (SELECT count(*) FROM baseline.claim_transition_events),
       (SELECT count(*) FROM (SELECT * FROM baseline.claim_transition_events EXCEPT SELECT * FROM claim_transition_events))
UNION ALL
SELECT 'curated_binary_metadata',
       (SELECT count(*) FROM baseline.binaries_cp),
       (SELECT count(*) FROM (SELECT name, description, experiment_id, source FROM baseline.binaries_cp EXCEPT SELECT name, description, experiment_id, source FROM binaries_cp))
UNION ALL
SELECT 'historical_artifact_metadata',
       (SELECT count(*) FROM baseline.artifacts),
       (SELECT count(*) FROM (SELECT id, key, title, citation, status, minimum_requirement_met, canonical_download_path FROM baseline.artifacts EXCEPT SELECT id, key, title, citation, status, minimum_requirement_met, canonical_download_path FROM artifacts))
UNION ALL
SELECT 'historical_artifact_links',
       (SELECT count(*) FROM baseline.artifact_links),
       (SELECT count(*) FROM (SELECT * FROM baseline.artifact_links EXCEPT SELECT * FROM artifact_links));
