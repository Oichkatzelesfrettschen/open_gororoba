-- Compare retained canonical rows against a reference snapshot.
ATTACH DATABASE '.cache/utility-before.sqlite3' AS prior;
SELECT 'original_claim_rows_changed_or_missing' AS check_name, count(*) AS violations
FROM (SELECT * FROM prior.claims EXCEPT SELECT * FROM main.claims)
UNION ALL
SELECT 'original_contracts_changed_or_missing', count(*)
FROM (SELECT * FROM prior.claim_evidence EXCEPT SELECT * FROM main.claim_evidence)
UNION ALL
SELECT 'original_transition_events_changed_or_missing', count(*)
FROM (SELECT * FROM prior.claim_transition_events EXCEPT SELECT * FROM main.claim_transition_events)
UNION ALL
SELECT 'original_artifacts_changed_or_missing', count(*)
FROM (SELECT * FROM prior.artifacts EXCEPT SELECT * FROM main.artifacts)
UNION ALL
SELECT 'original_experiment_statements_changed_or_missing', count(*)
FROM (SELECT id,title,status,binary_name,claim_refs_json,status_note FROM prior.experiments_cp
EXCEPT SELECT id,title,status,binary_name,claim_refs_json,status_note FROM main.experiments_cp)
UNION ALL
SELECT 'original_binary_metadata_changed_or_missing', count(*)
FROM (SELECT * FROM prior.binaries_cp EXCEPT SELECT * FROM main.binaries_cp);
DETACH DATABASE prior;
