.bail on
.mode json
ATTACH '.cache/artifact-provenance/before.sqlite3' AS baseline;
CREATE TEMP TABLE assertions(name TEXT PRIMARY KEY, ok INTEGER NOT NULL CHECK(ok = 1));
CREATE TEMP TABLE planned AS
SELECT value AS item FROM json_each(CAST(readfile('data/output/audit/artifact-document-path-provenance/repair-report.json') AS TEXT), '$.spec.repair');
CREATE TEMP TABLE expected_removed AS
SELECT json_extract(item,'$.id') AS artifact_id, value AS path, 'downloaded' AS relation
FROM planned, json_each(item,'$.expected_downloaded_paths');
CREATE TEMP TABLE expected_added AS
SELECT json_extract(item,'$.id') AS artifact_id, value AS path, json_extract(item,'$.old_path_relation') AS relation
FROM planned, json_each(item,'$.expected_downloaded_paths')
UNION ALL
SELECT json_extract(item,'$.id'), json_extract(item,'$.replacement_path'), 'downloaded' FROM planned
UNION ALL
SELECT json_extract(item,'$.id'), json_extract(value,'$.path'), json_extract(value,'$.relation')
FROM planned, json_each(item,'$.related_path');
CREATE TEMP TABLE observed_removed AS SELECT * FROM baseline.artifact_paths EXCEPT SELECT * FROM main.artifact_paths;
CREATE TEMP TABLE observed_added AS
SELECT * FROM main.artifact_paths WHERE artifact_id != 'LOCAL-ARTIFACT-DOCUMENT-PATH-PROVENANCE'
EXCEPT SELECT * FROM baseline.artifact_paths;
INSERT INTO assertions VALUES ('five_unique_artifact_ids', (SELECT count(*) = 5 AND count(DISTINCT json_extract(item,'$.id')) = 5 FROM planned));
INSERT INTO assertions VALUES ('seven_reclassified_old_paths', (SELECT count(*) = 7 FROM expected_removed));
INSERT INTO assertions VALUES ('thirteen_new_relations', (SELECT count(*) = 13 FROM expected_added));
INSERT INTO assertions VALUES ('exact_removed_paths', NOT EXISTS(SELECT * FROM observed_removed EXCEPT SELECT * FROM expected_removed) AND NOT EXISTS(SELECT * FROM expected_removed EXCEPT SELECT * FROM observed_removed));
INSERT INTO assertions VALUES ('exact_added_paths', NOT EXISTS(SELECT * FROM observed_added EXCEPT SELECT * FROM expected_added) AND NOT EXISTS(SELECT * FROM expected_added EXCEPT SELECT * FROM observed_added));
INSERT INTO assertions VALUES ('original_artifact_metadata_preserved', NOT EXISTS(
 SELECT id,key,title,citation,status,minimum_requirement_met,canonical_functional_url FROM baseline.artifacts
 EXCEPT SELECT id,key,title,citation,status,minimum_requirement_met,canonical_functional_url FROM main.artifacts));
INSERT INTO assertions VALUES ('canonical_paths_match_plan', NOT EXISTS(
 SELECT 1 FROM main.artifacts a JOIN baseline.artifacts b ON a.id=b.id
 WHERE a.canonical_download_path IS NOT COALESCE((SELECT json_extract(item,'$.replacement_path') FROM planned WHERE json_extract(item,'$.id')=a.id), b.canonical_download_path)));
INSERT INTO assertions VALUES ('only_declared_bundle_addition', NOT EXISTS(SELECT id FROM main.artifacts WHERE id != 'LOCAL-ARTIFACT-DOCUMENT-PATH-PROVENANCE' EXCEPT SELECT id FROM baseline.artifacts));
INSERT INTO assertions VALUES ('source_references_preserved', NOT EXISTS(SELECT * FROM baseline.record_sources EXCEPT SELECT * FROM main.record_sources));
INSERT INTO assertions VALUES ('lane_memberships_preserved', NOT EXISTS(SELECT * FROM baseline.lane_assignments EXCEPT SELECT * FROM main.lane_assignments));
INSERT INTO assertions VALUES ('citations_preserved', NOT EXISTS(SELECT * FROM baseline.citations EXCEPT SELECT * FROM main.citations));
INSERT INTO assertions VALUES ('claim_rows_preserved', NOT EXISTS(SELECT * FROM baseline.claims EXCEPT SELECT * FROM main.claims));
INSERT INTO assertions VALUES ('transition_events_preserved', NOT EXISTS(SELECT * FROM baseline.claim_transition_events EXCEPT SELECT * FROM main.claim_transition_events));
INSERT INTO assertions VALUES ('prior_export_history_preserved', NOT EXISTS(SELECT * FROM baseline.export_runs EXCEPT SELECT * FROM main.export_runs));
SELECT name,ok FROM assertions ORDER BY name;
