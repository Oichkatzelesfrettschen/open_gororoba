ATTACH '.cache/insight-admission/before.sqlite3' AS baseline;
SELECT 'admission_events' AS metric, count(*) AS value FROM insight_admissions
UNION ALL SELECT 'admitted_insights',count(DISTINCT insight_id) FROM insight_admissions
UNION ALL SELECT 'source_history_rows',count(*) FROM source_observations
UNION ALL SELECT 'source_observations',count(*) FROM source_observations WHERE json_extract(report_json,'$.corrects_observation_key') IS NULL
UNION ALL SELECT 'source_metadata_corrections',count(*) FROM source_observations WHERE json_extract(report_json,'$.corrects_observation_key') IS NOT NULL
UNION ALL SELECT 'lost_original_links',count(*) FROM (SELECT * FROM baseline.claim_insight_refs EXCEPT SELECT * FROM main.claim_insight_refs)
UNION ALL SELECT 'added_links',count(*) FROM (SELECT * FROM main.claim_insight_refs EXCEPT SELECT * FROM baseline.claim_insight_refs)
UNION ALL SELECT 'lost_original_insight_revisions',count(*) FROM (SELECT * FROM baseline.insight_revisions EXCEPT SELECT * FROM main.insight_revisions)
UNION ALL SELECT 'changed_nonpilot_insights',count(*) FROM (SELECT * FROM baseline.insights WHERE id NOT IN (SELECT insight_id FROM insight_admissions) EXCEPT SELECT * FROM main.insights)
UNION ALL SELECT 'missing_declared_links',count(*) FROM insights,json_each(insights.claim_refs_json) AS declared WHERE NOT EXISTS(SELECT 1 FROM claim_insight_refs WHERE insight_id=insights.id AND claim_id=declared.value)
UNION ALL SELECT 'undeclared_normalized_links',count(*) FROM claim_insight_refs WHERE NOT EXISTS(SELECT 1 FROM insights,json_each(insights.claim_refs_json) AS declared WHERE insights.id=claim_insight_refs.insight_id AND declared.value=claim_insight_refs.claim_id)
UNION ALL SELECT 'admitted_predicates',count(*) FROM insight_evidence,json_each(insight_evidence.spec_json,'$.predicates')
UNION ALL SELECT 'admitted_roles',count(*) FROM insight_evidence,json_each(insight_evidence.spec_json,'$.references')
UNION ALL SELECT 'scientific_replays_declared',count(*) FROM insight_evidence,json_each(insight_evidence.spec_json,'$.predicates') AS predicate WHERE json_extract(predicate.value,'$.execution_status')='replayed';
