ATTACH '.cache/macquart-conformance/before.sqlite3' AS baseline;
SELECT 'claims' AS metric, count(*) AS value FROM claims
UNION ALL SELECT 'insights',count(*) FROM insights
UNION ALL SELECT 'admission_events',count(*) FROM insight_admissions
UNION ALL SELECT 'admitted_insights',count(*) FROM insight_evidence
UNION ALL SELECT 'admitted_predicates',count(*) FROM insight_evidence,json_each(spec_json,'$.predicates')
UNION ALL SELECT 'admitted_roles',count(*) FROM insight_evidence,json_each(spec_json,'$.references')
UNION ALL SELECT 'source_history_rows',count(*) FROM source_observations
UNION ALL SELECT 'lost_admission_history',count(*) FROM (SELECT * FROM baseline.insight_admissions EXCEPT SELECT * FROM main.insight_admissions)
UNION ALL SELECT 'lost_source_history',count(*) FROM (SELECT * FROM baseline.source_observations EXCEPT SELECT * FROM main.source_observations)
UNION ALL SELECT 'lost_insight_revisions',count(*) FROM (SELECT * FROM baseline.insight_revisions EXCEPT SELECT * FROM main.insight_revisions)
UNION ALL SELECT 'changed_other_insights',count(*) FROM (SELECT * FROM baseline.insights WHERE id != 'I-001' EXCEPT SELECT * FROM main.insights)
UNION ALL SELECT 'changed_other_insight_evidence',count(*) FROM (SELECT * FROM baseline.insight_evidence WHERE insight_id != 'I-001' EXCEPT SELECT * FROM main.insight_evidence)
UNION ALL SELECT 'lost_links',count(*) FROM (SELECT * FROM baseline.claim_insight_refs EXCEPT SELECT * FROM main.claim_insight_refs)
UNION ALL SELECT 'added_links',count(*) FROM (SELECT * FROM main.claim_insight_refs EXCEPT SELECT * FROM baseline.claim_insight_refs)
UNION ALL SELECT 'missing_declared_links',count(*) FROM insights,json_each(insights.claim_refs_json) AS declared WHERE NOT EXISTS(SELECT 1 FROM claim_insight_refs WHERE insight_id=insights.id AND claim_id=declared.value)
UNION ALL SELECT 'undeclared_normalized_links',count(*) FROM claim_insight_refs WHERE NOT EXISTS(SELECT 1 FROM insights,json_each(insights.claim_refs_json) AS declared WHERE insights.id=claim_insight_refs.insight_id AND declared.value=claim_insight_refs.claim_id)
UNION ALL SELECT 'foreign_key_violations',count(*) FROM pragma_foreign_key_check;
