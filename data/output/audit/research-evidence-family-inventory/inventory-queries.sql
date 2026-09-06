-- Execute through a read-only SQLite connection.
SELECT 'claims' AS category, status, count(*) AS records FROM claims GROUP BY status
UNION ALL SELECT 'insights', status, count(*) FROM insights GROUP BY status
UNION ALL SELECT 'experiments', status, count(*) FROM experiments_cp GROUP BY status
UNION ALL SELECT 'lacunae', status, count(*) FROM lacunae GROUP BY status;

SELECT json_extract(spec_json, '$.evidence_layer') AS evidence_layer,
       count(*) AS contracts
FROM claim_evidence GROUP BY evidence_layer;

SELECT insight.id, insight.title, insight.status,
       claim.id AS claim_id, claim.status AS claim_status
FROM insights AS insight
JOIN json_each(insight.claim_refs_json) AS reference
JOIN claims AS claim ON claim.id = reference.value
WHERE claim.status IN ('Refuted', 'Closed/Refuted', 'Superseded',
                       'Closed/Methodology-Insufficient', 'Closed/Obstructed',
                       'Closed/Analogy')
ORDER BY insight.id, claim.id;

SELECT 'dangling_insight_claim_refs' AS metric, count(*) AS records
FROM insights AS insight, json_each(insight.claim_refs_json) AS reference
WHERE NOT EXISTS (SELECT 1 FROM claims WHERE id = reference.value)
UNION ALL
SELECT 'normalized_rows_missing_reverse_declaration', count(*)
FROM claim_insight_refs AS reference
LEFT JOIN insights AS insight ON insight.id = reference.insight_id
WHERE NOT EXISTS (
    SELECT 1 FROM json_each(insight.claim_refs_json) AS declared
    WHERE declared.value = reference.claim_id
)
UNION ALL
SELECT 'normalized_rows_missing_claim', count(*)
FROM claim_insight_refs AS reference
LEFT JOIN claims AS claim ON claim.id = reference.claim_id
WHERE claim.id IS NULL
UNION ALL
SELECT 'normalized_rows_missing_insight', count(*)
FROM claim_insight_refs AS reference
LEFT JOIN insights AS insight ON insight.id = reference.insight_id
WHERE insight.id IS NULL
UNION ALL
SELECT 'declared_rows_missing_normalized_reference', count(*)
FROM insights AS insight, json_each(insight.claim_refs_json) AS declared
WHERE NOT EXISTS (
    SELECT 1 FROM claim_insight_refs AS reference
    WHERE reference.insight_id = insight.id AND reference.claim_id = declared.value
);
